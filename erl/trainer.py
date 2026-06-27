from __future__ import annotations

import copy
import functools
import logging
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation

from erl.config import ERLConfig
from erl.memory import ReflectionMemory

# ---------------------------------------------------------------------------
# TRL 0.17.x data utilities — imported with fallbacks so unit tests that run
# without a full TRL install still work. The dep pin in pyproject.toml is
# trl>=0.17.0,<0.18.0; if you bump the pin, retest the helpers below and the
# parent ``_generate_and_score_completions`` / ``_compute_loss`` signatures.
# ---------------------------------------------------------------------------
try:
    from trl.data_utils import maybe_apply_chat_template, is_conversational
except ImportError:  # pragma: no cover
    def is_conversational(example: dict) -> bool:  # type: ignore[misc]
        return isinstance(example.get("prompt", ""), list)

    def maybe_apply_chat_template(example: dict, tokenizer: Any) -> dict:  # type: ignore[misc]
        prompt = example.get("prompt", "")
        if isinstance(prompt, list):
            text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            return {**example, "prompt": text}
        return example

try:
    from trl.data_utils import apply_chat_template as _apply_chat_template
except ImportError:  # pragma: no cover
    _apply_chat_template = None  # type: ignore[assignment]


logger = logging.getLogger("erl")

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_list_of_dicts(inputs: list[dict] | dict[str, Any]) -> list[dict]:
    """Normalise dict-of-lists or list-of-dicts to a canonical list-of-dicts."""
    if isinstance(inputs, list):
        return inputs
    keys = list(inputs.keys())
    if not keys:
        return []
    n = len(inputs[keys[0]])
    return [{k: inputs[k][i] for k in keys} for i in range(n)]


def _prompt_to_text(prompt: Any, tokenizer: Any, *, add_generation_prompt: bool = False) -> str:
    """Convert a prompt (string or list of message dicts) to plain text.

    By default this does *not* append the assistant generation header — the
    returned text is meant to be embedded as content inside another chat
    template (e.g., the reflection prompt). Pass ``add_generation_prompt=True``
    when you intend to use the result as a standalone prompt to ``generate``.
    """
    if isinstance(prompt, str):
        return prompt
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def _format_safe(template: str, **kwargs: Any) -> str:
    """``str.format`` that tolerates unrelated ``{...}`` patterns in values.

    User-supplied content (model attempts, feedback strings) may contain
    literal braces, e.g., LaTeX or JSON snippets. Using ``str.format`` directly
    raises ``KeyError`` / ``IndexError`` on those. This helper escapes braces
    in every substituted value so only the explicit template placeholders are
    interpreted.
    """
    safe = {
        k: (str(v).replace("{", "{{").replace("}", "}}") if isinstance(v, str) else v)
        for k, v in kwargs.items()
    }
    return template.format(**safe)


# ---------------------------------------------------------------------------
# Caching reward wrapper
# ---------------------------------------------------------------------------

class _CachingRewardWrapper:
    """Transparent wrapper that caches rewards and extracts feedback.

    TRL's parent calls the reward function internally during
    ``_generate_and_score_completions``.  This wrapper captures the results
    so ERL can read them without calling the function again.

    Supports two return formats from the wrapped function:

    * ``list[float]`` — plain scores; feedback defaults to empty strings.
    * ``list[tuple[float, str]]`` — ``(score, feedback)`` pairs; the score is
      returned to TRL as a plain float, and the feedback string is cached for
      use in the reflection prompt.
    """

    def __init__(self, func: Callable) -> None:
        self.func = func
        self.last_rewards: list[float] | None = None
        self.last_feedback: list[str] | None = None
        # Copy ``__name__``, ``__qualname__``, ``__doc__``, ``__module__`` so
        # TRL's reward_func_names lookup and any other attribute-based
        # introspection keeps working through the wrapper.
        functools.update_wrapper(self, func, updated=())

    def __call__(self, prompts, completions, **kwargs):
        raw = self.func(prompts=prompts, completions=completions, **kwargs)
        if raw and isinstance(raw[0], tuple):
            self.last_rewards = [r[0] for r in raw]
            self.last_feedback = [r[1] for r in raw]
        else:
            self.last_rewards = list(raw)
            self.last_feedback = [""] * len(raw)
        return self.last_rewards  # TRL always sees list[float]


# ---------------------------------------------------------------------------
# ERLTrainer
# ---------------------------------------------------------------------------

class ERLTrainer(GRPOTrainer):
    """Experiential Reinforcement Learning trainer for TRL 0.17.0+.

    Extends ``GRPOTrainer`` by overriding ``_generate_and_score_completions``
    to inject the full ERL reflection-retry-internalization loop after the
    first attempt.  The returned dict has the same keys as the parent's method,
    so ``_compute_loss`` works without modification.

    **Unified reward API** — the reward function doubles as a feedback source.
    Return plain floats for GRPO-compatible training (no feedback), or return
    ``(score, feedback)`` tuples for richer reflection prompts:

    .. code-block:: python

        # Format A — GRPO-compatible, no feedback:
        def reward_func(prompts, completions, **kwargs):
            return [compute_score(p, c) for p, c in zip(prompts, completions)]

        # Format B — (score, feedback) pairs for better reflections:
        def reward_func(prompts, completions, **kwargs):
            return [(compute_score(p, c), explain(p, c))
                    for p, c in zip(prompts, completions)]

        trainer = ERLTrainer(model=..., reward_funcs=reward_func, ...)

    The reward function is called **once** per training step for first-attempt
    completions (by the parent).  ERL reads the cached result without a second
    call, eliminating redundant inference.

    This implementation follows **Algorithm 2** from the ERL paper:

    * **y1 RL update** — delegated to the parent's GRPO loss on first-attempt
      completions.  Advantages are derived from r1 (y1 rewards) only.
    * **Δ + y2 RL update** — a separate GRPO loss computed in ``compute_loss``
      using the stored ``_erl_rl_data``.  Both Δ (reflections) and y2
      (second-attempt completions) receive reward r2; advantages are
      batch-wide-normalised.
    * **Internalization** — SFT cross-entropy on ``(original_prompt → y2)``
      for successful second attempts (r2 > 0).

    Args:
        model: Model to train (same as GRPOTrainer).
        reward_funcs: Reward function(s) (same as GRPOTrainer).  May return
            ``list[float]`` or ``list[tuple[float, str]]``; see above.
        args: ERLConfig with ERL-specific hyperparameters.
        **kwargs: Forwarded verbatim to ``GRPOTrainer.__init__``.
    """

    def __init__(
        self,
        model: str | nn.Module,
        reward_funcs: Callable | list[Callable],
        args: ERLConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if args is None:
            args = ERLConfig(output_dir="erl-output")
        super().__init__(model=model, reward_funcs=reward_funcs, args=args, **kwargs)

        # Wrap callable (non-Module) reward functions so their outputs are
        # cached after the parent's call and can be read without re-invoking
        # the function for y1 scoring.
        self._reward_wrappers: list[_CachingRewardWrapper] = []
        for i, func in enumerate(self.reward_funcs):
            if not isinstance(func, nn.Module):
                wrapper = _CachingRewardWrapper(func)
                self.reward_funcs[i] = wrapper
                self._reward_wrappers.append(wrapper)

        self.memory: ReflectionMemory | None = (
            ReflectionMemory(args.memory_size) if args.enable_memory else None
        )
        self._internalization_pairs: list[tuple[Any, str]] = []
        self._erl_rl_data: dict | None = None

        if self.args.erl_debug:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)
            logger.propagate = False

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------

    def _strip_chat_markers(self, text: str) -> str:
        """Remove chat-control tokens from a prompt before substitution into the
        reflection / retry template.

        When the caller's dataset emits pre-templated prompts (e.g. with Qwen3.5's
        `<|im_start|>system\\n...<|im_end|>` markers so that y1 generation can
        access a system prompt), the templated string flows through
        ``prompts_raw[i]`` into ``_build_reflection_prompt``. Naively substituting
        it leaves chat-control tokens *inside* the reflection user turn — the
        model then sees nested role tokens like:
            <|im_start|>user
            ## Brief
            <|im_start|>system
            ...
            <|im_start|>assistant
            (← the brief's leftover assistant tag)
            ...
            <|im_start|>assistant   (← the reflection's actual assistant tag)
        which breaks the role boundary the model was trained on and frequently
        causes empty or degenerate completions in batched generation.

        We strip the common chat markers from the substituted text. The brief
        content remains intact; only role wrappers are removed.
        """
        markers = (
            "<|im_start|>system\n", "<|im_start|>user\n", "<|im_start|>assistant\n",
            "<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant",
            "<|im_end|>", "<|im_start|>",
            "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
            "<|eot_id|>",
            "<start_of_turn>", "<end_of_turn>",
            "<think>\n\n</think>\n\n",  # Qwen3.5 enable_thinking=False empty block
            "<think>", "</think>",
        )
        out = text
        for m in markers:
            out = out.replace(m, "")
        return out.strip()

    def _build_reflection_prompt(
        self,
        prompt: str,
        attempt: str,
        feedback: str,
        reward: float,
        memory_entries: list[str],
    ) -> str:
        """Format the reflection prompt using the configured template.

        Uses ``_format_safe`` so braces in user content (LaTeX, JSON) cannot
        break the template substitution. Also strips chat-control tokens from
        ``prompt`` and ``attempt`` so that pre-templated dataset prompts do not
        embed nested role wrappers inside the reflection user turn (which would
        confuse role boundaries and cause empty completions).
        """
        memory_str = "\n\n".join(memory_entries) if memory_entries else "None available."
        return _format_safe(
            self.args.reflection_system_prompt,
            prompt=self._strip_chat_markers(prompt),
            attempt=self._strip_chat_markers(attempt),
            feedback=feedback,
            reward=reward,
            memory=memory_str,
        )

    def _build_retry_prompt(self, prompt: str, reflection: str) -> str:
        """Format the retry prompt using the configured template.

        Strips chat-control tokens from ``prompt`` and ``reflection`` so that
        pre-templated text does not embed nested role wrappers inside the retry
        user turn.
        """
        return _format_safe(
            self.args.retry_system_prompt,
            prompt=self._strip_chat_markers(prompt),
            reflection=self._strip_chat_markers(reflection),
        )

    # ------------------------------------------------------------------
    # TRL version compatibility
    # ------------------------------------------------------------------

    def _erl_get_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compatibility wrapper for TRL's per-token log-prob computation.

        ``_get_per_token_logps_and_entropies`` was added mid-0.17 series.
        Falls back to the older ``_get_per_token_logps`` (present in
        TRL 0.17.0) when the newer method is not available.
        """
        if hasattr(self, "_get_per_token_logps_and_entropies"):
            return self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep=logits_to_keep
            )
        logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep=logits_to_keep
        )
        return logps, None

    # ------------------------------------------------------------------
    # Internal generation helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _erl_generate(
        self,
        prompt_texts: list[str],
        *,
        compute_logps: bool = False,
        phase: str = "default",
    ) -> tuple:
        """Generate one completion per plain-text prompt string.

        Args:
            prompt_texts: Plain-text content for each prompt.
            compute_logps: When ``True``, also compute per-token log-probs from
                the current policy and reference model (if available).

        Returns:
            7-tuple ``(prompt_ids, prompt_mask, completion_ids, completion_texts,
            completion_mask, old_per_token_logps, ref_per_token_logps)``.
            ``completion_mask``, ``old_per_token_logps``, and
            ``ref_per_token_logps`` are ``None`` when ``compute_logps=False``.
        """
        if not prompt_texts:
            device = self.accelerator.device
            empty = torch.zeros(0, 0, dtype=torch.long, device=device)
            return empty, empty, empty, [], None, None, None

        device = self.accelerator.device
        tokenizer = self.processing_class

        # Format each prompt into the string the tokenizer will see. Three cases:
        #
        # 1) Plain text (str): wrap as a single user turn and apply chat template.
        # 2) Conversational (list of {role, content} dicts): apply chat template
        #    directly — caller has set up system/user roles intentionally.
        # 3) Pre-templated string (str containing chat-control tokens like
        #    "<|im_start|>"): DO NOT re-wrap. Re-templating a pre-templated
        #    string puts the chat control tokens inside another user turn
        #    ("double-templating"), which destroys the assistant generation
        #    context — the model sees nested user/system tags and produces
        #    garbage. Detect by looking for any known chat marker.
        #
        # For cases (1) and (2), pass `enable_thinking=False` if the tokenizer
        # supports it. Qwen3.5 (and other thinking-capable models) default to
        # opening a `<think>` block at the assistant turn — the model then
        # writes its reasoning instead of the requested completion, and the
        # max_completion_length budget is exhausted before the actual output.
        # Reflection and retry prompts especially need this off — we want
        # critique/ad text, not chain-of-thought.
        formatted_texts: list[str] = []
        chat_markers = (
            "<|im_start|>", "<|im_end|>",        # Qwen, ChatML
            "<|begin_of_text|>", "<|start_header_id|>",  # Llama 3
            "<start_of_turn>", "<end_of_turn>",  # Gemma
        )

        def _apply(messages: list[dict]) -> str:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # Older tokenizers without `enable_thinking` parameter
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

        for t in prompt_texts:
            if isinstance(t, list):
                # Case 2: caller passed message list directly
                formatted_texts.append(_apply(t))
            elif isinstance(t, str) and any(m in t for m in chat_markers):
                # Case 3: already templated, pass through unchanged
                formatted_texts.append(t)
            else:
                # Case 1: plain text, wrap and template
                formatted_texts.append(
                    _apply([{"role": "user", "content": t}])
                )

        prompt_inputs = tokenizer(
            text=formatted_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids: torch.Tensor = prompt_inputs["input_ids"].to(device)
        prompt_mask: torch.Tensor = prompt_inputs["attention_mask"].to(device)

        max_prompt_length = getattr(self, "max_prompt_length", None)
        if max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -max_prompt_length:]
            prompt_mask = prompt_mask[:, -max_prompt_length:]

        # Default matches GRPOConfig (True). Reading False here would break
        # generation under DeepSpeed ZeRO-3 because parameters would be sharded
        # across ranks.
        gather_ds3 = getattr(self.args, "ds3_gather_for_generation", True)

        # Per-phase generation config override. Reflection (Phase 3) generation
        # is prone to mode-collapse driven by the internalization SFT phase
        # reinforcing whatever phrasing happens to produce y2 > y1 lift early
        # in training. Higher temperature / top_p at the reflection step breaks
        # the prior on opening tokens and keeps reflections diverse.
        gen_config = self.generation_config
        if phase == "reflection":
            r_temp = getattr(self.args, "reflection_temperature", None)
            r_top_p = getattr(self.args, "reflection_top_p", None)
            r_top_k = getattr(self.args, "reflection_top_k", None)
            if r_temp is not None or r_top_p is not None or r_top_k is not None:
                from copy import deepcopy
                gen_config = deepcopy(self.generation_config)
                if r_temp is not None:
                    gen_config.temperature = r_temp
                if r_top_p is not None:
                    gen_config.top_p = r_top_p
                if r_top_k is not None:
                    gen_config.top_k = r_top_k

        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=gather_ds3,
        ) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=gen_config,
            )

        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        if not compute_logps:
            return prompt_ids, prompt_mask, completion_ids, completion_texts, None, None, None

        # ── EOS masking ────────────────────────────────────────────────────────
        C = completion_ids.shape[1]
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None:
            is_eos = completion_ids == eos_token_id
            eos_idx = is_eos.int().argmax(dim=1)
            has_eos = is_eos.any(dim=1)
            eos_idx = torch.where(has_eos, eos_idx, torch.full_like(eos_idx, C - 1))
        else:
            eos_idx = torch.full((completion_ids.shape[0],), C - 1, device=device)
        seq_idx = torch.arange(C, device=device).unsqueeze(0)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).long()

        # ── Old policy log-probs ───────────────────────────────────────────────
        full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        old_per_token_logps, _ = self._erl_get_logps(
            self.model, prompt_completion_ids, full_mask, logits_to_keep=C
        )

        # ── Reference model log-probs ──────────────────────────────────────────
        ref_model = getattr(self, "ref_model", None)
        beta = getattr(self, "beta", 0.0)
        if ref_model is not None and beta != 0.0:
            ref_per_token_logps, _ = self._erl_get_logps(
                ref_model, prompt_completion_ids, full_mask, logits_to_keep=C
            )
        else:
            ref_per_token_logps = None

        return (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_texts,
            completion_mask,
            old_per_token_logps,
            ref_per_token_logps,
        )

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _erl_compute_rewards(
        self,
        inputs: list[dict],
        prompts: list,
        completions: list,
    ) -> torch.Tensor:
        """Compute scalar rewards for a (sub)batch using ``self.reward_funcs``.

        Used for y2 scoring (second-attempt completions that the parent has
        never seen).  For y1, use ``_read_cached_y1_results`` instead.

        Args:
            inputs: Dataset dicts for this (sub)batch.
            prompts: Prompts in the format expected by reward functions.
            completions: Completions in the format expected by reward functions.

        Returns:
            1-D float tensor of weighted scalar rewards, shape ``(B,)``.
        """
        device = self.accelerator.device
        n = len(prompts)
        rewards_per_func = torch.zeros(n, len(self.reward_funcs), device=device)

        reward_processing_classes = getattr(
            self, "reward_processing_classes", [None] * len(self.reward_funcs)
        )
        reward_weights = getattr(self, "reward_weights", None)
        if reward_weights is None:
            reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)

        for i, (reward_func, rpc) in enumerate(
            zip(self.reward_funcs, reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    if _apply_chat_template is not None:
                        texts = [
                            _apply_chat_template(x, rpc)["text"] for x in messages
                        ]
                    else:
                        texts = [
                            rpc.apply_chat_template(x["messages"], tokenize=False)
                            for x in messages
                        ]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = rpc(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = {k: v.to(device) for k, v in reward_inputs.items()}
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Callable reward function (may be a _CachingRewardWrapper)
                keys = [k for k in inputs[0] if k not in ("prompt", "completion")]
                reward_kwargs = {k: [ex[k] for ex in inputs] for k in keys}
                output = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs
                )
                output = [r if r is not None else torch.nan for r in output]
                rewards_per_func[:, i] = torch.tensor(
                    output, dtype=torch.float32, device=device
                )

        return (rewards_per_func * reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

    def _read_cached_y1_results(
        self,
        inputs: list[dict],
        prompts: list,
        completions: list,
    ) -> tuple[torch.Tensor, list[str]]:
        """Read y1 rewards and feedback from the wrapper cache.

        The parent's ``_generate_and_score_completions`` calls each wrapped
        reward function exactly once.  This method reads the cached results so
        ERL does not call the functions a second time for y1 scoring.

        Falls back to ``_erl_compute_rewards`` when no wrapper caches are
        populated (e.g., all reward functions are ``nn.Module``-based, or the
        parent was bypassed in tests).

        Args:
            inputs: Dataset dicts for the full batch.
            prompts: Prompts for the full batch.
            completions: y1 completions for the full batch.

        Returns:
            ``(rewards, feedback)`` where ``rewards`` is a 1-D float tensor
            of shape ``(B,)`` and ``feedback`` is a list of strings.
        """
        n = len(prompts)

        # Check if we have cached results from any wrapper
        if not self._reward_wrappers or self._reward_wrappers[0].last_rewards is None:
            # No cache: fall back to recomputing
            return (
                self._erl_compute_rewards(inputs, prompts, completions),
                [""] * n,
            )

        device = self.accelerator.device
        reward_weights = getattr(self, "reward_weights", None)
        if reward_weights is None:
            reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)
        reward_weights = reward_weights.to(device)

        reward_processing_classes = getattr(
            self, "reward_processing_classes", [None] * len(self.reward_funcs)
        )
        rewards_per_func = torch.zeros(n, len(self.reward_funcs), device=device)

        for i, func in enumerate(self.reward_funcs):
            if isinstance(func, _CachingRewardWrapper):
                # Read from cache — no function call
                cached = func.last_rewards
                rewards_per_func[:, i] = torch.tensor(
                    [r if r is not None else float("nan") for r in cached],
                    dtype=torch.float32,
                    device=device,
                )
            elif isinstance(func, nn.Module):
                # Module-based reward: must re-call (no cache available)
                rpc = reward_processing_classes[i]
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    if _apply_chat_template is not None:
                        texts = [_apply_chat_template(x, rpc)["text"] for x in messages]
                    else:
                        texts = [
                            rpc.apply_chat_template(x["messages"], tokenize=False)
                            for x in messages
                        ]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = rpc(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = {k: v.to(device) for k, v in reward_inputs.items()}
                with torch.inference_mode():
                    rewards_per_func[:, i] = func(**reward_inputs).logits[:, 0]

        weighted = (rewards_per_func * reward_weights.unsqueeze(0)).nansum(dim=1)

        # Feedback from the first wrapper that has non-empty feedback strings
        feedback: list[str] = [""] * n
        for func in self.reward_funcs:
            if (
                isinstance(func, _CachingRewardWrapper)
                and func.last_feedback
                and any(f for f in func.last_feedback)
            ):
                feedback = func.last_feedback
                break

        return weighted, feedback

    # ------------------------------------------------------------------
    # Advantage normalisation for Δ and y2
    # ------------------------------------------------------------------

    @staticmethod
    def _erl_compute_r2_advantages(rewards: torch.Tensor) -> torch.Tensor:
        """Batch-wide advantage normalisation for Δ and y2 rewards.

        Unlike y1 which uses group-wise normalisation (GRPO), Δ and y2 use
        batch-wide normalisation because each gated sample produces exactly one
        Δ and one y2 (group size 1 makes per-group normalisation degenerate).

        Returns all-zeros when ``N <= 1``.
        """
        if rewards.shape[0] <= 1:
            return torch.zeros_like(rewards)
        mean = rewards.mean()
        std = rewards.std()
        return (rewards - mean) / (std + 1e-4)

    # ------------------------------------------------------------------
    # GRPO loss for Δ and y2 completions
    # ------------------------------------------------------------------

    def _erl_grpo_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """GRPO loss for Δ (reflection) or y2 (second-attempt) completions.

        Replicates TRL's ``_compute_loss`` PPO clip formula without metrics
        logging.  Called twice per step (once for Δ, once for y2).
        """
        prompt_ids  = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        completion_ids  = batch["completion_ids"]
        completion_mask = batch["completion_mask"].float()
        old_per_token_logps = batch["old_per_token_logps"]
        ref_per_token_logps = batch.get("ref_per_token_logps")
        advantages  = batch["advantages"]

        C = completion_ids.shape[1]
        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask.long()], dim=1)

        per_token_logps, _ = self._erl_get_logps(
            model, input_ids, attention_mask, logits_to_keep=C
        )

        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        log_ratio = per_token_logps - old_per_token_logps
        if getattr(self, "importance_sampling_level", "token") == "token":
            log_iw = log_ratio
        else:
            log_iw = (
                (log_ratio * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).unsqueeze(-1)

        coef_1 = torch.exp(log_iw)

        beta = getattr(self, "beta", 0.0)
        per_token_kl = None
        if beta != 0.0 and ref_per_token_logps is not None:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        loss_type = getattr(self, "loss_type", "grpo")
        eps_low  = getattr(self, "epsilon_low",  0.2)
        eps_high = getattr(self, "epsilon_high", 0.2)

        if loss_type in ("grpo", "bnpo", "dr_grpo", "dapo", "luspo"):
            coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
            per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
        elif loss_type == "cispo":
            clamped = torch.clamp(coef_1, max=eps_high).detach()
            per_token_loss = -clamped * advantages * per_token_logps
        elif loss_type == "sapo":
            temps = torch.where(
                advantages > 0,
                torch.full_like(advantages, self.args.sapo_temperature_pos),
                torch.full_like(advantages, self.args.sapo_temperature_neg),
            )
            soft_coef = torch.sigmoid(temps * (coef_1 - 1)) * 4 / temps
            per_token_loss = -soft_coef * advantages
        else:
            coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
            per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)

        if per_token_kl is not None:
            per_token_loss = per_token_loss + beta * per_token_kl

        mask = completion_mask
        # NOTE: do not divide by gradient_accumulation_steps here.
        # HF Trainer's training_step performs that division automatically on
        # the value returned by ``compute_loss``. Pre-dividing produces a 1/N²
        # scale and effectively suppresses the ERL gradient.

        if loss_type in ("grpo", "sapo", "dapo"):
            loss = (
                (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
        elif loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        elif loss_type == "luspo":
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
        else:
            loss = (
                (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            ).mean()

        return loss

    # ------------------------------------------------------------------
    # Main ERL loop
    # ------------------------------------------------------------------

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]] | dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Run the full ERL loop and return a batch dict for ``_compute_loss``."""
        inputs = _to_list_of_dicts(inputs)

        _debug = self.args.erl_debug
        if _debug:
            _step = self.state.global_step

        # Clear wrapper caches so stale data from the previous step cannot
        # leak into y1 reward/feedback reads for this step.
        for wrapper in self._reward_wrappers:
            wrapper.last_rewards = None
            wrapper.last_feedback = None

        inputs_orig = copy.deepcopy(inputs)
        prompts_raw = [x["prompt"] for x in inputs_orig]

        # ── Phase 1: First attempt ─────────────────────────────────────────
        y1_result = super()._generate_and_score_completions(inputs)

        y1_texts = self.processing_class.batch_decode(
            y1_result["completion_ids"], skip_special_tokens=True
        )

        if is_conversational(inputs_orig[0]):
            completions_y1 = [[{"role": "assistant", "content": t}] for t in y1_texts]
        else:
            completions_y1 = y1_texts

        # Read y1 rewards from the wrapper cache (no extra forward pass).
        # Fall back to recomputing only when no wrappers are populated.
        y1_rewards, y1_feedback = self._read_cached_y1_results(
            inputs_orig, prompts_raw, completions_y1
        )

        if _debug:
            n = len(y1_texts)
            r_list = [round(r, 3) for r in y1_rewards.tolist()]
            logger.debug(
                f"[ERL Step {_step}] Phase 1 — First Attempt\n"
                f"  Batch size: {n}\n"
                f"  Y1 rewards: {r_list}\n"
                f"  Y1 reward mean: {y1_rewards.mean().item():.3f}\n"
                f"  Y1 sample (first): \"{y1_texts[0][:100]}...\""
            )

        # ── Phase 2: Gating ─────────────────────────────────────────────────
        gated_mask = (y1_rewards < self.args.reward_threshold).tolist()
        gated_indices = [i for i, g in enumerate(gated_mask) if g]

        if _debug:
            logger.debug(
                f"[ERL Step {_step}] Phase 2 — Gating (threshold={self.args.reward_threshold})\n"
                f"  Gated: {len(gated_indices)}/{len(y1_texts)} samples\n"
                f"  Gated indices: {gated_indices}"
            )

        if not gated_indices:
            self._internalization_pairs = []
            self._erl_rl_data = None
            return y1_result

        compute_logps = self.args.erl_rl_coef > 0.0

        # ── Phase 3: Self-reflection (batched) ──────────────────────────────
        reflection_prompts: list[str] = []
        for i in gated_indices:
            prompt_str = _prompt_to_text(prompts_raw[i], self.processing_class)
            memory_entries: list[str] = (
                self.memory.retrieve(prompt_str, self.args.memory_top_k)
                if self.args.enable_memory and self.memory is not None
                else []
            )
            reflection_prompts.append(
                self._build_reflection_prompt(
                    prompt=prompt_str,
                    attempt=y1_texts[i],
                    feedback=y1_feedback[i],
                    reward=float(y1_rewards[i].item()),
                    memory_entries=memory_entries,
                )
            )

        (
            prompt_ids_d, prompt_mask_d, completion_ids_d, reflection_texts,
            completion_mask_d, old_logps_d, ref_logps_d,
        ) = self._erl_generate(
            reflection_prompts, compute_logps=compute_logps, phase="reflection"
        )

        if _debug:
            logger.debug(
                f"[ERL Step {_step}] Phase 3 — Reflection\n"
                f"  Generated {len(reflection_texts)} reflections\n"
                f"  Reflection sample (first): \"{reflection_texts[0][:150]}...\""
            )

        # ── Phase 4: Second attempt (batched) ───────────────────────────────
        retry_prompts: list[str] = [
            self._build_retry_prompt(
                prompt=_prompt_to_text(prompts_raw[i], self.processing_class),
                reflection=reflection_texts[j],
            )
            for j, i in enumerate(gated_indices)
        ]

        (
            prompt_ids_y2, prompt_mask_y2, completion_ids_y2, y2_texts,
            completion_mask_y2, old_logps_y2, ref_logps_y2,
        ) = self._erl_generate(retry_prompts, compute_logps=compute_logps)

        gated_inputs = [inputs_orig[i] for i in gated_indices]
        gated_prompts = [prompts_raw[i] for i in gated_indices]
        if is_conversational(inputs_orig[0]):
            completions_y2 = [[{"role": "assistant", "content": t}] for t in y2_texts]
        else:
            completions_y2 = y2_texts

        y2_rewards = self._erl_compute_rewards(gated_inputs, gated_prompts, completions_y2)

        if _debug:
            r2_list = [round(r, 3) for r in y2_rewards.tolist()]
            gated_r1 = [round(y1_rewards[i].item(), 3) for i in gated_indices]
            improved = sum(1 for r1, r2 in zip(gated_r1, r2_list) if r2 > r1)
            logger.debug(
                f"[ERL Step {_step}] Phase 4 — Second Attempt\n"
                f"  Y2 rewards: {r2_list}\n"
                f"  Y2 reward mean: {y2_rewards.mean().item():.3f}\n"
                f"  Y2 sample (first): \"{y2_texts[0][:100]}...\"\n"
                f"  Improvement: {improved}/{len(gated_indices)} samples improved"
            )

        # ── Phase 5: Memory update ───────────────────────────────────────────
        # ERL paper Alg. 2 line 18: memory is written when r2 > τ. With
        # continuous rewards in [0, 1] and τ at the top of that range, that
        # gate never fires and the memory feature is silently dead. Use a
        # separate `memory_add_threshold` when set; otherwise fall back to
        # the retry threshold (paper-faithful behaviour).
        mem_thresh = (
            self.args.memory_add_threshold
            if self.args.memory_add_threshold is not None
            else self.args.reward_threshold
        )
        if self.args.enable_memory and self.memory is not None:
            for j, i in enumerate(gated_indices):
                if y2_rewards[j].item() > mem_thresh:
                    prompt_str = _prompt_to_text(prompts_raw[i], self.processing_class)
                    self.memory.add(
                        reflection_texts[j], prompt_str, float(y2_rewards[j].item())
                    )

        if _debug:
            mem_added = sum(
                1 for j in range(len(gated_indices))
                if y2_rewards[j].item() > mem_thresh
            )
            mem_size = len(self.memory) if self.memory is not None else 0
            logger.debug(
                f"[ERL Step {_step}] Phase 5 — Memory Update\n"
                f"  New entries added: {mem_added}\n"
                f"  Total memory size: {mem_size}"
            )

        # ── Phase 6: Y1 advantages ───────────────────────────────────────────
        # The parent's ``_generate_and_score_completions`` has already written
        # group-normalised advantages into ``y1_result["advantages"]`` honouring
        # ``scale_rewards`` and the per-reward-function weights. We deliberately
        # do not recompute them here — re-deriving from cached ``y1_rewards``
        # silently drifts when ``scale_rewards != "group"`` and bypasses the
        # parent's per-func weighting.

        if _debug:
            adv_sample = [
                round(a, 3) for a in y1_result["advantages"][:4].tolist()
            ]
            logger.debug(
                f"[ERL Step {_step}] Phase 6 — Advantages\n"
                f"  Y1 advantages (first 4, from parent): {adv_sample}"
            )

        # ── Phase 6b: Store Δ + y2 RL data ───────────────────────────────────
        if compute_logps:
            y2_advantages = self._erl_compute_r2_advantages(y2_rewards)
            self._erl_rl_data = {
                "delta": {
                    "prompt_ids":          prompt_ids_d,
                    "prompt_mask":         prompt_mask_d,
                    "completion_ids":      completion_ids_d,
                    "completion_mask":     completion_mask_d,
                    "old_per_token_logps": old_logps_d,
                    "ref_per_token_logps": ref_logps_d,
                },
                "y2": {
                    "prompt_ids":          prompt_ids_y2,
                    "prompt_mask":         prompt_mask_y2,
                    "completion_ids":      completion_ids_y2,
                    "completion_mask":     completion_mask_y2,
                    "old_per_token_logps": old_logps_y2,
                    "ref_per_token_logps": ref_logps_y2,
                },
                "advantages": y2_advantages,
            }
        else:
            self._erl_rl_data = None

        # ── Phase 7: Store internalization pairs ─────────────────────────────
        self._internalization_pairs = []
        if self.args.enable_internalization:
            for j, i in enumerate(gated_indices):
                if y2_rewards[j].item() > 0:
                    self._internalization_pairs.append((prompts_raw[i], y2_texts[j]))

        return y1_result

    # ------------------------------------------------------------------
    # Phase 7: Internalization loss
    # ------------------------------------------------------------------

    def _compute_internalization_loss(self, model: nn.Module) -> torch.Tensor:
        """SFT cross-entropy on ``(original_prompt → y2)`` pairs.

        The prompt and the y2 completion are tokenised *separately* and then
        concatenated at the token level — tokenising ``prompt + y2`` as one
        string is unsafe because BPE/SentencePiece merges across the boundary
        can produce a different token at the join, silently shifting which
        positions are masked vs supervised.

        We do not divide by ``gradient_accumulation_steps`` here: HF Trainer's
        ``training_step`` performs that division on the value returned by
        ``compute_loss``. Pre-dividing produces a 1/N² scale.
        """
        pairs = self._internalization_pairs
        if not pairs:
            return torch.tensor(0.0, device=self.accelerator.device)

        device = self.accelerator.device
        tokenizer = self.processing_class
        pad_id: int = tokenizer.pad_token_id or 0

        all_input_ids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_attention_masks: list[torch.Tensor] = []

        for prompt, y2_text in pairs:
            # Keep the assistant generation header on the prompt side so the
            # boundary is well defined and y2 starts where the model would
            # actually produce its response at inference time.
            prompt_text = _prompt_to_text(
                prompt, tokenizer, add_generation_prompt=True
            )

            prompt_ids: torch.Tensor = tokenizer(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            y2_ids: torch.Tensor = tokenizer(
                y2_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]

            full_ids = torch.cat([prompt_ids, y2_ids], dim=0)
            labels = full_ids.clone()
            labels[: prompt_ids.shape[0]] = -100  # mask prompt tokens

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_attention_masks.append(torch.ones(full_ids.shape[0], dtype=torch.long))

        max_len = max(t.shape[0] for t in all_input_ids)

        batch_input_ids = torch.stack(
            [F.pad(t, (0, max_len - t.shape[0]), value=pad_id) for t in all_input_ids]
        ).to(device)
        batch_labels = torch.stack(
            [F.pad(t, (0, max_len - t.shape[0]), value=-100) for t in all_labels]
        ).to(device)
        batch_attention_mask = torch.stack(
            [F.pad(t, (0, max_len - t.shape[0]), value=0) for t in all_attention_masks]
        ).to(device)

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels,
        )
        return outputs.loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """y1 GRPO loss + optional ERL RL loss on Δ+y2 + optional internalization."""
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

        if return_outputs:
            loss, outputs = result
        else:
            loss = result

        erl_rl_loss = torch.tensor(0.0, device=loss.device)
        intern_loss = torch.tensor(0.0, device=loss.device)

        if self.args.erl_rl_coef > 0.0 and self._erl_rl_data is not None:
            delta_batch = {
                **self._erl_rl_data["delta"],
                "advantages": self._erl_rl_data["advantages"],
            }
            y2_batch = {
                **self._erl_rl_data["y2"],
                "advantages": self._erl_rl_data["advantages"],
            }
            erl_rl_loss = (
                self._erl_grpo_loss(model, delta_batch)
                + self._erl_grpo_loss(model, y2_batch)
            ) / 2.0
            loss = loss + self.args.erl_rl_coef * erl_rl_loss

        if self.args.enable_internalization and self._internalization_pairs:
            intern_loss = self._compute_internalization_loss(model)
            loss = loss + self.args.internalization_coef * intern_loss

        if self.args.erl_debug:
            logger.debug(
                f"[ERL Step {self.state.global_step}] Losses\n"
                f"  GRPO loss (y1): {(loss - self.args.erl_rl_coef * erl_rl_loss - self.args.internalization_coef * intern_loss).item():.4f}\n"
                f"  ERL RL loss (Δ+y2): {erl_rl_loss.item():.4f}\n"
                f"  Internalization loss: {intern_loss.item():.4f}\n"
                f"  Total loss: {loss.item():.4f}\n"
                f"  Internalization pairs: {len(self._internalization_pairs)}"
            )

        if return_outputs:
            return loss, outputs
        return loss
