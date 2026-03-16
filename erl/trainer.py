from __future__ import annotations

import copy
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
# TRL 0.17.0 data utilities — import with fallbacks so unit tests that run
# without a full TRL install still work.
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


def _prompt_to_text(prompt: Any, tokenizer: Any) -> str:
    """Convert a prompt (string or list of message dicts) to plain text."""
    if isinstance(prompt, str):
        return prompt
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )


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
        # Preserve name so TRL's reward_func_names lookup still works
        if hasattr(func, "__name__"):
            self.__name__ = func.__name__

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

    def _build_reflection_prompt(
        self,
        prompt: str,
        attempt: str,
        feedback: str,
        reward: float,
        memory_entries: list[str],
    ) -> str:
        """Format the reflection prompt using the configured template."""
        memory_str = "\n\n".join(memory_entries) if memory_entries else "None available."
        return self.args.reflection_system_prompt.format(
            prompt=prompt,
            attempt=attempt,
            feedback=feedback,
            reward=reward,
            memory=memory_str,
        )

    def _build_retry_prompt(self, prompt: str, reflection: str) -> str:
        """Format the retry prompt using the configured template."""
        return self.args.retry_system_prompt.format(
            prompt=prompt,
            reflection=reflection,
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
        """Thin wrapper around TRL 0.22.x's per-token log-prob computation."""
        return self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep=logits_to_keep
        )

    # ------------------------------------------------------------------
    # Internal generation helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _erl_generate(
        self,
        prompt_texts: list[str],
        *,
        compute_logps: bool = False,
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

        chat_inputs = [
            {"prompt": [{"role": "user", "content": t}]} for t in prompt_texts
        ]
        formatted_texts = [
            maybe_apply_chat_template(ex, tokenizer)["prompt"] for ex in chat_inputs
        ]

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

        gather_ds3 = getattr(self.args, "ds3_gather_for_generation", False)
        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=gather_ds3,
        ) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
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
        mode = "train" if model.training else "eval"
        _accum = (
            getattr(self, "current_gradient_accumulation_steps", None)
            or self.args.gradient_accumulation_steps
        )
        normalizer = _accum if mode == "train" else 1.0

        if loss_type in ("grpo", "sapo", "dapo"):
            loss = (
                (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            ).mean() / normalizer
        elif loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0) / normalizer
        elif loss_type == "dr_grpo":
            loss = (
                (per_token_loss * mask).sum()
                / (per_token_loss.size(0) * self.max_completion_length)
                / normalizer
            )
        elif loss_type == "luspo":
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean() / normalizer
        else:
            loss = (
                (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            ).mean() / normalizer

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
        ) = self._erl_generate(reflection_prompts, compute_logps=compute_logps)

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
        if self.args.enable_memory and self.memory is not None:
            for j, i in enumerate(gated_indices):
                if y2_rewards[j].item() > self.args.reward_threshold:
                    prompt_str = _prompt_to_text(prompts_raw[i], self.processing_class)
                    self.memory.add(
                        reflection_texts[j], prompt_str, float(y2_rewards[j].item())
                    )

        if _debug:
            mem_added = sum(
                1 for j in range(len(gated_indices))
                if y2_rewards[j].item() > self.args.reward_threshold
            )
            mem_size = len(self.memory) if self.memory is not None else 0
            logger.debug(
                f"[ERL Step {_step}] Phase 5 — Memory Update\n"
                f"  New entries added: {mem_added}\n"
                f"  Total memory size: {mem_size}"
            )

        # ── Phase 6: Re-normalise advantages with y1 rewards ─────────────────
        combined_rewards = y1_rewards.clone()
        mean_grouped = combined_rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped = combined_rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped = mean_grouped.repeat_interleave(self.num_generations, dim=0)
        std_grouped = std_grouped.repeat_interleave(self.num_generations, dim=0)
        new_advantages = combined_rewards - mean_grouped
        _scale = getattr(self, "scale_rewards", "group")
        if _scale is True or (isinstance(_scale, str) and _scale != "none"):
            new_advantages = new_advantages / (std_grouped + 1e-4)
        y1_result["advantages"] = new_advantages

        if _debug:
            adv_sample = [round(a, 3) for a in new_advantages[:4].tolist()]
            logger.debug(
                f"[ERL Step {_step}] Phase 6 — Advantages\n"
                f"  Y1 advantages (first 4): {adv_sample}"
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
        """SFT cross-entropy on ``(original_prompt → y2)`` pairs, normalised
        by gradient accumulation steps to match the GRPO loss scale."""
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
            prompt_text = _prompt_to_text(prompt, tokenizer)

            full_ids: torch.Tensor = tokenizer(
                prompt_text + y2_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]

            prompt_only_len: int = tokenizer(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].shape[1]

            labels = full_ids.clone()
            labels[:prompt_only_len] = -100

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
        loss = outputs.loss
        if model.training:
            _accum = (
                getattr(self, "current_gradient_accumulation_steps", None)
                or self.args.gradient_accumulation_steps
            )
            loss = loss / _accum
        return loss

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
