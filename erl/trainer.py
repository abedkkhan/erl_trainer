from __future__ import annotations

import copy
import warnings
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
# ERLTrainer
# ---------------------------------------------------------------------------

class ERLTrainer(GRPOTrainer):
    """Experiential Reinforcement Learning trainer for TRL 0.17.0+.

    Extends ``GRPOTrainer`` by overriding ``_generate_and_score_completions``
    to inject the full ERL reflection-retry-internalization loop after the
    first attempt.  The returned dict has the same keys as the parent's method,
    so ``_compute_loss`` works without modification.

    This implementation follows **Algorithm 2** from the ERL paper:

    * **y1 RL update** — delegated to the parent's GRPO loss on first-attempt
      completions.  Advantages are derived from r1 (y1 rewards) only (Algorithm
      1 style, consistent with the parent's batched normalisation).
    * **Δ + y2 RL update** — a separate GRPO loss computed in ``compute_loss``
      using the stored ``_erl_rl_data``.  Both Δ (reflections) and y2
      (second-attempt completions) receive reward r2; advantages are
      batch-wide-normalised (since each gated sample contributes exactly one Δ
      and one y2, making per-group normalisation degenerate).
    * **Internalization** — SFT cross-entropy on ``(original_prompt → y2)``
      for successful second attempts (r2 > 0), training the model to produce
      improved answers at inference time without any reflection scaffold.

    Args:
        model: Model to train (same as GRPOTrainer).
        reward_funcs: Reward function(s) (same as GRPOTrainer).
        args: ERLConfig with ERL-specific hyperparameters.
        feedback_func: Callable with the same signature as a reward function
            that returns a list of textual feedback strings.  May be ``None``;
            empty strings are used as feedback in that case.
        **kwargs: Forwarded verbatim to ``GRPOTrainer.__init__``.
    """

    def __init__(
        self,
        model: str | nn.Module,
        reward_funcs: Callable | list[Callable],
        args: ERLConfig | None = None,
        feedback_func: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        if args is None:
            args = ERLConfig(output_dir="erl-output")
        super().__init__(model=model, reward_funcs=reward_funcs, args=args, **kwargs)
        self.feedback_func = feedback_func
        self.memory: ReflectionMemory | None = (
            ReflectionMemory(args.memory_size) if args.enable_memory else None
        )
        # Populated each step by _generate_and_score_completions; consumed by
        # compute_loss for the internalization (distillation) loss.
        self._internalization_pairs: list[tuple[Any, str]] = []
        # Populated each step by _generate_and_score_completions; consumed by
        # compute_loss for the ERL RL loss on Δ and y2.  None when no samples
        # were gated or when erl_rl_coef == 0.
        self._erl_rl_data: dict | None = None

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

        Formats each prompt as a single-turn user message, applies the model's
        chat template, tokenises with left-padding, generates using the same
        ``generation_config`` as the parent, and returns IDs and decoded text.

        Used for both reflection (Phase 3) and second-attempt (Phase 4)
        generation.  Both phases are one batched ``model.generate`` call each,
        keeping GPU utilisation high regardless of how many samples are gated.

        Args:
            prompt_texts: Plain-text content for each prompt.
            compute_logps: When ``True``, also compute per-token log-probs from
                the current policy and reference model (if available).  These
                are used as the "old" log-probs for the ERL GRPO loss.

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

        # Wrap as single-turn chat inputs so apply_chat_template adds the
        # proper generation-prompt suffix (e.g. <|im_start|>assistant\n).
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
        old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
            self.model, prompt_completion_ids, full_mask, logits_to_keep=C
        )  # (B, C)

        # ── Reference model log-probs (for KL penalty when beta != 0) ─────────
        ref_model = getattr(self, "ref_model", None)
        beta = getattr(self, "beta", 0.0)
        if ref_model is not None and beta != 0.0:
            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                ref_model, prompt_completion_ids, full_mask, logits_to_keep=C
            )  # (B, C)
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
    # Reward and feedback helpers
    # ------------------------------------------------------------------

    def _erl_compute_rewards(
        self,
        inputs: list[dict],
        prompts: list,
        completions: list,
    ) -> torch.Tensor:
        """Compute scalar rewards for a (sub)batch using ``self.reward_funcs``.

        Mirrors the inline reward computation in TRL's
        ``_generate_and_score_completions`` so ERL can evaluate y2 completions
        against the original task's reward functions.

        Args:
            inputs: Dataset dicts for this (sub)batch.
            prompts: Prompts in the format expected by reward functions
                (string or list of message dicts).
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
                # Model-based reward function
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
                # Callable reward function
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

    def _erl_compute_feedback(
        self,
        inputs: list[dict],
        prompts: list,
        completions: list,
    ) -> list[str]:
        """Call ``self.feedback_func`` and return one feedback string per sample.

        Args:
            inputs: Dataset dicts for the current batch.
            prompts: Passed through to ``feedback_func``.
            completions: Passed through to ``feedback_func``.

        Returns:
            List of feedback strings.  Empty strings when ``feedback_func``
            is ``None``.
        """
        if self.feedback_func is None:
            return [""] * len(prompts)
        keys = [k for k in inputs[0] if k not in ("prompt", "completion")]
        feedback_kwargs = {k: [ex[k] for ex in inputs] for k in keys}
        return self.feedback_func(
            prompts=prompts, completions=completions, **feedback_kwargs
        )

    # ------------------------------------------------------------------
    # Advantage normalisation for Δ and y2
    # ------------------------------------------------------------------

    @staticmethod
    def _erl_compute_r2_advantages(rewards: torch.Tensor) -> torch.Tensor:
        """Batch-wide advantage normalisation for Δ and y2 rewards.

        Unlike y1 which uses group-wise normalisation (GRPO), Δ and y2 use
        batch-wide normalisation because each gated sample produces exactly one
        Δ and one y2 (group size 1), making per-group std identically zero.

        Args:
            rewards: 1-D tensor of r2 rewards for all gated samples, shape
                ``(N,)``.

        Returns:
            Normalised advantages, same shape.  Returns all-zeros when
            ``N <= 1`` (single sample → no meaningful normalisation possible).
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

        Replicates the parent's ``_compute_loss`` formula without the metrics
        logging, so this can be called twice per step (once for Δ, once for
        y2) without double-counting metrics.

        Args:
            model: Current training model (Accelerate-wrapped).
            batch: Dict with keys ``prompt_ids``, ``prompt_mask``,
                ``completion_ids``, ``completion_mask``,
                ``old_per_token_logps``, ``ref_per_token_logps`` (may be
                ``None``), ``advantages``.

        Returns:
            Scalar loss tensor with gradients attached.
        """
        prompt_ids  = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        completion_ids  = batch["completion_ids"]
        completion_mask = batch["completion_mask"].float()      # (N, C)
        old_per_token_logps = batch["old_per_token_logps"]     # (N, C)
        ref_per_token_logps = batch.get("ref_per_token_logps") # (N, C) or None
        advantages  = batch["advantages"]                       # (N,)

        C = completion_ids.shape[1]
        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask.long()], dim=1)

        # Current-policy log-probs (with gradient)
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep=C
        )  # (N, C)

        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)  # (N, 1) for broadcasting

        # Old log-probs fallback (shouldn't be None in normal ERL flow)
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        # Importance-sampling ratio
        log_ratio = per_token_logps - old_per_token_logps
        if getattr(self, "importance_sampling_level", "token") == "token":
            log_iw = log_ratio
        else:  # sequence
            log_iw = (
                (log_ratio * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).unsqueeze(-1)

        coef_1 = torch.exp(log_iw)

        # KL divergence penalty (mirrors parent when beta != 0)
        beta = getattr(self, "beta", 0.0)
        per_token_kl = None
        if beta != 0.0 and ref_per_token_logps is not None:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Per-token loss (PPO clip formula)
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
            # Unknown loss type — fall back to grpo formula
            coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
            per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)

        if per_token_kl is not None:
            per_token_loss = per_token_loss + beta * per_token_kl

        mask = completion_mask
        mode = "train" if model.training else "eval"
        normalizer = (
            self.current_gradient_accumulation_steps if mode == "train" else 1.0
        )

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
            # Fallback to grpo formula
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
        """Run the full ERL loop and return a batch dict for ``_compute_loss``.

        Delegates the first attempt entirely to the parent's
        ``_generate_and_score_completions``, which handles tokenisation,
        generation, EOS masking, log-probability computation, reward
        evaluation, advantage normalisation, and metrics logging.  ERL phases
        2–7 are then applied on top of the parent's output.

        The returned dict has the same keys as the parent's method
        (``prompt_ids``, ``prompt_mask``, ``completion_ids``,
        ``completion_mask``, ``advantages``, ``old_per_token_logps``,
        ``ref_per_token_logps``) so ``_compute_loss`` works unmodified.
        """
        inputs = _to_list_of_dicts(inputs)

        # Deep-copy so we preserve the original prompts before the parent can
        # mutate them (for conversational datasets the parent pops the last
        # assistant turn as a "bootstrap" prefix for the completion).
        inputs_orig = copy.deepcopy(inputs)
        prompts_raw = [x["prompt"] for x in inputs_orig]

        # ── Phase 1: First attempt (full parent behaviour) ─────────────────
        y1_result = super()._generate_and_score_completions(inputs)

        y1_texts = self.processing_class.batch_decode(
            y1_result["completion_ids"], skip_special_tokens=True
        )

        # Format completions for reward/feedback functions
        if is_conversational(inputs_orig[0]):
            completions_y1 = [[{"role": "assistant", "content": t}] for t in y1_texts]
        else:
            completions_y1 = y1_texts

        # Re-compute raw rewards for gating.  The parent only returns
        # advantages (normalized rewards), not the raw per-sample rewards.
        y1_rewards = self._erl_compute_rewards(inputs_orig, prompts_raw, completions_y1)
        y1_feedback = self._erl_compute_feedback(inputs_orig, prompts_raw, completions_y1)

        # ── Phase 2: Gating ─────────────────────────────────────────────────
        gated_mask = (y1_rewards < self.args.reward_threshold).tolist()
        gated_indices = [i for i, g in enumerate(gated_mask) if g]

        if not gated_indices:
            self._internalization_pairs = []
            self._erl_rl_data = None
            return y1_result

        # Whether to compute log-probs for the ERL RL loss on Δ + y2
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

        # ── Phase 5: Memory update ───────────────────────────────────────────
        if self.args.enable_memory and self.memory is not None:
            for j, i in enumerate(gated_indices):
                if y2_rewards[j].item() > self.args.reward_threshold:
                    prompt_str = _prompt_to_text(prompts_raw[i], self.processing_class)
                    self.memory.add(
                        reflection_texts[j], prompt_str, float(y2_rewards[j].item())
                    )

        # ── Phase 6: Re-normalise advantages with y1 rewards ─────────────────
        # completion_ids in y1_result are y1 TOKENS, so advantages must be
        # derived from r1 (y1 rewards).  Replacing with r2 here would credit
        # y1 tokens for the quality of an entirely different generation (y2),
        # which contradicts both Algorithm 1 and Algorithm 2 of the paper.
        combined_rewards = y1_rewards.clone()

        # Group-wise normalisation (same formula as parent).
        mean_grouped = combined_rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped = combined_rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped = mean_grouped.repeat_interleave(self.num_generations, dim=0)
        std_grouped = std_grouped.repeat_interleave(self.num_generations, dim=0)
        new_advantages = combined_rewards - mean_grouped
        if getattr(self, "scale_rewards", True):
            new_advantages = new_advantages / (std_grouped + 1e-4)

        y1_result["advantages"] = new_advantages

        # ── Phase 6b: Store Δ + y2 RL data for compute_loss ─────────────────
        # Advantages are batch-wide-normalised from r2 rewards because each
        # gated sample produces exactly one Δ and one y2 (group size 1 makes
        # per-group normalisation degenerate with std=0).
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

        Trains the model to produce the improved second-attempt answer
        directly from the original prompt without any reflection scaffold, so
        improvements transfer to deployment-time inference.

        Args:
            model: Current training model (may be Accelerate-wrapped).

        Returns:
            Scalar cross-entropy loss tensor.
        """
        pairs = self._internalization_pairs
        if not pairs:
            return torch.tensor(
                0.0, device=self.accelerator.device, requires_grad=True
            )

        device = self.accelerator.device
        tokenizer = self.processing_class
        pad_id: int = tokenizer.pad_token_id or 0

        all_input_ids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_attention_masks: list[torch.Tensor] = []

        for prompt, y2_text in pairs:
            prompt_text = _prompt_to_text(prompt, tokenizer)

            # Tokenize the full string as one piece so BPE merges at the
            # prompt/completion boundary are computed correctly.  Separate
            # tokenization can produce a different (invalid) token sequence
            # because context-dependent merges at the junction are missed.
            full_ids: torch.Tensor = tokenizer(
                prompt_text + y2_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]

            # Use prompt-only length to locate the label mask boundary.
            # Off-by-one at the boundary is at most 1-2 tokens and is
            # inconsequential — the input_ids correctness is the critical fix.
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
        return outputs.loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """y1 GRPO loss + optional ERL RL loss on Δ+y2 + optional internalization.

        Args:
            model: Training model passed by the Trainer loop.
            inputs: Batch dict produced by ``_generate_and_score_completions``.
            return_outputs: Whether to return model outputs alongside the loss.
            num_items_in_batch: Passed through to the parent's loss normalisation.

        Returns:
            Combined loss scalar, or ``(loss, outputs)`` when
            ``return_outputs=True``.
        """
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

        # ERL RL loss: GRPO on Δ (reflections) and y2 (second attempts)
        if self.args.erl_rl_coef > 0.0 and self._erl_rl_data is not None:
            delta_batch = {
                **self._erl_rl_data["delta"],
                "advantages": self._erl_rl_data["advantages"],
            }
            y2_batch = {
                **self._erl_rl_data["y2"],
                "advantages": self._erl_rl_data["advantages"],
            }
            erl_rl = (
                self._erl_grpo_loss(model, delta_batch)
                + self._erl_grpo_loss(model, y2_batch)
            ) / 2.0
            loss = loss + self.args.erl_rl_coef * erl_rl

        # Internalization loss: SFT on (original_prompt → y2)
        if self.args.enable_internalization and self._internalization_pairs:
            loss = loss + self.args.internalization_coef * self._compute_internalization_loss(model)

        if return_outputs:
            return loss, outputs
        return loss
