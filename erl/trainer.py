from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from trl import GRPOTrainer

from erl.config import ERLConfig
from erl.memory import ReflectionMemory


class ERLTrainer(GRPOTrainer):
    """Experiential Reinforcement Learning trainer extending TRL's GRPOTrainer.

    Implements the ERL algorithm which adds an experience-reflection-consolidation loop
    on top of standard GRPO. The loop consists of:

    1. First attempt (y1) with reward r1 and textual feedback f1.
    2. Gating: samples where r1 < reward_threshold enter the reflection loop.
    3. Self-reflection: the model reflects on its failure using f1 and memory.
    4. Second attempt (y2) guided by the reflection, with reward r2.
    5. Memory: successful reflections (r2 > threshold) are stored for future use.
    6. RL update (GRPO) on the combined batch: y1, reflections, and y2.
    7. Internalization: SFT on (prompt → y2) for successful second attempts (r2 > 0).

    Args:
        model: Model to train (same as GRPOTrainer).
        reward_funcs: Reward function(s) (same as GRPOTrainer).
        args: ERLConfig instance with ERL-specific hyperparameters.
        feedback_func: Callable that takes completions (and optionally prompts) and
            returns a list of textual feedback strings. Signature mirrors reward_funcs:
            ``feedback_func(prompts=..., completions=..., **kwargs) -> list[str]``.
        **kwargs: All remaining keyword arguments are forwarded to GRPOTrainer.
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
        self._erl_internalization_data: list[tuple[torch.Tensor, torch.Tensor]] | None = None

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
    # Single-completion generation for ERL reflection / retry phases
    # ------------------------------------------------------------------

    def _generate_erl_completion(
        self, prompt_text: str
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Generate exactly one completion for ERL reflection or retry phases.

        Uses the model's chat template to format the prompt and calls ``model.generate``
        with ``torch.no_grad()`` using the same temperature / length settings as the
        main GRPO generation.

        Args:
            prompt_text: Plain-text content for the user turn.

        Returns:
            Tuple of (prompt_ids, completion_ids, completion_text), all on CPU.
        """
        messages = [{"role": "user", "content": prompt_text}]
        tokenizer = self.processing_class

        input_ids: torch.Tensor = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.accelerator.device)

        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        unwrapped = self.accelerator.unwrap_model(self.model)
        was_training = unwrapped.training
        try:
            unwrapped.eval()
            with torch.no_grad():
                output_ids = unwrapped.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.args.max_completion_length,
                    temperature=max(self.args.temperature, 1e-2),
                    do_sample=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            if was_training:
                unwrapped.train()

        completion_ids = output_ids[0, prompt_len:].cpu()
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        return input_ids[0].cpu(), completion_ids, completion_text

    # ------------------------------------------------------------------
    # Batch packing: token lists → padded tensors + advantages + logprobs
    # ------------------------------------------------------------------

    def _pack_batch(
        self,
        prompt_ids_list: list[torch.Tensor],
        completion_ids_list: list[torch.Tensor],
        rewards: list[float],
    ) -> dict[str, torch.Tensor | int]:
        """Pack variable-length sequence tensors into a padded batch dict.

        Pads prompts on the left and completions on the right, computes global
        advantage normalisation over all sequences in the batch (appropriate for
        ERL's mixed batch of y1, reflections, and y2), and computes per-token
        log-probabilities under the current model weights (needed for PPO clipping).

        Args:
            prompt_ids_list: List of 1-D prompt token ID tensors (CPU).
            completion_ids_list: List of 1-D completion token ID tensors (CPU).
            rewards: Scalar reward for each sequence.

        Returns:
            Dict with keys expected by GRPOTrainer's ``_compute_loss``.
        """
        device = self.accelerator.device
        pad_id: int = self.processing_class.pad_token_id or 0

        max_prompt_len = max(p.shape[0] for p in prompt_ids_list)
        max_completion_len = max(c.shape[0] for c in completion_ids_list)

        prompt_ids = torch.stack(
            [F.pad(p, (max_prompt_len - p.shape[0], 0), value=pad_id) for p in prompt_ids_list]
        ).to(device)
        prompt_mask = (prompt_ids != pad_id).long()

        completion_ids = torch.stack(
            [
                F.pad(c, (0, max_completion_len - c.shape[0]), value=pad_id)
                for c in completion_ids_list
            ]
        ).to(device)
        completion_mask = torch.stack(
            [
                F.pad(
                    torch.ones(c.shape[0], dtype=torch.long),
                    (0, max_completion_len - c.shape[0]),
                    value=0,
                )
                for c in completion_ids_list
            ]
        ).to(device)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        std = rewards_t.std()
        advantages = (rewards_t - rewards_t.mean()) / (std + 1e-4)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        with torch.no_grad():
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                input_ids,
                attention_mask,
                max_completion_len,
            )

        result: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "num_items_in_batch": int(completion_mask.sum().item()),
        }

        if self.ref_model is not None:
            with torch.no_grad():
                ref_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    input_ids,
                    attention_mask,
                    max_completion_len,
                )
            result["ref_per_token_logps"] = ref_logps

        return result

    # ------------------------------------------------------------------
    # Core ERL training loop (overrides _prepare_inputs / generation)
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self, generation_batch: dict[str, Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Override to run ERL generation every step without buffering.

        ERL's combined batch size varies per step (depends on how many samples are
        gated), making TRL's fixed-size buffer slicing incompatible. Regenerating
        each step is simpler and ensures correctness.
        """
        return self._generate_and_score_completions(generation_batch)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """Run the full ERL loop for one training batch.

        Phases 1-7 of the ERL algorithm are executed here. The returned dict is
        consumed directly by GRPOTrainer's ``_compute_loss``.

        Args:
            inputs: List of dataset row dicts, each with at least a ``"prompt"`` key.

        Returns:
            Batch dict with padded tensors, advantages, and logprobs for all
            generated sequences (y1, reflections, y2).
        """
        prompts = [x["prompt"] for x in inputs]

        # ── Phase 1: First attempt ────────────────────────────────────────
        (
            prompt_ids_y1,
            completion_ids_y1,
            _tool_mask,
            completions_y1,
            _num_items,
            _logps,
            _extra,
        ) = self._generate(prompts)

        rewards_y1_per_func = self._calculate_rewards(
            inputs, prompts, completions_y1, completion_ids_y1
        )
        r1: torch.Tensor = (
            rewards_y1_per_func * self.reward_weights.to(rewards_y1_per_func.device)
        ).nansum(dim=1)

        if self.feedback_func is not None:
            f1: list[str] = self.feedback_func(prompts=prompts, completions=completions_y1)
        else:
            f1 = [""] * len(completions_y1)

        # ── Phase 2: Gating ───────────────────────────────────────────────
        gated_mask: list[bool] = (r1 < self.args.reward_threshold).tolist()

        all_prompt_ids: list[torch.Tensor] = [
            p.cpu() if isinstance(p, torch.Tensor) else torch.tensor(p)
            for p in prompt_ids_y1
        ]
        all_completion_ids: list[torch.Tensor] = [
            c.cpu() if isinstance(c, torch.Tensor) else torch.tensor(c)
            for c in completion_ids_y1
        ]
        all_rewards: list[float] = r1.cpu().tolist()

        internalization_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        # ── Phases 3-5: Reflection, retry, and memory (gated samples) ────
        for i, is_gated in enumerate(gated_mask):
            if not is_gated:
                continue

            prompt_idx = i // self.num_generations
            original_prompt = str(prompts[prompt_idx])

            memory_entries: list[str] = []
            if self.args.enable_memory and self.memory is not None:
                memory_entries = self.memory.retrieve(original_prompt, self.args.memory_top_k)

            reflection_content = self._build_reflection_prompt(
                prompt=original_prompt,
                attempt=completions_y1[i],
                feedback=f1[i],
                reward=float(r1[i].item()),
                memory_entries=memory_entries,
            )

            rfl_prompt_ids, delta_ids, delta_text = self._generate_erl_completion(
                reflection_content
            )

            retry_content = self._build_retry_prompt(
                prompt=original_prompt, reflection=delta_text
            )

            retry_prompt_ids, y2_ids, y2_text = self._generate_erl_completion(retry_content)

            rewards_y2_per_func = self._calculate_rewards(
                [inputs[prompt_idx]],
                [prompts[prompt_idx]],
                [y2_text],
                [y2_ids.tolist()],
            )
            r2: float = (
                rewards_y2_per_func * self.reward_weights.to(rewards_y2_per_func.device)
            ).nansum(dim=1).item()

            if self.args.enable_memory and self.memory is not None and r2 > self.args.reward_threshold:
                self.memory.add(delta_text, original_prompt, r2)

            all_prompt_ids.append(rfl_prompt_ids)
            all_completion_ids.append(delta_ids)
            all_rewards.append(r2)

            all_prompt_ids.append(retry_prompt_ids)
            all_completion_ids.append(y2_ids)
            all_rewards.append(r2)

            if r2 > 0 and self.args.enable_internalization:
                original_prompt_ids = all_prompt_ids[i]
                internalization_pairs.append((original_prompt_ids, y2_ids))

        self._erl_internalization_data = internalization_pairs or None

        # ── Phase 6: Pack combined batch for GRPO update ─────────────────
        return self._pack_batch(all_prompt_ids, all_completion_ids, all_rewards)

    # ------------------------------------------------------------------
    # Phase 7: Internalization loss
    # ------------------------------------------------------------------

    def _compute_internalization_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute cross-entropy SFT loss for internalizing y2 without reflection context.

        Trains the model to predict successful second-attempt completions directly from
        the original prompt, so the improved behaviour persists at inference time when
        no reflection scaffold is present.

        Args:
            model: The current training model (may be wrapped by Accelerate).

        Returns:
            Scalar cross-entropy loss tensor.
        """
        pairs = self._erl_internalization_data
        if not pairs:
            return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)

        device = self.accelerator.device
        pad_id: int = self.processing_class.pad_token_id or 0

        all_input_ids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_attention_masks: list[torch.Tensor] = []

        for prompt_ids, y2_ids in pairs:
            p = prompt_ids.tolist()
            c = y2_ids.tolist()
            seq_len = len(p) + len(c)

            all_input_ids.append(torch.tensor(p + c, dtype=torch.long))
            all_labels.append(torch.tensor([-100] * len(p) + c, dtype=torch.long))
            all_attention_masks.append(torch.ones(seq_len, dtype=torch.long))

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
        """Compute GRPO loss plus optional internalization loss.

        The GRPO loss is computed by the parent over the full combined batch
        (y1, reflections, y2 with their respective rewards / advantages). The
        internalization loss is cross-entropy on successful (prompt → y2) pairs.

        Args:
            model: Training model passed by the Trainer loop.
            inputs: Batch dict produced by ``_prepare_inputs``.
            return_outputs: Whether to return model outputs alongside the loss.
            num_items_in_batch: Passed through to the parent's loss normalisation.

        Returns:
            Combined loss scalar, or ``(loss, outputs)`` when ``return_outputs=True``.
        """
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

        if not (self.args.enable_internalization and self._erl_internalization_data):
            return result

        distill_loss = self._compute_internalization_loss(model)

        if return_outputs:
            loss, outputs = result
            return loss + self.args.internalization_coef * distill_loss, outputs

        return result + self.args.internalization_coef * distill_loss
