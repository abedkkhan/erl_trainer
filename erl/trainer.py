from __future__ import annotations

import warnings
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from trl import GRPOTrainer

from erl.config import ERLConfig
from erl.memory import ReflectionMemory


# Default key names used by GRPOTrainer's _compute_loss.  _discover_batch_keys
# pattern-matches against a reference dict to confirm (or correct) these at runtime.
_BATCH_KEY_DEFAULTS: dict[str, str] = {
    "prompt_ids": "prompt_ids",
    "prompt_mask": "prompt_mask",
    "completion_ids": "completion_ids",
    "completion_mask": "completion_mask",
}


def _to_list_of_dicts(inputs: list[dict] | dict[str, Any]) -> list[dict]:
    """Normalise the two formats TRL may pass to _generate_and_score_completions.

    TRL's DataLoader can deliver either a dict-of-lists (standard PyTorch collate)
    or a list-of-dicts (TRL's own collator in some versions).  Both are converted
    to a canonical list-of-dicts before any processing.
    """
    if isinstance(inputs, list):
        return inputs
    keys = list(inputs.keys())
    if not keys:
        return []
    n = len(inputs[keys[0]])
    return [{k: inputs[k][i] for k in keys} for i in range(n)]


def _extract_logps(result: Any) -> torch.Tensor:
    """Safely extract per-token log-probabilities from _get_per_token_logps_and_entropies.

    Different TRL versions return either a 2-tuple ``(logps, entropies)`` or a
    dict with a ``"logps"`` / ``"per_token_logps"`` key.  Both are handled here.
    """
    if isinstance(result, tuple):
        return result[0]
    if isinstance(result, dict):
        for key in ("logps", "per_token_logps"):
            if key in result:
                return result[key]
        raise KeyError(f"Cannot find logps in result dict with keys: {list(result.keys())}")
    return result


class ERLTrainer(GRPOTrainer):
    """Experiential Reinforcement Learning trainer extending TRL's GRPOTrainer.

    Implements Algorithm 1 (simplified combined update) from the ERL paper.
    All three output types — first attempt (y1), reflection (Δ), and second
    attempt (y2) — are combined into one batch and updated jointly.

    Note on Algorithm 1 vs Algorithm 2:
        The paper's Appendix A describes Algorithm 2 (full), which separates the
        RL update into two steps: one on y1 alone (line 9), then one on Δ + y2
        together (line 21).  Algorithm 1 (implemented here) combines all three
        into a single GRPO update.  The practical difference is in advantage
        normalisation: per-group normalisation over separate batches vs. global
        normalisation over the mixed batch.  The directional gradients are the
        same; only the magnitude scaling differs.  Algorithm 2 can be implemented
        by splitting the batch and calling super().compute_loss twice.

    Args:
        model: Model to train (same as GRPOTrainer).
        reward_funcs: Reward function(s) (same as GRPOTrainer).
        args: ERLConfig instance with ERL-specific hyperparameters.
        feedback_func: Callable that receives the same kwargs as a reward function
            and returns a list of textual feedback strings — one per completion.
            Signature: ``feedback_func(prompts=..., completions=..., **kwargs) -> list[str]``.
            May be ``None``; in that case empty strings are used as feedback (the
            reflection prompt will have blank feedback, which degrades quality but
            does not break training).
        **kwargs: Forwarded verbatim to GRPOTrainer.__init__.
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
        self._batch_keys: dict[str, str] | None = None

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
    # _generate return-value unpacking (Issue 1 fix)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_token_id_batch(item: Any) -> bool:
        """Return True if item looks like a batch of integer token ID sequences.

        Matches both a 2-D integer ``torch.Tensor`` (future TRL) and a list of
        1-D integer tensors (current TRL), since ``_generate`` has historically
        returned the latter.
        """
        int_dtypes = (torch.long, torch.int, torch.int16, torch.int32, torch.int64)
        if isinstance(item, torch.Tensor):
            return item.dim() == 2 and item.dtype in int_dtypes
        if isinstance(item, list) and item:
            first = item[0]
            if isinstance(first, torch.Tensor):
                return first.dtype in int_dtypes
            if isinstance(first, (list, tuple)) and first:
                return isinstance(first[0], int)
        return False

    @staticmethod
    def _is_string_list(item: Any) -> bool:
        """Return True if item is a non-empty list of strings or list of lists of strings."""
        if not isinstance(item, list) or not item:
            return False
        first = item[0]
        if isinstance(first, str):
            return True
        # nested: list[list[str]] — TRL may return per-group completions
        return isinstance(first, list) and bool(first) and isinstance(first[0], str)

    @staticmethod
    def _unpack_generate(result: Any) -> tuple[list, list, list[str]]:
        """Extract prompt IDs, completion IDs, and completion texts from ``_generate``.

        Uses type-based detection instead of positional indexing so the method
        remains correct if TRL adds, removes, or reorders return values, or
        changes from returning a plain tuple to a namedtuple or dataclass.

        Detection rules:
        - Token ID batches: the first two items that are either a 2-D integer
          ``torch.Tensor`` or a list of 1-D integer tensors.  Prompt IDs always
          precede completion IDs in TRL's return order.
        - Completion texts: the first item that is a non-empty list of strings.
          If the list is nested (list of list of str) it is flattened one level.

        Args:
            result: Raw return value of ``self._generate(prompts)``.

        Returns:
            Tuple of ``(prompt_ids, completion_ids, completion_texts)``.

        Raises:
            ValueError: If fewer than two token ID batches or no string list are
                found, with a message showing the types actually present.
        """
        try:
            items: list[Any] = list(result)
        except TypeError:
            items = [result]

        token_id_batches = [item for item in items if ERLTrainer._is_token_id_batch(item)]
        string_lists = [item for item in items if ERLTrainer._is_string_list(item)]

        type_names = [type(x).__name__ for x in items]

        if len(token_id_batches) < 2:
            raise ValueError(
                f"_generate returned fewer than 2 token ID batches. "
                f"Expected prompt_ids and completion_ids as the first two "
                f"integer tensor batches. "
                f"Found {len(token_id_batches)} token ID batch(es) among "
                f"{len(items)} items with types: {type_names}"
            )

        if not string_lists:
            raise ValueError(
                f"_generate returned no list of strings. "
                f"Expected completion texts as a list[str] in the return value. "
                f"Found items with types: {type_names}"
            )

        prompt_ids = token_id_batches[0]
        completion_ids = token_id_batches[1]
        completion_texts: list[str] = string_lists[0]

        if completion_texts and isinstance(completion_texts[0], list):
            completion_texts = [s for sub in completion_texts for s in sub]

        return prompt_ids, completion_ids, completion_texts

    # ------------------------------------------------------------------
    # Batch key discovery (Issue 2 fix)
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_batch_keys(reference_batch: dict[str, Any]) -> dict[str, str]:
        """Discover TRL's actual batch key names by pattern-matching a reference dict.

        Maps from the four canonical ERL key concepts to whatever key names
        appear in ``reference_batch``.  This insulates ``_pack_batch`` from TRL
        renaming internal dict keys (e.g. ``prompt_ids`` → ``prompt_input_ids``).

        Pattern rules (case-insensitive substring matching):
        - ``prompt_ids``    : contains ``"prompt"`` and ``"id"`` but not ``"mask"``
        - ``prompt_mask``   : contains ``"prompt"`` and ``"mask"``
        - ``completion_ids``: contains ``"completion"`` and ``"id"`` but not ``"mask"``
        - ``completion_mask``: contains ``"completion"`` and ``"mask"``

        Args:
            reference_batch: A dict produced by TRL's own
                ``_generate_and_score_completions`` (or our ``_pack_batch``).

        Returns:
            Mapping from canonical name → actual key name.  Falls back to
            ``_BATCH_KEY_DEFAULTS`` with a warning if any of the four keys
            cannot be identified with confidence.
        """
        found: dict[str, str] = {}
        for key in reference_batch:
            k = key.lower()
            if "prompt" in k and "mask" not in k and "id" in k:
                found.setdefault("prompt_ids", key)
            elif "prompt" in k and "mask" in k:
                found.setdefault("prompt_mask", key)
            elif "completion" in k and "mask" not in k and "id" in k:
                found.setdefault("completion_ids", key)
            elif "completion" in k and "mask" in k:
                found.setdefault("completion_mask", key)

        if len(found) < 4:
            missing = [k for k in _BATCH_KEY_DEFAULTS if k not in found]
            warnings.warn(
                f"Could not auto-discover TRL batch keys for: {missing}. "
                f"Using defaults. If training crashes, check TRL version compatibility. "
                f"Reference dict keys were: {list(reference_batch.keys())}",
                stacklevel=2,
            )
            return dict(_BATCH_KEY_DEFAULTS)

        return found

    # ------------------------------------------------------------------
    # Batched generation for reflection / retry phases
    # ------------------------------------------------------------------

    def _generate_erl_completions_batched(
        self, prompt_texts: list[str]
    ) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        """Generate one completion per prompt for ERL phases, in a single batched call.

        Prompts are formatted as single-turn user messages, left-padded for
        generation, then sliced apart after the forward pass so that each result
        carries its own (unpadded) prompt token IDs.  Reduces generation calls
        from ``2 × num_gated`` to 2 per training step.

        Args:
            prompt_texts: Plain-text content for each user turn.

        Returns:
            List of ``(prompt_ids, completion_ids, completion_text)`` tuples, all
            on CPU, in the same order as ``prompt_texts``.
        """
        if not prompt_texts:
            return []

        tokenizer = self.processing_class
        device = self.accelerator.device
        pad_token_id: int = tokenizer.pad_token_id or tokenizer.eos_token_id

        formatted_texts: list[str] = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in prompt_texts
        ]

        original_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        finally:
            tokenizer.padding_side = original_padding_side

        input_ids: torch.Tensor = encoded["input_ids"].to(device)
        attention_mask: torch.Tensor = encoded["attention_mask"].to(device)
        padded_prompt_len: int = input_ids.shape[1]

        unwrapped = self.accelerator.unwrap_model(self.model)
        was_training = unwrapped.training
        try:
            unwrapped.eval()
            with torch.no_grad():
                output_ids: torch.Tensor = unwrapped.generate(
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

        results: list[tuple[torch.Tensor, torch.Tensor, str]] = []
        for i in range(len(prompt_texts)):
            actual_prompt_len = int(attention_mask[i].sum().item())
            actual_prompt_ids = input_ids[i, -actual_prompt_len:].cpu()
            completion_ids = output_ids[i, padded_prompt_len:].cpu()
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            results.append((actual_prompt_ids, completion_ids, completion_text))

        return results

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

        Prompts are left-padded; completions are right-padded.  Advantages are
        computed by global mean/std normalisation across the entire mixed batch
        (y1 + reflections + y2), which is appropriate because the prompts are
        heterogeneous and GRPO's per-group normalisation does not apply.

        Key names in the returned dict are taken from ``self._batch_keys`` (set
        by ``_discover_batch_keys`` on the first training step) so they always
        match whatever names TRL's ``_compute_loss`` expects.

        Args:
            prompt_ids_list: List of 1-D CPU prompt token ID tensors.
            completion_ids_list: List of 1-D CPU completion token ID tensors.
            rewards: Scalar reward for each sequence.

        Returns:
            Dict consumed directly by ``_compute_loss``.
        """
        keys = self._batch_keys or _BATCH_KEY_DEFAULTS
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

        input_ids_full = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask_full = torch.cat([prompt_mask, completion_mask], dim=1)

        with torch.no_grad():
            old_per_token_logps = _extract_logps(
                self._get_per_token_logps_and_entropies(
                    self.model,
                    input_ids_full,
                    attention_mask_full,
                    max_completion_len,
                )
            )

        result: dict[str, Any] = {
            keys["prompt_ids"]: prompt_ids,
            keys["prompt_mask"]: prompt_mask,
            keys["completion_ids"]: completion_ids,
            keys["completion_mask"]: completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "num_items_in_batch": int(completion_mask.sum().item()),
        }

        if self.ref_model is not None:
            with torch.no_grad():
                result["ref_per_token_logps"] = _extract_logps(
                    self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        input_ids_full,
                        attention_mask_full,
                        max_completion_len,
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Core ERL training loop
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self, generation_batch: list[dict] | dict[str, Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Override to run ERL generation every step without buffering.

        ERL's combined batch size varies per step (depends on how many samples
        are gated), making TRL's fixed-size buffer slicing incompatible.
        Regenerating each step is simpler and ensures correctness.
        """
        return self._generate_and_score_completions(generation_batch)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]] | dict[str, Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Run the full ERL loop for one training batch.

        Implements Algorithm 1 (combined RL update) from the paper.  All seven
        ERL phases execute here; the result is consumed directly by
        GRPOTrainer._compute_loss.

        On the first call, ``_discover_batch_keys`` is invoked on the packed
        batch to confirm and cache the key names that ``_compute_loss`` expects.
        All subsequent calls reuse the cached mapping.

        Input normalisation:
            TRL may pass either a dict-of-lists (standard PyTorch DataLoader
            collate) or a list-of-dicts (TRL's own collator).  Both are accepted.

        Args:
            inputs: Dataset rows for the current batch.

        Returns:
            Batch dict with padded tensors, advantages, and logprobs covering all
            generated sequences (y1, reflections, y2).
        """
        inputs = _to_list_of_dicts(inputs)
        prompts = [x["prompt"] for x in inputs]

        # ── Phase 1: First attempt ────────────────────────────────────────
        prompt_ids_y1, completion_ids_y1, completions_y1 = self._unpack_generate(
            self._generate(prompts)
        )

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
        gated_indices: list[int] = [i for i, g in enumerate(gated_mask) if g]

        def _to_cpu_tensor(t: Any) -> torch.Tensor:
            if isinstance(t, torch.Tensor):
                return t.cpu()
            return torch.tensor(t, dtype=torch.long)

        all_prompt_ids: list[torch.Tensor] = [_to_cpu_tensor(p) for p in prompt_ids_y1]
        all_completion_ids: list[torch.Tensor] = [_to_cpu_tensor(c) for c in completion_ids_y1]
        all_rewards: list[float] = r1.cpu().tolist()

        internalization_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        if gated_indices:
            # ── Phase 3: Self-reflection (batched) ───────────────────────
            reflection_contents: list[str] = []
            for i in gated_indices:
                prompt_idx = i // self.num_generations
                original_prompt = str(prompts[prompt_idx])

                memory_entries: list[str] = []
                if self.args.enable_memory and self.memory is not None:
                    memory_entries = self.memory.retrieve(
                        original_prompt, self.args.memory_top_k
                    )

                reflection_contents.append(
                    self._build_reflection_prompt(
                        prompt=original_prompt,
                        attempt=completions_y1[i],
                        feedback=f1[i],
                        reward=float(r1[i].item()),
                        memory_entries=memory_entries,
                    )
                )

            reflection_results = self._generate_erl_completions_batched(reflection_contents)

            # ── Phase 4: Second attempt (batched) ─────────────────────────
            retry_contents: list[str] = []
            for idx, i in enumerate(gated_indices):
                prompt_idx = i // self.num_generations
                original_prompt = str(prompts[prompt_idx])
                _, _, delta_text = reflection_results[idx]
                retry_contents.append(
                    self._build_retry_prompt(prompt=original_prompt, reflection=delta_text)
                )

            retry_results = self._generate_erl_completions_batched(retry_contents)

            # ── Phases 5 + build combined batch ──────────────────────────
            for batch_pos, i in enumerate(gated_indices):
                prompt_idx = i // self.num_generations
                original_prompt = str(prompts[prompt_idx])

                rfl_prompt_ids, delta_ids, delta_text = reflection_results[batch_pos]
                retry_prompt_ids, y2_ids, y2_text = retry_results[batch_pos]

                rewards_y2_per_func = self._calculate_rewards(
                    [inputs[prompt_idx]],
                    [prompts[prompt_idx]],
                    [y2_text],
                    [y2_ids],
                )
                r2: float = (
                    rewards_y2_per_func
                    * self.reward_weights.to(rewards_y2_per_func.device)
                ).nansum(dim=1).item()

                if (
                    self.args.enable_memory
                    and self.memory is not None
                    and r2 > self.args.reward_threshold
                ):
                    self.memory.add(delta_text, original_prompt, r2)

                all_prompt_ids.append(rfl_prompt_ids)
                all_completion_ids.append(delta_ids)
                all_rewards.append(r2)

                all_prompt_ids.append(retry_prompt_ids)
                all_completion_ids.append(y2_ids)
                all_rewards.append(r2)

                if r2 > 0 and self.args.enable_internalization:
                    internalization_pairs.append((_to_cpu_tensor(prompt_ids_y1[i]), y2_ids))

        self._erl_internalization_data = internalization_pairs or None

        # ── Phase 6: Pack combined batch for GRPO update ─────────────────
        batch = self._pack_batch(all_prompt_ids, all_completion_ids, all_rewards)

        if self._batch_keys is None:
            self._batch_keys = self._discover_batch_keys(batch)

        return batch

    # ------------------------------------------------------------------
    # Phase 7: Internalization loss
    # ------------------------------------------------------------------

    def _compute_internalization_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute cross-entropy SFT loss for internalizing y2 without reflection context.

        Trains the model on ``(original_prompt → y2)`` pairs so that improved
        behaviour is preserved at deployment time when no reflection scaffold is
        present.  Labels for prompt tokens are set to ``-100`` (ignored by
        cross-entropy).

        Args:
            model: The current training model (may be Accelerate-wrapped).

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

            all_input_ids.append(torch.tensor(p + c, dtype=torch.long))
            all_labels.append(torch.tensor([-100] * len(p) + c, dtype=torch.long))
            all_attention_masks.append(torch.ones(len(p) + len(c), dtype=torch.long))

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
        """Compute GRPO loss (Phase 6) plus optional internalization loss (Phase 7).

        The GRPO loss runs over the full combined batch produced by
        ``_generate_and_score_completions`` (y1 with r1, reflections with r2,
        y2 with r2).  The internalization loss is standard cross-entropy on
        successful ``(prompt → y2)`` pairs.

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
