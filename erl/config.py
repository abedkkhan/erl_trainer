from dataclasses import dataclass, field

from trl import GRPOConfig

from erl.prompts import DEFAULT_REFLECTION_TEMPLATE, DEFAULT_RETRY_TEMPLATE


@dataclass
class ERLConfig(GRPOConfig):
    """Configuration for ERLTrainer, extending GRPOConfig with ERL-specific parameters.

    Args:
        reward_threshold: Gating threshold τ. Samples with r1 >= τ skip reflection and retry.
        memory_size: Maximum number of reflections stored in cross-episode memory.
        memory_top_k: Number of memory entries to include in the reflection prompt.
        reflection_system_prompt: Prompt template for generating reflections. Must contain
            placeholders for {prompt}, {attempt}, {feedback}, {reward}, {memory}.
        retry_system_prompt: Prompt template for the second attempt. Must contain
            placeholders for {prompt} and {reflection}.
        internalization_coef: Weight of the internalization (distillation) loss relative to RL loss.
        enable_memory: Toggle cross-episode memory on/off.
        enable_internalization: Toggle the distillation step on/off.
        erl_rl_coef: Weight of the ERL RL loss on Δ and y2 relative to the y1 RL loss.
            Set to 0.0 to disable RL updates on reflections and second attempts (Algorithm 1 mode).

    Note:
        The reward function passed to ``ERLTrainer`` may return either
        ``list[float]`` (GRPO-compatible) or ``list[tuple[float, str]]``
        (score + feedback pairs for richer reflections).  No separate
        ``feedback_func`` is needed.
    """

    reward_threshold: float = field(
        default=1.0,
        metadata={"help": "Gating threshold τ. Samples with r1 >= τ skip reflection."},
    )
    memory_add_threshold: float | None = field(
        default=None,
        metadata={
            "help": (
                "Threshold for writing a reflection into memory after retry. "
                "Per the ERL paper (Alg. 2 line 18), memory is written when "
                "r2 > τ — but with continuous rewards normalised to [0, 1] and "
                "τ = 1.0 (the default for retry-gating), this gate would never "
                "fire and the memory feature would be silently dead. "
                "Set this separately when reward_threshold sits at the top of "
                "the reward range. Falls back to reward_threshold when None."
            )
        },
    )
    distill_threshold: float | None = field(
        default=None,
        metadata={
            "help": (
                "Threshold for including a (prompt → y2) pair in the "
                "internalization SFT batch. Per the ERL paper, Ldistill uses "
                "I(r2 > 0) — designed for BINARY reward where >0 means "
                "succeeded. With continuous reward in [0, 1] every retry has "
                "r2 > 0 (even bad ones), so distillation degenerates into "
                "'SFT on every y2' — this was the v1 reflection-collapse "
                "driver. Set this to a calibrated value (e.g. 0.55) to only "
                "distill on retries that actually beat the bar. Falls back to "
                "paper-faithful r2 > 0 when None."
            )
        },
    )
    memory_size: int = field(
        default=50,
        metadata={"help": "Maximum number of reflections stored in cross-episode memory."},
    )
    memory_top_k: int = field(
        default=3,
        metadata={"help": "Number of memory entries to include in the reflection prompt."},
    )
    reflection_system_prompt: str = field(
        default=DEFAULT_REFLECTION_TEMPLATE,
        metadata={
            "help": (
                "Prompt template for generating reflections. "
                "Must contain {prompt}, {attempt}, {feedback}, {reward}, {memory}."
            )
        },
    )
    retry_system_prompt: str = field(
        default=DEFAULT_RETRY_TEMPLATE,
        metadata={
            "help": (
                "Prompt template for the second attempt. "
                "Must contain {prompt} and {reflection}."
            )
        },
    )
    internalization_coef: float = field(
        default=1.0,
        metadata={"help": "Weight of the internalization loss relative to the RL loss."},
    )
    enable_memory: bool = field(
        default=True,
        metadata={"help": "Toggle cross-episode memory on/off."},
    )
    enable_internalization: bool = field(
        default=True,
        metadata={"help": "Toggle the distillation step on/off."},
    )
    erl_rl_coef: float = field(
        default=1.0,
        metadata={
            "help": (
                "Weight of the ERL RL loss (Δ + y2 GRPO) relative to the y1 RL loss. "
                "Set to 0.0 to disable RL updates on reflections and second attempts."
            )
        },
    )
    erl_debug: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable detailed per-step logging of every ERL phase. "
                "Uses the 'erl' logger at DEBUG level."
            )
        },
    )
    reflection_temperature: float | None = field(
        default=None,
        metadata={
            "help": (
                "Sampling temperature override for the reflection (Phase 3) "
                "generation pass ONLY. When None, the global generation_config "
                "temperature is used. Raise above the policy temperature "
                "(e.g. 0.9-1.1) to fight reflection mode-collapse driven by "
                "the internalization SFT loop reinforcing one phrasing."
            )
        },
    )
    reflection_top_p: float | None = field(
        default=None,
        metadata={"help": "top_p override for reflection generation. None = use global."},
    )
    reflection_top_k: int | None = field(
        default=None,
        metadata={"help": "top_k override for reflection generation. None = use global."},
    )
    advantage_type: str = field(
        default="z_score",
        metadata={
            "help": (
                "Advantage normalization for Δ and y2 rewards. "
                "'z_score' (default) — paper-faithful batch-wide z-scoring. "
                "'rank' — rank-based advantages (Xu et al. 2025, 'Rank-based RLHF'). "
                "Rank-based is more robust to judge quantization noise and preserves "
                "diversity better for creative tasks; recommended when reward variance "
                "is small or judges output discrete scores."
            )
        },
    )
