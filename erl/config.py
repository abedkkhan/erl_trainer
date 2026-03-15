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
