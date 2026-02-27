# erl-trainer

**Experiential Reinforcement Learning (ERL)** — a thin wrapper on HuggingFace TRL's `GRPOTrainer` that adds a reflection-retry-internalization loop to standard GRPO training.

Install it, swap `GRPOTrainer` for `ERLTrainer`, add a `feedback_func`, and you get ERL training. Everything else — LoRA, quantization, datasets, reward functions — works exactly like TRL.

## Installation

```bash
pip install erl-trainer
```

## Quick Start

```python
from erl import ERLConfig, ERLTrainer
from datasets import load_dataset
from peft import LoraConfig  # optional

dataset = load_dataset("your_dataset", split="train")

# Standard reward function (same as TRL)
def reward_func(completions, **kwargs):
    return [compute_your_score(c) for c in completions]

# NEW: Textual feedback function (unique to ERL)
def feedback_func(completions, **kwargs):
    return [get_your_feedback(c) for c in completions]

config = ERLConfig(
    output_dir="erl-output",
    num_train_epochs=3,
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    num_generations=4,
    # ERL-specific params
    reward_threshold=1.0,
    memory_size=50,
    memory_top_k=3,
    internalization_coef=1.0,
)

# Optional: LoRA config (works exactly like TRL)
lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules="all-linear",
)

trainer = ERLTrainer(
    model="Qwen/Qwen2.5-3B-Instruct",
    args=config,
    train_dataset=dataset,
    reward_funcs=reward_func,
    feedback_func=feedback_func,   # NEW: the only addition vs TRL
    peft_config=lora_config,       # optional, same as TRL
)

trainer.train()
```

## The ERL Algorithm

Each training step runs seven phases:

| Phase | Description |
|-------|-------------|
| **1. First attempt** | Generate responses `y1` for the batch; compute reward `r1` and textual feedback `f1`. |
| **2. Gating** | Samples where `r1 < reward_threshold` enter the reflection loop; others are done. |
| **3. Self-reflection** | For gated samples, prompt the model to reflect on what went wrong, using `f1` and relevant entries from cross-episode memory. |
| **4. Second attempt** | Generate improved responses `y2` guided by the reflection; compute reward `r2`. |
| **5. Memory update** | Successful reflections (`r2 > threshold`) are stored in a FIFO reflection memory for future steps. |
| **6. GRPO update** | Policy gradient over the combined batch: `y1` (reward `r1`), reflections (reward `r2`), and `y2` (reward `r2`). |
| **7. Internalization** | SFT cross-entropy loss on `(prompt → y2)` pairs for successful second attempts, teaching the model to skip reflection at inference time. |

## Configuration

All `GRPOConfig` options are inherited. ERL adds:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reward_threshold` | `1.0` | Gating threshold τ. Samples with `r1 >= τ` skip reflection. |
| `memory_size` | `50` | Max reflections stored in cross-episode memory. |
| `memory_top_k` | `3` | Reflections retrieved per reflection prompt. |
| `reflection_system_prompt` | *(built-in)* | Template with `{prompt}`, `{attempt}`, `{feedback}`, `{reward}`, `{memory}`. |
| `retry_system_prompt` | *(built-in)* | Template with `{prompt}` and `{reflection}`. |
| `internalization_coef` | `1.0` | Weight of internalization loss relative to RL loss. |
| `enable_memory` | `True` | Toggle cross-episode memory on/off. |
| `enable_internalization` | `True` | Toggle the distillation step on/off. |

## Feedback Function

The only new concept vs TRL is `feedback_func`. It receives the same arguments as a reward function and must return a list of feedback strings — one per completion:

```python
def feedback_func(prompts, completions, **kwargs) -> list[str]:
    feedbacks = []
    for prompt, completion in zip(prompts, completions):
        feedbacks.append(f"Your answer was missing: {diagnose(prompt, completion)}")
    return feedbacks
```

## License

MIT
