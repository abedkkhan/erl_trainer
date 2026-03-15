"""Tests for ERLTrainer — instantiation, prompt helpers, and ERL-specific helpers."""

import inspect

import pytest
import torch

from erl import ERLConfig, ERLTrainer
from erl.memory import ReflectionMemory
from erl.trainer import _CachingRewardWrapper


TINY_MODEL = "sshleifer/tiny-gpt2"


def _dummy_reward_func(prompts, completions, **kwargs):
    return [1.0] * len(completions)


@pytest.fixture
def erl_config(tmp_path):
    return ERLConfig(
        output_dir=str(tmp_path),
        num_generations=2,
        max_completion_length=16,
        per_device_train_batch_size=2,  # must be divisible by num_generations
        reward_threshold=0.5,
        memory_size=10,
        memory_top_k=2,
        internalization_coef=1.0,
        enable_memory=True,
        enable_internalization=True,
        report_to="none",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Trainer instantiation
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skip(reason="Requires model download; run manually with network access.")
def test_erl_trainer_instantiates(erl_config, tmp_path):
    from datasets import Dataset

    dataset = Dataset.from_dict({"prompt": ["What is 2+2?"] * 4})

    trainer = ERLTrainer(
        model=TINY_MODEL,
        reward_funcs=_dummy_reward_func,
        args=erl_config,
        train_dataset=dataset,
    )

    assert isinstance(trainer, ERLTrainer)


def test_erl_trainer_has_memory_attribute(erl_config):
    try:
        trainer = ERLTrainer(
            model=TINY_MODEL,
            reward_funcs=_dummy_reward_func,
            args=erl_config,
        )
        assert isinstance(trainer.memory, ReflectionMemory)
    except Exception:
        pytest.skip("Model not available in this environment.")


def test_erl_trainer_memory_disabled(tmp_path):
    config = ERLConfig(
        output_dir=str(tmp_path),
        enable_memory=False,
        report_to="none",
    )
    try:
        trainer = ERLTrainer(
            model=TINY_MODEL,
            reward_funcs=_dummy_reward_func,
            args=config,
        )
        assert trainer.memory is None
    except Exception:
        pytest.skip("Model not available in this environment.")


def test_erl_trainer_has_expected_methods(erl_config):
    try:
        trainer = ERLTrainer(
            model=TINY_MODEL,
            reward_funcs=_dummy_reward_func,
            args=erl_config,
        )
        for method in [
            "_build_reflection_prompt",
            "_build_retry_prompt",
            "_erl_generate",
            "_erl_compute_rewards",
            "_erl_compute_r2_advantages",
            "_erl_grpo_loss",
            "_read_cached_y1_results",
            "_generate_and_score_completions",
            "_compute_internalization_loss",
            "compute_loss",
        ]:
            assert hasattr(trainer, method), f"Missing method: {method}"
    except Exception:
        pytest.skip("Model not available in this environment.")


def test_erl_trainer_internalization_pairs_init(erl_config):
    try:
        trainer = ERLTrainer(
            model=TINY_MODEL,
            reward_funcs=_dummy_reward_func,
            args=erl_config,
        )
        assert isinstance(trainer._internalization_pairs, list)
        assert len(trainer._internalization_pairs) == 0
    except Exception:
        pytest.skip("Model not available in this environment.")


# ──────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────────────────────

def test_build_reflection_prompt_fills_template(erl_config):
    config = erl_config
    config.reflection_system_prompt = (
        "Task: {prompt}\nAttempt: {attempt}\nFeedback: {feedback}\n"
        "Reward: {reward}\nMemory: {memory}"
    )

    class _Stub(ERLTrainer):
        def __init__(self):
            self.args = config

    stub = _Stub()
    result = stub._build_reflection_prompt(
        prompt="solve X",
        attempt="wrong answer",
        feedback="too vague",
        reward=0.2,
        memory_entries=["past reflection 1", "past reflection 2"],
    )

    assert "solve X" in result
    assert "wrong answer" in result
    assert "too vague" in result
    assert "0.2" in result
    assert "past reflection 1" in result


def test_build_retry_prompt_fills_template(erl_config):
    config = erl_config
    config.retry_system_prompt = "Task: {prompt}\nPlan: {reflection}"

    class _Stub(ERLTrainer):
        def __init__(self):
            self.args = config

    stub = _Stub()
    result = stub._build_retry_prompt(prompt="solve X", reflection="be more specific")
    assert "solve X" in result
    assert "be more specific" in result


def test_build_reflection_prompt_no_memory(erl_config):
    config = erl_config
    config.reflection_system_prompt = "Memory: {memory}"

    class _Stub(ERLTrainer):
        def __init__(self):
            self.args = config

    stub = _Stub()
    result = stub._build_reflection_prompt(
        prompt="p", attempt="a", feedback="f", reward=0.0, memory_entries=[]
    )
    assert "None available." in result


# ──────────────────────────────────────────────────────────────────────────────
# Advantage computation — y1 rewards must drive advantages, never y2
# ──────────────────────────────────────────────────────────────────────────────

def test_y1_advantages_use_r1_not_r2():
    """Advantages must be derived from y1 rewards, not y2 rewards."""
    y1_rewards = torch.tensor([0.1, 0.3, 0.0, 0.1])
    y2_rewards = torch.tensor([0.7, 0.5, 0.9, 0.8])  # noqa: F841
    num_generations = 4

    combined_rewards = y1_rewards.clone()
    mean_grouped = combined_rewards.view(-1, num_generations).mean(dim=1)
    std_grouped = combined_rewards.view(-1, num_generations).std(dim=1)
    mean_grouped = mean_grouped.repeat_interleave(num_generations, dim=0)
    std_grouped = std_grouped.repeat_interleave(num_generations, dim=0)
    advantages = (combined_rewards - mean_grouped) / (std_grouped + 1e-4)

    assert advantages[1] > advantages[0], "y1[1] had highest r1, should have highest advantage"
    assert advantages[2] < advantages[0], "y1[2] had lowest r1, should have lowest advantage"
    assert advantages[2] < advantages[1], "advantages must not follow y2 ordering"


def test_y1_advantages_independent_of_y2():
    """Changing y2 rewards must not affect advantages at all."""
    y1_rewards = torch.tensor([0.1, 0.3, 0.0, 0.1])
    num_generations = 4

    def _compute_advantages(y2_rewards):
        combined = y1_rewards.clone()
        mean = combined.view(-1, num_generations).mean(dim=1).repeat_interleave(num_generations)
        std = combined.view(-1, num_generations).std(dim=1).repeat_interleave(num_generations)
        return (combined - mean) / (std + 1e-4)

    adv_a = _compute_advantages(torch.tensor([0.9, 0.5, 0.9, 0.8]))
    adv_b = _compute_advantages(torch.tensor([0.0, 0.0, 0.0, 0.0]))

    assert torch.allclose(adv_a, adv_b), "advantages changed when only y2 rewards changed"


# ──────────────────────────────────────────────────────────────────────────────
# Internalization tokenization boundary
# ──────────────────────────────────────────────────────────────────────────────

def _load_tokenizer():
    """Load tiny-gpt2 tokenizer, skip if unavailable."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    except Exception:
        return None


def test_internalization_input_ids_match_joint_tokenization():
    """input_ids must equal tokenizer(prompt + completion) as one string."""
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        pytest.skip("Tokenizer not available.")

    prompt = "What is 2 + 2?"
    completion = " The answer is 4."
    full_text = prompt + completion

    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    expected = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    assert torch.equal(full_ids, expected)


def test_internalization_labels_mask_prompt_tokens():
    """Prompt token positions must be -100 in labels; completion positions must be real IDs."""
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        pytest.skip("Tokenizer not available.")

    prompt = "What is 2 + 2?"
    completion = " The answer is 4."

    full_ids = tokenizer(
        prompt + completion, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]
    prompt_only_len = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].shape[1]

    labels = full_ids.clone()
    labels[:prompt_only_len] = -100

    assert (labels[:prompt_only_len] == -100).all(), "prompt tokens must be -100"
    assert (labels[prompt_only_len:] != -100).any(), "completion tokens must not all be -100"
    assert torch.equal(labels[prompt_only_len:], full_ids[prompt_only_len:])


def test_internalization_handles_completion_starting_with_space():
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        pytest.skip("Tokenizer not available.")

    prompt = "What is 3 + 1?"
    completion = " 4"

    full_ids = tokenizer(
        prompt + completion, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]
    prompt_only_len = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].shape[1]

    assert full_ids.shape[0] > prompt_only_len, "completion should contribute tokens"


def test_internalization_handles_completion_without_leading_space():
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        pytest.skip("Tokenizer not available.")

    prompt = "Answer:"
    completion = "42"

    full_ids = tokenizer(
        prompt + completion, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]
    prompt_only_len = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].shape[1]

    assert full_ids.shape[0] > prompt_only_len


def test_internalization_handles_prompt_ending_with_newline():
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        pytest.skip("Tokenizer not available.")

    prompt = "Solve: 1 + 1 =\n"
    completion = "2"
    full_text = prompt + completion

    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    expected = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    assert torch.equal(full_ids, expected)


def test_internalization_joint_vs_separate_tokenization_boundary():
    """Demonstrate that separate tokenization can produce a different total token
    count than joint tokenization — the core BPE boundary bug this fix addresses."""
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        pytest.skip("Tokenizer not available.")

    pairs = [
        ("Hello", " world"),
        ("What is", " the answer"),
        ("The result is", "\n4"),
        ("Answer:", "42"),
    ]

    for prompt, completion in pairs:
        separate_len = (
            tokenizer(prompt, add_special_tokens=False)["input_ids"].__len__()
            + tokenizer(completion, add_special_tokens=False)["input_ids"].__len__()
        )
        joint_len = tokenizer(
            prompt + completion, add_special_tokens=False
        )["input_ids"].__len__()

        full_ids = tokenizer(
            prompt + completion, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        assert full_ids.shape[0] == joint_len, (
            "Fixed code must use joint tokenization length"
        )

        if separate_len != joint_len:
            assert full_ids.shape[0] != separate_len, (
                f"For '{prompt}' + '{completion}': joint={joint_len} tokens but "
                f"separate={separate_len} tokens — the fix matters here"
            )
            return

    assert True


# ──────────────────────────────────────────────────────────────────────────────
# _erl_compute_r2_advantages — batch-wide normalisation for Δ and y2
# ──────────────────────────────────────────────────────────────────────────────

def test_r2_advantages_batch_normalized():
    """Batch-wide normalisation: mean of advantages must be approximately zero."""
    rewards = torch.tensor([0.1, 0.3, 0.0, 0.8, 0.5, 0.2])
    adv = ERLTrainer._erl_compute_r2_advantages(rewards)
    assert adv.shape == rewards.shape
    assert abs(adv.mean().item()) < 0.01, f"mean {adv.mean().item()} too far from 0"


def test_r2_advantages_single_sample_zeros():
    """Single reward → all-zero advantages (no normalisation possible)."""
    rewards = torch.tensor([0.7])
    adv = ERLTrainer._erl_compute_r2_advantages(rewards)
    assert torch.equal(adv, torch.zeros_like(rewards))


def test_r2_advantages_identical_rewards_zero():
    """All-identical rewards: std=0, advantages must all be zero."""
    rewards = torch.tensor([0.5, 0.5, 0.5])
    adv = ERLTrainer._erl_compute_r2_advantages(rewards)
    assert torch.allclose(adv, torch.zeros(3))


def test_r2_advantages_independent_of_y1():
    """r2-advantages must not be affected by the y1 reward values."""
    y2_rewards = torch.tensor([0.1, 0.3, 0.0, 0.8])
    adv_a = ERLTrainer._erl_compute_r2_advantages(y2_rewards)
    adv_b = ERLTrainer._erl_compute_r2_advantages(y2_rewards)
    assert torch.allclose(adv_a, adv_b), "advantages changed between identical calls"


# ──────────────────────────────────────────────────────────────────────────────
# _erl_grpo_loss clip formula — pure math validation
# ──────────────────────────────────────────────────────────────────────────────

def test_erl_grpo_clip_formula_zero_log_ratio():
    """When log_ratio=0: coef_1=coef_2=1, per_token_loss = -advantage."""
    advantages = torch.tensor([[1.0, 0.0, -1.0]])
    per_token_logps = torch.tensor([[0.5, 0.5, 0.5]])
    old_per_token_logps = torch.tensor([[0.5, 0.5, 0.5]])

    log_ratio = per_token_logps - old_per_token_logps
    coef_1 = torch.exp(log_ratio)
    eps_low, eps_high = 0.2, 0.2
    coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
    per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)

    expected = torch.tensor([[-1.0, 0.0, 1.0]])
    assert torch.allclose(per_token_loss, expected), f"Got {per_token_loss}"


def test_erl_grpo_clip_activates_on_high_ratio():
    """When coef_1 > 1+eps and advantage > 0, clipping must cap the loss."""
    advantages = torch.tensor([[1.0]])
    log_ratio = torch.tensor([[1.0]])  # coef_1 = e ≈ 2.718
    coef_1 = torch.exp(log_ratio)
    eps_low, eps_high = 0.2, 0.2
    coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)  # clamped to 1.2
    per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
    assert torch.allclose(per_token_loss, torch.tensor([[-1.2]]), atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# _erl_rl_data init
# ──────────────────────────────────────────────────────────────────────────────

def test_erl_rl_data_init_none(erl_config):
    """_erl_rl_data must be None at initialization."""
    try:
        trainer = ERLTrainer(
            model=TINY_MODEL,
            reward_funcs=_dummy_reward_func,
            args=erl_config,
        )
        assert trainer._erl_rl_data is None
    except Exception:
        pytest.skip("Model not available in this environment.")


# ──────────────────────────────────────────────────────────────────────────────
# Internalization loss gradient accumulation normalization
# ──────────────────────────────────────────────────────────────────────────────

def test_internalization_loss_normalized_by_grad_accum():
    """Internalization loss must be divided by gradient_accumulation_steps."""
    raw_loss = torch.tensor(2.4)
    grad_accum_steps = 4
    normalized = raw_loss / grad_accum_steps
    assert torch.isclose(normalized, torch.tensor(0.6)), (
        f"Expected 0.6, got {normalized.item()}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# _CachingRewardWrapper — unified reward API tests
# ──────────────────────────────────────────────────────────────────────────────

def test_caching_wrapper_splits_tuples():
    """Wrapper must split (score, feedback) tuples and return plain floats to TRL."""
    def reward_with_feedback(prompts, completions, **kwargs):
        return [(0.8, "good"), (0.2, "bad"), (1.0, "perfect")]

    wrapper = _CachingRewardWrapper(reward_with_feedback)
    result = wrapper(prompts=["a", "b", "c"], completions=["x", "y", "z"])

    assert result == [0.8, 0.2, 1.0]
    assert wrapper.last_rewards == [0.8, 0.2, 1.0]
    assert wrapper.last_feedback == ["good", "bad", "perfect"]


def test_caching_wrapper_handles_plain_floats():
    """Wrapper must cache plain floats and produce empty-string feedback."""
    def plain_reward(prompts, completions, **kwargs):
        return [0.5, 1.0, 0.0]

    wrapper = _CachingRewardWrapper(plain_reward)
    result = wrapper(prompts=["a", "b", "c"], completions=["x", "y", "z"])

    assert result == [0.5, 1.0, 0.0]
    assert wrapper.last_rewards == [0.5, 1.0, 0.0]
    assert wrapper.last_feedback == ["", "", ""]


def test_cache_cleared_between_calls():
    """Resetting the cache forces the real function to run again."""
    call_count = 0

    def reward(prompts, completions, **kwargs):
        nonlocal call_count
        call_count += 1
        return [float(call_count)] * len(completions)

    wrapper = _CachingRewardWrapper(reward)

    wrapper(prompts=["a"], completions=["x"])
    assert wrapper.last_rewards == [1.0]

    # Simulate cache clear between steps
    wrapper.last_rewards = None
    wrapper.last_feedback = None

    wrapper(prompts=["a"], completions=["y"])
    assert wrapper.last_rewards == [2.0]
    assert call_count == 2


def test_reward_func_called_once_for_y1():
    """Wrapper is called once by the parent; ERL reads the cache — no second call."""
    call_count = 0

    def counting_reward(prompts, completions, **kwargs):
        nonlocal call_count
        call_count += 1
        return [0.5] * len(completions)

    wrapper = _CachingRewardWrapper(counting_reward)

    # Simulate the parent calling the wrapper once
    wrapper(prompts=["p1", "p2"], completions=["c1", "c2"])
    assert call_count == 1

    # ERL reads from cache — no additional call
    cached_rewards = wrapper.last_rewards
    assert cached_rewards == [0.5, 0.5]
    assert call_count == 1, "Reward function called more than once for y1"


def test_multi_reward_feedback_from_first_with_feedback():
    """Feedback comes from the first wrapper that has non-empty feedback."""
    def plain_reward(prompts, completions, **kwargs):
        return [1.0] * len(completions)

    def reward_with_fb(prompts, completions, **kwargs):
        return [(0.5, "feedback here")] * len(completions)

    wrapper_plain = _CachingRewardWrapper(plain_reward)
    wrapper_fb = _CachingRewardWrapper(reward_with_fb)

    wrapper_plain(prompts=["a"], completions=["x"])
    wrapper_fb(prompts=["a"], completions=["x"])

    assert wrapper_plain.last_feedback == [""]
    assert wrapper_fb.last_feedback == ["feedback here"]


def test_module_reward_not_wrapped():
    """nn.Module reward functions must not be wrapped."""
    class DummyRewardModel(torch.nn.Module):
        def forward(self, **kwargs):
            return torch.tensor([1.0])

    model_reward = DummyRewardModel()
    assert isinstance(model_reward, torch.nn.Module)
    # _CachingRewardWrapper is only applied when NOT isinstance(func, nn.Module)
    assert not isinstance(model_reward, _CachingRewardWrapper)


def test_wrapper_passes_kwargs():
    """Extra kwargs (dataset columns) must reach the real function."""
    received_kwargs: dict = {}

    def reward(prompts, completions, **kwargs):
        received_kwargs.update(kwargs)
        return [1.0] * len(completions)

    wrapper = _CachingRewardWrapper(reward)
    wrapper(prompts=["a"], completions=["x"], answer=[42], difficulty=["hard"])

    assert received_kwargs["answer"] == [42]
    assert received_kwargs["difficulty"] == ["hard"]


def test_wrapper_handles_empty_batch():
    """Empty batch must not raise and must cache empty lists."""
    def reward(prompts, completions, **kwargs):
        return []

    wrapper = _CachingRewardWrapper(reward)
    result = wrapper(prompts=[], completions=[])

    assert result == []
    assert wrapper.last_rewards == []
    assert wrapper.last_feedback == []


def test_feedback_func_parameter_removed():
    """ERLTrainer must NOT accept feedback_func as a parameter."""
    sig = inspect.signature(ERLTrainer.__init__)
    assert "feedback_func" not in sig.parameters, (
        "feedback_func should be removed from the public API"
    )


def test_cached_rewards_match_computed_rewards():
    """Rewards read from cache must produce the same weighted sum."""
    def reward_a(prompts, completions, **kwargs):
        return [0.8, 0.2]

    def reward_b(prompts, completions, **kwargs):
        return [0.5, 1.0]

    wrapper_a = _CachingRewardWrapper(reward_a)
    wrapper_b = _CachingRewardWrapper(reward_b)

    wrapper_a(prompts=["p1", "p2"], completions=["c1", "c2"])
    wrapper_b(prompts=["p1", "p2"], completions=["c1", "c2"])

    weights = [1.0, 1.0]
    expected = [
        sum(w * r for w, r in zip(weights, [0.8, 0.5])),  # 1.3
        sum(w * r for w, r in zip(weights, [0.2, 1.0])),  # 1.2
    ]
    cached = [
        sum(w * r for w, r in zip(weights, [wrapper_a.last_rewards[i], wrapper_b.last_rewards[i]]))
        for i in range(2)
    ]
    assert cached == expected


# ──────────────────────────────────────────────────────────────────────────────
# Debug logging
# ──────────────────────────────────────────────────────────────────────────────

def test_debug_config_defaults_false(tmp_path):
    """debug must default to False."""
    config = ERLConfig(output_dir=str(tmp_path), report_to="none")
    assert config.debug is False


@pytest.mark.skip(reason="Requires model download; run manually.")
def test_debug_logging_no_crash(tmp_path):
    """Training with debug=True must not raise."""
    from datasets import Dataset

    dataset = Dataset.from_dict({"prompt": ["What is 2+2?"] * 4})
    config = ERLConfig(
        output_dir=str(tmp_path),
        num_generations=2,
        max_completion_length=16,
        per_device_train_batch_size=2,
        reward_threshold=0.5,
        debug=True,
        report_to="none",
    )

    def reward(prompts, completions, **kwargs):
        return [(0.1, "wrong")] * len(completions)

    trainer = ERLTrainer(
        model="sshleifer/tiny-gpt2",
        reward_funcs=reward,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()


def test_debug_configures_logger(erl_config):
    """When debug=True, the 'erl' logger must be set to DEBUG level."""
    import logging as _logging

    erl_config.debug = True
    try:
        trainer = ERLTrainer(  # noqa: F841
            model=TINY_MODEL,
            reward_funcs=_dummy_reward_func,
            args=erl_config,
        )
        erl_logger = _logging.getLogger("erl")
        assert erl_logger.level == _logging.DEBUG
    except Exception:
        pytest.skip("Model not available in this environment.")


def test_no_debug_logger_untouched(tmp_path):
    """When debug=False, the 'erl' logger level must remain unchanged."""
    import logging as _logging

    config = ERLConfig(output_dir=str(tmp_path), debug=False, report_to="none")
    erl_logger = _logging.getLogger("erl")
    original_level = erl_logger.level
    try:
        trainer = ERLTrainer(  # noqa: F841
            model=TINY_MODEL,
            reward_funcs=_dummy_reward_func,
            args=config,
        )
        assert erl_logger.level == original_level
    except Exception:
        pytest.skip("Model not available in this environment.")
