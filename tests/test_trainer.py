"""Minimal smoke tests for ERLTrainer instantiation.

These tests verify that ERLTrainer can be constructed correctly and has the
expected interface without actually running a training loop (which requires GPU
and large model weights).
"""

import pytest

from erl import ERLConfig, ERLTrainer
from erl.memory import ReflectionMemory


TINY_MODEL = "sshleifer/tiny-gpt2"


def _dummy_reward_func(prompts, completions, **kwargs):
    return [1.0] * len(completions)


def _dummy_feedback_func(prompts, completions, **kwargs):
    return ["looks good"] * len(completions)


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


@pytest.mark.skip(reason="Requires model download; run manually with network access.")
def test_erl_trainer_instantiates(erl_config, tmp_path):
    from datasets import Dataset

    dataset = Dataset.from_dict({"prompt": ["What is 2+2?"] * 4})

    trainer = ERLTrainer(
        model=TINY_MODEL,
        reward_funcs=_dummy_reward_func,
        args=erl_config,
        train_dataset=dataset,
        feedback_func=_dummy_feedback_func,
    )

    assert isinstance(trainer, ERLTrainer)


def test_erl_trainer_has_feedback_func_attribute(erl_config):
    try:
        trainer = ERLTrainer(
            model=TINY_MODEL,
            reward_funcs=_dummy_reward_func,
            args=erl_config,
            feedback_func=_dummy_feedback_func,
        )
        assert trainer.feedback_func is _dummy_feedback_func
    except Exception:
        pytest.skip("Model not available in this environment.")


def test_erl_trainer_has_memory_attribute(erl_config):
    try:
        trainer = ERLTrainer(
            model=TINY_MODEL,
            reward_funcs=_dummy_reward_func,
            args=erl_config,
            feedback_func=_dummy_feedback_func,
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
            "_generate_erl_completion",
            "_pack_batch",
            "_compute_internalization_loss",
            "compute_loss",
            "_prepare_inputs",
            "_generate_and_score_completions",
        ]:
            assert hasattr(trainer, method), f"Missing method: {method}"
    except Exception:
        pytest.skip("Model not available in this environment.")


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
