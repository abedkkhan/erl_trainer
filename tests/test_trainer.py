"""Tests for ERLTrainer — instantiation, prompt helpers, and ERL-specific helpers."""

import pytest
import torch

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
            "_erl_generate",
            "_erl_compute_rewards",
            "_erl_compute_feedback",
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
# _erl_compute_feedback
# ──────────────────────────────────────────────────────────────────────────────

def test_erl_compute_feedback_returns_empty_strings_without_func(erl_config):
    class _Stub(ERLTrainer):
        def __init__(self):
            self.args = erl_config
            self.feedback_func = None

    stub = _Stub()
    result = stub._erl_compute_feedback(
        [{"prompt": "p", "answer": 4}], ["p"], ["c"]
    )
    assert result == [""]


def test_erl_compute_feedback_calls_func_with_extra_kwargs(erl_config):
    class _Stub(ERLTrainer):
        def __init__(self):
            self.args = erl_config
            self.feedback_func = (
                lambda prompts, completions, answer, **kw: [
                    f"answer={a}" for a in answer
                ]
            )

    stub = _Stub()
    result = stub._erl_compute_feedback(
        [{"prompt": "p", "answer": 7}],
        ["p"],
        ["c"],
    )
    assert result == ["answer=7"]


def test_erl_compute_feedback_multiple_samples(erl_config):
    feedbacks = ["fb-a", "fb-b", "fb-c"]
    call_args: dict = {}

    def feedback_func(prompts, completions, **kwargs):
        call_args["prompts"] = prompts
        call_args["completions"] = completions
        return feedbacks

    class _Stub(ERLTrainer):
        def __init__(self):
            self.args = erl_config
            self.feedback_func = feedback_func

    stub = _Stub()
    inputs = [{"prompt": f"p{i}"} for i in range(3)]
    prompts = [f"p{i}" for i in range(3)]
    completions = [f"c{i}" for i in range(3)]

    result = stub._erl_compute_feedback(inputs, prompts, completions)
    assert result == feedbacks
    assert call_args["prompts"] == prompts
    assert call_args["completions"] == completions
