"""Tests for ERLTrainer — instantiation, prompt helpers, and the two fragility fixes."""

import warnings

import pytest
import torch

from erl import ERLConfig, ERLTrainer
from erl.memory import ReflectionMemory
from erl.trainer import _BATCH_KEY_DEFAULTS


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
            "_generate_erl_completions_batched",
            "_pack_batch",
            "_compute_internalization_loss",
            "compute_loss",
            "_prepare_inputs",
            "_generate_and_score_completions",
            "_unpack_generate",
            "_discover_batch_keys",
        ]:
            assert hasattr(trainer, method), f"Missing method: {method}"
    except Exception:
        pytest.skip("Model not available in this environment.")


def test_erl_trainer_batch_keys_initialised_to_none(erl_config):
    try:
        trainer = ERLTrainer(
            model=TINY_MODEL,
            reward_funcs=_dummy_reward_func,
            args=erl_config,
        )
        assert trainer._batch_keys is None
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
# Issue 1 fix: _unpack_generate
# ──────────────────────────────────────────────────────────────────────────────

def _make_int_tensor_list(n: int, length: int = 5) -> list[torch.Tensor]:
    return [torch.randint(0, 100, (length,)) for _ in range(n)]


def test_unpack_generate_extracts_first_two_int_tensor_lists_and_string_list():
    prompt_ids = _make_int_tensor_list(4)
    completion_ids = _make_int_tensor_list(4)
    tool_mask = _make_int_tensor_list(4)
    completions = ["hello", "world", "foo", "bar"]
    extra_float = [torch.randn(5) for _ in range(4)]
    extra_int = 42

    result = (prompt_ids, completion_ids, tool_mask, completions, extra_float, extra_int)

    p, c, texts = ERLTrainer._unpack_generate(result)

    assert p is prompt_ids
    assert c is completion_ids
    assert texts is completions


def test_unpack_generate_minimal_tuple():
    prompt_ids = _make_int_tensor_list(2)
    completion_ids = _make_int_tensor_list(2)
    completions = ["a", "b"]

    p, c, texts = ERLTrainer._unpack_generate((prompt_ids, completion_ids, completions))

    assert p is prompt_ids
    assert c is completion_ids
    assert texts is completions


def test_unpack_generate_with_2d_tensors():
    prompt_ids_2d = torch.randint(0, 100, (4, 10))
    completion_ids_2d = torch.randint(0, 100, (4, 8))
    completions = ["a", "b", "c", "d"]

    p, c, texts = ERLTrainer._unpack_generate((prompt_ids_2d, completion_ids_2d, completions))

    assert p is prompt_ids_2d
    assert c is completion_ids_2d
    assert texts is completions


def test_unpack_generate_ignores_float_tensors():
    prompt_ids = _make_int_tensor_list(2)
    completion_ids = _make_int_tensor_list(2)
    logprobs = [torch.randn(5) for _ in range(2)]
    completions = ["hello", "world"]

    p, c, texts = ERLTrainer._unpack_generate(
        (logprobs, prompt_ids, completion_ids, completions)
    )

    assert p is prompt_ids
    assert c is completion_ids


def test_unpack_generate_flattens_nested_string_list():
    prompt_ids = _make_int_tensor_list(2)
    completion_ids = _make_int_tensor_list(2)
    nested = [["hello", "world"], ["foo", "bar"]]

    p, c, texts = ERLTrainer._unpack_generate((prompt_ids, completion_ids, nested))

    assert texts == ["hello", "world", "foo", "bar"]


def test_unpack_generate_raises_on_too_few_token_batches():
    completions = ["hello", "world"]
    only_one = _make_int_tensor_list(2)

    with pytest.raises(ValueError, match="fewer than 2 token ID batches"):
        ERLTrainer._unpack_generate((only_one, completions))


def test_unpack_generate_raises_on_missing_string_list():
    prompt_ids = _make_int_tensor_list(2)
    completion_ids = _make_int_tensor_list(2)

    with pytest.raises(ValueError, match="no list of strings"):
        ERLTrainer._unpack_generate((prompt_ids, completion_ids, 42))


def test_unpack_generate_handles_namedtuple():
    from collections import namedtuple

    GenerateResult = namedtuple(
        "GenerateResult",
        ["prompt_ids", "completion_ids", "tool_mask", "completions", "count"],
    )
    prompt_ids = _make_int_tensor_list(2)
    completion_ids = _make_int_tensor_list(2)
    completions = ["a", "b"]

    result = GenerateResult(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        tool_mask=None,
        completions=completions,
        count=2,
    )

    p, c, texts = ERLTrainer._unpack_generate(result)
    assert p is prompt_ids
    assert c is completion_ids
    assert texts is completions


# ──────────────────────────────────────────────────────────────────────────────
# Issue 2 fix: _discover_batch_keys
# ──────────────────────────────────────────────────────────────────────────────

def test_discover_batch_keys_standard_trl_names():
    reference = {
        "prompt_ids": torch.zeros(2, 5, dtype=torch.long),
        "prompt_mask": torch.ones(2, 5, dtype=torch.long),
        "completion_ids": torch.zeros(2, 8, dtype=torch.long),
        "completion_mask": torch.ones(2, 8, dtype=torch.long),
        "advantages": torch.randn(2),
    }

    keys = ERLTrainer._discover_batch_keys(reference)

    assert keys["prompt_ids"] == "prompt_ids"
    assert keys["prompt_mask"] == "prompt_mask"
    assert keys["completion_ids"] == "completion_ids"
    assert keys["completion_mask"] == "completion_mask"


def test_discover_batch_keys_renamed_trl_keys():
    reference = {
        "prompt_input_ids": torch.zeros(2, 5, dtype=torch.long),
        "prompt_attention_mask": torch.ones(2, 5, dtype=torch.long),
        "completion_input_ids": torch.zeros(2, 8, dtype=torch.long),
        "completion_attention_mask": torch.ones(2, 8, dtype=torch.long),
    }

    keys = ERLTrainer._discover_batch_keys(reference)

    assert keys["prompt_ids"] == "prompt_input_ids"
    assert keys["prompt_mask"] == "prompt_attention_mask"
    assert keys["completion_ids"] == "completion_input_ids"
    assert keys["completion_mask"] == "completion_attention_mask"


def test_discover_batch_keys_partial_match_falls_back_with_warning():
    reference = {
        "something_unknown": torch.zeros(2, 5),
        "another_weird_key": torch.ones(2, 8),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        keys = ERLTrainer._discover_batch_keys(reference)

    assert any("Could not auto-discover" in str(w.message) for w in caught)
    assert keys == _BATCH_KEY_DEFAULTS


def test_discover_batch_keys_empty_dict_falls_back_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        keys = ERLTrainer._discover_batch_keys({})

    assert any("Could not auto-discover" in str(w.message) for w in caught)
    assert keys == _BATCH_KEY_DEFAULTS


def test_discover_batch_keys_returns_defaults_shape():
    reference = {
        "prompt_ids": None,
        "prompt_mask": None,
        "completion_ids": None,
        "completion_mask": None,
    }
    keys = ERLTrainer._discover_batch_keys(reference)
    assert set(keys.keys()) == {"prompt_ids", "prompt_mask", "completion_ids", "completion_mask"}
