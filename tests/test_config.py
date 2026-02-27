import pytest
from trl import GRPOConfig

from erl.config import ERLConfig
from erl.prompts import DEFAULT_REFLECTION_TEMPLATE, DEFAULT_RETRY_TEMPLATE


def test_erl_config_is_subclass_of_grpo_config():
    assert issubclass(ERLConfig, GRPOConfig)


def test_erl_config_instantiates_with_defaults(tmp_path):
    config = ERLConfig(output_dir=str(tmp_path))
    assert isinstance(config, ERLConfig)
    assert isinstance(config, GRPOConfig)


def test_erl_config_reward_threshold_default(tmp_path):
    config = ERLConfig(output_dir=str(tmp_path))
    assert config.reward_threshold == 1.0


def test_erl_config_memory_size_default(tmp_path):
    config = ERLConfig(output_dir=str(tmp_path))
    assert config.memory_size == 50


def test_erl_config_memory_top_k_default(tmp_path):
    config = ERLConfig(output_dir=str(tmp_path))
    assert config.memory_top_k == 3


def test_erl_config_internalization_coef_default(tmp_path):
    config = ERLConfig(output_dir=str(tmp_path))
    assert config.internalization_coef == 1.0


def test_erl_config_enable_memory_default(tmp_path):
    config = ERLConfig(output_dir=str(tmp_path))
    assert config.enable_memory is True


def test_erl_config_enable_internalization_default(tmp_path):
    config = ERLConfig(output_dir=str(tmp_path))
    assert config.enable_internalization is True


def test_erl_config_reflection_prompt_default(tmp_path):
    config = ERLConfig(output_dir=str(tmp_path))
    assert config.reflection_system_prompt == DEFAULT_REFLECTION_TEMPLATE


def test_erl_config_retry_prompt_default(tmp_path):
    config = ERLConfig(output_dir=str(tmp_path))
    assert config.retry_system_prompt == DEFAULT_RETRY_TEMPLATE


def test_erl_config_custom_fields(tmp_path):
    config = ERLConfig(
        output_dir=str(tmp_path),
        reward_threshold=0.5,
        memory_size=100,
        memory_top_k=5,
        internalization_coef=0.5,
        enable_memory=False,
        enable_internalization=False,
        reflection_system_prompt="Custom reflection: {prompt} {attempt} {feedback} {reward} {memory}",
        retry_system_prompt="Custom retry: {prompt} {reflection}",
    )
    assert config.reward_threshold == 0.5
    assert config.memory_size == 100
    assert config.memory_top_k == 5
    assert config.internalization_coef == 0.5
    assert config.enable_memory is False
    assert config.enable_internalization is False
    assert config.reflection_system_prompt.startswith("Custom reflection")
    assert config.retry_system_prompt.startswith("Custom retry")


def test_erl_config_inherits_grpo_fields(tmp_path):
    config = ERLConfig(
        output_dir=str(tmp_path),
        num_generations=4,
        learning_rate=1e-5,
        max_completion_length=128,
    )
    assert config.num_generations == 4
    assert config.learning_rate == 1e-5
    assert config.max_completion_length == 128
