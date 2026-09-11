from pathlib import Path

import pytest

from router.config import load_model_registry


REGISTRY_PATH = Path("configs/models.yaml")


def test_model_registry_loads():
    registry = load_model_registry(REGISTRY_PATH)

    assert len(registry.models) == 3


def test_model_ids_are_expected():
    registry = load_model_registry(REGISTRY_PATH)

    assert [model.id for model in registry.models] == [
        "qwen3-1.7b",
        "qwen3-4b",
        "qwen3-8b",
    ]


def test_model_roles_are_expected():
    registry = load_model_registry(REGISTRY_PATH)

    roles = {model.id: model.role for model in registry.models}

    assert roles == {
        "qwen3-1.7b": "cheap",
        "qwen3-4b": "middle",
        "qwen3-8b": "strong",
    }


def test_all_models_are_enabled():
    registry = load_model_registry(REGISTRY_PATH)

    enabled = registry.enabled_models()

    assert len(enabled) == 3
    assert [model.id for model in enabled] == [
        "qwen3-1.7b",
        "qwen3-4b",
        "qwen3-8b",
    ]


def test_model_limits_are_positive():
    registry = load_model_registry(REGISTRY_PATH)

    for model in registry.models:
        assert model.limits.context_window_tokens > 0
        assert model.limits.max_output_tokens > 0


def test_duplicate_model_ids_are_rejected(tmp_path):
    duplicate_config = tmp_path / "models.yaml"

    duplicate_config.write_text(
        """
models:
  - id: duplicate
    display_name: Model A
    provider: local
    backend: vllm
    model_name: model-a
    role: cheap

    capabilities:
      supports_chat: true
      supports_reasoning: true
      supports_code: true
      supports_math: true

    limits:
      context_window_tokens: 1000
      max_output_tokens: 100

    pricing:
      input_per_1m_tokens_usd: 0
      output_per_1m_tokens_usd: 0

    routing:
      enabled: true
      priority: 1

  - id: duplicate
    display_name: Model B
    provider: local
    backend: vllm
    model_name: model-b
    role: strong

    capabilities:
      supports_chat: true
      supports_reasoning: true
      supports_code: true
      supports_math: true

    limits:
      context_window_tokens: 1000
      max_output_tokens: 100

    pricing:
      input_per_1m_tokens_usd: 0
      output_per_1m_tokens_usd: 0

    routing:
      enabled: true
      priority: 2
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Model IDs must be unique"):
        load_model_registry(duplicate_config)