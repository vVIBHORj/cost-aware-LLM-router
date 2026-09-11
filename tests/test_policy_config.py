import pytest

from router.policy_config import load_policy_config


def test_policy_config_loads():
    policy = load_policy_config()

    assert policy.routing.quality.target == 0.80
    assert policy.routing.quality.probability_threshold == 0.80


def test_latency_policy():
    policy = load_policy_config()

    assert policy.routing.latency.enabled is True
    assert policy.routing.latency.max_latency_ms == 10000


def test_objective_policy():
    policy = load_policy_config()

    assert policy.routing.objective.type == "cost"
    assert policy.routing.objective.latency_penalty == 0.0


def test_fallback_policy():
    policy = load_policy_config()

    assert policy.routing.fallback.enabled is True
    assert policy.routing.fallback.min_confidence == 0.60
    assert policy.routing.fallback.fallback_model == "qwen3-8b"


def test_invalid_quality_target_is_rejected(tmp_path):
    config = tmp_path / "policies.yaml"

    config.write_text(
        """
routing:
  quality:
    target: 1.5
    probability_threshold: 0.8

  latency:
    enabled: true
    max_latency_ms: 10000

  objective:
    type: cost
    latency_penalty: 0.0

  fallback:
    enabled: true
    min_confidence: 0.6
    fallback_model: qwen3-8b
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_policy_config(config)