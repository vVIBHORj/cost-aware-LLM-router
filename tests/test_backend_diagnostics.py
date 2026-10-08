import pytest
from eval.scoring_schema import ScoreOutcome
from router.benchmark_schema import BenchmarkPrompt
from runners.backend_diagnostics import (
    check_openai_endpoint,
    create_preflight_prompt,
    evaluate_preflight_outcome,
    get_backend_diagnostics,
    run_single_model_preflight,
)
from runners.outcome_schema import ModelOutcome
from runners.vllm_runner import MockInferenceProvider


def test_get_backend_diagnostics_keys():
    diag = get_backend_diagnostics()
    required_keys = [
        "timestamp_utc",
        "python_version",
        "platform_os",
        "torch_installed",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "gpu_names",
        "vllm_installed",
        "vllm_version",
        "openai_endpoint_configured",
        "openai_endpoint_reachable",
        "model_registry_loaded",
        "candidate_models",
    ]
    for k in required_keys:
        assert k in diag, f"Missing key in diagnostics: {k}"

    assert diag["model_registry_loaded"] is True
    assert len(diag["candidate_models"]) == 3
    model_ids = [m["id"] for m in diag["candidate_models"]]
    assert model_ids == ["qwen3-1.7b", "qwen3-4b", "qwen3-8b"]


def test_check_openai_endpoint_unreachable():
    assert check_openai_endpoint(None) is False
    assert check_openai_endpoint("http://127.0.0.1:9999") is False


def test_create_preflight_prompt():
    prompt = create_preflight_prompt()
    assert isinstance(prompt, BenchmarkPrompt)
    assert prompt.prompt == "Return exactly the word: TEST"
    assert prompt.expected_output == "TEST"
    assert prompt.evaluation_type == "exact_match"


def test_run_single_model_preflight_contract():
    outcome = run_single_model_preflight(model_id="qwen3-1.7b")
    assert isinstance(outcome, ModelOutcome)
    assert outcome.model_id == "qwen3-1.7b"
    assert outcome.backend == "vllm"
    assert outcome.model_name == "Qwen/Qwen3-1.7B"
    # In this environment without vllm/torch/endpoint, status should be BACKEND_UNAVAILABLE
    assert outcome.status in ["BACKEND_UNAVAILABLE", "SUCCESS"]
    if outcome.status == "BACKEND_UNAVAILABLE":
        assert outcome.error_type == "EngineUnavailableError"
        assert outcome.prompt_tokens is None
        assert outcome.output_tokens is None
        assert outcome.total_tokens is None


def test_evaluate_preflight_outcome_does_not_mutate():
    outcome = run_single_model_preflight(model_id="qwen3-1.7b")
    outcome_dict_before = outcome.model_dump()

    score = evaluate_preflight_outcome(outcome)
    assert isinstance(score, ScoreOutcome)
    assert outcome.model_dump() == outcome_dict_before

    if outcome.status == "BACKEND_UNAVAILABLE":
        assert score.scoring_status == "SCORING_UNAVAILABLE"
        assert score.quality_score is None


def test_evaluate_successful_outcome_contract(tmp_path):
    # Verify that a successful ModelOutcome evaluates cleanly through OutcomeScorer
    from router.config import load_model_registry
    registry = load_model_registry()
    model_cfg = next(m for m in registry.models if m.id == "qwen3-1.7b")

    mock_provider = MockInferenceProvider(model_config=model_cfg, fixed_output="TEST")
    prompt = create_preflight_prompt()
    outcome = mock_provider.execute_prompt(prompt, run_id="test-preflight-mock")

    assert outcome.status == "SUCCESS"
    assert outcome.generated_text == "TEST"
    assert outcome.prompt_tokens is not None
    assert outcome.output_tokens is not None
    assert outcome.total_tokens == outcome.prompt_tokens + outcome.output_tokens

    score = evaluate_preflight_outcome(outcome, prompt=prompt)
    assert score.scoring_status == "SUCCESS"
    assert score.quality_score == 1.0
    assert score.success_at_target is True
