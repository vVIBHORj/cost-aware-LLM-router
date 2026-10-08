from pathlib import Path
import pytest
from pydantic import ValidationError

from router.benchmark_schema import BenchmarkPrompt
from router.config import load_model_registry
from runners.inference import GenerationRequest, load_pricing_config
from runners.outcome_schema import ModelOutcome, RunManifest
from runners.smoke_runner import run_smoke_test, select_smoke_prompts
from runners.storage import load_outcomes_parquet, load_run_manifest, save_outcomes_parquet, save_run_manifest
from runners.vllm_runner import MockInferenceProvider, VLLMProvider


@pytest.fixture
def sample_prompt() -> BenchmarkPrompt:
    return BenchmarkPrompt(
        prompt_id="test-prompt-001",
        prompt="Solve 2x + 5 = 15.",
        task_type="math",
        domain="algebra",
        difficulty="easy",
        source_dataset="gsm8k",
        source_config="main",
        source_split="train",
        source_id="train_0001",
        split="train",
        expected_output="x = 5",
        evaluation_type="exact_match",
    )


@pytest.fixture
def candidate_model():
    registry = load_model_registry("configs/models.yaml")
    return registry.models[0]  # qwen3-1.7b


# 1. Outcome schema validation & required identity fields
def test_outcome_schema_valid_success(sample_prompt, candidate_model):
    outcome = ModelOutcome(
        run_id="run-001",
        prompt_id=sample_prompt.prompt_id,
        model_id=candidate_model.id,
        benchmark_split=sample_prompt.split,
        task_type=sample_prompt.task_type,
        domain=sample_prompt.domain,
        difficulty=sample_prompt.difficulty,
        source_dataset=sample_prompt.source_dataset,
        source_config=sample_prompt.source_config,
        source_split=sample_prompt.source_split,
        source_id=sample_prompt.source_id,
        request_timestamp="2026-10-08T12:00:00Z",
        prompt_tokens=15,
        max_output_tokens=4096,
        temperature=0.0,
        seed=42,
        generated_text="x = 5",
        finish_reason="stop",
        output_tokens=5,
        total_tokens=20,
        latency_ms=120.5,
        time_to_first_token_ms=30.2,
        tokens_per_second=41.49,
        status="SUCCESS",
        retry_count=0,
        backend=candidate_model.backend,
        model_name=candidate_model.model_name,
    )

    assert outcome.status == "SUCCESS"
    assert outcome.total_tokens == 20
    assert outcome.latency_ms == 120.5


def test_outcome_schema_required_identity_fields():
    with pytest.raises(ValidationError):
        # Missing prompt_id and run_id
        ModelOutcome(  # type: ignore
            model_id="qwen3-1.7b",
            benchmark_split="train",
            task_type="math",
            domain="algebra",
            difficulty="easy",
            source_dataset="gsm8k",
            source_split="train",
            source_id="train_0001",
            request_timestamp="2026-10-08T12:00:00Z",
            max_output_tokens=4096,
            temperature=0.0,
            status="SUCCESS",
            latency_ms=10.0,
            backend="vllm",
            model_name="Qwen/Qwen3-1.7B",
        )


def test_outcome_schema_status_validation(sample_prompt, candidate_model):
    valid_statuses = ["SUCCESS", "FAILED", "TIMEOUT", "RETRY_EXHAUSTED", "INVALID_OUTPUT", "BACKEND_UNAVAILABLE"]
    for st in valid_statuses:
        o = ModelOutcome(
            run_id="run-001",
            prompt_id=sample_prompt.prompt_id,
            model_id=candidate_model.id,
            benchmark_split=sample_prompt.split,
            task_type=sample_prompt.task_type,
            domain=sample_prompt.domain,
            difficulty=sample_prompt.difficulty,
            source_dataset=sample_prompt.source_dataset,
            source_config=sample_prompt.source_config,
            source_split=sample_prompt.source_split,
            source_id=sample_prompt.source_id,
            request_timestamp="2026-10-08T12:00:00Z",
            max_output_tokens=4096,
            temperature=0.0,
            status=st,  # type: ignore
            latency_ms=0.0,
            backend=candidate_model.backend,
            model_name=candidate_model.model_name,
        )
        assert o.status == st

    with pytest.raises(ValidationError):
        ModelOutcome(
            run_id="run-001",
            prompt_id=sample_prompt.prompt_id,
            model_id=candidate_model.id,
            benchmark_split=sample_prompt.split,
            task_type=sample_prompt.task_type,
            domain=sample_prompt.domain,
            difficulty=sample_prompt.difficulty,
            source_dataset=sample_prompt.source_dataset,
            source_split=sample_prompt.source_split,
            source_id=sample_prompt.source_id,
            request_timestamp="2026-10-08T12:00:00Z",
            max_output_tokens=4096,
            temperature=0.0,
            status="UNKNOWN_STATUS",  # type: ignore
            latency_ms=0.0,
            backend=candidate_model.backend,
            model_name=candidate_model.model_name,
        )


# 2. Token accounting & pricing cost calculation
def test_token_accounting_and_cost_calculation(sample_prompt, candidate_model):
    outcome = ModelOutcome(
        run_id="run-001",
        prompt_id=sample_prompt.prompt_id,
        model_id=candidate_model.id,
        benchmark_split=sample_prompt.split,
        task_type=sample_prompt.task_type,
        domain=sample_prompt.domain,
        difficulty=sample_prompt.difficulty,
        source_dataset=sample_prompt.source_dataset,
        source_split=sample_prompt.source_split,
        source_id=sample_prompt.source_id,
        request_timestamp="2026-10-08T12:00:00Z",
        prompt_tokens=1000,
        output_tokens=500,
        max_output_tokens=4096,
        temperature=0.0,
        status="SUCCESS",
        latency_ms=50.0,
        backend=candidate_model.backend,
        model_name=candidate_model.model_name,
    )
    assert outcome.total_tokens == 1500

    # Pricing: $2.00 per 1M input, $6.00 per 1M output
    outcome.calculate_cost(input_price_per_1m_usd=2.00, output_price_per_1m_usd=6.00)
    assert outcome.input_cost_usd == pytest.approx(0.002, rel=1e-5)
    assert outcome.output_cost_usd == pytest.approx(0.003, rel=1e-5)
    assert outcome.total_cost_usd == pytest.approx(0.005, rel=1e-5)


def test_token_accounting_unavailable_keeps_none(sample_prompt, candidate_model):
    outcome = ModelOutcome(
        run_id="run-001",
        prompt_id=sample_prompt.prompt_id,
        model_id=candidate_model.id,
        benchmark_split=sample_prompt.split,
        task_type=sample_prompt.task_type,
        domain=sample_prompt.domain,
        difficulty=sample_prompt.difficulty,
        source_dataset=sample_prompt.source_dataset,
        source_split=sample_prompt.source_split,
        source_id=sample_prompt.source_id,
        request_timestamp="2026-10-08T12:00:00Z",
        prompt_tokens=None,
        output_tokens=None,
        max_output_tokens=4096,
        temperature=0.0,
        status="FAILED",
        latency_ms=0.0,
        backend=candidate_model.backend,
        model_name=candidate_model.model_name,
    )
    outcome.calculate_cost(input_price_per_1m_usd=2.00, output_price_per_1m_usd=6.00)
    assert outcome.input_cost_usd is None
    assert outcome.output_cost_usd is None
    assert outcome.total_cost_usd is None


# 3. Provider retry accounting & bounded retries
def test_retry_accounting_success_after_retries(sample_prompt, candidate_model):
    # Fails 2 times, succeeds on 3rd attempt
    provider = MockInferenceProvider(candidate_model, should_fail_attempts=2)
    outcome = provider.execute_prompt(sample_prompt, run_id="retry-run-1", max_retries=3)

    assert outcome.status == "SUCCESS"
    assert outcome.retry_count == 2
    assert len(outcome.attempt_history) == 3
    assert outcome.attempt_history[0]["status"] == "FAILED"
    assert outcome.attempt_history[1]["status"] == "FAILED"
    assert outcome.attempt_history[2]["status"] == "SUCCESS"


def test_retry_exhausted_bounded(sample_prompt, candidate_model):
    # Fails all attempts, bounded by max_retries=2 (total 3 attempts: 1 initial + 2 retries)
    provider = MockInferenceProvider(candidate_model, should_fail_attempts=10)
    outcome = provider.execute_prompt(sample_prompt, run_id="retry-run-2", max_retries=2)

    assert outcome.status == "RETRY_EXHAUSTED"
    assert outcome.retry_count == 2
    assert len(outcome.attempt_history) == 3
    assert all(att["status"] == "FAILED" for att in outcome.attempt_history)


def test_timeout_handling(sample_prompt, candidate_model):
    provider = MockInferenceProvider(candidate_model, simulate_timeout=True)
    outcome = provider.execute_prompt(sample_prompt, run_id="timeout-run-1", max_retries=1)

    assert outcome.status == "TIMEOUT"
    assert outcome.error_type == "TimeoutError"
    assert "timeout" in outcome.error_message.lower()


def test_backend_unavailable_structured_failure(sample_prompt, candidate_model):
    provider = VLLMProvider(candidate_model)  # In this env, vllm is not installed
    assert provider.is_available() is False

    outcome = provider.execute_prompt(sample_prompt, run_id="unavailable-run-1")
    assert outcome.status == "BACKEND_UNAVAILABLE"
    assert outcome.error_type == "EngineUnavailableError"
    assert outcome.prompt_tokens is None
    assert outcome.latency_ms == 0.0


# 4. Parquet outcome storage & uniqueness enforcement
def test_parquet_storage_roundtrip_and_uniqueness(sample_prompt, candidate_model, tmp_path):
    parquet_file = tmp_path / "outcomes.parquet"
    provider = MockInferenceProvider(candidate_model)
    outcome1 = provider.execute_prompt(sample_prompt, run_id="run-roundtrip-1")

    prompt2 = sample_prompt.model_copy(update={"prompt_id": "test-prompt-002"})
    outcome2 = provider.execute_prompt(prompt2, run_id="run-roundtrip-1")

    save_outcomes_parquet([outcome1, outcome2], parquet_file)
    loaded = load_outcomes_parquet(parquet_file)

    assert len(loaded) == 2
    assert loaded[0].prompt_id == "test-prompt-001"
    assert loaded[1].prompt_id == "test-prompt-002"
    assert loaded[0].model_id == candidate_model.id

    # Test duplicate key rejection: (prompt_id, model_id, run_id)
    with pytest.raises(ValueError, match="Duplicate canonical outcome key"):
        save_outcomes_parquet([outcome1, outcome1], tmp_path / "dup.parquet")


# 5. Pricing configuration loader
def test_load_pricing_config():
    pricing = load_pricing_config("configs/pricing.yaml")
    assert "qwen3-1.7b" in pricing
    assert "qwen3-4b" in pricing
    assert "qwen3-8b" in pricing
    assert pricing["qwen3-1.7b"]["input_per_1m_tokens_usd"] == 0.0


# 6. Smoke runner 60-row matrix logic
def test_smoke_runner_mock_execution(tmp_path):
    outcomes, manifest = run_smoke_test(
        output_dir=tmp_path / "smoke",
        run_id="unit-smoke-test-run",
        use_mock=True,
    )

    assert len(outcomes) == 60
    assert manifest.expected_outcome_rows == 60
    assert manifest.actual_outcome_rows == 60
    assert manifest.success_count == 60
    assert manifest.failure_count == 0
    assert manifest.number_of_prompts == 20
    assert manifest.number_of_candidate_models == 3
    assert manifest.token_accounting_available is True
    assert manifest.latency_available is True

    # 20 per model
    for model_id, counts in manifest.per_model_counts.items():
        assert counts["total"] == 20
        assert counts["success"] == 20

    # 5 per split across 20 prompts
    assert manifest.per_split_counts == {
        "train": 5,
        "validation": 5,
        "test": 5,
        "stress": 5,
    }

    # Verify written files
    loaded_manifest = load_run_manifest(tmp_path / "smoke" / "smoke_manifest.json")
    assert loaded_manifest.run_id == "unit-smoke-test-run"

    loaded_parquet = load_outcomes_parquet(tmp_path / "smoke" / "smoke_outcomes.parquet")
    assert len(loaded_parquet) == 60
