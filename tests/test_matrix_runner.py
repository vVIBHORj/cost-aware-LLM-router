"""
Unit and integration tests for runners/matrix_runner.py.
Validates durability, crash-safe atomic persistence, resume semantics,
deterministic deduplication, failure preservation, and error handling.
"""

import json
import math
from pathlib import Path
from typing import Any
import pytest

from eval.scoring_schema import ScoreOutcome
from router.benchmark_schema import BenchmarkPrompt
from router.config import ModelConfig, load_model_registry
from runners.inference import GenerationRequest, GenerationResult, InferenceProvider
from runners.matrix_runner import (
    ConfigurationMismatchError,
    CorruptedArtifactError,
    PlannedAttempt,
    deduplicate_outcomes,
    deduplicate_scores,
    plan_matrix_attempts,
    run_matrix,
    safe_load_outcomes,
    safe_load_scores,
)
from runners.outcome_schema import ModelOutcome, RunManifest
from runners.storage import load_outcomes_parquet, load_run_manifest, save_outcomes_parquet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prompts() -> list[BenchmarkPrompt]:
    """Three deterministic synthetic prompts for matrix testing."""
    return [
        BenchmarkPrompt(
            prompt_id="synth-001",
            prompt="What is 2 + 2?",
            task_type="math",
            domain="arithmetic",
            difficulty="easy",
            source_dataset="synthetic",
            source_config="default",
            source_split="train",
            source_id="s1",
            split="train",
            expected_output="4",
            evaluation_type="exact_match",
        ),
        BenchmarkPrompt(
            prompt_id="synth-002",
            prompt="What is 3 * 3?",
            task_type="math",
            domain="arithmetic",
            difficulty="easy",
            source_dataset="synthetic",
            source_config="default",
            source_split="train",
            source_id="s2",
            split="train",
            expected_output="9",
            evaluation_type="exact_match",
        ),
        BenchmarkPrompt(
            prompt_id="synth-003",
            prompt="What is 10 / 2?",
            task_type="math",
            domain="arithmetic",
            difficulty="easy",
            source_dataset="synthetic",
            source_config="default",
            source_split="train",
            source_id="s3",
            split="train",
            expected_output="5",
            evaluation_type="exact_match",
        ),
    ]


@pytest.fixture
def synthetic_env(tmp_path: Path, synthetic_prompts: list[BenchmarkPrompt]) -> tuple[Path, Path]:
    """Setup mock processed benchmark directory and output directory."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    split_file = processed_dir / "benchmark_train.jsonl"
    with split_file.open("w", encoding="utf-8") as f:
        for p in synthetic_prompts:
            f.write(p.model_dump_json() + "\n")

    manifest_file = processed_dir / "benchmark_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump({"total_prompts": len(synthetic_prompts)}, f)

    output_dir = tmp_path / "matrix_output"
    return processed_dir, output_dir


class SpyMockProvider(InferenceProvider):
    """Inference provider that records calls and can simulate failures or crashes."""

    def __init__(
        self,
        model_config: ModelConfig,
        fail_prompts: set[str] | None = None,
        crash_on_prompt: str | None = None,
        truncate_prompts: set[str] | None = None,
    ) -> None:
        super().__init__(model_config=model_config)
        self.call_history: list[tuple[str, str]] = []  # (prompt_id, model_id)
        self.fail_prompts = fail_prompts or set()
        self.crash_on_prompt = crash_on_prompt
        self.truncate_prompts = truncate_prompts or set()

    def is_available(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            generated_text="4",
            finish_reason="stop",
            prompt_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_ms=120.0,
        )

    def execute_prompt(
        self,
        prompt: BenchmarkPrompt,
        run_id: str,
        max_retries: int = 3,
        temperature: float = 0.0,
        seed: int | None = 42,
        timeout_seconds: float = 60.0,
        max_output_tokens: int | None = None,
        think: bool | None = None,
        prompt_policy_version: str = "v0",
    ) -> ModelOutcome:
        self.call_history.append((prompt.prompt_id, self.model_config.id))

        if self.crash_on_prompt and prompt.prompt_id == self.crash_on_prompt:
            raise RuntimeError(f"Simulated crash on prompt {prompt.prompt_id}")

        if prompt.prompt_id in self.fail_prompts:
            return ModelOutcome(
                run_id=run_id,
                prompt_id=prompt.prompt_id,
                model_id=self.model_config.id,
                benchmark_split=prompt.split,
                task_type=prompt.task_type,
                domain=prompt.domain,
                difficulty=prompt.difficulty,
                source_dataset=prompt.source_dataset,
                source_config=prompt.source_config,
                source_split=prompt.source_split,
                source_id=prompt.source_id,
                request_timestamp="2026-10-21T12:00:00Z",
                prompt_tokens=15,
                max_output_tokens=max_output_tokens or 512,
                temperature=temperature,
                seed=seed,
                generated_text="",
                finish_reason="error",
                output_tokens=0,
                total_tokens=15,
                latency_ms=50.0,
                status="FAILED",
                error_type="InferenceExecutionError",
                error_message="Simulated inference backend failure.",
                retry_count=0,
                backend="mock",
                model_name=self.model_config.model_name,
                prompt_policy_version=prompt_policy_version,
            )

        if prompt.prompt_id in self.truncate_prompts:
            return ModelOutcome(
                run_id=run_id,
                prompt_id=prompt.prompt_id,
                model_id=self.model_config.id,
                benchmark_split=prompt.split,
                task_type=prompt.task_type,
                domain=prompt.domain,
                difficulty=prompt.difficulty,
                source_dataset=prompt.source_dataset,
                source_config=prompt.source_config,
                source_split=prompt.source_split,
                source_id=prompt.source_id,
                request_timestamp="2026-10-21T12:00:00Z",
                prompt_tokens=15,
                max_output_tokens=max_output_tokens or 64,
                temperature=temperature,
                seed=seed,
                generated_text="Thinking monologue overflowing budget...",
                finish_reason="length",
                output_tokens=64,
                total_tokens=79,
                is_truncated=True,
                latency_ms=1500.0,
                status="TRUNCATED",
                error_type="TokenLimitExceeded",
                error_message="Generation truncated by max_output_tokens.",
                retry_count=0,
                backend="mock",
                model_name=self.model_config.model_name,
                prompt_policy_version=prompt_policy_version,
            )

        return ModelOutcome(
            run_id=run_id,
            prompt_id=prompt.prompt_id,
            model_id=self.model_config.id,
            benchmark_split=prompt.split,
            task_type=prompt.task_type,
            domain=prompt.domain,
            difficulty=prompt.difficulty,
            source_dataset=prompt.source_dataset,
            source_config=prompt.source_config,
            source_split=prompt.source_split,
            source_id=prompt.source_id,
            request_timestamp="2026-10-21T12:00:00Z",
            prompt_tokens=15,
            max_output_tokens=max_output_tokens or 512,
            temperature=temperature,
            seed=seed,
            generated_text=prompt.expected_output or "",
            finish_reason="stop",
            output_tokens=5,
            total_tokens=20,
            latency_ms=100.0,
            status="SUCCESS",
            retry_count=0,
            backend="mock",
            model_name=self.model_config.model_name,
            prompt_policy_version=prompt_policy_version,
        )


# ---------------------------------------------------------------------------
# Test Cases (Requirement 3: 8 Core Test Cases)
# ---------------------------------------------------------------------------

def test_case_1_fresh_run_creates_expected_manifest(synthetic_env):
    """Test 1: Fresh run creates the expected attempt manifest and output artifacts."""
    processed_dir, output_dir = synthetic_env
    registry = load_model_registry("configs/models.yaml")
    two_models = registry.enabled_models()[:2]
    model_ids = [m.id for m in two_models]

    providers = {m.id: SpyMockProvider(model_config=m) for m in two_models}

    outcomes, scores, manifest = run_matrix(
        split="train",
        processed_dir=processed_dir,
        output_dir=output_dir,
        model_ids=model_ids,
        run_id="test-run-001",
        providers_override=providers,
        flush_interval=1,
    )

    # 3 prompts * 2 models = 6 expected attempts
    assert len(outcomes) == 6
    assert len(scores) == 6
    assert manifest.expected_outcome_rows == 6
    assert manifest.actual_outcome_rows == 6
    assert manifest.success_count == 6
    assert manifest.failure_count == 0

    # Verify physical files
    assert (output_dir / "matrix_outcomes.parquet").exists()
    assert (output_dir / "matrix_scores.parquet").exists()
    assert (output_dir / "matrix_manifest.json").exists()

    loaded_manifest = load_run_manifest(output_dir / "matrix_manifest.json")
    assert loaded_manifest.run_id == "test-run-001"
    assert loaded_manifest.actual_outcome_rows == 6


def test_case_2_completed_attempts_are_skipped_when_resuming(synthetic_env):
    """Test 2: Completed attempts are skipped when resuming an existing run."""
    processed_dir, output_dir = synthetic_env
    registry = load_model_registry("configs/models.yaml")
    two_models = registry.enabled_models()[:2]
    model_ids = [m.id for m in two_models]

    # Run first phase: limit to 3 attempts
    p1 = {m.id: SpyMockProvider(model_config=m) for m in two_models}
    outcomes_1, _, _ = run_matrix(
        split="train",
        processed_dir=processed_dir,
        output_dir=output_dir,
        model_ids=model_ids,
        run_id="resumable-run-001",
        providers_override=p1,
        limit=3,
        flush_interval=1,
    )
    assert len(outcomes_1) == 3
    total_calls_p1 = sum(len(p.call_history) for p in p1.values())
    assert total_calls_p1 == 3

    # Run second phase: resume without limit
    p2 = {m.id: SpyMockProvider(model_config=m) for m in two_models}
    outcomes_2, _, manifest_2 = run_matrix(
        split="train",
        processed_dir=processed_dir,
        output_dir=output_dir,
        model_ids=model_ids,
        providers_override=p2,
        flush_interval=1,
    )

    # Verify provider 2 was ONLY called for the remaining 3 attempts
    total_calls_p2 = sum(len(p.call_history) for p in p2.values())
    assert total_calls_p2 == 3

    # Total outcomes must be exactly 6 (all planned attempts, zero duplicates)
    assert len(outcomes_2) == 6
    assert manifest_2.actual_outcome_rows == 6
    unique_keys = {(o.prompt_id, o.model_id) for o in outcomes_2}
    assert len(unique_keys) == 6


def test_case_3_duplicate_outcome_rows_detected_and_handled_deterministically(synthetic_prompts):
    """Test 3: Duplicate outcome rows are detected and handled deterministically."""
    p = synthetic_prompts[0]

    # Create two outcomes with same canonical key: one failed earlier, one succeeded later
    failed_outcome = ModelOutcome(
        run_id="run-dup",
        prompt_id=p.prompt_id,
        model_id="qwen3-1.7b",
        benchmark_split="train",
        task_type="math",
        domain="arithmetic",
        difficulty="easy",
        source_dataset="synth",
        source_split="train",
        source_id="s1",
        request_timestamp="2026-10-21T10:00:00Z",
        prompt_tokens=10,
        max_output_tokens=64,
        temperature=0.0,
        latency_ms=50.0,
        status="FAILED",
        backend="mock",
        model_name="Qwen3-1.7B",
    )
    success_outcome = ModelOutcome(
        run_id="run-dup",
        prompt_id=p.prompt_id,
        model_id="qwen3-1.7b",
        benchmark_split="train",
        task_type="math",
        domain="arithmetic",
        difficulty="easy",
        source_dataset="synth",
        source_split="train",
        source_id="s1",
        request_timestamp="2026-10-21T11:00:00Z",
        prompt_tokens=10,
        max_output_tokens=64,
        temperature=0.0,
        generated_text="4",
        finish_reason="stop",
        output_tokens=5,
        total_tokens=15,
        latency_ms=100.0,
        status="SUCCESS",
        backend="mock",
        model_name="Qwen3-1.7B",
    )

    # 1. Test deduplicate_outcomes prefers SUCCESS over FAILED
    deduped, count = deduplicate_outcomes([failed_outcome, success_outcome])
    assert count == 1
    assert len(deduped) == 1
    assert deduped[0].status == "SUCCESS"

    # Even if success comes first
    deduped_reverse, count_rev = deduplicate_outcomes([success_outcome, failed_outcome])
    assert count_rev == 1
    assert len(deduped_reverse) == 1
    assert deduped_reverse[0].status == "SUCCESS"

    # 2. Test timestamp resolution between two identical statuses
    success_earlier = success_outcome.model_copy(update={"request_timestamp": "2026-10-21T09:00:00Z", "generated_text": "earlier"})
    success_later = success_outcome.model_copy(update={"request_timestamp": "2026-10-21T12:00:00Z", "generated_text": "later"})
    deduped_ts, count_ts = deduplicate_outcomes([success_earlier, success_later])
    assert count_ts == 1
    assert deduped_ts[0].generated_text == "later"


def test_case_4_interrupted_run_resumes_without_losing_persisted_outcomes(synthetic_env):
    """Test 4: An interrupted run resumes without losing previously persisted outcomes."""
    processed_dir, output_dir = synthetic_env
    registry = load_model_registry("configs/models.yaml")
    two_models = registry.enabled_models()[:2]
    model_ids = [m.id for m in two_models]

    # Setup provider for model 0 that crashes on 3rd prompt
    p1 = {
        two_models[0].id: SpyMockProvider(model_config=two_models[0], crash_on_prompt="synth-003"),
        two_models[1].id: SpyMockProvider(model_config=two_models[1]),
    }

    with pytest.raises(RuntimeError, match="Simulated crash on prompt synth-003"):
        run_matrix(
            split="train",
            processed_dir=processed_dir,
            output_dir=output_dir,
            model_ids=model_ids,
            run_id="crash-test-run",
            providers_override=p1,
            flush_interval=1,
        )

    # Verify that the 2 completed attempts prior to crash were persisted
    persisted_after_crash = load_outcomes_parquet(output_dir / "matrix_outcomes.parquet")
    assert len(persisted_after_crash) == 2
    assert {o.prompt_id for o in persisted_after_crash} == {"synth-001", "synth-002"}

    # Resume with healthy providers
    p2 = {m.id: SpyMockProvider(model_config=m) for m in two_models}
    outcomes_resumed, _, manifest_resumed = run_matrix(
        split="train",
        processed_dir=processed_dir,
        output_dir=output_dir,
        model_ids=model_ids,
        providers_override=p2,
        flush_interval=1,
    )

    # All 6 attempts are now present without duplicates
    assert len(outcomes_resumed) == 6
    assert manifest_resumed.actual_outcome_rows == 6
    unique_attempts = {(o.prompt_id, o.model_id) for o in outcomes_resumed}
    assert len(unique_attempts) == 6


def test_case_5_failed_inference_remains_recorded_and_not_scored_zero(synthetic_env):
    """Test 5: A failed or truncated inference remains recorded as a failure and is NOT scored as zero."""
    processed_dir, output_dir = synthetic_env
    registry = load_model_registry("configs/models.yaml")
    model = registry.enabled_models()[0]

    provider = SpyMockProvider(
        model_config=model,
        fail_prompts={"synth-002"},
        truncate_prompts={"synth-003"},
    )

    outcomes, scores, manifest = run_matrix(
        split="train",
        processed_dir=processed_dir,
        output_dir=output_dir,
        model_ids=[model.id],
        run_id="failure-test-run",
        providers_override={model.id: provider},
        flush_interval=1,
    )

    assert len(outcomes) == 3
    assert manifest.success_count == 1
    assert manifest.failure_count == 2

    # Map by prompt_id
    outcomes_by_id = {o.prompt_id: o for o in outcomes}
    scores_by_id = {s.prompt_id: s for s in scores}

    # synth-001: Success
    assert outcomes_by_id["synth-001"].status == "SUCCESS"
    assert scores_by_id["synth-001"].scoring_status == "SUCCESS"
    assert scores_by_id["synth-001"].quality_score == 1.0

    # synth-002: Failed inference
    assert outcomes_by_id["synth-002"].status == "FAILED"
    score_002 = scores_by_id["synth-002"]
    assert score_002.scoring_status == "SCORING_UNAVAILABLE"
    # Crucial assertion: quality_score must be None / NaN, NEVER 0.0
    assert score_002.quality_score is None or math.isnan(score_002.quality_score)
    assert score_002.success_at_target is None
    assert score_002.quality_score != 0.0

    # synth-003: Truncated inference
    assert outcomes_by_id["synth-003"].status == "TRUNCATED"
    score_003 = scores_by_id["synth-003"]
    assert score_003.scoring_status == "SCORING_UNAVAILABLE"
    assert score_003.quality_score is None or math.isnan(score_003.quality_score)
    assert score_003.success_at_target is None
    assert score_003.quality_score != 0.0


def test_case_6_partially_written_artifact_recovered_or_rejected(synthetic_env):
    """Test 6: Partially written/corrupted artifact is recovered safely or rejected with clear error."""
    processed_dir, output_dir = synthetic_env
    registry = load_model_registry("configs/models.yaml")
    model = registry.enabled_models()[0]

    # Run initial 2 attempts
    provider = SpyMockProvider(model_config=model)
    run_matrix(
        split="train",
        processed_dir=processed_dir,
        output_dir=output_dir,
        model_ids=[model.id],
        run_id="corrupt-test-run",
        providers_override={model.id: provider},
        limit=2,
        flush_interval=1,
    )

    outcomes_path = output_dir / "matrix_outcomes.parquet"
    bak_path = output_dir / "matrix_outcomes.parquet.bak"
    assert outcomes_path.exists()

    # Part A: Corrupted primary with valid backup -> safely recovered!
    # Ensure backup exists
    valid_data = load_outcomes_parquet(outcomes_path)
    save_outcomes_parquet(valid_data, bak_path)

    # Corrupt primary
    with outcomes_path.open("wb") as f:
        f.write(b"CORRUPTED_PARQUET_HEADER_GARBAGE_BYTES")

    recovered = safe_load_outcomes(outcomes_path)
    assert len(recovered) == 2

    # Part B: Corrupted primary with no valid backup -> raises CorruptedArtifactError
    if bak_path.exists():
        bak_path.unlink()
    tmp_path = output_dir / "matrix_outcomes.parquet.tmp"
    if tmp_path.exists():
        tmp_path.unlink()

    with pytest.raises(CorruptedArtifactError, match="is corrupted"):
        safe_load_outcomes(outcomes_path)


def test_case_7_configuration_mismatch_prevents_unsafe_resume(synthetic_env):
    """Test 7: A configuration or benchmark hash mismatch prevents unsafe resume."""
    processed_dir, output_dir = synthetic_env
    registry = load_model_registry("configs/models.yaml")
    model = registry.enabled_models()[0]

    provider = SpyMockProvider(model_config=model)

    # Initial run with policy v0, budget 512
    run_matrix(
        split="train",
        processed_dir=processed_dir,
        output_dir=output_dir,
        model_ids=[model.id],
        run_id="config-test-run",
        prompt_policy_version="v0",
        max_output_tokens=512,
        providers_override={model.id: provider},
        limit=1,
    )

    # 1. Policy mismatch
    with pytest.raises(ConfigurationMismatchError, match="prompt_policy_version mismatch"):
        run_matrix(
            split="train",
            processed_dir=processed_dir,
            output_dir=output_dir,
            model_ids=[model.id],
            prompt_policy_version="v4_structured_output",
            max_output_tokens=512,
            providers_override={model.id: provider},
        )

    # 2. Output budget mismatch
    with pytest.raises(ConfigurationMismatchError, match="output_budget mismatch"):
        run_matrix(
            split="train",
            processed_dir=processed_dir,
            output_dir=output_dir,
            model_ids=[model.id],
            prompt_policy_version="v0",
            max_output_tokens=64,
            providers_override={model.id: provider},
        )

    # 3. Backend mismatch
    with pytest.raises(ConfigurationMismatchError, match="backend mismatch"):
        run_matrix(
            split="train",
            processed_dir=processed_dir,
            output_dir=output_dir,
            model_ids=[model.id],
            backend="vllm",  # previous was mock
            prompt_policy_version="v0",
            max_output_tokens=512,
            providers_override={model.id: provider},
        )


def test_case_8_repeated_execution_does_not_silently_overwrite_incompatible_run(synthetic_env):
    """Test 8: Repeated execution does not silently overwrite a prior run with incompatible settings."""
    processed_dir, output_dir = synthetic_env
    registry = load_model_registry("configs/models.yaml")
    model = registry.enabled_models()[0]
    provider = SpyMockProvider(model_config=model)

    # Complete run with run_id = "run-original"
    run_matrix(
        split="train",
        processed_dir=processed_dir,
        output_dir=output_dir,
        model_ids=[model.id],
        run_id="run-original",
        providers_override={model.id: provider},
        flush_interval=1,
    )

    # Case A: Repeated execution with different run_id without --force raises error
    with pytest.raises(ConfigurationMismatchError, match="run_id mismatch"):
        run_matrix(
            split="train",
            processed_dir=processed_dir,
            output_dir=output_dir,
            model_ids=[model.id],
            run_id="run-conflicting",
            providers_override={model.id: provider},
        )

    # Case B: Repeated execution with matching settings detects run is already 100% complete
    spy_re_run = SpyMockProvider(model_config=model)
    outcomes_noop, _, manifest_noop = run_matrix(
        split="train",
        processed_dir=processed_dir,
        output_dir=output_dir,
        model_ids=[model.id],
        run_id="run-original",
        providers_override={model.id: spy_re_run},
    )
    # Zero new inference calls invoked
    assert len(spy_re_run.call_history) == 0
    assert len(outcomes_noop) == 3
    assert manifest_noop.actual_outcome_rows == 3


def test_dry_run_mode(synthetic_env):
    """Test dry-run mode accurately reports planned counts without calling inference."""
    processed_dir, output_dir = synthetic_env
    registry = load_model_registry("configs/models.yaml")
    two_models = registry.enabled_models()[:2]
    model_ids = [m.id for m in two_models]

    spy = {m.id: SpyMockProvider(model_config=m) for m in two_models}

    outcomes, scores, manifest = run_matrix(
        split="train",
        processed_dir=processed_dir,
        output_dir=output_dir,
        model_ids=model_ids,
        run_id="dry-run-001",
        dry_run=True,
        providers_override=spy,
    )

    # Zero inference calls made
    assert sum(len(p.call_history) for p in spy.values()) == 0
    # Zero files written
    assert not (output_dir / "matrix_outcomes.parquet").exists()
    assert not (output_dir / "matrix_manifest.json").exists()
    assert manifest.expected_outcome_rows == 6
