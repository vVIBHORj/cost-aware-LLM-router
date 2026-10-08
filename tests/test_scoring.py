import json
import pytest
from pydantic import ValidationError

from eval.scorer import OutcomeScorer
from eval.scorers.code_execution import CodeExecutionScorer
from eval.scorers.exact_match import score_exact_match
from eval.scorers.ifeval_scorer import score_ifeval
from eval.scorers.judge import JudgeConfig, JudgeResult
from eval.scorers.multiple_choice import extract_mc_choice, score_multiple_choice
from eval.scorers.reference_based import score_reference_based
from eval.scoring_schema import ScoreOutcome
from eval.storage import load_scores_parquet, save_scores_parquet
from runners.outcome_schema import ModelOutcome


@pytest.fixture
def base_outcome() -> ModelOutcome:
    return ModelOutcome(
        run_id="run-synth-001",
        prompt_id="prompt-synth-001",
        model_id="qwen3-4b",
        benchmark_split="test",
        task_type="math",
        domain="algebra",
        difficulty="medium",
        source_dataset="gsm8k",
        source_config="main",
        source_split="test",
        source_id="test_001",
        request_timestamp="2026-10-08T12:00:00Z",
        prompt_tokens=25,
        max_output_tokens=4096,
        temperature=0.0,
        seed=42,
        generated_text="42",
        finish_reason="stop",
        output_tokens=2,
        total_tokens=27,
        latency_ms=85.0,
        status="SUCCESS",
        backend="vllm",
        model_name="Qwen/Qwen3-4B",
    )


# 1. Exact match scorer tests
def test_exact_match_strict_and_normalized():
    # Strict match
    res1 = score_exact_match("42", "42")
    assert res1["quality_score"] == 1.0
    assert res1["exact_match"] is True

    # Normalized whitespace & punctuation
    res2 = score_exact_match("  hello, world! ", "Hello World")
    assert res2["quality_score"] == 1.0
    assert res2["exact_match"] is False
    assert res2["normalized_exact_match"] is True

    # LaTeX boxed extraction
    res3 = score_exact_match(r"The result is \boxed{3/4}", "3/4")
    assert res3["quality_score"] == 1.0

    # Numeric equivalence
    res4 = score_exact_match("12.0", "12")
    assert res4["quality_score"] == 1.0

    # Failure
    res5 = score_exact_match("apple", "banana")
    assert res5["quality_score"] == 0.0


# 2. Multiple choice scorer tests
def test_multiple_choice_extraction_and_scoring():
    # Various format extractions
    assert extract_mc_choice("The answer is (B).") == "B"
    assert extract_mc_choice("Answer: C") == "C"
    assert extract_mc_choice("Option D is correct.") == "D"
    assert extract_mc_choice("A") == "A"
    assert extract_mc_choice("(C)") == "C"

    # Scoring matches
    res1 = score_multiple_choice("The answer is (B).", "B")
    assert res1["quality_score"] == 1.0

    res2 = score_multiple_choice("Answer: C", "C")
    assert res2["quality_score"] == 1.0

    res3 = score_multiple_choice("Answer: A", "B")
    assert res3["quality_score"] == 0.0


# 3. Code execution safe stub tests
def test_code_execution_safe_stub():
    scorer = CodeExecutionScorer(sandbox_available=False)
    assert scorer.is_available() is False

    res = scorer.score("def foo(): return 1", "assert foo() == 1")
    assert res["quality_score"] is None
    assert res["scoring_status"] == "SANDBOX_UNAVAILABLE"
    assert "SandboxUnavailableError" in res["scoring_error_type"]


# 4. Reference-based scorer tests
def test_reference_based_scoring():
    pred = "The quick brown fox jumps over the lazy dog."
    ref = "A fast brown fox leaped over a lazy dog."

    res = score_reference_based(pred, ref)
    assert res["scoring_status"] == "SUCCESS"
    assert 0.0 < res["quality_score"] < 1.0
    assert 0.0 < res["reference_score"] < 1.0

    # Missing reference
    missing_res = score_reference_based(pred, "")
    assert missing_res["scoring_status"] == "INVALID_INPUT"
    assert missing_res["quality_score"] is None


# 5. IFEval deterministic rule evaluation tests
def test_ifeval_scorer_rules():
    # Valid rule set: no comma + lowercase
    spec = json.dumps({
        "instruction_id_list": ["punctuation:no_comma", "change_case:english_lowercase"],
        "kwargs": [{}, {}],
    })

    res_pass = score_ifeval("hello world this is a clean response", spec)
    assert res_pass["scoring_status"] == "SUCCESS"
    assert res_pass["quality_score"] == 1.0

    # Violates no_comma
    res_fail = score_ifeval("hello, world", spec)
    assert res_fail["scoring_status"] == "SUCCESS"
    assert res_fail["quality_score"] == 0.0

    # Unsupported instruction gracefully returns SCORING_UNAVAILABLE
    spec_unsupported = json.dumps({
        "instruction_id_list": ["language:response_language"],
        "kwargs": [{"language": "es"}],
    })
    res_unsupported = score_ifeval("hola mundo", spec_unsupported)
    assert res_unsupported["scoring_status"] == "SCORING_UNAVAILABLE"
    assert res_unsupported["quality_score"] is None


# 6. Judge interface schemas
def test_judge_interface_schemas():
    cfg = JudgeConfig(judge_model_id="gpt-4o", rubric_name="reasoning_rigor")
    assert cfg.temperature == 0.0

    result = JudgeResult(
        score=0.9,
        explanation="Well-structured reasoning.",
        passed=True,
        rubric_version="1.0.0",
    )
    assert result.score == 0.9
    assert result.passed is True


# 7. OutcomeScorer dispatcher and failure isolation
def test_dispatcher_handles_inference_failures(base_outcome):
    scorer = OutcomeScorer(target_quality=0.80)

    # Failed outcome
    failed_outcome = base_outcome.model_copy(update={
        "status": "BACKEND_UNAVAILABLE",
        "error_type": "EngineUnavailableError",
        "error_message": "vLLM not installed.",
    })

    scored = scorer.score_outcome(failed_outcome, evaluation_type="exact_match", expected_output="42")
    assert scored.scoring_status == "SCORING_UNAVAILABLE"
    assert scored.quality_score is None
    assert scored.success_at_target is None
    assert "InferenceStatus_BACKEND_UNAVAILABLE" in scored.scoring_error_type

    # Timeout outcome
    timeout_outcome = base_outcome.model_copy(update={
        "status": "TIMEOUT",
        "error_type": "TimeoutError",
        "error_message": "Timed out after 60s.",
    })
    scored_to = scorer.score_outcome(timeout_outcome, evaluation_type="exact_match", expected_output="42")
    assert scored_to.scoring_status == "SCORING_UNAVAILABLE"
    assert scored_to.quality_score is None
    assert scored_to.success_at_target is None


def test_dispatcher_successful_exact_match(base_outcome):
    scorer = OutcomeScorer(target_quality=0.80)

    # Exact match hit
    scored_hit = scorer.score_outcome(base_outcome, evaluation_type="exact_match", expected_output="42")
    assert scored_hit.scoring_status == "SUCCESS"
    assert scored_hit.quality_score == 1.0
    assert scored_hit.success_at_target is True

    # Exact match miss
    miss_outcome = base_outcome.model_copy(update={"generated_text": "99"})
    scored_miss = scorer.score_outcome(miss_outcome, evaluation_type="exact_match", expected_output="42")
    assert scored_miss.scoring_status == "SUCCESS"
    assert scored_miss.quality_score == 0.0
    assert scored_miss.success_at_target is False


def test_dispatcher_missing_expected_output(base_outcome):
    scorer = OutcomeScorer(target_quality=0.80)
    scored = scorer.score_outcome(base_outcome, evaluation_type="exact_match", expected_output=None)
    assert scored.scoring_status == "INVALID_INPUT"
    assert scored.quality_score is None
    assert scored.success_at_target is None


# 8. Quality target semantics
def test_quality_target_semantics():
    # Score outcome validator tests
    o1 = ScoreOutcome(
        run_id="run-1",
        prompt_id="p-1",
        model_id="qwen3-4b",
        benchmark_split="train",
        task_type="math",
        domain="algebra",
        difficulty="easy",
        canonical_key="p-1:qwen3-4b:run-1",
        evaluation_type="exact_match",
        expected_output_available=True,
        quality_score=0.85,
        target_quality=0.80,
        scorer_name="test_scorer",
        scorer_version="1.0.0",
        scoring_status="SUCCESS",
        source_dataset="gsm8k",
        source_split="train",
        source_id="1",
    )
    assert o1.success_at_target is True

    o2 = ScoreOutcome.model_validate({**o1.model_dump(), "quality_score": 0.75})
    assert o2.success_at_target is False

    o3 = ScoreOutcome.model_validate({**o1.model_dump(), "quality_score": None, "scoring_status": "SCORING_UNAVAILABLE"})
    assert o3.success_at_target is None


# 9. Parquet storage for ScoreOutcome
def test_score_storage_parquet_roundtrip(base_outcome, tmp_path):
    scorer = OutcomeScorer(target_quality=0.80)
    score1 = scorer.score_outcome(base_outcome, evaluation_type="exact_match", expected_output="42")

    outcome2 = base_outcome.model_copy(update={"prompt_id": "prompt-synth-002"})
    score2 = scorer.score_outcome(outcome2, evaluation_type="exact_match", expected_output="42")

    parquet_path = tmp_path / "scores.parquet"
    save_scores_parquet([score1, score2], parquet_path)

    loaded = load_scores_parquet(parquet_path)
    assert len(loaded) == 2
    assert loaded[0].canonical_key == score1.canonical_key
    assert loaded[1].canonical_key == score2.canonical_key

    # Duplicate key rejection
    with pytest.raises(ValueError, match="Duplicate canonical score key"):
        save_scores_parquet([score1, score1], tmp_path / "dup.parquet")
