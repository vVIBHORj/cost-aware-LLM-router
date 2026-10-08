from typing import Any
from pydantic import ValidationError

from eval.scoring_schema import ScoreOutcome, ScoringStatus
from eval.scorers.code_execution import CodeExecutionScorer
from eval.scorers.exact_match import score_exact_match
from eval.scorers.ifeval_scorer import score_ifeval
from eval.scorers.multiple_choice import score_multiple_choice
from eval.scorers.reference_based import score_reference_based
from router.benchmark_schema import BenchmarkPrompt
from runners.outcome_schema import ModelOutcome


class OutcomeScorer:
    """
    Dispatcher for evaluating ModelOutcome records against benchmark specifications.
    Decoupled from inference. Never crashes on individual scoring failures.
    Ensures inference failures are never recorded as successful quality observations.
    """

    def __init__(
        self,
        target_quality: float = 0.80,
        benchmark_manifest_checksum: str | None = None,
    ) -> None:
        self.target_quality = target_quality
        self.benchmark_manifest_checksum = benchmark_manifest_checksum
        self.code_scorer = CodeExecutionScorer(sandbox_available=False)

    def score_outcome(
        self,
        outcome: ModelOutcome,
        prompt: BenchmarkPrompt | None = None,
        evaluation_type: str | None = None,
        expected_output: str | None = None,
    ) -> ScoreOutcome:
        """
        Evaluate a single ModelOutcome and return a validated ScoreOutcome.
        """
        eval_type = evaluation_type or (prompt.evaluation_type if prompt else "exact_match")
        exp_out = expected_output if expected_output is not None else (prompt.expected_output if prompt else None)
        canonical_key = f"{outcome.prompt_id}:{outcome.model_id}:{outcome.run_id}"

        base_kwargs: dict[str, Any] = {
            "run_id": outcome.run_id,
            "prompt_id": outcome.prompt_id,
            "model_id": outcome.model_id,
            "benchmark_split": outcome.benchmark_split,
            "task_type": outcome.task_type,
            "domain": outcome.domain,
            "difficulty": outcome.difficulty,
            "canonical_key": canonical_key,
            "evaluation_type": eval_type,
            "expected_output_available": bool(exp_out and exp_out.strip()),
            "target_quality": self.target_quality,
            "benchmark_manifest_checksum": self.benchmark_manifest_checksum,
            "source_dataset": outcome.source_dataset,
            "source_split": outcome.source_split,
            "source_id": outcome.source_id,
        }

        # Guard: If inference failed, timed out, or backend was unavailable,
        # it MUST NOT become a valid quality observation.
        if outcome.status != "SUCCESS":
            return ScoreOutcome(
                **base_kwargs,
                quality_score=None,
                success_at_target=None,
                scorer_name="inference_failure_filter",
                scorer_version="1.0.0",
                scoring_status="SCORING_UNAVAILABLE",
                scoring_error_type=f"InferenceStatus_{outcome.status}",
                scoring_error_message=(
                    f"Candidate model inference status is '{outcome.status}': "
                    f"{outcome.error_message or 'No output generated.'}"
                ),
                details={"inference_status": outcome.status},
            )

        # Dispatch based on evaluation type
        try:
            if eval_type == "exact_match":
                if not exp_out:
                    return ScoreOutcome(
                        **base_kwargs,
                        quality_score=None,
                        scorer_name="exact_match_scorer",
                        scorer_version="1.0.0",
                        scoring_status="INVALID_INPUT",
                        scoring_error_type="MissingExpectedOutputError",
                        scoring_error_message="exact_match evaluation requires a ground-truth expected_output.",
                    )
                res = score_exact_match(outcome.generated_text, exp_out)
                return ScoreOutcome(
                    **base_kwargs,
                    quality_score=res["quality_score"],
                    scorer_name="exact_match_scorer",
                    scorer_version="1.0.0",
                    scoring_status="SUCCESS",
                    exact_match=res["exact_match"],
                    normalized_exact_match=res["normalized_exact_match"],
                    details=res["details"],
                )

            elif eval_type == "multiple_choice":
                if not exp_out:
                    return ScoreOutcome(
                        **base_kwargs,
                        quality_score=None,
                        scorer_name="multiple_choice_scorer",
                        scorer_version="1.0.0",
                        scoring_status="INVALID_INPUT",
                        scoring_error_type="MissingExpectedOutputError",
                        scoring_error_message="multiple_choice evaluation requires a ground-truth expected_output.",
                    )
                res = score_multiple_choice(outcome.generated_text, exp_out)
                return ScoreOutcome(
                    **base_kwargs,
                    quality_score=res["quality_score"],
                    scorer_name="multiple_choice_scorer",
                    scorer_version="1.0.0",
                    scoring_status="SUCCESS",
                    exact_match=res["exact_match"],
                    normalized_exact_match=res["normalized_exact_match"],
                    details=res["details"],
                )

            elif eval_type == "code_execution":
                res = self.code_scorer.score(outcome.generated_text, exp_out or "")
                return ScoreOutcome(
                    **base_kwargs,
                    quality_score=res["quality_score"],
                    pass_at_1=res["pass_at_1"],
                    scorer_name="code_execution_scorer",
                    scorer_version="1.0.0",
                    scoring_status=res["scoring_status"],
                    scoring_error_type=res["scoring_error_type"],
                    scoring_error_message=res["scoring_error_message"],
                    details=res["details"],
                )

            elif eval_type == "reference_based":
                res = score_reference_based(outcome.generated_text, exp_out or "")
                return ScoreOutcome(
                    **base_kwargs,
                    quality_score=res["quality_score"],
                    reference_score=res["reference_score"],
                    scorer_name="reference_based_scorer",
                    scorer_version="1.0.0",
                    scoring_status=res["scoring_status"],
                    scoring_error_type=res.get("scoring_error_type"),
                    scoring_error_message=res.get("scoring_error_message"),
                    details=res["details"],
                )

            elif eval_type == "ifeval":
                res = score_ifeval(outcome.generated_text, exp_out)
                return ScoreOutcome(
                    **base_kwargs,
                    quality_score=res["quality_score"],
                    scorer_name="ifeval_scorer",
                    scorer_version="1.0.0",
                    scoring_status=res["scoring_status"],
                    scoring_error_type=res["scoring_error_type"],
                    scoring_error_message=res["scoring_error_message"],
                    details=res["details"],
                )

            elif eval_type == "llm_judge":
                return ScoreOutcome(
                    **base_kwargs,
                    quality_score=None,
                    scorer_name="llm_judge_interface",
                    scorer_version="1.0.0",
                    scoring_status="SCORING_UNAVAILABLE",
                    scoring_error_type="JudgeNotConfiguredError",
                    scoring_error_message="LLM judge evaluation is decoupled and not executed in candidate inference phase.",
                )

            else:
                return ScoreOutcome(
                    **base_kwargs,
                    quality_score=None,
                    scorer_name="unknown_eval_type_handler",
                    scorer_version="1.0.0",
                    scoring_status="NOT_IMPLEMENTED",
                    scoring_error_type="UnknownEvaluationTypeError",
                    scoring_error_message=f"Evaluation type '{eval_type}' is not recognized.",
                )

        except Exception as e:
            return ScoreOutcome(
                **base_kwargs,
                quality_score=None,
                scorer_name="error_handler",
                scorer_version="1.0.0",
                scoring_status="SCORING_ERROR",
                scoring_error_type=type(e).__name__,
                scoring_error_message=f"Unexpected error during scoring: {e}",
            )
