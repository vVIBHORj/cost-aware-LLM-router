from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


ScoringStatus = Literal[
    "SUCCESS",
    "SCORING_UNAVAILABLE",
    "SCORING_ERROR",
    "INVALID_INPUT",
    "SANDBOX_UNAVAILABLE",
    "NOT_IMPLEMENTED",
]


class ScoreOutcome(BaseModel):
    """
    Canonical evaluated score for ONE ModelOutcome.
    Strictly separated from raw model inference telemetry.
    Immutable ModelOutcome rows are never overwritten.
    """
    model_config = ConfigDict(extra="forbid")

    # 1. Identity (mirrors ModelOutcome identity)
    run_id: str = Field(min_length=1, description="Unique identifier for the benchmark run.")
    prompt_id: str = Field(min_length=1, description="Prompt identifier from benchmark.")
    model_id: str = Field(min_length=1, description="Candidate model identifier.")
    benchmark_split: Literal["train", "validation", "test", "stress"] = Field(
        description="Benchmark split role."
    )
    task_type: str = Field(min_length=1, description="Benchmark task category.")
    domain: str = Field(min_length=1, description="Prompt domain.")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Problem difficulty level.")

    # 2. Input Linkage
    canonical_key: str = Field(min_length=1, description="Canonical outcome key: prompt_id:model_id:run_id.")
    evaluation_type: str = Field(min_length=1, description="Target evaluation paradigm from benchmark.")
    expected_output_available: bool = Field(description="Whether a ground-truth expected output was present.")

    # 3. Scoring
    quality_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Normalized quality score in [0.0, 1.0]. None if unavailable."
    )
    success_at_target: bool | None = Field(
        default=None, description="True if quality_score >= target_quality. None if quality is unavailable."
    )
    target_quality: float = Field(
        default=0.80, ge=0.0, le=1.0, description="Configured quality threshold (default 0.80 from policies.yaml)."
    )
    scorer_name: str = Field(min_length=1, description="Identifier of the scorer implementation used.")
    scorer_version: str = Field(min_length=1, description="Version string of scorer.")
    scoring_status: ScoringStatus = Field(description="Status of the scoring operation.")

    # 4. Optional Task-Specific Metrics
    exact_match: bool | None = Field(default=None, description="Strict character-by-character exact match.")
    normalized_exact_match: bool | None = Field(
        default=None, description="Normalized (lowercase, whitespace/punctuation stripped) match."
    )
    pass_at_1: bool | None = Field(default=None, description="Code execution pass@1 result if evaluated.")
    judge_score: float | None = Field(default=None, ge=0.0, le=1.0, description="LLM judge score if evaluated.")
    reference_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Reference overlap score (e.g., ROUGE/F1) if evaluated."
    )
    details: dict[str, Any] = Field(default_factory=dict, description="Fine-grained metric breakdowns.")

    # 5. Failure Handling
    scoring_error_type: str | None = Field(default=None, description="Diagnostic error type if scoring failed.")
    scoring_error_message: str | None = Field(default=None, description="Detailed error explanation.")

    # 6. Provenance
    scoring_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when scoring was computed.",
    )
    benchmark_manifest_checksum: str | None = Field(
        default=None, description="Checksum of benchmark manifest used for evaluation."
    )
    source_dataset: str = Field(min_length=1, description="Origin dataset name.")
    source_split: str = Field(min_length=1, description="Origin split name.")
    source_id: str = Field(min_length=1, description="Origin record identifier.")

    @model_validator(mode="after")
    def compute_success_at_target(self) -> "ScoreOutcome":
        """
        Enforce quality target semantics:
        success_at_target is True/False ONLY when quality_score is actually available.
        Never convert unavailable quality into 0 or false silently.
        """
        if self.quality_score is not None:
            self.success_at_target = bool(self.quality_score >= self.target_quality)
        else:
            self.success_at_target = None

        if self.scoring_status != "SUCCESS" and not self.scoring_error_type:
            self.scoring_error_type = f"{self.scoring_status}_ERROR"

        return self
