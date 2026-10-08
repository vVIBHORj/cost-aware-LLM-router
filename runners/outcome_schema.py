from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ExecutionStatus = Literal[
    "SUCCESS",
    "FAILED",
    "TIMEOUT",
    "RETRY_EXHAUSTED",
    "INVALID_OUTPUT",
    "BACKEND_UNAVAILABLE",
]


class ModelOutcome(BaseModel):
    """
    Strongly-validated canonical representation of ONE prompt × ONE candidate model evaluation.
    Captures complete identity, request telemetry, generation, performance metrics,
    execution status, retry history, and cost accounting.
    """
    model_config = ConfigDict(extra="forbid")

    # 1. Identity
    run_id: str = Field(min_length=1, description="Unique identifier for the benchmark run.")
    prompt_id: str = Field(min_length=1, description="Prompt identifier from benchmark.")
    model_id: str = Field(min_length=1, description="Candidate model identifier from models.yaml.")
    benchmark_split: Literal["train", "validation", "test", "stress"] = Field(
        description="Benchmark split role."
    )
    task_type: str = Field(min_length=1, description="Benchmark task category.")
    domain: str = Field(min_length=1, description="Prompt domain.")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Problem difficulty level.")
    source_dataset: str = Field(min_length=1, description="Canonical source dataset.")
    source_config: str | None = Field(default=None, description="Sub-task or source configuration.")
    source_split: str = Field(min_length=1, description="Source split name.")
    source_id: str = Field(min_length=1, description="Source record identifier.")

    # 2. Request Telemetry
    request_timestamp: str = Field(min_length=1, description="ISO-8601 UTC timestamp of request start.")
    prompt_tokens: int | None = Field(
        default=None, ge=0, description="Exact input tokens reported by backend. None if unavailable."
    )
    max_output_tokens: int = Field(gt=0, description="Max generated tokens configured for request.")
    temperature: float = Field(ge=0.0, description="Sampling temperature.")
    seed: int | None = Field(default=None, description="Random seed for generation reproducibility.")

    # 3. Generation Output
    generated_text: str = Field(default="", description="Generated response string.")
    finish_reason: str | None = Field(
        default=None, description="Generation stop reason: stop, length, error, or None."
    )
    output_tokens: int | None = Field(
        default=None, ge=0, description="Exact completion tokens reported by backend. None if unavailable."
    )
    total_tokens: int | None = Field(
        default=None, ge=0, description="Exact total tokens reported by backend. None if unavailable."
    )

    # 4. Performance Telemetry
    latency_ms: float = Field(ge=0.0, description="End-to-end inference latency in milliseconds.")
    time_to_first_token_ms: float | None = Field(
        default=None, ge=0.0, description="Time to first token in ms if streaming/available."
    )
    tokens_per_second: float | None = Field(
        default=None, ge=0.0, description="Generation throughput in tokens/sec if available."
    )

    # 5. Execution State & Retries
    status: ExecutionStatus = Field(description="Final execution outcome status.")
    error_type: str | None = Field(default=None, description="Standard error category if failed.")
    error_message: str | None = Field(default=None, description="Detailed error diagnostic message.")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted before final state.")
    attempt_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Detailed telemetry for individual retry attempts."
    )

    # 6. Cost Accounting
    input_cost_usd: float | None = Field(
        default=None, ge=0.0, description="Exact input cost in USD. None if tokens unavailable."
    )
    output_cost_usd: float | None = Field(
        default=None, ge=0.0, description="Exact output cost in USD. None if tokens unavailable."
    )
    total_cost_usd: float | None = Field(
        default=None, ge=0.0, description="Total cost in USD. None if tokens unavailable."
    )

    # 7. Metadata
    backend: str = Field(min_length=1, description="Inference engine backend: vllm, mock, openai, etc.")
    model_name: str = Field(min_length=1, description="Physical model weight repository / path.")
    inference_engine_version: str | None = Field(
        default=None, description="Version string of inference engine."
    )

    @model_validator(mode="after")
    def validate_tokens_and_status(self) -> "ModelOutcome":
        # Ensure total_tokens is consistent if both prompt and output are provided
        if self.prompt_tokens is not None and self.output_tokens is not None:
            computed_total = self.prompt_tokens + self.output_tokens
            if self.total_tokens is None:
                self.total_tokens = computed_total

        # Compute throughput if output_tokens and latency_ms are available and positive
        if self.output_tokens is not None and self.latency_ms > 0 and self.tokens_per_second is None:
            self.tokens_per_second = round((self.output_tokens / (self.latency_ms / 1000.0)), 2)

        # Status validation checks
        if self.status != "SUCCESS":
            if not self.error_type:
                self.error_type = f"{self.status}_ERROR"
        return self

    def calculate_cost(
        self,
        input_price_per_1m_usd: float = 0.0,
        output_price_per_1m_usd: float = 0.0,
    ) -> None:
        """Calculate and set cost fields using explicit token accounting."""
        if self.prompt_tokens is not None:
            self.input_cost_usd = round((self.prompt_tokens / 1_000_000.0) * input_price_per_1m_usd, 8)
        else:
            self.input_cost_usd = None

        if self.output_tokens is not None:
            self.output_cost_usd = round((self.output_tokens / 1_000_000.0) * output_price_per_1m_usd, 8)
        else:
            self.output_cost_usd = None

        if self.input_cost_usd is not None and self.output_cost_usd is not None:
            self.total_cost_usd = round(self.input_cost_usd + self.output_cost_usd, 8)
        else:
            self.total_cost_usd = None


class RunManifest(BaseModel):
    """Manifest describing a full or partial inference run over benchmark prompts."""
    model_config = ConfigDict(extra="forbid")

    run_id: str
    benchmark_manifest_checksum: str
    models_config_version: str
    timestamp: str
    seed: int
    number_of_prompts: int
    number_of_candidate_models: int
    expected_outcome_rows: int
    actual_outcome_rows: int
    success_count: int
    failure_count: int
    retry_count: int
    per_model_counts: dict[str, dict[str, int]]
    per_split_counts: dict[str, int]
    token_accounting_available: bool
    latency_available: bool
