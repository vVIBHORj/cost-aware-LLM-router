from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict, Field

from router.benchmark_schema import BenchmarkPrompt
from router.config import ModelConfig
from runners.outcome_schema import ExecutionStatus, ModelOutcome


def load_pricing_config(path: str | Path = "configs/pricing.yaml") -> dict[str, dict[str, float]]:
    """Load token pricing configuration per model from YAML."""
    path = Path(path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "models" not in data:
        return {}

    pricing_map: dict[str, dict[str, float]] = {}
    for model_id, prices in data["models"].items():
        if isinstance(prices, dict):
            pricing_map[model_id] = {
                "input_per_1m_tokens_usd": float(prices.get("input_per_1m_tokens_usd", 0.0)),
                "output_per_1m_tokens_usd": float(prices.get("output_per_1m_tokens_usd", 0.0)),
            }
    return pricing_map


class GenerationRequest(BaseModel):
    """Parameters for a single model completion request."""
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0, default=4096)
    temperature: float = Field(ge=0.0, default=0.0)
    seed: int | None = Field(default=42)
    timeout_seconds: float = Field(gt=0.0, default=60.0)
    stop: list[str] | None = None


class GenerationResult(BaseModel):
    """Raw result returned by an inference provider prior to outcome packaging."""
    model_config = ConfigDict(extra="forbid")

    generated_text: str = Field(default="")
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = Field(ge=0.0, default=0.0)
    time_to_first_token_ms: float | None = None
    tokens_per_second: float | None = None
    engine_version: str | None = None
    raw_response: Any = None


class InferenceProvider(ABC):
    """
    Provider-agnostic interface for candidate model execution.
    Handles configuration, execution, bounded retries, and telemetry generation.
    Strictly separated from evaluation and scoring.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        pricing: dict[str, float] | None = None,
    ) -> None:
        self.model_config = model_config
        self.pricing = pricing or {
            "input_per_1m_tokens_usd": model_config.pricing.input_per_1m_tokens_usd,
            "output_per_1m_tokens_usd": model_config.pricing.output_per_1m_tokens_usd,
        }

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the underlying backend engine or endpoint is available."""
        pass

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute a single generation request against the model."""
        pass

    def execute_prompt(
        self,
        prompt: BenchmarkPrompt,
        *,
        run_id: str,
        max_retries: int = 3,
        temperature: float = 0.0,
        seed: int = 42,
        timeout_seconds: float = 60.0,
    ) -> ModelOutcome:
        """
        Execute one benchmark prompt with bounded retries, telemetry tracking,
        and explicit cost accounting. Returns a validated ModelOutcome.
        """
        request_timestamp = datetime.now(timezone.utc).isoformat()
        max_output_tokens = self.model_config.limits.max_output_tokens

        # Check backend availability first
        if not self.is_available():
            outcome = ModelOutcome(
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
                request_timestamp=request_timestamp,
                prompt_tokens=None,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                seed=seed,
                generated_text="",
                finish_reason=None,
                output_tokens=None,
                total_tokens=None,
                latency_ms=0.0,
                time_to_first_token_ms=None,
                tokens_per_second=None,
                status="BACKEND_UNAVAILABLE",
                error_type="EngineUnavailableError",
                error_message=f"Backend '{self.model_config.backend}' is not installed or available in this environment.",
                retry_count=0,
                attempt_history=[],
                input_cost_usd=None,
                output_cost_usd=None,
                total_cost_usd=None,
                backend=self.model_config.backend,
                model_name=self.model_config.model_name,
                inference_engine_version=None,
            )
            return outcome

        gen_req = GenerationRequest(
            prompt=prompt.prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )

        attempts: list[dict[str, Any]] = []
        final_result: GenerationResult | None = None
        final_status: ExecutionStatus = "FAILED"
        final_error_type: str | None = None
        final_error_msg: str | None = None
        total_latency_ms = 0.0

        for attempt_idx in range(max_retries + 1):
            t_start = time.perf_counter()
            attempt_time = datetime.now(timezone.utc).isoformat()
            try:
                result = self.generate(gen_req)
                attempt_latency = result.latency_ms or ((time.perf_counter() - t_start) * 1000.0)
                total_latency_ms += attempt_latency

                if not isinstance(result.generated_text, str):
                    raise ValueError("Model output must be a string.")

                attempts.append({
                    "attempt": attempt_idx + 1,
                    "timestamp": attempt_time,
                    "status": "SUCCESS",
                    "latency_ms": round(attempt_latency, 2),
                    "prompt_tokens": result.prompt_tokens,
                    "output_tokens": result.output_tokens,
                })
                final_result = result
                final_status = "SUCCESS"
                break

            except TimeoutError as e:
                attempt_latency = (time.perf_counter() - t_start) * 1000.0
                total_latency_ms += attempt_latency
                final_error_type = "TimeoutError"
                final_error_msg = str(e) or "Inference request timed out."
                attempts.append({
                    "attempt": attempt_idx + 1,
                    "timestamp": attempt_time,
                    "status": "TIMEOUT",
                    "latency_ms": round(attempt_latency, 2),
                    "error": final_error_msg,
                })
                if attempt_idx == max_retries:
                    final_status = "TIMEOUT"

            except Exception as e:
                attempt_latency = (time.perf_counter() - t_start) * 1000.0
                total_latency_ms += attempt_latency
                final_error_type = type(e).__name__
                final_error_msg = str(e) or "Inference request failed."
                attempts.append({
                    "attempt": attempt_idx + 1,
                    "timestamp": attempt_time,
                    "status": "FAILED",
                    "latency_ms": round(attempt_latency, 2),
                    "error": final_error_msg,
                })
                if attempt_idx == max_retries:
                    final_status = "RETRY_EXHAUSTED" if max_retries > 0 else "FAILED"

        retry_count = max(0, len(attempts) - 1)

        outcome = ModelOutcome(
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
            request_timestamp=request_timestamp,
            prompt_tokens=final_result.prompt_tokens if final_result else None,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            seed=seed,
            generated_text=final_result.generated_text if final_result else "",
            finish_reason=final_result.finish_reason if final_result else None,
            output_tokens=final_result.output_tokens if final_result else None,
            total_tokens=final_result.total_tokens if final_result else None,
            latency_ms=round(total_latency_ms, 2),
            time_to_first_token_ms=final_result.time_to_first_token_ms if final_result else None,
            tokens_per_second=final_result.tokens_per_second if final_result else None,
            status=final_status,
            error_type=final_error_type if final_status != "SUCCESS" else None,
            error_message=final_error_msg if final_status != "SUCCESS" else None,
            retry_count=retry_count,
            attempt_history=attempts,
            backend=self.model_config.backend,
            model_name=self.model_config.model_name,
            inference_engine_version=final_result.engine_version if final_result else None,
        )

        outcome.calculate_cost(
            input_price_per_1m_usd=self.pricing.get("input_per_1m_tokens_usd", 0.0),
            output_price_per_1m_usd=self.pricing.get("output_per_1m_tokens_usd", 0.0),
        )

        return outcome
