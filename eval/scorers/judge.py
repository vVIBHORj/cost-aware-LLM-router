from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class JudgeConfig(BaseModel):
    """Configuration for an LLM evaluation judge."""
    model_config = ConfigDict(extra="forbid")

    judge_model_id: str = Field(default="gpt-4o", description="Judge model identifier.")
    judge_version: str = Field(default="1.0.0", description="Evaluation prompt/rubric version.")
    temperature: float = Field(default=0.0, ge=0.0, description="Judge sampling temperature.")
    max_tokens: int = Field(default=512, gt=0, description="Max tokens for judge explanation.")
    rubric_name: str = Field(default="general_quality_v1", description="Named evaluation rubric.")


class JudgeResult(BaseModel):
    """Structured evaluation output returned by a JudgeProvider."""
    model_config = ConfigDict(extra="forbid")

    score: float | None = Field(default=None, ge=0.0, le=1.0, description="Normalized score [0.0, 1.0].")
    explanation: str = Field(default="", description="Judge reasoning / rationale.")
    passed: bool | None = Field(default=None, description="Whether criteria was satisfied.")
    rubric_version: str = Field(description="Rubric version applied.")
    raw_response: Any = Field(default=None, description="Raw judge response.")


class JudgeProvider(ABC):
    """
    Abstract interface for LLM-as-a-Judge evaluations.
    Decoupled from candidate model inference to prevent external dependencies.
    """

    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()

    @abstractmethod
    def evaluate(
        self,
        prompt: str,
        response: str,
        reference: str | None = None,
    ) -> JudgeResult:
        """Evaluate a model generation using an LLM judge."""
        pass
