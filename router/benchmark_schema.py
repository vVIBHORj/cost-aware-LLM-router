from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TaskType = Literal[
    "qa",
    "reasoning",
    "math",
    "coding",
    "summarization",
    "classification",
    "instruction_following",
    "other",
]

Difficulty = Literal["easy", "medium", "hard"]

Split = Literal["train", "validation", "test", "stress"]

EvaluationType = Literal[
    "exact_match",
    "multiple_choice",
    "code_execution",
    "ifeval",
    "reference_based",
    "llm_judge",
]


class BenchmarkPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)

    task_type: TaskType
    domain: str = Field(min_length=1)
    difficulty: Difficulty

    # Dataset provenance
    source_dataset: str = Field(min_length=1)
    source_config: str | None = None
    source_split: str = Field(min_length=1)
    source_id: str = Field(min_length=1)

    # Our benchmark split
    split: Split

    # Optional expected/reference answer.
    # Some datasets do not provide one in a directly usable form.
    expected_output: str | None = None

    # Determines which scorer should be used.
    evaluation_type: EvaluationType

    @field_validator(
        "prompt",
        "domain",
        "source_dataset",
        "source_split",
        "source_id",
    )
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Value must not be empty or whitespace.")

        return value

    @field_validator("source_config", "expected_output")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value if value else None