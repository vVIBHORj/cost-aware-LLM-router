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

Difficulty = Literal[
    "easy",
    "medium",
    "hard",
]

Split = Literal[
    "train",
    "validation",
    "test",
    "stress",
]


class BenchmarkPrompt(BaseModel):
    """
    Canonical representation of a single benchmark prompt.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)

    task_type: TaskType
    domain: str = Field(min_length=1)
    difficulty: Difficulty

    source: str = Field(min_length=1)
    split: Split

    @field_validator("prompt", "domain", "source")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Value must not be empty or whitespace.")

        return value