from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt


def normalize_boolq_answer(answer: Any) -> str:
    """
    Normalize BoolQ boolean answer to canonical string representation ('True' or 'False').
    Rejects invalid types and arbitrary values.
    """
    if isinstance(answer, bool):
        return "True" if answer else "False"

    if isinstance(answer, str):
        cleaned = answer.strip().lower()
        if cleaned in {"true", "false"}:
            return "True" if cleaned == "true" else "False"

    raise ValueError(f"Invalid BoolQ answer value: {answer!r}")


def format_boolq_prompt(passage: str, question: str) -> str:
    """Format passage and question into a model-facing prompt."""
    if not isinstance(passage, str) or not passage.strip():
        raise ValueError("BoolQ record is missing a valid 'passage'.")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("BoolQ record is missing a valid 'question'.")

    return f"Passage:\n{passage.strip()}\n\nQuestion:\n{question.strip()}"


class BoolQAdapter(DatasetAdapter):
    """Adapter for BoolQ reading comprehension classification records."""

    source_dataset = "boolq"

    def __init__(self, source_config: str = "default") -> None:
        if not isinstance(source_config, str) or not source_config.strip():
            raise ValueError("source_config must be a non-empty string.")
        self.source_config = source_config.strip()

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        source_split: str,
        source_id: str,
        benchmark_split: str,
    ) -> BenchmarkPrompt:
        if not isinstance(record, dict):
            raise ValueError("BoolQ record must be a dictionary.")

        passage = record.get("passage")
        question = record.get("question")
        raw_answer = record.get("answer")

        prompt_text = format_boolq_prompt(passage, question)
        expected_output = normalize_boolq_answer(raw_answer)

        prompt_id = (
            f"boolq-{self.source_config}-"
            f"{source_split}-{source_id}"
        )

        return BenchmarkPrompt(
            prompt_id=prompt_id,
            prompt=prompt_text,
            task_type="classification",
            domain="reading_comprehension",
            difficulty="medium",
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=str(source_id).strip(),
            split=benchmark_split,
            expected_output=expected_output,
            evaluation_type="exact_match",
        )
