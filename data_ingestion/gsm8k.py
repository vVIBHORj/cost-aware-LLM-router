from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt


class GSM8KAdapter(DatasetAdapter):
    """Adapter for GSM8K records."""

    source_dataset = "gsm8k"
    source_config = "main"

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        source_split: str,
        source_id: str,
        benchmark_split: str,
    ) -> BenchmarkPrompt:

        question = record.get("question")
        answer = record.get("answer")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("GSM8K record is missing a valid question.")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("GSM8K record is missing a valid answer.")

        return BenchmarkPrompt(
            prompt_id=f"gsm8k-{source_split}-{source_id}",
            prompt=question,
            task_type="math",
            domain="grade_school_math",
            difficulty="medium",
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=str(source_id),
            split=benchmark_split,
            expected_output=answer,
            evaluation_type="exact_match",
        )