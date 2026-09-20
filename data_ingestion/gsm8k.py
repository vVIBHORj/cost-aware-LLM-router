import re
from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt


_FINAL_ANSWER_PATTERN = re.compile(
    r"####\s*(.+?)\s*$",
    re.DOTALL,
)


def extract_gsm8k_final_answer(answer: str) -> str:
    """
    Extract the final answer from the GSM8K answer field.

    GSM8K solutions conventionally end with:

        #### <final answer>
    """

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("GSM8K record is missing a valid answer.")

    match = _FINAL_ANSWER_PATTERN.search(answer)

    if not match:
        raise ValueError(
            "GSM8K answer does not contain a final '####' answer marker."
        )

    final_answer = match.group(1).strip()

    if not final_answer:
        raise ValueError(
            "GSM8K final answer marker is empty."
        )

    return final_answer


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
            raise ValueError(
                "GSM8K record is missing a valid question."
            )

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "GSM8K record is missing a valid answer."
            )

        final_answer = extract_gsm8k_final_answer(answer)

        prompt_id = (
            f"gsm8k-{self.source_config}-"
            f"{source_split}-{source_id}"
        )

        return BenchmarkPrompt(
            prompt_id=prompt_id,
            prompt=question,
            task_type="math",
            domain="grade_school_math",
            difficulty="medium",
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=str(source_id),
            split=benchmark_split,
            expected_output=final_answer,
            evaluation_type="exact_match",
        )