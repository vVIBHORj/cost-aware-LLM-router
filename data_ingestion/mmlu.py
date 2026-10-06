from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt

CHOICE_LABELS = ("A", "B", "C", "D", "E", "F", "G", "H")


def convert_mmlu_answer(answer: Any, num_choices: int = 4) -> str:
    """
    Convert raw MMLU answer representation (integer index or choice string)
    to a standardized choice label ('A', 'B', 'C', 'D', ...).
    """
    if isinstance(answer, bool):
        raise ValueError("Boolean value is not a valid MMLU answer.")

    if isinstance(answer, int):
        if 0 <= answer < num_choices:
            return CHOICE_LABELS[answer]
        raise ValueError(
            f"MMLU answer index {answer} is out of range for {num_choices} choices."
        )

    if isinstance(answer, str):
        cleaned = answer.strip().upper()
        if cleaned in CHOICE_LABELS[:num_choices]:
            return cleaned
        if cleaned.isdigit():
            idx = int(cleaned)
            if 0 <= idx < num_choices:
                return CHOICE_LABELS[idx]
        raise ValueError(f"MMLU answer string '{answer}' is invalid.")

    raise ValueError(f"Invalid MMLU answer type: {type(answer).__name__}")


def format_mmlu_prompt(question: str, choices: list[str]) -> str:
    """
    Format an MMLU question and list of choices into a single prompt string.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("MMLU record is missing a valid question.")

    if not isinstance(choices, list) or len(choices) < 2:
        raise ValueError("MMLU record choices must be a list with at least 2 options.")

    formatted_choices = []
    for idx, choice in enumerate(choices):
        if not isinstance(choice, str) or not choice.strip():
            raise ValueError(f"MMLU choice at index {idx} must be a non-empty string.")
        label = CHOICE_LABELS[idx]
        formatted_choices.append(f"{label}. {choice.strip()}")

    return f"{question.strip()}\n" + "\n".join(formatted_choices)


class MMLUAdapter(DatasetAdapter):
    """Adapter for MMLU multiple-choice records."""

    source_dataset = "mmlu"

    def __init__(self, source_config: str = "all") -> None:
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
        question = record.get("question")
        choices = record.get("choices")
        raw_answer = record.get("answer")
        subject = record.get("subject")

        prompt_text = format_mmlu_prompt(question, choices)
        expected_output = convert_mmlu_answer(
            raw_answer,
            num_choices=len(choices),
        )

        domain = (
            subject.strip()
            if isinstance(subject, str) and subject.strip()
            else self.source_config
        )

        prompt_id = (
            f"mmlu-{self.source_config}-"
            f"{source_split}-{source_id}"
        )

        return BenchmarkPrompt(
            prompt_id=prompt_id,
            prompt=prompt_text,
            task_type="qa",
            domain=domain,
            difficulty="medium",
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=str(source_id),
            split=benchmark_split,
            expected_output=expected_output,
            evaluation_type="multiple_choice",
        )
