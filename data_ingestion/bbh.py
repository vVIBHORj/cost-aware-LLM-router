import re
from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt, EvaluationType

SUPPORTED_TASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)

_MULTIPLE_CHOICE_PATTERN = re.compile(r"^\([A-Za-z]\)$")


def determine_bbh_evaluation_type(target: str) -> EvaluationType:
    """
    Determine evaluation type based on BBH target format.
    BBH multiple-choice tasks use target format '(A)', '(B)', etc.
    Other tasks use exact match (e.g. 'False', '24', 'No', sequences).
    """
    if _MULTIPLE_CHOICE_PATTERN.match(target.strip()):
        return "multiple_choice"
    return "exact_match"


class BBHAdapter(DatasetAdapter):
    """Adapter for BIG-Bench Hard (BBH) reasoning tasks."""

    source_dataset = "bbh"

    def __init__(self, source_config: str) -> None:
        if not isinstance(source_config, str) or not source_config.strip():
            raise ValueError("source_config must be a non-empty string.")

        source_config = source_config.strip()
        if source_config not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported BBH task '{source_config}'. Supported tasks: {sorted(SUPPORTED_TASKS)}"
            )

        self.source_config = source_config

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        source_split: str,
        source_id: str,
        benchmark_split: str,
    ) -> BenchmarkPrompt:
        if not isinstance(record, dict):
            raise ValueError("BBH record must be a dictionary.")

        prompt_text = record.get("input")
        target = record.get("target")

        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("BBH record is missing a valid 'input' prompt.")

        if not isinstance(target, str) or not target.strip():
            raise ValueError("BBH record is missing a valid 'target' answer.")

        target = target.strip()
        evaluation_type = determine_bbh_evaluation_type(target)

        prompt_id = (
            f"bbh-{self.source_config}-"
            f"{source_split}-{source_id}"
        )

        return BenchmarkPrompt(
            prompt_id=prompt_id,
            prompt=prompt_text.strip(),
            task_type="reasoning",
            domain=self.source_config,
            difficulty="hard",
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=str(source_id).strip(),
            split=benchmark_split,
            expected_output=target,
            evaluation_type=evaluation_type,
        )
