from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt


class HellaSwagAdapter(DatasetAdapter):
    """Adapter for Rowan/hellaswag records."""

    source_dataset = "hellaswag"
    source_config = "default"

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        source_split: str,
        source_id: str,
        benchmark_split: str,
    ) -> BenchmarkPrompt:
        if not isinstance(record, dict):
            raise ValueError("HellaSwag record must be a dictionary.")

        ctx = record.get("ctx") or record.get("ctx_a", "")
        endings = record.get("endings")
        label = record.get("label")
        activity_label = record.get("activity_label", "commonsense_reasoning")

        if not isinstance(ctx, str) or not ctx.strip():
            raise ValueError("HellaSwag record is missing context 'ctx'.")

        if not isinstance(endings, list) or len(endings) != 4:
            raise ValueError("HellaSwag record must contain exactly 4 'endings'.")

        if label is None or str(label).strip() == "":
            raise ValueError("HellaSwag record is missing a valid label.")

        try:
            label_idx = int(label)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid HellaSwag label: {label}") from exc

        if not (0 <= label_idx <= 3):
            raise ValueError(f"HellaSwag label out of range 0..3: {label_idx}")

        expected_letter = f"({chr(ord('A') + label_idx)})"

        prompt_text = (
            f"{ctx.strip()}\n\n"
            f"Which continuation makes the most sense?\n"
            f"(A) {endings[0].strip()}\n"
            f"(B) {endings[1].strip()}\n"
            f"(C) {endings[2].strip()}\n"
            f"(D) {endings[3].strip()}\n\n"
            f"Answer:"
        )

        domain = (
            str(activity_label).strip().lower().replace(" ", "_")
            if activity_label
            else "commonsense_reasoning"
        )

        prompt_id = f"hellaswag-{source_split}-{source_id}"

        return BenchmarkPrompt(
            prompt_id=prompt_id,
            prompt=prompt_text.strip(),
            task_type="other",
            domain=domain,
            difficulty="medium",
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=str(source_id).strip(),
            split=benchmark_split,
            expected_output=expected_letter,
            evaluation_type="multiple_choice",
        )
