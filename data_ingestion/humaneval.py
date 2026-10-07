from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt


class HumanEvalAdapter(DatasetAdapter):
    """Adapter for openai/openai_humaneval records."""

    source_dataset = "humaneval"
    source_config = "openai_humaneval"

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        source_split: str,
        source_id: str,
        benchmark_split: str,
    ) -> BenchmarkPrompt:
        if not isinstance(record, dict):
            raise ValueError("HumanEval record must be a dictionary.")

        prompt_text = record.get("prompt")
        canonical_solution = record.get("canonical_solution")
        task_id = record.get("task_id")

        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("HumanEval record is missing a valid 'prompt' string.")

        if not isinstance(canonical_solution, str) or not canonical_solution.strip():
            raise ValueError("HumanEval record is missing 'canonical_solution'.")

        resolved_source_id = (
            str(task_id).strip()
            if task_id is not None and str(task_id).strip()
            else str(source_id).strip()
        )

        # Clean prompt_id safe for filename/schema
        safe_id = resolved_source_id.replace("/", "_")
        prompt_id = f"humaneval-{source_split}-{safe_id}"

        return BenchmarkPrompt(
            prompt_id=prompt_id,
            prompt=prompt_text.strip(),
            task_type="coding",
            domain="python",
            difficulty="hard",
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=resolved_source_id,
            split=benchmark_split,
            expected_output=canonical_solution.strip(),
            evaluation_type="code_execution",
        )
