from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt


class MBPPAdapter(DatasetAdapter):
    """Adapter for MBPP (Mostly Basic Python Problems) records."""

    source_dataset = "mbpp"

    def __init__(self, source_config: str = "full") -> None:
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
            raise ValueError("MBPP record must be a dictionary.")

        # MBPP 'full' config uses 'text', while 'sanitized' config uses 'prompt'
        prompt_text = record.get("text") or record.get("prompt")
        code = record.get("code")
        native_task_id = record.get("task_id")

        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("MBPP record is missing a valid task prompt/description.")

        if not isinstance(code, str) or not code.strip():
            raise ValueError("MBPP record is missing valid reference code.")

        resolved_source_id = (
            str(native_task_id).strip()
            if native_task_id is not None and str(native_task_id).strip()
            else str(source_id).strip()
        )

        prompt_id = (
            f"mbpp-{self.source_config}-"
            f"{source_split}-{resolved_source_id}"
        )

        return BenchmarkPrompt(
            prompt_id=prompt_id,
            prompt=prompt_text.strip(),
            task_type="coding",
            domain="python",
            difficulty="medium",
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=resolved_source_id,
            split=benchmark_split,
            expected_output=code.strip(),
            evaluation_type="code_execution",
        )
