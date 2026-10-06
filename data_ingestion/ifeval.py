import json
from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt


def serialize_ifeval_constraints(
    instruction_id_list: list[str],
    kwargs: list[dict[str, Any]],
) -> str:
    """
    Serialize IFEval constraint verification metadata into a compact JSON string.
    Filters out null values from kwargs dicts for clean representation.
    """
    cleaned_kwargs = []
    if isinstance(kwargs, list):
        for kw in kwargs:
            if isinstance(kw, dict):
                cleaned_kw = {k: v for k, v in kw.items() if v is not None}
                cleaned_kwargs.append(cleaned_kw)
            else:
                cleaned_kwargs.append(kw)

    spec = {
        "instruction_id_list": instruction_id_list if isinstance(instruction_id_list, list) else [],
        "kwargs": cleaned_kwargs,
    }
    return json.dumps(spec, ensure_ascii=False)


class IFEvalAdapter(DatasetAdapter):
    """Adapter for IFEval (Instruction-Following Evaluation) records."""

    source_dataset = "ifeval"

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
            raise ValueError("IFEval record must be a dictionary.")

        prompt_text = record.get("prompt")
        native_key = record.get("key")
        instruction_id_list = record.get("instruction_id_list")
        kwargs = record.get("kwargs")

        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("IFEval record is missing a valid prompt.")

        if not isinstance(instruction_id_list, list) or len(instruction_id_list) == 0:
            raise ValueError("IFEval record is missing valid instruction_id_list.")

        resolved_source_id = (
            str(native_key).strip()
            if native_key is not None and str(native_key).strip()
            else str(source_id).strip()
        )

        expected_output = serialize_ifeval_constraints(
            instruction_id_list,
            kwargs if isinstance(kwargs, list) else [],
        )

        prompt_id = (
            f"ifeval-{self.source_config}-"
            f"{source_split}-{resolved_source_id}"
        )

        return BenchmarkPrompt(
            prompt_id=prompt_id,
            prompt=prompt_text.strip(),
            task_type="instruction_following",
            domain="general_instruction",
            difficulty="medium",
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=resolved_source_id,
            split=benchmark_split,
            expected_output=expected_output,
            evaluation_type="ifeval",
        )
