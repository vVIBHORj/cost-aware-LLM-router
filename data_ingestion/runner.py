from collections.abc import Iterable
from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt


def ingest_records(
    records: Iterable[dict[str, Any]],
    adapter: DatasetAdapter,
    *,
    source_split: str,
    benchmark_split: str,
) -> list[BenchmarkPrompt]:
    """
    Convert raw dataset records into validated BenchmarkPrompt objects.

    Guarantees:
    - Every record is converted through the supplied adapter.
    - Every resulting record is a valid BenchmarkPrompt.
    - prompt_id values are unique within this ingestion run.
    - Source provenance is preserved by the adapter.
    """

    results: list[BenchmarkPrompt] = []
    seen_ids: set[str] = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Dataset record at index {index} must be a dictionary."
            )

        item = adapter.convert_record(
            record,
            source_split=source_split,
            source_id=str(index),
            benchmark_split=benchmark_split,
        )

        if not isinstance(item, BenchmarkPrompt):
            raise TypeError(
                "DatasetAdapter.convert_record() must return "
                "a BenchmarkPrompt."
            )

        if item.prompt_id in seen_ids:
            raise ValueError(
                f"Duplicate prompt_id encountered: {item.prompt_id}"
            )

        seen_ids.add(item.prompt_id)
        results.append(item)

    return results