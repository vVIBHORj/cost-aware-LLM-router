from pathlib import Path
from typing import Any, Callable

from data_ingestion.gsm8k import GSM8KAdapter
from data_ingestion.runner import ingest_records
from data_ingestion.writer import write_benchmark_jsonl


DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"


def acquire_gsm8k_split(
    source_split: str,
    *,
    benchmark_split: str,
    output_path: str | Path,
    dataset_loader: Callable[..., Any] | None = None,
) -> Path:
    """
    Download/load one GSM8K split, convert it into BenchmarkPrompt
    objects, and write the standardized JSONL output.

    dataset_loader is injectable so this function can be tested without
    network access.
    """

    if source_split not in {"train", "test"}:
        raise ValueError(
            f"Unsupported GSM8K source split: {source_split}"
        )

    if benchmark_split not in {
        "train",
        "validation",
        "test",
        "stress",
    }:
        raise ValueError(
            f"Unsupported benchmark split: {benchmark_split}"
        )

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    dataset = dataset_loader(
        DATASET_NAME,
        DATASET_CONFIG,
        split=source_split,
    )

    records = dataset

    adapter = GSM8KAdapter()

    items = ingest_records(
        records,
        adapter,
        source_split=source_split,
        benchmark_split=benchmark_split,
    )

    return write_benchmark_jsonl(
        items,
        output_path,
    )