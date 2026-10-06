from pathlib import Path
from typing import Any, Callable

from data_ingestion.mmlu import MMLUAdapter
from data_ingestion.runner import ingest_records
from data_ingestion.writer import write_benchmark_jsonl

DATASET_NAME = "cais/mmlu"


def acquire_mmlu_split(
    subject: str,
    source_split: str,
    *,
    benchmark_split: str,
    output_path: str | Path,
    dataset_loader: Callable[..., Any] | None = None,
) -> Path:
    """
    Download/load one MMLU subject split, convert it into BenchmarkPrompt
    objects, and write the standardized JSONL output.

    dataset_loader is injectable so this function can be tested without
    network access.
    """

    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("MMLU subject / configuration must be a non-empty string.")

    subject = subject.strip()

    if source_split not in {"test", "validation", "dev"}:
        raise ValueError(
            f"Unsupported MMLU source split: {source_split}"
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
        subject,
        split=source_split,
    )

    records = dataset

    adapter = MMLUAdapter(source_config=subject)

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
