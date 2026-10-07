from pathlib import Path
from typing import Any, Callable

from data_ingestion.runner import ingest_records
from data_ingestion.super_glue import SUPPORTED_TASKS, SuperGlueAdapter
from data_ingestion.writer import write_benchmark_jsonl

DATASET_NAME = "aps/super_glue"
SUPPORTED_SOURCE_SPLITS = {"train", "validation", "test"}
SUPPORTED_BENCHMARK_SPLITS = {"train", "validation", "test", "stress"}


def acquire_super_glue_split(
    task: str,
    source_split: str,
    *,
    benchmark_split: str,
    output_path: str | Path,
    dataset_loader: Callable[..., Any] | None = None,
) -> Path:
    """
    Download/load one SuperGLUE task split, convert it into BenchmarkPrompt
    objects, and write the standardized JSONL output.

    dataset_loader is injectable so this function can be tested without
    network access.
    """

    if not isinstance(task, str) or not task.strip():
        raise ValueError("SuperGLUE task must be a non-empty string.")

    task = task.strip()
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported SuperGLUE task: '{task}'. Supported tasks: {sorted(SUPPORTED_TASKS)}"
        )

    if source_split not in SUPPORTED_SOURCE_SPLITS:
        raise ValueError(
            f"Unsupported SuperGLUE source split: '{source_split}'. Supported splits: {sorted(SUPPORTED_SOURCE_SPLITS)}"
        )

    if benchmark_split not in SUPPORTED_BENCHMARK_SPLITS:
        raise ValueError(
            f"Unsupported benchmark split: '{benchmark_split}'. Supported splits: {sorted(SUPPORTED_BENCHMARK_SPLITS)}"
        )

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    dataset = dataset_loader(
        DATASET_NAME,
        task,
        split=source_split,
    )

    records = dataset

    adapter = SuperGlueAdapter(source_config=task)

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
