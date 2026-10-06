from pathlib import Path
from typing import Any, Callable

from data_ingestion.ifeval import IFEvalAdapter
from data_ingestion.runner import ingest_records
from data_ingestion.writer import write_benchmark_jsonl

DATASET_NAME = "google/IFEval"
SUPPORTED_SOURCE_SPLITS = {"train"}
SUPPORTED_BENCHMARK_SPLITS = {"train", "validation", "test", "stress"}


def acquire_ifeval_split(
    source_split: str = "train",
    *,
    config: str = "default",
    benchmark_split: str,
    output_path: str | Path,
    dataset_loader: Callable[..., Any] | None = None,
) -> Path:
    """
    Download/load one IFEval split, convert it into BenchmarkPrompt
    objects, and write the standardized JSONL output.

    dataset_loader is injectable so this function can be tested without
    network access.
    """

    if not isinstance(config, str) or not config.strip():
        raise ValueError("IFEval config must be a non-empty string.")

    config = config.strip()

    if source_split not in SUPPORTED_SOURCE_SPLITS:
        raise ValueError(
            f"Unsupported IFEval source split: '{source_split}'. Supported splits: {sorted(SUPPORTED_SOURCE_SPLITS)}"
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
        config,
        split=source_split,
    )

    records = dataset

    adapter = IFEvalAdapter(source_config=config)

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
