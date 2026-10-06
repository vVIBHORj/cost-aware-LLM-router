import json

import pytest

from data_ingestion.acquire_gsm8k import (
    DATASET_CONFIG,
    DATASET_NAME,
    acquire_gsm8k_split,
)


class FakeDatasetLoader:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        return [
            {
                "question": "What is 2 + 2?",
                "answer": "2 + 2 = 4.\n#### 4",
            },
            {
                "question": "What is 10 + 5?",
                "answer": "10 + 5 = 15.\n#### 15",
            },
        ]


def test_acquire_gsm8k_split(tmp_path):
    loader = FakeDatasetLoader()

    output = tmp_path / "gsm8k.jsonl"

    result = acquire_gsm8k_split(
        "train",
        benchmark_split="train",
        output_path=output,
        dataset_loader=loader,
    )

    assert result == output
    assert output.exists()

    lines = output.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2

    records = [
        json.loads(line)
        for line in lines
    ]

    assert records[0]["prompt_id"] == "gsm8k-main-train-0"
    assert records[1]["prompt_id"] == "gsm8k-main-train-1"

    assert records[0]["source_dataset"] == "gsm8k"
    assert records[0]["source_config"] == "main"
    assert records[0]["source_split"] == "train"
    assert records[0]["split"] == "train"


def test_acquisition_calls_expected_dataset(tmp_path):
    loader = FakeDatasetLoader()

    acquire_gsm8k_split(
        "test",
        benchmark_split="validation",
        output_path=tmp_path / "unused.jsonl",
        dataset_loader=loader,
    )

    assert len(loader.calls) == 1

    args, kwargs = loader.calls[0]

    assert args == (
        DATASET_NAME,
        DATASET_CONFIG,
    )

    assert kwargs["split"] == "test"


def test_invalid_source_split_is_rejected(tmp_path):
    loader = FakeDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported GSM8K source split",
    ):
        acquire_gsm8k_split(
            "validation",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_benchmark_split_is_rejected(tmp_path):
    loader = FakeDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported benchmark split",
    ):
        acquire_gsm8k_split(
            "train",
            benchmark_split="invalid",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_output_preserves_expected_answers(tmp_path):
    loader = FakeDatasetLoader()

    output = tmp_path / "gsm8k.jsonl"

    acquire_gsm8k_split(
        "train",
        benchmark_split="train",
        output_path=output,
        dataset_loader=loader,
    )

    records = [
        json.loads(line)
        for line in output.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert records[0]["expected_output"] == "4"
    assert records[1]["expected_output"] == "15"


def test_output_parent_directory_is_created(tmp_path):
    loader = FakeDatasetLoader()

    output = (
        tmp_path
        / "processed"
        / "gsm8k"
        / "train.jsonl"
    )

    acquire_gsm8k_split(
        "train",
        benchmark_split="train",
        output_path=output,
        dataset_loader=loader,
    )

    assert output.exists()