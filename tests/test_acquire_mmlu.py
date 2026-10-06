import json

import pytest

from data_ingestion.acquire_mmlu import (
    DATASET_NAME,
    acquire_mmlu_split,
)


class FakeMMLUDatasetLoader:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        return [
            {
                "question": "What is 2 + 2?",
                "subject": "elementary_mathematics",
                "choices": ["1", "2", "3", "4"],
                "answer": 3,
            },
            {
                "question": "What is the capital of France?",
                "subject": "elementary_mathematics",
                "choices": ["Paris", "Rome", "Madrid", "Berlin"],
                "answer": 0,
            },
        ]


def test_acquire_mmlu_split(tmp_path):
    loader = FakeMMLUDatasetLoader()
    output = tmp_path / "mmlu.jsonl"

    result = acquire_mmlu_split(
        subject="elementary_mathematics",
        source_split="test",
        benchmark_split="train",
        output_path=output,
        dataset_loader=loader,
    )

    assert result == output
    assert output.exists()

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]

    assert records[0]["prompt_id"] == "mmlu-elementary_mathematics-test-0"
    assert records[1]["prompt_id"] == "mmlu-elementary_mathematics-test-1"

    assert records[0]["source_dataset"] == "mmlu"
    assert records[0]["source_config"] == "elementary_mathematics"
    assert records[0]["source_split"] == "test"
    assert records[0]["split"] == "train"
    assert records[0]["task_type"] == "qa"
    assert records[0]["domain"] == "elementary_mathematics"
    assert records[0]["evaluation_type"] == "multiple_choice"
    assert records[0]["expected_output"] == "D"
    assert records[1]["expected_output"] == "A"


def test_acquisition_calls_expected_dataset(tmp_path):
    loader = FakeMMLUDatasetLoader()

    acquire_mmlu_split(
        subject="anatomy",
        source_split="dev",
        benchmark_split="validation",
        output_path=tmp_path / "unused.jsonl",
        dataset_loader=loader,
    )

    assert len(loader.calls) == 1
    args, kwargs = loader.calls[0]

    assert args == (
        DATASET_NAME,
        "anatomy",
    )
    assert kwargs["split"] == "dev"


def test_invalid_source_split_is_rejected(tmp_path):
    loader = FakeMMLUDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported MMLU source split",
    ):
        acquire_mmlu_split(
            subject="anatomy",
            source_split="invalid_split",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_benchmark_split_is_rejected(tmp_path):
    loader = FakeMMLUDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported benchmark split",
    ):
        acquire_mmlu_split(
            subject="anatomy",
            source_split="test",
            benchmark_split="invalid_split",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_subject_is_rejected(tmp_path):
    loader = FakeMMLUDatasetLoader()

    with pytest.raises(
        ValueError,
        match="MMLU subject / configuration must be a non-empty string",
    ):
        acquire_mmlu_split(
            subject="   ",
            source_split="test",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_output_parent_directory_is_created(tmp_path):
    loader = FakeMMLUDatasetLoader()
    output = tmp_path / "nested" / "dir" / "mmlu" / "test.jsonl"

    acquire_mmlu_split(
        subject="anatomy",
        source_split="test",
        benchmark_split="test",
        output_path=output,
        dataset_loader=loader,
    )

    assert output.exists()
