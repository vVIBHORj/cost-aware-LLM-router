import json

import pytest

from data_ingestion.acquire_boolq import (
    DATASET_NAME,
    acquire_boolq_split,
)


class FakeBoolQDatasetLoader:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        return [
            {
                "passage": "First passage content.",
                "question": "is this first question",
                "answer": True,
            },
            {
                "passage": "Second passage content.",
                "question": "is this second question",
                "answer": False,
            },
        ]


def test_acquire_boolq_split(tmp_path):
    loader = FakeBoolQDatasetLoader()
    output = tmp_path / "boolq.jsonl"

    result = acquire_boolq_split(
        "validation",
        config="default",
        benchmark_split="train",
        output_path=output,
        dataset_loader=loader,
    )

    assert result == output
    assert output.exists()

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]

    assert records[0]["prompt_id"] == "boolq-default-validation-0"
    assert records[1]["prompt_id"] == "boolq-default-validation-1"

    assert records[0]["source_dataset"] == "boolq"
    assert records[0]["source_config"] == "default"
    assert records[0]["source_split"] == "validation"
    assert records[0]["split"] == "train"
    assert records[0]["task_type"] == "classification"
    assert records[0]["domain"] == "reading_comprehension"
    assert records[0]["evaluation_type"] == "exact_match"
    assert records[0]["expected_output"] == "True"
    assert records[1]["expected_output"] == "False"
    assert "Passage:\nFirst passage content." in records[0]["prompt"]
    assert "Question:\nis this first question" in records[0]["prompt"]


def test_acquisition_calls_expected_dataset(tmp_path):
    loader = FakeBoolQDatasetLoader()

    acquire_boolq_split(
        "train",
        config="default",
        benchmark_split="validation",
        output_path=tmp_path / "unused.jsonl",
        dataset_loader=loader,
    )

    assert len(loader.calls) == 1
    args, kwargs = loader.calls[0]

    assert args == (
        DATASET_NAME,
        "default",
    )
    assert kwargs["split"] == "train"


def test_invalid_source_split_is_rejected(tmp_path):
    loader = FakeBoolQDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported BoolQ source split",
    ):
        acquire_boolq_split(
            "test",
            config="default",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_benchmark_split_is_rejected(tmp_path):
    loader = FakeBoolQDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported benchmark split",
    ):
        acquire_boolq_split(
            "validation",
            config="default",
            benchmark_split="invalid_split",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_config_is_rejected(tmp_path):
    loader = FakeBoolQDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported BoolQ config",
    ):
        acquire_boolq_split(
            "validation",
            config="unsupported_config",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_output_parent_directory_is_created(tmp_path):
    loader = FakeBoolQDatasetLoader()
    output = tmp_path / "nested" / "dir" / "boolq" / "val.jsonl"

    acquire_boolq_split(
        "validation",
        config="default",
        benchmark_split="test",
        output_path=output,
        dataset_loader=loader,
    )

    assert output.exists()
