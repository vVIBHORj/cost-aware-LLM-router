import json

import pytest

from data_ingestion.acquire_mbpp import (
    DATASET_NAME,
    acquire_mbpp_split,
)


class FakeMBPPDatasetLoader:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        return [
            {
                "task_id": 101,
                "text": "Write a python function to add two numbers.",
                "code": "def add(a, b):\n    return a + b",
                "test_list": ["assert add(1, 2) == 3"],
            },
            {
                "task_id": 102,
                "text": "Write a python function to multiply two numbers.",
                "code": "def mul(a, b):\n    return a * b",
                "test_list": ["assert mul(2, 3) == 6"],
            },
        ]


def test_acquire_mbpp_split(tmp_path):
    loader = FakeMBPPDatasetLoader()
    output = tmp_path / "mbpp.jsonl"

    result = acquire_mbpp_split(
        "test",
        config="full",
        benchmark_split="train",
        output_path=output,
        dataset_loader=loader,
    )

    assert result == output
    assert output.exists()

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]

    assert records[0]["prompt_id"] == "mbpp-full-test-101"
    assert records[1]["prompt_id"] == "mbpp-full-test-102"

    assert records[0]["source_dataset"] == "mbpp"
    assert records[0]["source_config"] == "full"
    assert records[0]["source_split"] == "test"
    assert records[0]["split"] == "train"
    assert records[0]["task_type"] == "coding"
    assert records[0]["domain"] == "python"
    assert records[0]["evaluation_type"] == "code_execution"
    assert records[0]["expected_output"] == "def add(a, b):\n    return a + b"


def test_acquisition_calls_expected_dataset(tmp_path):
    loader = FakeMBPPDatasetLoader()

    acquire_mbpp_split(
        "validation",
        config="sanitized",
        benchmark_split="validation",
        output_path=tmp_path / "unused.jsonl",
        dataset_loader=loader,
    )

    assert len(loader.calls) == 1
    args, kwargs = loader.calls[0]

    assert args == (
        DATASET_NAME,
        "sanitized",
    )
    assert kwargs["split"] == "validation"


def test_invalid_source_split_is_rejected(tmp_path):
    loader = FakeMBPPDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported MBPP source split",
    ):
        acquire_mbpp_split(
            "unknown_split",
            config="full",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_benchmark_split_is_rejected(tmp_path):
    loader = FakeMBPPDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported benchmark split",
    ):
        acquire_mbpp_split(
            "test",
            config="full",
            benchmark_split="invalid_split",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_config_is_rejected(tmp_path):
    loader = FakeMBPPDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported MBPP config",
    ):
        acquire_mbpp_split(
            "test",
            config="invalid_config",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_output_parent_directory_is_created(tmp_path):
    loader = FakeMBPPDatasetLoader()
    output = tmp_path / "nested" / "dir" / "mbpp" / "test.jsonl"

    acquire_mbpp_split(
        "test",
        config="full",
        benchmark_split="test",
        output_path=output,
        dataset_loader=loader,
    )

    assert output.exists()
