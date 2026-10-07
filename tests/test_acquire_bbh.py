import json

import pytest

from data_ingestion.acquire_bbh import (
    DATASET_NAME,
    acquire_bbh_split,
)


class FakeBBHDatasetLoader:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        return [
            {
                "input": "Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 12/11/1937\n(B) 12/25/1937",
                "target": "(B)",
            },
            {
                "input": "Today is New Year's Eve of 2000. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 01/01/2001\n(B) 12/31/2000",
                "target": "(A)",
            },
        ]


def test_acquire_bbh_split(tmp_path):
    loader = FakeBBHDatasetLoader()
    output = tmp_path / "bbh.jsonl"

    result = acquire_bbh_split(
        task="date_understanding",
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

    assert records[0]["prompt_id"] == "bbh-date_understanding-test-0"
    assert records[1]["prompt_id"] == "bbh-date_understanding-test-1"

    assert records[0]["source_dataset"] == "bbh"
    assert records[0]["source_config"] == "date_understanding"
    assert records[0]["source_split"] == "test"
    assert records[0]["split"] == "train"
    assert records[0]["task_type"] == "reasoning"
    assert records[0]["domain"] == "date_understanding"
    assert records[0]["evaluation_type"] == "multiple_choice"
    assert records[0]["expected_output"] == "(B)"
    assert records[1]["expected_output"] == "(A)"


def test_acquisition_calls_expected_dataset(tmp_path):
    loader = FakeBBHDatasetLoader()

    acquire_bbh_split(
        task="boolean_expressions",
        source_split="test",
        benchmark_split="validation",
        output_path=tmp_path / "unused.jsonl",
        dataset_loader=loader,
    )

    assert len(loader.calls) == 1
    args, kwargs = loader.calls[0]

    assert args == (
        DATASET_NAME,
        "boolean_expressions",
    )
    assert kwargs["split"] == "test"


def test_invalid_source_split_is_rejected(tmp_path):
    loader = FakeBBHDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported BBH source split",
    ):
        acquire_bbh_split(
            task="boolean_expressions",
            source_split="invalid_split",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_benchmark_split_is_rejected(tmp_path):
    loader = FakeBBHDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported benchmark split",
    ):
        acquire_bbh_split(
            task="boolean_expressions",
            source_split="test",
            benchmark_split="invalid_split",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_task_is_rejected(tmp_path):
    loader = FakeBBHDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported BBH task",
    ):
        acquire_bbh_split(
            task="non_existent_task",
            source_split="test",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_output_parent_directory_is_created(tmp_path):
    loader = FakeBBHDatasetLoader()
    output = tmp_path / "nested" / "dir" / "bbh" / "test.jsonl"

    acquire_bbh_split(
        task="geometric_shapes",
        source_split="test",
        benchmark_split="test",
        output_path=output,
        dataset_loader=loader,
    )

    assert output.exists()
