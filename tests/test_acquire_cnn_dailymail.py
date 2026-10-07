import json

import pytest

from data_ingestion.acquire_cnn_dailymail import (
    DATASET_NAME,
    acquire_cnn_dailymail_split,
)


class FakeCNNDailyMailDatasetLoader:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        return [
            {
                "id": "abc123hash",
                "article": "First news story text.",
                "highlights": "First story summary.",
            },
            {
                "id": "def456hash",
                "article": "Second news story text.",
                "highlights": "Second story summary.",
            },
        ]


def test_acquire_cnn_dailymail_split(tmp_path):
    loader = FakeCNNDailyMailDatasetLoader()
    output = tmp_path / "cnn_dailymail.jsonl"

    result = acquire_cnn_dailymail_split(
        "test",
        config="3.0.0",
        benchmark_split="train",
        output_path=output,
        dataset_loader=loader,
    )

    assert result == output
    assert output.exists()

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]

    assert records[0]["prompt_id"] == "cnn_dailymail-3.0.0-test-abc123hash"
    assert records[1]["prompt_id"] == "cnn_dailymail-3.0.0-test-def456hash"

    assert records[0]["source_dataset"] == "cnn_dailymail"
    assert records[0]["source_config"] == "3.0.0"
    assert records[0]["source_split"] == "test"
    assert records[0]["split"] == "train"
    assert records[0]["task_type"] == "summarization"
    assert records[0]["domain"] == "news"
    assert records[0]["evaluation_type"] == "reference_based"
    assert records[0]["expected_output"] == "First story summary."
    assert "Summarize the following article:\n\nFirst news story text." in records[0]["prompt"]
    assert "First story summary." not in records[0]["prompt"]


def test_acquisition_calls_expected_dataset(tmp_path):
    loader = FakeCNNDailyMailDatasetLoader()

    acquire_cnn_dailymail_split(
        "validation",
        config="2.0.0",
        benchmark_split="validation",
        output_path=tmp_path / "unused.jsonl",
        dataset_loader=loader,
    )

    assert len(loader.calls) == 1
    args, kwargs = loader.calls[0]

    assert args == (
        DATASET_NAME,
        "2.0.0",
    )
    assert kwargs["split"] == "validation"


def test_invalid_source_split_is_rejected(tmp_path):
    loader = FakeCNNDailyMailDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported CNN/DailyMail source split",
    ):
        acquire_cnn_dailymail_split(
            "invalid_split",
            config="3.0.0",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_benchmark_split_is_rejected(tmp_path):
    loader = FakeCNNDailyMailDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported benchmark split",
    ):
        acquire_cnn_dailymail_split(
            "test",
            config="3.0.0",
            benchmark_split="invalid_split",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_config_is_rejected(tmp_path):
    loader = FakeCNNDailyMailDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported CNN/DailyMail config",
    ):
        acquire_cnn_dailymail_split(
            "test",
            config="9.9.9",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_output_parent_directory_is_created(tmp_path):
    loader = FakeCNNDailyMailDatasetLoader()
    output = tmp_path / "nested" / "dir" / "cnndm" / "test.jsonl"

    acquire_cnn_dailymail_split(
        "test",
        config="3.0.0",
        benchmark_split="test",
        output_path=output,
        dataset_loader=loader,
    )

    assert output.exists()
