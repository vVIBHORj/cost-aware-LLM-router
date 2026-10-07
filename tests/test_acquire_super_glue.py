import json

import pytest

from data_ingestion.acquire_super_glue import (
    DATASET_NAME,
    acquire_super_glue_split,
)


class FakeSuperGlueDatasetLoader:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        return [
            {
                "premise": "First premise text.",
                "hypothesis": "First hypothesis text.",
                "idx": 101,
                "label": 0,
            },
            {
                "premise": "Second premise text.",
                "hypothesis": "Second hypothesis text.",
                "idx": 102,
                "label": 1,
            },
        ]


def test_acquire_super_glue_split(tmp_path):
    loader = FakeSuperGlueDatasetLoader()
    output = tmp_path / "super_glue.jsonl"

    result = acquire_super_glue_split(
        task="cb",
        source_split="validation",
        benchmark_split="train",
        output_path=output,
        dataset_loader=loader,
    )

    assert result == output
    assert output.exists()

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]

    assert records[0]["prompt_id"] == "super_glue-cb-validation-101"
    assert records[1]["prompt_id"] == "super_glue-cb-validation-102"

    assert records[0]["source_dataset"] == "super_glue"
    assert records[0]["source_config"] == "cb"
    assert records[0]["source_split"] == "validation"
    assert records[0]["split"] == "train"
    assert records[0]["task_type"] == "classification"
    assert records[0]["domain"] == "super_glue_cb"
    assert records[0]["evaluation_type"] == "exact_match"
    assert records[0]["expected_output"] == "entailment"
    assert records[1]["expected_output"] == "contradiction"


def test_acquisition_calls_expected_dataset(tmp_path):
    loader = FakeSuperGlueDatasetLoader()

    acquire_super_glue_split(
        task="rte",
        source_split="train",
        benchmark_split="validation",
        output_path=tmp_path / "unused.jsonl",
        dataset_loader=loader,
    )

    assert len(loader.calls) == 1
    args, kwargs = loader.calls[0]

    assert args == (
        DATASET_NAME,
        "rte",
    )
    assert kwargs["split"] == "train"


def test_invalid_source_split_is_rejected(tmp_path):
    loader = FakeSuperGlueDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported SuperGLUE source split",
    ):
        acquire_super_glue_split(
            task="cb",
            source_split="invalid_split",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_benchmark_split_is_rejected(tmp_path):
    loader = FakeSuperGlueDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported benchmark split",
    ):
        acquire_super_glue_split(
            task="cb",
            source_split="train",
            benchmark_split="invalid_split",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_task_is_rejected(tmp_path):
    loader = FakeSuperGlueDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported SuperGLUE task",
    ):
        acquire_super_glue_split(
            task="unknown_task",
            source_split="train",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_output_parent_directory_is_created(tmp_path):
    loader = FakeSuperGlueDatasetLoader()
    output = tmp_path / "nested" / "dir" / "superglue" / "cb.jsonl"

    acquire_super_glue_split(
        task="cb",
        source_split="test",
        benchmark_split="test",
        output_path=output,
        dataset_loader=loader,
    )

    assert output.exists()
