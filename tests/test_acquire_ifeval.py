import json

import pytest

from data_ingestion.acquire_ifeval import (
    DATASET_NAME,
    acquire_ifeval_split,
)


class FakeIFEvalDatasetLoader:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        return [
            {
                "key": 1000,
                "prompt": "Write a 300+ word summary without commas.",
                "instruction_id_list": [
                    "punctuation:no_comma",
                    "length_constraints:number_words",
                ],
                "kwargs": [
                    {"num_highlights": None},
                    {"num_words": 300, "relation": "at least"},
                ],
            },
            {
                "key": 1001,
                "prompt": "Write a poem in Shakespearean style.",
                "instruction_id_list": ["punctuation:no_comma"],
                "kwargs": [{"language": None}],
            },
        ]


def test_acquire_ifeval_split(tmp_path):
    loader = FakeIFEvalDatasetLoader()
    output = tmp_path / "ifeval.jsonl"

    result = acquire_ifeval_split(
        "train",
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

    assert records[0]["prompt_id"] == "ifeval-default-train-1000"
    assert records[1]["prompt_id"] == "ifeval-default-train-1001"

    assert records[0]["source_dataset"] == "ifeval"
    assert records[0]["source_config"] == "default"
    assert records[0]["source_split"] == "train"
    assert records[0]["split"] == "train"
    assert records[0]["task_type"] == "instruction_following"
    assert records[0]["domain"] == "general_instruction"
    assert records[0]["evaluation_type"] == "ifeval"

    expected_0 = json.loads(records[0]["expected_output"])
    assert expected_0["instruction_id_list"] == [
        "punctuation:no_comma",
        "length_constraints:number_words",
    ]


def test_acquisition_calls_expected_dataset(tmp_path):
    loader = FakeIFEvalDatasetLoader()

    acquire_ifeval_split(
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
    loader = FakeIFEvalDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported IFEval source split",
    ):
        acquire_ifeval_split(
            "test",
            config="default",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_benchmark_split_is_rejected(tmp_path):
    loader = FakeIFEvalDatasetLoader()

    with pytest.raises(
        ValueError,
        match="Unsupported benchmark split",
    ):
        acquire_ifeval_split(
            "train",
            config="default",
            benchmark_split="invalid_split",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_invalid_config_is_rejected(tmp_path):
    loader = FakeIFEvalDatasetLoader()

    with pytest.raises(
        ValueError,
        match="IFEval config must be a non-empty string",
    ):
        acquire_ifeval_split(
            "train",
            config="   ",
            benchmark_split="train",
            output_path=tmp_path / "out.jsonl",
            dataset_loader=loader,
        )

    assert loader.calls == []


def test_output_parent_directory_is_created(tmp_path):
    loader = FakeIFEvalDatasetLoader()
    output = tmp_path / "nested" / "dir" / "ifeval" / "train.jsonl"

    acquire_ifeval_split(
        "train",
        config="default",
        benchmark_split="test",
        output_path=output,
        dataset_loader=loader,
    )

    assert output.exists()
