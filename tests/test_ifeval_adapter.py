import json
import pytest

from data_ingestion.ifeval import (
    IFEvalAdapter,
    serialize_ifeval_constraints,
)


def test_serialize_ifeval_constraints():
    raw_kwargs = [
        {"num_highlights": 3, "relation": None, "num_words": None},
        {"num_words": 300, "relation": "at least", "language": None},
    ]
    instruction_ids = [
        "detectable_format:number_highlighted_sections",
        "length_constraints:number_words",
    ]

    serialized = serialize_ifeval_constraints(instruction_ids, raw_kwargs)
    parsed = json.loads(serialized)

    assert parsed["instruction_id_list"] == instruction_ids
    assert parsed["kwargs"] == [
        {"num_highlights": 3},
        {"num_words": 300, "relation": "at least"},
    ]


def test_ifeval_adapter_record_conversion():
    adapter = IFEvalAdapter(source_config="default")

    record = {
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
    }

    item = adapter.convert_record(
        record,
        source_split="train",
        source_id="0",
        benchmark_split="test",
    )

    assert item.prompt_id == "ifeval-default-train-1000"
    assert item.prompt == "Write a 300+ word summary without commas."
    assert item.task_type == "instruction_following"
    assert item.domain == "general_instruction"
    assert item.difficulty == "medium"
    assert item.source_dataset == "ifeval"
    assert item.source_config == "default"
    assert item.source_split == "train"
    assert item.source_id == "1000"
    assert item.split == "test"
    assert item.evaluation_type == "ifeval"

    expected = json.loads(item.expected_output)
    assert expected["instruction_id_list"] == [
        "punctuation:no_comma",
        "length_constraints:number_words",
    ]
    assert expected["kwargs"] == [{}, {"num_words": 300, "relation": "at least"}]


def test_ifeval_adapter_fallback_to_source_id_if_key_missing():
    adapter = IFEvalAdapter(source_config="default")

    record = {
        "prompt": "Write a poem.",
        "instruction_id_list": ["detectable_content:number_sentences"],
        "kwargs": [{"num_sentences": 5}],
    }

    item = adapter.convert_record(
        record,
        source_split="train",
        source_id="42",
        benchmark_split="train",
    )

    assert item.prompt_id == "ifeval-default-train-42"
    assert item.source_id == "42"


def test_ifeval_adapter_missing_prompt_rejected():
    adapter = IFEvalAdapter()

    with pytest.raises(ValueError, match="missing a valid prompt"):
        adapter.convert_record(
            {"instruction_id_list": ["a"], "kwargs": []},
            source_split="train",
            source_id="1",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="missing a valid prompt"):
        adapter.convert_record(
            {"prompt": "   ", "instruction_id_list": ["a"], "kwargs": []},
            source_split="train",
            source_id="1",
            benchmark_split="train",
        )


def test_ifeval_adapter_missing_instruction_id_list_rejected():
    adapter = IFEvalAdapter()

    with pytest.raises(ValueError, match="missing valid instruction_id_list"):
        adapter.convert_record(
            {"prompt": "Write a text.", "instruction_id_list": []},
            source_split="train",
            source_id="1",
            benchmark_split="train",
        )


def test_ifeval_adapter_invalid_record_type_rejected():
    adapter = IFEvalAdapter()

    with pytest.raises(ValueError, match="must be a dictionary"):
        adapter.convert_record(
            "not a dict",  # type: ignore[arg-type]
            source_split="train",
            source_id="1",
            benchmark_split="train",
        )


def test_ifeval_adapter_invalid_config_rejected():
    with pytest.raises(ValueError, match="source_config must be a non-empty string"):
        IFEvalAdapter(source_config="   ")
