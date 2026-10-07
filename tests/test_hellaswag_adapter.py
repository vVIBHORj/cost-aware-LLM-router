import pytest

from data_ingestion.hellaswag import HellaSwagAdapter


def test_hellaswag_record_is_converted():
    adapter = HellaSwagAdapter()

    record = {
        "ind": 12,
        "activity_label": "Baking cookies",
        "ctx": "A person mixes flour and sugar in a bowl.",
        "endings": [
            "Then they put the mixture in the oven.",
            "They jump on the table.",
            "The car drives away.",
            "It starts raining.",
        ],
        "label": "0",
    }

    item = adapter.convert_record(
        record,
        source_split="validation",
        source_id="12",
        benchmark_split="validation",
    )

    assert item.prompt_id == "hellaswag-validation-12"
    assert "A person mixes flour" in item.prompt
    assert "(A) Then they put the mixture in the oven." in item.prompt
    assert item.task_type == "other"
    assert item.domain == "baking_cookies"
    assert item.difficulty == "medium"
    assert item.source_dataset == "hellaswag"
    assert item.source_config == "default"
    assert item.source_split == "validation"
    assert item.source_id == "12"
    assert item.split == "validation"
    assert item.expected_output == "(A)"
    assert item.evaluation_type == "multiple_choice"


def test_hellaswag_invalid_label_is_rejected():
    adapter = HellaSwagAdapter()
    record = {
        "ind": 1,
        "ctx": "Some text",
        "endings": ["1", "2", "3", "4"],
        "label": "9",  # out of range
    }
    with pytest.raises(ValueError, match="out of range"):
        adapter.convert_record(
            record,
            source_split="train",
            source_id="1",
            benchmark_split="train",
        )


def test_hellaswag_missing_label_is_rejected():
    adapter = HellaSwagAdapter()
    record = {
        "ind": 1,
        "ctx": "Some text",
        "endings": ["1", "2", "3", "4"],
        "label": "",
    }
    with pytest.raises(ValueError, match="missing a valid label"):
        adapter.convert_record(
            record,
            source_split="train",
            source_id="1",
            benchmark_split="train",
        )
