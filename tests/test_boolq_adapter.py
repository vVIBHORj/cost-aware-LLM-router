import pytest

from data_ingestion.boolq import (
    BoolQAdapter,
    format_boolq_prompt,
    normalize_boolq_answer,
)


def test_normalize_boolq_answer():
    assert normalize_boolq_answer(True) == "True"
    assert normalize_boolq_answer(False) == "False"
    assert normalize_boolq_answer("true") == "True"
    assert normalize_boolq_answer("False") == "False"
    assert normalize_boolq_answer(" TRUE ") == "True"


def test_normalize_boolq_answer_invalid():
    with pytest.raises(ValueError, match="Invalid BoolQ answer value"):
        normalize_boolq_answer(None)

    with pytest.raises(ValueError, match="Invalid BoolQ answer value"):
        normalize_boolq_answer("yes")

    with pytest.raises(ValueError, match="Invalid BoolQ answer value"):
        normalize_boolq_answer(1)


def test_format_boolq_prompt():
    passage = "Earth revolves around the Sun."
    question = "is the earth orbiting the sun"
    prompt = format_boolq_prompt(passage, question)
    expected = (
        "Passage:\n"
        "Earth revolves around the Sun.\n\n"
        "Question:\n"
        "is the earth orbiting the sun"
    )
    assert prompt == expected


def test_format_boolq_prompt_invalid():
    with pytest.raises(ValueError, match="missing a valid 'passage'"):
        format_boolq_prompt("", "A question?")

    with pytest.raises(ValueError, match="missing a valid 'question'"):
        format_boolq_prompt("A passage.", "   ")


def test_boolq_adapter_record_conversion():
    adapter = BoolQAdapter(source_config="default")

    record = {
        "passage": "All biomass goes through at least some of these steps.",
        "question": "does ethanol take more energy make that produces",
        "answer": False,
    }

    item = adapter.convert_record(
        record,
        source_split="validation",
        source_id="42",
        benchmark_split="test",
    )

    assert item.prompt_id == "boolq-default-validation-42"
    assert "Passage:\nAll biomass goes through" in item.prompt
    assert "Question:\ndoes ethanol take more energy make that produces" in item.prompt
    assert item.task_type == "classification"
    assert item.domain == "reading_comprehension"
    assert item.difficulty == "medium"
    assert item.source_dataset == "boolq"
    assert item.source_config == "default"
    assert item.source_split == "validation"
    assert item.source_id == "42"
    assert item.split == "test"
    assert item.expected_output == "False"
    assert item.evaluation_type == "exact_match"


def test_boolq_adapter_missing_fields_rejected():
    adapter = BoolQAdapter()

    with pytest.raises(ValueError, match="missing a valid 'passage'"):
        adapter.convert_record(
            {"question": "Valid?", "answer": True},
            source_split="train",
            source_id="0",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="missing a valid 'question'"):
        adapter.convert_record(
            {"passage": "Valid passage.", "answer": True},
            source_split="train",
            source_id="0",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="Invalid BoolQ answer value"):
        adapter.convert_record(
            {"passage": "Valid.", "question": "Valid?", "answer": None},
            source_split="train",
            source_id="0",
            benchmark_split="train",
        )


def test_boolq_adapter_invalid_record_type_rejected():
    adapter = BoolQAdapter()

    with pytest.raises(ValueError, match="must be a dictionary"):
        adapter.convert_record(
            "not a dict",  # type: ignore[arg-type]
            source_split="train",
            source_id="0",
            benchmark_split="train",
        )


def test_boolq_adapter_invalid_config_rejected():
    with pytest.raises(ValueError, match="source_config must be a non-empty string"):
        BoolQAdapter(source_config="   ")
