import pytest

from data_ingestion.bbh import (
    BBHAdapter,
    determine_bbh_evaluation_type,
)


def test_determine_bbh_evaluation_type():
    assert determine_bbh_evaluation_type("(A)") == "multiple_choice"
    assert determine_bbh_evaluation_type("(B)") == "multiple_choice"
    assert determine_bbh_evaluation_type("(d)") == "multiple_choice"
    assert determine_bbh_evaluation_type("False") == "exact_match"
    assert determine_bbh_evaluation_type("24") == "exact_match"
    assert determine_bbh_evaluation_type("No") == "exact_match"
    assert determine_bbh_evaluation_type("] ]") == "exact_match"


def test_bbh_adapter_multiple_choice_record_conversion():
    adapter = BBHAdapter(source_config="date_understanding")

    record = {
        "input": "Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 12/11/1937\n(B) 12/25/1937",
        "target": "(B)",
    }

    item = adapter.convert_record(
        record,
        source_split="test",
        source_id="0",
        benchmark_split="train",
    )

    assert item.prompt_id == "bbh-date_understanding-test-0"
    assert "Today is Christmas Eve of 1937." in item.prompt
    assert item.task_type == "reasoning"
    assert item.domain == "date_understanding"
    assert item.difficulty == "hard"
    assert item.source_dataset == "bbh"
    assert item.source_config == "date_understanding"
    assert item.source_split == "test"
    assert item.source_id == "0"
    assert item.split == "train"
    assert item.expected_output == "(B)"
    assert item.evaluation_type == "multiple_choice"


def test_bbh_adapter_exact_match_record_conversion():
    adapter = BBHAdapter(source_config="boolean_expressions")

    record = {
        "input": "not ( True ) and ( True ) is",
        "target": "False",
    }

    item = adapter.convert_record(
        record,
        source_split="test",
        source_id="12",
        benchmark_split="test",
    )

    assert item.prompt_id == "bbh-boolean_expressions-test-12"
    assert item.prompt == "not ( True ) and ( True ) is"
    assert item.task_type == "reasoning"
    assert item.domain == "boolean_expressions"
    assert item.difficulty == "hard"
    assert item.source_dataset == "bbh"
    assert item.source_config == "boolean_expressions"
    assert item.source_split == "test"
    assert item.source_id == "12"
    assert item.split == "test"
    assert item.expected_output == "False"
    assert item.evaluation_type == "exact_match"


def test_bbh_adapter_unsupported_task_rejected():
    with pytest.raises(ValueError, match="Unsupported BBH task"):
        BBHAdapter(source_config="non_existent_task")

    with pytest.raises(ValueError, match="non-empty string"):
        BBHAdapter(source_config="   ")


def test_bbh_adapter_missing_input_rejected():
    adapter = BBHAdapter(source_config="boolean_expressions")

    with pytest.raises(ValueError, match="missing a valid 'input' prompt"):
        adapter.convert_record(
            {"target": "False"},
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="missing a valid 'input' prompt"):
        adapter.convert_record(
            {"input": "   ", "target": "False"},
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )


def test_bbh_adapter_missing_target_rejected():
    adapter = BBHAdapter(source_config="boolean_expressions")

    with pytest.raises(ValueError, match="missing a valid 'target' answer"):
        adapter.convert_record(
            {"input": "What is true?"},
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="missing a valid 'target' answer"):
        adapter.convert_record(
            {"input": "What is true?", "target": "   "},
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )


def test_bbh_adapter_invalid_record_type_rejected():
    adapter = BBHAdapter(source_config="boolean_expressions")

    with pytest.raises(ValueError, match="must be a dictionary"):
        adapter.convert_record(
            "not a dict",  # type: ignore[arg-type]
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )
