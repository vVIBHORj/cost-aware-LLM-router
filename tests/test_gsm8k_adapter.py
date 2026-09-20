import pytest

from data_ingestion.gsm8k import (
    GSM8KAdapter,
    extract_gsm8k_final_answer,
)


def test_extract_gsm8k_final_answer():
    answer = (
        "Natalia sold 48/2 = 24 clips in May.\n"
        "Natalia sold 48+24 = 72 clips altogether.\n"
        "#### 72"
    )

    assert extract_gsm8k_final_answer(answer) == "72"


def test_extract_gsm8k_final_answer_with_whitespace():
    answer = "Some reasoning.\n####   42   "

    assert extract_gsm8k_final_answer(answer) == "42"


def test_missing_final_answer_marker_is_rejected():
    answer = "The answer is 42."

    with pytest.raises(
        ValueError,
        match="final '####' answer marker",
    ):
        extract_gsm8k_final_answer(answer)


def test_empty_final_answer_is_rejected():
    answer = "Some reasoning.\n####   "

    with pytest.raises(
        ValueError,
        match="final answer marker is empty",
    ):
        extract_gsm8k_final_answer(answer)


def test_gsm8k_record_is_converted():
    adapter = GSM8KAdapter()

    record = {
        "question": "A shop has 12 apples and sells 5. How many remain?",
        "answer": "12 - 5 = 7.\n#### 7",
    }

    item = adapter.convert_record(
        record,
        source_split="test",
        source_id="123",
        benchmark_split="validation",
    )

    assert item.prompt_id == "gsm8k-main-test-123"
    assert item.prompt == record["question"]
    assert item.task_type == "math"
    assert item.domain == "grade_school_math"

    assert item.source_dataset == "gsm8k"
    assert item.source_config == "main"
    assert item.source_split == "test"
    assert item.source_id == "123"

    assert item.split == "validation"
    assert item.expected_output == "7"
    assert item.evaluation_type == "exact_match"


def test_gsm8k_missing_question_is_rejected():
    adapter = GSM8KAdapter()

    record = {
        "answer": "7",
    }

    with pytest.raises(ValueError, match="valid question"):
        adapter.convert_record(
            record,
            source_split="test",
            source_id="123",
            benchmark_split="train",
        )


def test_gsm8k_missing_answer_is_rejected():
    adapter = GSM8KAdapter()

    record = {
        "question": "What is 2 + 2?",
    }

    with pytest.raises(ValueError, match="valid answer"):
        adapter.convert_record(
            record,
            source_split="test",
            source_id="123",
            benchmark_split="train",
        )


def test_gsm8k_whitespace_question_is_rejected():
    adapter = GSM8KAdapter()

    record = {
        "question": "   ",
        "answer": "2 + 2 = 4.\n#### 4",
    }

    with pytest.raises(ValueError, match="valid question"):
        adapter.convert_record(
            record,
            source_split="test",
            source_id="123",
            benchmark_split="train",
        )