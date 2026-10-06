import pytest

from data_ingestion.mmlu import (
    MMLUAdapter,
    convert_mmlu_answer,
    format_mmlu_prompt,
)


def test_convert_mmlu_answer_integer():
    assert convert_mmlu_answer(0) == "A"
    assert convert_mmlu_answer(1) == "B"
    assert convert_mmlu_answer(2) == "C"
    assert convert_mmlu_answer(3) == "D"


def test_convert_mmlu_answer_string():
    assert convert_mmlu_answer("A") == "A"
    assert convert_mmlu_answer("b") == "B"
    assert convert_mmlu_answer("2") == "C"
    assert convert_mmlu_answer("  D  ") == "D"


def test_convert_mmlu_answer_invalid():
    with pytest.raises(ValueError, match="Boolean value"):
        convert_mmlu_answer(True)

    with pytest.raises(ValueError, match="out of range"):
        convert_mmlu_answer(4, num_choices=4)

    with pytest.raises(ValueError, match="out of range"):
        convert_mmlu_answer(-1, num_choices=4)

    with pytest.raises(ValueError, match="invalid"):
        convert_mmlu_answer("E", num_choices=4)

    with pytest.raises(ValueError, match="Invalid MMLU answer type"):
        convert_mmlu_answer(None)


def test_format_mmlu_prompt():
    prompt = format_mmlu_prompt(
        "What is the capital of France?",
        ["London", "Berlin", "Paris", "Madrid"],
    )
    expected = (
        "What is the capital of France?\n"
        "A. London\n"
        "B. Berlin\n"
        "C. Paris\n"
        "D. Madrid"
    )
    assert prompt == expected


def test_format_mmlu_prompt_invalid():
    with pytest.raises(ValueError, match="valid question"):
        format_mmlu_prompt("", ["A", "B"])

    with pytest.raises(ValueError, match="at least 2 options"):
        format_mmlu_prompt("Question?", ["Only one"])

    with pytest.raises(ValueError, match="non-empty string"):
        format_mmlu_prompt("Question?", ["A", "  "])


def test_mmlu_adapter_record_conversion():
    adapter = MMLUAdapter(source_config="anatomy")

    record = {
        "question": "What is the embryological origin of the hyoid bone?",
        "subject": "anatomy",
        "choices": [
            "The first pharyngeal arch",
            "The first and second pharyngeal arches",
            "The second pharyngeal arch",
            "The second and third pharyngeal arches",
        ],
        "answer": 3,
    }

    item = adapter.convert_record(
        record,
        source_split="test",
        source_id="42",
        benchmark_split="train",
    )

    assert item.prompt_id == "mmlu-anatomy-test-42"
    assert "What is the embryological origin of the hyoid bone?" in item.prompt
    assert "D. The second and third pharyngeal arches" in item.prompt
    assert item.task_type == "qa"
    assert item.domain == "anatomy"
    assert item.difficulty == "medium"
    assert item.source_dataset == "mmlu"
    assert item.source_config == "anatomy"
    assert item.source_split == "test"
    assert item.source_id == "42"
    assert item.split == "train"
    assert item.expected_output == "D"
    assert item.evaluation_type == "multiple_choice"


def test_mmlu_adapter_missing_fields():
    adapter = MMLUAdapter(source_config="anatomy")

    with pytest.raises(ValueError, match="valid question"):
        adapter.convert_record(
            {
                "choices": ["A", "B"],
                "answer": 0,
            },
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="choices"):
        adapter.convert_record(
            {
                "question": "Test?",
                "choices": [],
                "answer": 0,
            },
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="answer"):
        adapter.convert_record(
            {
                "question": "Test?",
                "choices": ["A", "B"],
            },
            source_split="test",
            source_id="0",
            benchmark_split="train",
        )
