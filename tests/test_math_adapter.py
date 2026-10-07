import pytest

from data_ingestion.hendrycks_math import (
    HendrycksMATHAdapter,
    extract_boxed_answer,
    normalize_math_difficulty,
)


def test_extract_boxed_answer_simple():
    solution = "The answer is \\boxed{42}."
    assert extract_boxed_answer(solution) == "42"


def test_extract_boxed_answer_nested_braces():
    solution = "Therefore, $x = \\boxed{\\frac{3}{4}} + 1$."
    assert extract_boxed_answer(solution) == "\\frac{3}{4}"


def test_extract_boxed_answer_multiple():
    solution = "First \\boxed{1}, finally \\boxed{99}."
    # Takes the last one
    assert extract_boxed_answer(solution) == "99"


def test_extract_boxed_answer_missing():
    assert extract_boxed_answer("No boxed here") == ""


def test_normalize_math_difficulty():
    assert normalize_math_difficulty("Level 1") == "easy"
    assert normalize_math_difficulty("Level 2") == "easy"
    assert normalize_math_difficulty("Level 3") == "medium"
    assert normalize_math_difficulty("Level 4") == "medium"
    assert normalize_math_difficulty("Level 5") == "hard"
    assert normalize_math_difficulty(None) == "medium"


def test_math_record_is_converted():
    adapter = HendrycksMATHAdapter(source_config="algebra")

    record = {
        "problem": "Solve for x: 2x = 6.",
        "solution": "x = \\boxed{3}",
        "level": "Level 1",
        "type": "Algebra",
    }

    item = adapter.convert_record(
        record,
        source_split="train",
        source_id="42",
        benchmark_split="train",
    )

    assert item.prompt_id == "math-algebra-train-42"
    assert item.prompt == "Solve for x: 2x = 6."
    assert item.task_type == "math"
    assert item.domain == "algebra"
    assert item.difficulty == "easy"
    assert item.source_dataset == "math"
    assert item.source_config == "algebra"
    assert item.source_split == "train"
    assert item.source_id == "42"
    assert item.split == "train"
    assert item.expected_output == "3"
    assert item.evaluation_type == "exact_match"


def test_math_unsupported_config_is_rejected():
    with pytest.raises(ValueError, match="Unsupported MATH subject"):
        HendrycksMATHAdapter(source_config="invalid_subject")


def test_math_missing_problem_is_rejected():
    adapter = HendrycksMATHAdapter(source_config="algebra")
    with pytest.raises(ValueError, match="missing a valid 'problem'"):
        adapter.convert_record(
            {"solution": "x = 1"},
            source_split="train",
            source_id="1",
            benchmark_split="train",
        )
