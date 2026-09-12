import pytest
from pydantic import ValidationError

from router.benchmark_schema import BenchmarkPrompt


def valid_prompt() -> dict:
    return {
        "prompt_id": "test-001",
        "prompt": "What is 2 + 2?",
        "task_type": "math",
        "domain": "arithmetic",
        "difficulty": "easy",
        "source_dataset": "gsm8k",
        "source_config": "main",
        "source_split": "test",
        "source_id": "123",
        "split": "train",
        "expected_output": "4",
        "evaluation_type": "exact_match",
    }


def test_valid_prompt_is_accepted():
    item = BenchmarkPrompt.model_validate(valid_prompt())

    assert item.prompt_id == "test-001"
    assert item.source_dataset == "gsm8k"
    assert item.source_config == "main"
    assert item.source_split == "test"
    assert item.source_id == "123"
    assert item.split == "train"
    assert item.evaluation_type == "exact_match"


def test_prompt_id_is_required():
    data = valid_prompt()
    del data["prompt_id"]

    with pytest.raises(ValidationError):
        BenchmarkPrompt.model_validate(data)


def test_prompt_cannot_be_empty():
    data = valid_prompt()
    data["prompt"] = ""

    with pytest.raises(ValidationError):
        BenchmarkPrompt.model_validate(data)


def test_whitespace_prompt_is_rejected():
    data = valid_prompt()
    data["prompt"] = "   "

    with pytest.raises(ValidationError):
        BenchmarkPrompt.model_validate(data)


def test_source_dataset_cannot_be_empty():
    data = valid_prompt()
    data["source_dataset"] = ""

    with pytest.raises(ValidationError):
        BenchmarkPrompt.model_validate(data)


def test_source_split_cannot_be_empty():
    data = valid_prompt()
    data["source_split"] = "   "

    with pytest.raises(ValidationError):
        BenchmarkPrompt.model_validate(data)


def test_source_id_cannot_be_empty():
    data = valid_prompt()
    data["source_id"] = ""

    with pytest.raises(ValidationError):
        BenchmarkPrompt.model_validate(data)


def test_invalid_task_type_is_rejected():
    data = valid_prompt()
    data["task_type"] = "invalid_task"

    with pytest.raises(ValidationError):
        BenchmarkPrompt.model_validate(data)


def test_invalid_split_is_rejected():
    data = valid_prompt()
    data["split"] = "invalid_split"

    with pytest.raises(ValidationError):
        BenchmarkPrompt.model_validate(data)


def test_invalid_evaluation_type_is_rejected():
    data = valid_prompt()
    data["evaluation_type"] = "invalid_evaluation"

    with pytest.raises(ValidationError):
        BenchmarkPrompt.model_validate(data)


def test_optional_source_config_can_be_none():
    data = valid_prompt()
    data["source_config"] = None

    item = BenchmarkPrompt.model_validate(data)

    assert item.source_config is None


def test_optional_expected_output_can_be_none():
    data = valid_prompt()
    data["expected_output"] = None

    item = BenchmarkPrompt.model_validate(data)

    assert item.expected_output is None


def test_whitespace_optional_values_become_none():
    data = valid_prompt()
    data["source_config"] = "   "
    data["expected_output"] = "   "

    item = BenchmarkPrompt.model_validate(data)

    assert item.source_config is None
    assert item.expected_output is None


def test_unknown_fields_are_rejected():
    data = valid_prompt()
    data["unknown_field"] = "should fail"

    with pytest.raises(ValidationError):
        BenchmarkPrompt.model_validate(data)