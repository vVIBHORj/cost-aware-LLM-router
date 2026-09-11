import pytest
from pydantic import ValidationError

from router.benchmark_schema import BenchmarkPrompt


def make_prompt(**overrides):
    data = {
        "prompt_id": "prompt-001",
        "prompt": "Explain why the sky appears blue.",
        "task_type": "qa",
        "domain": "science",
        "difficulty": "easy",
        "source": "test",
        "split": "train",
    }

    data.update(overrides)

    return BenchmarkPrompt(**data)


def test_valid_benchmark_prompt():
    item = make_prompt()

    assert item.prompt_id == "prompt-001"
    assert item.task_type == "qa"
    assert item.domain == "science"
    assert item.difficulty == "easy"
    assert item.split == "train"


def test_prompt_text_is_stripped():
    item = make_prompt(prompt="  Explain this problem.  ")

    assert item.prompt == "Explain this problem."


def test_empty_prompt_is_rejected():
    with pytest.raises(ValidationError):
        make_prompt(prompt="")


def test_whitespace_prompt_is_rejected():
    with pytest.raises(ValidationError):
        make_prompt(prompt="   ")


def test_empty_prompt_id_is_rejected():
    with pytest.raises(ValidationError):
        make_prompt(prompt_id="")


def test_invalid_task_type_is_rejected():
    with pytest.raises(ValidationError):
        make_prompt(task_type="invalid")


def test_invalid_difficulty_is_rejected():
    with pytest.raises(ValidationError):
        make_prompt(difficulty="extreme")


def test_invalid_split_is_rejected():
    with pytest.raises(ValidationError):
        make_prompt(split="production")


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        make_prompt(unexpected_field="something")