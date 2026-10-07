import pytest

from data_ingestion.humaneval import HumanEvalAdapter


def test_humaneval_record_is_converted():
    adapter = HumanEvalAdapter()

    record = {
        "task_id": "HumanEval/0",
        "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    pass\n",
        "canonical_solution": "    return True\n",
        "test": "check(has_close_elements)",
        "entry_point": "has_close_elements",
    }

    item = adapter.convert_record(
        record,
        source_split="test",
        source_id="0",
        benchmark_split="test",
    )

    assert item.prompt_id == "humaneval-test-HumanEval_0"
    assert item.prompt == record["prompt"].strip()
    assert item.task_type == "coding"
    assert item.domain == "python"
    assert item.difficulty == "hard"
    assert item.source_dataset == "humaneval"
    assert item.source_config == "openai_humaneval"
    assert item.source_split == "test"
    assert item.source_id == "HumanEval/0"
    assert item.split == "test"
    assert item.expected_output == "return True"
    assert item.evaluation_type == "code_execution"


def test_humaneval_missing_prompt_is_rejected():
    adapter = HumanEvalAdapter()
    with pytest.raises(ValueError, match="missing a valid 'prompt'"):
        adapter.convert_record(
            {"canonical_solution": "pass"},
            source_split="test",
            source_id="0",
            benchmark_split="test",
        )


def test_humaneval_missing_solution_is_rejected():
    adapter = HumanEvalAdapter()
    with pytest.raises(ValueError, match="missing 'canonical_solution'"):
        adapter.convert_record(
            {"prompt": "def f(): pass"},
            source_split="test",
            source_id="0",
            benchmark_split="test",
        )
