import pytest

from data_ingestion.mbpp import MBPPAdapter


def test_mbpp_adapter_full_record_conversion():
    adapter = MBPPAdapter(source_config="full")

    record = {
        "task_id": 511,
        "text": "Write a python function to find minimum sum of factors of a given number.",
        "code": "def find_Min_Sum(num):\n    sum = 0\n    return sum",
        "test_list": ["assert find_Min_Sum(12) == 7"],
        "test_setup_code": "",
        "challenge_test_list": [],
    }

    item = adapter.convert_record(
        record,
        source_split="test",
        source_id="0",
        benchmark_split="validation",
    )

    assert item.prompt_id == "mbpp-full-test-511"
    assert item.prompt == "Write a python function to find minimum sum of factors of a given number."
    assert item.task_type == "coding"
    assert item.domain == "python"
    assert item.difficulty == "medium"
    assert item.source_dataset == "mbpp"
    assert item.source_config == "full"
    assert item.source_split == "test"
    assert item.source_id == "511"
    assert item.split == "validation"
    assert item.expected_output == "def find_Min_Sum(num):\n    sum = 0\n    return sum"
    assert item.evaluation_type == "code_execution"


def test_mbpp_adapter_sanitized_record_conversion():
    adapter = MBPPAdapter(source_config="sanitized")

    record = {
        "task_id": 554,
        "prompt": "Write a python function which takes a list of integers and only returns the odd ones.",
        "code": "def Split(list):\n    return [i for i in list if i % 2 != 0]",
        "test_list": ["assert Split([1,2,3,4,5,6]) == [1,3,5]"],
        "test_imports": [],
        "source_file": "Benchmark Questions Verification V2.ipynb",
    }

    item = adapter.convert_record(
        record,
        source_split="train",
        source_id="10",
        benchmark_split="train",
    )

    assert item.prompt_id == "mbpp-sanitized-train-554"
    assert item.prompt == "Write a python function which takes a list of integers and only returns the odd ones."
    assert item.source_config == "sanitized"
    assert item.source_id == "554"
    assert item.expected_output == "def Split(list):\n    return [i for i in list if i % 2 != 0]"
    assert item.evaluation_type == "code_execution"


def test_mbpp_adapter_fallback_to_positional_id_if_task_id_missing():
    adapter = MBPPAdapter(source_config="full")

    record = {
        "text": "Write a function to return 42.",
        "code": "def f(): return 42",
    }

    item = adapter.convert_record(
        record,
        source_split="test",
        source_id="99",
        benchmark_split="test",
    )

    assert item.prompt_id == "mbpp-full-test-99"
    assert item.source_id == "99"


def test_mbpp_adapter_missing_text_rejected():
    adapter = MBPPAdapter()

    with pytest.raises(ValueError, match="missing a valid task prompt"):
        adapter.convert_record(
            {"code": "def f(): pass"},
            source_split="test",
            source_id="1",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="missing a valid task prompt"):
        adapter.convert_record(
            {"text": "   ", "code": "def f(): pass"},
            source_split="test",
            source_id="1",
            benchmark_split="train",
        )


def test_mbpp_adapter_missing_code_rejected():
    adapter = MBPPAdapter()

    with pytest.raises(ValueError, match="missing valid reference code"):
        adapter.convert_record(
            {"text": "Write a function."},
            source_split="test",
            source_id="1",
            benchmark_split="train",
        )

    with pytest.raises(ValueError, match="missing valid reference code"):
        adapter.convert_record(
            {"text": "Write a function.", "code": "   "},
            source_split="test",
            source_id="1",
            benchmark_split="train",
        )


def test_mbpp_adapter_invalid_record_type_rejected():
    adapter = MBPPAdapter()

    with pytest.raises(ValueError, match="must be a dictionary"):
        adapter.convert_record(
            "not a dict",  # type: ignore[arg-type]
            source_split="test",
            source_id="1",
            benchmark_split="train",
        )


def test_mbpp_adapter_invalid_config_rejected():
    with pytest.raises(ValueError, match="source_config must be a non-empty string"):
        MBPPAdapter(source_config="   ")
