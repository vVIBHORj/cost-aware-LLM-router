import pytest

from data_ingestion.gsm8k import GSM8KAdapter
from data_ingestion.runner import ingest_records


def test_ingest_records():
    records = [
        {
            "question": "What is 2 + 2?",
            "answer": "2 + 2 = 4.\n#### 4",
        },
        {
            "question": "What is 10 + 5?",
            "answer": "10 + 5 = 15.\n#### 15",
        },
    ]

    items = ingest_records(
        records,
        GSM8KAdapter(),
        source_split="test",
        benchmark_split="train",
    )

    assert len(items) == 2

    assert items[0].prompt_id == "gsm8k-main-test-0"
    assert items[1].prompt_id == "gsm8k-main-test-1"

    assert items[0].source_dataset == "gsm8k"
    assert items[0].source_split == "test"
    assert items[0].split == "train"


def test_ingest_empty_records():
    items = ingest_records(
        [],
        GSM8KAdapter(),
        source_split="test",
        benchmark_split="train",
    )

    assert items == []


def test_invalid_record_type_is_rejected():
    records = [
        "not a dictionary",
    ]

    with pytest.raises(
        ValueError,
        match="must be a dictionary",
    ):
        ingest_records(
            records,
            GSM8KAdapter(),
            source_split="test",
            benchmark_split="train",
        )


def test_adapter_errors_propagate():
    records = [
        {
            "question": "",
            "answer": "The answer is 4.\n#### 4",
        },
    ]

    with pytest.raises(
        ValueError,
        match="valid question",
    ):
        ingest_records(
            records,
            GSM8KAdapter(),
            source_split="test",
            benchmark_split="train",
        )


def test_all_records_are_benchmark_prompts():
    records = [
        {
            "question": "What is 3 + 3?",
            "answer": "3 + 3 = 6.\n#### 6",
        },
        {
            "question": "What is 4 + 4?",
            "answer": "4 + 4 = 8.\n#### 8",
        },
    ]

    items = ingest_records(
        records,
        GSM8KAdapter(),
        source_split="validation",
        benchmark_split="validation",
    )

    for item in items:
        assert item.__class__.__name__ == "BenchmarkPrompt"
        assert item.source_dataset == "gsm8k"
        assert item.source_split == "validation"
        assert item.split == "validation"


def test_source_provenance_is_preserved():
    records = [
        {
            "question": "What is 100 - 40?",
            "answer": "100 - 40 = 60.\n#### 60",
        },
    ]

    items = ingest_records(
        records,
        GSM8KAdapter(),
        source_split="test",
        benchmark_split="test",
    )

    item = items[0]

    assert item.source_dataset == "gsm8k"
    assert item.source_config == "main"
    assert item.source_split == "test"
    assert item.source_id == "0"


def test_multiple_records_get_unique_ids():
    records = [
        {
            "question": "Question one",
            "answer": "The answer is 1.\n#### 1",
        },
        {
            "question": "Question two",
            "answer": "The answer is 2.\n#### 2",
        },
        {
            "question": "Question three",
            "answer": "The answer is 3.\n#### 3",
        },
    ]

    items = ingest_records(
        records,
        GSM8KAdapter(),
        source_split="train",
        benchmark_split="train",
    )

    ids = [item.prompt_id for item in items]

    assert len(ids) == len(set(ids))