import json

import pytest

from data_ingestion.gsm8k import GSM8KAdapter
from data_ingestion.runner import ingest_records
from data_ingestion.writer import write_benchmark_jsonl
from router.benchmark_loader import load_benchmark_jsonl
from router.benchmark_schema import BenchmarkPrompt


def create_items() -> list[BenchmarkPrompt]:
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

    return ingest_records(
        records,
        GSM8KAdapter(),
        source_split="test",
        benchmark_split="train",
    )


def test_write_benchmark_jsonl(tmp_path):
    items = create_items()

    output = tmp_path / "benchmark.jsonl"

    result = write_benchmark_jsonl(items, output)

    assert result == output
    assert output.exists()

    lines = output.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2


def test_written_records_are_valid_json(tmp_path):
    items = create_items()

    output = tmp_path / "benchmark.jsonl"

    write_benchmark_jsonl(items, output)

    lines = output.read_text(encoding="utf-8").splitlines()

    for line in lines:
        record = json.loads(line)

        assert isinstance(record, dict)
        assert "prompt_id" in record
        assert "prompt" in record
        assert "source_dataset" in record
        assert "source_id" in record


def test_writer_preserves_provenance(tmp_path):
    items = create_items()

    output = tmp_path / "benchmark.jsonl"

    write_benchmark_jsonl(items, output)

    record = json.loads(
        output.read_text(encoding="utf-8").splitlines()[0]
    )

    assert record["source_dataset"] == "gsm8k"
    assert record["source_config"] == "main"
    assert record["source_split"] == "test"
    assert record["source_id"] == "0"
    assert record["split"] == "train"


def test_writer_round_trip(tmp_path):
    items = create_items()

    output = tmp_path / "benchmark.jsonl"

    write_benchmark_jsonl(items, output)

    loaded = load_benchmark_jsonl(output)

    assert len(loaded) == len(items)

    for original, restored in zip(items, loaded):
        assert restored == original


def test_writer_creates_parent_directories(tmp_path):
    items = create_items()

    output = (
        tmp_path
        / "nested"
        / "directory"
        / "benchmark.jsonl"
    )

    write_benchmark_jsonl(items, output)

    assert output.exists()


def test_writer_rejects_invalid_objects(tmp_path):
    output = tmp_path / "benchmark.jsonl"

    with pytest.raises(
        TypeError,
        match="BenchmarkPrompt objects",
    ):
        write_benchmark_jsonl(
            ["not a BenchmarkPrompt"],
            output,
        )


def test_writer_supports_empty_input(tmp_path):
    output = tmp_path / "empty.jsonl"

    write_benchmark_jsonl([], output)

    assert output.exists()
    assert output.read_text(encoding="utf-8") == ""


def test_writer_supports_unicode(tmp_path):
    item = BenchmarkPrompt(
        prompt_id="unicode-001",
        prompt="भारत की राजधानी क्या है?",
        task_type="qa",
        domain="geography",
        difficulty="easy",
        source_dataset="custom_other",
        source_config=None,
        source_split="test",
        source_id="unicode-001",
        split="test",
        expected_output="नई दिल्ली",
        evaluation_type="exact_match",
    )

    output = tmp_path / "unicode.jsonl"

    write_benchmark_jsonl([item], output)

    content = output.read_text(encoding="utf-8")

    assert "भारत" in content
    assert "नई दिल्ली" in content

    loaded = load_benchmark_jsonl(output)

    assert loaded[0].prompt == "भारत की राजधानी क्या है?"
    assert loaded[0].expected_output == "नई दिल्ली"