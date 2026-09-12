from pathlib import Path

import pytest

from router.benchmark_loader import load_benchmark_jsonl


SAMPLE_PATH = Path("data/raw/sample_benchmark.jsonl")


def test_load_benchmark_jsonl():
    items = load_benchmark_jsonl(SAMPLE_PATH)

    assert len(items) == 3

    assert items[0].prompt_id == "sample-001"
    assert items[0].source_dataset == "mmlu"
    assert items[0].source_config == "global_facts"
    assert items[0].source_split == "test"
    assert items[0].source_id == "sample-001"
    assert items[0].split == "train"
    assert items[0].evaluation_type == "exact_match"


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "benchmark.jsonl"

    path.write_text(
        """
{"prompt_id":"p1","prompt":"What is 2 + 2?","task_type":"math","domain":"arithmetic","difficulty":"easy","source_dataset":"gsm8k","source_config":"main","source_split":"test","source_id":"p1","split":"train","expected_output":"4","evaluation_type":"exact_match"}

{"prompt_id":"p2","prompt":"Explain gravity.","task_type":"qa","domain":"science","difficulty":"easy","source_dataset":"mmlu","source_config":"conceptual_physics","source_split":"test","source_id":"p2","split":"test","expected_output":null,"evaluation_type":"reference_based"}

""",
        encoding="utf-8",
    )

    items = load_benchmark_jsonl(path)

    assert len(items) == 2
    assert items[0].prompt_id == "p1"
    assert items[1].prompt_id == "p2"


def test_duplicate_prompt_ids_are_rejected(tmp_path):
    path = tmp_path / "duplicates.jsonl"

    path.write_text(
        """
{"prompt_id":"duplicate","prompt":"First question","task_type":"qa","domain":"general","difficulty":"easy","source_dataset":"mmlu","source_config":"global_facts","source_split":"test","source_id":"1","split":"train","expected_output":"Paris","evaluation_type":"exact_match"}
{"prompt_id":"duplicate","prompt":"Second question","task_type":"qa","domain":"general","difficulty":"easy","source_dataset":"mmlu","source_config":"global_facts","source_split":"test","source_id":"2","split":"test","expected_output":"London","evaluation_type":"exact_match"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate prompt_id"):
        load_benchmark_jsonl(path)


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "invalid.jsonl"

    path.write_text(
        '{"prompt_id":"p1","prompt":invalid}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_benchmark_jsonl(path)


def test_invalid_benchmark_record_is_rejected(tmp_path):
    path = tmp_path / "invalid_record.jsonl"

    path.write_text(
        '{"prompt_id":"p1","prompt":"","task_type":"math","domain":"arithmetic","difficulty":"easy","source_dataset":"gsm8k","source_config":"main","source_split":"test","source_id":"1","split":"train","evaluation_type":"exact_match"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid benchmark record"):
        load_benchmark_jsonl(path)


def test_duplicate_ids_are_checked_after_validation(tmp_path):
    path = tmp_path / "duplicate_after_validation.jsonl"

    path.write_text(
        """
{"prompt_id":"duplicate","prompt":"First question","task_type":"qa","domain":"general","difficulty":"easy","source_dataset":"mmlu","source_config":"global_facts","source_split":"test","source_id":"1","split":"train","evaluation_type":"exact_match"}
{"prompt_id":"duplicate","prompt":"Second question","task_type":"qa","domain":"general","difficulty":"easy","source_dataset":"mmlu","source_config":"global_facts","source_split":"test","source_id":"2","split":"test","evaluation_type":"exact_match"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate prompt_id"):
        load_benchmark_jsonl(path)