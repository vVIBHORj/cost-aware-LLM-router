from pathlib import Path

import pytest

from router.benchmark_loader import load_benchmark_jsonl


SAMPLE_PATH = Path("data/raw/sample_benchmark.jsonl")


def test_load_benchmark_jsonl():
    items = load_benchmark_jsonl(SAMPLE_PATH)

    assert len(items) == 3
    assert items[0].prompt_id == "prompt-001"
    assert items[1].prompt_id == "prompt-002"
    assert items[2].prompt_id == "prompt-003"


def test_loaded_items_are_valid_benchmark_prompts():
    items = load_benchmark_jsonl(SAMPLE_PATH)

    assert [item.split for item in items] == [
        "train",
        "validation",
        "test",
    ]


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "benchmark.jsonl"

    path.write_text(
        """
{"prompt_id":"p1","prompt":"What is 2 + 2?","task_type":"math","domain":"arithmetic","difficulty":"easy","source":"test","split":"train"}

{"prompt_id":"p2","prompt":"Explain gravity.","task_type":"qa","domain":"science","difficulty":"easy","source":"test","split":"test"}

""",
        encoding="utf-8",
    )

    items = load_benchmark_jsonl(path)

    assert len(items) == 2


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "invalid.jsonl"

    path.write_text(
        '{"prompt_id":"p1","prompt":"Broken JSON"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_benchmark_jsonl(path)


def test_invalid_record_is_rejected(tmp_path):
    path = tmp_path / "invalid_record.jsonl"

    path.write_text(
        '{"prompt_id":"p1","prompt":"","task_type":"math","domain":"arithmetic","difficulty":"easy","source":"test","split":"train"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid benchmark record"):
        load_benchmark_jsonl(path)


def test_duplicate_prompt_ids_are_rejected(tmp_path):
    path = tmp_path / "duplicates.jsonl"

    path.write_text(
        """
{"prompt_id":"duplicate","prompt":"First question","task_type":"qa","domain":"general","difficulty":"easy","source":"test","split":"train"}
{"prompt_id":"duplicate","prompt":"Second question","task_type":"qa","domain":"general","difficulty":"easy","source":"test","split":"test"}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate prompt_id"):
        load_benchmark_jsonl(path)