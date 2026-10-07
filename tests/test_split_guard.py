import pytest

from router.benchmark_schema import BenchmarkPrompt
from data_ingestion.split_guard import SplitGuard, LeakageError, normalize_text


def make_prompt(
    prompt_id: str,
    prompt: str,
    source_dataset: str,
    source_id: str,
    split: str,
    task_type: str = "math",
    original_source_dataset: str | None = None,
    original_source_id: str | None = None,
) -> BenchmarkPrompt:
    return BenchmarkPrompt(
        prompt_id=prompt_id,
        prompt=prompt,
        task_type=task_type,
        domain="test",
        difficulty="medium",
        source_dataset=source_dataset,
        source_config=None,
        source_split="train",
        source_id=source_id,
        split=split,
        expected_output="1",
        evaluation_type="exact_match",
        original_source_dataset=original_source_dataset,
        original_source_id=original_source_id,
    )


def test_normalize_text():
    assert normalize_text("  Hello   World! \n\t ") == "hello world!"


def test_source_id_quarantine_clean():
    train = [make_prompt("p1", "What is 1+1?", "gsm8k", "1", "train")]
    val = [make_prompt("p2", "What is 2+2?", "gsm8k", "2", "validation")]
    test = [make_prompt("p3", "What is 3+3?", "gsm8k", "3", "test")]
    stress = [make_prompt("p4", "What is 4+4?", "gsm8k", "4", "stress")]

    SplitGuard.check_source_id_quarantine(train, val, test, stress)


def test_source_id_quarantine_train_val_overlap():
    train = [make_prompt("p1", "What is 1+1?", "gsm8k", "1", "train")]
    val = [make_prompt("p2", "What is 2+2?", "gsm8k", "1", "validation")]
    test = [make_prompt("p3", "What is 3+3?", "gsm8k", "3", "test")]

    with pytest.raises(LeakageError, match="train and validation"):
        SplitGuard.check_source_id_quarantine(train, val, test)


def test_source_id_quarantine_core_stress_overlap():
    train = [make_prompt("p1", "What is 1+1?", "gsm8k", "1", "train")]
    val = [make_prompt("p2", "What is 2+2?", "gsm8k", "2", "validation")]
    test = [make_prompt("p3", "What is 3+3?", "gsm8k", "3", "test")]
    stress = [
        make_prompt(
            "p4",
            "What is 1+1? Perturbed",
            "stress_gsm8k",
            "s1",
            "stress",
            original_source_dataset="gsm8k",
            original_source_id="1",
        )
    ]

    with pytest.raises(LeakageError, match="core and stress"):
        SplitGuard.check_source_id_quarantine(train, val, test, stress)


def test_prompt_deduplication_clean():
    items = [
        make_prompt("p1", "What is 1+1?", "gsm8k", "1", "train"),
        make_prompt("p2", "What is 2+2?", "gsm8k", "2", "train"),
    ]
    SplitGuard.check_prompt_deduplication(items)


def test_prompt_deduplication_duplicate_detected():
    items = [
        make_prompt("p1", "What is 1+1?", "gsm8k", "1", "train"),
        make_prompt("p2", "  what   is 1+1?  \n", "gsm8k", "2", "train"),
    ]
    with pytest.raises(LeakageError, match="duplicate prompts detected"):
        SplitGuard.check_prompt_deduplication(items)


def test_context_leakage_detected():
    by_split = {
        "train": [{"prompt_id": "p1", "context": "This is a long passage about the history of science and physics."}],
        "test": [{"prompt_id": "p2", "context": "  This is a LONG passage about the history of science and physics.  "}],
    }
    with pytest.raises(LeakageError, match="Context leakage violation"):
        SplitGuard.check_context_leakage(by_split)


def test_context_leakage_clean():
    by_split = {
        "train": [{"prompt_id": "p1", "context": "Passage A about biology and nature."}],
        "test": [{"prompt_id": "p2", "context": "Passage B about astronomy and telescopes."}],
    }
    SplitGuard.check_context_leakage(by_split)
