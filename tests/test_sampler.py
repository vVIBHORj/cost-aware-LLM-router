import json
from pathlib import Path
import pytest

from router.benchmark_schema import BenchmarkPrompt
from data_ingestion.split_guard import SplitGuard, normalize_text
from data_ingestion.sampler import (
    BenchmarkSampler,
    compute_ifeval_hash,
    MMLU_SUBJECTS,
    BBH_TASKS,
    MATH_SUBJECTS,
    SUPER_GLUE_CORE_TASKS,
)


PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = PROCESSED_DIR / "benchmark_manifest.json"
TRAIN_PATH = PROCESSED_DIR / "benchmark_train.jsonl"
VAL_PATH = PROCESSED_DIR / "benchmark_validation.jsonl"
TEST_PATH = PROCESSED_DIR / "benchmark_test.jsonl"
STRESS_PATH = PROCESSED_DIR / "benchmark_stress.jsonl"


def load_jsonl(path: Path) -> list[BenchmarkPrompt]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(BenchmarkPrompt.model_validate_json(line))
    return records


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}"
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def benchmark_data():
    return {
        "train": load_jsonl(TRAIN_PATH),
        "validation": load_jsonl(VAL_PATH),
        "test": load_jsonl(TEST_PATH),
        "stress": load_jsonl(STRESS_PATH),
    }


# 1. Exact split counts
def test_exact_split_counts(benchmark_data, manifest):
    assert len(benchmark_data["train"]) == 1400
    assert len(benchmark_data["validation"]) == 300
    assert len(benchmark_data["test"]) == 300
    assert len(benchmark_data["stress"]) == 300
    assert manifest["counts"]["grand_total"] == 2300
    assert manifest["counts"]["core_total"] == 2000


# 2. Exact category counts
def test_exact_category_quotas(benchmark_data, manifest):
    core_items = (
        benchmark_data["train"]
        + benchmark_data["validation"]
        + benchmark_data["test"]
    )
    cat_counts = manifest["category_distribution"]
    assert cat_counts["general_qa"]["total"] == 400
    assert cat_counts["reasoning"]["total"] == 300
    assert cat_counts["math"]["total"] == 300
    assert cat_counts["coding"]["total"] == 300
    assert cat_counts["instruction_following"]["total"] == 250
    assert cat_counts["summarization"]["total"] == 200
    assert cat_counts["classification_extraction"]["total"] == 150
    assert cat_counts["other"]["total"] == 100

    # Sum of all category totals must equal 2000
    assert sum(c["total"] for c in cat_counts.values()) == 2000


# 3. No source-ID overlap across splits
def test_no_source_id_overlap(benchmark_data):
    SplitGuard.check_source_id_quarantine(
        train_items=benchmark_data["train"],
        val_items=benchmark_data["validation"],
        test_items=benchmark_data["test"],
        stress_items=benchmark_data["stress"],
    )


# 4. No normalized prompt duplicates
def test_no_prompt_duplicates(benchmark_data):
    all_prompts = (
        benchmark_data["train"]
        + benchmark_data["validation"]
        + benchmark_data["test"]
        + benchmark_data["stress"]
    )
    assert len(all_prompts) == 2300
    SplitGuard.check_prompt_deduplication(all_prompts)


# 5. IFEval SHA-256 hash partition correctness
def test_ifeval_hash_partition_logic():
    # Deterministic SHA-256 partitioning with seed=42
    # hash < 70 -> train, 70 <= hash < 85 -> val, hash >= 85 -> test
    h_train = compute_ifeval_hash("10", seed=42)
    assert 0 <= h_train <= 100

    keys = [str(k) for k in range(541)]
    train_c = sum(1 for k in keys if compute_ifeval_hash(k, 42) < 70)
    val_c = sum(1 for k in keys if 70 <= compute_ifeval_hash(k, 42) < 85)
    test_c = sum(1 for k in keys if compute_ifeval_hash(k, 42) >= 85)

    assert train_c >= 175
    assert val_c >= 37
    assert test_c >= 38


# 6. MMLU subject stratification across 57 subjects
def test_mmlu_subject_stratification(benchmark_data):
    core_items = (
        benchmark_data["train"]
        + benchmark_data["validation"]
        + benchmark_data["test"]
    )
    mmlu_items = [p for p in core_items if p.source_dataset == "mmlu"]
    assert len(mmlu_items) == 400

    subjects_seen = {p.source_config for p in mmlu_items}
    assert subjects_seen == set(MMLU_SUBJECTS)
    assert len(subjects_seen) == 57

    # Strata allocation across 57 subjects: 400 total (mean 7.02) -> bounds [6, 9]
    counts = {}
    for p in mmlu_items:
        counts[p.source_config] = counts.get(p.source_config, 0) + 1
    for subj in MMLU_SUBJECTS:
        assert 6 <= counts[subj] <= 9


# 7. BBH per-task stratification across 27 tasks
def test_bbh_task_stratification(benchmark_data):
    core_items = (
        benchmark_data["train"]
        + benchmark_data["validation"]
        + benchmark_data["test"]
    )
    bbh_items = [p for p in core_items if p.source_dataset == "bbh"]
    assert len(bbh_items) == 300

    tasks_seen = {p.source_config for p in bbh_items}
    assert tasks_seen == set(BBH_TASKS)
    assert len(tasks_seen) == 27

    # Strategy B per task: 300 total across 27 tasks (mean 11.11) -> bounds [9, 12]
    counts = {}
    for p in bbh_items:
        counts[p.source_config] = counts.get(p.source_config, 0) + 1
    for task in BBH_TASKS:
        assert 9 <= counts[task] <= 12


# 8. HumanEval test-only enforcement (0 in train, 0 in val)
def test_humaneval_test_only_enforcement(benchmark_data):
    for p in benchmark_data["train"]:
        assert p.source_dataset != "humaneval", "HumanEval found in train split!"
    for p in benchmark_data["validation"]:
        assert p.source_dataset != "humaneval", "HumanEval found in val split!"

    humaneval_test = [p for p in benchmark_data["test"] if p.source_dataset == "humaneval"]
    assert len(humaneval_test) == 25


# 9. SuperGLUE boolq exclusion
def test_superglue_boolq_excluded(benchmark_data):
    all_prompts = (
        benchmark_data["train"]
        + benchmark_data["validation"]
        + benchmark_data["test"]
        + benchmark_data["stress"]
    )
    for p in all_prompts:
        if p.source_dataset == "super_glue":
            assert p.source_config != "boolq", "SuperGLUE boolq is prohibited!"


# 10. Unlabelled source-test exclusion
def test_unlabelled_source_test_excluded(benchmark_data):
    all_prompts = (
        benchmark_data["train"]
        + benchmark_data["validation"]
        + benchmark_data["test"]
        + benchmark_data["stress"]
    )
    for p in all_prompts:
        # HellaSwag test has no labels
        if p.source_dataset == "hellaswag":
            assert p.source_split != "test"
            assert p.expected_output in {"(A)", "(B)", "(C)", "(D)"}
        # SuperGLUE test has no labels (-1)
        if p.source_dataset == "super_glue":
            assert p.source_split != "test"
            assert p.expected_output is not None


# 11. Zero MMLU source-test in benchmark train or validation
def test_zero_mmlu_source_test_in_train_or_val(benchmark_data):
    for p in benchmark_data["train"]:
        if p.source_dataset == "mmlu":
            assert p.source_split != "test", "MMLU source test found in benchmark train!"
    for p in benchmark_data["validation"]:
        if p.source_dataset == "mmlu":
            assert p.source_split != "test", "MMLU source test found in benchmark val!"


# 12. Stress / Core total quarantine
def test_stress_core_quarantine(benchmark_data):
    core_items = (
        benchmark_data["train"]
        + benchmark_data["validation"]
        + benchmark_data["test"]
    )
    stress_items = benchmark_data["stress"]
    assert len(stress_items) == 300

    core_source_ids = {(p.source_dataset, p.source_config, p.source_split, p.source_id) for p in core_items}
    for sp in stress_items:
        orig_ds = sp.original_source_dataset or sp.source_dataset
        orig_cfg = sp.source_config or ""
        orig_split = sp.source_split or ""
        orig_id = sp.original_source_id or sp.source_id
        assert (orig_ds, orig_cfg, orig_split, orig_id) not in core_source_ids, (
            f"Stress prompt {sp.prompt_id} derived from core record {orig_id}!"
        )


# 13. Stress categories exact quotas
def test_stress_category_quotas(benchmark_data, manifest):
    stress_items = benchmark_data["stress"]
    stress_counts = manifest["stress_counts"]
    assert stress_counts == {
        "long_prompts": 60,
        "distractors": 50,
        "paraphrased": 50,
        "formatting_variations": 40,
        "harder_reasoning_math": 40,
        "code_variations": 30,
        "held_out_domain_task": 30,
        "total": 300,
    }


# 14. Provenance completeness
def test_provenance_completeness(benchmark_data):
    all_prompts = (
        benchmark_data["train"]
        + benchmark_data["validation"]
        + benchmark_data["test"]
        + benchmark_data["stress"]
    )
    for p in all_prompts:
        assert p.prompt_id and len(p.prompt_id.strip()) > 0
        assert p.prompt and len(p.prompt.strip()) > 0
        assert p.task_type in {"qa", "reasoning", "math", "coding", "instruction_following", "summarization", "classification", "other"}
        assert p.domain and len(p.domain.strip()) > 0
        assert p.difficulty in {"easy", "medium", "hard"}
        assert p.source_dataset and len(p.source_dataset.strip()) > 0
        assert p.source_split in {"train", "validation", "test", "dev"}
        assert p.source_id and len(p.source_id.strip()) > 0
        assert p.split in {"train", "validation", "test", "stress"}
        assert p.evaluation_type in {"exact_match", "multiple_choice", "code_execution", "rule_based", "reference_based", "ifeval"}

        if p.split == "stress":
            assert p.stress_transformation is not None
            assert p.original_source_dataset is not None
            assert p.original_source_id is not None
