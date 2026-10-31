"""
Deterministic, stratified prompt selection for Phase 4T Medium-Scale Validation.
Selects exactly 120 prompts from data/processed/benchmark_train.jsonl.
Preserves the 16 longitudinal pilot prompts and stratifies across all 8 benchmark categories.
Seed: 42
"""

import json
import hashlib
import random
from pathlib import Path
from collections import Counter


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent.parent
    train_file = root / "data" / "processed" / "benchmark_train.jsonl"
    pilot_file = root / "data" / "outcomes" / "pilot_phase4p" / "pilot_prompts.jsonl"
    out_dir = root / "data" / "outcomes" / "validation_phase4t"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prompts_file = out_dir / "validation_prompts.jsonl"
    out_manifest_file = out_dir / "selection_manifest.json"

    # Compute SHA-256 of training split
    train_sha256 = compute_file_sha256(train_file)

    # Load all training prompts
    train_prompts = []
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                assert item["split"] == "train", f"Found non-train prompt {item['prompt_id']}"
                train_prompts.append(item)

    train_by_id = {p["prompt_id"]: p for p in train_prompts}
    assert len(train_by_id) == len(train_prompts), "Duplicate prompt_id in benchmark_train.jsonl"

    # Load 16 pilot prompt IDs
    pilot_ids = []
    with pilot_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                pilot_ids.append(item["prompt_id"])

    assert len(pilot_ids) == 16, f"Expected 16 pilot prompts, got {len(pilot_ids)}"
    for pid in pilot_ids:
        assert pid in train_by_id, f"Pilot prompt {pid} not found in benchmark_train.jsonl"

    rng = random.Random(42)

    # Buckets to sample:
    # 1. qa (MMLU): 20 total (2 pilot + 18 sampled)
    # 2. reasoning (BBH): 16 total (2 pilot + 14 sampled: 11 MC + 3 exact match)
    # 3. math (GSM8K & MATH): 16 total (2 pilot GSM8K + 8 GSM8K + 6 MATH)
    # 4. coding (MBPP): 16 total (2 pilot + 14 sampled)
    # 5. instruction_following (IFEval): 14 total (2 pilot + 12 sampled)
    # 6. summarization (CNN/DailyMail): 16 total (2 pilot + 14 sampled)
    # 7. classification (BoolQ & SuperGLUE): 16 total (2 pilot BoolQ + 8 BoolQ + 5 SuperGLUE EM + 1 SuperGLUE COPA MC)
    # 8. other (HellaSwag): 6 total (2 pilot + 4 sampled)
    # Total = 20 + 16 + 16 + 16 + 14 + 16 + 16 + 6 = 120

    selected_ids = list(pilot_ids)
    selected_set = set(selected_ids)

    # 1. QA (MMLU)
    mmlu_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "qa" and p["source_dataset"] == "mmlu" and p["prompt_id"] not in selected_set
    ]
    rng.shuffle(mmlu_pool)
    qa_sampled = mmlu_pool[:18]
    selected_ids.extend(qa_sampled)
    selected_set.update(qa_sampled)

    # 2. Reasoning (BBH)
    bbh_mc_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "reasoning" and p["source_dataset"] == "bbh"
        and p["evaluation_type"] == "multiple_choice" and p["prompt_id"] not in selected_set
    ]
    bbh_em_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "reasoning" and p["source_dataset"] == "bbh"
        and p["evaluation_type"] == "exact_match" and p["prompt_id"] not in selected_set
    ]
    rng.shuffle(bbh_mc_pool)
    rng.shuffle(bbh_em_pool)
    reasoning_sampled = bbh_mc_pool[:11] + bbh_em_pool[:3]
    selected_ids.extend(reasoning_sampled)
    selected_set.update(reasoning_sampled)

    # 3. Math (GSM8K & MATH)
    gsm8k_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "math" and p["source_dataset"] == "gsm8k" and p["prompt_id"] not in selected_set
    ]
    math_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "math" and p["source_dataset"] == "math" and p["prompt_id"] not in selected_set
    ]
    rng.shuffle(gsm8k_pool)
    rng.shuffle(math_pool)
    math_sampled = gsm8k_pool[:8] + math_pool[:6]
    selected_ids.extend(math_sampled)
    selected_set.update(math_sampled)

    # 4. Coding (MBPP)
    mbpp_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "coding" and p["source_dataset"] == "mbpp" and p["prompt_id"] not in selected_set
    ]
    rng.shuffle(mbpp_pool)
    coding_sampled = mbpp_pool[:14]
    selected_ids.extend(coding_sampled)
    selected_set.update(coding_sampled)

    # 5. Instruction following (IFEval)
    ifeval_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "instruction_following" and p["source_dataset"] == "ifeval" and p["prompt_id"] not in selected_set
    ]
    rng.shuffle(ifeval_pool)
    ifeval_sampled = ifeval_pool[:12]
    selected_ids.extend(ifeval_sampled)
    selected_set.update(ifeval_sampled)

    # 6. Summarization (CNN/DailyMail)
    cnn_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "summarization" and p["source_dataset"] == "cnn_dailymail" and p["prompt_id"] not in selected_set
    ]
    rng.shuffle(cnn_pool)
    summarization_sampled = cnn_pool[:14]
    selected_ids.extend(summarization_sampled)
    selected_set.update(summarization_sampled)

    # 7. Classification (BoolQ & SuperGLUE)
    boolq_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "classification" and p["source_dataset"] == "boolq" and p["prompt_id"] not in selected_set
    ]
    sglue_em_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "classification" and p["source_dataset"] == "super_glue"
        and p["evaluation_type"] == "exact_match" and p["prompt_id"] not in selected_set
    ]
    sglue_mc_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "classification" and p["source_dataset"] == "super_glue"
        and p["evaluation_type"] == "multiple_choice" and p["prompt_id"] not in selected_set
    ]
    rng.shuffle(boolq_pool)
    rng.shuffle(sglue_em_pool)
    rng.shuffle(sglue_mc_pool)
    classification_sampled = boolq_pool[:8] + sglue_em_pool[:5] + sglue_mc_pool[:1]
    selected_ids.extend(classification_sampled)
    selected_set.update(classification_sampled)

    # 8. Other (HellaSwag)
    hellaswag_pool = [
        p["prompt_id"] for p in train_prompts
        if p["task_type"] == "other" and p["source_dataset"] == "hellaswag" and p["prompt_id"] not in selected_set
    ]
    rng.shuffle(hellaswag_pool)
    other_sampled = hellaswag_pool[:4]
    selected_ids.extend(other_sampled)
    selected_set.update(other_sampled)

    assert len(selected_ids) == 120, f"Expected 120 selected prompts, got {len(selected_ids)}"
    assert len(set(selected_ids)) == 120, "Duplicate prompts in selected list"

    # Write selected prompts to validation_prompts.jsonl
    selected_prompts = [train_by_id[pid] for pid in selected_ids]
    with out_prompts_file.open("w", encoding="utf-8") as f:
        for p in selected_prompts:
            f.write(json.dumps(p) + "\n")

    # Record manifest
    cat_counts = Counter(p["task_type"] for p in selected_prompts)
    dataset_counts = Counter(p["source_dataset"] for p in selected_prompts)
    eval_counts = Counter(p["evaluation_type"] for p in selected_prompts)

    manifest = {
        "experiment_stage": "Phase 4T - Medium-Scale Training-Only Validation",
        "seed": 42,
        "total_prompts": 120,
        "benchmark_source_file": "data/processed/benchmark_train.jsonl",
        "benchmark_source_sha256": train_sha256,
        "longitudinal_pilot_prompt_count": 16,
        "newly_sampled_prompt_count": 104,
        "category_counts": dict(cat_counts),
        "dataset_counts": dict(dataset_counts),
        "evaluation_type_counts": dict(eval_counts),
        "prompts": [
            {
                "prompt_id": p["prompt_id"],
                "task_type": p["task_type"],
                "evaluation_type": p["evaluation_type"],
                "source_dataset": p["source_dataset"],
                "source_config": p.get("source_config"),
                "difficulty": p.get("difficulty"),
                "is_longitudinal_pilot": p["prompt_id"] in pilot_ids,
            }
            for p in selected_prompts
        ],
    }

    with out_manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=== Phase 4T Prompt Selection Complete ===")
    print(f"Total Selected: {len(selected_prompts)}")
    print("Category Breakdown:", dict(cat_counts))
    print("Dataset Breakdown:", dict(dataset_counts))
    print("Evaluation Type Breakdown:", dict(eval_counts))
    print(f"Written to: {out_prompts_file}")
    print(f"Manifest: {out_manifest_file}")


if __name__ == "__main__":
    main()

