# Benchmark Generation & Sampling Report (v1)

**Project:** Cost-Aware LLM Model Router (`cost-aware-llm-router`)  
**Date:** October 2026  
**Status:** Verification Completed  
**Seed:** 42 (`configs/benchmark.yaml`)  
**Policy Document Reference:** `reports/benchmark_split_policy_v1.md`  
**Execution Environment:** Python 3.12.4, `pytest-8.4.2`, `datasets-5.0.1`

---

## 1. Executive Summary

A deterministic, reproducible benchmark sampling pipeline was executed to produce the official 2,300-prompt benchmark dataset (2,000 core + 300 held-out stress) strictly adhering to the approved specifications of `reports/benchmark_split_policy_v1.md`.

All allocations enforce strict directional split preservation, zero evaluation contamination, elimination of unlabelled source splits, prompt deduplication, and complete provenance tracking.

Two independent benchmark generations from clean initial states produced **100% byte-identical** artifacts and checksums.

---

## 2. Final Split & Category Counts

### 2.1 Benchmark Split Counts

| Benchmark Split | Core Prompts | Stress Prompts | Total Prompts | Policy Target | Status |
|---|---|---|---|---|---|
| `train` | 1,400 | 0 | 1,400 | 1,400 | **EXACT MATCH** |
| `validation` | 299 | 0 | 299 | 300* | **EXACT POLICY SUM** (See 2.3) |
| `test` | 301 | 0 | 301 | 300* | **EXACT POLICY SUM** (See 2.3) |
| `stress` | 0 | 300 | 300 | 300 | **EXACT MATCH** |
| **Total** | **2,000** | **300** | **2,300** | **2,300** | **EXACT MATCH** |

### 2.2 Category Quota & Split Distribution

| Category | Benchmark Train | Benchmark Validation | Benchmark Test | Category Total | Policy Quota |
|---|---|---|---|---|---|
| `general_qa` | 280 | 60 | 60 | **400** | 400 |
| `reasoning` | 210 | 45 | 45 | **300** | 300 |
| `math` | 210 | 45 | 45 | **300** | 300 |
| `coding` | 210 | 45 | 45 | **300** | 300 |
| `instruction_following` | 175 | 37 | 38 | **250** | 250 |
| `summarization` | 140 | 30 | 30 | **200** | 200 |
| `classification_extraction` | 105 | 22 | 23 | **150** | 150 |
| `other` | 70 | 15 | 15 | **100** | 100 |
| **Core Total** | **1,400** | **299** | **301** | **2,000** | **2,000** |

### 2.3 Analysis of Policy Table 7 Arithmetic Discrepancy

In `reports/benchmark_split_policy_v1.md`, Table 7 defines the exact per-category allocations:
- `instruction_following` (IFEval): 175 train, 37 validation, 38 test (quota: 250). Note: $250 \times 0.15 = 37.5$, so split partitioning allocated 37 to validation and 38 to test (-0.5 val, +0.5 test).
- `classification_extraction`: 105 train, 22 validation, 23 test (quota: 150). Note: $150 \times 0.15 = 22.5$, so split partitioning allocated 22 to validation and 23 to test (-0.5 val, +0.5 test).
- All other 6 categories are symmetric across validation and test: 60/60, 45/45, 45/45, 45/45, 30/30, 15/15.

Summing the explicit row entries in Table 7 gives:
- Validation column: $60 + 45 + 45 + 45 + 37 + 30 + 22 + 15 = 299$
- Test column: $60 + 45 + 45 + 45 + 38 + 30 + 23 + 15 = 301$

Table 7's bottom summary row stated `1,400 | 300 | 300 | 2,000` due to an arithmetic column-addition oversight in the policy specification itself. The sampler strictly implements the approved row-level specifications, resulting in 299 validation and 301 test.

---

## 3. Dataset Source Distribution & Split Mapping

| Source Dataset | Config / Task | Source Split | Benchmark Split | Count | Role & Policy Guard |
|---|---|---|---|---|---|
| `cais/mmlu` | 57 subjects | `validation` | `train` | 280 | Zero source test in train/val; 57 subjects stratified |
| `cais/mmlu` | 57 subjects | `validation` | `validation` | 60 | Zero source test in train/val; 57 subjects stratified |
| `cais/mmlu` | 57 subjects | `test` | `test` | 60 | Canonical source test; 57 subjects stratified |
| `lukaemon/bbh` | 27 tasks | `test` | `train` | 210 | Strategy B (intra-task slicing: 70% train) |
| `lukaemon/bbh` | 27 tasks | `test` | `validation` | 45 | Strategy B (intra-task slicing: 15% validation) |
| `lukaemon/bbh` | 27 tasks | `test` | `test` | 45 | Strategy B (intra-task slicing: 15% test) |
| `openai/gsm8k` | `main` | `train` | `train` | 105 | Directional train preservation |
| `openai/gsm8k` | `main` | `train` | `validation` | 23 | Directional train held-out validation |
| `openai/gsm8k` | `main` | `test` | `test` | 22 | Directional test preservation |
| `EleutherAI/hendrycks_math` | 7 subjects | `train` | `train` | 105 | Directional train preservation |
| `EleutherAI/hendrycks_math` | 7 subjects | `train` | `validation` | 22 | Directional train held-out validation |
| `EleutherAI/hendrycks_math` | 7 subjects | `test` | `test` | 23 | Directional test preservation |
| `google-research-datasets/mbpp` | `full` | `train` | `train` | 210 | Directional train preservation |
| `google-research-datasets/mbpp` | `full` | `validation` | `validation` | 45 | Directional validation preservation |
| `google-research-datasets/mbpp` | `full` | `test` | `test` | 20 | Directional test preservation |
| `openai/openai_humaneval` | `default` | `test` | `test` | 25 | **Test-only** (zero in train/val) |
| `google/ifeval` | `default` | `train` | `train` | 175 | SHA-256 partition `< 70` (seed=42) |
| `google/ifeval` | `default` | `train` | `validation` | 37 | SHA-256 partition `70 <= h < 85` (seed=42) |
| `google/ifeval` | `default` | `train` | `test` | 38 | SHA-256 partition `>= 85` (seed=42) |
| `abisee/cnn_dailymail` | `3.0.0` | `train` | `train` | 140 | Max tokens $\le 2048$, deduplicated |
| `abisee/cnn_dailymail` | `3.0.0` | `validation` | `validation` | 30 | Max tokens $\le 2048$, deduplicated |
| `abisee/cnn_dailymail` | `3.0.0` | `test` | `test` | 30 | Max tokens $\le 2048$, deduplicated |
| `google/boolq` | `default` | `train` | `train` | 53 | Directional train preservation |
| `google/boolq` | `default` | `validation` | `validation` | 11 | Validation slice A (0..10) |
| `google/boolq` | `default` | `validation` | `test` | 11 | Validation slice B (11..21) |
| `aps/super_glue` | `cb, copa, rte, wic, wsc, multirc` | `train` | `train` | 52 | Prohibits `boolq`; directional train |
| `aps/super_glue` | `cb, copa, rte, wic, wsc, multirc` | `validation` | `validation` | 11 | Validation slice A |
| `aps/super_glue` | `cb, copa, rte, wic, wsc, multirc` | `validation` | `test` | 12 | Validation slice B |
| `Rowan/hellaswag` | `default` | `train` | `train` | 70 | Directional train preservation |
| `Rowan/hellaswag` | `default` | `validation` | `validation` | 15 | Validation slice A (excludes unlabelled test) |
| `Rowan/hellaswag` | `default` | `validation` | `test` | 15 | Validation slice B (excludes unlabelled test) |

---

## 4. Stress Suite Composition & Provenance

The 300 stress prompts (`held_out: true`) test router robustness across 7 dedicated failure modes without plundering or perturbing any core benchmark record:

| Stress Category | Target Count | Actual Count | Source Dataset / Task | Transformation Applied | Quarantine Verification |
|---|---|---|---|---|---|
| `long_prompts` | 60 | 60 | `cnn_dailymail` (`3.0.0`) | Unselected articles offset $\ge 300$, token length filtered up to 2048 | 100% unselected records; 0 overlap with core |
| `distractors` | 50 | 50 | `boolq` (`train`) | Contextual distractor prefix injection | Sourced from unselected train offset $\ge 100$ |
| `paraphrased` | 50 | 50 | `gsm8k` (`train`) | Syntactic instruction paraphrasing | Sourced from unselected train offset $\ge 200$ |
| `formatting_variations` | 40 | 40 | `bbh` (10 diverse tasks) | Strict JSON specification schema wrapper | Sourced from unselected test offset $\ge 20$ |
| `harder_reasoning_math` | 40 | 40 | `math` (4 subjects) | Filtered Hendrycks MATH Level 5 competition problems | Sourced from unselected train offset $\ge 20$ |
| `code_variations` | 30 | 30 | `mbpp` (`full`) | Strict self-contained typed specification wrapper | Sourced from unselected test offset $\ge 30$ |
| `held_out_domain_task` | 30 | 30 | `super_glue` (`record`) | Reading Comprehension with Commonsense Reasoning | `record` task was completely withheld from core benchmark |
| **Total Stress** | **300** | **300** | — | — | **Zero core records perturbed** |

Every stress record carries complete provenance:
- `stress_transformation`: Explicit transformation label.
- `original_source_dataset`: Original source dataset identifier.
- `original_source_id`: Stable source identifier of the unselected underlying record.

---

## 5. Anti-Leakage & Integrity Verification

All three anti-leakage barriers mandated by `reports/benchmark_split_policy_v1.md` were evaluated and verified by `SplitGuard`:

1. **Source ID Quarantine:**
   - $\text{train} \cap \text{validation} = \emptyset$ (0 overlapping source keys)
   - $\text{train} \cap \text{test} = \emptyset$ (0 overlapping source keys)
   - $\text{validation} \cap \text{test} = \emptyset$ (0 overlapping source keys)
   - $\text{core} \cap \text{stress} = \emptyset$ (0 overlapping source keys)
   - **Result:** PASSED.

2. **Prompt Deduplication Guard:**
   - 2,300 total candidate records evaluated using lowercase, whitespace-collapsed normalization.
   - 0 duplicate prompts exist in the final generated benchmark.
   - **Result:** PASSED.

3. **Policy-Specific Invariants:**
   - HumanEval records in `train`: **0** (Policy requires 0).
   - HumanEval records in `validation`: **0** (Policy requires 0).
   - HumanEval records in `test`: **25** (Policy requires 25).
   - MMLU `source_split: test` in `train` or `validation`: **0**.
   - SuperGLUE `boolq` occurrences: **0** (Blacklisted).
   - Unlabelled test splits (e.g. HellaSwag test, SuperGLUE test): **0**.
   - **Result:** PASSED.

---

## 6. Rejected Records Audit

During sampling, candidate stream filtering identified and rejected **10 duplicate records** from `abisee/cnn_dailymail`:

| # | Dataset | Source ID / Hash | Reason for Rejection | Action Taken |
|---|---|---|---|---|
| 1 | `cnn_dailymail` | `8b0a818edcbbefb79dd40a19b08119f6848083db` | Duplicate of prior Amman article | Skipped; next candidate sampled |
| 2 | `cnn_dailymail` | `6235695e9e69aa35c488636fc63bae84d1e2ca1f` | Duplicate of prior Washington article | Skipped; next candidate sampled |
| 3 | `cnn_dailymail` | `f85e12129885fad0eabd6a1c6b6b24dbf8948db5` | Duplicate of prior North Carolina article | Skipped; next candidate sampled |
| 4 | `cnn_dailymail` | `eebebdc4fa08a6e927ab958a39d91f888550afec` | Duplicate prompt content | Skipped; next candidate sampled |
| 5 | `cnn_dailymail` | `c8ebf6f48257a704202c7b6df238bc1cbfa838d7` | Duplicate prompt content | Skipped; next candidate sampled |
| 6 | `cnn_dailymail` | `ea29e498c303964fc39c856ddf7037d5dfbc93fb` | Duplicate prompt content | Skipped; next candidate sampled |
| 7 | `cnn_dailymail` | `499fdb9bf024f6fbfd605792594372df22ec12f0` | Duplicate prompt content | Skipped; next candidate sampled |
| 8 | `cnn_dailymail` | `b037b44fc06a2d89d95dc7b4453b87371a34fe2f` | Duplicate prompt content | Skipped; next candidate sampled |
| 9 | `cnn_dailymail` | `8bd6dc9f3ebb23fe20f74c1956184064b5b95509` | Duplicate prompt content | Skipped; next candidate sampled |
| 10 | `cnn_dailymail` | `32f0c7b0845e88c7ff34bb9d358a86e029dc2261` | Duplicate prompt content | Skipped; next candidate sampled |

All rejected record metadata is recorded in `data/processed/benchmark_manifest.json`.

---

## 7. Deterministic Reproducibility Check

Benchmark sampling was run twice from scratch (`Run 1` into `data/processed`, `Run 2` into `data/processed_verify`). Every output file was verified:

| File Name | SHA-256 Checksum | Match Status |
|---|---|---|
| `benchmark_train.jsonl` | `f6794dabef15ca444c16ee3d38281d519431c9f33346111e3f62d053dfc14718` | **100% Byte-Identical** |
| `benchmark_validation.jsonl` | `965eb3ba28c2da511c064bdf4fb3058975bb878288d5171f14cc7dd7aecdc51a` | **100% Byte-Identical** |
| `benchmark_test.jsonl` | `b29d8cf9b050d0a4faae95170936b3973ee3adcab6ef26c00e5ccf603ef99f7d` | **100% Byte-Identical** |
| `benchmark_stress.jsonl` | `36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747` | **100% Byte-Identical** |
| `benchmark_manifest.json` | Manifest contents match (counts, checksums, categories, guards) | **100% Identical** |

---

## 8. Test Suite Status

Full repository test suite execution via `python -m pytest`:

```
collected 205 items
============================= 205 passed in 0.69s =============================
```

- Baseline before sampling layer: **169 passed**
- Added test files:
  - `tests/test_math_adapter.py` (8 tests)
  - `tests/test_humaneval_adapter.py` (3 tests)
  - `tests/test_hellaswag_adapter.py` (3 tests)
  - `tests/test_split_guard.py` (8 tests)
  - `tests/test_sampler.py` (14 tests)
- **Current Total:** **205 passed, 0 failed**.
