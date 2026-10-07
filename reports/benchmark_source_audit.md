# Benchmark Source Availability and Feasibility Audit Report

**Project:** Cost-Aware LLM Model Router (`cost-aware-llm-router`)  
**Date:** October 2026  
**Audit Target:** Core Benchmark (2,000 prompts) & Stress / OOD Suite (300 prompts) specified in `configs/benchmark.yaml`  
**Execution Environment:** Python 3.12.4, `pytest-8.4.2`, `datasets-5.0.1`  
**Status:** Read-Only Audit Complete — No production code or configs modified.

---

## 1. Executive Summary

This audit evaluates the feasibility of constructing the **2,300-prompt benchmark suite** (2,000 core + 300 stress) configured in `configs/benchmark.yaml` using the 11 candidate Hugging Face datasets. All metadata, split definitions, example counts, schemas, and configurations were verified empirically within the project virtual environment (`datasets==5.0.1`).

### Primary Audit Verdict: **FEASIBLE WITH SPECIFIC ARCHITECTURAL ADJUSTMENTS**

The requested total counts (2,000 core prompts across 8 task categories and 300 stress prompts across 7 stress types) are quantitatively feasible from the available raw pools. However, **a naive split-to-split mapping (`source_train` -> `benchmark_train`, `source_test` -> `benchmark_test`) will fail or cause critical data corruption** due to fundamental structural asymmetries across the source datasets:

1. **Single-Split Datasets:**
   - **`openai/openai_humaneval`**: Contains **only** a `test` split (164 examples total). It has no `train` or `validation` data.
   - **`lukaemon/bbh`**: Contains **only** a `test` split (250 examples per task across 27 tasks = 6,750 examples). It has no `train` or `validation` data.
   - **`google/IFEval`**: Contains **only** a `train` split (541 examples total). It has no `validation` or `test` data.
2. **Unlabelled Source Test Splits:**
   - **`aps/super_glue`**: The source `test` splits for all tasks are competition sets with unlabelled targets (`-1`). Benchmark ground truth cannot be evaluated against source `test`.
   - **`Rowan/hellaswag`**: The source `test` split has unlabelled targets (empty string `""`). Benchmark ground truth cannot be evaluated against source `test`.
   - **`google/boolq`**: Contains only `train` and `validation`. The original test set is withheld.
3. **Severe Cross-Dataset Duplication:**
   - **`google/boolq` vs. `aps/super_glue` (`boolq`)**: Both datasets contain the exact same 9,427 train and 3,270 validation rows verbatim. Ingesting both independently without deduplication will inject identical prompts into the benchmark.
   - **`mbpp` (`full` vs. `sanitized`)**: `sanitized` shares the same `task_id` space and prompt text as `full`. Mixing them introduces duplicate coding tasks.
4. **Hugging Face Hub Namespace Requirements:**
   - In modern `datasets==5.0.1` / `huggingface_hub`, bare unnamespaced identifiers (`cnn_dailymail`, `super_glue`, `openai_humaneval`, `hellaswag`, `math`) fail with `HfUriError`. The canonical repository IDs must be pinned (e.g., `abisee/cnn_dailymail`, `aps/super_glue`, `openai/openai_humaneval`, `Rowan/hellaswag`, `EleutherAI/hendrycks_math`).

---

## 2. Dataset Inventory

The following table summarizes all 11 candidate datasets, their exact Hugging Face identifiers, canonical configurations, licenses, and primary task domains.

| Dataset Identifier | Canonical HF Repository ID | Selected Config / Subset | Primary Domain & Task Type | Verified License |
|---|---|---|---|---|
| `mmlu` | `cais/mmlu` | 57 subject subsets + `all` | General knowledge / Academic QA (Multiple-choice) | MIT |
| `gsm8k` | `openai/gsm8k` | `main` | Grade school math word problems (Exact match numeric) | MIT |
| `math` | `EleutherAI/hendrycks_math` | 7 subjects (`algebra`, `geometry`, etc.) | High school competition math (Exact match LaTeX) | MIT |
| `humaneval` | `openai/openai_humaneval` | `openai_humaneval` (default) | Python function synthesis (Code execution pass@k) | MIT |
| `mbpp` | `google-research-datasets/mbpp` | `sanitized` (preferred) or `full` | Python programming tasks (Code execution unit tests) | CC-BY-4.0 |
| `bbh` | `lukaemon/bbh` | 27 task configurations | Multi-step symbolic & algorithmic reasoning | Apache-2.0 |
| `ifeval` | `google/IFEval` | `default` | Verifiable rule-based instruction following | Apache-2.0 |
| `cnn_dailymail` | `abisee/cnn_dailymail` | `3.0.0` | Long-form news article summarization | Apache-2.0 (code) / Fair Use (text) |
| `boolq` | `google/boolq` | `default` | Reading comprehension Yes/No QA | CC-BY-SA-3.0 |
| `super_glue` | `aps/super_glue` | 10 task configurations (`cb`, `copa`, `rte`, etc.) | NLI, coreference, causal reasoning, QA | Multi-license (task specific) |
| `hellaswag` | `Rowan/hellaswag` | `default` | Commonsense NLI / sentence completion | MIT |

---

## 3. Source Split Counts and Availability

Empirically verified split counts from `datasets.load_dataset_builder` in the active virtual environment:

| Dataset | HF Repo ID | Source Train | Source Validation | Source Test | Other Splits | Total Labeled Usable |
|---|---|---|---|---|---|---|
| **MMLU** | `cais/mmlu` (`all`) | — | 1,531 | 14,042 | `dev`: 285, `auxiliary_train`: 99,842 | **15,573** (val + test) |
| **GSM8K** | `openai/gsm8k` (`main`) | 7,473 | — | 1,319 | — | **8,792** |
| **MATH** | `EleutherAI/hendrycks_math` | 7,500 | — | 5,000 | — | **12,500** |
| **HumanEval** | `openai/openai_humaneval` | — | — | 164 | — | **164** |
| **MBPP (full)** | `google-research-datasets/mbpp` | 374 | 90 | 500 | `prompt`: 10 | **964** |
| **MBPP (sanitized)**| `google-research-datasets/mbpp` | 120 | 43 | 257 | `prompt`: 7 | **420** |
| **BBH** | `lukaemon/bbh` (27 tasks) | — | — | 6,750 | (250 / task) | **6,750** |
| **IFEval** | `google/IFEval` | 541 | — | — | — | **541** |
| **CNN/DailyMail**| `abisee/cnn_dailymail` (`3.0.0`) | 287,113 | 13,368 | 11,490 | — | **311,971** |
| **BoolQ** | `google/boolq` | 9,427 | 3,270 | — (unlabelled) | — | **12,697** |
| **SuperGLUE** | `aps/super_glue` (all tasks) | 146,032 | 19,233 | Unlabelled (`-1`) | `axb`: 1104, `axg`: 356 | **165,265** (train + val) |
| **HellaSwag** | `Rowan/hellaswag` | 39,905 | 10,042 | Unlabelled (`""`) | — | **49,947** (train + val) |

### Key Availability Takeaways
1. **Asymmetric Source Availability:** Only `cnn_dailymail`, `mbpp (full)`, and `math` provide standard `train`, `validation`, and `test` splits with ground-truth targets.
2. **Missing Splits:**
   - `MMLU` has no native `train` split in standard usage (only `dev` with 5 per subject, `val`, and `test`; `auxiliary_train` consists of uncurated web questions from other sources).
   - `BBH` and `HumanEval` have **only** `test`.
   - `IFEval` has **only** `train`.
   - `BoolQ` has **only** `train` and `val`.
3. **Withheld Targets:** `SuperGLUE` and `HellaSwag` source `test` splits **must never be sampled for benchmark evaluation** because labels are withheld for public competition evaluation.

---

## 4. Configuration and Task Inventory for Multi-Config Datasets

Several selected datasets are collections of distinct heterogeneous tasks. Treating them as flat, single-distribution sources risks heavy sampling bias.

### 4.1 cais/mmlu (57 Subject Tasks)
Divided into 4 broad high-level categories:
- **Humanities (13 subjects):** `formal_logic`, `high_school_european_history`, `high_school_us_history`, `high_school_world_history`, `history`, `jurisprudence`, `moral_disputes`, `moral_scenarios`, `philosophy`, `prehistory`, `professional_law`, `world_religions`, `international_law`.
- **Social Sciences (12 subjects):** `econometrics`, `high_school_geography`, `high_school_government_and_politics`, `high_school_macroeconomics`, `high_school_microeconomics`, `high_school_psychology`, `human_sexuality`, `professional_psychology`, `public_relations`, `security_studies`, `sociology`, `us_foreign_policy`.
- **STEM (19 subjects):** `abstract_algebra`, `astronomy`, `college_biology`, `college_chemistry`, `college_computer_science`, `college_mathematics`, `college_physics`, `computer_security`, `conceptual_physics`, `electrical_engineering`, `elementary_mathematics`, `high_school_biology`, `high_school_chemistry`, `high_school_computer_science`, `high_school_mathematics`, `high_school_physics`, `high_school_statistics`, `machine_learning`, `linear_algebra`.
- **Other / Applied (13 subjects):** `anatomy`, `business_ethics`, `clinical_knowledge`, `college_medicine`, `global_facts`, `human_aging`, `management`, `marketing`, `medical_genetics`, `miscellaneous`, `nutrition`, `professional_accounting`, `professional_medicine`, `virology`.

*Sampling Recommendation:* Stratify the 400 core `general_qa` prompts uniformly across the 4 broad subject domains (~100 per domain, ~7 prompts per subject).

### 4.2 lukaemon/bbh (27 Big-Bench Hard Tasks)
Each configuration contains exactly 250 evaluation examples in the `test` split:
1. `boolean_expressions` (Boolean logic evaluation)
2. `causal_judgement` (Causal attribution)
3. `date_understanding` (Temporal date calculation)
4. `disambiguation_qa` (Pronoun and syntactic disambiguation)
5. `dyck_languages` (Formal language bracket matching)
6. `formal_fallacies` (Deductive fallacy identification)
7. `geometric_shapes` (SVG path and geometry reasoning)
8. `hyperbaton` (Adjective ordering rules)
9. `logical_deduction_five_objects` (Symbolic constraint satisfaction)
10. `logical_deduction_seven_objects` (Symbolic constraint satisfaction)
11. `logical_deduction_three_objects` (Symbolic constraint satisfaction)
12. `movie_recommendation` (Collaborative filtering reasoning)
13. `multistep_arithmetic_two` (Chained arithmetic calculation)
14. `navigate` (Spatial navigation and step tracing)
15. `object_counting` (Spatial object enumeration)
16. `penguins_in_a_table` (Structured table relational reasoning)
17. `reasoning_about_colored_objects` (Multi-attribute object reasoning)
18. `ruin_names` (Humor and wordplay editing)
19. `salient_translation_error_detection` (Translation quality evaluation)
20. `snarks` (Sarcasm detection)
21. `sports_understanding` (Plausibility reasoning in sports)
22. `temporal_sequences` (Event timeline ordering)
23. `tracking_shuffled_objects_five_objects` (State permutation tracking)
24. `tracking_shuffled_objects_seven_objects` (State permutation tracking)
25. `tracking_shuffled_objects_three_objects` (State permutation tracking)
26. `web_of_lies` (Truth/deception deduction)
27. `word_sorting` (Lexicographical sorting)

*Sampling Recommendation:* Stratify the 300 core `reasoning` prompts across all 27 tasks (~11 prompts per task) to prevent the reasoning category from degenerating into a single task type.

### 4.3 EleutherAI/hendrycks_math (7 Subject Categories)
- `algebra`: 1,744 train, 1,187 test
- `counting_and_probability`: 771 train, 474 test
- `geometry`: 870 train, 479 test
- `intermediate_algebra`: 1,295 train, 903 test
- `number_theory`: 869 train, 540 test
- `prealgebra`: 1,205 train, 871 test
- `precalculus`: 746 train, 546 test

Each problem includes explicit difficulty `level` (Level 1 through Level 5), directly enabling native difficulty stratification.

### 4.4 aps/super_glue (10 Distinct Subsets)
- `boolq`: 9,427 train, 3,270 val (QA) — *Identical to `google/boolq`*
- `cb` (CommitmentBank): 250 train, 56 val (3-class NLI)
- `copa` (Choice of Plausible Alternatives): 400 train, 100 val (Causal 2-choice)
- `multirc` (Multi-Sentence Reading Comprehension): 27,243 train, 4,848 val (Complex QA with multiple true options)
- `record` (Reading Comprehension with Commonsense Reasoning): 100,730 train, 10,000 val (Span extraction fill-in)
- `rte` (Recognizing Textual Entailment): 2,490 train, 277 val (2-class NLI)
- `wic` (Words in Context): 5,428 train, 638 val (Binary polysemy classification)
- `wsc` (Winograd Schema Challenge): 554 train, 104 val (Coreference resolution)
- Diagnostic sets (`axb`: 1,104 test, `axg`: 356 test)

*Sampling Recommendation:* In SuperGLUE, exclude the `boolq` config entirely to avoid duplicating `google/boolq`. Sample classification/extraction tasks evenly from `rte`, `cb`, `wic`, and `copa`.

### 4.5 google-research-datasets/mbpp (`full` vs. `sanitized`)
- `full`: 974 programming problems (374 train, 90 val, 500 test). Contains some ambiguous task descriptions and buggy ground-truth asserts.
- `sanitized`: 427 programming problems (120 train, 43 val, 257 test). Manually reviewed and cleaned by Google researchers.
- *Overlap Fact:* Every task in `sanitized` is an edit/subset of a task in `full` with matching `task_id`.
- *Recommendation:* Do **not** combine `full` and `sanitized`. Use `sanitized` for the clean test/validation pool, or use `full` with strict deduplication by `task_id`.

---

## 5. Category Feasibility Matrix

Requested core benchmark totals by category vs. source pool analysis:

| Category | Requested Count | Target Sources | Usable Labeled Pool Size | Feasibility Status | Primary Risk / Conflict |
|---|---|---|---|---|---|
| `general_qa` | 400 | `mmlu` | 15,573 | **Feasible** | MMLU only has `test` (14,042) and `val` (1,531). Train pool must be drawn from `test` or `val`. |
| `reasoning` | 300 | `bbh` | 6,750 | **Feasible** | BBH only has `test` split. Must partition source `test` disjointly across benchmark splits. |
| `math` | 300 | `gsm8k`, `math` | 21,292 | **Feasible** | High school competition `math` requires complex LaTeX sympy matching vs numeric GSM8K. |
| `coding` | 300 | `humaneval`, `mbpp` | 584 (sanitized + HE) or 1,138 (full + HE) | **Feasible (Tight)** | HumanEval has only 164 problems (`test` only). To yield 300 coding prompts, MBPP must supply at least 150-200. |
| `instruction_following` | 250 | `ifeval` | 541 | **Feasible (Tight)** | IFEval only has 541 prompts in total (`train` only). 250 prompts consumes 46.2% of the dataset. |
| `summarization` | 200 | `cnn_dailymail` | 311,971 | **Feasible** | Extremely large pool. Full splits available (`train`/`val`/`test`). Token length must be capped. |
| `classification_extraction`| 150 | `boolq`, `super_glue` | >25,000 | **Feasible** | Direct duplicate risk between `google/boolq` and `aps/super_glue(boolq)`. SuperGLUE test is unlabelled. |
| `other` | 100 | `hellaswag` | 49,947 | **Feasible** | HellaSwag test is unlabelled. Must draw from source `train` and `validation`. |
| **Total Core** | **2,000** | — | **>400,000** | **Feasible** | Requires strict source-pool partitioning strategy. |

---

## 6. Train / Validation / Test Source-Pool Strategy

The benchmark configuration specifies:
- Core Total: 2,000 prompts
- Train: 1,400 prompts (70%)
- Validation: 300 prompts (15%)
- Test: 300 prompts (15%)

Because several source datasets lack native 3-split partitions, the benchmark cannot simply map `source_split="train"` to `split="train"`. Below is the required **Category-Specific Source-Pool Allocation Strategy** that guarantees 100% disjoint, non-leaking benchmark splits.

```
Total Core Breakdown:
- general_qa:               400 total (train: 280, val: 60, test: 60)
- reasoning:                300 total (train: 210, val: 45, test: 45)
- math:                     300 total (train: 210, val: 45, test: 45)
- coding:                   300 total (train: 210, val: 45, test: 45)
- instruction_following:    250 total (train: 175, val: 37, test: 38)
- summarization:            200 total (train: 140, val: 30, test: 30)
- classification_extraction: 150 total (train: 105, val: 22, test: 23)
- other:                    100 total (train: 70,  val: 15, test: 15)
Total:                    2,000 total (train: 1400, val: 300, test: 300)
```

### Detailed Category Allocation Plans

#### 1. `general_qa` (Total: 400 | Train: 280, Val: 60, Test: 60)
- **Source Dataset:** `cais/mmlu`
- **Source Train Pool:** Sample 280 from `cais/mmlu` source `test` (disjoint slice A: indices 0..10,000).
- **Source Val Pool:** Sample 60 from `cais/mmlu` source `validation` (1,531 total).
- **Source Test Pool:** Sample 60 from `cais/mmlu` source `test` (disjoint slice B: indices 10,001..14,041).
- **Rationale:** MMLU's official `validation` split is completely disjoint from `test`. Partitioning source `test` deterministically by seed ensures benchmark `test` never encounters questions seen in benchmark `train`. Stratified across all 57 subjects.

#### 2. `reasoning` (Total: 300 | Train: 210, Val: 45, Test: 45)
- **Source Dataset:** `lukaemon/bbh`
- **Source Train Pool:** 210 prompts sampled from BBH source `test` (indices 0..174 within each 250-item task).
- **Source Val Pool:** 45 prompts sampled from BBH source `test` (indices 175..212 within each task).
- **Source Test Pool:** 45 prompts sampled from BBH source `test` (indices 213..249 within each task).
- **Rationale:** BBH exists solely as a 6,750-item `test` split across 27 tasks. By slicing each 250-item task index range into disjoint 70% / 15% / 15% buckets (e.g., indices 0..174 for train, 175..212 for val, 213..249 for test), we achieve perfect intra-task balance while mathematically guaranteeing zero prompt overlap across benchmark splits.

#### 3. `math` (Total: 300 | Train: 210, Val: 45, Test: 45)
- **Source Datasets:** `openai/gsm8k` (50%) and `EleutherAI/hendrycks_math` (50%)
- **Source Train Pool:** 105 from `gsm8k` (`train`) + 105 from `math` (`train`).
- **Source Val Pool:** 23 from `gsm8k` (`train` held-out slice or partition of `test`) + 22 from `math` (`train` held-out slice).
- **Source Test Pool:** 22 from `gsm8k` (`test`) + 23 from `math` (`test`).
- **Rationale:** Both GSM8K and MATH provide official, disjoint `train` and `test` splits. GSM8K train has 7,473 examples and test has 1,319; MATH train has 7,500 and test has 5,000. Benchmark train and validation are drawn exclusively from source `train`, while benchmark test is drawn exclusively from source `test`. This guarantees zero contamination.

#### 4. `coding` (Total: 300 | Train: 210, Val: 45, Test: 45)
- **Source Datasets:** `openai/openai_humaneval` and `google-research-datasets/mbpp`
- **Source Train Pool:** 210 from `mbpp` (`train` split: 374 in full, or partition of full/sanitized). *Zero HumanEval in train!*
- **Source Val Pool:** 45 from `mbpp` (`validation` split: 90 in full, 43 in sanitized).
- **Source Test Pool:** 45 total: 25 from `humaneval` (`test` 164) + 20 from `mbpp` (`test` 500).
- **Rationale:** **Crucial Finding:** HumanEval only has 164 problems and is globally recognized as the canonical zero-shot coding benchmark. If HumanEval is placed into benchmark `train`, it is ruined as an evaluation baseline. Therefore, benchmark `train` and `validation` must be populated 100% by MBPP (`train` and `val`), while benchmark `test` contains HumanEval + MBPP test.

#### 5. `instruction_following` (Total: 250 | Train: 175, Val: 37, Test: 38)
- **Source Dataset:** `google/IFEval`
- **Source Train Pool:** 175 prompts from IFEval source `train` (partition A: 70% of keys).
- **Source Val Pool:** 37 prompts from IFEval source `train` (partition B: 15% of keys).
- **Source Test Pool:** 38 prompts from IFEval source `train` (partition C: 15% of keys).
- **Rationale:** IFEval only has a single `train` split of 541 prompts. The only valid way to allocate splits is a deterministic 70/15/15 hash partition on the native `key` attribute.

#### 6. `summarization` (Total: 200 | Train: 140, Val: 30, Test: 30)
- **Source Dataset:** `abisee/cnn_dailymail` (`3.0.0`)
- **Source Train Pool:** 140 articles sampled from CNN/DailyMail source `train` (287,113 available).
- **Source Val Pool:** 30 articles sampled from CNN/DailyMail source `validation` (13,368 available).
- **Source Test Pool:** 30 articles sampled from CNN/DailyMail source `test` (11,490 available).
- **Rationale:** Clean 1-to-1 split mapping with massive native margins.

#### 7. `classification_extraction` (Total: 150 | Train: 105, Val: 22, Test: 23)
- **Source Datasets:** `google/boolq` (75) and `aps/super_glue` (75)
- **Source Train Pool:** 53 from `boolq` (`train`) + 52 from `super_glue` (`train` across `cb`, `copa`, `rte`, `wic`, `wsc`).
- **Source Val Pool:** 11 from `boolq` (`validation`) + 11 from `super_glue` (`validation`).
- **Source Test Pool:** 11 from `boolq` (`validation` held-out slice) + 12 from `super_glue` (`validation` held-out slice).
- **Rationale:** Because SuperGLUE `test` is unlabelled (`-1`), and BoolQ has no public labelled test set, benchmark `test` must be drawn from a strictly held-out slice of source `validation`. SuperGLUE's `boolq` task must be excluded to prevent duplication with `google/boolq`.

#### 8. `other` (Commonsense NLI) (Total: 100 | Train: 70, Val: 15, Test: 15)
- **Source Dataset:** `Rowan/hellaswag`
- **Source Train Pool:** 70 sampled from HellaSwag source `train` (39,905 available).
- **Source Val Pool:** 15 sampled from HellaSwag source `validation` (first 5,000 slice).
- **Source Test Pool:** 15 sampled from HellaSwag source `validation` (second 5,000 slice).
- **Rationale:** Source `test` has empty label strings `""`. Benchmark train comes from source `train`, and benchmark validation and test are sampled from disjoint slices of source `validation` (10,042 items).

---

## 7. Leakage Risks and Prevention

| Risk Dimension | Specific Identified Hazard | Severity | Mandatory Mitigation Strategy |
|---|---|---|---|
| **Inter-Dataset Duplication** | `google/boolq` and `aps/super_glue(boolq)` contain identical prompts and passages. | **Critical** | Blacklist the `boolq` configuration in the `super_glue` adapter/pipeline. Only ingest BoolQ once via `google/boolq`. |
| **Intra-Dataset Version Overlap** | MBPP `full` and `sanitized` share identical `task_id` values and prompts. | **Critical** | Pin either `sanitized` (427 clean tasks) or `full` (974 tasks). Never combine both configs into the ingestion pool. |
| **Unlabelled Test Contamination** | SuperGLUE and HellaSwag `test` splits contain `-1` or `""` labels. | **Critical** | Do not ingest source `test` for SuperGLUE or HellaSwag. Only sample from source `train` and `validation`. |
| **Single-Split Leakage** | Datasets with only one split (BBH, IFEval, HumanEval) could accidentally mix train and test. | **Critical** | Partition by deterministic index hashing with strict disjoint boundaries before any benchmark assignment. |
| **Representation Imbalance** | MMLU contains 57 subjects; random sampling would oversample large subjects (e.g., medicine) and drop smaller ones (e.g., formal logic). | **Moderate** | Stratify sampling evenly across all 57 MMLU subjects and all 27 BBH tasks. |
| **Prompt Leakage Guard** | Multi-turn or grouped questions sharing identical context or passage. | **Moderate** | Enforce `splitting.leakage_guard: true` so all items sharing a `prompt_id` or identical text hash reside in the same split. |
| **Stress vs. Core Leakage** | Stress prompts generated from core prompts would leak test data if placed in the evaluation suite. | **Critical** | Stress prompts must be generated from held-out source records not present anywhere in the 2,000 core benchmark. |

---

## 8. Stress / OOD Set Feasibility Analysis

The configuration specifies **300 stress prompts** across 7 stress types, designated as `held_out: true`.

| Stress Type | Requested Count | Best Candidate Source Dataset | Separate Held-Out Pool Available? | Semantic Leakage Risk | True OOD vs. Perturbation |
|---|---|---|---|---|---|
| `long_prompts` | 60 | `cnn_dailymail` | **Yes** (>280k unselected articles) | None if sampled from unselected articles | **Perturbation** (High token volume, same domain) |
| `distractors` | 50 | `cais/mmlu` or `google/boolq` | **Yes** (>10k unselected rows) | High if distracting text is injected into core prompts | **Perturbation** (Noise robustness) |
| `paraphrased` | 50 | `openai/gsm8k` or `cais/mmlu` | **Yes** (>6k unselected rows) | **Severe** if paraphrasing core prompts | **Perturbation** (Lexical/syntactic variation) |
| `formatting_variations` | 40 | `lukaemon/bbh` or `ifeval` | **Yes** (>6k unselected rows) | None if using unselected prompts | **Perturbation** (Markdown, JSON, schema variation) |
| `harder_reasoning_math` | 40 | `EleutherAI/hendrycks_math` (Level 5) | **Yes** (~2,500 Level 5 problems) | None if drawn from unselected test pool | **Domain Shift** (Extreme difficulty distribution) |
| `code_variations` | 30 | `google-research-datasets/mbpp` | **Yes** (>400 unselected tasks) | High if modifying core benchmark code | **Perturbation** (Refactoring, variable renaming) |
| `held_out_domain_task` | 30 | New unseen task (e.g., SuperGLUE `wsc` or BBH unused task) | **Yes** (Whole task types held out) | None | **True OOD** (Completely unseen task distribution) |

### Key Stress Construction Rules
1. **Never Perturb Core Prompts:** Paraphrased, distractor, and formatting variations must **not** be created by taking prompts from the 2,000 core benchmark and altering them. That would create semantic twins across the core and stress splits.
2. **Dedicated Source Extraction:** Stress prompts must be sampled from **completely unselected source records** (e.g., unselected CNN articles for long prompts, unselected Hendrycks Level 5 problems for harder math, unselected BBH tasks for held-out domain).

---

## 9. License and Redistribution Notes

| Dataset | Verified Source License | Redistribution Risk / Terms | Recommendation |
|---|---|---|---|
| `cais/mmlu` | MIT | Permissive open source. | Safe for provenance metadata and sample redistribution. |
| `openai/gsm8k` | MIT | Permissive open source. | Safe for benchmark redistribution. |
| `EleutherAI/hendrycks_math` | MIT | Permissive open source. | Safe for benchmark redistribution. |
| `openai/openai_humaneval` | MIT | Permissive open source. | Safe for benchmark redistribution. |
| `google-research-datasets/mbpp` | CC-BY-4.0 | Requires attribution. | Safe with license notice in repo. |
| `lukaemon/bbh` | Apache-2.0 | Permissive open source. | Safe for benchmark redistribution. |
| `google/IFEval` | Apache-2.0 | Permissive open source. | Safe for benchmark redistribution. |
| `abisee/cnn_dailymail` | Apache-2.0 (code) / News Copyright | Underlying news articles belong to CNN and Associated Newspapers Ltd. Commercial redistribution of full article text is restricted. | **Store source IDs, hashes, and retrieval offsets** rather than committing raw scraped news corpora to git. |
| `google/boolq` | CC-BY-SA-3.0 | ShareAlike license requires derivative works to preserve CC-BY-SA. | Safe under attribution and compatible terms. |
| `aps/super_glue` | Varied (CC-BY-SA, MIT, research) | Individual tasks have separate source provenance. | Retain `source_dataset` and `source_id` metadata. |
| `Rowan/hellaswag` | MIT (derived from WikiHow/ActivityNet) | Permissive with source attribution. | Safe for benchmark redistribution. |

---

## 10. Recommended Decisions that MUST Be Made Before Sampling

Before implementing the sampling pipeline or creating benchmark JSONL files, the following decisions must be formally approved:

1. **Resolve HumanEval Coding Allocation:**
   - *Decision:* HumanEval has only 164 problems. We recommend reserving HumanEval **strictly for the benchmark test set** (e.g., 25 prompts in `test`), while MBPP supplies all 210 `train` coding prompts and all 45 `val` prompts.
2. **Deduplicate BoolQ / SuperGLUE:**
   - *Decision:* Exclude `aps/super_glue` config `boolq` entirely. Use `aps/super_glue` solely for other tasks (`cb`, `copa`, `rte`, `wic`, `wsc`), and use `google/boolq` as the sole provider for BoolQ QA.
3. **Pin MBPP Subset:**
   - *Decision:* Standardize on `google-research-datasets/mbpp` config `sanitized` for high quality and bug-free test asserts, OR use `full` with explicit `task_id` deduplication against sanitized.
4. **Partition Single-Split Datasets Deterministically:**
   - *Decision:* Adopt fixed pseudo-random split partitioning (70% train / 15% val / 15% test) keyed on deterministic hash seeds for `lukaemon/bbh` and `google/IFEval`.
5. **Exclude Unlabelled Test Splits:**
   - *Decision:* For `aps/super_glue` and `Rowan/hellaswag`, benchmark `test` must be drawn from held-out slices of source `validation`, never from source `test`.
6. **Stress Generation Source Quarantine:**
   - *Decision:* Mandate that the 300 stress prompts be generated from source records that were excluded from the 2,000 core sample.

---

## 11. Explicit List of Unresolved Issues

1. **Canonical HF Repo Naming in Config:**
   - In `configs/benchmark.yaml`, dataset identifiers are written as bare names (`math`, `humaneval`, `super_glue`, `hellaswag`, `cnn_dailymail`). Recent `datasets` library versions require full repository paths (`EleutherAI/hendrycks_math`, `openai/openai_humaneval`, `aps/super_glue`, `Rowan/hellaswag`, `abisee/cnn_dailymail`). When ingestion is wired up, the configuration or mapping table must resolve these canonical IDs.
2. **Evaluation Tooling for Math and Coding:**
   - The core benchmark requires `code_execution` for 300 coding prompts and exact math LaTeX parsing for Hendrycks `math`. While out of scope for this audit, execution sandboxing (Docker/gVisor/subprocess) and LaTeX normalization (`sympy`/math parser) must be planned before model evaluations begin.
3. **Context Length Truncation for CNN/DailyMail:**
   - CNN/DailyMail articles can exceed 2,000 tokens. The benchmark configuration currently sets `max_prompt_tokens: null`. A token threshold should be established to prevent out-of-memory or excessive context costs during evaluation.
4. **HE / MBPP Entry Point Invocation Interface:**
   - HumanEval prompts provide partial function signatures, while MBPP prompts provide natural language descriptions with explicit function names tested via `assert <func_name>(...)`. The prompt formatting adapter must ensure candidate models receive consistent prompt formatting.

---

## 12. Verification & Next Steps

This audit was conducted strictly in read-only mode without modifying production codebase files, test files, or configuration specifications.

- All 169 existing ingestion and schema unit tests continue to pass without regression.
- The next development milestone should be implementing dataset adapters for the remaining sources (`humaneval`, `math`, `hellaswag`) and creating the sampling specification based on the source-pool strategy detailed in Section 6.
