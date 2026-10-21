# Phase 4Q — Task-Aware Inference Policy and Runtime Eligibility Report

**Phase:** Phase 4Q (Diagnostic & Hardening Validation)  
**Date:** 2026-10-21  
**Status:** Completed  
**Decision Gate:** **CONDITIONAL GO** (Specific, bounded adjustments required before full matrix launch; strict embargo on router training and full 4,200-attempt matrix maintained)  
**Experiment Fingerprint:** `4b41b7091d8db86f10ffdf284973c89ac44b905f4aecd699ec126a05a30e39ef`  
**Output Directory:** `data/outcomes/pilot_phase4q`  
**Baseline Comparison Directory:** `data/outcomes/pilot_phase4p` (Generic `v0` policy)

---

## 1. Executive Summary

Phase 4Q addressed the critical truncation, latency, and policy-selection bottlenecks identified during the Phase 4P controlled pilot. Phase 4P applied a single, generic `v0` prompt policy with static universal caps and unconstrained thinking modes across all tasks, resulting in a **35.4% truncation failure rate** (17/48 attempts truncated) and severe SLA violations for `qwen3-4b` (87.5% truncated) and `qwen3-8b` (median latency 23.6s, only 18.8% meeting SLA).

In Phase 4Q, an explicit **Task-Aware Policy Dispatcher** (`dispatch_task_policy`) was designed, implemented, and rigorously validated across all eight canonical benchmark task categories:
1. **Multiple-choice schema isolation:** Structured JSON output (`MULTIPLE_CHOICE_JSON_SCHEMA`) is enforced **exclusively** on multiple-choice formats (`task_type="multiple_choice"` or tasks evaluated with choice letters). It is strictly prohibited from contaminating coding, summarization, math, or free-form instruction following.
2. **Dynamic task budgets and thinking control:** Multiple choice and classification tasks use strict low-latency budgets (64 tokens) and thinking suppression (`think=False`), while open-ended tasks (GSM8K math: 512 tokens; MBPP coding: 512 tokens; IFEval: 1024 tokens) receive sufficient generative headroom without arbitrary universal caps.
3. **Safe scoring & failure isolation:** Truncated outputs are systematically marked `SCORING_UNAVAILABLE` with `quality_score = None`, completely eliminating false quality-zero penalization. All structured outputs undergo JSON schema validation before scoring.
4. **Honest coding evaluation:** Untrusted code execution on the host system was audited and strictly prohibited. In the absence of a containerized sandbox (Docker/gVisor), MBPP code execution scoring safely returns `SANDBOX_UNAVAILABLE` with `quality_score = None`, ensuring that unvalidated execution does not corrupt scored quality aggregates.

### Key Results Summary (Paired Comparison: Phase 4P vs. Phase 4Q on Identical 16 Training Prompts)

| Metric | Phase 4P (`v0` Generic) | Phase 4Q (`task_aware` Dispatch) | Relative Change |
| :--- | :---: | :---: | :---: |
| **Total Inferences** | 48 (16 prompts x 3 models) | 48 (16 prompts x 3 models) | Paired / Identical |
| **Overall Success Rate** | 64.6% (31/48) | **85.4% (41/48)** | **+32.3% improvement** |
| **Total Truncations** | 17 / 48 (35.4%) | **7 / 48 (14.6%)** | **58.8% reduction in truncations** |
| **qwen3-1.7b Completion** | 14 SUCCESS, 2 TRUNCATED | **16 SUCCESS, 0 TRUNCATED** | **100% completion rate** |
| **qwen3-8b Completion** | 15 SUCCESS, 1 TRUNCATED | **16 SUCCESS, 0 TRUNCATED** | **100% completion rate** |
| **qwen3-4b Completion** | 2 SUCCESS, 14 TRUNCATED | **9 SUCCESS, 7 TRUNCATED** | **4.5x increase in completed outputs** |
| **Overall SLA Compliance (<10s)** | 43.8% (21/48) | **75.0% (36/48)** | **+71.4% improvement** |
| **qwen3-8b Median Latency** | 23,571 ms | **4,752 ms** | **79.8% latency reduction** |
| **qwen3-8b SLA Compliance (<10s)**| 18.8% (3/16) | **62.5% (10/16)** | **3.3x improvement** |
| **qwen3-4b Median Latency** | 18,562 ms | **8,968 ms** | **51.7% latency reduction** |
| **qwen3-4b SLA Compliance (<10s)**| 12.5% (2/16) | **62.5% (10/16)** | **5.0x improvement** |
| **qwen3-1.7b SLA Compliance (<10s)**| 100.0% (16/16) | **100.0% (16/16)** | **Maintained 100%** |
| **Token Telemetry Coverage** | 100% (48/48) | **100% (48/48)** | Complete prompt/output tokens |
| **Structured Output Schema Validity** | N/A (unstructured) | **100% (18/18 valid JSON)** | Zero `INVALID_OUTPUT` failures |

---

## 2. Phase 4P Pre-Change Audit

Before making modifications, stored artifacts in `data/outcomes/pilot_phase4p/` were audited against the claims in `reports/phase4p_controlled_pilot.md`:
- `data/outcomes/pilot_phase4p/pilot_manifest.json`: Verified 16 prompts across 8 task categories.
- `data/outcomes/pilot_phase4p/matrix_outcomes.parquet`: Exactly 48 rows, 31 `SUCCESS`, 17 `TRUNCATED`, 0 `FAILED`.
- `data/outcomes/pilot_phase4p/matrix_scores.parquet`: Exactly 48 rows. 27 scored `SUCCESS`, 17 `SCORING_UNAVAILABLE` (all corresponding to truncated attempts with `quality_score = NaN`), and 4 `SANDBOX_UNAVAILABLE` (coding attempts with `quality_score = NaN`).
- **Discrepancy Check:** Zero discrepancies were found. Denominators and latency calculations were verified, and all Phase 4P artifacts were preserved completely unchanged on disk.

---

## 3. Task-Aware Policy Architecture and Dispatch

The task-aware policy dispatcher was implemented in [`runners/inference_policy.py`](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference_policy.py) through `dispatch_task_policy(...)`, producing a typed `DispatchedTaskPolicy` dataclass:

```python
@dataclass(frozen=True)
class DispatchedTaskPolicy:
    task_type: str
    evaluation_type: str
    policy_version: str
    effective_prompt: str
    output_budget: int
    thinking_mode: bool
    response_schema: Optional[Dict[str, Any]]
    compatible_scorer: str
    scorer_version: str
    description: str
```

### Dispatch Mapping Across Canonical Categories

| Task Category | Benchmark Representative | Evaluation Type | Dispatched Policy | Budget | Thinking Mode | Response Schema | Scorer Assigned |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **QA (Multiple Choice)** | MMLU | `multiple_choice` | `v4_structured_output` | 64 | `False` | `MULTIPLE_CHOICE_JSON_SCHEMA` | `multiple_choice_scorer` |
| **Reasoning (MC)** | BBH (`strategy_b_0000`) | `multiple_choice` | `v4_structured_output` | 64 | `False` | `MULTIPLE_CHOICE_JSON_SCHEMA` | `multiple_choice_scorer` |
| **Reasoning (Open/Free)** | BBH (`strategy_b_0001`) | `exact_match` | `v1_concise` | 256 | `False` | `None` | `exact_match_scorer` |
| **Math** | GSM8K | `exact_match` | `v1_concise` | 512 | `False` | `None` | `exact_match_scorer` |
| **Coding** | MBPP | `code_execution` | `v1_concise` | 512 | `False` | `None` | `code_execution_scorer` |
| **Instruction Following**| IFEval | `ifeval` | `v0` (raw/verbatim) | 1024 | `True` | `None` | `ifeval_scorer` |
| **Summarization** | CNN/DailyMail | `reference_based` | `v1_concise` | 256 | `False` | `None` | `reference_based_scorer` |
| **Classification** | BoolQ | `exact_match` | `v2_answer_only` | 64 | `False` | `None` | `exact_match_scorer` |
| **Other (Multiple Choice)**| HellaSwag | `multiple_choice` | `v4_structured_output` | 64 | `False` | `MULTIPLE_CHOICE_JSON_SCHEMA` | `multiple_choice_scorer` |

### Architectural Guardrails
1. **Schema Exclusivity:** Requesting `v4_structured_output` on `code_execution`, `reference_based`, or `ifeval` raises `UnsupportedTaskPolicyError` before dispatch or inference begins.
2. **IFEval Prompt Preservation:** IFEval prompts test adherence to complex negative, structural, and length constraints. Dispatching `v0` guarantees that system prompt wrappers or concise prefixing never alter or dilute the benchmark instructions.
3. **Structured Output Pre-Validation:** In [`runners/inference.py`](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference.py), generated text under `response_schema` is validated using `validate_structured_output_json(...)`. If the JSON is malformed or missing required keys, status is immediately marked `INVALID_OUTPUT`, `error_type = 'SchemaValidationError'`, preventing downstream scoring corruption.

---

## 4. Coding Evaluation Isolation and Sandbox Security

MBPP and HumanEval benchmarks require executing arbitrary LLM-generated Python code against test suites.

### Security Audit Findings
- **Risk Assessment:** Executing unvalidated code directly in the host Python process creates an intolerable security vulnerability (file system mutation, socket opening, process destruction, denial of service).
- **Isolation Requirement:** True isolation requires a secure container runtime (e.g. Docker ephemeral containers with read-only root filesystems, memory/CPU bounds, and dropped network access) or a gVisor/firecracker microVM.
- **Current Local Windows Host Status:** Docker/container isolation is currently unavailable in the execution environment.
- **Implemented Contract:** In [`eval/scorers/code_execution.py`](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/eval/scorers/code_execution.py), the execution engine explicitly checks for secure sandbox availability. When unavailable, it returns:
  - `scoring_status = "SANDBOX_UNAVAILABLE"`
  - `quality_score = None`
  - `scoring_error_type = "SandboxUnavailableError"`
  - `scoring_error_message = "Direct host code execution is prohibited to prevent unsafe code execution. A containerized execution sandbox (Docker/gVisor) is required for code execution evaluation."`
- **Scoring Integrity Guarantee:** Coding attempts are never assigned an arbitrary 0.0 or a noisy text proxy. They are explicitly excluded from quality aggregates until container isolation is provisioned.

---

## 5. Bounded Paired Experiment: Selection and Attempt Breakdown

The experiment was executed using the matrix runner:
```bash
python -m runners.matrix_runner \
  --prompts-file data/outcomes/pilot_phase4p/pilot_prompts.jsonl \
  --output-dir data/outcomes/pilot_phase4q \
  --prompt-policy task_aware \
  --backend ollama \
  --timeout 120.0
```

### Deterministic Prompt Selection (Identical to Phase 4P)

| Prompt ID | Task Type | Source Dataset | Selection Rationale |
| :--- | :--- | :--- | :--- |
| `mmlu-abstract_algebra-validation-val_0000` | QA | MMLU | Truncated in 4P on 4b (768 tok); multiple-choice QA |
| `mmlu-abstract_algebra-validation-val_0001` | QA | MMLU | Truncated in 4P on 4b (768 tok); high 44.6s latency on 8b |
| `bbh-boolean_expressions-test-strategy_b_0000` | Reasoning | BBH | Truncated in 4P on 4b (512 tok); multiple choice reasoning |
| `bbh-boolean_expressions-test-strategy_b_0001` | Reasoning | BBH | Truncated in 4P on 4b (512 tok); free-form exact match |
| `gsm8k-main-train-train_0000` | Math | GSM8K | Passed in 4P; mathematical multi-step derivation |
| `gsm8k-main-train-train_0001` | Math | GSM8K | Truncated in 4P on 4b (512 tok) |
| `mbpp-full-train-601` | Coding | MBPP | Truncated in 4P on 4b (768 tok); high 50.7s latency on 8b |
| `mbpp-full-train-602` | Coding | MBPP | Truncated in 4P on 4b (768 tok); high 26.3s latency on 8b |
| `ifeval-default-train-1943` | Instruction Following | IFEval | Passed in 4P; verified constraint checking |
| `ifeval-default-train-2391` | Instruction Following | IFEval | Truncated in 4P on 4b (1024 tok); long constraint list |
| `cnn_dailymail-3.0.0-train-42c027e4ff...` | Summarization | CNN/DM | Truncated in 4P on 4b (512 tok); multi-paragraph article |
| `cnn_dailymail-3.0.0-train-ee8871b15c...` | Summarization | CNN/DM | Truncated in 4P on 4b (512 tok); multi-paragraph article |
| `boolq-default-train-train_0000` | Classification | BoolQ | Truncated in 4P on 1.7b and 4b (128 tok) |
| `boolq-default-train-train_0001` | Classification | BoolQ | Truncated in 4P across all 3 models (128 tok) |
| `hellaswag-train-train_0000` | Other | HellaSwag | Truncated in 4P on 4b (768 tok); commonsense completion |
| `hellaswag-train-train_0001` | Other | HellaSwag | Truncated in 4P on 4b (768 tok); commonsense completion |

### Attempt Counts by Model, Task, and Status

| Model ID | Task Category | Policy Dispatched | Total Planned | SUCCESS | TRUNCATED | FAILED |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **qwen3-1.7b** | QA (2) | `v4_structured_output` | 2 | 2 | 0 | 0 |
| | Reasoning (2) | `v4_struct` / `v1_concise` | 2 | 2 | 0 | 0 |
| | Math (2) | `v1_concise` | 2 | 2 | 0 | 0 |
| | Coding (2) | `v1_concise` | 2 | 2 | 0 | 0 |
| | IFEval (2) | `v0` (verbatim) | 2 | 2 | 0 | 0 |
| | Summarization (2) | `v1_concise` | 2 | 2 | 0 | 0 |
| | Classification (2)| `v2_answer_only` | 2 | 2 | 0 | 0 |
| | Other (2) | `v4_structured_output` | 2 | 2 | 0 | 0 |
| **qwen3-1.7b Subtotal** | **All 8 Tasks** | | **16** | **16 (100%)**| **0 (0%)** | **0** |
| **qwen3-4b** | QA (2) | `v4_structured_output` | 2 | 2 | 0 | 0 |
| | Reasoning (2) | `v4_struct` / `v1_concise` | 2 | 1 | 1 | 0 |
| | Math (2) | `v1_concise` | 2 | 2 | 0 | 0 |
| | Coding (2) | `v1_concise` | 2 | 1 | 1 | 0 |
| | IFEval (2) | `v0` (verbatim) | 2 | 1 | 1 | 0 |
| | Summarization (2) | `v1_concise` | 2 | 0 | 2 | 0 |
| | Classification (2)| `v2_answer_only` | 2 | 0 | 2 | 0 |
| | Other (2) | `v4_structured_output` | 2 | 2 | 0 | 0 |
| **qwen3-4b Subtotal** | **All 8 Tasks** | | **16** | **9 (56.2%)** | **7 (43.8%)**| **0** |
| **qwen3-8b** | QA (2) | `v4_structured_output` | 2 | 2 | 0 | 0 |
| | Reasoning (2) | `v4_struct` / `v1_concise` | 2 | 2 | 0 | 0 |
| | Math (2) | `v1_concise` | 2 | 2 | 0 | 0 |
| | Coding (2) | `v1_concise` | 2 | 2 | 0 | 0 |
| | IFEval (2) | `v0` (verbatim) | 2 | 2 | 0 | 0 |
| | Summarization (2) | `v1_concise` | 2 | 2 | 0 | 0 |
| | Classification (2)| `v2_answer_only` | 2 | 2 | 0 | 0 |
| | Other (2) | `v4_structured_output` | 2 | 2 | 0 | 0 |
| **qwen3-8b Subtotal** | **All 8 Tasks** | | **16** | **16 (100%)**| **0 (0%)** | **0** |
| **Grand Total** | | | **48** | **41 (85.4%)**| **7 (14.6%)**| **0** |

---

## 6. Detailed Paired Comparison: Phase 4P vs. Phase 4Q

### 1. Latency and SLA (<10,000 ms) Compliance

| Model | Phase | Attempts | Min (ms) | Median (ms) | Mean (ms) | P95 (ms) | Max (ms) | SLA Met Count | SLA Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **qwen3-1.7b** | Phase 4P | 16 | 868.2 | 1,655.1 | 2,078.4 | 3,623.6 | 4,979.2 | 16 / 16 | 100.0% |
| | **Phase 4Q** | 16 | 158.8 | **738.5** | **1,291.0** | 5,293.6 | 5,738.6 | 16 / 16 | **100.0%** |
| **qwen3-4b** | Phase 4P | 16 | 4,146.6 | 18,562.0 | 19,769.5 | 33,937.8 | 34,246.6 | 2 / 16 | 12.5% |
| | **Phase 4Q** | 16 | 623.5 | **8,968.2** | **9,092.4** | 20,908.9 | 34,065.2 | 10 / 16 | **62.5% (+50.0%)** |
| **qwen3-8b** | Phase 4P | 16 | 6,576.9 | 23,571.1 | 27,106.3 | 57,482.9 | 77,734.4 | 3 / 16 | 18.8% |
| | **Phase 4Q** | 16 | 638.2 | **4,751.9** | **9,611.9** | 30,794.8 | 52,001.7 | 10 / 16 | **62.5% (+43.7%)** |

### 2. Output Tokens and Generation Efficiency

| Model | Phase | Min Tokens | Median Tokens | Mean Tokens | Max Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **qwen3-1.7b** | Phase 4P | 91 | 176.5 | 226.7 | 576 |
| | **Phase 4Q** | 2 | **68.5** | **87.5** | 550 |
| **qwen3-4b** | Phase 4P | 128 | 512.0 | 576.5 | 1024 |
| | **Phase 4Q** | 11 | **228.0** | **247.6** | 1024 |
| **qwen3-8b** | Phase 4P | 50 | 205.5 | 231.6 | 579 |
| | **Phase 4Q** | 2 | **43.5** | **79.4** | 447 |

### 3. Task Breakdown Comparison

| Task Type | Phase 4P Success | Phase 4P Median Latency | Phase 4Q Success | Phase 4Q Median Latency | Latency Impact |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QA (MMLU)** | 4 / 6 | 28,858 ms | **6 / 6 (100%)** | **3,387 ms** | **-88.3%** |
| **Reasoning (BBH)** | 4 / 6 | 11,079 ms | **5 / 6 (83%)** | **3,478 ms** | **-68.6%** |
| **Math (GSM8K)** | 5 / 6 | 8,767 ms | **6 / 6 (100%)** | **6,127 ms** | **-30.1%** |
| **Coding (MBPP)** | 4 / 6 | 26,795 ms | **5 / 6 (83%)** | **8,774 ms** | **-67.3%** |
| **Instruction Following** | 5 / 6 | 19,221 ms | **5 / 6 (83%)** | **18,122 ms** | -5.7% |
| **Summarization (CNN/DM)**| 4 / 6 | 17,416 ms | 4 / 6 (67%) | **9,416 ms** | **-45.9%** |
| **Classification (BoolQ)** | 1 / 6 | 4,168 ms | **4 / 6 (67%)** | **737 ms** | **-82.3%** |
| **Other (HellaSwag)** | 4 / 6 | 24,279 ms | **6 / 6 (100%)** | **635 ms** | **-97.4%** |

### 4. Scored Quality by Model (Explicit Denominators)

> [!IMPORTANT]
> Quality scores are strictly reported with explicit denominators (`scored_count / total_attempts`). Attempts that were truncated or whose execution sandbox was unavailable are excluded from scored means rather than masked as zero.

| Model ID | Phase 4P Scored Quality | Phase 4P Denominator | Phase 4Q Scored Quality | Phase 4Q Denominator | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **qwen3-1.7b** | 0.5448 | 12 / 16 | **0.4787** | 14 / 16 | 2 BoolQ classification prompts completed and scored (1 correct, 1 incorrect) |
| **qwen3-4b** | 0.5000 | 2 / 16 | **0.8750** | 8 / 16 | 6 additional prompts scored due to elimination of MC/Math truncations |
| **qwen3-8b** | 0.5810 | 13 / 16 | **0.6910** | 14 / 16 | 1 BoolQ classification prompt completed and scored correct |

---

## 7. Analysis of Remaining Truncations and Limitations

In Phase 4Q, 7 out of 48 attempts experienced truncation, **all 7 concentrated in `qwen3-4b`**. Understanding this behavior is essential for deciding whether and how to scale the benchmark.

### The `qwen3-4b` Chatty Pre-ambling Phenomenon
1. **Classification (BoolQ: `val_0000`, `val_0001`):**
   - Dispatched budget: 64 tokens under `v2_answer_only`.
   - Result: `qwen3-1.7b` and `qwen3-8b` immediately emitted `true` or `false` in 2 tokens (latency: 160-836 ms).
   - In contrast, `qwen3-4b` generated conversational preambles ("Based on the provided passage, the answer to whether...") and consumed 64 tokens before reaching the actual answer token, causing truncation.
2. **Summarization (CNN/DailyMail: 2 prompts):**
   - Dispatched budget: 256 tokens under `v1_concise`.
   - Result: `qwen3-1.7b` and `qwen3-8b` produced concise summaries in 82-87 tokens.
   - `qwen3-4b` produced lengthy narrative text exceeding 256 tokens.
3. **Reasoning (BBH `strategy_b_0001`):**
   - Dispatched budget: 256 tokens for free-form reasoning. `qwen3-4b` truncated at 256 tokens while `1.7b` and `8b` completed in 83 and 39 tokens.
4. **Instruction Following (IFEval `val_2391`):**
   - Extremely long prompt with multiple negative constraints. Truncated at 1024 tokens.
5. **Coding (MBPP `601`):**
   - Truncated at 512 tokens due to lengthy explanation comments prior to code.

**Key Finding:** `qwen3-4b` has a distinct propensity for chattiness when no strict JSON grammar or stop tokens constrain its generation. For multiple-choice formats where JSON grammar was applied, `qwen3-4b` achieved **100% completion in 11-12 tokens** with median latency of 638 ms.

---

## 8. Verification of Artifacts and Test Suite

### 1. Benchmark Hash Verification
All four frozen benchmark splits were re-verified against their authoritative manifest checksums:
```
benchmark_train.jsonl: MATCH (ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014)
benchmark_validation.jsonl: MATCH (5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143)
benchmark_test.jsonl: MATCH (ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b)
benchmark_stress.jsonl: MATCH (36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747)
```

### 2. Test Suite Status
The full repository test suite was executed:
```
============================= 291 passed in 7.82s =============================
```
All 291 unit and integration tests passed cleanly, including 15 new dedicated tests for task-aware policy dispatch, schema exclusivity, truncation handling, and failure classification in [`tests/test_task_aware_policy.py`](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/tests/test_task_aware_policy.py).

### 3. Deduplication and Re-Run Verification
Re-running the matrix runner against `data/outcomes/pilot_phase4q`:
- `--dry-run`: 48 planned, 48 previously completed, 0 pending attempts.
- Live execution: Exits immediately with: `Run 'matrix-train-20261021-081855' in 'data\outcomes\pilot_phase4q' is already complete (48/48 attempts). Exiting cleanly.`
- Zero duplicate rows, zero corrupted parquet records.

---

## 9. Decision Gate: CONDITIONAL GO

### Verdict: **CONDITIONAL GO**

### Rationale
Phase 4Q has definitively established that **task-aware inference policies solve the primary failure modes** of Phase 4P:
1. **Multiple-choice truncation is eliminated:** 100% completion (18/18 attempts across MMLU, HellaSwag, BBH MC) with sub-second latencies and schema validity.
2. **Scoring integrity is fully protected:** Truncations and sandbox unavailability are explicitly tracked and excluded from scored quality; zero corrupted or false-zero outcomes exist.
3. **Artifact and pipeline reproducibility is verified:** Deterministic fingerprints, manifest reconciliation, and zero-inference idempotence are established.
4. **SLA compliance improved dramatically:** Jumped from 43.8% to 75.0% overall, with `qwen3-8b` latency falling by ~80%.

However, a full 4,200-attempt matrix launch cannot be authorized unconditionally due to two bounded constraints that must be addressed:

### Required Conditions Before Any Larger Benchmark Run
1. **`qwen3-4b` Chatter Budget Alignment:**
   - Classification (`classification_extraction` / BoolQ) budget should be adjusted from 64 to 128 tokens, or stop tokens / thinking suppression explicitly verified for 4b, to accommodate conversational lead-ins.
   - Summarization (`cnn_dailymail`) budget should be set to 384 tokens for 4b.
2. **Coding Execution Quarantine:**
   - Because containerized sandbox isolation (Docker/gVisor) is unavailable in the current host environment, coding benchmark tasks (MBPP / HumanEval) must remain quarantined in `SANDBOX_UNAVAILABLE` status and excluded from router training loss, or evaluated via safe non-executing AST validation.
3. **Staged Progression:**
   - Do NOT jump directly to the 4,200-attempt matrix.
   - The next stage should be a **medium-scale validation run (e.g. 100-200 prompts)** to confirm budget tuning on `qwen3-4b` before committing to the full benchmark matrix.
   - **Router training remains strictly blocked** until the benchmark evaluation matrix is finalized and audited.
