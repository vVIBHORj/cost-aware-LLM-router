# Phase 4N — Durable, Resumable Matrix Runner Reliability Report

**Execution Date**: 2026-10-21  
**Target Module**: `runners/matrix_runner.py`  
**Test Suite**: `tests/test_matrix_runner.py`  
**Decision Entering Phase**: **NO-GO for Matrix Execution**  
**Decision Exiting Phase**: **RUNNER RELIABLE — MATRIX EXECUTION CONDITIONAL ON TASK-POLICY RESOLUTION**

---

## 1. Executive Summary

Phase 4N implemented and rigorously validated `runners/matrix_runner.py` to resolve the architectural and data-safety gaps identified in the Phase 4J audit. Prior to this phase, candidate inference was executed via an in-memory accumulator (`smoke_runner.py`) that would lose all completed work upon interruption, overwrite prior directories unguardedly, and lack automated resume or canonical deduplication.

`runners/matrix_runner.py` establishes:
1. **Incremental, Crash-Safe Persistence**: Atomically flushes `matrix_outcomes.parquet`, `matrix_scores.parquet`, and `matrix_manifest.json` after every configurable interval (`--flush-interval`, default 1).
2. **Canonical Deduplication**: Deduplicates records using `(prompt_id, model_id)` deterministically (preferring `SUCCESS` over failure, and later timestamps for ties).
3. **Safe Resumption & Interruption Recovery**: Skips already completed attempts upon resumption; safely recovers from `.bak` or `.tmp` files if primary artifacts are corrupted.
4. **Configuration Lock & Overwrite Protection**: Strictly verifies benchmark manifest checksums, prompt policy versions, token budgets, splits, and backends against existing run manifests before allowing resumption.
5. **Score Decoupling & Failure Preservation**: Failed or truncated inferences are recorded as failures and evaluated with `scoring_status = SCORING_UNAVAILABLE`, `quality_score = NaN`, and `success_at_target = None`, strictly preventing unavailable scores from masquerading as `0.0`.
6. **Dry-Run Mode**: Verified against the frozen training split with all 3 candidate models: exactly **1,400 prompts × 3 models = 4,200 planned attempts** without invoking inference.

All 9 new test cases pass in 1.15s; the full test suite (269 tests) passes in 5.66s; and all four frozen benchmark SHA-256 hashes remain perfectly identical to established ground-truth values.

---

## 2. Architecture and Persistence Design

```
+-----------------------------------------------------------------------------------------+
|                                    run_matrix()                                         |
+-----------------------------------------------------------------------------------------+
        |                                                              |
        v                                                              v
 [1. Load Benchmark Split]                                     [2. Load Candidate Models]
  - benchmark_{split}.jsonl                                     - configs/models.yaml
  - SHA-256 Verification                                        - Enabled Models Filter
        |                                                              |
        +------------------------------+-------------------------------+
                                       |
                                       v
                        [3. Plan Canonical Attempts]
                         - Group by Candidate Model (minimize VRAM thrashing)
                         - Order prompts deterministically
                         - Total Planned = Prompts x Models
                                       |
                                       v
                        [4. Inspect Existing Output Dir]
                         - Check matrix_manifest.json (validate settings compatibility)
                         - Safe load matrix_outcomes.parquet (recover from .bak if corrupted)
                         - Deduplicate existing records by (prompt_id, model_id)
                         - Identify completed keys vs pending attempts
                                       |
                 +---------------------+---------------------+
                 | (If Dry-Run)                              | (If Pending > 0)
                 v                                           v
         [Print Planned Summary]                    [5. Execute Sequential Loop]
         - Exit cleanly (Code 0)                     - For attempt in pending_attempts:
                                                       * provider.execute_prompt()
                                                       * scorer.score_outcome()
                                                       * Telemetry & ETA logging
                                                       * If step % flush_interval == 0:
                                                           -> Atomic Parquet/JSON Flush
                                       |
                                       v
                        [6. Atomic Checkpoint Flush]
                         - Write matrix_outcomes.parquet.tmp
                         - Copy existing to matrix_outcomes.parquet.bak
                         - os.replace(tmp, target)
                         - Repeat for scores and manifest
```

### Key Architectural Elements:
1. **Model Grouping**: In local inference on edge hardware (NVIDIA RTX 3050 Laptop GPU 4 GB VRAM / 16 GB DDR5 RAM), reloading 1.7B, 4B, and 8B weights per prompt induces catastrophic thrashing. `plan_matrix_attempts` partitions attempts by candidate model first, executing all 1,400 prompts for Model 1 before switching to Model 2.
2. **Atomic Writes via `os.replace`**: Writes target Parquet tables to `.tmp` files on the local volume, backs up existing targets to `.bak`, and executes atomic directory entry replacement.
3. **Graceful Signal Handling**: Traps `KeyboardInterrupt` (SIGINT), immediately executes an atomic checkpoint flush of all in-memory outcomes and scores, prints recovery instructions, and exits with code 130.

---

## 3. Resume and Deduplication Semantics

### Canonical Attempt Identity
The canonical identity of an attempt is the 2-tuple:
$$\text{Attempt Key} = (\text{prompt\_id}, \text{model\_id})$$

When loading from disk or updating in-memory state:
- An attempt is **completed** if `(prompt_id, model_id)` exists in the loaded outcomes table.
- When resuming, the runner filters:
  $$\text{Pending Attempts} = \{a \in \text{Planned Attempts} \mid a.\text{key} \notin \text{Completed Keys}\}$$
- The provider is never invoked for an attempt whose canonical key already exists in the outcome table.

### Deterministic Deduplication Rule
If duplicate keys exist (e.g., from merged tables or partial writes):
1. **Status Precedence**: `SUCCESS` strictly takes precedence over non-`SUCCESS` (`FAILED`, `TRUNCATED`, `TIMEOUT`, `BACKEND_UNAVAILABLE`).
2. **Timestamp Precedence**: If both records have the same status class, the record with the strictly later `request_timestamp` is retained.
3. Both outcomes and scores are deduplicated prior to writing to Parquet, satisfying PyArrow uniqueness constraints.

---

## 4. Configuration Guardrails and Resume Safety

When an existing `matrix_manifest.json` is present in the output directory, `validate_resume_compatibility` verifies:
- `benchmark_manifest_checksum == expected_benchmark_checksum`
- `prompt_policy_version == current_prompt_policy_version`
- `output_budget == current_output_budget`
- `effective_inference_settings.backend == current_backend`
- `effective_inference_settings.benchmark_split == current_split`
- `manifest.run_id == current_run_id` (if custom `run_id` specified)

If any parameter conflicts, `ConfigurationMismatchError` is raised, halting execution before any files are modified or inference is called. Overwrite is blocked unless `--force` is explicitly provided.

---

## 5. Automated Test Cases and Empirical Results

The test suite in `tests/test_matrix_runner.py` implements all 8 mandatory verification cases specified in Phase 4N, plus dry-run validation:

| Test Case | Description | Tested Mechanism | Result |
| :--- | :--- | :--- | :---: |
| `test_case_1_fresh_run_creates_expected_manifest` | 3 prompts × 2 models fresh execution | Manifest generation, expected/actual row validation, Parquet creation | **PASS** |
| `test_case_2_completed_attempts_are_skipped_when_resuming` | Interrupted at 3/6 attempts, resumed | Verify mock provider only called 3 times on resume; exactly 6 final rows | **PASS** |
| `test_case_3_duplicate_outcome_rows_detected_and_handled_deterministically` | Duplicate keys with conflicting status and timestamps | Resolution: SUCCESS over FAILED, later timestamp over earlier | **PASS** |
| `test_case_4_interrupted_run_resumes_without_losing_persisted_outcomes` | Crash on prompt 3, persistence check, resume to completion | No loss of pre-crash attempts; exactly 6 unique rows in final table | **PASS** |
| `test_case_5_failed_inference_remains_recorded_and_not_scored_zero` | Injected FAILED and TRUNCATED inference outcomes | Verified: `scoring_status=SCORING_UNAVAILABLE`, `quality_score=NaN` (not 0.0), `success_at_target=None` | **PASS** |
| `test_case_6_partially_written_artifact_recovered_or_rejected` | Corrupted Parquet header with and without backup | Sub-case A: Recovered from `.bak`<br>Sub-case B: Raised `CorruptedArtifactError` | **PASS** |
| `test_case_7_configuration_mismatch_prevents_unsafe_resume` | Attempting resume with altered policy, budget, or backend | `ConfigurationMismatchError` raised on policy, budget, and backend drift | **PASS** |
| `test_case_8_repeated_execution_does_not_silently_overwrite_incompatible_run` | Repeated run on completed directory; conflicting run ID | Blocks conflicting run IDs; returns cleanly without inference if 100% complete | **PASS** |
| `test_dry_run_mode` | Dry run with spy provider | Zero inference calls; zero files written; exact planned attempt counts | **PASS** |

### Execution Command:
```powershell
.venv\Scripts\python -m pytest tests/test_matrix_runner.py -v
```
**Output**: `9 passed in 1.15s`

---

## 6. Training Split Dry-Run Validation

The dry-run command was executed against the production training dataset:
```powershell
.venv\Scripts\python -m runners.matrix_runner --split train --dry-run
```

### Dry-Run Telemetry:
```
=== Matrix Runner Dry Run ===
Benchmark Split: train (data\processed\benchmark_train.jsonl)
Prompts Loaded: 1400
Candidate Models: ['qwen3-1.7b', 'qwen3-4b', 'qwen3-8b']
Total Planned Attempts: 4200
Previously Completed Attempts: 0
Pending Attempts to Execute: 4200
Prompt Policy Version: v0
Output Budget: None tokens
Target Output Directory: data\outcomes\matrix_train
Dry run complete. No inference executed.
```

- **Prompts**: Exactly 1,400 prompts loaded from `benchmark_train.jsonl`.
- **Models**: Exactly 3 candidate models enabled (`qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`).
- **Matrix Dimensions**: Exactly **4,200 planned attempts**.
- **Backend Calls**: Zero inference requests sent to Ollama or local backends.

---

## 7. Integrity & Hash Verification

All four frozen benchmark splits were re-verified via SHA-256 computation:

| Benchmark Split | Expected Established SHA-256 | Observed SHA-256 | Status |
| :--- | :--- | :--- | :---: |
| `benchmark_train.jsonl` | `ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014` | `ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014` | **MATCH** |
| `benchmark_validation.jsonl` | `5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143` | `5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143` | **MATCH** |
| `benchmark_test.jsonl` | `ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b` | `ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b` | **MATCH** |
| `benchmark_stress.jsonl` | `36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747` | `36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747` | **MATCH** |

Full Pytest suite status:
```powershell
.venv\Scripts\python -m pytest
```
**Output**: `269 passed in 5.66s` (0 failures, 0 errors, 0 warnings).

---

## 8. Known Limitations & Unresolved Risks

While `runners/matrix_runner.py` is now fully hardened for crash recovery and atomic persistence, two fundamental operational blockers remain prior to executing the 4,200 training matrix:

1. **Policy-Task Divergence (Discovered in Phases 4K–4M)**:
   - For classification, boolq, and direct multiple-choice: `v4_structured_output` achieves 100% completion, zero truncation, and sub-second latency across all 3 models.
   - For complex mathematical reasoning (MMLU abstract algebra, GSM8K, Hendrycks MATH): Models fundamentally require thinking scratchpads. Enforcing direct zero-shot structured outputs without reasoning dropped 8B accuracy from 100% to 40%.
   - Running the entire 4,200 training matrix with a single blanket policy (`v0` or `v4_structured_output`) would either cause massive token limit truncation on 4B (under `v0`/`v1`) or severe reasoning impairment on math tasks (under zero-shot `v4`).
2. **`qwen3:8b` SLA Frontier**:
   - `qwen3:8b` exceeds 4 GB VRAM and spills into DDR5 system RAM, executing at ~9 tokens/second. When generating reasoning scratchpads, its latency is 21s–45s, violating the strict 10s SLA.
   - The matrix must decide whether `qwen3:8b` serves as an offline teacher/oracle (where latency SLA does not apply) or if tiered latency SLAs apply.

---

## 9. Next Checkpoint Recommendation

### Recommendation: **RUNNER INFRASTRUCTURE READY; PROCEED TO PHASE 4O (TASK-AWARE POLICY OR BOUNDED REASONING SCHEMA)**

The runner infrastructure is complete, reliable, and verified. Before launching 4,200 live inferences across ~21–34 compute hours, the project should define the exact policy configuration for the matrix:
- **Option A (Task-Aware Policy Mapping)**: Use `v4_structured_output` for multiple-choice/classification tasks (where it eliminates 4B truncation and guarantees sub-second latency) and bounded thinking (`<think>...<\think>`) or structured reasoning schemas `{"thought": str, "answer": str}` for generative math/code tasks.
- **Option B (Reasoning-in-Schema)**: Test a structured output schema that explicitly permits a bounded reasoning field before the answer choice:
  ```json
  {"steps": "string (concise reasoning)", "answer": "A | B | C | D"}
  ```

---

## 10. Artifact Summary

| File Path | Description |
| :--- | :--- |
| `runners/matrix_runner.py` | Production matrix runner with atomic persistence, resume, deduplication, and progress reporting. |
| `tests/test_matrix_runner.py` | 9 automated unit/integration tests verifying all 8 failure/resume modes. |
| `reports/phase4n_matrix_runner_reliability.md` | Comprehensive architectural, validation, and reliability audit report. |
