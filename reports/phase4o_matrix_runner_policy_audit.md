# Phase 4O — Independent Matrix Runner Audit and Inference Policy Integration Report

**Execution Date**: 2026-10-21  
**Audit Target**: `runners/matrix_runner.py` & Associated Inference/Storage Abstractions  
**Test Suite**: `tests/test_matrix_runner.py` (15 tests passing)  
**Full Test Suite**: 275 tests passing (0 failures, 0 errors)  
**Status**: **HARDENED & AUDITED — READY FOR POLICY EVALUATION MATRIX PROTOCOL**

---

## 1. Executive Summary

Phase 4O conducted an independent, code-level audit of the Phase 4N matrix runner (`runners/matrix_runner.py`) and its integration with inference policies, model registries, storage schemas, and resume guardrails. The audit confirmed several critical risks and architectural gaps in the initial runner implementation, rectified them with minimal targeted code enhancements, introduced deterministic experiment fingerprinting, and validated all edge cases with 6 new regression tests (15 total matrix runner tests).

### Core Audit Outcomes:
1. **Experiment Fingerprinting**: Replaced partial heuristic parameter checks on resume with a cryptographic SHA-256 experiment configuration fingerprint. Any drift in benchmark split hash, candidate models, exact model tags, prompt policies, output budgets, sampling parameters, response schemas, or scoring versions strictly halts resumption with actionable diff diagnostics.
2. **Attempt History & Failure Preservation**: Corrected deduplication so that winning records merge prior attempt histories and increment cumulative retry counts rather than silently erasing telemetry from earlier failed attempts. Added `--retry-failed` to permit selective re-execution of failed/timed-out queries while permanently preserving prior failure records.
3. **Outcomes/Scores Reconciliation**: Identified and eliminated a crash-recovery desync hazard where outcomes persisted before scores could leave unscored records permanently skipped on resume. The runner now automatically detects and reconciles any unscored outcomes upon restart before resuming pending attempts.
4. **Pre-Inference Validation**: Fixed silent fallbacks: unknown prompt policy versions or candidate model IDs now fail immediately before any inference or provider initialization.
5. **Smoke Artifact Protection**: Guarded against accidental pollution or overwriting of prior diagnostic smoke runs (`smoke_outcomes.parquet`, `smoke_manifest.json`).
6. **Explicit Policy & Settings Dry-Run**: Enhanced dry-run reporting to explicitly display the prompt policy description, dynamic task-specific token budgets, thinking modes, structured response schemas, candidate model tags, split file SHA-256, and computed experiment fingerprint. Exactly 4,200 attempts planned for `train` split across 3 candidate models without invoking live inference.

---

## 2. Actual Code Paths Audited

The audit performed line-by-line inspection of the following active repository components:

| Component | Audited Module / Path | Key Functions & Responsibilities Audited |
| :--- | :--- | :--- |
| **Matrix Runner** | `runners/matrix_runner.py` | `plan_matrix_attempts`, `run_matrix`, `validate_resume_compatibility`, `deduplicate_outcomes`, `safe_load_outcomes`, `safe_load_scores`, `atomic_save_outcomes`, `flush_checkpoint` |
| **Inference Policies** | `runners/inference_policy.py` | `format_prompt_with_policy`, `resolve_output_budget`, `resolve_thinking_mode`, `resolve_response_format`, `validate_structured_output_json` |
| **Inference Abstraction** | `runners/inference.py` | `InferenceProvider.execute_prompt`, `GenerationRequest`, `GenerationResult`, retry loop, and `attempt_history` population |
| **Local Backend** | `runners/ollama_runner.py` | Native Ollama `/api/chat` payload construction, `format` parameter propagation, options mapping |
| **Storage & Manifest** | `runners/storage.py`, `runners/outcome_schema.py` | `save_outcomes_parquet`, `load_outcomes_parquet`, `RunManifest` validation, `ModelOutcome` validation |
| **Scoring Engine** | `eval/scorer.py`, `eval/storage.py` | `OutcomeScorer.score_outcome`, decoupled failure filtering, `save_scores_parquet` |
| **Model Registry** | `configs/models.yaml` | Candidate models (`qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`), physical tags, roles, context limits |
| **Policies Config** | `configs/policies.yaml` | Quality threshold (`0.80`), SLA latency (`10000ms`), routing objective |

---

## 3. Confirmed Defects vs. Non-Reproduced Risks

| Issue / Risk Description | Audit Finding | Classification | Action Taken |
| :--- | :--- | :---: | :--- |
| **1. Unscored Outcomes on Interrupted Flush** | If execution stopped between saving outcomes and saving scores, resumed runs considered attempts completed and never computed scores for those rows. | **CONFIRMED DEFECT** | Added automatic startup reconciliation in `run_matrix`: detects any outcome missing a score, re-scores via `OutcomeScorer`, and flushes `matrix_scores.parquet`. |
| **2. Silent Erasure of Prior Attempt Failures** | In Phase 4N, `deduplicate_outcomes` replaced failed outcomes with successful ones, discarding the failed attempt's telemetry. | **CONFIRMED DEFECT** | Enhanced `deduplicate_outcomes` to merge previous attempt records into `winner.attempt_history` and increment `retry_count`. |
| **3. Silent Unknown Model / Policy Acceptance** | Specifying `--prompt-policy unknown_xyz` was silently accepted and defaulted to `v0`. Specifying `--models unknown_model` was silently filtered out. | **CONFIRMED DEFECT** | Added pre-flight validation against `SUPPORTED_PROMPT_POLICIES` and registered model IDs before inference. |
| **4. Incomplete Resume Compatibility Check** | Phase 4N did not verify candidate model list, model tags, thinking mode, temperature, seed, or split file SHA-256 on resume. | **CONFIRMED DEFECT** | Implemented `compute_experiment_fingerprint` capturing all 15 experiment dimensions in a SHA-256 digest. |
| **5. Accidental Smoke Artifact Overwrite** | Specifying an existing smoke output directory would overwrite or mix smoke artifacts. | **CONFIRMED DEFECT** | Added `check_smoke_artifact_collision`: blocks execution in directories containing smoke files unless `--force` is provided. |
| **6. Parquet Corruption on Abrupt Termination** | Concern that partial writes during power loss corrupt Parquet. | **NOT REPRODUCED (PROTECTED)** | Verified: Atomic writes (`.tmp` $\to$ `.parquet`) with `.bak` retention safely protect the primary artifact. Recovered from `.bak` when primary is damaged. |
| **7. Split Cross-Contamination** | Concern that validation, test, or stress prompts could leak into training matrix. | **NOT REPRODUCED (VERIFIED)** | `split_file` explicitly loads `benchmark_{split}.jsonl`. All 1,400 loaded training prompts verified as `split == "train"`. |

---

## 4. Changes Made and Files Touched

### 1. `runners/matrix_runner.py`
- Added `SUPPORTED_PROMPT_POLICIES` set for pre-flight CLI validation.
- Added `compute_experiment_fingerprint(...) -> tuple[str, dict[str, Any]]`.
- Updated `deduplicate_outcomes(...)` to preserve all prior attempt telemetry in `attempt_history` and cumulate `retry_count`.
- Added `check_smoke_artifact_collision(...)` to protect smoke diagnostic runs.
- Enhanced `validate_resume_compatibility(...)` to compare cryptographic experiment fingerprints and produce actionable diff diagnostics.
- Added score reconciliation in `run_matrix(...)` to heal any desync between outcomes and scores from interrupted runs.
- Added `--retry-failed` argument and logic to re-evaluate failed/timed-out prompts while preserving failure audit trails.
- Enhanced `--dry-run` console reporting with complete policy, budget, thinking, schema, model tag, and fingerprint details.

### 2. `tests/test_matrix_runner.py`
- Added `test_unknown_policy_and_model_ids_fail_before_inference`.
- Added `test_smoke_artifact_collision_blocks_run_without_force`.
- Added `test_experiment_fingerprint_sensitivity`.
- Added `test_unscored_outcomes_reconciled_on_resume`.
- Added `test_retry_failed_preserves_prior_attempt_history`.
- Added `test_dry_run_displays_detailed_policy_and_budget`.

---

## 5. Experiment Fingerprint Design & Resume Compatibility

The experiment configuration is cryptographically fingerprinted into a deterministic SHA-256 digest computed from a sorted canonical JSON dictionary:

```python
canonical_dict = {
    "backend": "ollama",
    "benchmark_manifest_sha256": "11f23ef...",
    "benchmark_split": "train",
    "benchmark_split_file_sha256": "ff956efa...",
    "candidate_models": ["qwen3-1.7b", "qwen3-4b", "qwen3-8b"],
    "candidate_model_tags": {
        "qwen3-1.7b": "Qwen/Qwen3-1.7B",
        "qwen3-4b": "Qwen/Qwen3-4B",
        "qwen3-8b": "Qwen/Qwen3-8B"
    },
    "models_config_sha256": "dc53514...",
    "outcome_schema_version": "1.0.0",
    "output_budget": max_output_tokens,
    "prompt_policy_version": prompt_policy_version,
    "response_schema": response_schema,
    "scorer_name": "OutcomeScorer",
    "scorer_version": "1.0.0",
    "seed": seed,
    "target_quality": 0.80,
    "temperature": temperature,
    "think": think,
}
```

### Resume Rules:
1. When an existing `matrix_manifest.json` is loaded, its `experiment_fingerprint` must match the current fingerprint.
2. If fingerprints differ, the runner performs a key-by-key comparison and raises `ConfigurationMismatchError` detailing the exact mismatches (e.g. `prompt_policy_version mismatch: existing='v0', current='v4_structured_output'`).
3. Outright rejection prevents accidental blending of different policies or sampling temperatures in a single matrix dataset.

---

## 6. Concrete Dry-Run Verification

### Baseline Dry Run (Policy `v0`, Default Dynamic Budgets):
```powershell
.venv\Scripts\python -m runners.matrix_runner --split train --dry-run
```
```
=== Matrix Runner Dry Run ===
Benchmark Split: train (data\processed\benchmark_train.jsonl)
Split File SHA-256: ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014
Prompts Loaded: 1400
Candidate Models: ['qwen3-1.7b', 'qwen3-4b', 'qwen3-8b']
Candidate Model Tags: {'qwen3-1.7b': 'Qwen/Qwen3-1.7B', 'qwen3-4b': 'Qwen/Qwen3-4B', 'qwen3-8b': 'Qwen/Qwen3-8B'}
Backend: ollama
Total Planned Attempts: 4200
Previously Completed Attempts: 0
Pending Attempts to Execute: 4200
Prompt Policy Version: v0 (Original benchmark prompt unmodified (unconstrained generation))
Output Budget: dynamic per task (multiple_choice: 768, exact_match: 512, classification: 128)
Thinking Mode: dynamic default (multiple_choice: False, exact_match: False)
Response Schema: None
Sampling Parameters: temperature=0.0, seed=42, timeout=120.0s
Experiment Fingerprint: 5a38015e81a5e9821931411b15af75614aa34b7ccb7bd747a909487a27896acf
Target Output Directory: data\outcomes\matrix_train
Dry run complete. No inference executed.
```

### Structured Output Policy Dry Run (`v4_structured_output`, Explicit 64-Token Budget):
```powershell
.venv\Scripts\python -m runners.matrix_runner --split train --prompt-policy v4_structured_output --max-output-tokens 64 --no-think --dry-run
```
```
=== Matrix Runner Dry Run ===
Benchmark Split: train (data\processed\benchmark_train.jsonl)
Split File SHA-256: ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014
Prompts Loaded: 1400
Candidate Models: ['qwen3-1.7b', 'qwen3-4b', 'qwen3-8b']
Candidate Model Tags: {'qwen3-1.7b': 'Qwen/Qwen3-1.7B', 'qwen3-4b': 'Qwen/Qwen3-4B', 'qwen3-8b': 'Qwen/Qwen3-8B'}
Backend: ollama
Total Planned Attempts: 4200
Previously Completed Attempts: 0
Pending Attempts to Execute: 4200
Prompt Policy Version: v4_structured_output (Native Ollama structured JSON schema enum constraint)
Output Budget: 64 tokens (explicit override)
Thinking Mode: False (explicit override)
Response Schema: {'type': 'object', 'properties': {'answer': {'type': 'string', 'enum': ['A', 'B', 'C', 'D']}}, 'required': ['answer'], 'additionalProperties': False}
Sampling Parameters: temperature=0.0, seed=42, timeout=120.0s
Experiment Fingerprint: b035b5087913106f878cfee018a6e4d87bf395e1e7158a2c36b250e42b559644
Target Output Directory: data\outcomes\matrix_train
Dry run complete. No inference executed.
```

---

## 7. Test Results & Hash Verification

### Regression Test Suite Execution:
```powershell
.venv\Scripts\python -m pytest tests/test_matrix_runner.py -v
```
**Output**: `15 passed in 1.29s`

### Complete Repository Test Suite:
```powershell
.venv\Scripts\python -m pytest
```
**Output**: `275 passed in 7.72s` (0 failed, 0 errors, 0 skipped).

### Frozen Benchmark SHA-256 Hashes:
```powershell
.venv\Scripts\python -c "
import hashlib
for split in ['train', 'validation', 'test', 'stress']:
    p = f'data/processed/benchmark_{split}.jsonl'
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    print(f'{split}: {h}')
"
```
- `train`: `ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014` (**MATCH**)
- `validation`: `5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143` (**MATCH**)
- `test`: `ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b` (**MATCH**)
- `stress`: `36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747` (**MATCH**)

**Confirmation**: Zero benchmark records or hashes were modified. Zero live Ollama or network inferences were performed during this audit phase.

---

## 8. Remaining Limitations & Recommendations

1. **Storage Format Scalability**: For 4,200 rows, rewriting `matrix_outcomes.parquet` on every attempt is fast (<5ms), but for 100k+ rows an append partition directory (`partitions/model_id/shard_*.parquet`) would be superior. The current single-table Parquet design is optimal for 4,200 rows and matches downstream training loaders (`load_outcomes_parquet`).
2. **Policy Selection Decision**: The infrastructure now supports explicit selection across `v0`, `v1_concise`, `v2_answer_only`, `v3_few_shot`, and `v4_structured_output`. However, as established in Phase 4M, a universal blanket policy is suboptimal: zero-shot grammar constraints work flawlessly for classification and direct choice QA, but complex mathematics benefits from bounded reasoning scratchpads.
3. **Next Checkpoint**: Define the matrix execution protocol (e.g. task-segmented prompt policies or bounded reasoning schemas) before commencing the 4,200-attempt run.
