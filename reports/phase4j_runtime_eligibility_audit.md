# Phase 4J: Runtime Eligibility Audit and Matrix Go/No-Go Decision

**Date**: 2026-10-20  
**Hardware Profile**: Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB dedicated VRAM), 16 GB DDR5 System RAM  
**Inference Backend**: Local Ollama v0.40.3 (`http://127.0.0.1:11434`)  
**Investigator**: Antigravity  

---

## 1. Executive Verdict: NOT READY FOR MATRIX LAUNCH

### **VERDICT: NOT READY (NO-GO)**

The candidate evaluation matrix across the 1,400 training split prompts (4,200 total inference attempts) **MUST NOT BE LAUNCHED** at this time. 

The audit reveals three insurmountable operational and scientific blockers:
1. **Candidate Infeasibility (`qwen3:4b`)**: Under the tested 512-token policy, `qwen3:4b` experiences a **100% truncation rate** (0/5 scorable). Launching a 1,400-prompt matrix would burn ~6.5 hours of continuous compute on `qwen3:4b` to produce 1,400 truncated, unscorable rows (`quality_score = NaN`), providing zero usable training signal for routing.
2. **SLA Non-Compliance (`qwen3:8b`)**: Due to hardware offloading into system DDR5 RAM, `qwen3:8b` generates tokens at ~9 tok/s, resulting in observed latencies between 26.3s and 48.2s. It has **0% compliance** with the established strict `< 10,000 ms` SLA. While its quality is 100% (5/5), it cannot be routed to under current runtime constraints.
3. **Infrastructure Deficits (No Checkpointing/Resume)**: There is currently no dedicated matrix runner. The existing `smoke_runner.py` accumulates all records in memory and writes Parquet files only upon full completion. An unexpected interruption 15 hours into a 21–34 hour run would result in catastrophic, unrecoverable data loss.

---

## 2. Audit of Phase 4I Artifacts & Per-Attempt Audit Table

### A. Provenance and File Deduplication
Inspection of directory contents and SHA-256 hashes:
- `data/outcomes/smoke_phase4i/budget_256/`: Contains Run `smoke-run-20261020-104821` (15 attempts).
- `data/outcomes/smoke_phase4i/budget_512/`: Contains Run `smoke-run-20261020-105211` (15 attempts).
- `data/outcomes/smoke_phase4i/`: Contains `smoke_manifest.json`, `smoke_outcomes.parquet`, and `smoke_scores.parquet`.
  - Checksums for files in root `smoke_phase4i/` are **byte-identical** to `smoke_phase4i/budget_512/` (`smoke_outcomes.parquet` SHA-256 prefix: `cb6a59d719a6`).
  - **Audit Finding**: Root `smoke_phase4i/` is a verbatim file copy of the 512-budget run, **not** an independent run. There are exactly two distinct runs (30 total inference attempts).

### B. Table 1: Initial 256-Token Multiple-Choice Budget
- **Run ID**: `smoke-run-20261020-104821`
- **Applied Prompt Policy**: `v1_concise`

| Run ID | Model ID | Prompt ID | Status | Finish Reason | Latency (ms) | Tokens | Truncated? | Scoring Status | Quality Score | Prompt Policy | Strict SLA (<10s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `smoke-run-20261020-104821` | `qwen3-1.7b` | `mmlu-abstract_algebra-validation-val_0000` | SUCCESS | stop | 7301.10 | 224 | No | SUCCESS | 1.0 | `v1_concise` | **PASS** |
| `smoke-run-20261020-104821` | `qwen3-1.7b` | `mmlu-abstract_algebra-validation-val_0001` | SUCCESS | stop | 1086.56 | 118 | No | SUCCESS | 1.0 | `v1_concise` | **PASS** |
| `smoke-run-20261020-104821` | `qwen3-1.7b` | `mmlu-abstract_algebra-validation-val_0002` | SUCCESS | stop | 1494.71 | 164 | No | SUCCESS | 0.0 | `v1_concise` | **PASS** |
| `smoke-run-20261020-104821` | `qwen3-1.7b` | `mmlu-abstract_algebra-validation-val_0003` | SUCCESS | stop | 1582.87 | 176 | No | SUCCESS | 0.0 | `v1_concise` | **PASS** |
| `smoke-run-20261020-104821` | `qwen3-1.7b` | `mmlu-abstract_algebra-validation-val_0004` | SUCCESS | stop | 1580.71 | 178 | No | SUCCESS | 1.0 | `v1_concise` | **PASS** |
| `smoke-run-20261020-104821` | `qwen3-4b` | `mmlu-abstract_algebra-validation-val_0000` | TRUNCATED | length | 16725.03 | 256 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **FAIL** |
| `smoke-run-20261020-104821` | `qwen3-4b` | `mmlu-abstract_algebra-validation-val_0001` | TRUNCATED | length | 8285.01 | 256 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **PASS** |
| `smoke-run-20261020-104821` | `qwen3-4b` | `mmlu-abstract_algebra-validation-val_0002` | TRUNCATED | length | 8016.39 | 256 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **PASS** |
| `smoke-run-20261020-104821` | `qwen3-4b` | `mmlu-abstract_algebra-validation-val_0003` | TRUNCATED | length | 8098.44 | 256 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **PASS** |
| `smoke-run-20261020-104821` | `qwen3-4b` | `mmlu-abstract_algebra-validation-val_0004` | TRUNCATED | length | 8110.45 | 256 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **PASS** |
| `smoke-run-20261020-104821` | `qwen3-8b` | `mmlu-abstract_algebra-validation-val_0000` | SUCCESS | stop | 35124.33 | 212 | No | SUCCESS | 1.0 | `v1_concise` | **FAIL** |
| `smoke-run-20261020-104821` | `qwen3-8b` | `mmlu-abstract_algebra-validation-val_0001` | SUCCESS | stop | 24094.38 | 222 | No | SUCCESS | 1.0 | `v1_concise` | **FAIL** |
| `smoke-run-20261020-104821` | `qwen3-8b` | `mmlu-abstract_algebra-validation-val_0002` | TRUNCATED | length | 29139.12 | 256 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **FAIL** |
| `smoke-run-20261020-104821` | `qwen3-8b` | `mmlu-abstract_algebra-validation-val_0003` | TRUNCATED | length | 27724.77 | 256 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **FAIL** |
| `smoke-run-20261020-104821` | `qwen3-8b` | `mmlu-abstract_algebra-validation-val_0004` | TRUNCATED | length | 29043.45 | 256 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **FAIL** |

### C. Table 2: Follow-up 512-Token Multiple-Choice Budget
- **Run ID**: `smoke-run-20261020-105211`
- **Applied Prompt Policy**: `v1_concise`

| Run ID | Model ID | Prompt ID | Status | Finish Reason | Latency (ms) | Tokens | Truncated? | Scoring Status | Quality Score | Prompt Policy | Strict SLA (<10s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `smoke-run-20261020-105211` | `qwen3-1.7b` | `mmlu-abstract_algebra-validation-val_0000` | SUCCESS | stop | **10038.51** | 224 | No | SUCCESS | 1.0 | `v1_concise` | **FAIL** |
| `smoke-run-20261020-105211` | `qwen3-1.7b` | `mmlu-abstract_algebra-validation-val_0001` | SUCCESS | stop | 1113.36 | 118 | No | SUCCESS | 1.0 | `v1_concise` | **PASS** |
| `smoke-run-20261020-105211` | `qwen3-1.7b` | `mmlu-abstract_algebra-validation-val_0002` | SUCCESS | stop | 1559.71 | 164 | No | SUCCESS | 0.0 | `v1_concise` | **PASS** |
| `smoke-run-20261020-105211` | `qwen3-1.7b` | `mmlu-abstract_algebra-validation-val_0003` | SUCCESS | stop | 1579.53 | 176 | No | SUCCESS | 0.0 | `v1_concise` | **PASS** |
| `smoke-run-20261020-105211` | `qwen3-1.7b` | `mmlu-abstract_algebra-validation-val_0004` | SUCCESS | stop | 1594.30 | 178 | No | SUCCESS | 1.0 | `v1_concise` | **PASS** |
| `smoke-run-20261020-105211` | `qwen3-4b` | `mmlu-abstract_algebra-validation-val_0000` | TRUNCATED | length | 23990.52 | 512 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **FAIL** |
| `smoke-run-20261020-105211` | `qwen3-4b` | `mmlu-abstract_algebra-validation-val_0001` | TRUNCATED | length | 16490.89 | 512 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **FAIL** |
| `smoke-run-20261020-105211` | `qwen3-4b` | `mmlu-abstract_algebra-validation-val_0002` | TRUNCATED | length | 16537.17 | 512 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **FAIL** |
| `smoke-run-20261020-105211` | `qwen3-4b` | `mmlu-abstract_algebra-validation-val_0003` | TRUNCATED | length | 16843.89 | 512 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **FAIL** |
| `smoke-run-20261020-105211` | `qwen3-4b` | `mmlu-abstract_algebra-validation-val_0004` | TRUNCATED | length | 16693.26 | 512 | Yes | SCORING_UNAVAILABLE | NaN | `v1_concise` | **FAIL** |
| `smoke-run-20261020-105211` | `qwen3-8b` | `mmlu-abstract_algebra-validation-val_0000` | SUCCESS | stop | 34968.03 | 212 | No | SUCCESS | 1.0 | `v1_concise` | **FAIL** |
| `smoke-run-20261020-105211` | `qwen3-8b` | `mmlu-abstract_algebra-validation-val_0001` | SUCCESS | stop | 26333.48 | 222 | No | SUCCESS | 1.0 | `v1_concise` | **FAIL** |
| `smoke-run-20261020-105211` | `qwen3-8b` | `mmlu-abstract_algebra-validation-val_0002` | SUCCESS | stop | 38509.35 | 320 | No | SUCCESS | 1.0 | `v1_concise` | **FAIL** |
| `smoke-run-20261020-105211` | `qwen3-8b` | `mmlu-abstract_algebra-validation-val_0003` | SUCCESS | stop | 48195.31 | 408 | No | SUCCESS | 1.0 | `v1_concise` | **FAIL** |
| `smoke-run-20261020-105211` | `qwen3-8b` | `mmlu-abstract_algebra-validation-val_0004` | SUCCESS | stop | 29829.73 | 259 | No | SUCCESS | 1.0 | `v1_concise` | **FAIL** |

---

## 3. Explanation of the 10.04-Second Discrepancy

In the Phase 4I report (`reports/phase4i_concise_policy_validation.md`), Section 4 stated:
> *"SLA Compliance Rates: `qwen3:1.7b`: **100% compliant** on warm queries (1.1s – 1.6s latency; 1 cold start at 10.04s)."*

This created an apparent contradiction with the strict `< 10,000 ms` SLA summary because the observed data record in `data/outcomes/smoke_phase4i/budget_512/smoke_outcomes.parquet` contains:
```
prompt_id: mmlu-abstract_algebra-validation-val_0000
model_id:  qwen3-1.7b
latency_ms: 10038.51
```

### Forensic Analysis:
1. **Verification of Attempt**: The 10,038.51 ms latency is verified as belonging to Run `smoke-run-20261020-105211`, row index 0. It is an authentic empirical measurement from the first prompt sent to `qwen3:1.7b` in that run.
2. **Strict SLA Compliance**: Under the invariant rule defined in `configs/policies.yaml` (`max_latency_ms: 10000.0`), an attempt meets the SLA if and only if `latency_ms < 10000.0`.
   - `10038.51 < 10000.0` evaluates strictly to **FALSE**.
   - Therefore, `qwen3-1.7b` achieved **4/5 (80.0%) SLA compliance**, NOT 5/5.
3. **Discrepancy Cause**: The Phase 4I report informally excluded the cold start when describing SLA compliance, filtering to warm queries. While warm queries did run in 1.1s–1.6s, substituting warm performance for observed telemetry violates rigorous reporting standards.
4. **Resolution**: The audit formally corrects the compliance record: under observed end-to-end latency, `qwen3:1.7b` is **80% compliant** on the 512-budget smoke test.

---

## 4. Latency Semantics & Telemetry Limitations

### A. What `latency_ms` Actually Represents
Inspection of [runners/inference.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference.py) (lines 203-207) and [runners/ollama_runner.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/ollama_runner.py) (lines 258-281) establishes that `latency_ms`:
- Is calculated as `(time.perf_counter() - t_start) * 1000.0` wrapping the HTTP request to Ollama's endpoint.
- Measures **Client-Observed End-to-End Wall-Clock Time**.
- Includes:
  1. TCP connection handshake.
  2. Model weight swapping / cold-start loading into VRAM / DDR5 RAM if the model was not already resident.
  3. Prompt evaluation time (`prompt_eval_duration`).
  4. Autoregressive token generation time (`eval_duration`).
  5. JSON response transmission and HTTP decoding.

### B. Current Telemetry Limitations
1. **No Cold/Warm Discrimination in Schema**: Neither `GenerationResult` nor `ModelOutcome` records a boolean `is_cold_start` or preserves the backend's breakdown metrics.
2. **Discarded Backend Telemetry**: Native Ollama `/api/chat` outputs `load_duration` (nanoseconds spent loading the model), but `OllamaProvider` currently parses only `prompt_eval_count` and `eval_count`.
3. **Audit Conclusion**: Because the existing artifacts do not record `load_duration`, the system cannot separate cold-start load overhead from generation latency post hoc without making speculative assumptions.

### C. Recommended Defensible Telemetry Policy
To preserve the integrity of the 10,000 ms SLA while providing actionable performance insights:
1. Maintain client-observed end-to-end `latency_ms` as the primary ground truth for SLA compliance.
2. In the telemetry schema, extract Ollama's native `load_duration` into an explicit field: `backend_load_duration_ms`.
3. Record an explicit flag `is_cold_start = (backend_load_duration_ms > 500.0)`.
4. Report both:
   - **Operational SLA Compliance**: Total observed latency against the 10-second ceiling.
   - **Steady-State (Warm) Latency**: Inference latency excluding model loading, strictly for capacity planning and router evaluation.

---

## 5. Candidate Eligibility Policy

Using existing empirical results, the routing framework must cleanly decouple the following five states:

```
                    ┌───────────────────────────┐
                    │     Inference Request     │
                    └─────────────┬─────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
            [status == SUCCESS]         [status != SUCCESS]
                   │                             │
        ┌──────────┴──────────┐                  │
        │                     │                  │
 [Scoring Available]   [Scoring Fails]           │
        │                     │                  │
 ┌──────┴──────┐       [quality_score: 0]        │
 │             │                                 │
[Correct] [Incorrect]                    [quality_score: NaN]
 (score=1)  (score=0)                    (SCORING_UNAVAILABLE)
        │
 ┌──────┴──────────────────────────────┐
 │                                     │
[latency_ms < 10,000]         [latency_ms >= 10,000]
 │                                     │
[SLA Eligible]                [SLA Non-Compliant]
```

### A. Definitions
1. **Inference Success (`status`)**:
   - `SUCCESS`: Complete, uncorrupted generation (`finish_reason: "stop"`).
   - `TRUNCATED`: Premature termination due to token budget exhaustion (`finish_reason: "length"`).
   - `FAILED` / `TIMEOUT`: Backend error, crash, or socket timeout.
2. **Scoring Availability (`scoring_status`)**:
   - Only evaluated when `status == SUCCESS`.
   - If `status == TRUNCATED` or `FAILED`, scoring is skipped, `scoring_status = "SCORING_UNAVAILABLE"`, and `quality_score = NaN` (None).
   - **Rule**: An inference failure or truncation **must never be scored as 0.0**. Assigning 0.0 corrupts accuracy statistics and conflates epistemic model errors with systemic budget starvation.
3. **Task Quality (`quality_score`)**:
   - Discrete `1.0` (correct) or `0.0` (incorrect), applicable exclusively to scorable generations.
4. **SLA Eligibility (`meets_sla`)**:
   - Evaluated as `latency_ms < 10000.0`. An accurate response that misses the SLA **must not** be marked SLA-compliant.
5. **Runtime Candidate Selection Eligibility**:
   - A model candidate is eligible for selection by the router on a prompt if and only if:
     $$\text{Eligible}(m) \iff (\text{status} = \text{SUCCESS}) \land (\text{latency\_ms} < 10000.0)$$

### B. Candidate Status Under Existing Telemetry

#### 1. `qwen3:1.7b` — CONDITIONAL PASS (Borderline Single Candidate)
- **Quality**: 60.0% (3/5 on MMLU abstract algebra).
- **Latency**: 1.1s–1.6s warm. 1 cold start at 10.04s.
- **SLA Compliance**: 80% (4/5 observed at 512 budget).
- **Selection Eligibility**: Eligible on warm requests; borderline on cold start.

#### 2. `qwen3:4b` — INELIGIBLE (100% Truncation Block)
- **Behavior**: Under Ollama v0.40.3, `think: false` suppresses the `<think>` tag from the response envelope, but model weights emit ~450 tokens of raw internal reasoning in plain text closing with `</think>` before generating answers.
- **Impact at 512 Budget**: 100% truncation rate (5/5). 0% scorable.
- **Consequence of Running Matrix**: If the 1,400 training matrix were executed, `qwen3:4b` would produce 1,400 truncated rows, consuming ~6.5 hours of GPU compute while producing **zero** valid quality scores.
- **Constraint Compliance**: The token budget cannot be silently increased to 1,536 tokens, as that would quadruple runtime and still violate the SLA (warm latency at 1,536 tokens is ~50 seconds).

#### 3. `qwen3:8b` — INELIGIBLE (100% SLA Violation)
- **Quality**: 100.0% (5/5 correct).
- **Latency**: 26.3s – 48.2s (mean: 35.7s).
- **SLA Compliance**: **0% (0/5)**.
- **Root Cause**: Offloaded to system DDR5 RAM across the PCIe bus, generating at ~9 tok/s.
- **Selection Eligibility**: Under the invariant 10,000 ms SLA, `qwen3:8b` is 100% ineligible. The 10-second SLA will **not** be relaxed to artificially inflate eligibility.

---

## 6. Matrix Technical Safety, Deduplication, and Resume Audit

An audit of the runner codebase reveals the following technical gaps:

| Safety Dimension | Current Status | Evidence / Audit Findings |
| :--- | :---: | :--- |
| **Split Isolation** | **VERIFIED** | `select_smoke_prompts` accepts explicit `splits=["train"]`. When loading from `data/processed/benchmark_train.jsonl`, all 1,400 prompts have `split == "train"`. No validation, test, or stress prompts are present. |
| **Separate Scores & Outcomes** | **VERIFIED** | `save_outcomes_parquet` and `save_scores_parquet` write to distinct Parquet tables (`smoke_outcomes.parquet` and `smoke_scores.parquet`), ensuring inference telemetry is not mixed with task evaluation. |
| **Canonical Deduplication** | **PARTIAL** | `storage.py` enforces canonical key uniqueness `(prompt_id, model_id, run_id)` within the in-memory batch. However, it does not detect cross-run duplicates or perform deduplicated updates against existing disk tables. |
| **Interruption & Resumption** | **FAILED (CRITICAL)** | `smoke_runner.py` holds all outcomes in an in-memory list `outcomes: list[ModelOutcome] = []`. Outcomes are written to Parquet only after all models and prompts finish. If aborted mid-run, **all executed inferences are lost**. There is no resume mechanism. |
| **Overwrite Protection** | **UNGUARDED** | Calling the existing runner with an existing `output_dir` truncates and overwrites `smoke_outcomes.parquet` and `smoke_manifest.json`. A full matrix run without dedicated directory isolation could overwrite Phase 4G, 4H, or 4I smoke directories. |

---

## 7. Matrix Runtime Estimate and Assumptions

### A. Empirical Timing Inputs
Using only observed wall-clock times from Phase 4H and Phase 4I:
- `qwen3:1.7b`:
  - Observed warm latency: ~1.46s per prompt.
  - Estimated 1,400 prompts: $1 \times 10\text{s (cold)} + 1,399 \times 1.46\text{s} \approx 2,053\text{s} \approx \mathbf{0.57\text{ hours}}$ (34.2 minutes).
- `qwen3:4b` (at 512 budget):
  - Observed latency (all truncated at 512 tokens): ~16.64s per prompt.
  - Estimated 1,400 prompts: $1,400 \times 16.64\text{s} \approx 23,296\text{s} \approx \mathbf{6.47\text{ hours}}$.
- `qwen3:8b` (at 512 budget):
  - Observed latency: ~35.72s per prompt.
  - Estimated 1,400 prompts: $1,400 \times 35.72\text{s} \approx 50,008\text{s} \approx \mathbf{13.89\text{ hours}}$.

### B. Total Estimated Runtime
$$\text{Total Matrix Runtime} \approx 0.57\text{h} + 6.47\text{h} + 13.89\text{h} \approx \mathbf{20.93\text{ hours}}$$

*(Note: If `qwen3:4b` were evaluated at 1,536 tokens where it finishes, its latency is ~50.0s/prompt, requiring 19.44 hours for 4B alone, bringing total matrix runtime to **33.90 hours**).*

### C. Explicit Assumptions
1. Sequential execution with zero model swapping between prompts (grouping all 1,400 prompts per model to minimize weight reloading).
2. Continuous, uninterrupted execution with zero network socket timeouts or process crashes.
3. Stable thermal performance on the laptop GPU/CPU with no severe thermal throttling during 21–34 hours of heavy sustained compute.
4. Prompt distribution in the 1,400 training set exhibits comparable token length and generation complexity to the audited MMLU abstract algebra subset.

---

## 8. Specific Blockers and Smallest Next Action

### Blockers:
1. **Blocker 1 (Candidate Starvation)**: Routing requires selecting between candidate models based on cost, latency, and quality. Currently:
   - `qwen3:1.7b` is the only model meeting the latency SLA, but achieves only 60% accuracy.
   - `qwen3:4b` produces 100% truncated output at 512 tokens.
   - `qwen3:8b` produces 100% accurate output but violates the 10-second SLA on 100% of queries.
   - Matrix execution in this state collapses the 3-model optimization into a single-model or empty decision set.
2. **Blocker 2 (Ollama Thinking Monologue on 4B)**: `think: false` fails to suppress raw reasoning tokens in `qwen3:4b`, overflowing the multiple-choice token budget.
3. **Blocker 3 (Missing Resilient Runner)**: No checkpointed matrix runner exists that can safely resume after an interruption.

### Smallest Next Action:
**Do not start the full matrix.**
1. Implement a dedicated, checkpointing matrix runner (`runners/matrix_runner.py`) that writes per-batch Parquet partitions and checks existing `(prompt_id, model_id)` records prior to execution.
2. Address the `qwen3:4b` thinking issue at the provider/modelfile level (e.g., investigating modelfile system prompt tuning or Ollama `/api/generate` options) before committing to 1,400 inferences.
3. Decide on the routing governance role for `qwen3:8b` (e.g., designating it as an offline teacher/oracle, or formally defining tiered latency classes).

---

## 9. Verification & Hash Integrity

### A. Pytest Suite Execution
```powershell
.venv\Scripts\python -m pytest
```
- **Result**: `258 passed in 5.25s` (0 failed, 0 errors, 0 skipped).

### B. Frozen Benchmark SHA-256 Hashes
All four frozen benchmark datasets were recomputed directly against their established expected values:
- `data/processed/benchmark_train.jsonl`: `ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014` (**MATCH: TRUE**)
- `data/processed/benchmark_validation.jsonl`: `5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143` (**MATCH: TRUE**)
- `data/processed/benchmark_test.jsonl`: `ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b` (**MATCH: TRUE**)
- `data/processed/benchmark_stress.jsonl`: `36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747` (**MATCH: TRUE**)

**Zero benchmark files or hashes were modified.**
