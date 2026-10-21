# Phase 4P — Controlled Training-Only Pilot Report

**Date:** 2026-10-21  
**Run ID:** `matrix-train-20261021-071538`  
**Target Split:** `train` (subset: `pilot_prompts.jsonl`)  
**Output Directory:** `data/outcomes/pilot_phase4p/`  
**Experiment Fingerprint:** `7b842144d7f031823102deae0acf753aa488b761ddcd66e40f92dcef8418dbef`  
**Backend:** Ollama (`http://127.0.0.1:11434`)  
**Candidate Models:** `qwen3:1.7b`, `qwen3:4b`, `qwen3:8b`  
**Inference Policy:** `v0` (original prompt unmodified, unconstrained generation, dynamic output budgets, dynamic thinking disabled `think=False`)  
**Sampling Configuration:** `temperature=0.0`, `seed=42`, `timeout=120.0s`, `flush_interval=1`  

---

## 1. Executive Summary

Phase 4P executed a strictly bounded, reproducible 48-attempt pilot matrix across the three local candidate models (`qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`) evaluating 16 deterministic prompts from `benchmark_train.jsonl` (2 per each of the 8 canonical benchmark categories).

The objective was to establish whether the end-to-end pipeline—including the hardened matrix runner, Ollama provider adapter, scoring engine, attempt history preservation, and atomic persistence—produces trustworthy outcomes before any full benchmark matrix or router training is attempted.

### Key Operational Findings:
1. **Pipeline & Runner Hardening:**
   - **Flawless Execution & Persistence:** Exactly 48 planned attempts were executed sequentially. Incremental atomic flushes persisted outcomes and scores after every single attempt.
   - **Failure & Retry Preservation:** All 48 initial preflight attempt histories (recording initial `BACKEND_UNAVAILABLE` before Ollama backend resolution) were preserved alongside the live inference outcomes with cumulative retry counters (`retry_count=1`).
   - **Interruption & Resumption Invariant:** Subsequent dry-run and live resume verification confirmed 48/48 attempts complete, 0 pending, and exactly 0 redundant inference calls executed.
   - **Telemetry Coverage:** 100% token usage (prompt, output, total) was captured across all 48 attempts with zero synthetic values or missing measurements.
2. **Model & Runtime Realities:**
   - **Truncation Pathology on `qwen3-4b`:** Under unconstrained policy `v0`, `qwen3-4b` generated repetitive chain-of-thought and explanations until hitting the output budget limit, resulting in a **87.5% truncation rate (14/16 prompts)**.
   - **Strict 10,000 ms SLA Compliance:**
     - `qwen3-1.7b`: **16/16 (100.0%)** met the strict SLA (median latency: 1,655 ms; max: 4,979 ms).
     - `qwen3-4b`: **2/16 (12.5%)** met the strict SLA (median latency: 18,562 ms; max: 34,247 ms).
     - `qwen3-8b`: **3/16 (18.8%)** met the strict SLA (median latency: 23,571 ms; max: 77,734 ms).
   - **Scoring Availability & Integrity:** No failed or truncated response was assigned a quality score of zero. Truncated responses (17) were assigned `SCORING_UNAVAILABLE` (`quality_score = None`), and code execution tasks without a container sandbox (4) were assigned `SANDBOX_UNAVAILABLE` (`quality_score = None`).
3. **Benchmark & Test Suite Integrity:**
   - All four frozen benchmark SHA-256 hashes remain perfectly identical to their canonical values.
   - The full test suite passed: **276 passed in 4.62s**.

---

## 2. Deterministic Pilot Selection & Manifest

The pilot subset was selected strictly from `data/processed/benchmark_train.jsonl` using the first two deterministic prompts appearing in file order for each of the eight canonical benchmark task categories. Selection was completely frozen prior to execution and saved to `data/outcomes/pilot_phase4p/pilot_manifest.json` and `pilot_prompts.jsonl` (SHA-256: `3bc02a8a1528fd0fdbb4f3ab29f603e0189d97bb5026e6a667188a450716076f`).

### Selected 16 Prompts:

| # | Prompt ID | Category / Task | Source Dataset | Source Config / Split | Output Budget |
|---|---|---|---|---|---|
| 1 | `mmlu-abstract_algebra-validation-val_0000` | `qa` | `mmlu` | `abstract_algebra` (val) | 768 |
| 2 | `mmlu-abstract_algebra-validation-val_0001` | `qa` | `mmlu` | `abstract_algebra` (val) | 768 |
| 3 | `bbh-boolean_expressions-test-strategy_b_0000` | `reasoning` | `bbh` | `boolean_expressions` (test) | 512 |
| 4 | `bbh-boolean_expressions-test-strategy_b_0001` | `reasoning` | `bbh` | `boolean_expressions` (test) | 512 |
| 5 | `gsm8k-main-train-train_0000` | `math` | `gsm8k` | `main` (train) | 512 |
| 6 | `gsm8k-main-train-train_0001` | `math` | `gsm8k` | `main` (train) | 512 |
| 7 | `mbpp-full-train-601` | `coding` | `mbpp` | `full` (train) | 768 |
| 8 | `mbpp-full-train-602` | `coding` | `mbpp` | `full` (train) | 768 |
| 9 | `ifeval-default-train-1943` | `instruction_following` | `ifeval` | `default` (train) | 1024 |
| 10 | `ifeval-default-train-2391` | `instruction_following` | `ifeval` | `default` (train) | 1024 |
| 11 | `cnn_dailymail-3.0.0-train-42c027e4ff9730fbb3de84c1af0d2c506e41c3e4` | `summarization` | `cnn_dailymail` | `3.0.0` (train) | 512 |
| 12 | `cnn_dailymail-3.0.0-train-ee8871b15c50d0db17b0179a6d2beab35065f1e9` | `summarization` | `cnn_dailymail` | `3.0.0` (train) | 512 |
| 13 | `boolq-default-train-train_0000` | `classification` | `boolq` | `default` (train) | 128 |
| 14 | `boolq-default-train-train_0001` | `classification` | `boolq` | `default` (train) | 128 |
| 15 | `hellaswag-train-train_0000` | `other` | `hellaswag` | `train` (train) | 768 |
| 16 | `hellaswag-train-train_0001` | `other` | `hellaswag` | `train` (train) | 768 |

---

## 3. Planned versus Actual Attempts

Execution strictly satisfied the initial 48-attempt cap. No model was over- or under-sampled.

| Model ID | Ollama Tag | Planned Attempts | Actual Attempts | Generation SUCCESS | Generation TRUNCATED | Timeouts | Failures |
|---|---|---|---|---|---|---|---|
| `qwen3-1.7b` | `qwen3:1.7b` | 16 | 16 | 14 (87.5%) | 2 (12.5%) | 0 (0.0%) | 0 (0.0%) |
| `qwen3-4b` | `qwen3:4b` | 16 | 16 | 2 (12.5%) | 14 (87.5%) | 0 (0.0%) | 0 (0.0%) |
| `qwen3-8b` | `qwen3:8b` | 16 | 16 | 15 (93.8%) | 1 (6.2%) | 0 (0.0%) | 0 (0.0%) |
| **Total** | — | **48** | **48** | **31 (64.6%)** | **17 (35.4%)** | **0 (0.0%)** | **0 (0.0%)** |

### Attempts by Task Category:

| Task Category | Planned | Actual | SUCCESS | TRUNCATED | Completion Rate |
|---|---|---|---|---|---|
| `qa` | 6 | 6 | 4 | 2 | 66.7% |
| `reasoning` | 6 | 6 | 4 | 2 | 66.7% |
| `math` | 6 | 6 | 5 | 1 | 83.3% |
| `coding` | 6 | 6 | 4 | 2 | 66.7% |
| `instruction_following` | 6 | 6 | 5 | 1 | 83.3% |
| `summarization` | 6 | 6 | 4 | 2 | 66.7% |
| `classification` | 6 | 6 | 1 | 5 | 16.7% |
| `other` | 6 | 6 | 4 | 2 | 66.7% |
| **Total** | **48** | **48** | **31** | **17** | **64.6%** |

---

## 4. Scoring Availability and Quality Metrics

### Explicit Denominator Clarifications:
- **Total Inferences:** 48
- **Unscored Inferences (Quality = None):** 21 (43.75%)
  - **17 Truncated Inferences:** Assigned `SCORING_UNAVAILABLE` (`quality_score = None`). Truncated outputs are incomplete and cannot be soundly scored. They are **not** treated as incorrect (zero).
  - **4 Code Execution Inferences:** Assigned `SANDBOX_UNAVAILABLE` (`quality_score = None`) because no Docker or container sandbox was present in this execution environment.
- **Scored Inferences:** 27 (56.25%)

### Quality Scores by Model (on Scored Subset):

| Model ID | Total Attempts | Scored Subset ($N$) | Sum Quality | Mean Quality Score | Standard Error |
|---|---|---|---|---|---|
| `qwen3-1.7b` | 16 | 12 | 6.537 | **0.5447 (54.5%)** | 0.141 |
| `qwen3-4b` | 16 | 2 | 1.000 | **0.5000 (50.0%)** | 0.354 |
| `qwen3-8b` | 16 | 13 | 7.553 | **0.5810 (58.1%)** | 0.133 |

*Note: For `qwen3-4b`, 14 of 16 inferences truncated, leaving only 2 scored instances (`gsm8k-0000` [0.0] and `ifeval-1943` [1.0]). The 50.0% score has an $N=2$ denominator and is purely diagnostic.*

### Quality Metrics by Category and Model:

| Category | `qwen3-1.7b` Quality ($N$) | `qwen3-4b` Quality ($N$) | `qwen3-8b` Quality ($N$) | Notes on Availability |
|---|---|---|---|---|
| `classification` (BoolQ) | N/A ($N=0$) | N/A ($N=0$) | 0.00 ($N=1$) | 5 of 6 attempts truncated at 128 tok budget |
| `coding` (MBPP) | N/A ($N=0$) | N/A ($N=0$) | N/A ($N=0$) | 4 `SANDBOX_UNAVAILABLE`, 2 truncated |
| `instruction_following` (IFEval) | 0.50 ($N=2$) | 1.00 ($N=1$) | 0.50 ($N=2$) | High format compliance on IFEval |
| `math` (GSM8K) | 1.00 ($N=2$) | 0.00 ($N=1$) | 0.00 ($N=2$) | Exact match extraction extracted final numbers |
| `other` (HellaSwag) | 0.50 ($N=2$) | N/A ($N=0$) | 1.00 ($N=2$) | Multiple choice letter extracted from text |
| `qa` (MMLU Abstract Algebra) | 1.00 ($N=2$) | N/A ($N=0$) | 1.00 ($N=2$) | Multiple choice letter extracted from text |
| `reasoning` (BBH Boolean) | 0.00 ($N=2$) | N/A ($N=0$) | 1.00 ($N=2$) | 8B correctly derived boolean values |
| `summarization` (CNN/DailyMail) | 0.2685 ($N=2$) | N/A ($N=0$) | 0.2763 ($N=2$) | ROUGE-1/L score on generated summaries |

---

## 5. Latency Distributions and SLA Compliance

The project enforces a strict, non-negotiable **10,000 ms SLA** for end-to-end client-observed generation latency.

### Summary Statistics (Client-Observed Latency in Milliseconds):

| Model ID | Min (ms) | Median (ms) | Mean (ms) | P90 (ms) | Max (ms) | $\le 10,000$ ms SLA Met | SLA % |
|---|---|---|---|---|---|---|---|
| `qwen3-1.7b` | 868.2 | 1,655.1 | 2,078.4 | 3,165.8 | 4,979.2 | **16 / 16** | **100.0%** |
| `qwen3-4b` | 4,146.6 | 18,562.0 | 19,769.5 | 31,104.0 | 34,246.6 | **2 / 16** | **12.5%** |
| `qwen3-8b` | 6,576.9 | 23,571.1 | 27,106.3 | 49,225.4 | 77,734.4 | **3 / 16** | **18.8%** |
| **Fleet Total** | **868.2** | **15,584.0** | **16,318.1** | **34,204.0** | **77,734.4** | **21 / 48** | **43.8%** |

### Operational Diagnosis of SLA Violations:
1. **`qwen3-1.7b`:** Fully compliant. Small model footprint (1.36 GB) runs entirely inside GPU VRAM, yielding generation throughput of ~100–120 tokens/sec. Even when generating 576 tokens on code, it completed in 4,979 ms.
2. **`qwen3-4b`:** Catastrophic failure under unconstrained policy `v0`. The model generated wordy preamble until exhausting 512–768 token limits. At ~25–30 tokens/sec, generating 768 tokens takes 24–34 seconds.
3. **`qwen3-8b`:** Succeeded on generation (15/16 SUCCESS) but severely failed the 10s SLA. Because the 8.2B model weights (~5.2 GB) exceed the local dedicated VRAM (2.3 GB VRAM allocated, balance in host RAM), llama.cpp operates in partial CPU offload. Generation speed is ~7–12 tokens/sec. Whenever the model generated >150 tokens, latency exceeded 15,000 ms; on 579 tokens, latency reached 77,734 ms.

---

## 6. Token Telemetry Coverage

Telemetry accounting was evaluated across all 48 attempts:

| Metric | `qwen3-1.7b` | `qwen3-4b` | `qwen3-8b` | Total Fleet |
|---|---|---|---|---|
| Prompt Tokens Recorded | 16 / 16 (100%) | 16 / 16 (100%) | 16 / 16 (100%) | **48 / 48 (100%)** |
| Output Tokens Recorded | 16 / 16 (100%) | 16 / 16 (100%) | 16 / 16 (100%) | **48 / 48 (100%)** |
| Total Tokens Recorded | 16 / 16 (100%) | 16 / 16 (100%) | 16 / 16 (100%) | **48 / 48 (100%)** |
| Missing / Fabricated Values | 0 | 0 | 0 | **0** |

All token counts reflect real backend generation metrics returned directly by the Ollama API.

---

## 7. Cold-Start versus Warm Latency

Because models are loaded on demand by Ollama into memory upon first invocation, the first attempt for each model reflects cold-start overhead (weight loading + context allocation).

| Model ID | Cold-Start Attempt | Cold Latency (ms) | Cold Tokens | Warm Attempts | Warm Median (ms) | Warm Min (ms) | Warm Max (ms) |
|---|---|---|---|---|---|---|---|
| `qwen3-1.7b` | `mmlu-...-val_0000` | 3,051.0 | 302 | 15 | 1,631.0 | 868.2 | 4,979.2 |
| `qwen3-4b` | `mmlu-...-val_0000` | 33,834.8 | 768 (trunc) | 15 | 18,091.8 | 4,146.6 | 34,246.6 |
| `qwen3-8b` | `mmlu-...-val_0000` | 77,734.4 | 579 | 15 | 23,092.1 | 6,576.9 | 50,732.4 |

For `qwen3-1.7b`, model load overhead was approximately 1.4 seconds. For `qwen3-8b`, loading ~5.2 GB of quantized weights from disk into memory took approximately 4.5 seconds of the initial 77.7s attempt.

---

## 8. Policy-Specific Results and Diagnostics

The pilot evaluated `v0` (original benchmark prompt, unconstrained output) because Phase 4M demonstrated that `v4_structured_output` is restricted to multiple choice schemas and cannot be applied to code synthesis, summarization, or IFEval.

### Findings on Policy `v0`:
- **Unconstrained Output Leads to Truncation:** Without prompt instructions enforcing brevity or schema boundaries, 4B and 8B default to rambling derivations. In BoolQ (128 token budget), 5 of 6 attempts hit the token ceiling.
- **Need for Task-Specific Policy Routing:** A single global policy is insufficient across diverse benchmark categories:
  - Multiple-choice and classification tasks require concise letter/label output policies (such as `v2_answer_only` or `v4_structured_output`) to fit within token budgets and satisfy latency SLAs.
  - Generative tasks (coding, summarization, IFEval) require standard generation budgets with thinking disabled to prevent runaway reasoning loops.

---

## 9. Resume and Recovery Verification

The matrix runner's recovery and deduplication mechanisms were verified under real operational conditions:

1. **Failure Preservation in Attempt History:**
   - In the initial preflight execution, all 48 attempts experienced `BACKEND_UNAVAILABLE` before Ollama provider resolution was fixed.
   - When retried with `--retry-failed`, the matrix runner preserved the initial failure record in `attempt_history` for all 48 outcome rows.
   - Every outcome recorded `retry_count = 1` and contained the exact timestamp, status (`BACKEND_UNAVAILABLE`), latency (`0.0 ms`), and error message (`Backend 'vllm' is not installed...`).
2. **Crash-Safe Checkpointing:**
   - The runner committed outcomes, scores, and manifest atomically to disk after every single prompt attempt (`flush_interval=1`).
3. **Idempotent Resume:**
   - A subsequent invocation verified:
     - 48 total planned attempts.
     - 48 previously completed attempts.
     - 0 pending attempts.
     - Matrix runner exited cleanly with 0 new inference requests.

---

## 10. Benchmark Hash & Test Suite Verification

### Frozen Benchmark SHA-256 Checksums:

| File | Canonical SHA-256 Hash | Post-Run SHA-256 Hash | Status |
|---|---|---|---|
| `benchmark_train.jsonl` | `ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014` | `ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014` | MATCH |
| `benchmark_validation.jsonl` | `5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143` | `5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143` | MATCH |
| `benchmark_test.jsonl` | `ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b` | `ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b` | MATCH |
| `benchmark_stress.jsonl` | `36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747` | `36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747` | MATCH |

### Full Test Suite Results:
- Command: `.venv\Scripts\python -m pytest`
- Result: **276 passed in 4.62s** (100% passing across 34 test modules).

---

## 11. Limitations of this Pilot

1. **Small Sample Size ($N=16$ Prompts):** This pilot was designed solely to diagnose pipeline stability, recovery, and runtime characteristics. It cannot establish general model quality benchmarks or justify router training.
2. **Local Hardware Constraints for 8B Model:** The 8.2B model operates in partial CPU offload on local hardware, limiting generation throughput to ~7–12 tokens/sec.
3. **No Container Sandbox for Code Execution:** MBPP prompts cannot currently be dynamically executed and verified in this environment, correctly resulting in `SANDBOX_UNAVAILABLE`.
4. **Policy `v0` Inefficiency:** Running unconstrained prompts across all task types induces severe truncation and latency bloat on 4B and 8B.

---

## 12. Final Decision

### **NO-GO for launching the full 4,200-attempt matrix or router training.**
### **CONDITIONAL GO for expanding to an intermediate evaluation phase, conditioned on resolving two specific blockers:**

1. **Resolution of Prompt Policy & Truncation on `qwen3-4b`:**
   - Launching a 4,200-inference matrix under `v0` would waste ~30 hours of compute and produce ~35% unusable, truncated rows.
   - **Requirement:** Task-appropriate prompt policies must be applied per task type (e.g. `v2_answer_only` or `v4_structured_output` for MC/classification; concise instructions for exact-match QA/math) so that models output their answers within their token budgets.
2. **SLA Realignment or Offload Strategy for `qwen3-8b`:**
   - `qwen3-8b` meets the 10,000 ms SLA on short responses (~80 tokens) but fails heavily on long responses when partially offloaded to CPU RAM.
   - **Requirement:** Before scaling up, the project must either enforce short output limits for 8B or formally document that 8B is ineligible for real-time 10s SLA routing on local hybrid CPU/GPU hardware.

**Do not train the router and do not launch the 4,200-attempt matrix until these conditions are met.**
