# Phase 4R — Phase 4Q Audit and Targeted 4B Policy Validation Report

**Phase:** Phase 4R (Diagnostic Audit & Targeted Budget Validation)  
**Date:** 2026-10-29  
**Status:** Completed  
**Decision Gate Recommendation:** **NO-GO** on naive token budget expansion for `qwen3-4b`; **GO** for structured/stop-token constraints in the planned 100–200 prompt validation  
**Embargo Status:** Router training remains strictly blocked. Full 4,200-attempt matrix launch remains strictly blocked.  
**Experiment Fingerprints:**  
- Baseline (`task_aware`): `b993f706c3413f74e15f733aa2baffc01bb8863d5bcd56a755f11031094e23a8`  
- Candidate (`task_aware_candidate`): `14e371a4003b9306beb8659935f8ec8a2cb4fc0e03e1b105ed64241e8b5a7925`  
**Output Directory:** `data/outcomes/pilot_phase4r`  
**Audit Baseline Directories:** `data/outcomes/pilot_phase4p`, `data/outcomes/pilot_phase4q`  

---

## 1. Executive Summary

Phase 4R performed two sequential, rigorously controlled objectives:
1. **Independent Comprehensive Audit of Phase 4Q Claims and Artifacts:** Independently recomputed model-level and overall outcome counts, truncations, scoring availability, quality denominators, latency distributions, and strict `<10,000 ms` SLA compliance directly from persisted Parquet files in `data/outcomes/pilot_phase4p/` and `data/outcomes/pilot_phase4q/`. All claims were verified, and unsupported or loose statements in the Phase 4Q report were formally identified and corrected.
2. **Targeted 4B Budget Experiment:** Conducted a strictly bounded paired experiment on the exact 7 prompts that truncated under Phase 4Q for `qwen3-4b`, testing whether expanding the token budget for classification (64 -> 128 tokens) and summarization (256 -> 384 tokens) resolves truncations without unacceptable latency or quality regressions. Exactly 14 new inference attempts (7 baseline, 7 candidate) were executed against Ollama.

### Key Findings:
- **Phase 4Q Claim Audit:** Verified 48 outcomes and 48 scores per phase across identical 16-prompt paired sets. Verified 41 `SUCCESS` and 7 `TRUNCATED` outcomes in Phase 4Q. Verified that all quality calculations strictly exclude unscorable attempts rather than falsifying them as 0.0.
- **Formal Correction to Phase 4Q Report:** Identified that the Phase 4Q report's claim that multiple-choice achieved "sub-second latencies" across all models was **factually unsupported for `qwen3-8b`**. `qwen3-8b` achieved only 2 sub-second inferences out of 16 (both were BoolQ classification), had a median latency of 4,751.9 ms, and had MC latencies up to 12,735.5 ms (violating the 10s SLA).
- **Targeted 4B Experiment Verdict:** Naive token budget expansion **completely failed** to resolve `qwen3-4b` truncations (0/7 completed in baseline, 0/7 completed in candidate). Qualitative analysis of generated tokens revealed that `qwen3-4b` generates extensive internal conversational reasoning preambles (*"Hmm, the user has given me a passage..."*, *"Looking at the article, it covers several key points... That's two sentences - perfect"*) in natural language when not constrained by formal grammar.
- **Latency Regression:** Expanding token budgets merely allowed `qwen3-4b` to emit more conversational reasoning tokens, doubling classification latency (~2.3s -> ~4.1s) and inflating summarization latency from ~9.6s (within SLA) to ~13.5s (violating the 10,000 ms SLA).

---

## 2. Comprehensive Audit of Phase 4Q Artifacts and Claims

All assertions in `reports/phase4q_task_aware_policy_validation.md` were cross-checked against the raw Parquet and JSON files on disk. Phase 4P and Phase 4Q files were preserved without modification.

### 2.1 Artifact and Record Integrity

| Check | Expected | Phase 4P Verified | Phase 4Q Verified | Status |
| :--- | :---: | :---: | :---: | :---: |
| `matrix_outcomes.parquet` rows | 48 | 48 | 48 | **VERIFIED** |
| `matrix_scores.parquet` rows | 48 | 48 | 48 | **VERIFIED** |
| Unique prompt count | 16 | 16 | 16 | **VERIFIED** |
| Prompt ID paired match | 100% | 16 / 16 identical | 16 / 16 identical | **VERIFIED** |
| Overall Success count | 31 (4P), 41 (4Q) | 31 `SUCCESS` | 41 `SUCCESS` | **VERIFIED** |
| Overall Truncation count | 17 (4P), 7 (4Q) | 17 `TRUNCATED` | 7 `TRUNCATED` | **VERIFIED** |
| Overall Failure / Timeout count | 0 | 0 `FAILED` | 0 `FAILED` | **VERIFIED** |

### 2.2 Model-Level Completion and Strict SLA (<10,000 ms) Compliance

| Model ID | Phase | Attempts | SUCCESS | TRUNCATED | SLA Met (<10s) | SLA Compliance | Median Latency | Mean Latency | Max Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **qwen3-1.7b** | Phase 4P | 16 | 14 | 2 | 16 / 16 | 100.0% | 1,655.1 ms | 2,078.4 ms | 4,979.2 ms |
| | **Phase 4Q** | 16 | **16** | **0** | **16 / 16** | **100.0%** | **738.5 ms** | **1,291.0 ms** | **5,738.6 ms** |
| **qwen3-4b** | Phase 4P | 16 | 2 | 14 | 2 / 16 | 12.5% | 18,562.0 ms | 19,769.5 ms | 34,246.6 ms |
| | **Phase 4Q** | 16 | **9** | **7** | **10 / 16** | **62.5%** | **8,968.2 ms** | **9,092.4 ms** | **34,065.2 ms** |
| **qwen3-8b** | Phase 4P | 16 | 15 | 1 | 3 / 16 | 18.8% | 23,571.1 ms | 27,106.3 ms | 77,734.4 ms |
| | **Phase 4Q** | 16 | **16** | **0** | **10 / 16** | **62.5%** | **4,751.9 ms** | **9,611.9 ms** | **52,001.7 ms** |
| **Overall** | Phase 4P | 48 | 31 | 17 | 21 / 48 | 43.8% | 13,515.2 ms | 16,318.1 ms | 77,734.4 ms |
| | **Phase 4Q** | 48 | **41** | **7** | **36 / 48** | **75.0%** | **2,778.6 ms** | **6,665.1 ms** | **52,001.7 ms** |

### 2.3 Quality Score Denominators and Non-Penalization Audit

The raw score distributions confirm that unscorable and sandbox-quarantined attempts are never imputed as zero:

| Model ID | Phase | Valid Scored / Total | Mean Quality (Valid Only) | If Masked as Zero | Scoring Status Breakdown |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **qwen3-1.7b** | Phase 4P | 12 / 16 | 0.5447 | 0.4086 | 12 SUCCESS, 2 SANDBOX_UNAVAILABLE, 2 SCORING_UNAVAILABLE |
| | Phase 4Q | 14 / 16 | 0.4787 | 0.4188 | 14 SUCCESS, 2 SANDBOX_UNAVAILABLE |
| **qwen3-4b** | Phase 4P | 2 / 16 | 0.5000 | 0.0625 | 2 SUCCESS, 14 SCORING_UNAVAILABLE |
| | Phase 4Q | 8 / 16 | 0.8750 | 0.4375 | 8 SUCCESS, 7 SCORING_UNAVAILABLE, 1 SANDBOX_UNAVAILABLE |
| **qwen3-8b** | Phase 4P | 13 / 16 | 0.5810 | 0.4720 | 13 SUCCESS, 2 SANDBOX_UNAVAILABLE, 1 SCORING_UNAVAILABLE |
| | Phase 4Q | 14 / 16 | 0.6910 | 0.6046 | 14 SUCCESS, 2 SANDBOX_UNAVAILABLE |

### 2.4 Formal Report Corrections and Discrepancy Reconciliation

1. **Rebuttal of "Sub-Second Latencies" for 8B:**
   - *Phase 4Q Report Statement (Line 285):* Claimed that multiple choice elimination achieved *"sub-second latencies and schema validity across 18/18 attempts"*.
   - *Audit Finding:* This statement is incorrect for `qwen3-8b`. In Phase 4Q, `qwen3-8b` achieved sub-second latencies on **zero** multiple-choice tasks. Its multiple-choice latencies were:
     - `mmlu-val_0000` (QA): **12,735.5 ms** (violates SLA)
     - `mmlu-val_0001` (QA): **1,035.4 ms**
     - `bbh-strategy_b_0000` (MC): **3,886.7 ms**
     - `hellaswag-train_0000` (MC): **1,324.4 ms**
     - `hellaswag-train_0001` (MC): **1,349.7 ms**
     - Sub-second latencies (<1,000 ms) for 8B occurred on only 2 attempts out of 16 overall (12.5%), and both were BoolQ classification tasks (`train_0000`: 836.4 ms; `train_0001`: 638.2 ms).
     - Sub-second multiple-choice latency was only achieved by `qwen3-1.7b` (all 6 MC attempts between 158.8 ms and 671.2 ms) and partially by `qwen3-4b` (4 of 6 MC attempts between 623.5 ms and 652.7 ms).
2. **Reconciliation of IFEval Thinking Mode:**
   - *Phase 4Q Report Table 75:* Listed IFEval thinking mode as `True`.
   - *Audit Finding:* In [`runners/inference_policy.py`](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference_policy.py) line 408, IFEval dispatch resolves to `thinking = user_think if user_think is not None else False`. Because `user_think` was `None` in the runner invocation, IFEval was executed with `thinking=False` in Ollama.
3. **Audit of Seven 4B Truncations:**
   - All 7 truncations in Phase 4Q for `qwen3-4b` were verified to match the stated task types and budgets:
     - 2 BoolQ classification tasks (`train_0000`, `train_0001`): max tokens 64 under `v2_answer_only`.
     - 2 CNN/DailyMail summarization tasks (`42c027...`, `ee8871...`): max tokens 256 under `v1_concise`.
     - 1 BBH reasoning task (`strategy_b_0001`): max tokens 256 under `v1_concise`.
     - 1 MBPP coding task (`601`): max tokens 512 under `v1_concise`.
     - 1 IFEval task (`2391`): max tokens 1024 under `v0`.
4. **Settings and Fingerprint Audit:**
   - `matrix_manifest.json` correctly stores runner-level parameters (`output_budget: null`, `think: null`, `response_schema: {"mode": "task_aware_dispatch"}`).
   - The cryptographic fingerprint `4b41b709...` hashes the run-level configuration.
   - Dynamic per-task settings actually delivered to Ollama are verified through `matrix_outcomes.parquet` row entries (`max_output_tokens`, `prompt_policy_version`, `temperature`, `seed`, `backend`).

---

## 3. Targeted 4B Budget Experiment Design

### 3.1 Prompts Selection
Extracted exclusively from `data/outcomes/pilot_phase4p/pilot_prompts.jsonl` into `data/outcomes/pilot_phase4r/pilot_prompts_affected.jsonl`. Exactly 7 prompts:
1. `bbh-boolean_expressions-test-strategy_b_0001` (reasoning, exact_match)
2. `mbpp-full-train-601` (coding, code_execution)
3. `ifeval-default-train-2391` (instruction_following, ifeval)
4. `cnn_dailymail-3.0.0-train-42c027e4ff9730fbb3de84c1af0d2c506e41c3e4` (summarization, reference_based)
5. `cnn_dailymail-3.0.0-train-ee8871b15c50d0db17b0179a6d2beab35065f1e9` (summarization, reference_based)
6. `boolq-default-train-train_0000` (classification, exact_match)
7. `boolq-default-train-train_0001` (classification, exact_match)

### 3.2 Paired Execution Settings

| Setting | Baseline Arm | Candidate Arm |
| :--- | :--- | :--- |
| **Output Directory** | `data/outcomes/pilot_phase4r/baseline` | `data/outcomes/pilot_phase4r/candidate` |
| **Model** | `qwen3-4b` (`Qwen/Qwen3-4B`) | `qwen3-4b` (`Qwen/Qwen3-4B`) |
| **Prompt Policy** | `task_aware` | `task_aware_candidate` |
| **Classification Budget** | **64 tokens** | **128 tokens** (+100%) |
| **Summarization Budget** | **256 tokens** | **384 tokens** (+50%) |
| **Reasoning Budget** | 256 tokens | 256 tokens (retained) |
| **Coding Budget** | 512 tokens | 512 tokens (retained) |
| **IFEval Budget** | 1024 tokens | 1024 tokens (retained) |
| **Temperature / Seed** | 0.0 / 42 | 0.0 / 42 |
| **Timeout / Backend** | 120.0s / `ollama` | 120.0s / `ollama` |
| **Schema Isolation** | No JSON schema applied to generative tasks | No JSON schema applied to generative tasks |
| **Planned Attempts** | Exactly 7 | Exactly 7 |

**Execution Commands:**
```bash
# Baseline
python -m runners.matrix_runner \
  --prompts-file data/outcomes/pilot_phase4r/pilot_prompts_affected.jsonl \
  --output-dir data/outcomes/pilot_phase4r/baseline \
  --models qwen3-4b \
  --prompt-policy task_aware \
  --backend ollama \
  --timeout 120.0

# Candidate
python -m runners.matrix_runner \
  --prompts-file data/outcomes/pilot_phase4r/pilot_prompts_affected.jsonl \
  --output-dir data/outcomes/pilot_phase4r/candidate \
  --models qwen3-4b \
  --prompt-policy task_aware_candidate \
  --backend ollama \
  --timeout 120.0
```

---

## 4. Evaluation Results and Paired Telemetry

### 4.1 Prompt-by-Prompt Paired Comparison

| Prompt ID | Task | Baseline Budget | Candidate Budget | Baseline Tokens | Candidate Tokens | Baseline Latency | Candidate Latency | Latency Delta | Baseline Status | Candidate Status | Baseline Score | Candidate Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bbh-strategy_b_0001` | reasoning | 256 | 256 | 256 | 256 | 15,275.9 ms | 8,452.2 ms | -6,823.7 ms | TRUNCATED | TRUNCATED | None | None |
| `mbpp-601` | coding | 512 | 512 | 512 | 512 | 15,999.2 ms | 16,915.6 ms | +916.4 ms | TRUNCATED | TRUNCATED | None | None |
| `ifeval-2391` | ifeval | 1024 | 1024 | 1024 | 1024 | 33,333.4 ms | 32,904.6 ms | -428.8 ms | TRUNCATED | TRUNCATED | None | None |
| `cnn_dailymail-42c027` | summarization | 256 | **384** | 256 | **384** | 9,332.5 ms | **13,295.2 ms** | **+3,962.6 ms** | TRUNCATED | TRUNCATED | None | None |
| `cnn_dailymail-ee8871` | summarization | 256 | **384** | 256 | **384** | 9,974.2 ms | **13,839.7 ms** | **+3,865.6 ms** | TRUNCATED | TRUNCATED | None | None |
| `boolq-train_0000` | classification | 64 | **128** | 64 | **128** | 2,320.1 ms | **4,162.7 ms** | **+1,842.6 ms** | TRUNCATED | TRUNCATED | None | None |
| `boolq-train_0001` | classification | 64 | **128** | 64 | **128** | 2,263.9 ms | **3,994.5 ms** | **+1,730.6 ms** | TRUNCATED | TRUNCATED | None | None |

### 4.2 Aggregate Comparison

| Metric | Baseline (`task_aware`) | Candidate (`task_aware_candidate`) | Impact / Delta |
| :--- | :---: | :---: | :---: |
| **Total Inferences** | 7 | 7 | Paired 1:1 |
| **Completion Rate** | 0.0% (0 / 7) | **0.0% (0 / 7)** | **No improvement (0 completions)** |
| **Truncation Rate** | 100.0% (7 / 7) | **100.0% (7 / 7)** | **Unchanged (100% truncated)** |
| **Valid Scored Count** | 0 / 7 (0.0%) | 0 / 7 (0.0%) | Excluded from quality average |
| **Mean Quality Score** | N/A (all truncated) | N/A (all truncated) | No false-zero penalties assigned |
| **SLA Met Count (<10,000 ms)** | 4 / 7 (57.1%) | **2 / 7 (28.6%)** | **-50% degradation in SLA compliance** |
| **Classification Median Latency** | 2,292.0 ms | **4,078.6 ms** | **+78.0% slower** |
| **Summarization Median Latency** | 9,653.4 ms (within SLA) | **13,567.5 ms (violates SLA)**| **+40.5% slower, breaches 10s SLA** |
| **Overall Median Latency** | 9,974.2 ms | **13,295.2 ms** | **+33.3% increase in latency** |

---

## 5. Root Cause Analysis: The `qwen3-4b` Conversational Preamble Trap

Why did increasing the budget from 64 to 128 for classification, and 256 to 384 for summarization, result in 100% truncation?

An inspection of the raw generated tokens reveals the exact mechanism:
1. **BoolQ Classification under `v2_answer_only`:**
   - Instruction: *"Answer true or false based on the text. Output only the final label ('true' or 'false'), with no explanation, reasoning, prefix, or other text."*
   - `qwen3-4b` Generation (128 tokens consumed):
     ```
     Hmm, the user has given me a passage about the Persian language and a question: "do iran and afghani...
     ```
   - Rather than adhering to the negative constraint, `qwen3-4b` behaves as an unconstrained conversational assistant, generating a streaming inner monologue explaining what the user wants. When given 64 tokens, it truncated at 64; when given 128 tokens, it simply continued its monologue to 128 tokens without ever emitting `true` or `false`.
2. **CNN/DailyMail Summarization under `v1_concise`:**
   - Instruction: *"Summarize the article concisely in 2-3 sentences. Do not include introductory preamble or conversational commentary."*
   - `qwen3-4b` Generation (384 tokens consumed):
     ```
     Hmm, the user wants me to summarize a Reuters article about Daniel Radcliffe gaining £20 million at age 18. They need a concise summary in 2-3 sentences without any preamble or conversational commentary.
     Looking at the article, it covers several key points: Daniel Radcliffe turns 18...
     I need to distill this into 2-3 tight sentences. I'll focus on the money aspect first...
     That's two sentences - perfect
     ```
   - The model spent over 300 tokens reasoning about *how* it would summarize the article before finally drafting sentences, running out of tokens at 384 right as it finished its internal reasoning.
3. **Contrast with Schema-Constrained Tasks:**
   - On Multiple-Choice tasks (MMLU, HellaSwag, BBH MC) where Ollama's grammar engine enforced `MULTIPLE_CHOICE_JSON_SCHEMA`, `qwen3-4b` completed in **11–12 tokens** with **623 ms latency** and **100% schema validity**.
   - Grammatical / schema constraints physically prevent the token sampling engine from choosing conversational tokens. Prompt instructions alone fail on `qwen3-4b`.

---

## 6. Verification and Integrity Check

### 6.1 Benchmark Integrity
All four frozen benchmark datasets match their authoritative checksums:
```
benchmark_train.jsonl: MATCH (ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014)
benchmark_validation.jsonl: MATCH (5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143)
benchmark_test.jsonl: MATCH (ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b)
benchmark_stress.jsonl: MATCH (36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747)
```

### 6.2 Pre-Existing Phase 4P and 4Q Artifact Immutability
All files in `data/outcomes/pilot_phase4p/` and `data/outcomes/pilot_phase4q/` were verified against pre-run SHA-256 hashes and confirmed **100% byte-for-byte unchanged**.

### 6.3 Matrix Runner Resume and Fingerprint Guard Verification
- **Clean Resume Test:** Re-running against completed baseline directory exited cleanly with `Run 'matrix-train-20261029-144102' in 'data\outcomes\pilot_phase4r\baseline' is already complete (7/7 attempts). Exiting cleanly.` (0 inferences re-executed).
- **Fingerprint Protection Test:** Attempting to execute candidate configuration into baseline directory correctly aborted with:
  ```
  [CONFIGURATION ERROR] Cannot resume matrix run due to configuration mismatch with existing manifest:
    - prompt_policy_version mismatch: existing='task_aware', current='task_aware_candidate'
  ```

### 6.4 Repository Test Suite
The full repository test suite was executed:
```
======================= 295 passed, 2 warnings in 7.59s =======================
```
All 295 unit and integration tests passed cleanly, including 4 new dedicated regression tests in [`tests/test_task_aware_policy.py`](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/tests/test_task_aware_policy.py) covering candidate policy dispatch, targeted budget isolation, and custom budget overrides.

---

## 7. Decision Gate & Recommendations

### 7.1 Decision Verdict: **NO-GO on Naive Budget Expansion**

Increasing output token budgets from 64 to 128 (classification) and 256 to 384 (summarization) for `qwen3-4b` is **firmly rejected**. It provides **zero** gain in completions (0/7 completed) while severely degrading latency (+78% on classification, +40% on summarization) and causing summarization to breach the 10-second SLA.

### 7.2 Actionable Recommendations for the 100–200 Prompt Validation Run

To successfully execute the upcoming 100–200 prompt validation run, the following architectural measures must be adopted:

1. **Classification Schema / Grammar Enforcement:**
   - Because `qwen3-4b` ignores negative instructions against conversational preambles in classification, classification tasks must either use a constrained JSON schema (e.g. `{"type": "object", "properties": {"label": {"type": "string", "enum": ["true", "false"]}}, "required": ["label"]}`) or 1-shot demonstration (`v3_few_shot`) to anchor the model on emitting labels immediately.
2. **Summarization Prompt Hardening:**
   - Add a firm stop token sequence or system instruction prefix (`/no_preamble`) to prevent `qwen3-4b` from generating planning thoughts in the output buffer.
3. **Coding Sandbox Quarantine:**
   - Maintain `SANDBOX_UNAVAILABLE` on MBPP / HumanEval until containerized execution sandbox isolation is provisioned.
4. **Strict Embargo Maintenance:**
   - Do NOT launch the full 4,200-attempt matrix.
   - Do NOT train the router until the 100–200 prompt validation confirms that the candidate policy achieves completion and SLA compliance across all candidate models.

