# Phase 4H: Pre-Matrix Readiness Audit Report

**Date**: 2026-10-20  
**Environment**: Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB dedicated VRAM), 16 GB DDR5 RAM  
**Inference Backend**: Ollama v0.40.3 (Native `/api/chat`, `http://127.0.0.1:11434`)  
**Status**: Infrastructure Audit Complete — **Candidate Matrix Readiness: NOT READY (Blocked)**

---

## 1. Prompt Provenance Audit

### Investigation of `val_0000` through `val_0004`
The Phase 4G smoke report evaluated five prompts from `data/processed/benchmark_train.jsonl` whose canonical `prompt_id`s ended in `val_0000` through `val_0004`:
- `mmlu-abstract_algebra-validation-val_0000`
- `mmlu-abstract_algebra-validation-val_0001`
- `mmlu-abstract_algebra-validation-val_0002`
- `mmlu-abstract_algebra-validation-val_0003`
- `mmlu-abstract_algebra-validation-val_0004`

### Audit Evidence
1. **Source Dataset Ingestion**:
   In `data_ingestion/acquire_mmlu.py` and `data_ingestion/mmlu.py`, records were sampled from Hugging Face dataset `cais/mmlu` (subject `abstract_algebra`). Upstream MMLU provides splits named `"dev"`, `"validation"`, and `"test"`. The benchmark builder sampled from MMLU's upstream `source_split = "validation"` to form the benchmark's training split (`split = "train"`).
2. **Canonical Schema Formatting**:
   The benchmark schema in [data_ingestion/mmlu.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/data_ingestion/mmlu.py) constructs `prompt_id` as:
   ```python
   prompt_id = f"mmlu-{self.source_config}-{source_split}-{source_id}"
   ```
   Thus, `source_split` (`validation`) and `source_id` (`val_0000`) reflect **upstream source provenance**, whereas the router's benchmark split is recorded in the `split` field.
3. **Partition Verification**:
   - `data/processed/benchmark_train.jsonl` contains `val_0000` through `val_0004` with `"split": "train"`.
   - `data/processed/benchmark_validation.jsonl` begins at `val_0005` with `"split": "validation"`.
   - `data/processed/benchmark_test.jsonl` and `data/processed/benchmark_stress.jsonl` contain no overlapping IDs.
4. **Conclusion**:
   There was **zero leakage** from held-out validation or test splits. The five prompts are 100% verified members of `benchmark_train.jsonl`.

---

## 2. Qwen3-4B Truncation Diagnostic

In Phase 4G, `qwen3:4b` truncated on all five prompts at a 768-token budget. In Phase 4H, controlled sequential diagnostics were executed using `think: false` at budgets of 1,024 and 1,536 tokens.

### Empirical Results Across Tested Budgets

| Output Token Budget | Truncation Count | Success Count | Truncation Rate | Latency Range | Output Tokens Range | Scoring Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **768 tokens (4G)** | 5 / 5 | 0 / 5 | **100%** | 23.3s – 31.5s | 768 (capped) | 5 SCORING_UNAVAILABLE |
| **1,024 tokens (4H)** | 5 / 5 | 0 / 5 | **100%** | 31.6s – 41.0s | 1,024 (capped) | 5 SCORING_UNAVAILABLE |
| **1,536 tokens (4H)** | 4 / 5 | 1 / 5 | **80%** | 49.0s – 55.2s | 1,518 – 1,536 | 1 Scored (1.0), 4 UNAVAILABLE |

### Per-Prompt Breakdown at 1,536 Budget (`qwen3-4b`)

| Prompt ID | Finish Reason | Output Tokens | Latency (s) | Truncated? | Scorer Result |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `mmlu-...-val_0000` | length | 1,536 | 50.29s | **Yes** | SCORING_UNAVAILABLE (`NaN`) |
| `mmlu-...-val_0001` | stop | 1,518 | 49.00s | **No** | **SUCCESS (1.0, choice B)** |
| `mmlu-...-val_0002` | length | 1,536 | 51.88s | **Yes** | SCORING_UNAVAILABLE (`NaN`) |
| `mmlu-...-val_0003` | length | 1,536 | 55.25s | **Yes** | SCORING_UNAVAILABLE (`NaN`) |
| `mmlu-...-val_0004` | length | 1,536 | 52.05s | **Yes** | SCORING_UNAVAILABLE (`NaN`) |

### Root Cause of 4B Truncation
Inspection of generated text revealed that although `think: false` successfully disables Ollama's `<think>` tags, `qwen3:4b`'s instruction tuning defaults to an extensive conversational chain-of-thought in `content` (*"Okay, let's try to figure out this problem. First, I need to remember what $Z_6$ is... Step 1... Step 2..."*). 
- It requires **over 1,500 tokens** to solve even basic abstract algebra multiple-choice questions.
- At 1,536 tokens, only 1 of 5 prompts concluded before running out of budget.
- Each 4B evaluation at 1,536 tokens consumes ~50 seconds of GPU time.

---

## 3. SLA Handling & Hardware Telemetry Audit

Configured policy in [configs/policies.yaml](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/configs/policies.yaml):
```yaml
routing:
  latency:
    enabled: true
    max_latency_ms: 10000
```

### Measured Latency vs. 10,000 ms SLA

| Model | Hardware Placement | Measured Generation Speed | Measured Latencies (5 prompts) | SLA Compliant (<10s) | Inference Success |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`qwen3:1.7b`** | 100% Dedicated GPU VRAM | ~60–80 tok/s | 8.2s (cold), 3.1s, 3.3s, 5.8s, 5.9s | **5 / 5 (100%)** | **5 / 5 (100%)** |
| **`qwen3:4b`** | 100% Dedicated GPU VRAM | ~25–30 tok/s | 31.6s – 41.0s (1024) / 49s–55s (1536) | **0 / 5 (0%)** | **0 / 5 (768/1024), 1/5 (1536)** |
| **`qwen3:8b`** | Hybrid GPU (2.3GB) + CPU RAM | ~8.5–10 tok/s | 74.0s (cold), 43.9s, 47.1s, 50.5s, 73.2s | **0 / 5 (0%)** | **5 / 5 (100%)** |

### Practical Hardware & Routing Implications
1. **Inference Success vs. SLA Compliance**:
   - `qwen3:8b` has **100% inference success** with zero timeouts and excellent generation quality, but **0% latency SLA compliance**.
   - `qwen3:8b` generates 400–600 tokens on mathematical QA. Offloaded into DDR5 RAM at ~9 tok/s, its wall-clock latency is physically bounded between **44 and 75 seconds**.
2. **Impact on Learned Router**:
   - If `routing.latency.enabled: true` with `max_latency_ms: 10000` is enforced as a strict hard filter during routing, `qwen3:8b` and `qwen3:4b` will be disqualified on nearly all reasoning prompts.
   - The router would be forced into a degenerate state where only `qwen3:1.7b` can be selected.
   - For offline batch benchmarking or quality-first routing, the latency policy must either treat latency as a soft penalty objective (`latency_penalty`) or use task-differentiated latency SLAs.

---

## 4. Inference & Scoring Pipeline Audit

### Scorer Vulnerability Discovered and Resolved
A deep audit of [eval/scorers/multiple_choice.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/eval/scorers/multiple_choice.py) uncovered two critical flaws:
1. **Unanchored Word Regex Fallback**:
   When `extract_mc_choice` failed to find a choice, lines 71–75 fell back to `re.search(rf"\b{ref_choice}\b", prediction, re.IGNORECASE)`. When the target answer was `"A"`, any English explanation containing the indefinite article `"a"` was awarded `quality_score = 1.0`.
   - On prompt `val_0002`, `qwen3:1.7b` actually predicted **C** (incorrect, target was **A**). The fallback regex matched the English word `"a"`, giving 1.7B an undeserved 1.0.
2. **Missing LaTeX & Markdown Choice Patterns**:
   The extractor previously did not parse nested LaTeX `\boxed{\text{A. 4}}` or markdown `### ✅ Final Answer: **C**`.

### Fix Implemented in [eval/scorers/multiple_choice.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/eval/scorers/multiple_choice.py)
- Added nested-brace LaTeX `\boxed{...}` extraction.
- Added markdown bold header `Final Answer: **...**` extraction.
- Completely removed the unanchored single-letter word fallback.
- Added comprehensive unit tests in [tests/test_scoring.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/tests/test_scoring.py).

### True Verified Quality on 5 Training Prompts

| Model | Phase 4G Reported Quality (Flawed Scorer) | Phase 4H Verified Quality (Robust Scorer) | Notes |
| :--- | :---: | :---: | :--- |
| **`qwen3:1.7b`** | 1.00 (5/5) | **0.80 (4/5)** | Corrected `val_0002` (model predicted C, target was A). |
| **`qwen3:4b`** | N/A (Guarded Truncation) | **N/A (0/5 at 1024) / 1.0 (1/1 scored at 1536)** | 4 of 5 truncated at 1536; 1 completed correctly. |
| **`qwen3:8b`** | 1.00 (5/5) | **1.00 (5/5)** | Genuinely answered all 5 correctly (`A`, `B`, `A`, `C`, `C`). |

---

## 5. Test Suite & Frozen Benchmark Integrity

1. **Full Pytest Execution**:
   ```powershell
   .venv\Scripts\python -m pytest
   ```
   **Result**: `257 passed in 3.61s` (0 failed, 0 skipped).
2. **Frozen Benchmark SHA-256 Hashes**:
   - `benchmark_train.jsonl`: `ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014` (**Match: True**)
   - `benchmark_validation.jsonl`: `5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143` (**Match: True**)
   - `benchmark_test.jsonl`: `ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b` (**Match: True**)
   - `benchmark_stress.jsonl`: `36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747` (**Match: True**)

---

## 6. Changed Files & Artifact Paths

### Code and Tests Modified
- [runners/smoke_runner.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/smoke_runner.py): Added optional `model_ids` filtering and `--models` CLI parameter.
- [eval/scorers/multiple_choice.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/eval/scorers/multiple_choice.py): Enhanced extraction with nested LaTeX `\boxed{...}` and markdown headers; removed single-letter regex fallback.
- [tests/test_scoring.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/tests/test_scoring.py): Added unit tests for boxed, markdown choices, and false-positive prevention.

### Diagnostic Artifacts Preserved
- `data/outcomes/smoke_phase4g/`: Untouched original Phase 4G outcomes and scores.
- `data/outcomes/smoke_phase4h/budget_1024/`: 4B diagnostic at 1,024-token budget (manifest, outcomes, scores).
- `data/outcomes/smoke_phase4h/budget_1536/`: 4B diagnostic at 1,536-token budget (manifest, outcomes, scores).

---

## 7. Decision: Candidate Outcome Matrix Readiness

### **DECISION: NOT READY FOR MATRIX (BLOCKED)**

Launching the 1,400-prompt training matrix (4,200 total inference attempts) is blocked by three concrete operational issues:

1. **Qwen3-4B Truncation & Compute Profile**:
   - At current budgets (768–1,024), Qwen3-4B exhibits an 80–100% truncation rate on multiple-choice reasoning due to unconstrained conversational preamble.
   - At 1,536 tokens, 4B takes ~50s per prompt. Running 1,400 prompts on 4B alone would take **~19.4 hours**, with the majority of outcomes still truncating.
   - **Required Action**: Either inject a concise prompt constraint for multiple-choice tasks (e.g. system instruction: *"Solve concisely and conclude with the letter."*) or adjust the task budget policy before matrix evaluation.
2. **Latency SLA Policy Conflict**:
   - `routing.latency.max_latency_ms: 10000` is impossible for `qwen3:8b` (44–75s) and `qwen3:4b` (30–55s) on this hardware.
   - **Required Action**: Clarify in policy configuration whether 10,000 ms is an online serving target or if matrix collection should record observed latency without hard-failing candidate evaluations.
3. **Execution Runtime Estimate**:
   - 1.7B (1,400 prompts $\times$ 5s) = ~2 hours
   - 4B (1,400 prompts $\times$ 35s) = ~13.6 hours
   - 8B (1,400 prompts $\times$ 50s) = ~19.4 hours
   - Total sequential runtime is **~35 hours**. A checkpointed batch runner with automatic resumption is mandatory before starting Phase 5.
