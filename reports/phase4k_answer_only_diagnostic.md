# Phase 4K: Answer-Only Policy Diagnostic

**Date**: 2026-10-20  
**Hardware Profile**: Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB dedicated VRAM), 16 GB DDR5 RAM  
**Inference Backend**: Local Ollama v0.40.3 (`http://127.0.0.1:11434`)  
**Run ID**: `smoke-run-20261020-111254`  
**Output Artifacts Directory**: `data/outcomes/smoke_phase4k/answer_only_64/`  
**Investigator**: Antigravity  

---

## 1. Executive Summary & Go/No-Go Decision

### **DECISION: NO-GO FOR MATRIX LAUNCH**

The strict multiple-choice answer-only policy (`v2_answer_only`) with a 64-token budget **does not make the candidate models operationally usable**. In fact, it severely degrades usability across all three candidate models compared to the 512-token concise policy (`v1_concise`).

### Core Findings:
1. **`qwen3:4b` Truncation Persists (100% Truncation Rate)**: `v2_answer_only` fails to suppress raw reasoning tokens in Qwen3-4B. The model completely ignores *"Output only its option letter, with no explanation, reasoning, prefix, or other text"* and begins emitting its conversational internal monologue (`"Okay, let's see. The question is about finding..."`). Capped at 64 tokens, it truncates mid-sentence on **100% of attempts (5/5)** without ever generating an answer letter.
2. **`qwen3:1.7b` Regresses to 100% Truncation**: While `qwen3:1.7b` completed 100% of prompts at a 256/512 budget under `v1_concise`, under `v2_answer_only` it insists on providing explanatory proofs or appending commentary after stating the letter, truncating at 64 tokens on **100% of attempts (5/5)**.
3. **`qwen3:8b` Quality Collapses**: While `qwen3:8b` complied with the format on 4/5 prompts—achieving fast warm latency (~712 ms to 891 ms)—its task accuracy on scorable prompts collapsed from **100.0% (5/5)** under `v1_concise` to **50.0% (2/4)** under `v2_answer_only`. Depriving the model of reasoning scratchpad capacity causes it to guess incorrectly on abstract algebra.

---

## 2. Versioned Prompt Policy Specification: `v2_answer_only`

Implemented in [runners/inference_policy.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference_policy.py):

- **Policy Identifier**: `PROMPT_POLICY_V2_ANSWER_ONLY = "v2_answer_only"`
- **Instruction String (Multiple Choice)**:
  > *"Select the single best answer. Output only its option letter, with no explanation, reasoning, prefix, or other text."*
- **Policy Invariants**:
  - The frozen benchmark files (`data/processed/benchmark_*.jsonl`) remain **100% byte-identical and unmutated**.
  - The instruction is deterministically appended at inference request time: `f"{prompt.prompt}\n\nInstruction: {instruction}"`.
  - The policy version `v2_answer_only` is recorded on every [ModelOutcome](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/outcome_schema.py) and in the [RunManifest](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/outcome_schema.py).
  - Existing prompt policies (`v0`, `v1_concise`) remain intact.

---

## 3. Phase 4K Per-Attempt Diagnostic Results (64-Token Budget)

- **Configuration**: `splits: ["train"]`, `prompts_per_split: 5`, `max_output_tokens: 64`, `think: False`, `temperature: 0.0`, `seed: 42`, `timeout: 120.0s`.
- **Target Prompts**: The same five verified MMLU abstract algebra training prompts (`val_0000` to `val_0004`) from `benchmark_train.jsonl`.

| Model ID | Prompt ID | Status | Finish Reason | Tokens | Latency (ms) | Truncated? | Extracted | Expected | Scoring Status | Quality Score | Strict SLA (<10s) | Policy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen3-1.7b` | `mmlu-...-val_0000` | TRUNCATED | length | 64 | 5699.11 | Yes | None | A | SCORING_UNAVAILABLE | NaN | **PASS** | `v2_answer_only` |
| `qwen3-1.7b` | `mmlu-...-val_0001` | TRUNCATED | length | 64 | 627.98 | Yes | None *(Trunc)* | B | SCORING_UNAVAILABLE | NaN | **PASS** | `v2_answer_only` |
| `qwen3-1.7b` | `mmlu-...-val_0002` | TRUNCATED | length | 64 | 610.77 | Yes | None | A | SCORING_UNAVAILABLE | NaN | **PASS** | `v2_answer_only` |
| `qwen3-1.7b` | `mmlu-...-val_0003` | TRUNCATED | length | 64 | 622.27 | Yes | None *(Trunc)* | C | SCORING_UNAVAILABLE | NaN | **PASS** | `v2_answer_only` |
| `qwen3-1.7b` | `mmlu-...-val_0004` | TRUNCATED | length | 64 | 635.48 | Yes | None | C | SCORING_UNAVAILABLE | NaN | **PASS** | `v2_answer_only` |
| `qwen3-4b` | `mmlu-...-val_0000` | TRUNCATED | length | 64 | 11118.89 | Yes | None | A | SCORING_UNAVAILABLE | NaN | **FAIL** | `v2_answer_only` |
| `qwen3-4b` | `mmlu-...-val_0001` | TRUNCATED | length | 64 | 1959.51 | Yes | None | B | SCORING_UNAVAILABLE | NaN | **PASS** | `v2_answer_only` |
| `qwen3-4b` | `mmlu-...-val_0002` | TRUNCATED | length | 64 | 1998.91 | Yes | None | A | SCORING_UNAVAILABLE | NaN | **PASS** | `v2_answer_only` |
| `qwen3-4b` | `mmlu-...-val_0003` | TRUNCATED | length | 64 | 1985.29 | Yes | None | C | SCORING_UNAVAILABLE | NaN | **PASS** | `v2_answer_only` |
| `qwen3-4b` | `mmlu-...-val_0004` | TRUNCATED | length | 64 | 1978.85 | Yes | None | C | SCORING_UNAVAILABLE | NaN | **PASS** | `v2_answer_only` |
| `qwen3-8b` | `mmlu-...-val_0000` | SUCCESS | stop | 5 | 13019.49 | No | **D** | A | SUCCESS | **0.0** | **FAIL** | `v2_answer_only` |
| `qwen3-8b` | `mmlu-...-val_0001` | TRUNCATED | length | 64 | 5306.03 | Yes | None *(Trunc)* | B | SCORING_UNAVAILABLE | NaN | **PASS** | `v2_answer_only` |
| `qwen3-8b` | `mmlu-...-val_0002` | SUCCESS | stop | 6 | 890.82 | No | **D** | A | SUCCESS | **0.0** | **PASS** | `v2_answer_only` |
| `qwen3-8b` | `mmlu-...-val_0003` | SUCCESS | stop | 5 | 712.34 | No | **C** | C | SUCCESS | **1.0** | **PASS** | `v2_answer_only` |
| `qwen3-8b` | `mmlu-...-val_0004` | SUCCESS | stop | 6 | 792.69 | No | **C** | C | SUCCESS | **1.0** | **PASS** | `v2_answer_only` |

*(Note: Per scoring integrity rules, truncated responses are strictly guarded with `scoring_status = SCORING_UNAVAILABLE` and `quality_score = NaN`. They are never scored as 0.0, and are excluded from the accuracy denominator).*

---

## 4. Inspection of Raw Generated Outputs

### A. Qwen3-4B Output Inspection (Unsuppressed Monologue)
Direct inspection of the `generated_text` strings emitted by `qwen3:4b` at 64 tokens confirms that answer-only prompting does **not** stop internal monologue generation:
- **`val_0000`**: `"Okay, let's see. The question is about finding the order of the cyclic subgroup of Z_24 generated by 18. Hmm, right. So first, I need to remember that in the additive group Z_n, the subgroup generated by an element k is the set {0, k, 2"`
- **`val_0001`**: `"Okay, let's see. I need to find the order of the factor group Z_6 divided by the subgroup <3>. Hmm, first, I should remember what Z_6 is. Z_6 is the cyclic group of integers modulo 6, right? So it has elements {0, 1,"`
- **`val_0002`**: `'Okay, let\'s tackle this problem step by step. So, there are two statements here, Statement 1 and Statement 2. I need to figure out if each is true or false and then pick the correct option from A to D.\n\nStarting with Statement 1: "A permutation that is a product of m'`
- **`val_0003`**: `"Okay, let's try to figure out this problem step by step. The question is asking for the order of the factor group (Z_4 x Z_12)/(<2> x <2>). First, I need to remember that the order of a factor group is equal to the order of the original group"`
- **`val_0004`**: `"Okay, let's see. The question is asking for the maximum possible order of some element in Z_4 x Z_6. Hmm, I need to remember how the order of elements works in direct products of cyclic groups. \n\nFirst, I recall that in a direct product of two cyclic groups, say Z_m"`

**Conclusion on 4B**: The model outputs conversational reasoning tokens identical to its natural reasoning behavior. Because the 64-token budget is exhausted before the model reaches its conclusion, zero option letters are produced.

### B. Qwen3-1.7B Output Inspection (Conversational Prose & Preamble)
`qwen3:1.7b` also fails the answer-only constraint:
- **`val_0000`**: Begins solving directly with explanation: `"The cyclic subgroup of Z_24 generated by 18 is the set {0, 18, 36, 54} in Z_24. Since 18 and 24 are not coprime, the subgroup is not cyclic..."`
- **`val_0001`**: Emits `"A. 2\n\nThe factor group $ \mathbb{Z}_6 / \langle 3 \rangle $ has order equal to the order of the quotient group..."` and continues generating proof until truncating.
- **`val_0002`**: Begins analysis: `"The first statement is **false**. A permutation that is a product of $ m $ even permutations..."`

### C. Qwen3-8B Output Inspection (Direct Option Selection)
When `qwen3:8b` obeyed the prompt (4 out of 5 prompts), it produced minimal tokens:
- **`val_0000`**: `'D. 6'` (5 tokens) — **Incorrect** (Expected `A`).
- **`val_0001`**: Explanatory lapse: `'A\n\nThe factor group $\mathbb{Z}_6/\langle 3\rangle$ has order equal to the index...'` (64 tokens, Truncated).
- **`val_0002`**: `'D. False, True'` (6 tokens) — **Incorrect** (Expected `A`).
- **`val_0003`**: `'C. 4'` (5 tokens) — **Correct** (Expected `C`).
- **`val_0004`**: `'C. 12'` (6 tokens) — **Correct** (Expected `C`).

---

## 5. Comparative Analysis: Phase 4I (`v1_concise`, 512 tokens) vs Phase 4K (`v2_answer_only`, 64 tokens)

| Metric | Model | Phase 4I (`v1_concise`, 512 Tokens) | Phase 4K (`v2_answer_only`, 64 Tokens) | Operational Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Completion Rate** | `qwen3:1.7b`<br>`qwen3:4b`<br>`qwen3:8b` | **100% (5/5)**<br>0% (0/5)<br>**100% (5/5)** | 0% (0/5)<br>0% (0/5)<br>**80% (4/5)** | 1.7B regressed from 100% to 0% completion.<br>4B remains 100% truncated under both policies.<br>8B truncated on 1 prompt. |
| **Truncation Rate** | `qwen3:1.7b`<br>`qwen3:4b`<br>`qwen3:8b` | 0% (0/5)<br>**100% (5/5)**<br>0% (0/5) | **100% (5/5)**<br>**100% (5/5)**<br>20% (1/5) | Budget of 64 tokens induces severe truncation on smaller models. |
| **Accuracy (Scorable)** | `qwen3:1.7b`<br>`qwen3:4b`<br>`qwen3:8b` | 60.0% (3/5)<br>N/A (unscorable)<br>**100.0% (5/5)** | N/A (unscorable)<br>N/A (unscorable)<br>**50.0% (2/4)** | 8B accuracy dropped by 50 percentage points without reasoning scratchpad. |
| **Warm Latency** | `qwen3:1.7b`<br>`qwen3:4b`<br>`qwen3:8b` | 1.1s – 1.6s<br>16.5s – 16.8s<br>26.3s – 48.2s | **0.61s – 0.64s**<br>**1.96s – 2.00s**<br>**0.71s – 0.89s** | Generating fewer tokens drops latency drastically across all models. |
| **Strict SLA (<10s)** | `qwen3:1.7b`<br>`qwen3:4b`<br>`qwen3:8b` | **80% (4/5)**<br>0% (0/5)<br>0% (0/5) | **100% (5/5)**<br>**80% (4/5)**<br>**60% (3/5)** | Latency meets SLA, but output is invalid/truncated for 1.7B and 4B. |

### Diagnostic Evaluation:
1. **Did 4B become usable?** **NO**. 4B is completely unusable under `v2_answer_only` with a 64-token budget. It generates pure reasoning monologue and truncates on 100% of queries.
2. **Did 8B solve its latency problem?** **PARTIALLY, but with unacceptable quality collapse**. When 8B generates only 5–6 tokens, its warm latency drops from ~35 seconds to ~750 ms, easily meeting the 10,000 ms SLA. However, without reasoning tokens, its accuracy on abstract algebra collapsed from 100% to 50%. Furthermore, on 20% of prompts it reverted to generating explanation, causing a truncation error.
3. **Trade-off Summary**: Attempting to force an answer-only policy via prompting alone creates a lose-lose dynamic: smaller models ignore the instruction and truncate, while the larger model loses the reasoning capacity necessary to solve complex tasks.

---

## 6. Smallest Recommended Next Experiment

Rather than repeatedly increasing token budgets or blindly starting the full training matrix, the root issue is architectural: **Qwen3 chat models in Ollama have unsuppressed internal thinking or strong conversational conversational priors that cannot be eliminated via a simple trailing user prompt instruction.**

### Recommended Actions:
1. **Stop Parameter Testing (`stop: ["\n"]` or `stop: ["</think>"]`)**:
   - Test whether setting Ollama's `stop` parameter to stop immediately on newline or after an option letter prevents models from continuing to generate explanations.
2. **Ollama Modelfile / System Prompt Format Binding**:
   - Investigate whether passing a system prompt (e.g. `system: "You are an automated multiple-choice scoring agent. Output only the single capital letter of the correct choice."`) or setting the Ollama Modelfile parameter `PARAMETER stop "\n"` forces direct output.
3. **Few-Shot Prompt Policy (`v3_few_shot`)**:
   - Test a 1-shot in-context exemplar format. LLMs follow structural demonstrations far more reliably than negative verbal constraints ("do not explain").
4. **Architectural Tiering for the Matrix**:
   - Acknowledge that complex reasoning tasks (like MMLU abstract algebra and GSM8K) fundamentally require reasoning budgets. If 4B and 8B require reasoning tokens to achieve acceptable accuracy, the router's role is precisely to decide when to pay the latency/cost for those tokens versus routing to a fast model.

---

## 7. Verification & Hash Integrity

### A. Full Pytest Suite Execution
```powershell
.venv\Scripts\python -m pytest
```
- **Result**: `258 passed in 3.72s` (0 failed, 0 errors, 0 skipped).

### B. Frozen Benchmark SHA-256 Hashes
Re-verified against established expected checksums:
- `data/processed/benchmark_train.jsonl`: `ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014` (**MATCH: TRUE**)
- `data/processed/benchmark_validation.jsonl`: `5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143` (**MATCH: TRUE**)
- `data/processed/benchmark_test.jsonl`: `ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b` (**MATCH: TRUE**)
- `data/processed/benchmark_stress.jsonl`: `36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747` (**MATCH: TRUE**)

### C. Changed Files and Artifacts
- **Modified Source**:
  - [runners/inference_policy.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference_policy.py): Added `PROMPT_POLICY_V2_ANSWER_ONLY` and instruction mappings.
  - [tests/test_inference_infrastructure.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/tests/test_inference_infrastructure.py): Added unit test coverage for `v2_answer_only` prompt formatting and propagation.
- **Created Deliverables**:
  - [reports/phase4k_answer_only_diagnostic.md](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/reports/phase4k_answer_only_diagnostic.md)
  - `data/outcomes/smoke_phase4k/answer_only_64/smoke_manifest.json`
  - `data/outcomes/smoke_phase4k/answer_only_64/smoke_outcomes.parquet`
  - `data/outcomes/smoke_phase4k/answer_only_64/smoke_scores.parquet`
