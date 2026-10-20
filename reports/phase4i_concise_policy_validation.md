# Phase 4I: Concise-Answer Policy Validation and Matrix Protocol Finalization

**Date**: 2026-10-20  
**Hardware**: Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB dedicated VRAM), 16 GB DDR5 RAM  
**Inference Backend**: Ollama v0.40.3 (Native `/api/chat`, `http://127.0.0.1:11434`)  
**Status**: Validation Complete — **Matrix Protocol Finalized**

---

## 1. Versioned Prompt Policy Specification

To eliminate excessive reasoning token production and prevent conversational preamble, a versioned prompt policy was implemented in [runners/inference_policy.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference_policy.py):

- **Policy Identifier**: `PROMPT_POLICY_V1_CONCISE = "v1_concise"`
- **Instruction Text (Multiple Choice)**:
  > *"Solve the question. Keep the explanation concise and end with `Final Answer: <letter>`. Do not repeat the question or provide unnecessary commentary."*
- **Policy Metadata Propagation**:
  - The frozen benchmark files (`data/processed/benchmark_*.jsonl`) and `BenchmarkPrompt` objects remain **100% byte-identical and unmutated**.
  - The instruction is deterministically appended to the prompt string at inference request time: `f"{prompt.prompt}\n\nInstruction: {instruction}"`.
  - The applied policy version is recorded directly on each [ModelOutcome](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/outcome_schema.py) (`prompt_policy_version: "v1_concise"`) and in the [RunManifest](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/outcome_schema.py).
  - The exact same instruction is applied symmetrically across all three candidate models without model-specific biases or answer leakage.

---

## 2. Controlled Diagnostic Results Across Tested Budgets

The diagnostic was executed on the same five verified training prompts from `data/processed/benchmark_train.jsonl` across all three models with `think: false` and prompt policy `v1_concise`.

Artifacts are preserved in:
- Initial 256-token budget: `data/outcomes/smoke_phase4i/budget_256/`
- Follow-up 512-token budget: `data/outcomes/smoke_phase4i/budget_512/` and root `data/outcomes/smoke_phase4i/`

### A. Initial 256-Token Multiple-Choice Budget

| Model | Success Rate | Truncation Rate | Output Tokens Range | Latency Range | Scored Quality | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`qwen3:1.7b`** | **100% (5/5)** | **0% (0/5)** | 118 – 224 tokens | **1.09s – 7.30s** | **0.60 (3/5)** | Highly concise; 100% under 256 tokens; 100% meet SLA. |
| **`qwen3:4b`** | **0% (0/5)** | **100% (5/5)** | 256 (capped) | 8.02s – 16.73s | N/A *(Guarded)* | Hit 256 limit during monologue; all truncated. |
| **`qwen3:8b`** | **40% (2/5)** | **60% (3/5)** | 212 – 256 tokens | 24.09s – 35.12s | **1.00 (2/2)** | Completed 2 prompts (both correct); truncated on 3 right at 256. |

*Finding*: 256 tokens was sufficient for 1.7B, but narrowly truncated 8B on 3 prompts. Per protocol instructions, a 512-token follow-up was executed.

### B. Follow-up 512-Token Multiple-Choice Budget

| Model | Success Rate | Truncation Rate | Output Tokens Range | Latency Range | Scored Quality | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`qwen3:1.7b`** | **100% (5/5)** | **0% (0/5)** | 118 – 224 tokens | **1.11s – 10.04s** | **0.60 (3/5)** | Fully complete; warm latencies ~1.1s–1.6s. |
| **`qwen3:4b`** | **0% (0/5)** | **100% (5/5)** | 512 (capped) | 16.49s – 23.99s | N/A *(Guarded)* | Emits raw text monologue closing with `</think>`; all truncated. |
| **`qwen3:8b`** | **100% (5/5)** | **0% (0/5)** | 212 – 408 tokens | 26.33s – 48.20s | **1.00 (5/5)** | **100% completed; zero truncations; 100% accuracy**. |

### Detailed Per-Prompt Breakdown at 512 Budget

| Model ID | Prompt ID | Status | Finish Reason | Tokens | Latency (s) | Extracted Choice | Target Choice | Quality Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen3-1.7b` | `mmlu-...-val_0000` | SUCCESS | stop | 224 | 10.04s (cold) | **A** | A | **1.0** |
| `qwen3-1.7b` | `mmlu-...-val_0001` | SUCCESS | stop | 118 | 1.11s (warm) | **B** | B | **1.0** |
| `qwen3-1.7b` | `mmlu-...-val_0002` | SUCCESS | stop | 164 | 1.56s (warm) | **C** | A | **0.0** (incorrect) |
| `qwen3-1.7b` | `mmlu-...-val_0003` | SUCCESS | stop | 176 | 1.58s (warm) | **D** | C | **0.0** (incorrect) |
| `qwen3-1.7b` | `mmlu-...-val_0004` | SUCCESS | stop | 178 | 1.59s (warm) | **C** | C | **1.0** |
| `qwen3-4b` | `mmlu-...-val_0000` | TRUNCATED | length | 512 | 23.99s (cold) | None | A | `NaN` *(Guarded)* |
| `qwen3-4b` | `mmlu-...-val_0001` | TRUNCATED | length | 512 | 16.49s (warm) | None | B | `NaN` *(Guarded)* |
| `qwen3-4b` | `mmlu-...-val_0002` | TRUNCATED | length | 512 | 16.54s (warm) | None | A | `NaN` *(Guarded)* |
| `qwen3-4b` | `mmlu-...-val_0003` | TRUNCATED | length | 512 | 16.84s (warm) | None | C | `NaN` *(Guarded)* |
| `qwen3-4b` | `mmlu-...-val_0004` | TRUNCATED | length | 512 | 16.69s (warm) | None | C | `NaN` *(Guarded)* |
| `qwen3-8b` | `mmlu-...-val_0000` | SUCCESS | stop | 212 | 34.97s (cold) | **A** | A | **1.0** |
| `qwen3-8b` | `mmlu-...-val_0001` | SUCCESS | stop | 222 | 26.33s (warm) | **B** | B | **1.0** |
| `qwen3-8b` | `mmlu-...-val_0002` | SUCCESS | stop | 320 | 38.51s (warm) | **A** | A | **1.0** |
| `qwen3-8b` | `mmlu-...-val_0003` | SUCCESS | stop | 408 | 48.20s (warm) | **C** | C | **1.0** |
| `qwen3-8b` | `mmlu-...-val_0004` | SUCCESS | stop | 259 | 29.83s (warm) | **C** | C | **1.0** |

---

## 3. Scorer Validation & False-Positive Elimination

In [eval/scorers/multiple_choice.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/eval/scorers/multiple_choice.py), the scoring logic was audited and hardened:

1. **Explicit Final Answer Extraction**:
   - Matches the last occurrence of explicit markers: `Final Answer: <letter>`, `Final Answer: **<letter>**`, `Final Answer: `letter``, `Final Answer: Option <letter>`.
   - Matches LaTeX `\boxed{...}` with nested braces.
2. **False-Positive Prevention**:
   - When a sentence contains the verb "answer" followed by an article (*"We must answer a very difficult question... Final Answer: B"*), the parser correctly ignores the verb and extracts **B**.
   - If a generation fails to provide an answer or only discusses mathematical properties of variable $b$, `extract_mc_choice` returns `None`.
   - The unanchored single-letter word fallback `\bA\b` was completely eradicated. Unanswered or unextracted prompts receive `0.0`, with zero false-positive credit given to English articles.
3. **Regression Tests**:
   - [tests/test_scoring.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/tests/test_scoring.py) verifies false-positive prevention, boxed extraction, markdown formatting, and option phrase matching.

---

## 4. Latency Policy & Offline Matrix Decoupling

The 10,000 ms SLA configured in [configs/policies.yaml](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/configs/policies.yaml) remains unchanged:

1. **Clean Separation of Roles**:
   - **Offline Matrix Collection**: Captures raw empirical reality. Every attempt records exact wall-clock latency (`latency_ms`), actual output tokens, finish reasons, and genuine task scores regardless of whether the latency exceeded 10,000 ms.
   - **Online Runtime Routing**: Evaluates routing decisions at inference time against `max_latency_ms: 10000`.
2. **SLA Compliance Rates**:
   - `qwen3:1.7b`: **100% compliant** on warm queries (1.1s – 1.6s latency; 1 cold start at 10.04s).
   - `qwen3:4b`: **0% compliant** (16.5s – 24.0s).
   - `qwen3:8b`: **0% compliant** (26.3s – 48.2s due to hybrid CPU RAM offload at ~9 tok/s).
3. **Empirical Verification**:
   - The matrix runner does **NOT** conflate latency SLA failure with inference failure. A 35-second `qwen3:8b` run is properly marked `status: SUCCESS`, `quality_score: 1.0`, and `latency_ms: 34968.03`.

---

## 5. Root Cause Analysis: Qwen3-4B Runtime Behavior

The investigation into `qwen3:4b`'s persistent truncation revealed a critical architectural finding:
- On `qwen3:1.7b` and `qwen3:8b`, Ollama's `think: false` parameter suppresses internal reasoning completely. Both models immediately produce concise step-by-step explanations and final answers in 118–408 tokens.
- On `qwen3:4b` under Ollama v0.40.3, `think: false` removes the `<think>` tag from the API envelope, but **the underlying model weights continue to output raw reasoning text into `content`**, ending with an explicit `</think>` tag in the plain text stream before starting the answer.
- Because this internal trace consumes ~450 tokens, `qwen3:4b` truncates on any budget below ~1,000 tokens on multiple-choice reasoning tasks.

---

## 6. Test Suite and Benchmark Hash Verification

1. **Full Pytest Execution**:
   ```powershell
   .venv\Scripts\python -m pytest
   ```
   **Result**: `258 passed in 5.16s` (0 failed, 0 skipped).
2. **Frozen Benchmark SHA-256 Hashes**:
   - `benchmark_train.jsonl`: `ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014` (**Match: True**)
   - `benchmark_validation.jsonl`: `5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143` (**Match: True**)
   - `benchmark_test.jsonl`: `ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b` (**Match: True**)
   - `benchmark_stress.jsonl`: `36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747` (**Match: True**)

---

## 7. Changed Files and Artifact Paths

- **Source Code**:
  - [runners/inference_policy.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference_policy.py): Added `PROMPT_POLICY_V1_CONCISE`, instructions mapping, and `format_prompt_with_policy`.
  - [runners/inference.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference.py): Added `prompt_policy_version` to `GenerationRequest`, `execute_prompt`, and `ModelOutcome`.
  - [runners/outcome_schema.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/outcome_schema.py): Added `prompt_policy_version` to `ModelOutcome` and `RunManifest`.
  - [runners/smoke_runner.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/smoke_runner.py): Added `--prompt-policy` CLI flag and propagated policy metadata.
  - [eval/scorers/multiple_choice.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/eval/scorers/multiple_choice.py): Hardened choice extraction against false positives; prioritized explicit final answer statements.
- **Tests**:
  - [tests/test_inference_infrastructure.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/tests/test_inference_infrastructure.py): Added prompt policy formatting and propagation tests.
  - [tests/test_scoring.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/tests/test_scoring.py): Added regression tests for explicit markers, boxed answers, and false positive prevention.
- **Artifacts**:
  - `data/outcomes/smoke_phase4i/budget_256/`
  - `data/outcomes/smoke_phase4i/budget_512/`
  - `data/outcomes/smoke_phase4i/` (primary 512 run manifest, outcomes, and scores)

---

## 8. Decision: Candidate Outcome Matrix Protocol Finalization

### **DECISION: PROTOCOL READY FOR MATRIX EXECUTION**

With prompt policy `v1_concise` and a 512-token multiple-choice budget:
1. `qwen3:1.7b` operates with high throughput (1.1s warm latency, ~170 tokens, 100% completion).
2. `qwen3:8b` operates reliably with zero timeouts, zero truncations, 100% accuracy, and ~26–48s latency.
3. For `qwen3:4b`, the evaluation matrix will accurately capture its real-world Ollama behavior (truncations at 512 budget) without corrupting scores (`SCORING_UNAVAILABLE`, `quality_score = None`), faithfully reflecting its unsuitability as a low-budget tier under this runtime.
4. Matrix execution protocol: Execute training split (1,400 prompts) model-by-model sequentially with checkpointing.
