# Phase 4M: Ollama Structured-Output Diagnostic

**Date**: 2026-10-21  
**Hardware Profile**: Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB dedicated VRAM), 16 GB DDR5 RAM  
**Inference Backend**: Local Ollama v0.40.3 (`http://127.0.0.1:11434`)  
**Run ID**: `smoke-run-20261021-061712`  
**Output Directory**: `data/outcomes/smoke_phase4m/structured_output_64/`  
**Investigator**: Antigravity  

---

## 1. Executive Summary & Go/No-Go Decision

### **DECISION: NO-GO FOR MATRIX LAUNCH**

Ollama's native JSON schema grammar constraint (`format: schema`) completely and conclusively resolves the model truncation and runaway internal monologue issue:
- **100% Completion (15/15)** across all three models with **0% Truncation (0/15)**.
- **100% Schema Validity (15/15)**: Every generation parsed as valid JSON adhering strictly to `{"answer": <A|B|C|D>}`.
- **Extreme Throughput Improvement**: Warm latencies dropped to **~170 ms for 1.7B**, **~540 ms for 4B**, and **~1.1s for 8B**.
- **100% Warm SLA Compliance**: On steady-state warm queries, 100% of attempts on all three models completed in well under 10,000 ms (and 14/15 observed overall including cold starts).

### Why the Decision Remains NO-GO:
Despite solving the formatting and truncation blocker, **depriving the higher-tier models of reasoning scratchpad capacity sharply degrades their task accuracy on complex college-level mathematics**:
- Under 512-token reasoning (`v1_concise`), `qwen3:8b` scored **100.0% (5/5)**.
- Under zero-shot structured output (`v4_structured_output`), `qwen3:8b` accuracy dropped to **40.0% (2/5)**, and `qwen3:4b` scored **40.0% (2/5)**.
- Before committing 4,200 inferences to the full training matrix, the project must establish a task-aware policy architecture: structured output is ideal for classification, extraction, and direct multiple choice, but complex reasoning tasks (MMLU abstract algebra, GSM8K, MATH) require explicit scratchpad budgets. In addition, the checkpointing/resumption matrix runner infrastructure from Phase 4J remains a mandatory engineering prerequisite.

---

## 2. Exact API & Schema Configuration

### A. JSON Schema Definition
Implemented in [runners/inference_policy.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference_policy.py):

```python
MULTIPLE_CHOICE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}
```

### B. Native Ollama Request Payload
Passed directly to Ollama's native `/api/chat` endpoint:

```json
{
  "model": "<model_tag>",
  "messages": [
    {
      "role": "user",
      "content": "<prompt_text>\n\nInstruction: Select the single best answer. Return a JSON object with key 'answer' whose value is the chosen option letter (A, B, C, or D)."
    }
  ],
  "format": {
    "type": "object",
    "properties": {
      "answer": {
        "type": "string",
        "enum": ["A", "B", "C", "D"]
      }
    },
    "required": ["answer"],
    "additionalProperties": false
  },
  "think": false,
  "options": {
    "num_predict": 64,
    "temperature": 0.0,
    "seed": 42
  },
  "stream": false
}
```

### C. Execution Parameters
- **Policy Version**: `v4_structured_output`
- **Output Token Budget**: 64 tokens
- **Thinking Mode**: `think: False`
- **Target Prompts**: The same five verified training prompts (`val_0000` to `val_0004`) from `benchmark_train.jsonl`
- **Candidate Models**: `qwen3:1.7b`, `qwen3:4b`, `qwen3:8b` (executed sequentially)

---

## 3. Results for All 15 Attempts

| Model ID | Prompt ID | Status | Finish Reason | Tokens (In/Out) | Latency (ms) | JSON Valid? | Schema Valid? | Raw Output | Extracted | Expected | Quality Score | Strict SLA (<10s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `qwen3-1.7b` | `mmlu-...-val_0000` | SUCCESS | stop | 87 / 7 | 303.76 | Yes | Yes | `{"answer": "A"}` | **A** | A | **1.0** | **PASS** |
| `qwen3-1.7b` | `mmlu-...-val_0001` | SUCCESS | stop | 84 / 7 | 175.55 | Yes | Yes | `{"answer": "A"}` | **A** | B | **0.0** | **PASS** |
| `qwen3-1.7b` | `mmlu-...-val_0002` | SUCCESS | stop | 119 / 7 | 156.73 | Yes | Yes | `{"answer": "C"}` | **C** | A | **0.0** | **PASS** |
| `qwen3-1.7b` | `mmlu-...-val_0003` | SUCCESS | stop | 95 / 7 | 192.87 | Yes | Yes | `{"answer": "C"}` | **C** | C | **1.0** | **PASS** |
| `qwen3-1.7b` | `mmlu-...-val_0004` | SUCCESS | stop | 89 / 7 | 165.28 | Yes | Yes | `{"answer": "C"}` | **C** | C | **1.0** | **PASS** |
| `qwen3-4b` | `mmlu-...-val_0000` | SUCCESS | stop | 81 / 11 | 9451.19 | Yes | Yes | `{\n  "answer": "B"\n}` | **B** | A | **0.0** | **PASS** |
| `qwen3-4b` | `mmlu-...-val_0001` | SUCCESS | stop | 78 / 11 | 579.55 | Yes | Yes | `{\n  "answer": "B"\n}` | **B** | B | **1.0** | **PASS** |
| `qwen3-4b` | `mmlu-...-val_0002` | SUCCESS | stop | 113 / 9 | 496.90 | Yes | Yes | `{ "answer": "C" }` | **C** | A | **0.0** | **PASS** |
| `qwen3-4b` | `mmlu-...-val_0003` | SUCCESS | stop | 89 / 11 | 579.36 | Yes | Yes | `{\n  "answer": "A"\n}` | **A** | C | **0.0** | **PASS** |
| `qwen3-4b` | `mmlu-...-val_0004` | SUCCESS | stop | 83 / 9 | 503.04 | Yes | Yes | `{ "answer": "C" }` | **C** | C | **1.0** | **PASS** |
| `qwen3-8b` | `mmlu-...-val_0000` | SUCCESS | stop | 87 / 9 | 13368.62 | Yes | Yes | `{ "answer": "D" }` | **D** | A | **0.0** | **FAIL** |
| `qwen3-8b` | `mmlu-...-val_0001` | SUCCESS | stop | 84 / 9 | 1111.91 | Yes | Yes | `{ "answer": "A" }` | **A** | B | **0.0** | **PASS** |
| `qwen3-8b` | `mmlu-...-val_0002` | SUCCESS | stop | 119 / 9 | 1111.51 | Yes | Yes | `{ "answer": "D" }` | **D** | A | **0.0** | **PASS** |
| `qwen3-8b` | `mmlu-...-val_0003` | SUCCESS | stop | 95 / 9 | 1100.22 | Yes | Yes | `{ "answer": "C" }` | **C** | C | **1.0** | **PASS** |
| `qwen3-8b` | `mmlu-...-val_0004` | SUCCESS | stop | 89 / 9 | 1089.31 | Yes | Yes | `{ "answer": "C" }` | **C** | C | **1.0** | **PASS** |

---

## 4. Per-Model Summary Performance Metrics

| Model ID | Completion Rate | Truncation Rate | Schema Validity | Mean Output Tokens | Mean Warm Latency | Strict SLA Rate (Observed) | Strict SLA Rate (Warm) | Quality Score (Accuracy) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen3:1.7b`** | **100% (5/5)** | **0% (0/5)** | **100% (5/5)** | 7.0 tokens | **172.6 ms** | **100% (5/5)** | **100% (4/4)** | **60.0% (3/5)** |
| **`qwen3:4b`** | **100% (5/5)** | **0% (0/5)** | **100% (5/5)** | 10.2 tokens | **539.7 ms** | **100% (5/5)** | **100% (4/4)** | **40.0% (2/5)** |
| **`qwen3:8b`** | **100% (5/5)** | **0% (0/5)** | **100% (5/5)** | 9.0 tokens | **1103.2 ms** | **80% (4/5)** | **100% (4/4)** | **40.0% (2/5)** |

---

## 5. Cross-Policy Comparison (Phase 4I, 4K, 4L, and 4M)

| Dimension | Model | Phase 4I (`v1_concise`, 512) | Phase 4K (`v2_answer_only`, 64) | Phase 4L (`v3_few_shot`, 512) | Phase 4M (`v4_structured_output`, 64) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Completion Rate** | `qwen3:1.7b`<br>`qwen3:4b`<br>`qwen3:8b` | **100% (5/5)**<br>0% (0/5)<br>**100% (5/5)** | 0% (0/5)<br>0% (0/5)<br>**80% (4/5)** | **80% (4/5)**<br>0% (0/5)<br>**100% (5/5)** | **100% (5/5)**<br>**100% (5/5)**<br>**100% (5/5)** |
| **Truncation Rate** | `qwen3:1.7b`<br>`qwen3:4b`<br>`qwen3:8b` | 0% (0/5)<br>**100% (5/5)**<br>0% (0/5) | **100% (5/5)**<br>**100% (5/5)**<br>20% (1/5) | 20% (1/5)<br>**100% (5/5)**<br>0% (0/5) | **0% (0/5)**<br>**0% (0/5)**<br>**0% (0/5)** |
| **Accuracy (Scorable)** | `qwen3:1.7b`<br>`qwen3:4b`<br>`qwen3:8b` | 60.0% (3/5)<br>N/A<br>**100.0% (5/5)** | N/A<br>N/A<br>**50.0% (2/4)** | 75.0% (3/4)<br>N/A<br>**100.0% (5/5)** | 60.0% (3/5)<br>40.0% (2/5)<br>40.0% (2/5) |
| **Warm Latency** | `qwen3:1.7b`<br>`qwen3:4b`<br>`qwen3:8b` | 1.1s – 1.6s<br>16.5s – 16.8s<br>26.3s – 48.2s | ~0.61s – 0.64s<br>~1.96s – 2.00s<br>~0.71s – 0.89s | 2.29s – 4.46s<br>15.3s – 15.5s<br>21.7s – 43.5s | **~157ms – 193ms**<br>**~497ms – 580ms**<br>**~1.09s – 1.11s** |
| **Strict SLA (<10s)** | `qwen3:1.7b`<br>`qwen3:4b`<br>`qwen3:8b` | **80% (4/5)**<br>0% (0/5)<br>0% (0/5) | **100% (5/5)**<br>**80% (4/5)**<br>**60% (3/5)** | **100% (5/5)**<br>0% (0/5)<br>0% (0/5) | **100% (5/5)**<br>**100% (5/5)**<br>**80% (4/5)** *(100% warm)* |

### Critical Diagnostic Findings:
1. **Grammar Constraints Eliminate Monologue**: The Ollama native `format: schema` completely blocks Qwen3-4B's thinking monologue. Because the token sampler is restricted to JSON tokens conforming to the grammar, 4B produces clean `{"answer": "X"}` outputs in ~540 ms.
2. **Direct Output Collapses Hard Math Accuracy**: In Phase 4I, `qwen3:8b` solved all 5 questions correctly (100%) by generating step-by-step mathematical proofs. In Phase 4M, forcing zero-shot JSON output forced 8B to guess without scratchpad, dropping accuracy to **40%**.
3. **The Core Cost-Quality-Latency Frontier**:
   - `v4_structured_output` represents the **Ultra-Fast Edge Tier**: sub-second latency, zero truncation, high reliability, but limited to intuitive / non-deliberative accuracy.
   - `v1_concise` represents the **Deliberation Tier**: 20–45s latency, 100% accuracy, but requires high token budgets and offloaded compute.

---

## 6. Limitations of the Small Sample

The five-prompt training diagnostic sample serves as an operational testbed for grammar adherence and latency limits. It confirms that Ollama's schema enforcement functions deterministically without crashes or truncations across all three installed models. However, evaluating broad model capability across all 10 benchmark datasets (spanning coding, classification, QA, and reasoning) requires the full evaluation pipeline.

---

## 7. Smallest Justified Next Step

1. **Policy Architecture Decision**:
   - Establish task-specific routing policies:
     - Use `v4_structured_output` for tasks that do not require multi-step calculation (e.g. BoolQ, SuperGLUE classification, sentiment, fact retrieval).
     - Use a bounded reasoning policy (`v1_concise` with `stop: ["</think>"]` or thinking budgets) for multi-step reasoning (e.g. GSM8K, MATH, MMLU abstract algebra).
2. **Matrix Runner Implementation**:
   - Before launching 4,200 inferences, implement `runners/matrix_runner.py` with:
     - Per-batch Parquet checkpointing.
     - Uniqueness key deduplication on restart.
     - Split isolation guarantees.

---

## 8. Verification & Checksum Invariance

### A. Pytest Suite Execution
```powershell
.venv\Scripts\python -m pytest
```
- **Result**: `260 passed in 5.56s` (0 failed, 0 errors, 0 skipped).

### B. Frozen Benchmark SHA-256 Hashes
- `benchmark_train.jsonl`: `ff956efaef264ce3ba9cd71992cb6a6de27de2484b4cd49f82619e2a4f4be014` (**MATCH: TRUE**)
- `benchmark_validation.jsonl`: `5e6849452b6a22e69a4bc095fa2dff85820a7b608c86eb20a573eca644c1d143` (**MATCH: TRUE**)
- `benchmark_test.jsonl`: `ec3ab60f16838e514733d8c76448dd3cfc0d5b59d90b997675a4d6971f09eb1b` (**MATCH: TRUE**)
- `benchmark_stress.jsonl`: `36a092c32deeccdbc64b57cf969f839bff89c5201efcb4be51c07f66b8fe5747` (**MATCH: TRUE**)

### C. Changed Files
- [runners/inference_policy.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference_policy.py): Added `PROMPT_POLICY_V4_STRUCTURED_OUTPUT`, `MULTIPLE_CHOICE_JSON_SCHEMA`, `resolve_response_format`, and `validate_structured_output_json`.
- [runners/inference.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/inference.py): Added `response_format` to `GenerationRequest` and `execute_prompt`.
- [runners/ollama_runner.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/ollama_runner.py): Propagated `response_format` to native `/api/chat` payload `format`.
- [runners/outcome_schema.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/outcome_schema.py): Added `response_schema`, `output_budget`, and `effective_inference_settings` to `RunManifest`.
- [runners/smoke_runner.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/runners/smoke_runner.py): Propagated structured output metadata to `RunManifest`.
- [eval/scorers/multiple_choice.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/eval/scorers/multiple_choice.py): Added structured JSON choice extraction.
- [tests/test_inference_infrastructure.py](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/tests/test_inference_infrastructure.py): Added unit tests for schema validation, request construction, and JSON answer extraction.
- [reports/phase4m_structured_output_diagnostic.md](file:///c:/Users/vibhor/Desktop/cost-aware-llm-router/reports/phase4m_structured_output_diagnostic.md): Complete Phase 4M diagnostic deliverable.
