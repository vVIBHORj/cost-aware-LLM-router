# Scoring Architecture & Baseline Policy Design (v1)

**Project:** Cost-Aware LLM Model Router (`cost-aware-llm-router`)  
**Phase:** Phase 3 — Candidate Evaluation & Baseline Routing Infrastructure  
**Status:** Verification Completed  
**Reference Policies:** `reports/benchmark_split_policy_v1.md`, `configs/policies.yaml`

---

## 1. Architectural Overview & Separation of Concerns

The evaluation architecture maintains strict separation between raw candidate-model inference and quality evaluation:

```
Frozen Benchmark Prompts (2,300)
             ↓
Candidate Model Inference (runners/inference.py)
             ↓
ModelOutcome (Immutable Parquet in data/outcomes/)
             ↓
Scoring Dispatcher (eval/scorer.py)
             ↓
ScoreOutcome (Separate Parquet in data/outcomes/scored/)
```

1. **Immutability of Inference Telemetry**: `ModelOutcome` records are never overwritten, modified, or re-interpreted by scorers.
2. **Failure Isolation**: An inference failure (`BACKEND_UNAVAILABLE`, `FAILED`, `TIMEOUT`, `RETRY_EXHAUSTED`) is strictly quarantined. It **never** becomes a quality observation (e.g. It is never marked as score = 0.0 or failure-at-target = False). It receives `scoring_status="SCORING_UNAVAILABLE"` and `quality_score=None`.
3. **Decoupled Evaluation**: Scorers evaluate predictions purely against ground truth specifications without knowledge of inference engines, network parameters, or hardware backends.

---

## 2. Canonical Scoring Schema (`ScoreOutcome`)

The scoring output is validated via Pydantic in `eval/scoring_schema.py` and stored in flat, queryable Parquet format (`data/outcomes/scored/`):

- **Identity**: `run_id`, `prompt_id`, `model_id`, `benchmark_split`, `task_type`, `domain`, `difficulty`.
- **Linkage**: `canonical_key` (`prompt_id:model_id:run_id`), `evaluation_type`, `expected_output_available`.
- **Quality Metrics**:
  - `quality_score`: Normalized continuous or binary score in `[0.0, 1.0]`, or `None` if unavailable.
  - `success_at_target`: `True` / `False` if `quality_score` is present; `None` if quality is unavailable.
  - `target_quality`: Configured target threshold (default `0.80` from `configs/policies.yaml`).
  - `scorer_name` & `scorer_version`: Full provenance tracking.
  - `scoring_status`: `SUCCESS`, `SCORING_UNAVAILABLE`, `SCORING_ERROR`, `INVALID_INPUT`, `SANDBOX_UNAVAILABLE`, `NOT_IMPLEMENTED`.
- **Specialized Metric Slots**: `exact_match`, `normalized_exact_match`, `pass_at_1`, `judge_score`, `reference_score`, and `details` dictionary.
- **Provenance**: `scoring_timestamp`, `benchmark_manifest_checksum`, `source_dataset`, `source_split`, `source_id`.

---

## 3. Scorer Implementations & Normalization Rules

The benchmark features five evaluation types across 2,300 prompts:

| Evaluation Type | Benchmark Count | Scorer Module | Normalization & Evaluation Rules |
|---|---|---|---|
| `exact_match` | 735 | `eval/scorers/exact_match.py` | Strict match, plus normalized match: lowercased, whitespace collapsed, punctuation removed, LaTeX `\boxed{...}` content extracted, and float numeric equivalence. Quality score: `1.0` or `0.0`. |
| `multiple_choice` | 725 | `eval/scorers/multiple_choice.py` | Robust choice option extraction (e.g., `"The answer is (A)"`, `"Answer: B"`, `"(C)"`, standalone `"D"`). Compares extracted choice letter against ground truth. Quality score: `1.0` or `0.0`. |
| `code_execution` | 330 | `eval/scorers/code_execution.py` | **Safe Host Isolation Stub**: Arbitrary model-generated code is never executed directly on the host system. Returns structured `SANDBOX_UNAVAILABLE` until an isolated Docker/gVisor runner is connected. |
| `reference_based`| 260 | `eval/scorers/reference_based.py` | Deterministic unigram word overlap (ROUGE-1 F1) as primary `quality_score`, plus Longest Common Subsequence (ROUGE-L F1) as `reference_score`. |
| `ifeval` | 250 | `eval/scorers/ifeval_scorer.py` | Deterministic rule checker verifying instruction constraints. Returns `1.0` if all rules pass, `0.0` if violated, and `SCORING_UNAVAILABLE` if an external NLP library is required. |

---

## 4. IFEval Instruction-Following Status

- **Representation in Frozen Benchmark**: Each IFEval prompt contains complete serialized constraint metadata in `expected_output`:
  ```json
  {"instruction_id_list": ["length_constraints:number_paragraphs", "startend:end_checker"], "kwargs": [{"num_paragraphs": 5}, {"end_phrase": "..."}]}
  ```
- **Supported Deterministic Rules**: `punctuation:no_comma`, `change_case:english_lowercase`, `change_case:english_capital`, `startend:quotation`, `startend:start_checker`, `startend:end_checker`, `keywords:existence`, `keywords:forbidden_words`, `keywords:frequency`, `keywords:letter_frequency`, `length_constraints:number_words`, `length_constraints:number_paragraphs`, `detectable_content:postscript`.
- **Unsupported Rules & Fallback**: Rules requiring external multilingual NLP models (e.g. `language:response_language`) are safely classified with `scoring_status="SCORING_UNAVAILABLE"`, preserving the raw model response intact for future external evaluation rather than guessing or replacing with an LLM judge.

---

## 5. LLM-Judge Interface (`JudgeProvider`)

Defined in `eval/scorers/judge.py`:
- `JudgeProvider(ABC)`: Clean abstract interface with `evaluate(prompt, response, reference) -> JudgeResult`.
- `JudgeConfig`: Versioned configuration (`judge_model_id`, `rubric_name`, `temperature`, `max_tokens`).
- `JudgeResult`: Standardized score, explanation, rubric version.
- **Quarantine**: No judge model is invoked during candidate inference or unit testing. Zero external API dependencies exist.

---

## 6. Baseline Routing Policies

Implemented in `router/baselines.py`:

| Policy Name | Selection Mechanism | Determinism & Cost Semantics |
|---|---|---|
| `always-small` | Always selects `qwen3-1.7b` (role: cheap). | Deterministic static selection. |
| `always-middle` | Always selects `qwen3-4b` (role: middle). | Deterministic static selection. |
| `always-strong` | Always selects `qwen3-8b` (role: strong). | Deterministic static selection. |
| `random` | Uniform random across enabled candidate models ($p = 1/3$). | Deterministic under fixed seed: $\text{SHA256}(\text{prompt} + \text{seed}) \pmod N$. |
| `cost-weighted-random` | Biases selection inversely proportional to cost. | **Zero Monetary Cost Handling**: When local models have 0.0 USD price, monetary cost weighting ($0/0$) is degenerate. The policy explicitly uses operational parameter count proxy ($1.7\text{B}, 4.0\text{B}, 8.0\text{B}$), weighting selection $w_i \propto 1 / \text{params}$, preventing division-by-zero. |
| `heuristic` | A priori rules based strictly on request metadata. | Rule-based (see Section 7). |

---

## 7. Heuristic Routing Policy Rules

The `HeuristicPolicy` in `router/baselines.py` is defined entirely *a priori* without machine learning, embeddings, or exposure to benchmark outcomes:

1. **High Complexity Rule $\to$ `qwen3-8b`**:
   - `difficulty == "hard"` OR
   - `task_type in ("reasoning", "coding")` OR
   - prompt word count $> 800$ words.
2. **Medium Complexity Rule $\to$ `qwen3-4b`**:
   - `difficulty == "medium"` OR
   - `task_type in ("math", "instruction_following")`.
3. **Low Complexity Rule $\to$ `qwen3-1.7b`**:
   - `difficulty == "easy"` AND `task_type in ("qa", "classification", "summarization", "other")`.
4. **Default Fallback**: `qwen3-4b`.

---

## 8. Quality Target Semantics (`target_quality = 0.80`)

In `eval/scoring_schema.py`:
$$\text{success\_at\_target} = \begin{cases} 
\text{True} & \text{if } \text{quality\_score} \ge 0.80 \\
\text{False} & \text{if } \text{quality\_score} < 0.80 \\
\text{None (null)} & \text{if } \text{quality\_score is None (unavailable)}
\end{cases}$$

- **Critical Rule**: An unavailable quality observation is **never** coerced into `False` or `0.0`. Inference failure does not imply poor reasoning; it represents an unobserved quality state.

---

## 9. Blocked Operations & Next Steps

The following operations remain intentionally blocked until candidate-model inference is performed on supported compute hardware:
1. **Full 6,900 Inference Matrix**: Requires vLLM / CUDA execution environment.
2. **Quality Predictor Training**: Requires valid model outcome observations.
3. **Router Optimization & Threshold Calibration**: Requires scored training splits.
4. **Cascade / Fallback Policy Evaluation**: Requires multi-model candidate scores per prompt.
