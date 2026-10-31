import pytest
from router.benchmark_schema import BenchmarkPrompt
from router.config import ModelConfig
from eval.scorer import OutcomeScorer
from eval.scorers.multiple_choice import extract_mc_choice, score_multiple_choice
from eval.scorers.exact_match import extract_final_answer, score_exact_match
from runners.inference_policy import (
    PROMPT_POLICY_TASK_AWARE,
    PROMPT_POLICY_TASK_AWARE_CANDIDATE,
    PROMPT_POLICY_TASK_AWARE_CONSTRAINED,
    PROMPT_POLICY_V0,
    PROMPT_POLICY_V1_CONCISE,
    PROMPT_POLICY_V2_ANSWER_ONLY,
    PROMPT_POLICY_V4_STRUCTURED_OUTPUT,
    MULTIPLE_CHOICE_JSON_SCHEMA,
    BOOLEAN_CLASSIFICATION_JSON_SCHEMA,
    UnsupportedTaskPolicyError,
    dispatch_task_policy,
    validate_structured_output_json,
)
from runners.outcome_schema import ModelOutcome
from runners.vllm_runner import MockInferenceProvider


# ---------------------------------------------------------------------------
# 1. Dispatch across all 8 canonical benchmark categories
# ---------------------------------------------------------------------------

def test_task_aware_dispatch_qa_multiple_choice():
    p = dispatch_task_policy("What is 1+1?\nA. 1\nB. 2\nC. 3\nD. 4", task_type="qa", evaluation_type="multiple_choice")
    assert p.policy_version == PROMPT_POLICY_V4_STRUCTURED_OUTPUT
    assert p.output_budget == 64
    assert p.thinking_mode is False
    assert p.response_schema == MULTIPLE_CHOICE_JSON_SCHEMA
    assert p.compatible_scorer == "multiple_choice_scorer"
    assert "JSON object" in p.effective_prompt


def test_task_aware_dispatch_reasoning_multiple_choice():
    p = dispatch_task_policy("Premise A. Premise B. What follows?", task_type="reasoning", evaluation_type="multiple_choice")
    assert p.policy_version == PROMPT_POLICY_V4_STRUCTURED_OUTPUT
    assert p.output_budget == 64
    assert p.response_schema == MULTIPLE_CHOICE_JSON_SCHEMA
    assert p.compatible_scorer == "multiple_choice_scorer"


def test_task_aware_dispatch_reasoning_exact_match():
    p = dispatch_task_policy("Evaluate True and False", task_type="reasoning", evaluation_type="exact_match")
    assert p.policy_version == PROMPT_POLICY_V1_CONCISE
    assert p.output_budget == 256
    assert p.response_schema is None
    assert p.compatible_scorer == "exact_match_scorer"
    assert "Final Answer: <answer>" in p.effective_prompt


def test_task_aware_dispatch_math_exact_match():
    p = dispatch_task_policy("Solve for x: 3x = 12", task_type="math", evaluation_type="exact_match")
    assert p.policy_version == PROMPT_POLICY_V1_CONCISE
    assert p.output_budget == 512
    assert p.response_schema is None
    assert p.compatible_scorer == "exact_match_scorer"
    assert "Final Answer: <answer>" in p.effective_prompt


def test_task_aware_dispatch_coding_code_execution():
    p = dispatch_task_policy("Write a python function to add two numbers.", task_type="coding", evaluation_type="code_execution")
    assert p.policy_version == PROMPT_POLICY_V1_CONCISE
    assert p.output_budget == 512
    assert p.response_schema is None
    assert p.compatible_scorer == "code_execution_scorer"
    assert "markdown code block" in p.effective_prompt


def test_task_aware_dispatch_instruction_following_ifeval():
    raw_prompt = "Write a 300 word essay without using the letter 'e'."
    p = dispatch_task_policy(raw_prompt, task_type="instruction_following", evaluation_type="ifeval")
    assert p.policy_version == PROMPT_POLICY_V0
    assert p.output_budget == 1024
    assert p.response_schema is None
    assert p.compatible_scorer == "ifeval_scorer"
    # Verbatim benchmark prompt preserved
    assert p.effective_prompt == raw_prompt


def test_task_aware_dispatch_summarization_reference_based():
    p = dispatch_task_policy("Long news article text...", task_type="summarization", evaluation_type="reference_based")
    assert p.policy_version == PROMPT_POLICY_V1_CONCISE
    assert p.output_budget == 256
    assert p.response_schema is None
    assert p.compatible_scorer == "reference_based_scorer"
    assert "Summarize the article concisely in 2-3 sentences" in p.effective_prompt


def test_task_aware_dispatch_classification_exact_match():
    p = dispatch_task_policy("Passage: ... Question: Was this event real?", task_type="classification", evaluation_type="exact_match")
    assert p.policy_version == PROMPT_POLICY_V2_ANSWER_ONLY
    assert p.output_budget == 64
    assert p.response_schema is None
    assert p.compatible_scorer == "exact_match_scorer"
    assert "true or false" in p.effective_prompt


def test_task_aware_dispatch_other_multiple_choice():
    p = dispatch_task_policy("A woman walks into a room. She...", task_type="other", evaluation_type="multiple_choice")
    assert p.policy_version == PROMPT_POLICY_V4_STRUCTURED_OUTPUT
    assert p.output_budget == 64
    assert p.response_schema == MULTIPLE_CHOICE_JSON_SCHEMA
    assert p.compatible_scorer == "multiple_choice_scorer"


# ---------------------------------------------------------------------------
# 2. Unsupported task-policy combinations fail with UnsupportedTaskPolicyError
# ---------------------------------------------------------------------------

def test_unsupported_policy_combinations_raise():
    # Structured output schema cannot be applied to coding
    with pytest.raises(UnsupportedTaskPolicyError, match="not supported for generative task"):
        dispatch_task_policy("def add(a, b):", "coding", "code_execution", requested_policy=PROMPT_POLICY_V4_STRUCTURED_OUTPUT)

    # Structured output schema cannot be applied to summarization
    with pytest.raises(UnsupportedTaskPolicyError, match="not supported for generative task"):
        dispatch_task_policy("Article...", "summarization", "reference_based", requested_policy=PROMPT_POLICY_V4_STRUCTURED_OUTPUT)

    # Structured output schema cannot be applied to ifeval
    with pytest.raises(UnsupportedTaskPolicyError, match="not supported for generative task"):
        dispatch_task_policy("Write text...", "instruction_following", "ifeval", requested_policy=PROMPT_POLICY_V4_STRUCTURED_OUTPUT)

    # Answer-only cannot be applied to code synthesis
    with pytest.raises(UnsupportedTaskPolicyError, match="not supported for generative task"):
        dispatch_task_policy("Code...", "coding", "code_execution", requested_policy=PROMPT_POLICY_V2_ANSWER_ONLY)

    # Unknown policy
    with pytest.raises(UnsupportedTaskPolicyError, match="Unknown prompt policy 'v99'"):
        dispatch_task_policy("Prompt", "qa", "multiple_choice", requested_policy="v99")

    # Unknown evaluation type
    with pytest.raises(UnsupportedTaskPolicyError, match="Unsupported evaluation_type 'imaginary'"):
        dispatch_task_policy("Prompt", "qa", "imaginary", requested_policy=PROMPT_POLICY_TASK_AWARE)


# ---------------------------------------------------------------------------
# 3. Truncation and Failed Generation vs Wrong Answer Distinction
# ---------------------------------------------------------------------------

def test_truncated_generation_is_never_scored_as_completed():
    scorer = OutcomeScorer()
    outcome = ModelOutcome(
        run_id="run-1",
        prompt_id="prompt-1",
        model_id="qwen3-4b",
        benchmark_split="train",
        task_type="qa",
        domain="abstract_algebra",
        difficulty="medium",
        source_dataset="mmlu",
        source_split="val",
        source_id="0",
        request_timestamp="2026-10-21T10:00:00Z",
        max_output_tokens=768,
        temperature=0.0,
        generated_text="In group theory, we know that the order of...",
        status="TRUNCATED",
        finish_reason="length",
        output_tokens=768,
        latency_ms=25000.0,
        backend="ollama",
        model_name="qwen3:4b",
    )
    score = scorer.score_outcome(outcome, evaluation_type="multiple_choice", expected_output="A")
    assert score.quality_score is None
    assert score.scoring_status == "SCORING_UNAVAILABLE"
    assert score.details.get("inference_status") == "TRUNCATED"


def test_schema_invalid_generation_is_distinguishable():
    scorer = OutcomeScorer()
    outcome = ModelOutcome(
        run_id="run-1",
        prompt_id="prompt-1",
        model_id="qwen3-4b",
        benchmark_split="train",
        task_type="qa",
        domain="abstract_algebra",
        difficulty="medium",
        source_dataset="mmlu",
        source_split="val",
        source_id="0",
        request_timestamp="2026-10-21T10:00:00Z",
        max_output_tokens=64,
        temperature=0.0,
        generated_text="{'answer': 'Z'}",  # enum violation
        status="INVALID_OUTPUT",
        error_type="SchemaValidationError",
        error_message="Property 'answer' value 'Z' not in allowed enum ['A', 'B', 'C', 'D']",
        latency_ms=500.0,
        backend="ollama",
        model_name="qwen3:4b",
    )
    score = scorer.score_outcome(outcome, evaluation_type="multiple_choice", expected_output="A")
    # Must be None, NOT zero
    assert score.quality_score is None
    assert score.scoring_status == "SCORING_UNAVAILABLE"


def test_wrong_answer_vs_failed_generation():
    scorer = OutcomeScorer()
    # A wrong answer that successfully generated is quality 0.0
    wrong_outcome = ModelOutcome(
        run_id="run-1",
        prompt_id="prompt-1",
        model_id="qwen3-1.7b",
        benchmark_split="train",
        task_type="qa",
        domain="abstract_algebra",
        difficulty="medium",
        source_dataset="mmlu",
        source_split="val",
        source_id="0",
        request_timestamp="2026-10-21T10:00:00Z",
        max_output_tokens=64,
        temperature=0.0,
        generated_text='{"answer": "C"}',
        status="SUCCESS",
        output_tokens=8,
        latency_ms=300.0,
        backend="ollama",
        model_name="qwen3:1.7b",
    )
    score_wrong = scorer.score_outcome(wrong_outcome, evaluation_type="multiple_choice", expected_output="A")
    assert score_wrong.quality_score == 0.0
    assert score_wrong.scoring_status == "SUCCESS"

    # A failed / timeout inference is None
    timeout_outcome = wrong_outcome.model_copy(update={"status": "TIMEOUT", "generated_text": ""})
    score_timeout = scorer.score_outcome(timeout_outcome, evaluation_type="multiple_choice", expected_output="A")
    assert score_timeout.quality_score is None
    assert score_timeout.scoring_status == "SCORING_UNAVAILABLE"


# ---------------------------------------------------------------------------
# 4. Complete answer followed by extra text
# ---------------------------------------------------------------------------

def test_complete_answer_followed_by_extra_text_mc():
    # Leading choice followed by punctuation and explanation
    assert extract_mc_choice("B. This is because the group order divides 24.") == "B"
    assert extract_mc_choice("(C) is the obvious answer.") == "C"
    assert extract_mc_choice("Option D is correct.\nProof follows below:") == "D"
    assert extract_mc_choice("Here is the solution.\nFinal Answer: A\nHope this helps!") == "A"

    score_res = score_multiple_choice("B. Explanation goes here.", "B")
    assert score_res["quality_score"] == 1.0


def test_complete_answer_followed_by_extra_text_exact_match():
    assert extract_final_answer("We calculate the total to be 42.\n\nFinal Answer: 42\nGood luck!") == "42"
    assert extract_final_answer("The answer is: True") == "True"

    res = score_exact_match("Step 1: calculate 5+7=12.\nFinal Answer: 12\nDone.", "12")
    assert res["quality_score"] == 1.0


# ---------------------------------------------------------------------------
# 5. Candidate policy dispatch and targeted budget adjustments
# ---------------------------------------------------------------------------

def test_task_aware_candidate_dispatch_classification_budget():
    p = dispatch_task_policy(
        "Passage: ... Question: Is this true?",
        task_type="classification",
        evaluation_type="exact_match",
        requested_policy=PROMPT_POLICY_TASK_AWARE_CANDIDATE,
    )
    assert p.policy_version == PROMPT_POLICY_V2_ANSWER_ONLY
    assert p.output_budget == 128
    assert p.thinking_mode is False
    assert p.response_schema is None
    assert p.compatible_scorer == "exact_match_scorer"


def test_task_aware_candidate_dispatch_summarization_budget():
    p = dispatch_task_policy(
        "Long news article text...",
        task_type="summarization",
        evaluation_type="reference_based",
        requested_policy=PROMPT_POLICY_TASK_AWARE_CANDIDATE,
    )
    assert p.policy_version == PROMPT_POLICY_V1_CONCISE
    assert p.output_budget == 384
    assert p.thinking_mode is False
    assert p.response_schema is None
    assert p.compatible_scorer == "reference_based_scorer"


def test_task_aware_candidate_dispatch_preserves_other_tasks():
    # MC: 64
    p_mc = dispatch_task_policy("Question\nA. 1\nB. 2", task_type="qa", evaluation_type="multiple_choice", requested_policy=PROMPT_POLICY_TASK_AWARE_CANDIDATE)
    assert p_mc.output_budget == 64
    assert p_mc.policy_version == PROMPT_POLICY_V4_STRUCTURED_OUTPUT

    # Reasoning: 256
    p_reas = dispatch_task_policy("Evaluate True and False", task_type="reasoning", evaluation_type="exact_match", requested_policy=PROMPT_POLICY_TASK_AWARE_CANDIDATE)
    assert p_reas.output_budget == 256

    # Coding: 512
    p_code = dispatch_task_policy("Write code", task_type="coding", evaluation_type="code_execution", requested_policy=PROMPT_POLICY_TASK_AWARE_CANDIDATE)
    assert p_code.output_budget == 512

    # Math: 512
    p_math = dispatch_task_policy("Solve math", task_type="math", evaluation_type="exact_match", requested_policy=PROMPT_POLICY_TASK_AWARE_CANDIDATE)
    assert p_math.output_budget == 512

    # IFEval: 1024
    p_if = dispatch_task_policy("Write essay", task_type="instruction_following", evaluation_type="ifeval", requested_policy=PROMPT_POLICY_TASK_AWARE_CANDIDATE)
    assert p_if.output_budget == 1024


def test_task_aware_custom_budgets_override():
    p = dispatch_task_policy(
        "Passage: ... Question: Is this true?",
        task_type="classification",
        evaluation_type="exact_match",
        requested_policy=PROMPT_POLICY_TASK_AWARE,
        custom_budgets={"classification": 200},
    )
    assert p.output_budget == 200


# ---------------------------------------------------------------------------
# 6. Constrained output diagnostic tests (Phase 4S)
# ---------------------------------------------------------------------------

def test_task_aware_constrained_dispatch_classification():
    p = dispatch_task_policy(
        "Passage: ... Question: Is this true?",
        task_type="classification",
        evaluation_type="exact_match",
        requested_policy=PROMPT_POLICY_TASK_AWARE_CONSTRAINED,
    )
    assert p.policy_version == PROMPT_POLICY_TASK_AWARE_CONSTRAINED
    assert p.output_budget == 64
    assert p.thinking_mode is False
    assert p.response_schema == BOOLEAN_CLASSIFICATION_JSON_SCHEMA
    assert p.compatible_scorer == "exact_match_scorer"
    assert 'JSON object with key \'label\' whose value is \'true\' or \'false\'' in p.effective_prompt


def test_task_aware_constrained_dispatch_summarization():
    p = dispatch_task_policy(
        "Long news article text...",
        task_type="summarization",
        evaluation_type="reference_based",
        requested_policy=PROMPT_POLICY_TASK_AWARE_CONSTRAINED,
    )
    assert p.policy_version == PROMPT_POLICY_TASK_AWARE_CONSTRAINED
    assert p.output_budget == 256
    assert p.thinking_mode is False
    assert p.response_schema is None
    assert p.compatible_scorer == "reference_based_scorer"
    assert "Provide only the final 2-3 sentence summary" in p.effective_prompt
    assert "no reasoning, no planning" in p.effective_prompt


def test_score_exact_match_with_json_label():
    res_true = score_exact_match('{"label": "true"}', "True")
    assert res_true["quality_score"] == 1.0
    assert res_true["normalized_exact_match"] is True

    res_false = score_exact_match('{"label": "false"}', "False")
    assert res_false["quality_score"] == 1.0
    assert res_false["normalized_exact_match"] is True

    res_mismatch = score_exact_match('{"label": "false"}', "True")
    assert res_mismatch["quality_score"] == 0.0
    assert res_mismatch["normalized_exact_match"] is False

    # Boolean literal in JSON
    res_bool = score_exact_match('{"label": true}', "True")
    assert res_bool["quality_score"] == 1.0
    assert res_bool["normalized_exact_match"] is True


def test_score_exact_match_with_malformed_json_fails():
    res_malformed = score_exact_match('{"label":', "True")
    assert res_malformed["quality_score"] == 0.0

    res_wrong_key = score_exact_match('{"other": "true"}', "True")
    assert res_wrong_key["quality_score"] == 0.0


