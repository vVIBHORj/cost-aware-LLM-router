from dataclasses import dataclass
from typing import Any

# Default output-token budgets per evaluation type and task category.
# Conservative budgets prevent runaway generations while allowing complete answers.
DEFAULT_TASK_OUTPUT_BUDGETS: dict[str, int] = {
    "multiple_choice": 768,      # Multiple choice: derivations + option selection
    "classification": 128,       # Binary / label classification
    "exact_match": 512,          # Short-answer / math / QA exact match
    "code_execution": 768,       # Code generation (HumanEval / MBPP)
    "ifeval": 1024,              # Complex instruction following
    "summarization": 512,        # Summarization (CNN/DailyMail)
    "reference_based": 512,      # Reference-based evaluation
    "default": 512,              # General fallback default
}

# Thinking mode policy per evaluation type and task category.
# Defaults to False (thinking disabled) for short-answer, multiple choice,
# classification, and extraction tasks.
DEFAULT_THINKING_POLICY: dict[str, bool] = {
    "multiple_choice": False,
    "classification": False,
    "exact_match": False,
    "summarization": False,
    "code_execution": False,
    "ifeval": False,
    "default": False,
}


def resolve_output_budget(
    task_type: str | None = None,
    evaluation_type: str | None = None,
    user_override: int | None = None,
    custom_budgets: dict[str, int] | None = None,
) -> int:
    """
    Resolve the maximum output token budget for a given prompt task/evaluation type.
    Priority: user_override > custom_budgets > DEFAULT_TASK_OUTPUT_BUDGETS.
    """
    if user_override is not None and user_override > 0:
        return user_override

    budgets = {**DEFAULT_TASK_OUTPUT_BUDGETS, **(custom_budgets or {})}

    if task_type and task_type in budgets:
        return budgets[task_type]
    if evaluation_type and evaluation_type in budgets:
        return budgets[evaluation_type]

    return budgets["default"]


def resolve_thinking_mode(
    task_type: str | None = None,
    evaluation_type: str | None = None,
    user_override: bool | None = None,
    custom_policy: dict[str, bool] | None = None,
) -> bool:
    """
    Resolve whether thinking / reasoning mode should be enabled for a given prompt.
    Priority: user_override > custom_policy > DEFAULT_THINKING_POLICY.
    """
    if user_override is not None:
        return user_override

    policy = {**DEFAULT_THINKING_POLICY, **(custom_policy or {})}

    if task_type and task_type in policy:
        return policy[task_type]
    if evaluation_type and evaluation_type in policy:
        return policy[evaluation_type]

    return policy["default"]


PROMPT_POLICY_V0 = "v0"
PROMPT_POLICY_V1_CONCISE = "v1_concise"
PROMPT_POLICY_V2_ANSWER_ONLY = "v2_answer_only"
PROMPT_POLICY_V3_FEW_SHOT = "v3_few_shot"
PROMPT_POLICY_V4_STRUCTURED_OUTPUT = "v4_structured_output"
PROMPT_POLICY_TASK_AWARE = "task_aware"
PROMPT_POLICY_TASK_AWARE_CANDIDATE = "task_aware_candidate"
PROMPT_POLICY_TASK_AWARE_CONSTRAINED = "task_aware_constrained"
PROMPT_POLICY_PHASE4T = "phase4t_frozen"

SUPPORTED_PROMPT_POLICIES = {
    PROMPT_POLICY_V0,
    PROMPT_POLICY_V1_CONCISE,
    PROMPT_POLICY_V2_ANSWER_ONLY,
    PROMPT_POLICY_V3_FEW_SHOT,
    PROMPT_POLICY_V4_STRUCTURED_OUTPUT,
    PROMPT_POLICY_TASK_AWARE,
    PROMPT_POLICY_TASK_AWARE_CANDIDATE,
    PROMPT_POLICY_TASK_AWARE_CONSTRAINED,
    PROMPT_POLICY_PHASE4T,
}

COMPATIBLE_SCORERS: dict[str, tuple[str, str]] = {
    "multiple_choice": ("multiple_choice_scorer", "1.0.0"),
    "exact_match": ("exact_match_scorer", "1.0.0"),
    "classification": ("exact_match_scorer", "1.0.0"),
    "code_execution": ("code_execution_scorer", "1.0.0"),
    "reference_based": ("reference_based_scorer", "1.0.0"),
    "ifeval": ("ifeval_scorer", "1.0.0"),
    "llm_judge": ("llm_judge_interface", "1.0.0"),
}


class UnsupportedTaskPolicyError(ValueError):
    """Raised when an unsupported prompt policy or schema is requested for a task."""
    pass


@dataclass(frozen=True)
class DispatchedTaskPolicy:
    """Explicit resolved policy specification for a single prompt evaluation."""
    task_type: str
    evaluation_type: str
    policy_version: str
    effective_prompt: str
    output_budget: int
    thinking_mode: bool
    response_schema: dict[str, Any] | None
    compatible_scorer: str
    scorer_version: str
    description: str


MULTIPLE_CHOICE_JSON_SCHEMA: dict[str, Any] = {
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

BOOLEAN_CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["true", "false", "True", "False"],
        }
    },
    "required": ["label"],
    "additionalProperties": False,
}

GENERAL_CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
        }
    },
    "required": ["label"],
    "additionalProperties": False,
}

STRUCTURED_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "multiple_choice": MULTIPLE_CHOICE_JSON_SCHEMA,
    "exact_match": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
    "classification": BOOLEAN_CLASSIFICATION_JSON_SCHEMA,
}

DEFAULT_PROMPT_POLICY_INSTRUCTIONS: dict[str, dict[str, str]] = {
    PROMPT_POLICY_V0: {},
    PROMPT_POLICY_V1_CONCISE: {
        "multiple_choice": (
            "Solve the question. Keep the explanation concise and end with `Final Answer: <letter>`. "
            "Do not repeat the question or provide unnecessary commentary."
        ),
        "exact_match": (
            "Solve the question. Keep the explanation concise and end with `Final Answer: <answer>`. "
            "Do not repeat the question or provide unnecessary commentary."
        ),
        "classification": (
            "Classify the text. Keep the output concise and state only the final label as `Final Answer: <label>`."
        ),
        "summarization": (
            "Summarize the article concisely in 2-3 sentences. Do not include introductory preamble or conversational commentary."
        ),
        "coding": (
            "Write python code to solve the task. Output the code cleanly inside a markdown code block ```python ... ``` without conversational commentary."
        ),
    },
    PROMPT_POLICY_V2_ANSWER_ONLY: {
        "multiple_choice": (
            "Select the single best answer. Output only its option letter, with no explanation, reasoning, prefix, or other text."
        ),
        "exact_match": (
            "Output only the final answer, with no explanation, reasoning, units, prefix, or other text."
        ),
        "classification": (
            "Answer true or false based on the text. Output only the final label ('true' or 'false'), with no explanation, reasoning, prefix, or other text."
        ),
    },
    PROMPT_POLICY_V3_FEW_SHOT: {
        "multiple_choice": (
            "Select the single best answer. Output one option letter as the final answer."
        ),
        "exact_match": (
            "Output only the final answer."
        ),
        "classification": (
            "Output only the final label."
        ),
    },
    PROMPT_POLICY_V4_STRUCTURED_OUTPUT: {
        "multiple_choice": (
            "Select the single best answer. Return a JSON object with key 'answer' whose value is the chosen option letter (A, B, C, or D)."
        ),
        "exact_match": (
            "Solve the question. Return a JSON object with key 'answer' containing the final answer."
        ),
        "classification": (
            "Answer the question based on the text. Return a JSON object with key 'label' whose value is 'true' or 'false'."
        ),
    },
    PROMPT_POLICY_TASK_AWARE_CONSTRAINED: {
        "classification": (
            "Answer the question based on the text. Return a JSON object with key 'label' whose value is 'true' or 'false'."
        ),
        "summarization": (
            "Provide only the final 2-3 sentence summary of the article. Output strictly the summary text alone with no introductory preamble, no reasoning, no planning, and no conversational commentary."
        ),
    },
}

FEW_SHOT_DEMONSTRATIONS: dict[str, str] = {
    "multiple_choice": (
        "Question: Which of these is a fruit?\n"
        "A. Carrot\n"
        "B. Apple\n"
        "C. Potato\n"
        "D. Onion\n\n"
        "Correct answer: B"
    ),
    "exact_match": (
        "Question: What is 2 + 2?\n\n"
        "Correct answer: 4"
    ),
    "classification": (
        "Text: I loved this film, it was wonderful!\n\n"
        "Correct label: positive"
    ),
}


def resolve_response_format(
    policy_version: str | None = None,
    evaluation_type: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any] | str | None:
    """
    Resolve the structured output schema for a given prompt policy version and evaluation type.
    Returns JSON schema dict for v4_structured_output, or None for unconstrained policies.
    """
    if policy_version == PROMPT_POLICY_V4_STRUCTURED_OUTPUT:
        key = evaluation_type or task_type or "multiple_choice"
        return STRUCTURED_OUTPUT_SCHEMAS.get(key, MULTIPLE_CHOICE_JSON_SCHEMA)
    return None


def validate_structured_output_json(
    text: str,
    schema: dict[str, Any] | None = None,
) -> tuple[bool, Any, str | None]:
    """
    Validate that text parses as valid JSON and conforms to the expected schema.
    Returns (is_valid, parsed_data_or_none, error_message_or_none).
    """
    import json
    if not text or not text.strip():
        return False, None, "Output is empty"

    cleaned = text.strip()
    if "```json" in cleaned:
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()

    try:
        data = json.loads(cleaned)
    except Exception as e:
        return False, None, f"JSONDecodeError: {e}"

    if schema is not None:
        if schema.get("type") == "object":
            if not isinstance(data, dict):
                return False, data, f"Expected JSON object dict, got {type(data).__name__}"
            for req in schema.get("required", []):
                if req not in data:
                    return False, data, f"Missing required property: '{req}'"
            properties = schema.get("properties", {})
            for key, val in data.items():
                prop_spec = properties.get(key)
                if prop_spec and "enum" in prop_spec:
                    if val not in prop_spec["enum"]:
                        return False, data, f"Property '{key}' value '{val}' not in allowed enum {prop_spec['enum']}"
            if schema.get("additionalProperties") is False:
                extra = set(data.keys()) - set(properties.keys())
                if extra:
                    return False, data, f"Unexpected additional properties: {extra}"

    return True, data, None


def format_prompt_with_policy(
    prompt_text: str,
    evaluation_type: str | None = None,
    task_type: str | None = None,
    policy_version: str = PROMPT_POLICY_V0,
) -> str:
    """
    Format prompt text according to the specified versioned prompt policy.
    Leaves original prompt unaltered if policy is 'v0' or unconfigured.
    """
    if policy_version == PROMPT_POLICY_V0 or not policy_version:
        return prompt_text

    if policy_version == PROMPT_POLICY_V3_FEW_SHOT:
        key = evaluation_type or task_type or "multiple_choice"
        demo = FEW_SHOT_DEMONSTRATIONS.get(key, FEW_SHOT_DEMONSTRATIONS["multiple_choice"])
        instruction = DEFAULT_PROMPT_POLICY_INSTRUCTIONS[PROMPT_POLICY_V3_FEW_SHOT].get(
            key, "Select the single best answer. Output one option letter as the final answer."
        )
        return f"{demo}\n\nQuestion:\n{prompt_text}\n\nInstruction: {instruction}"

    policy_map = DEFAULT_PROMPT_POLICY_INSTRUCTIONS.get(policy_version, {})
    instruction = policy_map.get(task_type) or policy_map.get(evaluation_type)
    if not instruction:
        return prompt_text

    return f"{prompt_text}\n\nInstruction: {instruction}"


def dispatch_task_policy(
    prompt_text: str,
    task_type: str,
    evaluation_type: str,
    requested_policy: str = PROMPT_POLICY_TASK_AWARE,
    user_output_budget: int | None = None,
    user_think: bool | None = None,
    response_format: dict[str, Any] | str | None = None,
    custom_budgets: dict[str, int] | None = None,
    source_dataset: str | None = None,
) -> DispatchedTaskPolicy:
    """
    Select explicit inference settings based on task type and evaluation type.
    Guarantees:
    - Multiple-choice JSON schema is only applied to genuine multiple-choice tasks.
    - Generative tasks (coding, summarization, ifeval) are never forced into MC schema or answer-only mode.
    - Unsupported task-policy combinations raise UnsupportedTaskPolicyError before inference.
    - Task-specific budgets prevent runaway generation while ensuring complete answers.
    - Preserves original benchmark prompt and source data.
    """
    if requested_policy not in SUPPORTED_PROMPT_POLICIES:
        raise UnsupportedTaskPolicyError(
            f"Unknown prompt policy '{requested_policy}'. Supported: {sorted(SUPPORTED_PROMPT_POLICIES)}"
        )

    if evaluation_type not in COMPATIBLE_SCORERS:
        raise UnsupportedTaskPolicyError(
            f"Unsupported evaluation_type '{evaluation_type}' for task '{task_type}'."
        )

    scorer_name, scorer_version = COMPATIBLE_SCORERS[evaluation_type]

    # Guard: v4_structured_output is only supported for multiple_choice, exact_match, and classification
    if requested_policy == PROMPT_POLICY_V4_STRUCTURED_OUTPUT:
        if evaluation_type not in ("multiple_choice", "exact_match", "classification"):
            raise UnsupportedTaskPolicyError(
                f"Policy '{PROMPT_POLICY_V4_STRUCTURED_OUTPUT}' is not supported for generative task '{task_type}' "
                f"(evaluation_type: '{evaluation_type}'). Structured output schema is not supported for coding, summarization, or IFEval."
            )

    # Guard: v2_answer_only cannot be applied to generative tasks
    if requested_policy == PROMPT_POLICY_V2_ANSWER_ONLY:
        if evaluation_type in ("code_execution", "reference_based", "ifeval"):
            raise UnsupportedTaskPolicyError(
                f"Policy '{PROMPT_POLICY_V2_ANSWER_ONLY}' is not supported for generative task '{task_type}' "
                f"(evaluation_type: '{evaluation_type}')."
            )

    # Guard: v3_few_shot cannot be applied to tasks without defined demonstrations
    if requested_policy == PROMPT_POLICY_V3_FEW_SHOT:
        if evaluation_type in ("code_execution", "reference_based", "ifeval"):
            raise UnsupportedTaskPolicyError(
                f"Policy '{PROMPT_POLICY_V3_FEW_SHOT}' has no valid demonstration for '{task_type}' "
                f"(evaluation_type: '{evaluation_type}')."
            )

    # Resolve settings under task-aware dispatcher
    if requested_policy in (
        PROMPT_POLICY_TASK_AWARE,
        PROMPT_POLICY_TASK_AWARE_CANDIDATE,
        PROMPT_POLICY_TASK_AWARE_CONSTRAINED,
        PROMPT_POLICY_PHASE4T,
    ):
        is_candidate = (requested_policy == PROMPT_POLICY_TASK_AWARE_CANDIDATE)
        is_constrained = (requested_policy == PROMPT_POLICY_TASK_AWARE_CONSTRAINED)
        is_phase4t = (requested_policy == PROMPT_POLICY_PHASE4T)
        default_cls_budget = 128 if is_candidate else 64
        default_sum_budget = 384 if is_candidate else 256
        policy_label = (
            "phase4t_frozen" if is_phase4t
            else ("constrained" if is_constrained
                  else ("candidate" if is_candidate else "baseline"))
        )

        if evaluation_type == "multiple_choice":
            resolved_policy = PROMPT_POLICY_V4_STRUCTURED_OUTPUT
            budget = user_output_budget or (custom_budgets.get("multiple_choice") if custom_budgets else None) or 64
            thinking = user_think if user_think is not None else False
            schema = MULTIPLE_CHOICE_JSON_SCHEMA
            description = f"Task-aware structured JSON output policy ({policy_label}, schema-constrained A/B/C/D)."

        elif task_type == "classification":
            is_bool = (source_dataset in ("boolq", "wic", "wsc", "multirc") or source_dataset is None or is_constrained)
            if is_constrained or (is_phase4t and is_bool):
                resolved_policy = PROMPT_POLICY_TASK_AWARE_CONSTRAINED
                budget = user_output_budget or (custom_budgets.get("classification") if custom_budgets else None) or 64
                thinking = user_think if user_think is not None else False
                schema = BOOLEAN_CLASSIFICATION_JSON_SCHEMA
                description = f"Task-aware grammar-constrained JSON output policy for classification ({policy_label})."
            elif is_phase4t:
                resolved_policy = PROMPT_POLICY_TASK_AWARE_CONSTRAINED
                budget = user_output_budget or (custom_budgets.get("classification") if custom_budgets else None) or 64
                thinking = user_think if user_think is not None else False
                schema = GENERAL_CLASSIFICATION_JSON_SCHEMA
                description = f"Task-aware general classification JSON schema policy ({policy_label})."
            else:
                resolved_policy = PROMPT_POLICY_V2_ANSWER_ONLY
                budget = user_output_budget or (custom_budgets.get("classification") if custom_budgets else None) or default_cls_budget
                thinking = user_think if user_think is not None else False
                schema = None
                description = f"Task-aware concise answer-only label policy for classification ({policy_label}, budget {budget})."

        elif task_type == "math" and evaluation_type == "exact_match":
            resolved_policy = PROMPT_POLICY_V1_CONCISE
            budget = user_output_budget or (custom_budgets.get("math") if custom_budgets else None) or 512
            thinking = user_think if user_think is not None else False
            schema = None
            description = f"Task-aware concise derivation with Final Answer marker for math ({policy_label})."

        elif task_type == "reasoning" and evaluation_type == "exact_match":
            resolved_policy = PROMPT_POLICY_V1_CONCISE
            budget = user_output_budget or (custom_budgets.get("reasoning") if custom_budgets else None) or 256
            thinking = user_think if user_think is not None else False
            schema = None
            description = f"Task-aware concise derivation with Final Answer marker for boolean/reasoning ({policy_label})."

        elif task_type == "instruction_following" and evaluation_type == "ifeval":
            resolved_policy = PROMPT_POLICY_V0
            budget = user_output_budget or (custom_budgets.get("instruction_following") if custom_budgets else None) or 1024
            thinking = user_think if user_think is not None else False
            schema = None
            description = f"Task-aware verbatim prompt policy for instruction-following constraints ({policy_label})."

        elif task_type == "summarization" and evaluation_type == "reference_based":
            if is_constrained:
                resolved_policy = PROMPT_POLICY_TASK_AWARE_CONSTRAINED
                budget = user_output_budget or (custom_budgets.get("summarization") if custom_budgets else None) or 256
                thinking = user_think if user_think is not None else False
                schema = None
                description = "Task-aware output-only constrained summarization policy (Phase 4S, budget 256)."
            else:
                resolved_policy = PROMPT_POLICY_V1_CONCISE
                budget = user_output_budget or (custom_budgets.get("summarization") if custom_budgets else None) or default_sum_budget
                thinking = user_think if user_think is not None else False
                schema = None
                description = f"Task-aware bounded 2-3 sentence summarization policy ({policy_label}, budget {budget})."

        elif task_type == "coding" and evaluation_type == "code_execution":
            resolved_policy = PROMPT_POLICY_V1_CONCISE
            budget = user_output_budget or (custom_budgets.get("coding") if custom_budgets else None) or 512
            thinking = user_think if user_think is not None else False
            schema = None
            description = f"Task-aware code block implementation policy ({policy_label})."

        else:
            resolved_policy = PROMPT_POLICY_V1_CONCISE
            budget = user_output_budget or resolve_output_budget(task_type, evaluation_type, custom_budgets=custom_budgets)
            thinking = user_think if user_think is not None else False
            schema = None
            description = f"Task-aware default concise policy ({policy_label})."

    else:
        # User specified an explicit policy override (v0, v1_concise, v2_answer_only, v3_few_shot, v4_structured_output)
        resolved_policy = requested_policy
        budget = user_output_budget or resolve_output_budget(task_type, evaluation_type)
        thinking = user_think if user_think is not None else resolve_thinking_mode(task_type, evaluation_type)
        schema = response_format or resolve_response_format(requested_policy, evaluation_type, task_type)
        description = f"Explicit user policy override '{requested_policy}'."

    effective_prompt = format_prompt_with_policy(
        prompt_text=prompt_text,
        evaluation_type=evaluation_type,
        task_type=task_type,
        policy_version=resolved_policy,
    )

    return DispatchedTaskPolicy(
        task_type=task_type,
        evaluation_type=evaluation_type,
        policy_version=resolved_policy,
        effective_prompt=effective_prompt,
        output_budget=budget,
        thinking_mode=thinking,
        response_schema=schema,
        compatible_scorer=scorer_name,
        scorer_version=scorer_version,
        description=description,
    )

