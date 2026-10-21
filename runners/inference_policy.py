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
    "classification": {
        "type": "object",
        "properties": {
            "label": {"type": "string"}
        },
        "required": ["label"],
        "additionalProperties": False,
    },
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
    },
    PROMPT_POLICY_V2_ANSWER_ONLY: {
        "multiple_choice": (
            "Select the single best answer. Output only its option letter, with no explanation, reasoning, prefix, or other text."
        ),
        "exact_match": (
            "Output only the final answer, with no explanation, reasoning, units, prefix, or other text."
        ),
        "classification": (
            "Output only the final label, with no explanation, reasoning, prefix, or other text."
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
            "Classify the text. Return a JSON object with key 'label' containing the classification label."
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
    instruction = policy_map.get(evaluation_type) or policy_map.get(task_type)
    if not instruction:
        return prompt_text

    return f"{prompt_text}\n\nInstruction: {instruction}"
