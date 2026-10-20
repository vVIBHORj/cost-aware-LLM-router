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
}


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

    policy_map = DEFAULT_PROMPT_POLICY_INSTRUCTIONS.get(policy_version, {})
    instruction = policy_map.get(evaluation_type) or policy_map.get(task_type)
    if not instruction:
        return prompt_text

    return f"{prompt_text}\n\nInstruction: {instruction}"
