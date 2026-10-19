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
