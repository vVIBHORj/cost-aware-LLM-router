import re
from typing import Any

from data_ingestion.base import DatasetAdapter
from router.benchmark_schema import BenchmarkPrompt, Difficulty

SUPPORTED_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def extract_boxed_answer(solution: str) -> str:
    """
    Extract the final answer from a Hendrycks MATH solution.

    MATH solutions conventionally format the final answer inside \\boxed{...}.
    Uses brace-depth matching to handle nested LaTeX braces properly.
    """
    if not isinstance(solution, str) or not solution.strip():
        return ""

    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        return ""

    i = idx + len(r"\boxed{")
    depth = 1
    chars: list[str] = []
    while i < len(solution) and depth > 0:
        ch = solution[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(ch)
        i += 1

    return "".join(chars).strip()


def normalize_math_difficulty(level_str: str | None) -> Difficulty:
    """Map Hendrycks MATH 'Level X' string to canonical Difficulty."""
    if not level_str:
        return "medium"

    match = re.search(r"Level\s*([1-5])", str(level_str), re.IGNORECASE)
    if not match:
        return "medium"

    level = int(match.group(1))
    if level <= 2:
        return "easy"
    elif level <= 4:
        return "medium"
    return "hard"


class HendrycksMATHAdapter(DatasetAdapter):
    """Adapter for EleutherAI/hendrycks_math records."""

    source_dataset = "math"

    def __init__(self, source_config: str = "algebra") -> None:
        if not isinstance(source_config, str) or not source_config.strip():
            raise ValueError("source_config must be a non-empty string.")

        source_config = source_config.strip()
        if source_config not in SUPPORTED_SUBJECTS:
            raise ValueError(
                f"Unsupported MATH subject '{source_config}'. Supported: {sorted(SUPPORTED_SUBJECTS)}"
            )

        self.source_config = source_config

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        source_split: str,
        source_id: str,
        benchmark_split: str,
    ) -> BenchmarkPrompt:
        if not isinstance(record, dict):
            raise ValueError("MATH record must be a dictionary.")

        problem = record.get("problem")
        solution = record.get("solution")
        level = record.get("level")

        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("MATH record is missing a valid 'problem' string.")

        if not isinstance(solution, str) or not solution.strip():
            raise ValueError("MATH record is missing a valid 'solution' string.")

        final_answer = extract_boxed_answer(solution)
        if not final_answer:
            final_answer = solution.strip()

        difficulty = normalize_math_difficulty(level)

        prompt_id = (
            f"math-{self.source_config}-"
            f"{source_split}-{source_id}"
        )

        return BenchmarkPrompt(
            prompt_id=prompt_id,
            prompt=problem.strip(),
            task_type="math",
            domain=self.source_config,
            difficulty=difficulty,
            source_dataset=self.source_dataset,
            source_config=self.source_config,
            source_split=source_split,
            source_id=str(source_id).strip(),
            split=benchmark_split,
            expected_output=final_answer,
            evaluation_type="exact_match",
        )
