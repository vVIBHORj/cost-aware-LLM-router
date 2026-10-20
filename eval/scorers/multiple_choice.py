import re
from typing import Any


def extract_mc_choice(text: str) -> str | None:
    """
    Extract multiple choice selection from model generation.
    Handles explicit answer phrases, parenthesized choices, standalone letters,
    LaTeX \\boxed{...}, and markdown formatted final answer headers.
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Pattern 0: LaTeX \\boxed{...} (handles nested braces like \\boxed{\\text{A. 4}})
    boxed_blocks = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", cleaned)
    if boxed_blocks:
        last_boxed = boxed_blocks[-1].strip()
        m = re.search(r"(?i)\b([A-Ha-h])(?:\.|\b)", last_boxed)
        if m:
            return m.group(1).upper()

    # Pattern 1: Explicit final answer headers/statements
    # e.g. "Final Answer: **A**", "### Final Answer: \n**C. 12**", "The answer is (B)"
    p_final = re.search(
        r"(?i)(?:final\s+answer|correct\s+answer|the\s+answer\s+is|answer)\s*[:\s]*[\*\s]*\(?([A-Ha-h])\)?(?:[\.\:\)\s\*]|$)",
        cleaned,
    )
    if p_final:
        return p_final.group(1).upper()

    # Pattern 2: Option / Choice indicators
    p2 = re.search(
        r"(?i)(?:option|choice)\s*[:\s]*\(?([A-Ha-h])\)?(?:[\.\:\)\s\*]|$)",
        cleaned,
    )
    if p2:
        return p2.group(1).upper()

    # Pattern 3: Standalone single character response
    if len(cleaned) <= 5:
        p3 = re.search(r"\(?([A-Ha-h])\)?", cleaned)
        if p3:
            return p3.group(1).upper()

    # Pattern 4: First character formatted as (A) or A. or A:
    p4 = re.match(r"^\s*\(?([A-Ha-h])\)?[\.\:\)\s]", cleaned)
    if p4:
        return p4.group(1).upper()

    # Pattern 5: Last line contains parenthesized or bold choice: '(A)' or '**A**'
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        for line in reversed(lines[-3:]):
            p5 = re.search(r"(?:\(|(?:\*\*))([A-Ha-h])(?:\)|(?:\*\*))", line)
            if p5:
                return p5.group(1).upper()

    return None


def score_multiple_choice(
    prediction: str,
    reference: str,
) -> dict[str, Any]:
    """
    Score multiple-choice answer prediction against reference choice.
    Returns:
        quality_score: 1.0 if extracted choice matches reference, else 0.0
        exact_match: bool
        normalized_exact_match: bool
        details: dict
    """
    ref_choice = reference.strip().upper()

    # Clean single letter reference if reference is e.g. "(A)" or "A."
    ref_extracted = extract_mc_choice(ref_choice) or ref_choice

    pred_choice = extract_mc_choice(prediction)

    is_match = (pred_choice is not None and pred_choice == ref_extracted)

    return {
        "quality_score": 1.0 if is_match else 0.0,
        "exact_match": (prediction.strip() == reference.strip()),
        "normalized_exact_match": is_match,
        "details": {
            "extracted_choice": pred_choice,
            "target_choice": ref_extracted,
            "raw_reference": reference,
        },
    }
