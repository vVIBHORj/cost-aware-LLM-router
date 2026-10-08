import re
from typing import Any


def extract_mc_choice(text: str) -> str | None:
    """
    Extract multiple choice selection from model generation.
    Handles explicit answer phrases, parenthesized choices, and standalone letters.
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Pattern 1: Explicit answer indicators
    # "The answer is (A)", "Answer: B", "The correct answer is Option C"
    p1 = re.search(r"(?i)(?:the\s+correct\s+answer\s+is|the\s+answer\s+is|answer)\s*[:\s]*\(?([A-Za-z0-9])\)?", cleaned)
    if p1:
        return p1.group(1).upper()

    # Pattern 2: Option / Choice indicators
    p2 = re.search(r"(?i)(?:option|choice)\s*[:\s]*\(?([A-Za-z0-9])\)?", cleaned)
    if p2:
        return p2.group(1).upper()

    # Pattern 3: Standalone single character response
    if len(cleaned) <= 3:
        p3 = re.search(r"[A-Za-z0-9]", cleaned)
        if p3:
            return p3.group(0).upper()

    # Pattern 4: First character formatted as (A) or A. or A:
    p4 = re.match(r"^\s*\(?([A-Za-z0-9])\)?[\.\:\)\s]", cleaned)
    if p4:
        return p4.group(1).upper()

    # Pattern 5: Last line contains parenthesized choice: "(A)"
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        p5 = re.search(r"\(([A-Za-z0-9])\)", last_line)
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

    # If extraction failed, also test if reference string itself occurs as full word
    is_match = False
    if pred_choice is not None and pred_choice == ref_extracted:
        is_match = True
    elif pred_choice is None and ref_extracted:
        # Fallback exact word boundary match
        pattern = rf"\b{re.escape(ref_extracted)}\b"
        if re.search(pattern, prediction, re.IGNORECASE):
            is_match = True

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
