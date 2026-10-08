import re
from typing import Any


def normalize_exact_text(text: str) -> str:
    """
    Standard normalization for exact match:
    1. Lowercase
    2. Strip trailing/leading whitespace
    3. Collapse multiple whitespaces to single space
    4. Strip common punctuation
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def extract_boxed_math(text: str) -> str | None:
    """Extract content from LaTeX \boxed{...} if present."""
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def score_exact_match(
    prediction: str,
    reference: str,
) -> dict[str, Any]:
    """
    Compute exact and normalized exact match scores.
    Returns:
        quality_score: 1.0 if normalized match, else 0.0
        exact_match: bool
        normalized_exact_match: bool
        details: dict
    """
    pred_clean = prediction.strip()
    ref_clean = reference.strip()

    is_strict = (pred_clean == ref_clean)

    # Check boxed match if reference or prediction has LaTeX boxed
    ref_boxed = extract_boxed_math(ref_clean) or ref_clean
    pred_boxed = extract_boxed_math(pred_clean) or pred_clean

    norm_pred = normalize_exact_text(pred_boxed)
    norm_ref = normalize_exact_text(ref_boxed)

    is_normalized = (norm_pred == norm_ref) if norm_ref else False

    # Also handle numeric equivalence if both are floats/ints
    is_numeric = False
    try:
        if abs(float(pred_boxed) - float(ref_boxed)) < 1e-6:
            is_numeric = True
            is_normalized = True
    except (ValueError, TypeError):
        pass

    final_match = is_strict or is_normalized

    return {
        "quality_score": 1.0 if final_match else 0.0,
        "exact_match": is_strict,
        "normalized_exact_match": final_match,
        "details": {
            "prediction_normalized": norm_pred,
            "reference_normalized": norm_ref,
            "numeric_match": is_numeric,
        },
    }
