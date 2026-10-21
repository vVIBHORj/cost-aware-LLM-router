import re
from typing import Any


def extract_mc_choice(text: str) -> str | None:
    """
    Extract multiple choice selection from model generation.
    Handles explicit answer phrases, parenthesized choices, standalone letters,
    LaTeX \\boxed{...}, and markdown formatted final answer headers.
    Prioritizes explicit final answer markers and ignores unanchored words.
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Pattern 0: Structured JSON output, e.g. {"answer": "B"} or Markdown ```json block
    json_candidate = cleaned
    if "```json" in json_candidate:
        m_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_candidate, re.DOTALL)
        if m_block:
            json_candidate = m_block.group(1).strip()
    if json_candidate.startswith("{") and json_candidate.endswith("}"):
        try:
            import json as _json
            data = _json.loads(json_candidate)
            if isinstance(data, dict) and "answer" in data:
                val = str(data["answer"]).strip().upper()
                if re.match(r"^[A-H]$", val):
                    return val
        except Exception:
            pass

    # JSON regex fallback for partial / unclosed JSON: "answer"\s*:\s*"([A-Ha-h])"
    p_json_field = re.findall(r'"answer"\s*:\s*"([A-Ha-h])"', cleaned, re.IGNORECASE)
    if p_json_field:
        return p_json_field[-1].upper()

    # Pattern 1: Explicit Final Answer statement (take the last occurrence in generation)
    # e.g. 'Final Answer: B', 'Final Answer: **A**', 'Final Answer: `C`', '### Final Answer:\n**C. 12**', 'Final Answer: Option D'
    p_final = re.findall(
        r"(?i)final\s+answer\s*[:\s]*[\*`\s]*(?:option\s+|choice\s+)?\(?([A-Ha-h])\)?(?:[\.\:\)\s\*`]|$)",
        cleaned,
    )
    if p_final:
        return p_final[-1].upper()

    # Pattern 2: LaTeX \\boxed{...} (take the last boxed block in generation)
    boxed_blocks = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", cleaned)
    if boxed_blocks:
        last_boxed = boxed_blocks[-1].strip()
        m = re.search(r"(?i)\b([A-Ha-h])(?:\.|\b)", last_boxed)
        if m:
            return m.group(1).upper()

    # Pattern 3: Explicit 'The correct answer is (X)' or 'The answer is (X)'
    p_statement = re.findall(
        r"(?i)(?:the\s+correct\s+answer\s+is|the\s+answer\s+is)\s*[:\s]*[\*`\s]*(?:option\s+|choice\s+)?\(?([A-Ha-h])\)?",
        cleaned,
    )
    if p_statement:
        return p_statement[-1].upper()

    # Pattern 4: Start of line 'Answer:\s*([A-H])' or 'Option:\s*([A-H])' or 'Option D is correct'
    p_option_correct = re.findall(
        r"(?i)(?:option|choice)\s+([A-Ha-h])(?:\s+is\s+correct|\b)",
        cleaned,
    )
    if p_option_correct:
        return p_option_correct[-1].upper()

    p_line_answer = re.findall(
        r"(?mi)^(?:[#\*\s]*)(?:correct\s+answer|final\s+answer|answer|option|choice)\s*:\s*[\*`\s]*(?:option\s+|choice\s+)?\(?([A-Ha-h])\)?",
        cleaned,
    )
    if p_line_answer:
        return p_line_answer[-1].upper()

    # Pattern 5: Standalone single character / parenthesized response (short string)
    if len(cleaned) <= 7:
        p_short = re.search(r"^\s*\(?([A-Ha-h])\)?[\.\:\s]*$", cleaned)
        if p_short:
            return p_short.group(1).upper()

    # Pattern 6: Leading choice formatted as (A) or A. or A:
    p_lead = re.match(r"^\s*\(?([A-Ha-h])\)?[\.\:\)\s]", cleaned)
    if p_lead:
        return p_lead.group(1).upper()

    # Pattern 7: Very last non-empty line contains parenthesized or bold choice: '(A)' or '**A**'
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        for line in reversed(lines[-2:]):
            p_last = re.search(r"(?:\(|(?:\*\*))([A-Ha-h])(?:\)|(?:\*\*))", line)
            if p_last:
                return p_last.group(1).upper()

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
