import json
import re
from typing import Any


def check_instruction_rule(instruction_id: str, kwargs: dict[str, Any], text: str) -> bool | None:
    """
    Evaluate a single IFEval instruction rule deterministically against generated text.
    Returns:
        True: rule followed
        False: rule violated
        None: instruction rule requires external evaluation library / unsupported
    """
    if instruction_id == "punctuation:no_comma":
        return "," not in text

    elif instruction_id == "change_case:english_lowercase":
        return text.islower() or (not any(c.isupper() for c in text))

    elif instruction_id == "change_case:english_capital":
        return text.isupper() or (not any(c.islower() for c in text))

    elif instruction_id == "startend:quotation":
        stripped = text.strip()
        return (stripped.startswith('"') and stripped.endswith('"')) or (
            stripped.startswith("“") and stripped.endswith("”")
        )

    elif instruction_id == "startend:end_checker":
        end_phrase = kwargs.get("end_phrase", "")
        return text.strip().endswith(end_phrase)

    elif instruction_id == "startend:start_checker":
        start_phrase = kwargs.get("start_phrase", "")
        return text.strip().startswith(start_phrase)

    elif instruction_id == "keywords:existence":
        keywords = kwargs.get("keywords", [])
        return all(kw.lower() in text.lower() for kw in keywords)

    elif instruction_id == "keywords:forbidden_words":
        forbidden_words = kwargs.get("forbidden_words", [])
        return not any(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in forbidden_words)

    elif instruction_id == "keywords:frequency":
        keyword = kwargs.get("keyword", "")
        freq = kwargs.get("frequency", 0)
        relation = kwargs.get("relation", "at least")
        actual = len(re.findall(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE))
        if relation in ("at least", "gte"):
            return actual >= freq
        elif relation in ("at most", "lte"):
            return actual <= freq
        elif relation in ("equal", "eq"):
            return actual == freq
        return None

    elif instruction_id == "keywords:letter_frequency":
        letter = kwargs.get("letter", "")
        letter_freq = kwargs.get("let_frequency", 0)
        relation = kwargs.get("relation", "at least")
        actual = text.lower().count(letter.lower())
        if relation in ("at least", "gte"):
            return actual >= letter_freq
        elif relation in ("at most", "lte"):
            return actual <= letter_freq
        elif relation in ("equal", "eq"):
            return actual == letter_freq
        return None

    elif instruction_id == "length_constraints:number_paragraphs":
        num_paragraphs = kwargs.get("num_paragraphs", 0)
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        return len(paragraphs) == num_paragraphs

    elif instruction_id == "length_constraints:number_words":
        num_words = kwargs.get("num_words", 0)
        relation = kwargs.get("relation", "at least")
        words = text.split()
        actual = len(words)
        if relation in ("at least", "gte"):
            return actual >= num_words
        elif relation in ("at most", "lte"):
            return actual <= num_words
        elif relation in ("equal", "eq"):
            return actual == num_words
        return None

    elif instruction_id == "detectable_content:postscript":
        postscript_marker = kwargs.get("postscript_marker", "P.S.")
        return postscript_marker.lower() in text.lower()

    # Other rules that require external NLP parse or language identification
    return None


def score_ifeval(
    prediction: str,
    expected_output_json: str | None,
) -> dict[str, Any]:
    """
    Score IFEval instruction following against serialized constraint metadata.
    If constraints cannot be fully evaluated deterministically, returns SCORING_UNAVAILABLE
    with diagnostic details while preserving the raw generation.
    """
    if not expected_output_json or not expected_output_json.strip():
        return {
            "quality_score": None,
            "scoring_status": "INVALID_INPUT",
            "scoring_error_type": "MissingExpectedOutputError",
            "scoring_error_message": "IFEval requires serialized constraint metadata in expected_output.",
            "details": {},
        }

    try:
        spec = json.loads(expected_output_json)
        instruction_ids = spec.get("instruction_id_list", [])
        kwargs_list = spec.get("kwargs", [])
    except Exception as e:
        return {
            "quality_score": None,
            "scoring_status": "SCORING_ERROR",
            "scoring_error_type": "ConstraintParsingError",
            "scoring_error_message": f"Failed to parse IFEval constraints JSON: {e}",
            "details": {},
        }

    if not instruction_ids:
        return {
            "quality_score": None,
            "scoring_status": "INVALID_INPUT",
            "scoring_error_type": "EmptyInstructionListError",
            "scoring_error_message": "IFEval record contains empty instruction_id_list.",
            "details": {},
        }

    rule_results: dict[str, bool | None] = {}
    unsupported_rules: list[str] = []

    for idx, inst_id in enumerate(instruction_ids):
        kw = kwargs_list[idx] if idx < len(kwargs_list) and isinstance(kwargs_list[idx], dict) else {}
        result = check_instruction_rule(inst_id, kw, prediction)
        rule_results[f"{inst_id}_{idx}"] = result
        if result is None:
            unsupported_rules.append(inst_id)

    if unsupported_rules:
        return {
            "quality_score": None,
            "scoring_status": "SCORING_UNAVAILABLE",
            "scoring_error_type": "UnsupportedInstructionRulesError",
            "scoring_error_message": (
                f"Instructions {unsupported_rules} require the full external IFEval evaluation suite. "
                "Preserving raw response for external evaluation."
            ),
            "details": {
                "rule_results": rule_results,
                "unsupported_rules": unsupported_rules,
            },
        }

    # All rules were evaluated
    all_passed = all(res is True for res in rule_results.values())
    quality = 1.0 if all_passed else 0.0

    return {
        "quality_score": quality,
        "scoring_status": "SUCCESS",
        "scoring_error_type": None,
        "scoring_error_message": None,
        "details": {
            "rule_results": rule_results,
            "total_rules": len(rule_results),
            "passed_rules": sum(1 for res in rule_results.values() if res is True),
        },
    }
