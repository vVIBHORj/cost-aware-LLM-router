import re
from collections import Counter
from typing import Any


def tokenize_words(text: str) -> list[str]:
    """Tokenize lowercased text into word alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_token_f1(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute deterministic unigram precision, recall, and F1 score (ROUGE-1 equivalent).
    """
    pred_tokens = tokenize_words(prediction)
    ref_tokens = tokenize_words(reference)

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    # Overlap count
    overlap = sum(min(count, ref_counts[token]) for token, count in pred_counts.items())

    prec = overlap / len(pred_tokens) if pred_tokens else 0.0
    rec = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def compute_longest_common_subsequence(seq1: list[str], seq2: list[str]) -> int:
    """Compute length of LCS for ROUGE-L."""
    m, n = len(seq1), len(seq2)
    # Space-optimized DP
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def score_reference_based(
    prediction: str,
    reference: str,
) -> dict[str, Any]:
    """
    Deterministic reference-based scoring using unigram overlap (ROUGE-1) and LCS (ROUGE-L).
    Returns:
        quality_score: float in [0.0, 1.0] (unigram F1)
        reference_score: float (LCS F1)
        details: dict with precision, recall, f1, lcs_f1
    """
    if not reference or not reference.strip():
        return {
            "quality_score": None,
            "reference_score": None,
            "scoring_status": "INVALID_INPUT",
            "scoring_error_type": "MissingReferenceError",
            "scoring_error_message": "Reference text is required for reference-based scoring.",
            "details": {},
        }

    unigram_metrics = compute_token_f1(prediction, reference)

    pred_tokens = tokenize_words(prediction)
    ref_tokens = tokenize_words(reference)

    lcs_len = compute_longest_common_subsequence(pred_tokens[:500], ref_tokens[:500]) if pred_tokens and ref_tokens else 0
    lcs_prec = lcs_len / len(pred_tokens) if pred_tokens else 0.0
    lcs_rec = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    lcs_f1 = (2 * lcs_prec * lcs_rec / (lcs_prec + lcs_rec)) if (lcs_prec + lcs_rec) > 0 else 0.0

    return {
        "quality_score": unigram_metrics["f1"],
        "reference_score": round(lcs_f1, 4),
        "scoring_status": "SUCCESS",
        "details": {
            "unigram_precision": unigram_metrics["precision"],
            "unigram_recall": unigram_metrics["recall"],
            "unigram_f1": unigram_metrics["f1"],
            "lcs_f1": round(lcs_f1, 4),
        },
    }
