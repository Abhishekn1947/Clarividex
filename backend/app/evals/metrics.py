"""Evaluation metrics for Clarividex predictions."""

from typing import Optional


def prediction_quality(
    summary: str,
    probability: float,
    bullish_count: int,
    bearish_count: int,
) -> dict:
    """
    Evaluate prediction quality.

    Checks: non-empty summary, valid probability, at least one factor.
    Returns dict with score (0-1) and details.
    """
    checks = {
        "has_summary": len(summary.strip()) > 20,
        "valid_probability": 0.15 <= probability <= 0.85,
        "has_factors": (bullish_count + bearish_count) > 0,
        "has_both_sides": bullish_count > 0 and bearish_count > 0,
    }
    score = sum(checks.values()) / len(checks)
    return {"score": score, "checks": checks}


def response_completeness(
    response: dict,
    required_fields: Optional[list[str]] = None,
) -> dict:
    """
    Check if a response contains all required fields.

    Returns dict with score (0-1) and missing fields.
    """
    if required_fields is None:
        required_fields = [
            "id", "query", "ticker", "probability",
            "confidence_level", "reasoning",
        ]

    present = [f for f in required_fields if f in response and response[f] is not None]
    missing = [f for f in required_fields if f not in response or response[f] is None]
    score = len(present) / len(required_fields) if required_fields else 1.0

    return {"score": score, "present": present, "missing": missing}


def keyword_recall(
    text: str,
    expected_keywords: list[str],
) -> dict:
    """
    Measure what fraction of expected keywords appear in the text.

    Returns dict with score (0-1) and found/missing keywords.
    """
    text_lower = text.lower()
    found = [kw for kw in expected_keywords if kw.lower() in text_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in text_lower]
    score = len(found) / len(expected_keywords) if expected_keywords else 1.0

    return {"score": score, "found": found, "missing": missing}


def latency_check(
    duration_seconds: float,
    max_acceptable: float = 30.0,
) -> dict:
    """
    Check if latency is within acceptable bounds.

    Returns dict with score (0-1) and details.
    """
    if duration_seconds <= max_acceptable:
        score = 1.0 - (duration_seconds / max_acceptable) * 0.5  # Partial credit
    else:
        score = max(0.0, 1.0 - (duration_seconds / max_acceptable))

    return {
        "score": min(1.0, score),
        "duration_seconds": duration_seconds,
        "max_acceptable": max_acceptable,
        "within_bounds": duration_seconds <= max_acceptable,
    }


def guardrail_accuracy(
    probability: float,
    warnings: list[str],
    modifications: list[str],
    expected_flags: list[str],
) -> dict:
    """
    Evaluate guardrail effectiveness.

    Checks: probability in bounds, expected flags triggered.
    """
    checks = {
        "probability_bounded": 0.15 <= probability <= 0.85,
    }

    for flag in expected_flags:
        if flag == "probability_clamped":
            checks["expected_clamp"] = len(modifications) > 0
        elif flag == "financial_advice":
            checks["advice_flagged"] = any("advice" in w.lower() for w in warnings)

    score = sum(checks.values()) / len(checks) if checks else 1.0
    return {"score": score, "checks": checks}


def probability_bounds_check(probability: float) -> dict:
    """
    Simple check that probability is within 15-85% bounds.

    Args:
        probability: Probability on 0-1 scale.

    Returns:
        Dict with pass/fail and details.
    """
    in_bounds = 0.15 <= probability <= 0.85
    return {
        "score": 1.0 if in_bounds else 0.0,
        "probability": probability,
        "in_bounds": in_bounds,
        "min_bound": 0.15,
        "max_bound": 0.85,
    }
