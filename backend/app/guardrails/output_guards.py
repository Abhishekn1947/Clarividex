"""
Output guardrails for validating and sanitizing AI responses.

Four guards:
1. Response quality — ensures minimum length and coherence
2. PII redaction — strips emails, phone numbers, SSNs
3. Financial advice detection — flags direct buy/sell recommendations
4. Probability bounds — clamps probability to 15-85 range
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger()


@dataclass
class OutputGuardResult:
    """Result from running all output guards."""
    text: str
    probability: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    modifications: list[str] = field(default_factory=list)
    passed: bool = True


def _guard_response_quality(text: str, warnings: list[str], modifications: list[str]) -> str:
    """Ensure response meets minimum quality standards."""
    if len(text.strip()) < 10:
        warnings.append("Response too short — may be incomplete")
    return text


def _guard_pii_redaction(text: str, warnings: list[str], modifications: list[str]) -> str:
    """Redact personally identifiable information from responses."""
    original = text

    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]', text)

    # Phone numbers (various formats)
    text = re.sub(r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE REDACTED]', text)

    # SSN patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', text)

    # Credit card patterns (basic)
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD REDACTED]', text)

    if text != original:
        modifications.append("PII redacted from response")

    return text


def _guard_financial_advice(text: str, warnings: list[str], modifications: list[str]) -> str:
    """Detect and flag direct financial advice language."""
    advice_patterns = [
        r'\byou should (buy|sell|invest|short)\b',
        r'\b(buy|sell|short) (this stock|immediately|now|right away)\b',
        r'\bguaranteed (return|profit|gain)\b',
        r'\brisk[- ]free\b',
        r'\bcannot (lose|fail)\b',
        r'\bsure thing\b',
    ]

    for pattern in advice_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            warnings.append(f"Response contains potential financial advice language (matched: {pattern})")
            break

    return text


def _guard_probability_bounds(
    probability: Optional[float],
    warnings: list[str],
    modifications: list[str],
) -> Optional[float]:
    """
    Clamp probability to valid bounds.

    Expects probability on 0-100 scale for checking, returns on same scale.
    The caller is responsible for converting to/from 0-1 scale if needed.
    """
    if probability is None:
        return None

    clamped = max(15.0, min(85.0, probability))
    if clamped != probability:
        modifications.append(
            f"Probability clamped from {probability:.1f} to {clamped:.1f} (bounds: 15-85)"
        )
    return clamped


def run_output_guards(
    text: str,
    probability: Optional[float] = None,
) -> OutputGuardResult:
    """
    Run all output guards on response text and optional probability.

    Args:
        text: The response text to validate.
        probability: Probability value on 0-100 scale to clamp.

    Returns:
        OutputGuardResult with potentially modified text, clamped probability,
        and any warnings/modifications logged.
    """
    warnings: list[str] = []
    modifications: list[str] = []

    # Run text guards in order
    text = _guard_response_quality(text, warnings, modifications)
    text = _guard_pii_redaction(text, warnings, modifications)
    text = _guard_financial_advice(text, warnings, modifications)

    # Run probability guard
    clamped_prob = _guard_probability_bounds(probability, warnings, modifications)

    result = OutputGuardResult(
        text=text,
        probability=clamped_prob,
        warnings=warnings,
        modifications=modifications,
        passed=len(warnings) == 0,
    )

    if warnings or modifications:
        logger.info(
            "Output guardrails applied",
            warnings=warnings,
            modifications=modifications,
        )

    return result
