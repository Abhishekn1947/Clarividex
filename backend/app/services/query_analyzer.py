"""
Smart Query Analyzer — two-stage analysis for incoming prediction queries.

Stage 1 (rule-based, ~0 ms): classify as clear / vague / non_financial.
Stage 2 (AI, ~300-800 ms):   generate improved query suggestions (vague/non_financial only).
"""

import re
import json
import asyncio
from dataclasses import dataclass, field
from typing import Literal

import structlog

from backend.app.services.query_validator import financial_query_validator

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryAnalysisResult:
    category: Literal["clear", "vague", "non_financial"]
    can_proceed: bool
    quality_score: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"\$?[A-Z]{1,5}\b")
_PRICE_RE = re.compile(r"\$[\d,]+(?:\.\d+)?")
_TIMEFRAME_RE = re.compile(
    r"\b(by|before|until|within|this month|next month|this quarter|next quarter|this year|next year"
    r"|january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
    r"|q[1-4]|20\d{2}|\d+ ?(day|week|month|year)s?)\b",
    re.IGNORECASE,
)

HARDCODED_FINANCIAL_SUGGESTIONS = [
    "Will NVDA reach $150 by March 2026?",
    "Will Tesla stock go up in the next month?",
    "Will Bitcoin hit $100k this year?",
]

HARDCODED_VAGUE_SUGGESTIONS = [
    "Try adding a ticker symbol, e.g. 'Will AAPL reach $200 by June?'",
    "Specify a price target: 'Will NVDA hit $150?'",
    "Add a timeframe: 'Will Tesla go up in the next 3 months?'",
]


class QueryAnalyzer:
    """Wraps FinancialQueryValidator with smarter two-stage analysis."""

    def __init__(self) -> None:
        self.logger = logger.bind(service="query_analyzer")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, query: str) -> QueryAnalysisResult:
        """Analyze *query* and return classification + suggestions."""

        if not query or not query.strip():
            return QueryAnalysisResult(
                category="non_financial",
                can_proceed=False,
                quality_score=0.0,
                issues=["Query is empty"],
                suggestions=HARDCODED_FINANCIAL_SUGGESTIONS,
                message="Please enter a prediction query about financial markets.",
            )

        # --- Stage 1: rule-based ----------------------------------------
        validation = financial_query_validator.validate_query(query)
        quality_score = self._compute_quality_score(query)

        if validation.is_valid and quality_score >= 0.7:
            return QueryAnalysisResult(
                category="clear",
                can_proceed=True,
                quality_score=quality_score,
                message="",
            )

        if validation.is_valid:
            category: Literal["vague", "non_financial"] = "vague"
            issues = self._identify_issues(query)
            message = "Your query could be more specific for better predictions."
        else:
            category = "non_financial"
            issues = [validation.rejection_reason or "Query does not appear to be about financial markets."]
            message = "This doesn't look like a financial prediction query."

        # --- Stage 2: AI suggestions (only for vague / non_financial) ---
        suggestions = await self._generate_suggestions(query, category)

        return QueryAnalysisResult(
            category=category,
            can_proceed=(category == "vague"),
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions,
            message=message,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_quality_score(self, query: str) -> float:
        score = 0.0
        if _TICKER_RE.search(query):
            score += 0.3
        if _PRICE_RE.search(query):
            score += 0.2
        if _TIMEFRAME_RE.search(query):
            score += 0.2
        query_lower = query.lower()
        if any(kw in query_lower for kw in financial_query_validator.FINANCIAL_KEYWORDS):
            score += 0.2
        if len(query.strip()) > 15:
            score += 0.1
        return min(score, 1.0)

    def _identify_issues(self, query: str) -> list[str]:
        issues: list[str] = []
        if not _TICKER_RE.search(query):
            issues.append("No ticker symbol detected (e.g. AAPL, NVDA)")
        if not _PRICE_RE.search(query):
            issues.append("No price target specified (e.g. $150)")
        if not _TIMEFRAME_RE.search(query):
            issues.append("No timeframe mentioned (e.g. 'by March 2026')")
        return issues

    async def _generate_suggestions(
        self, query: str, category: Literal["vague", "non_financial"]
    ) -> list[str]:
        """Try Claude -> Ollama -> hardcoded fallback."""
        try:
            return await asyncio.wait_for(
                self._call_claude_suggestions(query, category),
                timeout=3.0,
            )
        except Exception as e:
            self.logger.debug("Claude suggestion generation failed, trying Ollama", error=str(e))

        try:
            return await asyncio.wait_for(
                self._call_ollama_suggestions(query, category),
                timeout=3.0,
            )
        except Exception as e:
            self.logger.debug("Ollama suggestion generation failed, using fallback", error=str(e))

        return self._fallback_suggestions(category)

    async def _call_claude_suggestions(
        self, query: str, category: Literal["vague", "non_financial"]
    ) -> list[str]:
        from backend.app.services.prediction_engine import prediction_engine

        client = prediction_engine.claude_client
        if not client:
            raise RuntimeError("Claude client not available")

        if category == "vague":
            user_msg = f"User asked: '{query}'. Suggest more specific versions."
        else:
            user_msg = f"User asked: '{query}'. Suggest related financial prediction questions."

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            temperature=0.7,
            system=(
                "Generate 2-3 improved financial prediction queries. "
                "Each must include a ticker/asset, direction/target, and timeframe. "
                "Respond as a JSON array of strings only."
            ),
            messages=[{"role": "user", "content": user_msg}],
        )

        text = response.content[0].text.strip()
        suggestions = json.loads(text)
        if isinstance(suggestions, list) and len(suggestions) >= 1:
            return [str(s) for s in suggestions[:3]]
        raise ValueError("Unexpected Claude response format")

    async def _call_ollama_suggestions(
        self, query: str, category: Literal["vague", "non_financial"]
    ) -> list[str]:
        import httpx
        import os

        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

        if category == "vague":
            prompt = f"User asked: '{query}'. Suggest 2-3 more specific financial prediction queries as a JSON array of strings."
        else:
            prompt = f"User asked: '{query}'. Suggest 2-3 related financial prediction queries as a JSON array of strings."

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": "llama3.1:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 150},
                },
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip()
                # Try to extract JSON array from response
                match = re.search(r"\[.*\]", text, re.DOTALL)
                if match:
                    suggestions = json.loads(match.group())
                    if isinstance(suggestions, list) and len(suggestions) >= 1:
                        return [str(s) for s in suggestions[:3]]
        raise RuntimeError("Ollama failed to produce valid suggestions")

    def _fallback_suggestions(self, category: Literal["vague", "non_financial"]) -> list[str]:
        if category == "vague":
            return list(HARDCODED_VAGUE_SUGGESTIONS)
        return list(HARDCODED_FINANCIAL_SUGGESTIONS)


# Singleton
query_analyzer = QueryAnalyzer()
