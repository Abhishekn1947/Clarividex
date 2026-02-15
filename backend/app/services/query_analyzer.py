"""
Smart Query Analyzer — two-stage analysis for incoming prediction queries.

Stage 1 (rule-based, ~0 ms): classify as clear / vague / non_financial.
Stage 2 (AI, ~300-800 ms):   generate improved query suggestions (vague/non_financial only).

Includes query cleaning: spelling correction, gibberish removal, normalization.
"""

import re
import json
import asyncio
from dataclasses import dataclass, field
from typing import Literal

import structlog

from backend.app.services.query_validator import financial_query_validator
from backend.app.services.market_config import INDIAN_COMPANY_TO_TICKER, INDIAN_TICKER_TO_COMPANY

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
    cleaned_query: str = ""


# ---------------------------------------------------------------------------
# Company / ticker mapping for contextual suggestions
# ---------------------------------------------------------------------------

COMPANY_TICKER_MAP: dict[str, str] = {
    "tesla": "TSLA", "apple": "AAPL", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "microsoft": "MSFT", "meta": "META", "facebook": "META",
    "netflix": "NFLX", "nvidia": "NVDA", "amd": "AMD", "intel": "INTC",
    "disney": "DIS", "walmart": "WMT", "costco": "COST", "starbucks": "SBUX",
    "boeing": "BA", "salesforce": "CRM", "adobe": "ADBE", "paypal": "PYPL",
    "shopify": "SHOP", "spotify": "SPOT", "uber": "UBER", "lyft": "LYFT",
    "coinbase": "COIN", "robinhood": "HOOD", "palantir": "PLTR",
    "snowflake": "SNOW", "crowdstrike": "CRWD", "datadog": "DDOG",
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL",
    "jpmorgan": "JPM", "jp morgan": "JPM", "goldman": "GS",
    "goldman sachs": "GS", "morgan stanley": "MS", "bank of america": "BAC",
    "wells fargo": "WFC", "citigroup": "C", "visa": "V", "mastercard": "MA",
    "berkshire": "BRK.B", "johnson & johnson": "JNJ", "procter & gamble": "PG",
    "coca cola": "KO", "coca-cola": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "exxon": "XOM", "chevron": "CVX", "pfizer": "PFE", "moderna": "MRNA",
    "airbnb": "ABNB", "snap": "SNAP", "snapchat": "SNAP", "pinterest": "PINS",
    "zoom": "ZM", "twitter": "X", "ibm": "IBM", "oracle": "ORCL",
    "cisco": "CSCO", "qualcomm": "QCOM", "broadcom": "AVGO",
    "target": "TGT", "home depot": "HD", "lowes": "LOW", "lowe's": "LOW",
    "nike": "NKE", "adidas": "ADDYY", "ford": "F", "gm": "GM",
    "general motors": "GM", "general electric": "GE", "ge": "GE",
    "lockheed": "LMT", "raytheon": "RTX", "caterpillar": "CAT",
    "3m": "MMM", "honeywell": "HON", "deere": "DE", "john deere": "DE",
}

# Common misspellings → corrections
SPELLING_CORRECTIONS: dict[str, str] = {
    "tesle": "tesla", "tessla": "tesla", "teslaa": "tesla",
    "aplle": "apple", "appel": "apple", "aple": "apple",
    "googel": "google", "gogle": "google", "gooogle": "google",
    "amazn": "amazon", "amazone": "amazon", "amzon": "amazon",
    "microsfot": "microsoft", "mircosoft": "microsoft", "microsft": "microsoft",
    "netflex": "netflix", "netfilx": "netflix",
    "nvidea": "nvidia", "nividia": "nvidia", "nvida": "nvidia",
    "facebok": "facebook", "faceboook": "facebook",
    "bitconi": "bitcoin", "bitcion": "bitcoin", "bitcoing": "bitcoin",
    "etherium": "ethereum", "etheruem": "ethereum", "ethreum": "ethereum",
    "stokc": "stock", "sotck": "stock", "stoock": "stock", "stonk": "stock", "stonks": "stocks",
    "stonck": "stock", "stck": "stock",
    "prise": "price", "pirce": "price", "proce": "price",
    "taret": "target", "tagret": "target", "taregt": "target",
    "reech": "reach", "raech": "reach",
    "bullsih": "bullish", "bulish": "bullish",
    "bearsh": "bearish", "bearisch": "bearish",
    "monht": "month", "motnh": "month", "mnoth": "month",
    "yaer": "year", "yera": "year",
    "waek": "week", "wek": "week", "weeek": "week",
    "quater": "quarter", "quartr": "quarter", "qaurter": "quarter",
    "prediciton": "prediction", "preidction": "prediction", "predicton": "prediction",
    "anaylsis": "analysis", "analsis": "analysis", "anlaysis": "analysis",
    "divdend": "dividend", "dividned": "dividend",
    "earnigns": "earnings", "earings": "earnings",
    "investmnet": "investment", "invesment": "investment",
    "portoflio": "portfolio", "porfolio": "portfolio",
    "volatilty": "volatility", "voaltility": "volatility",
    "sentimnet": "sentiment", "sentment": "sentiment",
    "forcast": "forecast", "forcecast": "forecast",
    "recesion": "recession", "reccession": "recession",
    "inflaiton": "inflation", "infaltion": "inflation",
    "revenu": "revenue", "reveune": "revenue",
}

# Gibberish pattern: sequences of consonants with no vowels (4+ chars)
_GIBBERISH_RE = re.compile(r"\b[^aeiou\s\d$]{5,}\b", re.IGNORECASE)
# Repeated character runs (e.g. "aaaa", "!!!!")
_REPEATED_CHARS_RE = re.compile(r"(.)\1{3,}")
# Non-alphanumeric junk (excluding $ and common punctuation)
_JUNK_RE = re.compile(r"[^\w\s$%.,?!'\"-]")


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"\$?[A-Z]{1,5}\b")
_PRICE_RE = re.compile(r"[\$\u20b9][\d,]+(?:\.\d+)?")
_PRICE_RE_USD = re.compile(r"\$[\d,]+(?:\.\d+)?")
_PRICE_RE_INR = re.compile(r"\u20b9[\d,]+(?:\.\d+)?")
# Also match bare numbers that look like price targets (e.g. "reach 3000", "hit 1500")
_BARE_PRICE_RE = re.compile(r"\b(?:reach|hit|touch|cross|above|below)\s+(\d{3,6})\b", re.IGNORECASE)
_TIMEFRAME_RE = re.compile(
    r"\b(by|before|until|within|this month|next month|this quarter|next quarter|this year|next year"
    r"|january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
    r"|q[1-4]|20\d{2}|\d+ ?(day|week|month|year)s?)\b",
    re.IGNORECASE,
)

HARDCODED_FINANCIAL_SUGGESTIONS_US = [
    "Will NVDA reach $150 by March 2026?",
    "Will Tesla stock go up in the next month?",
    "Will Bitcoin hit $100k this year?",
]

HARDCODED_FINANCIAL_SUGGESTIONS_IN = [
    "Will Reliance reach \u20b93000 by March 2026?",
    "Will TCS stock go up in the next month?",
    "Will Infosys beat earnings next quarter?",
]


class QueryAnalyzer:
    """Wraps FinancialQueryValidator with smarter two-stage analysis."""

    def __init__(self) -> None:
        self.logger = logger.bind(service="query_analyzer")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, query: str, market: str = "US") -> QueryAnalysisResult:
        """Analyze *query* and return classification + suggestions."""

        hardcoded = HARDCODED_FINANCIAL_SUGGESTIONS_IN if market == "IN" else HARDCODED_FINANCIAL_SUGGESTIONS_US

        if not query or not query.strip():
            return QueryAnalysisResult(
                category="non_financial",
                can_proceed=False,
                quality_score=0.0,
                issues=["Query is empty"],
                suggestions=hardcoded,
                message="Please enter a prediction query about financial markets.",
            )

        # --- Clean the query first ------------------------------------
        cleaned = self._clean_query(query, market=market)
        self.logger.debug("Query cleaned", original=query, cleaned=cleaned)

        # --- Stage 1: rule-based ----------------------------------------
        validation = financial_query_validator.validate_query(cleaned)
        quality_score = self._compute_quality_score(cleaned, market=market)

        if validation.is_valid and quality_score >= 0.7:
            return QueryAnalysisResult(
                category="clear",
                can_proceed=True,
                quality_score=quality_score,
                message="",
                cleaned_query=cleaned,
            )

        if validation.is_valid:
            category: Literal["vague", "non_financial"] = "vague"
            issues = self._identify_issues(cleaned, market=market)
            message = "Your query could be more specific for better predictions."
        else:
            category = "non_financial"
            issues = [validation.rejection_reason or "Query does not appear to be about financial markets."]
            message = "This doesn't look like a financial prediction query."

        # --- Stage 2: AI suggestions (only for vague / non_financial) ---
        suggestions = await self._generate_suggestions(cleaned, category, market=market)

        return QueryAnalysisResult(
            category=category,
            can_proceed=(category == "vague"),
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions,
            message=message,
            cleaned_query=cleaned,
        )

    # ------------------------------------------------------------------
    # Query cleaning
    # ------------------------------------------------------------------

    def _clean_query(self, query: str, market: str = "US") -> str:
        """Fix spelling, strip gibberish, normalize the query."""
        text = query.strip()

        # Remove junk characters
        text = _JUNK_RE.sub("", text)

        # Collapse repeated characters (e.g. "gooooogle" → "google")
        text = _REPEATED_CHARS_RE.sub(r"\1\1", text)

        # Remove gibberish words (long consonant-only sequences)
        text = _GIBBERISH_RE.sub("", text)

        # Apply spelling corrections (word-level)
        words = text.split()
        corrected: list[str] = []
        for word in words:
            # Preserve $TICKER and all-caps words
            if word.startswith("$") or word.isupper():
                corrected.append(word)
                continue
            lower = word.lower().strip(".,?!\"'")
            if lower in SPELLING_CORRECTIONS:
                replacement = SPELLING_CORRECTIONS[lower]
                # Preserve original casing style
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Re-attach trailing punctuation
                trailing = ""
                for ch in reversed(word):
                    if ch in ".,?!\"'":
                        trailing = ch + trailing
                    else:
                        break
                corrected.append(replacement + trailing)
            else:
                corrected.append(word)
        text = " ".join(corrected)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Ensure it ends with ? if it starts with a question word
        if text and text.split()[0].lower() in ("will", "can", "should", "is", "are", "does", "do", "would", "could"):
            if not text.endswith("?"):
                text = text.rstrip(".!") + "?"

        return text if text else query.strip()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_quality_score(self, query: str, market: str = "US") -> float:
        score = 0.0
        query_lower = query.lower()

        # Check for ticker or known company name
        has_ticker = bool(_TICKER_RE.search(query))
        if market == "IN":
            has_ticker = has_ticker or any(c in query_lower for c in INDIAN_COMPANY_TO_TICKER)
        else:
            has_ticker = has_ticker or any(c in query_lower for c in COMPANY_TICKER_MAP)
        if has_ticker:
            score += 0.3

        # Check for price target (₹ or $ or bare number after "reach/hit")
        if _PRICE_RE.search(query) or _BARE_PRICE_RE.search(query):
            score += 0.2
        if _TIMEFRAME_RE.search(query):
            score += 0.2
        if any(kw in query_lower for kw in financial_query_validator.FINANCIAL_KEYWORDS):
            score += 0.2
        if len(query.strip()) > 15:
            score += 0.1
        return min(score, 1.0)

    def _identify_issues(self, query: str, market: str = "US") -> list[str]:
        issues: list[str] = []
        query_lower = query.lower()

        # Check for ticker or known company name
        has_ticker = bool(_TICKER_RE.search(query))
        if market == "IN":
            has_ticker = has_ticker or any(c in query_lower for c in INDIAN_COMPANY_TO_TICKER)
        else:
            has_ticker = has_ticker or any(c in query_lower for c in COMPANY_TICKER_MAP)

        if not has_ticker:
            if market == "IN":
                issues.append("No ticker or company detected (e.g. RELIANCE, TCS, Infosys)")
            else:
                issues.append("No ticker symbol detected (e.g. AAPL, NVDA)")

        if not _PRICE_RE.search(query) and not _BARE_PRICE_RE.search(query):
            if market == "IN":
                issues.append("No price target specified (e.g. \u20b93000)")
            else:
                issues.append("No price target specified (e.g. $150)")

        if not _TIMEFRAME_RE.search(query):
            issues.append("No timeframe mentioned (e.g. 'by March 2026')")
        return issues

    # ------------------------------------------------------------------
    # Subject / ticker extraction for contextual suggestions
    # ------------------------------------------------------------------

    def _extract_subject_and_ticker(self, query: str, market: str = "US") -> tuple[str, str | None]:
        """Extract the company/asset name and its ticker from the query."""
        query_lower = query.lower()

        # For India market, check Indian companies first
        if market == "IN":
            for company in sorted(INDIAN_COMPANY_TO_TICKER.keys(), key=len, reverse=True):
                if company in query_lower:
                    ticker = INDIAN_COMPANY_TO_TICKER[company]
                    display = INDIAN_TICKER_TO_COMPANY.get(ticker, company.title())
                    return display, ticker
            # Also check for ₹ prefix
            ticker_match = re.search(r"\u20b9([A-Z]{1,10})\b", query)
            if ticker_match:
                raw = ticker_match.group(1)
                ticker = f"{raw}.NS" if not raw.endswith(".NS") else raw
                display = INDIAN_TICKER_TO_COMPANY.get(ticker, raw)
                return display, ticker

        # Check for explicit ticker ($TSLA or standalone TSLA)
        ticker_match = re.search(r"\$([A-Z]{1,5})\b", query)
        if ticker_match:
            ticker = ticker_match.group(1)
            # Find company name from reverse lookup
            for company, t in COMPANY_TICKER_MAP.items():
                if t == ticker:
                    return company.title(), ticker
            return ticker, ticker

        # Check for company names in query
        for company in sorted(COMPANY_TICKER_MAP.keys(), key=len, reverse=True):
            if company in query_lower:
                return company.title(), COMPANY_TICKER_MAP[company]

        return "", None

    # ------------------------------------------------------------------
    # Suggestion generation
    # ------------------------------------------------------------------

    async def _generate_suggestions(
        self, query: str, category: Literal["vague", "non_financial"], market: str = "US"
    ) -> list[str]:
        """Use contextual fallback directly (skip Gemini to save API costs).

        The contextual fallback already produces good suggestions based on the
        user's query, extracting tickers and timeframes from their input.
        """
        return self._contextual_fallback_suggestions(query, category, market=market)

    async def _call_gemini_suggestions(
        self, query: str, category: Literal["vague", "non_financial"]
    ) -> list[str]:
        from google.genai import types as genai_types
        from backend.app.services.prediction_engine import prediction_engine

        client = prediction_engine.gemini_client
        if not client:
            raise RuntimeError("Gemini client not available")

        if category == "vague":
            user_msg = (
                f"The user asked: \"{query}\"\n"
                f"This query is about financial markets but is too vague for our prediction engine. "
                f"Rephrase it into 2-3 more specific versions that keep the user's original intent "
                f"(same stock/asset, same direction) but add a ticker symbol, price target, and/or timeframe."
            )
        else:
            user_msg = (
                f"The user asked: \"{query}\"\n"
                f"This is not a financial query. Suggest 2-3 financial prediction questions that are "
                f"loosely related to the user's topic, or common popular stock predictions."
            )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_msg,
            config=genai_types.GenerateContentConfig(
                system_instruction=(
                    "You rephrase user queries into well-formed financial prediction questions. "
                    "Each suggestion MUST include: a ticker symbol (e.g. TSLA, AAPL), "
                    "a price target or direction, and a timeframe. "
                    "Keep the user's original intent and asset. "
                    "Respond ONLY with a JSON array of strings, no other text."
                ),
                max_output_tokens=200,
                temperature=0.7,
            ),
        )

        text = response.text.strip()
        suggestions = json.loads(text)
        if isinstance(suggestions, list) and len(suggestions) >= 1:
            return [str(s) for s in suggestions[:3]]
        raise ValueError("Unexpected Gemini response format")

    def _contextual_fallback_suggestions(
        self, query: str, category: Literal["vague", "non_financial"], market: str = "US"
    ) -> list[str]:
        """Build suggestions based on what the user actually asked."""
        hardcoded = HARDCODED_FINANCIAL_SUGGESTIONS_IN if market == "IN" else HARDCODED_FINANCIAL_SUGGESTIONS_US

        if category == "non_financial":
            return list(hardcoded)

        subject, ticker = self._extract_subject_and_ticker(query, market=market)

        has_ticker = bool(ticker)
        has_price = bool(_PRICE_RE.search(query) or _BARE_PRICE_RE.search(query))
        has_timeframe = bool(_TIMEFRAME_RE.search(query))

        # Extract the timeframe the user used, if any
        tf_match = _TIMEFRAME_RE.search(query)
        user_timeframe = tf_match.group(0) if tf_match else None

        # Market-specific defaults
        if market == "IN":
            ticker_str = ticker or "RELIANCE.NS"
            company_str = subject or "Reliance"
            currency = "\u20b9"
            default_prices = ("3000", "2500", "3500")
        else:
            ticker_str = ticker or "TSLA"
            company_str = subject or "Tesla"
            currency = "$"
            default_prices = ("250", "200", "300")

        tf = user_timeframe or "next 3 months"

        suggestions: list[str] = []

        if not has_ticker and not has_price:
            suggestions.append(f"Will {company_str} reach {currency}{default_prices[0]} by {tf}?")
            suggestions.append(f"Will {company_str} stock go up in the {tf}?")
            if has_timeframe:
                suggestions.append(f"Will {company_str} hit {currency}{default_prices[2]} {user_timeframe}?")
            else:
                suggestions.append(f"Will {company_str} rise by end of this quarter?")
        elif not has_price:
            suggestions.append(f"Will {company_str} reach {currency}{default_prices[0]} by {tf}?")
            suggestions.append(f"Will {company_str} go above {currency}{default_prices[1]} {tf}?")
        elif not has_timeframe:
            suggestions.append(f"Will {company_str} hit its target by end of this month?")
            suggestions.append(f"Will {company_str} reach that price within the next 3 months?")
        else:
            suggestions.append(f"Will {company_str} reach {currency}{default_prices[0]} by {tf}?")
            suggestions.append(f"Will {company_str} go above {currency}{default_prices[1]} {tf}?")

        return suggestions[:3]


# Singleton
query_analyzer = QueryAnalyzer()
