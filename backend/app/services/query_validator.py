"""
Financial Query Validator - Guardrails for ensuring only financial queries are processed.

This module validates that incoming queries are related to financial markets
and rejects non-financial queries with helpful feedback.
"""

import re
from typing import Optional
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class QueryValidationResult:
    """Result of query validation."""

    is_valid: bool
    rejection_reason: Optional[str] = None
    suggestions: list[str] = None
    matched_keywords: list[str] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
        if self.matched_keywords is None:
            self.matched_keywords = []


class FinancialQueryValidator:
    """
    Validates that queries are related to financial markets.

    This validator ensures the prediction engine only processes
    legitimate financial queries and rejects off-topic requests.
    """

    # Financial keywords - query must match at least one
    FINANCIAL_KEYWORDS = {
        # Instruments
        "stock", "stocks", "share", "shares", "equity", "equities",
        "crypto", "cryptocurrency", "bitcoin", "btc", "ethereum", "eth",
        "forex", "fx", "currency", "currencies",
        "commodity", "commodities", "gold", "silver", "oil", "crude", "natural gas",
        "index", "indices", "s&p", "nasdaq", "dow", "russell",
        "etf", "etfs", "fund", "funds",
        "futures", "future", "contract", "contracts",
        "bond", "bonds", "treasury", "treasuries", "yield", "yields",
        "option", "options", "call", "put", "strike",

        # Price actions
        "price", "prices", "reach", "hit", "target", "level",
        "rise", "fall", "drop", "climb", "surge", "plunge", "rally", "crash",
        "increase", "decrease", "gain", "lose", "up", "down",
        "bull", "bullish", "bear", "bearish",
        "breakout", "breakdown", "support", "resistance",
        "high", "low", "ath", "all-time",

        # Trading
        "buy", "sell", "trade", "trading", "invest", "investment",
        "long", "short", "position", "portfolio",
        "market", "markets", "exchange",

        # Analysis
        "earnings", "revenue", "profit", "eps", "pe", "ratio",
        "dividend", "dividends", "payout",
        "analyst", "analysts", "forecast", "outlook", "prediction",
        "technical", "fundamental", "sentiment",
        "rsi", "macd", "moving average", "sma", "ema",
        "volatility", "vix", "beta",

        # Companies/tickers (common patterns)
        "ticker", "symbol", "company", "corp", "inc",
        "aapl", "msft", "googl", "amzn", "nvda", "tsla", "meta",
        "apple", "microsoft", "google", "amazon", "nvidia", "tesla",

        # Financial events
        "ipo", "merger", "acquisition", "split", "buyback",
        "fed", "fomc", "interest rate", "inflation",
        "gdp", "jobs", "employment", "economic",
    }

    # Blocked topics - reject if any of these are matched
    BLOCKED_TOPICS = {
        # Weather
        "weather", "forecast weather", "temperature", "rain", "snow", "sunny", "cloudy",
        "hurricane", "tornado", "climate",

        # Food/Recipes
        "recipe", "recipes", "cook", "cooking", "bake", "baking",
        "ingredient", "ingredients", "food", "meal", "dish",

        # Health/Medical
        "health", "medical", "doctor", "hospital", "symptom", "symptoms",
        "disease", "illness", "medicine", "treatment", "diagnosis",
        "diet", "exercise", "workout", "fitness",

        # Entertainment
        "movie", "movies", "film", "films", "tv", "television", "show", "shows",
        "music", "song", "songs", "album", "artist", "band",
        "game", "games", "gaming", "video game", "sports score",
        "celebrity", "celebrities", "actor", "actress",

        # General knowledge
        "history", "historical", "geography", "science", "math",
        "physics", "chemistry", "biology",

        # Creative writing
        "write me", "write a", "poem", "poetry", "story", "stories",
        "essay", "article", "blog", "creative",
        "joke", "jokes", "funny", "humor",

        # Programming (non-financial)
        "code", "coding", "programming", "program", "software",
        "python", "javascript", "java", "html", "css",
        "debug", "bug", "error", "function", "class",

        # Personal/Lifestyle
        "relationship", "dating", "love", "marriage",
        "travel", "vacation", "trip", "hotel", "flight",
        "fashion", "clothes", "outfit", "style",

        # Other
        "translate", "translation", "language",
        "homework", "assignment", "exam", "test",
        "summarize this", "explain this", "what is",
    }

    # Patterns that indicate non-financial queries
    BLOCKED_PATTERNS = [
        r"^(hi|hello|hey|howdy|greetings)\b",  # Greetings
        r"^(who|what|where|when|why|how) (is|are|was|were|do|does|did)\b(?!.*(stock|price|market|trade|crypto|forex|gold|oil|index|etf|bond))",  # General questions
        r"write (me )?(a |an )?",  # Writing requests
        r"tell (me )?(a |an )?(joke|story|fact)",  # Entertainment requests
        r"^can you\b(?!.*(predict|analyze|forecast|stock|price|market))",  # General capability questions
        r"^what do you think about\b(?!.*(stock|price|market|trade|crypto|company|ticker))",  # Opinion requests
    ]

    def __init__(self):
        """Initialize the validator."""
        self.logger = logger.bind(service="query_validator")
        # Compile blocked patterns for efficiency
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.BLOCKED_PATTERNS]

    def validate_query(self, query: str) -> QueryValidationResult:
        """
        Validate that a query is related to financial markets.

        Args:
            query: The user's prediction query

        Returns:
            QueryValidationResult with validation status and details
        """
        if not query or not query.strip():
            return QueryValidationResult(
                is_valid=False,
                rejection_reason="Query cannot be empty.",
                suggestions=["Try asking about a stock price prediction, e.g., 'Will AAPL reach $200?'"]
            )

        query_lower = query.lower().strip()

        # Check for blocked patterns first
        for pattern in self._compiled_patterns:
            if pattern.search(query_lower):
                self.logger.info("Query rejected by pattern", query=query[:50])
                return QueryValidationResult(
                    is_valid=False,
                    rejection_reason="I can only analyze financial market predictions. This query doesn't appear to be about financial markets.",
                    suggestions=[
                        "Ask about stock price predictions: 'Will NVDA reach $150?'",
                        "Ask about crypto: 'Will Bitcoin hit $100k?'",
                        "Ask about market direction: 'Will Tesla stock go up?'",
                    ]
                )

        # Check for blocked topics
        blocked_matches = []
        for topic in self.BLOCKED_TOPICS:
            if topic in query_lower:
                blocked_matches.append(topic)

        if blocked_matches:
            self.logger.info("Query contains blocked topics",
                          query=query[:50],
                          blocked=blocked_matches[:3])
            return QueryValidationResult(
                is_valid=False,
                rejection_reason=f"I can only analyze financial market predictions. Your query appears to be about: {', '.join(blocked_matches[:3])}.",
                suggestions=[
                    "Ask about stock price predictions: 'Will AAPL reach $200?'",
                    "Ask about crypto: 'Will Ethereum reach $5000?'",
                    "Ask about forex: 'Will EUR/USD reach 1.15?'",
                    "Ask about commodities: 'Will gold reach $2500?'",
                ]
            )

        # Check for financial keywords
        matched_keywords = []
        for keyword in self.FINANCIAL_KEYWORDS:
            # Use word boundary matching for short keywords
            if len(keyword) <= 3:
                pattern = rf"\b{re.escape(keyword)}\b"
                if re.search(pattern, query_lower):
                    matched_keywords.append(keyword)
            else:
                if keyword in query_lower:
                    matched_keywords.append(keyword)

        # Check for ticker patterns ($AAPL, NVDA, etc.)
        ticker_pattern = r"\$?[A-Z]{1,5}\b"
        if re.search(ticker_pattern, query):
            matched_keywords.append("ticker_symbol")

        # Check for price patterns ($150, $100k, etc.)
        price_pattern = r"\$[\d,]+(?:\.\d+)?[km]?\b"
        if re.search(price_pattern, query_lower):
            matched_keywords.append("price_target")

        if matched_keywords:
            self.logger.debug("Query validated as financial",
                           query=query[:50],
                           keywords=matched_keywords[:5])
            return QueryValidationResult(
                is_valid=True,
                matched_keywords=matched_keywords
            )

        # No financial keywords found
        self.logger.info("Query rejected - no financial keywords", query=query[:50])
        return QueryValidationResult(
            is_valid=False,
            rejection_reason="I couldn't identify any financial market context in your query. Please ask about stocks, crypto, forex, commodities, or other financial instruments.",
            suggestions=[
                "Include a ticker symbol: 'Will NVDA reach $150?'",
                "Mention the asset type: 'Will Bitcoin price increase?'",
                "Ask about market direction: 'Is Apple stock bullish?'",
                "Ask about price targets: 'Will gold hit $2500?'",
            ]
        )

    def is_financial_query(self, query: str) -> bool:
        """
        Quick check if a query is financial.

        Args:
            query: The user's query

        Returns:
            True if the query appears to be financial
        """
        return self.validate_query(query).is_valid


# Singleton instance
financial_query_validator = FinancialQueryValidator()
