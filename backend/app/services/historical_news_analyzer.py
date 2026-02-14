"""
Historical News Analyzer - Learn from past news impact on stock prices.

This service analyzes how similar news events in the past affected stock prices,
providing historical context for current news sentiment analysis.
"""

from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class NewsImpactEvent:
    """A historical news event and its price impact."""

    date: datetime
    headline: str
    sentiment_score: float  # -1 to +1
    category: str  # earnings, product, legal, macro, analyst, etc.
    price_before: float
    price_after_1d: Optional[float] = None
    price_after_5d: Optional[float] = None
    price_after_30d: Optional[float] = None
    impact_1d: Optional[float] = None  # % change
    impact_5d: Optional[float] = None
    impact_30d: Optional[float] = None


@dataclass
class HistoricalNewsPattern:
    """A pattern learned from historical news analysis."""

    pattern_type: str  # e.g., "earnings_beat", "product_launch", "analyst_upgrade"
    avg_impact_1d: float
    avg_impact_5d: float
    avg_impact_30d: float
    occurrence_count: int
    confidence: float  # How confident we are in this pattern
    description: str


@dataclass
class HistoricalNewsAnalysis:
    """Complete historical news analysis for a stock."""

    ticker: str
    analysis_period_days: int
    total_news_events: int
    patterns_found: list[HistoricalNewsPattern] = field(default_factory=list)
    similar_current_events: list[dict] = field(default_factory=list)
    historical_sentiment_accuracy: float = 0.5  # How predictive past sentiment was
    recommended_news_weight: float = 0.15  # Adjusted weight based on accuracy
    signal_score: float = 0.0  # Overall historical news signal
    reasoning: str = ""


class HistoricalNewsAnalyzer:
    """
    Analyzes historical news and their impact on stock prices.

    This service:
    1. Retrieves historical news for a stock
    2. Correlates news sentiment with subsequent price movements
    3. Identifies patterns (e.g., "earnings beat → +5% in 5 days")
    4. Applies learnings to current news sentiment
    """

    # News category keywords
    NEWS_CATEGORIES = {
        "earnings": ["earnings", "eps", "revenue", "profit", "quarter", "fiscal", "beat", "miss", "guidance"],
        "product": ["launch", "product", "release", "announce", "unveil", "new", "innovation"],
        "analyst": ["upgrade", "downgrade", "price target", "rating", "analyst", "buy", "sell", "hold"],
        "legal": ["lawsuit", "sec", "investigation", "settlement", "fine", "regulatory", "compliance"],
        "management": ["ceo", "cfo", "executive", "resign", "hire", "appoint", "board"],
        "macro": ["fed", "interest rate", "inflation", "recession", "economy", "tariff", "trade"],
        "acquisition": ["acquire", "merger", "buyout", "deal", "acquisition", "takeover"],
        "dividend": ["dividend", "payout", "yield", "distribution", "buyback", "repurchase"],
    }

    # Historical impact patterns (baseline from research)
    # These get refined per-stock based on actual data
    BASELINE_PATTERNS = {
        "earnings_beat": {"1d": 3.2, "5d": 4.1, "30d": 5.5, "confidence": 0.65},
        "earnings_miss": {"1d": -4.5, "5d": -5.2, "30d": -3.8, "confidence": 0.70},
        "analyst_upgrade": {"1d": 2.1, "5d": 3.5, "30d": 4.2, "confidence": 0.55},
        "analyst_downgrade": {"1d": -2.8, "5d": -3.9, "30d": -2.5, "confidence": 0.60},
        "product_launch": {"1d": 1.5, "5d": 2.8, "30d": 4.0, "confidence": 0.45},
        "acquisition_announced": {"1d": 5.5, "5d": 4.2, "30d": 3.0, "confidence": 0.50},
        "legal_negative": {"1d": -3.5, "5d": -4.0, "30d": -2.5, "confidence": 0.55},
        "management_change": {"1d": -1.0, "5d": 0.5, "30d": 1.5, "confidence": 0.40},
    }

    def __init__(self):
        """Initialize the historical news analyzer."""
        self.logger = logger.bind(service="historical_news_analyzer")

    def categorize_news(self, headline: str) -> str:
        """
        Categorize a news headline.

        Args:
            headline: News headline text

        Returns:
            Category string
        """
        headline_lower = headline.lower()

        for category, keywords in self.NEWS_CATEGORIES.items():
            if any(kw in headline_lower for kw in keywords):
                return category

        return "general"

    def analyze_historical_news(
        self,
        ticker: str,
        historical_prices: list[dict],  # [{date, open, high, low, close, volume}]
        historical_news: list[dict],  # [{date, title, sentiment_score}]
        current_news_sentiment: float,
        lookback_days: int = 180,
    ) -> HistoricalNewsAnalysis:
        """
        Analyze historical news and their price impact.

        Args:
            ticker: Stock ticker
            historical_prices: Historical OHLCV data
            historical_news: Historical news with sentiment scores
            current_news_sentiment: Current news sentiment score
            lookback_days: How far back to analyze

        Returns:
            HistoricalNewsAnalysis with patterns and recommendations
        """
        self.logger.info("Analyzing historical news", ticker=ticker, lookback_days=lookback_days)

        analysis = HistoricalNewsAnalysis(
            ticker=ticker,
            analysis_period_days=lookback_days,
            total_news_events=len(historical_news),
        )

        if not historical_prices or not historical_news:
            analysis.reasoning = "Insufficient historical data for analysis"
            analysis.signal_score = 0.0
            return analysis

        # Build price lookup by date
        price_by_date = {}
        sorted_prices = sorted(historical_prices, key=lambda x: x.get("date", ""))
        for i, p in enumerate(sorted_prices):
            date_str = str(p.get("date", ""))[:10]
            price_by_date[date_str] = {
                "close": p.get("close", p.get("Close", 0)),
                "index": i,
            }

        # Analyze each news event's impact
        news_impacts = []
        category_impacts = {}

        for news in historical_news:
            news_date = news.get("date") or news.get("published_at")
            if not news_date:
                continue

            if isinstance(news_date, datetime):
                date_str = news_date.strftime("%Y-%m-%d")
            else:
                date_str = str(news_date)[:10]

            if date_str not in price_by_date:
                continue

            price_data = price_by_date[date_str]
            price_before = price_data["close"]
            idx = price_data["index"]

            if price_before <= 0:
                continue

            # Get subsequent prices
            impact_1d = None
            impact_5d = None
            impact_30d = None

            if idx + 1 < len(sorted_prices):
                price_1d = sorted_prices[idx + 1].get("close", sorted_prices[idx + 1].get("Close", 0))
                if price_1d > 0:
                    impact_1d = ((price_1d - price_before) / price_before) * 100

            if idx + 5 < len(sorted_prices):
                price_5d = sorted_prices[idx + 5].get("close", sorted_prices[idx + 5].get("Close", 0))
                if price_5d > 0:
                    impact_5d = ((price_5d - price_before) / price_before) * 100

            if idx + 22 < len(sorted_prices):  # ~1 month of trading days
                price_30d = sorted_prices[idx + 22].get("close", sorted_prices[idx + 22].get("Close", 0))
                if price_30d > 0:
                    impact_30d = ((price_30d - price_before) / price_before) * 100

            headline = news.get("title", news.get("headline", ""))
            sentiment = news.get("sentiment_score", 0)
            category = self.categorize_news(headline)

            impact_event = {
                "date": date_str,
                "headline": headline,
                "sentiment": sentiment,
                "category": category,
                "impact_1d": impact_1d,
                "impact_5d": impact_5d,
                "impact_30d": impact_30d,
            }
            news_impacts.append(impact_event)

            # Aggregate by category
            if category not in category_impacts:
                category_impacts[category] = []
            category_impacts[category].append(impact_event)

        # Calculate sentiment-to-impact correlation
        sentiment_correct = 0
        sentiment_total = 0

        for event in news_impacts:
            if event["sentiment"] != 0 and event["impact_5d"] is not None:
                sentiment_total += 1
                # Check if sentiment direction matched price direction
                if (event["sentiment"] > 0 and event["impact_5d"] > 0) or \
                   (event["sentiment"] < 0 and event["impact_5d"] < 0):
                    sentiment_correct += 1

        if sentiment_total > 0:
            analysis.historical_sentiment_accuracy = sentiment_correct / sentiment_total
        else:
            analysis.historical_sentiment_accuracy = 0.5

        # Identify patterns from categories
        for category, events in category_impacts.items():
            if len(events) < 3:  # Need minimum events for pattern
                continue

            # Separate by sentiment
            positive_events = [e for e in events if e.get("sentiment", 0) > 0.1]
            negative_events = [e for e in events if e.get("sentiment", 0) < -0.1]

            for sentiment_group, group_name in [(positive_events, "positive"), (negative_events, "negative")]:
                if len(sentiment_group) < 2:
                    continue

                impacts_1d = [e["impact_1d"] for e in sentiment_group if e["impact_1d"] is not None]
                impacts_5d = [e["impact_5d"] for e in sentiment_group if e["impact_5d"] is not None]
                impacts_30d = [e["impact_30d"] for e in sentiment_group if e["impact_30d"] is not None]

                if not impacts_1d:
                    continue

                avg_1d = sum(impacts_1d) / len(impacts_1d)
                avg_5d = sum(impacts_5d) / len(impacts_5d) if impacts_5d else avg_1d
                avg_30d = sum(impacts_30d) / len(impacts_30d) if impacts_30d else avg_5d

                # Calculate confidence based on consistency
                if len(impacts_5d) > 1:
                    variance = sum((x - avg_5d) ** 2 for x in impacts_5d) / len(impacts_5d)
                    std_dev = variance ** 0.5
                    consistency = max(0.3, 1 - (std_dev / 10))  # Lower variance = higher confidence
                else:
                    consistency = 0.4

                pattern = HistoricalNewsPattern(
                    pattern_type=f"{category}_{group_name}",
                    avg_impact_1d=round(avg_1d, 2),
                    avg_impact_5d=round(avg_5d, 2),
                    avg_impact_30d=round(avg_30d, 2),
                    occurrence_count=len(sentiment_group),
                    confidence=round(consistency, 2),
                    description=f"{group_name.title()} {category} news historically led to {avg_5d:+.1f}% move over 5 days",
                )
                analysis.patterns_found.append(pattern)

        # Find similar current events based on current sentiment
        if current_news_sentiment > 0.1:
            relevant_patterns = [p for p in analysis.patterns_found if "positive" in p.pattern_type]
        elif current_news_sentiment < -0.1:
            relevant_patterns = [p for p in analysis.patterns_found if "negative" in p.pattern_type]
        else:
            relevant_patterns = []

        # Calculate historical news signal
        if relevant_patterns:
            weighted_impact = 0
            total_weight = 0

            for pattern in relevant_patterns:
                weight = pattern.confidence * pattern.occurrence_count
                weighted_impact += pattern.avg_impact_5d * weight
                total_weight += weight

            if total_weight > 0:
                expected_impact = weighted_impact / total_weight
                # Convert expected % impact to signal score (-1 to +1)
                # Assume ±10% is extreme
                analysis.signal_score = max(-1, min(1, expected_impact / 10))

                analysis.similar_current_events = [
                    {
                        "pattern": p.pattern_type,
                        "expected_impact": p.avg_impact_5d,
                        "confidence": p.confidence,
                        "occurrences": p.occurrence_count,
                    }
                    for p in relevant_patterns[:3]
                ]

        # Adjust recommended weight based on historical accuracy
        if analysis.historical_sentiment_accuracy > 0.6:
            analysis.recommended_news_weight = 0.18  # Increase weight
        elif analysis.historical_sentiment_accuracy < 0.4:
            analysis.recommended_news_weight = 0.10  # Decrease weight
        else:
            analysis.recommended_news_weight = 0.15  # Default

        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Analyzed {len(news_impacts)} news events over {lookback_days} days.")

        if analysis.historical_sentiment_accuracy != 0.5:
            acc_pct = analysis.historical_sentiment_accuracy * 100
            reasoning_parts.append(f"Historical sentiment accuracy: {acc_pct:.0f}%.")

        if analysis.patterns_found:
            top_pattern = max(analysis.patterns_found, key=lambda p: p.confidence * p.occurrence_count)
            reasoning_parts.append(f"Key pattern: {top_pattern.description}.")

        if analysis.signal_score != 0:
            direction = "bullish" if analysis.signal_score > 0 else "bearish"
            reasoning_parts.append(f"Historical news signal: {direction} ({analysis.signal_score:+.2f}).")

        analysis.reasoning = " ".join(reasoning_parts)

        self.logger.info(
            "Historical news analysis complete",
            ticker=ticker,
            patterns_found=len(analysis.patterns_found),
            signal_score=analysis.signal_score,
        )

        return analysis

    def get_baseline_pattern(self, pattern_type: str) -> Optional[dict]:
        """
        Get baseline pattern data when historical data is limited.

        Args:
            pattern_type: Type of pattern (e.g., "earnings_beat")

        Returns:
            Baseline pattern data or None
        """
        return self.BASELINE_PATTERNS.get(pattern_type)


# Singleton instance
historical_news_analyzer = HistoricalNewsAnalyzer()
