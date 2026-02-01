"""
Sentiment Analysis Service - Analyzes text sentiment from multiple sources.

Uses:
- VADER (specialized for social media)
- TextBlob (general purpose)
- Custom financial lexicon
"""

import re
from typing import Optional

from cachetools import TTLCache
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import structlog

from backend.app.config import settings

logger = structlog.get_logger()

# Cache sentiment results for 10 minutes
_sentiment_cache: TTLCache = TTLCache(maxsize=1000, ttl=600)


class SentimentService:
    """Service for analyzing text sentiment with financial focus."""

    # Financial-specific positive words (expanded)
    BULLISH_WORDS = {
        # Trading terms
        "buy", "bullish", "bulls", "long", "calls", "moon", "rocket", "breakthrough",
        "accumulate", "accumulating", "bullrun", "uptrend", "breakout",
        # Performance
        "beat", "beats", "beating", "exceeded", "exceeds", "exceeding",
        "outperform", "outperforms", "outperforming", "overweight",
        # Growth
        "strong", "stronger", "strength", "growth", "growing", "expand", "expanding",
        "surge", "surges", "surging", "rally", "rallies", "rallying",
        "soar", "soars", "soaring", "jump", "jumps", "jumping", "spike", "spiking",
        "gain", "gains", "gaining", "climb", "climbs", "climbing", "rise", "rises", "rising",
        # Financials
        "profit", "profits", "profitable", "profitability", "earnings",
        "record", "records", "high", "highs", "all-time", "boom", "booming",
        "revenue", "revenues", "dividend", "dividends",
        # Sentiment
        "positive", "optimistic", "optimism", "confident", "confidence",
        "undervalued", "opportunity", "opportunities", "winner", "winning", "wins",
        "success", "successful", "impressive", "excellent", "great", "amazing",
        # Innovation
        "innovative", "innovation", "disrupting", "disruptive", "revolutionary",
        "momentum", "tailwind", "tailwinds", "catalyst", "catalysts",
        # Analyst actions
        "upgrade", "upgrades", "upgraded", "upgrading", "raised", "raises", "raising",
        "reiterate", "reiterates", "reiterating", "overweight", "recommend", "recommends",
    }

    # Financial-specific negative words (expanded)
    BEARISH_WORDS = {
        # Trading terms
        "sell", "selling", "bearish", "bears", "short", "shorting", "puts",
        "crash", "crashes", "crashing", "dump", "dumping", "dumps",
        "downtrend", "breakdown", "capitulation",
        # Performance
        "miss", "misses", "missed", "missing", "disappoint", "disappoints", "disappointing",
        "underperform", "underperforms", "underperforming", "underweight",
        # Decline
        "decline", "declines", "declining", "fall", "falls", "falling", "fell",
        "drop", "drops", "dropping", "dropped", "plunge", "plunges", "plunging",
        "sink", "sinks", "sinking", "sank", "tumble", "tumbles", "tumbling",
        "slide", "slides", "sliding", "slump", "slumps", "slumping",
        "collapse", "collapses", "collapsing", "tank", "tanks", "tanking",
        # Financials
        "loss", "losses", "losing", "lost", "deficit", "deficits",
        "weak", "weaker", "weakness", "weakening", "soft", "softer", "softness",
        # Risk/Warning
        "concern", "concerns", "concerning", "concerned", "worried", "worries", "worry",
        "risk", "risks", "risky", "warning", "warnings", "warns", "warned", "caution",
        "fear", "fears", "fearing", "afraid", "panic", "panics", "panicking",
        "threat", "threatens", "threatening", "headwind", "headwinds",
        # Valuation
        "overvalued", "expensive", "bubble", "correction", "corrections",
        # Economic
        "recession", "recessionary", "inflation", "inflationary", "stagflation",
        "bankruptcy", "bankrupt", "insolvent", "default", "defaults", "defaulting",
        "layoffs", "layoff", "cuts", "cutting", "cut", "downsizing", "restructuring",
        # Sentiment
        "negative", "pessimistic", "pessimism", "gloomy", "bleak",
        "struggle", "struggles", "struggling", "struggled",
        "fail", "fails", "failed", "failing", "failure", "failures",
        # Legal/Scandal
        "investigation", "investigations", "investigating", "probe", "probes",
        "lawsuit", "lawsuits", "sue", "sues", "suing", "sued",
        "fraud", "fraudulent", "scandal", "scandals", "trouble", "troubles",
        "fine", "fines", "fined", "penalty", "penalties", "violation", "violations",
        # Analyst actions
        "downgrade", "downgrades", "downgraded", "downgrading",
        "lowered", "lowers", "lowering", "reduced", "reduces", "reducing",
        "underweight", "avoid", "avoids", "avoiding",
    }

    # Intensity modifiers
    INTENSIFIERS = {
        "very": 1.5,
        "extremely": 2.0,
        "significantly": 1.5,
        "slightly": 0.5,
        "somewhat": 0.7,
        "massive": 2.0,
        "huge": 1.8,
        "major": 1.5,
        "strongly": 1.5,
        "sharply": 1.5,
        "dramatically": 1.8,
        "substantially": 1.5,
    }

    # Bearish phrases (multi-word expressions)
    BEARISH_PHRASES = [
        "bears take control", "selling pressure", "profit taking",
        "death cross", "head and shoulders", "double top",
        "breaks support", "below support", "lost support",
        "red flags", "warning signs", "mixed signals",
        "take profits", "risk off", "flight to safety",
        "market correction", "bubble bursting", "liquidity crisis",
        "rate hike", "fed hawkish", "tightening policy",
        "earnings miss", "revenue miss", "guidance cut",
        "price target cut", "estimate cut", "rating cut",
    ]

    # Bullish phrases (multi-word expressions)
    BULLISH_PHRASES = [
        "bulls take control", "buying pressure", "short squeeze",
        "golden cross", "breakout above", "new highs",
        "breaks resistance", "above resistance", "clears resistance",
        "green flags", "positive signs", "strong signals",
        "buy the dip", "risk on", "flight to risk",
        "market rally", "bull market", "liquidity injection",
        "rate cut", "fed dovish", "easing policy",
        "earnings beat", "revenue beat", "guidance raise",
        "price target raise", "estimate raise", "rating upgrade",
        "record earnings", "record revenue", "record profit",
    ]

    def __init__(self):
        """Initialize sentiment analyzers."""
        self.logger = logger.bind(service="sentiment")
        self.vader = SentimentIntensityAnalyzer()

        # Add financial terms to VADER lexicon
        self._update_vader_lexicon()

    def _update_vader_lexicon(self):
        """Add financial terms to VADER's lexicon."""
        # Add bullish words with positive sentiment
        for word in self.BULLISH_WORDS:
            self.vader.lexicon[word] = 2.0

        # Add bearish words with negative sentiment
        for word in self.BEARISH_WORDS:
            self.vader.lexicon[word] = -2.0

    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of a text string.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score from -1 (bearish) to 1 (bullish)
        """
        if not text or not text.strip():
            return 0.0

        # Check cache
        cache_key = hash(text[:200])
        if cache_key in _sentiment_cache:
            return _sentiment_cache[cache_key]

        # Clean text
        cleaned = self._clean_text(text)

        # Get VADER sentiment
        vader_scores = self.vader.polarity_scores(cleaned)
        vader_compound = vader_scores["compound"]

        # Get TextBlob sentiment
        blob = TextBlob(cleaned)
        textblob_polarity = blob.sentiment.polarity

        # Get custom financial sentiment
        financial_score = self._analyze_financial_sentiment(cleaned)

        # Weighted average (VADER is best for social media)
        combined = (
            vader_compound * 0.5 +
            textblob_polarity * 0.2 +
            financial_score * 0.3
        )

        # Clamp to [-1, 1]
        result = max(-1, min(1, combined))

        # Cache result
        _sentiment_cache[cache_key] = result

        self.logger.debug(
            "Sentiment analyzed",
            text_preview=text[:50],
            vader=vader_compound,
            textblob=textblob_polarity,
            financial=financial_score,
            combined=result,
        )

        return round(result, 3)

    def _clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?'-]", " ", text)

        # Normalize whitespace
        text = " ".join(text.split())

        return text

    def _analyze_financial_sentiment(self, text: str) -> float:
        """
        Analyze sentiment using financial-specific lexicon and phrases.

        Args:
            text: Cleaned text to analyze

        Returns:
            Score from -1 to 1
        """
        text_lower = text.lower()
        words = text_lower.split()
        score = 0
        word_count = 0

        # First, check for multi-word phrases (stronger signal)
        for phrase in self.BEARISH_PHRASES:
            if phrase in text_lower:
                score -= 1.5  # Phrases are stronger signals
                word_count += 1

        for phrase in self.BULLISH_PHRASES:
            if phrase in text_lower:
                score += 1.5
                word_count += 1

        # Then check individual words
        i = 0
        while i < len(words):
            word = words[i]

            # Check for intensifier
            modifier = 1.0
            if i > 0 and words[i - 1] in self.INTENSIFIERS:
                modifier = self.INTENSIFIERS[words[i - 1]]

            if word in self.BULLISH_WORDS:
                score += 1 * modifier
                word_count += 1
            elif word in self.BEARISH_WORDS:
                score -= 1 * modifier
                word_count += 1

            i += 1

        if word_count == 0:
            return 0

        # Normalize to [-1, 1]
        return max(-1, min(1, score / max(word_count, 1)))

    def analyze_multiple(self, texts: list[str]) -> dict:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of text strings

        Returns:
            Dict with aggregate sentiment metrics
        """
        if not texts:
            return {
                "average_score": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "scores": [],
            }

        scores = [self.analyze_text(text) for text in texts]

        bullish = sum(1 for s in scores if s > 0.1)
        bearish = sum(1 for s in scores if s < -0.1)
        neutral = len(scores) - bullish - bearish

        return {
            "average_score": round(sum(scores) / len(scores), 3),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "scores": scores,
            "bullish_percentage": round(bullish / len(scores) * 100, 1) if scores else 0,
        }

    def get_sentiment_label(self, score: float) -> str:
        """
        Convert score to human-readable label.

        Args:
            score: Sentiment score (-1 to 1)

        Returns:
            Sentiment label
        """
        if score >= 0.5:
            return "very_bullish"
        elif score >= 0.15:
            return "bullish"
        elif score <= -0.5:
            return "very_bearish"
        elif score <= -0.15:
            return "bearish"
        else:
            return "neutral"


# Create singleton instance
sentiment_service = SentimentService()
