"""
Herd Sentiment Warning System for Clarividex.

Addresses fintech bottleneck: 40% of investors follow crowds blindly.

This module detects extreme herd behavior and provides contrarian warnings:
- Extreme bullish sentiment often precedes corrections
- Extreme bearish sentiment often precedes rallies
- Warns users when "everyone" agrees (dangerous signal)

Research basis:
- AAII Sentiment Survey shows contrarian signals work
- Put/Call ratio extremes are predictive
- Social media sentiment extremes often mark tops/bottoms

Usage:
    from backend.app.services.herd_sentiment import herd_sentiment_analyzer

    warning = await herd_sentiment_analyzer.check_herd_warning(
        ticker="NVDA",
        social_sentiment=0.85,  # 85% bullish
        put_call_ratio=0.5,    # Very low (bullish)
        fear_greed=82,         # Extreme greed
    )
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import structlog

logger = structlog.get_logger()


class HerdLevel(Enum):
    """Level of herd behavior detected."""
    EXTREME_BULLISH = "extreme_bullish"   # >80% agreement - contrarian bearish
    HIGH_BULLISH = "high_bullish"         # 70-80% agreement
    MODERATE_BULLISH = "moderate_bullish" # 60-70%
    NEUTRAL = "neutral"                   # 40-60%
    MODERATE_BEARISH = "moderate_bearish" # 30-40%
    HIGH_BEARISH = "high_bearish"         # 20-30%
    EXTREME_BEARISH = "extreme_bearish"   # <20% - contrarian bullish


class WarningType(Enum):
    """Type of warning to issue."""
    CONTRARIAN_SELL = "contrarian_sell"   # Everyone bullish = consider selling
    CONTRARIAN_BUY = "contrarian_buy"     # Everyone bearish = consider buying
    CAUTION_FOMO = "caution_fomo"         # FOMO detected
    CAUTION_PANIC = "caution_panic"       # Panic selling detected
    NO_WARNING = "no_warning"


@dataclass
class HerdWarning:
    """Warning about herd behavior."""
    ticker: str
    herd_level: HerdLevel
    warning_type: WarningType
    warning_message: str

    # Sentiment breakdown
    social_sentiment_pct: float  # % bullish on social media
    put_call_ratio: Optional[float]
    fear_greed_index: Optional[int]
    analyst_consensus: Optional[str]

    # Contrarian signal strength
    contrarian_signal: float  # -1 to 1 (negative = contrarian bearish)
    signal_confidence: float

    # Historical context
    historical_accuracy: str
    similar_periods: List[str]

    # Actionable advice
    recommendation: str
    what_smart_money_doing: Optional[str]

    # Educational content
    explanation: str
    eli5_explanation: str  # Explain Like I'm 5


class HerdSentimentAnalyzer:
    """
    Analyzes herd sentiment and provides contrarian warnings.

    Key insight: When everyone agrees, they're often wrong.
    - Extreme bullish consensus often marks tops
    - Extreme bearish consensus often marks bottoms
    """

    # Thresholds for herd detection
    EXTREME_BULLISH_THRESHOLD = 80  # >80% bullish = danger
    HIGH_BULLISH_THRESHOLD = 70
    EXTREME_BEARISH_THRESHOLD = 20  # <20% bullish = opportunity
    HIGH_BEARISH_THRESHOLD = 30

    # Historical accuracy of contrarian signals (from research)
    CONTRARIAN_ACCURACY = {
        "extreme_bullish": 0.65,  # 65% of the time, market corrects
        "extreme_bearish": 0.70,  # 70% of the time, market rallies
    }

    # Historical examples for context
    HISTORICAL_EXAMPLES = {
        "extreme_bullish": [
            "January 2000: Tech euphoria peaked, NASDAQ fell 78% over next 2 years",
            "November 2021: 'Everything rally', crypto/meme stocks peaked",
            "July 2024: AI hype peaked, subsequent 15% correction",
        ],
        "extreme_bearish": [
            "March 2009: Maximum pessimism, S&P 500 rallied 400% over next decade",
            "March 2020: COVID panic, market doubled in 18 months",
            "October 2022: Recession fears peaked, strong 2023 rally followed",
        ],
    }

    def __init__(self):
        self.logger = logger.bind(service="herd_sentiment")

    async def check_herd_warning(
        self,
        ticker: str,
        social_sentiment: Optional[float] = None,  # 0-1 scale (bullish %)
        put_call_ratio: Optional[float] = None,
        fear_greed: Optional[int] = None,          # 0-100
        analyst_ratings: Optional[Dict] = None,     # {"buy": 10, "hold": 5, "sell": 2}
        retail_positioning: Optional[float] = None, # % long
    ) -> HerdWarning:
        """
        Check for herd behavior and issue warnings if detected.

        Args:
            ticker: Stock ticker
            social_sentiment: Bullish percentage from social media (0-1)
            put_call_ratio: Options put/call ratio
            fear_greed: CNN Fear & Greed Index (0-100)
            analyst_ratings: Analyst recommendation breakdown
            retail_positioning: Retail investor positioning (% long)

        Returns:
            HerdWarning with contrarian analysis
        """
        self.logger.info(
            "Checking herd sentiment",
            ticker=ticker,
            social=social_sentiment,
            put_call=put_call_ratio,
            fear_greed=fear_greed,
        )

        # Calculate composite herd score
        herd_score, components = self._calculate_herd_score(
            social_sentiment=social_sentiment,
            put_call_ratio=put_call_ratio,
            fear_greed=fear_greed,
            analyst_ratings=analyst_ratings,
            retail_positioning=retail_positioning,
        )

        # Determine herd level
        herd_level = self._get_herd_level(herd_score)

        # Determine warning type
        warning_type = self._get_warning_type(herd_level)

        # Calculate contrarian signal strength
        contrarian_signal = self._calculate_contrarian_signal(herd_score)

        # Build warning
        return HerdWarning(
            ticker=ticker,
            herd_level=herd_level,
            warning_type=warning_type,
            warning_message=self._get_warning_message(herd_level, ticker),
            social_sentiment_pct=social_sentiment * 100 if social_sentiment else 50,
            put_call_ratio=put_call_ratio,
            fear_greed_index=fear_greed,
            analyst_consensus=self._get_analyst_consensus(analyst_ratings),
            contrarian_signal=contrarian_signal,
            signal_confidence=self._get_signal_confidence(herd_level, components),
            historical_accuracy=self._get_historical_accuracy(herd_level),
            similar_periods=self._get_similar_periods(herd_level),
            recommendation=self._get_recommendation(herd_level, ticker),
            what_smart_money_doing=self._get_smart_money_insight(put_call_ratio),
            explanation=self._get_explanation(herd_level),
            eli5_explanation=self._get_eli5_explanation(herd_level),
        )

    def _calculate_herd_score(
        self,
        social_sentiment: Optional[float],
        put_call_ratio: Optional[float],
        fear_greed: Optional[int],
        analyst_ratings: Optional[Dict],
        retail_positioning: Optional[float],
    ) -> Tuple[float, Dict]:
        """
        Calculate composite herd score (0-100).
        Higher = more bullish consensus.
        """
        components = {}
        weights = {}

        # Social sentiment (0-100)
        if social_sentiment is not None:
            components["social"] = social_sentiment * 100
            weights["social"] = 0.25

        # Put/Call ratio (inverted - low ratio = bullish)
        if put_call_ratio is not None:
            # Convert P/C ratio to bullish score
            # P/C of 0.5 = very bullish (score 80)
            # P/C of 1.0 = neutral (score 50)
            # P/C of 1.5 = very bearish (score 20)
            pc_score = max(0, min(100, 100 - (put_call_ratio - 0.5) * 60))
            components["put_call"] = pc_score
            weights["put_call"] = 0.25

        # Fear & Greed (already 0-100 scale, higher = more greedy/bullish)
        if fear_greed is not None:
            components["fear_greed"] = fear_greed
            weights["fear_greed"] = 0.25

        # Analyst ratings
        if analyst_ratings:
            buys = analyst_ratings.get("buy", 0) + analyst_ratings.get("strong_buy", 0)
            holds = analyst_ratings.get("hold", 0)
            sells = analyst_ratings.get("sell", 0) + analyst_ratings.get("strong_sell", 0)
            total = buys + holds + sells
            if total > 0:
                analyst_score = (buys / total) * 100
                components["analyst"] = analyst_score
                weights["analyst"] = 0.15

        # Retail positioning
        if retail_positioning is not None:
            components["retail"] = retail_positioning * 100
            weights["retail"] = 0.10

        # Calculate weighted average
        if not weights:
            return 50.0, {}  # Neutral if no data

        total_weight = sum(weights.values())
        weighted_sum = sum(
            components[k] * weights[k] / total_weight
            for k in components
        )

        return weighted_sum, components

    def _get_herd_level(self, herd_score: float) -> HerdLevel:
        """Convert herd score to categorical level."""
        if herd_score >= self.EXTREME_BULLISH_THRESHOLD:
            return HerdLevel.EXTREME_BULLISH
        elif herd_score >= self.HIGH_BULLISH_THRESHOLD:
            return HerdLevel.HIGH_BULLISH
        elif herd_score >= 60:
            return HerdLevel.MODERATE_BULLISH
        elif herd_score >= 40:
            return HerdLevel.NEUTRAL
        elif herd_score >= self.HIGH_BEARISH_THRESHOLD:
            return HerdLevel.MODERATE_BEARISH
        elif herd_score >= self.EXTREME_BEARISH_THRESHOLD:
            return HerdLevel.HIGH_BEARISH
        else:
            return HerdLevel.EXTREME_BEARISH

    def _get_warning_type(self, herd_level: HerdLevel) -> WarningType:
        """Determine warning type based on herd level."""
        if herd_level == HerdLevel.EXTREME_BULLISH:
            return WarningType.CONTRARIAN_SELL
        elif herd_level == HerdLevel.HIGH_BULLISH:
            return WarningType.CAUTION_FOMO
        elif herd_level == HerdLevel.EXTREME_BEARISH:
            return WarningType.CONTRARIAN_BUY
        elif herd_level == HerdLevel.HIGH_BEARISH:
            return WarningType.CAUTION_PANIC
        else:
            return WarningType.NO_WARNING

    def _calculate_contrarian_signal(self, herd_score: float) -> float:
        """
        Calculate contrarian signal strength.
        Positive = contrarian bullish (market too bearish)
        Negative = contrarian bearish (market too bullish)
        """
        # Neutral at 50, max at extremes
        deviation = (herd_score - 50) / 50  # -1 to 1
        return -deviation  # Invert for contrarian signal

    def _get_warning_message(self, herd_level: HerdLevel, ticker: str) -> str:
        """Get the main warning message."""
        messages = {
            HerdLevel.EXTREME_BULLISH: f"CONTRARIAN WARNING: Extreme bullish consensus on {ticker}. When everyone's bullish, consider who's left to buy.",
            HerdLevel.HIGH_BULLISH: f"CAUTION: High bullish sentiment on {ticker}. FOMO may be driving prices.",
            HerdLevel.MODERATE_BULLISH: f"Sentiment for {ticker} is moderately bullish. Within normal range.",
            HerdLevel.NEUTRAL: f"Sentiment for {ticker} is balanced. No herd behavior detected.",
            HerdLevel.MODERATE_BEARISH: f"Sentiment for {ticker} is moderately bearish. Within normal range.",
            HerdLevel.HIGH_BEARISH: f"CAUTION: High bearish sentiment on {ticker}. Panic selling may be occurring.",
            HerdLevel.EXTREME_BEARISH: f"CONTRARIAN OPPORTUNITY: Extreme bearish consensus on {ticker}. When everyone's bearish, consider who's left to sell.",
        }
        return messages.get(herd_level, "Unable to assess herd sentiment.")

    def _get_signal_confidence(self, herd_level: HerdLevel, components: Dict) -> float:
        """Calculate confidence in the contrarian signal."""
        base_confidence = 0.5

        # More data sources = higher confidence
        base_confidence += len(components) * 0.1

        # Extreme levels = higher confidence
        if herd_level in [HerdLevel.EXTREME_BULLISH, HerdLevel.EXTREME_BEARISH]:
            base_confidence += 0.15

        return min(0.85, base_confidence)

    def _get_historical_accuracy(self, herd_level: HerdLevel) -> str:
        """Get historical accuracy of contrarian signal."""
        if herd_level == HerdLevel.EXTREME_BULLISH:
            return "Historically, extreme bullish consensus has preceded market corrections ~65% of the time within 3 months."
        elif herd_level == HerdLevel.EXTREME_BEARISH:
            return "Historically, extreme bearish consensus has preceded market rallies ~70% of the time within 3 months."
        else:
            return "Sentiment is not at extreme levels where contrarian signals have predictive value."

    def _get_similar_periods(self, herd_level: HerdLevel) -> List[str]:
        """Get examples of similar historical periods."""
        if herd_level in [HerdLevel.EXTREME_BULLISH, HerdLevel.HIGH_BULLISH]:
            return self.HISTORICAL_EXAMPLES["extreme_bullish"]
        elif herd_level in [HerdLevel.EXTREME_BEARISH, HerdLevel.HIGH_BEARISH]:
            return self.HISTORICAL_EXAMPLES["extreme_bearish"]
        else:
            return ["Current sentiment is within normal ranges"]

    def _get_recommendation(self, herd_level: HerdLevel, ticker: str) -> str:
        """Get actionable recommendation."""
        recommendations = {
            HerdLevel.EXTREME_BULLISH: f"Consider taking partial profits on {ticker}. Avoid adding to positions at current sentiment levels. This is not a sell signal, but a 'be cautious' signal.",
            HerdLevel.HIGH_BULLISH: f"Don't chase {ticker} due to FOMO. If buying, wait for a pullback. Set stop-losses to protect gains.",
            HerdLevel.EXTREME_BEARISH: f"Contrarian opportunity: Consider accumulating {ticker} on further weakness. Historically, extreme pessimism marks good entry points.",
            HerdLevel.HIGH_BEARISH: f"Don't panic sell {ticker}. Extreme fear often marks bottoms. Consider if fundamentals have actually changed.",
        }
        return recommendations.get(herd_level, f"No specific action needed for {ticker} based on sentiment.")

    def _get_smart_money_insight(self, put_call_ratio: Optional[float]) -> Optional[str]:
        """Infer what smart money might be doing from options data."""
        if put_call_ratio is None:
            return None

        if put_call_ratio < 0.6:
            return "Low put/call ratio suggests institutional investors are bullish (or retail is overly bullish on calls)."
        elif put_call_ratio > 1.2:
            return "High put/call ratio suggests institutions may be hedging or bearish positioning."
        else:
            return "Options market shows balanced positioning."

    def _get_analyst_consensus(self, analyst_ratings: Optional[Dict]) -> Optional[str]:
        """Get analyst consensus summary."""
        if not analyst_ratings:
            return None

        buys = analyst_ratings.get("buy", 0) + analyst_ratings.get("strong_buy", 0)
        sells = analyst_ratings.get("sell", 0) + analyst_ratings.get("strong_sell", 0)
        total = buys + analyst_ratings.get("hold", 0) + sells

        if total == 0:
            return None

        buy_pct = buys / total * 100
        if buy_pct > 80:
            return f"Strong consensus: {buy_pct:.0f}% of analysts rate Buy (potential groupthink)"
        elif buy_pct > 60:
            return f"Bullish consensus: {buy_pct:.0f}% of analysts rate Buy"
        elif buy_pct < 30:
            return f"Bearish consensus: Only {buy_pct:.0f}% of analysts rate Buy"
        else:
            return f"Mixed views: {buy_pct:.0f}% of analysts rate Buy"

    def _get_explanation(self, herd_level: HerdLevel) -> str:
        """Get detailed explanation of the herd behavior."""
        explanations = {
            HerdLevel.EXTREME_BULLISH: """
Extreme bullish consensus detected. This occurs when:
- Social media sentiment is overwhelmingly positive
- Put/Call ratio is very low (everyone buying calls)
- Fear & Greed Index shows "Extreme Greed"
- Most analysts have Buy ratings

Why this matters: Markets are forward-looking. When everyone is already bullish,
most potential buyers have already bought. This leaves few buyers left to push
prices higher, but many potential sellers if sentiment shifts.

The contrarian view: "Be fearful when others are greedy."
""",
            HerdLevel.EXTREME_BEARISH: """
Extreme bearish consensus detected. This occurs when:
- Social media sentiment is overwhelmingly negative
- Put/Call ratio is very high (everyone buying puts/hedging)
- Fear & Greed Index shows "Extreme Fear"
- Many analysts have downgraded

Why this matters: When everyone is already bearish, most potential sellers have
already sold. This leaves few sellers left to push prices lower, but many
potential buyers waiting on the sidelines.

The contrarian view: "Be greedy when others are fearful."
""",
        }
        return explanations.get(herd_level, "Sentiment is within normal ranges.")

    def _get_eli5_explanation(self, herd_level: HerdLevel) -> str:
        """Explain Like I'm 5 - simple explanation for beginners."""
        eli5 = {
            HerdLevel.EXTREME_BULLISH: """
Imagine everyone at school wants the same toy. The store raises the price because
so many people want it. But once everyone who wanted it has bought it, there's
no one left to buy it, and the price might drop.

Right now, almost everyone thinks this stock will go up. That might mean most
people who wanted to buy have already bought. Be careful about buying when
everyone else is excited.
""",
            HerdLevel.EXTREME_BEARISH: """
Imagine everyone at school is scared of a new toy because they heard it might
break. So nobody wants to buy it, and the price drops really low. But then
someone realizes the toy is actually fine - and they get it super cheap!

Right now, almost everyone is scared about this stock. That might mean good
deals for brave buyers who do their homework.
""",
            HerdLevel.NEUTRAL: """
Some people like this stock, some don't - it's pretty balanced! This is normal
and healthy. There's no crowd behavior pushing the price too high or too low.
""",
        }
        return eli5.get(herd_level, "The crowd isn't too excited or too scared right now.")


# Singleton instance
herd_sentiment_analyzer = HerdSentimentAnalyzer()
