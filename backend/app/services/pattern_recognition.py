"""
Pattern Recognition Engine - Find similar historical scenarios.

This service identifies historical price patterns and technical setups
similar to the current situation, providing context for predictions.
"""

import math
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class HistoricalScenario:
    """A historical scenario similar to the current setup."""

    date: datetime
    similarity_score: float  # 0-1
    technical_match: float  # How well technicals matched
    sentiment_match: float  # How well sentiment matched
    setup_description: str
    outcome_5d: float  # % change over 5 days
    outcome_20d: float  # % change over 20 days
    outcome_60d: float  # % change over 60 days
    key_factors: list[str] = field(default_factory=list)


@dataclass
class PatternMatch:
    """A matched price pattern."""

    pattern_name: str
    confidence: float
    expected_move: float  # Expected % move
    typical_duration_days: int
    description: str
    historical_win_rate: float  # How often this pattern worked


@dataclass
class PatternAnalysis:
    """Complete pattern recognition analysis."""

    ticker: str
    current_setup: dict  # Current technical setup
    similar_scenarios: list[HistoricalScenario] = field(default_factory=list)
    pattern_matches: list[PatternMatch] = field(default_factory=list)
    signal_score: float = 0.0  # -1 to +1
    confidence: float = 0.5
    reasoning: str = ""


class PatternRecognitionEngine:
    """
    Identifies historical patterns and similar setups.

    This engine:
    1. Analyzes current technical setup
    2. Searches for similar historical setups
    3. Identifies chart patterns
    4. Returns expected outcomes based on history
    """

    # Technical indicator thresholds for pattern matching
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_NEUTRAL_LOW = 40
    RSI_NEUTRAL_HIGH = 60

    # Price pattern definitions
    PRICE_PATTERNS = {
        "oversold_bounce": {
            "conditions": {"rsi_below": 30, "below_sma20": True},
            "expected_move": 5.5,
            "duration": 10,
            "win_rate": 0.62,
            "description": "RSI oversold with price below SMA20 - historical bounce pattern",
        },
        "overbought_pullback": {
            "conditions": {"rsi_above": 70, "above_sma20": True},
            "expected_move": -3.5,
            "duration": 7,
            "win_rate": 0.58,
            "description": "RSI overbought - historical pullback pattern",
        },
        "golden_cross": {
            "conditions": {"sma20_above_sma50": True, "sma20_crossed_recently": True},
            "expected_move": 8.0,
            "duration": 30,
            "win_rate": 0.65,
            "description": "SMA20 crossed above SMA50 - bullish golden cross",
        },
        "death_cross": {
            "conditions": {"sma20_below_sma50": True, "sma20_crossed_recently": True},
            "expected_move": -6.5,
            "duration": 30,
            "win_rate": 0.60,
            "description": "SMA20 crossed below SMA50 - bearish death cross",
        },
        "bullish_macd_crossover": {
            "conditions": {"macd_above_signal": True, "macd_crossed_recently": True},
            "expected_move": 4.0,
            "duration": 15,
            "win_rate": 0.57,
            "description": "MACD bullish crossover",
        },
        "bearish_macd_crossover": {
            "conditions": {"macd_below_signal": True, "macd_crossed_recently": True},
            "expected_move": -3.5,
            "duration": 15,
            "win_rate": 0.55,
            "description": "MACD bearish crossover",
        },
        "support_bounce": {
            "conditions": {"near_support": True, "rsi_not_overbought": True},
            "expected_move": 4.0,
            "duration": 10,
            "win_rate": 0.58,
            "description": "Price near support level with room to run",
        },
        "resistance_rejection": {
            "conditions": {"near_resistance": True, "rsi_not_oversold": True},
            "expected_move": -3.0,
            "duration": 10,
            "win_rate": 0.55,
            "description": "Price near resistance level",
        },
        "bullish_trend_continuation": {
            "conditions": {"uptrend": True, "rsi_neutral": True, "above_sma50": True},
            "expected_move": 5.0,
            "duration": 20,
            "win_rate": 0.60,
            "description": "Established uptrend with healthy RSI - continuation expected",
        },
        "bearish_trend_continuation": {
            "conditions": {"downtrend": True, "rsi_neutral": True, "below_sma50": True},
            "expected_move": -4.5,
            "duration": 20,
            "win_rate": 0.58,
            "description": "Established downtrend - continuation expected",
        },
        "consolidation_breakout_pending": {
            "conditions": {"low_volatility": True, "rsi_neutral": True},
            "expected_move": 0.0,  # Direction uncertain
            "duration": 10,
            "win_rate": 0.50,
            "description": "Consolidation pattern - breakout pending but direction uncertain",
        },
    }

    def __init__(self):
        """Initialize the pattern recognition engine."""
        self.logger = logger.bind(service="pattern_recognition")

    def analyze_patterns(
        self,
        ticker: str,
        current_price: float,
        technicals: dict,  # Current technical indicators
        historical_prices: list[dict],  # Historical OHLCV data
        historical_technicals: Optional[list[dict]] = None,  # Historical technicals
    ) -> PatternAnalysis:
        """
        Analyze patterns and find similar historical scenarios.

        Args:
            ticker: Stock ticker
            current_price: Current price
            technicals: Current technical indicators
            historical_prices: Historical price data
            historical_technicals: Historical technical indicators

        Returns:
            PatternAnalysis with matches and scenarios
        """
        self.logger.info("Analyzing patterns", ticker=ticker)

        analysis = PatternAnalysis(
            ticker=ticker,
            current_setup=self._summarize_setup(technicals),
        )

        if not historical_prices:
            analysis.reasoning = "Insufficient historical data for pattern analysis"
            return analysis

        # 1. Identify current pattern matches
        pattern_matches = self._identify_patterns(technicals, historical_prices)
        analysis.pattern_matches = pattern_matches

        # 2. Find similar historical scenarios
        if historical_technicals:
            similar = self._find_similar_scenarios(
                technicals,
                historical_technicals,
                historical_prices,
            )
            analysis.similar_scenarios = similar

        # 3. Calculate signal score based on patterns
        if pattern_matches:
            weighted_move = 0
            total_weight = 0

            for pattern in pattern_matches:
                weight = pattern.confidence * pattern.historical_win_rate
                weighted_move += pattern.expected_move * weight
                total_weight += weight

            if total_weight > 0:
                expected_move = weighted_move / total_weight
                # Convert expected % move to signal (-1 to +1)
                analysis.signal_score = max(-1, min(1, expected_move / 10))

        # 4. Incorporate similar scenario outcomes
        if analysis.similar_scenarios:
            scenario_moves = [s.outcome_20d for s in analysis.similar_scenarios]
            avg_scenario_move = sum(scenario_moves) / len(scenario_moves)
            scenario_signal = max(-1, min(1, avg_scenario_move / 10))

            # Blend pattern signal with scenario signal
            if analysis.signal_score != 0:
                analysis.signal_score = (analysis.signal_score * 0.6) + (scenario_signal * 0.4)
            else:
                analysis.signal_score = scenario_signal

        # 5. Calculate confidence
        confidence_factors = []

        if pattern_matches:
            avg_win_rate = sum(p.historical_win_rate for p in pattern_matches) / len(pattern_matches)
            confidence_factors.append(avg_win_rate)

        if analysis.similar_scenarios:
            avg_similarity = sum(s.similarity_score for s in analysis.similar_scenarios) / len(analysis.similar_scenarios)
            confidence_factors.append(avg_similarity)

        if confidence_factors:
            analysis.confidence = sum(confidence_factors) / len(confidence_factors)
        else:
            analysis.confidence = 0.4

        # 6. Build reasoning
        analysis.reasoning = self._build_reasoning(analysis)

        self.logger.info(
            "Pattern analysis complete",
            ticker=ticker,
            patterns_found=len(pattern_matches),
            similar_scenarios=len(analysis.similar_scenarios),
            signal=analysis.signal_score,
        )

        return analysis

    def _summarize_setup(self, technicals: dict) -> dict:
        """Summarize the current technical setup."""
        rsi = technicals.get("rsi_14") or technicals.get("rsi")
        macd = technicals.get("macd")
        macd_signal = technicals.get("macd_signal")
        sma20 = technicals.get("sma_20")
        sma50 = technicals.get("sma_50")
        sma200 = technicals.get("sma_200")
        price = technicals.get("current_price") or technicals.get("price")

        setup = {
            "rsi": rsi,
            "rsi_zone": self._get_rsi_zone(rsi) if rsi else "unknown",
            "macd_bullish": macd > macd_signal if (macd and macd_signal) else None,
            "trend": self._determine_trend(price, sma20, sma50, sma200),
            "above_sma20": price > sma20 if (price and sma20) else None,
            "above_sma50": price > sma50 if (price and sma50) else None,
            "above_sma200": price > sma200 if (price and sma200) else None,
        }

        return setup

    def _get_rsi_zone(self, rsi: float) -> str:
        """Determine RSI zone."""
        if rsi <= self.RSI_OVERSOLD:
            return "oversold"
        elif rsi >= self.RSI_OVERBOUGHT:
            return "overbought"
        elif rsi <= self.RSI_NEUTRAL_LOW:
            return "neutral_low"
        elif rsi >= self.RSI_NEUTRAL_HIGH:
            return "neutral_high"
        else:
            return "neutral"

    def _determine_trend(self, price: float, sma20: float, sma50: float, sma200: float) -> str:
        """Determine overall trend."""
        if not all([price, sma20, sma50]):
            return "unknown"

        if price > sma20 > sma50:
            if sma200 and sma50 > sma200:
                return "strong_uptrend"
            return "uptrend"
        elif price < sma20 < sma50:
            if sma200 and sma50 < sma200:
                return "strong_downtrend"
            return "downtrend"
        else:
            return "sideways"

    def _identify_patterns(self, technicals: dict, historical_prices: list[dict]) -> list[PatternMatch]:
        """Identify matching price patterns."""
        matches = []

        rsi = technicals.get("rsi_14") or technicals.get("rsi")
        macd = technicals.get("macd")
        macd_signal = technicals.get("macd_signal")
        sma20 = technicals.get("sma_20")
        sma50 = technicals.get("sma_50")
        price = technicals.get("current_price") or technicals.get("price")
        support = technicals.get("support_level")
        resistance = technicals.get("resistance_level")

        # Check each pattern
        for pattern_name, pattern_def in self.PRICE_PATTERNS.items():
            conditions = pattern_def["conditions"]
            matched = True
            confidence = 0.6  # Base confidence

            # Check RSI conditions
            if "rsi_below" in conditions:
                if not rsi or rsi >= conditions["rsi_below"]:
                    matched = False
                else:
                    confidence += 0.1

            if "rsi_above" in conditions:
                if not rsi or rsi <= conditions["rsi_above"]:
                    matched = False
                else:
                    confidence += 0.1

            if "rsi_neutral" in conditions and conditions["rsi_neutral"]:
                if not rsi or rsi <= 35 or rsi >= 65:
                    matched = False

            if "rsi_not_overbought" in conditions and conditions["rsi_not_overbought"]:
                if rsi and rsi >= 70:
                    matched = False

            if "rsi_not_oversold" in conditions and conditions["rsi_not_oversold"]:
                if rsi and rsi <= 30:
                    matched = False

            # Check MA conditions
            if "below_sma20" in conditions and conditions["below_sma20"]:
                if not (price and sma20 and price < sma20):
                    matched = False

            if "above_sma20" in conditions and conditions["above_sma20"]:
                if not (price and sma20 and price > sma20):
                    matched = False

            if "above_sma50" in conditions and conditions["above_sma50"]:
                if not (price and sma50 and price > sma50):
                    matched = False

            if "below_sma50" in conditions and conditions["below_sma50"]:
                if not (price and sma50 and price < sma50):
                    matched = False

            if "sma20_above_sma50" in conditions and conditions["sma20_above_sma50"]:
                if not (sma20 and sma50 and sma20 > sma50):
                    matched = False

            if "sma20_below_sma50" in conditions and conditions["sma20_below_sma50"]:
                if not (sma20 and sma50 and sma20 < sma50):
                    matched = False

            # Check MACD conditions
            if "macd_above_signal" in conditions and conditions["macd_above_signal"]:
                if not (macd and macd_signal and macd > macd_signal):
                    matched = False

            if "macd_below_signal" in conditions and conditions["macd_below_signal"]:
                if not (macd and macd_signal and macd < macd_signal):
                    matched = False

            # Check trend conditions
            if "uptrend" in conditions and conditions["uptrend"]:
                trend = self._determine_trend(price, sma20, sma50, technicals.get("sma_200"))
                if "uptrend" not in trend:
                    matched = False

            if "downtrend" in conditions and conditions["downtrend"]:
                trend = self._determine_trend(price, sma20, sma50, technicals.get("sma_200"))
                if "downtrend" not in trend:
                    matched = False

            # Check support/resistance
            if "near_support" in conditions and conditions["near_support"]:
                if not (price and support and abs(price - support) / price < 0.03):
                    matched = False
                else:
                    confidence += 0.1

            if "near_resistance" in conditions and conditions["near_resistance"]:
                if not (price and resistance and abs(price - resistance) / price < 0.03):
                    matched = False
                else:
                    confidence += 0.1

            if matched:
                matches.append(PatternMatch(
                    pattern_name=pattern_name,
                    confidence=min(0.85, confidence),
                    expected_move=pattern_def["expected_move"],
                    typical_duration_days=pattern_def["duration"],
                    description=pattern_def["description"],
                    historical_win_rate=pattern_def["win_rate"],
                ))

        return matches

    def _find_similar_scenarios(
        self,
        current_technicals: dict,
        historical_technicals: list[dict],
        historical_prices: list[dict],
    ) -> list[HistoricalScenario]:
        """Find similar historical scenarios."""
        if not historical_technicals or not historical_prices:
            return []

        current_rsi = current_technicals.get("rsi_14") or current_technicals.get("rsi")
        current_price = current_technicals.get("current_price") or current_technicals.get("price")
        current_sma20 = current_technicals.get("sma_20")
        current_sma50 = current_technicals.get("sma_50")

        if not current_rsi:
            return []

        similar = []
        price_by_date = {str(p.get("date", ""))[:10]: p for p in historical_prices}

        for i, hist in enumerate(historical_technicals):
            if i >= len(historical_technicals) - 60:  # Need 60 days of forward data
                continue

            hist_rsi = hist.get("rsi_14") or hist.get("rsi")
            hist_date = hist.get("date")

            if not hist_rsi or not hist_date:
                continue

            # Calculate similarity
            rsi_diff = abs(current_rsi - hist_rsi) / 100
            rsi_similarity = 1 - rsi_diff

            # MA position similarity
            ma_similarity = 0.5  # Default
            hist_price = hist.get("price") or hist.get("close")
            hist_sma20 = hist.get("sma_20")
            hist_sma50 = hist.get("sma_50")

            if all([hist_price, hist_sma20, hist_sma50, current_price, current_sma20, current_sma50]):
                current_above_20 = current_price > current_sma20
                current_above_50 = current_price > current_sma50
                hist_above_20 = hist_price > hist_sma20
                hist_above_50 = hist_price > hist_sma50

                ma_matches = (current_above_20 == hist_above_20) + (current_above_50 == hist_above_50)
                ma_similarity = ma_matches / 2

            # Overall similarity
            similarity = (rsi_similarity * 0.5) + (ma_similarity * 0.5)

            if similarity < 0.7:  # Threshold for "similar"
                continue

            # Get outcomes
            date_str = str(hist_date)[:10]
            if date_str not in price_by_date:
                continue

            base_price = price_by_date[date_str].get("close", price_by_date[date_str].get("Close", 0))
            if base_price <= 0:
                continue

            # Find future prices
            future_dates = sorted([d for d in price_by_date.keys() if d > date_str])

            outcome_5d = 0
            outcome_20d = 0
            outcome_60d = 0

            if len(future_dates) >= 5:
                p5 = price_by_date[future_dates[4]].get("close", price_by_date[future_dates[4]].get("Close", 0))
                if p5 > 0:
                    outcome_5d = ((p5 - base_price) / base_price) * 100

            if len(future_dates) >= 20:
                p20 = price_by_date[future_dates[19]].get("close", price_by_date[future_dates[19]].get("Close", 0))
                if p20 > 0:
                    outcome_20d = ((p20 - base_price) / base_price) * 100

            if len(future_dates) >= 60:
                p60 = price_by_date[future_dates[59]].get("close", price_by_date[future_dates[59]].get("Close", 0))
                if p60 > 0:
                    outcome_60d = ((p60 - base_price) / base_price) * 100

            # Build setup description
            setup_parts = []
            if hist_rsi <= 30:
                setup_parts.append("oversold")
            elif hist_rsi >= 70:
                setup_parts.append("overbought")
            else:
                setup_parts.append(f"RSI {hist_rsi:.0f}")

            if hist_price and hist_sma20 and hist_sma50:
                if hist_price > hist_sma20 > hist_sma50:
                    setup_parts.append("uptrend")
                elif hist_price < hist_sma20 < hist_sma50:
                    setup_parts.append("downtrend")
                else:
                    setup_parts.append("sideways")

            scenario = HistoricalScenario(
                date=datetime.strptime(date_str, "%Y-%m-%d") if isinstance(hist_date, str) else hist_date,
                similarity_score=round(similarity, 2),
                technical_match=round(rsi_similarity, 2),
                sentiment_match=0.5,  # Would need sentiment data
                setup_description=" + ".join(setup_parts),
                outcome_5d=round(outcome_5d, 2),
                outcome_20d=round(outcome_20d, 2),
                outcome_60d=round(outcome_60d, 2),
                key_factors=[f"RSI: {hist_rsi:.0f}", f"Similarity: {similarity:.0%}"],
            )

            similar.append(scenario)

        # Sort by similarity and return top 5
        similar.sort(key=lambda s: s.similarity_score, reverse=True)
        return similar[:5]

    def _build_reasoning(self, analysis: PatternAnalysis) -> str:
        """Build reasoning text for the analysis."""
        parts = []

        if analysis.pattern_matches:
            top_pattern = max(analysis.pattern_matches, key=lambda p: p.confidence)
            parts.append(f"Identified pattern: {top_pattern.description}.")
            parts.append(f"Historical win rate: {top_pattern.historical_win_rate:.0%}.")

        if analysis.similar_scenarios:
            avg_outcome = sum(s.outcome_20d for s in analysis.similar_scenarios) / len(analysis.similar_scenarios)
            direction = "up" if avg_outcome > 0 else "down"
            parts.append(
                f"Found {len(analysis.similar_scenarios)} similar historical setups. "
                f"Average 20-day outcome: {avg_outcome:+.1f}% ({direction})."
            )

        if analysis.signal_score != 0:
            signal_type = "bullish" if analysis.signal_score > 0 else "bearish"
            parts.append(f"Pattern signal: {signal_type} ({analysis.signal_score:+.2f}).")

        return " ".join(parts) if parts else "Insufficient data for pattern analysis."


# Singleton instance
pattern_recognition_engine = PatternRecognitionEngine()
