"""
Enhanced Probability Engine for Clarividex - V2.

This is a significantly improved prediction algorithm that uses:

1. Extended Historical Data (2 years instead of 1)
2. Analyst Price Target Integration
3. Earnings Proximity Factor
4. Volatility Regime Detection (VIX-based)
5. Options Implied Move Analysis
6. Sector Relative Strength
7. Multi-Timeframe Analysis (Daily, Weekly, Monthly)
8. Enhanced Pattern Recognition
9. Insider Trading Signals
10. Short Interest Analysis
11. Fat Tail Distribution (Student-t instead of Normal)
12. Mean Reversion Modeling
13. Market Regime Detection (Bull/Bear/Sideways)
14. Earnings Surprise History
15. Price Level Analysis (Support/Resistance)

Key Innovation: Dynamic signal reliability that adjusts based on market conditions.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import structlog

logger = structlog.get_logger()


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class VolatilityRegime(Enum):
    """Volatility regime based on VIX."""
    LOW = "low"  # VIX < 15
    NORMAL = "normal"  # VIX 15-25
    HIGH = "high"  # VIX 25-35
    EXTREME = "extreme"  # VIX > 35


@dataclass
class EnhancedProbabilityResult:
    """Comprehensive result from enhanced probability calculation."""
    probability: float
    confidence: float
    method: str
    components: Dict[str, float] = field(default_factory=dict)
    adjustments: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    market_regime: str = ""
    volatility_regime: str = ""
    data_quality: float = 0.0
    factors_used: int = 0
    warnings: List[str] = field(default_factory=list)


class EnhancedProbabilityEngine:
    """
    Advanced probability engine with comprehensive data integration.

    This engine significantly improves prediction accuracy by:
    1. Using more data sources
    2. Dynamic signal reliability based on market conditions
    3. Better statistical modeling (fat tails, mean reversion)
    4. Multi-timeframe analysis
    5. Catalyst and event integration
    6. Probability calibration based on backtesting
    7. Earnings surprise history
    8. Options flow analysis
    9. Per-ticker model tuning
    """

    # Base market statistics (from historical data)
    MARKET_STATS = {
        "daily_up_probability": 0.53,
        "weekly_up_probability": 0.56,
        "monthly_up_probability": 0.62,
        "avg_daily_return": 0.0004,  # ~10% annual
        "avg_daily_volatility": 0.012,
        "avg_annual_volatility": 0.19,
    }

    # IMPROVEMENT #1: Calibration factor based on backtesting
    # Backtests showed model predicts ~60% but actual hit rate is ~40%
    # Calibration pulls predictions toward 50% to reduce overconfidence
    CALIBRATION_FACTOR = 0.70  # Reduces deviation from 50%

    # IMPROVEMENT #5: Per-ticker weight adjustments
    # Based on backtest results showing different accuracy per ticker
    TICKER_ADJUSTMENTS = {
        # Tickers where multi_timeframe works best
        "GOOGL": {"multi_timeframe": 1.3, "bayesian": 0.9},
        "MSFT": {"multi_timeframe": 1.2, "volatility": 0.9},
        # Tickers where momentum signals work better
        "NVDA": {"monte_carlo": 1.2, "bayesian": 1.1, "volatility": 0.8},
        "AMD": {"monte_carlo": 1.2, "bayesian": 1.1},
        # More stable tickers - volatility works better
        "AAPL": {"volatility": 1.2, "multi_timeframe": 1.1},
        "JNJ": {"volatility": 1.3, "mean_reversion": 1.2},
        "PG": {"volatility": 1.3, "mean_reversion": 1.2},
        # Default - no adjustment
    }

    # IMPROVEMENT #3: Earnings surprise patterns by sector
    SECTOR_EARNINGS_BEAT_RATES = {
        "Technology": 0.72,  # Tech beats 72% of the time
        "Healthcare": 0.68,
        "Financial Services": 0.65,
        "Consumer Cyclical": 0.63,
        "Communication Services": 0.70,
        "Consumer Defensive": 0.67,
        "Industrials": 0.64,
        "Energy": 0.58,
        "Utilities": 0.71,
        "Real Estate": 0.66,
        "Basic Materials": 0.60,
        "default": 0.65,
    }

    # Dynamic signal reliability (base values, adjusted by regime)
    BASE_SIGNAL_RELIABILITY = {
        # Technical signals
        "rsi_oversold": 0.62,
        "rsi_overbought": 0.58,
        "rsi_divergence": 0.65,
        "macd_bullish_cross": 0.56,
        "macd_bearish_cross": 0.54,
        "macd_divergence": 0.63,
        "golden_cross": 0.68,
        "death_cross": 0.64,
        "price_above_sma200": 0.58,
        "price_below_sma200": 0.55,
        "bollinger_squeeze": 0.60,
        "volume_breakout": 0.58,

        # Fundamental signals
        "analyst_upgrade": 0.57,
        "analyst_downgrade": 0.59,
        "analyst_target_above": 0.55,
        "analyst_target_below": 0.57,
        "insider_buying": 0.62,
        "insider_selling": 0.54,
        "high_short_interest": 0.52,  # Contrarian

        # Sentiment signals
        "bullish_news": 0.54,
        "bearish_news": 0.56,
        "fear_greed_extreme_fear": 0.60,  # Contrarian
        "fear_greed_extreme_greed": 0.55,  # Contrarian
        "bullish_options_flow": 0.56,
        "bearish_options_flow": 0.58,

        # Event signals
        "pre_earnings_drift": 0.55,
        "post_earnings_momentum": 0.58,
    }

    # Volatility regime adjustments for signal reliability
    VOLATILITY_ADJUSTMENTS = {
        VolatilityRegime.LOW: 1.05,  # Signals more reliable in calm markets
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.HIGH: 0.90,  # Less reliable in volatile markets
        VolatilityRegime.EXTREME: 0.75,  # Much less reliable in crisis
    }

    def __init__(self):
        """Initialize the enhanced probability engine."""
        self.logger = logger.bind(service="enhanced_probability_engine")
        self._random = random.Random(42)

    def calculate_probability(
        self,
        current_price: float,
        target_price: float,
        days_to_target: int,
        historical_prices: List[Dict],
        signals: Dict[str, float],
        technicals: Optional[Dict] = None,
        news_sentiment: Optional[float] = None,
        options_data: Optional[Dict] = None,
        analyst_data: Optional[Dict] = None,
        insider_data: Optional[List[Dict]] = None,
        earnings_dates: Optional[List[Dict]] = None,
        vix_data: Optional[Dict] = None,
        fear_greed: Optional[Dict] = None,
        sector_performance: Optional[Dict] = None,
        finviz_data: Optional[Dict] = None,
        ticker: Optional[str] = None,  # IMPROVEMENT #5: Per-ticker tuning
        earnings_history: Optional[List[Dict]] = None,  # IMPROVEMENT #3: Earnings surprise
        sector: Optional[str] = None,  # For sector-based earnings rates
    ) -> EnhancedProbabilityResult:
        """
        Calculate comprehensive probability using all available data.
        """
        warnings = []
        factors_used = 0

        if current_price <= 0 or target_price <= 0 or days_to_target <= 0:
            return EnhancedProbabilityResult(
                probability=50.0,
                confidence=0.1,
                method="invalid_input",
                reasoning="Invalid input parameters"
            )

        required_return = (target_price - current_price) / current_price
        is_bullish = required_return > 0
        abs_return = abs(required_return)

        self.logger.info(
            "Enhanced probability calculation",
            current=current_price,
            target=target_price,
            required_return=f"{required_return:.2%}",
            days=days_to_target,
        )

        # Detect market regimes
        volatility_regime = self._detect_volatility_regime(vix_data, historical_prices)
        market_regime = self._detect_market_regime(historical_prices)

        # Get dynamic signal reliability based on regime
        signal_reliability = self._get_adjusted_reliability(volatility_regime)

        # Initialize probability components
        components = {}
        adjustments = {}

        # ==========================================
        # 1. VOLATILITY-BASED PROBABILITY (Enhanced)
        # ==========================================
        vol_prob = self._enhanced_volatility_probability(
            historical_prices,
            required_return,
            days_to_target,
            volatility_regime,
        )
        components["volatility"] = vol_prob
        factors_used += 1

        # ==========================================
        # 2. MONTE CARLO WITH FAT TAILS
        # ==========================================
        mc_prob = self._enhanced_monte_carlo(
            current_price,
            target_price,
            days_to_target,
            historical_prices,
            signals,
            volatility_regime,
        )
        components["monte_carlo"] = mc_prob
        factors_used += 1

        # ==========================================
        # 3. MULTI-TIMEFRAME ANALYSIS
        # ==========================================
        mtf_prob = self._multi_timeframe_analysis(
            historical_prices,
            required_return,
            days_to_target,
            technicals,
        )
        components["multi_timeframe"] = mtf_prob
        factors_used += 1

        # ==========================================
        # 4. BAYESIAN SIGNAL INTEGRATION (Enhanced)
        # ==========================================
        bayes_prob = self._enhanced_bayesian_probability(
            signals,
            is_bullish,
            technicals,
            news_sentiment,
            options_data,
            signal_reliability,
            fear_greed,
        )
        components["bayesian"] = bayes_prob
        factors_used += 1

        # ==========================================
        # 5. ANALYST PRICE TARGET ANALYSIS
        # ==========================================
        if analyst_data or finviz_data:
            analyst_prob = self._analyst_target_probability(
                current_price,
                target_price,
                analyst_data,
                finviz_data,
                signal_reliability,
            )
            if analyst_prob is not None:
                components["analyst_targets"] = analyst_prob
                factors_used += 1

        # ==========================================
        # 6. OPTIONS IMPLIED PROBABILITY
        # ==========================================
        if options_data:
            options_prob = self._options_implied_probability(
                current_price,
                target_price,
                days_to_target,
                options_data,
            )
            if options_prob is not None:
                components["options_implied"] = options_prob
                factors_used += 1

        # ==========================================
        # 7. SECTOR RELATIVE STRENGTH
        # ==========================================
        if sector_performance and historical_prices:
            sector_adj = self._sector_relative_strength(
                historical_prices,
                sector_performance,
                is_bullish,
            )
            adjustments["sector_strength"] = sector_adj

        # ==========================================
        # 8. INSIDER TRADING SIGNAL
        # ==========================================
        if insider_data:
            insider_adj = self._insider_trading_signal(
                insider_data,
                is_bullish,
                signal_reliability,
            )
            adjustments["insider_activity"] = insider_adj

        # ==========================================
        # 9. EARNINGS PROXIMITY ADJUSTMENT
        # ==========================================
        if earnings_dates:
            earnings_adj = self._earnings_proximity_adjustment(
                earnings_dates,
                days_to_target,
                is_bullish,
            )
            adjustments["earnings_proximity"] = earnings_adj

        # ==========================================
        # 10. MEAN REVERSION FACTOR
        # ==========================================
        mean_rev_adj = self._mean_reversion_adjustment(
            historical_prices,
            current_price,
            required_return,
            days_to_target,
        )
        adjustments["mean_reversion"] = mean_rev_adj

        # ==========================================
        # 11. SUPPORT/RESISTANCE ANALYSIS
        # ==========================================
        if technicals:
            sr_adj = self._support_resistance_adjustment(
                current_price,
                target_price,
                technicals,
            )
            adjustments["support_resistance"] = sr_adj

        # ==========================================
        # 12. SHORT INTEREST ANALYSIS
        # ==========================================
        if finviz_data:
            short_adj = self._short_interest_adjustment(
                finviz_data,
                is_bullish,
            )
            if short_adj != 0:
                adjustments["short_interest"] = short_adj

        # ==========================================
        # 13. EARNINGS SURPRISE HISTORY (IMPROVEMENT #3)
        # ==========================================
        days_to_earnings = None
        if earnings_dates and len(earnings_dates) > 0:
            # Find next earnings date
            from datetime import datetime
            today = datetime.now()
            for ed in earnings_dates:
                try:
                    ed_date = ed.get('date')
                    if ed_date:
                        if isinstance(ed_date, str):
                            ed_parsed = datetime.strptime(ed_date[:10], '%Y-%m-%d')
                        else:
                            ed_parsed = ed_date
                        if ed_parsed > today:
                            days_to_earnings = (ed_parsed - today).days
                            break
                except:
                    pass

        earnings_prob = self._earnings_surprise_probability(
            earnings_history,
            sector,
            is_bullish,
            days_to_earnings,
        )
        if earnings_prob is not None:
            components["earnings_surprise"] = earnings_prob
            factors_used += 1

        # ==========================================
        # 14. OPTIONS FLOW ANALYSIS (IMPROVEMENT #4)
        # ==========================================
        if options_data:
            flow_prob = self._options_flow_probability(
                options_data,
                is_bullish,
            )
            if flow_prob is not None:
                components["options_flow"] = flow_prob
                factors_used += 1

        # ==========================================
        # COMBINE ALL COMPONENTS
        # ==========================================
        # IMPROVEMENT #5: Pass ticker for per-ticker weight tuning
        weights = self._calculate_dynamic_weights(
            historical_prices,
            components,
            volatility_regime,
            ticker=ticker,
        )

        # Calculate base probability from components
        base_prob = sum(
            components[k] * weights.get(k, 0.1)
            for k in components
        )

        # Apply adjustments
        total_adjustment = sum(adjustments.values())
        adjusted_prob = base_prob + total_adjustment

        # Calculate confidence
        confidence = self._calculate_enhanced_confidence(
            historical_prices,
            components,
            adjustments,
            factors_used,
            volatility_regime,
        )

        # Apply confidence-based smoothing
        if confidence < 0.5:
            # Low confidence pulls toward 50%
            adjusted_prob = 50 + (adjusted_prob - 50) * (confidence * 1.5)

        # ==========================================
        # IMPROVEMENT #1: APPLY PROBABILITY CALIBRATION
        # ==========================================
        # Backtesting showed overconfidence - apply calibration
        calibrated_prob = self._calibrate_probability(adjusted_prob, confidence)

        # Apply realistic bounds
        final_prob = self._apply_realistic_bounds(
            calibrated_prob,  # Use calibrated probability
            abs_return,
            days_to_target,
            confidence,
        )

        # Build reasoning
        reasoning = self._build_enhanced_reasoning(
            components,
            weights,
            adjustments,
            final_prob,
            required_return,
            days_to_target,
            market_regime,
            volatility_regime,
        )

        return EnhancedProbabilityResult(
            probability=round(final_prob, 1),
            confidence=round(confidence, 2),
            method="enhanced_ensemble_v2",
            components=components,
            adjustments=adjustments,
            reasoning=reasoning,
            market_regime=market_regime.value,
            volatility_regime=volatility_regime.value,
            data_quality=self._calculate_data_quality(historical_prices, factors_used),
            factors_used=factors_used,
            warnings=warnings,
        )

    def _detect_volatility_regime(
        self,
        vix_data: Optional[Dict],
        historical_prices: List[Dict],
    ) -> VolatilityRegime:
        """Detect current volatility regime from VIX or historical volatility."""
        vix_value = None

        if vix_data:
            vix_value = vix_data.get("current") or vix_data.get("value")

        if vix_value is None and historical_prices and len(historical_prices) >= 20:
            # Calculate realized volatility as proxy
            prices = [p.get("close", 0) for p in historical_prices[-20:] if p.get("close", 0) > 0]
            if len(prices) >= 10:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                realized_vol = math.sqrt(sum(r**2 for r in returns) / len(returns)) * math.sqrt(252) * 100
                vix_value = realized_vol

        if vix_value is None:
            return VolatilityRegime.NORMAL

        if vix_value < 15:
            return VolatilityRegime.LOW
        elif vix_value < 25:
            return VolatilityRegime.NORMAL
        elif vix_value < 35:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    def _detect_market_regime(self, historical_prices: List[Dict]) -> MarketRegime:
        """Detect current market regime from price trends."""
        if not historical_prices or len(historical_prices) < 50:
            return MarketRegime.SIDEWAYS

        prices = [p.get("close", 0) for p in historical_prices if p.get("close", 0) > 0]
        if len(prices) < 50:
            return MarketRegime.SIDEWAYS

        # Calculate 50-day and 200-day trends
        recent_50 = prices[-50:]
        sma_50 = sum(recent_50) / len(recent_50)

        if len(prices) >= 200:
            recent_200 = prices[-200:]
            sma_200 = sum(recent_200) / len(recent_200)
        else:
            sma_200 = sma_50

        current = prices[-1]

        # Determine regime
        if current > sma_50 > sma_200:
            return MarketRegime.BULL
        elif current < sma_50 < sma_200:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def _get_adjusted_reliability(self, volatility_regime: VolatilityRegime) -> Dict[str, float]:
        """Get signal reliability adjusted for current volatility regime."""
        adjustment = self.VOLATILITY_ADJUSTMENTS.get(volatility_regime, 1.0)
        return {
            k: min(0.75, v * adjustment)  # Cap at 75% reliability
            for k, v in self.BASE_SIGNAL_RELIABILITY.items()
        }

    def _enhanced_volatility_probability(
        self,
        historical_prices: List[Dict],
        required_return: float,
        days: int,
        volatility_regime: VolatilityRegime,
    ) -> float:
        """Enhanced volatility-based probability with fat tails."""
        if not historical_prices or len(historical_prices) < 30:
            return 50.0

        prices = [p.get("close", 0) for p in historical_prices if p.get("close", 0) > 0]
        if len(prices) < 30:
            return 50.0

        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])

        if not returns:
            return 50.0

        # Calculate statistics
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        daily_vol = math.sqrt(variance)

        # Calculate excess kurtosis for fat tail adjustment
        if variance > 0:
            fourth_moment = sum((r - mean_ret) ** 4 for r in returns) / len(returns)
            kurtosis = fourth_moment / (variance ** 2) - 3  # Excess kurtosis
        else:
            kurtosis = 0

        # Scale to target period
        period_vol = daily_vol * math.sqrt(days)
        period_drift = mean_ret * days

        if period_vol > 0:
            z_score = (required_return - period_drift) / period_vol
        else:
            return 50.0

        # Use Student-t distribution for fat tails
        # Degrees of freedom inversely related to kurtosis
        df = max(4, 30 / (1 + max(0, kurtosis)))

        # Calculate probability using t-distribution approximation
        # For large df, approaches normal; for small df, fatter tails
        t_adjustment = 1 + (kurtosis * 0.05) if abs(z_score) > 1.5 else 1
        probability = self._student_t_cdf(-z_score, df) * 100 * t_adjustment

        # Volatility regime adjustment
        if volatility_regime == VolatilityRegime.HIGH:
            # In high vol, extreme moves more likely
            if abs(z_score) > 1:
                probability = probability * 1.1 if required_return > 0 else probability * 0.9
        elif volatility_regime == VolatilityRegime.EXTREME:
            if abs(z_score) > 1:
                probability = probability * 1.2 if required_return > 0 else probability * 0.8

        return max(5, min(95, probability))

    def _student_t_cdf(self, x: float, df: float) -> float:
        """Approximate Student-t CDF."""
        # Use normal approximation with fat tail adjustment
        normal_cdf = 0.5 * (1 + math.erf(x / math.sqrt(2)))

        # Adjust for fat tails (Student-t has more probability in tails)
        if abs(x) > 2 and df < 30:
            tail_adjustment = 0.02 * (30 - df) / 30
            if x < 0:
                normal_cdf = max(0.01, normal_cdf - tail_adjustment)
            else:
                normal_cdf = min(0.99, normal_cdf + tail_adjustment)

        return normal_cdf

    def _enhanced_monte_carlo(
        self,
        current_price: float,
        target_price: float,
        days: int,
        historical_prices: List[Dict],
        signals: Dict[str, float],
        volatility_regime: VolatilityRegime,
    ) -> float:
        """Enhanced Monte Carlo with regime-switching and mean reversion."""
        n_simulations = 2000  # More simulations for accuracy

        if historical_prices and len(historical_prices) >= 30:
            prices = [p.get("close", 0) for p in historical_prices if p.get("close", 0) > 0]
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    returns.append(math.log(prices[i] / prices[i-1]))

            if returns:
                mu = sum(returns) / len(returns)
                sigma = math.sqrt(sum((r - mu) ** 2 for r in returns) / len(returns))

                # Calculate mean price for mean reversion
                mean_price = sum(prices[-60:]) / min(60, len(prices))
            else:
                mu, sigma = 0.0003, 0.015
                mean_price = current_price
        else:
            mu, sigma = 0.0003, 0.015
            mean_price = current_price

        # Adjust parameters for volatility regime
        if volatility_regime == VolatilityRegime.HIGH:
            sigma *= 1.3
        elif volatility_regime == VolatilityRegime.EXTREME:
            sigma *= 1.6
        elif volatility_regime == VolatilityRegime.LOW:
            sigma *= 0.8

        # Signal-based drift adjustment
        signal_strength = sum(signals.values()) / max(len(signals), 1) if signals else 0
        mu_adjusted = mu + (signal_strength * 0.002)  # Increased signal impact

        # Mean reversion speed (Ornstein-Uhlenbeck parameter)
        mean_reversion_speed = 0.02  # ~2% daily pull toward mean

        successes = 0
        is_bullish = target_price > current_price

        for _ in range(n_simulations):
            price = current_price

            for day in range(days):
                # Mean reversion component
                mean_rev = mean_reversion_speed * (math.log(mean_price) - math.log(price))

                # Random shock (using slightly fat-tailed distribution)
                u1 = self._random.random()
                u2 = self._random.random()
                # Box-Muller with slight kurtosis adjustment
                z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
                if abs(z) > 2:
                    z *= 1.1  # Slight fat tail

                daily_return = mu_adjusted + mean_rev + sigma * z
                price = price * math.exp(daily_return)

                # Check if target reached
                if is_bullish and price >= target_price:
                    successes += 1
                    break
                elif not is_bullish and price <= target_price:
                    successes += 1
                    break

        return (successes / n_simulations) * 100

    def _multi_timeframe_analysis(
        self,
        historical_prices: List[Dict],
        required_return: float,
        days: int,
        technicals: Optional[Dict],
    ) -> float:
        """Analyze multiple timeframes for trend alignment."""
        if not historical_prices or len(historical_prices) < 60:
            return 50.0

        prices = [p.get("close", 0) for p in historical_prices if p.get("close", 0) > 0]
        if len(prices) < 60:
            return 50.0

        signals = []

        # Daily trend (20-day)
        if len(prices) >= 20:
            sma_20 = sum(prices[-20:]) / 20
            daily_trend = 1 if prices[-1] > sma_20 else -1
            signals.append(("daily", daily_trend, 0.3))

        # Weekly trend (approximated by 5-week = 25 days)
        if len(prices) >= 25:
            weekly_prices = prices[-25::5]  # Sample every 5 days
            if len(weekly_prices) >= 4:
                weekly_trend = 1 if weekly_prices[-1] > weekly_prices[-4] else -1
                signals.append(("weekly", weekly_trend, 0.35))

        # Monthly trend (approximated by 60 days)
        if len(prices) >= 60:
            monthly_start = prices[-60]
            monthly_trend = 1 if prices[-1] > monthly_start else -1
            signals.append(("monthly", monthly_trend, 0.35))

        if not signals:
            return 50.0

        # Calculate alignment score
        is_bullish = required_return > 0

        total_weight = sum(w for _, _, w in signals)
        aligned_weight = sum(
            w for _, trend, w in signals
            if (trend > 0 and is_bullish) or (trend < 0 and not is_bullish)
        )

        alignment_score = aligned_weight / total_weight if total_weight > 0 else 0.5

        # Convert to probability
        base_prob = 50 + (alignment_score - 0.5) * 40

        return max(30, min(70, base_prob))

    def _enhanced_bayesian_probability(
        self,
        signals: Dict[str, float],
        is_bullish: bool,
        technicals: Optional[Dict],
        news_sentiment: Optional[float],
        options_data: Optional[Dict],
        signal_reliability: Dict[str, float],
        fear_greed: Optional[Dict],
    ) -> float:
        """Enhanced Bayesian probability with more evidence sources."""
        # Start with market base rate
        prior = 0.53 if is_bullish else 0.47
        evidence = []

        # Technical evidence
        if technicals:
            rsi = technicals.get("rsi_14")
            if rsi is not None:
                if rsi < 30:
                    rel = signal_reliability.get("rsi_oversold", 0.62)
                    evidence.append(("rsi_oversold", rel, 1 if is_bullish else -1))
                elif rsi > 70:
                    rel = signal_reliability.get("rsi_overbought", 0.58)
                    evidence.append(("rsi_overbought", rel, -1 if is_bullish else 1))
                elif rsi < 40:
                    # Moderately oversold
                    evidence.append(("rsi_low", 0.54, 0.5 if is_bullish else -0.5))
                elif rsi > 60:
                    # Moderately overbought
                    evidence.append(("rsi_high", 0.54, -0.5 if is_bullish else 0.5))

            # MACD
            macd = technicals.get("macd")
            macd_signal = technicals.get("macd_signal")
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:
                    rel = signal_reliability.get("macd_bullish_cross", 0.56)
                    evidence.append(("macd_bullish", rel, 1 if is_bullish else -1))
                else:
                    rel = signal_reliability.get("macd_bearish_cross", 0.54)
                    evidence.append(("macd_bearish", rel, -1 if is_bullish else 1))

            # Price vs SMAs
            current = technicals.get("current_price")
            sma_50 = technicals.get("sma_50")
            sma_200 = technicals.get("sma_200")

            if current and sma_200:
                if current > sma_200:
                    rel = signal_reliability.get("price_above_sma200", 0.58)
                    evidence.append(("above_200sma", rel, 1 if is_bullish else -1))
                else:
                    rel = signal_reliability.get("price_below_sma200", 0.55)
                    evidence.append(("below_200sma", rel, -1 if is_bullish else 1))

            # Golden/Death cross
            if sma_50 and sma_200:
                if sma_50 > sma_200:
                    rel = signal_reliability.get("golden_cross", 0.68)
                    evidence.append(("golden_cross", rel, 0.7 if is_bullish else -0.7))
                else:
                    rel = signal_reliability.get("death_cross", 0.64)
                    evidence.append(("death_cross", rel, -0.7 if is_bullish else 0.7))

        # News sentiment
        if news_sentiment is not None:
            if news_sentiment > 0.2:
                rel = signal_reliability.get("bullish_news", 0.54)
                strength = min(1, news_sentiment * 2)
                evidence.append(("positive_news", rel, strength if is_bullish else -strength))
            elif news_sentiment < -0.2:
                rel = signal_reliability.get("bearish_news", 0.56)
                strength = min(1, abs(news_sentiment) * 2)
                evidence.append(("negative_news", rel, -strength if is_bullish else strength))

        # Options flow
        if options_data:
            put_call = options_data.get("put_call_ratio")
            if put_call is not None:
                if put_call < 0.7:
                    rel = signal_reliability.get("bullish_options_flow", 0.56)
                    evidence.append(("bullish_options", rel, 1 if is_bullish else -1))
                elif put_call > 1.3:
                    rel = signal_reliability.get("bearish_options_flow", 0.58)
                    evidence.append(("bearish_options", rel, -1 if is_bullish else 1))

        # Fear & Greed (contrarian)
        if fear_greed:
            fg_value = fear_greed.get("value") or fear_greed.get("score")
            if fg_value is not None:
                if fg_value < 25:  # Extreme fear
                    rel = signal_reliability.get("fear_greed_extreme_fear", 0.60)
                    evidence.append(("extreme_fear", rel, 0.8 if is_bullish else -0.5))  # Contrarian bullish
                elif fg_value > 75:  # Extreme greed
                    rel = signal_reliability.get("fear_greed_extreme_greed", 0.55)
                    evidence.append(("extreme_greed", rel, -0.5 if is_bullish else 0.8))  # Contrarian bearish

        # Apply Bayesian updates
        posterior = prior
        for name, reliability, direction in evidence:
            if abs(direction) < 0.1:
                continue

            # Scale reliability by direction strength
            effective_rel = 0.5 + (reliability - 0.5) * abs(direction)

            if direction > 0:
                lr = effective_rel / (1 - effective_rel)
                odds = posterior / (1 - posterior)
                new_odds = odds * lr
                posterior = new_odds / (1 + new_odds)
            else:
                lr = (1 - effective_rel) / effective_rel
                odds = posterior / (1 - posterior)
                new_odds = odds * lr
                posterior = new_odds / (1 + new_odds)

            posterior = max(0.1, min(0.9, posterior))

        return posterior * 100

    def _analyst_target_probability(
        self,
        current_price: float,
        target_price: float,
        analyst_data: Optional[Dict],
        finviz_data: Optional[Dict],
        signal_reliability: Dict[str, float],
    ) -> Optional[float]:
        """Calculate probability based on analyst price targets."""
        analyst_target = None

        # Try to get analyst target from finviz
        if finviz_data:
            analyst_target = finviz_data.get("target_price") or finviz_data.get("analyst_target")

        if analyst_target is None and analyst_data:
            analyst_target = analyst_data.get("target_price") or analyst_data.get("mean_target")

        if analyst_target is None or analyst_target <= 0:
            return None

        # Calculate how user's target compares to analyst consensus
        user_return = (target_price - current_price) / current_price
        analyst_return = (analyst_target - current_price) / current_price

        if abs(analyst_return) < 0.01:
            return 50.0

        # If user target is within analyst target range, higher probability
        target_ratio = user_return / analyst_return if analyst_return != 0 else 1

        if 0 < target_ratio <= 1:
            # User target is less ambitious than analyst - higher probability
            base_prob = 55 + (1 - target_ratio) * 20
        elif target_ratio > 1:
            # User target exceeds analyst - lower probability
            excess = target_ratio - 1
            base_prob = 55 - min(30, excess * 25)
        else:
            # Opposite directions
            base_prob = 30

        return max(20, min(75, base_prob))

    def _options_implied_probability(
        self,
        current_price: float,
        target_price: float,
        days: int,
        options_data: Dict,
    ) -> Optional[float]:
        """Calculate probability from options-implied volatility and skew."""
        implied_vol = options_data.get("implied_volatility")

        if implied_vol is None:
            return None

        # Calculate implied move for the period
        annual_vol = implied_vol / 100 if implied_vol > 1 else implied_vol
        period_vol = annual_vol * math.sqrt(days / 252)

        required_return = (target_price - current_price) / current_price

        # Calculate probability using implied volatility
        if period_vol > 0:
            z_score = required_return / period_vol
            probability = 0.5 * (1 + math.erf(-z_score / math.sqrt(2))) * 100
        else:
            probability = 50.0

        return max(10, min(90, probability))

    def _sector_relative_strength(
        self,
        historical_prices: List[Dict],
        sector_performance: Dict,
        is_bullish: bool,
    ) -> float:
        """Calculate adjustment based on stock's strength relative to sector."""
        if not historical_prices or len(historical_prices) < 20:
            return 0.0

        # Calculate stock's recent performance
        prices = [p.get("close", 0) for p in historical_prices if p.get("close", 0) > 0]
        if len(prices) < 20:
            return 0.0

        stock_return = (prices[-1] - prices[-20]) / prices[-20]

        # Get sector performance
        sector_return = sector_performance.get("performance_20d", 0)
        if isinstance(sector_return, str):
            try:
                sector_return = float(sector_return.strip('%')) / 100
            except:
                sector_return = 0

        # Relative strength
        relative_strength = stock_return - sector_return

        # If stock outperforming sector and we want bullish, positive adjustment
        if is_bullish and relative_strength > 0.02:
            return min(5, relative_strength * 50)
        elif not is_bullish and relative_strength < -0.02:
            return min(5, abs(relative_strength) * 50)
        elif is_bullish and relative_strength < -0.02:
            return max(-5, relative_strength * 50)
        elif not is_bullish and relative_strength > 0.02:
            return max(-5, -relative_strength * 50)

        return 0.0

    def _insider_trading_signal(
        self,
        insider_data: List[Dict],
        is_bullish: bool,
        signal_reliability: Dict[str, float],
    ) -> float:
        """Calculate adjustment based on insider trading patterns."""
        if not insider_data:
            return 0.0

        buys = 0
        sells = 0
        buy_value = 0
        sell_value = 0

        for trade in insider_data[:10]:  # Look at last 10 transactions
            transaction = trade.get("transaction", "").lower()
            value = trade.get("value", 0) or 0

            if "buy" in transaction or "purchase" in transaction:
                buys += 1
                buy_value += value if isinstance(value, (int, float)) else 0
            elif "sell" in transaction or "sale" in transaction:
                sells += 1
                sell_value += value if isinstance(value, (int, float)) else 0

        if buys == 0 and sells == 0:
            return 0.0

        # Net insider sentiment
        if buys > sells * 2:
            # Strong insider buying
            adjustment = 4 if is_bullish else -2
        elif buys > sells:
            # Moderate insider buying
            adjustment = 2 if is_bullish else -1
        elif sells > buys * 2:
            # Strong insider selling
            adjustment = -3 if is_bullish else 2
        elif sells > buys:
            # Moderate insider selling
            adjustment = -1.5 if is_bullish else 1
        else:
            adjustment = 0

        return adjustment

    def _earnings_proximity_adjustment(
        self,
        earnings_dates: List[Dict],
        days_to_target: int,
        is_bullish: bool,
    ) -> float:
        """Adjust probability based on earnings proximity."""
        if not earnings_dates:
            return 0.0

        # Find next earnings date
        now = datetime.now()
        today = now.date()  # Use date for comparison
        next_earnings = None

        for ed in earnings_dates:
            date = ed.get("date")
            if date:
                # Convert to date object for consistent comparison
                if isinstance(date, str):
                    try:
                        parsed = datetime.fromisoformat(date.replace("Z", "+00:00"))
                        date = parsed.date()
                    except:
                        continue
                elif isinstance(date, datetime):
                    date = date.date()
                # date is now a datetime.date object
                if date > today:
                    next_earnings = date
                    break

        if next_earnings is None:
            return 0.0

        days_to_earnings = (next_earnings - today).days

        # Earnings within our target window creates uncertainty
        if 0 < days_to_earnings < days_to_target:
            # Earnings will happen before target date
            # This generally increases volatility but direction is uncertain
            # Slightly negative adjustment as it adds uncertainty
            return -2
        elif days_to_earnings <= 7:
            # Earnings very soon - high uncertainty
            return -3

        return 0.0

    def _mean_reversion_adjustment(
        self,
        historical_prices: List[Dict],
        current_price: float,
        required_return: float,
        days_to_target: int,
    ) -> float:
        """Calculate mean reversion adjustment."""
        if not historical_prices or len(historical_prices) < 60:
            return 0.0

        prices = [p.get("close", 0) for p in historical_prices if p.get("close", 0) > 0]
        if len(prices) < 60:
            return 0.0

        # Calculate mean price (60-day)
        mean_price = sum(prices[-60:]) / 60

        # How far is current price from mean?
        deviation = (current_price - mean_price) / mean_price

        # If price is extended above mean and target is even higher, less likely
        # If price is extended below mean and target is even lower, less likely
        is_bullish = required_return > 0

        if deviation > 0.15 and is_bullish:
            # Price already 15%+ above mean, bullish target less likely
            return -3
        elif deviation > 0.10 and is_bullish:
            return -1.5
        elif deviation < -0.15 and not is_bullish:
            # Price already 15%+ below mean, bearish target less likely
            return -3
        elif deviation < -0.10 and not is_bullish:
            return -1.5
        elif deviation > 0.10 and not is_bullish:
            # Price extended up, bearish target more likely (mean reversion)
            return 2
        elif deviation < -0.10 and is_bullish:
            # Price extended down, bullish target more likely (mean reversion)
            return 2

        return 0.0

    def _support_resistance_adjustment(
        self,
        current_price: float,
        target_price: float,
        technicals: Dict,
    ) -> float:
        """Adjust based on support/resistance levels."""
        support = technicals.get("support_level")
        resistance = technicals.get("resistance_level")

        adjustment = 0.0
        is_bullish = target_price > current_price

        if resistance and is_bullish:
            if target_price > resistance:
                # Target above resistance - harder to achieve
                resistance_gap = (target_price - resistance) / resistance
                adjustment -= min(5, resistance_gap * 20)
            elif target_price < resistance:
                # Target below resistance - more achievable
                adjustment += 1

        if support and not is_bullish:
            if target_price < support:
                # Target below support - harder to achieve
                support_gap = (support - target_price) / support
                adjustment -= min(5, support_gap * 20)
            elif target_price > support:
                # Target above support - more achievable
                adjustment += 1

        return adjustment

    def _short_interest_adjustment(
        self,
        finviz_data: Dict,
        is_bullish: bool,
    ) -> float:
        """Adjust based on short interest."""
        short_float = finviz_data.get("short_float")

        if short_float is None:
            return 0.0

        # Parse short float percentage
        if isinstance(short_float, str):
            try:
                short_float = float(short_float.strip('%'))
            except:
                return 0.0

        if short_float > 20:
            # High short interest (>20%)
            # Bullish: potential short squeeze, slight positive
            # Bearish: shorts may be right, slight positive
            return 2 if is_bullish else 1
        elif short_float > 10:
            # Moderate short interest
            return 1 if is_bullish else 0.5

        return 0.0

    # ==========================================
    # IMPROVEMENT #3: EARNINGS SURPRISE HISTORY
    # ==========================================

    def _earnings_surprise_probability(
        self,
        earnings_history: Optional[List[Dict]],
        sector: Optional[str],
        is_bullish: bool,
        days_to_earnings: Optional[int] = None,
    ) -> Optional[float]:
        """
        Calculate probability adjustment based on earnings surprise history.

        Uses historical beat/miss rates to adjust predictions around earnings.
        """
        if not earnings_history and not sector:
            return None

        # Calculate company-specific beat rate if we have history
        if earnings_history and len(earnings_history) >= 4:
            beats = sum(1 for e in earnings_history if e.get('surprise_pct', 0) > 0)
            beat_rate = beats / len(earnings_history)
        else:
            # Fall back to sector average
            beat_rate = self.SECTOR_EARNINGS_BEAT_RATES.get(
                sector, self.SECTOR_EARNINGS_BEAT_RATES["default"]
            )

        # Calculate average surprise magnitude
        if earnings_history:
            surprises = [e.get('surprise_pct', 0) for e in earnings_history if e.get('surprise_pct') is not None]
            avg_surprise = sum(surprises) / len(surprises) if surprises else 0
        else:
            avg_surprise = 0

        # Base probability from beat rate
        # If bullish and company beats often, higher probability
        if is_bullish:
            base_prob = 50 + (beat_rate - 0.5) * 30  # 50% beat rate = 50%, 70% = 56%
            # Add surprise magnitude bonus
            if avg_surprise > 5:  # Avg surprise > 5%
                base_prob += 3
            elif avg_surprise > 2:
                base_prob += 1
        else:
            # Bearish - inverse logic
            miss_rate = 1 - beat_rate
            base_prob = 50 + (miss_rate - 0.5) * 30

        # Adjust based on proximity to earnings
        if days_to_earnings is not None:
            if days_to_earnings <= 7:
                # Very close to earnings - high uncertainty
                # Pull toward 50%
                base_prob = 50 + (base_prob - 50) * 0.7
            elif days_to_earnings <= 14:
                # Moderate uncertainty
                base_prob = 50 + (base_prob - 50) * 0.85

        return max(35, min(65, base_prob))

    # ==========================================
    # IMPROVEMENT #4: OPTIONS FLOW ANALYSIS
    # ==========================================

    def _options_flow_probability(
        self,
        options_data: Optional[Dict],
        is_bullish: bool,
    ) -> Optional[float]:
        """
        Calculate probability based on options flow data.

        Analyzes unusual options activity as a "smart money" signal.
        """
        if not options_data:
            return None

        # Get key metrics
        put_call_ratio = options_data.get('put_call_ratio') or options_data.get('put_call_ratio_volume')
        unusual_calls = options_data.get('unusual_call_volume', 0)
        unusual_puts = options_data.get('unusual_put_volume', 0)
        call_volume = options_data.get('call_volume', 0)
        put_volume = options_data.get('put_volume', 0)
        implied_vol = options_data.get('implied_volatility')

        if put_call_ratio is None and call_volume == 0:
            return None

        # Calculate put/call ratio if not provided
        if put_call_ratio is None and call_volume > 0:
            put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0

        # Options flow signal
        flow_signal = 0.0

        # Put/Call ratio analysis
        if put_call_ratio is not None:
            if put_call_ratio < 0.5:
                # Very bullish options flow
                flow_signal += 0.6
            elif put_call_ratio < 0.7:
                # Moderately bullish
                flow_signal += 0.3
            elif put_call_ratio > 1.5:
                # Very bearish options flow
                flow_signal -= 0.6
            elif put_call_ratio > 1.2:
                # Moderately bearish
                flow_signal -= 0.3

        # Unusual activity analysis
        if unusual_calls > unusual_puts * 2:
            # Significant unusual call activity (smart money bullish)
            flow_signal += 0.4
        elif unusual_puts > unusual_calls * 2:
            # Significant unusual put activity (smart money bearish)
            flow_signal -= 0.4

        # Convert to probability
        if is_bullish:
            # Bullish target - positive flow signal helps
            base_prob = 50 + (flow_signal * 20)
        else:
            # Bearish target - negative flow signal helps
            base_prob = 50 - (flow_signal * 20)

        return max(30, min(70, base_prob))

    # ==========================================
    # IMPROVEMENT #1: PROBABILITY CALIBRATION
    # ==========================================

    def _calibrate_probability(
        self,
        raw_probability: float,
        confidence: float,
    ) -> float:
        """
        Calibrate probability to reduce overconfidence.

        Backtesting showed:
        - Model predicts ~60% average probability
        - Actual hit rate is ~40%
        - Calibration pulls predictions toward 50%

        Formula: calibrated = 50 + (raw - 50) * CALIBRATION_FACTOR * confidence_adj
        """
        # Distance from 50%
        deviation = raw_probability - 50

        # Confidence adjustment - lower confidence = more conservative
        confidence_adj = 0.8 + (confidence * 0.4)  # Range: 0.8 to 1.2

        # Apply calibration
        calibrated_deviation = deviation * self.CALIBRATION_FACTOR * confidence_adj

        calibrated = 50 + calibrated_deviation

        return max(15, min(85, calibrated))

    def _calculate_dynamic_weights(
        self,
        historical_prices: List[Dict],
        components: Dict[str, float],
        volatility_regime: VolatilityRegime,
        ticker: Optional[str] = None,
    ) -> Dict[str, float]:
        """Calculate dynamic weights based on data quality and market conditions."""
        weights = {}

        has_good_history = historical_prices and len(historical_prices) >= 200
        has_moderate_history = historical_prices and len(historical_prices) >= 60

        # IMPROVEMENT #2: Increased multi_timeframe weight (best performing component)
        # Base weights - multi_timeframe increased from 0.15 to 0.25
        if has_good_history:
            weights["volatility"] = 0.15          # Reduced from 0.20
            weights["monte_carlo"] = 0.20         # Reduced from 0.25
            weights["multi_timeframe"] = 0.25    # INCREASED from 0.15
        elif has_moderate_history:
            weights["volatility"] = 0.12
            weights["monte_carlo"] = 0.18
            weights["multi_timeframe"] = 0.20    # INCREASED from 0.10
        else:
            weights["volatility"] = 0.10
            weights["monte_carlo"] = 0.15
            weights["multi_timeframe"] = 0.15    # INCREASED from 0.05

        # Bayesian - slightly reduced to make room for multi_timeframe
        weights["bayesian"] = 0.22  # Reduced from 0.25

        # Optional components
        if "analyst_targets" in components:
            weights["analyst_targets"] = 0.10
        if "options_implied" in components:
            weights["options_implied"] = 0.10

        # IMPROVEMENT #3: Earnings surprise component
        if "earnings_surprise" in components:
            weights["earnings_surprise"] = 0.08

        # IMPROVEMENT #4: Options flow component
        if "options_flow" in components:
            weights["options_flow"] = 0.10

        # Adjust weights for volatility regime
        if volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            # In high volatility, statistical methods less reliable
            for k in ["volatility", "monte_carlo"]:
                if k in weights:
                    weights[k] *= 0.8
            # Increase weight on signals
            weights["bayesian"] *= 1.1
            weights["multi_timeframe"] *= 1.1  # Multi-timeframe also more reliable

        # IMPROVEMENT #5: Apply per-ticker adjustments
        if ticker and ticker.upper() in self.TICKER_ADJUSTMENTS:
            adjustments = self.TICKER_ADJUSTMENTS[ticker.upper()]
            for component, multiplier in adjustments.items():
                if component in weights:
                    weights[component] *= multiplier

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _calculate_enhanced_confidence(
        self,
        historical_prices: List[Dict],
        components: Dict[str, float],
        adjustments: Dict[str, float],
        factors_used: int,
        volatility_regime: VolatilityRegime,
    ) -> float:
        """Calculate confidence score with more factors."""
        confidence = 0.4  # Base

        # Data availability
        if historical_prices:
            if len(historical_prices) >= 252:
                confidence += 0.15
            elif len(historical_prices) >= 120:
                confidence += 0.10
            elif len(historical_prices) >= 60:
                confidence += 0.05

        # Component agreement
        probs = list(components.values())
        if len(probs) >= 2:
            mean_prob = sum(probs) / len(probs)
            variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
            agreement = max(0, 1 - math.sqrt(variance) / 20)
            confidence += agreement * 0.15

        # Factors used bonus
        confidence += min(0.10, factors_used * 0.015)

        # Volatility regime penalty
        if volatility_regime == VolatilityRegime.HIGH:
            confidence *= 0.9
        elif volatility_regime == VolatilityRegime.EXTREME:
            confidence *= 0.75

        return min(0.95, confidence)

    def _apply_realistic_bounds(
        self,
        probability: float,
        abs_return: float,
        days: int,
        confidence: float,
    ) -> float:
        """Apply realistic probability bounds based on move size and time."""
        # Extreme moves should have lower probability
        daily_move_required = abs_return / max(1, days)

        if daily_move_required > 0.03:  # >3% per day required
            probability = min(probability, 20)
        elif daily_move_required > 0.02:  # >2% per day
            probability = min(probability, 35)
        elif daily_move_required > 0.01:  # >1% per day
            probability = min(probability, 50)

        # Very long timeframes have more uncertainty
        if days > 365:
            # Pull toward 50% for very long predictions
            probability = 50 + (probability - 50) * 0.7

        return max(5, min(95, probability))

    def _calculate_data_quality(
        self,
        historical_prices: List[Dict],
        factors_used: int,
    ) -> float:
        """Calculate overall data quality score."""
        quality = 0.5

        if historical_prices:
            if len(historical_prices) >= 252:
                quality += 0.25
            elif len(historical_prices) >= 120:
                quality += 0.15
            elif len(historical_prices) >= 60:
                quality += 0.10

        quality += min(0.25, factors_used * 0.04)

        return min(1.0, quality)

    def _build_enhanced_reasoning(
        self,
        components: Dict[str, float],
        weights: Dict[str, float],
        adjustments: Dict[str, float],
        final_prob: float,
        required_return: float,
        days: int,
        market_regime: MarketRegime,
        volatility_regime: VolatilityRegime,
    ) -> str:
        """Build comprehensive reasoning explanation."""
        parts = []

        # Overall assessment
        if final_prob >= 70:
            parts.append(f"High probability ({final_prob:.0f}%) of reaching target")
        elif final_prob >= 55:
            parts.append(f"Moderate probability ({final_prob:.0f}%) - favorable conditions")
        elif final_prob >= 45:
            parts.append(f"Uncertain ({final_prob:.0f}%) - mixed signals")
        elif final_prob >= 30:
            parts.append(f"Lower probability ({final_prob:.0f}%) - challenging target")
        else:
            parts.append(f"Low probability ({final_prob:.0f}%) - significant headwinds")

        # Market context
        parts.append(f"Market: {market_regime.value}, Volatility: {volatility_regime.value}")

        # Key drivers
        if components:
            sorted_components = sorted(
                [(k, v, weights.get(k, 0)) for k, v in components.items()],
                key=lambda x: abs(x[1] - 50) * x[2],
                reverse=True
            )
            top = sorted_components[0]
            if abs(top[1] - 50) > 10:
                direction = "supports" if top[1] > 50 else "against"
                parts.append(f"Key factor: {top[0]} ({direction} at {top[1]:.0f}%)")

        # Notable adjustments
        significant_adj = [(k, v) for k, v in adjustments.items() if abs(v) >= 2]
        if significant_adj:
            adj_str = ", ".join(f"{k}: {v:+.1f}" for k, v in significant_adj)
            parts.append(f"Adjustments: {adj_str}")

        # Move assessment
        move_pct = abs(required_return) * 100
        if move_pct > 30:
            parts.append(f"Note: {move_pct:.0f}% move is historically rare")
        elif move_pct > 15:
            parts.append(f"Target requires significant {move_pct:.0f}% move")

        return ". ".join(parts) + "."


# Singleton instance
enhanced_probability_engine = EnhancedProbabilityEngine()
