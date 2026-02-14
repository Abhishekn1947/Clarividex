"""
Backtesting Engine for Clarividex.

This module validates prediction accuracy against historical data by:
1. Simulating predictions at past dates using only data available at that time
2. Comparing predicted probabilities against actual outcomes
3. Tracking accuracy metrics per signal type
4. Identifying which factors are most predictive

Usage:
    from backend.app.services.backtesting_engine import backtesting_engine

    results = await backtesting_engine.run_backtest(
        ticker="NVDA",
        target_return=0.10,  # +10%
        holding_period=30,   # 30 days
        lookback_days=252,   # Test over 1 year
    )
    print(f"Accuracy: {results.accuracy:.1%}")
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

import structlog

from backend.app.services.enhanced_probability_engine import (
    EnhancedProbabilityEngine,
    EnhancedProbabilityResult,
)

logger = structlog.get_logger()


class PredictionOutcome(Enum):
    """Outcome of a prediction."""
    CORRECT_BULLISH = "correct_bullish"      # Predicted high prob, target hit
    CORRECT_BEARISH = "correct_bearish"      # Predicted low prob, target missed
    FALSE_POSITIVE = "false_positive"         # Predicted high prob, target missed
    FALSE_NEGATIVE = "false_negative"         # Predicted low prob, target hit


@dataclass
class BacktestPrediction:
    """Single backtested prediction."""
    date: str
    current_price: float
    target_price: float
    holding_period: int
    predicted_probability: float
    predicted_confidence: float
    actual_hit: bool
    actual_return: float
    outcome: PredictionOutcome
    signals_used: Dict[str, float] = field(default_factory=dict)
    components: Dict[str, float] = field(default_factory=dict)
    market_regime: str = ""
    volatility_regime: str = ""


@dataclass
class SignalAccuracy:
    """Accuracy metrics for a specific signal."""
    signal_name: str
    times_fired: int
    times_correct: int
    accuracy: float
    avg_probability_when_fired: float
    contribution_to_accuracy: float


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    ticker: str
    target_return: float
    holding_period: int
    total_predictions: int
    correct_predictions: int
    accuracy: float

    # Detailed metrics
    precision: float          # True positives / (True positives + False positives)
    recall: float             # True positives / (True positives + False negatives)
    f1_score: float           # Harmonic mean of precision and recall

    # Probability calibration
    avg_probability: float
    avg_actual_hit_rate: float
    calibration_error: float  # How well probabilities match reality

    # By regime
    accuracy_by_regime: Dict[str, float] = field(default_factory=dict)
    accuracy_by_volatility: Dict[str, float] = field(default_factory=dict)

    # Signal analysis
    signal_accuracies: List[SignalAccuracy] = field(default_factory=list)

    # Component analysis
    component_correlations: Dict[str, float] = field(default_factory=dict)

    # All predictions
    predictions: List[BacktestPrediction] = field(default_factory=list)

    # Summary
    summary: str = ""


class BacktestingEngine:
    """
    Engine for backtesting prediction accuracy.

    Tests predictions against historical data to measure:
    - Overall accuracy
    - Probability calibration
    - Signal effectiveness
    - Regime-specific performance
    """

    def __init__(self):
        self.logger = logger.bind(service="backtesting_engine")
        self.probability_engine = EnhancedProbabilityEngine()

    async def run_backtest(
        self,
        ticker: str,
        target_return: float = 0.10,
        holding_period: int = 30,
        lookback_days: int = 252,
        step_days: int = 5,
        probability_threshold: float = 50.0,
    ) -> BacktestResults:
        """
        Run a comprehensive backtest.

        Args:
            ticker: Stock ticker symbol
            target_return: Target return (0.10 = +10%)
            holding_period: Days to hold before checking outcome
            lookback_days: How many days of history to test
            step_days: Days between each test (5 = weekly tests)
            probability_threshold: Threshold for "bullish" prediction

        Returns:
            BacktestResults with accuracy metrics
        """
        self.logger.info(
            "Starting backtest",
            ticker=ticker,
            target_return=f"{target_return:.1%}",
            holding_period=holding_period,
            lookback_days=lookback_days,
        )

        # Fetch historical data
        historical_prices = await self._fetch_historical_data(ticker, lookback_days + holding_period + 200)

        if not historical_prices or len(historical_prices) < lookback_days + holding_period:
            self.logger.error("Insufficient historical data")
            return BacktestResults(
                ticker=ticker,
                target_return=target_return,
                holding_period=holding_period,
                total_predictions=0,
                correct_predictions=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                avg_probability=0.0,
                avg_actual_hit_rate=0.0,
                calibration_error=0.0,
                summary="Insufficient historical data for backtest",
            )

        predictions = []

        # Run predictions at regular intervals
        # Start from lookback_days ago, end at holding_period days ago (so we can check outcome)
        start_idx = 200  # Need 200 days of history for indicators
        end_idx = len(historical_prices) - holding_period

        for i in range(start_idx, end_idx, step_days):
            # Get data available on this simulated date
            available_data = historical_prices[:i+1]
            current_price = available_data[-1]['close']

            # Calculate target price
            is_bullish = target_return > 0
            target_price = current_price * (1 + target_return)

            # Get actual outcome
            future_idx = i + holding_period
            if future_idx >= len(historical_prices):
                break

            future_price = historical_prices[future_idx]['close']
            actual_return = (future_price - current_price) / current_price

            if is_bullish:
                actual_hit = future_price >= target_price
            else:
                actual_hit = future_price <= target_price

            # Make prediction using only available data
            prediction_result = self._make_prediction(
                current_price=current_price,
                target_price=target_price,
                days=holding_period,
                historical_prices=available_data,
            )

            # Determine outcome
            predicted_bullish = prediction_result.probability > probability_threshold

            if predicted_bullish and actual_hit:
                outcome = PredictionOutcome.CORRECT_BULLISH
            elif not predicted_bullish and not actual_hit:
                outcome = PredictionOutcome.CORRECT_BEARISH
            elif predicted_bullish and not actual_hit:
                outcome = PredictionOutcome.FALSE_POSITIVE
            else:
                outcome = PredictionOutcome.FALSE_NEGATIVE

            prediction = BacktestPrediction(
                date=available_data[-1].get('date', str(i)),
                current_price=current_price,
                target_price=target_price,
                holding_period=holding_period,
                predicted_probability=prediction_result.probability,
                predicted_confidence=prediction_result.confidence,
                actual_hit=actual_hit,
                actual_return=actual_return,
                outcome=outcome,
                components=prediction_result.components,
                market_regime=prediction_result.market_regime,
                volatility_regime=prediction_result.volatility_regime,
            )

            predictions.append(prediction)

        # Calculate results
        results = self._calculate_results(
            ticker=ticker,
            target_return=target_return,
            holding_period=holding_period,
            predictions=predictions,
            probability_threshold=probability_threshold,
        )

        self.logger.info(
            "Backtest complete",
            ticker=ticker,
            total_predictions=results.total_predictions,
            accuracy=f"{results.accuracy:.1%}",
            calibration_error=f"{results.calibration_error:.2f}",
        )

        return results

    async def _fetch_historical_data(self, ticker: str, days: int) -> List[Dict]:
        """Fetch historical price data."""
        try:
            import yfinance as yf

            # Calculate period
            if days > 730:
                period = "max"
            elif days > 365:
                period = "2y"
            else:
                period = "1y"

            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                return []

            prices = []
            for idx, row in df.iterrows():
                prices.append({
                    'date': str(idx.date()) if hasattr(idx, 'date') else str(idx),
                    'open': float(row.get('Open', 0)),
                    'high': float(row.get('High', 0)),
                    'low': float(row.get('Low', 0)),
                    'close': float(row.get('Close', 0)),
                    'volume': int(row.get('Volume', 0)),
                })

            return prices

        except Exception as e:
            self.logger.error("Failed to fetch historical data", error=str(e))
            return []

    def _make_prediction(
        self,
        current_price: float,
        target_price: float,
        days: int,
        historical_prices: List[Dict],
    ) -> EnhancedProbabilityResult:
        """Make a prediction using only available historical data."""

        # Build signals from historical data
        signals = self._build_signals_from_history(historical_prices)

        # Build technicals
        technicals = self._build_technicals_from_history(historical_prices)

        # Calculate prediction
        result = self.probability_engine.calculate_probability(
            current_price=current_price,
            target_price=target_price,
            days_to_target=days,
            historical_prices=historical_prices,
            signals=signals,
            technicals=technicals,
            news_sentiment=None,  # Not available in backtest
            options_data=None,
            analyst_data=None,
            insider_data=None,
            earnings_dates=None,
            vix_data=None,
            fear_greed=None,
            sector_performance=None,
            finviz_data=None,
        )

        return result

    def _build_signals_from_history(self, prices: List[Dict]) -> Dict[str, float]:
        """Build trading signals from price history."""
        if len(prices) < 20:
            return {}

        signals = {}
        closes = [p['close'] for p in prices if p['close'] > 0]

        if len(closes) < 20:
            return signals

        # Calculate RSI
        rsi = self._calculate_rsi(closes)
        if rsi is not None:
            if rsi < 30:
                signals['rsi_oversold'] = 0.8
            elif rsi > 70:
                signals['rsi_overbought'] = -0.8
            elif rsi < 40:
                signals['rsi_low'] = 0.3
            elif rsi > 60:
                signals['rsi_high'] = -0.3

        # Price vs SMAs
        if len(closes) >= 50:
            sma_20 = sum(closes[-20:]) / 20
            sma_50 = sum(closes[-50:]) / 50
            current = closes[-1]

            if current > sma_20 > sma_50:
                signals['bullish_alignment'] = 0.6
            elif current < sma_20 < sma_50:
                signals['bearish_alignment'] = -0.6

        # Momentum
        if len(closes) >= 10:
            momentum = (closes[-1] - closes[-10]) / closes[-10]
            if momentum > 0.05:
                signals['strong_momentum'] = 0.5
            elif momentum < -0.05:
                signals['weak_momentum'] = -0.5

        return signals

    def _build_technicals_from_history(self, prices: List[Dict]) -> Dict:
        """Build technical indicators from price history."""
        if len(prices) < 20:
            return {}

        closes = [p['close'] for p in prices if p['close'] > 0]

        if len(closes) < 20:
            return {}

        technicals = {
            'current_price': closes[-1],
        }

        # RSI
        rsi = self._calculate_rsi(closes)
        if rsi is not None:
            technicals['rsi_14'] = rsi

        # SMAs
        if len(closes) >= 20:
            technicals['sma_20'] = sum(closes[-20:]) / 20
        if len(closes) >= 50:
            technicals['sma_50'] = sum(closes[-50:]) / 50
        if len(closes) >= 200:
            technicals['sma_200'] = sum(closes[-200:]) / 200

        # MACD
        if len(closes) >= 26:
            ema_12 = self._calculate_ema(closes, 12)
            ema_26 = self._calculate_ema(closes, 26)
            if ema_12 and ema_26:
                technicals['macd'] = ema_12 - ema_26
                technicals['macd_signal'] = self._calculate_ema(
                    [technicals['macd']] * 9, 9
                ) or 0

        # Support/Resistance (simple version)
        if len(closes) >= 20:
            recent = closes[-20:]
            technicals['support_level'] = min(recent) * 0.98
            technicals['resistance_level'] = max(recent) * 1.02

        return technicals

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return None

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]

        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate EMA."""
        if len(prices) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_results(
        self,
        ticker: str,
        target_return: float,
        holding_period: int,
        predictions: List[BacktestPrediction],
        probability_threshold: float,
    ) -> BacktestResults:
        """Calculate comprehensive backtest results."""

        if not predictions:
            return BacktestResults(
                ticker=ticker,
                target_return=target_return,
                holding_period=holding_period,
                total_predictions=0,
                correct_predictions=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                avg_probability=0.0,
                avg_actual_hit_rate=0.0,
                calibration_error=0.0,
                summary="No predictions generated",
            )

        total = len(predictions)

        # Count outcomes
        true_positives = sum(1 for p in predictions if p.outcome == PredictionOutcome.CORRECT_BULLISH)
        true_negatives = sum(1 for p in predictions if p.outcome == PredictionOutcome.CORRECT_BEARISH)
        false_positives = sum(1 for p in predictions if p.outcome == PredictionOutcome.FALSE_POSITIVE)
        false_negatives = sum(1 for p in predictions if p.outcome == PredictionOutcome.FALSE_NEGATIVE)

        correct = true_positives + true_negatives
        accuracy = correct / total if total > 0 else 0

        # Precision and Recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Probability calibration
        avg_prob = sum(p.predicted_probability for p in predictions) / total
        avg_hit_rate = sum(1 for p in predictions if p.actual_hit) / total * 100
        calibration_error = abs(avg_prob - avg_hit_rate)

        # By regime
        accuracy_by_regime = {}
        for regime in ['bull', 'bear', 'sideways']:
            regime_preds = [p for p in predictions if p.market_regime == regime]
            if regime_preds:
                regime_correct = sum(1 for p in regime_preds if p.outcome in [PredictionOutcome.CORRECT_BULLISH, PredictionOutcome.CORRECT_BEARISH])
                accuracy_by_regime[regime] = regime_correct / len(regime_preds)

        # By volatility
        accuracy_by_vol = {}
        for vol in ['low', 'normal', 'high', 'extreme']:
            vol_preds = [p for p in predictions if p.volatility_regime == vol]
            if vol_preds:
                vol_correct = sum(1 for p in vol_preds if p.outcome in [PredictionOutcome.CORRECT_BULLISH, PredictionOutcome.CORRECT_BEARISH])
                accuracy_by_vol[vol] = vol_correct / len(vol_preds)

        # Component correlations (which components predict accuracy best)
        component_correlations = self._calculate_component_correlations(predictions)

        # Build summary
        summary = self._build_summary(
            ticker=ticker,
            target_return=target_return,
            holding_period=holding_period,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            calibration_error=calibration_error,
            total=total,
            accuracy_by_regime=accuracy_by_regime,
        )

        return BacktestResults(
            ticker=ticker,
            target_return=target_return,
            holding_period=holding_period,
            total_predictions=total,
            correct_predictions=correct,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_probability=avg_prob,
            avg_actual_hit_rate=avg_hit_rate,
            calibration_error=calibration_error,
            accuracy_by_regime=accuracy_by_regime,
            accuracy_by_volatility=accuracy_by_vol,
            component_correlations=component_correlations,
            predictions=predictions,
            summary=summary,
        )

    def _calculate_component_correlations(
        self,
        predictions: List[BacktestPrediction],
    ) -> Dict[str, float]:
        """Calculate how well each component correlates with actual outcomes."""
        correlations = {}

        # Get all component names
        all_components = set()
        for p in predictions:
            all_components.update(p.components.keys())

        for component in all_components:
            # Get predictions where this component was used
            component_preds = [p for p in predictions if component in p.components]

            if len(component_preds) < 10:
                continue

            # Calculate correlation between component value and actual outcome
            # Simple approach: when component predicted bullish (>50), did outcome match?
            correct = 0
            for p in component_preds:
                comp_value = p.components[component]
                comp_bullish = comp_value > 50

                if (comp_bullish and p.actual_hit) or (not comp_bullish and not p.actual_hit):
                    correct += 1

            correlations[component] = correct / len(component_preds)

        # Sort by correlation
        correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))

        return correlations

    def _build_summary(
        self,
        ticker: str,
        target_return: float,
        holding_period: int,
        accuracy: float,
        precision: float,
        recall: float,
        calibration_error: float,
        total: int,
        accuracy_by_regime: Dict[str, float],
    ) -> str:
        """Build human-readable summary."""

        lines = [
            f"=== BACKTEST RESULTS FOR {ticker} ===",
            f"",
            f"Target: {target_return:+.1%} over {holding_period} days",
            f"Total Predictions: {total}",
            f"",
            f"ACCURACY METRICS:",
            f"  Overall Accuracy: {accuracy:.1%}",
            f"  Precision: {precision:.1%}",
            f"  Recall: {recall:.1%}",
            f"  Calibration Error: {calibration_error:.1f}%",
            f"",
        ]

        # Interpretation
        if accuracy >= 0.60:
            lines.append("INTERPRETATION: Good predictive power. Model is significantly better than random.")
        elif accuracy >= 0.55:
            lines.append("INTERPRETATION: Moderate predictive power. Model has an edge but needs improvement.")
        elif accuracy >= 0.52:
            lines.append("INTERPRETATION: Slight edge. Model is marginally better than random.")
        else:
            lines.append("INTERPRETATION: No predictive power. Model performs at or below random chance.")

        lines.append("")

        # By regime
        if accuracy_by_regime:
            lines.append("BY MARKET REGIME:")
            for regime, acc in accuracy_by_regime.items():
                lines.append(f"  {regime.capitalize()}: {acc:.1%}")
            lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS:")
        if accuracy < 0.55:
            lines.append("  - Consider ensemble methods to combine multiple models")
            lines.append("  - Add more data sources (options flow, earnings history)")
            lines.append("  - Implement adaptive signal weights based on recent accuracy")
        elif calibration_error > 10:
            lines.append("  - Calibrate probabilities (predictions are over/under confident)")
            lines.append("  - Adjust probability bounds based on historical performance")
        else:
            lines.append("  - Model performing well, consider expanding to more tickers")
            lines.append("  - Monitor for regime changes that may affect accuracy")

        return "\n".join(lines)

    async def run_multi_ticker_backtest(
        self,
        tickers: List[str],
        target_return: float = 0.10,
        holding_period: int = 30,
        lookback_days: int = 252,
    ) -> Dict[str, BacktestResults]:
        """Run backtest across multiple tickers."""
        results = {}

        for ticker in tickers:
            self.logger.info(f"Backtesting {ticker}...")
            result = await self.run_backtest(
                ticker=ticker,
                target_return=target_return,
                holding_period=holding_period,
                lookback_days=lookback_days,
            )
            results[ticker] = result

        # Calculate aggregate stats
        total_preds = sum(r.total_predictions for r in results.values())
        total_correct = sum(r.correct_predictions for r in results.values())

        if total_preds > 0:
            overall_accuracy = total_correct / total_preds
            self.logger.info(
                "Multi-ticker backtest complete",
                tickers=len(tickers),
                total_predictions=total_preds,
                overall_accuracy=f"{overall_accuracy:.1%}",
            )

        return results

    async def analyze_signal_effectiveness(
        self,
        ticker: str,
        lookback_days: int = 252,
    ) -> Dict[str, Dict]:
        """Analyze which signals are most effective for a ticker."""

        # Run backtest with detailed signal tracking
        results = await self.run_backtest(
            ticker=ticker,
            target_return=0.10,
            holding_period=30,
            lookback_days=lookback_days,
        )

        # Analyze component correlations
        analysis = {
            'ticker': ticker,
            'overall_accuracy': results.accuracy,
            'component_effectiveness': results.component_correlations,
            'best_components': [],
            'worst_components': [],
            'recommendations': [],
        }

        # Find best and worst
        sorted_components = sorted(
            results.component_correlations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if sorted_components:
            analysis['best_components'] = sorted_components[:3]
            analysis['worst_components'] = sorted_components[-3:]

            # Recommendations
            for comp, acc in sorted_components[:3]:
                if acc > 0.55:
                    analysis['recommendations'].append(
                        f"INCREASE weight for '{comp}' (accuracy: {acc:.1%})"
                    )

            for comp, acc in sorted_components[-3:]:
                if acc < 0.50:
                    analysis['recommendations'].append(
                        f"DECREASE weight for '{comp}' (accuracy: {acc:.1%})"
                    )

        return analysis


# Singleton instance
backtesting_engine = BacktestingEngine()


# CLI interface for running backtests
async def main():
    """Run backtest from command line."""
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"

    print(f"\nRunning backtest for {ticker}...")
    print("=" * 50)

    results = await backtesting_engine.run_backtest(
        ticker=ticker,
        target_return=0.10,  # +10%
        holding_period=30,   # 30 days
        lookback_days=252,   # 1 year
        step_days=5,         # Weekly tests
    )

    print(results.summary)
    print("\n" + "=" * 50)

    # Component analysis
    if results.component_correlations:
        print("\nCOMPONENT EFFECTIVENESS:")
        for comp, acc in list(results.component_correlations.items())[:5]:
            bar = "█" * int(acc * 20)
            print(f"  {comp:20} {acc:.1%} {bar}")

    print("\n" + "=" * 50)
    print(f"\nBacktest complete. {results.total_predictions} predictions analyzed.")


if __name__ == "__main__":
    asyncio.run(main())
