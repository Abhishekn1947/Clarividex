"""
Technical Analysis Service - Calculates technical indicators.

Uses pandas-ta for comprehensive technical analysis on price data.
"""

from typing import Optional

import pandas as pd
import numpy as np
import structlog

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

from backend.app.models.schemas import TechnicalIndicators
from backend.app.services.market_data import market_data_service

logger = structlog.get_logger()


class TechnicalAnalysisService:
    """Service for calculating technical indicators."""

    def __init__(self):
        """Initialize technical analysis service."""
        self.logger = logger.bind(service="technical_analysis")

        if not HAS_PANDAS_TA:
            self.logger.warning("pandas-ta not available, using fallback calculations")

    def calculate_indicators(self, ticker: str) -> Optional[TechnicalIndicators]:
        """
        Calculate all technical indicators for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            TechnicalIndicators object or None
        """
        self.logger.info("Calculating technical indicators", ticker=ticker)

        # Get historical data (3 months for good indicator calculation)
        df = market_data_service.get_historical_data(ticker, period="3mo", interval="1d")

        if df is None or df.empty:
            self.logger.warning("No historical data available", ticker=ticker)
            return None

        # Ensure column names are lowercase
        df.columns = df.columns.str.lower()

        try:
            # Calculate indicators using pandas-ta if available
            if HAS_PANDAS_TA:
                indicators = self._calculate_with_pandas_ta(df, ticker)
            else:
                indicators = self._calculate_fallback(df, ticker)

            self.logger.info(
                "Technical indicators calculated",
                ticker=ticker,
                rsi=indicators.rsi_14,
            )

            return indicators

        except Exception as e:
            self.logger.error("Failed to calculate indicators", ticker=ticker, error=str(e))
            return None

    def _calculate_with_pandas_ta(self, df: pd.DataFrame, ticker: str) -> TechnicalIndicators:
        """Calculate indicators using pandas-ta library."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI
        rsi = ta.rsi(close, length=14)
        rsi_14 = float(rsi.iloc[-1]) if rsi is not None and not rsi.empty else None

        # MACD
        macd_result = ta.macd(close, fast=12, slow=26, signal=9)
        macd_val = None
        macd_signal = None
        macd_hist = None
        if macd_result is not None and not macd_result.empty:
            macd_cols = macd_result.columns.tolist()
            if len(macd_cols) >= 3:
                macd_val = float(macd_result.iloc[-1, 0])
                macd_hist = float(macd_result.iloc[-1, 1])
                macd_signal = float(macd_result.iloc[-1, 2])

        # Moving Averages
        sma_20 = ta.sma(close, length=20)
        sma_50 = ta.sma(close, length=50)
        sma_200 = ta.sma(close, length=200) if len(close) >= 200 else None

        ema_12 = ta.ema(close, length=12)
        ema_26 = ta.ema(close, length=26)

        # Bollinger Bands
        bbands = ta.bbands(close, length=20, std=2)
        bb_upper = None
        bb_lower = None
        if bbands is not None and not bbands.empty:
            bb_cols = bbands.columns.tolist()
            # BBands returns: BBL, BBM, BBU, BBB, BBP
            for col in bb_cols:
                if "BBU" in col:
                    bb_upper = float(bbands[col].iloc[-1])
                elif "BBL" in col:
                    bb_lower = float(bbands[col].iloc[-1])

        # ATR (Average True Range)
        atr = ta.atr(high, low, close, length=14)
        atr_14 = float(atr.iloc[-1]) if atr is not None and not atr.empty else None

        # Volume SMA
        vol_sma = ta.sma(volume, length=20)
        vol_sma_20 = float(vol_sma.iloc[-1]) if vol_sma is not None and not vol_sma.empty else None

        # Support and Resistance (simple pivot points)
        support, resistance = self._calculate_support_resistance(df)

        return TechnicalIndicators(
            ticker=ticker,
            rsi_14=round(rsi_14, 2) if rsi_14 else None,
            macd=round(macd_val, 4) if macd_val else None,
            macd_signal=round(macd_signal, 4) if macd_signal else None,
            macd_histogram=round(macd_hist, 4) if macd_hist else None,
            sma_20=round(float(sma_20.iloc[-1]), 2) if sma_20 is not None and not sma_20.empty else None,
            sma_50=round(float(sma_50.iloc[-1]), 2) if sma_50 is not None and not sma_50.empty else None,
            sma_200=round(float(sma_200.iloc[-1]), 2) if sma_200 is not None and not sma_200.empty else None,
            ema_12=round(float(ema_12.iloc[-1]), 2) if ema_12 is not None and not ema_12.empty else None,
            ema_26=round(float(ema_26.iloc[-1]), 2) if ema_26 is not None and not ema_26.empty else None,
            bollinger_upper=round(bb_upper, 2) if bb_upper else None,
            bollinger_lower=round(bb_lower, 2) if bb_lower else None,
            atr_14=round(atr_14, 2) if atr_14 else None,
            volume_sma_20=round(vol_sma_20, 0) if vol_sma_20 else None,
            support_level=round(support, 2) if support else None,
            resistance_level=round(resistance, 2) if resistance else None,
        )

    def _calculate_fallback(self, df: pd.DataFrame, ticker: str) -> TechnicalIndicators:
        """Calculate indicators without pandas-ta (basic calculations)."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI (14-day)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

        # Simple Moving Averages
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=200).mean() if len(close) >= 200 else None

        # EMA
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        # MACD
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

        # Bollinger Bands
        bb_mid = sma_20
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_mid + (bb_std * 2)
        bb_lower = bb_mid - (bb_std * 2)

        # ATR
        tr = pd.DataFrame()
        tr["h-l"] = high - low
        tr["h-pc"] = abs(high - close.shift(1))
        tr["l-pc"] = abs(low - close.shift(1))
        tr["tr"] = tr[["h-l", "h-pc", "l-pc"]].max(axis=1)
        atr = tr["tr"].rolling(window=14).mean()

        # Volume SMA
        vol_sma = volume.rolling(window=20).mean()

        # Support/Resistance
        support, resistance = self._calculate_support_resistance(df)

        return TechnicalIndicators(
            ticker=ticker,
            rsi_14=round(rsi_14, 2) if rsi_14 and not np.isnan(rsi_14) else None,
            macd=round(float(macd_line.iloc[-1]), 4) if not pd.isna(macd_line.iloc[-1]) else None,
            macd_signal=round(float(signal_line.iloc[-1]), 4) if not pd.isna(signal_line.iloc[-1]) else None,
            macd_histogram=round(float(macd_hist.iloc[-1]), 4) if not pd.isna(macd_hist.iloc[-1]) else None,
            sma_20=round(float(sma_20.iloc[-1]), 2) if not pd.isna(sma_20.iloc[-1]) else None,
            sma_50=round(float(sma_50.iloc[-1]), 2) if not pd.isna(sma_50.iloc[-1]) else None,
            sma_200=round(float(sma_200.iloc[-1]), 2) if sma_200 is not None and not pd.isna(sma_200.iloc[-1]) else None,
            ema_12=round(float(ema_12.iloc[-1]), 2) if not pd.isna(ema_12.iloc[-1]) else None,
            ema_26=round(float(ema_26.iloc[-1]), 2) if not pd.isna(ema_26.iloc[-1]) else None,
            bollinger_upper=round(float(bb_upper.iloc[-1]), 2) if not pd.isna(bb_upper.iloc[-1]) else None,
            bollinger_lower=round(float(bb_lower.iloc[-1]), 2) if not pd.isna(bb_lower.iloc[-1]) else None,
            atr_14=round(float(atr.iloc[-1]), 2) if not pd.isna(atr.iloc[-1]) else None,
            volume_sma_20=round(float(vol_sma.iloc[-1]), 0) if not pd.isna(vol_sma.iloc[-1]) else None,
            support_level=round(support, 2) if support else None,
            resistance_level=round(resistance, 2) if resistance else None,
        )

    def _calculate_support_resistance(self, df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate basic support and resistance levels.

        Uses recent pivot points and price levels.
        """
        try:
            # Use last 20 days
            recent = df.tail(20)

            # Simple approach: recent lows and highs
            low_prices = recent["low"].values
            high_prices = recent["high"].values

            # Support: find clusters of lows
            support = float(np.percentile(low_prices, 25))

            # Resistance: find clusters of highs
            resistance = float(np.percentile(high_prices, 75))

            return support, resistance

        except Exception:
            return None, None

    def get_technical_signal(self, indicators: TechnicalIndicators) -> dict:
        """
        Get trading signal based on technical indicators.

        Args:
            indicators: TechnicalIndicators object

        Returns:
            Dict with signal and analysis
        """
        signals = []
        score = 0

        # RSI Signal
        if indicators.rsi_14:
            if indicators.rsi_14 < 30:
                signals.append({"indicator": "RSI", "signal": "bullish", "reason": "Oversold (RSI < 30)"})
                score += 1
            elif indicators.rsi_14 > 70:
                signals.append({"indicator": "RSI", "signal": "bearish", "reason": "Overbought (RSI > 70)"})
                score -= 1
            else:
                signals.append({"indicator": "RSI", "signal": "neutral", "reason": f"RSI at {indicators.rsi_14:.1f}"})

        # MACD Signal
        if indicators.macd is not None and indicators.macd_signal is not None:
            if indicators.macd > indicators.macd_signal:
                signals.append({"indicator": "MACD", "signal": "bullish", "reason": "MACD above signal line"})
                score += 1
            else:
                signals.append({"indicator": "MACD", "signal": "bearish", "reason": "MACD below signal line"})
                score -= 1

        # Moving Average Trend
        if indicators.sma_20 and indicators.sma_50:
            if indicators.sma_20 > indicators.sma_50:
                signals.append({"indicator": "SMA", "signal": "bullish", "reason": "SMA20 > SMA50 (uptrend)"})
                score += 1
            else:
                signals.append({"indicator": "SMA", "signal": "bearish", "reason": "SMA20 < SMA50 (downtrend)"})
                score -= 1

        # Bollinger Band Position
        if indicators.bollinger_upper and indicators.bollinger_lower and indicators.sma_20:
            current_price = indicators.sma_20  # Approximate
            if current_price < indicators.bollinger_lower:
                signals.append({"indicator": "Bollinger", "signal": "bullish", "reason": "Price below lower band"})
                score += 0.5
            elif current_price > indicators.bollinger_upper:
                signals.append({"indicator": "Bollinger", "signal": "bearish", "reason": "Price above upper band"})
                score -= 0.5

        # Calculate overall signal
        if score > 1:
            overall = "strong_bullish"
        elif score > 0:
            overall = "bullish"
        elif score < -1:
            overall = "strong_bearish"
        elif score < 0:
            overall = "bearish"
        else:
            overall = "neutral"

        return {
            "overall_signal": overall,
            "score": round(score, 2),
            "normalized_score": round(max(-1, min(1, score / 4)), 3),  # Normalize to [-1, 1]
            "signals": signals,
        }


# Create singleton instance
technical_analysis_service = TechnicalAnalysisService()
