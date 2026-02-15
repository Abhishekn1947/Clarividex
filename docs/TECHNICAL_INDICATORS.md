# Technical Indicators Guide

> Understanding the technical analysis indicators used by Clarividex.

---

## What is Technical Analysis?

Technical analysis studies historical price patterns and trading volumes to forecast future price movements. It's based on the principle that market prices reflect all available information and tend to move in trends.

---

## RSI (Relative Strength Index)

### What it is
A momentum oscillator that measures the speed and magnitude of recent price changes on a scale of 0-100.

### Formula
```
RSI = 100 - (100 / (1 + RS))

Where:
RS = Average Gain / Average Loss (over 14 periods)
```

### Interpretation

| RSI Value | Zone | Signal | Score |
|-----------|------|--------|-------|
| 0-30 | **Oversold** | Stock heavily sold, may be undervalued | Bullish (+0.85) |
| 30-40 | Near Oversold | Approaching oversold | Slightly Bullish (+0.50) |
| 40-60 | **Neutral** | Normal trading range | Neutral (0.00) |
| 60-70 | Strong Momentum | Not extreme | Slightly Bullish (+0.40) |
| 70-100 | **Overbought** | Stock heavily bought, may pull back | Bearish (-0.85) |

### Real-World Analogy
Think of RSI like a rubber band. When stretched too far in one direction (overbought or oversold), it tends to snap back.

---

## MACD (Moving Average Convergence Divergence)

### What it is
A trend-following momentum indicator showing the relationship between two moving averages.

### Components
```
MACD Line = 12-day EMA - 26-day EMA
Signal Line = 9-day EMA of MACD Line
Histogram = MACD Line - Signal Line
```

### Interpretation

| Signal | Meaning | Score |
|--------|---------|-------|
| MACD > Signal | Bullish crossover | +0.60 |
| MACD < Signal | Bearish crossover | -0.60 |
| Positive Histogram (growing) | Strong bullish momentum | +0.70 |
| Negative Histogram (shrinking) | Strong bearish momentum | -0.70 |

### Real-World Analogy
MACD compares short-term vs long-term market mood. When short-term mood overtakes long-term, it signals a trend change.

---

## Moving Averages (SMA)

### What it is
The average closing price over a specific number of days, smoothing out price fluctuations.

### Types Used
- **SMA 20** (20-day) - Short-term trend
- **SMA 50** (50-day) - Medium-term trend
- **SMA 200** (200-day) - Long-term trend (most important)

### Interpretation

| Signal | Meaning | Score |
|--------|---------|-------|
| Price > SMA200 | Long-term uptrend | +0.40 |
| Price < SMA200 | Long-term downtrend | -0.40 |
| SMA20 > SMA50 | Short-term bullish (Golden Cross forming) | +0.30 |
| SMA20 < SMA50 | Short-term bearish (Death Cross forming) | -0.30 |
| Price > SMA20 > SMA50 | Strong bullish alignment | +0.70 |
| Price < SMA20 < SMA50 | Strong bearish alignment | -0.70 |

### Key Patterns

**Golden Cross**
- 50-day SMA crosses above 200-day SMA
- Strong bullish signal
- Historical success rate: ~65%

**Death Cross**
- 50-day SMA crosses below 200-day SMA
- Strong bearish signal
- Historical success rate: ~60%

### Real-World Analogy
Moving averages are like looking at a photo vs. a time-lapse video. The 200-day shows the "big picture" direction.

---

## Bollinger Bands

### What it is
Volatility bands placed above and below a moving average. They widen when volatility increases and narrow when it decreases.

### Components
```
Middle Band = 20-day SMA
Upper Band = Middle + (2 × Standard Deviation)
Lower Band = Middle - (2 × Standard Deviation)
```

### Interpretation

| Price Position | Meaning |
|----------------|---------|
| Price touches Upper Band | Potentially overbought, may pull back |
| Price touches Lower Band | Potentially oversold, may bounce |
| Bands narrowing ("Squeeze") | Low volatility, big move coming |
| Bands widening | High volatility, trend in progress |

---

## Support and Resistance

### Definitions
- **Support** = Price level where buying interest prevents further decline
- **Resistance** = Price level where selling interest prevents further rise

### Why They Matter
- Prices tend to "bounce" off these levels
- Breaking through often leads to significant moves
- Former resistance becomes support (and vice versa)

### Signals

| Scenario | Score |
|----------|-------|
| Near strong support (bullish target) | +0.40 |
| Near strong resistance (bullish target) | -0.40 |
| Breakout above resistance | +0.60 |
| Breakdown below support | -0.60 |

---

## Volume Analysis

### What it is
The number of shares traded during a given period.

### Interpretation

| Signal | Meaning |
|--------|---------|
| Price up + High volume | Strong buying, bullish |
| Price up + Low volume | Weak rally, may fade |
| Price down + High volume | Strong selling, bearish |
| Price down + Low volume | Weak decline, may bounce |

### Volume Spike
Volume > 2x average often signals significant institutional activity.

---

## How Clarividex Combines Indicators

### Weighted Technical Score

```
Technical Score =
    (RSI_signal × 0.25) +
    (MACD_signal × 0.20) +
    (SMA_signal × 0.25) +
    (Support_Resistance × 0.15) +
    (Volume_signal × 0.15)
```

### Agreement Bonus
When multiple indicators agree:
- 4+ indicators aligned: +10% confidence boost
- 3 indicators aligned: +5% confidence boost
- Mixed signals: No adjustment

---

## Pattern Recognition

### Oversold Bounce
**Conditions:**
- RSI < 30
- Price > SMA200

**Expected:** +5.5% move, 62% win rate

### Overbought Reversal
**Conditions:**
- RSI > 70
- MACD bearish crossover

**Expected:** -4.5% move, 58% win rate

### Breakout Continuation
**Conditions:**
- Price breaks above resistance
- Volume > 2x average
- RSI 50-70

**Expected:** +6.5% move, 55% win rate

---

## Glossary

| Term | Definition |
|------|------------|
| **Bullish** | Expecting price to rise |
| **Bearish** | Expecting price to fall |
| **EMA** | Exponential Moving Average - more weight to recent prices |
| **SMA** | Simple Moving Average - equal weight to all prices |
| **Volatility** | How much price fluctuates |
| **Momentum** | Rate of price change |
| **Trend** | General direction of price movement |
| **Breakout** | Price moving outside a defined range |
| **Pullback** | Temporary reversal against the main trend |
| **Consolidation** | Period of sideways movement |

---

*Last Updated: February 2026*
