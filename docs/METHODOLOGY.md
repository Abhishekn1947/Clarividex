# Clarividex Prediction Methodology

> How Clarividex generates probability-based financial market predictions.

---

## Table of Contents

1. [Overview & Philosophy](#1-overview--philosophy)
2. [Main Prediction Flow](#2-main-prediction-flow)
3. [Eight-Factor Weighted Model](#3-eight-factor-weighted-model)
4. [Probability Calculation](#4-probability-calculation)
5. [Confidence Levels](#5-confidence-levels)
6. [Decision Trail Visualization](#6-decision-trail-visualization)
7. [AI Model Selection](#7-ai-model-selection)
8. [Supported Instruments](#8-supported-instruments)
9. [Output Interpretation](#9-output-interpretation)
10. [Disclaimers](#10-disclaimers)

---

## 1. Overview & Philosophy

Clarividex is designed to provide **transparent, data-driven probability assessments** for financial market predictions.

### Key Principles

1. **Transparency First**: Every prediction shows exactly how it was calculated
2. **Multi-Source Analysis**: We aggregate data from 10+ independent sources
3. **Weighted Consensus**: Different signals have different weights based on predictive value
4. **Calibrated Confidence**: We never claim certainty (probabilities capped at 15-85%)
5. **Full Reasoning Chain**: Every factor contributing to the prediction is visible

### What We DO NOT Do

- Provide financial advice
- Guarantee any outcomes
- Use insider information
- Make predictions beyond 85% certainty
- Ignore conflicting signals

---

## 2. Main Prediction Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER QUERY                                  │
│              "Will NVDA reach $150 by March 2026?"              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  QUERY VALIDATION                               │
│    ┌─────────────────┐                                          │
│    │ Financial Query │──Yes──▶ Continue                         │
│    │   Validator     │                                          │
│    └────────┬────────┘                                          │
│             │                                                   │
│            No                                                   │
│             ▼                                                   │
│    Return Error: "Non-financial query rejected"                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                INSTRUMENT DETECTION                             │
│                                                                 │
│    Query ──▶ Detect Type ──▶ Extract Symbol                     │
│                                                                 │
│    Types: Stock | Crypto | Forex | Commodity | Index |          │
│           ETF | Futures | Bond                                  │
│                                                                 │
│    Example: "NVDA" ──▶ STOCK ──▶ Symbol: NVDA                   │
│    Example: "Bitcoin" ──▶ CRYPTO ──▶ Symbol: BTC-USD            │
│    Example: "Gold" ──▶ COMMODITY ──▶ Symbol: GC=F               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATA AGGREGATION                              │
│                                                                 │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│    │   yfinance   │  │   Finnhub    │  │  News APIs   │         │
│    │  (quotes,    │  │  (if avail)  │  │  (headlines) │         │
│    │  technicals) │  │              │  │              │         │
│    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│           │                 │                 │                 │
│           └────────────┬────┴────────────────┘                  │
│                        ▼                                        │
│              ┌─────────────────┐                                │
│              │  Aggregated     │                                │
│              │  Market Data    │                                │
│              └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AI ANALYSIS                                   │
│                                                                 │
│         ┌────────────────────────────────────┐                  │
│         │  Try: Claude API (Primary)         │                  │
│         │  Fallback: Ollama (Local LLM)      │                  │
│         │  Last Resort: Rule-Based Engine    │                  │
│         └────────────────────────────────────┘                  │
│                                                                 │
│    Applies 8-Factor Weighted Model (see Section 3)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                PROBABILITY OUTPUT                               │
│                                                                 │
│    ┌──────────────────────────────────────────────────────┐     │
│    │  Probability: 62%                                    │     │
│    │  Confidence: Medium-High                             │     │
│    │  Sentiment: Bullish                                  │     │
│    │                                                      │     │
│    │  Bullish Factors:                                    │     │
│    │    - RSI at 45 (neutral, room to rise)              │     │
│    │    - Analyst target $165 (+18% upside)              │     │
│    │    - News sentiment positive (0.35 score)           │     │
│    │                                                      │     │
│    │  Bearish Factors:                                    │     │
│    │    - VIX at 22 (elevated volatility)                │     │
│    │    - Short interest at 3.2%                         │     │
│    └──────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Eight-Factor Weighted Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    FACTOR WEIGHTS (8-Factor Model)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ████████████████████████     Technical Analysis    20%         │
│  ██████████████████           News Sentiment        15%         │
│  █████████████                Historical News       10%         │
│  ███████████████              Options Flow          12%         │
│  ███████████████              Market Sentiment      12%         │
│  ████████████████             Analyst Ratings       13%         │
│  ██████████                   Social Media           8%         │
│  █████████████                Historical Patterns   10%         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Technical Analysis (20%)

| Indicator | Bullish Signal | Bearish Signal |
|-----------|----------------|----------------|
| RSI < 30 | Oversold (+0.85) | - |
| RSI > 70 | - | Overbought (-0.85) |
| Price > SMA20 > SMA50 | Bullish Trend (+0.70) | - |
| Price < SMA20 < SMA50 | - | Bearish Trend (-0.70) |
| MACD > Signal | Bullish (+0.50) | - |
| MACD < Signal | - | Bearish (-0.50) |

### 3.2 News Sentiment (15%)

| Aggregate Score | Interpretation | Signal |
|-----------------|----------------|--------|
| > +0.20 | Strongly Positive | Bullish (+0.70) |
| +0.10 to +0.20 | Positive | Mildly Bullish (+0.40) |
| -0.10 to +0.10 | Neutral | Neutral (0.00) |
| -0.20 to -0.10 | Negative | Mildly Bearish (-0.40) |
| < -0.20 | Strongly Negative | Bearish (-0.70) |

### 3.3 Options Flow (12%)

| P/C Ratio | Interpretation | Signal |
|-----------|----------------|--------|
| < 0.60 | Heavy call buying | Strong Bullish (+0.80) |
| 0.60-0.80 | More calls | Mildly Bullish (+0.50) |
| 0.80-1.10 | Balanced | Neutral (0.00) |
| 1.10-1.30 | More puts | Mildly Bearish (-0.50) |
| > 1.30 | Heavy put buying | Strong Bearish (-0.80) |

### 3.4 Market Sentiment (12%)

**VIX:**
| Level | Interpretation | Signal |
|-------|----------------|--------|
| < 15 | Low fear | Bullish (+0.30) |
| 15-20 | Normal | Neutral (0.00) |
| 20-25 | Elevated | Cautious (-0.15) |
| 25-30 | High fear | Bearish (-0.30) |
| > 30 | Extreme fear | Strong Bearish* |

**Fear & Greed Index:**
| Value | Interpretation | Signal |
|-------|----------------|--------|
| 0-25 | Extreme Fear | Contrarian Bullish (+0.40) |
| 25-40 | Fear | Mildly Bullish (+0.20) |
| 40-60 | Neutral | Neutral (0.00) |
| 60-75 | Greed | Bullish (+0.15) |
| 75-100 | Extreme Greed | Contrarian Bearish (-0.40) |

### 3.5 Analyst Ratings (13%)

| Upside % | Signal |
|----------|--------|
| > +20% | Bullish (+0.70) |
| +10% to +20% | Mildly Bullish (+0.40) |
| -10% to +10% | Neutral (0.00) |
| < -10% | Bearish (-0.50) |

### 3.6 Social Media (8%)

| Bullish % | Signal |
|-----------|--------|
| > 65% | Bullish (+0.50) |
| 55-65% | Mildly Bullish (+0.25) |
| 45-55% | Neutral (0.00) |
| 35-45% | Mildly Bearish (-0.25) |
| < 35% | Bearish (-0.50) |

### 3.7 Historical News Impact (10%)

Analyzes how similar news events affected the stock in the past.

| Pattern | 1-Day | 5-Day | Confidence |
|---------|-------|-------|------------|
| Earnings Beat | +3.2% | +4.1% | 65% |
| Earnings Miss | -4.5% | -5.2% | 70% |
| Analyst Upgrade | +2.1% | +3.5% | 55% |
| Analyst Downgrade | -2.8% | -3.9% | 60% |
| Product Launch | +1.5% | +2.8% | 45% |

### 3.8 Historical Patterns (10%)

| Pattern | Conditions | Expected Move | Win Rate |
|---------|------------|---------------|----------|
| Oversold Bounce | RSI < 30, Price > SMA200 | +5.5% | 62% |
| Golden Cross | SMA50 > SMA200 | +8.0% | 65% |
| Death Cross | SMA50 < SMA200 | -7.5% | 60% |
| Overbought Reversal | RSI > 70, MACD bearish | -4.5% | 58% |

---

## 4. Probability Calculation

### Formula

```
weighted_signal = Σ (signal_i × weight_i) / Σ weight_i

base_probability = 50 + (weighted_signal × 35)
```

This maps:
- signal = -1.0 → probability = 15%
- signal = 0.0 → probability = 50%
- signal = +1.0 → probability = 85%

### Price Gap Adjustments

| Gap % | Multiplier |
|-------|------------|
| < 10% | 1.00 |
| 10-20% | 0.90 |
| 20-30% | 0.80 |
| 30-40% | 0.70 |
| > 50% | 0.50 |

### Final Clamping

```
final_probability = CLAMP(adjusted_probability, 15%, 85%)
```

We never claim < 15% (impossible) or > 85% (certain).

---

## 5. Confidence Levels

### Calculation

```
confidence_score = (data_quality × 0.40) +
                   (signal_agreement × 0.40) +
                   (source_count_bonus × 0.20)
```

### Levels

| Score Range | Level | Interpretation |
|-------------|-------|----------------|
| 0.85 - 1.00 | HIGH | Strong data, signals agree |
| 0.70 - 0.85 | MEDIUM-HIGH | Good data, mostly agreement |
| 0.50 - 0.70 | MEDIUM | Adequate data, mixed signals |
| 0.35 - 0.50 | LOW | Limited data or conflict |
| 0.00 - 0.35 | VERY LOW | Poor data or strong conflict |

---

## 6. Decision Trail Visualization

Each prediction includes a full breakdown:

```
PREDICTION: 62% probability

├── TECHNICAL (20% weight) ─────────────────────── Score: +0.35
│   ├── RSI: 52.3 (Neutral) ──────────────────────── +0.00
│   ├── MACD: Bullish crossover ──────────────────── +0.50
│   ├── SMA: Price > SMA20 > SMA50 ───────────────── +0.70
│   └── Support: Near $125 support ───────────────── +0.20
│
├── NEWS (15% weight) ──────────────────────────── Score: +0.25
│   ├── 12 articles analyzed
│   ├── Sentiment: +0.32 (Positive)
│   └── Key: "AI chip demand remains strong"
│
├── OPTIONS (12% weight) ───────────────────────── Score: +0.30
│   ├── Put/Call Ratio: 0.72 (More calls)
│   └── Signal: Mildly Bullish
│
└── ANALYST (13% weight) ───────────────────────── Score: +0.55
    ├── Target: $165 (+18% upside)
    └── Consensus: Buy

WEIGHTED SIGNAL: +0.30
BASE PROBABILITY: 50 + (0.30 × 35) = 60%
FINAL: 60%
```

---

## 7. AI Model Selection

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Claude API Available? ──Yes──▶ Use Claude (Best Quality)   │
│           │                                                     │
│          No                                                     │
│           ▼                                                     │
│  2. Ollama Running? ────────Yes──▶ Use Ollama (Good Quality)   │
│           │                                                     │
│          No                                                     │
│           ▼                                                     │
│  3. Use Rule-Based Engine (Basic Quality)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Supported Instruments

| Type | Format | Examples |
|------|--------|----------|
| Stocks | TICKER | AAPL, NVDA, TSLA |
| Crypto | XXX-USD | BTC-USD, ETH-USD |
| Forex | XXXYYY=X | EURUSD=X, GBPUSD=X |
| Commodities | XX=F | GC=F (gold), CL=F (oil) |
| Indices | ^XXXX | ^GSPC, ^DJI, ^IXIC |
| ETFs | TICKER | SPY, QQQ, GLD |

---

## 9. Output Interpretation

### Probability Ranges

| Range | Meaning | Guidance |
|-------|---------|----------|
| 75-85% | Highly Likely | Strong signals |
| 65-75% | Likely | Favorable setup |
| 55-65% | Slightly Likely | Modest edge |
| 45-55% | Uncertain | No clear edge |
| 35-45% | Slightly Unlikely | Modest headwind |
| 25-35% | Unlikely | Unfavorable |
| 15-25% | Highly Unlikely | Strong headwinds |

**Important:** Even 80% probability means 1 in 5 times it fails.

---

## 10. Disclaimers

**THIS IS NOT FINANCIAL ADVICE**

- Past performance does NOT guarantee future results
- All investments carry risk of loss
- Markets can move against ANY prediction
- Never invest money you cannot afford to lose
- Always do your own research (DYOR)
- Consult a licensed financial advisor

**LIMITATIONS:**
- Cannot account for black swan events
- Cannot predict insider information
- Cannot forecast market manipulation
- Cannot react to breaking news after analysis

---

*Document Version: 2.0*
*Last Updated: January 2026*
