# Clarividex Prediction Engine

> Technical documentation of the algorithms and mathematical models powering Clarividex predictions.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Enhanced Probability Engine V2](#2-enhanced-probability-engine-v2)
3. [Monte Carlo Simulation](#3-monte-carlo-simulation)
4. [Bayesian Signal Integration](#4-bayesian-signal-integration)
5. [Multi-Timeframe Analysis](#5-multi-timeframe-analysis)
6. [Volatility-Based Probability](#6-volatility-based-probability)
7. [Market Regime Detection](#7-market-regime-detection)
8. [Signal Reliability System](#8-signal-reliability-system)
9. [Adjustment Factors](#9-adjustment-factors)
10. [Confidence Calculation](#10-confidence-calculation)
11. [Final Probability Bounds](#11-final-probability-bounds)

---

## 1. Architecture Overview

The Clarividex prediction engine uses a **multi-model fallback architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION ENGINE FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │  User Query     │                                            │
│  │  + Market Data  │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              AI MODEL SELECTION                          │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────────────────┐  │   │
│  │  │ Claude  │ -> │ Ollama  │ -> │ Rule-Based Engine   │  │   │
│  │  │  API    │    │ Local   │    │ (Always Available)  │  │   │
│  │  └─────────┘    └─────────┘    └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         ENHANCED PROBABILITY ENGINE V2                   │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ Monte Carlo  │  │  Bayesian    │  │ Multi-       │   │   │
│  │  │ Simulation   │  │  Analysis    │  │ Timeframe    │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ Volatility   │  │  Analyst     │  │   Options    │   │   │
│  │  │ Analysis     │  │  Targets     │  │   Implied    │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ADJUSTMENT FACTORS                          │   │
│  │  • Sector Relative Strength    • Insider Activity       │   │
│  │  • Earnings Proximity          • Mean Reversion         │   │
│  │  • Support/Resistance          • Short Interest         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │     FINAL PROBABILITY (15% - 85% bounds)                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Enhanced Probability Engine V2

The core algorithm integrates **15+ factors** for comprehensive probability calculation:

### 2.1 Core Components (Weighted)

| Component | Description | Base Weight |
|-----------|-------------|-------------|
| `volatility` | Historical volatility-based probability | 20% |
| `monte_carlo` | Monte Carlo simulation with fat tails | 25% |
| `multi_timeframe` | Daily/Weekly/Monthly trend alignment | 15% |
| `bayesian` | Bayesian signal integration | 25% |
| `analyst_targets` | Analyst price target analysis | 10% |
| `options_implied` | Options-implied probability | 10% |

### 2.2 Dynamic Weight Adjustment

Weights are adjusted based on data availability:

```python
if historical_data >= 200 days:
    volatility_weight = 0.20
    monte_carlo_weight = 0.25
elif historical_data >= 60 days:
    volatility_weight = 0.15
    monte_carlo_weight = 0.20
else:
    volatility_weight = 0.10
    monte_carlo_weight = 0.15
```

---

## 3. Monte Carlo Simulation

### 3.1 Algorithm

The Monte Carlo simulation runs **2,000 price paths** with:
- Fat-tailed distribution (Student-t approximation)
- Mean reversion component (Ornstein-Uhlenbeck)
- Signal-based drift adjustment

```
┌─────────────────────────────────────────────────────────────────┐
│              MONTE CARLO SIMULATION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each of 2,000 simulations:                                 │
│                                                                 │
│    price = current_price                                        │
│                                                                 │
│    For each day until target_date:                              │
│                                                                 │
│      1. Mean Reversion Component:                               │
│         mean_rev = 0.02 × (ln(mean_price) - ln(price))          │
│                                                                 │
│      2. Random Shock (with fat tails):                          │
│         z = Box_Muller_Normal()                                 │
│         if |z| > 2: z = z × 1.1  (fat tail adjustment)          │
│                                                                 │
│      3. Daily Return:                                           │
│         return = μ_adjusted + mean_rev + σ × z                  │
│                                                                 │
│      4. New Price:                                              │
│         price = price × exp(return)                             │
│                                                                 │
│      5. Check if target reached → count success                 │
│                                                                 │
│  Probability = (successes / 2,000) × 100                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Parameters

| Parameter | Formula | Description |
|-----------|---------|-------------|
| `μ` (drift) | Mean of historical log returns | Expected daily return |
| `σ` (volatility) | Std dev of historical log returns | Daily volatility |
| `μ_adjusted` | `μ + (signal_strength × 0.002)` | Signal-modified drift |
| `mean_reversion_speed` | 0.02 | ~2% daily pull toward mean |

### 3.3 Volatility Regime Adjustments

```python
if volatility_regime == HIGH:
    σ *= 1.3
elif volatility_regime == EXTREME:
    σ *= 1.6
elif volatility_regime == LOW:
    σ *= 0.8
```

---

## 4. Bayesian Signal Integration

### 4.1 Prior Probability

```python
prior = 0.53 if is_bullish else 0.47  # Market base rate
```

### 4.2 Evidence Sources

| Signal Type | Trigger | Reliability | Direction |
|-------------|---------|-------------|-----------|
| RSI Oversold | RSI < 30 | 62% | Bullish |
| RSI Overbought | RSI > 70 | 58% | Bearish |
| MACD Bullish | MACD > Signal | 56% | Bullish |
| MACD Bearish | MACD < Signal | 54% | Bearish |
| Golden Cross | SMA50 > SMA200 | 68% | Bullish |
| Death Cross | SMA50 < SMA200 | 64% | Bearish |
| Above SMA200 | Price > SMA200 | 58% | Bullish |
| Below SMA200 | Price < SMA200 | 55% | Bearish |
| Bullish News | Sentiment > 0.2 | 54% | Bullish |
| Bearish News | Sentiment < -0.2 | 56% | Bearish |
| Bullish Options | P/C < 0.7 | 56% | Bullish |
| Bearish Options | P/C > 1.3 | 58% | Bearish |
| Extreme Fear | F&G < 25 | 60% | Contrarian Bullish |
| Extreme Greed | F&G > 75 | 55% | Contrarian Bearish |

### 4.3 Bayesian Update Formula

For each piece of evidence:

```
┌─────────────────────────────────────────────────────────────────┐
│              BAYESIAN UPDATE                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  effective_reliability = 0.5 + (reliability - 0.5) × |direction│
│                                                                 │
│  If direction > 0 (supports hypothesis):                        │
│      likelihood_ratio = effective_rel / (1 - effective_rel)    │
│                                                                 │
│  If direction < 0 (against hypothesis):                         │
│      likelihood_ratio = (1 - effective_rel) / effective_rel    │
│                                                                 │
│  odds = posterior / (1 - posterior)                             │
│  new_odds = odds × likelihood_ratio                             │
│  posterior = new_odds / (1 + new_odds)                          │
│                                                                 │
│  Clamp: posterior = CLAMP(posterior, 0.1, 0.9)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Multi-Timeframe Analysis

Analyzes trend alignment across three timeframes:

```
┌─────────────────────────────────────────────────────────────────┐
│           MULTI-TIMEFRAME TREND ANALYSIS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Timeframe     │  Calculation              │  Weight            │
│  ──────────────┼───────────────────────────┼─────────────       │
│  Daily (20d)   │  Price vs SMA20           │  30%               │
│  Weekly (25d)  │  Current vs 4-week ago    │  35%               │
│  Monthly (60d) │  Current vs 60-day ago    │  35%               │
│                                                                 │
│  Alignment Score:                                               │
│  ─────────────────                                              │
│  aligned_weight = Σ(weight) for trends matching direction       │
│  alignment_score = aligned_weight / total_weight                │
│                                                                 │
│  Probability:                                                   │
│  ─────────────                                                  │
│  base_prob = 50 + (alignment_score - 0.5) × 40                  │
│  result = CLAMP(base_prob, 30, 70)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Volatility-Based Probability

### 6.1 Historical Volatility Calculation

```python
# Calculate daily returns
returns = [(price[i] - price[i-1]) / price[i-1] for i in range(1, len(prices))]

# Statistics
mean_return = sum(returns) / len(returns)
variance = sum((r - mean_return)² for r in returns) / len(returns)
daily_volatility = sqrt(variance)

# Scale to target period
period_volatility = daily_volatility × sqrt(days_to_target)
period_drift = mean_return × days_to_target
```

### 6.2 Fat Tail Adjustment (Excess Kurtosis)

```python
# Calculate excess kurtosis
fourth_moment = sum((r - mean)⁴ for r in returns) / len(returns)
kurtosis = fourth_moment / variance² - 3

# Degrees of freedom for Student-t
df = max(4, 30 / (1 + max(0, kurtosis)))

# Z-score
z_score = (required_return - period_drift) / period_volatility

# Probability with fat tail adjustment
t_adjustment = 1 + (kurtosis × 0.05) if |z_score| > 1.5 else 1
probability = Student_t_CDF(-z_score, df) × 100 × t_adjustment
```

### 6.3 Student-t CDF Approximation

```python
def student_t_cdf(x, df):
    # Normal CDF approximation
    normal_cdf = 0.5 × (1 + erf(x / sqrt(2)))

    # Fat tail adjustment for small df
    if |x| > 2 and df < 30:
        tail_adjustment = 0.02 × (30 - df) / 30
        if x < 0:
            normal_cdf = max(0.01, normal_cdf - tail_adjustment)
        else:
            normal_cdf = min(0.99, normal_cdf + tail_adjustment)

    return normal_cdf
```

---

## 7. Market Regime Detection

### 7.1 Volatility Regime (VIX-Based)

```
┌─────────────────────────────────────────────────────────────────┐
│              VOLATILITY REGIME CLASSIFICATION                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  VIX Level    │  Regime       │  Signal Reliability Multiplier │
│  ─────────────┼───────────────┼─────────────────────────────    │
│  < 15         │  LOW          │  1.05 (signals more reliable)  │
│  15 - 25      │  NORMAL       │  1.00                          │
│  25 - 35      │  HIGH         │  0.90 (signals less reliable)  │
│  > 35         │  EXTREME      │  0.75 (much less reliable)     │
│                                                                 │
│  If VIX unavailable: Calculate realized volatility as proxy    │
│  realized_vol = sqrt(sum(returns²) / n) × sqrt(252) × 100      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Market Regime (Trend-Based)

```python
def detect_market_regime(prices):
    sma_50 = average(prices[-50:])
    sma_200 = average(prices[-200:]) if len(prices) >= 200 else sma_50
    current = prices[-1]

    if current > sma_50 > sma_200:
        return BULL
    elif current < sma_50 < sma_200:
        return BEAR
    else:
        return SIDEWAYS
```

---

## 8. Signal Reliability System

### 8.1 Base Reliability Scores

Based on historical backtesting and academic research:

```python
BASE_SIGNAL_RELIABILITY = {
    # Technical signals
    "rsi_oversold": 0.62,
    "rsi_overbought": 0.58,
    "macd_bullish_cross": 0.56,
    "macd_bearish_cross": 0.54,
    "golden_cross": 0.68,
    "death_cross": 0.64,
    "price_above_sma200": 0.58,
    "price_below_sma200": 0.55,

    # Fundamental signals
    "analyst_upgrade": 0.57,
    "analyst_downgrade": 0.59,
    "insider_buying": 0.62,
    "insider_selling": 0.54,

    # Sentiment signals
    "bullish_news": 0.54,
    "bearish_news": 0.56,
    "fear_greed_extreme_fear": 0.60,  # Contrarian
    "fear_greed_extreme_greed": 0.55,  # Contrarian
    "bullish_options_flow": 0.56,
    "bearish_options_flow": 0.58,
}
```

### 8.2 Dynamic Adjustment

```python
def get_adjusted_reliability(volatility_regime):
    adjustment = VOLATILITY_ADJUSTMENTS[volatility_regime]
    return {
        k: min(0.75, v × adjustment)  # Cap at 75%
        for k, v in BASE_SIGNAL_RELIABILITY.items()
    }
```

---

## 9. Adjustment Factors

### 9.1 Sector Relative Strength

```python
stock_return_20d = (current - price_20d_ago) / price_20d_ago
relative_strength = stock_return_20d - sector_return_20d

if is_bullish and relative_strength > 0.02:
    adjustment = min(5, relative_strength × 50)
elif is_bullish and relative_strength < -0.02:
    adjustment = max(-5, relative_strength × 50)
```

### 9.2 Insider Trading Signal

```python
if insider_buys > insider_sells × 2:
    adjustment = +4 if is_bullish else -2
elif insider_sells > insider_buys × 2:
    adjustment = -3 if is_bullish else +2
```

### 9.3 Earnings Proximity

```python
if 0 < days_to_earnings < days_to_target:
    adjustment = -2  # Earnings within window adds uncertainty
elif days_to_earnings <= 7:
    adjustment = -3  # Earnings very soon = high uncertainty
```

### 9.4 Mean Reversion

```python
mean_price_60d = average(prices[-60:])
deviation = (current_price - mean_price) / mean_price

if deviation > 0.15 and is_bullish:
    adjustment = -3  # Already extended up, bullish target harder
elif deviation < -0.15 and not is_bullish:
    adjustment = -3  # Already extended down, bearish target harder
elif deviation > 0.10 and not is_bullish:
    adjustment = +2  # Mean reversion supports bearish
elif deviation < -0.10 and is_bullish:
    adjustment = +2  # Mean reversion supports bullish
```

### 9.5 Support/Resistance Analysis

```python
if target_price > resistance and is_bullish:
    resistance_gap = (target_price - resistance) / resistance
    adjustment = -min(5, resistance_gap × 20)

if target_price < support and not is_bullish:
    support_gap = (support - target_price) / support
    adjustment = -min(5, support_gap × 20)
```

### 9.6 Short Interest

```python
if short_float > 20%:
    adjustment = +2 if is_bullish else +1  # Squeeze potential
elif short_float > 10%:
    adjustment = +1 if is_bullish else +0.5
```

---

## 10. Confidence Calculation

### 10.1 Formula

```python
confidence = 0.4  # Base confidence

# Data availability bonus
if len(historical_prices) >= 252:
    confidence += 0.15
elif len(historical_prices) >= 120:
    confidence += 0.10
elif len(historical_prices) >= 60:
    confidence += 0.05

# Component agreement bonus
probs = list(components.values())
mean_prob = sum(probs) / len(probs)
variance = sum((p - mean_prob)² for p in probs) / len(probs)
agreement = max(0, 1 - sqrt(variance) / 20)
confidence += agreement × 0.15

# Factors used bonus
confidence += min(0.10, factors_used × 0.015)

# Volatility regime penalty
if volatility_regime == HIGH:
    confidence *= 0.9
elif volatility_regime == EXTREME:
    confidence *= 0.75

confidence = min(0.95, confidence)
```

### 10.2 Confidence-Based Smoothing

```python
if confidence < 0.5:
    # Low confidence pulls probability toward 50%
    adjusted_prob = 50 + (adjusted_prob - 50) × (confidence × 1.5)
```

---

## 11. Final Probability Bounds

### 11.1 Realistic Move Constraints

```python
daily_move_required = abs_return / max(1, days)

if daily_move_required > 0.03:  # >3% per day required
    probability = min(probability, 20)
elif daily_move_required > 0.02:  # >2% per day
    probability = min(probability, 35)
elif daily_move_required > 0.01:  # >1% per day
    probability = min(probability, 50)
```

### 11.2 Long Timeframe Uncertainty

```python
if days > 365:
    # Pull toward 50% for very long predictions
    probability = 50 + (probability - 50) × 0.7
```

### 11.3 Final Clamping

```python
final_probability = CLAMP(probability, 5, 95)
# Display to user: CLAMP(probability, 15, 85)
```

---

## Example Calculation

**Query:** "Will NVDA reach $150 by March 2026?"
- Current Price: $192.51
- Target Price: $150.00
- Days to Target: 56
- Required Return: -22.08% (bearish target)

```
Step 1: Component Calculations
──────────────────────────────
volatility_prob     = 95.0%   (based on historical vol)
monte_carlo_prob    = 5.85%   (simulation results)
multi_timeframe     = 44.0%   (trend alignment)
bayesian_prob       = 43.01%  (signal integration)
analyst_targets     = 30.0%   (target vs analyst consensus)

Step 2: Dynamic Weights (251 data points available)
───────────────────────────────────────────────────
volatility:      0.20 × 95.0  = 19.00
monte_carlo:     0.25 × 5.85  = 1.46
multi_timeframe: 0.15 × 44.0  = 6.60
bayesian:        0.25 × 43.01 = 10.75
analyst_targets: 0.10 × 30.0  = 3.00
                              ──────
base_probability              = 40.81%

Step 3: Adjustments
───────────────────
sector_strength:    -1.61%
insider_activity:   +0.00%
earnings_proximity: -2.00%
mean_reversion:     +0.00%
support_resistance: -3.66%
                    ──────
total_adjustment:   -7.27%

Step 4: Final Calculation
─────────────────────────
adjusted_prob = 40.81 - 7.27 = 33.54%

Confidence = 0.57 (medium)
Confidence smoothing: 50 + (33.54 - 50) × 0.855 = 35.9%

Final bounded: CLAMP(35.9, 15, 85) = 35.9%

RESULT: 36% probability
Market Regime: BULL
Volatility Regime: NORMAL
```

---

## Data Quality Score

```python
def calculate_data_quality(historical_prices, factors_used):
    quality = 0.5  # Base

    if len(historical_prices) >= 252:
        quality += 0.25
    elif len(historical_prices) >= 120:
        quality += 0.15
    elif len(historical_prices) >= 60:
        quality += 0.10

    quality += min(0.25, factors_used × 0.04)

    return min(1.0, quality)
```

---

*Document Version: 2.0*
*Algorithm Version: Enhanced Probability Engine V2*
*Last Updated: January 2026*
