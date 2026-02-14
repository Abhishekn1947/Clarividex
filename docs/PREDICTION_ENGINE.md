# Clarividex Prediction Engine

> How our prediction engine actually works — explained simply, with full technical details underneath.

---

## Table of Contents

1. [The Big Picture (ELI5)](#1-the-big-picture-eli5)
2. [Complete Flow Diagram](#2-complete-flow-diagram)
3. [Step-by-Step Walkthrough](#3-step-by-step-walkthrough)
4. [Data Collection — Where We Get Our Info](#4-data-collection--where-we-get-our-info)
5. [The Algorithms — Our 8 Crystal Balls](#5-the-algorithms--our-8-crystal-balls)
   - [5.1 Volatility Probability (The "How Wild Is This Stock?" Check)](#51-volatility-probability)
   - [5.2 Monte Carlo Simulation (The "Roll the Dice 2000 Times" Method)](#52-monte-carlo-simulation)
   - [5.3 Multi-Timeframe Trend Analysis (The "Zoom In, Zoom Out" Check)](#53-multi-timeframe-trend-analysis)
   - [5.4 Bayesian Analysis (The "Update Your Beliefs" Engine)](#54-bayesian-analysis)
   - [5.5 Analyst Target Probability (The "What Do the Pros Think?" Check)](#55-analyst-target-probability)
   - [5.6 Options Implied Probability (The "Follow the Smart Money" Signal)](#56-options-implied-probability)
   - [5.7 Earnings Surprise Probability (The "Report Card History" Check)](#57-earnings-surprise-probability)
   - [5.8 Options Flow Probability (The "Betting Volume" Signal)](#58-options-flow-probability)
6. [How The Algorithms Combine — The Ensemble](#6-how-the-algorithms-combine--the-ensemble)
7. [Adjustment Factors — Reality Checks](#7-adjustment-factors--reality-checks)
8. [Sentiment Analysis — Reading the Room](#8-sentiment-analysis--reading-the-room)
9. [Technical Analysis — Chart Reading](#9-technical-analysis--chart-reading)
10. [Pattern Recognition — "I've Seen This Before"](#10-pattern-recognition--ive-seen-this-before)
11. [Confidence Score — How Sure Are We?](#11-confidence-score--how-sure-are-we)
12. [Safety Rails — Keeping It Honest](#12-safety-rails--keeping-it-honest)
13. [Full Calculation Example](#13-full-calculation-example)
14. [Technical Reference](#14-technical-reference)

---

## 1. The Big Picture (ELI5)

Imagine you want to know: **"Will Tesla hit $300 by June?"**

Here's what Clarividex does in plain English:

1. **Gathers data from 12+ sources** — stock prices, news articles, social media buzz, insider trades, analyst opinions, options markets, economic indicators, and more. Think of it like a detective collecting every clue available.

2. **Runs 8 different prediction algorithms** — each one looks at the question from a different angle. One simulates 2,000 possible futures. Another reads the news to gauge mood. Another checks if chart patterns match historical setups. They're like 8 experts sitting at a table, each with a different specialty.

3. **Combines their answers** — the algorithms vote, but not equally. If one algorithm has better data to work with (e.g., we have a full year of price history), its vote counts more. If another algorithm has no data (e.g., no options data available), it sits out entirely.

4. **Applies reality checks** — is there an earnings report coming up? Is the stock near a resistance ceiling? Are insiders selling? These nudge the probability up or down by a few percentage points.

5. **Calibrates and bounds the answer** — we deliberately pull all predictions toward 50% (because nobody can predict the market with extreme certainty), and we never output anything below 15% or above 85%.

The result: a probability percentage with a confidence level, a list of supporting factors, and a full decision trail showing *exactly* how we got there.

---

## 2. Complete Flow Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        CLARIVIDEX PREDICTION PIPELINE                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  USER QUERY: "Will NVDA reach $150 by March 2026?"
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │  STEP 1: QUERY VALIDATION & PARSING                 │
  │                                                     │
  │  • Is this a financial question? (3-tier guidance)   │
  │  • Extract ticker → "NVDA"                          │
  │  • Extract target price → $150                      │
  │  • Extract timeframe → March 2026                   │
  │  • Direction → Bearish (price needs to DROP)         │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STEP 2: DATA COLLECTION (all happen at once)       │
  │                                                     │
  │  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐  │
  │  │ Stock Price  │ │  1-Year      │ │  Company    │  │
  │  │ Quote        │ │  History     │ │  Info       │  │
  │  └──────────────┘ └──────────────┘ └─────────────┘  │
  │  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐  │
  │  │ News from    │ │  Social      │ │  Insider    │  │
  │  │ 3+ Sources   │ │  Sentiment   │ │  Trades     │  │
  │  └──────────────┘ └──────────────┘ └─────────────┘  │
  │  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐  │
  │  │ Analyst      │ │  Options     │ │  VIX &      │  │
  │  │ Ratings      │ │  Chain       │ │  Fear/Greed │  │
  │  └──────────────┘ └──────────────┘ └─────────────┘  │
  │  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐  │
  │  │ SEC Filings  │ │  Sector      │ │  Economic   │  │
  │  │ (EDGAR)      │ │  Performance │ │  Indicators │  │
  │  └──────────────┘ └──────────────┘ └─────────────┘  │
  └──────────────────────┬──────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  ┌──────────────┐ ┌──────────┐ ┌─────────────────┐
  │  Technical   │ │Sentiment │ │  Pattern        │
  │  Analysis    │ │ Analysis │ │  Recognition    │
  │  RSI, MACD,  │ │ VADER +  │ │  11 patterns +  │
  │  SMA, Bollin.│ │ TextBlob │ │  historical     │
  │  ATR, Volume │ │ +Lexicon │ │  similarity     │
  └──────┬───────┘ └────┬─────┘ └────────┬────────┘
         │              │                │
         └──────────────┼────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────────────┐
  │  STEP 3: AI ANALYSIS (3-tier fallback)              │
  │                                                     │
  │  ① Claude API ──fail──▶ ② Ollama ──fail──▶ ③ Rules │
  │  (best: sees      (local model,     (8-factor       │
  │   everything,      same prompt)      weighted        │
  │   reasons over                       scoring)        │
  │   all data)                                          │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STEP 4: ENHANCED PROBABILITY ENGINE V2             │
  │  (8 algorithms running simultaneously)              │
  │                                                     │
  │  ┌─────────────────┐  ┌─────────────────┐           │
  │  │ ① Volatility    │  │ ② Monte Carlo   │           │
  │  │ "How wild is    │  │ "Simulate 2000  │           │
  │  │  this stock?"   │  │  possible       │           │
  │  │  Weight: 15-20% │  │  futures"       │           │
  │  └─────────────────┘  │  Weight: 15-25% │           │
  │                       └─────────────────┘           │
  │  ┌─────────────────┐  ┌─────────────────┐           │
  │  │ ③ Trend         │  │ ④ Bayesian      │           │
  │  │ "Are all time-  │  │ "Update beliefs │           │
  │  │  frames         │  │  with each new  │           │
  │  │  agreeing?"     │  │  clue"          │           │
  │  │  Weight: 15-25% │  │  Weight: ~22%   │           │
  │  └─────────────────┘  └─────────────────┘           │
  │  ┌─────────────────┐  ┌─────────────────┐           │
  │  │ ⑤ Analyst       │  │ ⑥ Options       │           │
  │  │ "What do Wall   │  │ "What does the  │           │
  │  │  Street pros    │  │  options market  │           │
  │  │  think?"        │  │  imply?"        │           │
  │  │  Weight: ~10%   │  │  Weight: ~10%   │           │
  │  └─────────────────┘  └─────────────────┘           │
  │  ┌─────────────────┐  ┌─────────────────┐           │
  │  │ ⑦ Earnings      │  │ ⑧ Options Flow  │           │
  │  │ "Does this co.  │  │ "Are traders    │           │
  │  │  usually beat   │  │  making big     │           │
  │  │  expectations?" │  │  directional    │           │
  │  │  Weight: ~8%    │  │  bets?"         │           │
  │  └─────────────────┘  │  Weight: ~10%   │           │
  │                       └─────────────────┘           │
  │                                                     │
  │  Weighted Average ──────────────▶ Base Probability  │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STEP 5: ADJUSTMENT FACTORS (±percentage points)    │
  │                                                     │
  │  • Sector strength vs market    → up to ±5%         │
  │  • Insider buying/selling       → up to ±4%         │
  │  • Earnings report coming?      → -2% to -3%        │
  │  • Mean reversion pressure      → up to ±3%         │
  │  • Near support/resistance?     → up to ±5%         │
  │  • Short squeeze potential?     → up to +2%          │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STEP 6: CALIBRATION & SAFETY RAILS                 │
  │                                                     │
  │  1. Calibrate: pull toward 50%                       │
  │     calibrated = 50 + (raw - 50) × 0.70             │
  │                                                     │
  │  2. Reality check: cap unrealistic daily moves       │
  │     Need >3%/day? → cap at 20%                       │
  │     Need >2%/day? → cap at 35%                       │
  │                                                     │
  │  3. Hard bounds: clamp to 15% — 85%                  │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  STEP 7: BUILD RESPONSE                             │
  │                                                     │
  │  ✦ Probability: 36%                                  │
  │  ✦ Confidence: Medium                                │
  │  ✦ Key Factors (bullish & bearish)                   │
  │  ✦ Catalysts & Risks                                 │
  │  ✦ Decision Trail (full audit of every signal)       │
  │  ✦ Data Sources & Limitations                        │
  └─────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Walkthrough

Here's what happens from the moment you type a question:

### Step 1: Understanding Your Question
**Files:** `routes.py`, `market_data.py`

Your question gets parsed to extract:
- **What stock?** — We use a multi-strategy ticker extractor that checks company names (~130 mapped), `$TICKER` prefixes, known tickers (~170), and regex patterns. It filters out common English words ("WILL", "THE") that look like tickers but aren't.
- **What price target?** — Extracted from the query text.
- **By when?** — Parsed into a date to calculate trading days remaining.
- **Up or down?** — If the target price is above the current price, it's a bullish question. Below = bearish.

### Step 2: Collecting All Available Data
**File:** `data_aggregator.py`

We fire off **12+ data requests simultaneously** (not one after another — all at once for speed). Each source is wrapped in error handling so if one fails, the rest still work. Think of it like sending 12 assistants to different libraries at the same time.

### Step 3: AI Analysis
**File:** `prediction_engine.py`

We try three approaches in order:
1. **Claude AI** — gets ALL the data as context and reasons over it like a human analyst would. This is the most nuanced analysis.
2. **Ollama (local AI)** — same prompt, but runs on your machine. Used if Claude API is down or unavailable.
3. **Rule-based scoring** — a deterministic 8-factor weighted formula. Always available as a guaranteed fallback. No AI needed.

### Step 4: Probability Calculation
**File:** `enhanced_probability_engine.py`

This is the mathematical core. Eight different algorithms each calculate their own probability estimate, then they get combined into one number. (Detailed in Section 5 below.)

### Steps 5-7: Adjust, Calibrate, Deliver
The raw probability gets nudged by real-world factors, pulled toward 50% for humility, bounded to prevent crazy numbers, and packaged with full transparency into the response.

---

## 4. Data Collection — Where We Get Our Info

Think of each data source as a different "expert witness" we consult:

| Source | What It Gives Us | Service File |
|--------|------------------|--------------|
| **Yahoo Finance (yfinance)** | Live stock price, 1-year price history, company info, insider trades, analyst recommendations, earnings dates, options chains | `market_data.py` |
| **Google News RSS** | Latest news headlines + article text for sentiment analysis | `news_service.py` |
| **Finnhub API** | Additional financial news coverage | `news_service.py` |
| **MarketWatch/Yahoo/CNBC RSS** | Even more news from major financial outlets | `additional_data_sources.py` |
| **StockTwits** | What retail traders are saying (bull/bear sentiment ratios) | `social_service.py` |
| **Reddit** | Social media discussion volume and sentiment | `social_service.py` |
| **Google Trends** | Is search interest in this stock rising or falling? | `social_service.py` |
| **SEC EDGAR** | Official company filings (10-K, 10-Q, 8-K) | `additional_data_sources.py` |
| **Finviz** | 30+ financial metrics (P/E ratio, short float, target price, etc.) | `additional_data_sources.py` |
| **CNN/alternative.me** | Fear & Greed Index — is the overall market fearful or greedy? | `additional_data_sources.py` |
| **VIX (^VIX)** | Market volatility index — how scared/calm is the market? | `additional_data_sources.py` |
| **Sector ETFs** | How the stock's sector is performing relative to the S&P 500 | `additional_data_sources.py` |
| **Treasury/FRED** | Economic indicators like interest rates | `additional_data_sources.py` |

**Data Quality Score:** Each source gets checked for availability. More sources available = higher data quality score = more confidence in the prediction. The quality score is a weighted sum — core data (prices, history) matters more than nice-to-haves (Google Trends, SEC filings).

---

## 5. The Algorithms — Our 8 Crystal Balls

Each algorithm answers the same question from a different angle. Here's what each one does, explained simply first, then technically.

---

### 5.1 Volatility Probability
**The "How Wild Is This Stock?" Check**

**Simple explanation:**
If a stock normally moves 1% per day, asking "will it move 50% in a week?" gets a very low probability. This algorithm looks at how much the stock *usually* moves and asks: "Given this stock's normal behavior, how likely is this target price?"

**The twist:** Real stock markets have "fat tails" — extreme moves happen more often than a simple bell curve would predict. Think of it like weather: a "once in a century" storm actually happens way more than once per century. We account for this by using a **Student-t distribution** instead of a normal bell curve, which has fatter tails (gives more probability to extreme moves).

**How it works:**
1. Calculate daily log returns from 1 year of price history
2. Measure the **excess kurtosis** (a fancy word for "how fat are the tails?")
3. Use kurtosis to set degrees of freedom: `df = max(3, 6/kurtosis + 4)` — higher kurtosis = fatter tails = lower df
4. Scale daily volatility to the target timeframe: `scaled_vol = daily_vol × √(trading_days)`
5. Calculate z-score: `z = required_return / scaled_vol` (how many standard deviations away is the target?)
6. Look up probability using the Student-t distribution's CDF
7. Adjust by volatility regime (quiet market = slightly reduce, volatile market = slightly boost)

**Base weight: 15-20%**

---

### 5.2 Monte Carlo Simulation
**The "Roll the Dice 2,000 Times" Method**

**Simple explanation:**
Imagine you could rewind time and replay the stock market 2,000 times with slightly different random events each time. Some replays the stock goes up, some it goes down, some it crashes, some it moons. Count how many of those 2,000 alternate realities hit the target price. That fraction is your probability.

**What makes ours special:** We don't just use random coin flips. Each simulated day includes:
- **Mean reversion** — stocks that go way up tend to pull back, and vice versa. We model this with an Ornstein-Uhlenbeck process (fancy name for "rubber band effect") with strength θ = 0.1.
- **Regime-aware volatility** — if the market is currently chaotic, we increase the randomness in our simulations. If calm, we decrease it.
- **Fat-tailed shocks** — we add extra randomness to large moves to simulate those "Black Swan" surprise days.

**How it works:**
```
For each of 2,000 simulations:
    Start at today's price

    For each trading day until target date:
        1. Calculate mean-reversion pull:
           pull = 0.1 × (average_price - current_price) / current_price

        2. Generate random shock with fat tails:
           shock = random_normal × daily_volatility
           if shock is extreme: amplify it by 1.1x

        3. Calculate the day's return:
           return = historical_avg_return + pull + shock

        4. New price = old price × e^(return)

        5. Did we hit the target at any point? Mark it.

    Probability = (simulations that hit target) / 2,000
```

**Base weight: 15-25%** (gets more weight when volatility is high, because simulation captures extreme scenarios better)

---

### 5.3 Multi-Timeframe Trend Analysis
**The "Zoom In, Zoom Out" Check**

**Simple explanation:**
Imagine looking at a stock chart zoomed into the last month, then zooming out to 6 weeks, then to 3 months. If ALL three views show the stock going up, that's a strong trend. If they disagree (going up short-term but down long-term), that's a confusing signal and reduces confidence.

**Why it matters:** A stock might be up today but in a long-term downtrend. Or it might be down this week but in a strong monthly uptrend. Looking at multiple timeframes helps us separate noise from real trends. Backtesting showed this is one of the most predictive signals — when all timeframes agree, the trend is much more likely to continue.

**How it works:**

| Timeframe | What We Measure | Weight |
|-----------|----------------|--------|
| **Daily** (20 days) | Slope of the 20-day simple moving average | 30% |
| **Weekly** (25 days) | Current price vs. price 25 days ago | 35% |
| **Monthly** (60 days) | Current price vs. price 60 days ago | 35% |

1. For each timeframe, determine: is the trend **up** or **down**?
2. Calculate **alignment score**: what percentage of weight agrees with the target direction?
   - All 3 agree → alignment = 100% → strong signal
   - 2 of 3 agree → alignment = 65% → moderate signal
   - Only 1 agrees → alignment = 30% → weak/conflicting signal
3. Convert to probability: `base_prob = 50 + (alignment - 0.5) × 40`

**Base weight: 15-25%** (boosted in V2 because backtesting proved its value)

---

### 5.4 Bayesian Analysis
**The "Update Your Beliefs With Each New Clue" Engine**

**Simple explanation:**
Start with a baseline belief: the stock market goes up about 53% of the time (historical S&P 500 average). Now, one by one, feed in each clue we've gathered:
- RSI says "oversold" → nudge probability up a bit
- MACD says "bearish crossover" → nudge it back down
- Insiders are buying → nudge up again
- Fear & Greed shows extreme fear → nudge up (contrarian signal — extreme fear often means the bottom is near)

Each clue adjusts the probability proportionally to how reliable that signal has historically been.

**Why this approach:** Unlike just averaging signals, Bayesian updating is mathematically optimal for combining uncertain evidence. Each piece of evidence properly accounts for its own reliability, and the order you process them doesn't matter — you get the same answer.

**How it works:**

Start with prior: `P(bullish) = 53%`

For each signal, we have a **likelihood ratio** (how much to adjust):

| Signal | What Triggers It | Likelihood Ratio | Direction |
|--------|-----------------|-------------------|-----------|
| RSI Oversold | RSI < 30 | 1.4× | Bullish |
| RSI Overbought | RSI > 70 | 1.3× | Bearish |
| MACD Bullish Cross | MACD line above signal line | 1.25× | Bullish |
| MACD Bearish Cross | MACD line below signal line | 1.2× | Bearish |
| Golden Cross | 50-day SMA crosses above 200-day SMA | 1.5× | Bullish |
| Death Cross | 50-day SMA crosses below 200-day SMA | 1.4× | Bearish |
| Above SMA 200 | Price above 200-day average | 1.3× | Bullish |
| Bullish News | Sentiment score > 0.2 | up to 1.35× | Bullish |
| Low Put/Call Ratio | P/C < 0.7 | 1.2× | Bullish |
| High Put/Call Ratio | P/C > 1.3 | 1.15× | Bearish |
| Extreme Fear | Fear & Greed < 25 | 1.25× | Contrarian Bullish |
| Extreme Greed | Fear & Greed > 75 | 1.2× | Contrarian Bearish |

**The update formula (applied for each signal):**
```
odds = probability / (1 - probability)
new_odds = odds × likelihood_ratio
new_probability = new_odds / (1 + new_odds)
```

Clamped to 10%-90% after each update to prevent any single signal from dominating.

**Base weight: ~22%**

---

### 5.5 Analyst Target Probability
**The "What Do the Pros Think?" Check**

**Simple explanation:**
Wall Street analysts publish price targets. If you're asking "will AAPL hit $200?" and the average analyst target is $220, that's a good sign — the pros think it's going higher than your target. If analysts say $180 and you're asking about $200, that's a headwind.

**How it works:**
- Compare your target price against the consensus analyst target from Finviz/Yahoo Finance
- If your target is *below* the analyst target (for bullish predictions): boost probability — you're asking for less than what the pros expect
- If your target *exceeds* the analyst target: reduce probability proportionally to the gap
- The further your target is from the analyst target, the bigger the adjustment

**Weight: ~10%** (only when analyst data is available)

---

### 5.6 Options Implied Probability
**The "Follow the Smart Money" Signal**

**Simple explanation:**
Options traders literally bet money on where a stock will be by a certain date. The prices of options bake in an "implied volatility" — basically the market's consensus estimate of how much the stock will move. We use this to calculate how likely the market *itself* thinks your target price is.

**How it works:**
1. Get implied volatility (IV) from the options chain
2. Scale IV to the target timeframe: `scaled_IV = IV × √(years_to_target)`
3. Calculate z-score: `z = required_return / scaled_IV`
4. Probability from the normal distribution CDF

**Why it's valuable:** Options IV reflects the collective wisdom (and money) of thousands of sophisticated traders. It's the market's own forecast.

**Weight: ~10%** (only when options data is available)

---

### 5.7 Earnings Surprise Probability
**The "Report Card History" Check**

**Simple explanation:**
Companies report earnings every quarter. Some companies (like most big tech) beat expectations most of the time — they sandbag their guidance. If a company has beaten earnings 80% of the time, and earnings are coming up within your prediction window, that history matters.

**How it works:**
- Look up the company's historical earnings beat rate
- Fall back to sector averages if company-specific data isn't available:

| Sector | Average Beat Rate |
|--------|------------------|
| Technology | 72% |
| Healthcare | 65% |
| Finance | 68% |
| Energy | 58% |
| Consumer | 63% |
| Industrial | 60% |
| Default | 62% |

- Adjust probability based on beat rate and how close earnings are to the prediction window

**Weight: ~8%** (only when earnings data is available)

---

### 5.8 Options Flow Probability
**The "Betting Volume" Signal**

**Simple explanation:**
Beyond just implied volatility, we look at the actual *volume* of calls vs. puts being traded. If way more people are buying calls (betting on up) than puts (betting on down), that's a bullish signal. We also look at open interest ratios and detect unusual activity.

**How it works:**
- Calculate put/call ratio from both volume and open interest
- Put/call < 0.7 = bullish (lots more call buying)
- Put/call > 1.3 = bearish (lots more put buying)
- Detect unusual options activity (spikes in volume relative to open interest)

**Weight: ~10%** (only when options data is available)

---

## 6. How The Algorithms Combine — The Ensemble

This is the key insight: **no single algorithm is reliable enough on its own.** By combining multiple independent approaches, errors in one tend to get cancelled out by the others.

### The Three Layers of Ensembling

```
LAYER 1: Within Each Data Source
════════════════════════════════
Sentiment = VADER (50%) + TextBlob (20%) + Custom Lexicon (30%)
Patterns  = Pattern signals (60%) + Historical similarity (40%)
Technicals = RSI + MACD + SMA + Bollinger (weighted by signal strength)

         ▼

LAYER 2: The 8-Algorithm Probability Engine
════════════════════════════════════════════
Each algorithm outputs a probability (0-100%)
Combined via DYNAMIC weighted average:

  Final = Σ(algorithm_probability × algorithm_weight) / Σ(weights)

Weights adapt based on:
  • Data availability (no data = weight 0)
  • Market regime (volatile → more Monte Carlo weight)
  • Ticker-specific tuning (TSLA → more Monte Carlo, AAPL → more Bayesian)

         ▼

LAYER 3: AI + Statistical Integration
══════════════════════════════════════
The Claude AI analysis and the statistical probability
engine cross-validate each other:
  • AI provides qualitative reasoning + its own probability estimate
  • Statistical engine provides mathematically rigorous probability
  • Enhanced probability engine's output is the final number
```

### Dynamic Weight Example

For a stock like **TSLA** (very volatile) in a **high-volatility market**:
```
Volatility:       15%  × 1.0  = base weight
Monte Carlo:      25%  × 1.3  = boosted (volatile stock + volatile market)
Trend:            15%  × 1.0  = base weight
Bayesian:         22%  × 1.0  = base weight
Analyst:          10%  × 1.0  = (if available)
Options:          10%  × 1.0  = (if available)
Earnings:          8%  × 1.0  = (if available)
Options Flow:     10%  × 1.0  = (if available)
                  ───
                  Weights are normalized to sum to 100%
```

For a stock like **GOOGL** (strong trending behavior):
```
Multi-Timeframe weight gets × 1.3 boost (trends are more predictive for GOOGL)
```

---

## 7. Adjustment Factors — Reality Checks

After the 8 algorithms produce a base probability, we apply real-world adjustments. Each one adds or subtracts a few percentage points:

### 7.1 Sector Relative Strength (±5%)
**Plain English:** Is this stock's sector outperforming or underperforming the overall market? A tech stock during a tech rally gets a bullish boost. A tech stock during a tech selloff gets a bearish nudge.

```
stock_return = how much THIS stock moved in 20 days
sector_return = how much the SECTOR moved in 20 days
relative_strength = stock_return - sector_return

If outperforming and bullish target: up to +5%
If underperforming and bullish target: up to -5%
```

### 7.2 Insider Trading Signal (±4%)
**Plain English:** Are the people who run the company buying or selling their own stock? Insiders know things we don't. If they're buying heavily, that's bullish. If they're dumping shares, watch out.

```
If insiders buying > 2× selling: +4% (bullish) or -2% (bearish)
If insiders selling > 2× buying: -3% (bullish) or +2% (bearish)
```

### 7.3 Earnings Proximity (-2% to -3%)
**Plain English:** Is an earnings report coming up within your prediction timeframe? Earnings are wildly unpredictable events — stocks can swing 10%+ in either direction. This always *reduces* confidence (adds uncertainty), never boosts it.

```
Earnings within prediction window: -2%
Earnings within 7 days: -3%
```

### 7.4 Mean Reversion (±3%)
**Plain English:** Stocks that have gone up a LOT recently tend to pull back, and stocks that have crashed tend to bounce. If a stock is 15%+ above its 60-day average and you're asking if it'll go even higher, we nudge the probability down.

```
If stock is >15% above average and bullish target: -3%
If stock is >15% below average and bearish target: -3%
If stock is >10% above average and bearish target: +2% (mean reversion supports this)
If stock is >10% below average and bullish target: +2% (bounce supports this)
```

### 7.5 Support/Resistance (±5%)
**Plain English:** Stocks tend to bounce off certain price levels (support below, resistance above) like a ball bouncing off a floor and ceiling. If your target price requires breaking through a strong resistance ceiling, that's harder. Calculated from the 25th percentile of lows (support) and 75th percentile of highs (resistance) over 20 days.

```
If bullish target is above resistance: penalty proportional to the gap (up to -5%)
If bearish target is below support: penalty proportional to the gap (up to -5%)
```

### 7.6 Short Interest (+2%)
**Plain English:** If many traders have bet against the stock (shorted it), there's "squeeze" potential — if the stock starts rising, shorts are forced to buy to cover their positions, which pushes the stock up even more. High short interest paradoxically makes a bullish outcome slightly more likely.

```
If >20% of shares are shorted: +2% for bullish, +1% for bearish
If >10% of shares are shorted: +1% for bullish, +0.5% for bearish
```

---

## 8. Sentiment Analysis — Reading the Room

**File:** `sentiment_service.py`

We analyze the *emotional tone* of news articles and text using three methods combined:

### The Triple-Method Ensemble

```
┌─────────────────────────────────────────────────┐
│         SENTIMENT ANALYSIS PIPELINE              │
│                                                  │
│  Input: News article text                        │
│                                                  │
│  ┌──────────────────┐                            │
│  │ ① VADER          │ ──── 50% weight            │
│  │ (Financial)      │                            │
│  │ Standard VADER + │                            │
│  │ 200+ finance     │                            │
│  │ terms added      │                            │
│  │ (e.g., "bullish" │                            │
│  │  = +2.0)         │                            │
│  └──────────────────┘                            │
│                                                  │
│  ┌──────────────────┐                            │
│  │ ② TextBlob       │ ──── 20% weight            │
│  │ (General NLP)    │                            │
│  │ Standard polarity│                            │
│  │ detection        │                            │
│  └──────────────────┘                            │
│                                                  │
│  ┌──────────────────┐                            │
│  │ ③ Custom Lexicon │ ──── 30% weight            │
│  │ (Hand-tuned)     │                            │
│  │ ~100 bullish     │                            │
│  │ ~120 bearish     │                            │
│  │ ~60 multi-word   │                            │
│  │ phrases checked  │                            │
│  │ first, 12        │                            │
│  │ intensity mods   │                            │
│  │ ("significantly" │                            │
│  │  = 1.5× boost)   │                            │
│  └──────────────────┘                            │
│                                                  │
│  Final Score = weighted average → [-1, +1]       │
│  -1 = very bearish, 0 = neutral, +1 = very bull  │
└─────────────────────────────────────────────────┘
```

**Why three methods?** VADER is great for social media but misses financial jargon (that's why we add 200+ finance terms). TextBlob catches general tone. The custom lexicon handles phrases specific to markets ("dead cat bounce", "short squeeze", "guidance raised"). By combining all three, we get a more robust sentiment reading than any single method.

---

## 9. Technical Analysis — Chart Reading

**File:** `technical_analysis.py`

Technical analysis is the art of reading price chart patterns to predict future movement. Here are the indicators we calculate:

### RSI (Relative Strength Index)
**Plain English:** Measures if a stock has been going up too much (overbought) or down too much (oversold) recently. Like checking if a rubber band is stretched too far.
- **Period:** 14 days
- **Below 30** = oversold → likely to bounce up (bullish signal, strength +1)
- **Above 70** = overbought → likely to pull back (bearish signal, strength -1)
- **30-45** = mildly oversold (+0.3)
- **55-70** = mildly overbought (-0.3)

### MACD (Moving Average Convergence/Divergence)
**Plain English:** Compares a fast moving average (12-day) with a slow one (26-day). When the fast one crosses above the slow one, momentum is turning bullish. Think of it like a speedometer — the MACD shows whether the stock is *accelerating* or *decelerating*.
- **Parameters:** Fast=12, Slow=26, Signal=9
- **MACD above Signal line** = bullish (+0.5 to +1.0)
- **MACD below Signal line** = bearish (-0.5 to -1.0)

### SMA (Simple Moving Average)
**Plain English:** The average price over a window of time. The 200-day SMA is the "big picture" trend. If the stock is above it, the long-term trend is up. If the 50-day crosses above the 200-day ("golden cross"), that's a famous bullish signal.
- **Periods:** 20, 50, 200 days
- **Price > SMA 200** = long-term uptrend (bullish +0.5)
- **SMA 50 > SMA 200 (Golden Cross)** = strong bullish (+1.0)
- **SMA 50 < SMA 200 (Death Cross)** = strong bearish (-1.0)

### Bollinger Bands
**Plain English:** Draws a band around the stock price — the band is 2 standard deviations wide around the 20-day average. When the price touches the bottom band, it's unusually low. When it touches the top band, it's unusually high.
- **Period:** 20, Width: 2 standard deviations
- **Price below lower band** = oversold → bullish signal
- **Price above upper band** = overbought → bearish signal
- **Weight in overall technical score:** 0.5× (less than RSI/MACD/SMA)

### ATR (Average True Range)
**Plain English:** Measures how much the stock price typically moves in a day. Used by the probability engine to gauge how volatile the stock is. A stock with ATR of $5 moves a lot more than one with ATR of $0.50.
- **Period:** 14 days
- Not a directional signal — purely a volatility measure used by other components

### Support & Resistance Levels
**Plain English:** Price "floors" and "ceilings" where the stock tends to bounce. Calculated as the 25th percentile of 20-day lows (support) and 75th percentile of 20-day highs (resistance).

### Overall Technical Signal
All indicators combine into a single score from **-1 (very bearish) to +1 (very bullish)**:
```
signal = (RSI × 1.0 + MACD × 1.0 + SMA_trend × 1.0 + Bollinger × 0.5) / 3.5
```

---

## 10. Pattern Recognition — "I've Seen This Before"

**File:** `pattern_recognition.py`

This module looks at the current technical setup and checks: "Have we seen this pattern before? What happened historically?"

### The 11 Recognized Patterns

| Pattern | What It Means (Simply) | Historical Win Rate | Signal Strength |
|---------|----------------------|--------------------:|:---------------:|
| **Oversold Bounce** | RSI < 30 — stock beaten down, likely to bounce | 62% | +0.6 |
| **Overbought Pullback** | RSI > 70 — stock ran too hot, likely to cool off | 58% | -0.6 |
| **Golden Cross** | 50-day avg crosses above 200-day avg — big bullish signal | 65% | +0.8 |
| **Death Cross** | 50-day avg crosses below 200-day avg — big bearish signal | 60% | -0.8 |
| **Bullish MACD Crossover** | MACD crosses above its signal line — momentum turning up | 57% | +0.5 |
| **Bearish MACD Crossover** | MACD crosses below its signal line — momentum turning down | 55% | -0.5 |
| **Support Bounce** | Price near support floor and holding — likely to bounce | 58% | +0.4 |
| **Resistance Rejection** | Price near resistance ceiling and failing — likely to fall back | 55% | -0.4 |
| **Bullish Trend Continuation** | Uptrend still intact — likely to keep going | 60% | +0.5 |
| **Bearish Trend Continuation** | Downtrend still intact — likely to keep falling | 58% | -0.5 |
| **Consolidation Breakout Pending** | Price squeezing into tight range — big move coming (direction unknown) | 50% | 0.0 |

### Historical Similarity Matching
Beyond fixed patterns, we also search for **similar past setups**:
1. Take the current RSI value and moving average positions
2. Compare against every historical window in the data
3. **Similarity score** = RSI similarity (50% weight) + MA position similarity (50% weight)
4. Windows with similarity > 0.7 are considered matches
5. Look at what happened *after* those similar historical setups

**Final pattern signal** = detected pattern signal (60%) + historical similarity signal (40%)

---

## 11. Confidence Score — How Sure Are We?

The confidence score (0-100%) tells you how much to trust the prediction. It's NOT the probability itself — it's how reliable we think our probability estimate is.

### Four Components of Confidence

```
┌─────────────────────────────────────────────────────────┐
│                 CONFIDENCE SCORE                         │
│                                                         │
│  ┌─────────────────────────┐                            │
│  │ Probability Engine      │ × 35%                      │
│  │ Confidence              │                            │
│  │ (Do the 8 algorithms    │                            │
│  │  agree with each other?)│                            │
│  └─────────────────────────┘                            │
│                                                         │
│  ┌─────────────────────────┐                            │
│  │ Data Quality Score      │ × 25%                      │
│  │ (How many of the 12+    │                            │
│  │  data sources actually  │                            │
│  │  returned data?)        │                            │
│  └─────────────────────────┘                            │
│                                                         │
│  ┌─────────────────────────┐                            │
│  │ Signal Agreement        │ × 25%                      │
│  │ (Do technicals,         │                            │
│  │  sentiment, patterns    │                            │
│  │  all point the          │                            │
│  │  same direction?)       │                            │
│  └─────────────────────────┘                            │
│                                                         │
│  ┌─────────────────────────┐                            │
│  │ Data Source Bonus        │ × 15%                     │
│  │ (Do we have premium     │                            │
│  │  data like options,     │                            │
│  │  insider trades,        │                            │
│  │  analyst targets?)      │                            │
│  └─────────────────────────┘                            │
│                                                         │
│  Confidence = sum of above → mapped to level:           │
│                                                         │
│  VERY_LOW → LOW → MODERATE → HIGH → VERY_HIGH → EXTREME│
└─────────────────────────────────────────────────────────┘
```

**Key insight:** High confidence means "our data is good and our models agree." Low confidence means "we're missing data or our models disagree — take this prediction with extra salt."

---

## 12. Safety Rails — Keeping It Honest

We deliberately build in humility. No prediction system can be certain about markets.

### 12.1 Calibration (Pull Toward 50%)
Every probability gets pulled toward 50% using:
```
calibrated = 50 + (raw_probability - 50) × 0.70 × confidence_adjustment
```
- A raw 80% becomes ~71%
- A raw 30% becomes ~36%
- This is the single most important step for avoiding overconfidence

### 12.2 Unrealistic Move Caps
If the target requires unrealistic daily moves:

| Required Daily Move | Max Probability |
|-------------------:|:--------------:|
| > 3% per day | 20% |
| > 2% per day | 35% |
| > 1% per day | 50% |

### 12.3 Long-Timeframe Uncertainty
For predictions > 1 year out, we pull further toward 50%:
```
probability = 50 + (probability - 50) × 0.7
```
Because predicting that far out is really just guessing.

### 12.4 Hard Bounds
No matter what the math says, the final output is always clamped to **15% — 85%**. We never say "this is basically certain" or "this is basically impossible."

### 12.5 Output Guardrails
The text response is checked by guardrails to ensure it includes proper disclaimers and doesn't make irresponsible claims.

---

## 13. Full Calculation Example

**Query:** "Will NVDA reach $150 by March 2026?"

- **Current Price:** $192.51
- **Target Price:** $150.00
- **Days to Target:** 56 trading days
- **Required Return:** -22.08% (this is a bearish bet — asking if the stock will DROP)

```
STEP 1: COMPONENT CALCULATIONS
═══════════════════════════════

① Volatility Probability:     95.0%
   → NVDA is volatile; a -22% move in 56 days is within
     its historical range (fat-tailed Student-t says "possible")

② Monte Carlo Simulation:      5.85%
   → Only 117 out of 2,000 simulated paths hit $150
     (most paths stayed above $150 due to upward drift)

③ Multi-Timeframe Trend:      44.0%
   → Daily and weekly trends are bullish (against bearish target)
     Monthly trend is mixed → partial alignment only

④ Bayesian Analysis:          43.01%
   → Started at 47% (bearish prior)
   → RSI not extreme: no update
   → MACD slightly bearish: nudged up to 48%
   → Above SMA200: nudged down to 43% (bullish signal hurts bearish case)
   → Moderately bullish news: nudged down to 43%

⑤ Analyst Target Prob:        30.0%
   → Analyst consensus target is ~$175
   → $150 is below most analyst estimates — but analysts are often wrong

⑥-⑧ Options/Earnings/Flow:    (limited data available)


STEP 2: WEIGHTED COMBINATION
═════════════════════════════

                    Weight    Prob     Contribution
volatility:          0.20  × 95.00  =  19.00
monte_carlo:         0.25  ×  5.85  =   1.46
multi_timeframe:     0.15  × 44.00  =   6.60
bayesian:            0.25  × 43.01  =  10.75
analyst_targets:     0.10  × 30.00  =   3.00
                                       ─────
BASE PROBABILITY                     = 40.81%


STEP 3: ADJUSTMENT FACTORS
══════════════════════════

Sector relative strength:   -1.61%  (tech sector outperforming → bearish target harder)
Insider activity:           +0.00%  (no significant insider signal)
Earnings proximity:         -2.00%  (earnings within window → uncertainty)
Mean reversion:             +0.00%  (stock near average → no mean-rev pressure)
Support/resistance:         -3.66%  (target below support → harder to reach)
                            ─────
TOTAL ADJUSTMENTS:          -7.27%

ADJUSTED PROBABILITY:       40.81% - 7.27% = 33.54%


STEP 4: CALIBRATION & BOUNDS
═════════════════════════════

Confidence score:           0.57 (MODERATE)
Confidence smoothing:       50 + (33.54 - 50) × 0.855 = 35.9%
Realistic move check:       -22% over 56 days = -0.39%/day → OK, under 1%
Hard bounds:                CLAMP(35.9%, 15%, 85%) = 35.9%

╔═══════════════════════════════════════╗
║  FINAL RESULT: 36% probability        ║
║  Confidence:   MODERATE               ║
║  Market Regime: BULL                  ║
║  Volatility:   NORMAL                 ║
╚═══════════════════════════════════════╝

Translation: "There's roughly a 1-in-3 chance NVDA drops to $150 by
March 2026. The current bullish trend, analyst targets, and support
levels all work against this bearish target."
```

---

## 14. Technical Reference

### Module Responsibilities

| Module | File | Role |
|--------|------|------|
| **Prediction Engine** | `prediction_engine.py` | Main orchestrator — coordinates the entire pipeline |
| **Enhanced Probability Engine** | `enhanced_probability_engine.py` | Core math — 8-component statistical ensemble |
| **Data Aggregator** | `data_aggregator.py` | Concurrent data fetching from 12+ sources |
| **Market Data Service** | `market_data.py` | Yahoo Finance wrapper, ticker extraction |
| **Technical Analysis** | `technical_analysis.py` | RSI, MACD, SMA, EMA, Bollinger, ATR |
| **Sentiment Service** | `sentiment_service.py` | Triple-method NLP sentiment scoring |
| **News Service** | `news_service.py` | News aggregation from Google/Finnhub |
| **Social Service** | `social_service.py` | StockTwits, Reddit, Google Trends |
| **Additional Data Sources** | `additional_data_sources.py` | SEC, VIX, Fear&Greed, Finviz, Options, Sectors |
| **Pattern Recognition** | `pattern_recognition.py` | 11 chart patterns + historical similarity |
| **Historical News Analyzer** | `historical_news_analyzer.py` | News-price correlation over time |
| **Decision Trail Builder** | `decision_trail_builder.py` | Full audit trail for transparency |
| **Stream Service** | `stream_service.py` | SSE streaming (9 progress stages) |
| **Schemas** | `schemas.py` | All data models and enums |

### Caching Strategy

| Cache | TTL | Purpose |
|-------|-----|---------|
| Stock Quotes | 1 min | Avoid hammering Yahoo Finance for real-time quotes |
| Company Info | 1 hour | Company metadata rarely changes |
| Price History | 5 min | Historical data is relatively stable |
| News Articles | 5 min | Avoid re-fetching the same headlines |
| Sentiment Scores | 10 min | NLP computation is expensive |
| Chat Responses | 5 min | Avoid re-calling Claude for identical chatbot questions |

### Key Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CALIBRATION_FACTOR` | 0.70 | How aggressively to pull predictions toward 50% |
| `MONTE_CARLO_SIMULATIONS` | 2,000 | Number of simulated price paths |
| `MEAN_REVERSION_SPEED` | 0.1 (θ) | Ornstein-Uhlenbeck mean reversion strength |
| `PROBABILITY_FLOOR` | 15% | Minimum displayed probability |
| `PROBABILITY_CEILING` | 85% | Maximum displayed probability |
| `BAYESIAN_CLAMP` | 10%-90% | Per-update clamp on Bayesian posterior |

---

*Document Version: 3.0*
*Algorithm Version: Enhanced Probability Engine V2*
*Last Updated: February 2026*
