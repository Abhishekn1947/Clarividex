<p align="center">
  <img src="clarividex-gold.png" alt="Clarividex Logo" width="120" height="120">
</p>

<h1 align="center">Clarividex</h1>

<p align="center">
  <strong>The Clairvoyant Index - AI-Powered Market Predictions</strong>
</p>

<p align="center">
  <em>See tomorrow's markets today with transparent, data-driven predictions</em>
</p>

<p align="center">
  <a href="https://dy9y276gfap4k.cloudfront.net"><strong>Live Demo</strong></a> •
  <a href="#why-clarividex">Why Clarividex</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#prediction-engine">Prediction Engine</a> •
  <a href="#demo">Demo</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#documentation">Docs</a>
</p>

<p align="center">
  <a href="https://github.com/Abhishekn1947/Clarividex/stargazers"><img src="https://img.shields.io/github/stars/Abhishekn1947/Clarividex?style=flat-square&color=gold" alt="Stars"></a>
  <a href="https://github.com/Abhishekn1947/Clarividex/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Abhishekn1947/Clarividex/ci.yml?style=flat-square&label=CI" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Next.js-16-black?style=flat-square&logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Gemini_API-Google-4285F4?style=flat-square&logo=google" alt="Gemini">
  <img src="https://img.shields.io/badge/Markets-US%20%2B%20India-orange?style=flat-square" alt="Markets">
  <img src="https://img.shields.io/badge/AWS-Lambda%20%2B%20CloudFront-FF9900?style=flat-square&logo=amazonaws" alt="AWS">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"></a>
</p>

---

## Why Clarividex?

### The Problem with Traditional Market Prediction Tools

| Traditional Tools | Clarividex |
|-------------------|------------|
| "Buy" / "Sell" / "Hold" ratings | **Quantified probability scores** (e.g., 73% chance) |
| Black-box algorithms | **Full transparency** - see every factor |
| Single data source | **12+ data sources** aggregated |
| No reasoning provided | **Decision trail** showing how predictions are made |
| Binary yes/no answers | **Confidence levels** with uncertainty quantified |

### What Sets Us Apart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CLARIVIDEX vs COMPETITORS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  1. TRANSPARENT REASONING                                             ║  │
│  ║     Every prediction shows exactly which factors contributed          ║  │
│  ║     and how much weight each signal carries.                          ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  2. PROBABILISTIC OUTPUT                                              ║  │
│  ║     No vague "bullish" or "bearish" - you get a specific              ║  │
│  ║     probability percentage with confidence bounds (15-85%).           ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  3. MULTI-MODEL ARCHITECTURE                                          ║  │
│  ║     Monte Carlo simulations + Bayesian analysis + Technical           ║  │
│  ║     indicators + Sentiment analysis = Comprehensive predictions.      ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  4. HERD SENTIMENT WARNINGS                                           ║  │
│  ║     Contrarian indicator alerts when retail sentiment reaches         ║  │
│  ║     extremes (extreme fear = bullish signal, extreme greed = caution) ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  5. SCENARIO ANALYSIS                                                 ║  │
│  ║     "What if" analysis for interest rate changes, earnings,           ║  │
│  ║     market crashes, and sector rotations.                             ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Disclaimer**: This is for educational and informational purposes only. Not financial advice.

---

## Supported Markets

Clarividex supports **two major markets** with a single toggle in the UI:

| Feature | US Market | India Market (NSE/BSE) |
|---------|-----------|----------------------|
| **Tickers** | AAPL, NVDA, TSLA, 160+ stocks | RELIANCE.NS, TCS.NS, INFY.NS, 40+ stocks |
| **Currency** | USD ($) | INR (₹) |
| **News Locale** | US English, CNBC, MarketWatch | India English, Economic Times |
| **Sentiment** | r/wallstreetbets, r/stocks | r/IndianStreetBets, r/IndiaInvestments |
| **Volatility** | VIX (^VIX) | India VIX (^INDIAVIX) |
| **Indices** | S&P 500, Dow Jones, NASDAQ | Nifty 50, Sensex, Bank Nifty |
| **Filings** | SEC EDGAR, Finviz | *(skipped — not available)* |
| **Fear & Greed** | CNN Fear & Greed Index | *(skipped — US-only)* |

**Cross-market detection**: Query an Indian stock while on the US tab? Clarividex suggests switching automatically.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLARIVIDEX PREDICTION FLOW                          │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────────────────────────────────────────────────────────┐
     │  USER QUERY: "Will NVDA reach $150 by March 2026?"               │
     │              "Will Reliance reach ₹3000 by June 2026?"           │
     └────────────────────────────────┬─────────────────────────────────┘
                                      │
                                      ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │                     QUERY VALIDATION                              │
     │  • Validate financial query format                               │
     │  • Extract ticker symbol (NVDA)                                  │
     │  • Parse target price ($150) and date (March 2026)               │
     └────────────────────────────────┬─────────────────────────────────┘
                                      │
                                      ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │                  PARALLEL DATA AGGREGATION                        │
     │                                                                   │
     │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
     │   │ Yahoo   │ │  News   │ │ Social  │ │ Options │ │   SEC   │   │
     │   │ Finance │ │   RSS   │ │  Media  │ │  Flow   │ │  EDGAR  │   │
     │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
     │        │           │           │           │           │         │
     │        └───────────┴───────────┴───────────┴───────────┘         │
     │                              │                                    │
     │                    ┌─────────▼─────────┐                         │
     │                    │  250+ Data Points │                         │
     │                    │    Aggregated     │                         │
     │                    └───────────────────┘                         │
     └────────────────────────────────┬─────────────────────────────────┘
                                      │
                                      ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │               ENHANCED PROBABILITY ENGINE V2                      │
     │                                                                   │
     │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
     │   │  Monte Carlo  │  │   Bayesian    │  │Multi-Timeframe│        │
     │   │  Simulation   │  │   Analysis    │  │   Analysis    │        │
     │   │  (2000 paths) │  │  (14 signals) │  │ (Daily/Wk/Mo) │        │
     │   └───────┬───────┘  └───────┬───────┘  └───────┬───────┘        │
     │           │                  │                  │                 │
     │           └──────────────────┼──────────────────┘                 │
     │                              │                                    │
     │   ┌───────────────┐  ┌───────▼───────┐  ┌───────────────┐        │
     │   │  Volatility   │  │   Weighted    │  │   Analyst     │        │
     │   │   Analysis    │  │   Consensus   │  │   Targets     │        │
     │   └───────────────┘  └───────────────┘  └───────────────┘        │
     └────────────────────────────────┬─────────────────────────────────┘
                                      │
                                      ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │                     ADJUSTMENT FACTORS                            │
     │                                                                   │
     │  • Sector Relative Strength      • Mean Reversion                │
     │  • Insider Trading Activity      • Short Interest                │
     │  • Earnings Proximity            • Support/Resistance Levels     │
     │  • Market Regime Detection       • VIX Volatility Regime         │
     └────────────────────────────────┬─────────────────────────────────┘
                                      │
                                      ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │                     AI REASONING (Gemini)                          │
     │                                                                   │
     │  Synthesizes all data into natural language explanation          │
     │  with bullish/bearish factors and actionable insights            │
     └────────────────────────────────┬─────────────────────────────────┘
                                      │
                                      ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │                      FINAL OUTPUT                                 │
     │  ┌──────────────────────────────────────────────────────────┐    │
     │  │  Probability: 62%          Confidence: Medium-High       │    │
     │  │  Sentiment: Bullish        Data Points: 287              │    │
     │  │                                                          │    │
     │  │  Bullish Factors:                                        │    │
     │  │    ✓ RSI at 45 (neutral, room to rise)                  │    │
     │  │    ✓ Analyst target $180 (+22% upside)                  │    │
     │  │    ✓ News sentiment positive (+0.35)                    │    │
     │  │                                                          │    │
     │  │  Bearish Factors:                                        │    │
     │  │    ✗ VIX elevated (22)                                  │    │
     │  │    ✗ Short interest 3.2%                                │    │
     │  └──────────────────────────────────────────────────────────┘    │
     └──────────────────────────────────────────────────────────────────┘
```

---

## Prediction Engine

### Core Algorithms

Clarividex uses a **multi-model ensemble approach** with 6 core probability components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROBABILITY ENGINE COMPONENTS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Component             │ Weight │ Description                               │
│  ──────────────────────┼────────┼─────────────────────────────────────────  │
│  Monte Carlo Sim       │  25%   │ 2,000 price path simulations with         │
│                        │        │ fat-tailed distributions                  │
│  ──────────────────────┼────────┼─────────────────────────────────────────  │
│  Bayesian Integration  │  25%   │ 14 signals with reliability scores        │
│                        │        │ (RSI, MACD, news, options, etc.)          │
│  ──────────────────────┼────────┼─────────────────────────────────────────  │
│  Volatility Analysis   │  20%   │ Historical volatility with Student-t      │
│                        │        │ fat tail adjustments                      │
│  ──────────────────────┼────────┼─────────────────────────────────────────  │
│  Multi-Timeframe       │  15%   │ Daily, Weekly, Monthly trend alignment    │
│  ──────────────────────┼────────┼─────────────────────────────────────────  │
│  Analyst Targets       │  10%   │ Consensus price targets vs current        │
│  ──────────────────────┼────────┼─────────────────────────────────────────  │
│  Options Implied       │  5%    │ Put/Call ratio implied probability        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Monte Carlo Simulation

```
For each of 2,000 simulations:

  price = current_price

  For each day until target_date:

    1. Mean Reversion:     mean_rev = 0.02 × (ln(mean) - ln(price))
    2. Random Shock:       z = Normal() with fat-tail adjustment
    3. Daily Return:       return = μ + mean_rev + σ × z
    4. New Price:          price = price × exp(return)
    5. Check target reached → count success

  Probability = (successes / 2,000) × 100
```

### Bayesian Signal Integration

| Signal | Trigger | Reliability | Direction |
|--------|---------|-------------|-----------|
| RSI Oversold | RSI < 30 | 62% | Bullish |
| RSI Overbought | RSI > 70 | 58% | Bearish |
| Golden Cross | SMA50 > SMA200 | 68% | Bullish |
| Death Cross | SMA50 < SMA200 | 64% | Bearish |
| Bullish Options | P/C < 0.7 | 56% | Bullish |
| Extreme Fear | F&G < 25 | 60% | Contrarian Bullish |
| Insider Buying | Buy > 2×Sell | 62% | Bullish |

### 8-Factor Weighted Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FACTOR WEIGHTS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ████████████████████████████████████████     Technical Analysis    20%    │
│  ██████████████████████████████               News Sentiment        15%    │
│  ██████████████████████████                   Analyst Ratings       13%    │
│  ████████████████████████                     Options Flow          12%    │
│  ████████████████████████                     Market Sentiment      12%    │
│  ████████████████████                         Historical News       10%    │
│  ████████████████████                         Historical Patterns   10%    │
│  ████████████████                             Social Media           8%    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Confidence Bounds

We **never claim certainty**. All probabilities are bounded:
- Minimum: **15%** (nothing is impossible)
- Maximum: **85%** (nothing is certain)

---

## Data Sources

Clarividex aggregates data from **12+ sources** in real-time:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MARKET DATA                    │  SENTIMENT DATA                           │
│  ───────────────────────────────┼────────────────────────────────────────   │
│  • Yahoo Finance (quotes)       │  • Google News RSS (US & India locale)    │
│  • Finviz (US fundamentals)     │  • StockTwits                             │
│  • VIX / India VIX              │  • Reddit (WSB, IndianStreetBets)         │
│  • Options Flow                 │  • CNN Fear & Greed Index (US)            │
│                                 │                                           │
│  FUNDAMENTAL DATA               │  TECHNICAL DATA                           │
│  ───────────────────────────────┼────────────────────────────────────────   │
│  • SEC EDGAR (US filings)       │  • RSI, MACD, Bollinger Bands             │
│  • Analyst Ratings              │  • Moving Averages (SMA/EMA)              │
│  • Insider Trading              │  • Support/Resistance Levels              │
│  • Earnings Calendar            │  • Volume Analysis                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Live Application

> **Try it now**: [https://dy9y276gfap4k.cloudfront.net](https://dy9y276gfap4k.cloudfront.net)

Hosted on AWS — fully serverless architecture with S3 + CloudFront (frontend) and Lambda (backend). Runs at ~$0/month under free tier.

---

## Demo

### Screenshots

<p align="center">
  <img src="docs/screenshots/main page ss.png" alt="Main Page" width="800">
  <br>
  <em>Clean, professional interface for making predictions</em>
</p>

<p align="center">
  <img src="docs/screenshots/how it works.png" alt="How It Works" width="800">
  <br>
  <em>Step-by-step prediction process with full transparency</em>
</p>

<p align="center">
  <img src="docs/screenshots/Data sources.png" alt="Data Sources" width="800">
  <br>
  <em>12+ data sources aggregated for comprehensive analysis</em>
</p>

### Video Demo

https://github.com/Abhishekn1947/Clarividex/raw/main/docs/Demo%20Video.mov

> Click the link above to watch the demo, or [download the video](docs/Demo%20Video.mov) directly.

---

## Documentation

For in-depth technical documentation, see the `/docs` folder:

| Document | Description |
|----------|-------------|
| [PREDICTION_ENGINE.md](docs/PREDICTION_ENGINE.md) | Complete algorithm documentation with formulas |
| [METHODOLOGY.md](docs/METHODOLOGY.md) | How predictions are generated step-by-step |
| [TECHNICAL_INDICATORS.md](docs/TECHNICAL_INDICATORS.md) | RSI, MACD, Moving Averages explained |
| [ENHANCEMENTS.md](docs/ENHANCEMENTS.md) | V2 enhancements: RAG, guardrails, SSE, evals |

---

## Tech Stack

### Frontend
| Technology | Purpose |
|------------|---------|
| **Next.js 16** | React 19 framework with App Router |
| **TypeScript** | Type-safe development |
| **Tailwind CSS** | Utility-first responsive styling |
| **Recharts** | Data visualization |
| **Lucide React** | Icon library |
| **S3 + CloudFront** | Static hosting (exported SPA) |

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance Python API |
| **Gemini API** | AI reasoning engine (Google) |
| **Pandas/NumPy** | Data processing & calculations |
| **yfinance** | Market data retrieval |
| **AWS Lambda** | Serverless deployment via Docker |

### AI Models
```
Gemini 2.0 Flash (Primary) → Rule-Based Engine (Fallback)
Architecture supports hot-swapping to Claude Opus 4.6, GPT-4o, or any frontier model
```

### Infrastructure
```
Frontend:  S3 (private) → CloudFront (OAC, PriceClass_100)
Backend:   ECR → Lambda (Function URL, public HTTPS)
IaC:       Terraform (modules: ecr, lambda, frontend, secrets, monitoring, warmup)
CI/CD:     GitHub Actions (test on PR, deploy on push to main)
```

### AWS Serverless Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Browser                                                                    │
│    │                                                                        │
│    ├── Static assets ──→ CloudFront (CDN) ──→ S3 (private bucket, OAC)     │
│    │                     • PriceClass_100 (US/Canada/Europe)                │
│    │                     • SPA routing via CloudFront Function              │
│    │                     • Immutable caching for hashed assets              │
│    │                                                                        │
│    └── API calls ──────→ Lambda Function URL (direct HTTPS)                │
│                          • FastAPI in Docker (ECR image)                    │
│                          • Mangum adapter for Lambda compatibility          │
│                          • CORS handled by FastAPI middleware               │
│                          • EventBridge warmup every 5 minutes              │
│                                                                             │
│  Supporting Services:                                                       │
│    • SSM Parameter Store — API keys & secrets                              │
│    • CloudWatch — alarms & logging                                         │
│    • SNS — email alerts                                                    │
│                                                                             │
│  Cost: ~$0/month (free tier) │ ~$3-10/month (post free tier)               │
│  No EC2, no ALB, no RDS — fully serverless                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Managed entirely with **Terraform** (~27 resources across 6 modules) and deployed via **GitHub Actions** CI/CD pipelines.

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Quick Start (Local Development)

```bash
# 1. Clone the repository
git clone https://github.com/Abhishekn1947/Clarividex.git
cd Clarividex

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Install & start backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Install & start frontend (new terminal)
cd frontend
npm install
npm run dev

# 5. Open in browser
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### Docker

```bash
docker-compose up --build
```

### Production Deployment (AWS)

```bash
# 1. Configure terraform/terraform.tfvars with your credentials (see .env.example)

# 2. Deploy infrastructure
cd terraform
terraform init && terraform apply

# 3. Build & push backend Docker image (see .github/workflows/deploy.yml)

# 4. Deploy frontend to S3 + CloudFront
./scripts/deploy-frontend.sh
```

---

## API Documentation

### Interactive Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` | POST | Generate full prediction |
| `/api/v1/health` | GET | Health check with API status |
| `/api/v1/stock/{ticker}/quote` | GET | Real-time stock quote |
| `/api/v1/stock/{ticker}/technicals` | GET | Technical indicators |
| `/api/v1/stock/{ticker}/news` | GET | News articles with sentiment |
| `/api/v1/stock/{ticker}/social` | GET | Social media sentiment |
| `/api/v1/analyze-query` | POST | Query quality analysis |
| `/api/v1/validate-ticker` | GET | Ticker extraction & validation |
| `/api/v1/chat` | POST | AI chat about predictions |
| `/api/v1/popular-tickers?market=US` | GET | List of supported tickers (US or IN) |

---

## Architecture

```
clarividex/
├── frontend/                 # Next.js 16 React application (static export)
│   ├── src/
│   │   ├── app/             # Next.js App Router pages
│   │   ├── components/      # React components (responsive 320px+)
│   │   └── lib/             # API client (NEXT_PUBLIC_API_URL baked at build)
│   └── public/              # Static assets (favicons, manifest)
│
├── backend/                  # FastAPI Python backend
│   ├── app/
│   │   ├── api/             # API routes
│   │   ├── models/          # Pydantic schemas
│   │   ├── services/        # Business logic
│   │   │   ├── prediction_engine.py
│   │   │   ├── market_config.py       # Market-specific config (US/India)
│   │   │   ├── market_data.py
│   │   │   ├── technical_analysis.py
│   │   │   ├── news_service.py
│   │   │   └── social_service.py
│   │   ├── middleware/      # Rate limiting
│   │   ├── rag/             # RAG pipeline (ChromaDB)
│   │   └── evals/           # Evaluation suite
│   └── lambda_handler.py    # AWS Lambda entry point (Mangum)
│
├── terraform/                # Infrastructure as Code
│   ├── modules/
│   │   ├── frontend/        # S3 + CloudFront + OAC
│   │   ├── lambda/          # Lambda + Function URL + IAM
│   │   ├── ecr/             # Container registry
│   │   ├── secrets/         # SSM Parameter Store
│   │   ├── monitoring/      # CloudWatch alarms + SNS
│   │   └── warmup/          # EventBridge keep-warm
│   └── main.tf
│
├── .github/workflows/        # CI/CD pipelines
│   ├── ci.yml               # PR validation (lint + build + test)
│   ├── deploy.yml           # Backend deploy (ECR + Lambda)
│   └── deploy-frontend.yml  # Frontend deploy (S3 + CloudFront)
│
├── scripts/
│   └── deploy-frontend.sh   # Manual frontend deployment
│
├── docker/                   # Docker configs
│   └── Dockerfile.lambda    # Lambda container image
│
└── docs/                     # Technical documentation
```

---

## V2 Enhancements

Clarividex V2 introduces 8 major enhancements:

| Enhancement | Description |
|-------------|-------------|
| **India Market Support** | Full NSE/BSE support with ₹ currency, India VIX, Indian news/sentiment sources |
| **Prompt Versioning** | YAML-based prompt templates with A/B testing support |
| **RAG-Powered Chat** | Documentation-grounded answers via ChromaDB + HuggingFace embeddings |
| **Output Guardrails** | PII redaction, financial advice detection, probability clamping (15-85%) |
| **SSE Streaming** | Real-time prediction progress via Server-Sent Events |
| **Singleton Caching** | Eliminated redundant API client instantiation |
| **Evaluation Suite** | 18-case golden dataset with automated metrics and experiment tracking |
| **Mobile Responsiveness** | Full responsive design across all components (320px - 1024px+) |

For detailed documentation, see [docs/ENHANCEMENTS.md](docs/ENHANCEMENTS.md).

---

## Roadmap & Future Work

### Prediction Engine V3 (In Development)

We're actively working on the next generation of the prediction engine with significant improvements:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREDICTION ENGINE V3 - COMING SOON                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENHANCED HYPERPARAMETERS                                                   │
│  • Optimized Monte Carlo simulation parameters                              │
│  • Fine-tuned Bayesian signal reliability scores                            │
│  • Improved volatility regime detection thresholds                          │
│                                                                             │
│  EXPANDED HISTORICAL DATA                                                   │
│  • 10+ years of historical price data for backtesting                       │
│  • Enhanced pattern recognition with larger training sets                   │
│  • Improved mean reversion and momentum calculations                        │
│                                                                             │
│  MODEL FINE-TUNING                                                          │
│  • Sector-specific model calibration                                        │
│  • Market regime-aware parameter adjustment                                 │
│  • Continuous learning from prediction outcomes                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Call for Contributors

We're inviting developers to contribute to the prediction engine! Areas of focus:

- **Algorithm Optimization** - Improve probability calculations and reduce latency
- **Data Source Integration** - Add new financial data APIs
- **Backtesting Framework** - Enhance historical accuracy validation
- **Model Calibration** - Fine-tune hyperparameters for different market conditions

**Before contributing, please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and code standards.**

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

### Data Providers
Yahoo Finance • SEC EDGAR • Finviz • Google News • StockTwits • Reddit • NSE/BSE (via yfinance)

---

## Author

**Abhishek Nandakumar**

Full Stack Developer | AI/ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/abhishek-nandakumar/)

---

<p align="center">
  <strong>Clarividex</strong> - The Clairvoyant Index
  <br>
  <em>See tomorrow's markets today</em>
</p>
