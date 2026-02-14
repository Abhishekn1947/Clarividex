# Clarividex - Complete Documentation

## Overview

Clarividex (The Clairvoyant Index) is an AI-powered financial prediction platform that generates probability-based stock forecasts with transparent reasoning. Unlike traditional "buy/sell" recommendations, this system provides:

- **Probability scores (0-100%)** for specific prediction queries
- **Confidence levels** (Low, Medium, High, Very High)
- **Full reasoning chains** with bullish and bearish factors
- **Historical accuracy tracking** to validate predictions over time

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  FRONTEND (Next.js 14 — Responsive)               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │   Search    │ │  Prediction │ │   Loading   │ │    How It  │ │
│  │    Form     │ │    Result   │ │  Skeleton   │ │    Works   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                │ API Calls
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI + Python 3.12)            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     API Routes (/api/v1)                 │   │
│  │  /predict  /health  /stock/{ticker}/*  /history  /accuracy  │
│  └─────────────────────────────────────────────────────────┘   │
│                                │                                 │
│  ┌─────────────────────────────┴─────────────────────────────┐ │
│  │                    PREDICTION ENGINE                       │ │
│  │  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐  │ │
│  │  │    Data      │  │  AI Analysis  │  │   Prediction   │  │ │
│  │  │  Aggregator  │  │   (Claude/    │  │    History     │  │ │
│  │  │              │  │    Ollama)    │  │    (SQLite)    │  │ │
│  │  └──────────────┘  └───────────────┘  └────────────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                │                                 │
│  ┌─────────────────────────────┴─────────────────────────────┐ │
│  │                      DATA SERVICES                         │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────┐  │ │
│  │  │ Market │ │ News   │ │ Social │ │ Tech   │ │Sentiment│  │ │
│  │  │ Data   │ │Service │ │Service │ │Analysis│ │ Service │  │ │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └─────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  FREE APIs    │      │  PREMIUM APIs │      │   AI Models   │
│               │      │   (Optional)  │      │               │
│ - Yahoo       │      │ - Finnhub     │      │ - Claude API  │
│   Finance     │      │ - Alpha       │      │   (Primary)   │
│ - Google News │      │   Vantage     │      │ - Ollama      │
│ - SEC EDGAR   │      │               │      │   (Fallback)  │
│ - StockTwits  │      │               │      │               │
│ - Reddit      │      │               │      │               │
│ - Finviz      │      │               │      │               │
│ - VIX         │      │               │      │               │
│ - Fear&Greed  │      │               │      │               │
└───────────────┘      └───────────────┘      └───────────────┘
```

---

## Tech Stack

### Frontend
- **Framework:** Next.js 14.2.5
- **Language:** TypeScript 5.5.4
- **Styling:** TailwindCSS 3.4.10 (mobile-first responsive)
- **Charts:** Recharts 2.12.7
- **HTTP Client:** Axios 1.7.4
- **Responsive:** Full mobile support from 320px (iPhone SE) to 1024px+

### Backend
- **Framework:** FastAPI 0.115.0
- **Server:** Uvicorn 0.30.6
- **Language:** Python 3.12
- **Database:** SQLite (local), PostgreSQL (production)
- **Cache:** Redis 7-alpine

### AI Models
- **Primary:** Claude API (claude-sonnet-4-20250514)
- **Fallback:** Ollama (llama3.1, mistral, deepseek-r1)
- **Last Resort:** Rule-based engine

---

## Data Sources (12+ APIs)

### Market Data
| Source | Data Type | Update Frequency |
|--------|-----------|------------------|
| **Yahoo Finance (yfinance)** | Real-time quotes, historical prices, company info | Real-time |
| **Finviz** | Stock screener data, target prices, analyst ratings | Daily |
| **VIX Index** | Market volatility/fear indicator | Real-time |

### News & Sentiment
| Source | Data Type | Update Frequency |
|--------|-----------|------------------|
| **Google News RSS** | Latest news articles | Real-time |
| **SEC EDGAR** | Official SEC filings (10-K, 10-Q, 8-K) | As filed |
| **CNN Fear & Greed Index** | Market sentiment indicator (0-100) | Daily |

### Social Media
| Source | Data Type | Update Frequency |
|--------|-----------|------------------|
| **StockTwits** | Social sentiment, trending stocks | Real-time |
| **Reddit (r/wallstreetbets, r/stocks)** | Retail sentiment, mentions | Real-time |

---

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- Ollama (optional, for offline mode)

### Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="/path/to/project"
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | No* | Claude API for AI analysis |
| `FINNHUB_API_KEY` | No | Additional market data |

*System falls back to Ollama if no API key provided

---

## API Endpoints

### Health & Status
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Service health and API availability |
| `/api/v1/` | GET | API information |

### Predictions
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` | POST | Generate full prediction with reasoning |
| `/api/v1/predict/simple` | GET | Quick prediction (query string) |

### Market Data
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/stock/{ticker}/quote` | GET | Real-time stock quote |
| `/api/v1/stock/{ticker}/info` | GET | Company fundamentals |
| `/api/v1/stock/{ticker}/technicals` | GET | Technical indicators |
| `/api/v1/stock/{ticker}/news` | GET | Recent news articles |
| `/api/v1/stock/{ticker}/social` | GET | Social media sentiment |

### Prediction History
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/history` | GET | List recent predictions |
| `/api/v1/history/{id}` | GET | Get specific prediction |
| `/api/v1/history/{id}/resolve` | POST | Manually resolve outcome |
| `/api/v1/accuracy` | GET | Accuracy statistics |

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [PREDICTION_ENGINE.md](./PREDICTION_ENGINE.md) | Algorithms and mathematical models |
| [METHODOLOGY.md](./METHODOLOGY.md) | How predictions are generated |
| [TECHNICAL_INDICATORS.md](./TECHNICAL_INDICATORS.md) | RSI, MACD, Moving Averages explained |
| [ENHANCEMENTS.md](./ENHANCEMENTS.md) | V2 enhancements: RAG, guardrails, SSE, mobile responsive |

---

## Disclaimer

**This application is for educational and informational purposes only.**

- Predictions are not financial advice
- Past performance does not guarantee future results
- Always conduct your own research before making investment decisions
- The system aggregates publicly available data and applies AI analysis
- No trading system can predict markets with certainty
- Use this tool as one of many inputs in your investment research

---

## License

MIT License - Free to use, modify, and distribute.

---

*Last updated: February 12, 2026*
