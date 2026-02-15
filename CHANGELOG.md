# Changelog

All notable changes to Clarividex are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.1.0] - 2026-02-15

### Added
- **India Market Support (NSE/BSE)** — Full support for Indian stocks with ₹ currency, India VIX, NSE/BSE tickers, Indian news locale, and Indian subreddits
- Market selector toggle (US/India) in the header nav
- Cross-market detection — suggests switching markets when querying the wrong one
- Market timing info bar with trading hours, short/long-term prediction notes
- ~40 Indian company ticker mappings (Reliance, TCS, Infosys, HDFC Bank, etc.)
- `market_config.py` module for per-market configuration
- Improved InfoTooltip popups with signal interpretation cards and pro tips
- Redesigned Architecture section (3-pillar layout with hot-swappable AI backbone)

### Changed
- Prediction engine now uses market-aware currency symbols (₹ vs $) throughout
- Gemini system prompt updated with Indian stock examples and currency rules
- Rule-based fallback analysis uses dynamic currency from market config
- VIX label adapts per market (VIX vs India VIX)
- Data aggregator skips US-only sources (SEC, Finviz, CNN F&G) for Indian stocks
- News service uses Indian locale for Google News RSS
- Social service uses Indian subreddits (r/IndianStreetBets, r/IndiaInvestments)

### Security
- Moved S3 bucket, CloudFront distribution ID, and Lambda function name to GitHub Secrets
- Removed hardcoded CloudFront URL from deploy verification step

---

## [2.0.0] - 2026-02-12

### Added
- **Prompt Versioning** — YAML-based prompt templates with A/B testing support
- **RAG-Powered Chat** — ChromaDB vector search grounding chatbot answers in documentation
- **Output Guardrails** — PII redaction, financial advice detection, probability clamping (15-85%)
- **SSE Streaming** — Real-time prediction progress via Server-Sent Events (8 event types)
- **Evaluation Suite** — 18-case golden dataset with 6 automated metrics
- **Mobile Responsiveness** — Full responsive design from 320px (iPhone SE) to 1024px+
- Smart Query Guidance System — three-tier guidance instead of hard rejection
- Contextual query suggestions based on trending stocks

### Changed
- Singleton caching for API client — eliminated redundant instantiation

---

## [1.2.0] - 2026-02-14

### Added
- AWS Lambda deployment with Docker (ECR image, Mangum adapter)
- S3 + CloudFront frontend hosting (static export, OAC, SPA routing)
- GitHub Actions CI/CD pipelines (ci.yml, deploy.yml, deploy-frontend.yml)
- Terraform infrastructure as code (6 modules, ~27 resources)
- EventBridge warmup rule (5-minute interval)
- CloudWatch alarms and SNS email alerts

### Changed
- Switched from Anthropic Claude to Google Gemini 2.0 Flash as primary AI
- Removed Ollama fallback — simplified to Gemini + rule-based engine
- Switched Lambda architecture to x86_64 for faster CI builds
- Added Docker layer caching (GHA cache) — builds reduced from 14min to ~5s

### Fixed
- Dual CORS headers issue — removed Lambda Function URL CORS config
- Double-slash in API URLs — strip trailing slash from base URL
- `.gitignore` `/lib/` pattern blocking `frontend/src/lib/`

---

## [1.1.0] - 2026-02-10

### Added
- Security hardening: input validation, rate limiting, error sanitization
- Next.js 16 upgrade with React 19

### Changed
- Cost optimization: eliminated double API call, added caches, reduced token usage (~65% cost reduction)

---

## [1.0.0] - 2026-02-08

### Added
- Initial release of Clarividex — AI-Powered Market Predictions
- 8-model probability ensemble (Monte Carlo, Bayesian, volatility, multi-timeframe, analyst targets, options implied, earnings surprise, options flow)
- 12+ data source aggregation (Yahoo Finance, Google News, SEC EDGAR, Finviz, StockTwits, Reddit, VIX, Fear & Greed, options, economic indicators)
- Technical analysis (RSI, MACD, SMA, EMA, Bollinger Bands, support/resistance)
- Sentiment analysis (VADER + TextBlob + custom financial lexicon)
- Pattern recognition (11 chart patterns + historical similarity matching)
- Decision trail visualization for full transparency
- Confidence scoring with probability clamping (15-85%)
- 160+ US stock ticker mappings
- Interactive Swagger UI at `/docs`
