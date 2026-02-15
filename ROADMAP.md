# Roadmap

Public roadmap for Clarividex. Contributions and feedback welcome via [GitHub Issues](https://github.com/Abhishekn1947/Clarividex/issues).

---

## Completed

- [x] 8-model probability ensemble with transparent decision trail
- [x] 12+ data source aggregation (Yahoo Finance, news, social, SEC, options)
- [x] RAG-powered chatbot grounded in documentation
- [x] Output guardrails (PII redaction, financial advice detection, probability bounds)
- [x] SSE streaming for real-time prediction progress
- [x] Mobile responsive design (320px - 1024px+)
- [x] AWS serverless deployment (Lambda + S3 + CloudFront)
- [x] India market support (NSE/BSE) with ₹ currency
- [x] Cross-market detection and market timing info
- [x] Prompt versioning with YAML templates

---

## In Progress

### Prediction Engine V3
- [ ] Sector-specific model calibration (tech, banking, pharma behave differently)
- [ ] Market regime-aware parameter adjustment (bull/bear/sideways auto-detection)
- [ ] Expanded historical data — 10+ years for backtesting
- [ ] Improved Monte Carlo with jump-diffusion model for tail events
- [ ] Fine-tuned Bayesian signal reliability scores from backtesting results

### Hot-Swappable AI Backbone
- [ ] Claude Opus 4.6 integration for deeper reasoning
- [ ] GPT-4o integration as alternative
- [ ] A/B testing framework for comparing AI model quality
- [ ] Per-query model routing based on complexity

---

## Planned

### More Markets
- [ ] London Stock Exchange (LSE) support
- [ ] Tokyo Stock Exchange (TSE) support
- [ ] Crypto-specific prediction models (BTC, ETH with on-chain data)
- [ ] Forex pairs with macro-economic indicators

### Backtesting & Accuracy
- [ ] Automated backtesting framework against historical data
- [ ] Public accuracy dashboard — track prediction outcomes over time
- [ ] Calibration curve visualization
- [ ] Brier score tracking for probability calibration

### User Features
- [ ] Prediction history (save and review past predictions)
- [ ] Watchlist with automated alerts
- [ ] Portfolio-level risk analysis
- [ ] Custom prediction timeframes with scenario analysis

### Infrastructure
- [ ] Custom domain (clarividex.com)
- [ ] PostgreSQL for prediction history persistence
- [ ] Redis caching layer for faster repeat queries
- [ ] WebSocket support for live price updates

---

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. The best way to get started:

1. Pick an item from the roadmap above
2. Open an issue to discuss your approach
3. Submit a PR when ready

We especially welcome contributions to:
- **Algorithm optimization** — improve probability calculations
- **New data source integrations** — add financial data APIs
- **Backtesting framework** — validate prediction accuracy
- **Market expansions** — add new stock exchanges
