# Contributing to Clarividex

First off, thank you for considering contributing to Clarividex! It's people like you that make Clarividex such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include screenshots if possible**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List any alternative solutions you've considered**

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. Ensure the test suite passes
4. Make sure your code follows the existing code style
5. Issue that pull request!

## Development Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

### Running Tests
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## Style Guides

### Python Style Guide
- Follow PEP 8
- Use type hints
- Write docstrings for all public functions

### TypeScript Style Guide
- Use TypeScript strict mode
- Prefer functional components with hooks
- Use meaningful variable and function names

### Git Commit Messages
- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less

## Multi-Market Contributions

Clarividex supports US and India (NSE/BSE) markets. When contributing:

- **Backend changes** should respect the `market` parameter (`"US"` | `"IN"`) that flows through all services
- **New data sources** should specify which markets they support in `market_config.py`
- **Indian tickers** use `.NS` suffix (NSE) — handled automatically by `market_data.py`
- **Currency**: Use the `currency_symbol` from `get_market_config(market)` — never hardcode `$` or `₹`
- **US-only sources** (SEC EDGAR, Finviz, CNN Fear & Greed) are skipped when `market="IN"`

## Questions?

Feel free to open an issue with your question or reach out to the maintainers.

Thank you for contributing! 🎉
