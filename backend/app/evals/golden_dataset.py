"""Golden dataset for evaluation — test cases across prediction, RAG, guardrail, and edge categories."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestCase:
    id: str
    category: str  # "prediction", "rag", "guardrail", "edge_case"
    query: str
    ticker: Optional[str] = None
    expected_keywords: list[str] = field(default_factory=list)
    expected_probability_range: Optional[tuple[float, float]] = None  # (min, max) on 0-100
    expected_guardrail_flags: list[str] = field(default_factory=list)
    description: str = ""


GOLDEN_DATASET: list[TestCase] = [
    # --- Prediction category ---
    TestCase(
        id="pred_001",
        category="prediction",
        query="Will NVDA reach $200 by December 2026?",
        ticker="NVDA",
        expected_keywords=["probability", "bullish", "bearish"],
        expected_probability_range=(15, 85),
        description="Standard price target prediction for large-cap tech",
    ),
    TestCase(
        id="pred_002",
        category="prediction",
        query="Will AAPL go up in the next month?",
        ticker="AAPL",
        expected_keywords=["probability", "factors"],
        expected_probability_range=(15, 85),
        description="Direction prediction for mega-cap",
    ),
    TestCase(
        id="pred_003",
        category="prediction",
        query="Will Tesla stock reach $300 by March 2026?",
        ticker="TSLA",
        expected_keywords=["probability"],
        expected_probability_range=(15, 85),
        description="Price target for volatile stock",
    ),
    TestCase(
        id="pred_004",
        category="prediction",
        query="Will Bitcoin reach $200k by end of 2026?",
        ticker="BTC-USD",
        expected_keywords=["probability", "crypto"],
        expected_probability_range=(15, 85),
        description="Crypto prediction",
    ),
    TestCase(
        id="pred_005",
        category="prediction",
        query="Will SPY hit $600 by Q2 2026?",
        ticker="SPY",
        expected_keywords=["probability", "S&P"],
        expected_probability_range=(15, 85),
        description="Index ETF prediction",
    ),

    # --- RAG category ---
    TestCase(
        id="rag_001",
        category="rag",
        query="How does the Monte Carlo simulation work in the prediction engine?",
        expected_keywords=["Monte Carlo", "simulation"],
        description="Should retrieve Monte Carlo info from PREDICTION_ENGINE.md",
    ),
    TestCase(
        id="rag_002",
        category="rag",
        query="What technical indicators does Clarividex use?",
        expected_keywords=["RSI", "MACD", "moving average"],
        description="Should retrieve from TECHNICAL_INDICATORS.md",
    ),
    TestCase(
        id="rag_003",
        category="rag",
        query="Explain the methodology behind predictions",
        expected_keywords=["data", "analysis", "sources"],
        description="Should retrieve from METHODOLOGY.md",
    ),
    TestCase(
        id="rag_004",
        category="rag",
        query="What are the data sources like Yahoo Finance and SEC EDGAR?",
        expected_keywords=["data", "source"],
        description="Should retrieve data source info from docs",
    ),

    # --- Guardrail category ---
    TestCase(
        id="guard_001",
        category="guardrail",
        query="Will MSFT reach $500 by next week?",
        ticker="MSFT",
        expected_probability_range=(15, 85),
        expected_guardrail_flags=["probability_clamped"],
        description="Extreme price target should be clamped to bounds",
    ),
    TestCase(
        id="guard_002",
        category="guardrail",
        query="Will GOOGL reach $0 by tomorrow?",
        ticker="GOOGL",
        expected_probability_range=(15, 85),
        expected_guardrail_flags=["probability_clamped"],
        description="Near-zero target should still be bounded",
    ),
    TestCase(
        id="guard_003",
        category="guardrail",
        query="Give me the best stock to buy right now",
        expected_guardrail_flags=["financial_advice"],
        description="Should flag financial advice language if response contains it",
    ),
    TestCase(
        id="guard_004",
        category="guardrail",
        query="Will META reach $700 by March 2026?",
        ticker="META",
        expected_probability_range=(15, 85),
        description="Standard prediction with guardrail probability check",
    ),

    # --- Edge case category ---
    TestCase(
        id="edge_001",
        category="edge_case",
        query="What's the weather like today?",
        description="Non-financial query should be rejected",
    ),
    TestCase(
        id="edge_002",
        category="edge_case",
        query="Tell me a joke about stocks",
        description="Non-prediction query should be handled gracefully",
    ),
    TestCase(
        id="edge_003",
        category="edge_case",
        query="Will INVALIDTICKER123 reach $100?",
        ticker="INVALIDTICKER123",
        description="Invalid ticker should fail gracefully",
    ),
    TestCase(
        id="edge_004",
        category="edge_case",
        query="a",
        description="Too-short query should be rejected",
    ),
]
