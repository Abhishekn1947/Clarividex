"""
Pydantic schemas for request/response validation.

These models define the structure of all data flowing through the API,
ensuring type safety and automatic documentation generation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================


class InstrumentType(str, Enum):
    """Types of financial instruments supported."""

    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    ETF = "etf"
    FUTURES = "futures"
    BOND = "bond"
    UNKNOWN = "unknown"


class PredictionType(str, Enum):
    """Types of predictions supported."""

    PRICE_TARGET = "price_target"  # Will stock X reach price Y by date Z?
    DIRECTION = "direction"  # Will stock X go up or down?
    EARNINGS = "earnings"  # Will company X beat earnings?
    EVENT = "event"  # Will event X happen?


class ConfidenceLevel(str, Enum):
    """Confidence levels for predictions."""

    VERY_LOW = "very_low"  # < 30%
    LOW = "low"  # 30-45%
    MEDIUM = "medium"  # 45-55%
    MEDIUM_HIGH = "medium_high"  # 55-70%
    HIGH = "high"  # 70-85%
    VERY_HIGH = "very_high"  # > 85%


class SentimentType(str, Enum):
    """Sentiment classifications."""

    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


# =============================================================================
# Request Models
# =============================================================================


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""

    message: str = Field(..., min_length=1, max_length=2000)
    context: str = Field(default="", max_length=50000)
    ticker: Optional[str] = Field(
        default=None,
        max_length=10,
        pattern=r"^[A-Z0-9.\-^=]{1,10}$",
    )


class AnalyzeQueryRequest(BaseModel):
    """Request model for the analyze-query endpoint."""

    query: str = Field(..., min_length=1, max_length=500)


class PredictionRequest(BaseModel):
    """Request model for making a prediction."""

    query: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The prediction question to analyze",
        examples=["Will NVDA reach $150 by March 2026?", "Will Bitcoin hit $100k?", "Will EUR/USD reach 1.15?"],
    )
    ticker: Optional[str] = Field(
        default=None,
        max_length=15,
        description="Instrument symbol (auto-detected if not provided). Supports stocks, crypto, forex, commodities, indices, ETFs, futures, and bonds.",
        examples=["NVDA", "BTC-USD", "EURUSD=X", "GC=F", "^GSPC"],
    )
    instrument_type: Optional[InstrumentType] = Field(
        default=None,
        description="Type of financial instrument (auto-detected if not provided)",
    )
    target_price: Optional[float] = Field(
        default=None,
        gt=0,
        description="Target price for price_target predictions",
    )
    target_date: Optional[datetime] = Field(
        default=None,
        description="Target date for the prediction",
    )
    include_technicals: bool = Field(
        default=True,
        description="Include technical analysis in reasoning",
    )
    include_sentiment: bool = Field(
        default=True,
        description="Include social sentiment analysis",
    )
    include_news: bool = Field(
        default=True,
        description="Include news analysis",
    )

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: Optional[str]) -> Optional[str]:
        """Ensure ticker is uppercase."""
        return v.upper() if v else None

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        """Clean and normalize query."""
        return v.strip()


# =============================================================================
# Data Source Models
# =============================================================================


class DataSource(BaseModel):
    """Information about a data source used in analysis."""

    name: str = Field(..., description="Name of the data source")
    url: Optional[str] = Field(default=None, description="URL to the source")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the data was fetched",
    )
    reliability_score: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="How reliable this source is (0-1)",
    )


class NewsArticle(BaseModel):
    """A news article related to the prediction."""

    title: str = Field(..., description="Article title")
    source: str = Field(..., description="News source name")
    url: Optional[str] = Field(default=None, description="Article URL")
    published_at: datetime = Field(..., description="Publication timestamp")
    summary: Optional[str] = Field(default=None, description="Article summary")
    sentiment_score: float = Field(
        default=0,
        ge=-1,
        le=1,
        description="Sentiment score (-1 to 1)",
    )
    relevance_score: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Relevance to the query (0-1)",
    )


class SocialSentiment(BaseModel):
    """Social media sentiment data."""

    platform: str = Field(..., description="Platform name (reddit, stocktwits, etc)")
    mentions_count: int = Field(default=0, ge=0, description="Number of mentions")
    sentiment_score: float = Field(
        default=0,
        ge=-1,
        le=1,
        description="Overall sentiment (-1 to 1)",
    )
    bullish_percentage: float = Field(
        default=50,
        ge=0,
        le=100,
        description="Percentage of bullish mentions",
    )
    trending: bool = Field(default=False, description="Is this trending?")
    sample_posts: list[str] = Field(
        default_factory=list,
        description="Sample posts/comments",
    )


# =============================================================================
# Market Data Models
# =============================================================================


class StockQuote(BaseModel):
    """Real-time stock quote data."""

    ticker: str = Field(..., description="Stock ticker symbol")
    current_price: float = Field(..., description="Current stock price")
    previous_close: float = Field(..., description="Previous closing price")
    open_price: float = Field(..., description="Today's opening price")
    day_high: float = Field(..., description="Today's high")
    day_low: float = Field(..., description="Today's low")
    volume: int = Field(..., description="Trading volume")
    market_cap: Optional[float] = Field(default=None, description="Market capitalization")
    pe_ratio: Optional[float] = Field(default=None, description="P/E ratio")
    fifty_two_week_high: Optional[float] = Field(default=None, description="52-week high")
    fifty_two_week_low: Optional[float] = Field(default=None, description="52-week low")
    change_percent: float = Field(..., description="Percent change today")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Quote timestamp",
    )


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""

    ticker: str = Field(..., description="Stock ticker symbol")
    rsi_14: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="14-day RSI",
    )
    macd: Optional[float] = Field(default=None, description="MACD value")
    macd_signal: Optional[float] = Field(default=None, description="MACD signal line")
    macd_histogram: Optional[float] = Field(default=None, description="MACD histogram")
    sma_20: Optional[float] = Field(default=None, description="20-day SMA")
    sma_50: Optional[float] = Field(default=None, description="50-day SMA")
    sma_200: Optional[float] = Field(default=None, description="200-day SMA")
    ema_12: Optional[float] = Field(default=None, description="12-day EMA")
    ema_26: Optional[float] = Field(default=None, description="26-day EMA")
    bollinger_upper: Optional[float] = Field(default=None, description="Bollinger upper band")
    bollinger_lower: Optional[float] = Field(default=None, description="Bollinger lower band")
    atr_14: Optional[float] = Field(default=None, description="14-day ATR")
    volume_sma_20: Optional[float] = Field(default=None, description="20-day volume SMA")
    support_level: Optional[float] = Field(default=None, description="Key support level")
    resistance_level: Optional[float] = Field(default=None, description="Key resistance level")

    @property
    def rsi_signal(self) -> Optional[str]:
        """Interpret RSI value."""
        if self.rsi_14 is None:
            return None
        if self.rsi_14 >= 70:
            return "overbought"
        if self.rsi_14 <= 30:
            return "oversold"
        return "neutral"

    @property
    def trend_signal(self) -> Optional[str]:
        """Determine trend based on moving averages."""
        if self.sma_20 is None or self.sma_50 is None:
            return None
        if self.sma_20 > self.sma_50:
            return "bullish"
        if self.sma_20 < self.sma_50:
            return "bearish"
        return "neutral"


class CompanyInfo(BaseModel):
    """Company fundamental information."""

    ticker: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    sector: Optional[str] = Field(default=None, description="Business sector")
    industry: Optional[str] = Field(default=None, description="Industry")
    description: Optional[str] = Field(default=None, description="Company description")
    website: Optional[str] = Field(default=None, description="Company website")
    employees: Optional[int] = Field(default=None, description="Number of employees")
    market_cap: Optional[float] = Field(default=None, description="Market capitalization")
    pe_ratio: Optional[float] = Field(default=None, description="P/E ratio")
    forward_pe: Optional[float] = Field(default=None, description="Forward P/E")
    peg_ratio: Optional[float] = Field(default=None, description="PEG ratio")
    price_to_book: Optional[float] = Field(default=None, description="Price to book")
    dividend_yield: Optional[float] = Field(default=None, description="Dividend yield")
    profit_margin: Optional[float] = Field(default=None, description="Profit margin")
    revenue_growth: Optional[float] = Field(default=None, description="Revenue growth YoY")
    earnings_growth: Optional[float] = Field(default=None, description="Earnings growth YoY")


# =============================================================================
# Reasoning & Prediction Models
# =============================================================================


class Factor(BaseModel):
    """A factor contributing to the prediction."""

    description: str = Field(..., description="Description of the factor")
    impact: str = Field(
        ...,
        description="Impact direction: bullish or bearish",
    )
    weight: float = Field(
        ...,
        ge=-1,
        le=1,
        description="Weight/importance (-1 to 1)",
    )
    source: str = Field(..., description="Data source for this factor")
    confidence: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Confidence in this factor",
    )


class Catalyst(BaseModel):
    """An upcoming event that could affect the prediction."""

    event: str = Field(..., description="Description of the catalyst")
    date: Optional[datetime] = Field(default=None, description="When it will occur")
    potential_impact: str = Field(
        ...,
        description="Potential impact: positive, negative, or uncertain",
    )
    importance: str = Field(
        default="medium",
        description="Importance: low, medium, high",
    )


class DecisionNode(BaseModel):
    """A single node in the decision tree showing a data source and its contribution."""

    id: str = Field(..., description="Unique node identifier")
    category: str = Field(..., description="Category: technical, news, options, market, analyst, social")
    source: str = Field(..., description="Data source name")
    data_point: str = Field(..., description="The specific data point analyzed")
    value: Optional[str] = Field(default=None, description="The value/result")
    signal: str = Field(..., description="Signal interpretation: bullish, bearish, or neutral")
    weight: float = Field(..., ge=0, le=1, description="Weight contribution to final score")
    score_contribution: float = Field(..., description="Actual contribution to probability")
    reasoning: str = Field(..., description="Why this data led to this signal")


class DecisionTrail(BaseModel):
    """Complete decision trail showing how the prediction was made."""

    nodes: list[DecisionNode] = Field(
        default_factory=list,
        description="All decision nodes that contributed to the prediction",
    )
    category_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Aggregated scores by category",
    )
    category_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Weights used for each category",
    )
    final_calculation: str = Field(
        default="",
        description="Human-readable final calculation formula",
    )


class ReasoningChain(BaseModel):
    """Complete reasoning chain for a prediction."""

    summary: str = Field(
        ...,
        description="Executive summary of the analysis",
    )
    bullish_factors: list[Factor] = Field(
        default_factory=list,
        description="Factors supporting positive outcome",
    )
    bearish_factors: list[Factor] = Field(
        default_factory=list,
        description="Factors supporting negative outcome",
    )
    catalysts: list[Catalyst] = Field(
        default_factory=list,
        description="Upcoming events to watch",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Key risks to the prediction",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Key assumptions made",
    )
    decision_trail: Optional[DecisionTrail] = Field(
        default=None,
        description="Full decision trail showing how prediction was calculated",
    )


class AccuracyMetrics(BaseModel):
    """Accuracy tracking metrics."""

    total_predictions: int = Field(default=0, description="Total predictions made")
    resolved_predictions: int = Field(
        default=0,
        description="Predictions that have reached their target date",
    )
    correct_predictions: int = Field(
        default=0,
        description="Predictions that were correct",
    )
    accuracy_rate: float = Field(
        default=0,
        ge=0,
        le=1,
        description="Overall accuracy (0-1)",
    )
    accuracy_by_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Accuracy broken down by confidence level",
    )
    accuracy_by_sector: dict[str, float] = Field(
        default_factory=dict,
        description="Accuracy broken down by sector",
    )


# =============================================================================
# Ticker Extraction Models
# =============================================================================


class TickerSuggestion(BaseModel):
    """A suggested ticker match when extraction is ambiguous."""

    ticker: str = Field(..., description="Suggested ticker symbol")
    company_name: str = Field(..., description="Company name for this ticker")
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence that this is the intended ticker (0-1)",
    )
    match_reason: str = Field(..., description="Why this ticker was suggested")


class TickerExtractionResult(BaseModel):
    """Result of attempting to extract a ticker from a query."""

    ticker: Optional[str] = Field(default=None, description="Extracted ticker (if confident)")
    company_name: Optional[str] = Field(default=None, description="Company name if known")
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Confidence in the extraction (0-1)",
    )
    needs_confirmation: bool = Field(
        default=False,
        description="True if user confirmation is recommended",
    )
    suggestions: list[TickerSuggestion] = Field(
        default_factory=list,
        description="Alternative ticker suggestions if ambiguous",
    )
    message: Optional[str] = Field(
        default=None,
        description="Message to show the user (e.g., 'Did you mean KTOS for Kratos Security?')",
    )


# =============================================================================
# Response Models
# =============================================================================


class DataLimitation(BaseModel):
    """Warning about data limitations."""

    category: str = Field(..., description="Category: data_quality, sources, recency, coverage")
    severity: str = Field(..., description="Severity: low, medium, high")
    message: str = Field(..., description="Human-readable warning message")
    recommendation: Optional[str] = Field(default=None, description="Recommended action")


class PredictionResponse(BaseModel):
    """Complete prediction response."""

    # Core prediction
    id: str = Field(..., description="Unique prediction ID")
    query: str = Field(..., description="Original query")
    ticker: Optional[str] = Field(default=None, description="Instrument symbol if applicable")
    instrument_type: InstrumentType = Field(
        default=InstrumentType.STOCK,
        description="Type of financial instrument (stock, crypto, forex, commodity, etc.)",
    )
    prediction_type: PredictionType = Field(..., description="Type of prediction")

    # Probability & confidence
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of outcome (0-1)",
    )
    confidence_level: ConfidenceLevel = Field(..., description="Confidence level")
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Numerical confidence (0-1)",
    )

    # Target info
    target_price: Optional[float] = Field(default=None, description="Target price")
    target_date: Optional[datetime] = Field(default=None, description="Target date")
    current_price: Optional[float] = Field(default=None, description="Current price")
    price_gap_percent: Optional[float] = Field(
        default=None,
        description="Gap between current and target price (%)",
    )

    # Reasoning
    reasoning: ReasoningChain = Field(..., description="Full reasoning chain")

    # Supporting data
    sentiment: SentimentType = Field(..., description="Overall sentiment")
    sentiment_score: float = Field(
        ...,
        ge=-1,
        le=1,
        description="Numerical sentiment (-1 to 1)",
    )
    technical_score: Optional[float] = Field(
        default=None,
        ge=-1,
        le=1,
        description="Technical analysis score",
    )
    fundamental_score: Optional[float] = Field(
        default=None,
        ge=-1,
        le=1,
        description="Fundamental analysis score",
    )

    # Data quality
    data_quality_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Quality of data used (0-1)",
    )
    data_points_analyzed: int = Field(
        ...,
        description="Number of data points analyzed",
    )
    sources_used: list[DataSource] = Field(
        default_factory=list,
        description="Data sources used",
    )
    data_limitations: list[DataLimitation] = Field(
        default_factory=list,
        description="Warnings about data limitations that may affect prediction accuracy",
    )
    has_limited_data: bool = Field(
        default=False,
        description="True if prediction is based on limited data",
    )

    # News & social
    news_articles: list[NewsArticle] = Field(
        default_factory=list,
        description="Relevant news articles",
    )
    social_sentiment: list[SocialSentiment] = Field(
        default_factory=list,
        description="Social media sentiment",
    )

    # Technical data
    technicals: Optional[TechnicalIndicators] = Field(
        default=None,
        description="Technical indicators",
    )

    # Track record
    historical_accuracy: Optional[AccuracyMetrics] = Field(
        default=None,
        description="Our historical accuracy",
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When prediction was made",
    )
    disclaimer: str = Field(
        default=(
            "This prediction is for informational purposes only and should not be "
            "considered financial advice. Past performance does not guarantee future results. "
            "Always do your own research and consult a financial advisor."
        ),
        description="Legal disclaimer",
    )


# =============================================================================
# Health & Status Models
# =============================================================================


class APIStatus(BaseModel):
    """Status of an external API."""

    name: str = Field(..., description="API name")
    available: bool = Field(..., description="Is API available?")
    last_check: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last health check",
    )
    error: Optional[str] = Field(default=None, description="Error message if unavailable")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Check timestamp",
    )
    apis: list[APIStatus] = Field(
        default_factory=list,
        description="Status of external APIs",
    )
    database_connected: bool = Field(
        default=False,
        description="Database connection status",
    )
    cache_connected: bool = Field(
        default=False,
        description="Cache connection status",
    )
