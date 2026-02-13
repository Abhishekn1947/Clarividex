"""
Configuration management for Clarividex.

Loads environment variables and provides typed configuration objects
for all application settings.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # API Keys
    # -------------------------------------------------------------------------
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude",
    )
    finnhub_api_key: str = Field(
        default="",
        description="Finnhub API key for market data",
    )
    alpha_vantage_api_key: str = Field(
        default="",
        description="Alpha Vantage API key",
    )
    reddit_client_id: str = Field(
        default="",
        description="Reddit OAuth client ID",
    )
    reddit_client_secret: str = Field(
        default="",
        description="Reddit OAuth client secret",
    )
    reddit_user_agent: str = Field(
        default="Clarividex/1.0",
        description="Reddit API user agent",
    )
    news_api_key: str = Field(
        default="",
        description="NewsAPI key",
    )

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/predictions_db",
        description="PostgreSQL connection URL",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_env: str = Field(
        default="development",
        description="Application environment",
    )
    app_debug: bool = Field(
        default=True,
        description="Debug mode",
    )
    app_secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for sessions/tokens",
    )
    app_host: str = Field(
        default="0.0.0.0",
        description="Application host",
    )
    app_port: int = Field(
        default=8000,
        description="Application port",
    )
    frontend_url: str = Field(
        default="http://localhost:3000",
        description="Frontend URL for CORS",
    )

    # -------------------------------------------------------------------------
    # Caching & Rate Limiting
    # -------------------------------------------------------------------------
    cache_ttl_seconds: int = Field(
        default=300,
        description="Default cache TTL in seconds",
    )
    max_requests_per_minute: int = Field(
        default=60,
        description="Rate limit per minute",
    )
    prediction_cache_hours: int = Field(
        default=1,
        description="How long to cache predictions",
    )

    # -------------------------------------------------------------------------
    # RAG Configuration
    # -------------------------------------------------------------------------
    chroma_dir: str = Field(
        default="data/chroma",
        description="ChromaDB persistence directory",
    )
    chroma_collection_name: str = Field(
        default="clarividex_docs",
        description="ChromaDB collection name",
    )
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    rag_chunk_size: int = Field(
        default=500,
        description="RAG document chunk size in characters",
    )
    rag_chunk_overlap: int = Field(
        default=100,
        description="RAG chunk overlap in characters",
    )
    rag_top_k: int = Field(
        default=3,
        description="Number of top chunks to retrieve for RAG",
    )

    # -------------------------------------------------------------------------
    # Claude Model Settings
    # -------------------------------------------------------------------------
    claude_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model to use",
    )
    max_tokens: int = Field(
        default=4096,
        description="Max tokens for Claude response",
    )
    temperature: float = Field(
        default=0.3,
        description="Temperature for Claude (lower = more deterministic)",
    )

    # -------------------------------------------------------------------------
    # External API URLs - News Sources
    # -------------------------------------------------------------------------
    google_news_rss_url: str = Field(
        default="https://news.google.com/rss/search",
        description="Google News RSS URL",
    )
    yahoo_finance_news_url: str = Field(
        default="https://finance.yahoo.com/quote",
        description="Yahoo Finance News URL",
    )
    yahoo_finance_rss_url: str = Field(
        default="https://feeds.finance.yahoo.com/rss/2.0/headline",
        description="Yahoo Finance RSS URL",
    )
    finnhub_api_url: str = Field(
        default="https://finnhub.io/api/v1",
        description="Finnhub API URL",
    )
    marketwatch_rss_url: str = Field(
        default="https://feeds.marketwatch.com/marketwatch/topstories/",
        description="MarketWatch RSS URL",
    )
    seeking_alpha_rss_url: str = Field(
        default="https://seekingalpha.com/market_currents.xml",
        description="Seeking Alpha RSS URL",
    )
    benzinga_rss_url: str = Field(
        default="https://www.benzinga.com/feed",
        description="Benzinga RSS URL",
    )
    yahoo_rss_url: str = Field(
        default="https://finance.yahoo.com/rss/topstories",
        description="Yahoo RSS URL",
    )
    reuters_rss_url: str = Field(
        default="https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        description="Reuters RSS URL",
    )
    cnbc_rss_url: str = Field(
        default="https://www.cnbc.com/id/100003114/device/rss/rss.html",
        description="CNBC RSS URL",
    )
    bloomberg_rss_url: str = Field(
        default="https://feeds.bloomberg.com/markets/news.rss",
        description="Bloomberg RSS URL",
    )
    wsj_rss_url: str = Field(
        default="https://feeds.wsj.com/xml/rss/3_7085.xml",
        description="WSJ RSS URL",
    )

    # -------------------------------------------------------------------------
    # External API URLs - Social & Sentiment
    # -------------------------------------------------------------------------
    stocktwits_api_url: str = Field(
        default="https://api.stocktwits.com/api/2/streams/symbol",
        description="StockTwits API URL",
    )
    reddit_rss_url: str = Field(
        default="https://www.reddit.com",
        description="Reddit RSS URL",
    )

    # -------------------------------------------------------------------------
    # External API URLs - SEC & Financial Data
    # -------------------------------------------------------------------------
    sec_edgar_search_url: str = Field(
        default="https://efts.sec.gov/LATEST/search-index",
        description="SEC EDGAR Search URL",
    )
    sec_submissions_url: str = Field(
        default="https://data.sec.gov/submissions",
        description="SEC Submissions URL",
    )
    sec_tickers_url: str = Field(
        default="https://www.sec.gov/files/company_tickers.json",
        description="SEC Tickers URL",
    )
    fred_api_url: str = Field(
        default="https://api.stlouisfed.org/fred/series/observations",
        description="FRED API URL",
    )
    finviz_url: str = Field(
        default="https://finviz.com/quote.ashx",
        description="Finviz URL",
    )
    finra_short_url: str = Field(
        default="https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data",
        description="FINRA Short Data URL",
    )

    # -------------------------------------------------------------------------
    # External API URLs - Market Indicators
    # -------------------------------------------------------------------------
    cnn_fear_greed_url: str = Field(
        default="https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
        description="CNN Fear & Greed URL",
    )
    alternative_fear_greed_url: str = Field(
        default="https://api.alternative.me/fng",
        description="Alternative Fear & Greed URL",
    )

    # -------------------------------------------------------------------------
    # External API URLs - Calendars
    # -------------------------------------------------------------------------
    nasdaq_earnings_url: str = Field(
        default="https://api.nasdaq.com/api/calendar/earnings",
        description="Nasdaq Earnings Calendar URL",
    )
    nasdaq_dividends_url: str = Field(
        default="https://api.nasdaq.com/api/calendar/dividends",
        description="Nasdaq Dividends Calendar URL",
    )
    investing_calendar_url: str = Field(
        default="https://www.investing.com/economic-calendar/",
        description="Investing.com Calendar URL",
    )
    forex_factory_url: str = Field(
        default="https://www.forexfactory.com/calendar",
        description="Forex Factory Calendar URL",
    )

    # -------------------------------------------------------------------------
    # Backend API URL
    # -------------------------------------------------------------------------
    backend_api_url: str = Field(
        default="http://localhost:8000",
        description="Backend API URL",
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        """Validate app environment."""
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"app_env must be one of: {allowed}")
        return v.lower()

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is in valid range."""
        if not 0 <= v <= 1:
            raise ValueError("temperature must be between 0 and 1")
        return v

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.app_env == "development"

    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key and self.anthropic_api_key != "your_anthropic_api_key_here")

    @property
    def has_finnhub_key(self) -> bool:
        """Check if Finnhub API key is configured."""
        return bool(self.finnhub_api_key and self.finnhub_api_key != "your_finnhub_api_key_here")

    @property
    def has_reddit_credentials(self) -> bool:
        """Check if Reddit credentials are configured."""
        return bool(self.reddit_client_id and self.reddit_client_secret)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Create a global settings instance for convenience
settings = get_settings()
