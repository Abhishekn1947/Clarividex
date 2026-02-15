"""
Data Aggregator Service - Combines data from ALL available sources.

This is the central service that coordinates data fetching from:
- Market data (yfinance) - Real-time quotes, fundamentals
- News (Google News, Finnhub, MarketWatch, Seeking Alpha, Benzinga, CNBC)
- Social sentiment (Reddit, StockTwits, Google Trends)
- Technical analysis (RSI, MACD, Bollinger Bands, etc.)
- SEC filings (10-K, 10-Q, 8-K, insider trades)
- Options data (Put/Call ratio, open interest)
- Market sentiment (Fear & Greed Index, VIX)
- Economic indicators (Treasury yields, Fed rates)
- Analyst data (Price targets, ratings from Finviz)
- Sector performance (Sector ETFs comparison)
"""

import asyncio
from datetime import datetime
from typing import Optional

import structlog

from backend.app.models.schemas import (
    StockQuote,
    CompanyInfo,
    TechnicalIndicators,
    NewsArticle,
    SocialSentiment,
    DataSource,
)
from backend.app.services.market_data import market_data_service
from backend.app.services.news_service import news_service
from backend.app.services.social_service import social_service
from backend.app.services.technical_analysis import technical_analysis_service
from backend.app.services.sentiment_service import sentiment_service
from backend.app.services.additional_data_sources import additional_data_sources
from backend.app.services.market_config import get_market_config

logger = structlog.get_logger()


class AggregatedData:
    """Container for all aggregated data for a prediction."""

    def __init__(
        self,
        ticker: str,
        quote: Optional[StockQuote] = None,
        company_info: Optional[CompanyInfo] = None,
        technicals: Optional[TechnicalIndicators] = None,
        news: Optional[list[NewsArticle]] = None,
        social: Optional[list[SocialSentiment]] = None,
        insider_transactions: Optional[list[dict]] = None,
        analyst_recommendations: Optional[dict] = None,
        earnings_dates: Optional[list[dict]] = None,
        # New data sources
        sec_filings: Optional[dict] = None,
        finviz_data: Optional[dict] = None,
        fear_greed: Optional[dict] = None,
        vix_data: Optional[dict] = None,
        options_data: Optional[dict] = None,
        economic_indicators: Optional[dict] = None,
        sector_performance: Optional[dict] = None,
        additional_news: Optional[list[dict]] = None,
        historical_prices: Optional[list[dict]] = None,
        # IMPROVEMENT #3: Earnings surprise history
        earnings_history: Optional[list[dict]] = None,
    ):
        self.ticker = ticker
        self.quote = quote
        self.company_info = company_info
        self.technicals = technicals
        self.news = news or []
        self.social = social or []
        self.insider_transactions = insider_transactions or []
        self.analyst_recommendations = analyst_recommendations or {}
        self.earnings_dates = earnings_dates or []
        # New data
        self.sec_filings = sec_filings or {}
        self.finviz_data = finviz_data or {}
        self.fear_greed = fear_greed or {}
        self.vix_data = vix_data or {}
        self.options_data = options_data or {}
        self.economic_indicators = economic_indicators or {}
        self.sector_performance = sector_performance or {}
        self.additional_news = additional_news or []
        self.historical_prices = historical_prices or []
        # IMPROVEMENT #3: Earnings history
        self.earnings_history = earnings_history or []
        self.data_sources: list[DataSource] = []
        self.fetched_at = datetime.utcnow()

    @property
    def current_price(self) -> Optional[float]:
        """Get current stock price."""
        return self.quote.current_price if self.quote else None

    @property
    def company_name(self) -> Optional[str]:
        """Get company name."""
        return self.company_info.name if self.company_info else None

    @property
    def data_quality_score(self) -> float:
        """
        Calculate data quality score based on what we have.

        Returns:
            Score from 0 to 1
        """
        score = 0
        max_score = 0

        # Quote data (essential)
        max_score += 3
        if self.quote:
            score += 3

        # Company info
        max_score += 1
        if self.company_info:
            score += 1

        # Technical indicators
        max_score += 2
        if self.technicals:
            score += 2

        # News
        max_score += 2
        if self.news and len(self.news) >= 5:
            score += 2
        elif self.news and len(self.news) > 0:
            score += 1

        # Social sentiment
        max_score += 2
        if self.social and len(self.social) >= 2:
            score += 2
        elif self.social:
            score += 1

        # NEW: SEC filings
        max_score += 1
        if self.sec_filings and self.sec_filings.get("filings"):
            score += 1

        # NEW: Finviz data (analyst targets, ratings)
        max_score += 2
        if self.finviz_data and not self.finviz_data.get("error"):
            score += 2

        # NEW: Market sentiment (Fear & Greed + VIX)
        max_score += 1
        if self.fear_greed and self.fear_greed.get("value"):
            score += 0.5
        if self.vix_data and self.vix_data.get("value"):
            score += 0.5

        # NEW: Options data
        max_score += 1
        if self.options_data and not self.options_data.get("error"):
            score += 1

        # NEW: Economic indicators
        max_score += 1
        if self.economic_indicators:
            score += 1

        # NEW: Additional news sources
        max_score += 1
        if self.additional_news and len(self.additional_news) >= 3:
            score += 1

        return round(score / max_score, 2) if max_score > 0 else 0

    @property
    def data_points_count(self) -> int:
        """Count total data points analyzed."""
        count = 0

        if self.quote:
            count += 10  # Multiple price points

        if self.company_info:
            count += 15  # Multiple fundamental metrics

        if self.technicals:
            count += 15  # Multiple indicators

        count += len(self.news) * 2  # Title + content

        for s in self.social:
            count += s.mentions_count

        count += len(self.insider_transactions) * 3

        # NEW: SEC filings
        if self.sec_filings:
            count += len(self.sec_filings.get("filings", [])) * 2
            count += self.sec_filings.get("insider_filings", 0)

        # NEW: Finviz data (25+ metrics)
        if self.finviz_data and not self.finviz_data.get("error"):
            count += 25

        # NEW: Market sentiment indicators
        if self.fear_greed:
            count += 5
        if self.vix_data:
            count += 5

        # NEW: Options data
        if self.options_data and not self.options_data.get("error"):
            count += 8

        # NEW: Economic indicators
        if self.economic_indicators:
            count += len(self.economic_indicators) * 2

        # NEW: Sector performance
        if self.sector_performance:
            count += len(self.sector_performance)

        # NEW: Additional news
        count += len(self.additional_news) * 2

        return count

    def get_news_sentiment(self) -> dict:
        """Get aggregated news sentiment."""
        return news_service.calculate_news_sentiment(self.news)

    def get_social_sentiment(self) -> dict:
        """Get aggregated social sentiment."""
        return social_service.calculate_aggregate_sentiment(self.social)

    def get_technical_signal(self) -> dict:
        """Get technical analysis signal."""
        if self.technicals:
            return technical_analysis_service.get_technical_signal(self.technicals)
        return {"overall_signal": "neutral", "score": 0, "signals": []}

    def to_context_string(self) -> str:
        """
        Convert aggregated data to a string for LLM context.

        Returns:
            Formatted string with all data
        """
        lines = []
        lines.append(f"=== DATA FOR {self.ticker} ===")
        lines.append(f"Data fetched at: {self.fetched_at.isoformat()}")
        lines.append(f"Data quality score: {self.data_quality_score}")
        lines.append(f"Total data points: {self.data_points_count}")
        lines.append("")

        # Price data
        if self.quote:
            lines.append("--- PRICE DATA ---")
            lines.append(f"Current Price: ${self.quote.current_price:.2f}")
            lines.append(f"Previous Close: ${self.quote.previous_close:.2f}")
            lines.append(f"Day Change: {self.quote.change_percent:+.2f}%")
            lines.append(f"Day Range: ${self.quote.day_low:.2f} - ${self.quote.day_high:.2f}")
            lines.append(f"Volume: {self.quote.volume:,}")
            if self.quote.market_cap:
                lines.append(f"Market Cap: ${self.quote.market_cap:,.0f}")
            if self.quote.pe_ratio:
                lines.append(f"P/E Ratio: {self.quote.pe_ratio:.2f}")
            if self.quote.fifty_two_week_high:
                lines.append(f"52-Week High: ${self.quote.fifty_two_week_high:.2f}")
            if self.quote.fifty_two_week_low:
                lines.append(f"52-Week Low: ${self.quote.fifty_two_week_low:.2f}")
            lines.append("")

        # Company info
        if self.company_info:
            lines.append("--- COMPANY INFO ---")
            lines.append(f"Name: {self.company_info.name}")
            if self.company_info.sector:
                lines.append(f"Sector: {self.company_info.sector}")
            if self.company_info.industry:
                lines.append(f"Industry: {self.company_info.industry}")
            if self.company_info.profit_margin:
                lines.append(f"Profit Margin: {self.company_info.profit_margin*100:.1f}%")
            if self.company_info.revenue_growth:
                lines.append(f"Revenue Growth: {self.company_info.revenue_growth*100:.1f}%")
            if self.company_info.earnings_growth:
                lines.append(f"Earnings Growth: {self.company_info.earnings_growth*100:.1f}%")
            lines.append("")

        # Technical indicators
        if self.technicals:
            lines.append("--- TECHNICAL INDICATORS ---")
            signal = self.get_technical_signal()
            lines.append(f"Overall Signal: {signal['overall_signal']}")
            lines.append(f"Technical Score: {signal['score']}")
            if self.technicals.rsi_14:
                lines.append(f"RSI (14): {self.technicals.rsi_14:.1f} ({self.technicals.rsi_signal})")
            if self.technicals.macd:
                lines.append(f"MACD: {self.technicals.macd:.4f}")
            if self.technicals.sma_20:
                lines.append(f"SMA 20: ${self.technicals.sma_20:.2f}")
            if self.technicals.sma_50:
                lines.append(f"SMA 50: ${self.technicals.sma_50:.2f}")
            if self.technicals.support_level:
                lines.append(f"Support: ${self.technicals.support_level:.2f}")
            if self.technicals.resistance_level:
                lines.append(f"Resistance: ${self.technicals.resistance_level:.2f}")
            lines.append("")

        # News sentiment
        if self.news:
            news_sentiment = self.get_news_sentiment()
            lines.append("--- NEWS ANALYSIS ---")
            lines.append(f"Articles Analyzed: {len(self.news)}")
            lines.append(f"News Sentiment Score: {news_sentiment['overall_score']:.3f}")
            lines.append(f"Bullish Articles: {news_sentiment['bullish_count']}")
            lines.append(f"Bearish Articles: {news_sentiment['bearish_count']}")
            lines.append(f"Neutral Articles: {news_sentiment['neutral_count']}")
            lines.append("")
            lines.append("Recent Headlines:")
            for article in self.news[:5]:
                sentiment_label = "📈" if article.sentiment_score > 0.1 else "📉" if article.sentiment_score < -0.1 else "➖"
                lines.append(f"  {sentiment_label} {article.title[:80]} ({article.source})")
            lines.append("")

        # Social sentiment
        if self.social:
            social_sentiment = self.get_social_sentiment()
            lines.append("--- SOCIAL SENTIMENT ---")
            lines.append(f"Platforms Analyzed: {social_sentiment['platforms_analyzed']}")
            lines.append(f"Total Mentions: {social_sentiment['total_mentions']}")
            lines.append(f"Overall Sentiment: {social_sentiment['overall_score']:.3f}")
            lines.append(f"Bullish %: {social_sentiment['overall_bullish_pct']:.1f}%")
            lines.append(f"Trending: {'Yes' if social_sentiment['is_trending'] else 'No'}")
            for s in self.social:
                lines.append(f"  - {s.platform}: {s.sentiment_score:.3f} ({s.mentions_count} mentions)")
            lines.append("")

        # Insider transactions
        if self.insider_transactions:
            lines.append("--- INSIDER ACTIVITY ---")
            for txn in self.insider_transactions[:5]:
                lines.append(f"  - {txn.get('insider', 'Unknown')}: {txn.get('transaction', 'Unknown')}")
            lines.append("")

        # Analyst recommendations
        if self.analyst_recommendations and self.analyst_recommendations.get("counts"):
            lines.append("--- ANALYST RECOMMENDATIONS ---")
            for grade, count in self.analyst_recommendations["counts"].items():
                lines.append(f"  - {grade}: {count}")
            lines.append("")

        # Upcoming catalysts
        if self.earnings_dates:
            lines.append("--- UPCOMING CATALYSTS ---")
            for earning in self.earnings_dates[:3]:
                lines.append(f"  - Earnings: {earning.get('date', 'TBD')}")
            lines.append("")

        # NEW: SEC Filings
        if self.sec_filings and self.sec_filings.get("filings"):
            lines.append("--- SEC FILINGS ---")
            lines.append(f"Total Recent Filings: {self.sec_filings.get('total_filings', 0)}")
            lines.append(f"Insider Filings (Form 4): {self.sec_filings.get('insider_filings', 0)}")
            lines.append(f"Recent 8-K Filings: {self.sec_filings.get('recent_8k_count', 0)}")
            if self.sec_filings.get("last_10k"):
                lines.append(f"Last 10-K: {self.sec_filings['last_10k'].get('date', 'N/A')}")
            if self.sec_filings.get("last_10q"):
                lines.append(f"Last 10-Q: {self.sec_filings['last_10q'].get('date', 'N/A')}")
            lines.append("")

        # NEW: Finviz Data (Analyst Targets)
        if self.finviz_data and not self.finviz_data.get("error"):
            lines.append("--- ANALYST DATA (FINVIZ) ---")
            if self.finviz_data.get("target_price"):
                lines.append(f"Analyst Target Price: ${self.finviz_data['target_price']:.2f}")
            if self.finviz_data.get("upside_pct"):
                lines.append(f"Upside to Target: {self.finviz_data['upside_pct']:.1f}%")
            if self.finviz_data.get("recommendation_text"):
                lines.append(f"Analyst Consensus: {self.finviz_data['recommendation_text']}")
            if self.finviz_data.get("short_float"):
                lines.append(f"Short Float: {self.finviz_data['short_float']}")
            if self.finviz_data.get("short_ratio"):
                lines.append(f"Short Ratio: {self.finviz_data['short_ratio']}")
            if self.finviz_data.get("insider_own"):
                lines.append(f"Insider Ownership: {self.finviz_data['insider_own']}")
            if self.finviz_data.get("inst_own"):
                lines.append(f"Institutional Ownership: {self.finviz_data['inst_own']}")
            lines.append("")

        # NEW: Market Sentiment
        if self.fear_greed or self.vix_data:
            lines.append("--- MARKET SENTIMENT ---")
            if self.fear_greed:
                lines.append(f"Fear & Greed Index: {self.fear_greed.get('value', 'N/A')} ({self.fear_greed.get('rating', 'N/A')})")
                if self.fear_greed.get("week_ago"):
                    lines.append(f"  Week Ago: {self.fear_greed['week_ago']}")
            if self.vix_data:
                lines.append(f"VIX (Volatility): {self.vix_data.get('value', 'N/A')}")
                lines.append(f"  Interpretation: {self.vix_data.get('interpretation', 'N/A')}")
                lines.append(f"  Market Sentiment: {self.vix_data.get('market_sentiment', 'N/A')}")
            lines.append("")

        # NEW: Options Data
        if self.options_data and not self.options_data.get("error"):
            lines.append("--- OPTIONS DATA ---")
            lines.append(f"Put/Call Ratio (Volume): {self.options_data.get('put_call_ratio_volume', 'N/A')}")
            lines.append(f"Put/Call Ratio (OI): {self.options_data.get('put_call_ratio_oi', 'N/A')}")
            lines.append(f"Options Sentiment: {self.options_data.get('sentiment', 'N/A')}")
            lines.append(f"Call Volume: {self.options_data.get('call_volume', 0):,}")
            lines.append(f"Put Volume: {self.options_data.get('put_volume', 0):,}")
            lines.append("")

        # NEW: Economic Indicators
        if self.economic_indicators:
            lines.append("--- ECONOMIC INDICATORS ---")
            if self.economic_indicators.get("treasury_10y"):
                lines.append(f"10-Year Treasury: {self.economic_indicators['treasury_10y'].get('value', 'N/A')}%")
            if self.economic_indicators.get("treasury_30y"):
                lines.append(f"30-Year Treasury: {self.economic_indicators['treasury_30y'].get('value', 'N/A')}%")
            if self.economic_indicators.get("yield_curve_spread"):
                spread = self.economic_indicators["yield_curve_spread"]
                lines.append(f"Yield Curve: {spread.get('value', 'N/A')} ({spread.get('interpretation', 'N/A')})")
            lines.append("")

        # NEW: Sector Performance
        if self.sector_performance:
            lines.append("--- SECTOR PERFORMANCE (5-DAY) ---")
            sorted_sectors = sorted(
                self.sector_performance.items(),
                key=lambda x: x[1].get("change_pct", 0),
                reverse=True
            )
            for sector, data in sorted_sectors[:5]:
                change = data.get("change_pct", 0)
                emoji = "📈" if change > 0 else "📉" if change < 0 else "➖"
                lines.append(f"  {emoji} {sector}: {change:+.2f}%")
            lines.append("")

        # NEW: Additional News Sources
        if self.additional_news:
            lines.append("--- ADDITIONAL NEWS SOURCES ---")
            for article in self.additional_news[:5]:
                source = article.get("source", "Unknown")
                title = article.get("title", "")[:70]
                lines.append(f"  [{source}] {title}")
            lines.append("")

        return "\n".join(lines)


class DataAggregator:
    """Service for aggregating data from multiple sources."""

    def __init__(self):
        """Initialize the data aggregator."""
        self.logger = logger.bind(service="data_aggregator")

    async def aggregate_data(
        self,
        ticker: str,
        include_technicals: bool = True,
        include_news: bool = True,
        include_social: bool = True,
        include_extended: bool = True,
        market: str = "US",
    ) -> AggregatedData:
        """
        Aggregate all data for a ticker from ALL available sources.

        Args:
            ticker: Stock ticker symbol
            include_technicals: Whether to include technical analysis
            include_news: Whether to include news data
            include_social: Whether to include social sentiment
            include_extended: Whether to include extended data (SEC, Finviz, etc.)
            market: Market context ("US" or "IN")

        Returns:
            AggregatedData object with all fetched data
        """
        self.logger.info(
            "Aggregating data from ALL sources",
            ticker=ticker,
            market=market,
            include_technicals=include_technicals,
            include_news=include_news,
            include_social=include_social,
            include_extended=include_extended,
        )

        # Create result container
        data = AggregatedData(ticker=ticker)

        # Fetch basic market data synchronously (fast)
        data.quote = market_data_service.get_quote(ticker)
        data.company_info = market_data_service.get_company_info(ticker)

        if data.quote:
            data.data_sources.append(DataSource(
                name="yfinance",
                reliability_score=0.9,
            ))

        # Build async tasks for slower operations
        tasks = {}

        market_config = get_market_config(market)

        if include_news and data.company_info:
            tasks["news"] = self._fetch_news(ticker, data.company_info.name, market=market)

        if include_social and data.company_info:
            tasks["social"] = self._fetch_social(ticker, data.company_info.name, market=market)

        # Extended data sources (market-aware: skip unavailable sources)
        if include_extended:
            # SEC filings only available for US market
            if market_config["has_sec_filings"]:
                tasks["sec_filings"] = self._fetch_sec_filings(ticker)
            # Finviz only available for US market
            if market_config["has_finviz"]:
                tasks["finviz"] = self._fetch_finviz_data(ticker)
            # Fear & Greed only available for US market
            if market_config["fear_greed_available"]:
                tasks["fear_greed"] = self._fetch_fear_greed()
            tasks["vix"] = self._fetch_vix_data(market=market)
            tasks["options"] = self._fetch_options_data(ticker)
            tasks["economic"] = self._fetch_economic_indicators()
            tasks["sectors"] = self._fetch_sector_performance(market=market)
            tasks["additional_news"] = self._fetch_additional_news(ticker, market=market)

        # Run ALL async tasks concurrently
        if tasks:
            task_list = list(tasks.values())
            task_names = list(tasks.keys())

            results = await asyncio.gather(*task_list, return_exceptions=True)

            for name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Task {name} failed", error=str(result))
                    continue

                # Process based on task type
                if name == "news" and isinstance(result, list):
                    if result and isinstance(result[0], NewsArticle):
                        data.news = result
                        data.data_sources.append(DataSource(
                            name="News Aggregator",
                            reliability_score=0.8,
                        ))

                elif name == "social" and isinstance(result, list):
                    if result and isinstance(result[0], SocialSentiment):
                        data.social = result
                        data.data_sources.append(DataSource(
                            name="Social Sentiment",
                            reliability_score=0.7,
                        ))

                elif name == "sec_filings" and isinstance(result, dict):
                    if not result.get("error"):
                        data.sec_filings = result
                        data.data_sources.append(DataSource(
                            name="SEC EDGAR",
                            reliability_score=0.95,
                        ))

                elif name == "finviz" and isinstance(result, dict):
                    if not result.get("error"):
                        data.finviz_data = result
                        data.data_sources.append(DataSource(
                            name="Finviz",
                            reliability_score=0.85,
                        ))

                elif name == "fear_greed" and isinstance(result, dict):
                    data.fear_greed = result
                    data.data_sources.append(DataSource(
                        name="Fear & Greed Index",
                        reliability_score=0.8,
                    ))

                elif name == "vix" and isinstance(result, dict):
                    if not result.get("error"):
                        data.vix_data = result
                        data.data_sources.append(DataSource(
                            name="VIX",
                            reliability_score=0.9,
                        ))

                elif name == "options" and isinstance(result, dict):
                    if not result.get("error"):
                        data.options_data = result
                        data.data_sources.append(DataSource(
                            name="Options Data",
                            reliability_score=0.85,
                        ))

                elif name == "economic" and isinstance(result, dict):
                    data.economic_indicators = result
                    data.data_sources.append(DataSource(
                        name="Economic Indicators",
                        reliability_score=0.95,
                    ))

                elif name == "sectors" and isinstance(result, dict):
                    data.sector_performance = result
                    data.data_sources.append(DataSource(
                        name="Sector ETFs",
                        reliability_score=0.9,
                    ))

                elif name == "additional_news" and isinstance(result, list):
                    data.additional_news = result
                    data.data_sources.append(DataSource(
                        name="Multi-Source News",
                        reliability_score=0.8,
                    ))

        # Calculate technical indicators (synchronous)
        if include_technicals:
            data.technicals = technical_analysis_service.calculate_indicators(ticker)
            if data.technicals:
                data.data_sources.append(DataSource(
                    name="Technical Analysis",
                    reliability_score=0.85,
                ))

        # Fetch additional yfinance data
        data.insider_transactions = market_data_service.get_insider_transactions(ticker)
        data.analyst_recommendations = market_data_service.get_analyst_recommendations(ticker)
        data.earnings_dates = market_data_service.get_earnings_dates(ticker)

        # IMPROVEMENT #3: Fetch earnings history
        data.earnings_history = self._fetch_earnings_history_sync(ticker)

        # Fetch historical prices for probability engine (1 year of data)
        historical_df = market_data_service.get_historical_data(ticker, period="1y")
        if historical_df is not None and not historical_df.empty:
            data.historical_prices = [
                {
                    "date": str(idx.date()) if hasattr(idx, 'date') else str(idx),
                    "open": float(row.get("Open", 0)),
                    "high": float(row.get("High", 0)),
                    "low": float(row.get("Low", 0)),
                    "close": float(row.get("Close", 0)),
                    "volume": int(row.get("Volume", 0)),
                }
                for idx, row in historical_df.iterrows()
            ]
            data.data_sources.append(DataSource(
                name="Historical Prices",
                reliability_score=0.95,
            ))
            self.logger.info("Historical prices fetched", count=len(data.historical_prices))

        self.logger.info(
            "Data aggregation complete",
            ticker=ticker,
            quality_score=data.data_quality_score,
            data_points=data.data_points_count,
            sources=len(data.data_sources),
        )

        return data

    async def _fetch_news(self, ticker: str, company_name: Optional[str], market: str = "US") -> list[NewsArticle]:
        """Fetch news articles."""
        try:
            return await news_service.get_news_for_ticker(ticker, company_name, market=market)
        except Exception as e:
            self.logger.error("News fetch failed", error=str(e))
            return []

    async def _fetch_social(self, ticker: str, company_name: Optional[str], market: str = "US") -> list[SocialSentiment]:
        """Fetch social sentiment."""
        try:
            return await social_service.get_social_sentiment(ticker, company_name, market=market)
        except Exception as e:
            self.logger.error("Social fetch failed", error=str(e))
            return []

    # NEW: Additional data source fetchers

    async def _fetch_sec_filings(self, ticker: str) -> dict:
        """Fetch SEC filings (10-K, 10-Q, 8-K, insider trades)."""
        try:
            return await additional_data_sources.get_sec_filings(ticker)
        except Exception as e:
            self.logger.error("SEC filings fetch failed", error=str(e))
            return {"error": str(e)}

    async def _fetch_finviz_data(self, ticker: str) -> dict:
        """Fetch Finviz screener data (analyst targets, ratings, etc.)."""
        try:
            return await additional_data_sources.get_finviz_data(ticker)
        except Exception as e:
            self.logger.error("Finviz fetch failed", error=str(e))
            return {"error": str(e)}

    async def _fetch_fear_greed(self) -> dict:
        """Fetch CNN Fear & Greed Index."""
        try:
            return await additional_data_sources.get_fear_greed_index()
        except Exception as e:
            self.logger.error("Fear & Greed fetch failed", error=str(e))
            return {"error": str(e)}

    async def _fetch_vix_data(self, market: str = "US") -> dict:
        """Fetch VIX volatility data."""
        try:
            return await additional_data_sources.get_vix_data(market=market)
        except Exception as e:
            self.logger.error("VIX fetch failed", error=str(e))
            return {"error": str(e)}

    async def _fetch_options_data(self, ticker: str) -> dict:
        """Fetch options data (put/call ratio, etc.)."""
        try:
            return await additional_data_sources.get_options_overview(ticker)
        except Exception as e:
            self.logger.error("Options fetch failed", error=str(e))
            return {"error": str(e)}

    async def _fetch_economic_indicators(self) -> dict:
        """Fetch economic indicators (Treasury yields, etc.)."""
        try:
            return await additional_data_sources.get_economic_indicators()
        except Exception as e:
            self.logger.error("Economic indicators fetch failed", error=str(e))
            return {}

    async def _fetch_sector_performance(self, market: str = "US") -> dict:
        """Fetch sector ETF performance."""
        try:
            return await additional_data_sources.get_sector_performance(market=market)
        except Exception as e:
            self.logger.error("Sector performance fetch failed", error=str(e))
            return {}

    async def _fetch_additional_news(self, ticker: str, market: str = "US") -> list[dict]:
        """Fetch news from additional sources (MarketWatch, Seeking Alpha, etc.)."""
        try:
            return await additional_data_sources.get_aggregated_news(ticker, limit=30, market=market)
        except Exception as e:
            self.logger.error("Additional news fetch failed", error=str(e))
            return []

    # IMPROVEMENT #3: Earnings history fetcher
    def _fetch_earnings_history_sync(self, ticker: str) -> list[dict]:
        """
        Fetch historical earnings surprise data.

        Returns list of earnings with:
        - date: Earnings date
        - eps_estimate: Expected EPS
        - eps_actual: Actual EPS
        - surprise_pct: Surprise percentage
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)

            # Try to get earnings history
            earnings_hist = stock.earnings_history

            if earnings_hist is None or earnings_hist.empty:
                # Fall back to quarterly earnings
                quarterly = stock.quarterly_earnings
                if quarterly is not None and not quarterly.empty:
                    return [
                        {
                            "date": str(idx) if not hasattr(idx, 'date') else str(idx.date()),
                            "earnings": float(row.get('Earnings', 0)) if 'Earnings' in row else 0,
                            "revenue": float(row.get('Revenue', 0)) if 'Revenue' in row else 0,
                        }
                        for idx, row in quarterly.iterrows()
                    ][-8:]  # Last 8 quarters
                return []

            # Convert earnings history to list
            result = []
            for idx, row in earnings_hist.iterrows():
                try:
                    eps_estimate = row.get('epsEstimate', 0)
                    eps_actual = row.get('epsActual', 0)

                    if eps_estimate and eps_actual:
                        surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
                    else:
                        surprise_pct = 0

                    result.append({
                        "date": str(idx) if not hasattr(idx, 'date') else str(idx.date()),
                        "eps_estimate": float(eps_estimate) if eps_estimate else 0,
                        "eps_actual": float(eps_actual) if eps_actual else 0,
                        "surprise_pct": round(surprise_pct, 2),
                    })
                except Exception:
                    continue

            self.logger.info("Earnings history fetched", ticker=ticker, count=len(result))
            return result[-8:]  # Last 8 quarters

        except Exception as e:
            self.logger.warning("Earnings history fetch failed", ticker=ticker, error=str(e))
            return []


# Create singleton instance
data_aggregator = DataAggregator()
