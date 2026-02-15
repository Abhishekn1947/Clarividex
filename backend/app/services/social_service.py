"""
Social Media Sentiment Service - Aggregates sentiment from social platforms.

Sources:
- Reddit (via PRAW or public RSS)
- StockTwits (public API)
- Google Trends (pytrends)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote

import httpx
import feedparser
from cachetools import TTLCache
import structlog

try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False

try:
    import praw
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False

from backend.app.config import settings
from backend.app.models.schemas import SocialSentiment
from backend.app.services.sentiment_service import sentiment_service
from backend.app.services.market_config import get_market_config

logger = structlog.get_logger()

# Cache social sentiment for 10 minutes
_social_cache: TTLCache = TTLCache(maxsize=200, ttl=600)


class SocialMediaService:
    """Service for aggregating social media sentiment."""

    # Subreddits to monitor
    STOCK_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "stockmarket"]

    def __init__(self):
        """Initialize social media service."""
        self.logger = logger.bind(service="social")
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "FuturePredictionAI/1.0"},
        )

        # Initialize Reddit client if credentials available
        self.reddit_client = None
        if HAS_PRAW and settings.has_reddit_credentials:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=settings.reddit_client_id,
                    client_secret=settings.reddit_client_secret,
                    user_agent=settings.reddit_user_agent,
                )
                self.logger.info("Reddit client initialized")
            except Exception as e:
                self.logger.warning("Failed to initialize Reddit client", error=str(e))

        # Initialize PyTrends
        self.pytrends = None
        if HAS_PYTRENDS:
            try:
                self.pytrends = TrendReq(hl="en-US", tz=360)
                self.logger.info("PyTrends initialized")
            except Exception as e:
                self.logger.warning("Failed to initialize PyTrends", error=str(e))

    async def get_social_sentiment(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        market: str = "US",
    ) -> list[SocialSentiment]:
        """
        Get social sentiment from multiple platforms.

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for search
            market: Market context ("US" or "IN")

        Returns:
            List of SocialSentiment objects from each platform
        """
        cache_key = f"social_{ticker}_{market}"

        if cache_key in _social_cache:
            self.logger.debug("Social cache hit", ticker=ticker)
            return _social_cache[cache_key]

        self.logger.info("Fetching social sentiment", ticker=ticker, market=market)

        # Use market-specific subreddits
        market_config = get_market_config(market)
        subreddits = market_config["subreddits"]

        # Fetch from multiple sources concurrently
        # For Indian stocks, StockTwits may have limited coverage but still try
        tasks = [
            self._fetch_stocktwits(ticker),
            self._fetch_reddit_sentiment(ticker, company_name, subreddits=subreddits),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        sentiments = []
        for result in results:
            if isinstance(result, SocialSentiment):
                sentiments.append(result)
            elif isinstance(result, Exception):
                self.logger.warning("Social fetch failed", error=str(result))

        # Add Google Trends data
        trends_data = await self._fetch_google_trends(ticker, company_name)
        if trends_data:
            sentiments.append(trends_data)

        # Cache results
        _social_cache[cache_key] = sentiments

        return sentiments

    async def _fetch_stocktwits(self, ticker: str) -> Optional[SocialSentiment]:
        """
        Fetch sentiment from StockTwits.

        Args:
            ticker: Stock ticker symbol

        Returns:
            SocialSentiment object or None
        """
        try:
            url = f"{settings.stocktwits_api_url}/{ticker.upper()}.json"
            self.logger.debug("Fetching StockTwits", ticker=ticker)

            response = await self.http_client.get(url)

            if response.status_code == 404:
                self.logger.debug("Ticker not found on StockTwits", ticker=ticker)
                return None

            response.raise_for_status()
            data = response.json()

            messages = data.get("messages", [])
            if not messages:
                return None

            # Analyze sentiment
            bullish_count = 0
            bearish_count = 0
            sample_posts = []

            for msg in messages[:50]:  # Analyze up to 50 messages
                body = msg.get("body", "")

                # StockTwits has its own sentiment labels
                sentiment = msg.get("entities", {}).get("sentiment", {})
                if sentiment:
                    if sentiment.get("basic") == "Bullish":
                        bullish_count += 1
                    elif sentiment.get("basic") == "Bearish":
                        bearish_count += 1
                else:
                    # Analyze ourselves if no label
                    score = sentiment_service.analyze_text(body)
                    if score > 0.1:
                        bullish_count += 1
                    elif score < -0.1:
                        bearish_count += 1

                if len(sample_posts) < 5:
                    sample_posts.append(body[:200])

            total = bullish_count + bearish_count
            bullish_pct = (bullish_count / total * 100) if total > 0 else 50

            # Calculate sentiment score
            sentiment_score = (bullish_pct - 50) / 50  # Normalize to [-1, 1]

            return SocialSentiment(
                platform="StockTwits",
                mentions_count=len(messages),
                sentiment_score=round(sentiment_score, 3),
                bullish_percentage=round(bullish_pct, 1),
                trending=len(messages) > 30,
                sample_posts=sample_posts,
            )

        except Exception as e:
            self.logger.error("StockTwits fetch failed", ticker=ticker, error=str(e))
            return None

    async def _fetch_reddit_sentiment(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        subreddits: list[str] = None,
    ) -> Optional[SocialSentiment]:
        """
        Fetch sentiment from Reddit.

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for search
            subreddits: List of subreddits to search

        Returns:
            SocialSentiment object or None
        """
        try:
            # Use PRAW if available, otherwise RSS
            if self.reddit_client:
                return await self._fetch_reddit_praw(ticker, company_name, subreddits=subreddits)
            else:
                return await self._fetch_reddit_rss(ticker, company_name, subreddits=subreddits)

        except Exception as e:
            self.logger.error("Reddit fetch failed", ticker=ticker, error=str(e))
            return None

    async def _fetch_reddit_praw(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        subreddits: list[str] = None,
    ) -> Optional[SocialSentiment]:
        """Fetch Reddit data using PRAW (authenticated)."""
        try:
            posts_text = []
            total_score = 0
            target_subreddits = subreddits or self.STOCK_SUBREDDITS

            for subreddit_name in target_subreddits[:2]:  # Limit to avoid rate limits
                subreddit = self.reddit_client.subreddit(subreddit_name)

                # Search for ticker mentions
                search_query = f"${ticker}" if len(ticker) <= 4 else ticker
                for post in subreddit.search(search_query, sort="new", time_filter="week", limit=10):
                    text = f"{post.title} {post.selftext}"
                    posts_text.append(text)
                    total_score += post.score

            if not posts_text:
                return None

            # Analyze sentiment
            analysis = sentiment_service.analyze_multiple(posts_text)

            return SocialSentiment(
                platform="Reddit",
                mentions_count=len(posts_text),
                sentiment_score=analysis["average_score"],
                bullish_percentage=analysis["bullish_percentage"],
                trending=len(posts_text) > 15,
                sample_posts=[p[:200] for p in posts_text[:5]],
            )

        except Exception as e:
            self.logger.error("Reddit PRAW fetch failed", error=str(e))
            return None

    async def _fetch_reddit_rss(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        subreddits: list[str] = None,
    ) -> Optional[SocialSentiment]:
        """Fetch Reddit data using public RSS feeds (no auth)."""
        try:
            posts_text = []
            # For Indian tickers, strip .NS suffix for search
            search_ticker = ticker.replace(".NS", "").replace(".BO", "")
            search_query = f"${search_ticker}" if len(search_ticker) <= 4 else search_ticker
            target_subreddits = subreddits or self.STOCK_SUBREDDITS

            # Try multiple subreddits
            for subreddit in target_subreddits[:2]:
                url = f"{settings.reddit_rss_url}/r/{subreddit}/search.rss?q={quote(search_query)}&sort=new&t=week"

                try:
                    response = await self.http_client.get(url)
                    if response.status_code == 200:
                        feed = feedparser.parse(response.text)
                        for entry in feed.entries[:10]:
                            text = f"{entry.title}"
                            posts_text.append(text)
                except Exception:
                    continue

            if not posts_text:
                return None

            # Analyze sentiment
            analysis = sentiment_service.analyze_multiple(posts_text)

            return SocialSentiment(
                platform="Reddit",
                mentions_count=len(posts_text),
                sentiment_score=analysis["average_score"],
                bullish_percentage=analysis["bullish_percentage"],
                trending=len(posts_text) > 10,
                sample_posts=[p[:200] for p in posts_text[:5]],
            )

        except Exception as e:
            self.logger.error("Reddit RSS fetch failed", error=str(e))
            return None

    async def _fetch_google_trends(
        self,
        ticker: str,
        company_name: Optional[str] = None,
    ) -> Optional[SocialSentiment]:
        """
        Fetch Google Trends data.

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for search

        Returns:
            SocialSentiment object with trends data
        """
        if not self.pytrends:
            return None

        try:
            # Run in thread pool since pytrends is synchronous
            loop = asyncio.get_event_loop()

            def fetch_trends():
                search_terms = [f"{ticker} stock"]
                if company_name:
                    search_terms.append(company_name)

                self.pytrends.build_payload(
                    search_terms[:1],  # Use primary term only
                    cat=0,
                    timeframe="now 7-d",
                    geo="US",
                )

                interest = self.pytrends.interest_over_time()
                return interest

            interest_df = await loop.run_in_executor(None, fetch_trends)

            if interest_df is None or interest_df.empty:
                return None

            # Calculate trend direction
            recent_values = interest_df.iloc[-7:, 0].values
            trend_direction = "stable"
            if len(recent_values) >= 2:
                if recent_values[-1] > recent_values[0] * 1.2:
                    trend_direction = "increasing"
                elif recent_values[-1] < recent_values[0] * 0.8:
                    trend_direction = "decreasing"

            # Use search interest as a proxy for sentiment (high interest can be bullish)
            avg_interest = float(recent_values.mean())
            sentiment_score = (avg_interest - 50) / 50  # Normalize

            return SocialSentiment(
                platform="Google Trends",
                mentions_count=int(avg_interest),
                sentiment_score=round(min(1, max(-1, sentiment_score)), 3),
                bullish_percentage=round(avg_interest, 1),
                trending=avg_interest > 70,
                sample_posts=[f"Search interest: {trend_direction}"],
            )

        except Exception as e:
            self.logger.debug("Google Trends fetch failed", error=str(e))
            return None

    def calculate_aggregate_sentiment(
        self,
        social_data: list[SocialSentiment],
    ) -> dict:
        """
        Calculate aggregate sentiment across all platforms.

        Args:
            social_data: List of SocialSentiment from different platforms

        Returns:
            Dict with aggregate metrics
        """
        if not social_data:
            return {
                "overall_score": 0,
                "overall_bullish_pct": 50,
                "total_mentions": 0,
                "platforms_analyzed": 0,
                "is_trending": False,
            }

        total_mentions = sum(s.mentions_count for s in social_data)
        weighted_sentiment = 0
        weighted_bullish = 0

        for s in social_data:
            weight = s.mentions_count / total_mentions if total_mentions > 0 else 1
            weighted_sentiment += s.sentiment_score * weight
            weighted_bullish += s.bullish_percentage * weight

        return {
            "overall_score": round(weighted_sentiment, 3),
            "overall_bullish_pct": round(weighted_bullish, 1),
            "total_mentions": total_mentions,
            "platforms_analyzed": len(social_data),
            "is_trending": any(s.trending for s in social_data),
        }

    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


# Create singleton instance
social_service = SocialMediaService()
