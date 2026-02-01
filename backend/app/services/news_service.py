"""
News Service - Aggregates news from multiple free sources.

Sources:
- Finnhub (if API key provided)
- Google News RSS (free, no API key)
- Yahoo Finance News (scraping)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote

import feedparser
import httpx
from bs4 import BeautifulSoup
from cachetools import TTLCache
import structlog

from backend.app.config import settings
from backend.app.models.schemas import NewsArticle
from backend.app.services.sentiment_service import sentiment_service

logger = structlog.get_logger()

# Cache news for 5 minutes
_news_cache: TTLCache = TTLCache(maxsize=200, ttl=300)


class NewsService:
    """Service for fetching and aggregating news from multiple sources."""

    def __init__(self):
        """Initialize news service."""
        self.logger = logger.bind(service="news")
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            },
        )

    # Industry keywords for broader news coverage
    INDUSTRY_KEYWORDS = {
        "BA": ["airplane crash", "aviation safety", "FAA", "airline accident", "Boeing 737", "Boeing 787"],
        "LMT": ["defense contract", "military spending", "Pentagon", "NATO", "aerospace defense"],
        "TSLA": ["electric vehicle", "EV market", "autonomous driving", "battery technology", "charging network"],
        "AAPL": ["iPhone", "App Store", "Apple lawsuit", "tech regulation", "smartphone market"],
        "NVDA": ["AI chips", "GPU shortage", "data center", "artificial intelligence", "semiconductor"],
        "GOOGL": ["search engine", "antitrust", "digital advertising", "AI regulation", "tech monopoly"],
        "META": ["social media", "metaverse", "Facebook", "Instagram", "digital privacy"],
        "AMZN": ["e-commerce", "AWS", "cloud computing", "retail", "logistics"],
        "JPM": ["banking", "interest rates", "Federal Reserve", "financial regulation", "credit"],
        "XOM": ["oil prices", "OPEC", "crude oil", "energy prices", "gas prices", "petroleum"],
        "PFE": ["vaccine", "FDA approval", "drug trial", "pharmaceutical", "healthcare"],
        "MRNA": ["vaccine", "mRNA technology", "clinical trial", "biotech", "COVID"],
    }

    async def get_news_for_ticker(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        limit: int = 50,
    ) -> list[NewsArticle]:
        """
        Get news articles for a stock ticker from multiple sources.

        Includes:
        - Direct stock/company news
        - Broader industry news that could affect the stock
        - Related events (e.g., crashes, recalls, lawsuits)

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for better search results
            limit: Maximum number of articles to return

        Returns:
            List of NewsArticle objects sorted by date
        """
        cache_key = f"news_{ticker}"

        # Check cache
        if cache_key in _news_cache:
            self.logger.debug("News cache hit", ticker=ticker)
            return _news_cache[cache_key][:limit]

        self.logger.info("Fetching news", ticker=ticker, company_name=company_name)

        # Fetch from multiple sources concurrently
        tasks = [
            self._fetch_google_news(ticker, company_name),
            self._fetch_finnhub_news(ticker),
            self._fetch_broader_news(ticker, company_name),  # New: broader impact news
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all articles
        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                self.logger.warning("News fetch failed", error=str(result))

        # Deduplicate by title similarity
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            # Simple dedup by normalized title
            normalized = article.title.lower()[:50]
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique_articles.append(article)

        # Sort by date (newest first)
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)

        # Analyze sentiment for each article
        for article in unique_articles:
            if article.sentiment_score == 0:
                text = f"{article.title} {article.summary or ''}"
                article.sentiment_score = sentiment_service.analyze_text(text)

        # Cache results
        _news_cache[cache_key] = unique_articles

        self.logger.info(
            "News fetched",
            ticker=ticker,
            article_count=len(unique_articles),
        )

        return unique_articles[:limit]

    async def _fetch_google_news(
        self,
        ticker: str,
        company_name: Optional[str] = None,
    ) -> list[NewsArticle]:
        """
        Fetch news from Google News RSS.

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for search

        Returns:
            List of news articles
        """
        try:
            # Build search query
            search_terms = [f"{ticker} stock"]
            if company_name:
                search_terms.append(company_name)

            query = " OR ".join(search_terms)
            url = f"{settings.google_news_rss_url}?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"

            self.logger.debug("Fetching Google News", url=url)

            response = await self.http_client.get(url)
            response.raise_for_status()

            # Parse RSS feed
            feed = feedparser.parse(response.text)
            articles = []

            for entry in feed.entries[:25]:
                try:
                    # Parse published date
                    published = datetime.now()
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])

                    # Extract source from title (Google News format: "Title - Source")
                    title = entry.title
                    source = "Google News"
                    if " - " in title:
                        parts = title.rsplit(" - ", 1)
                        title = parts[0]
                        source = parts[1] if len(parts) > 1 else source

                    article = NewsArticle(
                        title=title,
                        source=source,
                        url=entry.link,
                        published_at=published,
                        summary=entry.get("summary", "")[:500] if entry.get("summary") else None,
                        sentiment_score=0,  # Will be analyzed later
                        relevance_score=0.7,  # Default relevance
                    )
                    articles.append(article)

                except Exception as e:
                    self.logger.debug("Failed to parse news entry", error=str(e))
                    continue

            self.logger.debug("Google News fetched", count=len(articles))
            return articles

        except Exception as e:
            self.logger.error("Google News fetch failed", error=str(e))
            return []

    async def _fetch_finnhub_news(self, ticker: str) -> list[NewsArticle]:
        """
        Fetch news from Finnhub API.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of news articles
        """
        if not settings.has_finnhub_key:
            self.logger.debug("Finnhub API key not configured, skipping")
            return []

        try:
            # Get news from last 7 days
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)

            url = f"{settings.finnhub_api_url}/company-news"
            params = {
                "symbol": ticker.upper(),
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "token": settings.finnhub_api_key,
            }

            self.logger.debug("Fetching Finnhub news", ticker=ticker)

            response = await self.http_client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            articles = []

            for item in data[:25]:
                try:
                    article = NewsArticle(
                        title=item.get("headline", ""),
                        source=item.get("source", "Finnhub"),
                        url=item.get("url"),
                        published_at=datetime.fromtimestamp(item.get("datetime", 0)),
                        summary=item.get("summary", "")[:500],
                        sentiment_score=0,
                        relevance_score=0.85,  # Finnhub news is usually relevant
                    )
                    articles.append(article)

                except Exception as e:
                    self.logger.debug("Failed to parse Finnhub entry", error=str(e))
                    continue

            self.logger.debug("Finnhub news fetched", count=len(articles))
            return articles

        except Exception as e:
            self.logger.error("Finnhub news fetch failed", error=str(e))
            return []

    async def _fetch_broader_news(
        self,
        ticker: str,
        company_name: Optional[str] = None,
    ) -> list[NewsArticle]:
        """
        Fetch broader news that could impact the stock.

        This includes:
        - Industry-specific news
        - Events like crashes, recalls, lawsuits
        - Regulatory news affecting the sector

        Args:
            ticker: Stock ticker symbol
            company_name: Company name

        Returns:
            List of news articles with impact relevance
        """
        try:
            search_terms = []

            # Add industry-specific keywords if available
            ticker_upper = ticker.upper()
            if ticker_upper in self.INDUSTRY_KEYWORDS:
                keywords = self.INDUSTRY_KEYWORDS[ticker_upper]
                for keyword in keywords[:3]:  # Top 3 keywords
                    if company_name:
                        search_terms.append(f"{company_name} {keyword}")
                    else:
                        search_terms.append(f"{ticker} {keyword}")

            # Add generic impact news for the company
            if company_name:
                search_terms.extend([
                    f"{company_name} lawsuit",
                    f"{company_name} recall",
                    f"{company_name} accident",
                    f"{company_name} investigation",
                    f"{company_name} earnings",
                ])

            if not search_terms:
                return []

            # Fetch news for each search term
            articles = []
            for term in search_terms[:8]:  # Limit to 8 searches
                query = quote(term)
                url = f"{settings.google_news_rss_url}?q={query}&hl=en-US&gl=US&ceid=US:en"

                try:
                    response = await self.http_client.get(url)
                    response.raise_for_status()
                    feed = feedparser.parse(response.text)

                    for entry in feed.entries[:5]:  # Top 5 per term
                        try:
                            published = datetime.now()
                            if hasattr(entry, "published_parsed") and entry.published_parsed:
                                published = datetime(*entry.published_parsed[:6])

                            title = entry.title
                            source = "Google News"
                            if " - " in title:
                                parts = title.rsplit(" - ", 1)
                                title = parts[0]
                                source = parts[1] if len(parts) > 1 else source

                            # Calculate relevance based on keyword match
                            relevance = 0.6
                            title_lower = title.lower()
                            if company_name and company_name.lower() in title_lower:
                                relevance = 0.8

                            # Check for high-impact keywords
                            impact_keywords = ["crash", "accident", "recall", "lawsuit", "investigation", "ban", "fine"]
                            for kw in impact_keywords:
                                if kw in title_lower:
                                    relevance = 0.9
                                    break

                            article = NewsArticle(
                                title=title,
                                source=source,
                                url=entry.link,
                                published_at=published,
                                summary=entry.get("summary", "")[:500] if entry.get("summary") else None,
                                sentiment_score=0,
                                relevance_score=relevance,
                            )
                            articles.append(article)

                        except Exception:
                            continue

                except Exception:
                    continue

            self.logger.debug("Broader news fetched", count=len(articles))
            return articles

        except Exception as e:
            self.logger.error("Broader news fetch failed", error=str(e))
            return []

    async def get_general_market_news(self, limit: int = 10) -> list[NewsArticle]:
        """
        Get general market/financial news.

        Args:
            limit: Maximum articles to return

        Returns:
            List of market news articles
        """
        cache_key = "news_market_general"

        if cache_key in _news_cache:
            return _news_cache[cache_key][:limit]

        query = "stock market OR S&P 500 OR Federal Reserve OR inflation"
        url = f"{settings.google_news_rss_url}?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"

        try:
            response = await self.http_client.get(url)
            response.raise_for_status()

            feed = feedparser.parse(response.text)
            articles = []

            for entry in feed.entries[:limit]:
                try:
                    published = datetime.now()
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])

                    title = entry.title
                    source = "Google News"
                    if " - " in title:
                        parts = title.rsplit(" - ", 1)
                        title = parts[0]
                        source = parts[1] if len(parts) > 1 else source

                    article = NewsArticle(
                        title=title,
                        source=source,
                        url=entry.link,
                        published_at=published,
                        summary=entry.get("summary", "")[:500] if entry.get("summary") else None,
                        sentiment_score=sentiment_service.analyze_text(title),
                        relevance_score=0.6,
                    )
                    articles.append(article)

                except Exception:
                    continue

            _news_cache[cache_key] = articles
            return articles

        except Exception as e:
            self.logger.error("Market news fetch failed", error=str(e))
            return []

    def calculate_news_sentiment(self, articles: list[NewsArticle]) -> dict:
        """
        Calculate aggregate news sentiment.

        Args:
            articles: List of news articles

        Returns:
            Dict with sentiment metrics
        """
        if not articles:
            return {
                "overall_score": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "weighted_score": 0,
            }

        bullish = 0
        bearish = 0
        neutral = 0
        weighted_sum = 0
        total_weight = 0

        for article in articles:
            score = article.sentiment_score
            weight = article.relevance_score

            if score > 0.1:
                bullish += 1
            elif score < -0.1:
                bearish += 1
            else:
                neutral += 1

            weighted_sum += score * weight
            total_weight += weight

        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0

        return {
            "overall_score": round(weighted_score, 3),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "weighted_score": round(weighted_score, 3),
            "article_count": len(articles),
        }

    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


# Create singleton instance
news_service = NewsService()
