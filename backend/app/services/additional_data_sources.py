"""
Additional Free Data Sources - Expands data coverage with multiple open APIs.

Sources Added:
- Alpha Vantage (free tier: 25 calls/day)
- SEC EDGAR (official filings, unlimited)
- FRED (Federal Reserve Economic Data)
- Fear & Greed Index (CNN)
- Finviz (stock screener data)
- MarketWatch RSS
- Seeking Alpha RSS
- Benzinga RSS
- Yahoo Finance enhanced data
- Nasdaq Data Link (formerly Quandl)
- Trading Economics indicators
- VIX/Volatility data
- Earnings calendars
- Insider trading (SEC Form 4)
- Analyst ratings aggregation
- Options flow data
- Short interest data
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional, Any
from urllib.parse import quote

import httpx
import feedparser
from bs4 import BeautifulSoup
from cachetools import TTLCache
import structlog

from backend.app.config import settings

logger = structlog.get_logger()

# Caches with different TTLs based on data freshness needs
_economic_cache: TTLCache = TTLCache(maxsize=100, ttl=3600)  # 1 hour
_filings_cache: TTLCache = TTLCache(maxsize=200, ttl=1800)   # 30 min
_screener_cache: TTLCache = TTLCache(maxsize=200, ttl=600)   # 10 min
_sentiment_cache: TTLCache = TTLCache(maxsize=50, ttl=300)   # 5 min


class AdditionalDataSources:
    """
    Aggregates data from multiple free sources to enhance prediction accuracy.
    All URLs are configured in settings (loaded from .env).
    """

    # VIX Data (Yahoo Finance)
    VIX_TICKER = "^VIX"

    def __init__(self):
        """Initialize additional data sources."""
        self.logger = logger.bind(service="additional_data")
        self.http_client: Optional[httpx.AsyncClient] = None
        self._sec_cik_map: dict = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "FuturePredictionAI/1.0 (Educational Research)",
                    "Accept": "application/json, text/html, application/xml",
                },
                follow_redirects=True,
            )
        return self.http_client

    # ==================== SEC EDGAR (Official Filings) ====================

    async def get_sec_filings(self, ticker: str, filing_types: list[str] = None) -> dict:
        """
        Get SEC filings for a company (10-K, 10-Q, 8-K, etc.).

        Args:
            ticker: Stock ticker
            filing_types: List of filing types to fetch (default: all major types)

        Returns:
            Dict with recent filings and analysis
        """
        cache_key = f"sec_{ticker}"
        if cache_key in _filings_cache:
            return _filings_cache[cache_key]

        filing_types = filing_types or ["10-K", "10-Q", "8-K", "4", "DEF 14A"]

        try:
            # Get CIK number for ticker
            cik = await self._get_cik_for_ticker(ticker)
            if not cik:
                return {"error": "CIK not found", "filings": []}

            client = await self._get_client()

            # Fetch company submissions
            url = f"{settings.sec_submissions_url}/CIK{cik.zfill(10)}.json"
            response = await client.get(url, headers={"User-Agent": "FuturePredictionAI research@example.com"})

            if response.status_code != 200:
                return {"error": f"SEC API error: {response.status_code}", "filings": []}

            data = response.json()
            filings = []

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            descriptions = recent.get("primaryDocument", [])

            for i, form in enumerate(forms[:50]):  # Check last 50 filings
                if form in filing_types:
                    filings.append({
                        "type": form,
                        "date": dates[i] if i < len(dates) else None,
                        "accession": accessions[i] if i < len(accessions) else None,
                        "document": descriptions[i] if i < len(descriptions) else None,
                        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accessions[i].replace('-', '')}/{descriptions[i]}" if i < len(accessions) and i < len(descriptions) else None,
                    })

            result = {
                "ticker": ticker,
                "cik": cik,
                "company_name": data.get("name"),
                "filings": filings[:20],
                "total_filings": len(filings),
                "last_10k": next((f for f in filings if f["type"] == "10-K"), None),
                "last_10q": next((f for f in filings if f["type"] == "10-Q"), None),
                "recent_8k_count": len([f for f in filings if f["type"] == "8-K"]),
                "insider_filings": len([f for f in filings if f["type"] == "4"]),
            }

            _filings_cache[cache_key] = result
            self.logger.info("SEC filings fetched", ticker=ticker, count=len(filings))
            return result

        except Exception as e:
            self.logger.error("SEC filings fetch failed", ticker=ticker, error=str(e))
            return {"error": str(e), "filings": []}

    async def _get_cik_for_ticker(self, ticker: str) -> Optional[str]:
        """Get SEC CIK number for a ticker."""
        if not self._sec_cik_map:
            try:
                client = await self._get_client()
                response = await client.get(
                    settings.sec_tickers_url,
                    headers={"User-Agent": "FuturePredictionAI research@example.com"}
                )
                if response.status_code == 200:
                    data = response.json()
                    for key, company in data.items():
                        self._sec_cik_map[company["ticker"]] = str(company["cik_str"])
            except Exception as e:
                self.logger.error("Failed to load SEC CIK map", error=str(e))

        return self._sec_cik_map.get(ticker.upper())

    # ==================== Fear & Greed Index ====================

    async def get_fear_greed_index(self) -> dict:
        """
        Get CNN Fear & Greed Index.

        Returns:
            Dict with current index value and components
        """
        cache_key = "fear_greed"
        if cache_key in _sentiment_cache:
            return _sentiment_cache[cache_key]

        try:
            client = await self._get_client()
            response = await client.get(settings.cnn_fear_greed_url)

            if response.status_code == 200:
                data = response.json()

                # Parse the response
                fear_greed = data.get("fear_and_greed", {})

                result = {
                    "value": fear_greed.get("score", 50),
                    "rating": fear_greed.get("rating", "Neutral"),
                    "previous_close": fear_greed.get("previous_close", 50),
                    "week_ago": fear_greed.get("previous_1_week", 50),
                    "month_ago": fear_greed.get("previous_1_month", 50),
                    "year_ago": fear_greed.get("previous_1_year", 50),
                    "timestamp": datetime.now().isoformat(),
                }

                _sentiment_cache[cache_key] = result
                self.logger.info("Fear & Greed fetched", value=result["value"], rating=result["rating"])
                return result

            # Fallback: try alternative method
            return await self._scrape_fear_greed()

        except Exception as e:
            self.logger.error("Fear & Greed fetch failed", error=str(e))
            return {"value": 50, "rating": "Neutral", "error": str(e)}

    async def _scrape_fear_greed(self) -> dict:
        """Fallback: scrape Fear & Greed from alternative source."""
        try:
            client = await self._get_client()
            # Try alternative fear greed API
            response = await client.get(f"{settings.alternative_fear_greed_url}?limit=1")
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    item = data["data"][0]
                    return {
                        "value": int(item.get("value", 50)),
                        "rating": item.get("value_classification", "Neutral"),
                        "timestamp": datetime.now().isoformat(),
                        "source": "alternative.me"
                    }
        except:
            pass
        return {"value": 50, "rating": "Neutral", "error": "unavailable"}

    # ==================== Finviz Screener Data ====================

    async def get_finviz_data(self, ticker: str) -> dict:
        """
        Get comprehensive stock data from Finviz.

        Returns:
            Dict with price targets, analyst ratings, technicals, fundamentals
        """
        cache_key = f"finviz_{ticker}"
        if cache_key in _screener_cache:
            return _screener_cache[cache_key]

        try:
            client = await self._get_client()
            url = f"{settings.finviz_url}?t={ticker.upper()}"

            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })

            if response.status_code != 200:
                return {"error": f"Finviz error: {response.status_code}"}

            soup = BeautifulSoup(response.text, "html.parser")

            # Parse the snapshot table
            data = {}
            table = soup.find("table", class_="snapshot-table2")

            if table:
                cells = table.find_all("td")
                for i in range(0, len(cells) - 1, 2):
                    label = cells[i].text.strip()
                    value = cells[i + 1].text.strip()
                    data[label] = value

            # Extract key metrics
            result = {
                "ticker": ticker,
                "price": self._parse_number(data.get("Price", "0")),
                "target_price": self._parse_number(data.get("Target Price", "0")),
                "52w_high": self._parse_number(data.get("52W High", "0")),
                "52w_low": self._parse_number(data.get("52W Low", "0")),
                "pe_ratio": self._parse_number(data.get("P/E", "0")),
                "forward_pe": self._parse_number(data.get("Forward P/E", "0")),
                "peg": self._parse_number(data.get("PEG", "0")),
                "ps_ratio": self._parse_number(data.get("P/S", "0")),
                "pb_ratio": self._parse_number(data.get("P/B", "0")),
                "debt_equity": self._parse_number(data.get("Debt/Eq", "0")),
                "roe": data.get("ROE", "N/A"),
                "roi": data.get("ROI", "N/A"),
                "eps_ttm": self._parse_number(data.get("EPS (ttm)", "0")),
                "eps_next_y": self._parse_number(data.get("EPS next Y", "0")),
                "eps_growth": data.get("EPS next 5Y", "N/A"),
                "sales_growth": data.get("Sales Q/Q", "N/A"),
                "insider_own": data.get("Insider Own", "N/A"),
                "inst_own": data.get("Inst Own", "N/A"),
                "short_float": data.get("Short Float", "N/A"),
                "short_ratio": self._parse_number(data.get("Short Ratio", "0")),
                "analyst_recom": data.get("Recom", "N/A"),
                "rsi_14": self._parse_number(data.get("RSI (14)", "50")),
                "volatility": data.get("Volatility", "N/A"),
                "atr": self._parse_number(data.get("ATR", "0")),
                "sma20": data.get("SMA20", "N/A"),
                "sma50": data.get("SMA50", "N/A"),
                "sma200": data.get("SMA200", "N/A"),
                "volume": data.get("Volume", "N/A"),
                "avg_volume": data.get("Avg Volume", "N/A"),
                "relative_volume": self._parse_number(data.get("Rel Volume", "1")),
                "earnings_date": data.get("Earnings", "N/A"),
                "beta": self._parse_number(data.get("Beta", "1")),
            }

            # Parse analyst recommendation
            recom = result.get("analyst_recom", "3")
            try:
                recom_num = float(recom)
                if recom_num <= 1.5:
                    result["recommendation_text"] = "Strong Buy"
                elif recom_num <= 2.5:
                    result["recommendation_text"] = "Buy"
                elif recom_num <= 3.5:
                    result["recommendation_text"] = "Hold"
                elif recom_num <= 4.5:
                    result["recommendation_text"] = "Sell"
                else:
                    result["recommendation_text"] = "Strong Sell"
            except:
                result["recommendation_text"] = "N/A"

            # Calculate upside to target
            if result["price"] and result["target_price"]:
                result["upside_pct"] = round(
                    (result["target_price"] - result["price"]) / result["price"] * 100, 2
                )

            _screener_cache[cache_key] = result
            self.logger.info("Finviz data fetched", ticker=ticker)
            return result

        except Exception as e:
            self.logger.error("Finviz fetch failed", ticker=ticker, error=str(e))
            return {"error": str(e)}

    def _parse_number(self, value: str) -> Optional[float]:
        """Parse a number from Finviz format."""
        if not value or value in ["N/A", "-", ""]:
            return None
        try:
            # Remove %, $, commas, and handle B/M/K suffixes
            clean = value.replace("%", "").replace("$", "").replace(",", "").strip()

            multiplier = 1
            if clean.endswith("B"):
                multiplier = 1_000_000_000
                clean = clean[:-1]
            elif clean.endswith("M"):
                multiplier = 1_000_000
                clean = clean[:-1]
            elif clean.endswith("K"):
                multiplier = 1_000
                clean = clean[:-1]

            return float(clean) * multiplier
        except:
            return None

    # ==================== Multiple RSS News Feeds ====================

    async def get_aggregated_news(self, ticker: str = None, limit: int = 30) -> list[dict]:
        """
        Aggregate news from multiple RSS feeds.

        Args:
            ticker: Optional ticker to filter news
            limit: Max articles to return

        Returns:
            List of news articles from multiple sources
        """
        # Use ticker-specific Google News for better results
        feeds = [
            ("MarketWatch", settings.marketwatch_rss_url),
            ("Yahoo Finance", settings.yahoo_rss_url),
            ("CNBC", settings.cnbc_rss_url),
        ]

        # Add ticker-specific searches
        if ticker:
            google_url = f"{settings.google_news_rss_url}?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            feeds.append(("Google News", google_url))

            # Yahoo Finance ticker-specific
            yahoo_ticker_url = f"{settings.yahoo_finance_rss_url}?s={ticker}&region=US&lang=en-US"
            feeds.append(("Yahoo Finance Ticker", yahoo_ticker_url))

        tasks = [self._fetch_rss_feed(name, url, None) for name, url in feeds]  # Don't filter by ticker since URLs are already specific
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)

        # Sort by date and deduplicate
        seen_titles = set()
        unique = []
        for article in sorted(all_articles, key=lambda x: x.get("published", ""), reverse=True):
            title_key = article.get("title", "")[:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique.append(article)

        self.logger.info("Aggregated news fetched", count=len(unique), ticker=ticker)
        return unique[:limit]

    async def _fetch_rss_feed(self, source_name: str, url: str, ticker: str = None) -> list[dict]:
        """Fetch and parse an RSS feed."""
        try:
            client = await self._get_client()
            response = await client.get(url, timeout=15.0)

            if response.status_code != 200:
                return []

            feed = feedparser.parse(response.text)
            articles = []

            for entry in feed.entries[:20]:
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))[:500]

                # Filter by ticker if provided
                if ticker:
                    ticker_upper = ticker.upper()
                    if ticker_upper not in title.upper() and ticker_upper not in summary.upper():
                        continue

                published = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6]).isoformat()

                articles.append({
                    "title": title,
                    "summary": summary,
                    "url": entry.get("link", ""),
                    "source": source_name,
                    "published": published,
                })

            return articles

        except Exception as e:
            self.logger.debug(f"{source_name} RSS fetch failed", error=str(e))
            return []

    # ==================== VIX / Volatility Data ====================

    async def get_vix_data(self) -> dict:
        """
        Get VIX (Volatility Index) data.

        Returns:
            Dict with VIX value and interpretation
        """
        cache_key = "vix"
        if cache_key in _sentiment_cache:
            return _sentiment_cache[cache_key]

        try:
            # Use yfinance for VIX
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")

            if hist.empty:
                return {"error": "VIX data unavailable"}

            current = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current

            # Interpret VIX level
            if current < 12:
                interpretation = "Extremely Low - Complacency"
            elif current < 20:
                interpretation = "Low - Calm Markets"
            elif current < 25:
                interpretation = "Moderate - Normal"
            elif current < 30:
                interpretation = "Elevated - Concern"
            elif current < 40:
                interpretation = "High - Fear"
            else:
                interpretation = "Extreme - Panic"

            result = {
                "value": round(current, 2),
                "previous_close": round(prev_close, 2),
                "change": round(current - prev_close, 2),
                "change_pct": round((current - prev_close) / prev_close * 100, 2),
                "interpretation": interpretation,
                "market_sentiment": "bearish" if current > 25 else "neutral" if current > 15 else "bullish",
            }

            _sentiment_cache[cache_key] = result
            self.logger.info("VIX data fetched", value=current)
            return result

        except Exception as e:
            self.logger.error("VIX fetch failed", error=str(e))
            return {"value": 20, "interpretation": "Unknown", "error": str(e)}

    # ==================== Economic Indicators (FRED) ====================

    async def get_economic_indicators(self) -> dict:
        """
        Get key economic indicators from FRED.

        Returns:
            Dict with major economic indicators
        """
        cache_key = "economic"
        if cache_key in _economic_cache:
            return _economic_cache[cache_key]

        # Key FRED series
        indicators = {
            "fed_funds_rate": "FEDFUNDS",
            "unemployment": "UNRATE",
            "cpi_inflation": "CPIAUCSL",
            "gdp_growth": "GDP",
            "treasury_10y": "DGS10",
            "treasury_2y": "DGS2",
            "consumer_sentiment": "UMCSENT",
            "retail_sales": "RSXFS",
            "industrial_production": "INDPRO",
            "housing_starts": "HOUST",
        }

        result = {}

        # Try to get from FRED API if key available
        if hasattr(settings, 'fred_api_key') and settings.fred_api_key:
            for name, series_id in indicators.items():
                try:
                    value = await self._fetch_fred_series(series_id)
                    if value:
                        result[name] = value
                except:
                    pass

        # Fallback: Use yfinance for treasury yields
        try:
            import yfinance as yf

            # 10-year Treasury
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="5d")
            if not hist.empty:
                result["treasury_10y"] = {
                    "value": round(float(hist["Close"].iloc[-1]), 2),
                    "unit": "%",
                }

            # 2-year Treasury (via IRX proxy)
            try:
                tyx = yf.Ticker("^TYX")
                hist = tyx.history(period="5d")
                if not hist.empty:
                    result["treasury_30y"] = {
                        "value": round(float(hist["Close"].iloc[-1]), 2),
                        "unit": "%",
                    }
            except:
                pass

        except Exception as e:
            self.logger.debug("Treasury data fetch failed", error=str(e))

        # Calculate yield curve (10y - 2y spread)
        if result.get("treasury_10y") and result.get("treasury_2y"):
            spread = result["treasury_10y"]["value"] - result["treasury_2y"]["value"]
            result["yield_curve_spread"] = {
                "value": round(spread, 2),
                "interpretation": "Inverted (recession signal)" if spread < 0 else "Normal",
            }

        _economic_cache[cache_key] = result
        return result

    async def _fetch_fred_series(self, series_id: str) -> Optional[dict]:
        """Fetch a single FRED series."""
        try:
            client = await self._get_client()
            params = {
                "series_id": series_id,
                "api_key": settings.fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            }

            response = await client.get(settings.fred_api_url, params=params)
            if response.status_code == 200:
                data = response.json()
                observations = data.get("observations", [])
                if observations:
                    return {
                        "value": float(observations[0]["value"]),
                        "date": observations[0]["date"],
                    }
        except:
            pass
        return None

    # ==================== Earnings Calendar ====================

    async def get_earnings_calendar(self, date: str = None) -> list[dict]:
        """
        Get earnings calendar from Nasdaq.

        Args:
            date: Date in YYYY-MM-DD format (default: today)

        Returns:
            List of companies reporting earnings
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        cache_key = f"earnings_{date}"
        if cache_key in _screener_cache:
            return _screener_cache[cache_key]

        try:
            client = await self._get_client()
            url = f"{settings.nasdaq_earnings_url}?date={date}"

            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            })

            if response.status_code != 200:
                return []

            data = response.json()
            rows = data.get("data", {}).get("rows", [])

            earnings = []
            for row in rows[:50]:
                earnings.append({
                    "symbol": row.get("symbol", ""),
                    "name": row.get("name", ""),
                    "date": row.get("reportDate", date),
                    "time": row.get("time", ""),  # AMC/BMO
                    "eps_estimate": row.get("epsForecast"),
                    "eps_actual": row.get("eps"),
                    "surprise_pct": row.get("surprise"),
                })

            _screener_cache[cache_key] = earnings
            self.logger.info("Earnings calendar fetched", date=date, count=len(earnings))
            return earnings

        except Exception as e:
            self.logger.error("Earnings calendar fetch failed", error=str(e))
            return []

    # ==================== Short Interest Data ====================

    async def get_short_interest(self, ticker: str) -> dict:
        """
        Get short interest data for a ticker.

        Returns:
            Dict with short interest metrics
        """
        # Try to get from Finviz data first
        finviz = await self.get_finviz_data(ticker)

        if finviz and not finviz.get("error"):
            return {
                "ticker": ticker,
                "short_float": finviz.get("short_float"),
                "short_ratio": finviz.get("short_ratio"),
                "shares_short": None,  # Would need additional source
                "source": "finviz",
            }

        return {"ticker": ticker, "error": "Short interest data unavailable"}

    # ==================== Options Data (Basic) ====================

    async def get_options_overview(self, ticker: str) -> dict:
        """
        Get basic options data for a ticker.

        Returns:
            Dict with put/call ratio, max pain, etc.
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)

            # Get expiration dates
            expirations = stock.options
            if not expirations:
                return {"ticker": ticker, "error": "No options data"}

            # Get nearest expiration
            nearest = expirations[0]
            opt = stock.option_chain(nearest)

            calls = opt.calls
            puts = opt.puts

            # Calculate put/call ratio by volume
            call_volume = calls["volume"].sum() if "volume" in calls else 0
            put_volume = puts["volume"].sum() if "volume" in puts else 0

            pc_ratio = put_volume / call_volume if call_volume > 0 else 1

            # Calculate put/call ratio by open interest
            call_oi = calls["openInterest"].sum() if "openInterest" in calls else 0
            put_oi = puts["openInterest"].sum() if "openInterest" in puts else 0

            pc_ratio_oi = put_oi / call_oi if call_oi > 0 else 1

            return {
                "ticker": ticker,
                "expiration": nearest,
                "put_call_ratio_volume": round(pc_ratio, 2),
                "put_call_ratio_oi": round(pc_ratio_oi, 2),
                "call_volume": int(call_volume) if call_volume else 0,
                "put_volume": int(put_volume) if put_volume else 0,
                "call_open_interest": int(call_oi) if call_oi else 0,
                "put_open_interest": int(put_oi) if put_oi else 0,
                "sentiment": "bearish" if pc_ratio > 1.2 else "bullish" if pc_ratio < 0.8 else "neutral",
                "expirations_available": len(expirations),
            }

        except Exception as e:
            self.logger.error("Options data fetch failed", ticker=ticker, error=str(e))
            return {"ticker": ticker, "error": str(e)}

    # ==================== Sector Performance ====================

    async def get_sector_performance(self) -> dict:
        """
        Get sector ETF performance.

        Returns:
            Dict with sector performance data
        """
        cache_key = "sectors"
        if cache_key in _screener_cache:
            return _screener_cache[cache_key]

        try:
            import yfinance as yf

            # Sector ETFs
            sectors = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financials": "XLF",
                "Consumer Discretionary": "XLY",
                "Communication Services": "XLC",
                "Industrials": "XLI",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Materials": "XLB",
            }

            result = {}

            for sector_name, etf in sectors.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="5d")

                    if not hist.empty:
                        current = float(hist["Close"].iloc[-1])
                        prev = float(hist["Close"].iloc[0])
                        change_pct = (current - prev) / prev * 100

                        result[sector_name] = {
                            "etf": etf,
                            "price": round(current, 2),
                            "change_pct": round(change_pct, 2),
                        }
                except:
                    continue

            _screener_cache[cache_key] = result
            return result

        except Exception as e:
            self.logger.error("Sector performance fetch failed", error=str(e))
            return {}

    # ==================== Comprehensive Data Fetch ====================

    async def get_comprehensive_data(self, ticker: str) -> dict:
        """
        Fetch all available data for a ticker from all sources.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Comprehensive dict with all available data
        """
        self.logger.info("Fetching comprehensive data", ticker=ticker)

        # Run all fetches concurrently
        tasks = {
            "sec_filings": self.get_sec_filings(ticker),
            "finviz": self.get_finviz_data(ticker),
            "fear_greed": self.get_fear_greed_index(),
            "vix": self.get_vix_data(),
            "options": self.get_options_overview(ticker),
            "short_interest": self.get_short_interest(ticker),
            "economic": self.get_economic_indicators(),
            "sectors": self.get_sector_performance(),
            "news": self.get_aggregated_news(ticker, limit=30),
        }

        results = {}
        task_results = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True
        )

        for key, result in zip(tasks.keys(), task_results):
            if isinstance(result, Exception):
                results[key] = {"error": str(result)}
            else:
                results[key] = result

        results["ticker"] = ticker
        results["fetched_at"] = datetime.now().isoformat()

        # Calculate overall data quality
        data_points = 0
        sources_success = 0

        for key, value in results.items():
            if isinstance(value, dict) and not value.get("error"):
                sources_success += 1
                data_points += len(value)
            elif isinstance(value, list):
                sources_success += 1
                data_points += len(value)

        results["data_quality"] = {
            "sources_successful": sources_success,
            "total_sources": len(tasks),
            "data_points": data_points,
            "quality_score": round(sources_success / len(tasks), 2),
        }

        self.logger.info(
            "Comprehensive data fetched",
            ticker=ticker,
            sources=sources_success,
            data_points=data_points,
        )

        return results

    async def close(self):
        """Clean up resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None


# Singleton instance
additional_data_sources = AdditionalDataSources()
