"""
Market Data Service - Smart market data fetching with caching.

Uses yfinance (free, unlimited) as the primary source.
Includes intelligent ticker extraction and company name recognition.
"""

import re
from datetime import datetime, timedelta
from typing import Optional
from functools import lru_cache

import pandas as pd
import yfinance as yf
from cachetools import TTLCache
import structlog

from backend.app.models.schemas import StockQuote, CompanyInfo, TickerExtractionResult, TickerSuggestion, InstrumentType
from backend.app.services.instrument_detector import instrument_detector

logger = structlog.get_logger()

# Caches with TTL
_quote_cache: TTLCache = TTLCache(maxsize=500, ttl=60)  # 1 minute for quotes
_info_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)  # 1 hour for company info
_history_cache: TTLCache = TTLCache(maxsize=100, ttl=300)  # 5 min for historical data


class MarketDataService:
    """
    Smart market data service with intelligent ticker recognition.

    Features:
    - Company name to ticker mapping
    - Fuzzy ticker extraction from natural language
    - Multi-level caching for performance
    - Automatic retry on failures
    """

    # Company name to ticker mapping (common companies)
    COMPANY_TO_TICKER = {
        # Tech Giants
        "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
        "amazon": "AMZN", "nvidia": "NVDA", "meta": "META", "facebook": "META",
        "tesla": "TSLA", "netflix": "NFLX", "adobe": "ADBE", "salesforce": "CRM",
        "intel": "INTC", "amd": "AMD", "cisco": "CSCO", "oracle": "ORCL",
        "ibm": "IBM", "qualcomm": "QCOM", "broadcom": "AVGO", "texas instruments": "TXN",
        "paypal": "PYPL", "shopify": "SHOP", "uber": "UBER", "airbnb": "ABNB",
        "palantir": "PLTR", "snowflake": "SNOW", "crowdstrike": "CRWD", "datadog": "DDOG",
        "zoom": "ZM", "docusign": "DOCU", "twilio": "TWLO", "okta": "OKTA",
        "coinbase": "COIN", "robinhood": "HOOD", "rivian": "RIVN", "lucid": "LCID",

        # Defense/Aerospace
        "kratos": "KTOS", "kratos defense": "KTOS", "kratos security": "KTOS",
        "lockheed": "LMT", "lockheed martin": "LMT", "northrop": "NOC", "northrop grumman": "NOC",
        "raytheon": "RTX", "boeing": "BA", "general dynamics": "GD", "l3harris": "LHX",

        # Finance
        "jpmorgan": "JPM", "jp morgan": "JPM", "chase": "JPM",
        "bank of america": "BAC", "wells fargo": "WFC", "citigroup": "C", "citi": "C",
        "goldman sachs": "GS", "morgan stanley": "MS", "visa": "V", "mastercard": "MA",
        "american express": "AXP", "amex": "AXP", "paypal": "PYPL", "square": "SQ", "block": "SQ",
        "blackrock": "BLK", "berkshire": "BRK.B", "berkshire hathaway": "BRK.B",

        # Healthcare
        "johnson & johnson": "JNJ", "j&j": "JNJ", "pfizer": "PFE", "moderna": "MRNA",
        "unitedhealth": "UNH", "eli lilly": "LLY", "lilly": "LLY", "abbvie": "ABBV",
        "merck": "MRK", "bristol myers": "BMY", "amgen": "AMGN", "gilead": "GILD",

        # Consumer
        "walmart": "WMT", "costco": "COST", "target": "TGT", "home depot": "HD",
        "nike": "NKE", "starbucks": "SBUX", "mcdonald's": "MCD", "mcdonalds": "MCD",
        "coca cola": "KO", "coca-cola": "KO", "coke": "KO", "pepsi": "PEP", "pepsico": "PEP",
        "disney": "DIS", "procter & gamble": "PG", "p&g": "PG",

        # Energy
        "exxon": "XOM", "exxonmobil": "XOM", "chevron": "CVX", "conocophillips": "COP",
        "shell": "SHEL", "bp": "BP",

        # Auto
        "ford": "F", "gm": "GM", "general motors": "GM",

        # Telecom
        "verizon": "VZ", "at&t": "T", "att": "T", "t-mobile": "TMUS",

        # Cryptocurrencies
        "bitcoin": "BTC-USD", "btc": "BTC-USD",
        "ethereum": "ETH-USD", "eth": "ETH-USD", "ether": "ETH-USD",
        "ripple": "XRP-USD", "xrp": "XRP-USD",
        "cardano": "ADA-USD", "ada": "ADA-USD",
        "solana": "SOL-USD", "sol": "SOL-USD",
        "dogecoin": "DOGE-USD", "doge": "DOGE-USD",
        "polkadot": "DOT-USD", "litecoin": "LTC-USD",
        "chainlink": "LINK-USD", "avalanche": "AVAX-USD",
        "polygon": "MATIC-USD", "matic": "MATIC-USD",
        "uniswap": "UNI-USD", "shiba inu": "SHIB-USD", "shib": "SHIB-USD",

        # Forex (common references)
        "euro dollar": "EURUSD=X", "eur/usd": "EURUSD=X", "eurusd": "EURUSD=X",
        "pound dollar": "GBPUSD=X", "gbp/usd": "GBPUSD=X", "gbpusd": "GBPUSD=X",
        "dollar yen": "USDJPY=X", "usd/jpy": "USDJPY=X", "usdjpy": "USDJPY=X",
        "aussie dollar": "AUDUSD=X", "aud/usd": "AUDUSD=X",

        # Commodities
        "gold": "GC=F", "xau": "GC=F", "gold futures": "GC=F",
        "silver": "SI=F", "xag": "SI=F",
        "oil": "CL=F", "crude oil": "CL=F", "crude": "CL=F", "wti": "CL=F",
        "brent": "BZ=F", "brent oil": "BZ=F",
        "natural gas": "NG=F", "nat gas": "NG=F",
        "copper": "HG=F", "platinum": "PL=F",
        "corn": "ZC=F", "wheat": "ZW=F", "soybeans": "ZS=F",
        "coffee": "KC=F", "sugar": "SB=F", "cocoa": "CC=F",

        # Indices
        "s&p": "^GSPC", "s&p 500": "^GSPC", "sp500": "^GSPC", "spx": "^GSPC",
        "dow jones": "^DJI", "dow": "^DJI", "djia": "^DJI",
        "nasdaq composite": "^IXIC", "nasdaq 100": "^NDX", "ndx": "^NDX",
        "russell 2000": "^RUT", "russell": "^RUT",
        "vix": "^VIX", "volatility index": "^VIX",
        "ftse": "^FTSE", "dax": "^GDAXI", "nikkei": "^N225",

        # ETFs
        "spy": "SPY", "nasdaq": "QQQ", "qqq": "QQQ", "iwm": "IWM", "dia": "DIA",
        "gld": "GLD", "slv": "SLV", "uso": "USO", "ung": "UNG",
        "tlt": "TLT", "hyg": "HYG", "lqd": "LQD",
        "arkk": "ARKK", "soxx": "SOXX", "smh": "SMH",
        "xlk": "XLK", "xlf": "XLF", "xle": "XLE", "xlv": "XLV",

        # Futures (common references)
        "es futures": "ES=F", "e-mini": "ES=F", "es": "ES=F",
        "nq futures": "NQ=F", "nq": "NQ=F",

        # Bonds/Treasuries
        "10 year treasury": "^TNX", "10y treasury": "^TNX",
        "30 year treasury": "^TYX", "30y treasury": "^TYX",
    }

    # Popular tickers for quick validation
    POPULAR_TICKERS = {
        # Stocks
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
        "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "MRK",
        "ABBV", "KO", "PEP", "COST", "AVGO", "LLY", "MCD", "CSCO", "TMO", "ACN",
        "ABT", "DHR", "NFLX", "AMD", "INTC", "CRM", "NKE", "DIS", "VZ", "ADBE",
        "TXN", "PM", "CMCSA", "NEE", "RTX", "SPY", "QQQ", "IWM", "DIA", "VOO",
        "PLTR", "COIN", "HOOD", "RIVN", "LCID", "SNOW", "CRWD", "DDOG", "ZM",
        "PYPL", "SQ", "SHOP", "UBER", "ABNB", "MRNA", "PFE", "BAC", "WFC", "GS",
        # Defense stocks
        "KTOS", "LMT", "NOC", "GD", "LHX", "BA",
        # Cryptocurrencies
        "BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD",
        "DOT-USD", "LTC-USD", "LINK-USD", "AVAX-USD", "MATIC-USD", "UNI-USD",
        "SHIB-USD", "ATOM-USD", "XLM-USD", "NEAR-USD", "APT-USD",
        # Forex
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X",
        "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
        # Commodities
        "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "PL=F", "PA=F",
        "ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F", "CC=F", "CT=F",
        # Indices
        "^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT", "^VIX",
        "^FTSE", "^GDAXI", "^N225", "^HSI",
        # Futures
        "ES=F", "NQ=F", "YM=F", "RTY=F", "MES=F", "MNQ=F",
        # Bonds
        "^TNX", "^TYX", "^FVX", "ZN=F", "ZB=F",
        # Popular ETFs
        "GLD", "SLV", "USO", "UNG", "TLT", "HYG", "LQD", "ARKK", "SOXX", "SMH",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLRE",
        "EEM", "VWO", "EFA", "FXI", "TQQQ", "SQQQ",
    }

    # Ticker to company name mapping (reverse of COMPANY_TO_TICKER for lookup)
    TICKER_TO_COMPANY = {
        "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet (Google)", "AMZN": "Amazon",
        "NVDA": "NVIDIA", "META": "Meta (Facebook)", "TSLA": "Tesla", "NFLX": "Netflix",
        "ADBE": "Adobe", "CRM": "Salesforce", "INTC": "Intel", "AMD": "AMD",
        "CSCO": "Cisco", "ORCL": "Oracle", "IBM": "IBM", "QCOM": "Qualcomm",
        "AVGO": "Broadcom", "TXN": "Texas Instruments", "PYPL": "PayPal", "SHOP": "Shopify",
        "UBER": "Uber", "ABNB": "Airbnb", "PLTR": "Palantir", "SNOW": "Snowflake",
        "CRWD": "CrowdStrike", "DDOG": "Datadog", "ZM": "Zoom", "DOCU": "DocuSign",
        "TWLO": "Twilio", "OKTA": "Okta", "COIN": "Coinbase", "HOOD": "Robinhood",
        "RIVN": "Rivian", "LCID": "Lucid Motors",
        # Defense
        "KTOS": "Kratos Defense & Security", "LMT": "Lockheed Martin", "NOC": "Northrop Grumman",
        "RTX": "Raytheon", "BA": "Boeing", "GD": "General Dynamics", "LHX": "L3Harris",
        # Finance
        "JPM": "JPMorgan Chase", "BAC": "Bank of America", "WFC": "Wells Fargo",
        "C": "Citigroup", "GS": "Goldman Sachs", "MS": "Morgan Stanley",
        "V": "Visa", "MA": "Mastercard", "AXP": "American Express", "SQ": "Block (Square)",
        "BLK": "BlackRock", "BRK.B": "Berkshire Hathaway",
        # Healthcare
        "JNJ": "Johnson & Johnson", "PFE": "Pfizer", "MRNA": "Moderna",
        "UNH": "UnitedHealth", "LLY": "Eli Lilly", "ABBV": "AbbVie",
        "MRK": "Merck", "BMY": "Bristol Myers Squibb", "AMGN": "Amgen", "GILD": "Gilead",
        # Consumer
        "WMT": "Walmart", "COST": "Costco", "TGT": "Target", "HD": "Home Depot",
        "NKE": "Nike", "SBUX": "Starbucks", "MCD": "McDonald's",
        "KO": "Coca-Cola", "PEP": "PepsiCo", "DIS": "Disney", "PG": "Procter & Gamble",
        # Energy
        "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
        "SHEL": "Shell", "BP": "BP",
        # Auto
        "F": "Ford", "GM": "General Motors",
        # Telecom
        "VZ": "Verizon", "T": "AT&T", "TMUS": "T-Mobile",
        # ETFs
        "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "IWM": "Russell 2000 ETF", "DIA": "Dow Jones ETF",
    }

    # Words to exclude from ticker matching (common English words that could be tickers)
    EXCLUDED_WORDS = {
        # Common words
        "A", "I", "THE", "TO", "BY", "OR", "IF", "AT", "AS", "IT", "IS", "IN",
        "ON", "OF", "AN", "BE", "DO", "GO", "HE", "ME", "MY", "NO", "SO", "UP",
        "US", "WE", "AM", "ARE", "FOR", "HIT", "REACH", "STOCK", "PRICE",
        "BUY", "SELL", "ETF", "IPO", "CEO", "CFO", "EPS", "GDP", "FED", "SEC",
        "USA", "USD", "EUR", "NEW", "OLD", "TOP", "LOW", "HIGH", "ALL", "ANY",
        "CAN", "MAY", "NOW", "OUT", "OUR", "SAY", "WAY", "WHO", "HOW", "WHY",
        # Time-related words that are also tickers
        "NEXT", "LAST", "YEAR", "WEEK", "DAY", "DAYS", "MONTH", "TIME",
        # Action words that could match tickers
        "WILL", "HOLD", "MAKE", "TAKE", "RISE", "FALL", "DROP", "GAIN", "MOVE",
        "JUMP", "GROW", "WORK", "TURN", "KEEP", "COME", "STAY", "STOP", "RUN",
        # Question words
        "WHAT", "WHEN", "WHERE", "WHICH", "THINK", "DOES", "SHOULD", "COULD",
        # Other common words that are real tickers but likely not intended
        "WELL", "GOOD", "BEST", "REAL", "LIVE", "OPEN", "PLAY", "FREE", "FAST",
        "TRUE", "EVER", "EVEN", "MUCH", "MOST", "VERY", "JUST", "ONLY", "ALSO",
        "BACK", "OVER", "DOWN", "LONG", "SHORT", "CALL", "PUT",
        # Direction words
        "INCREASE", "DECREASE",
    }

    def __init__(self):
        """Initialize the market data service."""
        self.logger = logger.bind(service="market_data")

    def get_company_name(self, ticker: str) -> str:
        """Get company name for a ticker."""
        ticker = ticker.upper()
        if ticker in self.TICKER_TO_COMPANY:
            return self.TICKER_TO_COMPANY[ticker]
        # Try to fetch from yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get("longName") or info.get("shortName") or ticker
        except Exception:
            return ticker

    def extract_ticker_with_confidence(self, query: str) -> TickerExtractionResult:
        """
        Enhanced ticker extraction that returns confidence scores and suggestions.

        This method analyzes the query and returns:
        - The most likely ticker
        - Confidence score
        - Whether user confirmation is needed
        - Alternative suggestions if ambiguous

        Args:
            query: User's prediction query

        Returns:
            TickerExtractionResult with confidence info
        """
        query_clean = query.strip()
        query_lower = query_clean.lower()
        query_upper = query_clean.upper()

        suggestions = []

        # Strategy 1: Check company names first (high confidence)
        for company_name, ticker in self.COMPANY_TO_TICKER.items():
            if company_name in query_lower:
                company_display = self.get_company_name(ticker)
                self.logger.debug(
                    "Ticker found via company name",
                    company=company_name,
                    ticker=ticker,
                )
                return TickerExtractionResult(
                    ticker=ticker,
                    company_name=company_display,
                    confidence=0.95,  # High confidence for company name match
                    needs_confirmation=False,
                    message=None,
                )

        # Strategy 2: Check for $ prefix (very high confidence)
        dollar_match = re.search(r"\$([A-Z]{1,5})\b", query_upper)
        if dollar_match:
            ticker = dollar_match.group(1)
            if self._quick_validate_ticker(ticker):
                company_name = self.get_company_name(ticker)
                self.logger.debug("Ticker found via $ prefix", ticker=ticker)
                return TickerExtractionResult(
                    ticker=ticker,
                    company_name=company_name,
                    confidence=1.0,  # Very high confidence for $ prefix
                    needs_confirmation=False,
                    message=None,
                )

        # Strategy 3: Check for tickers in parentheses (very high confidence)
        paren_match = re.search(r"\(([A-Z]{1,5})\)", query_upper)
        if paren_match:
            ticker = paren_match.group(1)
            if ticker not in self.EXCLUDED_WORDS and self._quick_validate_ticker(ticker):
                company_name = self.get_company_name(ticker)
                self.logger.debug("Ticker found in parentheses", ticker=ticker)
                return TickerExtractionResult(
                    ticker=ticker,
                    company_name=company_name,
                    confidence=1.0,  # Very high confidence for explicit parentheses
                    needs_confirmation=False,
                    message=None,
                )

        # Strategy 4: Match popular tickers (medium-high confidence)
        sorted_tickers = sorted(self.POPULAR_TICKERS, key=len, reverse=True)
        for ticker in sorted_tickers:
            pattern = rf"\b{re.escape(ticker)}\b"
            if re.search(pattern, query_upper):
                company_name = self.get_company_name(ticker)
                self.logger.debug("Ticker found via popular match", ticker=ticker)
                return TickerExtractionResult(
                    ticker=ticker,
                    company_name=company_name,
                    confidence=0.85,  # Good confidence for popular ticker match
                    needs_confirmation=False,
                    message=None,
                )

        # Strategy 5: Pattern-based extraction (lower confidence, may need confirmation)
        potential_tickers = re.findall(r"\b([A-Z]{2,5})\b", query_upper)
        valid_candidates = []

        for ticker in potential_tickers:
            if ticker not in self.EXCLUDED_WORDS:
                if self._quick_validate_ticker(ticker):
                    company_name = self.get_company_name(ticker)
                    valid_candidates.append(
                        TickerSuggestion(
                            ticker=ticker,
                            company_name=company_name,
                            confidence=0.6,
                            match_reason="Pattern match - found ticker-like text in query",
                        )
                    )

        if len(valid_candidates) == 1:
            # Single candidate found via pattern - medium confidence
            candidate = valid_candidates[0]
            return TickerExtractionResult(
                ticker=candidate.ticker,
                company_name=candidate.company_name,
                confidence=0.7,
                needs_confirmation=True,
                suggestions=[candidate],
                message=f"Did you mean {candidate.ticker} ({candidate.company_name})?",
            )
        elif len(valid_candidates) > 1:
            # Multiple candidates - needs confirmation
            return TickerExtractionResult(
                ticker=valid_candidates[0].ticker,  # Best guess
                company_name=valid_candidates[0].company_name,
                confidence=0.5,
                needs_confirmation=True,
                suggestions=valid_candidates[:3],  # Top 3 suggestions
                message="Multiple possible tickers found. Which one did you mean?",
            )

        # No ticker found
        self.logger.warning("Could not extract ticker from query", query=query[:50])
        return TickerExtractionResult(
            ticker=None,
            company_name=None,
            confidence=0.0,
            needs_confirmation=True,
            suggestions=[],
            message="Could not identify a stock ticker in your question. Please specify a ticker symbol (e.g., AAPL, TSLA, NVDA).",
        )

    def extract_ticker_from_query(self, query: str) -> Optional[str]:
        """
        Smart ticker extraction from natural language query.

        Uses multiple strategies:
        1. Instrument detector (crypto, forex, commodities, indices, futures, bonds)
        2. Company name matching
        3. Direct ticker matching with word boundaries
        4. Pattern-based extraction with validation

        Args:
            query: User's prediction query

        Returns:
            Extracted ticker symbol or None
        """
        query_clean = query.strip()
        query_lower = query_clean.lower()
        query_upper = query_clean.upper()

        # Strategy 0: Use instrument detector for non-stock instruments
        instrument = instrument_detector.detect_instrument(query)
        if instrument:
            self.logger.debug(
                "Instrument detected",
                type=instrument.instrument_type.value,
                symbol=instrument.symbol,
            )
            return instrument.symbol

        # Strategy 1: Check company names first (most user-friendly)
        for company_name, ticker in self.COMPANY_TO_TICKER.items():
            if company_name in query_lower:
                self.logger.debug(
                    "Ticker found via company name",
                    company=company_name,
                    ticker=ticker,
                )
                return ticker

        # Strategy 2: Check for explicit ticker mentions with $ prefix
        dollar_match = re.search(r"\$([A-Z]{1,5})\b", query_upper)
        if dollar_match:
            ticker = dollar_match.group(1)
            if self._quick_validate_ticker(ticker):
                self.logger.debug("Ticker found via $ prefix", ticker=ticker)
                return ticker

        # Strategy 3: Check for tickers in parentheses like "(KTOS)" or "(AAPL)"
        # This is high priority since it's an explicit user reference
        paren_match = re.search(r"\(([A-Z]{1,5})\)", query_upper)
        if paren_match:
            ticker = paren_match.group(1)
            if ticker not in self.EXCLUDED_WORDS and self._quick_validate_ticker(ticker):
                self.logger.debug("Ticker found in parentheses", ticker=ticker)
                return ticker

        # Strategy 4: Match popular tickers with word boundaries (sorted by length)
        sorted_tickers = sorted(self.POPULAR_TICKERS, key=len, reverse=True)
        for ticker in sorted_tickers:
            pattern = rf"\b{re.escape(ticker)}\b"
            if re.search(pattern, query_upper):
                self.logger.debug("Ticker found via popular match", ticker=ticker)
                return ticker

        # Strategy 5: Pattern-based extraction for any ticker-like strings
        potential_tickers = re.findall(r"\b([A-Z]{2,5})\b", query_upper)
        for ticker in potential_tickers:
            if ticker not in self.EXCLUDED_WORDS:
                if self._quick_validate_ticker(ticker):
                    self.logger.debug("Ticker found via pattern", ticker=ticker)
                    return ticker

        self.logger.warning("Could not extract ticker from query", query=query[:50])
        return None

    def _quick_validate_ticker(self, ticker: str) -> bool:
        """
        Quick validation that a ticker exists.

        Uses cache to avoid repeated API calls.
        """
        if ticker in self.POPULAR_TICKERS:
            return True

        # Check cache
        cache_key = f"valid_{ticker}"
        if cache_key in _info_cache:
            return _info_cache[cache_key]

        # Quick check via yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            is_valid = info.get("regularMarketPrice") is not None or info.get("currentPrice") is not None
            _info_cache[cache_key] = is_valid
            return is_valid
        except Exception:
            _info_cache[cache_key] = False
            return False

    def get_quote(self, ticker: str) -> Optional[StockQuote]:
        """
        Get real-time stock quote with smart caching.

        Args:
            ticker: Stock ticker symbol

        Returns:
            StockQuote object or None
        """
        ticker = ticker.upper()
        cache_key = f"quote_{ticker}"

        # Check cache
        if cache_key in _quote_cache:
            self.logger.debug("Quote cache hit", ticker=ticker)
            return _quote_cache[cache_key]

        try:
            self.logger.info("Fetching quote", ticker=ticker)
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get current price from multiple possible fields
            current_price = (
                info.get("regularMarketPrice")
                or info.get("currentPrice")
                or info.get("previousClose")
            )

            if current_price is None:
                self.logger.warning("No price data available", ticker=ticker)
                return None

            # Get previous close
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or current_price

            # Calculate change
            change_percent = ((current_price - prev_close) / prev_close * 100) if prev_close else 0

            quote = StockQuote(
                ticker=ticker,
                current_price=round(current_price, 2),
                previous_close=round(prev_close, 2),
                open_price=round(info.get("regularMarketOpen") or info.get("open") or current_price, 2),
                day_high=round(info.get("regularMarketDayHigh") or info.get("dayHigh") or current_price, 2),
                day_low=round(info.get("regularMarketDayLow") or info.get("dayLow") or current_price, 2),
                volume=info.get("regularMarketVolume") or info.get("volume") or 0,
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                fifty_two_week_low=info.get("fiftyTwoWeekLow"),
                change_percent=round(change_percent, 2),
                timestamp=datetime.utcnow(),
            )

            _quote_cache[cache_key] = quote
            self.logger.info("Quote fetched", ticker=ticker, price=current_price)

            return quote

        except Exception as e:
            self.logger.error("Failed to fetch quote", ticker=ticker, error=str(e))
            return None

    def get_company_info(self, ticker: str) -> Optional[CompanyInfo]:
        """
        Get comprehensive company information.

        Args:
            ticker: Stock ticker symbol

        Returns:
            CompanyInfo object or None
        """
        ticker = ticker.upper()
        cache_key = f"info_{ticker}"

        if cache_key in _info_cache and isinstance(_info_cache[cache_key], CompanyInfo):
            self.logger.debug("Company info cache hit", ticker=ticker)
            return _info_cache[cache_key]

        try:
            self.logger.info("Fetching company info", ticker=ticker)
            stock = yf.Ticker(ticker)
            info = stock.info

            company = CompanyInfo(
                ticker=ticker,
                name=info.get("longName") or info.get("shortName") or ticker,
                sector=info.get("sector"),
                industry=info.get("industry"),
                description=info.get("longBusinessSummary"),
                website=info.get("website"),
                employees=info.get("fullTimeEmployees"),
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                peg_ratio=info.get("pegRatio"),
                price_to_book=info.get("priceToBook"),
                dividend_yield=info.get("dividendYield"),
                profit_margin=info.get("profitMargins"),
                revenue_growth=info.get("revenueGrowth"),
                earnings_growth=info.get("earningsGrowth"),
            )

            _info_cache[cache_key] = company
            self.logger.info("Company info fetched", ticker=ticker, name=company.name)

            return company

        except Exception as e:
            self.logger.error("Failed to fetch company info", ticker=ticker, error=str(e))
            return None

    def get_historical_data(
        self,
        ticker: str,
        period: str = "3mo",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data with caching.

        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)

        Returns:
            DataFrame with OHLCV data or None
        """
        ticker = ticker.upper()
        cache_key = f"hist_{ticker}_{period}_{interval}"

        if cache_key in _history_cache:
            self.logger.debug("Historical data cache hit", ticker=ticker)
            return _history_cache[cache_key]

        try:
            self.logger.info("Fetching historical data", ticker=ticker, period=period)
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                self.logger.warning("No historical data returned", ticker=ticker)
                return None

            _history_cache[cache_key] = df
            self.logger.info("Historical data fetched", ticker=ticker, rows=len(df))

            return df

        except Exception as e:
            self.logger.error("Failed to fetch historical data", ticker=ticker, error=str(e))
            return None

    def get_earnings_dates(self, ticker: str) -> list[dict]:
        """Get upcoming and past earnings dates."""
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar

            if calendar is None:
                return []

            earnings = []
            if isinstance(calendar, dict) and "Earnings Date" in calendar:
                dates = calendar["Earnings Date"]
                if isinstance(dates, list):
                    for date in dates:
                        earnings.append({"date": date, "type": "upcoming"})
                else:
                    earnings.append({"date": dates, "type": "upcoming"})

            return earnings

        except Exception as e:
            self.logger.error("Failed to fetch earnings dates", ticker=ticker, error=str(e))
            return []

    def get_insider_transactions(self, ticker: str, limit: int = 10) -> list[dict]:
        """Get recent insider transactions."""
        try:
            stock = yf.Ticker(ticker)
            insider_df = stock.insider_transactions

            if insider_df is None or insider_df.empty:
                return []

            transactions = []
            for _, row in insider_df.head(limit).iterrows():
                transactions.append({
                    "insider": row.get("Insider"),
                    "relation": row.get("Relation"),
                    "transaction": row.get("Transaction"),
                    "date": row.get("Start Date"),
                    "shares": row.get("Shares"),
                    "value": row.get("Value"),
                })

            return transactions

        except Exception as e:
            self.logger.error("Failed to fetch insider transactions", ticker=ticker, error=str(e))
            return []

    def get_analyst_recommendations(self, ticker: str) -> dict:
        """Get analyst recommendations summary."""
        try:
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations

            if recommendations is None or recommendations.empty:
                return {}

            recent = recommendations.tail(30)
            counts = recent["To Grade"].value_counts().to_dict() if "To Grade" in recent.columns else {}

            return {
                "counts": counts,
                "total": len(recent),
                "latest": recent.iloc[-1].to_dict() if len(recent) > 0 else None,
            }

        except Exception as e:
            self.logger.error("Failed to fetch recommendations", ticker=ticker, error=str(e))
            return {}

    def get_key_stats(self, ticker: str) -> dict:
        """Get key financial statistics."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "revenue": info.get("totalRevenue"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "current_ratio": info.get("currentRatio"),
                "debt_to_equity": info.get("debtToEquity"),
                "free_cash_flow": info.get("freeCashflow"),
                "beta": info.get("beta"),
                "52_week_change": info.get("52WeekChange"),
                "avg_volume": info.get("averageVolume"),
                "avg_volume_10d": info.get("averageVolume10days"),
            }

        except Exception as e:
            self.logger.error("Failed to fetch key stats", ticker=ticker, error=str(e))
            return {}


# Singleton instance
market_data_service = MarketDataService()
