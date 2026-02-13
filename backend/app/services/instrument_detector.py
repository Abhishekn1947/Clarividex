"""
Instrument Detector - Multi-asset class financial instrument detection.

Supports detection and classification of:
- Stocks (AAPL, MSFT)
- Cryptocurrencies (BTC-USD, ETH-USD)
- Forex pairs (EURUSD=X)
- Commodities (GC=F, CL=F)
- Indices (^GSPC, ^DJI)
- ETFs (SPY, QQQ)
- Futures (ES=F, NQ=F)
- Bonds/Treasuries (^TNX, ^TYX)
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass

import structlog

from backend.app.models.schemas import InstrumentType

logger = structlog.get_logger()


@dataclass
class InstrumentInfo:
    """Information about a detected financial instrument."""

    symbol: str  # Yahoo Finance compatible symbol
    instrument_type: InstrumentType
    name: str
    original_query: str  # What the user typed
    confidence: float = 1.0


class InstrumentDetector:
    """
    Detects and classifies financial instruments from natural language.

    Converts various formats to Yahoo Finance compatible symbols.
    """

    # =========================================================================
    # Cryptocurrency Mappings
    # =========================================================================
    CRYPTO_NAMES = {
        # Major cryptos
        "bitcoin": "BTC-USD",
        "btc": "BTC-USD",
        "ethereum": "ETH-USD",
        "eth": "ETH-USD",
        "ether": "ETH-USD",
        "ripple": "XRP-USD",
        "xrp": "XRP-USD",
        "cardano": "ADA-USD",
        "ada": "ADA-USD",
        "solana": "SOL-USD",
        "sol": "SOL-USD",
        "dogecoin": "DOGE-USD",
        "doge": "DOGE-USD",
        "polkadot": "DOT-USD",
        "dot": "DOT-USD",
        "litecoin": "LTC-USD",
        "ltc": "LTC-USD",
        "chainlink": "LINK-USD",
        "link": "LINK-USD",
        "avalanche": "AVAX-USD",
        "avax": "AVAX-USD",
        "polygon": "MATIC-USD",
        "matic": "MATIC-USD",
        "uniswap": "UNI-USD",
        "uni": "UNI-USD",
        "stellar": "XLM-USD",
        "xlm": "XLM-USD",
        "cosmos": "ATOM-USD",
        "atom": "ATOM-USD",
        "monero": "XMR-USD",
        "xmr": "XMR-USD",
        "tron": "TRX-USD",
        "trx": "TRX-USD",
        "near": "NEAR-USD",
        "near protocol": "NEAR-USD",
        "aptos": "APT-USD",
        "apt": "APT-USD",
        "sui": "SUI-USD",
        "arbitrum": "ARB-USD",
        "arb": "ARB-USD",
        "optimism": "OP-USD",
        "op": "OP-USD",
        "shiba": "SHIB-USD",
        "shiba inu": "SHIB-USD",
        "shib": "SHIB-USD",
        "pepe": "PEPE-USD",
    }

    # =========================================================================
    # Forex Mappings
    # =========================================================================
    FOREX_PAIRS = {
        # Major pairs
        "eur/usd": "EURUSD=X",
        "eurusd": "EURUSD=X",
        "eur usd": "EURUSD=X",
        "euro dollar": "EURUSD=X",
        "gbp/usd": "GBPUSD=X",
        "gbpusd": "GBPUSD=X",
        "gbp usd": "GBPUSD=X",
        "pound dollar": "GBPUSD=X",
        "cable": "GBPUSD=X",
        "usd/jpy": "USDJPY=X",
        "usdjpy": "USDJPY=X",
        "usd jpy": "USDJPY=X",
        "dollar yen": "USDJPY=X",
        "usd/chf": "USDCHF=X",
        "usdchf": "USDCHF=X",
        "swissy": "USDCHF=X",
        "usd/cad": "USDCAD=X",
        "usdcad": "USDCAD=X",
        "loonie": "USDCAD=X",
        "aud/usd": "AUDUSD=X",
        "audusd": "AUDUSD=X",
        "aussie": "AUDUSD=X",
        "nzd/usd": "NZDUSD=X",
        "nzdusd": "NZDUSD=X",
        "kiwi": "NZDUSD=X",

        # Cross pairs
        "eur/gbp": "EURGBP=X",
        "eurgbp": "EURGBP=X",
        "eur/jpy": "EURJPY=X",
        "eurjpy": "EURJPY=X",
        "gbp/jpy": "GBPJPY=X",
        "gbpjpy": "GBPJPY=X",
        "eur/chf": "EURCHF=X",
        "eurchf": "EURCHF=X",
        "aud/jpy": "AUDJPY=X",
        "audjpy": "AUDJPY=X",
        "cad/jpy": "CADJPY=X",
        "cadjpy": "CADJPY=X",

        # Emerging market pairs
        "usd/mxn": "USDMXN=X",
        "usdmxn": "USDMXN=X",
        "usd/zar": "USDZAR=X",
        "usdzar": "USDZAR=X",
        "usd/try": "USDTRY=X",
        "usdtry": "USDTRY=X",
        "usd/brl": "USDBRL=X",
        "usdbrl": "USDBRL=X",
        "usd/inr": "USDINR=X",
        "usdinr": "USDINR=X",
        "usd/cny": "USDCNY=X",
        "usdcny": "USDCNY=X",
    }

    # =========================================================================
    # Commodity Mappings
    # =========================================================================
    COMMODITIES = {
        # Precious metals
        "gold": "GC=F",
        "xau": "GC=F",
        "gold futures": "GC=F",
        "silver": "SI=F",
        "xag": "SI=F",
        "silver futures": "SI=F",
        "platinum": "PL=F",
        "palladium": "PA=F",

        # Energy
        "oil": "CL=F",
        "crude": "CL=F",
        "crude oil": "CL=F",
        "wti": "CL=F",
        "wti oil": "CL=F",
        "brent": "BZ=F",
        "brent oil": "BZ=F",
        "brent crude": "BZ=F",
        "natural gas": "NG=F",
        "nat gas": "NG=F",
        "natgas": "NG=F",
        "gasoline": "RB=F",
        "rbob": "RB=F",
        "heating oil": "HO=F",

        # Agriculture
        "corn": "ZC=F",
        "wheat": "ZW=F",
        "soybeans": "ZS=F",
        "soybean": "ZS=F",
        "coffee": "KC=F",
        "sugar": "SB=F",
        "cocoa": "CC=F",
        "cotton": "CT=F",
        "lumber": "LBS=F",
        "orange juice": "OJ=F",
        "oj": "OJ=F",

        # Livestock
        "cattle": "LE=F",
        "live cattle": "LE=F",
        "feeder cattle": "GF=F",
        "lean hogs": "HE=F",
        "hogs": "HE=F",

        # Industrial metals
        "copper": "HG=F",
        "aluminum": "ALI=F",
    }

    # =========================================================================
    # Index Mappings
    # =========================================================================
    INDICES = {
        # US indices
        "s&p 500": "^GSPC",
        "s&p": "^GSPC",
        "sp500": "^GSPC",
        "spx": "^GSPC",
        "dow jones": "^DJI",
        "dow": "^DJI",
        "djia": "^DJI",
        "nasdaq": "^IXIC",
        "nasdaq composite": "^IXIC",
        "nasdaq 100": "^NDX",
        "ndx": "^NDX",
        "russell 2000": "^RUT",
        "russell": "^RUT",
        "rut": "^RUT",
        "vix": "^VIX",
        "volatility index": "^VIX",
        "fear index": "^VIX",

        # International indices
        "ftse": "^FTSE",
        "ftse 100": "^FTSE",
        "dax": "^GDAXI",
        "german dax": "^GDAXI",
        "cac 40": "^FCHI",
        "cac": "^FCHI",
        "nikkei": "^N225",
        "nikkei 225": "^N225",
        "hang seng": "^HSI",
        "hsi": "^HSI",
        "shanghai": "000001.SS",
        "shanghai composite": "000001.SS",
        "kospi": "^KS11",
        "asx 200": "^AXJO",
        "asx": "^AXJO",
        "tsx": "^GSPTSE",
        "tsx composite": "^GSPTSE",
        "ibovespa": "^BVSP",
        "bovespa": "^BVSP",
        "sensex": "^BSESN",
        "nifty": "^NSEI",
        "nifty 50": "^NSEI",
    }

    # =========================================================================
    # Futures Mappings (Index Futures)
    # =========================================================================
    FUTURES = {
        # E-mini futures
        "es": "ES=F",
        "e-mini": "ES=F",
        "e-mini s&p": "ES=F",
        "es futures": "ES=F",
        "nq": "NQ=F",
        "e-mini nasdaq": "NQ=F",
        "nq futures": "NQ=F",
        "ym": "YM=F",
        "e-mini dow": "YM=F",
        "ym futures": "YM=F",
        "rty": "RTY=F",
        "e-mini russell": "RTY=F",

        # Micro futures
        "mes": "MES=F",
        "micro e-mini": "MES=F",
        "mnq": "MNQ=F",
        "micro nasdaq": "MNQ=F",

        # Currency futures
        "6e": "6E=F",
        "euro futures": "6E=F",
        "6b": "6B=F",
        "pound futures": "6B=F",
        "6j": "6J=F",
        "yen futures": "6J=F",
    }

    # =========================================================================
    # Bond/Treasury Mappings
    # =========================================================================
    BONDS = {
        # Treasury yields
        "10 year treasury": "^TNX",
        "10 year yield": "^TNX",
        "10y treasury": "^TNX",
        "tnx": "^TNX",
        "30 year treasury": "^TYX",
        "30 year yield": "^TYX",
        "30y treasury": "^TYX",
        "tyx": "^TYX",
        "5 year treasury": "^FVX",
        "5 year yield": "^FVX",
        "5y treasury": "^FVX",
        "fvx": "^FVX",
        "2 year treasury": "^IRX",
        "2 year yield": "^IRX",
        "2y treasury": "^IRX",

        # Treasury futures
        "zn": "ZN=F",
        "10 year note": "ZN=F",
        "zb": "ZB=F",
        "30 year bond": "ZB=F",
        "zf": "ZF=F",
        "5 year note": "ZF=F",
        "zt": "ZT=F",
        "2 year note": "ZT=F",

        # Bond ETFs
        "tlt": "TLT",
        "long term treasury": "TLT",
        "shy": "SHY",
        "short term treasury": "SHY",
        "ief": "IEF",
        "intermediate treasury": "IEF",
        "agg": "AGG",
        "aggregate bond": "AGG",
        "bnd": "BND",
        "total bond": "BND",
        "lqd": "LQD",
        "corporate bonds": "LQD",
        "hyg": "HYG",
        "high yield bonds": "HYG",
        "jnk": "JNK",
        "junk bonds": "JNK",
    }

    # =========================================================================
    # Popular ETFs
    # =========================================================================
    ETFS = {
        # Index ETFs
        "spy": ("SPY", "S&P 500 ETF"),
        "qqq": ("QQQ", "Nasdaq 100 ETF"),
        "iwm": ("IWM", "Russell 2000 ETF"),
        "dia": ("DIA", "Dow Jones ETF"),
        "voo": ("VOO", "Vanguard S&P 500 ETF"),
        "vti": ("VTI", "Vanguard Total Stock Market ETF"),
        "vtv": ("VTV", "Vanguard Value ETF"),
        "vug": ("VUG", "Vanguard Growth ETF"),

        # Sector ETFs
        "xlk": ("XLK", "Technology Select Sector"),
        "xlf": ("XLF", "Financial Select Sector"),
        "xle": ("XLE", "Energy Select Sector"),
        "xlv": ("XLV", "Health Care Select Sector"),
        "xli": ("XLI", "Industrial Select Sector"),
        "xlp": ("XLP", "Consumer Staples Select Sector"),
        "xly": ("XLY", "Consumer Discretionary Select Sector"),
        "xlu": ("XLU", "Utilities Select Sector"),
        "xlb": ("XLB", "Materials Select Sector"),
        "xlre": ("XLRE", "Real Estate Select Sector"),

        # Thematic ETFs
        "arkk": ("ARKK", "ARK Innovation ETF"),
        "arkg": ("ARKG", "ARK Genomic Revolution ETF"),
        "arkw": ("ARKW", "ARK Next Gen Internet ETF"),
        "soxx": ("SOXX", "iShares Semiconductor ETF"),
        "smh": ("SMH", "VanEck Semiconductor ETF"),
        "kweb": ("KWEB", "KraneShares CSI China Internet ETF"),

        # Commodity ETFs
        "gld": ("GLD", "SPDR Gold Shares"),
        "slv": ("SLV", "iShares Silver Trust"),
        "uso": ("USO", "United States Oil Fund"),
        "ung": ("UNG", "United States Natural Gas Fund"),
        "dba": ("DBA", "Invesco Agriculture Fund"),
        "dbc": ("DBC", "Invesco Commodity Index Fund"),

        # Volatility ETFs
        "uvxy": ("UVXY", "ProShares Ultra VIX"),
        "svxy": ("SVXY", "ProShares Short VIX"),
        "vixy": ("VIXY", "ProShares VIX Short-Term Futures"),

        # Leveraged ETFs
        "tqqq": ("TQQQ", "ProShares UltraPro QQQ 3x"),
        "sqqq": ("SQQQ", "ProShares UltraPro Short QQQ 3x"),
        "spxl": ("SPXL", "Direxion S&P 500 Bull 3x"),
        "spxs": ("SPXS", "Direxion S&P 500 Bear 3x"),
        "upro": ("UPRO", "ProShares UltraPro S&P 500 3x"),

        # International ETFs
        "eem": ("EEM", "iShares MSCI Emerging Markets"),
        "vwo": ("VWO", "Vanguard FTSE Emerging Markets"),
        "efa": ("EFA", "iShares MSCI EAFE"),
        "vea": ("VEA", "Vanguard FTSE Developed Markets"),
        "fxi": ("FXI", "iShares China Large-Cap"),
        "ewj": ("EWJ", "iShares MSCI Japan"),
        "ewz": ("EWZ", "iShares MSCI Brazil"),
    }

    def __init__(self):
        """Initialize the instrument detector."""
        self.logger = logger.bind(service="instrument_detector")

    def detect_instrument(self, query: str) -> Optional[InstrumentInfo]:
        """
        Detect and classify a financial instrument from a query.

        Args:
            query: Natural language query

        Returns:
            InstrumentInfo if an instrument is detected, None otherwise
        """
        query_lower = query.lower().strip()

        # Try each instrument type in order of specificity
        detectors = [
            (self._detect_forex, InstrumentType.FOREX),
            (self._detect_crypto, InstrumentType.CRYPTO),
            (self._detect_commodity, InstrumentType.COMMODITY),
            (self._detect_index, InstrumentType.INDEX),
            (self._detect_futures, InstrumentType.FUTURES),
            (self._detect_bond, InstrumentType.BOND),
            (self._detect_etf, InstrumentType.ETF),
        ]

        for detector, instrument_type in detectors:
            result = detector(query_lower)
            if result:
                symbol, name = result
                self.logger.info(
                    "Instrument detected",
                    type=instrument_type.value,
                    symbol=symbol,
                    name=name,
                )
                return InstrumentInfo(
                    symbol=symbol,
                    instrument_type=instrument_type,
                    name=name,
                    original_query=query,
                )

        return None

    def _detect_crypto(self, query: str) -> Optional[Tuple[str, str]]:
        """Detect cryptocurrency mentions."""
        for name, symbol in self.CRYPTO_NAMES.items():
            if re.search(rf"\b{re.escape(name)}\b", query):
                return symbol, name.title()

        # Check for direct symbol patterns like BTC-USD
        crypto_pattern = r"\b([A-Z]{2,5})-USD\b"
        match = re.search(crypto_pattern, query.upper())
        if match:
            return match.group(0), match.group(1)

        return None

    # Common ISO 4217 currency codes for forex pattern validation
    CURRENCY_CODES = {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN", "CZK",
        "HUF", "TRY", "ZAR", "MXN", "BRL", "INR", "KRW", "TWD",
        "THB", "MYR", "IDR", "PHP", "RUB", "ILS", "AED", "SAR",
    }

    def _detect_forex(self, query: str) -> Optional[Tuple[str, str]]:
        """Detect forex pair mentions."""
        for pair, symbol in self.FOREX_PAIRS.items():
            if pair in query:
                return symbol, pair.upper()

        # Check for patterns like EUR/USD or EURUSD
        forex_pattern = r"\b([A-Z]{3})[/\s]?([A-Z]{3})\b"
        match = re.search(forex_pattern, query.upper())
        if match:
            code1, code2 = match.group(1), match.group(2)
            # Only treat as forex if both parts are valid currency codes
            if code1 in self.CURRENCY_CODES and code2 in self.CURRENCY_CODES:
                pair = f"{code1}{code2}=X"
                return pair, f"{code1}/{code2}"

        return None

    def _detect_commodity(self, query: str) -> Optional[Tuple[str, str]]:
        """Detect commodity mentions."""
        for name, symbol in self.COMMODITIES.items():
            if re.search(rf"\b{re.escape(name)}\b", query):
                return symbol, name.title()
        return None

    def _detect_index(self, query: str) -> Optional[Tuple[str, str]]:
        """Detect market index mentions."""
        for name, symbol in self.INDICES.items():
            if re.search(rf"\b{re.escape(name)}\b", query):
                return symbol, name.title()
        return None

    def _detect_futures(self, query: str) -> Optional[Tuple[str, str]]:
        """Detect futures contract mentions."""
        for name, symbol in self.FUTURES.items():
            if re.search(rf"\b{re.escape(name)}\b", query):
                return symbol, name.upper()
        return None

    def _detect_bond(self, query: str) -> Optional[Tuple[str, str]]:
        """Detect bond/treasury mentions."""
        for name, symbol in self.BONDS.items():
            if re.search(rf"\b{re.escape(name)}\b", query):
                return symbol, name.title()
        return None

    def _detect_etf(self, query: str) -> Optional[Tuple[str, str]]:
        """Detect ETF mentions."""
        for name, (symbol, full_name) in self.ETFS.items():
            if re.search(rf"\b{re.escape(name)}\b", query):
                return symbol, full_name
        return None

    def get_yahoo_symbol(self, query: str) -> Optional[str]:
        """
        Get the Yahoo Finance compatible symbol from a query.

        Args:
            query: Natural language query

        Returns:
            Yahoo Finance symbol or None
        """
        instrument = self.detect_instrument(query)
        return instrument.symbol if instrument else None

    def get_instrument_type(self, symbol: str) -> InstrumentType:
        """
        Determine the instrument type from a symbol.

        Args:
            symbol: Financial instrument symbol

        Returns:
            InstrumentType enum value
        """
        symbol = symbol.upper()

        if symbol.endswith("-USD"):
            return InstrumentType.CRYPTO
        if symbol.endswith("=X"):
            return InstrumentType.FOREX
        if symbol.endswith("=F"):
            return InstrumentType.FUTURES
        if symbol.startswith("^"):
            if symbol in ("^TNX", "^TYX", "^FVX", "^IRX"):
                return InstrumentType.BOND
            return InstrumentType.INDEX

        # Check ETF list
        if symbol in [s for s, _ in self.ETFS.values()]:
            return InstrumentType.ETF

        # Default to stock
        return InstrumentType.STOCK


# Singleton instance
instrument_detector = InstrumentDetector()
