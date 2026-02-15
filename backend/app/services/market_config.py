"""
Market Configuration - Defines market-specific settings for US and India markets.

Each market config provides locale, data source availability, ticker formatting,
and other market-specific parameters used throughout the application.
"""

from typing import TypedDict


class MarketConfig(TypedDict):
    currency_symbol: str
    currency_code: str
    news_locale: str
    vix_ticker: str
    main_indices: list[str]
    ticker_suffix: str
    has_sec_filings: bool
    has_finviz: bool
    subreddits: list[str]
    fear_greed_available: bool
    popular_tickers: set[str]


MARKET_CONFIGS: dict[str, MarketConfig] = {
    "US": {
        "currency_symbol": "$",
        "currency_code": "USD",
        "news_locale": "hl=en-US&gl=US&ceid=US:en",
        "vix_ticker": "^VIX",
        "main_indices": ["^GSPC", "^DJI", "^IXIC"],
        "ticker_suffix": "",
        "has_sec_filings": True,
        "has_finviz": True,
        "subreddits": ["wallstreetbets", "stocks", "investing", "stockmarket"],
        "fear_greed_available": True,
        "popular_tickers": set(),  # Populated from MarketDataService.POPULAR_TICKERS
    },
    "IN": {
        "currency_symbol": "\u20b9",
        "currency_code": "INR",
        "news_locale": "hl=en-IN&gl=IN&ceid=IN:en",
        "vix_ticker": "^INDIAVIX",
        "main_indices": ["^NSEI", "^BSESN", "^NSEBANK"],
        "ticker_suffix": ".NS",
        "has_sec_filings": False,
        "has_finviz": False,
        "subreddits": ["IndianStreetBets", "IndiaInvestments"],
        "fear_greed_available": False,
        "popular_tickers": set(),  # Populated from Indian popular tickers
    },
}

# Indian company name to ticker mapping (NSE tickers)
INDIAN_COMPANY_TO_TICKER: dict[str, str] = {
    # IT / Tech
    "tcs": "TCS.NS", "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS", "infy": "INFY.NS",
    "wipro": "WIPRO.NS",
    "hcl tech": "HCLTECH.NS", "hcltech": "HCLTECH.NS", "hcl technologies": "HCLTECH.NS",
    "tech mahindra": "TECHM.NS", "techm": "TECHM.NS",
    "ltimindtree": "LTIM.NS", "lti mindtree": "LTIM.NS",

    # Banking & Finance
    "hdfc bank": "HDFCBANK.NS", "hdfc": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS", "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS", "state bank": "SBIN.NS", "state bank of india": "SBIN.NS",
    "kotak bank": "KOTAKBANK.NS", "kotak mahindra bank": "KOTAKBANK.NS", "kotak": "KOTAKBANK.NS",
    "axis bank": "AXISBANK.NS", "axis": "AXISBANK.NS",
    "indusind bank": "INDUSINDBK.NS", "indusind": "INDUSINDBK.NS",
    "bajaj finance": "BAJFINANCE.NS", "bajfinance": "BAJFINANCE.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "hdfc life": "HDFCLIFE.NS",
    "sbi life": "SBILIFE.NS",

    # Conglomerates / Energy
    "reliance": "RELIANCE.NS", "reliance industries": "RELIANCE.NS", "ril": "RELIANCE.NS",
    "adani enterprises": "ADANIENT.NS", "adani": "ADANIENT.NS",
    "adani ports": "ADANIPORTS.NS",
    "adani green": "ADANIGREEN.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "tata power": "TATAPOWER.NS",
    "tata consumer": "TATACONSUM.NS",

    # Oil & Gas
    "ongc": "ONGC.NS", "oil and natural gas": "ONGC.NS",
    "indian oil": "IOC.NS", "ioc": "IOC.NS",
    "bpcl": "BPCL.NS", "bharat petroleum": "BPCL.NS",

    # Telecom
    "bharti airtel": "BHARTIARTL.NS", "airtel": "BHARTIARTL.NS",
    "jio financial": "JIOFIN.NS",

    # Consumer / FMCG
    "hindustan unilever": "HINDUNILVR.NS", "hul": "HINDUNILVR.NS",
    "itc": "ITC.NS",
    "nestle india": "NESTLEIND.NS", "nestle": "NESTLEIND.NS",
    "asian paints": "ASIANPAINT.NS",
    "titan": "TITAN.NS", "titan company": "TITAN.NS",
    "britannia": "BRITANNIA.NS",
    "dabur": "DABUR.NS",
    "marico": "MARICO.NS",
    "godrej consumer": "GODREJCP.NS",

    # Pharma
    "sun pharma": "SUNPHARMA.NS", "sun pharmaceutical": "SUNPHARMA.NS",
    "dr reddy": "DRREDDY.NS", "dr reddys": "DRREDDY.NS",
    "cipla": "CIPLA.NS",
    "divi's": "DIVISLAB.NS", "divis lab": "DIVISLAB.NS", "divi's lab": "DIVISLAB.NS",

    # Auto
    "maruti": "MARUTI.NS", "maruti suzuki": "MARUTI.NS",
    "mahindra": "M&M.NS", "m&m": "M&M.NS", "mahindra and mahindra": "M&M.NS",
    "bajaj auto": "BAJAJ-AUTO.NS",
    "hero motocorp": "HEROMOTOCO.NS", "hero moto": "HEROMOTOCO.NS",
    "eicher motors": "EICHERMOT.NS", "eicher": "EICHERMOT.NS",

    # Infrastructure / Capital Goods
    "larsen & toubro": "LT.NS", "l&t": "LT.NS", "larsen": "LT.NS",
    "ultratech cement": "ULTRACEMCO.NS", "ultratech": "ULTRACEMCO.NS",
    "grasim": "GRASIM.NS",

    # Power / Utilities
    "ntpc": "NTPC.NS",
    "power grid": "POWERGRID.NS", "powergrid": "POWERGRID.NS",
    "coal india": "COALINDIA.NS",

    # Metals
    "hindalco": "HINDALCO.NS",
    "jswsteel": "JSWSTEEL.NS", "jsw steel": "JSWSTEEL.NS",

    # Insurance
    "lic": "LICI.NS", "lic of india": "LICI.NS",

    # Indices (common references)
    "nifty": "^NSEI", "nifty 50": "^NSEI",
    "sensex": "^BSESN",
    "bank nifty": "^NSEBANK", "banknifty": "^NSEBANK",
    "india vix": "^INDIAVIX",
}

# Indian popular tickers for quick validation
INDIAN_POPULAR_TICKERS: set[str] = {
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "ONGC.NS",
    "NESTLEIND.NS", "ULTRACEMCO.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "BAJAJFINSV.NS", "M&M.NS", "DRREDDY.NS", "CIPLA.NS", "BRITANNIA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "GRASIM.NS", "INDUSINDBK.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "JSWSTEEL.NS", "HINDALCO.NS",
    "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "TATACONSUM.NS",
    "TATAPOWER.NS", "ADANIGREEN.NS", "JIOFIN.NS", "LICI.NS",
    "LTIM.NS", "GODREJCP.NS", "DABUR.NS", "MARICO.NS",
    # Indices
    "^NSEI", "^BSESN", "^NSEBANK", "^INDIAVIX",
}

# Indian ticker to company name mapping
INDIAN_TICKER_TO_COMPANY: dict[str, str] = {
    "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys", "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever", "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel", "ITC.NS": "ITC", "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro", "AXISBANK.NS": "Axis Bank",
    "ASIANPAINT.NS": "Asian Paints", "MARUTI.NS": "Maruti Suzuki",
    "TITAN.NS": "Titan Company", "SUNPHARMA.NS": "Sun Pharmaceutical",
    "BAJFINANCE.NS": "Bajaj Finance", "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies", "TECHM.NS": "Tech Mahindra",
    "NTPC.NS": "NTPC", "POWERGRID.NS": "Power Grid Corporation",
    "TATAMOTORS.NS": "Tata Motors", "TATASTEEL.NS": "Tata Steel",
    "ONGC.NS": "Oil and Natural Gas Corporation", "NESTLEIND.NS": "Nestle India",
    "ULTRACEMCO.NS": "UltraTech Cement", "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports", "BAJAJFINSV.NS": "Bajaj Finserv",
    "M&M.NS": "Mahindra & Mahindra", "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "CIPLA.NS": "Cipla", "BRITANNIA.NS": "Britannia Industries",
    "COALINDIA.NS": "Coal India", "DIVISLAB.NS": "Divi's Laboratories",
    "GRASIM.NS": "Grasim Industries", "INDUSINDBK.NS": "IndusInd Bank",
    "HDFCLIFE.NS": "HDFC Life Insurance", "SBILIFE.NS": "SBI Life Insurance",
    "JSWSTEEL.NS": "JSW Steel", "HINDALCO.NS": "Hindalco Industries",
    "BAJAJ-AUTO.NS": "Bajaj Auto", "HEROMOTOCO.NS": "Hero MotoCorp",
    "EICHERMOT.NS": "Eicher Motors", "TATACONSUM.NS": "Tata Consumer Products",
    "TATAPOWER.NS": "Tata Power", "ADANIGREEN.NS": "Adani Green Energy",
    "JIOFIN.NS": "Jio Financial Services", "LICI.NS": "Life Insurance Corporation",
    "LTIM.NS": "LTIMindtree", "GODREJCP.NS": "Godrej Consumer Products",
    "DABUR.NS": "Dabur India", "MARICO.NS": "Marico",
    "IOC.NS": "Indian Oil Corporation", "BPCL.NS": "Bharat Petroleum",
}

# Indian industry keywords for news service
INDIAN_INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "RELIANCE.NS": ["Jio", "Reliance Retail", "oil refinery", "telecom India", "Mukesh Ambani"],
    "TCS.NS": ["IT services India", "Tata Group", "outsourcing"],
    "HDFCBANK.NS": ["private banking India", "HDFC merger", "retail banking"],
    "INFY.NS": ["IT services", "Infosys consulting", "Bangalore tech"],
    "ICICIBANK.NS": ["ICICI", "private bank India"],
    "SBIN.NS": ["public sector bank", "SBI", "government bank India"],
    "BHARTIARTL.NS": ["Airtel", "telecom India", "5G India"],
    "ITC.NS": ["FMCG India", "cigarettes", "hotels India"],
    "TATAMOTORS.NS": ["Jaguar Land Rover", "JLR", "EV India", "Tata Group"],
    "TATASTEEL.NS": ["steel India", "Tata Group", "metals India"],
    "ADANIENT.NS": ["Adani Group", "Gautam Adani", "infrastructure India"],
    "MARUTI.NS": ["Suzuki India", "automobile India", "car sales India"],
    "BAJFINANCE.NS": ["consumer lending", "NBFC India", "Bajaj Group"],
    "SUNPHARMA.NS": ["pharma India", "generics", "pharmaceutical"],
    "LT.NS": ["infrastructure India", "construction India", "engineering India"],
    "HINDUNILVR.NS": ["FMCG India", "Unilever", "consumer goods India"],
}


def get_market_config(market: str) -> MarketConfig:
    """Get config for the specified market, defaulting to US."""
    return MARKET_CONFIGS.get(market, MARKET_CONFIGS["US"])
