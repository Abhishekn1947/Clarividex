"""
Prediction Engine - Smart AI reasoning with multiple model support.

Architecture:
1. Claude API (Primary) - Best quality, requires API key with credits
2. Ollama Local (Fallback) - Good quality, free, requires local setup
3. Rule-based (Last resort) - Always works, basic quality

The engine automatically selects the best available option.
"""

import hashlib
import json
import re
import time
import uuid
from datetime import datetime
from typing import Optional

import anthropic
import structlog

from backend.app.config import settings
from backend.app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionType,
    ConfidenceLevel,
    SentimentType,
    InstrumentType,
    ReasoningChain,
    Factor,
    Catalyst,
    DataSource,
    DataLimitation,
    DecisionTrail,
)
from backend.app.services.data_aggregator import DataAggregator, AggregatedData
from backend.app.services.offline_model import offline_model_service
from backend.app.services.instrument_detector import instrument_detector
from backend.app.services.decision_trail_builder import create_decision_trail_builder
from backend.app.services.historical_news_analyzer import historical_news_analyzer
from backend.app.services.pattern_recognition import pattern_recognition_engine
from backend.app.services.enhanced_probability_engine import enhanced_probability_engine

logger = structlog.get_logger()


class PredictionEngine:
    """
    Intelligent prediction engine with multi-model support.

    Features:
    - Automatic model selection (Claude -> Ollama -> Rule-based)
    - Smart prompt engineering
    - Structured output parsing
    - Confidence calibration
    """

    @property
    def system_prompt(self) -> str:
        """Get the active system prompt, falling back to hardcoded version."""
        try:
            from backend.app.prompts.registry import get_active_prompt
            prompt = get_active_prompt("prediction_engine")
            if prompt:
                return prompt
        except Exception:
            pass
        return self._FALLBACK_SYSTEM_PROMPT

    _FALLBACK_SYSTEM_PROMPT = """You are an elite quantitative financial analyst AI with expertise in technical analysis, sentiment analysis, options flow, and market dynamics. Analyze ALL provided market data comprehensively to generate an accurate probability assessment.

CRITICAL CONSTRAINT: You ONLY analyze financial market instruments including:
- Stocks (AAPL, NVDA, TSLA, etc.)
- Cryptocurrencies (Bitcoin, Ethereum, etc.)
- Forex pairs (EUR/USD, GBP/JPY, etc.)
- Commodities (Gold, Oil, Silver, etc.)
- Indices (S&P 500, Nasdaq, Dow Jones, etc.)
- ETFs (SPY, QQQ, etc.)
- Futures (ES, NQ, etc.)
- Bonds and Treasuries

If a query is NOT about financial markets, respond ONLY with this JSON:
{"error": "non_financial_query", "message": "I can only analyze financial market predictions. Please ask about stocks, crypto, forex, commodities, or other financial instruments."}

ANALYSIS FRAMEWORK:
1. TECHNICAL ANALYSIS (25% weight):
   - RSI: <30 oversold (bullish), >70 overbought (bearish)
   - MACD: Positive crossover (bullish), negative crossover (bearish)
   - Moving averages: Price above SMA (bullish), below SMA (bearish)
   - Support/Resistance levels and price action

2. SENTIMENT ANALYSIS (20% weight):
   - News sentiment: Aggregate from multiple sources
   - Social media: Reddit, StockTwits sentiment scores
   - Fear & Greed Index: <25 fear (contrarian bullish), >75 greed (contrarian bearish)

3. OPTIONS FLOW (15% weight):
   - Put/Call ratio: <0.7 bullish, >1.3 bearish
   - Unusual options activity signals

4. FUNDAMENTAL DATA (20% weight):
   - Analyst price targets and consensus
   - P/E ratio vs sector average
   - Revenue/earnings growth trends
   - Insider buying/selling patterns

5. MARKET CONDITIONS (20% weight):
   - VIX level: <15 calm, >25 elevated fear
   - Sector performance relative to market
   - Economic indicators (Treasury yields, etc.)

PROBABILITY CALCULATION GUIDELINES:
- Start with 50% base probability
- Adjust +/- 10-15% for strong directional signals in each category
- Cap adjustments: max 85% (very confident), min 15% (very unlikely)
- Increase confidence if multiple signals align
- Decrease confidence if signals conflict

OUTPUT FORMAT (respond with valid JSON only):
{
    "probability": <15-85>,
    "confidence": "<very_low|low|medium|medium_high|high|very_high>",
    "summary": "<2-3 sentence executive summary citing specific data>",
    "bullish_factors": [
        {"description": "<specific factor with exact data point>", "weight": <0.1-1.0>, "source": "<data source>"}
    ],
    "bearish_factors": [
        {"description": "<specific factor with exact data point>", "weight": <0.1-1.0>, "source": "<data source>"}
    ],
    "catalysts": [
        {"event": "<upcoming event>", "date": "<date or null>", "impact": "<positive|negative|uncertain>", "importance": "<low|medium|high>"}
    ],
    "risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
    "assumptions": ["<assumption 1>", "<assumption 2>"],
    "sentiment_assessment": "<very_bearish|bearish|neutral|bullish|very_bullish>",
    "technical_assessment": "<strong_sell|sell|neutral|buy|strong_buy>",
    "fundamental_assessment": "<weak|neutral|strong>"
}"""

    def __init__(self):
        """Initialize prediction engine with available AI backends."""
        self.logger = logger.bind(service="prediction_engine")
        self.data_aggregator = DataAggregator()

        # Initialize Claude client
        self.claude_client = None
        self.claude_available = False
        if settings.has_anthropic_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                self.claude_available = True
                self.logger.info("Claude API client initialized")
            except Exception as e:
                self.logger.warning("Failed to initialize Claude client", error=str(e))

        # Offline model will be checked on first use
        self.offline_checked = False
        self.offline_available = False

        # Prediction result cache: {cache_key: (timestamp, PredictionResponse)}
        self._prediction_cache: dict[str, tuple[float, PredictionResponse]] = {}
        self._cache_max_size = 50
        self._cache_ttl = settings.prediction_cache_hours * 3600

    # ------------------------------------------------------------------
    # Prediction cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, ticker: str, query: str) -> str:
        """Generate cache key from ticker, query, and current date."""
        normalized = f"{ticker.upper()}:{query.lower().strip()}:{datetime.now().strftime('%Y-%m-%d')}"
        return hashlib.md5(normalized.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[PredictionResponse]:
        """Return cached prediction if still valid, else None."""
        if key in self._prediction_cache:
            ts, response = self._prediction_cache[key]
            if time.time() - ts < self._cache_ttl:
                return response
            del self._prediction_cache[key]
        return None

    def _set_cache(self, key: str, response: PredictionResponse) -> None:
        """Store prediction in cache, evicting oldest entry if at capacity."""
        if len(self._prediction_cache) >= self._cache_max_size:
            oldest_key = min(self._prediction_cache, key=lambda k: self._prediction_cache[k][0])
            del self._prediction_cache[oldest_key]
        self._prediction_cache[key] = (time.time(), response)

    async def generate_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate a prediction for the user's query (full pipeline).

        Extracts ticker, aggregates data, then generates the prediction.
        Uses cache when available.

        Args:
            request: PredictionRequest with query and options

        Returns:
            PredictionResponse with analysis
        """
        self.logger.info("Generating prediction", query=request.query[:50])

        # Extract ticker
        ticker = request.ticker
        if not ticker:
            from backend.app.services.market_data import market_data_service
            ticker = market_data_service.extract_ticker_from_query(request.query)

        if not ticker:
            raise ValueError(
                "Could not identify a stock ticker in your query. "
                "Please include a ticker symbol (e.g., AAPL) or company name (e.g., Apple)."
            )

        # Check cache before expensive data aggregation
        cache_key = self._cache_key(ticker, request.query)
        cached = self._get_cached(cache_key)
        if cached:
            self.logger.info("Returning cached prediction", ticker=ticker)
            return cached

        # Aggregate all market data
        data = await self.data_aggregator.aggregate_data(
            ticker=ticker,
            include_technicals=request.include_technicals,
            include_news=request.include_news,
            include_social=request.include_sentiment,
        )

        if not data.quote:
            raise ValueError(f"Could not fetch market data for {ticker}. Please verify the ticker symbol.")

        response = await self._generate_prediction_from_data(request, data, ticker)

        # Cache the result
        self._set_cache(cache_key, response)

        return response

    async def generate_prediction_with_data(
        self, request: PredictionRequest, data: AggregatedData, ticker: Optional[str] = None,
    ) -> PredictionResponse:
        """
        Generate a prediction using pre-aggregated data (skips data fetch).

        Use this when data has already been fetched (e.g. by the SSE stream service)
        to avoid redundant API calls.

        Args:
            request: PredictionRequest with query and options
            data: Pre-aggregated market data
            ticker: Optional ticker (extracted from request/query if not provided)

        Returns:
            PredictionResponse with analysis
        """
        self.logger.info("Generating prediction with pre-aggregated data", query=request.query[:50])

        if not ticker:
            ticker = request.ticker
        if not ticker:
            from backend.app.services.market_data import market_data_service
            ticker = market_data_service.extract_ticker_from_query(request.query)
        if not ticker:
            raise ValueError(
                "Could not identify a stock ticker in your query. "
                "Please include a ticker symbol (e.g., AAPL) or company name (e.g., Apple)."
            )

        if not data.quote:
            raise ValueError(f"Could not fetch market data for {ticker}. Please verify the ticker symbol.")

        # Check cache
        cache_key = self._cache_key(ticker, request.query)
        cached = self._get_cached(cache_key)
        if cached:
            self.logger.info("Returning cached prediction", ticker=ticker)
            return cached

        response = await self._generate_prediction_from_data(request, data, ticker)

        # Cache the result
        self._set_cache(cache_key, response)

        return response

    async def _generate_prediction_from_data(
        self, request: PredictionRequest, data: AggregatedData, ticker: str,
    ) -> PredictionResponse:
        """Core prediction logic using pre-aggregated data."""
        # Parse target price and date
        target_price = request.target_price or self._extract_price_from_query(request.query)
        target_date = request.target_date or self._extract_date_from_query(request.query)

        # Determine prediction type
        prediction_type = self._determine_prediction_type(request.query, target_price)

        # Generate analysis using best available AI
        analysis, model_used = await self._generate_analysis(
            query=request.query,
            data=data,
            target_price=target_price,
            target_date=target_date,
            ticker=ticker,
        )

        # Build response
        response = self._build_response(
            request=request,
            ticker=ticker,
            data=data,
            analysis=analysis,
            prediction_type=prediction_type,
            target_price=target_price,
            target_date=target_date,
            model_used=model_used,
        )

        self.logger.info(
            "Prediction complete",
            ticker=ticker,
            probability=f"{response.probability:.0%}",
            model=model_used,
        )

        return response

    async def _generate_analysis(
        self,
        query: str,
        data: AggregatedData,
        target_price: Optional[float],
        target_date: Optional[datetime],
        ticker: Optional[str] = None,
    ) -> tuple[dict, str]:
        """
        Generate analysis using the best available AI model.

        Returns:
            Tuple of (analysis_dict, model_name)
        """
        context = data.to_context_string()

        # Try Claude first
        if self.claude_available and self.claude_client:
            try:
                analysis = await self._call_claude(query, context, target_price, target_date)
                if analysis:
                    return analysis, "claude"
            except anthropic.APIStatusError as e:
                if "credit balance" in str(e).lower():
                    self.logger.warning("Claude API: insufficient credits")
                    self.claude_available = False  # Don't try again this session
                else:
                    self.logger.warning("Claude API error", error=str(e))
            except Exception as e:
                self.logger.warning("Claude failed", error=str(e))

        # Try Ollama offline model
        if not self.offline_checked:
            self.offline_available = await offline_model_service.check_availability()
            self.offline_checked = True

        if self.offline_available:
            try:
                analysis = await offline_model_service.generate_analysis(
                    query=query,
                    context_data=context,
                    target_price=target_price,
                    target_date=target_date,
                )
                if analysis:
                    self.logger.info("Using Ollama offline model")
                    return analysis, f"ollama:{offline_model_service.selected_model}"
            except Exception as e:
                self.logger.warning("Ollama failed", error=str(e))

        # Fallback to rule-based analysis
        self.logger.info("Using rule-based analysis (no AI available)")
        analysis = self._generate_rule_based_analysis(data, target_price, target_date, ticker=ticker)
        return analysis, "rule_based"

    async def _call_claude(
        self,
        query: str,
        context: str,
        target_price: Optional[float],
        target_date: Optional[datetime],
    ) -> Optional[dict]:
        """Call Claude API for analysis."""
        prompt = self._build_prompt(query, context, target_price, target_date)

        self.logger.debug("Calling Claude API")

        message = self.claude_client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text
        return self._parse_ai_response(response_text)

    def _build_prompt(
        self,
        query: str,
        context: str,
        target_price: Optional[float],
        target_date: Optional[datetime],
    ) -> str:
        """Build optimized prompt for AI analysis."""
        parts = [
            "PREDICTION REQUEST",
            "=" * 50,
            f"Question: {query}",
        ]

        if target_price:
            parts.append(f"Target Price: ${target_price:.2f}")
        if target_date:
            parts.append(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
            days = (target_date - datetime.now()).days
            parts.append(f"Days Until Target: {days}")

        parts.extend([
            "",
            "MARKET DATA",
            "=" * 50,
            context,
            "",
            "INSTRUCTIONS",
            "=" * 50,
            "Analyze ALL the data above and respond with a JSON object.",
            "Be specific - cite exact numbers from the data for each factor.",
        ])

        return "\n".join(parts)

    def _parse_ai_response(self, response_text: str) -> Optional[dict]:
        """Parse AI response with multiple fallback strategies."""
        if not response_text:
            return None

        response_text = response_text.strip()

        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        self.logger.warning("Failed to parse AI response as JSON")
        return None

    def _generate_rule_based_analysis(
        self,
        data: AggregatedData,
        target_price: Optional[float],
        target_date: Optional[datetime],
        ticker: Optional[str] = None,
    ) -> dict:
        """
        Generate comprehensive analysis using rule-based logic with all data sources.

        Uses weighted scoring across 8 factors:
        - Technical indicators (20%)
        - News sentiment (15%)
        - Historical news impact (10%) - NEW
        - Options data (12%)
        - Market sentiment (12%)
        - Analyst data (13%)
        - Social sentiment (8%)
        - Historical patterns (10%) - NEW
        """
        # Initialize decision trail builder for transparency
        trail_builder = create_decision_trail_builder()

        # Collect weighted signals
        signals = []
        bullish_factors = []
        bearish_factors = []
        prob_confidence = 0.5  # Default confidence, will be updated by probability engine

        # ===== 1. TECHNICAL ANALYSIS (20% weight) =====
        tech_signal = data.get_technical_signal()
        tech_score = tech_signal.get("normalized_score", 0)
        signals.append(("technical", tech_score, 0.20))

        # Add to decision trail
        if data.technicals and data.technicals.rsi_14:
            rsi = data.technicals.rsi_14
            rsi_signal = "bullish" if rsi < 40 else "bearish" if rsi > 60 else "neutral"
            rsi_score = (50 - rsi) / 50 if rsi < 50 else (50 - rsi) / 50  # Oversold = positive
            trail_builder.add_factor(
                category="technical",
                source="RSI (14-day)",
                data_point="Relative Strength Index",
                raw_value=f"{rsi:.1f}",
                signal=rsi_signal,
                score=max(-1, min(1, rsi_score)),
                reasoning=f"RSI at {rsi:.1f} - {'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral range'}",
            )

        if data.technicals:
            if data.technicals.rsi_14:
                rsi = data.technicals.rsi_14
                if rsi < 30:
                    bullish_factors.append({
                        "description": f"RSI at {rsi:.1f} indicates oversold conditions - historically leads to bounces",
                        "weight": 0.85,
                        "source": "Technical Analysis (RSI)",
                    })
                elif rsi < 40:
                    bullish_factors.append({
                        "description": f"RSI at {rsi:.1f} approaching oversold territory",
                        "weight": 0.5,
                        "source": "Technical Analysis (RSI)",
                    })
                elif rsi > 70:
                    bearish_factors.append({
                        "description": f"RSI at {rsi:.1f} indicates overbought conditions - potential pullback risk",
                        "weight": 0.85,
                        "source": "Technical Analysis (RSI)",
                    })
                elif rsi > 60:
                    bullish_factors.append({
                        "description": f"RSI at {rsi:.1f} shows strong momentum but not yet overbought",
                        "weight": 0.4,
                        "source": "Technical Analysis (RSI)",
                    })

            # Moving average analysis
            if data.technicals.sma_20 and data.technicals.sma_50 and data.quote:
                price = data.quote.current_price
                if price > data.technicals.sma_20 > data.technicals.sma_50:
                    bullish_factors.append({
                        "description": f"Price ${price:.2f} above rising moving averages (SMA20: ${data.technicals.sma_20:.2f}, SMA50: ${data.technicals.sma_50:.2f})",
                        "weight": 0.7,
                        "source": "Technical Analysis (Moving Averages)",
                    })
                elif price < data.technicals.sma_20 < data.technicals.sma_50:
                    bearish_factors.append({
                        "description": f"Price ${price:.2f} below declining moving averages (SMA20: ${data.technicals.sma_20:.2f}, SMA50: ${data.technicals.sma_50:.2f})",
                        "weight": 0.7,
                        "source": "Technical Analysis (Moving Averages)",
                    })

        # ===== 2. NEWS SENTIMENT (15% weight) =====
        news_sentiment = data.get_news_sentiment()
        if news_sentiment.get("article_count", 0) > 0:
            news_score = news_sentiment.get("overall_score", 0)
            signals.append(("news", news_score, 0.15))

            # Add to decision trail
            trail_builder.add_factor(
                category="news",
                source="News Aggregator",
                data_point="Current News Sentiment",
                raw_value=f"{news_sentiment.get('bullish_count', 0)} bullish / {news_sentiment.get('bearish_count', 0)} bearish",
                signal="bullish" if news_score > 0.1 else "bearish" if news_score < -0.1 else "neutral",
                score=news_score,
                reasoning=f"News sentiment score: {news_score:.2f} from {news_sentiment.get('article_count', 0)} articles",
            )

            if news_score > 0.2:
                bullish_factors.append({
                    "description": f"News sentiment strongly positive: {news_sentiment['bullish_count']} bullish vs {news_sentiment['bearish_count']} bearish articles (score: {news_score:.2f})",
                    "weight": min(abs(news_score) + 0.3, 1.0),
                    "source": "News Analysis",
                })
            elif news_score > 0.1:
                bullish_factors.append({
                    "description": f"News sentiment positive: {news_sentiment['bullish_count']} bullish articles",
                    "weight": abs(news_score) + 0.2,
                    "source": "News Analysis",
                })
            elif news_score < -0.2:
                bearish_factors.append({
                    "description": f"News sentiment strongly negative: {news_sentiment['bearish_count']} bearish vs {news_sentiment['bullish_count']} bullish articles (score: {news_score:.2f})",
                    "weight": min(abs(news_score) + 0.3, 1.0),
                    "source": "News Analysis",
                })
            elif news_score < -0.1:
                bearish_factors.append({
                    "description": f"News sentiment negative: {news_sentiment['bearish_count']} bearish articles",
                    "weight": abs(news_score) + 0.2,
                    "source": "News Analysis",
                })

        # ===== 3. SOCIAL SENTIMENT (8% weight) =====
        social_sentiment = data.get_social_sentiment()
        if social_sentiment.get("total_mentions", 0) > 0:
            social_score = social_sentiment.get("overall_score", 0)
            signals.append(("social", social_score, 0.08))

            # Add to decision trail
            trail_builder.add_factor(
                category="social",
                source="Social Media Aggregator",
                data_point="Social Sentiment",
                raw_value=f"{social_sentiment.get('overall_bullish_pct', 50):.0f}% bullish",
                signal="bullish" if social_score > 0.1 else "bearish" if social_score < -0.1 else "neutral",
                score=social_score,
                reasoning=f"Social media sentiment from {social_sentiment.get('platforms_analyzed', 1)} platforms",
            )

            bullish_pct = social_sentiment.get("overall_bullish_pct", 50)
            if bullish_pct > 65:
                bullish_factors.append({
                    "description": f"Social media strongly bullish: {bullish_pct:.0f}% bullish mentions across {social_sentiment.get('platforms_analyzed', 1)} platforms",
                    "weight": (bullish_pct - 50) / 40,
                    "source": "Social Media Sentiment",
                })
            elif bullish_pct < 35:
                bearish_factors.append({
                    "description": f"Social media bearish: {100 - bullish_pct:.0f}% bearish mentions",
                    "weight": (50 - bullish_pct) / 40,
                    "source": "Social Media Sentiment",
                })

        # ===== 4. OPTIONS DATA (12% weight) =====
        if data.options_data and not data.options_data.get("error"):
            pc_ratio = data.options_data.get("put_call_ratio_volume", 1.0)
            if pc_ratio:
                options_score = (1 - pc_ratio) / 2  # Convert P/C to bullish score
                options_score = max(-1, min(1, options_score))
                signals.append(("options", options_score, 0.12))

                # Add to decision trail
                trail_builder.add_factor(
                    category="options",
                    source="Options Flow Data",
                    data_point="Put/Call Ratio",
                    raw_value=f"{pc_ratio:.2f}",
                    signal="bullish" if pc_ratio < 0.8 else "bearish" if pc_ratio > 1.2 else "neutral",
                    score=options_score,
                    reasoning=f"P/C ratio {pc_ratio:.2f} - {'call buying dominance' if pc_ratio < 0.8 else 'put buying/hedging' if pc_ratio > 1.2 else 'balanced'}",
                )

                if pc_ratio < 0.6:
                    bullish_factors.append({
                        "description": f"Options flow bullish: Put/Call ratio {pc_ratio:.2f} indicates call buying dominance",
                        "weight": 0.8,
                        "source": "Options Flow",
                    })
                elif pc_ratio < 0.8:
                    bullish_factors.append({
                        "description": f"Options flow slightly bullish: Put/Call ratio {pc_ratio:.2f}",
                        "weight": 0.5,
                        "source": "Options Flow",
                    })
                elif pc_ratio > 1.3:
                    bearish_factors.append({
                        "description": f"Options flow bearish: Put/Call ratio {pc_ratio:.2f} indicates put buying/hedging",
                        "weight": 0.8,
                        "source": "Options Flow",
                    })
                elif pc_ratio > 1.1:
                    bearish_factors.append({
                        "description": f"Options flow slightly bearish: Put/Call ratio {pc_ratio:.2f}",
                        "weight": 0.5,
                        "source": "Options Flow",
                    })

        # ===== 5. MARKET SENTIMENT - VIX & Fear/Greed (12% weight) =====
        market_score = 0
        market_factors = 0

        if data.vix_data and data.vix_data.get("value"):
            vix = data.vix_data["value"]
            # VIX interpretation (contrarian in extremes)
            if vix < 15:
                market_score += 0.3  # Low fear = bullish
                bullish_factors.append({
                    "description": f"VIX at {vix:.1f} indicates calm markets - low volatility environment favorable for stocks",
                    "weight": 0.5,
                    "source": "VIX (Volatility Index)",
                })
            elif vix > 30:
                # High VIX can be contrarian bullish or bearish depending on trend
                bearish_factors.append({
                    "description": f"VIX at {vix:.1f} indicates elevated fear - market uncertainty high",
                    "weight": 0.6,
                    "source": "VIX (Volatility Index)",
                })
                market_score -= 0.3
            elif vix > 25:
                bearish_factors.append({
                    "description": f"VIX at {vix:.1f} above average - indicates market concern",
                    "weight": 0.4,
                    "source": "VIX (Volatility Index)",
                })
                market_score -= 0.15
            market_factors += 1

        if data.fear_greed and data.fear_greed.get("value"):
            fg = data.fear_greed["value"]
            rating = data.fear_greed.get("rating", "neutral")
            # Fear & Greed (contrarian indicator at extremes)
            if fg < 25:
                bullish_factors.append({
                    "description": f"Fear & Greed Index at {fg:.0f} ({rating}) - extreme fear often marks buying opportunities",
                    "weight": 0.6,
                    "source": "CNN Fear & Greed Index",
                })
                market_score += 0.4
            elif fg < 40:
                bullish_factors.append({
                    "description": f"Fear & Greed Index at {fg:.0f} ({rating}) - fear in market can present opportunities",
                    "weight": 0.3,
                    "source": "CNN Fear & Greed Index",
                })
                market_score += 0.2
            elif fg > 75:
                bearish_factors.append({
                    "description": f"Fear & Greed Index at {fg:.0f} ({rating}) - extreme greed often precedes corrections",
                    "weight": 0.6,
                    "source": "CNN Fear & Greed Index",
                })
                market_score -= 0.4
            elif fg > 60:
                bullish_factors.append({
                    "description": f"Fear & Greed Index at {fg:.0f} ({rating}) - greed indicates bullish market sentiment",
                    "weight": 0.3,
                    "source": "CNN Fear & Greed Index",
                })
                market_score += 0.15
            market_factors += 1

        if market_factors > 0:
            avg_market_score = market_score / market_factors
            signals.append(("market_sentiment", avg_market_score, 0.12))

            # Add to decision trail
            trail_builder.add_factor(
                category="market",
                source="Market Indicators",
                data_point="VIX & Fear/Greed",
                raw_value=f"VIX: {data.vix_data.get('value', 'N/A') if data.vix_data else 'N/A'}",
                signal="bullish" if avg_market_score > 0.1 else "bearish" if avg_market_score < -0.1 else "neutral",
                score=avg_market_score,
                reasoning="Combined VIX and Fear & Greed Index analysis",
            )

        # ===== 6. ANALYST DATA - Finviz (13% weight) =====
        if data.finviz_data and not data.finviz_data.get("error"):
            finviz = data.finviz_data

            # Price target analysis
            if finviz.get("target_price") and finviz.get("price"):
                target = finviz["target_price"]
                current = finviz["price"]
                upside = finviz.get("upside_pct", 0)

                analyst_score = 0
                if upside > 20:
                    bullish_factors.append({
                        "description": f"Analyst consensus target ${target:.2f} implies {upside:.1f}% upside from current ${current:.2f}",
                        "weight": 0.85,
                        "source": "Analyst Price Targets (Finviz)",
                    })
                    analyst_score = 0.7
                    signals.append(("analyst", 0.7, 0.13))
                elif upside > 10:
                    bullish_factors.append({
                        "description": f"Analyst target ${target:.2f} implies {upside:.1f}% upside",
                        "weight": 0.6,
                        "source": "Analyst Price Targets (Finviz)",
                    })
                    analyst_score = 0.4
                    signals.append(("analyst", 0.4, 0.13))
                elif upside < -10:
                    bearish_factors.append({
                        "description": f"Analyst target ${target:.2f} implies {upside:.1f}% downside - stock may be overvalued",
                        "weight": 0.7,
                        "source": "Analyst Price Targets (Finviz)",
                    })
                    analyst_score = -0.5
                    signals.append(("analyst", -0.5, 0.13))

                # Add to decision trail
                trail_builder.add_factor(
                    category="analyst",
                    source="Analyst Consensus (Finviz)",
                    data_point="Price Target",
                    raw_value=f"${target:.2f} ({upside:+.1f}%)",
                    signal="bullish" if upside > 10 else "bearish" if upside < -10 else "neutral",
                    score=analyst_score,
                    reasoning=f"Analyst target implies {upside:+.1f}% from current price",
                )

            # Recommendation
            recom = finviz.get("recommendation_text")
            if recom in ["Strong Buy", "Buy"]:
                bullish_factors.append({
                    "description": f"Analyst consensus rating: {recom}",
                    "weight": 0.5,
                    "source": "Analyst Recommendations",
                })
            elif recom in ["Sell", "Strong Sell"]:
                bearish_factors.append({
                    "description": f"Analyst consensus rating: {recom}",
                    "weight": 0.5,
                    "source": "Analyst Recommendations",
                })

            # Short interest
            short_float = finviz.get("short_float")
            if short_float and isinstance(short_float, str):
                try:
                    short_pct = float(short_float.replace("%", ""))
                    if short_pct > 20:
                        # High short interest - potential squeeze but also bearish signal
                        bullish_factors.append({
                            "description": f"High short interest at {short_pct:.1f}% - potential short squeeze candidate",
                            "weight": 0.4,
                            "source": "Short Interest Data",
                        })
                        bearish_factors.append({
                            "description": f"High short interest at {short_pct:.1f}% indicates significant bearish bets",
                            "weight": 0.5,
                            "source": "Short Interest Data",
                        })
                    elif short_pct > 10:
                        bearish_factors.append({
                            "description": f"Elevated short interest at {short_pct:.1f}%",
                            "weight": 0.3,
                            "source": "Short Interest Data",
                        })
                except ValueError:
                    pass

        # ===== 7. HISTORICAL NEWS IMPACT (10% weight) - NEW =====
        # Analyze how similar news events affected the stock historically
        try:
            if ticker and data.news:
                # Convert news to format expected by historical analyzer
                historical_news = [
                    {
                        "date": article.published_at,
                        "title": article.title,
                        "sentiment_score": article.sentiment_score,
                    }
                    for article in data.news
                ]

                # Get historical prices for correlation analysis
                historical_prices = []
                if hasattr(data, 'historical_prices') and data.historical_prices:
                    historical_prices = data.historical_prices

                # Analyze historical news impact
                hist_news_analysis = historical_news_analyzer.analyze_historical_news(
                    ticker=ticker,
                    historical_prices=historical_prices,
                    historical_news=historical_news,
                    current_news_sentiment=news_sentiment.get("overall_score", 0) if news_sentiment else 0,
                    lookback_days=180,
                )

                if hist_news_analysis.signal_score != 0:
                    signals.append(("historical_news", hist_news_analysis.signal_score, 0.10))

                    signal_type = "bullish" if hist_news_analysis.signal_score > 0 else "bearish"
                    if hist_news_analysis.signal_score > 0.2:
                        bullish_factors.append({
                            "description": f"Historical news impact analysis: {hist_news_analysis.reasoning}",
                            "weight": 0.6,
                            "source": "Historical News Correlation",
                        })
                    elif hist_news_analysis.signal_score < -0.2:
                        bearish_factors.append({
                            "description": f"Historical news impact analysis: {hist_news_analysis.reasoning}",
                            "weight": 0.6,
                            "source": "Historical News Correlation",
                        })

                    # Add to decision trail
                    trail_builder.add_factor(
                        category="historical_news",
                        source="Historical News Analyzer",
                        data_point="Past News Impact",
                        raw_value=f"{hist_news_analysis.historical_sentiment_accuracy:.0%} accuracy",
                        signal=signal_type,
                        score=hist_news_analysis.signal_score,
                        reasoning=hist_news_analysis.reasoning,
                    )
        except Exception as e:
            self.logger.warning("Historical news analysis failed", error=str(e))

        # ===== 8. HISTORICAL PATTERNS (10% weight) - NEW =====
        # Find similar historical technical setups and their outcomes
        try:
            if ticker and data.technicals:
                # Build technicals dict
                current_technicals = {
                    "rsi_14": data.technicals.rsi_14,
                    "macd": data.technicals.macd,
                    "macd_signal": data.technicals.macd_signal,
                    "sma_20": data.technicals.sma_20,
                    "sma_50": data.technicals.sma_50,
                    "sma_200": data.technicals.sma_200,
                    "support_level": data.technicals.support_level,
                    "resistance_level": data.technicals.resistance_level,
                    "current_price": data.current_price,
                }

                # Get historical prices
                historical_prices = []
                if hasattr(data, 'historical_prices') and data.historical_prices:
                    historical_prices = data.historical_prices

                # Analyze patterns
                pattern_analysis = pattern_recognition_engine.analyze_patterns(
                    ticker=ticker,
                    current_price=data.current_price or 0,
                    technicals=current_technicals,
                    historical_prices=historical_prices,
                )

                if pattern_analysis.signal_score != 0:
                    signals.append(("historical_patterns", pattern_analysis.signal_score, 0.10))

                    signal_type = "bullish" if pattern_analysis.signal_score > 0 else "bearish"
                    if pattern_analysis.pattern_matches:
                        top_pattern = pattern_analysis.pattern_matches[0]
                        if pattern_analysis.signal_score > 0.2:
                            bullish_factors.append({
                                "description": f"Pattern detected: {top_pattern.description} (Win rate: {top_pattern.historical_win_rate:.0%})",
                                "weight": 0.65,
                                "source": "Historical Pattern Recognition",
                            })
                        elif pattern_analysis.signal_score < -0.2:
                            bearish_factors.append({
                                "description": f"Pattern detected: {top_pattern.description} (Win rate: {top_pattern.historical_win_rate:.0%})",
                                "weight": 0.65,
                                "source": "Historical Pattern Recognition",
                            })

                    # Add to decision trail
                    trail_builder.add_factor(
                        category="historical_patterns",
                        source="Pattern Recognition Engine",
                        data_point="Technical Patterns",
                        raw_value=f"{len(pattern_analysis.pattern_matches)} patterns found",
                        signal=signal_type,
                        score=pattern_analysis.signal_score,
                        reasoning=pattern_analysis.reasoning,
                    )

                    # Add similar scenarios to decision trail
                    if pattern_analysis.similar_scenarios:
                        avg_outcome = sum(s.outcome_20d for s in pattern_analysis.similar_scenarios) / len(pattern_analysis.similar_scenarios)
                        trail_builder.add_factor(
                            category="historical_patterns",
                            source="Similar Historical Setups",
                            data_point="Past Scenario Outcomes",
                            raw_value=f"{len(pattern_analysis.similar_scenarios)} similar setups",
                            signal="bullish" if avg_outcome > 0 else "bearish",
                            score=max(-1, min(1, avg_outcome / 10)),
                            reasoning=f"Similar setups had avg {avg_outcome:+.1f}% outcome over 20 days",
                        )
        except Exception as e:
            self.logger.warning("Pattern recognition failed", error=str(e))

        # ===== CALCULATE WEIGHTED PROBABILITY USING ADVANCED ENGINE =====
        # Collect signal scores for Bayesian analysis
        signal_scores = {}
        if signals:
            for name, score, weight in signals:
                signal_scores[name] = score

            # Calculate weighted average for sentiment
            total_weight = sum(weight for _, _, weight in signals)
            normalized_signals = [(name, score, weight / total_weight) for name, score, weight in signals]
            avg_signal = sum(score * weight for _, score, weight in normalized_signals)
        else:
            avg_signal = 0

        # Calculate days until target
        days_until = 90  # Default
        if target_date:
            days_until = max(1, (target_date - datetime.now()).days)

        # Use the advanced probability engine for accurate predictions
        if target_price and data.current_price and data.current_price > 0:
            # Prepare technicals dict
            technicals_dict = None
            if data.technicals:
                technicals_dict = {
                    "rsi_14": data.technicals.rsi_14,
                    "macd": data.technicals.macd,
                    "macd_signal": data.technicals.macd_signal,
                    "sma_20": data.technicals.sma_20,
                    "sma_50": data.technicals.sma_50,
                    "sma_200": data.technicals.sma_200,
                    "current_price": data.current_price,
                }

            # Get news sentiment
            news_sent = news_sentiment.get("overall_score", 0) if news_sentiment else None

            # Get historical prices
            historical_prices = data.historical_prices if hasattr(data, 'historical_prices') else []

            # Calculate probability using advanced engine
            # Get earnings history and sector for improved predictions
            earnings_history = data.earnings_history if hasattr(data, 'earnings_history') else None
            sector = data.company_info.sector if data.company_info and hasattr(data.company_info, 'sector') else None

            prob_result = enhanced_probability_engine.calculate_probability(
                current_price=data.current_price,
                target_price=target_price,
                days_to_target=days_until,
                historical_prices=historical_prices,
                signals=signal_scores,
                technicals=technicals_dict,
                news_sentiment=news_sent,
                options_data=data.options_data if hasattr(data, 'options_data') else None,
                # New parameters for improved predictions
                ticker=data.ticker,  # For per-ticker tuning
                earnings_history=earnings_history,  # For earnings surprise analysis
                sector=sector,  # For sector-based predictions
            )

            probability = prob_result.probability
            prob_confidence = prob_result.confidence

            # Log the probability calculation details
            self.logger.info(
                "Advanced probability calculated",
                probability=probability,
                confidence=prob_confidence,
                method=prob_result.method,
                components=prob_result.components,
                historical_data_points=len(historical_prices),
            )

            # Add probability engine result to decision trail
            trail_builder.add_factor(
                category="probability_engine",
                source="Advanced Probability Engine",
                data_point="Monte Carlo + Bayesian",
                raw_value=f"{probability:.0f}%",
                signal="bullish" if probability > 55 else "bearish" if probability < 45 else "neutral",
                score=(probability - 50) / 50,  # Normalize to -1 to 1
                reasoning=prob_result.reasoning,
            )
        else:
            # Fallback to simple calculation if no target price
            base_probability = 50 + (avg_signal * 35)
            probability = max(15, min(85, base_probability))
            prob_confidence = 0.5

        # ===== DETERMINE CONFIDENCE =====
        # Based on: probability engine confidence, data quality, signal agreement
        if signals:
            signal_scores_list = [s for _, s, _ in signals]
            signal_std = (sum((s - avg_signal) ** 2 for s in signal_scores_list) / len(signal_scores_list)) ** 0.5
            signal_agreement = max(0, 1 - signal_std * 2)  # Higher std = less agreement
        else:
            signal_agreement = 0.5

        # Factor in data sources
        data_source_bonus = min(len(data.data_sources) / 12, 1)  # Max bonus at 12 sources

        # Incorporate probability engine confidence
        confidence_score = (
            prob_confidence * 0.35 +  # Probability engine confidence
            data.data_quality_score * 0.25 +
            signal_agreement * 0.25 +
            data_source_bonus * 0.15
        )

        if confidence_score >= 0.85:
            confidence = "high"
        elif confidence_score >= 0.7:
            confidence = "medium_high"
        elif confidence_score >= 0.5:
            confidence = "medium"
        elif confidence_score >= 0.35:
            confidence = "low"
        else:
            confidence = "very_low"

        # ===== DETERMINE OVERALL SENTIMENT =====
        if avg_signal > 0.4:
            sentiment = "very_bullish"
        elif avg_signal > 0.15:
            sentiment = "bullish"
        elif avg_signal < -0.4:
            sentiment = "very_bearish"
        elif avg_signal < -0.15:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        # ===== BUILD SUMMARY =====
        summary_parts = []

        # Lead with the key finding
        if probability >= 65:
            summary_parts.append(f"Analysis indicates {probability:.0f}% probability with {sentiment} outlook")
        elif probability <= 35:
            summary_parts.append(f"Analysis indicates low {probability:.0f}% probability")
        else:
            summary_parts.append(f"Analysis suggests mixed signals with {probability:.0f}% probability")

        # Add data context
        summary_parts.append(f"Based on {data.data_points_count} data points from {len(data.data_sources)} sources")

        # Add key driver
        if bullish_factors and (not bearish_factors or len(bullish_factors) > len(bearish_factors)):
            key_driver = bullish_factors[0]["description"][:80]
            summary_parts.append(f"Key bullish factor: {key_driver}")
        elif bearish_factors:
            key_driver = bearish_factors[0]["description"][:80]
            summary_parts.append(f"Key bearish factor: {key_driver}")

        # ===== BUILD DECISION TRAIL =====
        # Calculate probability through trail builder for consistency
        days_until = None
        if target_date:
            days_until = (target_date - datetime.now()).days

        trail_builder.calculate_probability(
            target_price=target_price,
            current_price=data.current_price,
            days_until_target=days_until,
        )

        # Build the final decision trail
        decision_trail = trail_builder.build_decision_trail()

        # Generate ASCII visualization for logging
        try:
            ascii_tree = trail_builder.generate_ascii_tree()
            self.logger.debug("Decision trail generated", ascii_tree=ascii_tree[:500])
        except Exception:
            pass

        return {
            "probability": round(probability),
            "confidence": confidence,
            "summary": ". ".join(summary_parts) + ".",
            "bullish_factors": bullish_factors,
            "bearish_factors": bearish_factors,
            "catalysts": self._extract_catalysts(data),
            "risks": [
                "Market volatility may cause unexpected price movements",
                "Unforeseen news events could change the outlook",
                "This analysis is based on current data which may change",
            ],
            "assumptions": [
                "Current market conditions persist",
                "No major unexpected events occur",
                "Historical patterns continue to hold",
            ],
            "sentiment_assessment": sentiment,
            "technical_assessment": tech_signal.get("overall_signal", "neutral"),
            "fundamental_assessment": "neutral",
            "decision_trail": decision_trail,  # NEW: Full decision trail for transparency
        }

    def _build_decision_trail_from_data(
        self,
        data: AggregatedData,
        target_price: Optional[float],
        target_date: Optional[datetime],
    ) -> "DecisionTrail":
        """
        Build a decision trail from raw data for transparency.

        This is used when AI models don't provide their own decision trail,
        ensuring users always see how factors contributed to the prediction.
        """
        trail_builder = create_decision_trail_builder()

        # 1. Technical Analysis (20% weight)
        if data.technicals:
            if data.technicals.rsi_14 is not None:
                rsi = data.technicals.rsi_14
                rsi_signal = "bullish" if rsi < 40 else "bearish" if rsi > 60 else "neutral"
                rsi_score = (50 - rsi) / 50
                trail_builder.add_factor(
                    category="technical",
                    source="RSI (14-day)",
                    data_point="Relative Strength Index",
                    raw_value=f"{rsi:.1f}",
                    signal=rsi_signal,
                    score=max(-1, min(1, rsi_score)),
                    reasoning=f"RSI at {rsi:.1f} - {'oversold (bullish)' if rsi < 30 else 'overbought (bearish)' if rsi > 70 else 'neutral range'}",
                )

            if data.technicals.macd is not None:
                macd = data.technicals.macd
                macd_signal_val = data.technicals.macd_signal or 0
                macd_diff = macd - macd_signal_val
                signal = "bullish" if macd_diff > 0 else "bearish" if macd_diff < 0 else "neutral"
                score = max(-1, min(1, macd_diff / 5))
                trail_builder.add_factor(
                    category="technical",
                    source="MACD",
                    data_point="MACD vs Signal Line",
                    raw_value=f"{macd:.2f} vs {macd_signal_val:.2f}",
                    signal=signal,
                    score=score,
                    reasoning=f"MACD {'above' if macd_diff > 0 else 'below'} signal line by {abs(macd_diff):.2f}",
                )

            if data.technicals.sma_20 and data.technicals.sma_50:
                sma20 = data.technicals.sma_20
                sma50 = data.technicals.sma_50
                signal = "bullish" if sma20 > sma50 else "bearish"
                score = 0.5 if sma20 > sma50 else -0.5
                trail_builder.add_factor(
                    category="technical",
                    source="Moving Averages",
                    data_point="SMA20 vs SMA50",
                    raw_value=f"SMA20: ${sma20:.2f}, SMA50: ${sma50:.2f}",
                    signal=signal,
                    score=score,
                    reasoning=f"Short-term MA {'above' if sma20 > sma50 else 'below'} long-term MA - {'bullish trend' if sma20 > sma50 else 'bearish trend'}",
                )

        # 2. News Sentiment (15% weight)
        news_sentiment = data.get_news_sentiment()
        if news_sentiment.get("overall_score") is not None:
            score = news_sentiment["overall_score"]
            signal = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
            trail_builder.add_factor(
                category="news",
                source="News Aggregator",
                data_point="News Sentiment Score",
                raw_value=f"{score:.2f} ({len(data.news)} articles)",
                signal=signal,
                score=score,
                reasoning=f"Aggregated news sentiment: {score:.2f} from {len(data.news)} recent articles",
            )

        # 3. Options Flow (12% weight)
        if data.options_data and not data.options_data.get("error"):
            pcr = data.options_data.get("put_call_ratio")
            if pcr is not None:
                signal = "bullish" if pcr < 0.7 else "bearish" if pcr > 1.0 else "neutral"
                score = (1.0 - pcr) * 0.8  # Lower PCR = more bullish
                trail_builder.add_factor(
                    category="options",
                    source="Options Data",
                    data_point="Put/Call Ratio",
                    raw_value=f"{pcr:.2f}",
                    signal=signal,
                    score=max(-1, min(1, score)),
                    reasoning=f"Put/Call ratio of {pcr:.2f} indicates {'bullish' if pcr < 0.7 else 'bearish' if pcr > 1.0 else 'neutral'} options sentiment",
                )

        # 4. Market Sentiment (12% weight)
        if data.vix_data and data.vix_data.get("value") is not None:
            vix = data.vix_data["value"]
            signal = "bullish" if vix < 18 else "bearish" if vix > 25 else "neutral"
            score = (20 - vix) / 20  # Lower VIX = more bullish
            trail_builder.add_factor(
                category="market",
                source="VIX Index",
                data_point="Volatility Index",
                raw_value=f"{vix:.1f}",
                signal=signal,
                score=max(-1, min(1, score)),
                reasoning=f"VIX at {vix:.1f} indicates {'low fear' if vix < 18 else 'elevated fear' if vix > 25 else 'normal'} market conditions",
            )

        if data.fear_greed and data.fear_greed.get("value") is not None:
            fgi = data.fear_greed["value"]
            # Contrarian: extreme fear = bullish, extreme greed = bearish
            signal = "bullish" if fgi < 30 else "bearish" if fgi > 70 else "neutral"
            score = (50 - fgi) / 50
            trail_builder.add_factor(
                category="market",
                source="Fear & Greed Index",
                data_point="Market Sentiment",
                raw_value=f"{fgi}",
                signal=signal,
                score=max(-1, min(1, score)),
                reasoning=f"Fear & Greed at {fgi} - {'extreme fear (contrarian bullish)' if fgi < 25 else 'extreme greed (contrarian bearish)' if fgi > 75 else 'neutral'}",
            )

        # 5. Analyst Ratings (13% weight)
        if data.finviz_data and not data.finviz_data.get("error"):
            # Get analyst recommendation
            rec = data.finviz_data.get("recommendation", "").lower()
            score_map = {"strong buy": 0.8, "buy": 0.5, "hold": 0, "sell": -0.5, "strong sell": -0.8}
            if rec in score_map:
                score = score_map[rec]
                signal = "bullish" if score > 0.3 else "bearish" if score < -0.3 else "neutral"
                trail_builder.add_factor(
                    category="analyst",
                    source="Analyst Ratings",
                    data_point="Consensus Rating",
                    raw_value=rec.title(),
                    signal=signal,
                    score=score,
                    reasoning=f"Analyst consensus: {rec.title()} (score: {score:.2f})",
                )

            # Get target price upside
            target = data.finviz_data.get("target_price")
            if target and data.current_price:
                upside = ((target - data.current_price) / data.current_price) * 100
                signal = "bullish" if upside > 10 else "bearish" if upside < -10 else "neutral"
                score = min(1, max(-1, upside / 30))
                trail_builder.add_factor(
                    category="analyst",
                    source="Price Targets",
                    data_point="Target Upside",
                    raw_value=f"{upside:+.1f}%",
                    signal=signal,
                    score=score,
                    reasoning=f"Analyst price target implies {upside:+.1f}% upside",
                )

        # 6. Social Sentiment (8% weight)
        social_sentiment = data.get_social_sentiment()
        if social_sentiment.get("overall_score") is not None:
            score = social_sentiment["overall_score"]
            signal = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"
            bullish_pct = social_sentiment.get("bullish_percentage", 50)
            trail_builder.add_factor(
                category="social",
                source="Social Media",
                data_point="Social Sentiment",
                raw_value=f"{bullish_pct:.0f}% bullish",
                signal=signal,
                score=score,
                reasoning=f"Social sentiment shows {bullish_pct:.0f}% bullish mentions",
            )

        # Calculate the probability with all factors
        days_until = None
        if target_date:
            days_until = (target_date - datetime.now()).days

        trail_builder.calculate_probability(
            target_price=target_price,
            current_price=data.current_price,
            days_until_target=days_until,
        )

        return trail_builder.build_decision_trail()

    def _extract_catalysts(self, data: AggregatedData) -> list[dict]:
        """Extract upcoming catalysts from data."""
        catalysts = []

        for earning in data.earnings_dates[:2]:
            catalysts.append({
                "event": "Earnings Report",
                "date": str(earning.get("date")) if earning.get("date") else None,
                "impact": "uncertain",
                "importance": "high",
            })

        return catalysts

    def _calculate_data_limitations(self, data: AggregatedData) -> tuple[list[DataLimitation], bool]:
        """
        Calculate data limitations based on available data.

        Returns:
            Tuple of (list of limitations, has_limited_data flag)
        """
        limitations = []
        has_limited_data = False

        # Check data quality score
        if data.data_quality_score < 0.5:
            limitations.append(DataLimitation(
                category="data_quality",
                severity="high",
                message="Low data quality score ({:.0f}%). Prediction reliability may be significantly reduced.".format(
                    data.data_quality_score * 100
                ),
                recommendation="Consider waiting for more data or verify the ticker symbol is correct.",
            ))
            has_limited_data = True
        elif data.data_quality_score < 0.7:
            limitations.append(DataLimitation(
                category="data_quality",
                severity="medium",
                message="Moderate data quality score ({:.0f}%). Some data sources may be unavailable.".format(
                    data.data_quality_score * 100
                ),
                recommendation="Results should be used with caution.",
            ))
            has_limited_data = True

        # Check number of data sources
        source_count = len(data.data_sources)
        if source_count < 5:
            limitations.append(DataLimitation(
                category="sources",
                severity="high",
                message="Limited data sources ({} of 12+ available). Prediction is based on incomplete information.".format(
                    source_count
                ),
                recommendation="Some data feeds may be unavailable. Consider checking API status.",
            ))
            has_limited_data = True
        elif source_count < 8:
            limitations.append(DataLimitation(
                category="sources",
                severity="medium",
                message="Fewer data sources than optimal ({} of 12+).".format(source_count),
                recommendation="Some secondary data sources could not be fetched.",
            ))

        # Check news availability
        if len(data.news) == 0:
            limitations.append(DataLimitation(
                category="coverage",
                severity="medium",
                message="No recent news articles found for this stock.",
                recommendation="News sentiment analysis is unavailable. Prediction relies on other data sources.",
            ))
            has_limited_data = True
        elif len(data.news) < 5:
            limitations.append(DataLimitation(
                category="coverage",
                severity="low",
                message="Limited news coverage ({} articles found).".format(len(data.news)),
                recommendation="News sentiment may not be fully representative.",
            ))

        # Check social media data
        if not data.social or len(data.social) == 0:
            limitations.append(DataLimitation(
                category="coverage",
                severity="low",
                message="No social media sentiment data available.",
                recommendation="Social sentiment analysis is unavailable for this prediction.",
            ))

        # Check technical data
        if not data.technicals:
            limitations.append(DataLimitation(
                category="coverage",
                severity="medium",
                message="Technical indicators could not be calculated.",
                recommendation="Technical analysis is unavailable. Prediction relies on fundamental and sentiment data.",
            ))
            has_limited_data = True
        elif data.technicals.rsi_14 is None and data.technicals.macd is None:
            limitations.append(DataLimitation(
                category="coverage",
                severity="low",
                message="Some technical indicators are unavailable.",
                recommendation="Technical analysis may be incomplete.",
            ))

        # Check for missing fundamental data
        if not data.company_info or not data.company_info.market_cap:
            limitations.append(DataLimitation(
                category="coverage",
                severity="low",
                message="Company fundamental data is limited or unavailable.",
                recommendation="Fundamental analysis may be incomplete.",
            ))

        # Check if this is a less-traded/obscure stock
        if data.data_points_count < 50:
            limitations.append(DataLimitation(
                category="data_quality",
                severity="high",
                message="Very limited data points ({}) available for analysis.".format(data.data_points_count),
                recommendation="This may be a thinly traded stock. Predictions for such stocks are less reliable.",
            ))
            has_limited_data = True
        elif data.data_points_count < 100:
            limitations.append(DataLimitation(
                category="data_quality",
                severity="medium",
                message="Limited data points ({}) available for analysis.".format(data.data_points_count),
                recommendation="Some analysis categories may have incomplete data.",
            ))

        # Check VIX/market data
        if not data.vix_data or not data.fear_greed:
            limitations.append(DataLimitation(
                category="coverage",
                severity="low",
                message="Market sentiment indicators (VIX/Fear & Greed) unavailable.",
                recommendation="Market context analysis is limited.",
            ))

        return limitations, has_limited_data

    def _build_response(
        self,
        request: PredictionRequest,
        ticker: str,
        data: AggregatedData,
        analysis: dict,
        prediction_type: PredictionType,
        target_price: Optional[float],
        target_date: Optional[datetime],
        model_used: str,
    ) -> PredictionResponse:
        """Build the final prediction response."""
        # Always build decision trail for transparency (even if AI model was used)
        if not analysis.get("decision_trail"):
            decision_trail = self._build_decision_trail_from_data(data, target_price, target_date)
            analysis["decision_trail"] = decision_trail

        # Map confidence
        confidence_map = {
            "very_low": ConfidenceLevel.VERY_LOW,
            "low": ConfidenceLevel.LOW,
            "medium": ConfidenceLevel.MEDIUM,
            "medium_high": ConfidenceLevel.MEDIUM_HIGH,
            "high": ConfidenceLevel.HIGH,
            "very_high": ConfidenceLevel.VERY_HIGH,
        }
        confidence_level = confidence_map.get(
            analysis.get("confidence", "medium"),
            ConfidenceLevel.MEDIUM,
        )

        # Map sentiment
        sentiment_map = {
            "very_bearish": SentimentType.VERY_BEARISH,
            "bearish": SentimentType.BEARISH,
            "neutral": SentimentType.NEUTRAL,
            "bullish": SentimentType.BULLISH,
            "very_bullish": SentimentType.VERY_BULLISH,
        }
        sentiment = sentiment_map.get(
            analysis.get("sentiment_assessment", "neutral"),
            SentimentType.NEUTRAL,
        )

        # Build factors
        bullish_factors = [
            Factor(
                description=f["description"],
                impact="bullish",
                weight=f.get("weight", 0.5),
                source=f.get("source", "Analysis"),
            )
            for f in analysis.get("bullish_factors", [])
            if isinstance(f, dict)
        ]

        bearish_factors = [
            Factor(
                description=f["description"],
                impact="bearish",
                weight=f.get("weight", 0.5),
                source=f.get("source", "Analysis"),
            )
            for f in analysis.get("bearish_factors", [])
            if isinstance(f, dict)
        ]

        # Build catalysts
        catalysts = []
        for c in analysis.get("catalysts", []):
            if isinstance(c, dict):
                date = None
                if c.get("date"):
                    try:
                        date = datetime.fromisoformat(str(c["date"]).replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass
                catalysts.append(Catalyst(
                    event=c.get("event", "Upcoming event"),
                    date=date,
                    potential_impact=c.get("impact", "uncertain"),
                    importance=c.get("importance", "medium"),
                ))

        # Build reasoning chain with decision trail
        decision_trail = analysis.get("decision_trail")
        reasoning = ReasoningChain(
            summary=analysis.get("summary", "Analysis complete."),
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            catalysts=catalysts,
            risks=analysis.get("risks", []),
            assumptions=analysis.get("assumptions", []),
            decision_trail=decision_trail,  # Include full decision trail for transparency
        )

        # Calculate scores using ADVANCED PROBABILITY ENGINE
        news_sent = data.get_news_sentiment()
        social_sent = data.get_social_sentiment()
        tech_signal = data.get_technical_signal()

        # Price gap
        price_gap = None
        if target_price and data.current_price:
            price_gap = ((target_price - data.current_price) / data.current_price) * 100

        # USE ENHANCED PROBABILITY ENGINE V2 for accurate predictions
        # This uses comprehensive data integration with 15+ factors
        if target_price and data.current_price and data.current_price > 0:
            # Calculate days until target
            days_until = 90  # Default
            if target_date:
                days_until = max(1, (target_date - datetime.now()).days)

            # Build signals dict from analysis
            signal_scores = {}
            if tech_signal:
                signal_scores["technical"] = tech_signal.get("normalized_score", 0)
            if news_sent:
                signal_scores["news"] = news_sent.get("overall_score", 0)
            if social_sent:
                signal_scores["social"] = social_sent.get("overall_score", 0)

            # Prepare technicals dict with support/resistance
            technicals_dict = None
            if data.technicals:
                technicals_dict = {
                    "rsi_14": data.technicals.rsi_14,
                    "macd": data.technicals.macd,
                    "macd_signal": data.technicals.macd_signal,
                    "sma_20": data.technicals.sma_20,
                    "sma_50": data.technicals.sma_50,
                    "sma_200": data.technicals.sma_200,
                    "current_price": data.current_price,
                    "support_level": getattr(data.technicals, 'support_level', None),
                    "resistance_level": getattr(data.technicals, 'resistance_level', None),
                }

            # Get all available data
            historical_prices = data.historical_prices if hasattr(data, 'historical_prices') else []

            # Get earnings history and sector for improved predictions
            earnings_history = data.earnings_history if hasattr(data, 'earnings_history') else None
            sector = data.company_info.sector if data.company_info and hasattr(data.company_info, 'sector') else None

            # Calculate probability using ENHANCED engine with ALL available data
            prob_result = enhanced_probability_engine.calculate_probability(
                current_price=data.current_price,
                target_price=target_price,
                days_to_target=days_until,
                historical_prices=historical_prices,
                signals=signal_scores,
                technicals=technicals_dict,
                news_sentiment=news_sent.get("overall_score") if news_sent else None,
                options_data=data.options_data if hasattr(data, 'options_data') else None,
                analyst_data=data.analyst_recommendations if hasattr(data, 'analyst_recommendations') else None,
                insider_data=data.insider_transactions if hasattr(data, 'insider_transactions') else None,
                earnings_dates=data.earnings_dates if hasattr(data, 'earnings_dates') else None,
                vix_data=data.vix_data if hasattr(data, 'vix_data') else None,
                fear_greed=data.fear_greed if hasattr(data, 'fear_greed') else None,
                sector_performance=data.sector_performance if hasattr(data, 'sector_performance') else None,
                finviz_data=data.finviz_data if hasattr(data, 'finviz_data') else None,
                # New parameters for improved predictions
                ticker=data.ticker,  # For per-ticker tuning
                earnings_history=earnings_history,  # For earnings surprise analysis
                sector=sector,  # For sector-based predictions
            )

            probability = prob_result.probability / 100  # Convert to 0-1 range

            self.logger.info(
                "Enhanced probability engine V2 result",
                probability=f"{prob_result.probability:.1f}%",
                confidence=prob_result.confidence,
                method=prob_result.method,
                market_regime=prob_result.market_regime,
                volatility_regime=prob_result.volatility_regime,
                factors_used=prob_result.factors_used,
                components=prob_result.components,
                adjustments=prob_result.adjustments,
                historical_data_points=len(historical_prices),
            )
        else:
            # Fallback to AI-generated probability
            probability = analysis.get("probability", 50) / 100

        # Add model info to sources
        sources = data.data_sources.copy()
        sources.append(DataSource(
            name=f"AI Model ({model_used})",
            reliability_score=0.9 if "claude" in model_used else 0.75 if "ollama" in model_used else 0.6,
        ))

        # Calculate data limitations
        data_limitations, has_limited_data = self._calculate_data_limitations(data)

        # Determine instrument type
        detected_instrument_type = instrument_detector.get_instrument_type(ticker)

        return PredictionResponse(
            id=str(uuid.uuid4()),
            query=request.query,
            ticker=ticker,
            instrument_type=detected_instrument_type,
            prediction_type=prediction_type,
            probability=probability,
            confidence_level=confidence_level,
            confidence_score=data.data_quality_score,
            target_price=target_price,
            target_date=target_date,
            current_price=data.current_price,
            price_gap_percent=round(price_gap, 2) if price_gap else None,
            reasoning=reasoning,
            sentiment=sentiment,
            sentiment_score=round((news_sent.get("overall_score", 0) + social_sent.get("overall_score", 0)) / 2, 3),
            technical_score=tech_signal.get("normalized_score"),
            fundamental_score=None,
            data_quality_score=data.data_quality_score,
            data_points_analyzed=data.data_points_count,
            sources_used=sources,
            data_limitations=data_limitations,
            has_limited_data=has_limited_data,
            news_articles=data.news[:10],
            social_sentiment=data.social,
            technicals=data.technicals,
            historical_accuracy=None,
            created_at=datetime.utcnow(),
        )

    def _determine_prediction_type(self, query: str, target_price: Optional[float]) -> PredictionType:
        """Determine prediction type from query."""
        query_lower = query.lower()

        if target_price or "reach" in query_lower or "hit" in query_lower or "$" in query:
            return PredictionType.PRICE_TARGET
        if "earnings" in query_lower or "beat" in query_lower or "miss" in query_lower:
            return PredictionType.EARNINGS
        if any(word in query_lower for word in ["up", "down", "rise", "fall", "increase", "decrease"]):
            return PredictionType.DIRECTION

        return PredictionType.DIRECTION

    def _extract_price_from_query(self, query: str) -> Optional[float]:
        """Extract target price from query."""
        patterns = [
            r"\$(\d+(?:\.\d{1,2})?)",
            r"(\d+(?:\.\d{1,2})?)\s*dollars?",
            r"reach\s+(\d+(?:\.\d{1,2})?)",
            r"hit\s+(\d+(?:\.\d{1,2})?)",
            r"to\s+(\d+(?:\.\d{1,2})?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1))
                    if price > 0:
                        return price
                except ValueError:
                    continue

        return None

    def _extract_date_from_query(self, query: str) -> Optional[datetime]:
        """Extract target date from query."""
        import calendar

        query_lower = query.lower()

        # Match "by March 2026"
        month_pattern = r"by\s+(\w+)\s+(\d{4})"
        match = re.search(month_pattern, query_lower)
        if match:
            month_name = match.group(1).capitalize()
            year = int(match.group(2))
            try:
                month_num = list(calendar.month_name).index(month_name)
                return datetime(year, month_num, 28)
            except (ValueError, IndexError):
                pass

        # Match "Q1 2026"
        quarter_pattern = r"q(\d)\s+(\d{4})"
        match = re.search(quarter_pattern, query_lower)
        if match:
            quarter = int(match.group(1))
            year = int(match.group(2))
            month = quarter * 3
            return datetime(year, month, 28)

        # Match "end of 2026" or "2026"
        year_pattern = r"(?:by|end of|in)\s+(\d{4})"
        match = re.search(year_pattern, query_lower)
        if match:
            year = int(match.group(1))
            return datetime(year, 12, 31)

        return None


# Singleton instance
prediction_engine = PredictionEngine()
