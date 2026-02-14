"""
API Routes for Clarividex.

Defines all REST API endpoints for the prediction service.
"""

import hashlib
import re
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import structlog

from backend.app.config import settings
from backend.app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    APIStatus,
    StockQuote,
    CompanyInfo,
    TechnicalIndicators,
    TickerExtractionResult,
    ChatRequest,
    AnalyzeQueryRequest,
)
from backend.app.services.prediction_engine import prediction_engine
from backend.app.services.market_data import market_data_service
from backend.app.services.technical_analysis import technical_analysis_service
from backend.app.services.news_service import news_service
from backend.app.services.social_service import social_service
from backend.app.services.prediction_history import prediction_history
from backend.app.services.query_validator import financial_query_validator
from backend.app.services.query_analyzer import query_analyzer

logger = structlog.get_logger()

router = APIRouter()

# ---------------------------------------------------------------------------
# Chat response cache — avoids redundant Claude calls for identical questions
# ---------------------------------------------------------------------------
_chat_cache: dict[str, tuple[float, dict]] = {}
_CHAT_CACHE_TTL = 300  # 5 minutes
_CHAT_CACHE_MAX_SIZE = 100


def _chat_cache_key(message: str, ticker: str | None) -> str:
    normalized = f"{message.lower().strip()}:{(ticker or '').upper()}"
    return hashlib.sha256(normalized.encode()).hexdigest()


def _chat_cache_get(key: str) -> dict | None:
    if key in _chat_cache:
        ts, response = _chat_cache[key]
        if time.time() - ts < _CHAT_CACHE_TTL:
            return response
        del _chat_cache[key]
    return None


def _chat_cache_set(key: str, response: dict) -> None:
    if len(_chat_cache) >= _CHAT_CACHE_MAX_SIZE:
        oldest = min(_chat_cache, key=lambda k: _chat_cache[k][0])
        del _chat_cache[oldest]
    _chat_cache[key] = (time.time(), response)


# ---------------------------------------------------------------------------
# Ticker path-parameter validation
# ---------------------------------------------------------------------------
_TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")


def _validate_ticker(ticker: str) -> str:
    """Validate and normalize a ticker path parameter."""
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=400, detail="Invalid ticker format")
    return t


# =============================================================================
# Health & Status Endpoints
# =============================================================================


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check the health status of the API and its dependencies.

    Returns:
        HealthResponse with status of all services
    """
    apis = []

    # Check Anthropic API
    apis.append(APIStatus(
        name="Anthropic (Claude)",
        available=settings.has_anthropic_key,
        error=None if settings.has_anthropic_key else "API key not configured",
    ))

    # Check Finnhub API
    apis.append(APIStatus(
        name="Finnhub",
        available=settings.has_finnhub_key,
        error=None if settings.has_finnhub_key else "API key not configured (optional)",
    ))

    # Check yfinance (always available)
    apis.append(APIStatus(
        name="yfinance",
        available=True,
    ))

    return HealthResponse(
        status="healthy" if settings.has_anthropic_key else "degraded",
        version="1.0.0",
        timestamp=datetime.utcnow(),
        apis=apis,
        database_connected=True,  # SQLite always available
        cache_connected=True,  # In-memory cache always available
    )


@router.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Clarividex",
        "tagline": "The Clairvoyant Index",
        "version": "1.0.0",
        "description": "AI-powered financial predictions with transparent reasoning",
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
    }


# =============================================================================
# Prediction Endpoints
# =============================================================================


@router.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def create_prediction(request: PredictionRequest):
    """
    Generate a prediction for a financial query.

    This endpoint:
    1. Validates the query is about financial markets
    2. Extracts the stock ticker from the query
    3. Aggregates data from multiple sources
    4. Analyzes using AI reasoning
    5. Returns probability with full reasoning chain

    Args:
        request: PredictionRequest with the query and options

    Returns:
        PredictionResponse with probability, reasoning, and supporting data

    Example queries:
        - "Will NVDA reach $150 by March 2026?"
        - "Will Tesla stock go up in the next month?"
        - "Will Apple beat earnings next quarter?"
        - "Will Bitcoin reach $100k?"
        - "Will EUR/USD hit 1.15?"
    """
    logger.info("Prediction request received", query=request.query)

    # Log non-standard queries but allow them through (frontend handles guidance)
    validation = financial_query_validator.validate_query(request.query)
    if not validation.is_valid:
        logger.info("Non-standard query proceeding after user confirmation",
                     query=request.query[:50])

    try:
        prediction = await prediction_engine.generate_prediction(request)

        # Apply output guardrails
        from backend.app.guardrails import run_output_guards
        guard_result = run_output_guards(
            prediction.reasoning.summary,
            probability=prediction.probability * 100,
        )
        # Clamp probability to 0.15-0.85 (0-1 scale)
        prediction.probability = max(0.15, min(0.85, prediction.probability))

        # Record prediction in history
        prediction_history.record_prediction(
            prediction_id=prediction.id,
            query=prediction.query,
            ticker=prediction.ticker,
            prediction_type=prediction.prediction_type.value,
            probability=prediction.probability,
            confidence_level=prediction.confidence_level.value,
            target_price=prediction.target_price,
            target_date=prediction.target_date,
            current_price=prediction.current_price,
            sentiment=prediction.sentiment.value,
            data_quality_score=prediction.data_quality_score,
            data_points_analyzed=prediction.data_points_analyzed,
            sources_count=len(prediction.sources_used),
            reasoning_summary=prediction.reasoning.summary,
            bullish_factors_count=len(prediction.reasoning.bullish_factors),
            bearish_factors_count=len(prediction.reasoning.bearish_factors),
        )

        return prediction

    except ValueError as e:
        logger.warning("Invalid prediction request", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error("Prediction generation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to generate prediction. Please try again.",
        )


@router.get("/predict/simple", tags=["Predictions"])
async def simple_prediction(
    query: str = Query(..., description="Your prediction question", min_length=10),
    ticker: Optional[str] = Query(None, description="Stock ticker (auto-detected if not provided)"),
):
    """
    Simple GET endpoint for predictions.

    Use this for quick predictions without full options.

    Args:
        query: Your prediction question
        ticker: Optional stock ticker

    Returns:
        Simplified prediction response
    """
    request = PredictionRequest(
        query=query,
        ticker=ticker,
    )

    prediction = await create_prediction(request)

    # Return simplified response
    return {
        "query": prediction.query,
        "ticker": prediction.ticker,
        "probability": f"{prediction.probability * 100:.0f}%",
        "confidence": prediction.confidence_level.value,
        "sentiment": prediction.sentiment.value,
        "summary": prediction.reasoning.summary,
        "current_price": prediction.current_price,
        "target_price": prediction.target_price,
        "bullish_factors": len(prediction.reasoning.bullish_factors),
        "bearish_factors": len(prediction.reasoning.bearish_factors),
        "data_quality": f"{prediction.data_quality_score * 100:.0f}%",
        "disclaimer": prediction.disclaimer,
    }


# =============================================================================
# Market Data Endpoints
# =============================================================================


@router.get("/stock/{ticker}/quote", response_model=StockQuote, tags=["Market Data"])
async def get_stock_quote(ticker: str):
    """
    Get real-time quote for a stock.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, NVDA)

    Returns:
        StockQuote with current price and market data
    """
    ticker = _validate_ticker(ticker)
    quote = market_data_service.get_quote(ticker)

    if not quote:
        raise HTTPException(
            status_code=404,
            detail=f"Could not find quote for ticker {ticker}",
        )

    return quote


@router.get("/stock/{ticker}/info", response_model=CompanyInfo, tags=["Market Data"])
async def get_company_info(ticker: str):
    """
    Get company fundamental information.

    Args:
        ticker: Stock ticker symbol

    Returns:
        CompanyInfo with company details and fundamentals
    """
    ticker = _validate_ticker(ticker)
    info = market_data_service.get_company_info(ticker)

    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Could not find info for ticker {ticker}",
        )

    return info


@router.get("/stock/{ticker}/technicals", response_model=TechnicalIndicators, tags=["Market Data"])
async def get_technical_indicators(ticker: str):
    """
    Get technical analysis indicators for a stock.

    Args:
        ticker: Stock ticker symbol

    Returns:
        TechnicalIndicators with RSI, MACD, moving averages, etc.
    """
    ticker = _validate_ticker(ticker)
    technicals = technical_analysis_service.calculate_indicators(ticker)

    if not technicals:
        raise HTTPException(
            status_code=404,
            detail=f"Could not calculate technicals for ticker {ticker}",
        )

    return technicals


@router.get("/stock/{ticker}/news", tags=["Market Data"])
async def get_stock_news(
    ticker: str,
    limit: int = Query(10, ge=1, le=50, description="Number of articles to return"),
):
    """
    Get recent news articles for a stock.

    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of articles

    Returns:
        List of news articles with sentiment scores
    """
    ticker = _validate_ticker(ticker)
    # Get company name for better search
    info = market_data_service.get_company_info(ticker)
    company_name = info.name if info else None

    articles = await news_service.get_news_for_ticker(
        ticker,
        company_name,
        limit=limit,
    )

    sentiment = news_service.calculate_news_sentiment(articles)

    return {
        "ticker": ticker.upper(),
        "articles": [
            {
                "title": a.title,
                "source": a.source,
                "url": a.url,
                "published_at": a.published_at.isoformat(),
                "sentiment_score": a.sentiment_score,
            }
            for a in articles
        ],
        "aggregate_sentiment": sentiment,
    }


@router.get("/stock/{ticker}/social", tags=["Market Data"])
async def get_social_sentiment(ticker: str):
    """
    Get social media sentiment for a stock.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Social sentiment data from multiple platforms
    """
    ticker = _validate_ticker(ticker)
    # Get company name for better search
    info = market_data_service.get_company_info(ticker)
    company_name = info.name if info else None

    social_data = await social_service.get_social_sentiment(
        ticker,
        company_name,
    )

    aggregate = social_service.calculate_aggregate_sentiment(social_data)

    return {
        "ticker": ticker.upper(),
        "platforms": [
            {
                "platform": s.platform,
                "mentions": s.mentions_count,
                "sentiment_score": s.sentiment_score,
                "bullish_percentage": s.bullish_percentage,
                "trending": s.trending,
                "sample_posts": s.sample_posts,
            }
            for s in social_data
        ],
        "aggregate": aggregate,
    }


# =============================================================================
# Utility Endpoints
# =============================================================================


@router.get("/extract-ticker", tags=["Utilities"])
async def extract_ticker_from_text(
    text: str = Query(..., description="Text to extract ticker from"),
):
    """
    Extract a stock ticker from natural language text.

    Args:
        text: Text that may contain a ticker symbol

    Returns:
        Extracted ticker or null
    """
    ticker = market_data_service.extract_ticker_from_query(text)

    return {
        "input": text,
        "extracted_ticker": ticker,
        "valid": ticker is not None,
    }


@router.get("/validate-ticker", response_model=TickerExtractionResult, tags=["Utilities"])
async def validate_ticker_extraction(
    query: str = Query(..., description="Query to extract and validate ticker from"),
):
    """
    Extract and validate a stock ticker from a query with confidence information.

    This endpoint analyzes the query and returns:
    - The extracted ticker (if found)
    - Confidence score
    - Whether user confirmation is recommended
    - Alternative suggestions if ambiguous

    Use this before making a prediction to ensure the correct ticker is used.

    Args:
        query: The prediction query to analyze

    Returns:
        TickerExtractionResult with confidence info and suggestions
    """
    result = market_data_service.extract_ticker_with_confidence(query)
    return result


@router.get("/popular-tickers", tags=["Utilities"])
async def get_popular_tickers():
    """
    Get a list of popular stock tickers.

    Returns:
        List of commonly traded tickers
    """
    return {
        "tickers": sorted(list(market_data_service.POPULAR_TICKERS)),
        "count": len(market_data_service.POPULAR_TICKERS),
    }


# =============================================================================
# Prediction History & Accuracy Endpoints
# =============================================================================


@router.get("/history", tags=["History"])
async def get_prediction_history(
    limit: int = Query(20, ge=1, le=100, description="Number of predictions to return"),
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    outcome: Optional[str] = Query(None, description="Filter by outcome (correct, incorrect, pending)"),
):
    """
    Get prediction history.

    Returns:
        List of recent predictions with optional filters
    """
    predictions = prediction_history.get_recent_predictions(
        limit=limit,
        ticker=ticker.upper() if ticker else None,
        outcome=outcome,
    )

    return {
        "predictions": predictions,
        "count": len(predictions),
    }


@router.get("/history/{prediction_id}", tags=["History"])
async def get_prediction_by_id(prediction_id: str):
    """
    Get a specific prediction by ID.

    Returns:
        Prediction details
    """
    prediction = prediction_history.get_prediction(prediction_id)

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return prediction


@router.post("/history/{prediction_id}/resolve", tags=["History"])
async def resolve_prediction(
    prediction_id: str,
    outcome: str = Query(..., description="Outcome: correct or incorrect"),
    actual_price: Optional[float] = Query(None, description="Actual price at resolution"),
    notes: Optional[str] = Query(None, description="Additional notes"),
):
    """
    Manually resolve a prediction outcome.

    Use this to mark whether a prediction was correct or incorrect.
    """
    if outcome not in ["correct", "incorrect"]:
        raise HTTPException(status_code=400, detail="Outcome must be 'correct' or 'incorrect'")

    success = prediction_history.update_outcome(
        prediction_id=prediction_id,
        outcome=outcome,
        actual_price=actual_price,
        notes=notes,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update prediction outcome. Please try again.")

    return {"success": True, "prediction_id": prediction_id, "outcome": outcome}


@router.get("/accuracy", tags=["History"])
async def get_accuracy_stats():
    """
    Get prediction accuracy statistics.

    Returns:
        Overall accuracy rate, breakdown by ticker, confidence level, etc.
    """
    stats = prediction_history.get_accuracy_stats()

    return {
        "total_predictions": stats.total_predictions,
        "resolved_predictions": stats.resolved_predictions,
        "correct_predictions": stats.correct_predictions,
        "incorrect_predictions": stats.incorrect_predictions,
        "pending_predictions": stats.pending_predictions,
        "accuracy_rate": round(stats.accuracy_rate, 1),
        "average_probability": {
            "correct": round(stats.average_probability_correct * 100, 1),
            "incorrect": round(stats.average_probability_incorrect * 100, 1),
        },
        "by_ticker": stats.by_ticker,
        "by_confidence": stats.by_confidence,
        "recent_predictions": stats.recent_predictions,
    }


@router.post("/history/auto-resolve", tags=["History"])
async def auto_resolve_predictions():
    """
    Automatically resolve predictions that have passed their target date.

    Checks current prices against target prices for predictions
    that have expired.
    """
    resolved_count = prediction_history.auto_resolve_predictions()

    return {
        "resolved_count": resolved_count,
        "message": f"Automatically resolved {resolved_count} predictions",
    }


# =============================================================================
# Chat Endpoint for Results Assistant
# =============================================================================


@router.post("/chat", tags=["Chat"])
async def chat_about_results(request: ChatRequest):
    """
    Chat endpoint for asking questions about prediction results.
    Uses Claude API if available, otherwise falls back to local Ollama model.

    Enhanced to detect and handle:
    - Scenario/what-if questions ("What if rates go up?")
    - Herd sentiment questions ("Is everyone bullish?")
    - Risk analysis questions ("What are the risks?")

    Args:
        request: ChatRequest with message, context, and optional ticker

    Returns:
        AI response about the prediction results
    """
    import httpx

    message = request.message
    context = request.context
    ticker = request.ticker

    # Check chat cache for identical question
    cache_key = _chat_cache_key(message, ticker)
    cached = _chat_cache_get(cache_key)
    if cached:
        logger.info("Chat: returning cached response", ticker=ticker)
        return cached

    # Detect special question types and enhance context
    enhanced_context = context
    special_data = None

    # Detect scenario questions
    scenario_keywords = ["what if", "what happens if", "scenario", "if rates", "if there's a recession", "market crash", "interest rate"]
    is_scenario_question = any(kw in message.lower() for kw in scenario_keywords)

    # Detect herd sentiment questions
    herd_keywords = ["everyone", "crowd", "herd", "sentiment", "bullish", "bearish", "fear", "greed", "contrarian"]
    is_herd_question = any(kw in message.lower() for kw in herd_keywords)

    # Detect risk questions
    risk_keywords = ["risk", "danger", "warning", "concern", "worried", "downside"]
    is_risk_question = any(kw in message.lower() for kw in risk_keywords)

    # Enhance context based on question type
    if ticker and (is_scenario_question or is_herd_question or is_risk_question):
        try:
            if is_scenario_question:
                from backend.app.services.scenario_analyzer import scenario_analyzer
                # Try to detect specific scenario from message
                scenario_map = {
                    "rate": "interest_rate_hike",
                    "recession": "recession",
                    "crash": "market_crash",
                    "oil": "oil_spike",
                    "inflation": "inflation_surge",
                    "vix": "vix_spike",
                }
                detected_scenario = "interest_rate_hike"  # default
                for keyword, scenario in scenario_map.items():
                    if keyword in message.lower():
                        detected_scenario = scenario
                        break

                scenario_result = await scenario_analyzer.analyze_scenario(
                    ticker=ticker.upper(),
                    scenario=detected_scenario,
                )
                enhanced_context += f"\n\nScenario Analysis ({scenario_result.scenario_description}):\n"
                enhanced_context += f"- Expected impact: {scenario_result.expected_move_pct:+.1f}%\n"
                enhanced_context += f"- Confidence: {scenario_result.confidence * 100:.0f}%\n"
                enhanced_context += f"- Historical precedent: {scenario_result.historical_precedent or 'N/A'}\n"
                enhanced_context += f"- Recommendation: {scenario_result.recommendation}\n"
                special_data = {"type": "scenario", "scenario": detected_scenario}

            if is_herd_question:
                from backend.app.services.herd_sentiment import herd_sentiment_analyzer
                # Get social sentiment data
                info = market_data_service.get_company_info(ticker.upper())
                company_name = info.name if info else None
                social_data = await social_service.get_social_sentiment(ticker.upper(), company_name)
                social_sentiment = None
                if social_data:
                    aggregate = social_service.calculate_aggregate_sentiment(social_data)
                    social_sentiment = (aggregate + 1) / 2

                herd_result = await herd_sentiment_analyzer.check_herd_warning(
                    ticker=ticker.upper(),
                    social_sentiment=social_sentiment,
                )
                enhanced_context += f"\n\nHerd Sentiment Analysis:\n"
                enhanced_context += f"- Herd level: {herd_result.herd_level.value}\n"
                enhanced_context += f"- Warning: {herd_result.warning_message}\n"
                enhanced_context += f"- Contrarian signal: {herd_result.contrarian_signal:+.2f}\n"
                enhanced_context += f"- Historical accuracy: {herd_result.historical_accuracy}\n"
                enhanced_context += f"- Recommendation: {herd_result.recommendation}\n"
                special_data = {"type": "herd", "level": herd_result.herd_level.value}

            if is_risk_question:
                from backend.app.services.scenario_analyzer import scenario_analyzer
                risks = await scenario_analyzer.get_current_risk_scenarios(ticker=ticker.upper())
                enhanced_context += f"\n\nCurrent Risk Scenarios for {ticker.upper()}:\n"
                for r in risks:
                    enhanced_context += f"- {r['scenario']}: {r['probability']*100:.0f}% probability, {r['impact_on_ticker'].expected_move_pct:+.1f}% impact\n"
                special_data = {"type": "risk", "risk_count": len(risks)}

        except Exception as e:
            logger.warning("Failed to enhance context", error=str(e))

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # RAG: Detect methodology questions and augment context with doc chunks
    METHODOLOGY_KEYWORDS = [
        "how does", "how do", "methodology", "prediction engine", "algorithm",
        "technical indicator", "rsi", "macd", "data source", "how it works",
        "explain", "what is", "monte carlo", "bayesian", "probability",
        "bollinger", "moving average", "sentiment", "approach",
    ]
    is_methodology_question = any(kw in message.lower() for kw in METHODOLOGY_KEYWORDS)

    if is_methodology_question:
        try:
            from backend.app.rag.service import rag_service
            chunks = rag_service.query(message, top_k=3)
            if chunks:
                enhanced_context += "\n\nRelevant documentation:\n"
                for i, chunk in enumerate(chunks, 1):
                    enhanced_context += f"\n--- Doc Chunk {i} ---\n{chunk}\n"
        except Exception as e:
            logger.warning("RAG query failed", error=str(e))

    # Load chat system prompt from prompt registry
    chat_system_prompt = None
    try:
        from backend.app.prompts.registry import get_active_prompt
        chat_system_prompt = get_active_prompt("chat_assistant")
    except Exception:
        pass

    if chat_system_prompt:
        enhanced_context = f"{chat_system_prompt}\n\nContext:\n{enhanced_context}"

    # Import guardrails for chat response filtering
    try:
        from backend.app.guardrails import run_output_guards
    except Exception as e:
        logger.error("Failed to import guardrails", error=str(e))
        run_output_guards = None

    # Try Claude API first if available — reuse singleton client
    logger.info("Chat: checking Claude availability", has_key=settings.has_anthropic_key, has_client=bool(prediction_engine.claude_client))
    if settings.has_anthropic_key:
        try:
            client = prediction_engine.claude_client
            if not client:
                logger.warning("Chat: has_anthropic_key is True but claude_client is None")
                raise HTTPException(status_code=503, detail="Claude API not available")

            logger.info("Chat: calling Claude API")
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=enhanced_context,
                messages=[
                    {"role": "user", "content": message}
                ]
            )

            response_text = response.content[0].text
            logger.info("Chat: Claude response received", length=len(response_text))

            # Apply guardrails to chat response
            if run_output_guards:
                guard_result = run_output_guards(response_text)
                response_text = guard_result.text

            result = {
                "response": response_text,
                "model": "claude"
            }
            if special_data:
                result["enhanced_with"] = special_data
            _chat_cache_set(cache_key, result)
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.warning("Claude API failed, falling back to Ollama", error=str(e), error_type=type(e).__name__)
    else:
        logger.warning("Chat: Anthropic key not available, skipping Claude")

    # Fallback to local Ollama model
    import os
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    logger.info("Chat: trying Ollama", ollama_host=ollama_host)
    try:
        async with httpx.AsyncClient(timeout=120.0) as ollama_client:
            # First check which models are available
            models_response = await ollama_client.get(f"{ollama_host}/api/tags")
            available_models = []
            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = [m.get("name", "") for m in models_data.get("models", [])]

            # Pick the best available model
            preferred_models = ["llama3.1:latest", "llama3.1", "mistral:latest", "mistral", "deepseek-r1:latest"]
            model_to_use = "llama3.1:latest"
            for pm in preferred_models:
                if pm in available_models:
                    model_to_use = pm
                    break

            logger.info("Using Ollama for chat", model=model_to_use)

            ollama_response = await ollama_client.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model_to_use,
                    "prompt": f"{enhanced_context}\n\nUser Question: {message}\n\nProvide a helpful, concise answer:",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500
                    }
                }
            )

            if ollama_response.status_code == 200:
                data = ollama_response.json()
                response_text = data.get("response", "").strip()
                if response_text:
                    # Apply guardrails to Ollama response
                    guard_result = run_output_guards(response_text)

                    result = {
                        "response": guard_result.text,
                        "model": f"ollama:{model_to_use}"
                    }
                    if special_data:
                        result["enhanced_with"] = special_data
                    _chat_cache_set(cache_key, result)
                    return result

    except Exception as e:
        logger.error("Ollama chat failed", error=str(e))

    # Final fallback - basic response
    return {
        "response": "I can help answer questions about your prediction results. Based on the data shown, you can see the probability, confidence level, and various factors that contributed to this prediction. What specific aspect would you like to understand better?",
        "model": "fallback"
    }


# =============================================================================
# Backtesting Endpoints
# =============================================================================


@router.post("/backtest", tags=["Backtesting"])
async def run_backtest(
    ticker: str = Query(..., description="Stock ticker to backtest"),
    target_return: float = Query(0.10, description="Target return (0.10 = +10%)"),
    holding_period: int = Query(30, ge=5, le=365, description="Days to hold before checking outcome"),
    lookback_days: int = Query(252, ge=60, le=500, description="Days of history to test"),
):
    """
    Run a backtest to validate prediction accuracy against historical data.

    This tests how well the prediction engine would have performed in the past by:
    1. Simulating predictions at regular intervals over the lookback period
    2. Comparing predicted probabilities against actual outcomes
    3. Calculating accuracy, precision, recall, and calibration metrics

    Args:
        ticker: Stock ticker to backtest (e.g., "NVDA", "AAPL")
        target_return: Target return to predict (0.10 = +10%, -0.05 = -5%)
        holding_period: Days to wait before checking if target was hit
        lookback_days: How many days of history to test (more = better but slower)

    Returns:
        Comprehensive backtest results with accuracy metrics
    """
    from backend.app.services.backtesting_engine import backtesting_engine

    logger.info(
        "Starting backtest",
        ticker=ticker,
        target_return=target_return,
        holding_period=holding_period,
        lookback_days=lookback_days,
    )

    try:
        results = await backtesting_engine.run_backtest(
            ticker=ticker.upper(),
            target_return=target_return,
            holding_period=holding_period,
            lookback_days=lookback_days,
            step_days=5,  # Test every 5 days (weekly)
        )

        return {
            "ticker": results.ticker,
            "target_return": f"{results.target_return:+.1%}",
            "holding_period": results.holding_period,
            "total_predictions": results.total_predictions,
            "correct_predictions": results.correct_predictions,
            "accuracy": round(results.accuracy * 100, 1),
            "precision": round(results.precision * 100, 1),
            "recall": round(results.recall * 100, 1),
            "f1_score": round(results.f1_score * 100, 1),
            "calibration": {
                "avg_predicted_probability": round(results.avg_probability, 1),
                "actual_hit_rate": round(results.avg_actual_hit_rate, 1),
                "calibration_error": round(results.calibration_error, 1),
            },
            "by_market_regime": {
                k: round(v * 100, 1) for k, v in results.accuracy_by_regime.items()
            },
            "by_volatility": {
                k: round(v * 100, 1) for k, v in results.accuracy_by_volatility.items()
            },
            "component_effectiveness": {
                k: round(v * 100, 1) for k, v in list(results.component_correlations.items())[:5]
            },
            "summary": results.summary,
        }

    except Exception as e:
        logger.error("Backtest failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Backtest failed: {str(e)}" if settings.app_debug else "Backtest failed. Please try again.",
        )


@router.post("/backtest/multi", tags=["Backtesting"])
async def run_multi_ticker_backtest(
    tickers: str = Query(..., description="Comma-separated list of tickers"),
    target_return: float = Query(0.10, description="Target return"),
    holding_period: int = Query(30, ge=5, le=365, description="Holding period in days"),
):
    """
    Run backtest across multiple tickers to get aggregate accuracy.

    Args:
        tickers: Comma-separated tickers (e.g., "NVDA,AAPL,MSFT,GOOGL")
        target_return: Target return to predict
        holding_period: Days to hold

    Returns:
        Individual and aggregate accuracy metrics
    """
    from backend.app.services.backtesting_engine import backtesting_engine

    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    if len(ticker_list) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 tickers allowed")

    results = await backtesting_engine.run_multi_ticker_backtest(
        tickers=ticker_list,
        target_return=target_return,
        holding_period=holding_period,
        lookback_days=252,
    )

    # Calculate aggregate
    total_preds = sum(r.total_predictions for r in results.values())
    total_correct = sum(r.correct_predictions for r in results.values())
    overall_accuracy = total_correct / total_preds if total_preds > 0 else 0

    return {
        "tickers_tested": len(ticker_list),
        "overall_accuracy": round(overall_accuracy * 100, 1),
        "total_predictions": total_preds,
        "by_ticker": {
            ticker: {
                "accuracy": round(r.accuracy * 100, 1),
                "predictions": r.total_predictions,
                "correct": r.correct_predictions,
            }
            for ticker, r in results.items()
        },
    }


@router.get("/backtest/signals/{ticker}", tags=["Backtesting"])
async def analyze_signal_effectiveness(
    ticker: str,
):
    """
    Analyze which prediction signals are most effective for a specific ticker.

    This helps identify which factors (RSI, MACD, sentiment, etc.) are
    actually predictive for this stock, allowing for weight optimization.

    Args:
        ticker: Stock ticker to analyze

    Returns:
        Signal effectiveness analysis with recommendations
    """
    from backend.app.services.backtesting_engine import backtesting_engine

    ticker = _validate_ticker(ticker)
    analysis = await backtesting_engine.analyze_signal_effectiveness(
        ticker=ticker,
        lookback_days=252,
    )

    return {
        "ticker": analysis['ticker'],
        "overall_accuracy": round(analysis['overall_accuracy'] * 100, 1),
        "best_signals": [
            {"signal": s, "accuracy": round(a * 100, 1)}
            for s, a in analysis['best_components']
        ],
        "worst_signals": [
            {"signal": s, "accuracy": round(a * 100, 1)}
            for s, a in analysis['worst_components']
        ],
        "recommendations": analysis['recommendations'],
    }


# =============================================================================
# Scenario Analysis Endpoints
# =============================================================================


@router.get("/scenario/{ticker}", tags=["Scenario Analysis"])
async def analyze_scenario(
    ticker: str,
    scenario: str = Query(..., description="Scenario type (e.g., 'interest_rate_hike', 'recession', 'market_crash')"),
    magnitude: float = Query(1.0, ge=0.1, le=3.0, description="Severity multiplier (1.0 = standard, 2.0 = severe)"),
):
    """
    Analyze how a hypothetical scenario would impact a stock.

    This addresses a key fintech bottleneck: users want "what-if" analysis.

    Available scenarios:
    - interest_rate_hike / interest_rate_cut
    - recession
    - market_crash
    - earnings_beat / earnings_miss
    - oil_spike
    - vix_spike
    - inflation_surge
    - trade_war

    Args:
        ticker: Stock ticker symbol
        scenario: Scenario type to analyze
        magnitude: Severity multiplier (1.0 = standard)

    Returns:
        Detailed scenario impact analysis with recommendations
    """
    from backend.app.services.scenario_analyzer import scenario_analyzer

    ticker = _validate_ticker(ticker)
    try:
        result = await scenario_analyzer.analyze_scenario(
            ticker=ticker,
            scenario=scenario,
            magnitude=magnitude,
        )

        return {
            "ticker": result.ticker,
            "scenario": result.scenario,
            "description": result.scenario_description,
            "expected_impact": {
                "move_pct": result.expected_move_pct,
                "confidence": round(result.confidence * 100, 1),
                "timeframe_days": result.timeframe_days,
            },
            "sector_impact": result.sector_impact,
            "relative_to_market": result.relative_to_market,
            "primary_factors": result.primary_factors,
            "risk_factors": result.risk_factors,
            "historical_precedent": result.historical_precedent,
            "recommendation": result.recommendation,
            "hedging_suggestions": result.hedging_suggestions,
            "caveats": result.caveats,
        }

    except Exception as e:
        logger.error("Scenario analysis failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Scenario analysis failed: {str(e)}" if settings.app_debug else "Scenario analysis failed. Please try again.",
        )


@router.get("/scenario/{ticker}/compare", tags=["Scenario Analysis"])
async def compare_scenarios(
    ticker: str,
    scenarios: str = Query(..., description="Comma-separated scenarios to compare"),
):
    """
    Compare multiple scenarios for a single stock.

    Useful for stress testing a position against multiple risks.

    Args:
        ticker: Stock ticker symbol
        scenarios: Comma-separated list of scenarios

    Returns:
        Comparison of multiple scenario impacts
    """
    from backend.app.services.scenario_analyzer import scenario_analyzer

    ticker = _validate_ticker(ticker)
    scenario_list = [s.strip().lower() for s in scenarios.split(",")]

    if len(scenario_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 scenarios allowed")

    results = await scenario_analyzer.analyze_multiple_scenarios(
        ticker=ticker.upper(),
        scenarios=scenario_list,
    )

    return {
        "ticker": ticker.upper(),
        "scenarios_analyzed": len(results),
        "comparison": [
            {
                "scenario": r.scenario,
                "expected_move_pct": r.expected_move_pct,
                "confidence": round(r.confidence * 100, 1),
                "sector_impact": r.sector_impact,
                "recommendation": r.recommendation,
            }
            for r in results
        ],
        "worst_case": min(results, key=lambda x: x.expected_move_pct).scenario,
        "best_case": max(results, key=lambda x: x.expected_move_pct).scenario,
    }


@router.get("/scenario/{ticker}/risks", tags=["Scenario Analysis"])
async def get_current_risks(ticker: str):
    """
    Get the most relevant risk scenarios for current market conditions.

    Proactively warns users about scenarios they should monitor.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List of current risk scenarios with probabilities and impacts
    """
    from backend.app.services.scenario_analyzer import scenario_analyzer

    ticker = _validate_ticker(ticker)
    try:
        risks = await scenario_analyzer.get_current_risk_scenarios(ticker=ticker)

        return {
            "ticker": ticker,
            "risk_count": len(risks),
            "risks": [
                {
                    "scenario": r["scenario"],
                    "probability": round(r["probability"] * 100, 1),
                    "description": r["description"],
                    "expected_impact_pct": r["impact_on_ticker"].expected_move_pct,
                    "recommendation": r["impact_on_ticker"].recommendation,
                }
                for r in risks
            ],
        }

    except Exception as e:
        logger.error("Risk analysis failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Risk analysis failed: {str(e)}" if settings.app_debug else "Risk analysis failed. Please try again.",
        )


# =============================================================================
# Herd Sentiment Warning Endpoints
# =============================================================================


@router.get("/herd-warning/{ticker}", tags=["Herd Sentiment"])
async def check_herd_warning(
    ticker: str,
    social_sentiment: Optional[float] = Query(None, ge=0, le=1, description="Social media bullish % (0-1)"),
    put_call_ratio: Optional[float] = Query(None, ge=0.1, le=3.0, description="Options put/call ratio"),
    fear_greed: Optional[int] = Query(None, ge=0, le=100, description="CNN Fear & Greed Index (0-100)"),
):
    """
    Check for herd behavior and get contrarian warnings.

    Addresses fintech bottleneck: 40% of investors follow crowds blindly.

    This endpoint:
    - Detects extreme bullish/bearish consensus
    - Provides contrarian warnings (when "everyone" agrees)
    - Includes historical accuracy of contrarian signals
    - Offers ELI5 explanations for beginners

    Args:
        ticker: Stock ticker symbol
        social_sentiment: Bullish percentage from social media (0-1 scale)
        put_call_ratio: Options put/call ratio (low = bullish)
        fear_greed: CNN Fear & Greed Index value

    Returns:
        Herd behavior analysis with contrarian signals
    """
    from backend.app.services.herd_sentiment import herd_sentiment_analyzer

    ticker = _validate_ticker(ticker)
    try:
        # If no sentiment data provided, try to fetch it
        if social_sentiment is None:
            info = market_data_service.get_company_info(ticker)
            company_name = info.name if info else None
            social_data = await social_service.get_social_sentiment(ticker, company_name)
            if social_data:
                aggregate = social_service.calculate_aggregate_sentiment(social_data)
                social_sentiment = (aggregate + 1) / 2  # Convert -1 to 1 scale to 0-1

        result = await herd_sentiment_analyzer.check_herd_warning(
            ticker=ticker,
            social_sentiment=social_sentiment,
            put_call_ratio=put_call_ratio,
            fear_greed=fear_greed,
        )

        return {
            "ticker": result.ticker,
            "herd_level": result.herd_level.value,
            "warning_type": result.warning_type.value,
            "warning_message": result.warning_message,
            "sentiment_data": {
                "social_bullish_pct": result.social_sentiment_pct,
                "put_call_ratio": result.put_call_ratio,
                "fear_greed_index": result.fear_greed_index,
                "analyst_consensus": result.analyst_consensus,
            },
            "contrarian_analysis": {
                "signal_strength": round(result.contrarian_signal, 2),
                "confidence": round(result.signal_confidence * 100, 1),
                "historical_accuracy": result.historical_accuracy,
            },
            "similar_historical_periods": result.similar_periods,
            "recommendation": result.recommendation,
            "smart_money_insight": result.what_smart_money_doing,
            "explanations": {
                "detailed": result.explanation.strip(),
                "eli5": result.eli5_explanation.strip(),
            },
        }

    except Exception as e:
        logger.error("Herd sentiment analysis failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Herd analysis failed: {str(e)}" if settings.app_debug else "Herd analysis failed. Please try again.",
        )


# =============================================================================
# Query Analysis Endpoint
# =============================================================================


@router.post("/analyze-query", tags=["Utilities"])
async def analyze_query(request: AnalyzeQueryRequest):
    """
    Analyze a query for quality and financial relevance.

    Returns classification (clear/vague/non_financial), quality score,
    issues, and AI-generated improvement suggestions.
    """
    query = request.query.strip()

    result = await query_analyzer.analyze(query)
    return {
        "category": result.category,
        "can_proceed": result.can_proceed,
        "quality_score": result.quality_score,
        "issues": result.issues,
        "suggestions": result.suggestions,
        "message": result.message,
        "cleaned_query": result.cleaned_query,
    }


# =============================================================================
# SSE Streaming Endpoint
# =============================================================================


@router.post("/predict/stream", tags=["Predictions"])
async def stream_prediction(request: PredictionRequest):
    """
    Stream a prediction via Server-Sent Events.

    Emits events as each stage completes: data_fetch, technical_analysis,
    sentiment_analysis, social_analysis, market_conditions, ai_reasoning,
    probability_calculation, done, error.
    """
    from backend.app.services.stream_service import prediction_stream_service

    return StreamingResponse(
        prediction_stream_service.stream_prediction(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Prompt Versioning Endpoints
# =============================================================================


@router.get("/prompts", tags=["Prompts"])
async def list_prompt_versions():
    """List all available prompt templates with their versions and status."""
    from backend.app.prompts.registry import list_prompts

    prompts = list_prompts()
    return {
        "prompts": [
            {
                "name": p.name,
                "version": p.version,
                "status": p.status,
                "description": p.description,
                "tags": p.tags,
                "prompt_length": len(p.system_prompt),
            }
            for p in prompts
        ],
        "count": len(prompts),
    }


@router.get("/prompts/{name}/compare", tags=["Prompts"])
async def compare_prompt_versions(name: str):
    """Compare all versions of a prompt by name prefix."""
    from backend.app.prompts.registry import compare_prompts

    comparisons = compare_prompts(name)
    if not comparisons:
        raise HTTPException(status_code=404, detail=f"No prompts found matching '{name}'")

    return {
        "name_prefix": name,
        "versions": comparisons,
        "count": len(comparisons),
    }


# =============================================================================
# Evaluation Suite Endpoint
# =============================================================================


@router.get("/eval/run", tags=["Evaluation"])
async def run_evaluation(
    categories: Optional[str] = Query(None, description="Comma-separated categories: prediction,rag,guardrail,edge_case"),
):
    """
    Run the evaluation suite and return results.

    This runs a lightweight subset of the golden dataset to validate
    system health. For full evals, use: python -m backend.app.evals.runner
    """
    from backend.app.evals.runner import run_full_eval

    cat_list = None
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]

    try:
        results = await run_full_eval(categories=cat_list)
        return results
    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}" if settings.app_debug else "Evaluation failed. Please try again.",
        )


@router.get("/available-scenarios", tags=["Scenario Analysis"])
async def list_available_scenarios():
    """
    List all available scenario types for analysis.

    Returns:
        List of scenario types with descriptions
    """
    from backend.app.services.scenario_analyzer import ScenarioType

    scenarios = []
    for st in ScenarioType:
        if st != ScenarioType.CUSTOM:
            scenarios.append({
                "id": st.value,
                "name": st.value.replace("_", " ").title(),
            })

    return {
        "scenarios": scenarios,
        "count": len(scenarios),
        "usage": "Use scenario ID in /scenario/{ticker}?scenario={id}",
    }
