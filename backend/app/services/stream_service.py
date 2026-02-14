"""
SSE Streaming service for real-time prediction progress.

Emits Server-Sent Events as each stage of prediction completes:
data_fetch, technical_analysis, sentiment_analysis, social_analysis,
market_conditions, ai_reasoning, probability_calculation, done, error.
"""

import json
import asyncio
from typing import AsyncGenerator, Optional

import structlog

from backend.app.config import settings
from backend.app.models.schemas import PredictionRequest

logger = structlog.get_logger()


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class PredictionStreamService:
    """Streams prediction progress via SSE events."""

    async def stream_prediction(
        self,
        request: PredictionRequest,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields SSE events as prediction progresses.

        Args:
            request: The prediction request.

        Yields:
            SSE-formatted strings for each stage.
        """
        try:
            from backend.app.services.prediction_engine import prediction_engine
            from backend.app.services.data_aggregator import DataAggregator
            from backend.app.services.market_data import market_data_service

            # Stage 0: Check query relevance
            from backend.app.services.query_validator import financial_query_validator
            validation = financial_query_validator.validate_query(request.query)

            # Stage 1: Extract ticker
            ticker = request.ticker
            if not ticker:
                ticker = market_data_service.extract_ticker_from_query(request.query)
            if not ticker:
                if not validation.is_valid:
                    yield _sse_event("error", {
                        "message": "This doesn't appear to be a financial query and no ticker could be identified.",
                        "suggestions": validation.suggestions or [
                            "Will NVDA reach $150 by March 2026?",
                            "Will Tesla stock go up in the next month?",
                            "Will Bitcoin hit $100k this year?",
                        ],
                    })
                else:
                    yield _sse_event("error", {"message": "Could not identify a stock ticker in your query."})
                return

            # Stage 2: Data fetch
            yield _sse_event("data_fetch", {"status": "started", "ticker": ticker})
            data_aggregator = DataAggregator()
            data = await data_aggregator.aggregate_data(
                ticker=ticker,
                include_technicals=request.include_technicals,
                include_news=request.include_news,
                include_social=request.include_sentiment,
            )
            if not data.quote:
                yield _sse_event("error", {"message": f"Could not fetch market data for {ticker}."})
                return
            yield _sse_event("data_fetch", {
                "status": "complete",
                "current_price": data.current_price,
                "data_points": data.data_points_count,
            })

            # Stage 3: Technical analysis
            yield _sse_event("technical_analysis", {"status": "started"})
            tech_signal = data.get_technical_signal()
            yield _sse_event("technical_analysis", {
                "status": "complete",
                "signal": tech_signal.get("overall_signal", "neutral"),
                "score": tech_signal.get("normalized_score", 0),
            })

            # Stage 4: Sentiment analysis
            yield _sse_event("sentiment_analysis", {"status": "started"})
            news_sentiment = data.get_news_sentiment()
            yield _sse_event("sentiment_analysis", {
                "status": "complete",
                "news_score": news_sentiment.get("overall_score", 0),
                "article_count": news_sentiment.get("article_count", 0),
            })

            # Stage 5: Social analysis
            yield _sse_event("social_analysis", {"status": "started"})
            social_sentiment = data.get_social_sentiment()
            yield _sse_event("social_analysis", {
                "status": "complete",
                "score": social_sentiment.get("overall_score", 0),
                "mentions": social_sentiment.get("total_mentions", 0),
            })

            # Stage 6: Market conditions
            yield _sse_event("market_conditions", {"status": "started"})
            vix_value = data.vix_data.get("value") if data.vix_data else None
            fg_value = data.fear_greed.get("value") if data.fear_greed else None
            yield _sse_event("market_conditions", {
                "status": "complete",
                "vix": vix_value,
                "fear_greed": fg_value,
            })

            # Stage 7: AI reasoning (uses pre-aggregated data to avoid redundant fetch)
            yield _sse_event("ai_reasoning", {"status": "started"})
            prediction = await prediction_engine.generate_prediction_with_data(request, data, ticker)
            yield _sse_event("ai_reasoning", {"status": "complete"})

            # Stage 8: Probability calculation + guardrails
            yield _sse_event("probability_calculation", {"status": "started"})

            # Apply guardrails
            from backend.app.guardrails import run_output_guards
            guard_result = run_output_guards(
                prediction.reasoning.summary,
                probability=prediction.probability * 100,
            )

            # Clamp probability on 0-1 scale
            clamped_probability = max(0.15, min(0.85, prediction.probability))

            # Apply guardrail results back to the prediction object
            prediction.probability = clamped_probability
            prediction.reasoning.summary = guard_result.text

            yield _sse_event("probability_calculation", {
                "status": "complete",
                "probability": clamped_probability,
                "confidence": prediction.confidence_level.value,
            })

            # Stage 9: Done — send FULL PredictionResponse so frontend
            # doesn't need a second API call
            prediction_data = prediction.model_dump(mode="json")
            prediction_data["guardrail_warnings"] = guard_result.warnings
            prediction_data["guardrail_modifications"] = guard_result.modifications
            yield _sse_event("done", {"prediction": prediction_data})

        except Exception as e:
            logger.error("Stream prediction error", error=str(e))
            yield _sse_event("error", {
                "message": "Prediction failed. Please try again.",
            })


# Singleton
prediction_stream_service = PredictionStreamService()
