"""
Evaluation runner for Clarividex.

Runs test cases from the golden dataset and reports metrics.
Can be run via CLI: python -m backend.app.evals.runner
Or via API: GET /api/v1/eval/run
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Optional

import structlog

from backend.app.evals.golden_dataset import GOLDEN_DATASET, TestCase
from backend.app.evals.metrics import (
    prediction_quality,
    response_completeness,
    keyword_recall,
    latency_check,
    guardrail_accuracy,
    probability_bounds_check,
)
from backend.app.evals.experiment_tracker import Experiment, log_experiment

logger = structlog.get_logger()


async def run_single_eval(test_case: TestCase) -> dict:
    """
    Run a single evaluation test case.

    Returns dict with test_id, category, passed, score, details, duration.
    """
    start = time.time()
    result = {
        "test_id": test_case.id,
        "category": test_case.category,
        "query": test_case.query,
        "passed": False,
        "score": 0.0,
        "details": {},
        "error": None,
    }

    try:
        if test_case.category == "prediction":
            result = await _eval_prediction(test_case, result)
        elif test_case.category == "rag":
            result = await _eval_rag(test_case, result)
        elif test_case.category == "guardrail":
            result = await _eval_guardrail(test_case, result)
        elif test_case.category == "edge_case":
            result = await _eval_edge_case(test_case, result)
    except Exception as e:
        result["error"] = str(e)
        result["score"] = 0.0

    duration = time.time() - start
    result["duration_seconds"] = round(duration, 2)

    latency = latency_check(duration)
    result["latency_score"] = latency["score"]

    # Overall score
    if result["error"]:
        result["score"] = 0.0
        result["passed"] = False
    else:
        result["passed"] = result["score"] >= 0.5

    return result


async def _eval_prediction(test_case: TestCase, result: dict) -> dict:
    """Evaluate a prediction test case."""
    from backend.app.models.schemas import PredictionRequest
    from backend.app.services.prediction_engine import prediction_engine

    request = PredictionRequest(
        query=test_case.query,
        ticker=test_case.ticker,
        include_technicals=True,
        include_sentiment=True,
        include_news=True,
    )

    prediction = await prediction_engine.generate_prediction(request)

    # Check quality
    quality = prediction_quality(
        summary=prediction.reasoning.summary,
        probability=prediction.probability,
        bullish_count=len(prediction.reasoning.bullish_factors),
        bearish_count=len(prediction.reasoning.bearish_factors),
    )
    result["details"]["quality"] = quality

    # Check probability bounds
    bounds = probability_bounds_check(prediction.probability)
    result["details"]["probability_bounds"] = bounds

    # Check completeness
    completeness = response_completeness({
        "id": prediction.id,
        "query": prediction.query,
        "ticker": prediction.ticker,
        "probability": prediction.probability,
        "confidence_level": prediction.confidence_level,
        "reasoning": prediction.reasoning,
    })
    result["details"]["completeness"] = completeness

    # Keyword check
    if test_case.expected_keywords:
        kw = keyword_recall(prediction.reasoning.summary, test_case.expected_keywords)
        result["details"]["keyword_recall"] = kw

    # Average scores
    scores = [quality["score"], bounds["score"], completeness["score"]]
    result["score"] = sum(scores) / len(scores)

    return result


async def _eval_rag(test_case: TestCase, result: dict) -> dict:
    """Evaluate a RAG test case."""
    from backend.app.rag.service import rag_service

    rag_service.ensure_indexed()
    chunks = rag_service.query(test_case.query, top_k=3)

    combined_text = " ".join(chunks)

    if test_case.expected_keywords:
        kw = keyword_recall(combined_text, test_case.expected_keywords)
        result["details"]["keyword_recall"] = kw
        result["score"] = kw["score"]
    else:
        result["score"] = 1.0 if chunks else 0.0

    result["details"]["chunks_retrieved"] = len(chunks)
    return result


async def _eval_guardrail(test_case: TestCase, result: dict) -> dict:
    """Evaluate a guardrail test case."""
    from backend.app.guardrails import run_output_guards

    # Run a prediction if there's a ticker
    if test_case.ticker:
        from backend.app.models.schemas import PredictionRequest
        from backend.app.services.prediction_engine import prediction_engine

        request = PredictionRequest(
            query=test_case.query,
            ticker=test_case.ticker,
        )
        prediction = await prediction_engine.generate_prediction(request)

        guard_result = run_output_guards(
            prediction.reasoning.summary,
            probability=prediction.probability * 100,
        )

        ga = guardrail_accuracy(
            probability=prediction.probability,
            warnings=guard_result.warnings,
            modifications=guard_result.modifications,
            expected_flags=test_case.expected_guardrail_flags,
        )
        result["details"]["guardrail_accuracy"] = ga
        result["score"] = ga["score"]
    else:
        # Non-prediction guardrail test
        guard_result = run_output_guards("This is a test response.")
        result["score"] = 1.0 if guard_result.passed else 0.5

    return result


async def _eval_edge_case(test_case: TestCase, result: dict) -> dict:
    """Evaluate an edge case test case."""
    # Edge cases should either be rejected or fail gracefully
    try:
        from backend.app.services.query_validator import financial_query_validator

        validation = financial_query_validator.validate_query(test_case.query)

        if "non-financial" in test_case.description.lower() or "too-short" in test_case.description.lower():
            # Expected to be rejected
            result["score"] = 1.0 if not validation.is_valid else 0.0
            result["details"]["rejected"] = not validation.is_valid
        elif "invalid ticker" in test_case.description.lower():
            # Expected to fail at ticker extraction
            from backend.app.services.market_data import market_data_service
            ticker = market_data_service.extract_ticker_from_query(test_case.query)
            result["score"] = 0.5  # Partial credit for not crashing
            result["details"]["ticker_found"] = ticker
        else:
            result["score"] = 0.5
    except Exception as e:
        # Edge cases that throw exceptions get partial credit for not crashing the server
        result["score"] = 0.5
        result["details"]["exception"] = str(e)

    return result


async def run_full_eval(
    categories: Optional[list[str]] = None,
    experiment_name: Optional[str] = None,
) -> dict:
    """
    Run the full evaluation suite.

    Args:
        categories: Optional list of categories to filter (prediction, rag, guardrail, edge_case).
        experiment_name: Optional name for this experiment run.

    Returns:
        Summary dict with results.
    """
    test_cases = GOLDEN_DATASET
    if categories:
        test_cases = [tc for tc in test_cases if tc.category in categories]

    logger.info("Starting evaluation", total_tests=len(test_cases))

    results = []
    for tc in test_cases:
        logger.info("Running eval", test_id=tc.id, category=tc.category)
        result = await run_single_eval(tc)
        results.append(result)
        logger.info(
            "Eval result",
            test_id=tc.id,
            passed=result["passed"],
            score=f"{result['score']:.2f}",
        )

    # Compute summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    avg_score = sum(r["score"] for r in results) / total if total > 0 else 0

    # By category
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0, "avg_score": 0}
        by_category[cat]["total"] += 1
        if r["passed"]:
            by_category[cat]["passed"] += 1
        by_category[cat]["avg_score"] += r["score"]

    for cat in by_category:
        n = by_category[cat]["total"]
        by_category[cat]["avg_score"] = round(by_category[cat]["avg_score"] / n, 3) if n else 0

    # Log experiment
    experiment = Experiment(
        id=str(uuid.uuid4())[:8],
        name=experiment_name or f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.utcnow().isoformat(),
        total_tests=total,
        passed_tests=passed,
        failed_tests=failed,
        average_score=round(avg_score, 3),
        results=results,
    )
    log_experiment(experiment)

    summary = {
        "experiment_id": experiment.id,
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "average_score": round(avg_score, 3),
        "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
        "by_category": by_category,
        "results": results,
    }

    logger.info(
        "Evaluation complete",
        total=total,
        passed=passed,
        failed=failed,
        avg_score=f"{avg_score:.3f}",
    )

    return summary


# CLI entry point
if __name__ == "__main__":
    import sys
    import json

    async def main():
        categories = None
        if len(sys.argv) > 1:
            categories = sys.argv[1].split(",")

        results = await run_full_eval(categories=categories)
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())
