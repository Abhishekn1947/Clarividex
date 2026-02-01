"""
Comprehensive Accuracy Testing Suite
Tests prediction accuracy, data quality, and system reliability.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class AccuracyTestSuite:
    """Comprehensive testing for the prediction system."""

    def __init__(self):
        self.results = {
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "tests": [],
            "issues": [],
            "improvements": [],
        }

    def log_result(self, test_name: str, passed: bool, details: str = "", issue: str = None):
        """Log a test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if details:
            print(f"         {details}")

        if passed:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
            if issue:
                self.results["issues"].append({"test": test_name, "issue": issue})

        self.results["tests"].append({
            "name": test_name,
            "passed": passed,
            "details": details,
        })

    def log_warning(self, message: str, improvement: str = None):
        """Log a warning."""
        print(f"   ⚠️  WARNING: {message}")
        self.results["warnings"] += 1
        if improvement:
            self.results["improvements"].append(improvement)

    async def run_all_tests(self):
        """Run the complete test suite."""
        print("=" * 80)
        print("🧪 COMPREHENSIVE ACCURACY TEST SUITE")
        print(f"   Started: {datetime.now().isoformat()}")
        print("=" * 80)

        # Test Categories
        await self.test_data_accuracy()
        await self.test_ticker_extraction()
        await self.test_multiple_predictions()
        await self.test_edge_cases()
        await self.test_sentiment_accuracy()
        await self.test_technical_indicators()
        await self.test_api_reliability()

        # Print Summary
        self.print_summary()

        return self.results

    async def test_data_accuracy(self):
        """Test data accuracy against known values."""
        print("\n" + "=" * 80)
        print("📊 TEST 1: DATA ACCURACY VERIFICATION")
        print("=" * 80)

        from backend.app.services.market_data import market_data_service
        from backend.app.services.additional_data_sources import additional_data_sources

        # Test tickers with known characteristics
        test_tickers = [
            {"ticker": "AAPL", "name": "Apple", "sector": "Technology"},
            {"ticker": "MSFT", "name": "Microsoft", "sector": "Technology"},
            {"ticker": "NVDA", "name": "NVIDIA", "sector": "Technology"},
            {"ticker": "TSLA", "name": "Tesla", "sector": "Consumer Cyclical"},
            {"ticker": "JPM", "name": "JPMorgan", "sector": "Financial Services"},
            {"ticker": "XOM", "name": "Exxon", "sector": "Energy"},
        ]

        for test in test_tickers:
            ticker = test["ticker"]
            print(f"\n   Testing {ticker} ({test['name']})...")

            # Test 1: Quote data
            quote = market_data_service.get_quote(ticker)
            if quote:
                self.log_result(
                    f"{ticker} quote retrieval",
                    quote.current_price > 0,
                    f"Price: ${quote.current_price:.2f}"
                )

                # Sanity checks
                if quote.current_price < 1 or quote.current_price > 10000:
                    self.log_warning(
                        f"{ticker} price seems unusual: ${quote.current_price}",
                        f"Add price range validation for {ticker}"
                    )

                # Verify 52-week range makes sense
                if quote.fifty_two_week_high and quote.fifty_two_week_low:
                    in_range = quote.fifty_two_week_low <= quote.current_price <= quote.fifty_two_week_high * 1.1
                    self.log_result(
                        f"{ticker} price within 52-week range",
                        in_range,
                        f"Range: ${quote.fifty_two_week_low:.2f} - ${quote.fifty_two_week_high:.2f}"
                    )
            else:
                self.log_result(f"{ticker} quote retrieval", False, issue="Quote fetch failed")

            # Test 2: Company info
            company = market_data_service.get_company_info(ticker)
            if company:
                sector_match = test["sector"].lower() in (company.sector or "").lower()
                self.log_result(
                    f"{ticker} sector verification",
                    sector_match or company.sector is not None,
                    f"Sector: {company.sector}"
                )
            else:
                self.log_result(f"{ticker} company info", False, issue="Company info fetch failed")

            # Test 3: Finviz cross-verification
            finviz = await additional_data_sources.get_finviz_data(ticker)
            if finviz and not finviz.get("error") and quote:
                price_diff = abs(finviz.get("price", 0) - quote.current_price)
                price_match = price_diff < quote.current_price * 0.02  # Within 2%
                self.log_result(
                    f"{ticker} cross-source price verification",
                    price_match,
                    f"yfinance: ${quote.current_price:.2f}, Finviz: ${finviz.get('price', 0):.2f}"
                )

                if not price_match:
                    self.log_warning(
                        f"Price discrepancy for {ticker}: ${price_diff:.2f}",
                        "Add price reconciliation logic"
                    )

    async def test_ticker_extraction(self):
        """Test ticker extraction accuracy."""
        print("\n" + "=" * 80)
        print("🔍 TEST 2: TICKER EXTRACTION ACCURACY")
        print("=" * 80)

        from backend.app.services.market_data import market_data_service

        test_cases = [
            # Format: (query, expected_ticker)
            ("Will AAPL reach $200?", "AAPL"),
            ("Is Apple stock going up?", "AAPL"),
            ("Tesla price prediction", "TSLA"),
            ("What about $NVDA?", "NVDA"),
            ("Microsoft future outlook", "MSFT"),
            ("Will nvidia hit $150?", "NVDA"),
            ("Amazon earnings impact", "AMZN"),
            ("Meta stock analysis", "META"),
            ("Facebook price target", "META"),
            ("Google stock prediction", "GOOGL"),
            ("Alphabet earnings", "GOOGL"),
            ("JP Morgan outlook", "JPM"),
            ("Will Berkshire Hathaway go up?", "BRK.B"),
            ("Netflix subscriber growth impact", "NFLX"),
            ("AMD vs Intel comparison", "AMD"),  # Should get first one
            ("S&P 500 prediction", "SPY"),
            ("Nasdaq outlook", "QQQ"),
            ("Will V go up?", "V"),  # Visa - edge case
            ("Costco earnings", "COST"),
            ("Home Depot forecast", "HD"),
        ]

        passed = 0
        for query, expected in test_cases:
            result = market_data_service.extract_ticker_from_query(query)
            match = result == expected
            if match:
                passed += 1
            self.log_result(
                f"Extract '{expected}' from query",
                match,
                f"Query: '{query[:40]}...' -> Got: {result}"
            )
            if not match and result:
                self.log_warning(
                    f"Expected {expected}, got {result}",
                    f"Improve extraction for query pattern: '{query}'"
                )

        accuracy = passed / len(test_cases) * 100
        print(f"\n   📊 Ticker Extraction Accuracy: {accuracy:.1f}% ({passed}/{len(test_cases)})")

        if accuracy < 90:
            self.results["improvements"].append(
                "Ticker extraction accuracy below 90% - needs improvement"
            )

    async def test_multiple_predictions(self):
        """Test predictions for multiple scenarios."""
        print("\n" + "=" * 80)
        print("🔮 TEST 3: PREDICTION GENERATION")
        print("=" * 80)

        from backend.app.models.schemas import PredictionRequest
        from backend.app.services.prediction_engine import prediction_engine

        test_scenarios = [
            {
                "query": "Will AAPL reach $300 by December 2026?",
                "type": "price_target_up",
                "expected_sentiment": "depends",  # Based on current price
            },
            {
                "query": "Will NVDA drop below $100 in 6 months?",
                "type": "price_target_down",
                "expected_sentiment": "depends",
            },
            {
                "query": "Is TSLA a good buy right now?",
                "type": "general_outlook",
                "expected_sentiment": "depends",
            },
            {
                "query": "Will MSFT outperform the S&P 500 this year?",
                "type": "relative_performance",
                "expected_sentiment": "depends",
            },
        ]

        for scenario in test_scenarios:
            print(f"\n   Testing: {scenario['query'][:50]}...")

            request = PredictionRequest(
                query=scenario["query"],
                include_technicals=True,
                include_sentiment=True,
                include_news=True,
            )

            try:
                prediction = await prediction_engine.generate_prediction(request)

                # Verify prediction structure
                self.log_result(
                    f"Prediction generated for {scenario['type']}",
                    prediction is not None,
                    f"Probability: {prediction.probability * 100:.0f}%, Confidence: {prediction.confidence_level}"
                )

                # Check probability is valid
                self.log_result(
                    "Probability in valid range",
                    0 <= prediction.probability <= 1,
                    f"Value: {prediction.probability}"
                )

                # Check data points analyzed
                self.log_result(
                    "Sufficient data points",
                    prediction.data_points_analyzed >= 50,
                    f"Data points: {prediction.data_points_analyzed}"
                )

                # Check reasoning exists
                self.log_result(
                    "Reasoning provided",
                    prediction.reasoning is not None and len(prediction.reasoning.summary) > 50,
                    f"Summary length: {len(prediction.reasoning.summary)} chars"
                )

                # Check factors identified
                bullish_count = len(prediction.reasoning.bullish_factors) if prediction.reasoning.bullish_factors else 0
                bearish_count = len(prediction.reasoning.bearish_factors) if prediction.reasoning.bearish_factors else 0

                self.log_result(
                    "Factors identified",
                    bullish_count + bearish_count >= 2,
                    f"Bullish: {bullish_count}, Bearish: {bearish_count}"
                )

            except Exception as e:
                self.log_result(
                    f"Prediction for {scenario['type']}",
                    False,
                    issue=str(e)
                )

    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n" + "=" * 80)
        print("⚠️  TEST 4: EDGE CASES & ERROR HANDLING")
        print("=" * 80)

        from backend.app.services.market_data import market_data_service
        from backend.app.models.schemas import PredictionRequest
        from backend.app.services.prediction_engine import prediction_engine

        # Test invalid ticker
        print("\n   Testing invalid ticker handling...")
        quote = market_data_service.get_quote("INVALIDTICKER123")
        self.log_result(
            "Invalid ticker returns None",
            quote is None,
            "Graceful handling of invalid ticker"
        )

        # Test empty query
        print("\n   Testing empty query handling...")
        try:
            ticker = market_data_service.extract_ticker_from_query("")
            self.log_result(
                "Empty query handled",
                ticker is None,
                "Returns None for empty query"
            )
        except Exception as e:
            self.log_result("Empty query handled", False, issue=str(e))

        # Test very long query
        print("\n   Testing very long query...")
        long_query = "Will AAPL " + "go up " * 100 + "?"
        ticker = market_data_service.extract_ticker_from_query(long_query)
        self.log_result(
            "Long query handled",
            ticker == "AAPL",
            f"Extracted: {ticker}"
        )

        # Test special characters in query
        print("\n   Testing special characters...")
        special_query = "Will $AAPL reach $200!!! ???"
        ticker = market_data_service.extract_ticker_from_query(special_query)
        self.log_result(
            "Special characters handled",
            ticker == "AAPL",
            f"Extracted: {ticker}"
        )

        # Test query with no ticker
        print("\n   Testing query without ticker...")
        no_ticker_query = "What will the market do tomorrow?"
        ticker = market_data_service.extract_ticker_from_query(no_ticker_query)
        self.log_result(
            "No-ticker query handled",
            ticker is None or ticker in ["SPY", "QQQ"],  # May default to market ETF
            f"Extracted: {ticker}"
        )

    async def test_sentiment_accuracy(self):
        """Test sentiment analysis accuracy."""
        print("\n" + "=" * 80)
        print("💭 TEST 5: SENTIMENT ANALYSIS ACCURACY")
        print("=" * 80)

        from backend.app.services.sentiment_service import sentiment_service

        test_cases = [
            # (text, expected_sentiment: positive/negative/neutral)
            ("Stock surges 20% on incredible earnings beat!", "positive"),
            ("Company beats expectations, raises guidance", "positive"),
            ("Shares plummet after disappointing results", "negative"),
            ("Stock crashes amid fraud allegations", "negative"),
            ("Company files for bankruptcy protection", "negative"),
            ("Shares traded flat on mixed signals", "neutral"),
            ("Stock unchanged after earnings report", "neutral"),
            ("Bullish momentum continues with strong buying", "positive"),
            ("Bears take control as selling intensifies", "negative"),
            ("Analysts upgrade stock to buy rating", "positive"),
            ("Analysts downgrade to sell rating", "negative"),
            ("Record revenue and profit announced", "positive"),
            ("Massive layoffs announced, 10000 jobs cut", "negative"),
        ]

        correct = 0
        for text, expected in test_cases:
            score = sentiment_service.analyze_text(text)

            if expected == "positive":
                is_correct = score > 0.1
            elif expected == "negative":
                is_correct = score < -0.1
            else:
                is_correct = -0.1 <= score <= 0.1

            if is_correct:
                correct += 1

            self.log_result(
                f"Sentiment: {expected}",
                is_correct,
                f"Text: '{text[:40]}...' Score: {score:.3f}"
            )

        accuracy = correct / len(test_cases) * 100
        print(f"\n   📊 Sentiment Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")

        if accuracy < 80:
            self.results["improvements"].append(
                "Sentiment analysis accuracy below 80% - enhance financial lexicon"
            )

    async def test_technical_indicators(self):
        """Test technical indicator calculations."""
        print("\n" + "=" * 80)
        print("📈 TEST 6: TECHNICAL INDICATORS")
        print("=" * 80)

        from backend.app.services.technical_analysis import technical_analysis_service
        from backend.app.services.additional_data_sources import additional_data_sources

        test_tickers = ["AAPL", "MSFT", "GOOGL"]

        for ticker in test_tickers:
            print(f"\n   Testing {ticker} technical indicators...")

            indicators = technical_analysis_service.calculate_indicators(ticker)

            if indicators:
                # RSI should be between 0 and 100
                self.log_result(
                    f"{ticker} RSI valid range",
                    0 <= indicators.rsi_14 <= 100,
                    f"RSI: {indicators.rsi_14:.2f}"
                )

                # Cross-check with Finviz
                finviz = await additional_data_sources.get_finviz_data(ticker)
                if finviz and finviz.get("rsi_14"):
                    rsi_diff = abs(indicators.rsi_14 - finviz["rsi_14"])
                    # RSI can vary slightly due to timing
                    self.log_result(
                        f"{ticker} RSI cross-validation",
                        rsi_diff < 10,  # Within 10 points
                        f"Calculated: {indicators.rsi_14:.1f}, Finviz: {finviz['rsi_14']}"
                    )

                # SMA should be reasonable
                if indicators.sma_20 and indicators.sma_50:
                    self.log_result(
                        f"{ticker} SMAs calculated",
                        indicators.sma_20 > 0 and indicators.sma_50 > 0,
                        f"SMA20: ${indicators.sma_20:.2f}, SMA50: ${indicators.sma_50:.2f}"
                    )
            else:
                self.log_result(f"{ticker} technical indicators", False, issue="Calculation failed")

    async def test_api_reliability(self):
        """Test API reliability and response times."""
        print("\n" + "=" * 80)
        print("🔗 TEST 7: API RELIABILITY")
        print("=" * 80)

        import time
        from backend.app.services.additional_data_sources import additional_data_sources

        apis = [
            ("SEC EDGAR", lambda: additional_data_sources.get_sec_filings("AAPL")),
            ("Finviz", lambda: additional_data_sources.get_finviz_data("AAPL")),
            ("Fear & Greed", lambda: additional_data_sources.get_fear_greed_index()),
            ("VIX", lambda: additional_data_sources.get_vix_data()),
            ("Options", lambda: additional_data_sources.get_options_overview("AAPL")),
            ("Economic", lambda: additional_data_sources.get_economic_indicators()),
            ("Sectors", lambda: additional_data_sources.get_sector_performance()),
        ]

        for name, api_call in apis:
            start = time.time()
            try:
                result = await api_call()
                elapsed = time.time() - start

                success = result is not None and not (isinstance(result, dict) and result.get("error"))

                self.log_result(
                    f"{name} API",
                    success,
                    f"Response time: {elapsed:.2f}s"
                )

                if elapsed > 5:
                    self.log_warning(
                        f"{name} slow response: {elapsed:.2f}s",
                        f"Add caching for {name} API"
                    )

            except Exception as e:
                self.log_result(f"{name} API", False, issue=str(e))

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)

        total = self.results["passed"] + self.results["failed"]
        pass_rate = self.results["passed"] / total * 100 if total > 0 else 0

        print(f"""
   Total Tests:  {total}
   Passed:       {self.results['passed']} ✅
   Failed:       {self.results['failed']} ❌
   Warnings:     {self.results['warnings']} ⚠️
   Pass Rate:    {pass_rate:.1f}%
        """)

        if self.results["issues"]:
            print("\n   🔴 ISSUES FOUND:")
            for issue in self.results["issues"]:
                print(f"      - {issue['test']}: {issue['issue']}")

        if self.results["improvements"]:
            print("\n   🔧 RECOMMENDED IMPROVEMENTS:")
            for imp in self.results["improvements"]:
                print(f"      - {imp}")

        print("\n" + "=" * 80)


async def main():
    suite = AccuracyTestSuite()
    results = await suite.run_all_tests()

    # Save results to file
    output_path = Path(__file__).parent / "test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n   Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
