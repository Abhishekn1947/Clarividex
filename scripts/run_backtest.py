#!/usr/bin/env python3
"""
Backtest CLI for Clarividex.

Run prediction engine backtests from the command line.

Usage:
    python scripts/run_backtest.py NVDA                    # Basic backtest
    python scripts/run_backtest.py NVDA --return 0.15      # Test +15% predictions
    python scripts/run_backtest.py NVDA,AAPL,MSFT --multi  # Multiple tickers
    python scripts/run_backtest.py NVDA --analyze          # Signal analysis

Examples:
    # Test +10% predictions over 30 days for NVDA
    python scripts/run_backtest.py NVDA

    # Test -5% predictions (bearish) over 14 days
    python scripts/run_backtest.py TSLA --return -0.05 --days 14

    # Test multiple tickers
    python scripts/run_backtest.py "NVDA,AAPL,MSFT,GOOGL,AMZN" --multi

    # Analyze which signals work best for a ticker
    python scripts/run_backtest.py NVDA --analyze
"""

import asyncio
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.services.backtesting_engine import backtesting_engine


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str):
    """Print a section header."""
    print(f"\n--- {text} ---")


def print_bar(value: float, max_width: int = 30):
    """Print a progress bar."""
    filled = int(value * max_width)
    bar = "█" * filled + "░" * (max_width - filled)
    return bar


async def run_single_backtest(
    ticker: str,
    target_return: float,
    holding_period: int,
    lookback_days: int,
):
    """Run backtest for a single ticker."""
    print_header(f"BACKTEST: {ticker}")
    print(f"Target: {target_return:+.1%} over {holding_period} days")
    print(f"Testing {lookback_days} days of history...")
    print()

    results = await backtesting_engine.run_backtest(
        ticker=ticker,
        target_return=target_return,
        holding_period=holding_period,
        lookback_days=lookback_days,
        step_days=5,
    )

    if results.total_predictions == 0:
        print("ERROR: No predictions generated. Check ticker symbol.")
        return

    # Print results
    print_section("ACCURACY METRICS")
    print(f"  Total Predictions:  {results.total_predictions}")
    print(f"  Correct:            {results.correct_predictions}")
    print()

    # Accuracy with visual bar
    acc_bar = print_bar(results.accuracy)
    print(f"  Accuracy:   {results.accuracy:>6.1%}  {acc_bar}")

    prec_bar = print_bar(results.precision)
    print(f"  Precision:  {results.precision:>6.1%}  {prec_bar}")

    rec_bar = print_bar(results.recall)
    print(f"  Recall:     {results.recall:>6.1%}  {rec_bar}")

    f1_bar = print_bar(results.f1_score)
    print(f"  F1 Score:   {results.f1_score:>6.1%}  {f1_bar}")

    print_section("PROBABILITY CALIBRATION")
    print(f"  Avg Predicted Probability: {results.avg_probability:.1f}%")
    print(f"  Actual Hit Rate:           {results.avg_actual_hit_rate:.1f}%")
    print(f"  Calibration Error:         {results.calibration_error:.1f}%")

    if results.calibration_error > 10:
        print("  ⚠️  High calibration error - probabilities need adjustment")
    elif results.calibration_error < 5:
        print("  ✓  Good calibration - probabilities are well-calibrated")

    if results.accuracy_by_regime:
        print_section("ACCURACY BY MARKET REGIME")
        for regime, acc in sorted(results.accuracy_by_regime.items(), key=lambda x: x[1], reverse=True):
            bar = print_bar(acc, 20)
            print(f"  {regime.capitalize():12} {acc:>6.1%}  {bar}")

    if results.accuracy_by_volatility:
        print_section("ACCURACY BY VOLATILITY")
        for vol, acc in sorted(results.accuracy_by_volatility.items(), key=lambda x: x[1], reverse=True):
            bar = print_bar(acc, 20)
            print(f"  {vol.capitalize():12} {acc:>6.1%}  {bar}")

    if results.component_correlations:
        print_section("COMPONENT EFFECTIVENESS (Top 5)")
        for comp, acc in list(results.component_correlations.items())[:5]:
            bar = print_bar(acc, 20)
            indicator = "✓" if acc > 0.55 else "⚠️" if acc > 0.50 else "✗"
            print(f"  {indicator} {comp:20} {acc:>6.1%}  {bar}")

    print_section("INTERPRETATION")
    if results.accuracy >= 0.60:
        print("  ✓ GOOD: Model has significant predictive power")
        print("    The prediction engine performs well on this ticker.")
    elif results.accuracy >= 0.55:
        print("  ⚠️ MODERATE: Model has some predictive power")
        print("    Consider adding more data sources or adjusting weights.")
    elif results.accuracy >= 0.52:
        print("  ⚠️ SLIGHT EDGE: Model is marginally better than random")
        print("    Significant improvements needed for production use.")
    else:
        print("  ✗ NO EDGE: Model performs at or below random chance")
        print("    Major algorithm or data improvements required.")

    print_section("RECOMMENDATIONS")
    if results.accuracy < 0.55:
        print("  1. Add ensemble voting (combine multiple models)")
        print("  2. Integrate options flow data (Polygon.io)")
        print("  3. Add earnings surprise history")
        print("  4. Implement adaptive signal weights")
    elif results.calibration_error > 10:
        print("  1. Adjust probability bounds based on historical accuracy")
        print("  2. Implement probability calibration layer")
    else:
        print("  1. Expand testing to more tickers")
        print("  2. Monitor for regime changes")
        print("  3. Consider production deployment")

    print("\n" + "=" * 60 + "\n")

    return results


async def run_multi_backtest(
    tickers: list,
    target_return: float,
    holding_period: int,
):
    """Run backtest for multiple tickers."""
    print_header(f"MULTI-TICKER BACKTEST")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Target: {target_return:+.1%} over {holding_period} days")
    print()

    results = await backtesting_engine.run_multi_ticker_backtest(
        tickers=tickers,
        target_return=target_return,
        holding_period=holding_period,
        lookback_days=252,
    )

    # Calculate aggregate
    total_preds = sum(r.total_predictions for r in results.values())
    total_correct = sum(r.correct_predictions for r in results.values())
    overall_accuracy = total_correct / total_preds if total_preds > 0 else 0

    print_section("RESULTS BY TICKER")
    print(f"  {'Ticker':<8} {'Accuracy':>10} {'Predictions':>12} {'Correct':>10}")
    print("  " + "-" * 45)

    for ticker, r in sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True):
        bar = print_bar(r.accuracy, 15)
        print(f"  {ticker:<8} {r.accuracy:>9.1%}  {r.total_predictions:>11}  {r.correct_predictions:>9}  {bar}")

    print_section("AGGREGATE RESULTS")
    agg_bar = print_bar(overall_accuracy)
    print(f"  Overall Accuracy: {overall_accuracy:>6.1%}  {agg_bar}")
    print(f"  Total Predictions: {total_preds}")
    print(f"  Total Correct: {total_correct}")

    print("\n" + "=" * 60 + "\n")


async def run_signal_analysis(ticker: str):
    """Analyze signal effectiveness for a ticker."""
    print_header(f"SIGNAL ANALYSIS: {ticker}")
    print("Analyzing which signals are most predictive...")
    print()

    analysis = await backtesting_engine.analyze_signal_effectiveness(
        ticker=ticker,
        lookback_days=252,
    )

    overall_bar = print_bar(analysis['overall_accuracy'])
    print(f"Overall Accuracy: {analysis['overall_accuracy']:>6.1%}  {overall_bar}")

    print_section("BEST PERFORMING SIGNALS")
    for signal, acc in analysis['best_components']:
        bar = print_bar(acc, 20)
        print(f"  ✓ {signal:<25} {acc:>6.1%}  {bar}")

    print_section("WORST PERFORMING SIGNALS")
    for signal, acc in analysis['worst_components']:
        bar = print_bar(acc, 20)
        indicator = "✗" if acc < 0.50 else "⚠️"
        print(f"  {indicator} {signal:<25} {acc:>6.1%}  {bar}")

    print_section("RECOMMENDATIONS")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest the Clarividex prediction engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_backtest.py NVDA                      # Basic backtest
  python scripts/run_backtest.py NVDA --return 0.15        # Test +15% predictions
  python scripts/run_backtest.py NVDA --return -0.05       # Test bearish predictions
  python scripts/run_backtest.py NVDA,AAPL,MSFT --multi    # Multiple tickers
  python scripts/run_backtest.py NVDA --analyze            # Signal analysis
        """,
    )

    parser.add_argument(
        "ticker",
        help="Stock ticker(s) to backtest (comma-separated for multi)",
    )
    parser.add_argument(
        "--return", "-r",
        dest="target_return",
        type=float,
        default=0.10,
        help="Target return (default: 0.10 = +10%%)",
    )
    parser.add_argument(
        "--days", "-d",
        dest="holding_period",
        type=int,
        default=30,
        help="Holding period in days (default: 30)",
    )
    parser.add_argument(
        "--lookback", "-l",
        dest="lookback_days",
        type=int,
        default=252,
        help="Days of history to test (default: 252 = 1 year)",
    )
    parser.add_argument(
        "--multi", "-m",
        action="store_true",
        help="Run multi-ticker backtest",
    )
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Analyze signal effectiveness instead of running backtest",
    )

    args = parser.parse_args()

    # Parse tickers
    tickers = [t.strip().upper() for t in args.ticker.split(",")]

    if args.analyze:
        # Run signal analysis
        asyncio.run(run_signal_analysis(tickers[0]))
    elif args.multi or len(tickers) > 1:
        # Run multi-ticker backtest
        asyncio.run(run_multi_backtest(
            tickers=tickers,
            target_return=args.target_return,
            holding_period=args.holding_period,
        ))
    else:
        # Run single backtest
        asyncio.run(run_single_backtest(
            ticker=tickers[0],
            target_return=args.target_return,
            holding_period=args.holding_period,
            lookback_days=args.lookback_days,
        ))


if __name__ == "__main__":
    main()
