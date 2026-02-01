"""
Test all data sources for accuracy and availability.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


async def test_all_data_sources():
    """Test all data sources for AAPL."""
    ticker = "AAPL"

    print("=" * 70)
    print(f"TESTING ALL DATA SOURCES FOR {ticker}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Test 1: Basic Market Data (yfinance)
    print("\n📊 1. MARKET DATA (yfinance)")
    print("-" * 50)
    from backend.app.services.market_data import market_data_service

    quote = market_data_service.get_quote(ticker)
    if quote:
        print(f"   ✅ Current Price: ${quote.current_price:.2f}")
        print(f"   ✅ Day Change: {quote.change_percent:+.2f}%")
        print(f"   ✅ 52W High: ${quote.fifty_two_week_high:.2f}")
        print(f"   ✅ 52W Low: ${quote.fifty_two_week_low:.2f}")
        print(f"   ✅ P/E Ratio: {quote.pe_ratio:.2f}" if quote.pe_ratio else "   ⚠️  P/E: N/A")
    else:
        print("   ❌ Failed to fetch quote")

    company = market_data_service.get_company_info(ticker)
    if company:
        print(f"   ✅ Company: {company.name}")
        print(f"   ✅ Sector: {company.sector}")

    # Test 2: Technical Analysis
    print("\n📈 2. TECHNICAL ANALYSIS")
    print("-" * 50)
    from backend.app.services.technical_analysis import technical_analysis_service

    technicals = technical_analysis_service.calculate_indicators(ticker)
    if technicals:
        print(f"   ✅ RSI (14): {technicals.rsi_14:.2f}")
        print(f"   ✅ SMA 20: ${technicals.sma_20:.2f}")
        print(f"   ✅ SMA 50: ${technicals.sma_50:.2f}")
        print(f"   ✅ MACD: {technicals.macd:.4f}")
        print(f"   ✅ Signal: {technicals.rsi_signal}")
    else:
        print("   ❌ Failed to calculate technicals")

    # Test 3: News (Google News)
    print("\n📰 3. NEWS (Google News RSS)")
    print("-" * 50)
    from backend.app.services.news_service import news_service

    news = await news_service.get_news_for_ticker(ticker, "Apple Inc")
    print(f"   ✅ Articles fetched: {len(news)}")
    for article in news[:3]:
        sentiment = "📈" if article.sentiment_score > 0.1 else "📉" if article.sentiment_score < -0.1 else "➖"
        print(f"   {sentiment} {article.title[:60]}...")

    # Test 4: Social Sentiment
    print("\n💬 4. SOCIAL SENTIMENT")
    print("-" * 50)
    from backend.app.services.social_service import social_service

    social = await social_service.get_social_sentiment(ticker, "Apple")
    print(f"   ✅ Platforms: {len(social)}")
    for s in social:
        print(f"   ✅ {s.platform}: score={s.sentiment_score:.3f}, mentions={s.mentions_count}")

    # Test 5: Additional Data Sources
    print("\n🔍 5. ADDITIONAL DATA SOURCES")
    print("-" * 50)
    from backend.app.services.additional_data_sources import additional_data_sources

    # SEC Filings
    print("\n   📋 SEC EDGAR Filings:")
    sec = await additional_data_sources.get_sec_filings(ticker)
    if sec and not sec.get("error"):
        print(f"      ✅ Company: {sec.get('company_name')}")
        print(f"      ✅ CIK: {sec.get('cik')}")
        print(f"      ✅ Total Filings: {sec.get('total_filings', 0)}")
        print(f"      ✅ Insider Filings: {sec.get('insider_filings', 0)}")
        if sec.get("last_10k"):
            print(f"      ✅ Last 10-K: {sec['last_10k'].get('date')}")
    else:
        print(f"      ⚠️  SEC: {sec.get('error', 'unavailable')}")

    # Finviz Data
    print("\n   📊 Finviz Screener Data:")
    finviz = await additional_data_sources.get_finviz_data(ticker)
    if finviz and not finviz.get("error"):
        print(f"      ✅ Price: ${finviz.get('price', 0):.2f}")
        print(f"      ✅ Target Price: ${finviz.get('target_price', 0):.2f}")
        print(f"      ✅ Upside: {finviz.get('upside_pct', 0):.1f}%")
        print(f"      ✅ Analyst Rating: {finviz.get('recommendation_text', 'N/A')}")
        print(f"      ✅ Short Float: {finviz.get('short_float', 'N/A')}")
        print(f"      ✅ RSI: {finviz.get('rsi_14', 'N/A')}")
    else:
        print(f"      ⚠️  Finviz: {finviz.get('error', 'unavailable')}")

    # Fear & Greed
    print("\n   😨 Fear & Greed Index:")
    fg = await additional_data_sources.get_fear_greed_index()
    if fg:
        print(f"      ✅ Value: {fg.get('value', 'N/A')}")
        print(f"      ✅ Rating: {fg.get('rating', 'N/A')}")

    # VIX
    print("\n   📉 VIX (Volatility Index):")
    vix = await additional_data_sources.get_vix_data()
    if vix and not vix.get("error"):
        print(f"      ✅ VIX: {vix.get('value', 'N/A')}")
        print(f"      ✅ Interpretation: {vix.get('interpretation', 'N/A')}")
        print(f"      ✅ Market Sentiment: {vix.get('market_sentiment', 'N/A')}")
    else:
        print(f"      ⚠️  VIX: {vix.get('error', 'unavailable')}")

    # Options
    print("\n   📊 Options Data:")
    options = await additional_data_sources.get_options_overview(ticker)
    if options and not options.get("error"):
        print(f"      ✅ Put/Call Ratio: {options.get('put_call_ratio_volume', 'N/A')}")
        print(f"      ✅ Options Sentiment: {options.get('sentiment', 'N/A')}")
        print(f"      ✅ Call Volume: {options.get('call_volume', 0):,}")
        print(f"      ✅ Put Volume: {options.get('put_volume', 0):,}")
    else:
        print(f"      ⚠️  Options: {options.get('error', 'unavailable')}")

    # Economic Indicators
    print("\n   🏛️  Economic Indicators:")
    econ = await additional_data_sources.get_economic_indicators()
    if econ:
        if econ.get("treasury_10y"):
            print(f"      ✅ 10Y Treasury: {econ['treasury_10y'].get('value', 'N/A')}%")
        if econ.get("treasury_30y"):
            print(f"      ✅ 30Y Treasury: {econ['treasury_30y'].get('value', 'N/A')}%")

    # Sector Performance
    print("\n   📊 Sector Performance:")
    sectors = await additional_data_sources.get_sector_performance()
    if sectors:
        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1].get("change_pct", 0), reverse=True)
        for sector, data in sorted_sectors[:5]:
            change = data.get("change_pct", 0)
            emoji = "📈" if change > 0 else "📉"
            print(f"      {emoji} {sector}: {change:+.2f}%")

    # Additional News
    print("\n   📰 Multi-Source News:")
    add_news = await additional_data_sources.get_aggregated_news(ticker, limit=5)
    print(f"      ✅ Articles: {len(add_news)}")
    for article in add_news[:3]:
        print(f"      [{article.get('source', 'Unknown')}] {article.get('title', '')[:50]}...")

    # Test 6: Full Data Aggregation
    print("\n🔄 6. FULL DATA AGGREGATION")
    print("-" * 50)
    from backend.app.services.data_aggregator import data_aggregator

    aggregated = await data_aggregator.aggregate_data(ticker)
    print(f"   ✅ Data Sources Used: {len(aggregated.data_sources)}")
    for source in aggregated.data_sources:
        print(f"      - {source.name} (reliability: {source.reliability_score})")
    print(f"   ✅ Data Quality Score: {aggregated.data_quality_score:.2f}")
    print(f"   ✅ Total Data Points: {aggregated.data_points_count}")

    # Summary
    print("\n" + "=" * 70)
    print("📊 DATA SOURCES SUMMARY")
    print("=" * 70)

    sources = [
        ("yfinance (Market Data)", quote is not None),
        ("Technical Analysis", technicals is not None),
        ("Google News RSS", len(news) > 0),
        ("Social Sentiment", len(social) > 0),
        ("SEC EDGAR", sec and not sec.get("error")),
        ("Finviz", finviz and not finviz.get("error")),
        ("Fear & Greed Index", fg and fg.get("value")),
        ("VIX Data", vix and not vix.get("error")),
        ("Options Data", options and not options.get("error")),
        ("Economic Indicators", bool(econ)),
        ("Sector Performance", bool(sectors)),
        ("Multi-Source News", len(add_news) > 0),
    ]

    working = sum(1 for _, status in sources if status)
    print(f"\nWorking Sources: {working}/{len(sources)}")
    print()
    for name, status in sources:
        emoji = "✅" if status else "❌"
        print(f"   {emoji} {name}")

    print(f"\n📈 Overall Data Quality: {aggregated.data_quality_score * 100:.0f}%")
    print(f"📊 Total Data Points: {aggregated.data_points_count}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_all_data_sources())
