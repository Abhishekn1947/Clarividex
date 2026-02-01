"""
Scenario Analysis Engine for Clarividex.

Addresses fintech bottleneck: Users want "what-if" analysis.

This module allows users to ask hypothetical questions:
- "What if interest rates rise 1%?"
- "What if there's a recession?"
- "What if earnings beat by 20%?"
- "What if oil prices spike?"

Usage:
    from backend.app.services.scenario_analyzer import scenario_analyzer

    result = await scenario_analyzer.analyze_scenario(
        ticker="NVDA",
        scenario="interest_rate_hike",
        magnitude=0.25  # 0.25% rate hike
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import structlog

logger = structlog.get_logger()


class ScenarioType(Enum):
    """Types of scenarios users can analyze."""
    INTEREST_RATE_HIKE = "interest_rate_hike"
    INTEREST_RATE_CUT = "interest_rate_cut"
    RECESSION = "recession"
    MARKET_CRASH = "market_crash"
    BULL_MARKET = "bull_market"
    EARNINGS_BEAT = "earnings_beat"
    EARNINGS_MISS = "earnings_miss"
    OIL_SPIKE = "oil_spike"
    DOLLAR_STRENGTH = "dollar_strength"
    DOLLAR_WEAKNESS = "dollar_weakness"
    INFLATION_SURGE = "inflation_surge"
    TRADE_WAR = "trade_war"
    SECTOR_ROTATION = "sector_rotation"
    VIX_SPIKE = "vix_spike"
    CUSTOM = "custom"


@dataclass
class ScenarioImpact:
    """Impact of a scenario on a stock."""
    scenario: str
    scenario_description: str
    ticker: str

    # Expected impact
    expected_move_pct: float  # e.g., -5.5 means -5.5%
    confidence: float  # How confident we are in this estimate
    timeframe_days: int  # Over what period

    # Reasoning
    primary_factors: List[str]
    risk_factors: List[str]
    historical_precedent: Optional[str]

    # Sector context
    sector_impact: str  # "high_negative", "moderate_negative", "neutral", etc.
    relative_to_market: str  # "outperform", "underperform", "inline"

    # Actionable insights
    recommendation: str
    hedging_suggestions: List[str]

    # Limitations
    caveats: List[str]


class ScenarioAnalyzer:
    """
    Analyzes hypothetical scenarios and their impact on stocks.

    Addresses key fintech bottleneck: Users want to ask "what if" questions
    but most tools only provide static point-in-time predictions.
    """

    # Historical impact data by scenario and sector
    # Based on academic research and historical market data
    SCENARIO_IMPACTS = {
        ScenarioType.INTEREST_RATE_HIKE: {
            "Technology": {"impact": -3.5, "confidence": 0.7, "timeframe": 30},
            "Financial Services": {"impact": 2.0, "confidence": 0.75, "timeframe": 30},
            "Real Estate": {"impact": -5.0, "confidence": 0.8, "timeframe": 30},
            "Utilities": {"impact": -2.5, "confidence": 0.7, "timeframe": 30},
            "Consumer Discretionary": {"impact": -2.0, "confidence": 0.65, "timeframe": 30},
            "Healthcare": {"impact": -1.0, "confidence": 0.6, "timeframe": 30},
            "Consumer Staples": {"impact": -0.5, "confidence": 0.65, "timeframe": 30},
            "Energy": {"impact": 0.5, "confidence": 0.5, "timeframe": 30},
            "default": {"impact": -2.0, "confidence": 0.5, "timeframe": 30},
        },
        ScenarioType.INTEREST_RATE_CUT: {
            "Technology": {"impact": 4.0, "confidence": 0.7, "timeframe": 30},
            "Financial Services": {"impact": -1.5, "confidence": 0.7, "timeframe": 30},
            "Real Estate": {"impact": 5.5, "confidence": 0.8, "timeframe": 30},
            "Utilities": {"impact": 3.0, "confidence": 0.7, "timeframe": 30},
            "Consumer Discretionary": {"impact": 3.0, "confidence": 0.65, "timeframe": 30},
            "Healthcare": {"impact": 1.5, "confidence": 0.6, "timeframe": 30},
            "default": {"impact": 2.5, "confidence": 0.5, "timeframe": 30},
        },
        ScenarioType.RECESSION: {
            "Technology": {"impact": -25.0, "confidence": 0.6, "timeframe": 180},
            "Financial Services": {"impact": -30.0, "confidence": 0.65, "timeframe": 180},
            "Consumer Discretionary": {"impact": -35.0, "confidence": 0.7, "timeframe": 180},
            "Consumer Staples": {"impact": -10.0, "confidence": 0.7, "timeframe": 180},
            "Healthcare": {"impact": -15.0, "confidence": 0.65, "timeframe": 180},
            "Utilities": {"impact": -8.0, "confidence": 0.7, "timeframe": 180},
            "Energy": {"impact": -30.0, "confidence": 0.6, "timeframe": 180},
            "default": {"impact": -20.0, "confidence": 0.5, "timeframe": 180},
        },
        ScenarioType.MARKET_CRASH: {
            "Technology": {"impact": -30.0, "confidence": 0.5, "timeframe": 30},
            "Financial Services": {"impact": -35.0, "confidence": 0.55, "timeframe": 30},
            "Consumer Discretionary": {"impact": -28.0, "confidence": 0.55, "timeframe": 30},
            "Consumer Staples": {"impact": -15.0, "confidence": 0.6, "timeframe": 30},
            "Healthcare": {"impact": -18.0, "confidence": 0.55, "timeframe": 30},
            "Utilities": {"impact": -12.0, "confidence": 0.6, "timeframe": 30},
            "default": {"impact": -25.0, "confidence": 0.4, "timeframe": 30},
        },
        ScenarioType.EARNINGS_BEAT: {
            "default": {"impact": 5.0, "confidence": 0.7, "timeframe": 5},
        },
        ScenarioType.EARNINGS_MISS: {
            "default": {"impact": -8.0, "confidence": 0.75, "timeframe": 5},
        },
        ScenarioType.OIL_SPIKE: {
            "Energy": {"impact": 15.0, "confidence": 0.75, "timeframe": 30},
            "Airlines": {"impact": -12.0, "confidence": 0.8, "timeframe": 30},
            "Consumer Discretionary": {"impact": -5.0, "confidence": 0.6, "timeframe": 30},
            "Industrials": {"impact": -3.0, "confidence": 0.55, "timeframe": 30},
            "default": {"impact": -2.0, "confidence": 0.4, "timeframe": 30},
        },
        ScenarioType.VIX_SPIKE: {
            "Technology": {"impact": -8.0, "confidence": 0.7, "timeframe": 14},
            "Financial Services": {"impact": -6.0, "confidence": 0.65, "timeframe": 14},
            "Consumer Staples": {"impact": -2.0, "confidence": 0.7, "timeframe": 14},
            "Utilities": {"impact": -1.5, "confidence": 0.7, "timeframe": 14},
            "default": {"impact": -5.0, "confidence": 0.6, "timeframe": 14},
        },
        ScenarioType.INFLATION_SURGE: {
            "Technology": {"impact": -8.0, "confidence": 0.65, "timeframe": 60},
            "Real Estate": {"impact": -6.0, "confidence": 0.7, "timeframe": 60},
            "Consumer Staples": {"impact": 2.0, "confidence": 0.6, "timeframe": 60},
            "Energy": {"impact": 8.0, "confidence": 0.7, "timeframe": 60},
            "Basic Materials": {"impact": 6.0, "confidence": 0.65, "timeframe": 60},
            "default": {"impact": -3.0, "confidence": 0.5, "timeframe": 60},
        },
    }

    # Historical precedents for context
    HISTORICAL_PRECEDENTS = {
        ScenarioType.INTEREST_RATE_HIKE: "2022 Fed rate hikes: Tech stocks fell 33%, banks initially rose 15% before falling",
        ScenarioType.RECESSION: "2008 Financial Crisis: S&P 500 fell 57% peak-to-trough; 2020 COVID: 34% drop in 33 days",
        ScenarioType.MARKET_CRASH: "Black Monday 1987: 22% single-day drop; COVID crash 2020: 34% in 33 days",
        ScenarioType.OIL_SPIKE: "2022 Russia-Ukraine: Oil spiked 60%, energy stocks +45%, airlines -25%",
        ScenarioType.VIX_SPIKE: "March 2020: VIX hit 82, S&P 500 fell 12% in a single week",
        ScenarioType.INFLATION_SURGE: "2022: 9.1% CPI peak, growth stocks fell 30%+, value outperformed",
    }

    def __init__(self):
        self.logger = logger.bind(service="scenario_analyzer")

    async def analyze_scenario(
        self,
        ticker: str,
        scenario: str,
        magnitude: float = 1.0,
        sector: Optional[str] = None,
        current_price: Optional[float] = None,
    ) -> ScenarioImpact:
        """
        Analyze how a hypothetical scenario would impact a stock.

        Args:
            ticker: Stock ticker symbol
            scenario: Scenario type (e.g., "interest_rate_hike", "recession")
            magnitude: Severity multiplier (1.0 = standard, 2.0 = severe)
            sector: Stock's sector (auto-detected if not provided)
            current_price: Current stock price

        Returns:
            ScenarioImpact with expected move and reasoning
        """
        self.logger.info(
            "Analyzing scenario",
            ticker=ticker,
            scenario=scenario,
            magnitude=magnitude,
        )

        # Parse scenario type
        try:
            scenario_type = ScenarioType(scenario.lower().replace(" ", "_"))
        except ValueError:
            scenario_type = ScenarioType.CUSTOM

        # Get sector if not provided
        if not sector:
            sector = await self._get_ticker_sector(ticker)

        # Get base impact for this scenario and sector
        scenario_data = self.SCENARIO_IMPACTS.get(scenario_type, {})
        sector_impact = scenario_data.get(sector, scenario_data.get("default", {}))

        if not sector_impact:
            # Custom scenario - use AI to estimate
            return await self._analyze_custom_scenario(ticker, scenario, sector)

        # Calculate adjusted impact based on magnitude
        base_impact = sector_impact.get("impact", 0)
        adjusted_impact = base_impact * magnitude

        # Get historical precedent
        precedent = self.HISTORICAL_PRECEDENTS.get(scenario_type)

        # Build response
        return ScenarioImpact(
            scenario=scenario,
            scenario_description=self._get_scenario_description(scenario_type, magnitude),
            ticker=ticker,
            expected_move_pct=round(adjusted_impact, 1),
            confidence=sector_impact.get("confidence", 0.5),
            timeframe_days=sector_impact.get("timeframe", 30),
            primary_factors=self._get_primary_factors(scenario_type, sector),
            risk_factors=self._get_risk_factors(scenario_type),
            historical_precedent=precedent,
            sector_impact=self._categorize_impact(adjusted_impact),
            relative_to_market=self._get_relative_performance(scenario_type, sector),
            recommendation=self._get_recommendation(adjusted_impact, scenario_type),
            hedging_suggestions=self._get_hedging_suggestions(scenario_type, adjusted_impact),
            caveats=self._get_caveats(scenario_type),
        )

    async def analyze_multiple_scenarios(
        self,
        ticker: str,
        scenarios: List[str],
        sector: Optional[str] = None,
    ) -> List[ScenarioImpact]:
        """Analyze multiple scenarios for comparison."""
        results = []
        for scenario in scenarios:
            result = await self.analyze_scenario(ticker, scenario, sector=sector)
            results.append(result)
        return results

    async def get_current_risk_scenarios(
        self,
        ticker: str,
        sector: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get the most relevant risk scenarios for current market conditions.

        This addresses the "Risk Radar" feature - proactively warning users
        about scenarios they should be watching.
        """
        # TODO: Integrate with real-time macro data
        # For now, return static risk scenarios

        risks = [
            {
                "scenario": "interest_rate_hike",
                "probability": 0.35,
                "description": "Fed may raise rates if inflation persists",
                "impact_on_ticker": await self.analyze_scenario(ticker, "interest_rate_hike", sector=sector),
            },
            {
                "scenario": "vix_spike",
                "probability": 0.25,
                "description": "Market volatility may increase due to geopolitical tensions",
                "impact_on_ticker": await self.analyze_scenario(ticker, "vix_spike", sector=sector),
            },
            {
                "scenario": "recession",
                "probability": 0.20,
                "description": "Economic slowdown risk from tight monetary policy",
                "impact_on_ticker": await self.analyze_scenario(ticker, "recession", sector=sector),
            },
        ]

        return risks

    async def _get_ticker_sector(self, ticker: str) -> str:
        """Get the sector for a ticker."""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get("sector", "Unknown")
        except:
            return "Unknown"

    async def _analyze_custom_scenario(
        self,
        ticker: str,
        scenario: str,
        sector: str,
    ) -> ScenarioImpact:
        """Analyze a custom user-defined scenario using AI."""
        # For custom scenarios, provide a generic response
        # In production, this would call Claude API for analysis
        return ScenarioImpact(
            scenario=scenario,
            scenario_description=f"Custom scenario: {scenario}",
            ticker=ticker,
            expected_move_pct=0.0,
            confidence=0.3,
            timeframe_days=30,
            primary_factors=["Custom scenario requires manual analysis"],
            risk_factors=["Uncertainty is high for custom scenarios"],
            historical_precedent=None,
            sector_impact="unknown",
            relative_to_market="unknown",
            recommendation="Consult with a financial advisor for custom scenario analysis",
            hedging_suggestions=["Consider diversification", "Use stop-loss orders"],
            caveats=[
                "Custom scenarios have high uncertainty",
                "Historical data may not be available",
                "This is not financial advice",
            ],
        )

    def _get_scenario_description(self, scenario_type: ScenarioType, magnitude: float) -> str:
        """Get human-readable scenario description."""
        descriptions = {
            ScenarioType.INTEREST_RATE_HIKE: f"Federal Reserve raises interest rates by {magnitude * 0.25:.2f}%",
            ScenarioType.INTEREST_RATE_CUT: f"Federal Reserve cuts interest rates by {magnitude * 0.25:.2f}%",
            ScenarioType.RECESSION: f"Economy enters {'mild' if magnitude < 1.5 else 'severe'} recession",
            ScenarioType.MARKET_CRASH: f"Market experiences {'correction' if magnitude < 1.5 else 'crash'} (>20% decline)",
            ScenarioType.BULL_MARKET: "Market enters sustained bull run",
            ScenarioType.EARNINGS_BEAT: f"Company beats earnings expectations by {magnitude * 10:.0f}%",
            ScenarioType.EARNINGS_MISS: f"Company misses earnings expectations by {magnitude * 10:.0f}%",
            ScenarioType.OIL_SPIKE: f"Oil prices spike {magnitude * 30:.0f}%",
            ScenarioType.VIX_SPIKE: f"VIX (volatility) spikes to {20 + magnitude * 20:.0f}",
            ScenarioType.INFLATION_SURGE: f"Inflation rises to {magnitude * 5:.1f}%",
            ScenarioType.TRADE_WAR: "Major trade war escalation",
        }
        return descriptions.get(scenario_type, str(scenario_type.value))

    def _get_primary_factors(self, scenario_type: ScenarioType, sector: str) -> List[str]:
        """Get primary factors driving the impact."""
        factors = {
            ScenarioType.INTEREST_RATE_HIKE: [
                "Higher discount rates reduce present value of future earnings",
                "Increased borrowing costs for companies",
                "Consumer spending may decrease",
                f"{sector} sensitivity to rate changes",
            ],
            ScenarioType.RECESSION: [
                "Reduced consumer spending",
                "Corporate earnings decline",
                "Credit conditions tighten",
                "Unemployment rises",
            ],
            ScenarioType.MARKET_CRASH: [
                "Panic selling and margin calls",
                "Liquidity crisis",
                "Flight to safety assets",
                "Correlation spike across assets",
            ],
            ScenarioType.VIX_SPIKE: [
                "Increased hedging demand",
                "Risk-off sentiment",
                "Portfolio rebalancing",
                "Options market stress",
            ],
        }
        return factors.get(scenario_type, ["Market uncertainty", "Sector-specific factors"])

    def _get_risk_factors(self, scenario_type: ScenarioType) -> List[str]:
        """Get risk factors that could make impact worse."""
        return [
            "Actual impact may differ significantly from historical patterns",
            "Multiple scenarios occurring simultaneously amplifies effects",
            "Company-specific factors may override sector trends",
            "Market sentiment can cause overshooting",
        ]

    def _categorize_impact(self, impact_pct: float) -> str:
        """Categorize the impact level."""
        if impact_pct <= -20:
            return "severe_negative"
        elif impact_pct <= -10:
            return "high_negative"
        elif impact_pct <= -5:
            return "moderate_negative"
        elif impact_pct < 0:
            return "mild_negative"
        elif impact_pct == 0:
            return "neutral"
        elif impact_pct < 5:
            return "mild_positive"
        elif impact_pct < 10:
            return "moderate_positive"
        else:
            return "high_positive"

    def _get_relative_performance(self, scenario_type: ScenarioType, sector: str) -> str:
        """Determine if stock will outperform or underperform market."""
        # Defensive sectors in negative scenarios
        defensive_sectors = ["Consumer Staples", "Utilities", "Healthcare"]
        cyclical_sectors = ["Technology", "Consumer Discretionary", "Industrials"]

        negative_scenarios = [
            ScenarioType.INTEREST_RATE_HIKE,
            ScenarioType.RECESSION,
            ScenarioType.MARKET_CRASH,
            ScenarioType.VIX_SPIKE,
        ]

        if scenario_type in negative_scenarios:
            if sector in defensive_sectors:
                return "outperform"
            elif sector in cyclical_sectors:
                return "underperform"

        return "inline"

    def _get_recommendation(self, impact_pct: float, scenario_type: ScenarioType) -> str:
        """Get actionable recommendation."""
        if impact_pct <= -15:
            return "Consider reducing position or hedging before this scenario materializes"
        elif impact_pct <= -5:
            return "Monitor closely and have exit strategy ready"
        elif impact_pct < 0:
            return "Minor negative impact expected; maintain position with awareness"
        elif impact_pct == 0:
            return "Neutral impact; no action required for this scenario"
        elif impact_pct < 10:
            return "Potential upside; consider maintaining or adding to position"
        else:
            return "Significant upside potential if scenario occurs"

    def _get_hedging_suggestions(self, scenario_type: ScenarioType, impact_pct: float) -> List[str]:
        """Get hedging suggestions based on scenario."""
        if impact_pct >= 0:
            return ["No hedging needed for positive scenario"]

        suggestions = {
            ScenarioType.INTEREST_RATE_HIKE: [
                "Consider TLT puts (Treasury bond ETF)",
                "Financial sector ETFs (XLF) may hedge tech exposure",
                "Short-duration bond funds",
            ],
            ScenarioType.RECESSION: [
                "Defensive sector ETFs (XLP, XLU)",
                "Treasury bonds (TLT)",
                "Gold (GLD) as safe haven",
                "Put options on market indices",
            ],
            ScenarioType.MARKET_CRASH: [
                "VIX calls for volatility protection",
                "Put options on SPY/QQQ",
                "Treasury bonds",
                "Cash position increase",
            ],
            ScenarioType.VIX_SPIKE: [
                "Reduce position size",
                "Collar strategy (buy puts, sell calls)",
                "Move to lower-beta stocks",
            ],
        }
        return suggestions.get(scenario_type, ["Diversification", "Stop-loss orders"])

    def _get_caveats(self, scenario_type: ScenarioType) -> List[str]:
        """Get important caveats for the analysis."""
        return [
            "This analysis is based on historical patterns and may not predict future outcomes",
            "Individual stock performance can deviate significantly from sector averages",
            "Multiple simultaneous scenarios can amplify or offset impacts",
            "This is not financial advice - consult a professional for investment decisions",
            "Past performance does not guarantee future results",
        ]


# Singleton instance
scenario_analyzer = ScenarioAnalyzer()
