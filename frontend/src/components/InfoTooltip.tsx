"use client";

import { useState } from "react";
import { Info, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface Signal {
  condition: string;
  meaning: string;
  type: "bullish" | "bearish" | "neutral";
}

interface TermDefinition {
  title: string;
  description: string;
  signals?: Signal[];
  tip?: string;
}

const TERM_DEFINITIONS: Record<string, TermDefinition> = {
  // Technical Indicators
  rsi: {
    title: "RSI (Relative Strength Index)",
    description:
      "Momentum oscillator measuring speed and magnitude of recent price changes on a 0\u2013100 scale. One of the most widely used technical indicators for identifying reversal points.",
    signals: [
      { condition: "Below 30", meaning: "Oversold \u2014 stock may be undervalued, potential bounce", type: "bullish" },
      { condition: "30\u201350", meaning: "Weak momentum \u2014 watch for trend continuation", type: "neutral" },
      { condition: "50\u201370", meaning: "Healthy momentum \u2014 trend is intact", type: "neutral" },
      { condition: "Above 70", meaning: "Overbought \u2014 stock may be overvalued, potential pullback", type: "bearish" },
    ],
    tip: "RSI divergence (price making new highs while RSI makes lower highs) is a powerful reversal signal. Works best when combined with volume analysis.",
  },
  macd: {
    title: "MACD (Moving Average Convergence Divergence)",
    description:
      "Trend-following momentum indicator showing the relationship between two exponential moving averages (EMA 12 and EMA 26). The histogram shows distance between the MACD line and signal line.",
    signals: [
      { condition: "Positive & rising", meaning: "Bullish momentum increasing", type: "bullish" },
      { condition: "Positive & falling", meaning: "Bullish momentum weakening", type: "neutral" },
      { condition: "Negative & falling", meaning: "Bearish momentum increasing", type: "bearish" },
      { condition: "Negative & rising", meaning: "Bearish momentum weakening \u2014 watch for crossover", type: "neutral" },
    ],
    tip: "A MACD crossover above the signal line is a classic buy signal. The larger the histogram bar, the stronger the momentum.",
  },
  sma: {
    title: "SMA (Simple Moving Average)",
    description:
      "Average closing price over a specific period. SMA 20 = short-term trend, SMA 50 = medium-term, SMA 200 = long-term. Price position relative to these averages reveals trend direction.",
    signals: [
      { condition: "Price > SMA", meaning: "Stock is in an uptrend relative to that period", type: "bullish" },
      { condition: "Price < SMA", meaning: "Stock is in a downtrend relative to that period", type: "bearish" },
      { condition: "SMA 50 > SMA 200", meaning: "Golden Cross \u2014 long-term bullish signal", type: "bullish" },
      { condition: "SMA 50 < SMA 200", meaning: "Death Cross \u2014 long-term bearish signal", type: "bearish" },
    ],
    tip: "The SMA 200 is the most watched long-term trend indicator. Institutions often use it as a buy/sell threshold.",
  },
  ema: {
    title: "EMA (Exponential Moving Average)",
    description:
      "Like SMA but gives more weight to recent prices, making it more responsive to new price action. EMA 12 and EMA 26 are used in the MACD calculation.",
    signals: [
      { condition: "Price > EMA", meaning: "Short-term trend is bullish", type: "bullish" },
      { condition: "Price < EMA", meaning: "Short-term trend is bearish", type: "bearish" },
    ],
    tip: "EMA crossovers (EMA 12 crossing EMA 26) are the same signals used by MACD. EMA reacts faster than SMA to sudden price changes.",
  },
  bollinger: {
    title: "Bollinger Bands",
    description:
      "Bands placed 2 standard deviations above and below a 20-period SMA. They widen during volatile periods and contract during calm periods.",
    signals: [
      { condition: "Price at upper band", meaning: "May be overbought \u2014 potential pullback", type: "bearish" },
      { condition: "Price at lower band", meaning: "May be oversold \u2014 potential bounce", type: "bullish" },
      { condition: "Bands squeezing", meaning: "Low volatility \u2014 big move coming (direction TBD)", type: "neutral" },
    ],
    tip: "A Bollinger squeeze (very narrow bands) often precedes a major breakout. Combine with RSI or MACD to predict direction.",
  },
  support: {
    title: "Support Level",
    description:
      "Price level where buying interest is historically strong enough to prevent further decline. Acts as a floor \u2014 when price approaches support, buyers tend to step in.",
    signals: [
      { condition: "Price near support", meaning: "Potential bounce \u2014 good entry point if support holds", type: "bullish" },
      { condition: "Price breaks below", meaning: "Support broken \u2014 next support level becomes the target", type: "bearish" },
    ],
    tip: "Support levels strengthen each time they are tested successfully. A break below on high volume is a strong bearish signal.",
  },
  resistance: {
    title: "Resistance Level",
    description:
      "Price level where selling pressure is historically strong enough to prevent further rise. Acts as a ceiling \u2014 when price approaches resistance, sellers tend to step in.",
    signals: [
      { condition: "Price near resistance", meaning: "May struggle to break through \u2014 potential pullback", type: "bearish" },
      { condition: "Breakout above", meaning: "Bullish \u2014 old resistance becomes new support", type: "bullish" },
    ],
    tip: "Breakouts above resistance on high volume are strong buy signals. Failed breakouts often lead to sharp reversals downward.",
  },

  // Market Sentiment
  vix: {
    title: "VIX (Volatility Index)",
    description:
      "The \"Fear Index\" \u2014 measures expected S&P 500 volatility over the next 30 days. India uses India VIX (based on Nifty 50 options). Higher values = more fear and uncertainty.",
    signals: [
      { condition: "Below 15", meaning: "Low fear \u2014 calm market, favorable for stocks", type: "bullish" },
      { condition: "15\u201325", meaning: "Normal volatility \u2014 standard market conditions", type: "neutral" },
      { condition: "25\u201330", meaning: "Elevated fear \u2014 increased caution warranted", type: "bearish" },
      { condition: "Above 30", meaning: "High fear \u2014 often a contrarian buying opportunity", type: "bullish" },
    ],
    tip: "Extreme VIX spikes (above 35) historically mark market bottoms. VIX tends to mean-revert, making extremes actionable.",
  },
  feargreed: {
    title: "Fear & Greed Index",
    description:
      "CNN\u2019s composite sentiment indicator combining 7 market factors: momentum, strength, breadth, put/call ratio, junk bond demand, volatility, and safe haven demand. Scale: 0 (Extreme Fear) to 100 (Extreme Greed).",
    signals: [
      { condition: "0\u201325", meaning: "Extreme Fear \u2014 contrarian buy signal", type: "bullish" },
      { condition: "25\u201345", meaning: "Fear \u2014 market is cautious", type: "neutral" },
      { condition: "45\u201355", meaning: "Neutral \u2014 balanced sentiment", type: "neutral" },
      { condition: "55\u201375", meaning: "Greed \u2014 market is optimistic", type: "neutral" },
      { condition: "75\u2013100", meaning: "Extreme Greed \u2014 contrarian sell signal", type: "bearish" },
    ],
    tip: "\"Be fearful when others are greedy, and greedy when others are fearful.\" \u2014 Warren Buffett. Extreme readings often precede reversals.",
  },
  putcall: {
    title: "Put/Call Ratio",
    description:
      "Ratio of put option volume to call option volume. Puts = bearish bets, Calls = bullish bets. Reveals what options traders are positioning for.",
    signals: [
      { condition: "Below 0.7", meaning: "Bullish sentiment \u2014 more calls being bought", type: "bullish" },
      { condition: "0.7\u20131.0", meaning: "Balanced \u2014 no clear directional bias", type: "neutral" },
      { condition: "1.0\u20131.3", meaning: "Bearish sentiment \u2014 more puts being bought", type: "bearish" },
      { condition: "Above 1.3", meaning: "Extreme bearish \u2014 possible contrarian buy signal", type: "bullish" },
    ],
    tip: "Smart money often uses puts for hedging rather than directional bets. Extremely high put/call ratios can actually signal a market bottom.",
  },

  // Fundamental Terms
  pe: {
    title: "P/E Ratio (Price-to-Earnings)",
    description:
      "Stock price divided by earnings per share. Shows how much investors pay per unit of profit. Used to assess relative valuation within a sector.",
    signals: [
      { condition: "Below 15", meaning: "May be undervalued \u2014 or low growth expected", type: "neutral" },
      { condition: "15\u201325", meaning: "Average valuation \u2014 typical for established companies", type: "neutral" },
      { condition: "Above 25", meaning: "Premium valuation \u2014 high growth expected", type: "neutral" },
    ],
    tip: "Always compare P/E within the same sector. A tech stock at P/E 30 may be cheap vs. peers, while a utility at P/E 30 is expensive.",
  },
  marketcap: {
    title: "Market Cap",
    description:
      "Total market value of a company\u2019s outstanding shares (Share Price \u00d7 Total Shares). Larger companies tend to be more stable and have more liquid markets.",
    signals: [
      { condition: "Mega cap (>$200B)", meaning: "Most stable, highest data quality", type: "neutral" },
      { condition: "Large cap ($10B\u2013$200B)", meaning: "Established, liquid", type: "neutral" },
      { condition: "Mid cap ($2B\u2013$10B)", meaning: "Growth potential, moderate risk", type: "neutral" },
      { condition: "Small cap (<$2B)", meaning: "Higher risk, higher potential reward", type: "neutral" },
    ],
    tip: "Market cap affects prediction reliability. Large-cap stocks have more data sources, more analyst coverage, and more predictable behavior.",
  },
  eps: {
    title: "EPS (Earnings Per Share)",
    description:
      "Company\u2019s net profit divided by outstanding shares. Shows how much money the company earns for each share you own.",
    signals: [
      { condition: "Rising EPS", meaning: "Company is growing profitably", type: "bullish" },
      { condition: "Falling EPS", meaning: "Declining profitability", type: "bearish" },
    ],
    tip: "Earnings beats (actual > expected) often drive stock price up. Misses can trigger sharp selloffs, especially for growth stocks.",
  },

  // Analysis Terms
  bullish: {
    title: "Bullish",
    description:
      "Expecting prices to rise. The term comes from how a bull attacks \u2014 thrusting its horns upward. Used globally across all stock markets (NYSE, NSE, BSE, LSE, and more).",
    signals: [
      { condition: "Bullish signal", meaning: "Data supports price increase", type: "bullish" },
      { condition: "Very Bullish", meaning: "Multiple strong signals agree on upward movement", type: "bullish" },
    ],
    tip: "BSE (Bombay Stock Exchange) uses a bull as its logo. Bullish/bearish terminology is universal across Indian and US markets.",
  },
  bearish: {
    title: "Bearish",
    description:
      "Expecting prices to fall. The term comes from how a bear attacks \u2014 swiping its paw downward. Bear markets (20%+ decline) create long-term buying opportunities.",
    signals: [
      { condition: "Bearish signal", meaning: "Data supports price decrease", type: "bearish" },
      { condition: "Very Bearish", meaning: "Multiple strong signals agree on downward movement", type: "bearish" },
    ],
    tip: "Markets have recovered from every bear market in history. Bear phases are typically shorter than bull phases.",
  },
  volatility: {
    title: "Volatility",
    description:
      "How much a stock\u2019s price fluctuates over time. Measured by standard deviation of returns. High volatility = bigger price swings in both directions.",
    signals: [
      { condition: "High volatility", meaning: "Larger price swings \u2014 higher risk and reward", type: "neutral" },
      { condition: "Low volatility", meaning: "Smaller price swings \u2014 more predictable", type: "neutral" },
    ],
    tip: "Volatility is not the same as risk. It creates opportunities for active traders but increases uncertainty for fixed price targets.",
  },
  shortinterest: {
    title: "Short Interest",
    description:
      "Percentage of a company\u2019s shares sold short (borrowed and sold, betting price will drop). High short interest means many traders are betting against the stock.",
    signals: [
      { condition: "Below 5%", meaning: "Low \u2014 minimal bearish bets", type: "neutral" },
      { condition: "5\u201310%", meaning: "Moderate \u2014 some bearish sentiment", type: "neutral" },
      { condition: "10\u201320%", meaning: "Elevated \u2014 significant bearish sentiment", type: "bearish" },
      { condition: "Above 20%", meaning: "Very high \u2014 short squeeze candidate", type: "bullish" },
    ],
    tip: "Short squeezes happen when heavily shorted stocks spike, forcing short sellers to buy back and pushing price even higher.",
  },

  // Data Quality
  dataquality: {
    title: "Data Quality Score",
    description:
      "Measures completeness and reliability of all data used in the prediction. Factors: number of sources available, data freshness, and cross-source consistency.",
    signals: [
      { condition: "Above 80%", meaning: "Excellent \u2014 comprehensive data from most sources", type: "bullish" },
      { condition: "60\u201380%", meaning: "Good \u2014 some sources unavailable, still reliable", type: "neutral" },
      { condition: "Below 60%", meaning: "Limited \u2014 interpret prediction with caution", type: "bearish" },
    ],
    tip: "Data quality is lower outside market hours (delayed quotes) and for thinly-traded stocks. Large-cap stocks always have the best data quality.",
  },
  confidence: {
    title: "Confidence Level",
    description:
      "How certain the model is about its prediction. Based on: probability engine output, data quality score, signal agreement across sources, and total data points analyzed.",
    signals: [
      { condition: "High", meaning: "Multiple data sources strongly agree \u2014 reliable", type: "bullish" },
      { condition: "Medium", meaning: "Some signal agreement \u2014 reasonable prediction", type: "neutral" },
      { condition: "Low", meaning: "Conflicting signals or limited data \u2014 less reliable", type: "bearish" },
    ],
    tip: "High confidence means the data is consistent, not that the prediction is guaranteed. Always consider probability alongside confidence.",
  },
  probability: {
    title: "Probability Score",
    description:
      "Likelihood (15\u201385%) that the predicted outcome occurs. Calculated by an 8-model ensemble: Monte Carlo simulation, Bayesian inference, technical analysis, and more.",
    signals: [
      { condition: "65\u201385%", meaning: "Likely \u2014 strong signals support this outcome", type: "bullish" },
      { condition: "45\u201365%", meaning: "Uncertain \u2014 signals are mixed", type: "neutral" },
      { condition: "15\u201345%", meaning: "Unlikely \u2014 most signals oppose this outcome", type: "bearish" },
    ],
    tip: "Probabilities are capped at 15\u201385% because financial markets are inherently uncertain. No model can predict markets with >85% confidence.",
  },
};

interface InfoTooltipProps {
  term: keyof typeof TERM_DEFINITIONS;
  className?: string;
  size?: "sm" | "md";
}

export function InfoTooltip({ term, className, size = "sm" }: InfoTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);
  const definition = TERM_DEFINITIONS[term];

  if (!definition) return null;

  const iconSize = size === "sm" ? "w-3.5 h-3.5" : "w-4 h-4";

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className={cn(
          "inline-flex items-center justify-center rounded-full text-slate-400 hover:text-amber-600 hover:bg-amber-50 transition-colors",
          size === "sm" ? "w-4 h-4" : "w-5 h-5",
          className
        )}
        title={`What is ${definition.title}?`}
      >
        <Info className={iconSize} />
      </button>

      {/* Modal */}
      {isOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
          onClick={() => setIsOpen(false)}
        >
          <div
            className="bg-white rounded-2xl shadow-2xl max-w-sm w-full mx-4 overflow-hidden animate-scale-in"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-slate-800 to-slate-900 px-5 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center">
                    <Info className="w-4 h-4 text-amber-400" />
                  </div>
                  <h3 className="font-semibold text-white text-sm leading-tight">
                    {definition.title}
                  </h3>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="text-slate-400 hover:text-white transition-colors ml-2 shrink-0"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="p-5 space-y-4 max-h-[60vh] overflow-y-auto">
              <p className="text-sm text-slate-600 leading-relaxed">
                {definition.description}
              </p>

              {/* Signal Interpretation */}
              {definition.signals && definition.signals.length > 0 && (
                <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2.5">
                    Signal Interpretation
                  </p>
                  <div className="space-y-2">
                    {definition.signals.map((signal, i) => (
                      <div
                        key={i}
                        className={cn(
                          "flex items-start gap-2.5 p-2.5 rounded-xl text-sm border",
                          signal.type === "bullish"
                            ? "bg-emerald-50/50 border-emerald-100"
                            : signal.type === "bearish"
                            ? "bg-red-50/50 border-red-100"
                            : "bg-slate-50/50 border-slate-100"
                        )}
                      >
                        <div
                          className={cn(
                            "w-2 h-2 rounded-full mt-1.5 shrink-0",
                            signal.type === "bullish"
                              ? "bg-emerald-500"
                              : signal.type === "bearish"
                              ? "bg-red-500"
                              : "bg-slate-400"
                          )}
                        />
                        <div className="flex-1 min-w-0">
                          <span className="font-semibold text-slate-700 text-xs">
                            {signal.condition}
                          </span>
                          <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
                            {signal.meaning}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Pro Tip */}
              {definition.tip && (
                <div className="bg-amber-50 rounded-xl p-3.5 border border-amber-100">
                  <p className="text-[10px] font-bold text-amber-700 uppercase tracking-widest mb-1">
                    Pro Tip
                  </p>
                  <p className="text-xs text-amber-600 leading-relaxed">
                    {definition.tip}
                  </p>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="px-5 py-3 border-t border-slate-100">
              <button
                onClick={() => setIsOpen(false)}
                className="w-full px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-slate-800 to-slate-900 rounded-xl hover:from-slate-700 hover:to-slate-800 transition-all"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// Inline tooltip (smaller, for use in text)
interface InlineTooltipProps {
  term: keyof typeof TERM_DEFINITIONS;
  children: React.ReactNode;
}

export function InlineTooltip({ term, children }: InlineTooltipProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const definition = TERM_DEFINITIONS[term];

  if (!definition) return <>{children}</>;

  return (
    <span className="relative inline-flex items-center gap-1">
      {children}
      <button
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onClick={() => setShowTooltip(!showTooltip)}
        className="inline-flex items-center justify-center w-4 h-4 rounded-full text-slate-400 hover:text-amber-600 hover:bg-amber-50 transition-colors"
      >
        <Info className="w-3 h-3" />
      </button>
      {showTooltip && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 max-w-[calc(100vw-2rem)] p-3 bg-slate-800 text-white text-xs rounded-lg shadow-lg z-50">
          <div className="font-medium mb-1">{definition.title}</div>
          <div className="text-slate-300 leading-relaxed">
            {definition.description}
          </div>
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 w-2 h-2 bg-slate-800 rotate-45" />
        </div>
      )}
    </span>
  );
}
