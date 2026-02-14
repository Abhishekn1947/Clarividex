"use client";

import { useState } from "react";
import { Info, X } from "lucide-react";
import { cn } from "@/lib/utils";

// Technical term definitions from HOW_IT_WORKS.md
export const TERM_DEFINITIONS: Record<string, { title: string; description: string; example?: string }> = {
  // Technical Indicators
  rsi: {
    title: "RSI (Relative Strength Index)",
    description: "Measures the speed and magnitude of price changes on a scale of 0-100. It helps identify overbought (above 70) or oversold (below 30) conditions.",
    example: "RSI of 25 suggests the stock is oversold and may bounce back. RSI of 75 suggests overbought conditions and potential pullback.",
  },
  macd: {
    title: "MACD (Moving Average Convergence Divergence)",
    description: "A trend-following momentum indicator that shows the relationship between two moving averages. MACD crossing above the signal line is bullish; crossing below is bearish.",
    example: "Positive MACD histogram = bullish momentum. Negative = bearish momentum.",
  },
  sma: {
    title: "SMA (Simple Moving Average)",
    description: "The average price over a specific period. SMA20 = 20-day average, SMA50 = 50-day average. Price above moving averages suggests uptrend.",
    example: "Price above SMA50 and SMA200 indicates a strong uptrend (bullish).",
  },
  ema: {
    title: "EMA (Exponential Moving Average)",
    description: "Similar to SMA but gives more weight to recent prices, making it more responsive to new information.",
    example: "EMA12 crossing above EMA26 is a bullish signal (used in MACD calculation).",
  },
  bollinger: {
    title: "Bollinger Bands",
    description: "Bands placed 2 standard deviations above and below a moving average. Price touching the upper band may indicate overbought; lower band may indicate oversold.",
    example: "Price at lower Bollinger Band with RSI < 30 = strong oversold signal.",
  },
  support: {
    title: "Support Level",
    description: "A price level where buying pressure typically prevents further decline. Think of it as a 'floor' for the stock price.",
    example: "Stock at $95 with support at $90 has a 'safety net' $5 below.",
  },
  resistance: {
    title: "Resistance Level",
    description: "A price level where selling pressure typically prevents further rise. Think of it as a 'ceiling' for the stock price.",
    example: "Stock at $95 with resistance at $100 needs strong buying to break through.",
  },

  // Market Sentiment
  vix: {
    title: "VIX (Volatility Index)",
    description: "Known as the 'Fear Index', measures expected market volatility over the next 30 days. Higher VIX = more fear/uncertainty in the market.",
    example: "VIX < 15 = calm market. VIX 15-25 = normal. VIX > 25 = elevated fear. VIX > 30 = high fear (often a buying opportunity).",
  },
  feargreed: {
    title: "Fear & Greed Index",
    description: "Measures investor sentiment on a scale of 0 (Extreme Fear) to 100 (Extreme Greed). Extreme readings often signal market turning points.",
    example: "Index at 20 (Fear) = potential buying opportunity. Index at 80 (Greed) = potential time for caution.",
  },
  putcall: {
    title: "Put/Call Ratio",
    description: "Ratio of put options to call options traded. High ratio (>1.0) indicates bearish sentiment; low ratio (<0.7) indicates bullish sentiment.",
    example: "P/C ratio of 0.5 = more calls than puts being bought (bullish). Ratio of 1.5 = more puts (bearish/hedging).",
  },

  // Fundamental Terms
  pe: {
    title: "P/E Ratio (Price-to-Earnings)",
    description: "Stock price divided by earnings per share. Shows how much investors are willing to pay per dollar of earnings. Lower P/E may indicate value; higher P/E may indicate growth expectations.",
    example: "P/E of 15 is considered average. Tech stocks often have P/E of 30-50 due to growth expectations.",
  },
  marketcap: {
    title: "Market Cap (Market Capitalization)",
    description: "Total market value of a company (share price × total shares). Large cap > $10B, Mid cap $2B-$10B, Small cap < $2B.",
    example: "Apple has ~$3T market cap (mega cap). A $500M company is small cap.",
  },
  eps: {
    title: "EPS (Earnings Per Share)",
    description: "Company profit divided by outstanding shares. Shows how much profit the company makes per share.",
    example: "EPS of $5 with stock at $100 = P/E ratio of 20.",
  },

  // Analysis Terms
  bullish: {
    title: "Bullish",
    description: "Expecting the price to go UP. A bullish signal or factor supports higher prices.",
    example: "Bullish factors: positive earnings, RSI oversold, analyst upgrades.",
  },
  bearish: {
    title: "Bearish",
    description: "Expecting the price to go DOWN. A bearish signal or factor suggests lower prices.",
    example: "Bearish factors: negative news, RSI overbought, high short interest.",
  },
  volatility: {
    title: "Volatility",
    description: "How much a stock's price fluctuates. High volatility = bigger price swings (both up and down).",
    example: "TSLA is highly volatile (large daily moves). Utility stocks have low volatility.",
  },
  shortinterest: {
    title: "Short Interest",
    description: "Percentage of shares that have been sold short (betting on price decline). High short interest (>20%) can lead to short squeezes.",
    example: "Short interest of 25% means 1 in 4 shares are being shorted - high bearish sentiment.",
  },

  // Data Quality
  dataquality: {
    title: "Data Quality Score",
    description: "Measures the completeness and reliability of data used in the prediction. Higher scores mean more confidence in the analysis.",
    example: "100% = all data sources available. 70% = some data missing or limited.",
  },
  confidence: {
    title: "Confidence Level",
    description: "How certain the model is about its prediction based on data agreement, source count, and signal clarity.",
    example: "High confidence = multiple data sources agree. Low confidence = conflicting signals.",
  },
  probability: {
    title: "Probability",
    description: "The likelihood that the predicted outcome will occur. 50% = neutral/uncertain, >65% = likely, <35% = unlikely.",
    example: "65% probability of increase means the model sees more bullish than bearish signals.",
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
          "inline-flex items-center justify-center rounded-full text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors",
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
            className="bg-white rounded-xl shadow-xl max-w-md w-full mx-4 animate-scale-in"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between p-4 border-b border-slate-200">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center">
                  <Info className="w-4 h-4 text-slate-600" />
                </div>
                <h3 className="font-semibold text-slate-800">{definition.title}</h3>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-slate-400 hover:text-slate-600 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 space-y-3">
              <p className="text-sm text-slate-600 leading-relaxed">
                {definition.description}
              </p>
              {definition.example && (
                <div className="bg-slate-50 rounded-lg p-3">
                  <p className="text-xs text-slate-500 uppercase tracking-wide font-medium mb-1">
                    Example
                  </p>
                  <p className="text-sm text-slate-700">
                    {definition.example}
                  </p>
                </div>
              )}
            </div>
            <div className="px-4 py-3 border-t border-slate-100 bg-slate-50 rounded-b-xl">
              <button
                onClick={() => setIsOpen(false)}
                className="w-full px-4 py-2 text-sm font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors"
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
        className="inline-flex items-center justify-center w-4 h-4 rounded-full text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
      >
        <Info className="w-3 h-3" />
      </button>
      {showTooltip && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 max-w-[calc(100vw-2rem)] p-3 bg-slate-800 text-white text-xs rounded-lg shadow-lg z-50">
          <div className="font-medium mb-1">{definition.title}</div>
          <div className="text-slate-300 leading-relaxed">{definition.description}</div>
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 w-2 h-2 bg-slate-800 rotate-45" />
        </div>
      )}
    </span>
  );
}
