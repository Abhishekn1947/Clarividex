"use client";

import { useState, useEffect, useRef } from "react";
import {
  Brain,
  Zap,
  Shield,
  BarChart3,
  TrendingUp,
  Database,
  LineChart,
  PieChart,
  Activity,
  ArrowRight,
  Search,
  Cpu,
  FileText,
  CheckCircle,
  XCircle,
  Clock,
  Target,
  Layers,
  RefreshCw,
} from "lucide-react";
import { PredictionForm } from "@/components/PredictionForm";
import { PredictionResult } from "@/components/PredictionResult";
import { LoadingSkeleton } from "@/components/LoadingSkeleton";
import { TickerConfirmation } from "@/components/TickerConfirmation";
import { api, PredictionResponse, HealthStatus, TickerExtractionResult, SSEEvent } from "@/lib/api";
import { cn } from "@/lib/utils";

const FEATURES = [
  {
    icon: Database,
    title: "12+ Data Sources",
    description: "Real-time aggregation from Yahoo Finance, SEC EDGAR, Finviz, News APIs, and more.",
  },
  {
    icon: Brain,
    title: "AI Analysis",
    description: "Advanced reasoning engine analyzes 250+ data points per prediction.",
  },
  {
    icon: LineChart,
    title: "Technical Indicators",
    description: "RSI, MACD, Bollinger Bands, moving averages, support/resistance.",
  },
  {
    icon: Activity,
    title: "Market Sentiment",
    description: "VIX, Fear & Greed Index, options flow, and social sentiment.",
  },
  {
    icon: PieChart,
    title: "Transparent Reasoning",
    description: "Every prediction shows factors with weighted evidence.",
  },
  {
    icon: Zap,
    title: "Real-Time Data",
    description: "Live stock prices, breaking news, up-to-the-minute data.",
  },
];

const DATA_SOURCES = [
  { name: "Yahoo Finance", reliability: 90 },
  { name: "SEC EDGAR", reliability: 95 },
  { name: "Finviz", reliability: 85 },
  { name: "Google News", reliability: 80 },
  { name: "Reddit", reliability: 70 },
  { name: "VIX Index", reliability: 90 },
  { name: "Fear & Greed", reliability: 80 },
  { name: "Options Data", reliability: 85 },
  { name: "Treasury Yields", reliability: 95 },
  { name: "Sector ETFs", reliability: 90 },
  { name: "Technical Analysis", reliability: 85 },
  { name: "Clarividex AI", reliability: 75 },
];

const HOW_IT_WORKS = [
  {
    step: 1,
    icon: Search,
    title: "Enter Your Question",
    description: "Ask about any US stock prediction. Include ticker, target price, or timeframe for best results.",
    example: '"Will NVDA reach $180 by March 2026?"',
  },
  {
    step: 2,
    icon: Database,
    title: "Data Aggregation",
    description: "We fetch real-time data from 12+ sources including Yahoo Finance, SEC filings, news, and social media.",
    details: ["Stock prices & volume", "News sentiment", "Technical indicators", "Options flow"],
  },
  {
    step: 3,
    icon: Cpu,
    title: "AI Analysis",
    description: "Our AI engine processes 250+ data points using weighted scoring across multiple factors.",
    details: ["Technical: 25%", "News: 20%", "Options: 15%", "Market: 15%", "Analyst: 15%", "Social: 10%"],
  },
  {
    step: 4,
    icon: FileText,
    title: "Prediction Generation",
    description: "Receive a probability-based prediction with full transparency on bullish/bearish factors.",
    details: ["Probability score", "Confidence level", "Key factors", "Risk assessment"],
  },
];

export default function Home() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<"online" | "offline" | "checking">("checking");
  const [tickerConfirmation, setTickerConfirmation] = useState<{
    result: TickerExtractionResult;
    query: string;
  } | null>(null);
  const [sseEvents, setSseEvents] = useState<SSEEvent[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await api.getHealth();
        setApiStatus(health.status === "healthy" ? "online" : "offline");
      } catch {
        setApiStatus("offline");
      }
    };
    checkHealth();
  }, []);

  const handleSubmit = async (query: string) => {
    setError(null);
    setPrediction(null);
    setTickerConfirmation(null);

    try {
      // First, validate the ticker extraction
      const validation = await api.validateTicker(query);

      // If needs confirmation and confidence is low, show confirmation dialog
      if (validation.needs_confirmation && validation.confidence < 0.8) {
        setTickerConfirmation({ result: validation, query });
        return;
      }

      // Otherwise, proceed with prediction
      await executePrediction(query, validation.ticker || undefined);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to validate ticker. Please try again."
      );
    }
  };

  const executePrediction = async (query: string, ticker?: string) => {
    setIsLoading(true);
    setError(null);
    setTickerConfirmation(null);
    setSseEvents([]);

    // Create abort controller for cleanup
    const abortController = new AbortController();
    abortRef.current = abortController;

    try {
      // Use SSE streaming for real-time progress
      const streamResult = await api.streamPrediction(
        {
          query,
          ticker,
          include_technicals: true,
          include_sentiment: true,
          include_news: true,
        },
        (event: SSEEvent) => {
          setSseEvents((prev) => [...prev, event]);

          // Check for errors in the SSE stream
          if (event.event === "error") {
            setError((event.data as { message?: string }).message || "Prediction failed");
          }
        },
        abortController.signal,
      );

      // If streaming returned a prediction, fetch the full one via the standard endpoint
      // since the SSE done event only has a summary
      if (streamResult) {
        // Use the standard endpoint to get the full PredictionResponse
        const fullResult = await api.createPrediction({
          query,
          ticker,
          include_technicals: true,
          include_sentiment: true,
          include_news: true,
        });
        setPrediction(fullResult);
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setError(
        err instanceof Error
          ? err.message
          : "Failed to generate prediction. Please try again."
      );
    } finally {
      setIsLoading(false);
      abortRef.current = null;
    }
  };

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const handleTickerConfirm = (ticker: string, query: string) => {
    executePrediction(query, ticker);
  };

  const handleTickerCancel = () => {
    setTickerConfirmation(null);
  };

  const handleNewPrediction = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <main className="min-h-screen bg-slate-50">
      {/* Ticker Confirmation Dialog */}
      {tickerConfirmation && (
        <TickerConfirmation
          result={tickerConfirmation.result}
          originalQuery={tickerConfirmation.query}
          onConfirm={handleTickerConfirm}
          onCancel={handleTickerCancel}
        />
      )}

      {/* Header */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-50">
        <div className="container-app py-2">
          <div className="flex items-center justify-between">
            <a
              href="/"
              onClick={(e) => {
                e.preventDefault();
                window.scrollTo({ top: 0, behavior: 'smooth' });
                setPrediction(null);
                setError(null);
              }}
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            >
              <img
                src="/clarividex-logo.png"
                alt="Clarividex"
                className="w-32 h-32 sm:w-40 sm:h-40 object-contain"
              />
              <div>
                <h1 className="text-xl sm:text-2xl font-semibold text-slate-800 tracking-tight">
                  Clarividex
                </h1>
                <div className="flex items-center gap-1.5 text-xs text-slate-500">
                  <span className={cn(
                    "status-dot",
                    apiStatus === "online" ? "online" : apiStatus === "offline" ? "offline" : "bg-amber-400"
                  )} />
                  <span>
                    {apiStatus === "online" ? "Live" : apiStatus === "offline" ? "Offline" : "Checking..."}
                  </span>
                </div>
              </div>
            </a>

            <nav className="flex items-center gap-4">
              <a
                href="#how-it-works"
                className="text-sm font-medium text-slate-600 hover:text-slate-800 transition-colors hidden sm:block"
              >
                How It Works
              </a>
              <a
                href="#features"
                className="text-sm font-medium text-slate-600 hover:text-slate-800 transition-colors hidden sm:block"
              >
                Features
              </a>
              <a
                href="#sources"
                className="text-sm font-medium text-slate-600 hover:text-slate-800 transition-colors hidden sm:block"
              >
                Data Sources
              </a>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-12 lg:py-16 px-4 sm:px-6 lg:px-8 bg-white border-b border-slate-200">
        <div className="max-w-4xl mx-auto text-center mb-10">
          <h2 className="heading-1 mb-2">
            Stock Predictions with AI Analysis
          </h2>
          <p className="text-amber-600 font-medium mb-4">
            The Clairvoyant Index
          </p>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
            Get probability-based forecasts backed by real-time data from 12+ sources.
            Transparent reasoning and quantified predictions.
          </p>

          {/* Stats - muted design */}
          <div className="flex flex-wrap justify-center gap-8 mt-8">
            {[
              { value: "12+", label: "Data Sources" },
              { value: "250+", label: "Data Points" },
              { value: "94%", label: "Data Quality" },
              { value: "<15s", label: "Analysis Time" },
            ].map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-2xl font-semibold text-slate-700">{stat.value}</div>
                <div className="text-xs text-slate-500 mt-0.5 uppercase tracking-wide">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Search Form */}
        <PredictionForm onSubmit={handleSubmit} isLoading={isLoading} />

        {/* Error Display */}
        {error && (
          <div className="max-w-2xl mx-auto mt-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
              <div className="w-6 h-6 bg-red-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-3 h-3 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </div>
              <div>
                <p className="font-medium text-red-800 text-sm">Error</p>
                <p className="text-red-700 text-sm mt-0.5">{error}</p>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Results Section */}
      {(prediction || isLoading) && (
        <section className="py-10 px-4 sm:px-6 lg:px-8">
          <div className="container-app">
            {prediction && (
              <button
                onClick={handleNewPrediction}
                className="mb-6 flex items-center gap-2 text-slate-600 hover:text-slate-800 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                <span className="text-sm font-medium">New Prediction</span>
              </button>
            )}

            {isLoading ? (
              <LoadingSkeleton sseEvents={sseEvents} />
            ) : prediction ? (
              <div className="animate-fade-in">
                <PredictionResult prediction={prediction} />
              </div>
            ) : null}
          </div>
        </section>
      )}

      {/* How It Works Section */}
      {!prediction && !isLoading && (
        <>
          <section id="how-it-works" className="section px-4 sm:px-6 lg:px-8 bg-slate-50">
            <div className="container-app">
              <div className="text-center mb-12">
                <h3 className="heading-2 mb-3">How It Works</h3>
                <p className="body-text max-w-xl mx-auto">
                  Our prediction engine combines real-time data aggregation with AI analysis
                  to generate probability-based forecasts.
                </p>
              </div>

              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl mx-auto">
                {HOW_IT_WORKS.map((item, index) => (
                  <div key={index} className="step-card">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="step-number">{item.step}</div>
                      <item.icon className="w-5 h-5 text-slate-400" />
                    </div>
                    <h4 className="heading-3 mb-2">{item.title}</h4>
                    <p className="text-sm text-slate-600 mb-3">{item.description}</p>

                    {item.example && (
                      <div className="text-xs text-slate-500 bg-slate-50 px-3 py-2 rounded-md font-mono">
                        {item.example}
                      </div>
                    )}

                    {item.details && (
                      <ul className="space-y-1 mt-2">
                        {item.details.map((detail, i) => (
                          <li key={i} className="text-xs text-slate-500 flex items-center gap-1.5">
                            <CheckCircle className="w-3 h-3 text-slate-400" />
                            {detail}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>

              {/* Methodology Details */}
              <div className="mt-12 max-w-3xl mx-auto">
                <div className="bg-white rounded-xl border border-slate-200 p-6">
                  <h4 className="heading-3 mb-4 flex items-center gap-2">
                    <Target className="w-5 h-5 text-slate-400" />
                    Prediction Methodology
                  </h4>
                  <div className="grid sm:grid-cols-2 gap-4 text-sm">
                    <div className="space-y-3">
                      <div>
                        <div className="font-medium text-slate-700 mb-1">Weighted Scoring Model</div>
                        <p className="text-slate-500">Each data source contributes to the final probability based on its reliability and relevance.</p>
                      </div>
                      <div>
                        <div className="font-medium text-slate-700 mb-1">Multi-Source Validation</div>
                        <p className="text-slate-500">Cross-references data points across sources to identify consensus and divergence.</p>
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <div className="font-medium text-slate-700 mb-1">Timeframe Adjustment</div>
                        <p className="text-slate-500">Probabilities are calibrated based on the prediction timeframe (day, week, month).</p>
                      </div>
                      <div>
                        <div className="font-medium text-slate-700 mb-1">Confidence Calculation</div>
                        <p className="text-slate-500">Confidence reflects data agreement, source count, and signal clarity.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Features Section */}
          <section id="features" className="section px-4 sm:px-6 lg:px-8 bg-white">
            <div className="container-app">
              <div className="text-center mb-12">
                <h3 className="heading-2 mb-3">Features</h3>
                <p className="body-text max-w-xl mx-auto">
                  Unlike generic chatbots, we aggregate real market data and provide
                  transparent, quantified predictions.
                </p>
              </div>

              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-4xl mx-auto">
                {FEATURES.map((feature, index) => (
                  <div
                    key={index}
                    className="p-5 rounded-xl border border-slate-200 bg-white hover:border-slate-300 transition-colors"
                  >
                    <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center mb-3">
                      <feature.icon className="w-5 h-5 text-slate-600" />
                    </div>
                    <h4 className="font-semibold text-slate-800 mb-1">{feature.title}</h4>
                    <p className="text-sm text-slate-500">{feature.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Data Sources Section */}
          <section id="sources" className="section px-4 sm:px-6 lg:px-8 bg-slate-50">
            <div className="container-app">
              <div className="text-center mb-12">
                <h3 className="heading-2 mb-3">Data Sources</h3>
                <p className="body-text max-w-xl mx-auto">
                  We aggregate data from 12+ trusted sources with reliability scoring.
                </p>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-w-3xl mx-auto">
                {DATA_SOURCES.map((source, index) => (
                  <div
                    key={index}
                    className={cn(
                      "source-badge justify-center py-2",
                      source.reliability >= 90 && "high-reliability",
                      source.reliability >= 80 && source.reliability < 90 && "medium-reliability"
                    )}
                  >
                    <span>{source.name}</span>
                    <span className="text-xs opacity-60">({source.reliability}%)</span>
                  </div>
                ))}
              </div>

              {/* Why Us vs AI Agents */}
              <div className="mt-12 max-w-4xl mx-auto">
                <div className="text-center mb-8">
                  <h4 className="heading-2 mb-3">Why Use Us Instead of AI Agents?</h4>
                  <p className="body-text max-w-2xl mx-auto">
                    Generic AI agents can search the web, but they can&apos;t give you what we provide.
                    See the difference:
                  </p>
                </div>

                {/* Response Comparison */}
                <div className="grid md:grid-cols-2 gap-6 mb-8">
                  {/* AI Agent Response */}
                  <div className="bg-slate-100 rounded-xl p-5 border border-slate-200">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-8 h-8 rounded-lg bg-slate-300 flex items-center justify-center">
                        <svg className="w-4 h-4 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                      </div>
                      <span className="font-medium text-slate-600">Generic AI Agent</span>
                    </div>
                    <div className="bg-white rounded-lg p-4 text-sm text-slate-600 italic">
                      &quot;Tesla stock could go up or down depending on various factors like market conditions,
                      earnings reports, and investor sentiment. I cannot provide specific financial advice
                      or predictions. Please consult a financial advisor.&quot;
                    </div>
                    <div className="mt-4 space-y-2">
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <XCircle className="w-4 h-4" />
                        No specific probability
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <XCircle className="w-4 h-4" />
                        No real-time data
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <XCircle className="w-4 h-4" />
                        No technical analysis
                      </div>
                    </div>
                  </div>

                  {/* Our Response */}
                  <div className="bg-amber-50 rounded-xl p-5 border border-amber-200">
                    <div className="flex items-center gap-2 mb-4">
                      <img src="/clarividex-logo.png" alt="" className="w-20 h-20 object-contain" />
                      <span className="font-semibold text-amber-700 text-xl">Clarividex</span>
                    </div>
                    <div className="bg-white rounded-lg p-4 text-sm text-slate-700">
                      <div className="text-2xl font-bold text-emerald-600 mb-2">62% Probability</div>
                      <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                          <span className="text-slate-500">RSI (14):</span>
                          <span className="font-medium text-emerald-600">34.2 (Oversold)</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">MACD:</span>
                          <span className="font-medium text-emerald-600">+2.4 (Bullish)</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">News Sentiment:</span>
                          <span className="font-medium text-emerald-600">+0.32</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Put/Call Ratio:</span>
                          <span className="font-medium text-emerald-600">0.65</span>
                        </div>
                      </div>
                      <div className="mt-2 pt-2 border-t border-slate-100 text-xs text-slate-500">
                        Based on 287 data points from 13 sources
                      </div>
                    </div>
                    <div className="mt-4 space-y-2">
                      <div className="flex items-center gap-2 text-xs text-amber-600">
                        <CheckCircle className="w-4 h-4" />
                        Quantified probability score
                      </div>
                      <div className="flex items-center gap-2 text-xs text-amber-600">
                        <CheckCircle className="w-4 h-4" />
                        Live technical indicators
                      </div>
                      <div className="flex items-center gap-2 text-xs text-amber-600">
                        <CheckCircle className="w-4 h-4" />
                        Full transparency
                      </div>
                    </div>
                  </div>
                </div>

                {/* Feature Comparison Table */}
                <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
                  <div className="p-5 bg-slate-800">
                    <h4 className="text-lg font-semibold text-white">
                      Feature Comparison
                    </h4>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-slate-50 border-b border-slate-200">
                        <tr>
                          <th className="text-left px-5 py-3 font-medium text-slate-600">Feature</th>
                          <th className="text-center px-5 py-3 font-medium text-slate-400">AI Agents</th>
                          <th className="text-center px-5 py-3 font-medium text-amber-600">Clarividex</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        <tr>
                          <td className="px-5 py-3 text-slate-700">Probability Score</td>
                          <td className="px-5 py-3 text-center text-slate-400">&quot;Could go up or down&quot;</td>
                          <td className="px-5 py-3 text-center font-medium text-amber-600">62% with confidence level</td>
                        </tr>
                        <tr>
                          <td className="px-5 py-3 text-slate-700">Real-time RSI, MACD</td>
                          <td className="px-5 py-3 text-center"><XCircle className="w-4 h-4 text-slate-300 mx-auto" /></td>
                          <td className="px-5 py-3 text-center"><CheckCircle className="w-4 h-4 text-amber-500 mx-auto" /></td>
                        </tr>
                        <tr>
                          <td className="px-5 py-3 text-slate-700">Live Options Flow (Put/Call)</td>
                          <td className="px-5 py-3 text-center"><XCircle className="w-4 h-4 text-slate-300 mx-auto" /></td>
                          <td className="px-5 py-3 text-center"><CheckCircle className="w-4 h-4 text-amber-500 mx-auto" /></td>
                        </tr>
                        <tr>
                          <td className="px-5 py-3 text-slate-700">VIX & Fear/Greed Index</td>
                          <td className="px-5 py-3 text-center"><XCircle className="w-4 h-4 text-slate-300 mx-auto" /></td>
                          <td className="px-5 py-3 text-center"><CheckCircle className="w-4 h-4 text-amber-500 mx-auto" /></td>
                        </tr>
                        <tr>
                          <td className="px-5 py-3 text-slate-700">12+ Data Sources</td>
                          <td className="px-5 py-3 text-center text-slate-400">Random web results</td>
                          <td className="px-5 py-3 text-center font-medium text-amber-600">SEC, Yahoo, Finviz, etc.</td>
                        </tr>
                        <tr>
                          <td className="px-5 py-3 text-slate-700">Consistent Methodology</td>
                          <td className="px-5 py-3 text-center text-slate-400">Varies each time</td>
                          <td className="px-5 py-3 text-center font-medium text-amber-600">Same weighted scoring</td>
                        </tr>
                        <tr>
                          <td className="px-5 py-3 text-slate-700">Transparent Reasoning</td>
                          <td className="px-5 py-3 text-center"><XCircle className="w-4 h-4 text-slate-300 mx-auto" /></td>
                          <td className="px-5 py-3 text-center"><CheckCircle className="w-4 h-4 text-amber-500 mx-auto" /></td>
                        </tr>
                        <tr>
                          <td className="px-5 py-3 text-slate-700">Impact News Analysis</td>
                          <td className="px-5 py-3 text-center text-slate-400">General search</td>
                          <td className="px-5 py-3 text-center font-medium text-amber-600">Crashes, lawsuits, recalls</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* CTA Section */}
          <section className="py-12 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-slate-900 via-slate-800 to-amber-900">
            <div className="container-app text-center">
              <div className="max-w-xl mx-auto">
                <h3 className="text-2xl font-semibold text-white mb-3">
                  Ready to See the Future?
                </h3>
                <p className="text-slate-300 mb-6 text-sm">
                  Ask Clarividex about any US stock. No signup required.
                </p>
                <button
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-amber-500 text-white font-medium rounded-lg hover:bg-amber-600 transition-colors text-sm"
                >
                  Start Predicting
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </section>
        </>
      )}

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-6 px-4 sm:px-6 lg:px-8">
        <div className="container-app">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <img
                src="/clarividex-logo.png"
                alt=""
                className="w-28 h-28 object-contain"
              />
              <span className="font-semibold text-slate-700 text-xl">Clarividex</span>
            </div>

            <div className="flex items-center gap-1.5 text-slate-500 text-xs">
              <Shield className="w-3.5 h-3.5" />
              <span>Not financial advice. For informational purposes only.</span>
            </div>

            <div className="text-xs text-slate-400">
              &copy; {new Date().getFullYear()} Clarividex
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}
