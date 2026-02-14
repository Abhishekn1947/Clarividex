"use client";

import { useState, useEffect, useRef } from "react";
import {
  Brain,
  Zap,
  Shield,
  TrendingUp,
  Database,
  ArrowRight,
  Search,
  Cpu,
  FileText,
  CheckCircle,
  XCircle,
  Target,
  Layers,
  RefreshCw,
  MessageSquare,
} from "lucide-react";
import { PredictionForm } from "@/components/PredictionForm";
import { PredictionResult } from "@/components/PredictionResult";
import { LoadingSkeleton } from "@/components/LoadingSkeleton";
import { TickerConfirmation } from "@/components/TickerConfirmation";
import { QueryGuidance } from "@/components/QueryGuidance";
import { api, PredictionResponse, TickerExtractionResult, QueryAnalysisResult, SSEEvent } from "@/lib/api";
import { cn } from "@/lib/utils";

/* ============================================
   HOOKS
   ============================================ */

function useScrollReveal() {
  useEffect(() => {
    const observed = new Set<Element>();

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
    );

    // Observe all current + future .section-reveal elements
    const scan = () => {
      document.querySelectorAll(".section-reveal").forEach((el) => {
        if (!observed.has(el)) {
          observed.add(el);
          io.observe(el);
        }
      });
    };

    scan();

    // Watch for new elements added to DOM (conditional renders)
    const mo = new MutationObserver(scan);
    mo.observe(document.body, { childList: true, subtree: true });

    return () => {
      io.disconnect();
      mo.disconnect();
    };
  }, []);
}

function useAnimatedCounter(target: number, duration: number = 1500) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const hasAnimated = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !hasAnimated.current) {
          hasAnimated.current = true;
          const start = performance.now();
          const animate = (now: number) => {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            setCount(Math.round(eased * target));
            if (progress < 1) requestAnimationFrame(animate);
          };
          requestAnimationFrame(animate);
        }
      },
      { threshold: 0.3 }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [target, duration]);

  return { count, ref };
}

/* ============================================
   DATA
   ============================================ */

const FEATURES: {
  icon: typeof Database;
  title: string;
  description: string;
  badge?: string;
}[] = [
  {
    icon: Database,
    title: "12+ Data Sources",
    description: "Real-time aggregation from Yahoo Finance, SEC EDGAR, Finviz, News APIs, and more.",
  },
  {
    icon: Brain,
    title: "AI-Powered Reasoning",
    description: "Versioned prompt engine analyzes 250+ data points with reproducible, A/B-testable logic.",
  },
  {
    icon: Zap,
    title: "Live Streaming",
    description: "SSE real-time prediction pipeline streams each analysis stage as it happens.",
    badge: "New",
  },
  {
    icon: Shield,
    title: "Output Guardrails",
    description: "PII redaction, advice-language detection, and probability clamping between 15-85%.",
    badge: "New",
  },
  {
    icon: MessageSquare,
    title: "RAG Chat Assistant",
    description: "Ask follow-up questions grounded in the prediction data and source documents.",
    badge: "New",
  },
  {
    icon: Target,
    title: "Eval-Tested Quality",
    description: "17 golden test cases with 100% pass rate ensure consistent, reliable predictions.",
    badge: "New",
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
    title: "AI Analysis + Streaming",
    description: "Our AI engine processes 250+ data points via SSE, streaming each analysis stage to you in real time.",
    details: ["Technical: 25%", "News: 20%", "Options: 15%", "Market: 15%", "Analyst: 15%", "Social: 10%"],
  },
  {
    step: 4,
    icon: FileText,
    title: "Guardrailed Prediction",
    description: "Output guardrails enforce probability bounds (15-85%), redact PII, and flag advice language before delivery.",
    details: ["Probability clamped 15-85%", "PII redaction", "Advice detection", "Risk assessment"],
  },
];

/* ============================================
   COMPONENT
   ============================================ */

export default function Home() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<"online" | "offline" | "checking">("checking");
  const [tickerConfirmation, setTickerConfirmation] = useState<{
    result: TickerExtractionResult;
    query: string;
  } | null>(null);
  const [sseEvents, setSseEvents] = useState<SSEEvent[]>([]);
  const [activeQuery, setActiveQuery] = useState<string>("");
  const [queryGuidance, setQueryGuidance] = useState<{
    analysis: QueryAnalysisResult;
    query: string;
  } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Activate scroll reveals
  useScrollReveal();

  // Animated counters for hero stats
  const counter1 = useAnimatedCounter(12);
  const counter2 = useAnimatedCounter(250);
  const counter3 = useAnimatedCounter(17);

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

  const proceedWithTickerValidation = async (query: string) => {
    try {
      const validation = await api.validateTicker(query);

      if (validation.needs_confirmation && validation.confidence < 0.8) {
        setTickerConfirmation({ result: validation, query });
        return;
      }

      await executePrediction(query, validation.ticker || undefined);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to validate ticker. Please try again."
      );
    }
  };

  const handleSubmit = async (query: string) => {
    setError(null);
    setPrediction(null);
    setTickerConfirmation(null);
    setQueryGuidance(null);
    setIsAnalyzing(true);

    try {
      const analysis = await api.analyzeQuery(query);
      // Use the cleaned query (spelling fixes, gibberish removed) if available
      const effectiveQuery = analysis.cleaned_query || query;

      if (analysis.category === "clear") {
        setIsAnalyzing(false);
        await proceedWithTickerValidation(effectiveQuery);
      } else {
        setIsAnalyzing(false);
        setQueryGuidance({ analysis, query: effectiveQuery });
      }
    } catch {
      setIsAnalyzing(false);
      // If analyze-query fails, fall through to ticker validation directly
      await proceedWithTickerValidation(query);
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

      // The SSE done event now sends the full PredictionResponse,
      // so no second API call is needed
      if (streamResult) {
        setPrediction(streamResult);
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

  const handleGuidanceUseSuggestion = (suggestion: string) => {
    setQueryGuidance(null);
    setActiveQuery(suggestion);
    handleSubmit(suggestion);
  };

  const handleGuidanceProceedAnyway = () => {
    if (queryGuidance) {
      const query = queryGuidance.query;
      setQueryGuidance(null);
      proceedWithTickerValidation(query);
    }
  };

  const handleGuidanceCancel = () => {
    setQueryGuidance(null);
  };

  const handleNewPrediction = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <main className="min-h-screen bg-slate-50">
      {/* Query Guidance Dialog */}
      {queryGuidance && (
        <QueryGuidance
          analysis={queryGuidance.analysis}
          originalQuery={queryGuidance.query}
          onUseSuggestion={handleGuidanceUseSuggestion}
          onProceedAnyway={handleGuidanceProceedAnyway}
          onCancel={handleGuidanceCancel}
        />
      )}

      {/* Ticker Confirmation Dialog */}
      {tickerConfirmation && (
        <TickerConfirmation
          result={tickerConfirmation.result}
          originalQuery={tickerConfirmation.query}
          onConfirm={handleTickerConfirm}
          onCancel={handleTickerCancel}
        />
      )}

      {/* ============================================
          HEADER — Glassmorphism
          ============================================ */}
      <header className="border-b border-white/10 bg-white/80 backdrop-blur-xl sticky top-0 z-50 shadow-sm">
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
              className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            >
              <img
                src="/clarividex-logo.png"
                alt="Clarividex"
                className="w-8 h-8 object-contain"
              />
              <h1 className="text-lg font-bold text-slate-800 tracking-tight">
                Clarividex
              </h1>
              <span className={cn(
                "status-dot",
                apiStatus === "online" ? "online" : apiStatus === "offline" ? "offline" : "bg-amber-400"
              )} />
            </a>

            <nav className="flex items-center gap-6">
              {[
                { href: "#how-it-works", label: "How It Works" },
                { href: "#features", label: "Features" },
                { href: "#sources", label: "Data Sources" },
              ].map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  onClick={(e) => {
                    e.preventDefault();
                    setPrediction(null);
                    setError(null);
                    setIsLoading(false);
                    setTimeout(() => {
                      document.getElementById(link.href.slice(1))?.scrollIntoView({ behavior: "smooth" });
                    }, 50);
                  }}
                  className="relative text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors hidden sm:block group"
                >
                  {link.label}
                  <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-amber-500 transition-all duration-300 group-hover:w-full" />
                </a>
              ))}
            </nav>
          </div>
        </div>
      </header>

      {/* ============================================
          HERO — Dark gradient with ambient orbs
          ============================================ */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-amber-950 py-16 lg:py-24 px-4 sm:px-6 lg:px-8">
        {/* Ambient glow orbs */}
        <div className="absolute top-20 left-1/4 w-72 h-72 bg-amber-500/10 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-10 right-1/4 w-96 h-96 bg-orange-500/8 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-amber-400/5 rounded-full blur-3xl animate-float" style={{ animationDelay: "4s" }} />

        <div className="relative z-10 max-w-4xl mx-auto text-center mb-12">
          {/* Pill badge */}
          <div className="pill-badge mb-6 mx-auto w-fit">
            <Zap className="w-3 h-3" />
            The Clairvoyant Index
          </div>

          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white mb-4">
            Stock Predictions with{" "}
            <span className="gradient-text-vibrant">AI Analysis</span>
          </h2>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed mb-10">
            Probability-based forecasts backed by real-time data from 12+ sources,
            streamed live with production-grade guardrails and transparent reasoning.
          </p>

          {/* Animated stat counters */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 sm:gap-6 max-w-2xl mx-auto">
            <div ref={counter1.ref} className="glass-card-dark p-4 text-center">
              <div className="text-3xl font-bold text-white">{counter1.count}+</div>
              <div className="text-sm text-slate-400 mt-1">Data Sources</div>
            </div>
            <div ref={counter2.ref} className="glass-card-dark p-4 text-center">
              <div className="text-3xl font-bold text-white">{counter2.count}+</div>
              <div className="text-sm text-slate-400 mt-1">Data Points</div>
            </div>
            <div ref={counter3.ref} className="glass-card-dark p-4 text-center">
              <div className="text-3xl font-bold text-white">{counter3.count}/17</div>
              <div className="text-sm text-slate-400 mt-1">Eval Pass Rate</div>
            </div>
            <div className="glass-card-dark p-4 text-center">
              <div className="text-3xl font-bold text-emerald-400 flex items-center justify-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                Live
              </div>
              <div className="text-sm text-slate-400 mt-1">SSE Streaming</div>
            </div>
          </div>
        </div>

        {/* Search Form — white card pops on dark */}
        <div className="relative z-10">
          <PredictionForm onSubmit={handleSubmit} isLoading={isLoading} isAnalyzing={isAnalyzing} externalQuery={activeQuery} />
        </div>

        {/* Error Display — adjusted for dark bg */}
        {error && (
          <div className="relative z-10 max-w-2xl mx-auto mt-6">
            <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-4 flex items-start gap-3 backdrop-blur-sm">
              <div className="w-6 h-6 bg-red-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-3 h-3 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </div>
              <div>
                <p className="font-medium text-red-300 text-sm">Error</p>
                <p className="text-red-400 text-sm mt-0.5">{error}</p>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* ============================================
          RESULTS
          ============================================ */}
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

      {/* ============================================
          LANDING CONTENT (hidden when showing results)
          ============================================ */}
      {!prediction && !isLoading && (
        <>
          {/* ============================================
              HOW IT WORKS
              ============================================ */}
          <section id="how-it-works" className="section px-4 sm:px-6 lg:px-8 bg-slate-50">
            <div className="container-app">
              <div className="text-center mb-14 section-reveal">
                <h3 className="heading-2 mb-4">How It Works</h3>
                <p className="body-text max-w-xl mx-auto text-lg">
                  Our prediction engine combines real-time data aggregation with AI analysis
                  to generate probability-based forecasts.
                </p>
              </div>

              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 max-w-5xl mx-auto">
                {HOW_IT_WORKS.map((item, index) => (
                  <div key={index} className="step-card section-reveal" style={{ transitionDelay: `${index * 100}ms` }}>
                    <div className="flex items-center gap-3 mb-4">
                      <div className="step-number">{item.step}</div>
                      <item.icon className="w-5 h-5 text-slate-400" />
                    </div>
                    {/* Arrow connector (desktop) */}
                    {index < HOW_IT_WORKS.length - 1 && (
                      <div className="hidden lg:block absolute -right-3 top-1/2 -translate-y-1/2 z-10">
                        <ArrowRight className="w-5 h-5 text-amber-400/50" />
                      </div>
                    )}
                    <h4 className="heading-3 mb-2">{item.title}</h4>
                    <p className="text-sm text-slate-600 mb-3 leading-relaxed">{item.description}</p>

                    {item.example && (
                      <div className="text-sm text-slate-500 bg-slate-50 px-3 py-2 rounded-xl font-mono">
                        {item.example}
                      </div>
                    )}

                    {item.details && (
                      <ul className="space-y-1.5 mt-3">
                        {item.details.map((detail, i) => (
                          <li key={i} className="text-sm text-slate-500 flex items-center gap-2">
                            <CheckCircle className="w-3.5 h-3.5 text-amber-500" />
                            {detail}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>

              {/* Methodology Details */}
              <div className="mt-14 max-w-3xl mx-auto section-reveal">
                <div className="glass-card p-6 sm:p-8">
                  <h4 className="heading-3 mb-5 flex items-center gap-2">
                    <Target className="w-5 h-5 text-amber-500" />
                    Prediction Methodology
                  </h4>
                  <div className="grid sm:grid-cols-2 gap-5 text-sm">
                    <div className="space-y-4">
                      <div>
                        <div className="font-semibold text-slate-700 mb-1">Weighted Scoring Model</div>
                        <p className="text-slate-500 leading-relaxed">Each data source contributes to the final probability based on its reliability and relevance.</p>
                      </div>
                      <div>
                        <div className="font-semibold text-slate-700 mb-1">Multi-Source Validation</div>
                        <p className="text-slate-500 leading-relaxed">Cross-references data points across sources to identify consensus and divergence.</p>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div>
                        <div className="font-semibold text-slate-700 mb-1">Timeframe Adjustment</div>
                        <p className="text-slate-500 leading-relaxed">Probabilities are calibrated based on the prediction timeframe (day, week, month).</p>
                      </div>
                      <div>
                        <div className="font-semibold text-slate-700 mb-1">Confidence Calculation</div>
                        <p className="text-slate-500 leading-relaxed">Confidence reflects data agreement, source count, and signal clarity.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ============================================
              V2 TECHNOLOGY BANNER — Full bleed dark
              ============================================ */}
          <section className="relative overflow-hidden px-4 sm:px-6 lg:px-8 py-14 bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900">
            {/* Ambient orb */}
            <div className="absolute top-1/2 right-0 -translate-y-1/2 w-80 h-80 bg-amber-500/10 rounded-full blur-3xl animate-float" />

            <div className="container-app relative z-10">
              <div className="flex items-center gap-3 mb-6 section-reveal">
                <span className="pill-badge">
                  <Zap className="w-3 h-3" />
                  V2 Enhancements
                </span>
              </div>
              <h3 className="text-3xl sm:text-4xl font-bold text-white mb-8 section-reveal">
                Production-Grade AI Infrastructure
              </h3>
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {[
                  { icon: Zap, label: "SSE Streaming", desc: "Real-time prediction pipeline", featured: true },
                  { icon: Shield, label: "Output Guardrails", desc: "PII redaction, advice detection", featured: false },
                  { icon: MessageSquare, label: "RAG Chatbot", desc: "Doc-grounded follow-up Q&A", featured: false },
                  { icon: Layers, label: "Prompt Versioning", desc: "YAML-based, A/B testable", featured: false },
                  { icon: Target, label: "Eval Suite", desc: "17 test cases, 100% pass rate", featured: false },
                  { icon: RefreshCw, label: "Singleton Caching", desc: "Optimized inference pipeline", featured: false },
                ].map((item, i) => (
                  <div
                    key={i}
                    className={cn(
                      "glass-card-dark p-4 flex items-start gap-4 transition-all duration-300 hover:-translate-y-0.5 section-reveal",
                      i === 0 && "sm:col-span-2 lg:col-span-1"
                    )}
                    style={{ transitionDelay: `${i * 80}ms` }}
                  >
                    <div className="w-11 h-11 rounded-xl bg-amber-500/10 flex items-center justify-center shrink-0">
                      <item.icon className="w-5 h-5 text-amber-400" />
                    </div>
                    <div>
                      <div className="text-base font-semibold text-white">{item.label}</div>
                      <div className="text-sm text-slate-400 mt-0.5">{item.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* ============================================
              FEATURES — Bento grid
              ============================================ */}
          <section id="features" className="section px-4 sm:px-6 lg:px-8 bg-white">
            <div className="container-app">
              <div className="text-center mb-14 section-reveal">
                <h3 className="heading-2 mb-4">Features</h3>
                <p className="body-text max-w-xl mx-auto text-lg">
                  Unlike generic chatbots, we aggregate real market data and provide
                  transparent, quantified predictions.
                </p>
              </div>

              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-4xl mx-auto">
                {FEATURES.map((feature, index) => (
                  <div
                    key={index}
                    className={cn(
                      "bento-item relative section-reveal",
                      index === 0 && "lg:col-span-2"
                    )}
                    style={{ transitionDelay: `${index * 80}ms` }}
                  >
                    {feature.badge && (
                      <span className="absolute top-4 right-4 px-2.5 py-1 bg-gradient-to-r from-amber-500 to-orange-500 text-white text-xs font-semibold rounded-full">
                        {feature.badge}
                      </span>
                    )}
                    <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-amber-100 to-orange-100 flex items-center justify-center mb-4">
                      <feature.icon className="w-5 h-5 text-amber-600" />
                    </div>
                    <h4 className="font-semibold text-slate-800 text-base mb-2">{feature.title}</h4>
                    <p className="text-sm text-slate-500 leading-relaxed">{feature.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* ============================================
              DATA SOURCES
              ============================================ */}
          <section id="sources" className="section px-4 sm:px-6 lg:px-8 bg-slate-50">
            <div className="container-app">
              <div className="text-center mb-14 section-reveal">
                <h3 className="heading-2 mb-4">Data Sources</h3>
                <p className="body-text max-w-xl mx-auto text-lg">
                  We aggregate data from 12+ trusted sources with reliability scoring.
                </p>
              </div>

              {/* Source cards with progress bars */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-w-3xl mx-auto section-reveal">
                {DATA_SOURCES.map((source, index) => (
                  <div
                    key={index}
                    className="glass-card p-3 text-center"
                  >
                    <span className="text-sm font-medium text-slate-700 block mb-2">{source.name}</span>
                    <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={cn(
                          "h-full rounded-full transition-all duration-1000",
                          source.reliability >= 90 ? "bg-emerald-500" :
                          source.reliability >= 80 ? "bg-amber-500" :
                          "bg-slate-400"
                        )}
                        style={{ width: `${source.reliability}%` }}
                      />
                    </div>
                    <span className="text-xs text-slate-400 mt-1 block">{source.reliability}%</span>
                  </div>
                ))}
              </div>

              {/* Why Us vs AI Agents */}
              <div className="mt-16 max-w-4xl mx-auto">
                <div className="text-center mb-10 section-reveal">
                  <h4 className="heading-2 mb-4">Why Use Us Instead of AI Agents?</h4>
                  <p className="body-text max-w-2xl mx-auto text-lg">
                    Generic AI agents can search the web, but they can&apos;t give you what we provide.
                    See the difference:
                  </p>
                </div>

                {/* Response Comparison */}
                <div className="grid md:grid-cols-2 gap-4 sm:gap-6 mb-10 section-reveal">
                  {/* AI Agent Response */}
                  <div className="glass-card p-4 sm:p-6 bg-slate-100/70">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-9 h-9 rounded-xl bg-slate-300 flex items-center justify-center">
                        <svg className="w-4 h-4 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                      </div>
                      <span className="font-semibold text-slate-600">Generic AI Agent</span>
                    </div>
                    <div className="bg-white rounded-xl p-4 text-sm text-slate-600 italic">
                      &quot;Tesla stock could go up or down depending on various factors like market conditions,
                      earnings reports, and investor sentiment. I cannot provide specific financial advice
                      or predictions. Please consult a financial advisor.&quot;
                    </div>
                    <div className="mt-4 space-y-2">
                      <div className="flex items-center gap-2 text-sm text-slate-400">
                        <XCircle className="w-4 h-4" />
                        No specific probability
                      </div>
                      <div className="flex items-center gap-2 text-sm text-slate-400">
                        <XCircle className="w-4 h-4" />
                        No real-time data
                      </div>
                      <div className="flex items-center gap-2 text-sm text-slate-400">
                        <XCircle className="w-4 h-4" />
                        No technical analysis
                      </div>
                    </div>
                  </div>

                  {/* Our Response */}
                  <div className="glass-card p-4 sm:p-6 animate-glow-pulse">
                    <div className="flex items-center gap-2 mb-4">
                      <img src="/clarividex-logo.png" alt="" className="w-12 h-12 sm:w-16 sm:h-16 object-contain" />
                      <span className="font-bold text-amber-700 text-xl">Clarividex</span>
                    </div>
                    <div className="bg-white rounded-xl p-4 text-sm text-slate-700">
                      <div className="text-2xl font-bold text-emerald-600 mb-2">62% Probability</div>
                      <div className="space-y-1.5 text-sm">
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
                      <div className="mt-3 pt-3 border-t border-slate-100 text-sm text-slate-500">
                        Based on 287 data points from 13 sources
                      </div>
                    </div>
                    <div className="mt-4 space-y-2">
                      <div className="flex items-center gap-2 text-sm text-amber-600">
                        <CheckCircle className="w-4 h-4" />
                        Quantified probability score
                      </div>
                      <div className="flex items-center gap-2 text-sm text-amber-600">
                        <CheckCircle className="w-4 h-4" />
                        Live technical indicators
                      </div>
                      <div className="flex items-center gap-2 text-sm text-amber-600">
                        <CheckCircle className="w-4 h-4" />
                        Full transparency
                      </div>
                    </div>
                  </div>
                </div>

                {/* Feature Comparison Table */}
                <div className="glass-card overflow-hidden section-reveal">
                  <div className="p-4 sm:p-6 bg-gradient-to-r from-slate-800 to-slate-900 rounded-t-2xl">
                    <h4 className="text-lg font-bold text-white">
                      Feature Comparison
                    </h4>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-slate-50/80 border-b border-slate-200">
                        <tr>
                          <th className="text-left px-3 py-3 sm:px-5 sm:py-4 font-semibold text-slate-600">Feature</th>
                          <th className="text-center px-3 py-3 sm:px-5 sm:py-4 font-semibold text-slate-400">AI Agents</th>
                          <th className="text-center px-3 py-3 sm:px-5 sm:py-4 font-semibold text-amber-600">Clarividex</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {[
                          { feature: "Probability Score", agent: '"Could go up or down"', us: "62% with confidence level" },
                          { feature: "Real-time RSI, MACD", agent: null, us: true },
                          { feature: "Live Options Flow (Put/Call)", agent: null, us: true },
                          { feature: "VIX & Fear/Greed Index", agent: null, us: true },
                          { feature: "12+ Data Sources", agent: "Random web results", us: "SEC, Yahoo, Finviz, etc." },
                          { feature: "Consistent Methodology", agent: "Varies each time", us: "Same weighted scoring" },
                          { feature: "Transparent Reasoning", agent: null, us: true },
                          { feature: "Impact News Analysis", agent: "General search", us: "Crashes, lawsuits, recalls" },
                          { feature: "Real-Time Streaming", agent: null, us: true },
                          { feature: "Output Guardrails", agent: null, us: true },
                        ].map((row, i) => (
                          <tr key={i} className={i % 2 === 1 ? "bg-slate-50/50" : ""}>
                            <td className="px-3 py-3 sm:px-5 sm:py-3.5 text-slate-700 font-medium">{row.feature}</td>
                            <td className="px-3 py-3 sm:px-5 sm:py-3.5 text-center">
                              {row.agent === null ? (
                                <XCircle className="w-4 h-4 text-slate-300 mx-auto" />
                              ) : (
                                <span className="text-slate-400">{row.agent}</span>
                              )}
                            </td>
                            <td className="px-3 py-3 sm:px-5 sm:py-3.5 text-center">
                              {row.us === true ? (
                                <CheckCircle className="w-4 h-4 text-amber-500 mx-auto" />
                              ) : (
                                <span className="font-medium text-amber-600">{row.us}</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ============================================
              CTA — Dark gradient with radial mesh
              ============================================ */}
          <section className="relative overflow-hidden py-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-slate-900 via-slate-800 to-amber-900">
            {/* Radial mesh */}
            <div className="absolute inset-0 opacity-30" style={{
              backgroundImage: "radial-gradient(circle at 30% 50%, rgba(245,158,11,0.15) 0%, transparent 50%), radial-gradient(circle at 70% 50%, rgba(234,88,12,0.1) 0%, transparent 50%)"
            }} />

            <div className="container-app text-center relative z-10">
              <div className="max-w-xl mx-auto section-reveal">
                <h3 className="text-3xl sm:text-4xl font-bold text-white mb-4">
                  Ready to See the Future?
                </h3>
                <p className="text-slate-300 mb-8 text-lg">
                  Ask Clarividex about any US stock. No signup required.
                </p>
                <button
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                  className="group inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-semibold rounded-2xl text-lg transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-amber-500/25"
                >
                  Start Predicting
                  <ArrowRight className="w-5 h-5 transition-transform duration-300 group-hover:translate-x-1" />
                </button>
              </div>
            </div>
          </section>
        </>
      )}

      {/* ============================================
          FOOTER — Glassmorphism
          ============================================ */}
      <footer className="border-t border-slate-200/50 bg-white/80 backdrop-blur-xl py-8 px-4 sm:px-6 lg:px-8">
        <div className="container-app">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <img
                src="/clarividex-logo.png"
                alt=""
                className="w-8 h-8 object-contain"
              />
              <span className="font-bold text-slate-700 text-lg">Clarividex</span>
            </div>

            <div className="flex items-center gap-2 text-slate-500 text-sm">
              <Shield className="w-4 h-4" />
              <span>Not financial advice. For informational purposes only.</span>
            </div>

            <div className="text-sm text-slate-400">
              &copy; {new Date().getFullYear()} Clarividex
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}
