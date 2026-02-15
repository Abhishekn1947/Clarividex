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
  Linkedin,
} from "lucide-react";
import { PredictionForm } from "@/components/PredictionForm";
import { PredictionResult } from "@/components/PredictionResult";
import { LoadingSkeleton } from "@/components/LoadingSkeleton";
import { TickerConfirmation } from "@/components/TickerConfirmation";
import { QueryGuidance } from "@/components/QueryGuidance";
import { api, PredictionResponse, TickerExtractionResult, QueryAnalysisResult, SSEEvent, Market } from "@/lib/api";
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

const getArchitecturePillars = (market: Market) => [
  {
    title: "Data Ingestion",
    subtitle: "Real-time market data pipeline",
    icon: Database,
    gradient: "from-emerald-500 to-teal-600",
    iconBg: "bg-emerald-100",
    iconColor: "text-emerald-600",
    dotColor: "bg-emerald-400",
    features: [
      {
        label: market === "IN" ? "10+ Live Sources" : "12+ Live Sources",
        detail: market === "IN"
          ? "Yahoo Finance, NSE/BSE data, India VIX, Google News India, Nifty sector indices, Reddit"
          : "Yahoo Finance, SEC EDGAR, Finviz, VIX, Fear & Greed, options flow, Google News, Reddit",
      },
      {
        label: "SSE Streaming",
        detail: "Server-Sent Events pipe each analysis stage to the client in real time",
      },
      {
        label: "Data Quality Scoring",
        detail: "Every source scored for reliability; cross-validated before aggregation",
      },
    ],
  },
  {
    title: "Intelligence Engine",
    subtitle: "8-model probability matrix with hot-swappable AI backbone",
    icon: Brain,
    gradient: "from-amber-500 to-orange-600",
    iconBg: "bg-amber-100",
    iconColor: "text-amber-600",
    dotColor: "bg-amber-400",
    features: [
      {
        label: "Swappable AI Core",
        detail: "Currently Gemini 2.0 Flash with versioned YAML prompts. Architecture supports hot-swapping to Claude Opus 4.6, GPT-4o, or any frontier model for deeper reasoning without changing the pipeline",
      },
      {
        label: "Probability Matrix",
        detail: "8 independent models (Monte Carlo, Bayesian inference, fat-tail distribution, technical scoring, sentiment regression, historical pattern matching, options flow analysis, multi-timeframe trend) each output a probability. Weighted ensemble combines them via confidence-adjusted averaging into a single calibrated score (15\u201385%)",
      },
      {
        label: "2,000 Monte Carlo Simulations",
        detail: "Stochastic price path simulations using historical volatility and drift. Bayesian priors update in real time as new data arrives, adjusting the posterior probability distribution before final scoring",
      },
    ],
  },
  {
    title: "Production Safety",
    subtitle: "Guardrails and quality assurance",
    icon: Shield,
    gradient: "from-blue-500 to-indigo-600",
    iconBg: "bg-blue-100",
    iconColor: "text-blue-600",
    dotColor: "bg-blue-400",
    features: [
      {
        label: "Output Guardrails",
        detail: "PII redaction, advice-language detection, probability clamping (15-85%)",
      },
      {
        label: "RAG Chat Assistant",
        detail: "ChromaDB + LangChain for document-grounded follow-up Q&A",
      },
      {
        label: "Evaluation Suite",
        detail: "17 golden test cases, 100% pass rate, automated regression testing",
      },
    ],
  },
];

const DATA_SOURCES_US = [
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

const DATA_SOURCES_IN = [
  { name: "Yahoo Finance", reliability: 90 },
  { name: "NSE/BSE Data", reliability: 95 },
  { name: "Google News India", reliability: 80 },
  { name: "Reddit India", reliability: 70 },
  { name: "India VIX", reliability: 90 },
  { name: "Nifty Indices", reliability: 95 },
  { name: "Options Data", reliability: 85 },
  { name: "Technical Analysis", reliability: 85 },
  { name: "Sector Indices", reliability: 90 },
  { name: "Clarividex AI", reliability: 75 },
];

const getHowItWorks = (market: Market) => [
  {
    step: 1,
    icon: Search,
    title: "Enter Your Question",
    description: market === "IN"
      ? "Ask about any Indian stock prediction. Include NSE ticker, target price, or timeframe for best results."
      : "Ask about any US stock prediction. Include ticker, target price, or timeframe for best results.",
    example: market === "IN"
      ? '"Will Reliance reach \u20b93000 by March 2026?"'
      : '"Will NVDA reach $180 by March 2026?"',
  },
  {
    step: 2,
    icon: Database,
    title: "Data Aggregation",
    description: market === "IN"
      ? "We fetch real-time data from 10+ sources including Yahoo Finance, NSE/BSE data, Indian news, and social media."
      : "We fetch real-time data from 12+ sources including Yahoo Finance, SEC filings, news, and social media.",
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
   CROSS-MARKET DETECTION
   ============================================ */

function detectCrossMarket(
  query: string,
  market: Market
): { targetMarket: Market; message: string } | null {
  // Indian stock signals when user is on US side
  if (market === "US") {
    const hasRupee = query.includes("₹");
    const hasNSBO = /\.\s*(?:ns|bo)\b/i.test(query);
    const indianTerms =
      /\b(reliance|infosys|tcs|hdfc\s*bank|icici|wipro|nifty|sensex|nse|bse|bharti\s*airtel|itc\b|maruti|bajaj\s*finance|adani|kotak|sun\s*pharma|asian\s*paints|titan|ntpc|ongc|coal\s*india|sbi|axis\s*bank|hindustan\s*unilever|power\s*grid|larsen|hul)\b/i;
    if (hasRupee || hasNSBO || indianTerms.test(query)) {
      return {
        targetMarket: "IN",
        message:
          "This looks like an Indian stock. Switch to India mode for accurate NSE/BSE data and ₹ pricing.",
      };
    }
  }

  // US stock signals when user is on India side
  if (market === "IN") {
    const usTickerPattern =
      /\b(AAPL|TSLA|NVDA|MSFT|AMZN|GOOG|GOOGL|META|AMD|NFLX|SPY|QQQ|DIS|BA|JPM|GS|INTC|CRM|ORCL|PYPL|UBER|ABNB|COIN)\b/;
    const usCompanyNames =
      /\b(tesla|apple|nvidia|microsoft|amazon|google|alphabet|meta|facebook|netflix|disney|boeing|intel|salesforce|oracle|paypal|uber|airbnb|coinbase|amd)\b/i;
    const usTerms = /\b(s&p\s*500|nasdaq|dow\s*jones|nyse|wall\s*street)\b/i;
    const hasDollarPrice = /\$\d/.test(query);
    if (usTickerPattern.test(query) || usCompanyNames.test(query) || usTerms.test(query) || hasDollarPrice) {
      return {
        targetMarket: "US",
        message:
          "This looks like a US stock. Switch to USA mode for accurate NYSE/NASDAQ data and $ pricing.",
      };
    }
  }

  return null;
}

/* ============================================
   COMPONENT
   ============================================ */

export default function Home() {
  const [market, setMarketState] = useState<Market>(() => {
    if (typeof window !== "undefined") {
      const saved = sessionStorage.getItem("clarividex_market");
      if (saved === "US" || saved === "IN") return saved;
    }
    return "US";
  });
  const setMarket = (m: Market) => {
    setMarketState(m);
    if (typeof window !== "undefined") {
      sessionStorage.setItem("clarividex_market", m);
    }
  };
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
  const [crossMarketSuggestion, setCrossMarketSuggestion] = useState<{
    targetMarket: Market;
    message: string;
    query: string;
  } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Activate scroll reveals
  useScrollReveal();

  // Animated counters for hero stats
  const counter1 = useAnimatedCounter(12);
  const counter2 = useAnimatedCounter(250);
  const counter3 = useAnimatedCounter(8);

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
      const validation = await api.validateTicker(query, market);

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

  const runQueryAnalysis = async (query: string) => {
    setIsAnalyzing(true);
    try {
      const analysis = await api.analyzeQuery(query, market);
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

  const handleSubmit = async (query: string) => {
    setError(null);
    setPrediction(null);
    setTickerConfirmation(null);
    setQueryGuidance(null);
    setCrossMarketSuggestion(null);

    // Check for cross-market query
    const crossMarket = detectCrossMarket(query, market);
    if (crossMarket) {
      setCrossMarketSuggestion({ ...crossMarket, query });
      return;
    }

    await runQueryAnalysis(query);
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
      const result = await api.createPrediction({
        query,
        ticker,
        include_technicals: true,
        include_sentiment: true,
        include_news: true,
        market,
      });

      setPrediction(result);
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

  const handleCrossMarketSwitch = () => {
    if (!crossMarketSuggestion) return;
    setMarket(crossMarketSuggestion.targetMarket);
    setActiveQuery(crossMarketSuggestion.query);
    setCrossMarketSuggestion(null);
  };

  const handleCrossMarketContinue = () => {
    if (!crossMarketSuggestion) return;
    const query = crossMarketSuggestion.query;
    setCrossMarketSuggestion(null);
    runQueryAnalysis(query);
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

            <nav className="flex items-center gap-4 sm:gap-6">
              {/* Market Selector — Animated Toggle */}
              <div className="relative flex items-center bg-slate-100 rounded-full p-1 w-[136px] sm:w-[168px]">
                {/* Sliding indicator */}
                <div
                  className={cn(
                    "absolute top-1 h-[calc(100%-8px)] w-[calc(50%-4px)] rounded-full bg-white shadow-md transition-all duration-300 ease-in-out",
                    market === "US" ? "left-1" : "left-[calc(50%+2px)]"
                  )}
                />
                <button
                  onClick={() => { setMarket("US"); setPrediction(null); setError(null); setActiveQuery(""); setCrossMarketSuggestion(null); }}
                  className={cn(
                    "relative z-10 flex items-center justify-center gap-1.5 w-1/2 py-1.5 rounded-full text-xs font-semibold transition-colors duration-300",
                    market === "US" ? "text-slate-800" : "text-slate-400 hover:text-slate-600"
                  )}
                >
                  <span className="text-base leading-none">🇺🇸</span>
                  <span>USA</span>
                </button>
                <button
                  onClick={() => { setMarket("IN"); setPrediction(null); setError(null); setActiveQuery(""); setCrossMarketSuggestion(null); }}
                  className={cn(
                    "relative z-10 flex items-center justify-center gap-1.5 w-1/2 py-1.5 rounded-full text-xs font-semibold transition-colors duration-300",
                    market === "IN" ? "text-slate-800" : "text-slate-400 hover:text-slate-600"
                  )}
                >
                  <span className="text-base leading-none">🇮🇳</span>
                  <span>India</span>
                </button>
              </div>

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
            The Clairvoyant Index — AI Stock Prediction Analysis
          </div>

          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white mb-4 whitespace-nowrap">
            Market Intelligence, <span className="gradient-text-vibrant">AI Engineered</span>
          </h2>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed mb-10">
            8 algorithms. 2,000 Monte Carlo simulations. Bayesian inference, fat-tail modeling,
            and multi-timeframe trend analysis — fused into one calibrated probability,
            powered by the Clarividex engine.
          </p>

          {/* Animated stat counters */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 sm:gap-6 max-w-2xl mx-auto">
            <div ref={counter1.ref} className="glass-card-dark p-4 text-center">
              <div className="text-3xl font-bold text-white">{market === "IN" ? "10+" : `${counter1.count}+`}</div>
              <div className="text-sm text-slate-400 mt-1">{market === "IN" ? "NSE/BSE Sources" : "Data Sources"}</div>
            </div>
            <div ref={counter2.ref} className="glass-card-dark p-4 text-center">
              <div className="text-3xl font-bold text-white">{counter2.count}+</div>
              <div className="text-sm text-slate-400 mt-1">Data Points</div>
            </div>
            <div ref={counter3.ref} className="glass-card-dark p-4 text-center">
              <div className="text-3xl font-bold text-white">{counter3.count}</div>
              <div className="text-sm text-slate-400 mt-1">Prediction Algorithms</div>
            </div>
            <div className="glass-card-dark p-4 text-center">
              <div className="text-3xl font-bold text-emerald-400 flex items-center justify-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                2,000
              </div>
              <div className="text-sm text-slate-400 mt-1">Monte Carlo Simulations</div>
            </div>
          </div>
        </div>

        {/* Search Form — white card pops on dark */}
        <div className="relative z-10">
          <PredictionForm onSubmit={handleSubmit} isLoading={isLoading} isAnalyzing={isAnalyzing} externalQuery={activeQuery} market={market} />
        </div>

        {/* Market Timing Info */}
        <div className="relative z-10 max-w-2xl mx-auto mt-4">
          <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-[11px] text-slate-500">
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>
                {market === "IN"
                  ? "NSE/BSE: 9:15 AM \u2013 3:30 PM IST"
                  : "NYSE/NASDAQ: 9:30 AM \u2013 4:00 PM ET"}
              </span>
            </div>
            <span className="text-slate-600 hidden sm:inline">&middot;</span>
            <span>Short-term & long-term predictions</span>
            <span className="text-slate-600 hidden sm:inline">&middot;</span>
            <span>Data freshness varies with market hours</span>
          </div>
        </div>

        {/* Cross-Market Suggestion */}
        {crossMarketSuggestion && (
          <div className="relative z-10 max-w-2xl mx-auto mt-6">
            <div className="bg-amber-500/10 border border-amber-500/30 rounded-2xl p-4 flex flex-col sm:flex-row items-start sm:items-center gap-3 backdrop-blur-sm">
              <div className="flex items-start gap-3 flex-1">
                <span className="text-xl leading-none mt-0.5">
                  {crossMarketSuggestion.targetMarket === "IN" ? "\ud83c\uddee\ud83c\uddf3" : "\ud83c\uddfa\ud83c\uddf8"}
                </span>
                <p className="text-sm text-amber-200">{crossMarketSuggestion.message}</p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  onClick={handleCrossMarketSwitch}
                  className="px-4 py-1.5 bg-amber-500 text-white text-xs font-semibold rounded-full hover:bg-amber-600 transition-colors"
                >
                  Switch to {crossMarketSuggestion.targetMarket === "IN" ? "India" : "USA"}
                </button>
                <button
                  onClick={handleCrossMarketContinue}
                  className="px-4 py-1.5 text-amber-300 text-xs font-medium hover:text-amber-200 transition-colors"
                >
                  Continue
                </button>
              </div>
            </div>
          </div>
        )}

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
                <PredictionResult prediction={prediction} market={market} />
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
                {getHowItWorks(market).map((item, index) => (
                  <div key={index} className="step-card section-reveal" style={{ transitionDelay: `${index * 100}ms` }}>
                    <div className="flex items-center gap-3 mb-4">
                      <div className="step-number">{item.step}</div>
                      <item.icon className="w-5 h-5 text-slate-400" />
                    </div>
                    {/* Arrow connector (desktop) */}
                    {index < getHowItWorks(market).length - 1 && (
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
                  { icon: Zap, label: "8-Model Ensemble", desc: "Monte Carlo, Bayesian, trend analysis & more", featured: true },
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
              FEATURES — Three-pillar architecture
              ============================================ */}
          <section id="features" className="section px-4 sm:px-6 lg:px-8 bg-white">
            <div className="container-app">
              <div className="text-center mb-14 section-reveal">
                <h3 className="heading-2 mb-4">Architecture</h3>
                <p className="body-text max-w-2xl mx-auto text-lg">
                  Three-layer pipeline: ingest real-time market data, run it through an 8-model
                  AI ensemble, and deliver guardrailed predictions with full transparency.
                </p>
              </div>

              <div className="grid lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
                {getArchitecturePillars(market).map((pillar, i) => (
                  <div key={i} className="relative section-reveal" style={{ transitionDelay: `${i * 120}ms` }}>
                    {/* Connector arrow between pillars (desktop) */}
                    {i < 2 && (
                      <div className="hidden lg:flex absolute -right-3 top-1/2 -translate-y-1/2 z-10 w-6 h-6 items-center justify-center">
                        <ArrowRight className="w-5 h-5 text-slate-300" />
                      </div>
                    )}
                    <div className="glass-card p-6 h-full hover:-translate-y-1 transition-transform duration-300">
                      {/* Pillar header */}
                      <div className="flex items-center gap-3 mb-5">
                        <div className={cn("w-10 h-10 rounded-xl flex items-center justify-center", pillar.iconBg)}>
                          <pillar.icon className={cn("w-5 h-5", pillar.iconColor)} />
                        </div>
                        <div>
                          <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Layer {i + 1}</div>
                          <h4 className="font-bold text-slate-800 text-base">{pillar.title}</h4>
                        </div>
                      </div>
                      <p className="text-sm text-slate-500 mb-5 leading-relaxed">{pillar.subtitle}</p>

                      {/* Sub-features */}
                      <div className="space-y-4">
                        {pillar.features.map((feat, j) => (
                          <div key={j} className="flex items-start gap-3">
                            <div className={cn("w-1.5 h-1.5 rounded-full mt-[7px] shrink-0", pillar.dotColor)} />
                            <div>
                              <div className="text-sm font-semibold text-slate-700">{feat.label}</div>
                              <div className="text-xs text-slate-400 mt-0.5 leading-relaxed">{feat.detail}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
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
                  {market === "IN"
                    ? "We aggregate data from 10+ trusted sources covering NSE, BSE, India VIX, and Indian financial news — each scored for reliability."
                    : "We aggregate data from 12+ trusted sources including SEC EDGAR, options flow, and VIX — each scored for reliability."}
                </p>
              </div>

              {/* Source cards with progress bars */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-w-3xl mx-auto section-reveal">
                {(market === "IN" ? DATA_SOURCES_IN : DATA_SOURCES_US).map((source, index) => (
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
                          { feature: market === "IN" ? "India VIX & Nifty Indices" : "VIX & Fear/Greed Index", agent: null, us: true },
                          { feature: market === "IN" ? "10+ Data Sources" : "12+ Data Sources", agent: "Random web results", us: market === "IN" ? "NSE, Yahoo, India VIX, etc." : "SEC, Yahoo, Finviz, etc." },
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
                  Ask Clarividex about any {market === "IN" ? "Indian" : "US"} stock. No signup required.
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

            <div className="flex items-center gap-4">
              <a
                href="https://www.linkedin.com/in/abhin1998/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-400 hover:text-blue-600 transition-colors"
                aria-label="LinkedIn"
              >
                <Linkedin className="w-5 h-5" />
              </a>
              <span className="text-sm text-slate-400">
                &copy; {new Date().getFullYear()} Clarividex
              </span>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}
