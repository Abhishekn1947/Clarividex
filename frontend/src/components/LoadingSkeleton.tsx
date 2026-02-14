"use client";

import { Brain, BarChart3, Newspaper, MessageSquare, TrendingUp, Database, Zap } from "lucide-react";
import { cn } from "@/lib/utils";
import { useEffect, useState, useMemo } from "react";
import type { SSEEvent } from "@/lib/api";

const LOADING_STEPS = [
  { icon: Database, label: "Fetching stock data..." },
  { icon: BarChart3, label: "Calculating technical indicators..." },
  { icon: Newspaper, label: "Analyzing news sentiment..." },
  { icon: MessageSquare, label: "Processing social media..." },
  { icon: TrendingUp, label: "Evaluating market conditions..." },
  { icon: Brain, label: "Clarividex is analyzing..." },
  { icon: Zap, label: "Generating prediction..." },
];

// Map SSE event names to loading step indices
const SSE_EVENT_TO_STEP: Record<string, number> = {
  data_fetch: 0,
  technical_analysis: 1,
  sentiment_analysis: 2,
  social_analysis: 3,
  market_conditions: 4,
  ai_reasoning: 5,
  probability_calculation: 6,
  done: 7,
};

const DATA_SOURCE_LABELS = ["Yahoo Finance", "SEC EDGAR", "News APIs", "Reddit", "Finviz", "VIX", "Options"];

interface LoadingSkeletonProps {
  sseEvents?: SSEEvent[];
}

export function LoadingSkeleton({ sseEvents }: LoadingSkeletonProps) {
  const [timerStep, setTimerStep] = useState(0);

  // Compute SSE-driven step from events
  const sseStep = useMemo(() => {
    if (!sseEvents || sseEvents.length === 0) return -1;
    let maxStep = -1;
    for (const evt of sseEvents) {
      const step = SSE_EVENT_TO_STEP[evt.event];
      if (step !== undefined && step > maxStep) {
        maxStep = step;
      }
    }
    return maxStep;
  }, [sseEvents]);

  // Use SSE step when available, otherwise fall back to timer
  const currentStep = sseStep >= 0 ? Math.min(sseStep, LOADING_STEPS.length - 1) : timerStep;
  const progressPercent = ((currentStep + 1) / LOADING_STEPS.length) * 100;

  useEffect(() => {
    // Only run timer if no SSE events are driving the UI
    if (sseStep >= 0) return;

    const interval = setInterval(() => {
      setTimerStep((prev) => (prev + 1) % LOADING_STEPS.length);
    }, 2000);

    return () => clearInterval(interval);
  }, [sseStep]);

  // SVG progress ring calculations
  const ringRadius = 44;
  const ringCircumference = 2 * Math.PI * ringRadius;
  const ringOffset = ringCircumference - (progressPercent / 100) * ringCircumference;

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Main loading card — dark premium */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl border border-white/10 overflow-hidden shadow-2xl">
        {/* Header skeleton — dark */}
        <div className="bg-gradient-to-r from-slate-800 to-slate-900 p-4 sm:p-6">
          <div className="skeleton-premium h-5 w-3/4 rounded-lg mb-2" />
          <div className="skeleton-premium h-3 w-1/2 rounded-lg" />
        </div>

        {/* Main content */}
        <div className="p-6 sm:p-8">
          {/* Central orbital visualization */}
          <div className="flex flex-col items-center justify-center mb-8">
            <div className="relative w-28 h-28">
              {/* Outer glow ping */}
              <div className="absolute inset-0 rounded-full bg-amber-500/10 animate-ping" style={{ animationDuration: "3s" }} />

              {/* SVG progress ring */}
              <svg className="absolute inset-0 w-full h-full -rotate-90" viewBox="0 0 100 100">
                {/* Track */}
                <circle
                  cx="50" cy="50" r={ringRadius}
                  fill="none"
                  stroke="rgba(255,255,255,0.05)"
                  strokeWidth="4"
                />
                {/* Progress */}
                <circle
                  cx="50" cy="50" r={ringRadius}
                  fill="none"
                  stroke="url(#progressGradient)"
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeDasharray={ringCircumference}
                  strokeDashoffset={ringOffset}
                  className="transition-all duration-700 ease-out"
                />
                <defs>
                  <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#f59e0b" />
                    <stop offset="100%" stopColor="#ea580c" />
                  </linearGradient>
                </defs>
              </svg>

              {/* Orbiting dot */}
              <div className="absolute inset-0 animate-spin" style={{ animationDuration: "4s" }}>
                <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1 w-2.5 h-2.5 rounded-full bg-amber-400 shadow-lg shadow-amber-400/50" />
              </div>

              {/* Center content */}
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <Brain className="w-7 h-7 text-amber-400 mb-1" />
                <span className="text-white text-sm font-bold">{currentStep + 1}/{LOADING_STEPS.length}</span>
              </div>
            </div>

            <div className="mt-6 text-center">
              <p className="text-base font-semibold text-white mb-1">
                Clarividex is Reading the Markets
              </p>
              <p className="text-slate-400 text-sm">
                {sseStep >= 0 ? "Receiving live analysis updates..." : "This usually takes 10-15 seconds"}
              </p>
            </div>
          </div>

          {/* Loading steps — dark cards */}
          <div className="bg-white/5 rounded-2xl p-4 sm:p-5 border border-white/5">
            <div className="space-y-2">
              {LOADING_STEPS.map((step, index) => {
                const Icon = step.icon;
                const isActive = index === currentStep;
                const isCompleted = index < currentStep;

                return (
                  <div
                    key={index}
                    className={cn(
                      "flex items-center gap-3 p-3 rounded-xl transition-all duration-300",
                      isActive && "bg-gradient-to-r from-amber-500/10 to-orange-500/10 border border-amber-500/20",
                      isCompleted && "opacity-40",
                      !isActive && !isCompleted && "opacity-60"
                    )}
                  >
                    <div
                      className={cn(
                        "w-7 h-7 rounded-lg flex items-center justify-center transition-all duration-300",
                        isActive && "bg-amber-500/20",
                        isCompleted && "bg-amber-500/10",
                        !isActive && !isCompleted && "bg-white/5"
                      )}
                    >
                      {isCompleted ? (
                        <svg className="w-3.5 h-3.5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <Icon
                          className={cn(
                            "w-3.5 h-3.5",
                            isActive ? "text-amber-400" : "text-slate-500"
                          )}
                        />
                      )}
                    </div>
                    <span
                      className={cn(
                        "text-sm transition-all duration-300 flex-1",
                        isActive && "text-white font-medium",
                        isCompleted && "text-slate-500 line-through",
                        !isActive && !isCompleted && "text-slate-500"
                      )}
                    >
                      {step.label}
                    </span>
                    {isActive && (
                      <>
                        {/* Shimmer progress bar */}
                        <div className="flex-1 max-w-[80px] h-1 rounded-full overflow-hidden bg-white/5">
                          <div className="h-full w-full bg-gradient-to-r from-amber-500/50 via-amber-400 to-amber-500/50 animate-shimmer rounded-full" />
                        </div>
                        <div className="flex gap-0.5">
                          <span className="w-1 h-1 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                          <span className="w-1 h-1 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                          <span className="w-1 h-1 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Data source badges — glow indicators */}
          <div className="mt-5 grid grid-cols-3 sm:flex sm:flex-wrap sm:justify-center gap-2">
            {DATA_SOURCE_LABELS.map((source, index) => (
              <span
                key={source}
                className={cn(
                  "px-3 py-1.5 rounded-xl text-xs font-medium transition-all duration-500 text-center border",
                  index <= currentStep
                    ? "bg-amber-500/10 text-amber-400 border-amber-500/20 shadow-sm shadow-amber-500/10"
                    : "bg-white/3 text-slate-600 border-white/5"
                )}
              >
                {source}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
