"use client";

import { Brain, BarChart3, Newspaper, MessageSquare, TrendingUp, Database, Zap } from "lucide-react";
import { cn } from "@/lib/utils";
import { useEffect, useState, useMemo } from "react";
import type { SSEEvent } from "@/lib/api";

const LOADING_STEPS = [
  { icon: Database, label: "Fetching stock data...", color: "text-slate-600" },
  { icon: BarChart3, label: "Calculating technical indicators...", color: "text-slate-600" },
  { icon: Newspaper, label: "Analyzing news sentiment...", color: "text-slate-600" },
  { icon: MessageSquare, label: "Processing social media...", color: "text-slate-600" },
  { icon: TrendingUp, label: "Evaluating market conditions...", color: "text-slate-600" },
  { icon: Brain, label: "Clarividex is analyzing...", color: "text-slate-600" },
  { icon: Zap, label: "Generating prediction...", color: "text-slate-600" },
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

  useEffect(() => {
    // Only run timer if no SSE events are driving the UI
    if (sseStep >= 0) return;

    const interval = setInterval(() => {
      setTimerStep((prev) => (prev + 1) % LOADING_STEPS.length);
    }, 2000);

    return () => clearInterval(interval);
  }, [sseStep]);

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Main loading card */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        {/* Header skeleton */}
        <div className="bg-slate-800 p-3 sm:p-5">
          <div className="skeleton h-5 w-3/4 bg-slate-700 rounded mb-2" />
          <div className="skeleton h-3 w-1/2 bg-slate-700 rounded" />
        </div>

        {/* Main content */}
        <div className="p-4 sm:p-6">
          {/* Animated loading spinner */}
          <div className="flex flex-col items-center justify-center mb-6">
            <div className="relative w-20 h-20">
              {/* Outer ring */}
              <div className="absolute inset-0 rounded-full border-2 border-slate-200" />
              {/* Spinning ring */}
              <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-amber-500 animate-spin" />
              {/* Center icon */}
              <div className="absolute inset-0 flex items-center justify-center">
                <Brain className="w-6 h-6 text-amber-500 animate-pulse-subtle" />
              </div>
            </div>

            <div className="mt-4 text-center">
              <p className="text-sm font-medium text-slate-700 mb-1">
                Clarividex is Reading the Markets
              </p>
              <p className="text-slate-400 text-xs">
                {sseStep >= 0 ? "Receiving live analysis updates..." : "This usually takes 10-15 seconds"}
              </p>
            </div>
          </div>

          {/* Loading steps */}
          <div className="bg-slate-50 rounded-lg p-4">
            <div className="space-y-2">
              {LOADING_STEPS.map((step, index) => {
                const Icon = step.icon;
                const isActive = index === currentStep;
                const isCompleted = index < currentStep;

                return (
                  <div
                    key={index}
                    className={cn(
                      "flex items-center gap-3 p-2.5 rounded-lg transition-all duration-200",
                      isActive && "bg-white border border-slate-200",
                      isCompleted && "opacity-40"
                    )}
                  >
                    <div
                      className={cn(
                        "w-6 h-6 rounded-full flex items-center justify-center transition-all duration-200",
                        isActive && "bg-amber-100",
                        isCompleted && "bg-amber-50",
                        !isActive && !isCompleted && "bg-slate-100"
                      )}
                    >
                      {isCompleted ? (
                        <svg className="w-3 h-3 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <Icon
                          className={cn(
                            "w-3 h-3",
                            isActive ? "text-amber-600" : "text-slate-400"
                          )}
                        />
                      )}
                    </div>
                    <span
                      className={cn(
                        "text-xs transition-all duration-200",
                        isActive && "text-slate-700 font-medium",
                        isCompleted && "text-slate-400 line-through",
                        !isActive && !isCompleted && "text-slate-500"
                      )}
                    >
                      {step.label}
                    </span>
                    {isActive && (
                      <div className="ml-auto flex gap-0.5">
                        <span className="w-1 h-1 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-1 h-1 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-1 h-1 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Data sources being queried */}
          <div className="mt-4 grid grid-cols-3 sm:flex sm:flex-wrap sm:justify-center gap-1.5">
            {["Yahoo Finance", "SEC EDGAR", "News APIs", "Reddit", "Finviz", "VIX", "Options"].map((source, index) => (
              <span
                key={source}
                className={cn(
                  "px-2 py-1 rounded text-xs font-medium transition-all duration-300",
                  index <= currentStep
                    ? "bg-slate-100 text-slate-600"
                    : "bg-slate-50 text-slate-400"
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
