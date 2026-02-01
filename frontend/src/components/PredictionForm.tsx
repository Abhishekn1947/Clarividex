"use client";

import { useState, useRef, useEffect } from "react";
import { Search, Loader2, Sparkles, ArrowRight, TrendingUp, HelpCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface PredictionFormProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
}

const EXAMPLE_QUERIES = [
  { text: "Will NVDA reach $150 by March 2026?", icon: TrendingUp },
  { text: "Will Tesla stock go up this month?", icon: TrendingUp },
  { text: "Will Apple beat earnings next quarter?", icon: TrendingUp },
  { text: "Will Boeing stock drop after recent news?", icon: TrendingUp },
];

const PLACEHOLDER_TEXTS = [
  "Ask Clarividex: Will NVDA reach $150 by March?",
  "Ask Clarividex: Will Tesla stock go up this month?",
  "Ask Clarividex: Should I buy Apple stock?",
  "Ask Clarividex: Will Boeing recover after the news?",
  "Ask Clarividex: What will happen to Microsoft stock?",
];

export function PredictionForm({ onSubmit, isLoading }: PredictionFormProps) {
  const [query, setQuery] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [placeholderIndex, setPlaceholderIndex] = useState(0);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Rotate placeholder text
  useEffect(() => {
    if (!isFocused && !query) {
      const interval = setInterval(() => {
        setPlaceholderIndex((prev) => (prev + 1) % PLACEHOLDER_TEXTS.length);
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [isFocused, query]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 120) + "px";
    }
  }, [query]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit(query.trim());
    }
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (query.trim() && !isLoading) {
        onSubmit(query.trim());
      }
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Main Search Form */}
      <form onSubmit={handleSubmit} className="relative">
        <div
          className={cn(
            "relative bg-white rounded-2xl transition-all duration-300",
            "border-2",
            isFocused
              ? "border-slate-400 shadow-lg shadow-slate-200/80 ring-4 ring-slate-100"
              : "border-slate-200 shadow-md shadow-slate-100/50 hover:border-slate-300 hover:shadow-lg"
          )}
        >
          {/* Icon */}
          <div className="absolute left-4 top-4">
            <Search className={cn(
              "w-5 h-5 transition-colors duration-200",
              isFocused ? "text-slate-500" : "text-slate-400"
            )} />
          </div>

          {/* Textarea Input */}
          <textarea
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            onKeyDown={handleKeyDown}
            placeholder={PLACEHOLDER_TEXTS[placeholderIndex]}
            rows={1}
            className={cn(
              "w-full pl-12 pr-4 py-4 text-base resize-none",
              "bg-transparent rounded-2xl",
              "focus:outline-none",
              "placeholder:text-slate-400 placeholder:transition-opacity",
              "text-slate-700",
              "min-h-[56px] max-h-[120px]"
            )}
            disabled={isLoading}
          />

          {/* Submit Button */}
          <div className="absolute right-3 bottom-3">
            <button
              type="submit"
              disabled={!query.trim() || isLoading}
              className={cn(
                "flex items-center gap-2 px-4 py-2.5",
                "rounded-xl font-medium text-sm",
                "transition-all duration-200 transform",
                query.trim() && !isLoading
                  ? "bg-slate-800 text-white hover:bg-slate-700 active:scale-95 shadow-md hover:shadow-lg"
                  : "bg-slate-100 text-slate-400 cursor-not-allowed"
              )}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Analyzing</span>
                  <span className="animate-pulse">...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Predict</span>
                  <ArrowRight className={cn(
                    "w-4 h-4 transition-transform duration-200",
                    query.trim() ? "translate-x-0 opacity-100" : "-translate-x-1 opacity-0"
                  )} />
                </>
              )}
            </button>
          </div>
        </div>

        {/* Helper text */}
        <div className="flex items-center justify-center gap-1 mt-2 text-xs text-slate-400">
          <HelpCircle className="w-3 h-3" />
          <span>Include a stock ticker (AAPL, TSLA) or company name for best results</span>
        </div>
      </form>

      {/* Example Queries */}
      <div className="mt-6">
        <div className="flex items-center justify-center gap-2 mb-3">
          <span className="text-xs text-slate-500 uppercase tracking-wide font-medium">
            Quick examples
          </span>
        </div>
        <div className="flex flex-wrap gap-2 justify-center">
          {EXAMPLE_QUERIES.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example.text)}
              disabled={isLoading}
              className={cn(
                "group flex items-center gap-2 px-4 py-2",
                "bg-white border border-slate-200 rounded-full",
                "text-sm text-slate-600",
                "hover:bg-slate-50 hover:border-slate-300 hover:text-slate-800",
                "transition-all duration-200",
                "disabled:opacity-50 disabled:cursor-not-allowed",
                "shadow-sm hover:shadow"
              )}
            >
              <example.icon className="w-3.5 h-3.5 text-slate-400 group-hover:text-amber-500 transition-colors" />
              <span className="line-clamp-1">{example.text}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Loading State Overlay */}
      {isLoading && (
        <div className="mt-6 text-center">
          <div className="inline-flex items-center gap-3 px-6 py-3 bg-slate-50 rounded-full border border-slate-200">
            <div className="flex gap-1">
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
            <span className="text-sm text-slate-600">
              Clarividex is reading the markets...
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
