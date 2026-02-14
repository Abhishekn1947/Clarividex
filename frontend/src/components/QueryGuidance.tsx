"use client";

import { AlertTriangle, Compass, X, Lightbulb, ArrowRight } from "lucide-react";
import { QueryAnalysisResult } from "@/lib/api";
import { cn } from "@/lib/utils";

interface QueryGuidanceProps {
  analysis: QueryAnalysisResult;
  originalQuery: string;
  onUseSuggestion: (query: string) => void;
  onProceedAnyway: () => void;
  onCancel: () => void;
}

export function QueryGuidance({
  analysis,
  originalQuery,
  onUseSuggestion,
  onProceedAnyway,
  onCancel,
}: QueryGuidanceProps) {
  const isVague = analysis.category === "vague";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-xl shadow-xl max-w-md w-full animate-scale-in">
        {/* Header */}
        <div className="flex items-start justify-between p-4 sm:p-5 border-b border-slate-200">
          <div className="flex items-start gap-3">
            <div
              className={cn(
                "w-10 h-10 rounded-lg flex items-center justify-center border",
                isVague
                  ? "text-amber-600 bg-amber-50 border-amber-200"
                  : "text-slate-600 bg-slate-100 border-slate-200"
              )}
            >
              {isVague ? (
                <AlertTriangle className="w-5 h-5" />
              ) : (
                <Compass className="w-5 h-5" />
              )}
            </div>
            <div>
              <h3 className="font-semibold text-slate-800">
                {isVague
                  ? "Your query could be more specific"
                  : "This doesn't look like a financial query"}
              </h3>
              <p className="text-sm text-slate-500 mt-0.5">
                {analysis.message}
              </p>
            </div>
          </div>
          <button
            onClick={onCancel}
            className="text-slate-400 hover:text-slate-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 sm:p-5 space-y-4">
          {/* Original query */}
          <div>
            <label className="text-xs text-slate-500 uppercase tracking-wide font-medium">
              Your Question
            </label>
            <p className="text-sm text-slate-700 mt-1 bg-slate-50 p-3 rounded-lg">
              {originalQuery}
            </p>
          </div>

          {/* Issues (vague only) */}
          {isVague && analysis.issues.length > 0 && (
            <div>
              <label className="text-xs text-slate-500 uppercase tracking-wide font-medium">
                Issues Detected
              </label>
              <ul className="mt-2 space-y-1.5">
                {analysis.issues.map((issue, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-amber-700 bg-amber-50 px-3 py-2 rounded-lg"
                  >
                    <Lightbulb className="w-4 h-4 mt-0.5 shrink-0" />
                    {issue}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Quality score (vague only) */}
          {isVague && (
            <div>
              <label className="text-xs text-slate-500 uppercase tracking-wide font-medium">
                Query Quality
              </label>
              <div className="mt-2 flex items-center gap-3">
                <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all",
                      analysis.quality_score >= 0.5
                        ? "bg-amber-400"
                        : "bg-red-400"
                    )}
                    style={{ width: `${analysis.quality_score * 100}%` }}
                  />
                </div>
                <span className="text-xs font-medium text-slate-600">
                  {Math.round(analysis.quality_score * 100)}%
                </span>
              </div>
            </div>
          )}

          {/* Suggestions */}
          {analysis.suggestions.length > 0 && (
            <div>
              <label className="text-xs text-slate-500 uppercase tracking-wide font-medium">
                {isVague ? "Try a more specific query" : "Try a financial query instead"}
              </label>
              <div className="mt-2 space-y-2">
                {analysis.suggestions.map((suggestion, i) => (
                  <button
                    key={i}
                    onClick={() => onUseSuggestion(suggestion)}
                    className="w-full flex items-center justify-between p-3 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 transition-all text-left group"
                  >
                    <span className="text-sm text-slate-700 pr-2">
                      {suggestion}
                    </span>
                    <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-slate-600 shrink-0 transition-colors" />
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-3 p-4 sm:p-5 border-t border-slate-200 bg-slate-50 rounded-b-xl">
          <button
            onClick={onCancel}
            className="flex-1 px-4 py-2.5 text-sm font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors"
          >
            Cancel
          </button>
          {isVague ? (
            <button
              onClick={onProceedAnyway}
              className="flex-1 px-4 py-2.5 text-sm font-medium text-white bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors"
            >
              Predict Anyway
            </button>
          ) : (
            <button
              onClick={onProceedAnyway}
              className="text-sm text-slate-500 hover:text-slate-700 transition-colors underline underline-offset-2"
            >
              Try Anyway
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
