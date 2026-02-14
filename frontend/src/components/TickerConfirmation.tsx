"use client";

import { useState } from "react";
import { AlertCircle, CheckCircle, HelpCircle, X } from "lucide-react";
import { TickerExtractionResult } from "@/lib/api";
import { cn } from "@/lib/utils";

interface TickerConfirmationProps {
  result: TickerExtractionResult;
  originalQuery: string;
  onConfirm: (ticker: string, query: string) => void;
  onCancel: () => void;
}

export function TickerConfirmation({
  result,
  originalQuery,
  onConfirm,
  onCancel,
}: TickerConfirmationProps) {
  const [selectedTicker, setSelectedTicker] = useState<string>(
    result.ticker || (result.suggestions[0]?.ticker ?? "")
  );
  const [customTicker, setCustomTicker] = useState("");
  const [useCustom, setUseCustom] = useState(false);

  const handleConfirm = () => {
    const ticker = useCustom ? customTicker.toUpperCase() : selectedTicker;
    if (ticker) {
      // Replace any detected ticker in the query with the confirmed one
      // or append it if not present
      let modifiedQuery = originalQuery;
      if (result.ticker && result.ticker !== ticker) {
        modifiedQuery = originalQuery.replace(
          new RegExp(`\\b${result.ticker}\\b`, "gi"),
          ticker
        );
      }
      onConfirm(ticker, modifiedQuery);
    }
  };

  const getSeverityColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-emerald-600 bg-emerald-50 border-emerald-200";
    if (confidence >= 0.6) return "text-amber-600 bg-amber-50 border-amber-200";
    return "text-red-600 bg-red-50 border-red-200";
  };

  const getSeverityIcon = (confidence: number) => {
    if (confidence >= 0.8) return <CheckCircle className="w-5 h-5" />;
    if (confidence >= 0.6) return <HelpCircle className="w-5 h-5" />;
    return <AlertCircle className="w-5 h-5" />;
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-xl shadow-xl max-w-md w-full animate-scale-in">
        {/* Header */}
        <div className="flex items-start justify-between p-4 sm:p-5 border-b border-slate-200">
          <div className="flex items-start gap-3">
            <div className={cn(
              "w-10 h-10 rounded-lg flex items-center justify-center border",
              getSeverityColor(result.confidence)
            )}>
              {getSeverityIcon(result.confidence)}
            </div>
            <div>
              <h3 className="font-semibold text-slate-800">Confirm Stock Ticker</h3>
              <p className="text-sm text-slate-500 mt-0.5">
                {result.message || "Please confirm the ticker symbol"}
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

          {/* Suggestions */}
          {result.suggestions.length > 0 && (
            <div>
              <label className="text-xs text-slate-500 uppercase tracking-wide font-medium">
                Suggested Tickers
              </label>
              <div className="mt-2 space-y-2">
                {result.suggestions.map((suggestion) => (
                  <button
                    key={suggestion.ticker}
                    onClick={() => {
                      setSelectedTicker(suggestion.ticker);
                      setUseCustom(false);
                    }}
                    className={cn(
                      "w-full flex items-center justify-between p-3 rounded-lg border transition-all text-left",
                      selectedTicker === suggestion.ticker && !useCustom
                        ? "border-slate-800 bg-slate-50 ring-1 ring-slate-800"
                        : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                    )}
                  >
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        "w-5 h-5 rounded-full border-2 flex items-center justify-center",
                        selectedTicker === suggestion.ticker && !useCustom
                          ? "border-slate-800"
                          : "border-slate-300"
                      )}>
                        {selectedTicker === suggestion.ticker && !useCustom && (
                          <div className="w-2.5 h-2.5 bg-slate-800 rounded-full" />
                        )}
                      </div>
                      <div>
                        <span className="font-semibold text-slate-800">
                          {suggestion.ticker}
                        </span>
                        <span className="text-slate-500 ml-2 text-sm truncate">
                          {suggestion.company_name}
                        </span>
                      </div>
                    </div>
                    <span className={cn(
                      "text-xs px-2 py-1 rounded-full",
                      suggestion.confidence >= 0.8
                        ? "bg-emerald-100 text-emerald-700"
                        : suggestion.confidence >= 0.6
                        ? "bg-amber-100 text-amber-700"
                        : "bg-slate-100 text-slate-600"
                    )}>
                      {Math.round(suggestion.confidence * 100)}%
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Primary suggested ticker (if no suggestions array but has ticker) */}
          {result.suggestions.length === 0 && result.ticker && (
            <div>
              <label className="text-xs text-slate-500 uppercase tracking-wide font-medium">
                Detected Ticker
              </label>
              <button
                onClick={() => {
                  setSelectedTicker(result.ticker!);
                  setUseCustom(false);
                }}
                className={cn(
                  "w-full mt-2 flex items-center justify-between p-3 rounded-lg border transition-all text-left",
                  selectedTicker === result.ticker && !useCustom
                    ? "border-slate-800 bg-slate-50 ring-1 ring-slate-800"
                    : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                )}
              >
                <div className="flex items-center gap-3">
                  <div className={cn(
                    "w-5 h-5 rounded-full border-2 flex items-center justify-center",
                    selectedTicker === result.ticker && !useCustom
                      ? "border-slate-800"
                      : "border-slate-300"
                  )}>
                    {selectedTicker === result.ticker && !useCustom && (
                      <div className="w-2.5 h-2.5 bg-slate-800 rounded-full" />
                    )}
                  </div>
                  <div>
                    <span className="font-semibold text-slate-800">
                      {result.ticker}
                    </span>
                    {result.company_name && (
                      <span className="text-slate-500 ml-2 text-sm">
                        {result.company_name}
                      </span>
                    )}
                  </div>
                </div>
              </button>
            </div>
          )}

          {/* Custom ticker input */}
          <div>
            <button
              onClick={() => setUseCustom(!useCustom)}
              className="text-sm text-slate-600 hover:text-slate-800 transition-colors flex items-center gap-1.5"
            >
              <span className={cn(
                "w-4 h-4 rounded border flex items-center justify-center text-xs",
                useCustom ? "bg-slate-800 border-slate-800 text-white" : "border-slate-300"
              )}>
                {useCustom && "✓"}
              </span>
              Enter a different ticker
            </button>

            {useCustom && (
              <div className="mt-2">
                <input
                  type="text"
                  value={customTicker}
                  onChange={(e) => setCustomTicker(e.target.value.toUpperCase())}
                  placeholder="e.g., AAPL, TSLA, NVDA"
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-slate-400 focus:border-slate-400 uppercase"
                  maxLength={5}
                />
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex gap-3 p-4 sm:p-5 border-t border-slate-200 bg-slate-50 rounded-b-xl">
          <button
            onClick={onCancel}
            className="flex-1 px-4 py-2.5 text-sm font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={useCustom ? !customTicker : !selectedTicker}
            className="flex-1 px-4 py-2.5 text-sm font-medium text-white bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Confirm & Predict
          </button>
        </div>
      </div>
    </div>
  );
}
