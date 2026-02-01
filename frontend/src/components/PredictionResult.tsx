"use client";

import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  ExternalLink,
  Clock,
  BarChart3,
  MessageSquare,
  Newspaper,
  Target,
  Shield,
  ChevronDown,
  ChevronUp,
  Activity,
  Database,
  Zap,
  Info,
  CheckCircle,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { useState } from "react";
import { PredictionResponse } from "@/lib/api";
import {
  cn,
  formatCurrency,
  formatPercent,
  formatRelativeTime,
  getConfidenceLabel,
  getSentimentLabel,
} from "@/lib/utils";
import { ProbabilityGauge } from "./ProbabilityGauge";
import { InfoTooltip } from "./InfoTooltip";
import DecisionTreeModal from "./DecisionTreeModal";
import FloatingChatbot from "./FloatingChatbot";

interface PredictionResultProps {
  prediction: PredictionResponse;
}

export function PredictionResult({ prediction }: PredictionResultProps) {
  const [showAllNews, setShowAllNews] = useState(false);
  const [showAllFactors, setShowAllFactors] = useState(false);

  const probabilityPercent = Math.round(prediction.probability * 100);
  const confidence = getConfidenceLabel(prediction.confidence_level);
  const sentiment = getSentimentLabel(prediction.sentiment);
  const isBullish = prediction.probability >= 0.5;

  return (
    <div className="w-full max-w-3xl mx-auto space-y-4 animate-fade-in">
      {/* Main Prediction Card */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        {/* Header */}
        <div className="bg-slate-800 p-5 text-white">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className="text-xs text-slate-400 mb-1">Prediction</p>
              <h2 className="text-lg font-medium">{prediction.query}</h2>
              {prediction.ticker && (
                <div className="flex flex-wrap items-center gap-3 mt-2 text-sm text-slate-300">
                  <span className="font-mono bg-slate-700 px-2 py-0.5 rounded text-xs">
                    {prediction.ticker}
                  </span>
                  {prediction.current_price && (
                    <span>Current: {formatCurrency(prediction.current_price)}</span>
                  )}
                  {prediction.target_price && (
                    <span>Target: {formatCurrency(prediction.target_price)}</span>
                  )}
                </div>
              )}
            </div>
            <div className="flex flex-col items-end gap-2">
              <span className="text-xs text-slate-400">
                {formatRelativeTime(prediction.created_at)}
              </span>
              <div className={cn(
                "flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium",
                isBullish ? "bg-emerald-600/20 text-emerald-300" : "bg-red-600/20 text-red-300"
              )}>
                {isBullish ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                {isBullish ? "Bullish" : "Bearish"}
              </div>
            </div>
          </div>
        </div>

        {/* Probability Display */}
        <div className="p-6">
          <div className="flex flex-col md:flex-row items-center justify-center gap-6">
            {/* Probability Gauge */}
            <ProbabilityGauge
              probability={prediction.probability}
              size="lg"
              showLabel={true}
              animated={true}
            />

            {/* Key Metrics Grid */}
            <div className="flex-1 grid grid-cols-2 gap-3 w-full max-w-xs">
              <div className="metric-card">
                <div className="text-xs text-slate-500 mb-1 flex items-center gap-1">
                  Confidence
                  <InfoTooltip term="confidence" />
                </div>
                <div className={cn("text-sm font-medium", confidence.color)}>
                  {confidence.label}
                </div>
              </div>

              <div className="metric-card">
                <div className="text-xs text-slate-500 mb-1 flex items-center gap-1">
                  Sentiment
                  <InfoTooltip term="bullish" />
                </div>
                <div className={cn("text-sm font-medium flex items-center gap-1", sentiment.color)}>
                  <span>{sentiment.emoji}</span>
                  <span>{sentiment.label}</span>
                </div>
              </div>

              <div className="metric-card">
                <div className="text-xs text-slate-500 mb-1 flex items-center gap-1">
                  Data Quality
                  <InfoTooltip term="dataquality" />
                  {prediction.has_limited_data && (
                    <AlertCircle className="w-3 h-3 text-amber-500" />
                  )}
                </div>
                <div className={cn(
                  "text-lg font-semibold",
                  prediction.data_quality_score >= 0.8 ? "text-slate-700" :
                  prediction.data_quality_score >= 0.6 ? "text-amber-600" :
                  "text-red-600"
                )}>
                  {Math.round(prediction.data_quality_score * 100)}%
                </div>
              </div>

              <div className="metric-card">
                <div className="text-xs text-slate-500 mb-1 flex items-center gap-1">
                  Probability
                  <InfoTooltip term="probability" />
                </div>
                <div className="text-lg font-semibold text-slate-700">
                  {prediction.data_points_analyzed.toLocaleString()}
                </div>
              </div>
            </div>
          </div>

          {/* Price Gap */}
          {prediction.price_gap_percent !== null && (
            <div className={cn(
              "mt-4 p-3 rounded-lg border",
              prediction.price_gap_percent >= 0
                ? "bg-emerald-50 border-emerald-200"
                : "bg-red-50 border-red-200"
            )}>
              <div className="flex items-center justify-between">
                <span className={prediction.price_gap_percent >= 0 ? "text-emerald-700" : "text-red-700"}>
                  Gap to Target
                </span>
                <span className={cn(
                  "text-lg font-semibold",
                  prediction.price_gap_percent >= 0 ? "text-emerald-700" : "text-red-700"
                )}>
                  {formatPercent(prediction.price_gap_percent)}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Data Limitations Warning */}
      {prediction.has_limited_data && prediction.data_limitations && prediction.data_limitations.length > 0 && (
        <div className="bg-amber-50 border border-amber-300 rounded-xl p-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="font-semibold text-amber-800 text-sm mb-2">
                Limited Data Available
              </h3>
              <p className="text-xs text-amber-700 mb-3">
                This prediction is based on incomplete data. Results should be interpreted with caution.
              </p>
              <div className="space-y-2">
                {prediction.data_limitations
                  .filter((l) => l.severity === "high" || l.severity === "medium")
                  .slice(0, 4)
                  .map((limitation, index) => (
                    <div
                      key={index}
                      className={cn(
                        "p-2.5 rounded-lg text-xs",
                        limitation.severity === "high"
                          ? "bg-red-100 border border-red-200"
                          : "bg-amber-100 border border-amber-200"
                      )}
                    >
                      <div className="flex items-start gap-2">
                        <span className={cn(
                          "px-1.5 py-0.5 rounded text-xs font-medium uppercase",
                          limitation.severity === "high"
                            ? "bg-red-200 text-red-700"
                            : "bg-amber-200 text-amber-700"
                        )}>
                          {limitation.severity}
                        </span>
                        <div className="flex-1">
                          <p className={limitation.severity === "high" ? "text-red-800" : "text-amber-800"}>
                            {limitation.message}
                          </p>
                          {limitation.recommendation && (
                            <p className={cn(
                              "mt-1 italic",
                              limitation.severity === "high" ? "text-red-600" : "text-amber-600"
                            )}>
                              {limitation.recommendation}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
              {prediction.data_limitations.filter((l) => l.severity === "high" || l.severity === "medium").length > 4 && (
                <p className="text-xs text-amber-600 mt-2">
                  +{prediction.data_limitations.filter((l) => l.severity === "high" || l.severity === "medium").length - 4} more limitations
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* AI Summary */}
      <div className="card">
        <h3 className="heading-3 mb-3 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-slate-400" />
          Analysis Summary
        </h3>
        <p className="text-sm text-slate-600 leading-relaxed">
          {prediction.reasoning.summary}
        </p>
      </div>

      {/* Decision Tree Button - Opens modal with full transparency */}
      {prediction.reasoning.decision_trail && prediction.reasoning.decision_trail.nodes.length > 0 && (
        <DecisionTreeModal
          trail={prediction.reasoning.decision_trail}
          probability={prediction.probability}
        />
      )}

      {/* Factors Grid */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Bullish Factors */}
        <div className="card">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-slate-700">
            <TrendingUp className="w-4 h-4 text-emerald-600" />
            Bullish Factors
            <span className="ml-auto text-xs font-normal text-slate-400">
              {prediction.reasoning.bullish_factors.length}
            </span>
          </h3>
          <div className="space-y-2">
            {prediction.reasoning.bullish_factors
              .slice(0, showAllFactors ? undefined : 3)
              .map((factor, index) => (
                <div key={index} className="factor-card bullish">
                  <div className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <p className="text-slate-700 text-xs leading-relaxed">{factor.description}</p>
                      <div className="flex items-center gap-2 mt-1.5">
                        <span className="text-xs text-slate-400">{factor.source}</span>
                        <span className="text-xs text-slate-300">|</span>
                        <span className="text-xs text-slate-400">
                          {(factor.weight * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            {prediction.reasoning.bullish_factors.length > 3 && (
              <button
                onClick={() => setShowAllFactors(!showAllFactors)}
                className="w-full text-xs text-slate-500 hover:text-slate-700 flex items-center justify-center gap-1 py-1.5"
              >
                {showAllFactors ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                {showAllFactors ? "Less" : `Show all ${prediction.reasoning.bullish_factors.length}`}
              </button>
            )}
          </div>
        </div>

        {/* Bearish Factors */}
        <div className="card">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-slate-700">
            <TrendingDown className="w-4 h-4 text-red-500" />
            Bearish Factors
            <span className="ml-auto text-xs font-normal text-slate-400">
              {prediction.reasoning.bearish_factors.length}
            </span>
          </h3>
          <div className="space-y-2">
            {prediction.reasoning.bearish_factors
              .slice(0, showAllFactors ? undefined : 3)
              .map((factor, index) => (
                <div key={index} className="factor-card bearish">
                  <div className="flex items-start gap-2">
                    <XCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <p className="text-slate-700 text-xs leading-relaxed">{factor.description}</p>
                      <div className="flex items-center gap-2 mt-1.5">
                        <span className="text-xs text-slate-400">{factor.source}</span>
                        <span className="text-xs text-slate-300">|</span>
                        <span className="text-xs text-slate-400">
                          {Math.abs(factor.weight * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            {prediction.reasoning.bearish_factors.length === 0 && (
              <div className="text-slate-400 text-xs italic p-3 bg-slate-50 rounded-lg">
                No significant bearish factors identified.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Catalysts */}
      {prediction.reasoning.catalysts.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-slate-700">
            <Target className="w-4 h-4 text-slate-400" />
            Upcoming Catalysts
          </h3>
          <div className="space-y-2">
            {prediction.reasoning.catalysts.map((catalyst, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                <div className="flex items-center gap-2">
                  <div className={cn(
                    "w-6 h-6 rounded-full flex items-center justify-center",
                    catalyst.potential_impact === "positive" ? "bg-emerald-100" :
                    catalyst.potential_impact === "negative" ? "bg-red-100" : "bg-amber-100"
                  )}>
                    {catalyst.potential_impact === "positive" ? (
                      <TrendingUp className="w-3 h-3 text-emerald-600" />
                    ) : catalyst.potential_impact === "negative" ? (
                      <TrendingDown className="w-3 h-3 text-red-600" />
                    ) : (
                      <Activity className="w-3 h-3 text-amber-600" />
                    )}
                  </div>
                  <div>
                    <p className="font-medium text-slate-700 text-xs">{catalyst.event}</p>
                    {catalyst.date && (
                      <p className="text-xs text-slate-400 flex items-center gap-1">
                        <Clock className="w-2.5 h-2.5" />
                        {catalyst.date}
                      </p>
                    )}
                  </div>
                </div>
                <span className={cn(
                  "px-2 py-0.5 rounded text-xs font-medium capitalize",
                  catalyst.potential_impact === "positive" ? "bg-emerald-100 text-emerald-700" :
                  catalyst.potential_impact === "negative" ? "bg-red-100 text-red-700" :
                  "bg-amber-100 text-amber-700"
                )}>
                  {catalyst.potential_impact}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risks & Assumptions */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="card">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-slate-700">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            Key Risks
          </h3>
          <ul className="space-y-2">
            {prediction.reasoning.risks.map((risk, index) => (
              <li key={index} className="flex items-start gap-2 text-xs text-slate-600">
                <span className="w-4 h-4 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">
                  {index + 1}
                </span>
                {risk}
              </li>
            ))}
          </ul>
        </div>

        <div className="card">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-slate-700">
            <Shield className="w-4 h-4 text-slate-400" />
            Assumptions
          </h3>
          <ul className="space-y-2">
            {prediction.reasoning.assumptions.map((assumption, index) => (
              <li key={index} className="flex items-start gap-2 text-xs text-slate-600">
                <span className="w-4 h-4 rounded-full bg-slate-100 text-slate-600 flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">
                  {index + 1}
                </span>
                {assumption}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* News Articles */}
      {prediction.news_articles.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-slate-700">
            <Newspaper className="w-4 h-4 text-slate-400" />
            Recent News
            <span className="ml-auto text-xs font-normal text-slate-400">
              {prediction.news_articles.length} articles
            </span>
          </h3>
          <div className="space-y-2">
            {prediction.news_articles
              .slice(0, showAllNews ? undefined : 10)
              .map((article, index) => (
                <div key={index} className="flex items-start gap-3 p-2.5 rounded-lg bg-slate-50 hover:bg-slate-100 transition-colors">
                  <div className={cn(
                    "w-6 h-6 rounded flex items-center justify-center flex-shrink-0",
                    article.sentiment_score > 0.1 ? "bg-emerald-100" :
                    article.sentiment_score < -0.1 ? "bg-red-100" : "bg-slate-100"
                  )}>
                    {article.sentiment_score > 0.1 ? (
                      <TrendingUp className="w-3 h-3 text-emerald-600" />
                    ) : article.sentiment_score < -0.1 ? (
                      <TrendingDown className="w-3 h-3 text-red-600" />
                    ) : (
                      <span className="text-slate-400 text-xs">-</span>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-slate-700 line-clamp-2">{article.title}</p>
                    <div className="flex items-center gap-2 mt-1 text-xs text-slate-400">
                      <span>{article.source}</span>
                      <span>{formatRelativeTime(article.published_at)}</span>
                    </div>
                  </div>
                  {article.url && (
                    <a
                      href={article.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="w-6 h-6 rounded bg-slate-100 hover:bg-slate-200 flex items-center justify-center text-slate-400 hover:text-slate-600 transition-colors flex-shrink-0"
                    >
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  )}
                </div>
              ))}
            {prediction.news_articles.length > 10 && (
              <button
                onClick={() => setShowAllNews(!showAllNews)}
                className="w-full text-xs text-slate-500 hover:text-slate-700 flex items-center justify-center gap-1 py-1.5"
              >
                {showAllNews ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                {showAllNews ? "Less" : `Show all ${prediction.news_articles.length}`}
              </button>
            )}
          </div>
        </div>
      )}

      {/* Technical Indicators */}
      {prediction.technicals && (
        <div className="card">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-slate-700">
            <BarChart3 className="w-4 h-4 text-slate-400" />
            Technical Indicators
            <span className="ml-auto text-xs font-normal text-slate-400">
              Click (i) for explanations
            </span>
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {/* RSI */}
            {prediction.technicals.rsi_14 !== null && (
              <div className="text-center p-3 rounded-lg bg-slate-50 border border-slate-100">
                <div className="text-xs text-slate-500 mb-1 flex items-center justify-center gap-1">
                  RSI (14)
                  <InfoTooltip term="rsi" />
                </div>
                <div className={cn(
                  "text-xl font-semibold",
                  prediction.technicals.rsi_14 < 30 ? "text-emerald-600" :
                  prediction.technicals.rsi_14 > 70 ? "text-red-600" : "text-slate-700"
                )}>
                  {prediction.technicals.rsi_14.toFixed(1)}
                </div>
                <div className={cn(
                  "text-xs mt-1 px-2 py-0.5 rounded-full inline-block",
                  prediction.technicals.rsi_14 < 30 ? "bg-emerald-100 text-emerald-700" :
                  prediction.technicals.rsi_14 > 70 ? "bg-red-100 text-red-700" : "bg-slate-100 text-slate-600"
                )}>
                  {prediction.technicals.rsi_14 < 30 ? "Oversold" :
                   prediction.technicals.rsi_14 > 70 ? "Overbought" : "Neutral"}
                </div>
              </div>
            )}

            {/* MACD */}
            {prediction.technicals.macd !== null && (
              <div className="text-center p-3 rounded-lg bg-slate-50 border border-slate-100">
                <div className="text-xs text-slate-500 mb-1 flex items-center justify-center gap-1">
                  MACD
                  <InfoTooltip term="macd" />
                </div>
                <div className={cn(
                  "text-xl font-semibold",
                  prediction.technicals.macd > 0 ? "text-emerald-600" :
                  prediction.technicals.macd < 0 ? "text-red-600" : "text-slate-700"
                )}>
                  {prediction.technicals.macd.toFixed(2)}
                </div>
                <div className={cn(
                  "text-xs mt-1 px-2 py-0.5 rounded-full inline-block",
                  prediction.technicals.macd > 0 ? "bg-emerald-100 text-emerald-700" :
                  prediction.technicals.macd < 0 ? "bg-red-100 text-red-700" : "bg-slate-100 text-slate-600"
                )}>
                  {prediction.technicals.macd > 0 ? "Bullish" :
                   prediction.technicals.macd < 0 ? "Bearish" : "Neutral"}
                </div>
              </div>
            )}

            {/* SMA 20 */}
            {prediction.technicals.sma_20 !== null && (
              <div className="text-center p-3 rounded-lg bg-slate-50 border border-slate-100">
                <div className="text-xs text-slate-500 mb-1 flex items-center justify-center gap-1">
                  SMA 20
                  <InfoTooltip term="sma" />
                </div>
                <div className="text-xl font-semibold text-slate-700">
                  {formatCurrency(prediction.technicals.sma_20)}
                </div>
                <div className="text-xs text-slate-400 mt-1">20-day avg</div>
              </div>
            )}

            {/* SMA 50 */}
            {prediction.technicals.sma_50 !== null && (
              <div className="text-center p-3 rounded-lg bg-slate-50 border border-slate-100">
                <div className="text-xs text-slate-500 mb-1 flex items-center justify-center gap-1">
                  SMA 50
                  <InfoTooltip term="sma" />
                </div>
                <div className="text-xl font-semibold text-slate-700">
                  {formatCurrency(prediction.technicals.sma_50)}
                </div>
                <div className="text-xs text-slate-400 mt-1">50-day avg</div>
              </div>
            )}

            {/* SMA 200 */}
            {prediction.technicals.sma_200 !== null && (
              <div className="text-center p-3 rounded-lg bg-slate-50 border border-slate-100">
                <div className="text-xs text-slate-500 mb-1 flex items-center justify-center gap-1">
                  SMA 200
                  <InfoTooltip term="sma" />
                </div>
                <div className="text-xl font-semibold text-slate-700">
                  {formatCurrency(prediction.technicals.sma_200)}
                </div>
                <div className="text-xs text-slate-400 mt-1">200-day avg</div>
              </div>
            )}

            {/* Support */}
            {prediction.technicals.support_level !== null && (
              <div className="text-center p-3 rounded-lg bg-emerald-50 border border-emerald-100">
                <div className="text-xs text-emerald-600 mb-1 flex items-center justify-center gap-1">
                  Support Level
                  <InfoTooltip term="support" />
                </div>
                <div className="text-xl font-semibold text-emerald-700">
                  {formatCurrency(prediction.technicals.support_level)}
                </div>
                <div className="text-xs text-emerald-500 mt-1">Price floor</div>
              </div>
            )}

            {/* Resistance */}
            {prediction.technicals.resistance_level !== null && (
              <div className="text-center p-3 rounded-lg bg-red-50 border border-red-100">
                <div className="text-xs text-red-600 mb-1 flex items-center justify-center gap-1">
                  Resistance Level
                  <InfoTooltip term="resistance" />
                </div>
                <div className="text-xl font-semibold text-red-700">
                  {formatCurrency(prediction.technicals.resistance_level)}
                </div>
                <div className="text-xs text-red-500 mt-1">Price ceiling</div>
              </div>
            )}
          </div>

          {/* Legend */}
          <div className="mt-4 pt-3 border-t border-slate-100">
            <div className="flex flex-wrap gap-4 text-xs text-slate-500">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                <span>Bullish signal</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-red-500"></span>
                <span>Bearish signal</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-slate-400"></span>
                <span>Neutral</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Data Sources */}
      <div className="card">
        <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-slate-700">
          <Database className="w-4 h-4 text-slate-400" />
          Data Sources
          <span className="ml-auto text-xs font-normal text-slate-400">
            {prediction.sources_used.length} sources
          </span>
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {prediction.sources_used.map((source, index) => (
            <span
              key={index}
              className={cn(
                "source-badge text-xs",
                source.reliability_score >= 0.9 && "high-reliability",
                source.reliability_score >= 0.8 && source.reliability_score < 0.9 && "medium-reliability"
              )}
            >
              {source.name}
              <span className="opacity-60 ml-1">({(source.reliability_score * 100).toFixed(0)}%)</span>
            </span>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
        <div className="flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-amber-800 text-xs mb-0.5">Disclaimer</p>
            <p className="text-xs text-amber-700">{prediction.disclaimer}</p>
          </div>
        </div>
      </div>

      {/* Floating Chatbot */}
      <FloatingChatbot prediction={prediction} />
    </div>
  );
}
