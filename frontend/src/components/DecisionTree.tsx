"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  Newspaper,
  BarChart3,
  LineChart,
  Users,
  MessageSquare,
  Clock,
  Target,
} from "lucide-react";
import { DecisionTrail, DecisionNode } from "@/lib/api";

interface DecisionTreeProps {
  trail: DecisionTrail;
  probability: number;
}

const categoryIcons: Record<string, React.ReactNode> = {
  technical: <Activity className="w-4 h-4" />,
  news: <Newspaper className="w-4 h-4" />,
  historical_news: <Clock className="w-4 h-4" />,
  options: <BarChart3 className="w-4 h-4" />,
  market: <LineChart className="w-4 h-4" />,
  analyst: <Target className="w-4 h-4" />,
  social: <MessageSquare className="w-4 h-4" />,
  historical_patterns: <Users className="w-4 h-4" />,
};

const categoryColors: Record<string, string> = {
  technical: "bg-blue-500/10 text-blue-400 border-blue-500/30",
  news: "bg-purple-500/10 text-purple-400 border-purple-500/30",
  historical_news: "bg-indigo-500/10 text-indigo-400 border-indigo-500/30",
  options: "bg-amber-500/10 text-amber-400 border-amber-500/30",
  market: "bg-cyan-500/10 text-cyan-400 border-cyan-500/30",
  analyst: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
  social: "bg-pink-500/10 text-pink-400 border-pink-500/30",
  historical_patterns: "bg-orange-500/10 text-orange-400 border-orange-500/30",
};

function SignalIcon({ signal }: { signal: string }) {
  if (signal === "bullish") {
    return <TrendingUp className="w-4 h-4 text-emerald-400" />;
  }
  if (signal === "bearish") {
    return <TrendingDown className="w-4 h-4 text-red-400" />;
  }
  return <Minus className="w-4 h-4 text-gray-400" />;
}

function CategorySection({
  category,
  nodes,
  categoryScore,
  categoryWeight,
}: {
  category: string;
  nodes: DecisionNode[];
  categoryScore: number;
  categoryWeight: number;
}) {
  const [isExpanded, setIsExpanded] = useState(true);
  const colorClass = categoryColors[category] || "bg-gray-500/10 text-gray-400 border-gray-500/30";
  const icon = categoryIcons[category] || <Activity className="w-4 h-4" />;

  const formattedCategory = category
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

  const absScore = Math.abs(categoryScore);
  const barWidth = Math.min(absScore / 15, 1) * 100;

  return (
    <div className="mb-3">
      {/* Category Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`w-full flex items-center justify-between p-2 sm:p-3 rounded-lg border ${colorClass} hover:opacity-80 transition-opacity`}
      >
        <div className="flex items-center gap-2">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
          {icon}
          <span className="font-medium">{formattedCategory}</span>
          <span className="text-xs opacity-70">({(categoryWeight * 100).toFixed(0)}% weight)</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`font-mono text-sm ${
              categoryScore > 0
                ? "text-emerald-400"
                : categoryScore < 0
                ? "text-red-400"
                : "text-gray-400"
            }`}
          >
            {categoryScore > 0 ? "+" : ""}
            {categoryScore.toFixed(2)}%
          </span>
        </div>
      </button>

      {/* Impact Bar */}
      <div className="mt-1 h-1 bg-gray-700/30 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${
            categoryScore > 0 ? "bg-emerald-500/60" : categoryScore < 0 ? "bg-red-500/60" : "bg-gray-500/30"
          }`}
          style={{ width: `${barWidth}%` }}
        />
      </div>

      {/* Expanded Nodes */}
      {isExpanded && nodes.length > 0 && (
        <div className="ml-4 mt-2 space-y-2 border-l-2 border-gray-700 pl-4">
          {nodes.map((node) => (
            <div
              key={node.id}
              className="bg-gray-800/50 rounded-lg p-2 sm:p-3 text-sm border border-gray-700/30 hover:border-gray-600/50 transition-colors"
            >
              <div className="flex items-start justify-between mb-1">
                <div className="flex items-center gap-2">
                  <SignalIcon signal={node.signal} />
                  <span className="font-medium text-gray-200">
                    {node.data_point}
                  </span>
                </div>
                <span
                  className={`font-mono text-xs px-2 py-0.5 rounded ${
                    node.score_contribution > 0
                      ? "bg-emerald-500/20 text-emerald-400"
                      : node.score_contribution < 0
                      ? "bg-red-500/20 text-red-400"
                      : "bg-gray-500/20 text-gray-400"
                  }`}
                >
                  {node.score_contribution > 0 ? "+" : ""}
                  {node.score_contribution.toFixed(2)}%
                </span>
              </div>
              {node.value && (
                <div className="text-xs mb-1">
                  <span className="bg-gray-700/30 rounded px-2 py-1 inline-block text-gray-300">{node.value}</span>
                </div>
              )}
              <div className="text-gray-500 text-xs line-clamp-2">{node.reasoning}</div>
              <div className="flex items-center gap-2 mt-1.5">
                <span className="text-xs bg-gray-700/30 px-2 py-0.5 rounded text-gray-400">{node.source}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function DecisionTree({ trail, probability }: DecisionTreeProps) {
  // Group nodes by category
  const nodesByCategory: Record<string, DecisionNode[]> = {};
  trail.nodes.forEach((node) => {
    if (!nodesByCategory[node.category]) {
      nodesByCategory[node.category] = [];
    }
    nodesByCategory[node.category].push(node);
  });

  // Sort categories by weight
  const sortedCategories = Object.keys(nodesByCategory).sort(
    (a, b) => (trail.category_weights[b] || 0) - (trail.category_weights[a] || 0)
  );

  // Calculate total positive and negative contributions
  const totalPositive = Object.values(trail.category_scores)
    .filter((score) => score > 0)
    .reduce((sum, score) => sum + score, 0);
  const totalNegative = Object.values(trail.category_scores)
    .filter((score) => score < 0)
    .reduce((sum, score) => sum + score, 0);

  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-700/50 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 px-4 sm:px-6 py-3 sm:py-4 border-b border-gray-700/50">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <Activity className="w-5 h-5 text-blue-400" />
          Decision Trail
        </h3>
        <p className="text-sm text-gray-400 mt-1">
          See exactly how each factor contributed to the prediction
        </p>
      </div>

      {/* Summary Bar */}
      <div className="px-4 sm:px-6 py-3 sm:py-4 bg-gray-800/30 border-b border-gray-700/50">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-400">Base Probability</span>
          <span className="text-sm font-mono text-gray-300">50%</span>
        </div>

        {/* Contribution Bar */}
        <div className="flex items-center gap-3 mb-1.5">
          <span className="text-xs font-mono text-red-400 w-14 text-right shrink-0">
            {totalNegative.toFixed(1)}%
          </span>
          <div className="flex h-3 rounded-lg overflow-hidden bg-gray-700/50 flex-1">
            <div
              className="bg-red-500/60 transition-all"
              style={{ width: `${(Math.abs(totalNegative) / (Math.abs(totalNegative) + totalPositive + 0.01)) * 100}%` }}
            />
            <div
              className="bg-emerald-500/60 transition-all"
              style={{ width: `${(totalPositive / (Math.abs(totalNegative) + totalPositive + 0.01)) * 100}%` }}
            />
          </div>
          <span className="text-xs font-mono text-emerald-400 w-14 shrink-0">
            +{totalPositive.toFixed(1)}%
          </span>
        </div>

        <div className="flex items-center justify-between text-xs text-gray-500 px-[4.25rem]">
          <span>Bearish</span>
          <span>Bullish</span>
        </div>
      </div>

      {/* Category Breakdown */}
      <div className="p-4 sm:p-6">
        {sortedCategories.map((category) => (
          <CategorySection
            key={category}
            category={category}
            nodes={nodesByCategory[category]}
            categoryScore={trail.category_scores[category] || 0}
            categoryWeight={trail.category_weights[category] || 0}
          />
        ))}
      </div>

      {/* Final Calculation */}
      {trail.final_calculation && (
        <div className="px-4 sm:px-6 py-3 sm:py-4 bg-gray-800/50 border-t border-gray-700/50">
          <div className="text-sm text-gray-400 mb-2">Calculation Flow:</div>
          <div className="bg-gray-900/70 rounded-lg p-3 font-mono text-xs text-gray-300 overflow-x-auto whitespace-pre-wrap">
            {trail.final_calculation}
          </div>
        </div>
      )}

      {/* Final Probability */}
      <div className="px-4 sm:px-6 py-3 sm:py-4 bg-gradient-to-r from-gray-800/80 to-gray-800/50 border-t border-gray-700/50">
        <div className="flex items-center justify-between">
          <span className="text-gray-400 font-medium">Final Probability</span>
          <span
            className={`text-2xl font-bold ${
              probability >= 0.6
                ? "text-emerald-400"
                : probability >= 0.4
                ? "text-amber-400"
                : "text-red-400"
            }`}
          >
            {(probability * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );
}
