"use client";

import { useState, useEffect } from "react";
import { X, GitBranch, TrendingUp, TrendingDown, Minus, ChevronDown } from "lucide-react";
import { DecisionTrail, DecisionNode } from "@/lib/api";
import { cn } from "@/lib/utils";

interface DecisionTreeModalProps {
  trail: DecisionTrail;
  probability: number;
}

function Branch({
  category,
  nodes,
  score,
}: {
  category: string;
  nodes: DecisionNode[];
  score: number;
  weight: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const label = category.split("_").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
  const isPositive = score > 0;
  const isNegative = score < 0;

  return (
    <div className="relative">
      <div className={cn(
        "absolute left-0 top-0 w-6 h-4 border-l-2 border-b-2 rounded-bl-lg",
        isPositive ? "border-emerald-300" : isNegative ? "border-red-300" : "border-slate-300"
      )} />

      <div className="ml-6 pt-0.5">
        <button
          onClick={() => setExpanded(!expanded)}
          className={cn(
            "w-full flex items-center gap-2 p-2 rounded-lg border transition-all text-left",
            isPositive ? "bg-emerald-50 border-emerald-200 hover:bg-emerald-100" :
            isNegative ? "bg-red-50 border-red-200 hover:bg-red-100" :
            "bg-slate-50 border-slate-200 hover:bg-slate-100"
          )}
        >
          <div className={cn(
            "w-5 h-5 rounded-full flex items-center justify-center shrink-0",
            isPositive ? "bg-emerald-200" : isNegative ? "bg-red-200" : "bg-slate-200"
          )}>
            {isPositive ? <TrendingUp className="w-3 h-3 text-emerald-700" /> :
             isNegative ? <TrendingDown className="w-3 h-3 text-red-700" /> :
             <Minus className="w-3 h-3 text-slate-600" />}
          </div>

          <div className="flex-1 min-w-0">
            <span className="font-medium text-sm text-slate-800">{label}</span>
            <span className="text-xs text-slate-500 ml-2">({nodes.length})</span>
          </div>

          <span className={cn(
            "font-mono text-sm font-semibold",
            isPositive ? "text-emerald-600" : isNegative ? "text-red-600" : "text-slate-500"
          )}>
            {score > 0 ? "+" : ""}{score.toFixed(1)}%
          </span>

          <ChevronDown className={cn(
            "w-4 h-4 text-slate-400 transition-transform shrink-0",
            expanded && "rotate-180"
          )} />
        </button>

        {expanded && nodes.length > 0 && (
          <div className="mt-1.5 space-y-1 ml-2">
            {nodes.map((node) => (
              <div key={node.id} className="relative">
                <div className={cn(
                  "absolute left-0 top-0 w-3 h-2.5 border-l border-b rounded-bl",
                  node.score_contribution > 0 ? "border-emerald-200" :
                  node.score_contribution < 0 ? "border-red-200" : "border-slate-200"
                )} />

                <div className="ml-3 flex items-center gap-2 p-1.5 bg-white rounded border border-slate-100 text-xs">
                  {node.signal === "bullish" ? <TrendingUp className="w-3 h-3 text-emerald-500 shrink-0" /> :
                   node.signal === "bearish" ? <TrendingDown className="w-3 h-3 text-red-500 shrink-0" /> :
                   <Minus className="w-3 h-3 text-slate-400 shrink-0" />}

                  <span className="flex-1 text-slate-700 truncate">{node.data_point}</span>

                  <span className={cn(
                    "font-mono shrink-0",
                    node.score_contribution > 0 ? "text-emerald-600" :
                    node.score_contribution < 0 ? "text-red-600" : "text-slate-500"
                  )}>
                    {node.score_contribution > 0 ? "+" : ""}{node.score_contribution.toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function DecisionTreeModal({ trail, probability }: DecisionTreeModalProps) {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsOpen(false);
    };
    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
      document.body.style.overflow = "hidden";
    }
    return () => {
      document.removeEventListener("keydown", handleEscape);
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  const nodesByCategory: Record<string, DecisionNode[]> = {};
  trail.nodes.forEach((node) => {
    if (!nodesByCategory[node.category]) nodesByCategory[node.category] = [];
    nodesByCategory[node.category].push(node);
  });

  const sortedCategories = Object.keys(nodesByCategory).sort(
    (a, b) => (trail.category_weights[b] || 0) - (trail.category_weights[a] || 0)
  );

  const totalPositive = Object.values(trail.category_scores).filter(s => s > 0).reduce((a, b) => a + b, 0);
  const totalNegative = Object.values(trail.category_scores).filter(s => s < 0).reduce((a, b) => a + b, 0);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="w-full flex items-center justify-between p-4 bg-white border border-slate-200 rounded-xl hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <GitBranch className="w-5 h-5 text-slate-600" />
          <span className="font-medium text-slate-700">View Decision Tree</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className="text-emerald-600">+{totalPositive.toFixed(1)}%</span>
          <span className="text-slate-300">/</span>
          <span className="text-red-600">{totalNegative.toFixed(1)}%</span>
        </div>
      </button>

      {isOpen && (
        <div
          className="fixed inset-0 z-[9999] overflow-hidden"
          style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0 }}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setIsOpen(false)}
          />

          {/* Modal - Centered with safe margins */}
          <div className="absolute inset-0 flex items-center justify-center p-6">
            <div
              className="relative w-full max-w-md bg-white rounded-xl shadow-2xl flex flex-col"
              style={{ maxHeight: 'calc(100vh - 120px)', marginTop: '20px', marginBottom: '20px' }}
            >
              {/* Header */}
              <div className="shrink-0 flex items-center justify-between px-4 py-3 border-b border-slate-200 bg-slate-50 rounded-t-xl">
                <h2 className="font-semibold text-slate-800 flex items-center gap-2">
                  <GitBranch className="w-4 h-4" />
                  Decision Tree
                </h2>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1 hover:bg-slate-200 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-slate-500" />
                </button>
              </div>

              {/* Scrollable Content */}
              <div className="flex-1 overflow-y-auto overscroll-contain p-4">
                {/* Start */}
                <div className="flex justify-center mb-3">
                  <div className="px-3 py-1.5 bg-slate-800 text-white rounded-lg text-sm font-medium">
                    Base: 50%
                  </div>
                </div>

                {/* Trunk */}
                <div className="flex justify-center">
                  <div className="w-0.5 h-3 bg-slate-300" />
                </div>

                {/* Branches */}
                <div className="border-l-2 border-slate-300 ml-3 space-y-2 py-1">
                  {sortedCategories.map((category) => (
                    <Branch
                      key={category}
                      category={category}
                      nodes={nodesByCategory[category]}
                      score={trail.category_scores[category] || 0}
                      weight={trail.category_weights[category] || 0}
                    />
                  ))}
                </div>

                {/* Bottom trunk */}
                <div className="flex justify-center">
                  <div className="w-0.5 h-3 bg-slate-300" />
                </div>

                {/* Result */}
                <div className="flex justify-center mt-2">
                  <div className={cn(
                    "px-4 py-2 rounded-lg text-white text-center",
                    probability >= 0.5 ? "bg-emerald-600" : "bg-red-600"
                  )}>
                    <div className="text-xs opacity-80">Result</div>
                    <div className="text-lg font-bold font-mono">{(probability * 100).toFixed(0)}%</div>
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="shrink-0 px-4 py-2 border-t border-slate-200 bg-slate-50 text-xs text-slate-500 text-center rounded-b-xl">
                Click branches to expand
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
