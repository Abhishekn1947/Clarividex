"use client";

import { useState, useEffect, useRef } from "react";
import { X, GitBranch, TrendingUp, TrendingDown, Minus, ChevronRight } from "lucide-react";
import { DecisionTrail, DecisionNode } from "@/lib/api";
import { cn } from "@/lib/utils";

interface DecisionTreeModalProps {
  trail: DecisionTrail;
  probability: number;
}

function NodeCard({ node }: { node: DecisionNode }) {
  const isPositive = node.score_contribution > 0;
  const isNegative = node.score_contribution < 0;

  return (
    <div className={cn(
      "shrink-0 w-44 sm:w-52 p-2.5 rounded-lg border text-xs space-y-1.5",
      isPositive ? "bg-emerald-50/50 border-emerald-200" :
      isNegative ? "bg-red-50/50 border-red-200" :
      "bg-slate-50 border-slate-200"
    )}>
      <div className="flex items-center gap-1.5">
        {node.signal === "bullish" ? <TrendingUp className="w-3 h-3 text-emerald-500 shrink-0" /> :
         node.signal === "bearish" ? <TrendingDown className="w-3 h-3 text-red-500 shrink-0" /> :
         <Minus className="w-3 h-3 text-slate-400 shrink-0" />}
        <span className="font-medium text-slate-700 truncate">{node.data_point}</span>
      </div>

      {node.value && (
        <div className="text-slate-600 bg-white/80 rounded px-1.5 py-0.5 truncate">
          {node.value}
        </div>
      )}

      {node.reasoning && (
        <p className="text-slate-500 line-clamp-2 leading-relaxed">{node.reasoning}</p>
      )}

      <div className="flex items-center justify-between pt-0.5">
        {node.source && (
          <span className="text-slate-400 bg-white/80 border border-slate-100 px-1.5 py-0.5 rounded text-[10px] truncate max-w-[60%]">
            {node.source}
          </span>
        )}
        <span className={cn(
          "font-mono px-1.5 py-0.5 rounded font-medium ml-auto",
          isPositive ? "text-emerald-700 bg-emerald-100" :
          isNegative ? "text-red-700 bg-red-100" : "text-slate-500 bg-slate-100"
        )}>
          {node.score_contribution > 0 ? "+" : ""}{node.score_contribution.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

function Branch({
  category,
  nodes,
  score,
  weight,
  isLast,
}: {
  category: string;
  nodes: DecisionNode[];
  score: number;
  weight: number;
  isLast: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const label = category.split("_").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
  const isPositive = score > 0;
  const isNegative = score < 0;

  return (
    <div className="flex items-stretch">
      {/* Vertical trunk connector */}
      <div className="relative w-8 shrink-0 flex items-center justify-center">
        {/* Vertical line (full height unless last) */}
        <div className={cn(
          "absolute left-1/2 -translate-x-1/2 w-0.5 bg-slate-300",
          isLast ? "top-0 h-1/2" : "inset-y-0"
        )} />
        {/* Horizontal connector to branch */}
        <div className="absolute left-1/2 right-0 h-0.5 bg-slate-300 top-1/2" />
        {/* Dot at junction */}
        <div className={cn(
          "relative z-10 w-2.5 h-2.5 rounded-full border-2",
          isPositive ? "bg-emerald-400 border-emerald-300" :
          isNegative ? "bg-red-400 border-red-300" :
          "bg-slate-400 border-slate-300"
        )} />
      </div>

      {/* Branch content: category button + expanded nodes */}
      <div className="flex items-center gap-0 py-1.5 min-w-0 flex-1">
        {/* Category button */}
        <button
          onClick={() => setExpanded(!expanded)}
          className={cn(
            "shrink-0 flex items-center gap-2 px-2 sm:px-3 py-2 rounded-lg border transition-all text-left",
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

          <div className="min-w-0">
            <div className="font-medium text-xs sm:text-sm text-slate-800 whitespace-nowrap">{label}</div>
            <div className="text-[10px] text-slate-400">{weight}% weight</div>
          </div>

          <span className={cn(
            "font-mono text-sm font-semibold ml-2 whitespace-nowrap",
            isPositive ? "text-emerald-600" : isNegative ? "text-red-600" : "text-slate-500"
          )}>
            {score > 0 ? "+" : ""}{score.toFixed(1)}%
          </span>

          <ChevronRight className={cn(
            "w-4 h-4 text-slate-400 transition-transform shrink-0 ml-1",
            expanded && "rotate-90"
          )} />
        </button>

        {/* Horizontal connector + expanded nodes */}
        {expanded && nodes.length > 0 && (
          <div className="flex items-center min-w-0 flex-1">
            {/* Connector line */}
            <div className={cn(
              "w-4 h-0.5 shrink-0",
              isPositive ? "bg-emerald-300" : isNegative ? "bg-red-300" : "bg-slate-300"
            )} />

            {/* Scrollable node cards */}
            <div
              ref={scrollRef}
              className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin min-w-0"
              style={{ scrollbarWidth: "thin" }}
            >
              {nodes.map((node) => (
                <NodeCard key={node.id} node={node} />
              ))}
            </div>
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
          className="fixed inset-0 z-[9999]"
          style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0 }}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setIsOpen(false)}
          />

          {/* Modal — pinned within viewport, scrolls internally */}
          <div className="absolute inset-2 sm:inset-4 lg:inset-6 flex items-center justify-center pointer-events-none">
            <div
              className="w-full max-w-5xl max-h-full bg-white rounded-xl shadow-2xl flex flex-col pointer-events-auto"
              onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="shrink-0 flex items-center justify-between px-3 sm:px-5 py-3 border-b border-slate-200 bg-slate-50 rounded-t-xl">
                  <h2 className="font-semibold text-slate-800 flex items-center gap-2">
                    <GitBranch className="w-4 h-4" />
                    Decision Tree
                  </h2>
                  <div className="flex items-center gap-4">
                    {/* Contribution summary in header */}
                    <div className="hidden sm:flex items-center gap-3 text-xs">
                      <span className="text-slate-400">Balance:</span>
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-red-600 font-medium">{totalNegative.toFixed(1)}%</span>
                        <div className="flex h-1.5 w-20 rounded-full overflow-hidden bg-slate-200">
                          <div
                            className="bg-red-400"
                            style={{ width: `${(Math.abs(totalNegative) / (Math.abs(totalNegative) + totalPositive + 0.01)) * 100}%` }}
                          />
                          <div
                            className="bg-emerald-400"
                            style={{ width: `${(totalPositive / (Math.abs(totalNegative) + totalPositive + 0.01)) * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-emerald-600 font-medium">+{totalPositive.toFixed(1)}%</span>
                      </div>
                    </div>
                    <button
                      onClick={() => setIsOpen(false)}
                      className="p-1 hover:bg-slate-200 rounded-lg transition-colors"
                    >
                      <X className="w-5 h-5 text-slate-500" />
                    </button>
                  </div>
                </div>

                {/* Scrollable content — both axes */}
                <div className="flex-1 overflow-auto overscroll-contain">
                  <div className="p-3 sm:p-5 min-w-max">
                    {/* Horizontal tree layout */}
                    <div className="flex items-start gap-0">

                      {/* Left: Base + trunk + Result column */}
                      <div className="flex flex-col items-center shrink-0">
                        {/* Base pill */}
                        <div className="px-4 py-2 bg-slate-800 text-white rounded-lg text-sm font-semibold font-mono">
                          Base: 50%
                        </div>

                        {/* Vertical trunk going down to branches */}
                        <div className="w-0.5 h-4 bg-slate-300" />
                      </div>
                    </div>

                    {/* Branches area */}
                    <div className="ml-0">
                      {sortedCategories.map((category, index) => (
                        <Branch
                          key={category}
                          category={category}
                          nodes={nodesByCategory[category]}
                          score={trail.category_scores[category] || 0}
                          weight={trail.category_weights[category] || 0}
                          isLast={index === sortedCategories.length - 1}
                        />
                      ))}
                    </div>

                    {/* Result node */}
                    <div className="flex items-center mt-0">
                      {/* Trunk end */}
                      <div className="w-8 flex justify-center shrink-0">
                        <div className="w-0.5 h-4 bg-slate-300" />
                      </div>
                    </div>
                    <div className="flex items-center">
                      <div className="w-8 shrink-0" />
                      <div className={cn(
                        "px-5 py-3 rounded-xl text-white text-center shadow-sm",
                        probability >= 0.5
                          ? "bg-gradient-to-r from-emerald-500 to-emerald-600"
                          : "bg-gradient-to-r from-red-500 to-red-600"
                      )}>
                        <div className="text-xs opacity-80 mb-0.5">Final Probability</div>
                        <div className="text-2xl font-bold font-mono">{(probability * 100).toFixed(0)}%</div>
                        <div className="text-xs opacity-70 mt-1 font-mono">
                          50%{totalPositive > 0 ? ` + ${totalPositive.toFixed(1)}` : ""}{totalNegative < 0 ? ` - ${Math.abs(totalNegative).toFixed(1)}` : ""}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Footer */}
                <div className="shrink-0 px-3 sm:px-5 py-2 border-t border-slate-200 bg-slate-50 text-xs text-slate-500 text-center rounded-b-xl">
                  Click branches to expand &middot; Scroll horizontally to see all factors
                </div>
              </div>
            </div>
        </div>
      )}
    </>
  );
}
