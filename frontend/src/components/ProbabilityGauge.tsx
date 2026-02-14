"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface ProbabilityGaugeProps {
  probability: number; // 0-1
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
  animated?: boolean;
}

export function ProbabilityGauge({
  probability,
  size = "md",
  showLabel = true,
  animated = true,
}: ProbabilityGaugeProps) {
  const [displayValue, setDisplayValue] = useState(animated ? 0 : probability);

  useEffect(() => {
    if (!animated) {
      setDisplayValue(probability);
      return;
    }

    // Animate the number counting up
    const duration = 1200;
    const steps = 50;
    const increment = probability / steps;
    let current = 0;

    const timer = setInterval(() => {
      current += increment;
      if (current >= probability) {
        setDisplayValue(probability);
        clearInterval(timer);
      } else {
        setDisplayValue(current);
      }
    }, duration / steps);

    return () => clearInterval(timer);
  }, [probability, animated]);

  const displayPercent = Math.round(displayValue * 100);

  // Professional muted colors
  const getColor = () => {
    if (probability >= 0.65) return { primary: "#4a7c59", secondary: "#5e8c64" }; // sage green
    if (probability >= 0.5) return { primary: "#b48c46", secondary: "#a07c3d" }; // muted amber
    if (probability >= 0.35) return { primary: "#b27a59", secondary: "#9f6d4e" }; // terracotta
    return { primary: "#b25959", secondary: "#9f4e4e" }; // muted red
  };

  const color = getColor();

  const dimensions = {
    sm: { width: 100, height: 100, stroke: 6, fontSize: "text-xl" },
    md: { width: 140, height: 140, stroke: 8, fontSize: "text-3xl" },
    lg: { width: 180, height: 180, stroke: 10, fontSize: "text-4xl" },
  };

  const dim = dimensions[size];
  const radius = (dim.width - dim.stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (displayValue * circumference);

  return (
    <div className="flex flex-col items-center scale-[0.85] sm:scale-100 origin-top">
      <div
        className="relative"
        style={{ width: dim.width, height: dim.height }}
      >
        {/* SVG Circle */}
        <svg
          className="transform -rotate-90"
          width={dim.width}
          height={dim.height}
        >
          <defs>
            <linearGradient id={`gauge-gradient-${size}`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={color.primary} />
              <stop offset="100%" stopColor={color.secondary} />
            </linearGradient>
          </defs>

          {/* Background circle */}
          <circle
            cx={dim.width / 2}
            cy={dim.height / 2}
            r={radius}
            fill="none"
            stroke="#e2e8f0"
            strokeWidth={dim.stroke}
          />

          {/* Progress circle */}
          <circle
            cx={dim.width / 2}
            cy={dim.height / 2}
            r={radius}
            fill="none"
            stroke={`url(#gauge-gradient-${size})`}
            strokeWidth={dim.stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className={cn(
              animated && "transition-all duration-700 ease-out"
            )}
          />
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className={cn(
              dim.fontSize,
              "font-semibold tabular-nums",
              probability >= 0.5 ? "text-slate-700" : "text-slate-600"
            )}
          >
            {displayPercent}%
          </span>
          {showLabel && (
            <span className="text-xs text-slate-500 mt-0.5">Probability</span>
          )}
        </div>
      </div>

      {/* Sentiment indicator */}
      <div className={cn(
        "mt-3 px-3 py-1 rounded-md text-xs font-medium",
        probability >= 0.65 && "bg-emerald-50 text-emerald-700 border border-emerald-200",
        probability >= 0.5 && probability < 0.65 && "bg-amber-50 text-amber-700 border border-amber-200",
        probability >= 0.35 && probability < 0.5 && "bg-orange-50 text-orange-700 border border-orange-200",
        probability < 0.35 && "bg-red-50 text-red-700 border border-red-200",
      )}>
        {probability >= 0.65 && "Likely"}
        {probability >= 0.5 && probability < 0.65 && "Possible"}
        {probability >= 0.35 && probability < 0.5 && "Unlikely"}
        {probability < 0.35 && "Very Unlikely"}
      </div>
    </div>
  );
}

// Mini probability bar for inline use
export function ProbabilityBar({ probability }: { probability: number }) {
  const percent = Math.round(probability * 100);

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-500">Probability</span>
        <span className={cn(
          "font-medium",
          probability >= 0.5 ? "text-slate-700" : "text-slate-600"
        )}>
          {percent}%
        </span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-700 ease-out",
            probability >= 0.65 && "bg-emerald-500",
            probability >= 0.5 && probability < 0.65 && "bg-amber-500",
            probability >= 0.35 && probability < 0.5 && "bg-orange-500",
            probability < 0.35 && "bg-red-400",
          )}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
