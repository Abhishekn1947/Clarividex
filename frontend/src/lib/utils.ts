/**
 * Utility functions for the frontend.
 */

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge Tailwind CSS classes with proper precedence.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format a number as currency.
 */
export function formatCurrency(value: number, decimals: number = 2): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

/**
 * Format a number as percentage.
 */
export function formatPercent(value: number, decimals: number = 1): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(decimals)}%`;
}

/**
 * Get confidence label.
 */
export function getConfidenceLabel(level: string): {
  label: string;
  color: string;
} {
  const labels: Record<string, { label: string; color: string }> = {
    very_low: { label: "Very Low", color: "text-red-600 bg-red-100" },
    low: { label: "Low", color: "text-orange-600 bg-orange-100" },
    medium: { label: "Medium", color: "text-yellow-600 bg-yellow-100" },
    medium_high: { label: "Medium-High", color: "text-lime-600 bg-lime-100" },
    high: { label: "High", color: "text-green-600 bg-green-100" },
    very_high: { label: "Very High", color: "text-emerald-600 bg-emerald-100" },
  };
  return labels[level] || { label: level, color: "text-gray-600 bg-gray-100" };
}

/**
 * Get sentiment label and color.
 */
export function getSentimentLabel(sentiment: string): {
  label: string;
  color: string;
  emoji: string;
} {
  const labels: Record<string, { label: string; color: string; emoji: string }> = {
    very_bearish: { label: "Very Bearish", color: "text-red-700 bg-red-100", emoji: "📉" },
    bearish: { label: "Bearish", color: "text-red-600 bg-red-50", emoji: "🔻" },
    neutral: { label: "Neutral", color: "text-gray-600 bg-gray-100", emoji: "➖" },
    bullish: { label: "Bullish", color: "text-green-600 bg-green-50", emoji: "🔼" },
    very_bullish: { label: "Very Bullish", color: "text-green-700 bg-green-100", emoji: "📈" },
  };
  return labels[sentiment] || { label: sentiment, color: "text-gray-600 bg-gray-100", emoji: "❓" };
}

/**
 * Format a date string.
 */
function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
}

/**
 * Format relative time.
 */
export function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return formatDate(dateString);
}
