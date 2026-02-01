"use client";

import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { MessageCircle, X, Send, Sparkles, Loader2, RotateCcw } from "lucide-react";
import { PredictionResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

// Format assistant message with proper structure
function FormattedMessage({ content }: { content: string }) {
  // Parse the content into structured elements
  const formatContent = (text: string) => {
    const lines = text.split('\n');
    const elements: JSX.Element[] = [];
    let listItems: string[] = [];
    let listKey = 0;

    const flushList = () => {
      if (listItems.length > 0) {
        elements.push(
          <ul key={`list-${listKey++}`} className="ml-3 my-1.5 space-y-1">
            {listItems.map((item, i) => (
              <li key={i} className="flex gap-1.5">
                <span className="text-amber-500 mt-0.5">•</span>
                <span>{formatInlineText(item)}</span>
              </li>
            ))}
          </ul>
        );
        listItems = [];
      }
    };

    const formatInlineText = (text: string) => {
      // Bold text with **text** or __text__
      const parts = text.split(/(\*\*[^*]+\*\*|__[^_]+__)/g);
      return parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={i} className="font-semibold text-slate-800">{part.slice(2, -2)}</strong>;
        }
        if (part.startsWith('__') && part.endsWith('__')) {
          return <strong key={i} className="font-semibold text-slate-800">{part.slice(2, -2)}</strong>;
        }
        // Numbers and percentages
        return part.replace(/(\d+\.?\d*%?)/g, (match) => {
          if (match.includes('%') || parseFloat(match) > 0) {
            return match;
          }
          return match;
        });
      });
    };

    lines.forEach((line, index) => {
      const trimmedLine = line.trim();

      // Empty line - flush list and add spacing
      if (!trimmedLine) {
        flushList();
        if (elements.length > 0) {
          elements.push(<div key={`space-${index}`} className="h-2" />);
        }
        return;
      }

      // Headers with ### or **Header:**
      if (trimmedLine.startsWith('###')) {
        flushList();
        const headerText = trimmedLine.replace(/^#+\s*/, '');
        elements.push(
          <h4 key={`h-${index}`} className="font-semibold text-slate-800 mt-2 mb-1 text-sm">
            {headerText}
          </h4>
        );
        return;
      }

      // Bold header lines (e.g., "**Summary:**")
      if (/^\*\*[^*]+:\*\*$/.test(trimmedLine) || /^__[^_]+:__$/.test(trimmedLine)) {
        flushList();
        const headerText = trimmedLine.replace(/^\*\*|\*\*$|^__|__$/g, '');
        elements.push(
          <h4 key={`bh-${index}`} className="font-semibold text-amber-600 mt-2 mb-1 text-sm">
            {headerText}
          </h4>
        );
        return;
      }

      // Bullet points
      if (trimmedLine.startsWith('- ') || trimmedLine.startsWith('• ') || trimmedLine.startsWith('* ')) {
        listItems.push(trimmedLine.substring(2));
        return;
      }

      // Numbered lists
      if (/^\d+[\.\)]\s/.test(trimmedLine)) {
        const itemText = trimmedLine.replace(/^\d+[\.\)]\s*/, '');
        listItems.push(itemText);
        return;
      }

      // Regular paragraph
      flushList();
      elements.push(
        <p key={`p-${index}`} className="my-1">
          {formatInlineText(trimmedLine)}
        </p>
      );
    });

    flushList();
    return elements;
  };

  return <div className="text-sm leading-relaxed">{formatContent(content)}</div>;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface FloatingChatbotProps {
  prediction: PredictionResponse;
  onClose?: () => void;
}

// JSON structure for the prediction context
interface PredictionContextJSON {
  meta: {
    generatedAt: string;
    sessionId: string;
  };
  prediction: {
    query: string;
    ticker: string;
    companyName: string;
    currentPrice: number | null;
    targetPrice: number | null;
    targetDate: string | null;
    probability: number;
    confidenceLevel: string;
    sentiment: string;
    dataQualityScore: number;
    dataPointsAnalyzed: number;
    sourcesCount: number;
  };
  technicals: {
    rsi: number | null;
    rsiSignal: string;
    macd: number | null;
    macdSignal: number | null;
    macdHistogram: number | null;
    macdTrend: string;
    sma20: number | null;
    sma50: number | null;
    sma200: number | null;
    bbUpper: number | null;
    bbLower: number | null;
    atr: number | null;
    trend: string;
  } | null;
  analysis: {
    summary: string;
    bullishFactors: Array<{
      description: string;
      source: string;
      weight: number;
    }>;
    bearishFactors: Array<{
      description: string;
      source: string;
      weight: number;
    }>;
    risks: string[];
  };
  news: Array<{
    title: string;
    source: string;
    date: string;
    sentimentScore: number;
    sentimentLabel: string;
  }>;
  decisionTree: {
    technicalWeight: number;
    newsWeight: number;
    optionsWeight: number;
    marketWeight: number;
    analystWeight: number;
    socialWeight: number;
    netSignal: string;
    totalBullish: number;
    totalBearish: number;
  };
}

export default function FloatingChatbot({ prediction, onClose }: FloatingChatbotProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showTooltip, setShowTooltip] = useState(true);
  const [contextJSON, setContextJSON] = useState<PredictionContextJSON | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Generate session ID for this chat session
  const sessionId = useMemo(() => `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`, []);

  // Build JSON context when prediction changes
  const buildContextJSON = useCallback((): PredictionContextJSON => {
    const ta = prediction.technical_analysis;

    return {
      meta: {
        generatedAt: new Date().toISOString(),
        sessionId,
      },
      prediction: {
        query: prediction.query,
        ticker: prediction.ticker || "N/A",
        companyName: prediction.company_name || "",
        currentPrice: prediction.current_price || null,
        targetPrice: prediction.target_price || null,
        targetDate: prediction.target_date || null,
        probability: Math.round(prediction.probability * 100),
        confidenceLevel: prediction.confidence_level,
        sentiment: prediction.sentiment,
        dataQualityScore: Math.round(prediction.data_quality_score * 100),
        dataPointsAnalyzed: prediction.data_points_analyzed,
        sourcesCount: prediction.sources_used?.length || 0,
      },
      technicals: ta ? {
        rsi: ta.rsi || null,
        rsiSignal: ta.rsi ? (ta.rsi < 30 ? "OVERSOLD" : ta.rsi > 70 ? "OVERBOUGHT" : "NEUTRAL") : "N/A",
        macd: ta.macd || null,
        macdSignal: ta.macd_signal || null,
        macdHistogram: ta.macd_histogram || null,
        macdTrend: ta.macd_histogram ? (ta.macd_histogram > 0 ? "BULLISH" : "BEARISH") : "NEUTRAL",
        sma20: ta.sma_20 || null,
        sma50: ta.sma_50 || null,
        sma200: ta.sma_200 || null,
        bbUpper: ta.bb_upper || null,
        bbLower: ta.bb_lower || null,
        atr: ta.atr || null,
        trend: ta.trend || "N/A",
      } : null,
      analysis: {
        summary: prediction.reasoning.summary,
        bullishFactors: prediction.reasoning.bullish_factors.map(f => ({
          description: f.description,
          source: f.source,
          weight: f.weight,
        })),
        bearishFactors: prediction.reasoning.bearish_factors.map(f => ({
          description: f.description,
          source: f.source,
          weight: f.weight,
        })),
        risks: prediction.reasoning.risks,
      },
      news: prediction.news_articles.slice(0, 15).map(a => ({
        title: a.title,
        source: a.source,
        date: a.published_at ? new Date(a.published_at).toLocaleDateString("en-US", {
          month: "short", day: "numeric", year: "numeric"
        }) : "Unknown",
        sentimentScore: a.sentiment_score,
        sentimentLabel: a.sentiment_score > 0.1 ? "Positive" : a.sentiment_score < -0.1 ? "Negative" : "Neutral",
      })),
      decisionTree: {
        technicalWeight: 25,
        newsWeight: 20,
        optionsWeight: 15,
        marketWeight: 15,
        analystWeight: 15,
        socialWeight: 10,
        netSignal: prediction.reasoning.bullish_factors.length > prediction.reasoning.bearish_factors.length
          ? "BULLISH"
          : prediction.reasoning.bearish_factors.length > prediction.reasoning.bullish_factors.length
            ? "BEARISH"
            : "NEUTRAL",
        totalBullish: prediction.reasoning.bullish_factors.length,
        totalBearish: prediction.reasoning.bearish_factors.length,
      },
    };
  }, [prediction, sessionId]);

  // Initialize context JSON when chat opens
  useEffect(() => {
    if (isOpen && !contextJSON) {
      const json = buildContextJSON();
      setContextJSON(json);
      // Debug: [Clarividex Chat] Context JSON created:", json.meta.sessionId);
    }
  }, [isOpen, contextJSON, buildContextJSON]);

  // Handle chat close - keep history, only clear on explicit reset
  const handleClose = useCallback(() => {
    setIsOpen(false);
    // DON'T clear messages or context - keep them for when user reopens
    // Debug: [Clarividex Chat] Chat minimized (history preserved)");
    onClose?.();
  }, [onClose]);

  // Explicit reset function to clear everything
  const handleReset = useCallback(() => {
    setMessages([]);
    setContextJSON(null);
    // Debug: [Clarividex Chat] Chat reset - history cleared");
  }, []);

  // Hide tooltip after 5 seconds
  useEffect(() => {
    const timer = setTimeout(() => setShowTooltip(false), 5000);
    return () => clearTimeout(timer);
  }, []);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
    }
  }, [isOpen]);

  // Build context string from JSON for the API
  const buildContextFromJSON = (): string => {
    if (!contextJSON) return "";

    const { prediction: p, technicals: t, analysis, news, decisionTree } = contextJSON;

    return `You are Clarividex AI, a helpful financial analyst assistant. Answer questions based on this JSON data:

=== PREDICTION DATA (JSON) ===
${JSON.stringify(contextJSON, null, 2)}

=== QUICK REFERENCE ===

PREDICTION SUMMARY:
- Query: "${p.query}"
- Stock: ${p.ticker} (${p.companyName})
- Current Price: $${p.currentPrice?.toFixed(2) || "N/A"}
- Target Price: $${p.targetPrice?.toFixed(2) || "N/A"}
- Probability: ${p.probability}%
- Confidence: ${p.confidenceLevel}
- Sentiment: ${p.sentiment}

TECHNICAL ANALYSIS:
${t ? `- RSI: ${t.rsi?.toFixed(1) || "N/A"} (${t.rsiSignal})
- MACD Trend: ${t.macdTrend}
- Trend: ${t.trend}
- SMA50: $${t.sma50?.toFixed(2) || "N/A"}
- SMA200: $${t.sma200?.toFixed(2) || "N/A"}` : "No technical data available"}

DECISION TREE:
- Total Bullish Factors: ${decisionTree.totalBullish}
- Total Bearish Factors: ${decisionTree.totalBearish}
- Net Signal: ${decisionTree.netSignal}
- Final Probability: ${p.probability}%

ANALYSIS SUMMARY:
${analysis.summary}

BULLISH FACTORS:
${analysis.bullishFactors.map(f => `- [${f.source}] ${f.description}`).join("\n")}

BEARISH FACTORS:
${analysis.bearishFactors.map(f => `- [${f.source}] ${f.description}`).join("\n")}

RECENT NEWS:
${news.slice(0, 5).map((n, i) => `${i + 1}. "${n.title}" - ${n.source} (${n.date}) [${n.sentimentLabel}]`).join("\n")}

RESPONSE FORMAT INSTRUCTIONS:
- Structure your responses clearly with sections
- Use **Bold:** for section headers (e.g., **Summary:** or **Key Points:**)
- Use bullet points (- ) for lists of items
- Include specific numbers, percentages, and dates from the data
- Keep responses concise but informative (3-5 short paragraphs max)
- Use **bold** for important numbers or terms

EXAMPLE RESPONSE FORMAT:
**Summary:**
The prediction shows a 65% probability based on technical and sentiment analysis.

**Key Factors:**
- RSI at 28 indicates oversold conditions (bullish)
- MACD showing positive momentum
- 3 bullish news articles vs 1 bearish

**Recommendation:**
The data suggests moderate confidence in reaching the target.

CONTENT INSTRUCTIONS:
- Answer questions using the JSON data provided
- Explain how factors contributed to the probability
- Be professional and objective
- Do not provide financial advice - only explain the data`;
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input.trim(),
          context: buildContextFromJSON(),
        }),
      });

      if (!response.ok) throw new Error("Failed to get response");

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response || "I apologize, I couldn't process that question. Please try again.",
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* Floating Button */}
      <div className="fixed bottom-6 right-6 z-50">
        {/* Tooltip */}
        {showTooltip && !isOpen && (
          <div className="absolute bottom-full right-0 mb-2 w-48 p-2 bg-slate-800 text-white text-xs rounded-lg shadow-lg animate-fade-in">
            <div className="flex items-center gap-1.5">
              <Sparkles className="w-3 h-3 text-amber-400" />
              <span>Have questions about your results? Ask me!</span>
            </div>
            <div className="absolute bottom-0 right-4 translate-y-1/2 rotate-45 w-2 h-2 bg-slate-800" />
          </div>
        )}

        <button
          onClick={() => {
            if (isOpen) {
              handleClose(); // Minimize - keeps history
            } else {
              setIsOpen(true);
              setShowTooltip(false);
            }
          }}
          className={cn(
            "w-14 h-14 rounded-full shadow-lg flex items-center justify-center transition-all",
            isOpen
              ? "bg-slate-700 hover:bg-slate-600"
              : "bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-400 hover:to-orange-500"
          )}
        >
          {isOpen ? (
            <X className="w-6 h-6 text-white" />
          ) : (
            <div className="relative">
              <MessageCircle className="w-6 h-6 text-white" />
              {messages.length > 0 && (
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full border-2 border-white" />
              )}
            </div>
          )}
        </button>
      </div>

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-24 right-6 z-50 w-80 sm:w-96 bg-white rounded-xl shadow-2xl border border-slate-200 flex flex-col animate-scale-in" style={{ maxHeight: "500px" }}>
          {/* Header */}
          <div className="shrink-0 px-4 py-3 bg-gradient-to-r from-amber-500 to-orange-600 rounded-t-xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-white">
                <Sparkles className="w-4 h-4" />
                <span className="font-medium">Clarividex Assistant</span>
              </div>
              <div className="flex items-center gap-2">
                {messages.length > 0 && (
                  <button
                    onClick={handleReset}
                    className="p-1 rounded hover:bg-white/20 transition-colors"
                    title="Clear chat history"
                  >
                    <RotateCcw className="w-3.5 h-3.5 text-white/80" />
                  </button>
                )}
                <button
                  onClick={handleClose}
                  className="p-1 rounded hover:bg-white/20 transition-colors"
                  title="Minimize chat"
                >
                  <X className="w-4 h-4 text-white/80" />
                </button>
              </div>
            </div>
            <p className="text-xs text-amber-100 mt-0.5">Ask anything about your prediction</p>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-[200px] bg-white">
            {messages.length === 0 && (
              <div className="text-center py-6 text-slate-400 text-sm">
                <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-amber-50 flex items-center justify-center">
                  <MessageCircle className="w-6 h-6 text-amber-400" />
                </div>
                <p className="font-medium text-slate-600">Ask me anything about</p>
                <p className="text-amber-600 font-semibold">{prediction.ticker} prediction</p>
                <div className="mt-4 space-y-2">
                  <p className="text-xs text-slate-400">Try asking:</p>
                  <div className="flex flex-wrap justify-center gap-1.5">
                    {["Why this probability?", "Explain the signals", "Latest news?"].map((q) => (
                      <button
                        key={q}
                        onClick={() => setInput(q)}
                        className="text-xs px-2 py-1 bg-slate-100 hover:bg-slate-200 rounded-full text-slate-600 transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={cn(
                  "max-w-[85%] p-2.5 rounded-lg",
                  msg.role === "user"
                    ? "ml-auto bg-amber-500 text-white rounded-br-none text-sm"
                    : "bg-slate-50 text-slate-700 rounded-bl-none border border-slate-200"
                )}
              >
                {msg.role === "assistant" ? (
                  <FormattedMessage content={msg.content} />
                ) : (
                  msg.content
                )}
              </div>
            ))}

            {isLoading && (
              <div className="flex items-center gap-2 text-slate-400 text-sm">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Thinking...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="shrink-0 p-3 border-t border-slate-200">
            <div className="flex items-center gap-2">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question..."
                className="flex-1 px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                disabled={isLoading}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="p-2 bg-amber-500 text-white rounded-lg hover:bg-amber-400 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
