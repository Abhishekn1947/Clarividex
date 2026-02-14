/**
 * API client for Future Prediction AI backend.
 */

// Use relative URLs in the browser so requests go through the Next.js rewrite
// proxy (works from any device on the network). Server-side uses the explicit URL.
const API_BASE_URL =
  typeof window !== "undefined"
    ? ""
    : process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PredictionRequest {
  query: string;
  ticker?: string;
  target_price?: number;
  target_date?: string;
  include_technicals?: boolean;
  include_sentiment?: boolean;
  include_news?: boolean;
}

export interface Factor {
  description: string;
  impact: string;
  weight: number;
  source: string;
  confidence: number;
}

export interface Catalyst {
  event: string;
  date: string | null;
  potential_impact: string;
  importance: string;
}

export interface DecisionNode {
  id: string;
  category: string;
  source: string;
  data_point: string;
  value: string | null;
  signal: string;
  weight: number;
  score_contribution: number;
  reasoning: string;
}

export interface DecisionTrail {
  nodes: DecisionNode[];
  category_scores: Record<string, number>;
  category_weights: Record<string, number>;
  final_calculation: string;
}

export interface ReasoningChain {
  summary: string;
  bullish_factors: Factor[];
  bearish_factors: Factor[];
  catalysts: Catalyst[];
  risks: string[];
  assumptions: string[];
  decision_trail?: DecisionTrail;
}

export interface NewsArticle {
  title: string;
  source: string;
  url: string | null;
  published_at: string;
  sentiment_score: number;
}

export interface SocialSentiment {
  platform: string;
  mentions_count: number;
  sentiment_score: number;
  bullish_percentage: number;
  trending: boolean;
}

export interface TechnicalIndicators {
  ticker: string;
  rsi_14: number | null;
  macd: number | null;
  macd_signal: number | null;
  macd_histogram: number | null;
  sma_20: number | null;
  sma_50: number | null;
  sma_200: number | null;
  ema_12: number | null;
  ema_26: number | null;
  bollinger_upper: number | null;
  bollinger_lower: number | null;
  support_level: number | null;
  resistance_level: number | null;
}

export interface DataSource {
  name: string;
  url: string | null;
  timestamp: string;
  reliability_score: number;
}

export interface DataLimitation {
  category: string;
  severity: string;
  message: string;
  recommendation: string | null;
}

export interface PredictionResponse {
  id: string;
  query: string;
  ticker: string | null;
  prediction_type: string;
  probability: number;
  confidence_level: string;
  confidence_score: number;
  target_price: number | null;
  target_date: string | null;
  current_price: number | null;
  price_gap_percent: number | null;
  reasoning: ReasoningChain;
  sentiment: string;
  sentiment_score: number;
  technical_score: number | null;
  data_quality_score: number;
  data_points_analyzed: number;
  sources_used: DataSource[];
  data_limitations: DataLimitation[];
  has_limited_data: boolean;
  news_articles: NewsArticle[];
  social_sentiment: SocialSentiment[];
  technicals: TechnicalIndicators | null;
  created_at: string;
  disclaimer: string;
}

export interface StockQuote {
  ticker: string;
  current_price: number;
  previous_close: number;
  change_percent: number;
  volume: number;
  market_cap: number | null;
  pe_ratio: number | null;
  fifty_two_week_high: number | null;
  fifty_two_week_low: number | null;
}

export interface HealthStatus {
  status: string;
  version: string;
  apis: Array<{
    name: string;
    available: boolean;
    error: string | null;
  }>;
}

export interface TickerSuggestion {
  ticker: string;
  company_name: string;
  confidence: number;
  match_reason: string;
}

export interface TickerExtractionResult {
  ticker: string | null;
  company_name: string | null;
  confidence: number;
  needs_confirmation: boolean;
  suggestions: TickerSuggestion[];
  message: string | null;
}

export interface SSEEvent {
  event: string;
  data: Record<string, unknown>;
  timestamp: number;
}

export interface QueryAnalysisResult {
  category: "clear" | "vague" | "non_financial";
  can_proceed: boolean;
  quality_score: number;
  issues: string[];
  suggestions: string[];
  message: string;
  cleaned_query: string;
}

class APIClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      const detail = error.detail;
      const message = typeof detail === "object" && detail !== null
        ? detail.message || JSON.stringify(detail)
        : detail || `API error: ${response.status}`;
      throw new Error(message);
    }

    return response.json();
  }

  // Health check
  async getHealth(): Promise<HealthStatus> {
    return this.request<HealthStatus>("/api/v1/health");
  }

  // Create prediction
  async createPrediction(
    request: PredictionRequest
  ): Promise<PredictionResponse> {
    return this.request<PredictionResponse>("/api/v1/predict", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  // Get stock quote
  async getQuote(ticker: string): Promise<StockQuote> {
    return this.request<StockQuote>(`/api/v1/stock/${ticker}/quote`);
  }

  // Get technical indicators
  async getTechnicals(ticker: string): Promise<TechnicalIndicators> {
    return this.request<TechnicalIndicators>(`/api/v1/stock/${ticker}/technicals`);
  }

  // Get news for stock
  async getNews(
    ticker: string,
    limit: number = 10
  ): Promise<{ articles: NewsArticle[]; aggregate_sentiment: object }> {
    return this.request(`/api/v1/stock/${ticker}/news?limit=${limit}`);
  }

  // Get social sentiment
  async getSocialSentiment(
    ticker: string
  ): Promise<{ platforms: SocialSentiment[]; aggregate: object }> {
    return this.request(`/api/v1/stock/${ticker}/social`);
  }

  // Extract ticker from text
  async extractTicker(text: string): Promise<{ extracted_ticker: string | null }> {
    return this.request(`/api/v1/extract-ticker?text=${encodeURIComponent(text)}`);
  }

  // Validate and extract ticker with confidence info
  async validateTicker(query: string): Promise<TickerExtractionResult> {
    return this.request<TickerExtractionResult>(
      `/api/v1/validate-ticker?query=${encodeURIComponent(query)}`
    );
  }

  // Get popular tickers
  async getPopularTickers(): Promise<{ tickers: string[] }> {
    return this.request("/api/v1/popular-tickers");
  }

  // Analyze query quality and relevance
  async analyzeQuery(query: string): Promise<QueryAnalysisResult> {
    return this.request<QueryAnalysisResult>("/api/v1/analyze-query", {
      method: "POST",
      body: JSON.stringify({ query }),
    });
  }

  // Chat about prediction results
  async chat(message: string, context: string): Promise<{ response: string; model: string }> {
    return this.request<{ response: string; model: string }>("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({ message, context }),
    });
  }

  // Stream prediction via SSE
  async streamPrediction(
    request: PredictionRequest,
    onEvent: (event: SSEEvent) => void,
    signal?: AbortSignal,
  ): Promise<PredictionResponse | null> {
    const url = `${this.baseUrl}/api/v1/predict/stream`;

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      const detail = error.detail;
      const message = typeof detail === "object" && detail !== null
        ? detail.message || JSON.stringify(detail)
        : detail || `API error: ${response.status}`;
      throw new Error(message);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";
    let finalPrediction: PredictionResponse | null = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete line in buffer

      let currentEvent = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            const sseEvent: SSEEvent = {
              event: currentEvent,
              data,
              timestamp: Date.now(),
            };
            onEvent(sseEvent);

            // Capture the final prediction from 'done' event
            if (currentEvent === "done" && data.prediction) {
              finalPrediction = data.prediction;
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }

    return finalPrediction;
  }
}

// Export singleton instance
export const api = new APIClient();

// Export class for custom instances
export { APIClient };
