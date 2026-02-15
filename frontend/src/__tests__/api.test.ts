import { describe, it, expect } from "vitest";
import type {
  PredictionRequest,
  Market,
  PredictionResponse,
  QueryAnalysisResult,
} from "../lib/api";

describe("API types", () => {
  it("should define valid Market types", () => {
    const us: Market = "US";
    const india: Market = "IN";
    expect(us).toBe("US");
    expect(india).toBe("IN");
  });

  it("should define valid PredictionRequest", () => {
    const request: PredictionRequest = {
      query: "Will NVDA reach $150?",
      ticker: "NVDA",
      target_price: 150,
      include_technicals: true,
      include_sentiment: true,
      include_news: true,
      market: "US",
    };
    expect(request.query).toBe("Will NVDA reach $150?");
    expect(request.market).toBe("US");
  });

  it("should support India market in PredictionRequest", () => {
    const request: PredictionRequest = {
      query: "Will Reliance reach ₹3000?",
      ticker: "RELIANCE.NS",
      target_price: 3000,
      market: "IN",
    };
    expect(request.ticker).toBe("RELIANCE.NS");
    expect(request.market).toBe("IN");
  });
});

describe("API URL construction", () => {
  it("should strip trailing slash from API base URL", () => {
    const url = "https://example.com/api/".replace(/\/+$/, "");
    expect(url).toBe("https://example.com/api");
  });

  it("should handle URL without trailing slash", () => {
    const url = "https://example.com/api".replace(/\/+$/, "");
    expect(url).toBe("https://example.com/api");
  });
});
