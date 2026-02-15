import { describe, it, expect } from "vitest";

// Replicate the cross-market detection logic from page.tsx for unit testing
type Market = "US" | "IN";

function detectCrossMarket(
  query: string,
  market: Market
): { targetMarket: Market; message: string } | null {
  if (market === "US") {
    const hasRupee = query.includes("₹");
    const hasNSBO = /\.\s*(?:ns|bo)\b/i.test(query);
    const indianTerms =
      /\b(reliance|infosys|tcs|hdfc\s*bank|icici|wipro|nifty|sensex|nse|bse|bharti\s*airtel|itc\b|maruti|bajaj\s*finance|adani|kotak|sun\s*pharma|asian\s*paints|titan|ntpc|ongc|coal\s*india|sbi|axis\s*bank|hindustan\s*unilever|power\s*grid|larsen|hul)\b/i;
    if (hasRupee || hasNSBO || indianTerms.test(query)) {
      return {
        targetMarket: "IN",
        message:
          "This looks like an Indian stock. Switch to India mode for accurate NSE/BSE data and ₹ pricing.",
      };
    }
  }

  if (market === "IN") {
    const usTickerPattern =
      /\b(AAPL|TSLA|NVDA|MSFT|AMZN|GOOG|GOOGL|META|AMD|NFLX|SPY|QQQ|DIS|BA|JPM|GS|INTC|CRM|ORCL|PYPL|UBER|ABNB|COIN)\b/;
    const usCompanyNames =
      /\b(tesla|apple|nvidia|microsoft|amazon|google|alphabet|meta|facebook|netflix|disney|boeing|intel|salesforce|oracle|paypal|uber|airbnb|coinbase|amd)\b/i;
    const usTerms = /\b(s&p\s*500|nasdaq|dow\s*jones|nyse|wall\s*street)\b/i;
    const hasDollarPrice = /\$\d/.test(query);
    if (
      usTickerPattern.test(query) ||
      usCompanyNames.test(query) ||
      usTerms.test(query) ||
      hasDollarPrice
    ) {
      return {
        targetMarket: "US",
        message:
          "This looks like a US stock. Switch to USA mode for accurate NYSE/NASDAQ data and $ pricing.",
      };
    }
  }

  return null;
}

describe("Cross-market detection", () => {
  describe("US mode detecting Indian stocks", () => {
    it("detects rupee symbol", () => {
      const result = detectCrossMarket("Will stock reach ₹3000?", "US");
      expect(result).not.toBeNull();
      expect(result!.targetMarket).toBe("IN");
    });

    it("detects .NS suffix", () => {
      const result = detectCrossMarket("Will RELIANCE.NS go up?", "US");
      expect(result).not.toBeNull();
      expect(result!.targetMarket).toBe("IN");
    });

    it("detects Indian company names", () => {
      const companies = ["Reliance", "TCS", "Infosys", "HDFC Bank", "Wipro", "Nifty", "Sensex"];
      for (const company of companies) {
        const result = detectCrossMarket(`Will ${company} go up?`, "US");
        expect(result).not.toBeNull();
        expect(result!.targetMarket).toBe("IN");
      }
    });

    it("does not flag US stocks on US mode", () => {
      const result = detectCrossMarket("Will AAPL reach $200?", "US");
      expect(result).toBeNull();
    });
  });

  describe("India mode detecting US stocks", () => {
    it("detects US ticker symbols", () => {
      const result = detectCrossMarket("Will AAPL go up?", "IN");
      expect(result).not.toBeNull();
      expect(result!.targetMarket).toBe("US");
    });

    it("detects US company names", () => {
      const companies = ["Tesla", "Apple", "Nvidia", "Microsoft", "Amazon", "Google", "Netflix"];
      for (const company of companies) {
        const result = detectCrossMarket(`Will ${company} stock go up?`, "IN");
        expect(result).not.toBeNull();
        expect(result!.targetMarket).toBe("US");
      }
    });

    it("detects dollar prices", () => {
      const result = detectCrossMarket("Will stock reach $200?", "IN");
      expect(result).not.toBeNull();
      expect(result!.targetMarket).toBe("US");
    });

    it("detects US market terms", () => {
      const terms = ["S&P 500", "NASDAQ", "Dow Jones", "NYSE", "Wall Street"];
      for (const term of terms) {
        const result = detectCrossMarket(`How is ${term} doing?`, "IN");
        expect(result).not.toBeNull();
        expect(result!.targetMarket).toBe("US");
      }
    });

    it("does not flag Indian stocks on India mode", () => {
      const result = detectCrossMarket("Will Reliance reach ₹3000?", "IN");
      expect(result).toBeNull();
    });
  });
});
