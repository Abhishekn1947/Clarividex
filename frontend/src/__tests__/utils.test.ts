import { describe, it, expect } from "vitest";
import { cn, formatCurrency } from "../lib/utils";

describe("cn utility", () => {
  it("merges class names", () => {
    expect(cn("foo", "bar")).toBe("foo bar");
  });

  it("handles conditional classes", () => {
    expect(cn("base", false && "hidden", "visible")).toBe("base visible");
  });

  it("handles undefined and null", () => {
    expect(cn("base", undefined, null, "end")).toBe("base end");
  });

  it("merges Tailwind classes correctly", () => {
    expect(cn("p-4", "p-2")).toBe("p-2");
    expect(cn("text-red-500", "text-blue-500")).toBe("text-blue-500");
  });
});

describe("formatCurrency", () => {
  it("formats USD values", () => {
    const result = formatCurrency(1234.56, 2, "US");
    expect(result).toContain("1");
    expect(result).toContain("234");
  });

  it("formats INR values", () => {
    const result = formatCurrency(1234.56, 2, "IN");
    expect(result).toContain("1");
    expect(result).toContain("234");
  });

  it("handles zero", () => {
    expect(formatCurrency(0, 2, "US")).toBeDefined();
  });

  it("handles negative values", () => {
    const result = formatCurrency(-100, 2, "US");
    expect(result).toBeDefined();
  });
});
