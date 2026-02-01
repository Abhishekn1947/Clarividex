import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"]
});

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#E67E22",
};

export const metadata: Metadata = {
  title: "Clarividex - AI-Powered Market Predictions",
  description:
    "Clarividex - The Clairvoyant Index. Advanced AI analyzes market data to provide intelligent stock predictions. See tomorrow's markets today.",
  keywords: [
    "stock prediction",
    "AI trading",
    "market analysis",
    "sentiment analysis",
    "technical analysis",
    "clarividex",
    "clairvoyant index",
  ],
  icons: {
    icon: "/favicon.ico",
    apple: "/apple-touch-icon.png",
  },
  manifest: "/site.webmanifest",
  openGraph: {
    title: "Clarividex - The Clairvoyant Index",
    description: "AI-Powered Market Predictions. See tomorrow's markets today.",
    siteName: "Clarividex",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
          {children}
        </div>
      </body>
    </html>
  );
}
