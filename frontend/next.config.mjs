/** @type {import('next').NextConfig} */
const backendUrl = process.env.BACKEND_API_URL || "http://localhost:8000";

const nextConfig = {
  // Enable React strict mode for better development
  reactStrictMode: true,

  // API proxy to backend
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },

  // Environment variables exposed to browser
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || backendUrl,
  },
};

export default nextConfig;
