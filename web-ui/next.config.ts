import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      // TensorBoard proxy routes should be handled by Next.js API routes
      {
        source: '/api/tensorboard/:path*',
        destination: '/api/tensorboard/:path*'
      },
      // All other API routes go to FastAPI backend
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/:path*'
      }
    ];
  }
};

export default nextConfig;
