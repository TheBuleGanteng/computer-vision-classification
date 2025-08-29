import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Configure for both Webpack and Turbopack
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Handle cytoscape-elk worker files
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
      };

      // Handle ELK.js web worker imports
      config.module.rules.push({
        test: /elk-worker\.(js|min\.js)$/,
        type: 'asset/resource',
        generator: {
          filename: 'static/chunks/[name].[hash][ext]',
        },
      });
    }
    
    return config;
  },
  
  // Turbopack configuration
  turbopack: {
    // Configuration for handling cytoscape-elk modules
    resolveAlias: {},
  },
  
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
