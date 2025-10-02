import type { NextConfig } from "next";

// Detect if running on GCP VM vs local development
// GCP_DEPLOYMENT: true = use WEB_BASEPATH, false = use root path
const isGCPDeployment = process.env.GCP_DEPLOYMENT === 'true';
const basePath = isGCPDeployment ? (process.env.WEB_BASEPATH || '') : '';

const nextConfig: NextConfig = {
  // Base path for serving from subpath on GCP, empty for local development
  basePath: basePath,

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
  // Turbopack handles Node.js polyfills automatically, no need to explicitly disable them
  turbopack: {},
  
  async rewrites() {
    // Use internal Docker hostname in containerized mode, localhost for local dev
    const backendUrl = process.env.BACKEND_INTERNAL_URL || 'http://localhost:8000';

    return [
      // TensorBoard proxy routes should be handled by Next.js API routes
      {
        source: '/api/tensorboard/:path*',
        destination: '/api/tensorboard/:path*'
      },
      // All other API routes go to FastAPI backend
      {
        source: '/api/:path*',
        destination: `${backendUrl}/:path*`
      }
    ];
  }
};

export default nextConfig;
