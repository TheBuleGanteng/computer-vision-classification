import type { NextConfig } from "next";

// Detect if running on GCP VM vs local development
// GCP_DEPLOYMENT: true = use WEB_BASEPATH, false = use root path
const isGCPDeployment = process.env.GCP_DEPLOYMENT === 'true';
const basePath = isGCPDeployment && process.env.WEB_BASEPATH ? process.env.WEB_BASEPATH : undefined;

const nextConfig: NextConfig = {
  // Base path for serving from subpath on GCP, undefined for local development
  basePath: basePath,

  // Asset prefix must match basePath for static files to work correctly
  assetPrefix: basePath,

  // Disable image optimization when using basePath (workaround for production)
  images: {
    unoptimized: true,
  },

  // Configure for both Webpack and Turbopack
  webpack: (config, { isServer }) => {
    // Exclude .legacy files from build
    config.plugins = config.plugins || [];
    config.plugins.push(
      new (require('webpack').IgnorePlugin)({
        resourceRegExp: /\.legacy\.(tsx?|jsx?)$/
      })
    );

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
