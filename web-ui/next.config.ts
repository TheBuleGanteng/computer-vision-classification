import type { NextConfig } from "next";

// Detect if running on GCP VM vs local development
// GCP_DEPLOYMENT: true = use WEB_BASEPATH, false = use root path
const isGCPDeployment = process.env.GCP_DEPLOYMENT === 'true';
const basePath = isGCPDeployment && process.env.WEB_BASEPATH ? process.env.WEB_BASEPATH : undefined;

// Content-Security-Policy in Report-Only mode: enforces nothing, only reports
// violations. Broad union allow-list enumerated from what the UI actually loads
// (self-hosted next/font, same-origin API via rewrites, the TensorBoard subdomain
// for live-training views, cytoscape-elk web workers, canvas/blob image exports).
// Promote to an enforcing `Content-Security-Policy` header once reports are clean.
const cspReportOnly = [
  "default-src 'self'",
  "base-uri 'self'",
  "object-src 'none'",
  "frame-ancestors 'none'",
  // Next.js injects an inline bootstrap script; dev/Turbopack needs eval.
  "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  // connect-src: same-origin API (path-routed through nginx / Next rewrites) plus
  // the TensorBoard subdomain and its websocket for live updates.
  "connect-src 'self' https://tensorboard.kebayorantechnologies.com wss://tensorboard.kebayorantechnologies.com",
  // cytoscape-elk runs layout in a web worker loaded as a blob.
  "worker-src 'self' blob:",
  // TensorBoard is embedded in an iframe in the dashboard.
  "frame-src 'self' https://tensorboard.kebayorantechnologies.com",
  "form-action 'self'",
].join("; ");

// Browser security headers applied to every response. HSTS is intentionally
// omitted — it is terminated/owned at the edge (nginx + Let's Encrypt) and a
// second header here could conflict.
const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  // The demo trains on pre-bundled datasets — no webcam/getUserMedia, no geo/mic.
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "Content-Security-Policy-Report-Only", value: cspReportOnly },
];

const nextConfig: NextConfig = {
  // Base path for serving from subpath on GCP, undefined for local development
  basePath: basePath,

  // Asset prefix must match basePath for static files to work correctly
  assetPrefix: basePath,

  // Remove the X-Powered-By: Next.js version-disclosure header
  poweredByHeader: false,

  // Apply browser security headers to all routes
  async headers() {
    return [
      {
        source: "/:path*",
        headers: securityHeaders,
      },
    ];
  },

  // Disable image optimization when using basePath (workaround for production)
  images: {
    unoptimized: true,
  },

  // Configure for both Webpack and Turbopack
  webpack: (config, { isServer }) => {
    // Exclude .legacy files from build
    config.plugins = config.plugins || [];
    config.plugins.push(
      // webpack is provided transitively by Next.js; require avoids adding a
      // direct dependency just for its types. This branch only runs under the
      // webpack fallback — the Turbopack build ignores it.
      // eslint-disable-next-line @typescript-eslint/no-require-imports
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
