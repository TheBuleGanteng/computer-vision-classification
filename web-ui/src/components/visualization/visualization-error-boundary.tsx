'use client';

import React from 'react';

interface VisualizationErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

interface VisualizationErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export class VisualizationErrorBoundary extends React.Component<
  VisualizationErrorBoundaryProps,
  VisualizationErrorBoundaryState
> {
  constructor(props: VisualizationErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error): Partial<VisualizationErrorBoundaryState> {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log the error to console with detailed information
    console.error('3D Visualization Error Caught by Error Boundary:', error);
    console.error('Component Stack:', errorInfo.componentStack);
    
    this.setState({
      error,
      errorInfo
    });
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI for 3D visualization errors
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="w-full h-full flex items-center justify-center bg-gray-900 rounded-lg">
          <div className="text-center text-red-400 max-w-md">
            <div className="text-6xl mb-4">🔧</div>
            <p className="text-lg font-semibold mb-2">3D Visualization Error</p>
            <p className="text-sm text-gray-300 mb-4">
              The 3D model visualization encountered an error and couldn't render.
            </p>
            <details className="text-xs text-gray-400 mb-4 text-left">
              <summary className="cursor-pointer hover:text-gray-300">
                Error Details (click to expand)
              </summary>
              <pre className="mt-2 p-2 bg-gray-800 rounded overflow-auto max-h-32">
                {this.state.error?.message || 'Unknown error'}
                {this.state.error?.stack && '\n\nStack trace:\n' + this.state.error.stack}
              </pre>
            </details>
            <button
              onClick={this.handleReset}
              className="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Functional wrapper for easier use
export const WithVisualizationErrorBoundary: React.FC<{
  children: React.ReactNode;
  fallback?: React.ReactNode;
}> = ({ children, fallback }) => (
  <VisualizationErrorBoundary fallback={fallback}>
    {children}
  </VisualizationErrorBoundary>
);