'use client';

import React from 'react';
import { Model3DViewer } from './model-3d-viewer';
import { useModelVisualization } from '@/hooks/use-model-visualization';

interface BestModelVisualizationProps {
  jobId: string;
}

export const BestModelVisualization: React.FC<BestModelVisualizationProps> = ({
  jobId
}) => {
  const {
    data: bestModelResponse,
    isLoading,
    error
  } = useModelVisualization(jobId, true);

  // Extract visualization data from the best model response
  const visualizationData = bestModelResponse?.visualization_data;

  // Handle loading state
  if (isLoading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-300">Loading best model visualization...</p>
        </div>
      </div>
    );
  }

  // Handle 404 - visualization not ready yet
  if (error && error.includes('404')) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900 rounded-lg">
        <div className="text-center text-gray-400">
          <div className="text-6xl mb-4">⏳</div>
          <p className="text-lg">Generating visualization...</p>
          <p className="text-sm mt-2">3D model will appear once processing completes</p>
        </div>
      </div>
    );
  }

  // Handle other errors
  if (error) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900 rounded-lg">
        <div className="text-center text-red-400">
          <div className="text-6xl mb-4">⚠️</div>
          <p className="text-lg">Failed to load visualization</p>
          <p className="text-sm mt-2">{error}</p>
        </div>
      </div>
    );
  }

  // Render the 3D visualization
  return (
    <Model3DViewer
      visualizationData={visualizationData}
      isLoading={false}
      error={undefined}
      className="w-full h-full"
    />
  );
};