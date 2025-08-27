'use client';

import React from 'react';
import { Model3DViewer } from './model-3d-viewer';
import { useModelVisualization } from '@/hooks/use-model-visualization';
import { TrialData } from '@/types/optimization';

interface Model3DVisualizationContainerProps {
  jobId: string | null;
  selectedTrial: TrialData | null;
}

export const Model3DVisualizationContainer: React.FC<Model3DVisualizationContainerProps> = ({
  jobId,
  selectedTrial
}) => {
  // Only fetch visualization data if we have a jobId and selected trial
  const shouldFetch = Boolean(jobId && selectedTrial?.status === 'completed');
  
  const {
    data: visualizationData,
    isLoading,
    error
  } = useModelVisualization(
    shouldFetch ? jobId! : null,
    {
      enabled: shouldFetch,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    }
  );

  // Show loading state
  if (!jobId || !selectedTrial) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900 rounded-lg">
        <div className="text-center text-gray-400">
          <div className="text-6xl mb-4">📊</div>
          <p className="text-lg">No trial selected</p>
          <p className="text-sm mt-2">Select a completed trial to view 3D model</p>
        </div>
      </div>
    );
  }

  if (selectedTrial.status !== 'completed') {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900 rounded-lg">
        <div className="text-center text-gray-400">
          <div className="text-6xl mb-4">⏳</div>
          <p className="text-lg">Trial in progress</p>
          <p className="text-sm mt-2">3D visualization available after completion</p>
        </div>
      </div>
    );
  }

  // Handle case where visualization data is not available yet
  if (error && error.includes('404')) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900 rounded-lg">
        <div className="text-center text-gray-400">
          <div className="text-6xl mb-4">⏳</div>
          <p className="text-lg">Generating visualization...</p>
          <p className="text-sm mt-2">3D model will appear shortly</p>
        </div>
      </div>
    );
  }

  return (
    <Model3DViewer
      visualizationData={visualizationData}
      isLoading={isLoading}
      error={error}
      className="w-full h-full"
    />
  );
};