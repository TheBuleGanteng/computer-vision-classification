"use client"

import React, { useState, useCallback, useRef, useMemo, Suspense, useEffect } from 'react';
import { Activity, Zap, Brain, Target, Skull, TrendingUp, LineChart, Network, Download, ExternalLink, Maximize2, Minimize2 } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import TensorBoardPanel from './tensorboard-panel';

// Lazy load the heavy ModelGraph component to prevent blocking renders
const ModelGraph = React.lazy(() => import('./model-graph'));
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import FullscreenPopup from './fullscreen-popup';
import { logger } from '@/lib/logger';

// Use the same API base URL as the main api client
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TensorBoardLog {
  trial_directory: string;
  trial_name: string;
  log_files: Array<{
    file_path: string;
    file_name: string;
    size_bytes: number;
    modified: string;
  }>;
}

interface TensorBoardData {
  job_id: string;
  tensorboard_logs: TensorBoardLog[];
  total_trials: number;
  base_log_directory: string;
}

interface MetricsTabsProps {
  jobId: string;
  trialId?: string;
  className?: string;
  onExpandClick?: () => void;
}

const MetricsTabs: React.FC<MetricsTabsProps> = React.memo(({
  jobId,
  trialId,
  className = "",
  onExpandClick
}) => {
  const [activeTab, setActiveTab] = useState<string>('model_architecture');
  const [showArchitecturePopup, setShowArchitecturePopup] = useState(false);
  const tabModelGraphRef = useRef<{ exportToPNG: () => Promise<Blob | null> } | null>(null);
  const lastRenderTime = useRef<number>(0);
  const [tensorboardPreloaded, setTensorboardPreloaded] = useState(false);

  // Preload TensorBoard once logs are ready so it's warm when user clicks
  useEffect(() => {
    if (tensorboardPreloaded) return;

    const checkAndPreload = async () => {
      try {
        const status = await fetch(`${API_BASE_URL}/jobs/${jobId}/tensorboard/url`);
        const statusData = await status.json();

        // Once logs are ready and server is running, prefetch TensorBoard to warm it up
        if (statusData.logs_ready && statusData.running && !tensorboardPreloaded) {
          const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
          const tensorboardUrl = basePath
            ? `${basePath}/api/tensorboard/${jobId}`
            : statusData.tensorboard_url;

          // Make a background request to warm up TensorBoard
          fetch(tensorboardUrl, { mode: 'no-cors' }).catch(() => {
            // Ignore errors - this is just a prefetch
          });

          setTensorboardPreloaded(true);
        }
      } catch (error) {
        // Ignore errors - this is optional optimization
      }
    };

    // Check every 5 seconds until preloaded
    const interval = setInterval(checkAndPreload, 5000);
    checkAndPreload(); // Check immediately too

    return () => clearInterval(interval);
  }, [jobId, tensorboardPreloaded]);

  // Throttle heavy renders to prevent setTimeout violations
  const shouldRender = useMemo(() => {
    const now = Date.now();
    const timeSinceLastRender = now - lastRenderTime.current;
    const isMobile = typeof window !== 'undefined' && window.innerWidth <= 768;
    const minRenderInterval = isMobile ? 500 : 250; // Throttle more aggressively on mobile
    
    if (timeSinceLastRender >= minRenderInterval) {
      lastRenderTime.current = now;
      return true;
    }
    return false;
  }, []); // Dependencies removed as computation doesn't depend on these values
  
  // Download function for Model Architecture tab using Save As dialog
  const downloadArchitecturePNG = useCallback(async () => {
    try {
      if (!tabModelGraphRef.current?.exportToPNG) {
        logger.error('Tab ModelGraph export function not available');
        return;
      }
      const pngBlob = await tabModelGraphRef.current.exportToPNG();
      if (!pngBlob) {
        logger.error('Failed to generate PNG blob');
        return;
      }

      const defaultFilename = `model_architecture_${jobId}_${trialId || 'latest'}_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;

      // Use File System Access API for modern browsers with Save As dialog
      if ('showSaveFilePicker' in window) {
        try {
          const fileHandle = await (window as unknown as { showSaveFilePicker: (options: {
            suggestedName: string;
            types: Array<{ description: string; accept: Record<string, string[]> }>;
          }) => Promise<{ createWritable: () => Promise<{ write: (data: Blob) => Promise<void>; close: () => Promise<void> }> }> }).showSaveFilePicker({
            suggestedName: defaultFilename,
            types: [{
              description: 'PNG images',
              accept: { 'image/png': ['.png'] },
            }],
          });
          const writable = await fileHandle.createWritable();
          await writable.write(pngBlob);
          await writable.close();
          logger.log('Architecture PNG saved successfully');
          return;
        } catch (saveError: unknown) {
          if ((saveError as Error).name === 'AbortError') {
            logger.log('Save dialog was cancelled by user');
            return;
          }
          logger.warn('Save As dialog failed, falling back to download:', saveError);
        }
      }

      // Fallback for browsers that don't support File System Access API
      const url = URL.createObjectURL(pngBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = defaultFilename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      logger.log('Architecture PNG downloaded successfully (fallback method)');
    } catch (error) {
      logger.error('Failed to download architecture PNG:', error);
    }
  }, [jobId, trialId]);

  // Optimized TensorBoard logs fetching with React Query
  const { isLoading: loading } = useQuery({
    queryKey: ['tensorboard-logs', jobId],
    queryFn: async (): Promise<TensorBoardData> => {
      const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/tensorboard/logs`);
      if (!response.ok) {
        throw new Error('Failed to load TensorBoard logs');
      }
      return response.json();
    },
    enabled: !!jobId,
    staleTime: 30000, // 30 seconds
    refetchInterval: false, // Don't auto-refetch
  });

  // Architecture data fetching for Model Architecture tab - using same endpoint as main section
  const { data: architectureResponse, isLoading: architectureLoading } = useQuery({
    queryKey: ['cytoscape-architecture-tab', jobId, trialId],
    queryFn: async () => {
      if (!jobId) return null;
      
      const url = trialId 
        ? `${API_BASE_URL}/jobs/${jobId}/cytoscape/architecture?trial_id=${trialId}`
        : `${API_BASE_URL}/jobs/${jobId}/cytoscape/architecture`;
      
      const response = await fetch(url);
      if (!response.ok) {
        logger.warn('Architecture data not available:', response.status);
        return null;
      }
      return response.json();
    },
    enabled: !!jobId && activeTab === 'model_architecture',
    staleTime: 60000, // 1 minute
    refetchInterval: false,
  });
  
  const architectureData = architectureResponse?.cytoscape_data;

  const tabs = [
    {
      id: 'model_architecture',
      label: 'Model Architecture',
      icon: <Network className="w-4 h-4" />,
      description: 'Interactive neural network structure visualization'
    },
    {
      id: 'training_progress',
      label: 'Training Progress',
      icon: <LineChart className="w-4 h-4" />,
      description: 'Loss and accuracy curves with overfitting detection'
    },
    {
      id: 'weights_bias',
      label: 'Weights + Bias',
      icon: <Activity className="w-4 h-4" />,
      description: 'Weight distributions and parameter health analysis'
    },
    {
      id: 'activation_maps',
      label: 'Activation Maps',
      icon: <Brain className="w-4 h-4" />,
      description: 'CNN layer activations and filter visualizations'
    },
    {
      id: 'confusion_matrix',
      label: 'Confusion Matrix',
      icon: <Target className="w-4 h-4" />,
      description: 'Classification accuracy and error analysis'
    },
    {
      id: 'dead_neuron_analysis',
      label: 'Dead Neurons',
      icon: <Skull className="w-4 h-4" />,
      description: 'Dead neuron detection and analysis'
    },
    {
      id: 'gradient_flow',
      label: 'Gradient Flow',
      icon: <Zap className="w-4 h-4" />,
      description: 'Gradient flow analysis'
    },
    {
      id: 'gradient_distributions',
      label: 'Gradient Distrib.',
      icon: <TrendingUp className="w-4 h-4" />,
      description: 'Gradient distribution patterns'
    },
    {
      id: 'activation_summary',
      label: 'Activation Summary',
      icon: <Brain className="w-4 h-4" />,
      description: 'Activation pattern summary and analysis'
    },
    {
      id: 'activation_progression',
      label: 'Activation Progression',
      icon: <TrendingUp className="w-4 h-4" />,
      description: 'Layer-by-layer activation progression analysis'
    }
  ];


  if (loading || (activeTab === 'model_architecture' && architectureLoading)) {
    return (
      <div className={`flex items-center justify-center h-96 ${className}`}>
        <div className="text-center">
          <div className="text-gray-400 text-lg mb-2">Loading TensorBoard Data</div>
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={`h-full flex flex-col ${className}`}>
      {/* Tab Headers */}
      <div className="flex flex-wrap justify-start sm:justify-center gap-1 sm:gap-0 bg-gray-800 border-b border-gray-600 pt-2 px-2 sm:px-0 overflow-x-visible scrollbar-hide">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1 px-2 sm:px-3 py-1 text-xs font-medium transition-all duration-200 relative
              ${activeTab === tab.id
                ? 'text-blue-300 bg-gray-900 border-t-2 border-l border-r border-blue-400 border-b-0 rounded-t-md -mb-px z-10'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/50 border border-gray-500 rounded-t-md'
              }`}
          >
            {tab.icon}
            <span className="hidden sm:inline">{tab.label}</span>
            <span className="sm:hidden text-xs truncate max-w-16" title={tab.label}>
              {tab.label.split(' ')[0]}
            </span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1">
        {activeTab === 'model_architecture' ? (
          <div className="h-full">
            <Card className="h-full bg-gray-900 border-gray-700">
              <CardHeader className="pb-0">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-base font-semibold flex items-center gap-1">
                      <Network className="w-4 h-4" />
                      Model Architecture
                    </h3>
                    <p className="text-xs text-muted-foreground">
                      {architectureData?.metadata ? (
                        `${architectureData.metadata.architecture_type} Architecture - ${architectureData.metadata.total_parameters.toLocaleString()} parameters`
                      ) : (
                        'Interactive neural network structure visualization'
                      )}
                    </p>
                  </div>
                  <div className="flex flex-col gap-1 w-24">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={async () => {
                        // Check if TensorBoard is running and logs are ready
                        const status = await fetch(`${API_BASE_URL}/jobs/${jobId}/tensorboard/url`);
                        const statusData = await status.json();

                        // Don't open if logs aren't ready yet
                        if (!statusData.logs_ready) {
                          alert('TensorBoard logs are still downloading. Please wait a moment and try again.');
                          return;
                        }

                        if (!statusData.running) {
                          // Start TensorBoard first
                          await fetch(`${API_BASE_URL}/jobs/${jobId}/tensorboard/start`, { method: 'POST' });
                          // Wait longer for TensorBoard to fully initialize and be ready to serve requests
                          await new Promise(resolve => setTimeout(resolve, 3000));
                          // Fetch updated status to get the URL
                          const updatedStatus = await fetch(`${API_BASE_URL}/jobs/${jobId}/tensorboard/url`);
                          const updatedData = await updatedStatus.json();

                          // Use proxy route only for GCP production (not local containerized)
                          const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
                          const tensorboardUrl = basePath
                            ? `${basePath}/api/tensorboard/${jobId}`
                            : updatedData.tensorboard_url;
                          window.open(tensorboardUrl, '_blank');
                        } else {
                          // Use proxy route only for GCP production (not local containerized)
                          const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
                          const tensorboardUrl = basePath
                            ? `${basePath}/api/tensorboard/${jobId}`
                            : statusData.tensorboard_url;
                          window.open(tensorboardUrl, '_blank');
                        }
                      }}
                      className="flex items-center justify-center gap-1 w-full h-6 text-xs px-2 py-1"
                      title="Open TensorBoard interface"
                    >
                      <ExternalLink className="w-2 h-2" />
                      TensorBoard
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={downloadArchitecturePNG}
                      className="flex items-center justify-center gap-1 w-full h-6 text-xs px-2 py-1"
                      disabled={!architectureData}
                      title="Download architecture as PNG"
                    >
                      <Download className="w-2 h-2" />
                      Download
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-0 h-[450px]">
                {!architectureData ? (
                  <div className="flex items-center justify-center h-full text-center">
                    <div>
                      <div className="text-gray-400 text-lg mb-2">No Architecture Data</div>
                      <div className="text-gray-500 text-sm">
                        Architecture visualization will appear when trial data is available
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-full w-full group relative">
                    {shouldRender ? (
                      <Suspense fallback={
                        <div className="h-full w-full flex items-center justify-center bg-gray-50">
                          <div className="text-center">
                            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto mb-2"></div>
                            <div className="text-gray-400 text-sm">Loading model...</div>
                          </div>
                        </div>
                      }>
                        <ModelGraph
                          architectureData={architectureData}
                          className="h-full"
                          showLegend={false}
                          ref={tabModelGraphRef}
                        />
                      </Suspense>
                    ) : (
                      <div className="h-full w-full flex items-center justify-center bg-gray-50">
                        <div className="text-gray-400 text-sm">Rendering...</div>
                      </div>
                    )}
                    {/* Expand icon in top-right corner */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setShowArchitecturePopup(true);
                      }}
                      className="absolute top-2 right-2 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10"
                      title="Expand visualization"
                    >
                      <Maximize2 className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        ) : (
          tabs.filter(tab => tab.id !== 'model_architecture').map((tab) => (
            activeTab === tab.id && (
              <div key={tab.id} className="h-full">
                <TensorBoardPanel 
                  jobId={jobId} 
                  trialId={trialId} 
                  height={500}
                  onExpandClick={onExpandClick}
                  defaultPlotType={tab.id}
                />
              </div>
            )
          ))
        )}
      </div>

      {/* Architecture Fullscreen Popup */}
      <FullscreenPopup
        isOpen={showArchitecturePopup}
        onClose={() => setShowArchitecturePopup(false)}
        title="Model Architecture - Full View"
      >
        <div className="h-[80vh] p-4">
          {architectureData ? (
            <div className="h-full w-full group relative">
              {shouldRender ? (
                <Suspense fallback={
                  <div className="h-full w-full flex items-center justify-center bg-gray-50">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
                      <div className="text-gray-400">Loading architecture...</div>
                    </div>
                  </div>
                }>
                  <ModelGraph
                    architectureData={architectureData}
                    className="h-full"
                    showLegend={false}
                  />
                </Suspense>
              ) : (
                <div className="h-full w-full flex items-center justify-center bg-gray-50">
                  <div className="text-gray-400">Rendering architecture...</div>
                </div>
              )}
              {/* Minimize icon in top-right corner */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowArchitecturePopup(false);
                }}
                className="absolute top-2 right-2 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-20"
                title="Minimize architecture"
              >
                <Minimize2 className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-center">
              <div>
                <div className="text-gray-400 text-lg mb-2">No Architecture Data</div>
                <div className="text-gray-500 text-sm">
                  Architecture visualization will appear when trial data is available
                </div>
              </div>
            </div>
          )}
        </div>
      </FullscreenPopup>
    </div>
  );
});

MetricsTabs.displayName = 'MetricsTabs'

export default MetricsTabs;