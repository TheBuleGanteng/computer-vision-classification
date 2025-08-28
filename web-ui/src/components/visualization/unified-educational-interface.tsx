"use client"

import React, { useState } from 'react';
import { LayoutGrid, Maximize2, Minimize2 } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { Badge } from '@/components/ui/badge';
import ModelGraph from './model-graph';
import LayerInfoPanel from './layer-info-panel-new';
import MetricsTabs from './metrics-tabs';
import FullscreenPopup from './fullscreen-popup';
import TensorBoardFullscreen from './tensorboard-fullscreen';

// Use the same API base URL as the main api client
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface LayerData {
  id: string;
  type: string;
  label: string;
  parameters?: number;
  color_intensity?: number;
  opacity?: number;
  units?: number;
  filters?: number;
  kernel_size?: number[];
  pool_size?: number[];
  activation?: string;
  dropout_rate?: number;
}

interface CytoscapeData {
  nodes: Array<{ data: LayerData }>;
  edges: Array<{ data: { source: string; target: string; tensor_transform: string } }>;
  metadata?: {
    architecture_type: string;
    total_parameters: number;
  };
}

interface PlotInfo {
  plot_type: string;
  filename: string;
  url: string;
  size_bytes: number;
  created_time: number;
}

interface TrialPlotsData {
  job_id: string;
  trial_id: string;
  plots: PlotInfo[];
  total_plots: number;
}

interface UnifiedEducationalInterfaceProps {
  jobId: string;
  trialId?: string;
  className?: string;
}

type LayoutMode = 'split' | 'architecture-focus' | 'metrics-focus';

// Layer type definitions for dynamic legend - matches model-graph.tsx styling
interface LayerTypeDefinition {
  color: string;
  shape: 'rounded-full' | 'rounded' | 'rounded-lg';
  label: string;
  order: number;
}

const LAYER_TYPE_DEFINITIONS: Record<string, LayerTypeDefinition> = {
  'input': {
    color: 'bg-green-600', // Matches #059669 from model-graph.tsx
    shape: 'rounded-full',
    label: 'Input',
    order: 1
  },
  'conv2d': {
    color: 'bg-blue-500', // Matches #3B82F6 from model-graph.tsx
    shape: 'rounded',
    label: 'Conv2D',
    order: 2
  },
  'conv': {
    color: 'bg-blue-500', // Matches #3B82F6 from model-graph.tsx
    shape: 'rounded',
    label: 'Conv2D',
    order: 2
  },
  'pooling': {
    color: 'bg-cyan-500', // Changed from emerald to cyan for better distinction from input
    shape: 'rounded-full',
    label: 'Pooling',
    order: 3
  },
  'pool': {
    color: 'bg-cyan-500', // Changed from emerald to cyan for better distinction from input
    shape: 'rounded-full',
    label: 'Pooling',
    order: 3
  },
  'flatten': {
    color: 'bg-gray-500', // Default for flatten - future enhancement
    shape: 'rounded',
    label: 'Flatten',
    order: 3.5
  },
  'lstm': {
    color: 'bg-purple-500', // Matches #8B5CF6 from model-graph.tsx
    shape: 'rounded',
    label: 'LSTM',
    order: 4
  },
  'dense': {
    color: 'bg-amber-500', // Matches #F59E0B from model-graph.tsx
    shape: 'rounded',
    label: 'Dense',
    order: 5
  },
  'dropout': {
    color: 'bg-red-500', // Matches #EF4444 from model-graph.tsx
    shape: 'rounded-lg',
    label: 'Dropout',
    order: 6
  },
  'activation': {
    color: 'bg-lime-500', // Matches #84CC16 from model-graph.tsx
    shape: 'rounded-lg',
    label: 'Activation',
    order: 7
  },
  'output': {
    color: 'bg-red-600', // Matches #DC2626 from model-graph.tsx
    shape: 'rounded-full',
    label: 'Output',
    order: 10
  }
};

// Default style for unknown layer types
const DEFAULT_LAYER_TYPE: LayerTypeDefinition = {
  color: 'bg-gray-400',
  shape: 'rounded',
  label: 'Unknown',
  order: 99
};

// Function to generate dynamic legend based on actual layers present
const generateDynamicLegend = (architectureData: CytoscapeData | null): LayerTypeDefinition[] => {
  if (!architectureData?.nodes || architectureData.nodes.length === 0) {
    return [];
  }
  
  // Extract unique layer types from the architecture data
  const presentTypes = new Set<string>();
  architectureData.nodes.forEach(node => {
    const layerType = node.data.type?.toLowerCase() || 'unknown';
    presentTypes.add(layerType);
  });
  
  // Map to legend definitions
  const legendItems: LayerTypeDefinition[] = [];
  presentTypes.forEach(type => {
    const definition = LAYER_TYPE_DEFINITIONS[type] || {
      ...DEFAULT_LAYER_TYPE,
      label: type.charAt(0).toUpperCase() + type.slice(1)
    };
    legendItems.push(definition);
  });
  
  // Sort by flow order (input → conv → pool → lstm → dense → dropout → output)
  // then alphabetically by label for consistency
  return legendItems.sort((a, b) => {
    if (a.order !== b.order) return a.order - b.order;
    return a.label.localeCompare(b.label);
  });
};

// Dynamic Legend Component
const DynamicLegend: React.FC<{ architectureData: CytoscapeData | null }> = React.memo(({ architectureData }) => {
  const legendItems = generateDynamicLegend(architectureData);
  
  if (legendItems.length === 0) {
    return (
      <div className="text-xs text-gray-400">
        No layers to display
      </div>
    );
  }
  
  return (
    <div className="z-10 bg-gray-800 bg-opacity-90 rounded-lg p-0 pt-1">
      <div className="text-xs text-gray-300 mb-2 font-semibold">Layer Types</div>
      <div className="flex flex-wrap gap-2 text-xs">
        {legendItems.map((item) => (
          <div key={item.label} className="flex items-center gap-1">
            <div className={`w-3 h-3 ${item.color} ${item.shape}`}></div>
            <span className="text-gray-300">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
});

DynamicLegend.displayName = 'DynamicLegend';

const UnifiedEducationalInterface: React.FC<UnifiedEducationalInterfaceProps> = React.memo(({
  jobId,
  trialId,
  className = ""
}) => {
  const [layout, setLayout] = useState<LayoutMode>('split');
  const [selectedLayer, setSelectedLayer] = useState<LayerData | null>(null);
  const [showArchitecturePopup, setShowArchitecturePopup] = useState(false);
  const [showMetricsPopup, setShowMetricsPopup] = useState(false);

  // Use React Query for optimized data fetching with caching and deduplication
  const {
    data: architectureResponse,
    isLoading: loading,
    error: queryError
  } = useQuery({
    queryKey: ['cytoscape-architecture', jobId, trialId],
    queryFn: async () => {
      if (!jobId) return null;
      
      const url = trialId 
        ? `${API_BASE_URL}/jobs/${jobId}/cytoscape/architecture?trial_id=${trialId}`
        : `${API_BASE_URL}/jobs/${jobId}/cytoscape/architecture`;
        
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error('Failed to load architecture data');
      }
      
      return response.json();
    },
    enabled: !!jobId,
    staleTime: 30000, // 30 seconds - architecture doesn't change often
    refetchInterval: false, // Don't auto-refetch, let parent components handle updates
    refetchOnWindowFocus: false,
    placeholderData: (previousData) => previousData, // Keep previous data while refetching
  });

  const architectureData = architectureResponse?.cytoscape_data;
  const error = queryError?.message;

  // Query to get available plots for the badge
  const { data: plotsData } = useQuery({
    queryKey: ['trial-plots', jobId, trialId],
    queryFn: async (): Promise<TrialPlotsData | null> => {
      const endpoint = trialId 
        ? `${API_BASE_URL}/jobs/${jobId}/plots/${trialId}`
        : `${API_BASE_URL}/jobs/${jobId}/plots`;
      
      const response = await fetch(endpoint);
      if (!response.ok) {
        if (response.status === 404) {
          return null; // No plots available yet
        }
        throw new Error('Failed to get plots data');
      }
      const data = await response.json();
      return data;
    },
    enabled: !!jobId,
    staleTime: 30000, // 30 seconds
    refetchInterval: 10000, // Check every 10 seconds for new plots
  });

  // Determine which plots are available
  const availablePlots = plotsData?.plots || [];
  const hasPlots = availablePlots.length > 0;

  // Debug logging
  React.useEffect(() => {
    if (architectureData) {
      console.log('Architecture data received:', architectureData);
      console.log('Nodes:', architectureData.nodes?.length);
      console.log('Edges:', architectureData.edges?.length);
    }
  }, [architectureData]);

  const layoutOptions = [
    {
      id: 'split' as LayoutMode,
      label: 'Split View',
      icon: <LayoutGrid className="w-4 h-4" />,
      description: 'Architecture and metrics side by side'
    },
    {
      id: 'architecture-focus' as LayoutMode,
      label: 'Architecture Focus',
      icon: <Maximize2 className="w-4 h-4" />,
      description: 'Large architecture view with metrics sidebar'
    },
    {
      id: 'metrics-focus' as LayoutMode,
      label: 'Metrics Focus',
      icon: <Minimize2 className="w-4 h-4" />,
      description: 'Large metrics view with architecture sidebar'
    }
  ];

  if (loading) {
    return (
      <div className={`flex items-center justify-center h-96 bg-gray-900 rounded-lg ${className}`}>
        <div className="text-center">
          <div className="text-gray-400 text-lg mb-2">Loading Educational Interface</div>
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`flex items-center justify-center h-96 bg-gray-900 rounded-lg ${className}`}>
        <div className="text-center">
          <div className="text-red-400 text-lg mb-2">Error Loading Interface</div>
          <div className="text-gray-500 text-sm">{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-[600px] flex flex-col bg-gray-900 ${className}`}>
      {/* Header with Layout Controls */}
      <div className="flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700">
        <div>
          <h2 className="text-lg font-semibold text-white">Educational Model Interface</h2>
          <p className="text-sm text-gray-400">
            Job: <code className="font-mono">{jobId}</code>
            {trialId && <span> • Trial: <code className="font-mono">{trialId}</code></span>}
            {hasPlots && <span> • Updated: <code className="font-mono">{new Date().toLocaleTimeString()}</code></span>}
          </p>
        </div>

        {/* Layout Selector */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400 mr-2">Layout:</span>
          <div className="flex rounded-lg bg-gray-700 p-1">
            {layoutOptions.map((option) => (
              <button
                key={option.id}
                onClick={() => setLayout(option.id)}
                className={`flex items-center gap-2 px-3 py-1.5 text-xs rounded-md transition-colors
                  ${layout === option.id
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:text-white hover:bg-gray-600'
                  }`}
                title={option.description}
              >
                {option.icon}
                <span className="hidden sm:inline">{option.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      {layout === 'split' ? (
        <div className="flex flex-1 min-h-[200px]">
          {/* Left Panel - Architecture */}
          <div className="w-1/2 border-r border-gray-700 relative">
            <div className="bg-gray-800 px-4 py-2 border-b border-gray-700 flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-white">Model Architecture</h3>
                {architectureData?.metadata ? (
                  <div className="text-gray-400 text-xs">
                    {architectureData.metadata.architecture_type} Architecture - {architectureData.metadata.total_parameters.toLocaleString()} parameters
                    <DynamicLegend architectureData={architectureData} />
                  </div>
                ) : (
                  <p className="text-gray-400 text-sm">
                    Interactive neural network structure
                  </p>
                )}
              </div>
              
            </div>
            <div className="h-[480px] relative bg-gray-900">
              {loading ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-white text-center">
                    <div className="text-lg mb-2">Loading Architecture...</div>
                    <div className="text-sm text-gray-400">Fetching model data</div>
                  </div>
                </div>
              ) : error ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-red-400 text-center">
                    <div className="text-lg mb-2">Error Loading Architecture</div>
                    <div className="text-sm">{error}</div>
                  </div>
                </div>
              ) : architectureData ? (
                <div className="h-full w-full group relative">
                  <ModelGraph 
                    architectureData={architectureData}
                    onNodeClick={setSelectedLayer}
                    className="h-full w-full"
                  />
                  {/* Expand icon in top-right corner */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowArchitecturePopup(true);
                    }}
                    className="absolute top-2 right-2 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-20"
                    title="Expand architecture"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </button>
                  
                </div>
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-gray-500 text-center">
                    <div className="text-lg mb-2">No Architecture Data</div>
                    <div className="text-sm">Start a trial to see the model architecture</div>
                  </div>
                </div>
              )}
            </div>
            
            {/* Layer Info Overlay */}
            {selectedLayer && (
              <LayerInfoPanel 
                layer={selectedLayer}
                onClose={() => setSelectedLayer(null)}
              />
            )}
            
            {/* Educational Tips Footer - Architecture Only */}
            <div className="absolute bottom-0 left-0 right-0 bg-gray-800 border-t border-gray-700 p-2">
              <div className="flex items-center gap-4 text-xs text-gray-400">
                <span>💡 <strong>Tip:</strong> Click layers to learn about their functions</span>
                <span>🎯 <strong>Animation:</strong> Use &quot;Forward Pass&quot; to see data flow</span>
              </div>
            </div>
          </div>

          {/* Right Panel - Metrics */}
          <div className="w-1/2 relative">
            <div className="bg-gray-800 px-4 py-2 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="font-semibold text-white">Training Metrics & Diagnostics</h3>
                </div>
              </div>
              
              
            </div>
            <div className="group relative">
              <MetricsTabs 
                jobId={jobId} 
                trialId={trialId} 
                className="h-[480px]"
                onExpandClick={() => setShowMetricsPopup(true)}
              />
              
            </div>
          </div>
        </div>
      ) : layout === 'architecture-focus' ? (
        <div className="flex h-full">
          {/* Main Area - Architecture */}
          <div className="flex-1 relative">
            <div className="bg-gray-800 px-4 py-2 border-b border-gray-700">
              <h3 className="font-semibold text-white">Model Architecture (Focus Mode)</h3>
            </div>
            <div className="h-full">
              {architectureData ? (
                <ModelGraph 
                  architectureData={architectureData}
                  onNodeClick={setSelectedLayer}
                  className="h-full"
                />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-gray-500 text-center">
                    <div className="text-lg mb-2">No Architecture Data</div>
                    <div className="text-sm">Start a trial to see the model architecture</div>
                  </div>
                </div>
              )}
            </div>
            
            {selectedLayer && (
              <LayerInfoPanel 
                layer={selectedLayer}
                onClose={() => setSelectedLayer(null)}
              />
            )}
          </div>

          {/* Sidebar - Metrics Summary */}
          <div className="w-80 border-l border-gray-700">
            <MetricsTabs jobId={jobId} trialId={trialId} />
          </div>
        </div>
      ) : (
        <div className="flex h-full">
          {/* Sidebar - Architecture */}
          <div className="w-80 border-r border-gray-700 relative">
            <div className="bg-gray-800 px-4 py-2 border-b border-gray-700">
              <h3 className="font-semibold text-white text-sm">Architecture</h3>
            </div>
            <div className="h-full">
              {architectureData ? (
                <ModelGraph 
                  architectureData={architectureData}
                  onNodeClick={setSelectedLayer}
                  className="h-full"
                />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-gray-500 text-center text-sm">
                    <div className="mb-1">No Data</div>
                    <div className="text-xs">Start a trial</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Main Area - Metrics */}
          <div className="flex-1">
            <div className="bg-gray-800 px-4 py-2 border-b border-gray-700">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="font-semibold text-white">Training Metrics (Focus Mode)</h3>
                  <Badge variant={hasPlots ? "default" : "secondary"}>
                    {hasPlots ? `${availablePlots.length} plots` : "No plots"}
                  </Badge>
                </div>
              </div>
            </div>
            <MetricsTabs jobId={jobId} trialId={trialId} />
          </div>
        </div>
      )}


      {/* Popup Modals */}
      <FullscreenPopup
        isOpen={showArchitecturePopup}
        onClose={() => setShowArchitecturePopup(false)}
        title="Model Architecture - Full View"
      >
        <div className="h-[80vh] p-4">
          {architectureData ? (
            <div className="h-full w-full group relative">
              <ModelGraph 
                architectureData={architectureData}
                onNodeClick={setSelectedLayer}
                className="h-full"
              />
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
            <div className="flex items-center justify-center h-full">
              <div className="text-gray-500 text-center">
                <div className="text-lg mb-2">No Architecture Data</div>
                <div className="text-sm">Load a completed trial to view the architecture</div>
              </div>
            </div>
          )}
        </div>
      </FullscreenPopup>

      <FullscreenPopup
        isOpen={showMetricsPopup}
        onClose={() => setShowMetricsPopup(false)}
        title=""
      >
        <div className="h-[85vh]">
          <TensorBoardFullscreen 
            jobId={jobId} 
            trialId={trialId}
            onClose={() => setShowMetricsPopup(false)}
          />
        </div>
      </FullscreenPopup>
    </div>
  );
});

UnifiedEducationalInterface.displayName = 'UnifiedEducationalInterface'

export { UnifiedEducationalInterface }
export default UnifiedEducationalInterface;