"use client"

import React, { useState, useEffect, useMemo } from 'react';
import Image from 'next/image';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
// import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Loader2, Download, ExternalLink, AlertCircle, BarChart3, Activity, Zap, Maximize2, Minimize2, Brain, Target, Play, Skull, TrendingUp, LineChart } from 'lucide-react';

interface TrainingMetricsPanelProps {
  jobId: string;
  trialId?: string;
  height?: number;
  onExpandClick?: () => void;
  isFullscreen?: boolean;
  defaultPlotType?: string;
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

interface TensorBoardConfig {
  tensorboard_url: string;
  port: number;
  status: string;
  running?: boolean;
  job_id?: string;
  pid?: number;
}

export const TrainingMetricsPanel: React.FC<TrainingMetricsPanelProps> = ({ 
  jobId, 
  trialId, 
  onExpandClick,
  isFullscreen = false,
  defaultPlotType
}) => {
  const [activeTab, setActiveTab] = useState<string>(defaultPlotType || 'training_history');

  // Use the same API base URL as other components
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // Query to get available plots for the specific trial or all trials
  const { data: plotsData, isLoading: plotsLoading, error: plotsError } = useQuery({
    queryKey: ['trial-plots', jobId, trialId],
    queryFn: async (): Promise<TrialPlotsData | null> => {
      const endpoint = trialId 
        ? `${API_BASE_URL}/jobs/${jobId}/plots/${trialId}`
        : `${API_BASE_URL}/jobs/${jobId}/plots`;
      
      const response = await fetch(endpoint);
      if (!response.ok) {
        console.warn('⚠️ Plots API response not ok:', response.status, response.statusText);
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

  // Query to keep TensorBoard as fallback option
  const { data: tbConfig } = useQuery({
    queryKey: ['tensorboard-config', jobId],
    queryFn: async (): Promise<TensorBoardConfig | null> => {
      const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/tensorboard/url`);
      if (!response.ok) {
        console.warn('⚠️ TensorBoard config not available:', response.status, response.statusText);
        return null; // TensorBoard not available
      }
      const config = await response.json();
      return config;
    },
    enabled: !!jobId,
    staleTime: 60000, // 1 minute
  });

  // Memoize plots to prevent dependency warning
  const availablePlots = useMemo(() => plotsData?.plots || [], [plotsData?.plots]);
  const hasPlots = availablePlots.length > 0;
  
  // Set active tab to defaultPlotType or first available plot if current tab doesn't exist
  useEffect(() => {
    if (hasPlots) {
      // If defaultPlotType is provided and available, use it
      if (defaultPlotType && availablePlots.find(p => p.plot_type === defaultPlotType)) {
        setActiveTab(defaultPlotType);
      } 
      // Otherwise, if current tab doesn't exist, use first available
      else if (!availablePlots.find(p => p.plot_type === activeTab)) {
        setActiveTab(availablePlots[0].plot_type);
      }
    }
  }, [availablePlots, activeTab, hasPlots, defaultPlotType]);

  // Helper to get plot icon
  const getPlotIcon = (plotType: string) => {
    switch (plotType) {
      case 'training_history':
        return <BarChart3 className="w-4 h-4" />;
      case 'weights_bias':
        return <Activity className="w-4 h-4" />;
      case 'gradient_flow':
        return <Zap className="w-4 h-4" />;
      case 'dead_neuron_analysis':
        return <Skull className="w-4 h-4" />;
      case 'gradient_distributions':
        return <TrendingUp className="w-4 h-4" />;
      case 'training_progress':
        return <LineChart className="w-4 h-4" />;
      case 'activation_maps':
        return <Brain className="w-4 h-4" />;
      case 'activation_summary':
        return <Brain className="w-4 h-4" />;
      case 'confusion_matrix':
        return <Target className="w-4 h-4" />;
      case 'training_animation':
        return <Play className="w-4 h-4" />;
      default:
        return <BarChart3 className="w-4 h-4" />;
    }
  };

  // Helper to get plot display name
  const getPlotDisplayName = (plotType: string) => {
    switch (plotType) {
      case 'training_history':
        return 'Training History';
      case 'weights_bias':
        return 'Weights + Bias';
      case 'gradient_flow':
        return 'Gradient Flow';
      case 'dead_neuron_analysis':
        return 'Dead Neurons';
      case 'gradient_distributions':
        return 'Gradient Distrib.';
      case 'training_progress':
        return 'Training Progress';
      case 'activation_maps':
        return 'Activation Maps';
      case 'activation_summary':
        return 'Activation Summary';
      case 'confusion_matrix':
        return 'Confusion Matrix';
      case 'training_animation':
        return 'Training Animation';
      default:
        return plotType.replace('_', ' ').toUpperCase();
    }
  };

  // Helper to get plot description
  const getPlotDescription = (plotType: string) => {
    switch (plotType) {
      case 'training_history':
        return 'Historical training metrics and trends';
      case 'weights_bias':
        return 'Weight distributions and parameter health analysis';
      case 'gradient_flow':
        return 'Gradient flow analysis';
      case 'dead_neuron_analysis':
        return 'Dead neuron detection and analysis';
      case 'gradient_distributions':
        return 'Gradient distribution patterns';
      case 'training_progress':
        return 'Loss and accuracy curves with overfitting detection';
      case 'activation_maps':
        return 'CNN layer activations and filter visualizations';
      case 'activation_summary':
        return 'Activation pattern summary and analysis';
      case 'confusion_matrix':
        return 'Classification accuracy and error analysis';
      case 'training_animation':
        return 'Animated training progress visualization';
      default:
        return 'Training visualization';
    }
  };

  if (plotsLoading) {
    return (
      <Card className="w-full h-full border-0">
        <CardContent className="flex items-center justify-center h-full">
          <div className="text-center">
            <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-500" />
            <p className="text-muted-foreground">Loading training visualizations...</p>
          </div>
        </CardContent>
      </Card>
    );
  }


  return (
    <Card className="w-full h-full border-0">
      <CardHeader className="pb-0">
        <div className="flex justify-between items-start">
          <div>
            {(() => {
              const currentPlot = (plotsData?.plots || []).find(plot => plot.plot_type === activeTab);
              return currentPlot ? (
                <div>
                  <h3 className="text-base font-semibold flex items-center gap-1">
                    {getPlotIcon(currentPlot.plot_type)}
                    {getPlotDisplayName(currentPlot.plot_type)}
                  </h3>
                  <p className="text-xs text-muted-foreground">
                    {getPlotDescription(currentPlot.plot_type)}
                  </p>
                </div>
              ) : null;
            })()}
          </div>
          <div className="flex flex-col gap-1 w-24">
            {tbConfig?.tensorboard_url && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  // Open TensorBoard without filtering - users can select runs in the TensorBoard UI
                  window.open(tbConfig.tensorboard_url, '_blank');
                }}
                className="flex items-center justify-center gap-1 w-full h-6 text-xs px-2 py-1"
                title="Open full TensorBoard interface for deep analysis"
              >
                <ExternalLink className="w-2 h-2" />
                TensorBoard
              </Button>
            )}
            {(() => {
              // Find the current plot to show download button
              const currentPlot = (plotsData?.plots || []).find(plot => plot.plot_type === activeTab);
              return currentPlot ? (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={async () => {
                    try {
                      const fullUrl = `${API_BASE_URL}${currentPlot.url}`;
                      
                      // Fetch the plot image as blob
                      const response = await fetch(fullUrl);
                      if (!response.ok) {
                        console.error('Failed to fetch plot image:', response.status);
                        window.open(fullUrl, '_blank'); // Fallback to open in new tab
                        return;
                      }
                      const plotBlob = await response.blob();
                      
                      const defaultFilename = currentPlot.filename || `${getPlotDisplayName(currentPlot.plot_type).replace(/\s+/g, '_').toLowerCase()}_${jobId}_${trialId || 'latest'}_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;

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
                          await writable.write(plotBlob);
                          await writable.close();
                          console.log('Plot saved successfully');
                          return;
                        } catch (saveError: unknown) {
                          if ((saveError as Error).name === 'AbortError') {
                            console.log('Save dialog was cancelled by user');
                            return;
                          }
                          console.warn('Save As dialog failed, falling back to download:', saveError);
                        }
                      }

                      // Fallback for browsers that don't support File System Access API
                      const url = URL.createObjectURL(plotBlob);
                      const link = document.createElement('a');
                      link.href = url;
                      link.download = defaultFilename;
                      document.body.appendChild(link);
                      link.click();
                      document.body.removeChild(link);
                      URL.revokeObjectURL(url);
                      console.log('Plot downloaded successfully (fallback method)');
                    } catch (error) {
                      console.error('Failed to download plot:', error);
                      // Final fallback - open in new tab
                      const fullUrl = `${API_BASE_URL}${currentPlot.url}`;
                      window.open(fullUrl, '_blank');
                    }
                  }}
                  className="flex items-center justify-center gap-1 w-full h-6 text-xs px-2 py-1"
                >
                  <Download className="w-2 h-2" />
                  Download
                </Button>
              ) : null;
            })()}
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        {!hasPlots ? (
          <div className="flex items-center justify-center h-96 text-center">
            <div>
              <AlertCircle className="w-12 h-12 mx-auto mb-4 text-yellow-500 opacity-50" />
              <h3 className="text-lg font-medium mb-2">No Training Visualizations Available</h3>
              <p className="text-sm text-muted-foreground max-w-md mb-4">
                Training visualizations will appear here after trials complete. 
                Educational plots are automatically generated to help you understand model behavior.
              </p>
              
              <div className="text-xs text-muted-foreground bg-blue-50 dark:bg-blue-950/20 rounded-lg p-3 max-w-md">
                <p className="font-medium text-blue-600 dark:text-blue-400 mb-1">📊 Available visualizations:</p>
                <p>• Training Progress - Loss/accuracy curves with overfitting detection</p>
                <p>• Weights + Bias - Weight distributions and parameter analysis</p>
                <p>• Activation Maps - CNN layer activations and filter visualizations</p>
                <p>• Confusion Matrix - Classification accuracy and error analysis</p>
                <p>• Dead Neurons - Dead neuron detection and analysis</p>
                <p>• Gradient Flow - Gradient flow analysis</p>
                <p>• Gradient Distrib. - Gradient distribution patterns</p>
                <p>• Activation Summary - Activation pattern summary</p>
              </div>
              
              {plotsError && (
                <div className="mt-4 text-xs text-red-500">
                  Error loading plots: {plotsError.message}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="p-4">
            {(() => {
              // Find the plot to display based on activeTab
              const currentPlot = availablePlots.find(plot => plot.plot_type === activeTab);
              if (!currentPlot) return null;

              return (
                <div className="space-y-4">
                  <div 
                    className="border border-border rounded-lg overflow-hidden relative group bg-white"
                  >
                    <Image
                      src={`${API_BASE_URL}${currentPlot.url}`}
                      alt={`${getPlotDisplayName(currentPlot.plot_type)} visualization`}
                      className="w-full h-auto max-h-[600px] object-contain"
                      width={800}
                      height={600}
                      unoptimized={true}
                      onError={(e) => {
                        const fullUrl = `${API_BASE_URL}${currentPlot.url}`;
                        console.error('❌ Failed to load plot:', fullUrl);
                        console.error('❌ Plot data:', { plot_type: currentPlot.plot_type, filename: currentPlot.filename, url: currentPlot.url });
                        console.error('❌ API_BASE_URL:', API_BASE_URL);
                        const target = e.target as HTMLImageElement;
                        target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDIwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjIwMCIgaGVpZ2h0PSIxMDAiIGZpbGw9IiNmM2Y0ZjYiLz48dGV4dCB4PSIxMDAiIHk9IjUwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkb21pbmFudC1iYXNlbGluZT0ibWlkZGxlIiBmb250LXNpemU9IjEyIiBmaWxsPSIjNjM2MzYzIj5QbG90IGZhaWxlZCB0byBsb2FkPC90ZXh0Pjwvc3ZnPg==';
                        target.style.maxHeight = '200px';
                      }}
                    />
                    
                    {/* Expand/Minimize icon in top-right corner */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onExpandClick?.();
                      }}
                      className="absolute top-2 right-2 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10"
                      title={isFullscreen ? "Minimize image" : "Expand image"}
                    >
                      {isFullscreen ? (
                        <Minimize2 className="w-4 h-4" />
                      ) : (
                        <Maximize2 className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                  
                  <div className="text-xs text-muted-foreground flex items-center justify-between">
                    <span>
                      Size: {(currentPlot.size_bytes / 1024).toFixed(1)} KB
                    </span>
                    <span>
                      Created: {new Date(currentPlot.created_time * 1000).toLocaleString()}
                    </span>
                  </div>
                </div>
              );
            })()}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Keep backward compatibility by exporting as TensorBoardPanel
export const TensorBoardPanel = TrainingMetricsPanel;
export default TrainingMetricsPanel;