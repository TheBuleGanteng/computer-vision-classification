"use client"

import React, { useState } from 'react';
import { Activity, Zap, Brain, Target, Skull, TrendingUp, LineChart } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import TensorBoardPanel from './tensorboard-panel';

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
  const [activeTab, setActiveTab] = useState<string>('training_progress');

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

  const tabs = [
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
    }
  ];


  if (loading) {
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
      <div className="flex flex-wrap justify-center gap-0 bg-gray-800 border-b border-gray-600 pt-2 px-0">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1 px-2 py-1 text-xs font-medium transition-all duration-200 whitespace-nowrap relative
              ${activeTab === tab.id
                ? 'text-blue-300 bg-gray-900 border-t-2 border-l border-r border-blue-400 border-b-0 rounded-t-md -mb-px z-10'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/50 border border-gray-500 rounded-t-md'
              }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1">
        {tabs.map((tab) => (
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
        ))}
      </div>
    </div>
  );
});

MetricsTabs.displayName = 'MetricsTabs'

export default MetricsTabs;