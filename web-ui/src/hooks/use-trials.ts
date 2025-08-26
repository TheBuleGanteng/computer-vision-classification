'use client';

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useDashboard } from '@/components/dashboard/dashboard-provider';
import { apiClient } from '@/lib/api-client';
import { TrialProgress } from '@/types/optimization';

/**
 * Centralized hook for all trial data fetching and management
 * Eliminates redundant polling and provides optimized updates
 */
export function useTrials() {
  const { currentJobId, isOptimizationRunning } = useDashboard();

  // Single source of truth for trial data with smart polling
  const {
    data: trialsResponse,
    isLoading,
    error,
    refetch,
    dataUpdatedAt
  } = useQuery({
    queryKey: ['trials', currentJobId],
    queryFn: async () => {
      if (!currentJobId) return { trials: [] };
      return await apiClient.getTrialHistory(currentJobId);
    },
    enabled: !!currentJobId,
    
    // Smart polling configuration
    refetchInterval: isOptimizationRunning ? 3000 : 10000, // 3s active, 10s idle
    refetchIntervalInBackground: false, // Don't poll when tab is inactive
    
    // Stability optimizations
    refetchOnWindowFocus: false,
    refetchOnReconnect: true,
    
    // Keep previous data to prevent flashing
    placeholderData: (previousData) => previousData,
    
    // Cache configuration
    staleTime: isOptimizationRunning ? 1000 : 30000,
    gcTime: 10 * 60 * 1000, // 10 minutes
    
    // Data transformation and deduplication at query level
    select: (data) => {
      const rawTrials = data?.trials || [];
      
      // Deduplicate trials by trial_id and trial_number
      const uniqueTrials = rawTrials.filter((trial, index, array) => {
        const firstIndex = array.findIndex(t => 
          (t.trial_id && trial.trial_id && t.trial_id === trial.trial_id) ||
          (t.trial_number !== undefined && trial.trial_number !== undefined && t.trial_number === trial.trial_number)
        );
        return firstIndex === index;
      });

      return uniqueTrials;
    },
    
    // Error handling
    retry: (failureCount, error) => {
      // Don't retry on 404 (no job found)
      if (error && 'status' in error && error.status === 404) {
        return false;
      }
      return failureCount < 2;
    }
  });

  // Memoized computations to prevent unnecessary re-calculations
  const processedData = useMemo(() => {
    const trials = trialsResponse || [];
    
    // Find best trial
    const completedTrials = trials.filter(trial => 
      trial.status === 'completed' && 
      trial.performance?.total_score !== undefined && 
      trial.performance?.total_score !== null
    );
    
    const bestTrial = completedTrials.length > 0 
      ? completedTrials.reduce((best, current) => {
          const bestScore = best.performance?.total_score || 0;
          const currentScore = current.performance?.total_score || 0;
          return currentScore > bestScore ? current : best;
        })
      : null;

    // Calculate summary statistics
    const completedCount = completedTrials.length;
    const totalCount = trials.length;
    const runningCount = trials.filter(t => t.status === 'running').length;
    const failedCount = trials.filter(t => t.status === 'failed').length;
    
    const averageScore = completedTrials.length > 0
      ? completedTrials.reduce((sum, trial) => sum + (trial.performance?.total_score || 0), 0) / completedTrials.length
      : 0;

    return {
      trials,
      bestTrial,
      completedTrials,
      stats: {
        total: totalCount,
        completed: completedCount,
        running: runningCount,
        failed: failedCount,
        averageScore
      }
    };
  }, [trialsResponse]);

  return {
    // Raw data
    trials: processedData.trials,
    bestTrial: processedData.bestTrial,
    completedTrials: processedData.completedTrials,
    
    // Statistics
    stats: processedData.stats,
    
    // Query state
    isLoading,
    error: error instanceof Error ? error.message : null,
    lastUpdated: dataUpdatedAt,
    
    // Actions
    refetch,
    
    // Helper functions
    getTrial: (trialNumber: number) => processedData.trials.find(t => t.trial_number === trialNumber),
    getTrialById: (trialId: string) => processedData.trials.find(t => t.trial_id === trialId),
  };
}

// Legacy hook for backward compatibility - now uses centralized data
export function useBestTrial() {
  const { bestTrial, isLoading, error, refetch } = useTrials();
  
  return {
    bestTrial,
    trials: [], // Deprecated - use useTrials() directly
    isLoading,
    error,
    refetch
  };
}