'use client'

import { useQuery } from '@tanstack/react-query'
import { useDashboard } from '@/components/dashboard/dashboard-provider'
import { apiClient } from '@/lib/api-client'
import { TrialProgress } from '@/types/optimization'

/**
 * Unified hook that replaces separate polling for job status, trials, and elapsed time
 * Single API call eliminates React setTimeout violations from multiple concurrent timers
 */
export function useComprehensiveStatus() {
  const { currentJobId, isOptimizationRunning } = useDashboard()

  const {
    data,
    isLoading,
    error,
    refetch,
    dataUpdatedAt
  } = useQuery({
    queryKey: ['comprehensive-status', currentJobId],
    queryFn: () => currentJobId ? apiClient.getComprehensiveStatus(currentJobId) : null,
    enabled: !!currentJobId,
    
    // Single polling interval replaces all separate timers
    refetchInterval: isOptimizationRunning ? 2000 : 10000,
    refetchIntervalInBackground: false,
    
    refetchOnWindowFocus: false,
    refetchOnReconnect: true,
    
    // Keep previous data to prevent UI flashing
    placeholderData: (previousData) => previousData,
    
    staleTime: isOptimizationRunning ? 1000 : 30000,
    gcTime: 10 * 60 * 1000,
    
    retry: (failureCount, error) => {
      if (error && 'status' in error && error.status === 404) {
        return false
      }
      return failureCount < 2
    }
  })

  // Extract individual components from unified response
  const jobStatus = data?.job_status
  const trials = data?.trials || []
  const elapsedSeconds = data?.elapsed_seconds || 0
  const isComplete = data?.is_complete || false

  // Compute derived data
  const bestTrial = trials.reduce<TrialProgress | null>((best, trial) => {
    if (trial.status === 'completed' && trial.performance?.total_score) {
      if (!best || (trial.performance.total_score > (best.performance?.total_score || 0))) {
        return trial
      }
    }
    return best
  }, null)

  const completedTrials = trials.filter(trial => trial.status === 'completed')
  
  const stats = {
    total: trials.length,
    completed: completedTrials.length,
    running: trials.filter(trial => trial.status === 'running').length,
    failed: trials.filter(trial => trial.status === 'failed').length,
    averageScore: completedTrials.length > 0 
      ? completedTrials.reduce((sum, trial) => sum + (trial.performance?.total_score || 0), 0) / completedTrials.length
      : 0
  }

  return {
    // Job status data
    jobStatus,
    isComplete,
    
    // Trial data  
    trials,
    bestTrial,
    completedTrials,
    stats,
    
    // Elapsed time (replaces separate timer)
    elapsedSeconds,
    
    // Query state
    isLoading,
    error: error instanceof Error ? error.message : null,
    lastUpdated: dataUpdatedAt,
    
    // Actions
    refetch,
    
    // Helper functions
    getTrial: (trialNumber: number) => trials.find(t => t.trial_number === trialNumber),
    getTrialById: (trialId: string) => trials.find(t => t.trial_id === trialId),
  }
}