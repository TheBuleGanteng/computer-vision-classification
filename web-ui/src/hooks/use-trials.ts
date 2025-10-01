'use client';

import { useComprehensiveStatus } from './use-comprehensive-status';

/**
 * Legacy hook for backward compatibility - now uses unified comprehensive status
 * All trial data fetching is now handled by useComprehensiveStatus to eliminate 
 * React setTimeout violations from multiple concurrent polling systems
 */
export function useTrials() {
  const {
    trials,
    bestTrial,
    completedTrials,
    stats,
    isLoading,
    error,
    lastUpdated,
    refetch
  } = useComprehensiveStatus();

  return {
    // Raw data
    trials,
    bestTrial,
    completedTrials,
    
    // Statistics
    stats,
    
    // Query state
    isLoading,
    error,
    lastUpdated,
    isPending: false, // Not used in unified approach
    
    // Actions
    refetch,
    
    // Helper functions
    getTrial: (trialNumber: number) => trials.find(t => t.trial_number === trialNumber),
    getTrialById: (trialId: string) => trials.find(t => t.trial_id === trialId),
  };
}

// Legacy hook for backward compatibility - now uses unified comprehensive status
export function useBestTrial() {
  const { bestTrial, trials, isLoading, error, refetch } = useTrials();
  
  return {
    bestTrial,
    trials, // Now populated from unified system
    isLoading,
    error,
    refetch
  };
}