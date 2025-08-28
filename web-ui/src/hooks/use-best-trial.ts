'use client';

import { useState, useEffect, useCallback } from 'react';
import { useDashboard } from '@/components/dashboard/dashboard-provider';
import { apiClient } from '@/lib/api-client';
import { TrialProgress } from '@/types/optimization';

export function useBestTrial() {
  const { currentJobId, isOptimizationRunning } = useDashboard();
  const [bestTrial, setBestTrial] = useState<TrialProgress | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trials, setTrials] = useState<TrialProgress[]>([]);

  // Find the best trial based on highest total_score
  const findBestTrial = (trialList: TrialProgress[]) => {
    if (!trialList || trialList.length === 0) return null;
    
    // Only consider completed trials for best trial selection
    const completedTrials = trialList.filter(trial => 
      trial.status === 'completed' && 
      trial.performance?.total_score !== undefined && 
      trial.performance?.total_score !== null
    );
    
    if (completedTrials.length === 0) return null;
    
    return completedTrials.reduce((best, current) => {
      const bestScore = best.performance?.total_score || 0;
      const currentScore = current.performance?.total_score || 0;
      return currentScore > bestScore ? current : best;
    });
  };

  // Fetch trial data from API
  const fetchTrials = useCallback(async (jobId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiClient.getTrialHistory(jobId);
      const rawTrials = response.trials || [];
      
      // Deduplicate trials by trial_id and trial_number
      const uniqueTrials = rawTrials.filter((trial, index, array) => {
        const firstIndex = array.findIndex(t => 
          (t.trial_id && trial.trial_id && t.trial_id === trial.trial_id) ||
          (t.trial_number !== undefined && trial.trial_number !== undefined && t.trial_number === trial.trial_number)
        );
        return firstIndex === index;
      });

      setTrials(uniqueTrials);
      
      // Find and set the best trial
      const currentBest = findBestTrial(uniqueTrials);
      if (currentBest && currentBest.performance?.total_score && 
          (!bestTrial || currentBest.performance.total_score > (bestTrial.performance?.total_score || 0))) {
        setBestTrial(currentBest);
        console.log(`🏆 NEW BEST TRIAL: Trial ${currentBest.trial_number} with score ${(currentBest.performance.total_score * 100).toFixed(1)}%`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch trials');
    } finally {
      setIsLoading(false);
    }
  }, [bestTrial]);

  // Poll for trial updates
  useEffect(() => {
    if (!currentJobId) {
      setBestTrial(null);
      setTrials([]);
      return;
    }

    // Fetch immediately
    fetchTrials(currentJobId);

    // Poll more frequently during optimization, less frequently when not running
    const pollFrequency = isOptimizationRunning ? 2000 : 10000; // 2s when running, 10s when idle
    
    const pollInterval = setInterval(() => {
      fetchTrials(currentJobId);
    }, pollFrequency);

    return () => clearInterval(pollInterval);
  }, [currentJobId, isOptimizationRunning, fetchTrials]);

  return {
    bestTrial,
    trials,
    isLoading,
    error,
    refetch: () => currentJobId ? fetchTrials(currentJobId) : Promise.resolve()
  };
}