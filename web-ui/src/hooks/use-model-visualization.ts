// React Query hooks for 3D Model Visualization
// Provides data fetching, caching, and state management

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getBestModelVisualization,
  downloadVisualizationFile,
  hasVisualizationData,
  VisualizationAPIError,
  generateVisualizationFilename
} from '@/lib/api/visualization';
import { BestModelResponse } from '@/types/visualization';
import { toast } from 'sonner';
import { logger } from '@/lib/logger';

/**
 * Hook to fetch 3D visualization data for a job's best model
 */
export function useModelVisualization(jobId: string | null, enabled: boolean = true) {
  return useQuery<BestModelResponse, VisualizationAPIError>({
    queryKey: ['modelVisualization', jobId],
    queryFn: () => {
      if (!jobId) {
        throw new VisualizationAPIError('Job ID is required');
      }
      return getBestModelVisualization(jobId);
    },
    enabled: enabled && !!jobId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: (failureCount, error) => {
      // Don't retry on 404 (no completed trials)
      if (error instanceof VisualizationAPIError && error.status === 404) {
        return false;
      }
      return failureCount < 3;
    },
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });
}

/**
 * Hook to check if visualization data is available for a job
 */
export function useVisualizationAvailability(jobId: string | null) {
  return useQuery<boolean, Error>({
    queryKey: ['visualizationAvailable', jobId],
    queryFn: () => {
      if (!jobId) return false;
      return hasVisualizationData(jobId);
    },
    enabled: !!jobId,
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: () => {
      // Coordinate with other polling to prevent simultaneous requests  
      const isMobile = typeof window !== 'undefined' && window.innerWidth <= 768;
      // Offset from other polling systems to stagger requests
      return isMobile ? false : 15000; // 15s desktop, disabled on mobile
    },
    refetchIntervalInBackground: false
  });
}

/**
 * Hook to handle visualization file downloads
 */
export function useVisualizationDownload() {
  const queryClient = useQueryClient();

  return useMutation<void, VisualizationAPIError, { jobId: string; customFilename?: string }>({
    mutationFn: async ({ jobId, customFilename }) => {
      // Get visualization data for filename generation
      const cachedData = queryClient.getQueryData<BestModelResponse>([
        'modelVisualization', 
        jobId
      ]);

      const filename = customFilename || generateVisualizationFilename(
        jobId,
        cachedData?.visualization_data.type,
        cachedData?.performance_score
      );

      await downloadVisualizationFile(jobId, filename);
    },
    onSuccess: () => {
      toast.success('Visualization data downloaded successfully');
    },
    onError: (error) => {
      logger.error('Download failed:', error);
      toast.error(`Download failed: ${error.message}`);
    },
  });
}

/**
 * Hook to preload visualization data for better UX
 */
export function usePrefetchVisualization() {
  const queryClient = useQueryClient();

  return {
    prefetch: (jobId: string) => {
      queryClient.prefetchQuery({
        queryKey: ['modelVisualization', jobId],
        queryFn: () => getBestModelVisualization(jobId),
        staleTime: 5 * 60 * 1000,
      });
    },
    
    prefetchAvailability: (jobId: string) => {
      queryClient.prefetchQuery({
        queryKey: ['visualizationAvailable', jobId], 
        queryFn: () => hasVisualizationData(jobId),
        staleTime: 30 * 1000,
      });
    }
  };
}

/**
 * Hook to invalidate and refetch visualization data
 */
export function useRefreshVisualization() {
  const queryClient = useQueryClient();

  return {
    refresh: (jobId: string) => {
      queryClient.invalidateQueries({ 
        queryKey: ['modelVisualization', jobId] 
      });
      queryClient.invalidateQueries({ 
        queryKey: ['visualizationAvailable', jobId] 
      });
    },

    refreshAll: () => {
      queryClient.invalidateQueries({ 
        queryKey: ['modelVisualization'] 
      });
      queryClient.invalidateQueries({ 
        queryKey: ['visualizationAvailable'] 
      });
    }
  };
}

/**
 * Utility hook to get visualization status for UI decisions
 */
export function useVisualizationStatus(jobId: string | null) {
  const { data: hasData, isLoading: checkingAvailability } = useVisualizationAvailability(jobId);
  const { data: vizData, isLoading: loadingVisualization, error } = useModelVisualization(
    jobId, 
    hasData === true
  );

  return {
    isAvailable: hasData === true,
    isLoading: checkingAvailability || loadingVisualization,
    hasError: !!error,
    error: error?.message,
    canView3D: hasData === true && !error && !!vizData,
    visualizationData: vizData,
  };
}