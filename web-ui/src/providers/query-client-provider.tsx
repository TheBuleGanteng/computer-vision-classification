'use client';

import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Create a client optimized for smooth polling UX
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Disable automatic refetching for stability during user interaction
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
      refetchOnMount: true,
      
      // Optimize for smooth updates during polling
      staleTime: 2000, // 2 seconds - faster updates for real-time data
      gcTime: 10 * 60 * 1000, // 10 minutes cache retention
      
      // Keep previous data while fetching to prevent UI flashing
      placeholderData: (previousData: unknown) => previousData,
      
      // Retry configuration for robustness
      retry: (failureCount, error) => {
        // Don't retry on 4xx errors (client errors)
        if (error && 'status' in error && typeof error.status === 'number') {
          if (error.status >= 400 && error.status < 500) {
            return false;
          }
        }
        // Retry up to 2 times for other errors
        return failureCount < 2;
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 5000), // Faster retry
      
      // Network mode for better offline handling
      networkMode: 'online',
    },
    mutations: {
      // Retry mutations once on failure
      retry: 1,
      // Don't show network error immediately for mutations
      networkMode: 'online',
    }
  },
});

interface QueryProviderProps {
  children: React.ReactNode;
}

export function QueryProvider({ children }: QueryProviderProps) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}