'use client';

import React from 'react';

/**
 * Utility components and hooks for preventing unnecessary re-renders
 * during polling and real-time updates
 */

// Higher-order component for memoizing components that receive trial data
export function withTrialMemoization<T extends Record<string, unknown>>(
  Component: React.ComponentType<T>
) {
  const MemoizedComponent = React.memo(Component, (prevProps, nextProps) => {
    // Custom comparison for trial data to prevent unnecessary re-renders
    // Only re-render if trial data actually changed
    if (prevProps.trials !== nextProps.trials) {
      // Deep comparison for trial arrays
      if (Array.isArray(prevProps.trials) && Array.isArray(nextProps.trials)) {
        if (prevProps.trials.length !== nextProps.trials.length) {
          return false; // Re-render
        }
        
        // Compare trial IDs and modification times
        for (let i = 0; i < prevProps.trials.length; i++) {
          const prev = prevProps.trials[i];
          const next = nextProps.trials[i];
          if (prev.trial_id !== next.trial_id || 
              prev.completed_at !== next.completed_at ||
              prev.status !== next.status) {
            return false; // Re-render
          }
        }
      }
    }
    
    // Check other props for changes
    for (const key in nextProps) {
      if (key !== 'trials' && prevProps[key] !== nextProps[key]) {
        return false; // Re-render
      }
    }
    
    return true; // Don't re-render
  });
  
  MemoizedComponent.displayName = `withTrialMemoization(${Component.displayName || Component.name})`;
  return MemoizedComponent;
}

// Memoized wrapper for components that display individual trial data
export const TrialCard = React.memo<{
  trial: Record<string, unknown>;
  onSelect?: (trial: Record<string, unknown>) => void;
  isSelected?: boolean;
}>(({ }) => {
  // This will only re-render when trial data, selection state, or handler changes
  return null; // Placeholder - implement actual card content
}, (prevProps, nextProps) => {
  return (
    prevProps.trial?.trial_id === nextProps.trial?.trial_id &&
    prevProps.trial?.status === nextProps.trial?.status &&
    prevProps.trial?.completed_at === nextProps.trial?.completed_at &&
    prevProps.isSelected === nextProps.isSelected &&
    prevProps.onSelect === nextProps.onSelect
  );
});

TrialCard.displayName = 'TrialCard';

// Stable callback hook to prevent callback recreation on every render
export function useStableCallback<T extends (...args: unknown[]) => unknown>(callback: T): T {
  const callbackRef = React.useRef(callback);
  
  // Update the callback ref on every render
  React.useEffect(() => {
    callbackRef.current = callback;
  });
  
  // Return stable callback that calls the latest version
  return React.useCallback(((...args: Parameters<T>) => {
    return callbackRef.current(...args);
  }) as T, []);
}