"use client"

import React, { createContext, useContext, useState, ReactNode } from 'react'

export interface ProgressData {
  trials_performed?: number
  best_accuracy?: number
  best_total_score?: number
  average_duration_per_trial?: number
  current_trial?: number
  total_trials?: number
  completed_trials?: number
  elapsed_time?: number
  current_epoch?: number
  total_epochs?: number
  epoch_progress?: number
  status_message?: string
  is_gpu_mode?: boolean
  plot_generation?: {
    status: 'generating' | 'completed' | 'failed'
    current_plot: string
    completed_plots: number
    total_plots: number
    plot_progress: number
  }
  final_model_building?: {
    status: 'building' | 'completed' | 'failed'
    current_step: string
    progress: number
    detailed_info?: string
  }
}

interface DashboardContextType {
  progress: ProgressData | null
  optimizationMode: "simple" | "health"
  healthWeight: number
  isOptimizationRunning: boolean
  currentJobId: string | null
  setProgress: (progress: ProgressData | null) => void
  setOptimizationMode: (mode: "simple" | "health") => void
  setHealthWeight: (weight: number) => void
  setIsOptimizationRunning: (running: boolean) => void
  setCurrentJobId: (jobId: string | null) => void
}

const DashboardContext = createContext<DashboardContextType | undefined>(undefined)

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [progress, setProgress] = useState<ProgressData | null>(null)
  const [optimizationMode, setOptimizationMode] = useState<"simple" | "health">("health")
  const [healthWeight, setHealthWeight] = useState<number>(0.3) // Default from API
  const [isOptimizationRunning, setIsOptimizationRunning] = useState(false)
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)

  return (
    <DashboardContext.Provider value={{
      progress,
      optimizationMode,
      healthWeight,
      isOptimizationRunning,
      currentJobId,
      setProgress,
      setOptimizationMode,
      setHealthWeight,
      setIsOptimizationRunning,
      setCurrentJobId
    }}>
      {children}
    </DashboardContext.Provider>
  )
}

export function useDashboard() {
  const context = useContext(DashboardContext)
  if (context === undefined) {
    throw new Error('useDashboard must be used within a DashboardProvider')
  }
  return context
}