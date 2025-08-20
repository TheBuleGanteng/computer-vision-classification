import { OptimizationResults, TrialData } from './optimization'

export interface ApiResponse<T> {
  data: T
  success: boolean
  message?: string
  timestamp: string
}

export interface OptimizationSession {
  id: string
  name: string
  dataset: string
  mode: 'simple' | 'health'
  status: 'running' | 'completed' | 'failed'
  created_at: string
  completed_at?: string
  trial_count: number
  best_accuracy?: number
  best_health?: number
}

export interface OptimizationSessionsResponse {
  sessions: OptimizationSession[]
  total: number
  page: number
  per_page: number
}

export interface TrialDataResponse {
  trials: TrialData[]
  total: number
  session_id: string
}

export interface LiveProgressUpdate {
  session_id: string
  trial_number: number
  status: 'running' | 'completed' | 'failed'
  progress: number
  current_epoch?: number
  total_epochs?: number
  current_accuracy?: number
  current_loss?: number
  eta_seconds?: number
}

export interface SystemHealth {
  cpu_usage: number
  memory_usage: number
  gpu_usage?: number
  active_sessions: number
  queue_length: number
  last_updated: string
}

// API endpoints configuration
export interface ApiConfig {
  baseUrl: string
  timeout: number
  retries: number
}