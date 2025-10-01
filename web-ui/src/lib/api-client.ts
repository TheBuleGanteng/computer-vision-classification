/**
 * API Client for Hyperparameter Optimization Backend
 * 
 * Provides TypeScript interface for communicating with the FastAPI backend
 * for optimization job management, progress tracking, and result retrieval.
 */

import { TrialProgress } from '@/types/optimization'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface OptimizationRequest {
  // Core parameters
  dataset_name: string
  mode: 'simple' | 'health'
  optimize_for?: string

  // Optimization control
  trials?: number
  max_epochs_per_trial?: number
  min_epochs_per_trial?: number

  // Scoring weights (replaces deprecated health_weight)
  accuracy_weight?: number
  health_overall_weight?: number
  health_component_proportions?: {
    neuron_utilization: number
    parameter_efficiency: number
    training_stability: number
    gradient_health: number
    convergence_quality: number
    accuracy_consistency: number
  }
  
  // Training parameters
  validation_split?: number
  test_size?: number
  batch_size?: number
  learning_rate?: number
  optimizer_name?: string
  
  // Architecture constraints
  max_parameters?: number
  min_accuracy_threshold?: number
  
  // Advanced optimization settings
  activation_functions?: string[]
  n_startup_trials?: number
  n_warmup_steps?: number
  early_stopping_patience?: number
  enable_early_stopping?: boolean
  
  // Resource and timing constraints
  max_training_time_minutes?: number
  gpu_proxy_sample_percentage?: number
  
  // Reproducibility
  random_seed?: number
  
  // Validation and health analysis
  health_analysis_sample_size?: number
  enable_stability_checks?: boolean
  stability_window?: number
  
  // RunPod service settings
  use_runpod_service?: boolean
  concurrent_workers?: number
  use_multi_gpu?: boolean
  target_gpus_per_worker?: number
  
  // Legacy compatibility
  config_overrides?: Record<string, unknown>
}

export interface JobResponse {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: string
  started_at?: string
  completed_at?: string
  progress?: {
    current_trial: number
    total_trials: number
    completed_trials: number
    success_rate: number
    best_value: number
    elapsed_time: number
    status_message: string
    current_epoch?: number
    total_epochs?: number
    epoch_progress?: number
  }
  result?: Record<string, unknown>
  error?: string
}

export interface ApiError {
  detail: string
}

export interface JobResults {
  model_result?: {
    model_path?: string
    [key: string]: unknown
  }
  [key: string]: unknown
}

class OptimizationApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  /**
   * Start a new optimization job
   */
  async startOptimization(request: OptimizationRequest): Promise<JobResponse> {
    console.log('Sending optimization request:', request)
    console.log('API URL:', `${this.baseUrl}/optimize`)
    console.log('Request body:', JSON.stringify(request, null, 2))
    
    const response = await fetch(`${this.baseUrl}/optimize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    console.log('API Response status:', response.status, response.statusText)
    console.log('Response headers:', Object.fromEntries(response.headers.entries()))

    if (!response.ok) {
      // Clone the response so we can read it multiple times if needed
      const responseClone = response.clone()
      
      try {
        const error: ApiError = await response.json()
        console.error('API Error details:', error)
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
      } catch {
        // If JSON parsing fails, use the cloned response to get text
        try {
          const errorText = await responseClone.text()
          console.error('Failed to parse JSON, raw response:', errorText)
          throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`)
        } catch (textError) {
          console.error('Failed to read response as text:', textError)
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }
      }
    }

    const result = await response.json()
    console.log('Optimization started successfully:', result)
    return result
  }

  /**
   * Get job status and progress
   */
  async getJobStatus(jobId: string): Promise<JobResponse> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}`)

    if (!response.ok) {
      const error: ApiError = await response.json()
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Cancel a running optimization job
   */
  async cancelJob(jobId: string): Promise<{ message: string }> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      const error: ApiError = await response.json()
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get detailed results from completed job
   */
  async getJobResults(jobId: string): Promise<JobResults> {
    const response = await fetch(`${this.baseUrl}/results/${jobId}`)

    if (!response.ok) {
      const error: ApiError = await response.json()
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get trial history for visualization
   */
  async getTrialHistory(jobId: string): Promise<{ trials: TrialProgress[], total_trials: number }> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}/trials`)

    if (!response.ok) {
      const error: ApiError = await response.json()
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get currently running trial data
   */
  async getCurrentTrial(jobId: string): Promise<{ current_trial: TrialProgress }> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}/current-trial`)

    if (!response.ok) {
      const error: ApiError = await response.json()
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get best performing trial so far
   */
  async getBestTrial(jobId: string): Promise<{ best_trial: TrialProgress }> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}/best-trial`)

    if (!response.ok) {
      const error: ApiError = await response.json()
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get available datasets
   */
  async getAvailableDatasets(): Promise<{ datasets: string[] }> {
    const response = await fetch(`${this.baseUrl}/datasets`)

    if (!response.ok) {
      const error: ApiError = await response.json()
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get available optimization modes
   */
  async getAvailableModes(): Promise<{ modes: string[], descriptions: Record<string, string> }> {
    const response = await fetch(`${this.baseUrl}/modes`)

    if (!response.ok) {
      const error: ApiError = await response.json()
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Download trained model from completed job
   */
  getModelDownloadUrl(jobId: string): string {
    return `${this.baseUrl}/download/${jobId}`
  }

  /**
   * Get comprehensive job status including trials and elapsed time
   */
  async getComprehensiveStatus(jobId: string): Promise<{
    job_id: string
    job_status: JobResponse
    trials: TrialProgress[]
    elapsed_seconds: number
    is_complete: boolean
    total_trials: number
    timestamp: string
  }> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}/comprehensive`)

    if (!response.ok) {
      const error: ApiError = await response.json()
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Get default scoring weights
   */
  async getDefaultScoringWeights(): Promise<{
    accuracy_weight: number
    health_overall_weight: number
    health_component_proportions: {
      neuron_utilization: number
      parameter_efficiency: number
      training_stability: number
      gradient_health: number
      convergence_quality: number
      accuracy_consistency: number
    }
  }> {
    const response = await fetch(`${this.baseUrl}/default-scoring-weights`)

    if (!response.ok) {
      const error: ApiError = await response.json()
      throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string, timestamp: string }> {
    const response = await fetch(`${this.baseUrl}/health`)

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`)
    }

    return response.json()
  }
}

// Export singleton instance
export const apiClient = new OptimizationApiClient()

// Export class for testing or custom instances
export { OptimizationApiClient }