export interface OptimizationMetadata {
  dataset_name: string
  optimization_mode: 'simple' | 'health'
  optimization_objective: string
  health_weight?: number
  total_trials: number
  successful_trials: number
  optimization_time_hours: number
  timestamp: string
}

export interface OptimizationRequest {
  dataset: string
  target_metric: 'simple' | 'health'
  session_id?: string
}

export interface OptimizationStatus {
  session_id: string
  is_running: boolean
  is_completed: boolean
  current_trial?: number
  total_trials?: number
  dataset: string
  target_metric: 'simple' | 'health'
  start_time?: string
  estimated_completion?: string
}

export interface Hyperparameters {
  // CNN/Architecture parameters
  num_layers_conv: number
  kernel_size: number
  pool_size: number
  padding: 'valid' | 'same'
  filters_per_conv_layer: number
  activation: string
  kernel_initializer: string
  batch_normalization: boolean
  use_global_pooling: boolean
  
  // Dense layer parameters
  num_layers_hidden: number
  first_hidden_layer_nodes: number
  subsequent_hidden_layer_nodes_decrease: number
  hidden_layer_activation_algo: string
  first_hidden_layer_dropout: number
  subsequent_hidden_layer_dropout_decrease: number
  
  // Training parameters
  epochs: number
  optimizer: string
  learning_rate: number
  enable_gradient_clipping: boolean
  gradient_clip_norm?: number
}

export interface HealthMetrics {
  neuron_utilization: number
  parameter_efficiency: number
  training_stability: number
  gradient_health: number
  convergence_quality: number
  accuracy_consistency: number
  overall_health: number
  health_breakdown: {
    [key: string]: {
      score: number
      weight: number
    }
  }
  recommendations: string[]
  trial_number?: number
}

export interface BestResults {
  best_total_score: number
  best_params: Hyperparameters
  best_trial_health: HealthMetrics
}

export interface ParameterImportance {
  [parameter: string]: number
}

export interface OptimizationAnalysis {
  parameter_importance: ParameterImportance
  objective_history: number[]
  average_health_metrics: {
    overall_health: number
    training_stability: number
    gradient_health: number
    convergence_quality: number
    parameter_efficiency: number
    accuracy_consistency: number
    neuron_utilization: number
  }
}

export interface TrialHealthHistory {
  trial_number: number
  health_metrics: HealthMetrics
  timestamp: string
  base_objective: number
  final_objective: number
}

export interface HealthMonitoring {
  trial_health_history: TrialHealthHistory[]
  health_analysis_enabled: boolean
  health_weighting_applied: boolean
}

export interface OptimizationConfiguration {
  mode: 'simple' | 'health'
  health_weight?: number
  trials: number
  max_epochs_per_trial: number
  max_training_time_minutes: number
  optimization_objective: string
}

export interface OptimizationResults {
  optimization_metadata: OptimizationMetadata
  best_results: BestResults
  analysis: OptimizationAnalysis
  health_monitoring: HealthMonitoring
  configuration: OptimizationConfiguration
}

// Trial progress from real-time API
export interface TrialProgress {
  trial_id: string
  trial_number: number
  status: 'running' | 'completed' | 'failed' | 'pruned'
  started_at: string
  completed_at?: string
  duration_seconds?: number
  
  // Epoch-level progress tracking
  current_epoch?: number
  total_epochs?: number
  epoch_progress?: number
  
  // Architecture Information
  architecture?: {
    type?: string
    conv_layers?: number
    dense_layers?: number
    activation?: string
    filters_per_layer?: number
    first_dense_nodes?: number
    batch_normalization?: boolean
    kernel_size?: number[] | number | string
    convLayers?: number | string
    denseLayers?: number | string
    totalLayers?: number | string
    filterSizes?: number[]
    activations?: string[]
    keyFeatures?: string[]
    parameters?: number
    [key: string]: unknown
  }
  hyperparameters?: {
    [key: string]: unknown
  }
  model_size?: {
    [key: string]: unknown
  }
  
  // Health Metrics (includes test_loss, overall_health, etc.)
  health_metrics?: {
    test_loss?: number
    test_accuracy?: number
    overall_health?: number
    neuron_utilization?: number
    parameter_efficiency?: number
    training_stability?: number
    gradient_health?: number
    convergence_quality?: number
    accuracy_consistency?: number
    [key: string]: unknown
  }
  training_stability?: Record<string, unknown>
  
  // Performance Data
  performance?: {
    total_score?: number
    accuracy?: number
    [key: string]: unknown
  }
  training_history?: Record<string, unknown>
  
  // Pruning Information
  pruning_info?: Record<string, unknown>
  
  // Plot Generation Progress
  plot_generation?: {
    status: 'generating' | 'completed' | 'failed'
    current_plot: string
    completed_plots: number
    total_plots: number
    plot_progress: number
  }
}

// Trial data from CSV
export interface TrialData {
  trial_number: number
  total_score: number
  state: 'COMPLETE' | 'FAILED' | 'PRUNED'
  duration_seconds: number
  overall_health: number
  hyperparameters: Hyperparameters
}

// Simplified trial for gallery view
export interface TrialSummary {
  id: number
  trial_number: number
  total_score: number
  overall_health: number
  duration_seconds: number
  architecture_type: 'CNN' | 'LSTM'
  layer_count: number
  parameter_count?: number
  key_features: string[]
}

// 3D Architecture representation
export interface LayerNode {
  id: string
  type: 'conv' | 'dense' | 'dropout' | 'pooling' | 'activation'
  position: [number, number, number]
  size: [number, number, number]
  parameters: {
    [key: string]: unknown
  }
  connections: string[]
  health_score?: number
  importance_score?: number
}

export interface Architecture3D {
  id: string
  trial_number: number
  layers: LayerNode[]
  performance: {
    accuracy: number
    health: number
    duration: number
  }
  metadata: {
    dataset: string
    total_score: number
    hyperparameters: Hyperparameters
  }
}