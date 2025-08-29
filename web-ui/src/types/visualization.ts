// TypeScript types for 3D Model Visualization
// Matches the backend data structures from model_visualizer.py

export interface LayerVisualization {
  layer_id: string;
  layer_type: string;
  position_z: number;
  width: number;
  height: number;
  depth: number;
  parameters: number;
  filters?: number;
  kernel_size?: [number, number];
  units?: number;
  color_intensity: number;
  opacity: number;
}

export interface ArchitectureVisualization {
  type: string; // "CNN" | "LSTM" | "Generic"
  layers: LayerVisualization[];
  total_parameters: number;
  model_depth: number;
  max_layer_width: number;
  max_layer_height: number;
  performance_score: number;
  health_score?: number;
}

export interface BestModelResponse {
  trial_id: string;
  trial_number: number;
  architecture: {
    type: string;
    layers: Array<{
      class_name: string;
      config: Record<string, unknown>;
    }>;
  };
  performance_score: number;
  health_score?: number;
  health_metrics?: {
    neuron_utilization: number;
    parameter_efficiency: number;
    training_stability: number;
    gradient_health: number;
    convergence_quality: number;
    accuracy_consistency: number;
  };
  visualization_data: ArchitectureVisualization;
}

export interface Model3DViewerProps {
  visualizationData: ArchitectureVisualization;
  isLoading?: boolean;
  error?: string;
  className?: string;
}

export interface Layer3DProps {
  layer: LayerVisualization;
  maxWidth: number;
  maxHeight: number;
  onClick?: (layer: LayerVisualization) => void;
  onHover?: (layer: LayerVisualization | null) => void;
}

export interface PerformanceColor {
  performance: string; // hex color based on performance_score
  health: string;      // hex color based on health_score  
}

export const getPerformanceColor = (score: number): string => {
  if (score >= 0.8) return "#10B981"; // Green for good performance
  if (score >= 0.6) return "#F59E0B"; // Yellow for medium performance
  return "#EF4444"; // Red for poor performance
};

export const getHealthColor = (score?: number): string => {
  if (score === undefined || score === null) return "#6B7280"; // Gray if no health data
  if (score >= 0.8) return "#10B981"; // Green for good health
  if (score >= 0.6) return "#F59E0B"; // Yellow for medium health
  return "#EF4444"; // Red for poor health
};