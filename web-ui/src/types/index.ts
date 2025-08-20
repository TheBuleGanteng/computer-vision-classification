export * from './optimization'
export * from './api'

// UI-specific types
export interface ThemeConfig {
  mode: 'light' | 'dark'
  primaryColor: string
  accentColor: string
}

export interface ViewConfig {
  show3D: boolean
  showHealthMetrics: boolean
  showParameterImportance: boolean
  autoRotate: boolean
  cameraSpeed: number
  animationSpeed: number
}

export interface FilterConfig {
  minAccuracy: number
  maxAccuracy: number
  minHealth: number
  maxHealth: number
  datasets: string[]
  optimizers: string[]
  activations: string[]
}

export interface SortConfig {
  field: 'accuracy' | 'health' | 'duration' | 'trial_number'
  direction: 'asc' | 'desc'
}

export interface ChartData {
  labels: string[]
  datasets: {
    label: string
    data: number[]
    backgroundColor?: string | string[]
    borderColor?: string | string[]
    borderWidth?: number
  }[]
}