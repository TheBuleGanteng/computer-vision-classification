"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Select, SelectItem } from "@/components/ui/select"
import { Card, CardContent } from "@/components/ui/card"
import { Tooltip } from "@/components/ui/tooltip"
import { apiClient } from "@/lib/api-client"
import { useDashboard } from "./dashboard-provider"
import { 
  Play, 
  Square, 
  Download,
  Database,
  Target,
  Info
} from "lucide-react"

// Supported datasets from your optimization system
const DATASETS = [
  { value: "mnist", label: "MNIST (Images - Handwritten Digits)" },
  { value: "cifar10", label: "CIFAR-10 (Images - Objects)" },
  { value: "cifar100", label: "CIFAR-100 (Images - Objects)" },
  { value: "fashion-mnist", label: "Fashion-MNIST (Images - Clothing)" },
  { value: "gtsrb", label: "GTSRB (Images - Traffic Signs)" },
  { value: "imdb", label: "IMDB (Text - Sentiment)" },
  { value: "reuters", label: "Reuters (Text - Topics)" }
]

// Target metric options for optimization
const TARGET_METRICS = [
  { value: "Accuracy + model health", label: "Accuracy + model health", mode: "health" },
  { value: "Accuracy", label: "Accuracy", mode: "simple" }
]

export function OptimizationControls() {
  const { progress, optimizationMode, isOptimizationRunning, currentJobId, setProgress, setOptimizationMode, setIsOptimizationRunning, setCurrentJobId } = useDashboard()
  
  const [selectedDataset, setSelectedDataset] = useState("")
  const [selectedTargetMetric, setSelectedTargetMetric] = useState("") // Default to empty (placeholder state)
  const [isOptimizationCompleted, setIsOptimizationCompleted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [clientElapsedTime, setClientElapsedTime] = useState<number>(0)
  const [optimizationStartTime, setOptimizationStartTime] = useState<number | null>(null)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Real-time elapsed time counter
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    
    if (isOptimizationRunning && optimizationStartTime) {
      interval = setInterval(() => {
        const now = Date.now()
        const elapsedSeconds = Math.floor((now - optimizationStartTime) / 1000)
        setClientElapsedTime(elapsedSeconds)
      }, 1000) // Update every second
    } else {
      setClientElapsedTime(0)
    }

    return () => {
      if (interval) {
        clearInterval(interval)
      }
    }
  }, [isOptimizationRunning, optimizationStartTime])

  // Sync selectedTargetMetric with shared optimization mode
  useEffect(() => {
    if (selectedTargetMetric && selectedTargetMetric !== "") {
      const targetMetric = TARGET_METRICS.find(m => m.value === selectedTargetMetric)
      if (targetMetric) {
        setOptimizationMode(targetMetric.mode as "simple" | "health")
      }
    }
  }, [selectedTargetMetric, setOptimizationMode])

  const handleOptimizationToggle = async () => {
    if (isOptimizationRunning && currentJobId) {
      // Cancel optimization
      try {
        await apiClient.cancelJob(currentJobId)
        
        // Clear polling interval to prevent further updates
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null
        }
        
        setIsOptimizationRunning(false)
        setCurrentJobId(null)
        setOptimizationStartTime(null) // Clear timer state
        setClientElapsedTime(0)
        console.log("Optimization cancelled successfully")
      } catch (err) {
        console.error("Failed to cancel optimization:", err)
        setError(err instanceof Error ? err.message : "Failed to cancel optimization")
      }
    } else {
      // Start optimization
      try {
        setError(null)
        
        const targetMetric = TARGET_METRICS.find(m => m.value === selectedTargetMetric)
        const mode = targetMetric?.mode as 'simple' | 'health'
        
        const request = {
          // Core parameters
          dataset_name: selectedDataset,
          mode: mode
        }
        // All other parameters will use API defaults from OptimizationRequest

        console.log(`Starting optimization:`, request)
        
        const response = await apiClient.startOptimization(request)
        
        setIsOptimizationRunning(true)
        setIsOptimizationCompleted(false)
        setCurrentJobId(response.job_id)
        setOptimizationStartTime(Date.now()) // Record start time for real-time counter
        
        console.log(`Optimization started with job ID: ${response.job_id}`)
        
        // Start polling for progress updates
        startProgressPolling(response.job_id)
        
      } catch (err) {
        console.error("Failed to start optimization:", err)
        const errorMessage = err instanceof Error ? err.message : 
          (typeof err === 'string' ? err : `Unknown error: ${JSON.stringify(err)}`)
        setError(errorMessage)
      }
    }
  }

  const startProgressPolling = (jobId: string) => {
    // Clear any existing polling interval
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
    }
    
    const pollInterval = setInterval(async () => {
      try {
        const status = await apiClient.getJobStatus(jobId)
        console.log(`Polling update for job ${jobId}:`, status.progress)
        
        // Force state update by creating new object to ensure React re-renders
        if (status.progress) {
          setProgress({...status.progress})
        }
        
        if (status.status === 'completed') {
          setIsOptimizationRunning(false)
          setIsOptimizationCompleted(true)
          // Keep currentJobId for 3D visualization access - don't set to null
          setOptimizationStartTime(null) // Clear timer state
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
            pollingIntervalRef.current = null
          }
          console.log("Optimization completed successfully")
        } else if (status.status === 'failed' || status.status === 'cancelled') {
          setIsOptimizationRunning(false)
          setCurrentJobId(null)
          setProgress(null) // Reset progress data to clear UI statistics
          setOptimizationStartTime(null) // Clear timer state
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
            pollingIntervalRef.current = null
          }
          if (status.error) {
            setError(status.error)
          }
          console.log(`Optimization ${status.status}`)
        }
      } catch (err) {
        console.error("Failed to poll job status:", err)
        
        // If job is not found (e.g., server restarted), stop polling and reset state
        if (err instanceof Error && err.message.includes('not found')) {
          console.log("Job not found - likely server restarted. Stopping optimization state.")
          setIsOptimizationRunning(false)
          setIsOptimizationCompleted(false)
          setCurrentJobId(null)
          setProgress(null)
          setOptimizationStartTime(null) // Clear timer state
          setError("Optimization was interrupted (server restarted). Please start a new optimization.")
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
            pollingIntervalRef.current = null
          }
        }
        // Don't clear interval on other polling errors - backend might be temporarily unavailable
      }
    }, 2000) // Poll every 2 seconds
    
    // Store the interval reference
    pollingIntervalRef.current = pollInterval

    // Cleanup polling on component unmount or job completion
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
        pollingIntervalRef.current = null
      }
    }
  }

  // Helper function to format elapsed time and choose appropriate time source
  const formatElapsedTime = () => {
    // Use real-time client elapsed time when optimization is running, otherwise use server time
    const timeToUse = isOptimizationRunning ? clientElapsedTime : (progress?.elapsed_time || 0)
    const minutes = Math.floor(timeToUse / 60)
    const seconds = Math.round(timeToUse % 60)
    return `${minutes}m ${seconds}s`
  }

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex flex-col sm:flex-row items-center gap-4">
          {/* Dataset Selection */}
          <div className="flex items-center gap-3">
            <Database className="h-5 w-5 text-muted-foreground flex-shrink-0" />
            <div className="w-auto min-w-[280px]">
              <Select
                value={selectedDataset}
                onValueChange={setSelectedDataset}
                placeholder="Select dataset"
                disabled={isOptimizationRunning}
              >
                {DATASETS.map((dataset) => (
                  <SelectItem key={dataset.value} value={dataset.value}>
                    {dataset.label}
                  </SelectItem>
                ))}
              </Select>
            </div>
          </div>

          {/* Target Metric Selection */}
          <div className="flex items-center gap-3">
            <Target className="h-5 w-5 text-muted-foreground flex-shrink-0" />
            <div className="w-auto min-w-[240px]">
              <Select
                value={selectedTargetMetric}
                onValueChange={setSelectedTargetMetric}
                placeholder="Select target metric"
                disabled={isOptimizationRunning}
              >
                {TARGET_METRICS.map((metric) => (
                  <SelectItem key={metric.value} value={metric.value}>
                    {metric.label}
                  </SelectItem>
                ))}
              </Select>
            </div>
            <Tooltip
              content={
                <div className="space-y-2">
                  <p className="font-medium">Model health vs accuracy</p>
                  <p>
                    <strong>Accuracy only:</strong> Optimizes solely for prediction correctness on test data.
                  </p>
                  <p>
                    <strong>Accuracy + model health (recommended):</strong> Considers both accuracy and model robustness including gradient stability, loss convergence, dead/saturated filters, and training dynamics.
                  </p>
                  <p>
                    Model health metrics help identify architectures that not only perform well but are also stable during training and less prone to overfitting or gradient issues.
                  </p>
                  <div className="pt-2 border-t">
                    <p className="text-xs">
                      To learn more about this topic, click{" "}
                      <a 
                        href="https://cs231n.github.io/neural-networks-3/" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-700 underline font-medium"
                      >
                        here
                      </a>
                    </p>
                  </div>
                </div>
              }
            >
              <div className="flex items-center justify-center w-5 h-5 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-colors">
                <Info className="h-3 w-3" />
              </div>
            </Tooltip>
          </div>

          {/* Control Buttons */}
          <div className="flex items-center gap-3 flex-shrink-0">
            {/* Start/Cancel Optimization Toggle */}
            <Button
              onClick={handleOptimizationToggle}
              disabled={!selectedDataset || selectedDataset === "" || !selectedTargetMetric || selectedTargetMetric === ""}
              className={`min-w-[140px] ${
                isOptimizationRunning 
                  ? "bg-red-600 hover:bg-red-700 text-white" 
                  : "bg-green-600 hover:bg-green-700 text-white"
              }`}
            >
              {isOptimizationRunning ? (
                <>
                  <Square className="h-4 w-4 mr-2" />
                  Cancel optimization
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Start optimization
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Status Indicator */}
        {error && (
          <div className="mt-4 flex items-center gap-2 text-sm text-red-600 bg-red-50 p-3 rounded-md">
            <div className="w-2 h-2 bg-red-600 rounded-full" />
            <span>Error: {error}</span>
          </div>
        )}
        
        {isOptimizationRunning && (
          <div className="mt-4 space-y-2">
            <div className="flex items-center gap-2 text-sm text-blue-600">
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" />
              Optimization in progress for {DATASETS.find(d => d.value === selectedDataset)?.label}{" "} 
              using {selectedTargetMetric.toLowerCase()}...
            </div>
            
            {progress && (
              <div className="text-xs text-gray-600 space-y-1">
                <div>
                  <span>Progress: </span>
                </div>
                <div className="pl-2 space-y-1">
                  <div>
                    <span>Trials: </span>
                    <span className="font-medium">{progress.current_trial || 0}/{progress.total_trials || 20}</span>
                  </div>
                  {progress.current_epoch !== undefined && progress.total_epochs !== undefined && (
                    <div>
                      <span>Epoch: </span>
                      <span className="font-medium">{progress.current_epoch}/{progress.total_epochs}</span>
                      {progress.epoch_progress !== undefined && (
                        <div className="ml-2 mt-1">
                          <div className="w-32 bg-gray-200 rounded-full h-1.5">
                            <div 
                              className="bg-blue-600 h-1.5 rounded-full transition-all duration-300" 
                              style={{width: `${Math.max(0, Math.min(100, (progress.epoch_progress || 0) * 100))}%`}}
                            ></div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                {progress.best_total_score !== null && progress.best_total_score !== undefined && (
                  <div>
                    <span>Best total score {selectedTargetMetric === 'Accuracy + model health' ? 'accuracy + model health' : 'accuracy'}: </span>
                    <span className="font-medium">{(progress.best_total_score * 100).toFixed(1)}%</span>
                  </div>
                )}
                {(isOptimizationRunning || progress?.elapsed_time) && (
                  <div>
                    <span>Elapsed time: </span>
                    <span className="font-medium">{formatElapsedTime()}</span>
                  </div>
                )}
                {progress.status_message && (
                  <div className="text-xs text-gray-500 italic">
                    {progress.status_message}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        
        {isOptimizationCompleted && (
          <div className="mt-4 flex items-center gap-2 text-sm text-green-600 bg-green-50 p-3 rounded-md">
            <div className="w-2 h-2 bg-green-600 rounded-full" />
            <span>Optimization completed successfully! Results are available in the dashboard.</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

/* 
IMPLEMENTATION NOTES:
- Dataset dropdown includes all supported datasets from optimizer.py with classification type indicators
- Target metric dropdown allows selection between "simple" (accuracy only) and "health" (accuracy + model health)
- Default target metric is "health" mode (accuracy + model health) as recommended
- Start/Cancel button requires both dataset and target metric selection to be enabled
- Status indicator shows both selected dataset and target metric during optimization
- Mobile responsive design with stacked layout on small screens
- Both dataset and target metric values are logged and ready for backend API integration
- Currently uses mock state - will be connected to real optimization state in Phase 1.5
*/