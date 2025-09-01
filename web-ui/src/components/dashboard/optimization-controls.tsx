"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Select, SelectItem } from "@/components/ui/select"
import { Card, CardContent } from "@/components/ui/card"
import { Tooltip } from "@/components/ui/tooltip"
import { apiClient } from "@/lib/api-client"
import { useDashboard } from "./dashboard-provider"
import { useComprehensiveStatus } from "@/hooks/use-comprehensive-status"
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
  const { progress, isOptimizationRunning, currentJobId, setProgress, setOptimizationMode, setIsOptimizationRunning, setCurrentJobId } = useDashboard()
  const { jobStatus, elapsedSeconds, trials } = useComprehensiveStatus()
  
  // Sync comprehensive status data with dashboard context
  useEffect(() => {
    if (jobStatus?.progress) {
      // Create enhanced progress object with proper typing
      const enhancedProgress: any = { ...jobStatus.progress }
      
      // Extract plot_generation from the current running trial
      if (trials && trials.length > 0) {
        const runningTrial = trials.find(trial => trial.status === 'running')
        if (runningTrial && (runningTrial as any).plot_generation) {
          enhancedProgress.plot_generation = (runningTrial as any).plot_generation
        }
      }
      
      // Final model building progress comes from job_status.progress directly (not from trials)
      if (jobStatus.progress && (jobStatus.progress as any).final_model_building) {
        enhancedProgress.final_model_building = (jobStatus.progress as any).final_model_building
      }
      
      setProgress(enhancedProgress)
    }
  }, [jobStatus?.progress, trials, setProgress])

  // Monitor job status changes for completion/failure
  useEffect(() => {
    if (!jobStatus) return

    if (jobStatus.status === 'completed') {
      setIsOptimizationRunning(false)
      setIsOptimizationCompleted(true)
      // Keep currentJobId for 3D visualization access
      
      // Check if final optimized model is available for download
      if (currentJobId) {
        checkModelAvailability(currentJobId)
      }
      console.log("Optimization completed successfully")
    } else if (jobStatus.status === 'failed' || jobStatus.status === 'cancelled') {
      setIsOptimizationRunning(false)
      setCurrentJobId(null)
      setIsModelAvailable(false)
      setProgress(null)
      if (jobStatus.error) {
        setError(jobStatus.error)
      }
      console.log(`Optimization ${jobStatus.status}`)
    }
  }, [jobStatus?.status, jobStatus?.error, currentJobId])
  
  const [selectedDataset, setSelectedDataset] = useState("")
  const [selectedTargetMetric, setSelectedTargetMetric] = useState("") // Default to empty (placeholder state)
  const [isOptimizationCompleted, setIsOptimizationCompleted] = useState(false)
  const [isModelAvailable, setIsModelAvailable] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)

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
        setIsModelAvailable(false) // Reset model availability on cancellation
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
        setIsModelAvailable(false) // Reset model availability for new optimization
        setCurrentJobId(response.job_id)
        
        console.log(`Optimization started with job ID: ${response.job_id}`)
        
        // Start polling for progress updates
        startProgressPolling()
        
      } catch (err) {
        console.error("Failed to start optimization:", err)
        const errorMessage = err instanceof Error ? err.message : 
          (typeof err === 'string' ? err : `Unknown error: ${JSON.stringify(err)}`)
        setError(errorMessage)
      }
    }
  }

  const checkModelAvailability = async (jobId: string) => {
    try {
      const results = await apiClient.getJobResults(jobId)
      // Check if the final model has been built and is available for download
      const hasModelPath = results?.model_result?.model_path
      setIsModelAvailable(!!hasModelPath)
      console.log("Model availability check:", hasModelPath ? "Available" : "Not available")
    } catch (err) {
      console.error("Failed to check model availability:", err)
      setIsModelAvailable(false)
    }
  }

  const handleDownloadModel = async () => {
    if (!currentJobId) return
    
    try {
      const downloadUrl = apiClient.getModelDownloadUrl(currentJobId)
      const defaultFilename = `optimized_model_${currentJobId}.zip`
      
      // Try modern File System Access API first (Chrome/Edge)
      if ('showSaveFilePicker' in window) {
        try {
          const fileHandle = await (window as unknown as {showSaveFilePicker: (options: unknown) => Promise<FileSystemFileHandle>}).showSaveFilePicker({
            suggestedName: defaultFilename,
            types: [{
              description: 'ZIP Archive',
              accept: { 'application/zip': ['.zip'] }
            }]
          })
          
          // Fetch the file data
          const response = await fetch(downloadUrl)
          if (!response.ok) throw new Error(`Download failed: ${response.statusText}`)
          
          const blob = await response.blob()
          const writable = await fileHandle.createWritable()
          await writable.write(blob)
          await writable.close()
          
          console.log("Model package saved to user-selected location")
          return
        } catch (err: unknown) {
          // User cancelled the save dialog or API not supported
          if ((err as Error).name === 'AbortError') {
            console.log("User cancelled save dialog")
            return
          }
          console.log("File System Access API failed, falling back to traditional method")
        }
      }
      
      // Fallback: Traditional download with "Save As" dialog
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = defaultFilename
      // Force download dialog by setting these attributes
      link.style.display = 'none'
      document.body.appendChild(link)
      link.click()
      
      // Clean up immediately using non-blocking approach
      if ('requestIdleCallback' in window) {
        requestIdleCallback(() => {
          document.body.removeChild(link)
        }, { timeout: 150 })
      } else {
        // Fallback for older browsers
        setTimeout(() => {
          document.body.removeChild(link)
        }, 100)
      }
      
      console.log("Model package download initiated - save dialog should appear")
    } catch (err) {
      console.error("Failed to download model package:", err)
      setError(err instanceof Error ? err.message : "Failed to download model package")
    }
  }

  const startProgressPolling = () => {
    // Clear any existing polling interval
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
    }
    
    // No longer need separate polling - comprehensive status hook handles this
    return () => {}  // Cleanup not needed since useComprehensiveStatus handles polling
  }

  // Helper function to format elapsed time from unified status
  const formatElapsedTime = () => {
    const timeToUse = elapsedSeconds || (progress?.elapsed_time || 0)
    const minutes = Math.floor(timeToUse / 60)
    const seconds = Math.round(timeToUse % 60)
    return `${minutes}m ${seconds}s`
  }

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-4">
          {/* Dataset Selection */}
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <Database className="h-5 w-5 text-muted-foreground flex-shrink-0" />
            <div className="w-full sm:w-auto sm:min-w-[280px]">
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
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <Target className="h-5 w-5 text-muted-foreground flex-shrink-0" />
            <div className="w-full sm:w-auto sm:min-w-[240px]">
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
          <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto sm:flex-shrink-0">
            {/* Start/Cancel Optimization Toggle */}
            <Button
              onClick={handleOptimizationToggle}
              disabled={!selectedDataset || selectedDataset === "" || !selectedTargetMetric || selectedTargetMetric === ""}
              className={`w-full sm:w-auto sm:min-w-[140px] ${
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

            {/* Download Optimized Model Button */}
            <Button
              onClick={handleDownloadModel}
              disabled={!isModelAvailable || !currentJobId}
              className="w-full sm:w-auto sm:min-w-[180px] bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white"
              title={
                !isModelAvailable 
                  ? "Model package will be available after optimization completes and final model is built"
                  : "Download ZIP archive containing .keras model file, hyperparameters metadata, and usage guide"
              }
            >
              <Download className="h-4 w-4 mr-2" />
              Download Model
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
                  {progress.plot_generation && progress.plot_generation.status === 'generating' && (
                    <div>
                      <span>Plots: </span>
                      <span className="font-medium">{progress.plot_generation.current_plot}</span>
                      <span className="ml-1 text-gray-500">({progress.plot_generation.completed_plots}/{progress.plot_generation.total_plots})</span>
                      <div className="ml-2 mt-1">
                        <div className="w-32 bg-gray-200 rounded-full h-1.5">
                          <div 
                            className="bg-blue-600 h-1.5 rounded-full transition-all duration-300" 
                            style={{width: `${Math.max(0, Math.min(100, (progress.plot_generation.plot_progress || 0) * 100))}%`}}
                          ></div>
                        </div>
                      </div>
                    </div>
                  )}
                  {progress.final_model_building && progress.final_model_building.status === 'building' && (
                    <div>
                      <span>Final Model: </span>
                      <span className="font-medium">{progress.final_model_building.current_step}</span>
                      {progress.final_model_building.detailed_info && (
                        <span className="ml-1 text-gray-500">({progress.final_model_building.detailed_info})</span>
                      )}
                      <div className="ml-2 mt-1">
                        <div className="w-32 bg-gray-200 rounded-full h-1.5">
                          <div 
                            className="bg-blue-600 h-1.5 rounded-full transition-all duration-300" 
                            style={{width: `${Math.max(0, Math.min(100, (progress.final_model_building.progress || 0) * 100))}%`}}
                          ></div>
                        </div>
                      </div>
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