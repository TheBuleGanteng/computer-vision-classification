"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from "@/components/ui/dialog"
import { useDashboard } from "./dashboard-provider"
import { apiClient } from "@/lib/api-client"
import { TrialProgress } from "@/types/optimization"
import { 
  Eye,
  Target,
  Activity,
  Clock,
  Layers,
  Hash,
  Download,
  Loader2
} from "lucide-react"
import { UnifiedEducationalInterface } from "@/components/visualization/unified-educational-interface"
import { useModelVisualization, useVisualizationDownload } from "@/hooks/use-model-visualization"

// Modern Educational Visualization Container Component
function EducationalVisualizationContainer({ 
  jobId, 
  selectedTrial 
}: { 
  jobId: string | null; 
  selectedTrial: TrialProgress | null;
}) {
  const { data: visualizationData } = useModelVisualization(
    jobId, 
    !!jobId && !!selectedTrial
  );

  const downloadMutation = useVisualizationDownload();

  const handleDownload = () => {
    if (jobId) {
      downloadMutation.mutate({ jobId });
    }
  };

  return (
    <>
      <div className="w-full h-96 bg-gradient-to-br from-gray-900 to-black rounded-lg border overflow-hidden">
        <UnifiedEducationalInterface
          jobId={jobId || ''}
          trialId={selectedTrial?.trial_id}
          className="w-full h-full"
        />
      </div>
      
      {/* Visualization Controls */}
      <div className="absolute top-4 right-4 flex gap-2">
        <Button 
          variant="outline" 
          size="sm"
          onClick={handleDownload}
          disabled={!visualizationData || downloadMutation.isPending}
        >
          {downloadMutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Download className="h-4 w-4" />
          )}
        </Button>
      </div>
    </>
  );
}

export function TrialGallery() {
  const { isOptimizationRunning, currentJobId } = useDashboard()
  const [selectedTrial, setSelectedTrial] = useState<TrialProgress | null>(null)
  const [trials, setTrials] = useState<TrialProgress[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastJobId, setLastJobId] = useState<string | null>(null)

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
  }

  // Fetch trial data from API
  const fetchTrialHistory = async (jobId: string) => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await apiClient.getTrialHistory(jobId)
      const rawTrials = response.trials || []
      
      // 🔍 DEBUG: Log all trial data received from API
      console.log('🔍 FRONTEND TRIAL DATA DEBUG:')
      console.log(`  📊 Total trials received: ${rawTrials.length}`)
      if (rawTrials.length > 0) {
        const sampleTrial = rawTrials[0]
        console.log(`  📊 Sample trial structure:`, sampleTrial)
        console.log(`  📊 Sample performance:`, sampleTrial.performance)
        console.log(`  📊 Sample health_metrics:`, sampleTrial.health_metrics)
        if (sampleTrial.health_metrics) {
          console.log(`  📊 Health metrics keys:`, Object.keys(sampleTrial.health_metrics))
          console.log(`  📊 convergence_quality:`, sampleTrial.health_metrics.convergence_quality)
          console.log(`  📊 accuracy_consistency:`, sampleTrial.health_metrics.accuracy_consistency)
          console.log(`  📊 gradient_health:`, sampleTrial.health_metrics.gradient_health)
        }
      }

      // Deduplicate trials by trial_id and trial_number to prevent duplicate keys
      const uniqueTrials = rawTrials.filter((trial, index, array) => {
        const firstIndex = array.findIndex(t => 
          (t.trial_id && trial.trial_id && t.trial_id === trial.trial_id) ||
          (t.trial_number !== undefined && trial.trial_number !== undefined && t.trial_number === trial.trial_number)
        )
        return firstIndex === index
      })
      
      setTrials(uniqueTrials)
      console.log('Fetched trial history:', response)
      console.log('Deduplicated trials:', uniqueTrials)
    } catch (err) {
      console.error('Failed to fetch trial history:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch trial data')
      // Fallback to empty array on error
      setTrials([])
    } finally {
      setIsLoading(false)
    }
  }

  // Effect to fetch trial data when job ID changes
  useEffect(() => {
    if (currentJobId) {
      // If this is a new job ID (different from the last one), clear previous trials
      if (lastJobId && currentJobId !== lastJobId) {
        setTrials([])
      }
      setLastJobId(currentJobId)
      fetchTrialHistory(currentJobId)
    }
    // Note: Don't clear trials when currentJobId becomes null to preserve them after cancellation
  }, [currentJobId, lastJobId])

  // Polling effect to update trial data during optimization
  useEffect(() => {
    if (!currentJobId) return

    // Poll more frequently during optimization, less frequently when not running
    const pollFrequency = isOptimizationRunning ? 2000 : 10000 // 2s when running, 10s when idle
    
    const pollInterval = setInterval(() => {
      fetchTrialHistory(currentJobId)
    }, pollFrequency)

    return () => clearInterval(pollInterval)
  }, [currentJobId, isOptimizationRunning])

  // Also poll immediately when optimization status changes
  useEffect(() => {
    if (currentJobId && isOptimizationRunning) {
      fetchTrialHistory(currentJobId)
    }
  }, [isOptimizationRunning, currentJobId])

  const handleTrialClick = (trial: TrialProgress) => {
    setSelectedTrial(trial)
  }

  const handleCloseDialog = () => {
    setSelectedTrial(null)
  }

  // Find the best trial based on highest total_score
  const findBestTrial = () => {
    if (!trials || trials.length === 0) return null
    
    // Only consider completed trials for best trial selection
    const completedTrials = trials.filter(trial => 
      trial.status === 'completed' && 
      trial.performance?.total_score !== undefined && 
      trial.performance?.total_score !== null
    )
    
    if (completedTrials.length === 0) return null
    
    return completedTrials.reduce((best, current) => {
      const bestScore = best.performance?.total_score || 0
      const currentScore = current.performance?.total_score || 0
      return currentScore > bestScore ? current : best
    })
  }

  const bestTrial = findBestTrial()

  // Debug logging for best trial identification
  useEffect(() => {
    if (bestTrial) {
      console.log(`🏆 BEST TRIAL IDENTIFIED: Trial ${bestTrial.trial_number} with score ${((bestTrial.performance?.total_score ?? 0) * 100).toFixed(1)}%`)
    }
  }, [bestTrial])

  return (
    <>
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-xl font-semibold">Trial results</h3>
              <p className="text-sm text-muted-foreground">
                Click any trial to view its 3D architecture visualization
              </p>
            </div>
            <Badge variant="outline">
              {trials.filter(trial => trial.status === 'completed').length} trials completed
            </Badge>
          </div>

          {/* Loading State */}
          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin mr-2" />
              <span>Loading trial data...</span>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
              <p className="text-red-700 text-sm">
                Error loading trial data: {error}
              </p>
            </div>
          )}

          {/* Empty State */}
          {!isLoading && !error && trials.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <Layers className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No trials available yet.</p>
              <p className="text-sm">Trials will appear here as optimization progresses.</p>
            </div>
          )}

          {/* Trial Grid */}
          {!isLoading && !error && trials.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {trials.map((trial, index) => {
                // Calculate trial number for display
                const displayTrialNumber = trial.trial_number !== undefined && trial.trial_number !== null 
                  ? trial.trial_number 
                  : trials.length - index  // For in-progress trials, calculate based on position
                
                // Create a robust unique key combining multiple identifiers
                const uniqueKey = `trial_${trial.trial_id || ''}_${displayTrialNumber}_${index}`
                
                // Check if this is the best trial
                const isBestTrial = bestTrial && (
                  (trial.trial_id && trial.trial_id === bestTrial.trial_id) ||
                  (trial.trial_number !== undefined && trial.trial_number === bestTrial.trial_number)
                )
                
                return (
                <Card 
                  key={uniqueKey} 
                  className={`cursor-pointer transition-all duration-300 min-h-[200px] ${
                    isBestTrial 
                      ? "border-4 border-orange-500 shadow-lg shadow-orange-500/25 hover:shadow-xl hover:shadow-orange-500/30 ring-4 ring-orange-200 dark:ring-orange-500/20" 
                      : "border hover:shadow-md hover:border-primary/50"
                  }`}
                  onClick={() => handleTrialClick(trial)}
              >
                <CardContent className="p-4 flex flex-col h-full">
                  {/* Trial Header */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Hash className="h-4 w-4 text-muted-foreground" />
                      <span className="font-semibold">Trial {displayTrialNumber}</span>
                      {isBestTrial && (
                        <Badge className="bg-orange-500 hover:bg-orange-600 text-white text-xs font-bold animate-pulse">
                          🏆 BEST
                        </Badge>
                      )}
                    </div>
                    <Badge 
                      variant={
                        trial.status === 'completed' ? "default" : 
                        trial.status === 'running' ? "secondary" : 
                        trial.status === 'failed' ? "destructive" : 
                        trial.status === 'pruned' ? "outline" : 
                        "secondary"
                      }
                      className={
                        trial.status === 'completed' ? "bg-green-500 hover:bg-green-600 text-white font-normal" :
                        trial.status === 'running' ? "bg-yellow-500 hover:bg-yellow-600 text-white font-normal" :
                        trial.status === 'failed' ? "bg-red-500 hover:bg-red-600 text-white font-normal" :
                        trial.status === 'pruned' ? "bg-gray-500 hover:bg-gray-600 text-white font-normal" :
                        "bg-gray-400 hover:bg-gray-500 text-white font-normal"
                      }
                    >
                      {trial.status === 'running' ? 'In Progress' : 
                       trial.status === 'completed' ? 'Completed' :
                       trial.status === 'failed' ? 'Failed' :
                       trial.status === 'pruned' ? 'Pruned' :
                       'Unknown'}
                    </Badge>
                  </div>

                  {/* Basic Info */}
                  <div className="space-y-2 mb-3">
                    {trial.started_at && (
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-1">
                          <Eye className="h-3 w-3 text-blue-500" />
                          <span className="text-muted-foreground">Started:</span>
                        </div>
                        <span className="font-medium text-xs">{new Date(trial.started_at).toLocaleTimeString()}</span>
                      </div>
                    )}
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3 text-orange-500" />
                        <span className="text-muted-foreground">Duration:</span>
                      </div>
                      <span className="font-medium">
                        {trial.duration_seconds ? formatDuration(trial.duration_seconds) : 'N/A'}
                      </span>
                    </div>
                  </div>

                  {/* Architecture Info */}
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center gap-1 text-sm">
                      <Layers className="h-3 w-3 text-indigo-500" />
                      <span className="text-muted-foreground">Architecture</span>
                    </div>
                    
                    {trial.architecture ? (
                      <div className="text-xs space-y-1">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Type:</span>
                          <span className="font-medium">{trial.architecture.type}</span>
                        </div>
                        
                        {trial.architecture.type === 'CNN' && (
                          <>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Conv Layers:</span>
                              <span className="font-medium">{trial.architecture.conv_layers?.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Filters:</span>
                              <span className="font-medium">{trial.architecture.filters_per_layer?.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Kernel Size (pixels):</span>
                              <span className="font-medium">
                                {Array.isArray(trial.architecture.kernel_size) 
                                  ? `${trial.architecture.kernel_size[0]}×${trial.architecture.kernel_size[1]}`
                                  : String(trial.architecture.kernel_size || 'N/A')}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Dense Layers:</span>
                              <span className="font-medium">{trial.architecture.dense_layers?.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Dense Nodes:</span>
                              <span className="font-medium">{trial.architecture.first_dense_nodes?.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Activation:</span>
                              <span className="font-medium capitalize">{trial.architecture.activation}</span>
                            </div>
                          </>
                        )}
                        
                        {trial.architecture.type === 'LSTM' && (
                          <>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">LSTM Units:</span>
                              <span className="font-medium">{trial.architecture.lstm_units?.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Dense Layers:</span>
                              <span className="font-medium">{trial.architecture.dense_layers?.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Dense Nodes:</span>
                              <span className="font-medium">{trial.architecture.first_dense_nodes?.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Activation:</span>
                              <span className="font-medium capitalize">{trial.architecture.activation}</span>
                            </div>
                          </>
                        )}
                      </div>
                    ) : (
                      <div className="text-xs text-muted-foreground">
                        Architecture data pending...
                      </div>
                    )}
                  </div>

                  {/* Performance Metrics */}
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center gap-1 text-sm">
                      <Target className="h-3 w-3 text-green-500" />
                      <span className="text-muted-foreground">Performance</span>
                    </div>
                    
                    {/* Core Performance - Always Show Structure */}
                    <div className="text-xs space-y-1">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total Score:</span>
                        <span className="font-medium">
                          {trial.performance?.total_score ? `${(trial.performance.total_score * 100).toFixed(1)}%` : 'N/A'}
                        </span>
                      </div>
                      
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Accuracy:</span>
                        <span className="font-medium">
                          {trial.performance?.accuracy ? `${(trial.performance.accuracy * 100).toFixed(1)}%` : 'N/A'}
                        </span>
                      </div>
                      
                      {/* Show loss for completed trials */}
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Final Loss:</span>
                        <span className="font-medium">
                          {trial.health_metrics?.test_loss ? trial.health_metrics.test_loss.toFixed(4) : 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Health Analysis Section (Health Mode Only) */}
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center gap-1 text-sm">
                      <Activity className="h-3 w-3 text-purple-500" />
                      <span className="text-muted-foreground">Health Analysis</span>
                    </div>
                    
                    <div className="text-xs space-y-1">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Overall Health:</span>
                        <span className="font-medium text-purple-600">
                          {trial.health_metrics?.overall_health ? `${(trial.health_metrics.overall_health * 100).toFixed(1)}%` : 'N/A'}
                        </span>
                      </div>
                      
                      {/* Health Components with Weights */}
                      <div className="pl-2 space-y-1 border-l border-purple-200">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Neuron Usage:</span>
                          <span className="font-medium">
                            {trial.health_metrics?.neuron_utilization ? `${(trial.health_metrics.neuron_utilization * 100).toFixed(1)}%` : 'N/A'}
                          </span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Efficiency:</span>
                          <span className="font-medium">
                            {trial.health_metrics?.parameter_efficiency ? `${(trial.health_metrics.parameter_efficiency * 100).toFixed(1)}%` : 'N/A'}
                          </span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Stability:</span>
                          <span className="font-medium">
                            {trial.health_metrics?.training_stability ? `${(trial.health_metrics.training_stability * 100).toFixed(1)}%` : 'N/A'}
                          </span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Gradients:</span>
                          <span className="font-medium">
                            {trial.health_metrics?.gradient_health ? `${(trial.health_metrics.gradient_health * 100).toFixed(1)}%` : 'N/A'}
                          </span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Convergence:</span>
                          <span className="font-medium">
                            {trial.health_metrics?.convergence_quality ? `${(trial.health_metrics.convergence_quality * 100).toFixed(1)}%` : 'N/A'}
                          </span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Consistency:</span>
                          <span className="font-medium">
                            {trial.health_metrics?.accuracy_consistency ? `${(trial.health_metrics.accuracy_consistency * 100).toFixed(1)}%` : 'N/A'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Spacer to push Trial ID to bottom */}
                  <div className="flex-grow"></div>

                  {/* Trial ID and 3D Model Button */}
                  <div className="mt-3 pt-2 border-t border-border/50 space-y-2">
                    <p className="text-xs text-muted-foreground">
                      ID: {trial.trial_id || 'N/A'}
                    </p>
                    
                    {/* 3D Model Button for Best Trial */}
                    {isBestTrial && (
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="w-full text-xs bg-orange-50 border-orange-200 hover:bg-orange-100 dark:bg-orange-950/20 dark:border-orange-800 dark:hover:bg-orange-900/30"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleTrialClick(trial);
                        }}
                      >
                        <Layers className="h-3 w-3 mr-1" />
                        View 3D Model
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
              )
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 3D Architecture Modal */}
      <Dialog open={!!selectedTrial} onOpenChange={() => setSelectedTrial(null)}>
        <DialogContent className="max-w-4xl w-full max-h-[90vh] overflow-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-3">
              Model visualization: Trial #{selectedTrial?.trial_number}
              <Badge variant="outline">{selectedTrial?.architecture?.type || 'Unknown'}</Badge>
            </DialogTitle>
            <DialogClose onClose={handleCloseDialog} />
          </DialogHeader>
          
          {selectedTrial && (
            <div className="space-y-6">
              {/* Trial Summary */}
              <div className="grid grid-cols-3 gap-4 p-4 bg-muted/20 rounded-lg">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Total Score</p>
                  <p className="text-lg font-bold">{selectedTrial.performance?.total_score ? `${(selectedTrial.performance.total_score * 100).toFixed(1)}%` : 'N/A'}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Accuracy</p>
                  <p className="text-lg font-bold">{selectedTrial.performance?.accuracy ? `${(selectedTrial.performance.accuracy * 100).toFixed(1)}%` : 'N/A'}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Total Loss</p>
                  <p className="text-lg font-bold">{selectedTrial.health_metrics?.test_loss ? selectedTrial.health_metrics.test_loss.toFixed(4) : 'N/A'}</p>
                </div>
              </div>

              {/* 3D Visualization Area */}
              <div className="relative">
                <EducationalVisualizationContainer
                  jobId={currentJobId}
                  selectedTrial={selectedTrial}
                />
              </div>

              {/* Architecture Details */}
              <div className="space-y-4">
                <h4 className="font-medium">Architecture details</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Conv Layers</p>
                    <p className="font-medium">{String(selectedTrial.architecture?.convLayers || 'N/A')}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Dense Layers</p>
                    <p className="font-medium">{String(selectedTrial.architecture?.denseLayers || 'N/A')}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Total Layers</p>
                    <p className="font-medium">{String(selectedTrial.architecture?.totalLayers || 'N/A')}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Parameters</p>
                    <p className="font-medium">{selectedTrial.architecture?.parameters?.toLocaleString() || 'N/A'}</p>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Filter Sizes:</p>
                  <div className="flex gap-2">
                    {(selectedTrial.architecture?.filterSizes || []).map((size: number, index: number) => (
                      <Badge key={index} variant="outline">{size}</Badge>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Activations:</p>
                  <div className="flex gap-2">
                    {(selectedTrial.architecture?.activations || []).map((activation: string, index: number) => (
                      <Badge key={index} variant="outline">{activation}</Badge>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Key Features:</p>
                  <div className="flex flex-wrap gap-2">
                    {(selectedTrial.architecture?.keyFeatures || []).map((feature: string, index: number) => (
                      <Badge key={index} variant="secondary">{feature}</Badge>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}

/* 
IMPLEMENTATION NOTES:
- Grid layout: responsive from 1 column (mobile) to 4 columns (desktop)
- Trial tiles sorted by most recent first (trial number descending)
- Click behavior opens modal overlay with 3D visualization
- Architecture summary matches format from Best Architecture display
- Modal includes:
  - Trial performance summary
  - 3D visualization area (placeholder for React Three Fiber)
  - Interactive 3D controls (zoom, rotate, reset, download)
  - Detailed architecture breakdown
- Mobile responsive design with single column on small screens
- Currently uses mock trial data matching your optimization results structure
- Will be connected to real trial data in Phase 1.5
- 3D visualization placeholder will be replaced with actual Three.js implementation in Phase 2
*/