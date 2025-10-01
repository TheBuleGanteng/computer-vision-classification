"use client"

import React, { useState, useMemo, Suspense } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Info,
  Layers,
  Activity,
  Timer,
  Loader2
} from "lucide-react"
import { useTrials } from "@/hooks/use-trials"
import { useDashboard } from "@/components/dashboard/dashboard-provider"
// Lazy load heavy visualization component to prevent blocking renders
const UnifiedEducationalInterface = React.lazy(() => 
  import("@/components/visualization/unified-educational-interface").then(module => ({
    default: module.UnifiedEducationalInterface
  }))
)

const BestArchitectureView = React.memo(() => {
  const { currentJobId, isOptimizationRunning } = useDashboard()
  const { bestTrial, isLoading, error } = useTrials()
  const [isNewBest, setIsNewBest] = useState(false)

  // Memoize stable props to prevent unnecessary re-renders of expensive components
  const stableTrialId = useMemo(() => 
    bestTrial?.trial_number?.toString() || bestTrial?.trial_id, 
    [bestTrial?.trial_number, bestTrial?.trial_id]
  )
  
  const stableJobId = useMemo(() => currentJobId, [currentJobId])

  // Trigger animation when new best trial is detected (only when trial ID changes)
  React.useEffect(() => {
    if (bestTrial?.trial_id) {
      setIsNewBest(true)
      // Use requestIdleCallback for non-urgent animation cleanup
      const cleanup = () => setIsNewBest(false)
      
      if ('requestIdleCallback' in window) {
        const handle = requestIdleCallback(cleanup, { timeout: 3100 }) // 3 seconds + buffer
        return () => cancelIdleCallback(handle)
      } else {
        // Fallback to setTimeout for older browsers
        const timer = setTimeout(cleanup, 3000)
        return () => clearTimeout(timer)
      }
    }
  }, [bestTrial?.trial_id]) // Only depend on trial_id, not the entire object

  // Loading state
  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center h-96">
            <div className="text-center">
              <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading best model...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Error state
  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center h-96">
            <div className="text-center text-red-600">
              <div className="text-6xl mb-4">⚠️</div>
              <p className="text-lg">Error loading best model</p>
              <p className="text-sm text-gray-400 mt-2">{error}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Empty state when no optimization has been run or no completed trials
  if (!currentJobId || !bestTrial) {
    // Show different states based on whether optimization is running
    const isRunning = isOptimizationRunning && currentJobId;
    
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-3">
                Best architecture
                {isRunning && (
                  <Badge variant="outline" className="text-blue-600 border-blue-600">
                    <Timer className="h-3 w-3 mr-1" />
                    In Progress
                  </Badge>
                )}
              </CardTitle>
              <CardDescription className="mt-2">
                {isRunning 
                  ? "Optimization in progress - results will appear as trials complete"
                  : "Start an optimization to see results"
                }
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          <div className="relative">
            <div className={`w-full h-96 bg-muted/20 rounded-lg border-2 border-dashed flex items-center justify-center ${
              isRunning ? 'border-blue-300 bg-blue-50/50' : ''
            }`}>
              <div className="text-center text-muted-foreground">
                {isRunning ? (
                  <>
                    <Loader2 className="h-12 w-12 mx-auto mb-4 animate-spin text-blue-500" />
                    <p className="text-lg font-medium">Optimization running...</p>
                    <p className="text-sm mt-2">Best architecture will appear when first trial completes</p>
                  </>
                ) : (
                  <>
                    <Info className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p className="text-lg font-medium">Awaiting optimization results</p>
                    <p className="text-sm mt-2">Start an optimization to see architecture visualization</p>
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium mb-3">Architecture layers</h4>
              <div className="text-sm text-muted-foreground italic">
                {isRunning 
                  ? "Layers will appear when optimization completes first trial"
                  : "No layers to display - run optimization first"
                }
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium mb-3">Model health metrics</h4>
              <div className="text-sm text-muted-foreground italic">
                {isRunning 
                  ? "Health metrics will appear when optimization completes first trial"
                  : "No health metrics available - run optimization first"
                }
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={`transition-all duration-500 ${isNewBest ? 'ring-2 ring-green-500 ring-opacity-75 shadow-lg' : ''}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-3">
              Best architecture - Trial #{bestTrial.trial_number}
              <Badge variant="outline" className="text-green-600 border-green-600">
                {bestTrial.architecture?.type || 'Unknown'}
              </Badge>
              {isNewBest && (
                <Badge className="bg-green-500 text-white animate-pulse">
                  🏆 NEW BEST!
                </Badge>
              )}
            </CardTitle>
            <CardDescription className="mt-2">
              Total Score: {bestTrial.performance?.total_score ? `${(bestTrial.performance.total_score * 100).toFixed(2)}%` : 'N/A'} • 
              Accuracy: {bestTrial.performance?.accuracy ? `${(bestTrial.performance.accuracy * 100).toFixed(2)}%` : 'N/A'} • 
              Health: {bestTrial.health_metrics?.overall_health ? `${(bestTrial.health_metrics.overall_health * 100).toFixed(2)}%` : 'N/A'}
            </CardDescription>
          </div>
          
        </div>
      </CardHeader>
      
      <CardContent>
        {/* Real-time 3D Visualization */}
        <div className="relative min-h-[300px] sm:min-h-[400px] lg:min-h-[500px] bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-lg overflow-hidden">
          {stableJobId && stableTrialId ? (
            <Suspense fallback={
              <div className="w-full h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
                  <p className="text-sm text-gray-600">Loading visualization...</p>
                </div>
              </div>
            }>
              <UnifiedEducationalInterface 
                jobId={stableJobId} 
                trialId={stableTrialId}
                className="w-full h-full"
              />
            </Suspense>
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-center text-gray-400">
                <div className="text-6xl mb-4">📊</div>
                <p className="text-lg">No visualization available</p>
                <p className="text-sm mt-2">Complete a trial to view 3D model</p>
              </div>
            </div>
          )}
        </div>

        {/* Real-time Trial Details */}
        <div className="mt-4 sm:mt-6 grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
          {/* Architecture Details */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Architecture Details
            </h4>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 text-sm">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Type:</span>
                  <span className="font-medium">{bestTrial.architecture?.type || 'N/A'}</span>
                </div>
                {bestTrial.architecture?.conv_layers && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Conv Layers:</span>
                    <span className="font-medium">{bestTrial.architecture.conv_layers}</span>
                  </div>
                )}
                {bestTrial.architecture?.dense_layers && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Dense Layers:</span>
                    <span className="font-medium">{bestTrial.architecture.dense_layers}</span>
                  </div>
                )}
                {bestTrial.architecture?.activation && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Activation:</span>
                    <span className="font-medium">{bestTrial.architecture.activation}</span>
                  </div>
                )}
                {bestTrial.hyperparameters?.kernel_size !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Kernel Size:</span>
                    <span className="font-medium">
                      {Array.isArray(bestTrial.hyperparameters.kernel_size) 
                        ? (bestTrial.hyperparameters.kernel_size as (string | number)[]).join('×') 
                        : String(bestTrial.hyperparameters.kernel_size)}
                    </span>
                  </div>
                )}
                {bestTrial.hyperparameters?.kernel_initializer !== undefined && typeof bestTrial.hyperparameters.kernel_initializer === 'string' && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Kernel Init:</span>
                    <span className="font-medium">{bestTrial.hyperparameters.kernel_initializer}</span>
                  </div>
                )}
              </div>
              
              <div className="space-y-2">
                {bestTrial.architecture?.filters_per_layer && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Filters:</span>
                    <span className="font-medium">{bestTrial.architecture.filters_per_layer}</span>
                  </div>
                )}
                {bestTrial.architecture?.first_dense_nodes && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Dense Nodes:</span>
                    <span className="font-medium">{bestTrial.architecture.first_dense_nodes.toLocaleString()}</span>
                  </div>
                )}
                {bestTrial.architecture?.batch_normalization !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Batch Norm:</span>
                    <span className="font-medium">{bestTrial.architecture.batch_normalization ? 'Yes' : 'No'}</span>
                  </div>
                )}
                {bestTrial.architecture?.use_global_pooling !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Flattening:</span>
                    <span className="font-medium">{bestTrial.architecture.use_global_pooling ? 'No' : 'Yes'}</span>
                  </div>
                )}
                {bestTrial.hyperparameters?.epochs !== undefined && typeof bestTrial.hyperparameters.epochs === 'number' && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Epochs:</span>
                    <span className="font-medium">{bestTrial.hyperparameters.epochs}</span>
                  </div>
                )}
                {bestTrial.hyperparameters?.optimizer !== undefined && typeof bestTrial.hyperparameters.optimizer === 'string' && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Optimizer:</span>
                    <span className="font-medium">{bestTrial.hyperparameters.optimizer}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Performance & Health Metrics */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Performance & Health
            </h4>
            
            {/* Responsive layout: cards on large screens, key-value pairs on small screens */}
            <div className="hidden sm:grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-2 bg-muted/50 rounded">
                <div className="font-medium text-muted-foreground">Total Score</div>
                <div className="text-sm font-semibold">{bestTrial.performance?.total_score ? `${(bestTrial.performance.total_score * 100).toFixed(1)}%` : 'N/A'}</div>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <div className="font-medium text-muted-foreground">Accuracy</div>
                <div className="text-sm font-semibold">{bestTrial.performance?.accuracy ? `${(bestTrial.performance.accuracy * 100).toFixed(1)}%` : 'N/A'}</div>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <div className="font-medium text-muted-foreground">Test Loss</div>
                <div className="text-sm font-semibold">{bestTrial.health_metrics?.test_loss ? bestTrial.health_metrics.test_loss.toFixed(4) : 'N/A'}</div>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <div className="font-medium text-muted-foreground">Overall Health</div>
                <div className="text-sm font-semibold">{bestTrial.health_metrics?.overall_health ? `${(bestTrial.health_metrics.overall_health * 100).toFixed(1)}%` : 'N/A'}</div>
              </div>
              {bestTrial.health_metrics?.training_stability && (
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Training Stability</div>
                  <div className="text-sm font-semibold">{(bestTrial.health_metrics.training_stability * 100).toFixed(1)}%</div>
                </div>
              )}
              {bestTrial.health_metrics?.parameter_efficiency && (
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Param Efficiency</div>
                  <div className="text-sm font-semibold">{(bestTrial.health_metrics.parameter_efficiency * 100).toFixed(1)}%</div>
                </div>
              )}
              {bestTrial.hyperparameters?.learning_rate !== undefined && typeof bestTrial.hyperparameters.learning_rate === 'number' && (
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Learning Rate</div>
                  <div className="text-sm font-semibold">{bestTrial.hyperparameters.learning_rate}</div>
                </div>
              )}
              {bestTrial.hyperparameters?.first_hidden_layer_dropout !== undefined && typeof bestTrial.hyperparameters.first_hidden_layer_dropout === 'number' && (
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Dropout Rate</div>
                  <div className="text-sm font-semibold">{(bestTrial.hyperparameters.first_hidden_layer_dropout * 100).toFixed(1)}%</div>
                </div>
              )}
            </div>
            
            {/* Mobile layout with left-justified labels and right-justified values */}
            <div className="sm:hidden space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Score:</span>
                <span className="font-semibold">{bestTrial.performance?.total_score ? `${(bestTrial.performance.total_score * 100).toFixed(1)}%` : 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Accuracy:</span>
                <span className="font-semibold">{bestTrial.performance?.accuracy ? `${(bestTrial.performance.accuracy * 100).toFixed(1)}%` : 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Test Loss:</span>
                <span className="font-semibold">{bestTrial.health_metrics?.test_loss ? bestTrial.health_metrics.test_loss.toFixed(4) : 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Overall Health:</span>
                <span className="font-semibold">{bestTrial.health_metrics?.overall_health ? `${(bestTrial.health_metrics.overall_health * 100).toFixed(1)}%` : 'N/A'}</span>
              </div>
              {bestTrial.health_metrics?.training_stability && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Training Stability:</span>
                  <span className="font-semibold">{(bestTrial.health_metrics.training_stability * 100).toFixed(1)}%</span>
                </div>
              )}
              {bestTrial.health_metrics?.parameter_efficiency && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Param Efficiency:</span>
                  <span className="font-semibold">{(bestTrial.health_metrics.parameter_efficiency * 100).toFixed(1)}%</span>
                </div>
              )}
              {bestTrial.hyperparameters?.learning_rate !== undefined && typeof bestTrial.hyperparameters.learning_rate === 'number' && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Learning Rate:</span>
                  <span className="font-semibold">{bestTrial.hyperparameters.learning_rate}</span>
                </div>
              )}
              {bestTrial.hyperparameters?.first_hidden_layer_dropout !== undefined && typeof bestTrial.hyperparameters.first_hidden_layer_dropout === 'number' && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Dropout Rate:</span>
                  <span className="font-semibold">{(bestTrial.hyperparameters.first_hidden_layer_dropout * 100).toFixed(1)}%</span>
                </div>
              )}
            </div>
          </div>
        </div>

      </CardContent>
    </Card>
  )
})

BestArchitectureView.displayName = 'BestArchitectureView'

export { BestArchitectureView }