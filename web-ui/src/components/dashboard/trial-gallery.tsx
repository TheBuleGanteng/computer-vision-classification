"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from "@/components/ui/dialog"
import { useDashboard } from "./dashboard-provider"
import { apiClient } from "@/lib/api-client"
import { 
  Eye,
  Target,
  Activity,
  Clock,
  Layers,
  Hash,
  Download,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Loader2
} from "lucide-react"

export function TrialGallery() {
  const { progress, isOptimizationRunning, currentJobId } = useDashboard()
  const [selectedTrial, setSelectedTrial] = useState<any | null>(null)
  const [trials, setTrials] = useState<any[]>([])
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

  const handleTrialClick = (trial: any) => {
    setSelectedTrial(trial)
  }

  const handleCloseDialog = () => {
    setSelectedTrial(null)
  }

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
                return (
                <Card 
                  key={uniqueKey} 
                  className="cursor-pointer hover:shadow-md transition-all duration-200 hover:border-primary/50"
                  onClick={() => handleTrialClick(trial)}
              >
                <CardContent className="p-4">
                  {/* Trial Header */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Hash className="h-4 w-4 text-muted-foreground" />
                      <span className="font-semibold">Trial {displayTrialNumber}</span>
                    </div>
                    <Badge 
                      variant={
                        trial.status === 'completed' ? "default" : 
                        trial.status === 'running' ? "secondary" : 
                        trial.status === 'failed' ? "destructive" : 
                        trial.status === 'cancelled' ? "outline" : 
                        "secondary"
                      }
                      className={
                        trial.status === 'completed' ? "bg-green-500 hover:bg-green-600 text-white font-normal" :
                        trial.status === 'running' ? "bg-yellow-500 hover:bg-yellow-600 text-white font-normal" :
                        trial.status === 'failed' ? "bg-red-500 hover:bg-red-600 text-white font-normal" :
                        trial.status === 'cancelled' ? "bg-gray-500 hover:bg-gray-600 text-white font-normal" :
                        "bg-gray-400 hover:bg-gray-500 text-white font-normal"
                      }
                    >
                      {trial.status === 'running' ? 'In Progress' : 
                       trial.status === 'completed' ? 'Completed' :
                       trial.status === 'failed' ? 'Failed' :
                       trial.status === 'cancelled' ? 'Cancelled' :
                       'Unknown'}
                    </Badge>
                  </div>

                  {/* Basic Info */}
                  <div className="space-y-2 mb-3">
                    {trial.started_at && (
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-1">
                          <Eye className="h-3 w-3 text-blue-500" />
                          <span className="text-muted-foreground">Started</span>
                        </div>
                        <span className="font-medium text-xs">{new Date(trial.started_at).toLocaleTimeString()}</span>
                      </div>
                    )}
                    {trial.duration_seconds && (
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-1">
                          <Clock className="h-3 w-3 text-orange-500" />
                          <span className="text-muted-foreground">Duration</span>
                        </div>
                        <span className="font-medium">{formatDuration(trial.duration_seconds)}</span>
                      </div>
                    )}
                  </div>

                  {/* Architecture Info */}
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center gap-1 text-sm">
                      <Layers className="h-3 w-3 text-indigo-500" />
                      <span className="text-muted-foreground">Architecture</span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {trial.architecture ? 'Architecture details available' : 'Architecture data pending...'}
                    </div>
                  </div>

                  {/* Trial ID */}
                  <div className="mt-3 pt-2 border-t border-border/50">
                    <p className="text-xs text-muted-foreground">
                      ID: {trial.trial_id || 'N/A'}
                    </p>
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
              Trial #{selectedTrial?.trialNumber} - 3D Architecture
              <Badge variant="outline">{selectedTrial?.architecture.type}</Badge>
            </DialogTitle>
            <DialogClose onClose={handleCloseDialog} />
          </DialogHeader>
          
          {selectedTrial && (
            <div className="space-y-6">
              {/* Trial Summary */}
              <div className="grid grid-cols-3 gap-4 p-4 bg-muted/20 rounded-lg">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Overall Score</p>
                  <p className="text-lg font-bold">{(selectedTrial.overallScore * 100).toFixed(1)}%</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Accuracy</p>
                  <p className="text-lg font-bold">{(selectedTrial.accuracyScore * 100).toFixed(1)}%</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Duration</p>
                  <p className="text-lg font-bold">{formatDuration(selectedTrial.duration)}</p>
                </div>
              </div>

              {/* 3D Visualization Area */}
              <div className="relative">
                <div className="w-full h-96 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-lg border flex items-center justify-center">
                  <div className="text-center">
                    <Layers className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-xl font-medium text-muted-foreground">3D Architecture Visualization</p>
                    <p className="text-sm text-muted-foreground mt-2">
                      Interactive 3D model for Trial #{selectedTrial.trialNumber}
                    </p>
                    <p className="text-xs text-muted-foreground mt-2">
                      Will be implemented with React Three Fiber in Phase 2
                    </p>
                  </div>
                </div>
                
                {/* 3D Controls */}
                <div className="absolute top-4 right-4 flex gap-2">
                  <Button variant="outline" size="sm">
                    <ZoomOut className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <ZoomIn className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* Architecture Details */}
              <div className="space-y-4">
                <h4 className="font-medium">Architecture details</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Conv Layers</p>
                    <p className="font-medium">{selectedTrial.architecture.convLayers}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Dense Layers</p>
                    <p className="font-medium">{selectedTrial.architecture.denseLayers}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Total Layers</p>
                    <p className="font-medium">{selectedTrial.architecture.totalLayers}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Parameters</p>
                    <p className="font-medium">{selectedTrial.architecture.parameters.toLocaleString()}</p>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Filter Sizes:</p>
                  <div className="flex gap-2">
                    {selectedTrial.architecture.filterSizes.map((size, index) => (
                      <Badge key={index} variant="outline">{size}</Badge>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Activations:</p>
                  <div className="flex gap-2">
                    {selectedTrial.architecture.activations.map((activation, index) => (
                      <Badge key={index} variant="outline">{activation}</Badge>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Key Features:</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedTrial.architecture.keyFeatures.map((feature, index) => (
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