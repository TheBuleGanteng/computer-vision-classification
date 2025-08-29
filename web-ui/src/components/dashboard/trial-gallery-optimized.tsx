"use client"

import React, { useState, useMemo } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from "@/components/ui/dialog"
import { useDashboard } from "./dashboard-provider"
import { useTrials } from "@/hooks/use-trials"
import { TrialProgress } from "@/types/optimization"
import { 
  Eye,
  Target,
  Activity,
  Clock,
  Layers,
  Hash,
  Loader2
} from "lucide-react"
import { UnifiedEducationalInterface } from "@/components/visualization/unified-educational-interface"

// Optimized Educational Visualization Container Component
const EducationalVisualizationContainer = React.memo(({ 
  jobId, 
  selectedTrial 
}: { 
  jobId: string | null; 
  selectedTrial: TrialProgress | null;
}) => {
  if (!jobId || !selectedTrial) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        <div className="text-center">
          <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>Select a trial to view educational visualization</p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full h-full min-h-[600px]">
      <UnifiedEducationalInterface 
        jobId={jobId} 
        trialId={selectedTrial.trial_number?.toString() || selectedTrial.trial_id}
        className="h-full"
      />
    </div>
  )
})

EducationalVisualizationContainer.displayName = 'EducationalVisualizationContainer'

const TrialGallery = React.memo(() => {
  const { currentJobId } = useDashboard()
  const [selectedTrial, setSelectedTrial] = useState<TrialProgress | null>(null)
  const { trials, bestTrial, isLoading, error } = useTrials()

  // Memoized helper functions to prevent unnecessary re-calculations
  const formatDuration = useMemo(() => (seconds: number | null | undefined) => {
    if (!seconds) return "N/A"
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
  }, [])

  const formatScore = useMemo(() => (score: number | null | undefined) => {
    if (score === null || score === undefined) return "N/A"
    return `${(score * 100).toFixed(1)}%`
  }, [])

  // Memoized trial status badge to prevent re-renders
  const TrialStatusBadge = useMemo(() => {
    const StatusBadge = React.memo(({ status }: { status: string }) => {
      const statusConfig = {
        completed: { variant: "default" as const, color: "text-green-600", label: "Completed" },
        running: { variant: "secondary" as const, color: "text-blue-600", label: "Running" },
        failed: { variant: "destructive" as const, color: "text-red-600", label: "Failed" },
      }
      
      const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.completed
      
      return (
        <Badge variant={config.variant} className={`${config.color} text-xs`}>
          {config.label}
        </Badge>
      )
    })
    
    StatusBadge.displayName = 'TrialStatusBadge'
    return StatusBadge
  }, [])

  // Loading state
  if (isLoading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-center h-32">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
            <span className="ml-2 text-muted-foreground">Loading trials...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Error state
  if (error) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="text-center text-muted-foreground">
            <p>Error loading trials: {error}</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Empty state
  if (!trials || trials.length === 0) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="text-center text-muted-foreground">
            <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-medium mb-2">No Trials Yet</h3>
            <p className="text-sm">Start an optimization to see trials here</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <>
      <Card>
        <CardContent className="pt-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {trials.map((trial) => {
              const isBestTrial = bestTrial && trial.trial_id === bestTrial.trial_id
              
              return (
                <Card 
                  key={`trial-${trial.trial_id || trial.trial_number}`}
                  className={`cursor-pointer transition-all duration-200 hover:shadow-md ${
                    isBestTrial 
                      ? "ring-2 ring-orange-500 shadow-lg bg-orange-50 dark:bg-orange-900/20" 
                      : "hover:ring-2 hover:ring-blue-300"
                  }`}
                  onClick={() => setSelectedTrial(trial)}
                >
                  <CardContent className="p-4">
                    <div className="space-y-3">
                      {/* Header */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Hash className="w-4 h-4 text-muted-foreground" />
                          <span className="font-medium">Trial {trial.trial_number}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <TrialStatusBadge status={trial.status} />
                          {isBestTrial && (
                            <Badge className="bg-orange-500 text-white text-xs">
                              Best Trial
                            </Badge>
                          )}
                        </div>
                      </div>

                      {/* Performance Metrics */}
                      {trial.performance && (
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-muted-foreground">Accuracy</span>
                            <span className="text-sm font-medium">
                              {formatScore(trial.performance.accuracy)}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-muted-foreground">Total Score</span>
                            <span className="text-sm font-medium">
                              {formatScore(trial.performance.total_score)}
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Health Metrics */}
                      {trial.health_metrics && (
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-muted-foreground">Health Score</span>
                            <span className="text-sm font-medium">
                              {formatScore(trial.health_metrics.overall_health)}
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Architecture Info */}
                      {trial.architecture && (
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <Layers className="w-4 h-4" />
                          <span>{trial.architecture.type || "Unknown"} Architecture</span>
                        </div>
                      )}

                      {/* Duration */}
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Clock className="w-4 h-4" />
                        <span>Duration: {formatDuration(trial.duration_seconds)}</span>
                      </div>

                      {/* View Button */}
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="w-full mt-3"
                        onClick={(e) => {
                          e.stopPropagation()
                          setSelectedTrial(trial)
                        }}
                      >
                        <Eye className="w-4 h-4 mr-2" />
                        View Details
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Educational Visualization Modal */}
      <Dialog open={!!selectedTrial} onOpenChange={() => setSelectedTrial(null)}>
        <DialogContent className="max-w-7xl max-h-[90vh] w-[95vw]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Trial {selectedTrial?.trial_number} Educational Visualization
              {bestTrial && selectedTrial?.trial_id === bestTrial.trial_id && (
                <Badge className="bg-orange-500 text-white ml-2">Best Trial</Badge>
              )}
            </DialogTitle>
          </DialogHeader>
          
          <div className="mt-4">
            <EducationalVisualizationContainer 
              jobId={currentJobId} 
              selectedTrial={selectedTrial} 
            />
          </div>
          
          <DialogClose />
        </DialogContent>
      </Dialog>
    </>
  )
})

TrialGallery.displayName = 'TrialGallery'

export { TrialGallery }