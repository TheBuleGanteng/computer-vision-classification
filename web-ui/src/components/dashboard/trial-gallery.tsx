"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from "@/components/ui/dialog"
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
  ZoomOut
} from "lucide-react"

// Mock trial data - matches your optimization results structure
const mockTrials = [
  {
    trialNumber: 25,
    overallScore: 0.8956,
    accuracyScore: 0.9247,
    duration: 142,
    architecture: {
      type: "CNN",
      convLayers: 3,
      denseLayers: 2,
      totalLayers: 8,
      filterSizes: ["3x3", "5x5", "3x3"],
      activations: ["relu", "sigmoid"],
      parameters: 124567,
      keyFeatures: ["Batch Norm", "Global Pool", "Dropout: 0.35"]
    },
    timestamp: "2 minutes ago"
  },
  {
    trialNumber: 24,
    overallScore: 0.8834,
    accuracyScore: 0.9156,
    duration: 156,
    architecture: {
      type: "CNN",
      convLayers: 2,
      denseLayers: 3,
      totalLayers: 9,
      filterSizes: ["5x5", "3x3"],
      activations: ["relu", "tanh"],
      parameters: 98432,
      keyFeatures: ["Batch Norm", "Max Pool", "Dropout: 0.42"]
    },
    timestamp: "5 minutes ago"
  },
  {
    trialNumber: 23,
    overallScore: 0.8723,
    accuracyScore: 0.9034,
    duration: 134,
    architecture: {
      type: "CNN",
      convLayers: 4,
      denseLayers: 1,
      totalLayers: 7,
      filterSizes: ["3x3", "3x3", "5x5", "3x3"],
      activations: ["relu"],
      parameters: 156789,
      keyFeatures: ["No Batch Norm", "Average Pool", "Dropout: 0.28"]
    },
    timestamp: "8 minutes ago"
  },
  {
    trialNumber: 22,
    overallScore: 0.8567,
    accuracyScore: 0.8945,
    duration: 98,
    architecture: {
      type: "CNN",
      convLayers: 2,
      denseLayers: 2,
      totalLayers: 6,
      filterSizes: ["5x5", "5x5"],
      activations: ["swish", "sigmoid"],
      parameters: 67890,
      keyFeatures: ["Batch Norm", "Global Pool", "Dropout: 0.50"]
    },
    timestamp: "12 minutes ago"
  },
  {
    trialNumber: 21,
    overallScore: 0.8445,
    accuracyScore: 0.8823,
    duration: 189,
    architecture: {
      type: "CNN",
      convLayers: 3,
      denseLayers: 3,
      totalLayers: 10,
      filterSizes: ["3x3", "3x3", "3x3"],
      activations: ["relu", "tanh"],
      parameters: 203456,
      keyFeatures: ["Batch Norm", "Max Pool", "Dropout: 0.33"]
    },
    timestamp: "15 minutes ago"
  },
  {
    trialNumber: 20,
    overallScore: 0.8234,
    accuracyScore: 0.8756,
    duration: 167,
    architecture: {
      type: "CNN",
      convLayers: 1,
      denseLayers: 4,
      totalLayers: 8,
      filterSizes: ["7x7"],
      activations: ["relu", "sigmoid"],
      parameters: 89012,
      keyFeatures: ["No Batch Norm", "Average Pool", "Dropout: 0.45"]
    },
    timestamp: "18 minutes ago"
  }
]

export function TrialGallery() {
  const [selectedTrial, setSelectedTrial] = useState<typeof mockTrials[0] | null>(null)

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
  }

  const handleTrialClick = (trial: typeof mockTrials[0]) => {
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
              {mockTrials.length} trials completed
            </Badge>
          </div>

          {/* Trial Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {mockTrials.map((trial) => (
              <Card 
                key={trial.trialNumber} 
                className="cursor-pointer hover:shadow-md transition-all duration-200 hover:border-primary/50"
                onClick={() => handleTrialClick(trial)}
              >
                <CardContent className="p-4">
                  {/* Trial Header */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Hash className="h-4 w-4 text-muted-foreground" />
                      <span className="font-semibold">Trial {trial.trialNumber}</span>
                    </div>
                    <Badge variant={trial.overallScore > 0.85 ? "default" : "secondary"}>
                      {trial.architecture.type}
                    </Badge>
                  </div>

                  {/* Scores */}
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-1">
                        <Activity className="h-3 w-3 text-purple-500" />
                        <span className="text-muted-foreground">Overall Score</span>
                      </div>
                      <span className="font-medium">{(trial.overallScore * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-1">
                        <Target className="h-3 w-3 text-green-500" />
                        <span className="text-muted-foreground">Accuracy</span>
                      </div>
                      <span className="font-medium">{(trial.accuracyScore * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3 text-orange-500" />
                        <span className="text-muted-foreground">Duration</span>
                      </div>
                      <span className="font-medium">{formatDuration(trial.duration)}</span>
                    </div>
                  </div>

                  {/* Architecture Summary */}
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center gap-1 text-sm">
                      <Layers className="h-3 w-3 text-indigo-500" />
                      <span className="text-muted-foreground">Architecture</span>
                    </div>
                    <div className="text-xs space-y-1">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Conv Layers:</span>
                        <span>{trial.architecture.convLayers}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Dense Layers:</span>
                        <span>{trial.architecture.denseLayers}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Filters:</span>
                        <span>{trial.architecture.filterSizes.join(", ")}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Parameters:</span>
                        <span>{(trial.architecture.parameters / 1000).toFixed(0)}K</span>
                      </div>
                    </div>
                  </div>

                  {/* Key Features */}
                  <div className="space-y-2">
                    <p className="text-xs text-muted-foreground">Key features:</p>
                    <div className="flex flex-wrap gap-1">
                      {trial.architecture.keyFeatures.map((feature, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {feature}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Timestamp */}
                  <div className="mt-3 pt-2 border-t border-border/50">
                    <p className="text-xs text-muted-foreground flex items-center gap-1">
                      <Eye className="h-3 w-3" />
                      {trial.timestamp}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
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