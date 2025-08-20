"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Select, SelectItem } from "@/components/ui/select"
import { Card, CardContent } from "@/components/ui/card"
import { Tooltip } from "@/components/ui/tooltip"
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
  { value: "health", label: "Accuracy + model health (recommended)" },
  { value: "simple", label: "Accuracy only" }
]

// Mock optimization state - in real app this would come from context/state management
const mockOptimizationState = {
  isRunning: false,
  isCompleted: true, // Set to true to show download button enabled
  selectedDataset: "cifar10",
  sessionId: "2025-08-20-10:30:15_cifar10_health"
}

export function OptimizationControls() {
  const [selectedDataset, setSelectedDataset] = useState("")
  const [selectedTargetMetric, setSelectedTargetMetric] = useState("") // Default to empty (placeholder state)
  const [isOptimizationRunning, setIsOptimizationRunning] = useState(mockOptimizationState.isRunning)
  const [isOptimizationCompleted, setIsOptimizationCompleted] = useState(mockOptimizationState.isCompleted)

  const handleOptimizationToggle = () => {
    if (isOptimizationRunning) {
      // Cancel optimization
      setIsOptimizationRunning(false)
      console.log("Cancelling optimization...")
    } else {
      // Start optimization
      setIsOptimizationRunning(true)
      setIsOptimizationCompleted(false)
      console.log(`Starting optimization for dataset: ${selectedDataset}, target metric: ${selectedTargetMetric}`)
      
      // Simulate optimization completion after 5 seconds (for demo)
      setTimeout(() => {
        setIsOptimizationRunning(false)
        setIsOptimizationCompleted(true)
      }, 5000)
    }
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
        {isOptimizationRunning && (
          <div className="mt-4 flex items-center gap-2 text-sm text-blue-600">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" />
            Optimization in progress for {DATASETS.find(d => d.value === selectedDataset)?.label} 
            using {TARGET_METRICS.find(m => m.value === selectedTargetMetric)?.label.toLowerCase()}...
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