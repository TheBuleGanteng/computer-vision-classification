"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tooltip } from "@/components/ui/tooltip"
import { 
  Target,
  Activity,
  Clock,
  Hash,
  Info
} from "lucide-react"
import React, { useMemo } from "react"
import { useDashboard } from "./dashboard-provider"
import { useTrials } from "@/hooks/use-trials"

// Health component weights (matches backend health_analyzer.py)
const HEALTH_COMPONENT_WEIGHTS = {
  neuron_health: 0.25,
  parameter_efficiency: 0.15,
  training_stability: 0.20,
  gradient_health: 0.15,
  convergence_quality: 0.15,
  accuracy_consistency: 0.10
}

const SummaryStats = React.memo(() => {
  const { optimizationMode, healthWeight } = useDashboard()
  const { stats, bestTrial } = useTrials()
  
  // Memoized calculations to prevent unnecessary re-renders
  const formattedStats = useMemo(() => {
    const bestAccuracy = bestTrial?.performance?.accuracy ?? 0
    const bestTotalScore = bestTrial?.performance?.total_score ?? 0
    

    const formatScore = (score: number) => `${(score * 100).toFixed(1)}%`
    
    return {
      trialsCompleted: stats.completed,
      bestAccuracy: formatScore(bestAccuracy),
      bestTotalScore: formatScore(bestTotalScore),
      avgScore: formatScore(stats.averageScore)
    }
  }, [stats, bestTrial])

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Trials Completed */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-2">
            <Hash className="h-5 w-5 text-blue-500" />
            <div>
              <p className="text-sm font-medium text-muted-foreground">Trials Completed</p>
              <p className="text-2xl font-bold">{formattedStats.trialsCompleted}</p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Best Total Score */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-purple-500" />
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-medium text-muted-foreground">Best Total Score</span>
                <Tooltip
                  content={
                    <div className="space-y-3">
                      <p className="font-bold">Best Total Score Calculation</p>
                      {optimizationMode === "simple" ? (
                        <div className="space-y-2">
                          <p>
                            <strong>Simple Mode:</strong> Pure categorical accuracy optimization.
                          </p>
                          <p>
                            The score represents the model's prediction accuracy on the test dataset.
                          </p>
                          <p className="text-xs">
                             <a 
                              href="https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy" 
                              target="_blank" 
                              rel="noopener noreferrer" 
                              className="text-blue-500 hover:text-blue-700 underline"
                            >
                              Click here to learn more
                            </a>
                          </p>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <p>
                            <strong>Health-Aware Mode:</strong> Balanced accuracy + model health optimization.
                          </p>
                          <p>
                            {optimizationMode === "health" && healthWeight > 0 ? (
                              <span>
                                Score = Accuracy Weight ({((1 - healthWeight) * 100).toFixed(1)}%) × Accuracy + Health Weight ({(healthWeight * 100).toFixed(1)}%) × Health Score
                              </span>
                            ) : (
                              <span>
                                Score = (Accuracy Weight × Accuracy) + (Health Weight × Health Score)
                              </span>
                            )}
                          </p>
                          <p className="text-sm">
                            <strong>Health Score Components:</strong>
                          </p>
                          <ul className="text-xs space-y-1 ml-4">
                            <li>• Neuron utilization: {(HEALTH_COMPONENT_WEIGHTS.neuron_health * 100).toFixed(1)}% (active vs inactive neurons)</li>
                            <li>• Parameter efficiency: {(HEALTH_COMPONENT_WEIGHTS.parameter_efficiency * 100).toFixed(1)}% (performance per parameter)</li>
                            <li>• Training stability: {(HEALTH_COMPONENT_WEIGHTS.training_stability * 100).toFixed(1)}% (loss convergence quality)</li>
                            <li>• Gradient health: {(HEALTH_COMPONENT_WEIGHTS.gradient_health * 100).toFixed(1)}% (gradient flow quality)</li>
                            <li>• Convergence quality: {(HEALTH_COMPONENT_WEIGHTS.convergence_quality * 100).toFixed(1)}% (training smoothness)</li>
                            <li>• Accuracy consistency: {(HEALTH_COMPONENT_WEIGHTS.accuracy_consistency * 100).toFixed(1)}% (cross-validation stability)</li>
                          </ul>
                          <p className="text-xs">
                            Learn more: <a 
                              href="https://www.tensorflow.org/guide/keras/train_and_evaluate" 
                              target="_blank" 
                              rel="noopener noreferrer" 
                              className="text-blue-500 hover:text-blue-700 underline"
                            >
                              TensorFlow Model Evaluation
                            </a> | <a 
                              href="https://www.tensorflow.org/guide/keras/custom_callback" 
                              target="_blank" 
                              rel="noopener noreferrer" 
                              className="text-blue-500 hover:text-blue-700 underline"
                            >
                              Training Callbacks
                            </a>
                          </p>
                        </div>
                      )}
                    </div>
                  }
                >
                  <div className="flex items-center justify-center w-4 h-4 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-colors">
                    <Info className="h-2.5 w-2.5" />
                  </div>
                </Tooltip>
                <Badge variant="outline" className="text-xs">
                  {optimizationMode}
                </Badge>
              </div>
              <p className="text-2xl font-bold">{formattedStats.bestTotalScore}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Best Accuracy */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-2">
            <Target className="h-5 w-5 text-green-500" />
            <div>
              <p className="text-sm font-medium text-muted-foreground">Best Accuracy</p>
              <p className="text-2xl font-bold">{formattedStats.bestAccuracy}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Average Duration */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-2">
            <Clock className="h-5 w-5 text-orange-500" />
            <div>
              <p className="text-sm font-medium text-muted-foreground">Average Score</p>
              <p className="text-2xl font-bold">{formattedStats.avgScore}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
})

SummaryStats.displayName = 'SummaryStats'

export { SummaryStats }

/* 
IMPLEMENTATION NOTES:
- Displays all requested summary statistics in a responsive grid
- Best Total Score shows different values for simple vs health mode (as per your optimizer.py logic)
- Best Architecture summary includes:
  - Number of Conv vs Dense layers
  - Filter sizes (e.g., "3x3, 5x5")
  - Activation functions used  
  - Total parameters count
  - Key architectural features as badges
- Mobile responsive: stacks into single column on small screens
- Icons provide visual hierarchy and quick recognition
- Currently uses mock data that matches your optimization results structure
- Will be connected to real optimization results in Phase 1.5
*/