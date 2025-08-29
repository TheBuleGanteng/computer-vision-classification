"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { 
  Target,
  Activity,
  Clock,
  Hash
} from "lucide-react"
import React, { useMemo } from "react"
import { useDashboard } from "./dashboard-provider"
import { useTrials } from "@/hooks/use-trials"

const SummaryStats = React.memo(() => {
  const { optimizationMode } = useDashboard()
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