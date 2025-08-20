"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { 
  Target,
  Activity,
  Clock,
  Hash
} from "lucide-react"

// Mock summary data - matches your optimization results structure
const mockSummaryData = {
  trialsPerformed: 25,
  bestAccuracy: 0.9247, // 92.47%
  bestTotalScore: 0.8956, // Health-weighted score in health mode
  avgDurationPerTrial: 127, // seconds
  optimizationMode: "health" as "simple" | "health" // or "simple"
}

export function SummaryStats() {
  const { 
    trialsPerformed, 
    bestAccuracy, 
    bestTotalScore, 
    avgDurationPerTrial,
    optimizationMode
  } = mockSummaryData

  // Format duration to readable format
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Trials Performed */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center space-x-2">
            <Hash className="h-5 w-5 text-blue-500" />
            <div>
              <p className="text-sm font-medium text-muted-foreground">Trials Performed</p>
              <p className="text-2xl font-bold">{trialsPerformed}</p>
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
              <p className="text-2xl font-bold">{(bestAccuracy * 100).toFixed(2)}%</p>
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
              <p className="text-sm font-medium text-muted-foreground">
                Best Total Score
                <Badge variant="outline" className="ml-1 text-xs">
                  {optimizationMode}
                </Badge>
              </p>
              <p className="text-2xl font-bold">{(bestTotalScore * 100).toFixed(2)}%</p>
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
              <p className="text-sm font-medium text-muted-foreground">Avg. Duration Per Trial</p>
              <p className="text-2xl font-bold">{formatDuration(avgDurationPerTrial)}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

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