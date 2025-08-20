"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { 
  TrendingUp, 
  TrendingDown, 
  Activity,
  Clock,
  Target,
  Cpu
} from "lucide-react"

// Mock data - in real app this would come from API
const mockStats = {
  totalSessions: 47,
  activeOptimizations: 3,
  averageAccuracy: 0.9247,
  bestAccuracy: 0.9849,
  totalTrials: 1253,
  avgTrialDuration: 127, // seconds
  successRate: 0.94,
  topDataset: "CIFAR-10"
}

const statCards = [
  {
    title: "Total Sessions",
    value: mockStats.totalSessions,
    description: "Optimization sessions completed",
    icon: Activity,
    trend: "+12% from last month",
    trendUp: true,
  },
  {
    title: "Active Optimizations",
    value: mockStats.activeOptimizations,
    description: "Currently running",
    icon: Cpu,
    trend: "3 workers active",
    trendUp: true,
  },
  {
    title: "Best Accuracy",
    value: `${(mockStats.bestAccuracy * 100).toFixed(2)}%`,
    description: "Highest model performance",
    icon: Target,
    trend: "+2.3% improvement",
    trendUp: true,
  },
  {
    title: "Avg Trial Duration",
    value: `${Math.floor(mockStats.avgTrialDuration / 60)}m ${mockStats.avgTrialDuration % 60}s`,
    description: "Per trial training time",
    icon: Clock,
    trend: "-15s optimization",
    trendUp: true,
  },
]

export function DashboardOverview() {
  return (
    <div className="space-y-6">
      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {statCards.map((stat) => {
          const Icon = stat.icon
          return (
            <Card key={stat.title}>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-muted-foreground">
                      {stat.title}
                    </p>
                    <p className="text-2xl font-bold">
                      {stat.value}
                    </p>
                    <div className="flex items-center space-x-1 text-xs">
                      {stat.trendUp ? (
                        <TrendingUp className="h-3 w-3 text-green-500" />
                      ) : (
                        <TrendingDown className="h-3 w-3 text-red-500" />
                      )}
                      <span className={stat.trendUp ? "text-green-600" : "text-red-600"}>
                        {stat.trend}
                      </span>
                    </div>
                  </div>
                  <Icon className="h-8 w-8 text-muted-foreground" />
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Performance Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Optimization Performance Trends</CardTitle>
          <CardDescription>
            Model accuracy and health metrics over time
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center bg-muted/20 rounded-lg">
            <div className="text-center text-muted-foreground">
              <BarChart3 className="h-12 w-12 mx-auto mb-4" />
              <p className="text-lg font-medium">Performance Chart</p>
              <p className="text-sm">Interactive chart will be implemented with Chart.js</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function BarChart3({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
      />
    </svg>
}