"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { 
  Eye,
  Clock,
  Target,
  Activity,
  ArrowRight
} from "lucide-react"
import Link from "next/link"

// Mock recent optimization data
const recentOptimizations = [
  {
    id: "opt_001",
    name: "CIFAR-10 CNN Optimization",
    dataset: "CIFAR-10",
    status: "completed",
    bestAccuracy: 0.9247,
    healthScore: 0.87,
    trialsCompleted: 25,
    duration: "2h 34m",
    completedAt: "2 hours ago",
    mode: "health"
  },
  {
    id: "opt_002", 
    name: "MNIST Dense Network",
    dataset: "MNIST",
    status: "running",
    bestAccuracy: 0.9849,
    healthScore: 0.92,
    trialsCompleted: 12,
    totalTrials: 30,
    duration: "45m",
    startedAt: "45 minutes ago",
    mode: "simple"
  },
  {
    id: "opt_003",
    name: "GTSRB Traffic Signs",
    dataset: "GTSRB", 
    status: "completed",
    bestAccuracy: 0.9156,
    healthScore: 0.79,
    trialsCompleted: 40,
    duration: "4h 12m",
    completedAt: "1 day ago",
    mode: "health"
  },
  {
    id: "opt_004",
    name: "Fashion-MNIST Classifier",
    dataset: "Fashion-MNIST",
    status: "failed",
    bestAccuracy: 0.8234,
    healthScore: 0.65,
    trialsCompleted: 8,
    duration: "23m",
    failedAt: "3 days ago",
    mode: "simple"
  }
]

function getStatusColor(status: string) {
  switch (status) {
    case "completed":
      return "text-green-600 border-green-600"
    case "running":
      return "text-blue-600 border-blue-600"
    case "failed":
      return "text-red-600 border-red-600"
    default:
      return "text-gray-600 border-gray-600"
  }
}

function getModeColor(mode: string) {
  return mode === "health" 
    ? "text-purple-600 border-purple-600" 
    : "text-orange-600 border-orange-600"
}

export function RecentOptimizations() {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Recent Optimizations</CardTitle>
            <CardDescription>
              Latest hyperparameter optimization sessions
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" asChild>
            <Link href="/sessions">
              View All
              <ArrowRight className="h-4 w-4 ml-2" />
            </Link>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {recentOptimizations.map((opt) => (
            <div key={opt.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors">
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-3 mb-2">
                  <h4 className="font-medium text-sm truncate">{opt.name}</h4>
                  <Badge variant="outline" className={getStatusColor(opt.status)}>
                    {opt.status}
                  </Badge>
                  <Badge variant="outline" className={getModeColor(opt.mode)}>
                    {opt.mode}
                  </Badge>
                </div>
                
                <div className="flex items-center space-x-6 text-xs text-muted-foreground">
                  <div className="flex items-center space-x-1">
                    <Target className="h-3 w-3" />
                    <span>Acc: {(opt.bestAccuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Activity className="h-3 w-3" />
                    <span>Health: {(opt.healthScore * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Clock className="h-3 w-3" />
                    <span>{opt.duration}</span>
                  </div>
                  <div>
                    {opt.status === "running" 
                      ? `${opt.trialsCompleted}/${opt.totalTrials} trials`
                      : `${opt.trialsCompleted} trials`
                    }
                  </div>
                </div>
                
                <div className="text-xs text-muted-foreground mt-1">
                  {opt.dataset} • {
                    opt.status === "running" ? `Started ${opt.startedAt}` :
                    opt.status === "failed" ? `Failed ${opt.failedAt}` :
                    `Completed ${opt.completedAt}`
                  }
                </div>
              </div>
              
              <div className="flex items-center space-x-2 ml-4">
                <Button variant="ghost" size="sm" asChild>
                  <Link href={`/sessions/${opt.id}`}>
                    <Eye className="h-4 w-4" />
                  </Link>
                </Button>
                {opt.status === "completed" && (
                  <Button variant="ghost" size="sm" asChild>
                    <Link href={`/architecture?session=${opt.id}`}>
                      <Layers className="h-4 w-4" />
                    </Link>
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function Layers({ className }: { className?: string }) {
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