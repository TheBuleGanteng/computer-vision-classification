"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { 
  Cpu, 
  HardDrive, 
  Zap,
  Server,
  Clock,
  CheckCircle
} from "lucide-react"

// Mock system data
const systemStats = {
  cpuUsage: 45,
  memoryUsage: 67,
  gpuUsage: 78,
  diskUsage: 34,
  runpodWorkers: {
    active: 3,
    total: 6,
    status: "healthy"
  },
  uptime: "2d 14h 32m",
  lastOptimization: "12 minutes ago",
  queueLength: 2
}

export function SystemStats() {
  return (
    <div className="space-y-6">
      {/* Resource Usage */}
      <Card>
        <CardHeader>
          <CardTitle>System Resources</CardTitle>
          <CardDescription>
            Current resource utilization
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Cpu className="h-4 w-4" />
                <span className="text-sm font-medium">CPU</span>
              </div>
              <span className="text-sm text-muted-foreground">
                {systemStats.cpuUsage}%
              </span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${systemStats.cpuUsage}%` }}
              />
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <HardDrive className="h-4 w-4" />
                <span className="text-sm font-medium">Memory</span>
              </div>
              <span className="text-sm text-muted-foreground">
                {systemStats.memoryUsage}%
              </span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${systemStats.memoryUsage}%` }}
              />
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Zap className="h-4 w-4" />
                <span className="text-sm font-medium">GPU</span>
              </div>
              <span className="text-sm text-muted-foreground">
                {systemStats.gpuUsage}%
              </span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2">
              <div 
                className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${systemStats.gpuUsage}%` }}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* RunPod Status */}
      <Card>
        <CardHeader>
          <CardTitle>RunPod Workers</CardTitle>
          <CardDescription>
            Cloud GPU worker status
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Server className="h-4 w-4" />
              <span className="text-sm font-medium">Active Workers</span>
            </div>
            <Badge variant="secondary">
              {systemStats.runpodWorkers.active}/{systemStats.runpodWorkers.total}
            </Badge>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span className="text-sm font-medium">Status</span>
            </div>
            <Badge variant="outline" className="text-green-600 border-green-600">
              {systemStats.runpodWorkers.status}
            </Badge>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4" />
              <span className="text-sm font-medium">Queue</span>
            </div>
            <span className="text-sm text-muted-foreground">
              {systemStats.queueLength} pending
            </span>
          </div>
        </CardContent>
      </Card>

      {/* System Info */}
      <Card>
        <CardHeader>
          <CardTitle>System Info</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-4 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Uptime</span>
              <span className="font-medium">{systemStats.uptime}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Last Optimization</span>
              <span className="font-medium">{systemStats.lastOptimization}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Disk Usage</span>
              <span className="font-medium">{systemStats.diskUsage}%</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}