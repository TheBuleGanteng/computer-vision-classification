"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { 
  Play, 
  Upload, 
  Eye, 
  BarChart3,
  Layers,
  Settings
} from "lucide-react"
import Link from "next/link"

const quickActions = [
  {
    title: "Start New Optimization",
    description: "Begin a new hyperparameter optimization session",
    icon: Play,
    href: "/optimize/new",
    color: "bg-green-500",
  },
  {
    title: "View 3D Architectures",
    description: "Explore neural networks in interactive 3D",
    icon: Layers,
    href: "/architecture",
    color: "bg-blue-500",
  },
  {
    title: "Compare Trials",
    description: "Analyze and compare optimization results",
    icon: BarChart3,
    href: "/trials",
    color: "bg-purple-500",
  },
  {
    title: "Import Results",
    description: "Upload existing optimization data",
    icon: Upload,
    href: "/import",
    color: "bg-orange-500",
  },
  {
    title: "System Monitor",
    description: "Check system health and resources",
    icon: Eye,
    href: "/system",
    color: "bg-indigo-500",
  },
  {
    title: "Configuration",
    description: "Adjust settings and preferences",
    icon: Settings,
    href: "/settings",
    color: "bg-gray-500",
  },
]

export function QuickActions() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Quick Actions</CardTitle>
        <CardDescription>
          Common tasks and navigation shortcuts
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {quickActions.map((action) => {
            const Icon = action.icon
            return (
              <Button
                key={action.href}
                variant="outline"
                className="h-auto p-4 flex flex-col items-center space-y-2"
                asChild
              >
                <Link href={action.href}>
                  <div className={`p-2 rounded-lg ${action.color}`}>
                    <Icon className="h-5 w-5 text-white" />
                  </div>
                  <div className="text-center">
                    <div className="font-medium text-sm">{action.title}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {action.description}
                    </div>
                  </div>
                </Link>
              </Button>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}