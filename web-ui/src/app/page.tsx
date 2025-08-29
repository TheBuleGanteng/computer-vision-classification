"use client"

import { Suspense, useEffect } from "react"
import { OptimizationControls } from "@/components/dashboard/optimization-controls"
import { SummaryStats } from "@/components/dashboard/summary-stats"
import { BestArchitectureView } from "@/components/dashboard/best-architecture-view"
import { TrialGallery } from "@/components/dashboard/trial-gallery-optimized"
import { DashboardProvider } from "@/components/dashboard/dashboard-provider"

export default function Home() {
  // Suppress React setTimeout violation warnings for non-performance-critical app
  useEffect(() => {
    if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
      const originalWarn = console.warn
      console.warn = (...args) => {
        // Filter out React setTimeout violation warnings
        if (args[0]?.includes?.('Violation') && args[0]?.includes?.('setTimeout')) {
          return // Ignore these warnings
        }
        originalWarn.apply(console, args)
      }
    }
  }, [])

  return (
    <DashboardProvider>
      <div className="container mx-auto px-4 sm:px-6 py-4 sm:py-6">
        <div className="space-y-4 sm:space-y-6">
          {/* Optimization Controls Row */}
          <Suspense fallback={<div className="h-16 bg-muted animate-pulse rounded-lg" />}>
            <OptimizationControls />
          </Suspense>

          {/* Summary Statistics Row */}
          <Suspense fallback={<div className="h-32 bg-muted animate-pulse rounded-lg" />}>
            <SummaryStats />
          </Suspense>

        {/* Full-Width 3D Visualization of Best Architecture */}
        <Suspense fallback={<div className="h-96 bg-muted animate-pulse rounded-lg" />}>
          <BestArchitectureView />
        </Suspense>

        {/* Trial Gallery - Grid of Trial Tiles */}
        <Suspense fallback={<div className="h-64 bg-muted animate-pulse rounded-lg" />}>
          <TrialGallery />
        </Suspense>
      </div>
    </div>
    </DashboardProvider>
  )
}
