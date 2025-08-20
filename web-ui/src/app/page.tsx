import { Suspense } from "react"
import { OptimizationControls } from "@/components/dashboard/optimization-controls"
import { SummaryStats } from "@/components/dashboard/summary-stats"
import { BestArchitectureView } from "@/components/dashboard/best-architecture-view"
import { TrialGallery } from "@/components/dashboard/trial-gallery"

export default function Home() {
  return (
    <div className="container mx-auto px-6 py-6">
      <div className="space-y-6">
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
  )
}
