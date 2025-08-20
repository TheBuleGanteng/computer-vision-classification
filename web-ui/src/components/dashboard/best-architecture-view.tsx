"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { 
  Download,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Maximize,
  Info
} from "lucide-react"

// Mock best architecture data - will be replaced with real data in Phase 1.5
const mockBestArchitecture = {
  hasOptimization: true, // Set to false to see empty state
  trialNumber: 18,
  accuracy: 0.9247,
  healthScore: 0.8956,
  architecture: {
    type: "CNN",
    layers: [
      { type: "Conv2D", filters: 32, kernelSize: "3x3", activation: "relu" },
      { type: "BatchNormalization" },
      { type: "Conv2D", filters: 64, kernelSize: "5x5", activation: "relu" },
      { type: "MaxPooling2D", poolSize: "2x2" },
      { type: "Conv2D", filters: 128, kernelSize: "3x3", activation: "relu" },
      { type: "GlobalAveragePooling2D" },
      { type: "Dense", units: 512, activation: "sigmoid" },
      { type: "Dropout", rate: 0.35 },
      { type: "Dense", units: 10, activation: "softmax" }
    ],
    totalParams: 124567
  },
  healthMetrics: {
    gradientNorm: 0.0342,
    finalLoss: 0.2156,
    trainingStability: 0.94,
    deadFilters: 3,
    saturatedFilters: 1,
    overfittingScore: 0.15,
    convergenceEpoch: 23,
    lossVariance: 0.008
  }
}

export function BestArchitectureView() {
  const [cameraAngle, setCameraAngle] = useState({ x: 0, y: 0, z: 0 })
  const [zoomLevel, setZoomLevel] = useState(1)
  
  const { hasOptimization, trialNumber, accuracy, healthScore, architecture, healthMetrics } = mockBestArchitecture

  const handleDownload3D = () => {
    console.log("Downloading 3D model as GLTF...")
    // In real implementation, this would trigger GLTF download
    alert("3D model download (.gltf format) would start here")
  }

  const handleDownloadModel = () => {
    console.log("Downloading best model (.keras format)...")
    // In real implementation, this would trigger download from the API
    const sessionId = "2025-08-20-10:30:15_cifar10_health"
    const link = document.createElement('a')
    link.href = `/api/download/model/${sessionId}`
    link.download = `best_model_${sessionId}.keras`
    // link.click() // Uncomment when API is ready
    alert("Model download would start here (API not yet implemented)")
  }

  const handleResetCamera = () => {
    setCameraAngle({ x: 0, y: 0, z: 0 })
    setZoomLevel(1)
  }

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev + 0.2, 3))
  }

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev - 0.2, 0.3))
  }

  // Empty state when no optimization has been run
  if (!hasOptimization) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-3">
                Best architecture
              </CardTitle>
              <CardDescription className="mt-2">
                Start an optimization to see results
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          {/* Empty 3D Visualization Container */}
          <div className="relative">
            <div className="w-full h-96 bg-muted/20 rounded-lg border-2 border-dashed flex items-center justify-center">
              <div className="text-center text-muted-foreground">
                <Info className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="text-lg font-medium">Awaiting optimization results</p>
                <p className="text-sm mt-2">Start an optimization to see architecture visualization</p>
              </div>
            </div>
          </div>

          {/* Empty Architecture Details - Two Column Layout */}
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Empty Architecture Layers */}
            <div>
              <h4 className="text-sm font-medium mb-3">Architecture layers</h4>
              <div className="text-sm text-muted-foreground italic">
                No layers to display - run optimization first
              </div>
            </div>

            {/* Empty Model Health Metrics */}
            <div>
              <h4 className="text-sm font-medium mb-3">Model health metrics</h4>
              <div className="text-sm text-muted-foreground italic">
                No health metrics available - run optimization first
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-3">
              Best architecture - Trial #{trialNumber}
              <Badge variant="outline" className="text-green-600 border-green-600">
                {architecture.type}
              </Badge>
            </CardTitle>
            <CardDescription className="mt-2">
              Accuracy: {(accuracy * 100).toFixed(2)}% • Health Score: {(healthScore * 100).toFixed(2)}% • 
              {architecture.totalParams.toLocaleString()} parameters
            </CardDescription>
          </div>
          
          {/* 3D Controls */}
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleZoomOut}>
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleZoomIn}>
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleResetCamera}>
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleDownload3D}>
              <Download className="h-4 w-4 mr-2" />
              Download 3D
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        {/* 3D Visualization Container */}
        <div className="relative">
          <div 
            className="w-full h-96 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-lg border flex items-center justify-center overflow-hidden"
            style={{
              transform: `scale(${zoomLevel}) rotateX(${cameraAngle.x}deg) rotateY(${cameraAngle.y}deg)`,
              transition: 'transform 0.3s ease'
            }}
          >
            {/* Placeholder for 3D Architecture Visualization */}
            <div className="text-center">
              <div className="flex items-center justify-center space-x-4 mb-6">
                {/* Simplified visual representation of the architecture */}
                {architecture.layers.map((layer, index) => (
                  <div key={index} className="text-center">
                    <div 
                      className={`
                        w-16 h-12 rounded border-2 flex items-center justify-center text-xs font-medium
                        ${layer.type.includes('Conv') ? 'bg-blue-100 border-blue-300 text-blue-700' : ''}
                        ${layer.type.includes('Dense') ? 'bg-green-100 border-green-300 text-green-700' : ''}
                        ${layer.type.includes('Pool') || layer.type.includes('Dropout') ? 'bg-yellow-100 border-yellow-300 text-yellow-700' : ''}
                        ${layer.type.includes('Batch') ? 'bg-purple-100 border-purple-300 text-purple-700' : ''}
                      `}
                    >
                      {layer.type.replace(/2D|Pool|Normalization/g, '').substring(0, 4)}
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 max-w-16 truncate">
                      {layer.type}
                    </p>
                  </div>
                ))}
              </div>
              
              {/* Interactive 3D placeholder */}
              <div className="bg-white/80 dark:bg-black/20 rounded-lg p-6 max-w-md mx-auto">
                <Maximize className="h-8 w-8 mx-auto mb-3 text-muted-foreground" />
                <p className="text-lg font-medium text-muted-foreground">Interactive 3D Visualization</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Three.js/React Three Fiber implementation will render here
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  Click and drag to rotate • Scroll to zoom • Download as GLTF
                </p>
              </div>
            </div>
          </div>
          
          {/* Zoom indicator */}
          <div className="absolute bottom-4 right-4 bg-white/90 dark:bg-black/90 px-2 py-1 rounded text-xs">
            Zoom: {Math.round(zoomLevel * 100)}%
          </div>
        </div>

        {/* Architecture Details - Two Column Layout */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Architecture Layers */}
          <div>
            <h4 className="text-sm font-medium mb-3">Architecture layers ({architecture.layers.length})</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-2">
              {architecture.layers.map((layer, index) => (
                <div key={index} className="p-2 bg-muted/50 rounded text-xs">
                  <div className="font-medium">{index + 1}. {layer.type}</div>
                  {layer.filters && <div className="text-muted-foreground">Filters: {layer.filters}</div>}
                  {layer.kernelSize && <div className="text-muted-foreground">Kernel: {layer.kernelSize}</div>}
                  {layer.units && <div className="text-muted-foreground">Units: {layer.units}</div>}
                  {layer.rate && <div className="text-muted-foreground">Rate: {layer.rate}</div>}
                  {layer.activation && <div className="text-muted-foreground">Act: {layer.activation}</div>}
                </div>
              ))}
            </div>
          </div>

          {/* Model Health Metrics */}
          <div>
            <h4 className="text-sm font-medium mb-3">Model health metrics</h4>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Gradient norm</div>
                  <div className="text-sm font-semibold">{healthMetrics.gradientNorm.toFixed(4)}</div>
                </div>
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Final loss</div>
                  <div className="text-sm font-semibold">{healthMetrics.finalLoss.toFixed(4)}</div>
                </div>
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Training stability</div>
                  <div className="text-sm font-semibold">{(healthMetrics.trainingStability * 100).toFixed(1)}%</div>
                </div>
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Overfitting score</div>
                  <div className="text-sm font-semibold">{(healthMetrics.overfittingScore * 100).toFixed(1)}%</div>
                </div>
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Dead filters</div>
                  <div className="text-sm font-semibold">{healthMetrics.deadFilters}</div>
                </div>
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Saturated filters</div>
                  <div className="text-sm font-semibold">{healthMetrics.saturatedFilters}</div>
                </div>
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Convergence epoch</div>
                  <div className="text-sm font-semibold">{healthMetrics.convergenceEpoch}</div>
                </div>
                <div className="p-2 bg-muted/50 rounded">
                  <div className="font-medium text-muted-foreground">Loss variance</div>
                  <div className="text-sm font-semibold">{healthMetrics.lossVariance.toFixed(3)}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Download Model Button */}
        <div className="mt-6 flex justify-center">
          <Button 
            onClick={handleDownloadModel}
            className="bg-green-600 hover:bg-green-700 text-white min-w-[160px]"
          >
            <Download className="h-4 w-4 mr-2" />
            Download model (.keras)
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

/* 
IMPLEMENTATION NOTES:
- Full-width 3D visualization area with interactive controls
- Zoom in/out, rotate, and reset camera functionality
- Download 3D model in GLTF format (.gltf/.glb)
- Fallback state shows "Awaiting optimization results" when no data
- Responsive layer details grid showing architecture breakdown
- Mock 3D visualization placeholder - will be replaced with React Three Fiber in Phase 2
- Interactive camera controls with smooth transitions
- Currently uses mock architecture data matching your model structure
- Will integrate with real trial data in Phase 1.5
*/