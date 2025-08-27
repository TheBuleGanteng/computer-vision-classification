'use client';

import React, { useState, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import { Model3DViewerProps, LayerVisualization } from '@/types/visualization';
import { Architecture3D } from './architecture-3d';
import { LayerInfoPanel } from './layer-info-panel';

export const Model3DViewer: React.FC<Model3DViewerProps> = ({
  visualizationData,
  isLoading = false,
  error,
  className = ''
}) => {
  const [selectedLayer, setSelectedLayer] = useState<LayerVisualization | null>(null);
  const [hoveredLayer, setHoveredLayer] = useState<LayerVisualization | null>(null);

  if (isLoading) {
    return (
      <div className={`w-full h-full flex items-center justify-center bg-gray-900 ${className}`}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-300">Loading 3D Model...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`w-full h-full flex items-center justify-center bg-gray-900 ${className}`}>
        <div className="text-center text-red-400">
          <div className="text-6xl mb-4">⚠️</div>
          <p className="text-lg">Failed to load 3D visualization</p>
          <p className="text-sm text-gray-400 mt-2">{error}</p>
        </div>
      </div>
    );
  }

  if (!visualizationData || !visualizationData.layers || !visualizationData.layers.length) {
    return (
      <div className={`w-full h-full flex items-center justify-center bg-gray-900 ${className}`}>
        <div className="text-center text-gray-400">
          <div className="text-6xl mb-4">📊</div>
          <p className="text-lg">No architecture data available</p>
          <p className="text-sm mt-2">Complete an optimization to view 3D model</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative w-full h-full bg-gradient-to-b from-gray-900 to-black ${className}`}>
      {/* 3D Canvas */}
      <Canvas
        camera={{ 
          position: [8, 8, 8], 
          fov: 60,
          near: 0.1,
          far: 1000
        }}
        style={{ background: 'transparent' }}
      >
        {/* Lighting Setup */}
        <ambientLight intensity={0.4} />
        <pointLight 
          position={[10, 10, 10]} 
          intensity={0.8} 
          color="#ffffff"
        />
        <pointLight 
          position={[-10, -10, 10]} 
          intensity={0.3} 
          color="#4F46E5"
        />
        
        {/* Camera Controls */}
        <OrbitControls 
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={5}
          maxDistance={30}
          maxPolarAngle={Math.PI}
        />

        {/* 3D Architecture */}
        <Suspense fallback={
          <Html center>
            <div className="text-white">Loading layers...</div>
          </Html>
        }>
          <Architecture3D
            visualizationData={visualizationData}
            onLayerClick={setSelectedLayer}
            onLayerHover={setHoveredLayer}
          />
        </Suspense>


      </Canvas>

      {/* Layer Information Panel */}
      <LayerInfoPanel
        selectedLayer={selectedLayer}
        hoveredLayer={hoveredLayer}
        onClose={() => setSelectedLayer(null)}
      />

      {/* Controls Help */}
      <div className="absolute bottom-4 left-4 text-gray-400 text-sm space-y-1 z-10">
        <div className="bg-black/50 rounded px-2 py-1">
          <p>🖱️ Click and drag to rotate</p>
          <p>⚙️ Scroll to zoom</p>
          <p>📦 Click layers for details</p>
        </div>
      </div>

      {/* Architecture Info */}
      <div className="absolute top-4 right-4 text-white text-sm z-10">
        <div className="bg-black/70 rounded-lg p-3 space-y-2">
          <div className="font-semibold text-blue-400">
            {visualizationData.type} Model
          </div>
          <div>Layers: {visualizationData.layers.length}</div>
          <div>Parameters: {visualizationData.total_parameters.toLocaleString()}</div>
          <div>Performance: {(visualizationData.performance_score * 100).toFixed(1)}%</div>
          {visualizationData.health_score && (
            <div>Health: {(visualizationData.health_score * 100).toFixed(1)}%</div>
          )}
        </div>
      </div>
    </div>
  );
};