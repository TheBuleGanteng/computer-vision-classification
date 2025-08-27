'use client';

import React, { useMemo } from 'react';
import { Text } from '@react-three/drei';
import { ArchitectureVisualization, LayerVisualization } from '@/types/visualization';
import { Layer3D } from './layer-3d';
import { AnimatedArrow } from './animated-arrow';

interface Architecture3DProps {
  visualizationData: ArchitectureVisualization;
  onLayerClick?: (layer: LayerVisualization) => void;
  onLayerHover?: (layer: LayerVisualization | null) => void;
}

export const Architecture3D: React.FC<Architecture3DProps> = ({
  visualizationData,
  onLayerClick,
  onLayerHover
}) => {
  // Create animated arrows showing data flow between layers
  const dataFlowArrows = useMemo(() => {
    const arrows = [];
    const layers = visualizationData?.layers;
    
    if (!layers || !Array.isArray(layers)) {
      return arrows;
    }
    
    for (let i = 0; i < layers.length - 1; i++) {
      const currentLayer = layers[i];
      const nextLayer = layers[i + 1];
      
      if (!currentLayer || !nextLayer) continue;
      
      // Position arrows to show flow from current layer to next
      const startZ = currentLayer.position_z + currentLayer.depth / 2;
      const endZ = nextLayer.position_z - nextLayer.depth / 2;
      
      arrows.push({
        start: [0, 0, startZ] as [number, number, number],
        end: [0, 0, endZ] as [number, number, number],
        id: `arrow-${i}`,
        speed: 1 + i * 0.1 // Slightly different speeds for visual interest
      });
    }
    
    return arrows;
  }, [visualizationData?.layers]);

  return (
    <group>
      {/* Animated Data Flow Arrows */}
      {dataFlowArrows.map((arrow) => (
        <AnimatedArrow
          key={arrow.id}
          start={arrow.start}
          end={arrow.end}
          color="#10B981"
          speed={arrow.speed}
        />
      ))}

      {/* Layer Geometries with Sequential Numbering */}
      {visualizationData?.layers?.map((layer, index) => (
        <group key={layer.layer_id}>
          {/* Layer 3D Geometry */}
          <Layer3D
            layer={layer}
            maxWidth={visualizationData?.max_layer_width || 0}
            maxHeight={visualizationData?.max_layer_height || 0}
            onClick={onLayerClick}
            onHover={onLayerHover}
            index={index}
          />
          
          {/* Sequential Layer Number */}
          <Text
            position={[
              -(visualizationData?.max_layer_width || 0) / 2 - 1,
              (visualizationData?.max_layer_height || 0) / 2 + 0.5,
              layer.position_z
            ]}
            fontSize={0.4}
            color="#ffffff"
            anchorX="center"
            anchorY="middle"
          >
            {index + 1}
          </Text>
          
          {/* Layer Type Label */}
          <Text
            position={[
              -(visualizationData?.max_layer_width || 0) / 2 - 1,
              (visualizationData?.max_layer_height || 0) / 2,
              layer.position_z
            ]}
            fontSize={0.25}
            color="#9CA3AF"
            anchorX="center"
            anchorY="middle"
          >
            {layer.layer_type.replace('2d', '').toUpperCase()}
          </Text>
        </group>
      ))}

      {/* Coordinate System Axes (for debugging) */}
      {process.env.NODE_ENV === 'development' && (
        <group>
          {/* X-axis - Red */}
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([0, 0, 0, 5, 0, 0])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#FF0000" />
          </line>
          
          {/* Y-axis - Green */}
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([0, 0, 0, 0, 5, 0])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#00FF00" />
          </line>
          
          {/* Z-axis - Blue */}
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([0, 0, 0, 0, 0, 5])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#0000FF" />
          </line>
        </group>
      )}
    </group>
  );
};