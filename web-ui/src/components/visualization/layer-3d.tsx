'use client';

import React, { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import { LayerVisualization, getPerformanceColor } from '@/types/visualization';
import * as THREE from 'three';

interface Layer3DProps {
  layer: LayerVisualization;
  maxWidth: number;
  maxHeight: number;
  onClick?: (layer: LayerVisualization) => void;
  onHover?: (layer: LayerVisualization | null) => void;
  index: number;
}

export const Layer3D: React.FC<Layer3DProps> = ({
  layer,
  maxWidth,
  maxHeight,
  onClick,
  onHover,
  index
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  
  // Gentle floating animation
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.position.y = Math.sin(state.clock.elapsedTime + index * 0.5) * 0.1;
    }
  });

  // Calculate scaling based on layer properties
  const scaleX = Math.max(0.5, layer.width / maxWidth * 4);
  const scaleY = Math.max(0.5, layer.height / maxHeight * 4);
  const scaleZ = Math.max(0.3, layer.depth);

  // Get color based on performance
  const baseColor = getPerformanceColor(layer.color_intensity);
  const emissiveColor = hovered ? baseColor : '#000000';
  const emissiveIntensity = hovered ? 0.2 : 0;

  // Handle mouse events
  const handlePointerOver = () => {
    setHovered(true);
    onHover?.(layer);
    document.body.style.cursor = 'pointer';
  };

  const handlePointerOut = () => {
    setHovered(false);
    onHover?.(null);
    document.body.style.cursor = 'default';
  };

  const handleClick = () => {
    onClick?.(layer);
  };

  // Choose geometry based on layer type
  const getLayerGeometry = () => {
    switch (layer.layer_type.toLowerCase()) {
      case 'conv':
      case 'conv2d':
      case 'conv1d':
        // Cylinder for convolutional layers
        return (
          <cylinderGeometry 
            args={[scaleX * 0.8, scaleX * 0.8, scaleZ, 16]} 
          />
        );
      
      case 'lstm':
      case 'gru':
      case 'rnn':
        // Sphere for recurrent layers
        return (
          <sphereGeometry 
            args={[Math.max(scaleX, scaleY) * 0.6, 16, 12]} 
          />
        );
      
      case 'pooling':
      case 'maxpooling2d':
      case 'averagepooling2d':
      case 'globalmaxpooling2d':
      case 'globalaveragepooling2d':
        // Octahedron for pooling layers
        return (
          <octahedronGeometry 
            args={[Math.max(scaleX, scaleY) * 0.7, 0]} 
          />
        );
      
      case 'dropout':
      case 'batchnormalization':
        // Thin box for regularization layers
        return (
          <boxGeometry 
            args={[scaleX, 0.2, scaleZ]} 
          />
        );
      
      case 'dense':
      case 'linear':
      default:
        // Box for dense/linear layers
        return (
          <boxGeometry 
            args={[scaleX, scaleY, scaleZ]} 
          />
        );
    }
  };

  // Get layer type symbol for labeling
  const getLayerSymbol = () => {
    switch (layer.layer_type.toLowerCase()) {
      case 'conv':
      case 'conv2d':
      case 'conv1d':
        return '⚡';
      case 'lstm':
      case 'gru':
      case 'rnn':
        return '🔄';
      case 'dense':
      case 'linear':
        return '▦';
      case 'pooling':
      case 'maxpooling2d':
      case 'averagepooling2d':
      case 'globalmaxpooling2d':
      case 'globalaveragepooling2d':
        return '⬇️';
      case 'dropout':
        return '🎲';
      case 'batchnormalization':
        return '📊';
      default:
        return '📦';
    }
  };

  return (
    <group position={[0, 0, layer.position_z]}>
      {/* Main layer geometry */}
      <mesh
        ref={meshRef}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
        onClick={handleClick}
        scale={hovered ? [1.1, 1.1, 1.1] : [1, 1, 1]}
      >
        {getLayerGeometry()}
        <meshStandardMaterial
          color={baseColor}
          emissive={emissiveColor}
          emissiveIntensity={emissiveIntensity}
          opacity={layer.opacity}
          transparent={layer.opacity < 1}
          roughness={0.3}
          metalness={0.2}
        />
      </mesh>

      {/* Layer label */}
      <Text
        position={[0, scaleY + 0.8, 0]}
        fontSize={0.3}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        visible={hovered}
      >
        {getLayerSymbol()} {layer.layer_type}
      </Text>

      {/* Parameter count */}
      {layer.parameters > 0 && (
        <Text
          position={[0, -scaleY - 0.8, 0]}
          fontSize={0.2}
          color="#9CA3AF"
          anchorX="center"
          anchorY="middle"
          visible={hovered}
        >
          {layer.parameters.toLocaleString()} params
        </Text>
      )}

      {/* Connection point indicator */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.05, 8, 6]} />
        <meshBasicMaterial 
          color="#ffffff" 
          opacity={0.6} 
          transparent 
        />
      </mesh>
    </group>
  );
};