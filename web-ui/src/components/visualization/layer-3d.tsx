'use client';

import React, { useRef, useState, useMemo } from 'react';
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
  // Expand depth for convolutional layers to accommodate filter spacing
  const isConvLayer = layer.layer_type.toLowerCase().includes('conv');
  const scaleZ = isConvLayer ? Math.max(1.5, layer.depth * 2) : Math.max(0.3, layer.depth);

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
        // Rectangular prism for convolutional layers to contain filters
        return (
          <boxGeometry 
            args={[scaleX, scaleY, scaleZ]} 
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

  // Internal filter stack visualization for Conv2D layers (memoized to prevent re-creation)
  const internalFilterStack = useMemo(() => {
    const layerType = layer.layer_type.toLowerCase();
    const isConvLayer = layerType.includes('conv');
    
    // Only show for conv layers with valid filter data
    if (!isConvLayer || !layer.filters || !layer.kernel_size || !Array.isArray(layer.kernel_size)) {
      return null;
    }

    const filterCount = layer.filters;
    const [kernelWidth, kernelHeight] = layer.kernel_size;
    
    // Simple validation
    if (filterCount <= 0 || filterCount > 100 || kernelWidth <= 0 || kernelHeight <= 0) {
      return null;
    }
    
    // Show all filters for accurate representation
    const maxFiltersToShow = filterCount; // Show actual number of filters in the layer
    
    // Filter dimensions - scaled based on actual kernel shape relative to layer
    const kernelScale = 0.6; // Scale factor for filter size within layer
    const filterWidth = (scaleX * kernelScale) * (kernelWidth / Math.max(kernelWidth, kernelHeight));
    const filterHeight = (scaleY * kernelScale) * (kernelHeight / Math.max(kernelWidth, kernelHeight));
    const filterThickness = 0.03; // Slightly thicker for better visibility
    
    // Z-axis spacing - much more space between filters
    const availableDepth = scaleZ * 0.8; // Use 80% of expanded layer depth
    const filterSpacing = maxFiltersToShow > 1 ? availableDepth / (maxFiltersToShow - 1) : 0; // Even larger gaps between filters
    
    // Filter color with slight variation
    const filterColor = getPerformanceColor(layer.color_intensity);
    
    // Create stacked filter representations
    const filters = [];
    for (let i = 0; i < maxFiltersToShow; i++) {
      // Position filters along Z-axis, centered in X and Y
      const x = 0; // Centered horizontally
      const y = 0; // Centered vertically  
      const z = -availableDepth/2 + (i * filterSpacing); // Evenly distributed with larger gaps
      
      // Create screen-like filter with heavy outer border and fine inner grid
      const screenElements = [];
      
      // Filter screen background
      screenElements.push(
        <mesh key="screen-bg" position={[x, y, z]}>
          <boxGeometry args={[filterWidth, filterHeight, filterThickness]} />
          <meshStandardMaterial
            color={filterColor}
            opacity={0.8}
            transparent
            roughness={0.3}
            metalness={0.1}
          />
        </mesh>
      );
      
      // Heavy outer border - create 3D rectangular frame for visibility from all angles
      
      // Top border
      screenElements.push(
        <mesh key="border-top" position={[x, y + filterHeight/2 + 0.02, z]}>
          <boxGeometry args={[filterWidth * 1.1, 0.04, filterThickness * 1.2]} />
          <meshStandardMaterial
            color="#ffffff"
            opacity={0.9}
            transparent
            roughness={0.1}
            metalness={0.3}
          />
        </mesh>
      );
      
      // Bottom border
      screenElements.push(
        <mesh key="border-bottom" position={[x, y - filterHeight/2 - 0.02, z]}>
          <boxGeometry args={[filterWidth * 1.1, 0.04, filterThickness * 1.2]} />
          <meshStandardMaterial
            color="#ffffff"
            opacity={0.9}
            transparent
            roughness={0.1}
            metalness={0.3}
          />
        </mesh>
      );
      
      // Left border
      screenElements.push(
        <mesh key="border-left" position={[x - filterWidth/2 - 0.02, y, z]}>
          <boxGeometry args={[0.04, filterHeight * 1.1, filterThickness * 1.2]} />
          <meshStandardMaterial
            color="#ffffff"
            opacity={0.9}
            transparent
            roughness={0.1}
            metalness={0.3}
          />
        </mesh>
      );
      
      // Right border
      screenElements.push(
        <mesh key="border-right" position={[x + filterWidth/2 + 0.02, y, z]}>
          <boxGeometry args={[0.04, filterHeight * 1.1, filterThickness * 1.2]} />
          <meshStandardMaterial
            color="#ffffff"
            opacity={0.9}
            transparent
            roughness={0.1}
            metalness={0.3}
          />
        </mesh>
      );
      
      // Fine inner grid lines - horizontal (3D boxes for better visibility)
      for (let row = 1; row < kernelHeight; row++) {
        const lineY = y + (row - kernelHeight/2) * (filterHeight/kernelHeight);
        screenElements.push(
          <mesh key={`h-line-${row}`} position={[x, lineY, z]}>
            <boxGeometry args={[filterWidth * 0.95, 0.015, filterThickness * 0.8]} />
            <meshStandardMaterial
              color="#ffffff"
              opacity={0.8}
              transparent
              roughness={0.2}
              metalness={0.1}
            />
          </mesh>
        );
      }
      
      // Fine inner grid lines - vertical (3D boxes for better visibility)
      for (let col = 1; col < kernelWidth; col++) {
        const lineX = x + (col - kernelWidth/2) * (filterWidth/kernelWidth);
        screenElements.push(
          <mesh key={`v-line-${col}`} position={[lineX, y, z]}>
            <boxGeometry args={[0.015, filterHeight * 0.95, filterThickness * 0.8]} />
            <meshStandardMaterial
              color="#ffffff"
              opacity={0.8}
              transparent
              roughness={0.2}
              metalness={0.1}
            />
          </mesh>
        );
      }
      
      filters.push(
        <group key={`internal-filter-${i}`}>
          {screenElements}
        </group>
      );
    }
    
    return (
      <group>
        {filters}
        {/* Filter count label shown on hover */}
        {hovered && (
          <Text
            position={[0, scaleY * 0.6, 0]}
            fontSize={0.12}
            color="#60A5FA"
            anchorX="center"
            anchorY="middle"
          >
            {filterCount} filters
          </Text>
        )}
      </group>
    );
  }, [layer.filters, layer.kernel_size, layer.layer_type, scaleX, scaleY, scaleZ, hovered]); // Memoize based on static layer properties

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
          opacity={layer.layer_type.toLowerCase().includes('conv') ? 0.15 : layer.opacity}
          transparent={true}
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

      {/* Filter count for Conv layers */}
      {layer.filters && (
        <Text
          position={[0, -scaleY - 1.2, 0]}
          fontSize={0.18}
          color="#60A5FA"
          anchorX="center"
          anchorY="middle"
          visible={hovered}
        >
          {layer.filters} filters {layer.kernel_size && `(${layer.kernel_size[0]}×${layer.kernel_size[1]})`}
        </Text>
      )}

      {/* Internal Filter Stack Visualization */}
      {internalFilterStack}

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