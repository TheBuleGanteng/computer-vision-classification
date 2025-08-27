'use client';

import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Mesh, Vector3 } from 'three';

interface AnimatedArrowProps {
  start: [number, number, number];
  end: [number, number, number];
  color?: string;
  speed?: number;
}

export const AnimatedArrow: React.FC<AnimatedArrowProps> = ({
  start,
  end,
  color = "#10B981",
  speed = 1
}) => {
  const arrowRef = useRef<Mesh>(null);
  const glowRef = useRef<Mesh>(null);
  
  // Calculate arrow direction and position
  const startVector = new Vector3(...start);
  const endVector = new Vector3(...end);
  const direction = endVector.clone().sub(startVector).normalize();
  const distance = startVector.distanceTo(endVector);
  const midpoint = startVector.clone().add(endVector).multiplyScalar(0.5);
  
  // Animation: pulsing and moving glow effect
  useFrame(({ clock }) => {
    const time = clock.getElapsedTime() * speed;
    
    if (arrowRef.current) {
      // Gentle pulsing
      const pulse = 1 + Math.sin(time * 2) * 0.1;
      arrowRef.current.scale.setScalar(pulse);
    }
    
    if (glowRef.current) {
      // Moving glow along the arrow path
      const progress = (Math.sin(time) + 1) / 2; // 0 to 1
      const glowPosition = startVector.clone().lerp(endVector, progress);
      glowRef.current.position.copy(glowPosition);
      
      // Fade in/out based on position
      const material = glowRef.current.material as THREE.MeshBasicMaterial;
      if (material && 'opacity' in material) {
        material.opacity = Math.sin(progress * Math.PI) * 0.8;
      }
    }
  });

  return (
    <group>
      {/* Main Arrow Body */}
      <mesh
        ref={arrowRef}
        position={[midpoint.x, midpoint.y, midpoint.z]}
        lookAt={endVector}
      >
        <cylinderGeometry args={[0.02, 0.02, distance * 0.8, 8]} />
        <meshStandardMaterial
          color={color}
          opacity={0.7}
          transparent
          emissive={color}
          emissiveIntensity={0.2}
        />
      </mesh>

      {/* Arrow Head */}
      <mesh
        position={[
          end[0] - direction.x * 0.3,
          end[1] - direction.y * 0.3,
          end[2] - direction.z * 0.3
        ]}
        lookAt={endVector}
        rotation={[Math.PI / 2, 0, 0]}
      >
        <coneGeometry args={[0.08, 0.3, 6]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.3}
        />
      </mesh>

      {/* Animated Glow Effect */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Connection Line (subtle) */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([...start, ...end])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color={color}
          opacity={0.3}
          transparent
        />
      </line>
    </group>
  );
};