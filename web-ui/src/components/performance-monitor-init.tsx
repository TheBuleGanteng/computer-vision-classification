"use client"

import { useEffect } from 'react';

export function PerformanceMonitorInit() {
  useEffect(() => {
    // Performance monitor disabled - was causing overhead
    console.log('⏸️ Performance Monitor disabled to eliminate monitoring overhead');
  }, []);

  return null; // This component renders nothing
}