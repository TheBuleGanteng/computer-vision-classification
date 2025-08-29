"use client"

import { useEffect } from 'react';
import { performanceMonitor } from '@/lib/performance-monitor';

export function PerformanceMonitorInit() {
  useEffect(() => {
    // Performance monitor disabled - was causing overhead
    console.log('⏸️ Performance Monitor disabled to eliminate monitoring overhead');
  }, []);

  return null; // This component renders nothing
}