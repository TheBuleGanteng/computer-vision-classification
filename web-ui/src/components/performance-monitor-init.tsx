"use client"

import { useEffect } from 'react';
import { logger } from '@/lib/logger';

export function PerformanceMonitorInit() {
  useEffect(() => {
    // Performance monitor disabled - was causing overhead
    logger.log('⏸️ Performance Monitor disabled to eliminate monitoring overhead');
  }, []);

  return null; // This component renders nothing
}