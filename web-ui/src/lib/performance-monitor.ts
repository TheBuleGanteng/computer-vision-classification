/**
 * Performance Monitor - Debug setTimeout violations and identify bottlenecks
 */

interface PerformanceEntry {
  name: string;
  duration: number;
  timestamp: number;
  source: string;
  stackTrace?: string;
}

class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private entries: PerformanceEntry[] = [];
  private isMonitoring = false;
  private originalSetTimeout: typeof setTimeout = setTimeout;
  private originalRequestAnimationFrame: typeof requestAnimationFrame = requestAnimationFrame;

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  startMonitoring() {
    if (this.isMonitoring) return;
    this.isMonitoring = true;

    console.log('🔍 Performance Monitor: Starting violation tracking...');
    
    // Store original functions
    this.originalSetTimeout = window.setTimeout;
    this.originalRequestAnimationFrame = window.requestAnimationFrame;

    // Monitor setTimeout violations
    this.monitorSetTimeout();
    
    // Monitor requestAnimationFrame violations  
    this.monitorRequestAnimationFrame();
    
    // Monitor React scheduler if available
    this.monitorReactScheduler();
    
    // Monitor fetch requests
    this.monitorFetchRequests();

    // Log performance entries periodically
    setInterval(() => this.logPerformanceReport(), 10000);
  }

  private monitorSetTimeout() {
    type TimeoutCallback = (...args: unknown[]) => void;
    window.setTimeout = ((callback: TimeoutCallback, delay?: number) => {
      const stackTrace = new Error().stack?.split('\n').slice(2, 5).join('\n');
      
      const wrappedCallback = (...callbackArgs: unknown[]) => {
        const execStart = performance.now();
        try {
          return callback.apply(this, callbackArgs);
        } finally {
          const duration = performance.now() - execStart;
          if (duration > 50) {
            this.recordEntry({
              name: 'setTimeout handler',
              duration,
              timestamp: Date.now(),
              source: 'setTimeout',
              stackTrace: stackTrace || 'Unknown stack'
            });
            console.warn(`⚠️  setTimeout violation: ${duration.toFixed(2)}ms\n`, stackTrace);
          }
        }
      };
      
      return this.originalSetTimeout.call(window, wrappedCallback, delay || 0)
    }) as typeof setTimeout;
  }

  private monitorRequestAnimationFrame() {
    window.requestAnimationFrame = ((callback: FrameRequestCallback) => {
      const wrappedCallback = (timestamp: number) => {
        const start = performance.now();
        const stackTrace = new Error().stack?.split('\n').slice(2, 5).join('\n');
        
        try {
          return callback(timestamp);
        } finally {
          const duration = performance.now() - start;
          if (duration > 16.67) { // 60fps threshold
            this.recordEntry({
              name: 'requestAnimationFrame handler',
              duration,
              timestamp: Date.now(),
              source: 'requestAnimationFrame',
              stackTrace: stackTrace || 'Unknown stack'
            });
            console.warn(`⚠️  RAF violation: ${duration.toFixed(2)}ms\n`, stackTrace);
          }
        }
      };
      
      return this.originalRequestAnimationFrame.call(window, wrappedCallback);
    });
  }

  private monitorReactScheduler() {
    // Monitor React's internal scheduler if available
    interface WindowWithReact extends Window {
      React?: {
        __SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED?: {
          Scheduler?: unknown;
        };
      };
    }
    
    const windowWithReact = window as WindowWithReact;
    if (typeof window !== 'undefined' && windowWithReact.React) {
      const originalScheduler = windowWithReact.React?.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED?.Scheduler;
      if (originalScheduler) {
        console.log('🔍 Monitoring React scheduler...');
      }
    }

    // Monitor message events that might be React-related
    const originalAddEventListener = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function(type, listener, options) {
      if (type === 'message') {
        const wrappedListener = (event: Event) => {
          const start = performance.now();
          const stackTrace = new Error().stack?.split('\n').slice(2, 5).join('\n');
          
          try {
            return (listener as EventListener).call(this, event);
          } finally {
            const duration = performance.now() - start;
            if (duration > 50) {
              PerformanceMonitor.getInstance().recordEntry({
                name: 'message handler',
                duration,
                timestamp: Date.now(),
                source: 'messageEvent',
                stackTrace: stackTrace || 'Unknown stack'
              });
              console.warn(`⚠️  Message handler violation: ${duration.toFixed(2)}ms\n`, stackTrace);
            }
          }
        };
        
        return originalAddEventListener.call(this, type, wrappedListener, options);
      }
      return originalAddEventListener.call(this, type, listener, options);
    };
  }

  private monitorFetchRequests() {
    const originalFetch = window.fetch;
    
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const start = performance.now();
      const url = typeof input === 'string' ? input : input.toString();
      
      try {
        const response = await originalFetch.call(window, input, init);
        const duration = performance.now() - start;
        
        if (duration > 100) {
          this.recordEntry({
            name: `fetch: ${url}`,
            duration,
            timestamp: Date.now(),
            source: 'fetch'
          });
          console.log(`🌐 Slow fetch: ${url} took ${duration.toFixed(2)}ms`);
        }
        
        return response;
      } catch (error) {
        const duration = performance.now() - start;
        console.error(`❌ Fetch error: ${url} (${duration.toFixed(2)}ms)`, error);
        throw error;
      }
    };
  }

  private recordEntry(entry: PerformanceEntry) {
    this.entries.push(entry);
    
    // Keep only last 100 entries to prevent memory leaks
    if (this.entries.length > 100) {
      this.entries.shift();
    }
  }

  private logPerformanceReport() {
    if (this.entries.length === 0) return;

    const recent = this.entries.filter(e => Date.now() - e.timestamp < 10000);
    if (recent.length === 0) return;

    console.group('📊 Performance Report (last 10s)');
    
    const bySource = recent.reduce((acc, entry) => {
      if (!acc[entry.source]) acc[entry.source] = [];
      acc[entry.source].push(entry);
      return acc;
    }, {} as Record<string, PerformanceEntry[]>);

    Object.entries(bySource).forEach(([source, entries]) => {
      const avgDuration = entries.reduce((sum, e) => sum + e.duration, 0) / entries.length;
      const maxDuration = Math.max(...entries.map(e => e.duration));
      
      console.log(`${source}: ${entries.length} violations, avg: ${avgDuration.toFixed(2)}ms, max: ${maxDuration.toFixed(2)}ms`);
      
      // Log the worst offender
      const worst = entries.reduce((max, e) => e.duration > max.duration ? e : max);
      if (worst.stackTrace) {
        console.log(`  Worst offender (${worst.duration.toFixed(2)}ms):`, worst.stackTrace);
      }
    });
    
    console.groupEnd();
  }

  getReport() {
    return {
      totalViolations: this.entries.length,
      recentViolations: this.entries.filter(e => Date.now() - e.timestamp < 30000),
      bySource: this.entries.reduce((acc, entry) => {
        if (!acc[entry.source]) acc[entry.source] = 0;
        acc[entry.source]++;
        return acc;
      }, {} as Record<string, number>)
    };
  }

  stopMonitoring() {
    if (!this.isMonitoring) return;
    
    console.log('🔍 Performance Monitor: Stopping...');
    this.isMonitoring = false;
    
    // Restore original functions
    if (this.originalSetTimeout) {
      window.setTimeout = this.originalSetTimeout;
    }
    if (this.originalRequestAnimationFrame) {
      window.requestAnimationFrame = this.originalRequestAnimationFrame;
    }
  }
}

export const performanceMonitor = PerformanceMonitor.getInstance();

// Auto-start monitoring in development
if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
  performanceMonitor.startMonitoring();
}