/**
 * Request Coordinator - Prevents simultaneous API calls from overwhelming React
 * Coordinates multiple polling systems to stagger requests and prevent setTimeout violations
 */

class RequestCoordinator {
  private static instance: RequestCoordinator;
  private activeRequests = new Set<string>();
  private requestQueue: Array<{ key: string; fn: () => Promise<unknown>; resolve: (value: unknown) => void; reject: (error: unknown) => void }> = [];
  private processingQueue = false;

  static getInstance(): RequestCoordinator {
    if (!RequestCoordinator.instance) {
      RequestCoordinator.instance = new RequestCoordinator();
    }
    return RequestCoordinator.instance;
  }

  /**
   * Coordinates API requests to prevent overlapping calls that cause setTimeout violations
   */
  async coordinateRequest<T>(key: string, requestFn: () => Promise<T>): Promise<T> {
    // If same request is already active, wait for it
    if (this.activeRequests.has(key)) {
      await this.waitForRequest(key);
    }

    // Check if too many requests are active (mobile optimization)
    const isMobile = typeof window !== 'undefined' && window.innerWidth <= 768;
    const maxConcurrent = isMobile ? 1 : 2; // Aggressive limiting on mobile

    if (this.activeRequests.size >= maxConcurrent) {
      return new Promise<T>((resolve, reject) => {
        this.requestQueue.push({ key, fn: requestFn, resolve: resolve as (value: unknown) => void, reject: reject as (error: unknown) => void });
        this.processQueue();
      });
    }

    return this.executeRequest(key, requestFn);
  }

  private async executeRequest<T>(key: string, requestFn: () => Promise<T>): Promise<T> {
    this.activeRequests.add(key);
    
    try {
      const result = await requestFn();
      return result;
    } finally {
      this.activeRequests.delete(key);
      this.processQueue();
    }
  }

  private async processQueue() {
    if (this.processingQueue || this.requestQueue.length === 0) return;
    
    const isMobile = typeof window !== 'undefined' && window.innerWidth <= 768;
    const maxConcurrent = isMobile ? 1 : 2;
    
    if (this.activeRequests.size >= maxConcurrent) return;

    this.processingQueue = true;
    const { key, fn, resolve, reject } = this.requestQueue.shift()!;
    
    try {
      const result = await this.executeRequest(key, fn);
      resolve(result);
    } catch (error) {
      reject(error);
    } finally {
      this.processingQueue = false;
    }
  }

  private async waitForRequest(key: string): Promise<void> {
    return new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        if (!this.activeRequests.has(key)) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 100);
    });
  }

  /**
   * Stagger polling intervals to prevent simultaneous requests
   */
  getStaggeredInterval(baseInterval: number, offset: number): number {
    return baseInterval + (offset * 1000); // Add offset in seconds
  }

  /**
   * Check if system is under heavy load
   */
  isUnderHeavyLoad(): boolean {
    return this.activeRequests.size > 1 || this.requestQueue.length > 0;
  }
}

export const requestCoordinator = RequestCoordinator.getInstance();