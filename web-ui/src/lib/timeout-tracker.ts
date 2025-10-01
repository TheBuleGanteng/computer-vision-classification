'use client'

/**
 * setTimeout tracking utility to identify sources of React violations
 * This helps debug which setTimeout calls are causing performance issues
 */

interface TimeoutInfo {
  id: number
  file: string
  line: number
  startTime: number
  callback: (...args: unknown[]) => void
  delay: number
}

class TimeoutTracker {
  private timeouts: Map<number, TimeoutInfo> = new Map()
  private originalSetTimeout: typeof setTimeout
  private originalClearTimeout: typeof clearTimeout
  private enabled = false

  constructor() {
    this.originalSetTimeout = setTimeout
    this.originalClearTimeout = clearTimeout
  }

  enable() {
    if (this.enabled || typeof window === 'undefined') return

    this.enabled = true
    console.log('🔍 setTimeout tracker enabled - watching for violations...')

    // Override global setTimeout
    window.setTimeout = ((callback: (...args: unknown[]) => void, delay: number = 0, ...args: unknown[]) => {
      const stack = new Error().stack || ''
      const stackLines = stack.split('\n')
      
      // Find the first line that's not from this tracker or browser internals
      let callerInfo = 'unknown:0'
      for (let i = 2; i < stackLines.length; i++) {
        const line = stackLines[i]
        if (line && 
            !line.includes('timeout-tracker') && 
            !line.includes('node_modules') &&
            !line.includes('react-dom') &&
            !line.includes('webpack') &&
            (line.includes('.tsx') || line.includes('.ts') || line.includes('.jsx') || line.includes('.js'))) {
          
          // Extract file and line number
          const match = line.match(/\((.*):(\d+):(\d+)\)/) || line.match(/at (.*):(\d+):(\d+)/)
          if (match) {
            const fullPath = match[1]
            const fileName = fullPath.split('/').pop() || fullPath
            callerInfo = `${fileName}:${match[2]}`
            break
          }
        }
      }

      const startTime = performance.now()
      const id = this.originalSetTimeout(() => {
        const duration = performance.now() - startTime
        
        // Log if it takes longer than React's threshold (usually ~5-16ms)
        if (duration > 16) {
          console.warn(`⚠️ SLOW setTimeout detected:`)
          console.warn(`   📁 File: ${callerInfo}`)
          console.warn(`   ⏱️ Duration: ${duration.toFixed(2)}ms`)
          console.warn(`   🔢 Delay: ${delay}ms`)
          console.warn(`   📊 Stack trace:`, stack)
        }
        
        this.timeouts.delete(id)
        return callback.apply(this, args)
      }, delay) as unknown as number

      // Store timeout info
      this.timeouts.set(id, {
        id,
        file: callerInfo,
        line: parseInt(callerInfo.split(':')[1]) || 0,
        startTime,
        callback,
        delay
      })

      return id
    }) as typeof setTimeout

    // Override clearTimeout to clean up tracking
    window.clearTimeout = (id?: number | string | NodeJS.Timeout) => {
      if (typeof id === 'number') {
        this.timeouts.delete(id)
      }
      return this.originalClearTimeout(id as number)
    }
  }

  disable() {
    if (!this.enabled || typeof window === 'undefined') return

    this.enabled = false
    window.setTimeout = this.originalSetTimeout
    window.clearTimeout = this.originalClearTimeout
    this.timeouts.clear()
    console.log('🔍 setTimeout tracker disabled')
  }

  getActiveTimeouts() {
    return Array.from(this.timeouts.values())
  }

  logStats() {
    const active = this.getActiveTimeouts()
    console.log(`📊 setTimeout tracker stats:`)
    console.log(`   Active timeouts: ${active.length}`)
    
    if (active.length > 0) {
      console.log(`   Files with active timeouts:`)
      const byFile = active.reduce((acc, timeout) => {
        acc[timeout.file] = (acc[timeout.file] || 0) + 1
        return acc
      }, {} as Record<string, number>)
      
      Object.entries(byFile)
        .sort(([,a], [,b]) => b - a)
        .forEach(([file, count]) => {
          console.log(`     ${file}: ${count} timeouts`)
        })
    }
  }
}

// Export singleton instance
export const timeoutTracker = new TimeoutTracker()

// Helper to enable tracking in development
export const enableTimeoutTracking = () => {
  if (process.env.NODE_ENV === 'development') {
    timeoutTracker.enable()
    
    // Log stats every 10 seconds in development
    setInterval(() => {
      timeoutTracker.logStats()
    }, 10000)
  }
}

export const disableTimeoutTracking = () => {
  timeoutTracker.disable()
}