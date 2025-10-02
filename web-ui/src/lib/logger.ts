/**
 * Environment-based logger utility
 *
 * In development: All logs are output to console
 * In production: Logs are suppressed (or can be configured to send to monitoring service)
 *
 * Usage:
 *   import { logger } from '@/lib/logger'
 *   logger.log('message', data)
 *   logger.warn('warning', data)
 *   logger.error('error', data)
 */

type LogLevel = 'log' | 'warn' | 'error' | 'info' | 'debug'

const isDevelopment = process.env.NODE_ENV === 'development'

class Logger {
  private shouldLog(level: LogLevel): boolean {
    // In production, only log errors
    if (!isDevelopment && level !== 'error') {
      return false
    }
    return true
  }

  log(...args: unknown[]) {
    if (this.shouldLog('log')) {
      console.log(...args)
    }
  }

  warn(...args: unknown[]) {
    if (this.shouldLog('warn')) {
      console.warn(...args)
    }
  }

  error(...args: unknown[]) {
    if (this.shouldLog('error')) {
      console.error(...args)
    }
  }

  info(...args: unknown[]) {
    if (this.shouldLog('info')) {
      console.info(...args)
    }
  }

  debug(...args: unknown[]) {
    if (this.shouldLog('debug')) {
      console.debug(...args)
    }
  }

  /**
   * Group logs together (collapsed by default)
   * Only works in development
   */
  group(label: string, callback: () => void) {
    if (this.shouldLog('log')) {
      console.group(label)
      callback()
      console.groupEnd()
    }
  }

  /**
   * Group logs together (expanded by default)
   * Only works in development
   */
  groupCollapsed(label: string, callback: () => void) {
    if (this.shouldLog('log')) {
      console.groupCollapsed(label)
      callback()
      console.groupEnd()
    }
  }

  /**
   * Table view for structured data
   * Only works in development
   */
  table(data: unknown) {
    if (this.shouldLog('log')) {
      console.table(data)
    }
  }
}

export const logger = new Logger()
