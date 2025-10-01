import * as React from "react"
import { cn } from "@/lib/utils"

interface TooltipProps {
  content: React.ReactNode
  children: React.ReactNode
  className?: string
}

export function Tooltip({ content, children, className }: TooltipProps) {
  const [isVisible, setIsVisible] = React.useState(false)
  const [isMobile, setIsMobile] = React.useState(false)
  const [position, setPosition] = React.useState<'left' | 'center' | 'right'>('center')
  const tooltipRef = React.useRef<HTMLDivElement>(null)
  const triggerRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  // Calculate optimal position when tooltip becomes visible
  React.useEffect(() => {
    if (isVisible && triggerRef.current && !isMobile) {
      const triggerRect = triggerRef.current.getBoundingClientRect()
      const viewportWidth = window.innerWidth
      const tooltipWidth = 320 // w-80 = 320px
      const padding = 16 // Account for some padding from screen edges
      
      // Calculate space on left and right
      const spaceOnLeft = triggerRect.left
      const spaceOnRight = viewportWidth - triggerRect.right
      
      // Determine optimal position
      if (spaceOnLeft >= tooltipWidth / 2 + padding && spaceOnRight >= tooltipWidth / 2 + padding) {
        // Enough space to center
        setPosition('center')
      } else if (spaceOnRight >= tooltipWidth + padding) {
        // Not enough space to center, but enough to align left
        setPosition('left')
      } else if (spaceOnLeft >= tooltipWidth + padding) {
        // Not enough space on right, align right
        setPosition('right')
      } else {
        // Very little space, use center and let it be constrained
        setPosition('center')
      }
    }
  }, [isVisible, isMobile])

  const handleToggle = () => {
    setIsVisible(!isVisible)
  }

  // Handle click outside to close tooltip
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (isVisible && 
          triggerRef.current && 
          tooltipRef.current && 
          !triggerRef.current.contains(event.target as Node) && 
          !tooltipRef.current.contains(event.target as Node)) {
        setIsVisible(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isVisible])

  // Get positioning classes based on calculated position
  const getPositionClasses = () => {
    if (isMobile) {
      // On mobile, use full-width centered positioning
      return "fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[calc(100vw-2rem)] max-w-sm mx-4 z-50"
    }
    
    switch (position) {
      case 'left':
        return "absolute z-50 top-full left-0 mt-2 w-80"
      case 'right': 
        return "absolute z-50 top-full right-0 mt-2 w-80"
      case 'center':
      default:
        return "absolute z-50 top-full left-1/2 transform -translate-x-1/2 mt-2 w-80"
    }
  }

  const getArrowClasses = () => {
    if (isMobile) return "hidden" // Hide arrow on mobile
    
    switch (position) {
      case 'left':
        return "absolute bottom-full left-4 w-2 h-2 bg-white border-l border-t rotate-45"
      case 'right':
        return "absolute bottom-full right-4 w-2 h-2 bg-white border-l border-t rotate-45"
      case 'center':
      default:
        return "absolute bottom-full left-1/2 transform -translate-x-1/2 w-2 h-2 bg-white border-l border-t rotate-45"
    }
  }

  return (
    <div className="relative inline-block">
      <div
        ref={triggerRef}
        onClick={handleToggle}
        className="cursor-help"
      >
        {children}
      </div>
      
      {isVisible && (
        <div 
          ref={tooltipRef}
          className={cn(
            getPositionClasses(),
            "p-3 bg-white text-gray-900 text-sm rounded-md border shadow-lg opacity-100 backdrop-blur-none",
            className
          )}
          style={{ 
            backgroundColor: 'rgb(255, 255, 255)',
            opacity: 1,
            backdropFilter: 'none',
            WebkitBackdropFilter: 'none',
            ...(isMobile && {
              backgroundColor: 'rgba(255, 255, 255, 1)',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)'
            })
          }}
        >
          {content}
          <div className={getArrowClasses()} />
        </div>
      )}
      
      {/* Mobile backdrop */}
      {isMobile && isVisible && (
        <div 
          className="fixed inset-0 z-40 bg-black/20" 
          onClick={() => setIsVisible(false)}
        />
      )}
    </div>
  )
}