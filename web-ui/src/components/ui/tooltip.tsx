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

  React.useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  const handleToggle = () => {
    if (isMobile) {
      setIsVisible(!isVisible)
    }
  }

  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => !isMobile && setIsVisible(true)}
        onMouseLeave={() => !isMobile && setIsVisible(false)}
        onClick={handleToggle}
        className="cursor-help"
      >
        {children}
      </div>
      
      {isVisible && (
        <div className={cn(
          "absolute z-50 top-full left-1/2 transform -translate-x-1/2 mt-2 w-80 p-3 bg-gray-50 text-gray-900 text-sm rounded-md border shadow-lg",
          className
        )}>
          {content}
          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 w-2 h-2 bg-gray-50 border-l border-t rotate-45" />
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