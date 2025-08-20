import * as React from "react"
import { ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"

interface SelectProps {
  placeholder?: string
  value?: string
  onValueChange?: (value: string) => void
  children: React.ReactNode
  disabled?: boolean
}

interface SelectItemProps {
  value: string
  children: React.ReactNode
}

export function SelectItem({ value, children, ...props }: SelectItemProps & { onClick?: () => void }) {
  return (
    <div
      className="relative flex cursor-pointer select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-gray-200 hover:text-gray-900"
      {...props}
    >
      {children}
    </div>
  )
}

// Helper function to check if a React element is a SelectItem
const isSelectItemElement = (element: React.ReactNode): element is React.ReactElement<SelectItemProps> => {
  return React.isValidElement(element) && element.type === SelectItem
}

export function Select({ placeholder, value, onValueChange, children, disabled }: SelectProps) {
  const [isOpen, setIsOpen] = React.useState(false)
  const [selectedValue, setSelectedValue] = React.useState(value || "")
  const selectRef = React.useRef<HTMLDivElement>(null)

  const handleSelect = (itemValue: string) => {
    setSelectedValue(itemValue)
    onValueChange?.(itemValue)
    setIsOpen(false)
  }

  // Click outside to close dropdown
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (selectRef.current && !selectRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  return (
    <div className="relative" ref={selectRef}>
      <button
        type="button"
        className={cn(
          "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          disabled && "cursor-not-allowed opacity-50"
        )}
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
      >
        <span className={selectedValue ? "" : "text-muted-foreground"}>
          {selectedValue || placeholder}
        </span>
        <ChevronDown className="h-4 w-4 opacity-50" />
      </button>
      
      {isOpen && (
        <div className="absolute z-50 mt-1 w-full rounded-md border bg-gray-50 text-gray-900 shadow-lg p-1">
          {React.Children.map(children, (child) => {
            if (isSelectItemElement(child)) {
              return React.cloneElement(child, {
                ...child.props,
                onClick: () => handleSelect(child.props.value)
              })
            }
            return child
          })}
        </div>
      )}
    </div>
  )
}