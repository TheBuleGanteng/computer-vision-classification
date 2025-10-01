"use client"

import * as React from "react"

export interface SliderProps {
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  disabled?: boolean
  className?: string
  label?: string
  showValue?: boolean
}

export function Slider({
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  className = "",
  label,
  showValue = true
}: SliderProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(parseFloat(e.target.value))
  }

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      {label && (
        <div className="flex justify-between items-center">
          <label className="text-sm font-medium text-gray-700">{label}</label>
          {showValue && (
            <span className="text-sm font-medium text-gray-600">{value.toFixed(1)}%</span>
          )}
        </div>
      )}
      <div className="flex items-center gap-3">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleChange}
          disabled={disabled}
          className={`
            flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer
            disabled:opacity-50 disabled:cursor-not-allowed
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-blue-600
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:transition-colors
            [&::-webkit-slider-thumb]:hover:bg-blue-700
            [&::-moz-range-thumb]:w-4
            [&::-moz-range-thumb]:h-4
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:bg-blue-600
            [&::-moz-range-thumb]:cursor-pointer
            [&::-moz-range-thumb]:border-0
            [&::-moz-range-thumb]:transition-colors
            [&::-moz-range-thumb]:hover:bg-blue-700
          `}
        />
        {!label && showValue && (
          <span className="text-xs font-medium text-gray-300 w-12 text-right">{value.toFixed(1)}%</span>
        )}
      </div>
    </div>
  )
}
