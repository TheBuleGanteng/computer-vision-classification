"use client"

import { useState, useEffect, useCallback } from "react"
import { Slider } from "@/components/ui/slider"
import { Card, CardContent } from "@/components/ui/card"
import { Tooltip } from "@/components/ui/tooltip"
import { Info, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"

export interface WeightConfig {
  accuracyWeight: number
  healthOverallWeight: number
  healthComponentProportions: {
    neuron_utilization: number
    parameter_efficiency: number
    training_stability: number
    gradient_health: number
    convergence_quality: number
    accuracy_consistency: number
  }
}

interface WeightSlidersProps {
  mode: "simple" | "health"
  onChange: (config: WeightConfig) => void
  defaults?: WeightConfig
}

export function WeightSliders({ mode, onChange, defaults }: WeightSlidersProps) {
  // Tier 1: Accuracy vs Health Overall (0-100 for UI display)
  const [accuracyWeight, setAccuracyWeight] = useState(
    defaults ? defaults.accuracyWeight * 100 : (mode === "simple" ? 100.0 : 70.0)
  )

  // Tier 2: Health sub-component PROPORTIONS (0-1 range, sum to 1.0)
  // These are multiplied by healthOverallWeight to get actual percentages
  const [healthProportions, setHealthProportions] = useState(
    defaults?.healthComponentProportions || {
      neuron_utilization: 0.25,
      parameter_efficiency: 0.15,
      training_stability: 0.20,
      gradient_health: 0.15,
      convergence_quality: 0.15,
      accuracy_consistency: 0.10
    }
  )

  // Initialize from defaults if provided
  useEffect(() => {
    if (defaults) {
      setAccuracyWeight(defaults.accuracyWeight * 100)
      setHealthProportions(defaults.healthComponentProportions)
    }
  }, [defaults])

  // Reset to defaults when mode changes
  useEffect(() => {
    if (defaults) {
      if (mode === "simple") {
        setAccuracyWeight(100.0)
      } else {
        setAccuracyWeight(defaults.accuracyWeight * 100)
      }
      setHealthProportions(defaults.healthComponentProportions)
    }
  }, [mode, defaults])

  // Calculate health overall weight
  const healthOverallWeight = 100.0 - accuracyWeight

  // Notify parent of weight changes - memoize to prevent infinite loops
  const notifyParent = useCallback(() => {
    const config = {
      accuracyWeight: accuracyWeight / 100,
      healthOverallWeight: healthOverallWeight / 100,
      healthComponentProportions: healthProportions
    }
    console.log('[WeightSliders] Notifying parent with weight configuration:', {
      accuracyWeight: `${accuracyWeight.toFixed(1)}% (${config.accuracyWeight.toFixed(3)})`,
      healthOverallWeight: `${healthOverallWeight.toFixed(1)}% (${config.healthOverallWeight.toFixed(3)})`,
      healthComponentProportions: Object.entries(config.healthComponentProportions).map(([key, value]) =>
        `${key}: ${(value * 100).toFixed(1)}% of health (${value.toFixed(3)} proportion)`
      ),
      totalCheck: `Accuracy + Health = ${(config.accuracyWeight + config.healthOverallWeight).toFixed(3)} (should be 1.0)`,
      healthProportionsSum: `Health proportions sum = ${Object.values(config.healthComponentProportions).reduce((sum, val) => sum + val, 0).toFixed(3)} (should be 1.0)`
    })
    onChange(config)
  }, [accuracyWeight, healthOverallWeight, healthProportions, onChange])

  useEffect(() => {
    notifyParent()
  }, [notifyParent])

  // Auto-balance health sub-component proportions
  // newAbsolutePercent is the absolute percentage (e.g., 7.5% out of total 100%)
  const handleHealthComponentChange = (component: keyof typeof healthProportions, newAbsolutePercent: number) => {
    console.log(`[WeightSliders] User adjusted ${component}: ${newAbsolutePercent.toFixed(1)}% (absolute)`)

    // Constrain to not exceed health overall weight
    let clampedAbsolutePercent = newAbsolutePercent;
    if (clampedAbsolutePercent > healthOverallWeight) {
      console.log(`[WeightSliders] Clamping ${component} from ${clampedAbsolutePercent.toFixed(1)}% to ${healthOverallWeight.toFixed(1)}% (max health overall weight)`)
      clampedAbsolutePercent = healthOverallWeight
    }

    // Convert from absolute percentage to proportion (0-1) of health overall
    const newProportion = healthOverallWeight > 0 ? clampedAbsolutePercent / healthOverallWeight : 0
    const oldProportion = healthProportions[component]
    const delta = newProportion - oldProportion

    console.log(`[WeightSliders] Converting ${component}: ${clampedAbsolutePercent.toFixed(1)}% absolute → ${newProportion.toFixed(3)} proportion (delta: ${delta.toFixed(3)})`)

    // Calculate total of other components
    const otherComponents = Object.entries(healthProportions).filter(([key]) => key !== component)
    const otherTotal = otherComponents.reduce((sum, [, value]) => sum + value, 0)

    // If other components sum to zero, can't redistribute
    if (otherTotal === 0) {
      return
    }

    // Distribute the delta proportionally across other components
    const newProportions = { ...healthProportions }
    newProportions[component] = newProportion

    otherComponents.forEach(([key]) => {
      const proportion = healthProportions[key as keyof typeof healthProportions] / otherTotal
      newProportions[key as keyof typeof healthProportions] = healthProportions[key as keyof typeof healthProportions] - (delta * proportion)
    })

    // Ensure all proportions are within bounds and sum to 1.0
    const total = Object.values(newProportions).reduce((sum, val) => sum + val, 0)
    if (Math.abs(total - 1.0) > 0.01) {
      // Normalize to exactly 1.0
      const factor = 1.0 / total
      Object.keys(newProportions).forEach(key => {
        newProportions[key as keyof typeof healthProportions] *= factor
      })
    }

    setHealthProportions(newProportions)
  }

  const handleReset = () => {
    if (defaults) {
      if (mode === "simple") {
        setAccuracyWeight(100.0)
      } else {
        setAccuracyWeight(defaults.accuracyWeight * 100)
      }
      setHealthProportions(defaults.healthComponentProportions)
    }
  }

  return (
    <Card className="bg-gray-900 border-gray-800">
      <CardContent className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-100">Calculation of Total Score</h3>
          <Button
            onClick={handleReset}
            variant="outline"
            size="sm"
            className="text-xs"
            disabled={mode === "simple"}
          >
            <RotateCcw className="h-3 w-3 mr-1" />
            Reset to defaults
          </Button>
        </div>

        {/* Tier 1: Accuracy vs Health Overall */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <h4 className="text-xs font-bold text-gray-200">Primary Weights</h4>
            <Tooltip
              content={
                <div className="space-y-2 text-xs">
                  <p><strong>Accuracy Weight:</strong> How much to prioritize test accuracy in final score</p>
                  <p><strong>Health Overall Weight:</strong> How much to prioritize model health metrics in final score</p>
                  <p className="pt-2 border-t text-gray-400">These weights must sum to 100%</p>
                </div>
              }
            >
              <div className="flex items-center justify-center w-4 h-4 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-colors">
                <Info className="h-2.5 w-2.5" />
              </div>
            </Tooltip>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-xs font-medium text-gray-300 w-32">Accuracy</span>
            <div className="flex-1">
              <Slider
                value={accuracyWeight}
                onChange={setAccuracyWeight}
                min={0}
                max={100}
                step={0.5}
                disabled={mode === "simple"}
                showValue
              />
            </div>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-xs font-medium text-gray-300 w-32">Health Overall</span>
            <div className="flex-1">
              <Slider
                value={healthOverallWeight}
                onChange={(val) => setAccuracyWeight(100.0 - val)}
                min={0}
                max={100}
                step={0.5}
                disabled={mode === "simple"}
                showValue
              />
            </div>
          </div>
        </div>

        {/* Tier 2: Health Sub-Components */}
        {mode === "health" && healthOverallWeight > 0 && (
          <div className="space-y-3 pt-4 border-t border-gray-700">
            <div className="flex items-center gap-2">
              <h4 className="text-xs font-bold text-gray-200">Health Component Proportions</h4>
              <Tooltip
                content={
                  <div className="space-y-2 text-xs">
                    <p><strong>Neuron Utilization:</strong> Active vs inactive neurons</p>
                    <p><strong>Parameter Efficiency:</strong> Performance per parameter</p>
                    <p><strong>Training Stability:</strong> Loss convergence quality</p>
                    <p><strong>Gradient Health:</strong> Gradient flow quality</p>
                    <p><strong>Convergence Quality:</strong> Training smoothness</p>
                    <p><strong>Accuracy Consistency:</strong> Cross-validation stability</p>
                    <p className="pt-2 border-t text-gray-400">These proportions must sum to the value for "Health Overall"</p>
                  </div>
                }
              >
                <div className="flex items-center justify-center w-4 h-4 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-colors">
                  <Info className="h-2.5 w-2.5" />
                </div>
              </Tooltip>
            </div>

            <div className="space-y-2 pl-2">
              <div className="flex items-center gap-4">
                <span className="text-xs font-medium text-gray-300 w-32">Neuron Utilization</span>
                <div className="flex-1">
                  <Slider
                    value={healthProportions.neuron_utilization * healthOverallWeight}
                    onChange={(val) => handleHealthComponentChange('neuron_utilization', val)}
                    min={0}
                    max={100}
                    step={0.1}
                    showValue
                  />
                </div>
              </div>

              <div className="flex items-center gap-4">
                <span className="text-xs font-medium text-gray-300 w-32">Parameter Efficiency</span>
                <div className="flex-1">
                  <Slider
                    value={healthProportions.parameter_efficiency * healthOverallWeight}
                    onChange={(val) => handleHealthComponentChange('parameter_efficiency', val)}
                    min={0}
                    max={100}
                    step={0.1}
                    showValue
                  />
                </div>
              </div>

              <div className="flex items-center gap-4">
                <span className="text-xs font-medium text-gray-300 w-32">Training Stability</span>
                <div className="flex-1">
                  <Slider
                    value={healthProportions.training_stability * healthOverallWeight}
                    onChange={(val) => handleHealthComponentChange('training_stability', val)}
                    min={0}
                    max={100}
                    step={0.1}
                    showValue
                  />
                </div>
              </div>

              <div className="flex items-center gap-4">
                <span className="text-xs font-medium text-gray-300 w-32">Gradient Health</span>
                <div className="flex-1">
                  <Slider
                    value={healthProportions.gradient_health * healthOverallWeight}
                    onChange={(val) => handleHealthComponentChange('gradient_health', val)}
                    min={0}
                    max={100}
                    step={0.1}
                    showValue
                  />
                </div>
              </div>

              <div className="flex items-center gap-4">
                <span className="text-xs font-medium text-gray-300 w-32">Convergence Quality</span>
                <div className="flex-1">
                  <Slider
                    value={healthProportions.convergence_quality * healthOverallWeight}
                    onChange={(val) => handleHealthComponentChange('convergence_quality', val)}
                    min={0}
                    max={100}
                    step={0.1}
                    showValue
                  />
                </div>
              </div>

              <div className="flex items-center gap-4">
                <span className="text-xs font-medium text-gray-300 w-32">Accuracy Consistency</span>
                <div className="flex-1">
                  <Slider
                    value={healthProportions.accuracy_consistency * healthOverallWeight}
                    onChange={(val) => handleHealthComponentChange('accuracy_consistency', val)}
                    min={0}
                    max={100}
                    step={0.1}
                    showValue
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
