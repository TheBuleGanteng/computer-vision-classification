'use client';

import React from 'react';
import { X } from 'lucide-react';
import { LayerVisualization, getPerformanceColor } from '@/types/visualization';

interface LayerInfoPanelProps {
  selectedLayer: LayerVisualization | null;
  hoveredLayer: LayerVisualization | null;
  onClose: () => void;
}

export const LayerInfoPanel: React.FC<LayerInfoPanelProps> = ({
  selectedLayer,
  hoveredLayer,
  onClose
}) => {
  const displayLayer = selectedLayer || hoveredLayer;
  
  if (!displayLayer) return null;

  const isSelected = !!selectedLayer;
  const performanceColor = getPerformanceColor(displayLayer.color_intensity);

  // Get layer-specific details
  const getLayerDetails = () => {
    const details: Array<{ label: string; value: string | number }> = [
      { label: 'Type', value: displayLayer.layer_type },
      { label: 'Position', value: `Z: ${displayLayer.position_z.toFixed(2)}` },
      { label: 'Dimensions', value: `${displayLayer.width.toFixed(2)} × ${displayLayer.height.toFixed(2)} × ${displayLayer.depth.toFixed(2)}` }
    ];

    if (displayLayer.parameters > 0) {
      details.push({ label: 'Parameters', value: displayLayer.parameters.toLocaleString() });
    }

    if (displayLayer.filters) {
      details.push({ label: 'Filters', value: displayLayer.filters });
    }

    if (displayLayer.kernel_size) {
      details.push({ 
        label: 'Kernel Size', 
        value: `${displayLayer.kernel_size[0]} × ${displayLayer.kernel_size[1]}` 
      });
    }

    if (displayLayer.units) {
      details.push({ label: 'Units', value: displayLayer.units });
    }

    details.push(
      { label: 'Color Intensity', value: `${(displayLayer.color_intensity * 100).toFixed(1)}%` },
      { label: 'Opacity', value: `${(displayLayer.opacity * 100).toFixed(0)}%` }
    );

    return details;
  };

  // Get layer type icon
  const getLayerIcon = () => {
    switch (displayLayer.layer_type.toLowerCase()) {
      case 'conv':
      case 'conv2d':
      case 'conv1d':
        return '⚡';
      case 'lstm':
      case 'gru':
      case 'rnn':
        return '🔄';
      case 'dense':
      case 'linear':
        return '▦';
      case 'pooling':
      case 'maxpooling2d':
      case 'averagepooling2d':
      case 'globalmaxpooling2d':
      case 'globalaveragepooling2d':
        return '⬇️';
      case 'dropout':
        return '🎲';
      case 'batchnormalization':
        return '📊';
      default:
        return '📦';
    }
  };

  // Get layer description
  const getLayerDescription = () => {
    switch (displayLayer.layer_type.toLowerCase()) {
      case 'conv':
      case 'conv2d':
      case 'conv1d':
        return 'Convolutional layer that applies filters to extract features from the input';
      case 'lstm':
        return 'Long Short-Term Memory layer for processing sequential data';
      case 'gru':
        return 'Gated Recurrent Unit layer for sequence processing';
      case 'rnn':
        return 'Recurrent Neural Network layer for temporal patterns';
      case 'dense':
      case 'linear':
        return 'Fully connected layer that performs linear transformation';
      case 'maxpooling2d':
        return 'Max pooling layer that reduces spatial dimensions by taking maximum values';
      case 'averagepooling2d':
        return 'Average pooling layer that reduces spatial dimensions by averaging values';
      case 'globalmaxpooling2d':
        return 'Global max pooling that reduces each feature map to a single value';
      case 'globalaveragepooling2d':
        return 'Global average pooling that reduces each feature map to a single value';
      case 'dropout':
        return 'Regularization layer that randomly sets inputs to zero during training';
      case 'batchnormalization':
        return 'Normalization layer that stabilizes and accelerates training';
      default:
        return 'Neural network layer for data processing';
    }
  };

  return (
    <div className={`
      absolute top-4 left-4 
      bg-gray-900/95 backdrop-blur-sm border border-gray-700 rounded-lg
      text-white text-sm max-w-80
      transition-all duration-200 ease-in-out
      ${isSelected ? 'opacity-100' : 'opacity-90'}
    `}>
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-gray-700">
        <div className="flex items-center space-x-2">
          <span className="text-lg">{getLayerIcon()}</span>
          <h3 className="font-semibold text-blue-400">
            {displayLayer.layer_type}
          </h3>
          <div 
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: performanceColor }}
            title={`Performance: ${(displayLayer.color_intensity * 100).toFixed(1)}%`}
          />
        </div>
        {isSelected && (
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X size={16} />
          </button>
        )}
      </div>

      {/* Content */}
      <div className="p-3 space-y-3">
        {/* Description */}
        <p className="text-gray-300 text-xs leading-relaxed">
          {getLayerDescription()}
        </p>

        {/* Details */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          {getLayerDetails().map((detail, index) => (
            <div key={index} className="space-y-1">
              <div className="text-gray-400 font-medium">
                {detail.label}
              </div>
              <div className="text-white font-mono">
                {detail.value}
              </div>
            </div>
          ))}
        </div>

        {/* Performance indicator */}
        <div className="pt-2 border-t border-gray-700">
          <div className="flex justify-between items-center text-xs">
            <span className="text-gray-400">Performance Score</span>
            <span 
              className="font-semibold"
              style={{ color: performanceColor }}
            >
              {(displayLayer.color_intensity * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mt-1">
            <div 
              className="h-2 rounded-full transition-all duration-300"
              style={{ 
                width: `${displayLayer.color_intensity * 100}%`,
                backgroundColor: performanceColor
              }}
            />
          </div>
        </div>
      </div>

      {/* Footer hint */}
      {!isSelected && (
        <div className="px-3 pb-3 text-xs text-gray-500">
          Click to pin this information
        </div>
      )}
    </div>
  );
};