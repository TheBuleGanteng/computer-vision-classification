"use client"

import React from 'react';
import { X } from 'lucide-react';

interface LayerData {
  id: string;
  type: string;
  label: string;
  parameters?: number;
  color_intensity?: number;
  opacity?: number;
  units?: number;
  filters?: number;
  kernel_size?: number[];
  pool_size?: number[];
  activation?: string;
  dropout_rate?: number;
}

interface LayerInfoPanelProps {
  layer: LayerData;
  onClose: () => void;
}

export const LayerInfoPanel: React.FC<LayerInfoPanelProps> = ({ layer, onClose }) => {
  const getLayerTypeDescription = (type: string): string => {
    switch (type.toLowerCase()) {
      case 'input':
        return 'Input layer that receives the raw data into the neural network';
      case 'conv2d':
        return 'Convolutional layer that applies filters to detect features in the input';
      case 'dense':
        return 'Fully connected layer where each neuron connects to all neurons in the previous layer';
      case 'lstm':
        return 'Long Short-Term Memory layer for processing sequential data with memory';
      case 'maxpooling2d':
      case 'maxpool2d':
        return 'Max pooling layer that reduces spatial dimensions by taking maximum values';
      case 'dropout':
        return 'Regularization layer that randomly sets input units to 0 during training';
      case 'activation':
        return 'Activation function layer that applies non-linear transformations';
      case 'output':
        return 'Output layer that produces the final predictions of the network';
      default:
        return 'Neural network layer component';
    }
  };

  const getLayerColor = (type: string): string => {
    switch (type.toLowerCase()) {
      case 'input': return 'bg-green-600';
      case 'conv2d': return 'bg-blue-500';
      case 'dense': return 'bg-amber-500';
      case 'lstm': return 'bg-purple-500';
      case 'maxpooling2d':
      case 'maxpool2d': return 'bg-emerald-500';
      case 'dropout': return 'bg-red-500';
      case 'activation': return 'bg-lime-500';
      case 'output': return 'bg-red-600';
      default: return 'bg-gray-500';
    }
  };

  const formatNumber = (num: number | undefined): string => {
    if (num === undefined) return 'N/A';
    return num.toLocaleString();
  };

  return (
    <div className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50">
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm" 
        onClick={onClose}
      />
      
      <div className="relative bg-gray-800 border border-gray-600 rounded-lg p-6 max-w-md w-full mx-4 shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`w-4 h-4 rounded ${getLayerColor(layer.type)}`} />
            <div>
              <h3 className="text-lg font-semibold text-white">{layer.label}</h3>
              <p className="text-sm text-gray-400 capitalize">{layer.type} Layer</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white p-1 rounded transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Description */}
        <div className="mb-4 p-3 bg-gray-700 rounded-lg">
          <p className="text-sm text-gray-300">
            {getLayerTypeDescription(layer.type)}
          </p>
        </div>

        {/* Layer Details */}
        <div className="space-y-3">
          <div className="text-sm">
            <span className="text-gray-400">Layer ID:</span>
            <span className="ml-2 text-white font-mono text-xs">{layer.id}</span>
          </div>

          {layer.parameters !== undefined && (
            <div className="text-sm">
              <span className="text-gray-400">Parameters:</span>
              <span className="ml-2 text-blue-400 font-medium">
                {formatNumber(layer.parameters)}
              </span>
            </div>
          )}

          {layer.units !== undefined && (
            <div className="text-sm">
              <span className="text-gray-400">Units/Neurons:</span>
              <span className="ml-2 text-green-400 font-medium">
                {formatNumber(layer.units)}
              </span>
            </div>
          )}

          {layer.filters !== undefined && (
            <div className="text-sm">
              <span className="text-gray-400">Filters:</span>
              <span className="ml-2 text-purple-400 font-medium">
                {formatNumber(layer.filters)}
              </span>
            </div>
          )}

          {layer.kernel_size && (
            <div className="text-sm">
              <span className="text-gray-400">Kernel Size:</span>
              <span className="ml-2 text-cyan-400 font-medium">
                {layer.kernel_size.join(' × ')}
              </span>
            </div>
          )}

          {layer.pool_size && (
            <div className="text-sm">
              <span className="text-gray-400">Pool Size:</span>
              <span className="ml-2 text-teal-400 font-medium">
                {layer.pool_size.join(' × ')}
              </span>
            </div>
          )}

          {layer.activation && (
            <div className="text-sm">
              <span className="text-gray-400">Activation:</span>
              <span className="ml-2 text-yellow-400 font-medium capitalize">
                {layer.activation}
              </span>
            </div>
          )}

          {layer.dropout_rate !== undefined && (
            <div className="text-sm">
              <span className="text-gray-400">Dropout Rate:</span>
              <span className="ml-2 text-red-400 font-medium">
                {(layer.dropout_rate * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
        
        {/* Performance indicator */}
        {layer.color_intensity !== undefined && (
          <div className="mt-4 pt-3 border-t border-gray-700">
            <div className="flex justify-between text-xs mb-2">
              <span className="text-gray-400">Performance Impact</span>
              <span className="text-blue-400">
                {(layer.color_intensity * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-blue-400 h-2 rounded-full transition-all duration-500"
                style={{ width: `${layer.color_intensity * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Educational Tips */}
        <div className="mt-4 p-3 bg-blue-900 bg-opacity-30 rounded-lg border border-blue-700">
          <div className="text-xs text-blue-300 font-medium mb-1">💡 Educational Tip</div>
          <div className="text-xs text-blue-200">
            {layer.type.toLowerCase() === 'conv2d' && 
              'Convolutional layers learn spatial features like edges, textures, and patterns through their filters.'
            }
            {layer.type.toLowerCase() === 'dense' && 
              'Dense layers combine features from previous layers to make final decisions or predictions.'
            }
            {layer.type.toLowerCase() === 'lstm' && 
              'LSTM layers can remember important information across time steps, making them perfect for sequences.'
            }
            {layer.type.toLowerCase() === 'maxpooling2d' && 
              'Pooling reduces the size of feature maps while keeping the most important information.'
            }
            {layer.type.toLowerCase() === 'dropout' && 
              'Dropout prevents overfitting by forcing the network to not rely too heavily on any single neuron.'
            }
            {layer.type.toLowerCase() === 'input' && 
              'The input layer shapes and normalizes your data before it flows through the network.'
            }
            {layer.type.toLowerCase() === 'output' && 
              'The output layer produces your final predictions using the features learned by previous layers.'
            }
          </div>
        </div>
      </div>
    </div>
  );
};

export default LayerInfoPanel;