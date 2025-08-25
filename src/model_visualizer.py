"""
Model 3D Visualization Data Preparation

This module handles the conversion of neural network architecture data into 
3D visualization-ready formats for the React Three Fiber frontend.

Converts model architecture information from TrialProgress objects into
LayerVisualization objects suitable for interactive 3D rendering, including
parameter calculations, spatial positioning, and performance-based styling.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from utils.logger import logger

@dataclass
class LayerVisualization:
    """3D visualization data for a single neural network layer"""
    layer_id: str
    layer_type: str  # "input", "conv2d", "dense", "lstm", "pooling", "dropout"
    
    # 3D positioning
    position_z: float  # Depth position in model
    width: float       # Visual width (nodes/filters)
    height: float      # Visual height
    depth: float       # Visual depth
    
    # Layer-specific data
    parameters: int = 0
    activation: Optional[str] = None
    filters: Optional[int] = None
    kernel_size: Optional[Tuple[int, int]] = None
    units: Optional[int] = None
    
    # Visual properties
    color_intensity: float = 1.0  # Based on parameter count or importance
    opacity: float = 0.8

@dataclass
class ArchitectureVisualization:
    """Complete 3D visualization data for a neural network architecture"""
    architecture_type: str  # "CNN" or "LSTM" 
    layers: List[LayerVisualization]
    
    # 3D rendering parameters
    total_parameters: int
    model_depth: int  # Number of layers for 3D depth
    max_layer_width: float  # Largest layer for scaling
    max_layer_height: float  # Largest layer height for scaling
    
    # Performance overlay data
    performance_score: float  # For color coding (0.0-1.0)
    health_score: Optional[float] = None  # For health mode visualization
    
    @property
    def performance_color(self) -> str:
        """Get color code based on performance score"""
        if self.performance_score >= 0.8:
            return "#10B981"  # Green for good performance
        elif self.performance_score >= 0.6:
            return "#F59E0B"  # Yellow for medium performance
        else:
            return "#EF4444"  # Red for poor performance
    
    @property
    def health_color(self) -> str:
        """Get color code based on health score"""
        if self.health_score is None:
            return "#6B7280"  # Gray if no health data
        elif self.health_score >= 0.8:
            return "#10B981"  # Green for good health
        elif self.health_score >= 0.6:
            return "#F59E0B"  # Yellow for medium health
        else:
            return "#EF4444"  # Red for poor health

class ModelVisualizer:
    """Handles conversion of model architecture to 3D visualization data"""
    
    def __init__(self):
        self.layer_spacing = 2.0  # Standard spacing between layers in 3D space
        self.scale_factor = 0.1   # Scale down large numbers for reasonable 3D sizes
        logger.debug("Initialized ModelVisualizer with default parameters")
        
    def prepare_visualization_data(self, 
                                 architecture: Dict[str, Any], 
                                 performance_score: float = 0.0,
                                 health_score: Optional[float] = None) -> ArchitectureVisualization:
        """
        Convert architecture data into 3D visualization format
        
        Args:
            architecture: Architecture dictionary from TrialProgress
            performance_score: Performance score (0.0-1.0) for color coding
            health_score: Optional health score for additional visual coding
            
        Returns:
            ArchitectureVisualization object ready for 3D rendering
        """
        arch_type = architecture.get('architecture_type') or architecture.get('type', 'unknown')
        logger.debug(f"Preparing 3D visualization for {arch_type} architecture")
        
        if arch_type == 'CNN':
            return self._prepare_cnn_visualization(architecture, performance_score, health_score)
        elif arch_type == 'LSTM':
            return self._prepare_lstm_visualization(architecture, performance_score, health_score)
        else:
            logger.warning(f"Unknown architecture type: {arch_type}")
            return self._prepare_generic_visualization(architecture, performance_score, health_score)
    
    def _prepare_cnn_visualization(self, 
                                 architecture: Dict[str, Any], 
                                 performance_score: float,
                                 health_score: Optional[float]) -> ArchitectureVisualization:
        """Prepare 3D visualization for CNN architecture"""
        
        layers = []
        total_params = 0
        z_position = 0
        
        # Extract architecture parameters
        conv_layers = architecture.get('conv_layers', 0)
        filters_per_layer = architecture.get('filters_per_layer', 32)
        kernel_size = architecture.get('kernel_size', [3, 3])
        dense_layers = architecture.get('dense_layers', 1)
        first_dense_nodes = architecture.get('first_dense_nodes', 128)
        use_global_pooling = architecture.get('use_global_pooling', False)
        
        # Ensure kernel_size is a tuple/list
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size, kernel_size]
        
        logger.debug(f"CNN parameters: {conv_layers} conv layers, {filters_per_layer} filters, {dense_layers} dense layers")
        
        # 1. Input layer (implicit)
        input_layer = LayerVisualization(
            layer_id='input',
            layer_type='input',
            position_z=z_position,
            width=3.2,  # Representing input channels (scaled for visibility)
            height=3.2,  # Representing spatial dimensions
            depth=0.5,
            parameters=0,
            color_intensity=0.6,
            opacity=0.7
        )
        layers.append(input_layer)
        z_position += self.layer_spacing
        
        # 2. Convolutional layers
        prev_channels = 3  # RGB input
        spatial_size = 32  # Assume 32x32 input, common for CIFAR datasets
        
        for i in range(conv_layers):
            # Calculate parameters for this conv layer
            layer_params = (
                kernel_size[0] * kernel_size[1] * 
                filters_per_layer * prev_channels +  # Weights
                filters_per_layer  # Biases
            )
            total_params += layer_params
            
            # Calculate visual dimensions
            # Width represents number of filters
            layer_width = min(filters_per_layer * self.scale_factor, 8.0)  # Cap at reasonable size
            # Height represents spatial dimensions (decreases with pooling/stride)
            layer_height = max(spatial_size * self.scale_factor, 0.5)
            # Depth represents filter depth
            layer_depth = 0.8
            
            # Color intensity based on parameter count
            color_intensity = min(layer_params / 10000.0, 1.0)  # Normalize to 0-1
            
            conv_layer = LayerVisualization(
                layer_id=f'conv_{i+1}',
                layer_type='conv2d',
                position_z=z_position,
                width=layer_width,
                height=layer_height,
                depth=layer_depth,
                parameters=layer_params,
                activation=architecture.get('activation', 'relu'),
                filters=filters_per_layer,
                kernel_size=(kernel_size[0], kernel_size[1]),
                color_intensity=0.8 + color_intensity * 0.2,  # Vary intensity
                opacity=0.85
            )
            layers.append(conv_layer)
            
            # Update for next layer
            prev_channels = filters_per_layer
            spatial_size = max(spatial_size // 2, 1)  # Assume some spatial reduction
            z_position += self.layer_spacing
        
        # 3. Global pooling or flatten layer
        if use_global_pooling:
            pooling_layer = LayerVisualization(
                layer_id='global_pooling',
                layer_type='pooling',
                position_z=z_position,
                width=filters_per_layer * self.scale_factor * 0.5,
                height=0.8,
                depth=0.8,
                parameters=0,
                color_intensity=0.7,
                opacity=0.6
            )
        else:
            # Flatten layer
            flattened_size = filters_per_layer * spatial_size * spatial_size
            pooling_layer = LayerVisualization(
                layer_id='flatten',
                layer_type='pooling',
                position_z=z_position,
                width=min(flattened_size * self.scale_factor * 0.1, 6.0),
                height=0.6,
                depth=0.6,
                parameters=0,
                color_intensity=0.7,
                opacity=0.6
            )
        
        layers.append(pooling_layer)
        z_position += self.layer_spacing
        
        # 4. Dense layers
        prev_units = filters_per_layer if use_global_pooling else (filters_per_layer * spatial_size * spatial_size)
        
        for i in range(dense_layers):
            # Calculate current layer units (decreasing pattern)
            if i == dense_layers - 1:
                # Last dense layer - assume classification layer
                current_units = 10  # Common for CIFAR-10, MNIST
            else:
                current_units = max(first_dense_nodes // (2 ** i), 10)
            
            # Calculate parameters
            layer_params = prev_units * current_units + current_units  # Weights + biases
            total_params += layer_params
            
            # Visual dimensions
            layer_width = min(current_units * self.scale_factor, 6.0)
            layer_height = 1.2 if i < dense_layers - 1 else 1.0  # Output layer slightly smaller
            layer_depth = 1.0
            
            # Color intensity based on parameter count
            color_intensity = min(layer_params / 50000.0, 1.0)
            
            # Determine if this is output layer
            is_output = (i == dense_layers - 1)
            activation = 'softmax' if is_output else architecture.get('activation', 'relu')
            
            dense_layer = LayerVisualization(
                layer_id=f'dense_{i+1}' if not is_output else 'output',
                layer_type='dense',
                position_z=z_position,
                width=layer_width,
                height=layer_height,
                depth=layer_depth,
                parameters=layer_params,
                units=current_units,
                activation=activation,
                color_intensity=0.9 if is_output else (0.7 + color_intensity * 0.3),
                opacity=0.95 if is_output else 0.8
            )
            layers.append(dense_layer)
            
            prev_units = current_units
            z_position += self.layer_spacing
        
        # Calculate max dimensions for scaling
        max_width = max(layer.width for layer in layers)
        max_height = max(layer.height for layer in layers)
        
        logger.debug(f"Generated CNN visualization: {len(layers)} layers, {total_params:,} parameters")
        
        return ArchitectureVisualization(
            architecture_type='CNN',
            layers=layers,
            total_parameters=total_params,
            model_depth=len(layers),
            max_layer_width=max_width,
            max_layer_height=max_height,
            performance_score=performance_score,
            health_score=health_score
        )
    
    def _prepare_lstm_visualization(self, 
                                  architecture: Dict[str, Any], 
                                  performance_score: float,
                                  health_score: Optional[float]) -> ArchitectureVisualization:
        """Prepare 3D visualization for LSTM architecture"""
        
        layers = []
        total_params = 0
        z_position = 0
        
        # Extract LSTM parameters
        lstm_units = architecture.get('lstm_units', 128)
        dense_layers = architecture.get('dense_layers', 1)
        first_dense_nodes = architecture.get('first_dense_nodes', 64)
        
        # Assume some input dimension for LSTM (e.g., for text data)
        input_features = architecture.get('input_features', 100)  # Vocabulary size or feature count
        
        logger.debug(f"LSTM parameters: {lstm_units} LSTM units, {dense_layers} dense layers")
        
        # 1. Input layer
        input_layer = LayerVisualization(
            layer_id='input',
            layer_type='input',
            position_z=z_position,
            width=min(input_features * self.scale_factor, 5.0),
            height=2.0,
            depth=0.5,
            parameters=0,
            color_intensity=0.6,
            opacity=0.7
        )
        layers.append(input_layer)
        z_position += self.layer_spacing
        
        # 2. LSTM layer
        # LSTM has 4 gates, each with their own weights
        lstm_params = 4 * (lstm_units * input_features + lstm_units * lstm_units + lstm_units)
        total_params += lstm_params
        
        lstm_layer = LayerVisualization(
            layer_id='lstm',
            layer_type='lstm',
            position_z=z_position,
            width=min(lstm_units * self.scale_factor, 8.0),
            height=3.0,  # LSTM layers are visually thicker
            depth=1.5,   # More depth to show complexity
            parameters=lstm_params,
            units=lstm_units,
            activation='tanh',  # LSTM internal activation
            color_intensity=0.9,
            opacity=0.85
        )
        layers.append(lstm_layer)
        z_position += self.layer_spacing
        
        # 3. Dense layers
        prev_units = lstm_units
        
        for i in range(dense_layers):
            # Calculate current layer units
            if i == dense_layers - 1:
                current_units = 10  # Output layer
            else:
                current_units = max(first_dense_nodes // (2 ** i), 10)
            
            layer_params = prev_units * current_units + current_units
            total_params += layer_params
            
            is_output = (i == dense_layers - 1)
            activation = 'softmax' if is_output else architecture.get('activation', 'relu')
            
            dense_layer = LayerVisualization(
                layer_id=f'dense_{i+1}' if not is_output else 'output',
                layer_type='dense',
                position_z=z_position,
                width=min(current_units * self.scale_factor, 6.0),
                height=1.2 if not is_output else 1.0,
                depth=1.0,
                parameters=layer_params,
                units=current_units,
                activation=activation,
                color_intensity=0.9 if is_output else 0.8,
                opacity=0.95 if is_output else 0.8
            )
            layers.append(dense_layer)
            
            prev_units = current_units
            z_position += self.layer_spacing
        
        max_width = max(layer.width for layer in layers)
        max_height = max(layer.height for layer in layers)
        
        logger.debug(f"Generated LSTM visualization: {len(layers)} layers, {total_params:,} parameters")
        
        return ArchitectureVisualization(
            architecture_type='LSTM',
            layers=layers,
            total_parameters=total_params,
            model_depth=len(layers),
            max_layer_width=max_width,
            max_layer_height=max_height,
            performance_score=performance_score,
            health_score=health_score
        )
    
    def _prepare_generic_visualization(self, 
                                     architecture: Dict[str, Any], 
                                     performance_score: float,
                                     health_score: Optional[float]) -> ArchitectureVisualization:
        """Fallback visualization for unknown architecture types"""
        
        logger.warning("Using generic visualization for unknown architecture type")
        
        # Create a simple generic representation
        layers = [
            LayerVisualization(
                layer_id='generic_input',
                layer_type='input',
                position_z=0,
                width=2.0,
                height=2.0,
                depth=0.5,
                parameters=0,
                color_intensity=0.6,
                opacity=0.7
            ),
            LayerVisualization(
                layer_id='generic_hidden',
                layer_type='dense',
                position_z=2.0,
                width=4.0,
                height=2.0,
                depth=1.0,
                parameters=1000,  # Estimate
                color_intensity=0.8,
                opacity=0.8
            ),
            LayerVisualization(
                layer_id='generic_output',
                layer_type='dense',
                position_z=4.0,
                width=1.0,
                height=1.0,
                depth=1.0,
                parameters=100,  # Estimate
                activation='softmax',
                color_intensity=0.9,
                opacity=0.9
            )
        ]
        
        return ArchitectureVisualization(
            architecture_type='Generic',
            layers=layers,
            total_parameters=1100,
            model_depth=3,
            max_layer_width=4.0,
            max_layer_height=2.0,
            performance_score=performance_score,
            health_score=health_score
        )
    
    def get_performance_color_scheme(self, performance_score: float, health_score: Optional[float] = None) -> Dict[str, str]:
        """
        Get color scheme based on performance and health scores
        
        Args:
            performance_score: Performance score (0.0-1.0)
            health_score: Optional health score (0.0-1.0)
            
        Returns:
            Dictionary with color information for 3D rendering
        """
        # Base color based on performance
        if performance_score >= 0.9:
            base_color = "#10b981"  # green-500
        elif performance_score >= 0.8:
            base_color = "#f59e0b"  # amber-500
        elif performance_score >= 0.7:
            base_color = "#ef4444"  # red-500
        else:
            base_color = "#6b7280"  # gray-500
        
        # Modify based on health score if available
        if health_score is not None:
            if health_score >= 0.8:
                accent_color = "#059669"  # green-600 (healthy)
            elif health_score >= 0.6:
                accent_color = "#d97706"  # amber-600 (moderate)
            else:
                accent_color = "#dc2626"  # red-600 (unhealthy)
        else:
            accent_color = base_color
        
        logger.debug(f"Generated color scheme - Performance: {performance_score:.2f} -> {base_color}, Health: {health_score} -> {accent_color}")
        
        return {
            'primary': base_color,
            'accent': accent_color,
            'opacity': min(0.6 + performance_score * 0.4, 1.0)  # Higher performance = less transparent
        }

# Factory function for easy import
def create_model_visualizer() -> ModelVisualizer:
    """Factory function to create a ModelVisualizer instance"""
    return ModelVisualizer()

# Example usage and testing
if __name__ == "__main__":
    # Test CNN architecture
    cnn_arch = {
        'type': 'CNN',
        'conv_layers': 2,
        'filters_per_layer': 32,
        'kernel_size': [3, 3],
        'dense_layers': 1,
        'first_dense_nodes': 128,
        'use_global_pooling': True,
        'activation': 'relu'
    }
    
    # Test LSTM architecture
    lstm_arch = {
        'type': 'LSTM',
        'lstm_units': 128,
        'dense_layers': 2,
        'first_dense_nodes': 64,
        'activation': 'relu'
    }
    
    visualizer = create_model_visualizer()
    
    print("Testing CNN visualization:")
    cnn_viz = visualizer.prepare_visualization_data(cnn_arch, 0.85, 0.75)
    print(f"  Layers: {len(cnn_viz.layers)}")
    print(f"  Total parameters: {cnn_viz.total_parameters:,}")
    print(f"  Max width: {cnn_viz.max_layer_width:.2f}")
    
    print("\nTesting LSTM visualization:")
    lstm_viz = visualizer.prepare_visualization_data(lstm_arch, 0.92, 0.80)
    print(f"  Layers: {len(lstm_viz.layers)}")
    print(f"  Total parameters: {lstm_viz.total_parameters:,}")
    print(f"  Max width: {lstm_viz.max_layer_width:.2f}")
    
    print("\nColor scheme example:")
    colors = visualizer.get_performance_color_scheme(0.85, 0.75)
    print(f"  Primary: {colors['primary']}")
    print(f"  Accent: {colors['accent']}")
    print(f"  Opacity: {colors['opacity']:.2f}")