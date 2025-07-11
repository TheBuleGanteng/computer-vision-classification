"""
Activation Map Analysis Module

This module provides comprehensive analysis of CNN activation maps (feature maps) including:
- Layer-by-layer activation visualization for sample inputs
- Individual filter response analysis across convolutional layers
- Feature hierarchy progression from low-level to high-level features
- Dead filter detection and activation statistics
- Professional-quality visualizations for presentations and debugging
- Sample selection strategies for representative analysis

Designed to work with any Keras CNN model architecture and provides insights into
what different layers and filters have learned during training.

FUNDAMENTAL CONCEPTS:

What Are Activation Maps?
    Activation maps (also called feature maps) are visual representations of how
    individual filters in convolutional layers respond to input images. They show
    which parts of an input image activate each filter most strongly.

    Think of each filter as a specialized "pattern detector":
    - Filter 1: "I detect horizontal edges"
    - Filter 5: "I detect circular shapes"  
    - Filter 12: "I detect red color patches"
    - Filter 23: "I detect vertical edges"

How Activation Maps Work:
    For each filter in each convolutional layer:
    1. Filter scans the input image with a sliding window (e.g., 3x3 kernel)
    2. Pattern matching occurs - filter responds strongly where its pattern appears
    3. Activation values generated - high values = strong pattern match
    4. Feature map created - 2D map showing activation strength at each location

Visual Interpretation:
    - Bright/Hot areas: Strong activation = "This filter found its pattern here!"
    - Dark/Cold areas: Weak activation = "This filter's pattern is not present here"
    - Spatial preservation: Maps retain spatial relationships from original image

LAYER HIERARCHY PROGRESSION:

Layer 1 (Early Features):
    - Edge Detectors: Show where horizontal/vertical/diagonal lines appear
    - Color Detectors: Highlight regions of specific colors (red, blue, etc.)
    - Texture Detectors: Respond to rough/smooth/patterned surfaces

Layer 2+ (Mid-Level Features):
    - Shape Combiners: Detect circles, rectangles, triangles formed from edges
    - Pattern Detectors: Find repeated elements or specific textures
    - Object Parts: Recognize components like "wheel-like shapes" or "text regions"

Layer N (High-Level Features):
    - Object Detectors: Recognize "stop sign-like shapes" or "vehicle outlines"
    - Context Combiners: Understand spatial relationships between parts
    - Classification Features: Patterns specific to final decision making

QUANTITATIVE HEALTH INDICATORS:

Healthy Activation Patterns:
    - Diverse activation ranges across filters (not all active or all dead)
    - Clear spatial localization for relevant features
    - Progressive abstraction through layers
    - Reasonable activation magnitudes (not exploded or vanished)

Problem Patterns:
    - Dead Filters: No significant activation (max < 0.1) - filter learned nothing
    - Saturated Filters: Always highly active (mean > 0.8) - filter too generic
    - Sparse Activation: Very few filters active - model underutilizing capacity
    - Uniform Activation: All filters respond similarly - lack of specialization

PRACTICAL APPLICATIONS:

Model Debugging:
    - Identify layers that aren't learning useful features
    - Detect filters that have "died" during training
    - Verify model is learning expected feature hierarchy
    - Find filters that are too generic or too specific

Feature Understanding:
    - Visualize what patterns each layer learns
    - Understand feature evolution through network depth
    - Discover unexpected patterns the model has learned
    - Validate feature learning matches domain expectations

Educational/Research:
    - Demonstrate CNN hierarchical learning concepts
    - Show how simple features combine into complex ones
    - Create compelling visualizations for presentations
    - Compare feature learning across different architectures
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import tensorflow as tf
from tensorflow import keras  # type: ignore
import traceback
from typing import Dict, Any, List, Tuple, Optional, Union, NamedTuple
from utils.logger import logger
from dataclasses import dataclass


# Note: All activation map configuration parameters are now centralized in ModelConfig class
# This follows the established pattern for gradient flow, weights/bias, and other analysis configurations


class LayerActivationStats(NamedTuple):
    """Statistics for activations in a single layer"""
    layer_name: str
    layer_index: int
    num_filters: int
    dead_filters: int
    saturated_filters: int
    mean_activation: float
    std_activation: float
    max_activation: float
    min_activation: float
    activation_range: float
    sparsity_ratio: float  # Fraction of activations near zero


class ActivationMapAnalyzer:
    """
    Comprehensive activation map analysis for CNN models
    
    Provides detailed analysis of how convolutional filters respond to input images,
    including layer-by-layer progression, filter health assessment, and feature
    hierarchy visualization.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize activation map analyzer
        
        Args:
            model_name: Name of the model for documentation and file naming
        
        Note: Configuration parameters are accessed from ModelConfig via ModelBuilder,
              following the established pattern used by other analyzers in the project.
        """
        self.model_name = model_name
        self.conv_layers: List[Tuple[int, keras.layers.Layer]] = []  # List of (index, layer) tuples
        self.activation_models: Dict[str, keras.Model] = {}  # Models for extracting activations
        
        logger.debug(f"running ActivationMapAnalyzer.__init__ ... Initialized for model: {model_name}")
    
    
    def analyze_and_visualize(
        self,
        model: keras.Model,
        sample_images: np.ndarray,
        sample_labels: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        dataset_name: str = "dataset",
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None,
        model_config = None  # ModelConfig instance for accessing configuration parameters
    ) -> Dict[str, Any]:
        """
        Perform comprehensive activation map analysis and create visualizations
        
        Args:
            model: Trained Keras model to analyze
            sample_images: Input images for activation analysis (batch_size, height, width, channels)
            sample_labels: Optional labels for sample images (for class-based selection)
            class_names: Optional class names for labeling
            dataset_name: Name of dataset for documentation
            run_timestamp: Timestamp for file naming
            plot_dir: Directory to save plots
            model_config: ModelConfig instance containing activation map configuration parameters
            
        Returns:
            Dictionary containing analysis results and file paths
        """
        logger.debug("running analyze_and_visualize ... Starting activation map analysis...")
        
        # Validate model_config is provided
        if model_config is None:
            return {'error': 'ModelConfig is required for activation map analysis'}
        
        try:
            # Setup analysis
            analysis_results = {
                'layer_stats': [],
                'activation_insights': [],
                'recommendations': [],
                'visualization_paths': [],
                'sample_info': {},
                'filter_health': {}
            }
            
            # Discover and setup convolutional layers
            self._discover_conv_layers(model, model_config)
            if not self.conv_layers:
                return {'error': 'No convolutional layers found in model'}
            
            # Select representative samples for analysis
            selected_samples, selected_labels, sample_info = self._select_samples(
                sample_images, sample_labels, class_names, model_config
            )
            analysis_results['sample_info'] = sample_info
            
            # Create activation extraction models
            self._create_activation_models(model)
            
            # Extract activations for all selected samples and layers
            layer_activations = self._extract_all_activations(selected_samples)
            
            # Analyze each layer
            for layer_idx, layer in self.conv_layers:
                layer_name = layer.name
                logger.debug(f"running analyze_and_visualize ... Analyzing layer {layer_idx}: {layer_name}")
                
                # Get activations for this layer
                if layer_name not in layer_activations:
                    logger.warning(f"running analyze_and_visualize ... No activations found for layer {layer_name}")
                    continue
                
                activations = layer_activations[layer_name]
                
                # Compute layer statistics
                layer_stats = self._compute_layer_statistics(layer_name, layer_idx, activations, model_config)
                analysis_results['layer_stats'].append(layer_stats)
                
                # Create visualizations for this layer
                if plot_dir:
                    viz_paths = self._visualize_layer_activations(
                        layer_name, layer_idx, activations, selected_samples,
                        plot_dir, run_timestamp or "analysis", model_config
                    )
                    analysis_results['visualization_paths'].extend(viz_paths)
            
            # Generate comprehensive insights and recommendations
            self._generate_insights_and_recommendations(analysis_results)
            
            # Create overview visualizations
            if plot_dir and model_config.show_activation_maps:
                overview_paths = self._create_overview_visualizations(
                    analysis_results, layer_activations, selected_samples,
                    plot_dir, run_timestamp or "analysis", model_config
                )
                analysis_results['visualization_paths'].extend(overview_paths)
            
            logger.debug("running analyze_and_visualize ... Activation map analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            error_msg = f"Activation map analysis failed: {str(e)}"
            logger.error(f"running analyze_and_visualize ... {error_msg}")
            logger.debug(f"running analyze_and_visualize ... Error traceback: {traceback.format_exc()}")
            return {'error': error_msg}
    
    
    def _discover_conv_layers(self, model: keras.Model, model_config) -> None:
        """
        Discover all convolutional layers in the model
        
        Args:
            model: Keras model to analyze
            model_config: ModelConfig instance containing layer selection parameters
        """
        logger.debug("running _discover_conv_layers ... Discovering convolutional layers...")
        
        self.conv_layers = []
        layer_count = 0
        
        for i, layer in enumerate(model.layers):
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D, keras.layers.DepthwiseConv2D)):
                # Apply layer frequency filter
                if layer_count % model_config.activation_layer_frequency == 0:
                    self.conv_layers.append((i, layer))
                    logger.debug(f"running _discover_conv_layers ... Found conv layer {i}: {layer.name} (filters: {layer.filters})")
                layer_count += 1
                
                # Respect maximum layer limit
                if len(self.conv_layers) >= model_config.activation_max_layers_to_analyze:
                    logger.debug(f"running _discover_conv_layers ... Reached maximum layer limit ({model_config.activation_max_layers_to_analyze})")
                    break
        
        logger.debug(f"running _discover_conv_layers ... Discovered {len(self.conv_layers)} convolutional layers for analysis")
    
    
    def _select_samples(
    self,
    sample_images: np.ndarray,
    sample_labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    model_config = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Select representative samples for activation analysis
        
        Args:
            sample_images: All available sample images
            sample_labels: Optional labels for samples
            class_names: Optional class names
            model_config: ModelConfig instance containing sample selection parameters
            
        Returns:
            Tuple of (selected_images, selected_labels, sample_info)
        """
        # Add validation for model_config
        if model_config is None:
            logger.error("running _select_samples ... ModelConfig is required but was None")
            # Provide fallback behavior
            max_samples = min(10, len(sample_images))  # Default fallback
            indices = np.random.choice(len(sample_images), size=max_samples, replace=False)
            selected_images = sample_images[indices]
            selected_labels = sample_labels[indices] if sample_labels is not None else None
            sample_info = {
                'total_available': len(sample_images),
                'selected_count': len(indices),
                'selection_strategy': 'fallback_random',
                'selection_details': {'indices': indices.tolist()}
            }
            return selected_images, selected_labels, sample_info
        
        logger.debug(f"running _select_samples ... Selecting samples using strategy: {model_config.activation_sample_selection_strategy}")
        
        # Initialize missing variables
        total_samples = len(sample_images)
        max_samples = min(model_config.activation_max_total_samples, total_samples)
        
        sample_info = {
            'total_available': total_samples,
            'selected_count': 0,
            'selection_strategy': model_config.activation_sample_selection_strategy,
            'selection_details': {}
        }
        
        
        if model_config.activation_sample_selection_strategy == "random":
            # Simple random selection
            indices = np.random.choice(total_samples, size=max_samples, replace=False)
            selected_images = sample_images[indices]
            selected_labels = sample_labels[indices] if sample_labels is not None else None
            sample_info['selected_count'] = len(indices)
            sample_info['selection_details'] = {'indices': indices.tolist()}
            
        elif model_config.activation_sample_selection_strategy == "representative" and sample_labels is not None:
            # Select representative samples from each class
            unique_classes = np.unique(sample_labels)
            samples_per_class = min(
                model_config.activation_num_samples_per_class,
                max_samples // len(unique_classes)
            )
            
            selected_indices = []
            class_details = {}
            
            for class_label in unique_classes:
                class_indices = np.where(sample_labels == class_label)[0]
                if len(class_indices) > 0:
                    # Select first few samples from this class (could be made more sophisticated)
                    selected_from_class = np.random.choice(
                        class_indices, 
                        size=min(samples_per_class, len(class_indices)), 
                        replace=False
                    )
                    selected_indices.extend(selected_from_class)
                    
                    class_name = class_names[class_label] if class_names and class_label < len(class_names) else f"Class_{class_label}"
                    class_details[class_name] = {
                        'class_label': int(class_label),
                        'samples_selected': len(selected_from_class),
                        'indices': selected_from_class.tolist()
                    }
            
            selected_indices = np.array(selected_indices[:max_samples])
            selected_images = sample_images[selected_indices]
            selected_labels = sample_labels[selected_indices]
            sample_info['selected_count'] = len(selected_indices)
            sample_info['selection_details'] = class_details
            
        else:
            # Mixed or fallback strategy - combination of approaches
            # For now, implement as random selection with some deterministic elements
            if total_samples <= max_samples:
                selected_images = sample_images
                selected_labels = sample_labels
                indices = list(range(total_samples))
            else:
                # Select first sample, last sample, and random samples in between
                fixed_indices = [0, total_samples - 1]  # First and last
                remaining_slots = max_samples - 2
                
                if remaining_slots > 0:
                    available_indices = list(range(1, total_samples - 1))
                    random_indices = np.random.choice(
                        available_indices, 
                        size=min(remaining_slots, len(available_indices)), 
                        replace=False
                    )
                    indices = fixed_indices + random_indices.tolist()
                else:
                    indices = fixed_indices
                
                indices = sorted(indices)
                selected_images = sample_images[indices]
                selected_labels = sample_labels[indices] if sample_labels is not None else None
            
            sample_info['selected_count'] = len(indices)
            sample_info['selection_details'] = {'indices': indices}
        
        logger.debug(f"running _select_samples ... Selected {sample_info['selected_count']} samples from {total_samples} available")
        return selected_images, selected_labels, sample_info
    
    
    def _create_activation_models(self, model: keras.Model) -> None:
        """
        Create models for extracting activations from each convolutional layer
        
        Args:
            model: Original trained model
        """
        logger.debug("running _create_activation_models ... Creating activation extraction models...")
        
        # Store the original model reference
        self.original_model = model
        
        # Validate that we have convolutional layers to analyze
        if not self.conv_layers:
            logger.warning("running _create_activation_models ... No convolutional layers found for activation extraction")
            return
        
        # Test that the model is functional and build it if necessary
        try:
            # Get input shape from model
            if hasattr(model, 'input_shape') and model.input_shape is not None:
                input_shape = model.input_shape
                logger.debug(f"running _create_activation_models ... Got input shape: {input_shape}")
                
                # CRITICAL: Ensure the model is built by calling it
                # This is essential for Sequential models
                logger.debug("running _create_activation_models ... Testing model with dummy input...")
                dummy_input = tf.zeros((1,) + input_shape[1:])
                test_output = model(dummy_input, training=False)
                logger.debug(f"running _create_activation_models ... Model test successful, output shape: {test_output.shape}")
                
                logger.debug(f"running _create_activation_models ... Setup complete for {len(self.conv_layers)} convolutional layers")
                
            else:
                logger.error("running _create_activation_models ... Could not get input shape from model")
                return
                
        except Exception as e:
            logger.error(f"running _create_activation_models ... Failed to set up activation extraction: {e}")
            logger.debug(f"running _create_activation_models ... Error traceback: {traceback.format_exc()}")
            return
    
    
    def _extract_all_activations(self, sample_images: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract activations from all layers for all sample images
        ROBUST SOLUTION using Keras model reconstruction
        
        Args:
            sample_images: Input images for activation extraction
            
        Returns:
            Dictionary mapping layer names to activation arrays
        """
        logger.debug("running _extract_all_activations ... Extracting activations for all layers...")
        
        layer_activations = {}
        
        try:
            # Ensure model is built by calling it
            logger.debug("running _extract_all_activations ... Building model by calling with sample data...")
            sample_input = sample_images[:1]
            _ = self.original_model(sample_input, training=False)
            logger.debug("running _extract_all_activations ... Model successfully built, proceeding with activation extraction...")
            
            # Get input shape for reconstructing models
            input_shape = self.original_model.input_shape[1:]  # Remove batch dimension
            
            # Extract activations for each target layer
            for layer_idx, target_layer in self.conv_layers:
                layer_name = target_layer.name
                
                try:
                    logger.debug(f"running _extract_all_activations ... Creating intermediate model for layer {layer_name} (index {layer_idx})")
                    
                    # SOLUTION: Reconstruct a new Functional API model up to the target layer
                    # This avoids Sequential model limitations
                    
                    # Create input tensor
                    inputs = keras.Input(shape=input_shape)
                    x = inputs
                    
                    # Forward through layers up to and including target layer
                    for i in range(layer_idx + 1):
                        current_layer = self.original_model.layers[i]
                        
                        # Create a new layer with same config and weights
                        layer_config = current_layer.get_config()
                        new_layer = current_layer.__class__.from_config(layer_config)
                        
                        # Apply the layer
                        x = new_layer(x)
                        
                        # Copy weights after layer is built
                        if current_layer.weights:
                            new_layer.set_weights(current_layer.get_weights())
                    
                    # Create intermediate model
                    intermediate_model = keras.Model(inputs=inputs, outputs=x)
                    
                    # Extract activations for this layer
                    logger.debug(f"running _extract_all_activations ... Extracting activations for layer {layer_name}")
                    activations = intermediate_model.predict(sample_images, verbose=0)
                    layer_activations[layer_name] = activations
                    
                    activation_shape = activations.shape
                    logger.debug(f"running _extract_all_activations ... Layer {layer_name}: activations shape {activation_shape}")
                    
                except Exception as e:
                    logger.warning(f"running _extract_all_activations ... Failed to extract activations for layer {layer_name}: {e}")
                    logger.debug(f"running _extract_all_activations ... Layer extraction error traceback: {traceback.format_exc()}")
                    continue
            
            logger.debug(f"running _extract_all_activations ... Extracted activations for {len(layer_activations)} layers")
            return layer_activations
            
        except Exception as e:
            logger.error(f"running _extract_all_activations ... Overall extraction failed: {e}")
            logger.debug(f"running _extract_all_activations ... Error traceback: {traceback.format_exc()}")
            return {}
    
    
    def _compute_layer_statistics(
        self, 
        layer_name: str, 
        layer_idx: int, 
        activations: np.ndarray, 
        model_config
    ) -> LayerActivationStats:
        """
        Compute comprehensive statistics for activations in a single layer
        
        Args:
            layer_name: Name of the layer
            layer_idx: Index of the layer in the model
            activations: Activation array (batch_size, height, width, num_filters)
            model_config: ModelConfig instance for threshold parameters
            
        Returns:
            LayerActivationStats object with computed statistics
        """
        logger.debug(f"running _compute_layer_statistics ... Computing statistics for layer {layer_name}")
        
        # Get dimensions
        if len(activations.shape) == 4:  # Standard conv layer: (batch, height, width, filters)
            batch_size, height, width, num_filters = activations.shape
        else:
            logger.warning(f"running _compute_layer_statistics ... Unexpected activation shape for {layer_name}: {activations.shape}")
            # Handle edge cases
            if len(activations.shape) == 2:  # Flattened: (batch, features)
                batch_size, features = activations.shape
                num_filters = features
                height = width = 1
            else:
                # Fallback
                batch_size = activations.shape[0]
                num_filters = activations.shape[-1] if len(activations.shape) > 1 else 1
                height = width = 1
        
        # Compute filter-wise statistics
        dead_filters = 0
        saturated_filters = 0
        filter_max_activations = []
        filter_mean_activations = []
        
        for filter_idx in range(num_filters):
            if len(activations.shape) == 4:
                filter_activations = activations[:, :, :, filter_idx]
            else:
                filter_activations = activations[:, filter_idx] if len(activations.shape) > 1 else activations
            
            filter_max = np.max(filter_activations)
            filter_mean = np.mean(filter_activations)
            
            filter_max_activations.append(filter_max)
            filter_mean_activations.append(filter_mean)
            
            # Check for dead filters (no significant activation)
            if filter_max < model_config.activation_dead_filter_threshold:
                dead_filters += 1
            
            # Check for saturated filters (always highly active)
            if filter_mean > model_config.activation_saturated_filter_threshold:
                saturated_filters += 1
        
        # Overall statistics
        mean_activation = np.mean(activations)
        std_activation = np.std(activations)
        max_activation = np.max(activations)
        min_activation = np.min(activations)
        activation_range = max_activation - min_activation
        
        # Compute sparsity (fraction of activations near zero)
        sparsity_threshold = 0.01 * max_activation if max_activation > 0 else 0.01
        near_zero = np.sum(np.abs(activations) < sparsity_threshold)
        total_activations = activations.size
        sparsity_ratio = near_zero / total_activations
        
        stats = LayerActivationStats(
            layer_name=layer_name,
            layer_index=layer_idx,
            num_filters=num_filters,
            dead_filters=dead_filters,
            saturated_filters=saturated_filters,
            mean_activation=float(mean_activation),
            std_activation=float(std_activation),
            max_activation=float(max_activation),
            min_activation=float(min_activation),
            activation_range=float(activation_range),
            sparsity_ratio=float(sparsity_ratio)
        )
        
        logger.debug(f"running _compute_layer_statistics ... Layer {layer_name}: "
                    f"{dead_filters}/{num_filters} dead filters, "
                    f"{saturated_filters}/{num_filters} saturated filters, "
                    f"sparsity: {sparsity_ratio:.3f}")
        
        return stats
    
    
    def _visualize_layer_activations(
        self,
        layer_name: str,
        layer_idx: int,
        activations: np.ndarray,
        sample_images: np.ndarray,
        plot_dir: Path,
        run_timestamp: str,
        model_config
    ) -> List[str]:
        """
        Create visualizations for a single layer's activations
        
        Args:
            layer_name: Name of the layer
            layer_idx: Index of the layer
            activations: Activation array for this layer
            sample_images: Original input images
            plot_dir: Directory to save plots
            run_timestamp: Timestamp for file naming
            model_config: ModelConfig instance for visualization parameters
            
        Returns:
            List of file paths for created visualizations
        """
        logger.debug(f"running _visualize_layer_activations ... Creating visualizations for layer {layer_name}")
        
        visualization_paths = []
        
        try:
            # Ensure we have 4D activations (batch, height, width, filters)
            if len(activations.shape) != 4:
                logger.warning(f"running _visualize_layer_activations ... Skipping visualization for layer {layer_name} (shape: {activations.shape})")
                return visualization_paths
            
            batch_size, height, width, num_filters = activations.shape
            
            # Limit number of filters to visualize for performance
            max_filters = min(num_filters, model_config.activation_max_filters_per_layer)
            
            # Create filter grid visualization for first sample
            if batch_size > 0:
                fig_path = self._create_filter_grid_visualization(
                    layer_name, layer_idx, activations[0], sample_images[0],
                    plot_dir, run_timestamp, model_config, max_filters
                )
                if fig_path:
                    visualization_paths.append(fig_path)
            
            # Create sample comparison visualization if multiple samples
            if batch_size > 1:
                comparison_path = self._create_sample_comparison_visualization(
                    layer_name, layer_idx, activations, sample_images,
                    plot_dir, run_timestamp, model_config
                )
                if comparison_path:
                    visualization_paths.append(comparison_path)
        
        except Exception as e:
            logger.warning(f"running _visualize_layer_activations ... Failed to create visualizations for layer {layer_name}: {e}")
            logger.debug(f"running _visualize_layer_activations ... Error traceback: {traceback.format_exc()}")
        
        return visualization_paths
    
    
    def _create_filter_grid_visualization(
        self,
        layer_name: str,
        layer_idx: int,
        single_activation: np.ndarray,
        original_image: np.ndarray,
        plot_dir: Path,
        run_timestamp: str,
        model_config,
        max_filters: int
    ) -> Optional[str]:
        """
        Create a grid visualization showing all filters for a single input
        
        Args:
            layer_name: Name of the layer
            layer_idx: Index of the layer
            single_activation: Activations for single input (height, width, filters)
            original_image: Original input image
            plot_dir: Directory to save plot
            run_timestamp: Timestamp for file naming
            model_config: ModelConfig instance for visualization parameters
            max_filters: Maximum number of filters to show
            
        Returns:
            Path to saved visualization file, or None if failed
        """
        try:
            logger.debug(f"running _create_filter_grid_visualization ... Creating filter grid for layer {layer_name}")
            
            height, width, num_filters = single_activation.shape
            filters_to_show = min(max_filters, num_filters)
            
            # Calculate grid dimensions
            filters_per_row = model_config.activation_filters_per_row
            num_rows = (filters_to_show + filters_per_row - 1) // filters_per_row
            
            # Create figure
            fig, axes = plt.subplots(
                num_rows + 1, filters_per_row, 
                figsize=model_config.activation_figsize_individual
            )
            
            # Handle different axes configurations to ensure consistent access pattern
            # Convert to a function that always allows [row][col] indexing
            def get_axis(row: int, col: int):
                """Helper function to safely access axes regardless of subplot configuration"""
                if num_rows + 1 == 1 and filters_per_row == 1:
                    # Single subplot case - axes is just one Axes object
                    return axes
                elif num_rows + 1 == 1:
                    # Single row case - axes is 1D array of Axes
                    return axes[col]  # type: ignore
                elif filters_per_row == 1:
                    # Single column case - axes is 1D array of Axes  
                    return axes[row]  # type: ignore
                else:
                    # Standard 2D case - axes is 2D array of Axes
                    return axes[row, col]  # type: ignore
            
            # Show original image in first row, first position
            ax = get_axis(0, 0)
            if original_image.shape[-1] == 3:  # RGB
                ax.imshow(original_image)
            else:  # Grayscale
                ax.imshow(original_image.squeeze(), cmap=model_config.activation_cmap_original)
            ax.set_title('Original Image')
            ax.axis('off')
            
            # Hide unused positions in first row
            for col in range(1, filters_per_row):
                ax = get_axis(0, col)
                ax.axis('off')
            
            # Show filter activations
            for filter_idx in range(filters_to_show):
                row = (filter_idx // filters_per_row) + 1
                col = filter_idx % filters_per_row
                
                if row < num_rows + 1 and col < filters_per_row:
                    ax = get_axis(row, col)
                    activation_map = single_activation[:, :, filter_idx]
                    
                    im = ax.imshow(
                        activation_map, 
                        cmap=model_config.activation_cmap,
                        interpolation='nearest'
                    )
                    ax.set_title(f'Filter {filter_idx}')
                    ax.axis('off')
                    
                    # Add colorbar for this filter
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Hide any remaining unused subplots
            for row in range(num_rows + 1):
                for col in range(filters_per_row):
                    if row == 0 and col > 0:  # First row except first column (already handled)
                        continue
                    elif row > 0:  # Filter rows
                        filter_idx = (row - 1) * filters_per_row + col
                        if filter_idx >= filters_to_show:
                            ax = get_axis(row, col)
                            ax.axis('off')
            
            # Set overall title
            plt.suptitle(f'Activation Maps - {layer_name} (Layer {layer_idx})', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            filename = f"activation_maps_{run_timestamp}_{layer_name}_layer_{layer_idx:02d}.png"
            filepath = plot_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"running _create_filter_grid_visualization ... Saved filter grid to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.warning(f"running _create_filter_grid_visualization ... Failed to create filter grid for {layer_name}: {e}")
            plt.close()
            return None
    
    def _create_sample_comparison_visualization(
        self,
        layer_name: str,
        layer_idx: int,
        activations: np.ndarray,
        sample_images: np.ndarray,
        plot_dir: Path,
        run_timestamp: str,
        model_config
    ) -> Optional[str]:
        """
        Create a visualization comparing how different samples activate the same filters
        
        Args:
            layer_name: Name of the layer
            layer_idx: Index of the layer
            activations: All activations for this layer (batch, height, width, filters)
            sample_images: Original input images
            plot_dir: Directory to save plot
            run_timestamp: Timestamp for file naming
            model_config: ModelConfig instance for visualization parameters
            
        Returns:
            Path to saved visualization file, or None if failed
        """
        try:
            batch_size, height, width, num_filters = activations.shape
            
            # Select a few interesting filters (highest variance across samples)
            filter_variances = []
            for filter_idx in range(num_filters):
                filter_activations_flat = activations[:, :, :, filter_idx].reshape(batch_size, -1)
                filter_means = np.mean(filter_activations_flat, axis=1)
                variance = np.var(filter_means)
                filter_variances.append(variance)
            
            # Get top filters with highest variance
            top_filter_indices = np.argsort(filter_variances)[-4:][::-1]  # Top 4 filters
            
            # Determine subplot dimensions
            num_filter_rows = len(top_filter_indices)
            num_sample_cols = min(batch_size, 4)
            
            # Create comparison plot
            fig, axes = plt.subplots(
                num_filter_rows, num_sample_cols, 
                figsize=(16, 4 * num_filter_rows)
            )
            
            # Handle different axes configurations to ensure 2D indexing always works
            if num_filter_rows == 1 and num_sample_cols == 1:
                # Single subplot case
                axes = np.array([[axes]])
            elif num_filter_rows == 1:
                # Single row, multiple columns
                axes = axes.reshape(1, -1)
            elif num_sample_cols == 1:
                # Multiple rows, single column
                axes = axes.reshape(-1, 1)
            # If both > 1, axes is already properly shaped as 2D
            
            for filter_row, filter_idx in enumerate(top_filter_indices):
                for sample_col in range(num_sample_cols):
                    activation_map = activations[sample_col, :, :, filter_idx]
                    
                    im = axes[filter_row, sample_col].imshow(
                        activation_map,
                        cmap=model_config.activation_cmap,
                        interpolation='nearest'
                    )
                    
                    if filter_row == 0:
                        axes[filter_row, sample_col].set_title(f'Sample {sample_col + 1}')
                    
                    if sample_col == 0:
                        axes[filter_row, sample_col].set_ylabel(f'Filter {filter_idx}')
                    
                    axes[filter_row, sample_col].axis('off')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=axes[filter_row, sample_col], fraction=0.046, pad=0.04)
            
            plt.suptitle(f'Sample Comparison - {layer_name} (Layer {layer_idx})', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            filename = f"activation_comparison_{run_timestamp}_{layer_name}_layer_{layer_idx:02d}.png"
            filepath = plot_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"running _create_sample_comparison_visualization ... Saved sample comparison to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.warning(f"running _create_sample_comparison_visualization ... Failed to create sample comparison for {layer_name}: {e}")
            plt.close()
            return None
    
    
    def _generate_insights_and_recommendations(self, analysis_results: Dict[str, Any]) -> None:
        """
        Generate insights and recommendations based on activation analysis
        
        Args:
            analysis_results: Dictionary containing analysis results to update
        """
        logger.debug("running _generate_insights_and_recommendations ... Generating insights and recommendations...")
        
        insights = []
        recommendations = []
        layer_stats = analysis_results['layer_stats']
        
        if not layer_stats:
            insights.append("No layer statistics available for analysis")
            return
        
        # Analyze dead filters across layers
        total_filters = sum(stats.num_filters for stats in layer_stats)
        total_dead_filters = sum(stats.dead_filters for stats in layer_stats)
        dead_filter_ratio = total_dead_filters / total_filters if total_filters > 0 else 0
        
        if dead_filter_ratio > 0.3:
            insights.append(f"High proportion of dead filters detected: {dead_filter_ratio:.1%} ({total_dead_filters}/{total_filters})")
            recommendations.append("Consider reducing learning rate or adjusting initialization to prevent filter death")
        elif dead_filter_ratio > 0.1:
            insights.append(f"Moderate dead filter presence: {dead_filter_ratio:.1%} ({total_dead_filters}/{total_filters})")
            recommendations.append("Monitor dead filters and consider regularization adjustments")
        else:
            insights.append(f"Healthy filter utilization: only {dead_filter_ratio:.1%} dead filters")
        
        # Analyze saturation across layers
        total_saturated_filters = sum(stats.saturated_filters for stats in layer_stats)
        saturation_ratio = total_saturated_filters / total_filters if total_filters > 0 else 0
        
        if saturation_ratio > 0.2:
            insights.append(f"High filter saturation detected: {saturation_ratio:.1%} ({total_saturated_filters}/{total_filters})")
            recommendations.append("Consider adding regularization or adjusting activation functions to reduce saturation")
        
        # Analyze sparsity patterns
        sparsity_values = [stats.sparsity_ratio for stats in layer_stats]
        avg_sparsity = np.mean(sparsity_values)
        
        if avg_sparsity > 0.8:
            insights.append(f"Very sparse activations detected: {avg_sparsity:.1%} average sparsity")
            recommendations.append("High sparsity may indicate underutilized network capacity")
        elif avg_sparsity < 0.3:
            insights.append(f"Dense activations detected: {avg_sparsity:.1%} average sparsity")
            recommendations.append("Consider adding dropout or other regularization to prevent overfitting")
        
        # Analyze layer progression
        if len(layer_stats) > 1:
            # Check if activation ranges decrease appropriately through layers
            activation_ranges = [stats.activation_range for stats in layer_stats]
            if activation_ranges[0] > 0:
                range_trend = [(activation_ranges[i] / activation_ranges[0]) for i in range(len(activation_ranges))]
                
                # Look for unexpected patterns
                if any(range_trend[i] > range_trend[i-1] * 2 for i in range(1, len(range_trend))):
                    insights.append("Unexpected activation range increase detected between layers")
                    recommendations.append("Check for gradient explosion or inappropriate layer scaling")
        
        # Layer-specific insights
        for stats in layer_stats:
            if stats.dead_filters > stats.num_filters * 0.5:
                insights.append(f"Layer {stats.layer_name}: Majority of filters are dead ({stats.dead_filters}/{stats.num_filters})")
                recommendations.append(f"Layer {stats.layer_name}: Consider architectural changes or different initialization")
            
            if stats.max_activation > 100:
                insights.append(f"Layer {stats.layer_name}: Very high activation values detected (max: {stats.max_activation:.2f})")
                recommendations.append(f"Layer {stats.layer_name}: Check for gradient explosion or normalization issues")
        
        # Store results
        analysis_results['activation_insights'] = insights
        analysis_results['recommendations'] = recommendations
        
        # Compute overall health assessment
        if dead_filter_ratio < 0.1 and saturation_ratio < 0.1 and 0.3 <= avg_sparsity <= 0.8:
            health_status = "healthy"
        elif dead_filter_ratio < 0.3 and saturation_ratio < 0.2:
            health_status = "moderate"
        else:
            health_status = "concerning"
        
        analysis_results['filter_health'] = {
            'overall_status': health_status,
            'dead_filter_ratio': dead_filter_ratio,
            'saturation_ratio': saturation_ratio,
            'average_sparsity': avg_sparsity,
            'total_filters': total_filters
        }
        
        logger.debug(f"running _generate_insights_and_recommendations ... Generated {len(insights)} insights and {len(recommendations)} recommendations")
    
    
    def _create_overview_visualizations(
        self,
        analysis_results: Dict[str, Any],
        layer_activations: Dict[str, np.ndarray],
        sample_images: np.ndarray,
        plot_dir: Path,
        run_timestamp: str,
        model_config
    ) -> List[str]:
        """
        Create overview visualizations summarizing activation patterns across all layers
        
        Args:
            analysis_results: Dictionary containing analysis results
            layer_activations: Activations for all layers
            sample_images: Original input images
            plot_dir: Directory to save plots
            run_timestamp: Timestamp for file naming
            model_config: ModelConfig instance for visualization parameters
            
        Returns:
            List of file paths for created overview visualizations
        """
        logger.debug("running _create_overview_visualizations ... Creating overview visualizations...")
        
        overview_paths = []
        
        try:
            # Create layer statistics summary plot
            stats_path = self._create_layer_statistics_plot(
                analysis_results['layer_stats'], plot_dir, run_timestamp
            )
            if stats_path:
                overview_paths.append(stats_path)
            
            # Create activation progression plot
            progression_path = self._create_activation_progression_plot(
                analysis_results['layer_stats'], layer_activations, 
                plot_dir, run_timestamp, model_config
            )
            if progression_path:
                overview_paths.append(progression_path)
            
        except Exception as e:
            logger.warning(f"running _create_overview_visualizations ... Failed to create overview visualizations: {e}")
        
        return overview_paths
    
    
    def _create_layer_statistics_plot(
        self,
        layer_stats: List[LayerActivationStats],
        plot_dir: Path,
        run_timestamp: str
    ) -> Optional[str]:
        """
        Create a plot summarizing statistics across all layers
        
        Args:
            layer_stats: List of layer statistics
            plot_dir: Directory to save plot
            run_timestamp: Timestamp for file naming
            
        Returns:
            Path to saved plot file, or None if failed
        """
        try:
            if not layer_stats:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            layer_names = [stats.layer_name for stats in layer_stats]
            layer_indices = list(range(len(layer_stats)))
            
            # Plot 1: Dead and saturated filters
            dead_counts = [stats.dead_filters for stats in layer_stats]
            saturated_counts = [stats.saturated_filters for stats in layer_stats]
            total_filters = [stats.num_filters for stats in layer_stats]
            
            x = np.arange(len(layer_names))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, dead_counts, width, label='Dead Filters', color='red', alpha=0.7)
            axes[0, 0].bar(x + width/2, saturated_counts, width, label='Saturated Filters', color='orange', alpha=0.7)
            axes[0, 0].set_xlabel('Layer')
            axes[0, 0].set_ylabel('Number of Filters')
            axes[0, 0].set_title('Filter Health Across Layers')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Activation statistics
            mean_activations = [stats.mean_activation for stats in layer_stats]
            max_activations = [stats.max_activation for stats in layer_stats]
            
            axes[0, 1].plot(layer_indices, mean_activations, 'b-o', label='Mean Activation', linewidth=2)
            axes[0, 1].plot(layer_indices, max_activations, 'r-s', label='Max Activation', linewidth=2)
            axes[0, 1].set_xlabel('Layer Index')
            axes[0, 1].set_ylabel('Activation Value')
            axes[0, 1].set_title('Activation Magnitudes Across Layers')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Sparsity ratios
            sparsity_ratios = [stats.sparsity_ratio for stats in layer_stats]
            
            axes[1, 0].bar(layer_indices, sparsity_ratios, color='green', alpha=0.7)
            axes[1, 0].set_xlabel('Layer Index')
            axes[1, 0].set_ylabel('Sparsity Ratio')
            axes[1, 0].set_title('Activation Sparsity Across Layers')
            axes[1, 0].set_xticks(layer_indices)
            axes[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Filter utilization summary
            utilization_ratios = [(stats.num_filters - stats.dead_filters) / stats.num_filters 
                                 for stats in layer_stats]
            
            colors = ['red' if ratio < 0.7 else 'orange' if ratio < 0.9 else 'green' 
                     for ratio in utilization_ratios]
            
            axes[1, 1].bar(layer_indices, utilization_ratios, color=colors, alpha=0.7)
            axes[1, 1].set_xlabel('Layer Index')
            axes[1, 1].set_ylabel('Filter Utilization Ratio')
            axes[1, 1].set_title('Filter Utilization Across Layers')
            axes[1, 1].set_xticks(layer_indices)
            axes[1, 1].set_xticklabels(layer_names, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
            
            plt.suptitle('Activation Map Analysis Summary', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            filename = f"activation_summary_{run_timestamp}.png"
            filepath = plot_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"running _create_layer_statistics_plot ... Saved layer statistics plot to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.warning(f"running _create_layer_statistics_plot ... Failed to create layer statistics plot: {e}")
            plt.close()
            return None
    
    
    def _create_activation_progression_plot(
        self,
        layer_stats: List[LayerActivationStats],
        layer_activations: Dict[str, np.ndarray],
        plot_dir: Path,
        run_timestamp: str,
        model_config
    ) -> Optional[str]:
        """
        Create a plot showing how activations progress through the network
        
        Args:
            layer_stats: List of layer statistics
            layer_activations: Activations for all layers
            plot_dir: Directory to save plot
            run_timestamp: Timestamp for file naming
            model_config: ModelConfig instance for visualization parameters
            
        Returns:
            Path to saved plot file, or None if failed
        """
        try:
            if not layer_stats:
                return None
            
            # Create a progression visualization showing how features evolve
            num_layers = len(layer_stats)
            fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 6))
            
            if num_layers == 1:
                axes = [axes]
            
            for i, stats in enumerate(layer_stats):
                layer_name = stats.layer_name
                if layer_name not in layer_activations:
                    continue
                
                activations = layer_activations[layer_name]
                if len(activations.shape) != 4:
                    continue
                
                # Show the mean activation across all filters for the first sample
                sample_activations = activations[0]  # First sample
                mean_activation_map = np.mean(sample_activations, axis=2)  # Average across filters
                
                im = axes[i].imshow(
                    mean_activation_map, 
                    cmap=model_config.activation_cmap,
                    interpolation='nearest'
                )
                axes[i].set_title(f'{layer_name}\n({stats.num_filters} filters)')
                axes[i].axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            plt.suptitle('Feature Progression Through Network Layers', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            filename = f"activation_progression_{run_timestamp}.png"
            filepath = plot_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"running _create_activation_progression_plot ... Saved progression plot to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.warning(f"running _create_activation_progression_plot ... Failed to create progression plot: {e}")
            plt.close()
            return None