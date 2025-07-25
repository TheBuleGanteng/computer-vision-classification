"""
Shared Health Analysis Module

Provides comprehensive model health assessment capabilities for use across
multiple components including hyperparameter optimizers and API endpoints.

This module extracts and centralizes health calculation logic to enable:
- Consistent health metrics across all optimization strategies
- Reusable health assessment for API visualization
- Centralized health analysis logic with proper separation of concerns
- Standardized health metric definitions and calculations

Key Health Metrics:
- Dead Neuron Analysis: Identifies inactive neurons that don't contribute to learning
- Parameter Efficiency: Measures accuracy achieved per model parameter
- Training Stability: Assesses smoothness and consistency of training process
- Gradient Health: Evaluates gradient flow quality and distribution
- Convergence Quality: Analyzes learning progress and final performance
- Accuracy Consistency: Measures prediction stability across training

Usage:
    # For optimization (with sample data)
    analyzer = HealthAnalyzer()
    health_metrics = analyzer.calculate_comprehensive_health(model, history, sample_data)
    
    # For API visualization (without sample data)
    health_metrics = analyzer.calculate_basic_health(model, history)
"""

import numpy as np
from tensorflow import keras  # type: ignore
import traceback
from typing import Dict, Any, List, Optional, Tuple
from utils.logger import logger


class HealthAnalyzer:
    """
    Comprehensive model health analysis system
    
    Provides standardized health assessment capabilities that can be used
    by optimization systems, API endpoints, and visualization tools.
    
    The analyzer supports two modes:
    1. Comprehensive analysis (with sample data for activation-based metrics)
    2. Basic analysis (weight-based metrics only, suitable for API endpoints)
    
    All health metrics are normalized to 0-1 scale where higher values
    indicate better health, enabling consistent interpretation across
    different analysis contexts.
    
    Attributes:
        None (stateless analyzer for thread safety)
        
    Example Usage:
        analyzer = HealthAnalyzer()
        
        # Comprehensive analysis with sample data
        health = analyzer.calculate_comprehensive_health(
            model=trained_model,
            history=training_history,
            sample_data=validation_samples
        )
        
        # Basic analysis for API endpoints
        health = analyzer.calculate_basic_health(
            model=trained_model,
            history=training_history
        )
    """
    
    def __init__(self):
        """
        Initialize the health analyzer
        
        Creates a stateless analyzer instance that can be safely used
        across multiple threads and components.
        """
        logger.debug("running HealthAnalyzer.__init__ ... Health analyzer initialized")
    
    def calculate_comprehensive_health(
        self,
        model: Any,
        history: Any,
        sample_data: Optional[np.ndarray] = None,
        training_time_minutes: Optional[float] = None,
        total_params: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive health metrics with activation-based analysis
        
        Performs complete health assessment including activation-based dead neuron
        detection when sample data is available. This is the most accurate health
        assessment method and should be used during optimization.
        
        Args:
            model: Trained Keras model
            history: Training history object with loss/accuracy curves
            sample_data: Optional sample data for activation-based analysis
            training_time_minutes: Optional training time for efficiency metrics
            total_params: Optional parameter count (calculated if None)
            
        Returns:
            Dictionary with comprehensive health metrics:
            {
                'neuron_utilization': float,      # 0-1, higher = better neuron usage
                'parameter_efficiency': float,    # 0-1, higher = better efficiency
                'training_stability': float,      # 0-1, higher = more stable training
                'gradient_health': float,         # 0-1, higher = healthier gradients
                'convergence_quality': float,     # 0-1, higher = better convergence
                'accuracy_consistency': float,    # 0-1, higher = more consistent
                'overall_health': float,          # 0-1, weighted average of all metrics
                'health_breakdown': dict,         # Detailed breakdown of each metric
                'recommendations': list           # Specific improvement recommendations
            }
        """
        logger.debug("running HealthAnalyzer.calculate_comprehensive_health ... Starting comprehensive health analysis")
        
        try:
            # Calculate individual health components
            neuron_health = self._calculate_neuron_utilization(model, sample_data)
            parameter_efficiency = self._calculate_parameter_efficiency(model, history, training_time_minutes, total_params)
            training_stability = self._calculate_training_stability(history)
            gradient_health = self._calculate_gradient_health(model)
            convergence_quality = self._calculate_convergence_quality(history)
            accuracy_consistency = self._calculate_accuracy_consistency(history)
            
            # Calculate overall health score (weighted average)
            overall_health = self._calculate_overall_health_score(
                neuron_health, parameter_efficiency, training_stability,
                gradient_health, convergence_quality, accuracy_consistency
            )
            
            # Generate recommendations based on health metrics
            recommendations = self._generate_health_recommendations(
                neuron_health, parameter_efficiency, training_stability,
                gradient_health, convergence_quality, accuracy_consistency
            )
            
            # Compile comprehensive results
            health_metrics = {
                'neuron_utilization': neuron_health,
                'parameter_efficiency': parameter_efficiency,
                'training_stability': training_stability,
                'gradient_health': gradient_health,
                'convergence_quality': convergence_quality,
                'accuracy_consistency': accuracy_consistency,
                'overall_health': overall_health,
                'health_breakdown': {
                    'neuron_utilization': {'score': neuron_health, 'weight': 0.25},
                    'parameter_efficiency': {'score': parameter_efficiency, 'weight': 0.15},
                    'training_stability': {'score': training_stability, 'weight': 0.20},
                    'gradient_health': {'score': gradient_health, 'weight': 0.15},
                    'convergence_quality': {'score': convergence_quality, 'weight': 0.15},
                    'accuracy_consistency': {'score': accuracy_consistency, 'weight': 0.10}
                },
                'recommendations': recommendations
            }
            
            logger.debug(f"running HealthAnalyzer.calculate_comprehensive_health ... Overall health score: {overall_health:.3f}")
            logger.debug(f"running HealthAnalyzer.calculate_comprehensive_health ... Generated {len(recommendations)} recommendations")
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"running HealthAnalyzer.calculate_comprehensive_health ... Health analysis failed: {e}")
            return self._get_default_health_metrics(error=str(e))
    
    def calculate_basic_health(
        self,
        model: Any,
        history: Any,
        training_time_minutes: Optional[float] = None,
        total_params: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate basic health metrics without activation-based analysis
        
        Provides health assessment suitable for API endpoints and visualization
        where sample data may not be available. Uses weight-based analysis
        for neuron utilization and focuses on training dynamics.
        
        Args:
            model: Trained Keras model
            history: Training history object
            training_time_minutes: Optional training time for efficiency metrics
            total_params: Optional parameter count (calculated if None)
            
        Returns:
            Dictionary with basic health metrics (same structure as comprehensive)
        """
        logger.debug("running HealthAnalyzer.calculate_basic_health ... Starting basic health analysis")
        
        try:
            # Calculate health components (weight-based analysis)
            neuron_health = self._calculate_neuron_utilization(model, sample_data=None)
            parameter_efficiency = self._calculate_parameter_efficiency(model, history, training_time_minutes, total_params)
            training_stability = self._calculate_training_stability(history)
            gradient_health = self._calculate_gradient_health(model)
            convergence_quality = self._calculate_convergence_quality(history)
            accuracy_consistency = self._calculate_accuracy_consistency(history)
            
            # Calculate overall health score
            overall_health = self._calculate_overall_health_score(
                neuron_health, parameter_efficiency, training_stability,
                gradient_health, convergence_quality, accuracy_consistency
            )
            
            # Generate basic recommendations
            recommendations = self._generate_health_recommendations(
                neuron_health, parameter_efficiency, training_stability,
                gradient_health, convergence_quality, accuracy_consistency
            )
            
            # Compile basic results
            health_metrics = {
                'neuron_utilization': neuron_health,
                'parameter_efficiency': parameter_efficiency,
                'training_stability': training_stability,
                'gradient_health': gradient_health,
                'convergence_quality': convergence_quality,
                'accuracy_consistency': accuracy_consistency,
                'overall_health': overall_health,
                'health_breakdown': {
                    'neuron_utilization': {'score': neuron_health, 'weight': 0.25, 'method': 'weight_based'},
                    'parameter_efficiency': {'score': parameter_efficiency, 'weight': 0.15},
                    'training_stability': {'score': training_stability, 'weight': 0.20},
                    'gradient_health': {'score': gradient_health, 'weight': 0.15},
                    'convergence_quality': {'score': convergence_quality, 'weight': 0.15},
                    'accuracy_consistency': {'score': accuracy_consistency, 'weight': 0.10}
                },
                'recommendations': recommendations,
                'analysis_mode': 'basic'
            }
            
            logger.debug(f"running HealthAnalyzer.calculate_basic_health ... Basic health score: {overall_health:.3f}")
            return health_metrics
            
        except Exception as e:
            logger.error(f"running HealthAnalyzer.calculate_basic_health ... Basic health analysis failed: {e}")
            return self._get_default_health_metrics(error=str(e))
    
    def _calculate_neuron_utilization(self, model: Any, sample_data: Optional[np.ndarray] = None) -> float:
        """
        Calculate neuron utilization score (1 - dead_neuron_ratio)
        
        Uses activation-based analysis if sample data is provided,
        otherwise falls back to weight-based analysis.
        
        Args:
            model: Trained Keras model
            sample_data: Optional sample data for activation analysis
            
        Returns:
            Float between 0 and 1 (higher = better neuron utilization)
        """
        try:
            if sample_data is not None:
                # Use activation-based analysis (more accurate)
                dead_ratio = self._calculate_dead_neuron_ratio_activations(model, sample_data)
                logger.debug(f"running HealthAnalyzer._calculate_neuron_utilization ... Activation-based dead ratio: {dead_ratio:.3f}")
            else:
                # Use weight-based analysis (fallback)
                dead_ratio = self._calculate_dead_neuron_ratio_weights(model)
                logger.debug(f"running HealthAnalyzer._calculate_neuron_utilization ... Weight-based dead ratio: {dead_ratio:.3f}")
            
            neuron_utilization = max(0.0, 1.0 - dead_ratio)
            return neuron_utilization
            
        except Exception as e:
            logger.warning(f"running HealthAnalyzer._calculate_neuron_utilization ... Failed to calculate neuron utilization: {e}")
            return 0.5  # Conservative fallback
    
    def _calculate_dead_neuron_ratio_activations(self, model: Any, sample_data: np.ndarray) -> float:
        """
        Calculate dead neuron ratio using activation-based detection
        
        Uses get_layer method with adaptive shape handling for different datasets.
        
        Args:
            model: Trained Keras model
            sample_data: Sample data for activation analysis
            
        Returns:
            Float between 0 and 1 (proportion of dead neurons)
        """
        try:
            # Use subset of samples for performance
            sample_size = min(50, len(sample_data))
            samples = sample_data[:sample_size]
            
            logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Starting activation analysis with {sample_size} samples")
            logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Original sample shape: {samples.shape}")
            
            # ADAPTIVE SHAPE FIX: Handle different dataset types
            if len(samples.shape) == 3:
                # 3D data (grayscale): (batch, height, width) -> (batch, height, width, 1)
                samples_fixed = np.expand_dims(samples, axis=-1)
                logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Detected grayscale data, added channel dimension: {samples.shape} -> {samples_fixed.shape}")
            elif len(samples.shape) == 4:
                # 4D data (RGB): (batch, height, width, channels) -> use as-is
                samples_fixed = samples
                logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Detected RGB data, using original shape: {samples.shape}")
            else:
                # Unexpected shape - fallback to weight-based analysis
                logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Unexpected data shape {samples.shape}, using weight-based analysis")
                return self._calculate_dead_neuron_ratio_weights(model)
            
            # Verify model can process the fixed data
            try:
                test_output = model(samples_fixed[:1])
                logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Model inference successful with fixed shape, output: {test_output.shape}")
            except Exception as inference_error:
                logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Model inference failed even with shape fix: {inference_error}")
                return self._calculate_dead_neuron_ratio_weights(model)
            
            # Check if we can use get_layer method
            if not hasattr(model, 'get_layer'):
                logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... No get_layer method available")
                return self._calculate_dead_neuron_ratio_weights(model)
            
            # Process each layer to get activations
            dead_neurons = 0
            total_neurons = 0
            current_activations = samples_fixed
            
            for layer_idx in range(len(model.layers)):
                layer = model.get_layer(index=layer_idx)
                layer_name = getattr(layer, 'name', f'layer_{layer_idx}')
                
                logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Processing layer {layer_idx}: {layer_name} ({type(layer).__name__})")
                
                try:
                    # Apply the layer to get activations
                    current_activations = layer(current_activations)
                    activation_shape = current_activations.shape
                    logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Layer {layer_name} output shape: {activation_shape}")
                    
                    # Analyze activations for this layer
                    if hasattr(layer, 'filters') and 'conv' in layer_name.lower():
                        # Convolutional layer analysis
                        activations_np = current_activations.numpy() if hasattr(current_activations, 'numpy') else current_activations
                        
                        num_filters = activations_np.shape[-1]
                        layer_dead = 0
                        
                        for filter_idx in range(num_filters):
                            filter_activations = activations_np[:, :, :, filter_idx]
                            
                            # Multiple criteria for dead filter detection
                            max_activation = np.max(filter_activations)
                            mean_activation = np.mean(filter_activations)
                            activation_frequency = np.mean(filter_activations > 0.001)
                            activation_std = np.std(filter_activations)
                            
                            is_dead = (
                                max_activation < 0.01 or
                                activation_frequency < 0.01 or
                                activation_std < 0.001 or
                                mean_activation < 0.001
                            )
                            
                            if is_dead:
                                dead_neurons += 1
                                layer_dead += 1
                            total_neurons += 1
                        
                        logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Conv layer {layer_name}: {layer_dead}/{num_filters} dead filters")
                    
                    elif hasattr(layer, 'units') and 'dense' in layer_name.lower():
                        # Dense layer analysis
                        activations_np = current_activations.numpy() if hasattr(current_activations, 'numpy') else current_activations
                        
                        # Handle different activation shapes
                        if len(activations_np.shape) > 2:
                            # Flatten if needed (shouldn't happen after Flatten layer, but just in case)
                            activations_np = activations_np.reshape(activations_np.shape[0], -1)
                        
                        num_neurons = activations_np.shape[-1]
                        layer_dead = 0
                        
                        for neuron_idx in range(num_neurons):
                            neuron_activations = activations_np[:, neuron_idx]
                            
                            max_activation = np.max(neuron_activations)
                            activation_frequency = np.mean(neuron_activations > 0.01)
                            
                            if max_activation < 0.05 or activation_frequency < 0.02:
                                dead_neurons += 1
                                layer_dead += 1
                            total_neurons += 1
                        
                        logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Dense layer {layer_name}: {layer_dead}/{num_neurons} dead neurons")
                    
                    else:
                        # Skip non-analyzable layers (Dropout, Flatten, etc.)
                        logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Skipping non-analyzable layer: {layer_name} ({type(layer).__name__})")
                        continue
                    
                except Exception as layer_error:
                    logger.warning(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Failed to process layer {layer_name}: {layer_error}")
                    logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Layer error details: {traceback.format_exc()}")
                    # Continue processing other layers
                    continue
            
            if total_neurons == 0:
                logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... No analyzable neurons found, using weight-based analysis")
                return self._calculate_dead_neuron_ratio_weights(model)
            
            dead_ratio = dead_neurons / total_neurons
            logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Activation analysis complete: {dead_neurons}/{total_neurons} dead = {dead_ratio:.3f}")
            return dead_ratio
            
        except Exception as e:
            logger.warning(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Activation analysis failed: {e}")
            logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_activations ... Error traceback: {traceback.format_exc()}")
            return self._calculate_dead_neuron_ratio_weights(model)

    
    
    def _calculate_dead_neuron_ratio_weights(self, model: Any, threshold: float = 1e-3) -> float:
        """
        Calculate dead neuron ratio using weight variance analysis
        
        Analyzes weight distributions to identify neurons with low variance
        (indicating they may not be contributing meaningfully to learning).
        
        Args:
            model: Trained Keras model
            threshold: Variance threshold for dead neuron detection
            
        Returns:
            Float between 0 and 1 (proportion of dead neurons)
        """
        try:
            total_neurons = 0
            dead_neurons = 0
            
            for layer in model.layers:
                # Dense layers
                if hasattr(layer, 'units') and hasattr(layer, 'activation'):
                    layer_neurons = layer.units
                    total_neurons += layer_neurons
                    
                    if layer.get_weights():
                        weights = layer.get_weights()[0]  # Weight matrix
                        weight_variances = np.var(weights, axis=0)
                        layer_dead = np.sum(weight_variances < threshold)
                        dead_neurons += layer_dead
                
                # Convolutional layers
                elif hasattr(layer, 'filters') and 'conv' in layer.name.lower():
                    layer_neurons = layer.filters
                    total_neurons += layer_neurons
                    
                    if layer.get_weights():
                        weights = layer.get_weights()[0]  # Conv weight tensor
                        filter_variances = np.var(weights, axis=(0, 1, 2))
                        layer_dead = np.sum(filter_variances < threshold)
                        dead_neurons += layer_dead
            
            dead_ratio = dead_neurons / total_neurons if total_neurons > 0 else 0.0
            logger.debug(f"running HealthAnalyzer._calculate_dead_neuron_ratio_weights ... Weight analysis: {dead_neurons}/{total_neurons} = {dead_ratio:.3f}")
            return dead_ratio
            
        except Exception as e:
            logger.warning(f"running HealthAnalyzer._calculate_dead_neuron_ratio_weights ... Weight analysis failed: {e}")
            return 0.3  # Conservative estimate
    
    def _calculate_parameter_efficiency(
        self,
        model: Any,
        history: Any,
        training_time_minutes: Optional[float] = None,
        total_params: Optional[int] = None
    ) -> float:
        """
        Calculate parameter efficiency (accuracy per parameter)
        
        Args:
            model: Trained Keras model
            history: Training history
            training_time_minutes: Optional training time
            total_params: Optional parameter count
            
        Returns:
            Float between 0 and 1 (higher = better efficiency)
        """
        try:
            if total_params is None:
                total_params = model.count_params()
            
            # Get best validation accuracy
            val_accuracy = 0.0
            if history and hasattr(history, 'history'):
                if 'val_accuracy' in history.history:
                    val_accuracy = max(history.history['val_accuracy'])
                elif 'accuracy' in history.history:
                    val_accuracy = max(history.history['accuracy'])
            
            if total_params is not None and total_params > 0:
                # Use log scale to avoid penalizing larger models too heavily
                params_millions = total_params / 1_000_000
                efficiency_raw = val_accuracy / (1 + np.log10(max(params_millions, 0.001)))
                parameter_efficiency = min(1.0, max(0.0, efficiency_raw))
            else:
                parameter_efficiency = val_accuracy
            
            logger.debug(f"running HealthAnalyzer._calculate_parameter_efficiency ... Efficiency: {parameter_efficiency:.3f} (acc: {val_accuracy:.3f}, params: {total_params:,})")
            return parameter_efficiency
            
        except Exception as e:
            logger.warning(f"running HealthAnalyzer._calculate_parameter_efficiency ... Failed to calculate efficiency: {e}")
            return 0.5
    
    def _calculate_training_stability(self, history: Any) -> float:
        """
        Calculate training stability based on loss curve smoothness
        
        Args:
            history: Training history
            
        Returns:
            Float between 0 and 1 (higher = more stable training)
        """
        try:
            if not history or not hasattr(history, 'history') or 'loss' not in history.history:
                return 0.5
            
            losses = np.array(history.history['loss'])
            if len(losses) < 3:
                return 0.5
            
            # Overall trend (should be decreasing)
            overall_trend = (losses[-1] - losses[0]) / max(abs(losses[0]), 1e-6)
            trend_score = max(0.0, min(1.0, -overall_trend))
            
            # Smoothness (low variance in loss changes)
            loss_changes = np.diff(losses)
            smoothness_mad = np.median(np.abs(loss_changes - np.median(loss_changes)))
            smoothness = 1.0 / (1.0 + smoothness_mad * 3)
            
            # Recent improvement
            if len(losses) >= 5:
                recent_losses = losses[-5:]
                recent_trend = recent_losses[-1] - recent_losses[0]
                recent_improvement = max(0.0, min(1.0, -recent_trend + 0.5))
            else:
                recent_improvement = 0.5
            
            # Combine scores
            stability = (trend_score * 0.5 + smoothness * 0.2 + recent_improvement * 0.3)
            stability = min(1.0, max(0.0, float(stability)))
            
            logger.debug(f"running HealthAnalyzer._calculate_training_stability ... Stability: {stability:.3f}")
            return stability
            
        except Exception as e:
            logger.warning(f"running HealthAnalyzer._calculate_training_stability ... Failed to calculate stability: {e}")
            return 0.5
    
    def _calculate_gradient_health(self, model: Any) -> float:
        """
        Calculate gradient health by examining weight distributions
        
        Args:
            model: Trained Keras model
            
        Returns:
            Float between 0 and 1 (higher = healthier gradients)
        """
        try:
            weight_stats = []
            
            for layer in model.layers:
                if layer.get_weights():
                    weights = layer.get_weights()[0]  # Main weight matrix
                    
                    weight_mean = np.mean(np.abs(weights))
                    weight_std = np.std(weights)
                    weight_max = np.max(np.abs(weights))
                    
                    # Healthy weights should have reasonable magnitudes
                    mean_score = 1.0 / (1.0 + weight_mean * 5)
                    std_score = min(1.0, float(weight_std) * 5)
                    max_score = 1.0 / (1.0 + weight_max * 0.5)
                    
                    layer_health = (mean_score + std_score + max_score) / 3
                    weight_stats.append(layer_health)
            
            if weight_stats:
                gradient_health = np.mean(weight_stats)
                gradient_health = min(1.0, max(0.0, float(gradient_health)))
                logger.debug(f"running HealthAnalyzer._calculate_gradient_health ... Gradient health: {gradient_health:.3f}")
                return gradient_health
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"running HealthAnalyzer._calculate_gradient_health ... Failed to calculate gradient health: {e}")
            return 0.5
    
    def _calculate_convergence_quality(self, history: Any) -> float:
        """
        Calculate convergence quality based on learning progress
        
        Args:
            history: Training history
            
        Returns:
            Float between 0 and 1 (higher = better convergence)
        """
        try:
            if not history or not hasattr(history, 'history'):
                return 0.5
            
            # Use validation accuracy if available, otherwise training accuracy
            if 'val_accuracy' in history.history and history.history['val_accuracy']:
                accuracies = np.array(history.history['val_accuracy'])
            elif 'accuracy' in history.history and history.history['accuracy']:
                accuracies = np.array(history.history['accuracy'])
            else:
                return 0.5
            
            if len(accuracies) < 3:
                return 0.5
            
            # Final performance
            final_performance = accuracies[-1]
            
            # Improvement over training
            total_improvement = accuracies[-1] - accuracies[0]
            improvement_score = max(0.0, min(1.0, total_improvement * 5))
            
            # Stability in final epochs
            if len(accuracies) >= 5:
                final_epochs = accuracies[-5:]
                final_stability = 1.0 / (1.0 + np.var(final_epochs) * 20)
            else:
                final_stability = 0.5
            
            # Combine scores
            convergence = (final_performance * 0.6 + improvement_score * 0.3 + final_stability * 0.1)
            convergence = min(1.0, max(0.0, convergence))
            
            logger.debug(f"running HealthAnalyzer._calculate_convergence_quality ... Convergence: {convergence:.3f}")
            return convergence
            
        except Exception as e:
            logger.warning(f"running HealthAnalyzer._calculate_convergence_quality ... Failed to calculate convergence: {e}")
            return 0.5
    
    def _calculate_accuracy_consistency(self, history: Any) -> float:
        """
        Calculate accuracy consistency across training
        
        Args:
            history: Training history
            
        Returns:
            Float between 0 and 1 (higher = more consistent)
        """
        try:
            if not history or not hasattr(history, 'history'):
                return 0.5
            
            # Use validation accuracy if available
            if 'val_accuracy' in history.history and len(history.history['val_accuracy']) >= 5:
                accuracies = np.array(history.history['val_accuracy'])
            elif 'accuracy' in history.history and len(history.history['accuracy']) >= 5:
                accuracies = np.array(history.history['accuracy'])
            else:
                return 0.5
            
            # Calculate consistency as inverse of coefficient of variation
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            if mean_accuracy > 0:
                cv = std_accuracy / mean_accuracy
                consistency = 1.0 / (1.0 + cv * 10)
            else:
                consistency = 0.0
            
            consistency = min(1.0, max(0.0, float(consistency)))
            logger.debug(f"running HealthAnalyzer._calculate_accuracy_consistency ... Consistency: {consistency:.3f}")
            return consistency
            
        except Exception as e:
            logger.warning(f"running HealthAnalyzer._calculate_accuracy_consistency ... Failed to calculate consistency: {e}")
            return 0.5
    
    def _calculate_overall_health_score(
        self,
        neuron_health: float,
        parameter_efficiency: float,
        training_stability: float,
        gradient_health: float,
        convergence_quality: float,
        accuracy_consistency: float
    ) -> float:
        """
        Calculate weighted overall health score
        
        Args:
            neuron_health: Neuron utilization score
            parameter_efficiency: Parameter efficiency score
            training_stability: Training stability score
            gradient_health: Gradient health score
            convergence_quality: Convergence quality score
            accuracy_consistency: Accuracy consistency score
            
        Returns:
            Float between 0 and 1 (overall health score)
        """
        # Define weights for each health component
        weights = {
            'neuron_health': 0.25,        # Most important for model capacity
            'training_stability': 0.20,   # Critical for reliable training
            'parameter_efficiency': 0.15, # Important for model practicality
            'gradient_health': 0.15,      # Important for training dynamics
            'convergence_quality': 0.15,  # Important for learning effectiveness
            'accuracy_consistency': 0.10  # Important for prediction reliability
        }
        
        overall_health = (
            neuron_health * weights['neuron_health'] +
            parameter_efficiency * weights['parameter_efficiency'] +
            training_stability * weights['training_stability'] +
            gradient_health * weights['gradient_health'] +
            convergence_quality * weights['convergence_quality'] +
            accuracy_consistency * weights['accuracy_consistency']
        )
        
        return min(1.0, max(0.0, overall_health))
    
    def _generate_health_recommendations(
        self,
        neuron_health: float,
        parameter_efficiency: float,
        training_stability: float,
        gradient_health: float,
        convergence_quality: float,
        accuracy_consistency: float
    ) -> List[str]:
        """
        Generate specific recommendations based on health metrics
        
        Args:
            neuron_health: Neuron utilization score
            parameter_efficiency: Parameter efficiency score
            training_stability: Training stability score
            gradient_health: Gradient health score
            convergence_quality: Convergence quality score
            accuracy_consistency: Accuracy consistency score
            
        Returns:
            List of specific improvement recommendations
        """
        recommendations = []
        
        # Neuron utilization recommendations
        if neuron_health < 0.7:
            if neuron_health < 0.4:
                recommendations.append("CRITICAL: High dead neuron ratio detected. Consider reducing model size or adjusting learning rate.")
            else:
                recommendations.append("Consider reducing dropout rates or adjusting activation functions to improve neuron utilization.")
        
        # Parameter efficiency recommendations
        if parameter_efficiency < 0.6:
            recommendations.append("Model may be over-parameterized. Consider using smaller architectures or global pooling.")
        
        # Training stability recommendations
        if training_stability < 0.6:
            recommendations.append("Training appears unstable. Consider reducing learning rate or adding batch normalization.")
        
        # Gradient health recommendations
        if gradient_health < 0.6:
            recommendations.append("Gradient health issues detected. Consider gradient clipping or weight initialization adjustments.")
        
        # Convergence quality recommendations
        if convergence_quality < 0.7:
            recommendations.append("Poor convergence detected. Consider increasing training epochs or adjusting optimizer settings.")
        
        # Accuracy consistency recommendations
        if accuracy_consistency < 0.7:
            recommendations.append("Accuracy inconsistency detected. Consider adjusting validation split or training data quality.")
        
        # Overall health recommendations
        overall_health = self._calculate_overall_health_score(
            neuron_health, parameter_efficiency, training_stability,
            gradient_health, convergence_quality, accuracy_consistency
        )
        
        if overall_health < 0.5:
            recommendations.append("URGENT: Multiple health issues detected. Consider comprehensive model architecture review.")
        elif overall_health < 0.7:
            recommendations.append("Model health below optimal. Focus on addressing the lowest-scoring health metrics.")
        
        return recommendations
    
    def _get_default_health_metrics(self, error: Optional[str] = None) -> Dict[str, Any]:
        """
        Get default health metrics for error cases
        
        Args:
            error: Optional error message
            
        Returns:
            Dictionary with default health metrics
        """
        default_metrics = {
            'neuron_utilization': 0.5,
            'parameter_efficiency': 0.5,
            'training_stability': 0.5,
            'gradient_health': 0.5,
            'convergence_quality': 0.5,
            'accuracy_consistency': 0.5,
            'overall_health': 0.5,
            'health_breakdown': {
                'neuron_utilization': {'score': 0.5, 'weight': 0.25},
                'parameter_efficiency': {'score': 0.5, 'weight': 0.15},
                'training_stability': {'score': 0.5, 'weight': 0.20},
                'gradient_health': {'score': 0.5, 'weight': 0.15},
                'convergence_quality': {'score': 0.5, 'weight': 0.15},
                'accuracy_consistency': {'score': 0.5, 'weight': 0.10}
            },
            'recommendations': ["Health analysis failed. Check model and training data quality."],
            'analysis_mode': 'error'
        }
        
        if error:
            default_metrics['error'] = error
        
        return default_metrics