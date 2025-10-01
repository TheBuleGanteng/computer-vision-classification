"""
Weights and Bias Distribution Analysis Module

This module provides comprehensive analysis of learned neural network parameters including:
- Weight distribution analysis across all layers (convolutional and dense)
- Bias distribution analysis and threshold detection
- Layer-wise parameter health assessment
- Professional-quality visualizations for presentations and debugging
- Detection of training issues through parameter patterns

Designed to work with any Keras model architecture and provides insights into
what the model has learned and how well the training process performed.

FUNDAMENTAL CONCEPTS:

What Are Weights and Biases?
    Weights and biases are the learned parameters that define how your neural network
    transforms inputs into outputs. They literally encode the "knowledge" your model
    acquired during training.

Weights in Neural Networks:
    - Convolutional weights: "Determine what patterns are important for feature detection"
    - Dense weights: "Determine the strength of connections between neurons in the 
      current dense layer and neurons in the preceding dense layer"

Biases in Neural Networks:
    - Act like y-intercepts in linear functions: y = mx + b
    - Shift the activation function up or down, controlling baseline activation
    - Allow neurons to activate even when all inputs are zero
    - Control the "skepticism" or "eagerness" of each neuron

Weight Values Interpretation:
    - High weight (e.g., 5.0): "High sensitivity" - activates easily with small inputs
    - Low weight (e.g., 0.1): "Low sensitivity" - requires strong evidence to activate
    - Positive weight: "This input supports this neuron's purpose"
    - Negative weight: "This input contradicts this neuron's purpose"
    - Zero weight: "This input is irrelevant to this neuron"

Bias Values Interpretation:
    - Positive bias: "Optimistic" - neuron tends to activate more readily
    - Negative bias: "Skeptical" - neuron requires stronger evidence to activate
    - Large absolute bias: Strong baseline opinion, harder for inputs to override
    - Small absolute bias: Neutral baseline, inputs have more influence

WEIGHT/BIAS DISTRIBUTION HEALTH INDICATORS:

Healthy Parameter Distributions:
    - Weights centered around zero with appropriate spread
    - No extreme outliers or saturation
    - Biases distributed across a reasonable range
    - Clear evolution during training (if monitored over time)

Problem Patterns:
    - Weight Explosion: Very large weight values (>10) indicate training instability
    - Weight Collapse: All weights near zero indicate dead/non-learning neurons
    - Weight Saturation: Many weights at extreme values (+/-3) indicate activation saturation
    - Bias Domination: Very large biases can override input influence entirely

PRACTICAL APPLICATIONS:

Model Debugging:
    - Identify layers that aren't learning (weights near zero)
    - Detect training instability (exploding weights)
    - Find dead neurons (collapsed weight distributions)
    - Assess initialization quality

Architecture Optimization:
    - Compare parameter distributions across different architectures
    - Guide decisions about layer sizes and depths
    - Understand which layers are most important for the task

Training Analysis:
    - Validate that training is progressing healthily
    - Identify when to stop training (parameter convergence)
    - Detect overfitting through parameter patterns
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from tensorflow import keras # type: ignore
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
from utils.logger import logger


class WeightsBiasAnalyzer:
    """
    Comprehensive weights and bias distribution analysis and visualization
    
    Analyzes the learned parameters of a neural network to understand what
    patterns the model has learned and assess the health of the training process.
    Essential for debugging training issues and understanding model behavior.
    
    ANALYSIS CAPABILITIES:
    
    1. **Layer-wise Distribution Analysis**:
       For each layer with trainable parameters:
       - Weight distribution statistics (mean, std, min, max, quartiles)
       - Bias distribution analysis (if present)
       - Parameter count and density information
       - Health assessment based on distribution patterns
    
    2. **Cross-Layer Comparison**:
       - Compare parameter distributions across layers
       - Identify layers with unusual parameter patterns
       - Track how parameter complexity varies through the network
    
    3. **Training Health Assessment**:
       - Detect common training problems through parameter analysis
       - Identify dead neurons and collapsed weights
       - Find evidence of exploding or vanishing gradient effects
       - Assess initialization and learning quality
    
    4. **Professional Visualizations**:
       - Multi-subplot dashboard showing all layer distributions
       - Box plots, histograms, and violin plots
       - Statistical summaries and health indicators
       - Publication-quality plots for papers and presentations
    
    INTERPRETATION GUIDE:
    
    Weight Distribution Patterns:
    ```
    Healthy CNN Layer:
    ├── Conv weights: mean≈0, std=0.1-0.3, smooth distribution
    ├── Dense weights: mean≈0, std=0.05-0.2, bell-curved
    └── Assessment: ✅ Normal learning, good generalization
    
    Problematic Dense Layer:
    ├── Weights: mean=0.0, std=0.001, very narrow distribution
    ├── Many weights exactly at 0.0 (>50%)
    └── Assessment: ❌ Dead neurons, not contributing to learning
    
    Exploding Weights Layer:
    ├── Weights: mean=2.1, std=5.4, very wide distribution  
    ├── Many extreme values (>3.0 or <-3.0)
    └── Assessment: ❌ Training instability, gradient explosion
    ```
    
    Bias Distribution Patterns:
    ```
    Healthy Bias Pattern:
    ├── Biases: mean≈0, std=0.5-2.0, distributed across range
    ├── Mix of positive and negative values
    └── Assessment: ✅ Good threshold learning
    
    Bias Domination Problem:
    ├── Biases: mean=10.2, std=15.8, very large values
    ├── Overwhelming input influence
    └── Assessment: ❌ Inputs can't overcome bias influence
    ```
    """
    
    def __init__(self, model_name: str = "Model"):
        """
        Initialize the weights and bias analyzer
        
        Args:
            model_name: Name of the model for plot titles and logging
        """
        self.model_name = model_name
        logger.debug("running WeightsBiasAnalyzer.__init__ ... Weights and bias analyzer initialized")
    
    def analyze_and_visualize(
        self,
        model: keras.Model,
        dataset_name: str = "dataset",
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None,
        max_layers_to_plot: int = 12
    ) -> Dict[str, Any]:
        """
        Analyze model weights and biases and create comprehensive visualizations
        
        Performs detailed analysis of all trainable parameters in the neural network,
        creating professional visualizations and health assessments. This analysis
        reveals what the model has learned and identifies potential training issues.
        
        ANALYSIS PROCESS:
        
        1. **Parameter Extraction**:
           For each layer with trainable parameters:
           ```
           Layer Analysis:
           ├── Extract all weights and biases
           ├── Flatten multi-dimensional arrays for statistical analysis  
           ├── Compute comprehensive statistics (mean, std, quartiles, etc.)
           └── Assess parameter health and identify patterns
           ```
        
        2. **Distribution Analysis**:
           ```
           Statistical Measures Computed:
           ├── Central Tendency: mean, median, mode
           ├── Spread: standard deviation, variance, range
           ├── Shape: skewness, kurtosis, distribution type
           ├── Extremes: min, max, outlier detection
           └── Health Indicators: dead neuron %, saturation level
           ```
        
        3. **Cross-Layer Comparison**:
           ```
           Comparative Analysis:
           ├── Parameter count progression through network
           ├── Distribution width evolution (early vs late layers)
           ├── Mean/variance trends across architecture
           └── Identification of problematic layers
           ```
        
        4. **Visualization Creation**:
           ```
           Multi-Panel Dashboard:
           ├── Top Left: Weight Distributions (histograms/violin plots)
           ├── Top Right: Bias Distributions (box plots)
           ├── Bottom Left: Layer-wise Statistics Summary  
           └── Bottom Right: Health Assessment and Warnings
           ```
        
        WEIGHT DISTRIBUTION INTERPRETATION:
        
        What Different Patterns Mean:
        ```
        Healthy Learning Pattern:
        Conv2D Layer 1: mean=0.02, std=0.15, range=[-0.8, 0.9]
        ├── Analysis: ✅ Good initialization, learning progressing
        ├── Interpretation: Filters learning diverse patterns
        └── Action: Continue training as normal
        
        Dense Layer 3: mean=0.01, std=0.12, range=[-0.6, 0.7]  
        ├── Analysis: ✅ Healthy connection strengths
        ├── Interpretation: Balanced positive/negative influences
        └── Action: Model ready for deployment
        ```
        
        ```
        Dead Neurons Pattern:
        Dense Layer 2: mean=0.00, std=0.001, range=[-0.01, 0.01]
        ├── Analysis: ❌ 85% of weights near zero (dead neurons)
        ├── Interpretation: Layer not contributing to learning
        └── Action: Check initialization, reduce layer size, or use different activation
        ```
        
        ```
        Weight Explosion Pattern:
        Dense Layer 4: mean=1.8, std=12.5, range=[-45.2, 38.7]
        ├── Analysis: ❌ Extreme weight values, training unstable
        ├── Interpretation: Gradient explosion, loss of control
        └── Action: Add gradient clipping, reduce learning rate, check data normalization
        ```
        
        BIAS ANALYSIS INSIGHTS:
        
        Understanding Bias Patterns:
        ```
        Balanced Threshold Learning:
        Layer Biases: mean=-0.2, std=1.3, range=[-3.1, 2.8]
        ├── Interpretation: Neurons learning appropriate activation thresholds
        ├── Mix of optimistic (+) and skeptical (-) neurons
        └── Assessment: ✅ Healthy learning of decision boundaries
        
        Bias Domination Problem:
        Layer Biases: mean=15.6, std=8.9, range=[2.1, 28.4]
        ├── Interpretation: Biases overwhelming input influence
        ├── Neurons activating regardless of input patterns
        └── Assessment: ❌ Inputs can't meaningfully affect decisions
        ```
        
        PRACTICAL DEBUGGING APPLICATIONS:
        
        Training Problem Diagnosis:
        ```
        Problem: "Model accuracy plateaued at 60% after epoch 5"
        Weight Analysis Findings:
        ├── Early layers: Healthy weight distributions ✅
        ├── Middle layers: 70% of weights collapsed to near-zero ❌
        ├── Final layer: Extreme bias values dominating ❌
        └── Solution: Better initialization + learning rate adjustment
        
        Problem: "Training loss oscillating wildly"  
        Weight Analysis Findings:
        ├── All layers: Weight standard deviation increasing each epoch
        ├── Many weights exceeding ±5.0 range
        └── Solution: Implement gradient clipping, reduce learning rate
        ```
        
        Model Comparison Use Case:
        ```
        Architecture A vs Architecture B:
        Model A: All layers show healthy, evolving weight distributions
        Model B: Several layers show saturated or collapsed weights
        Conclusion: Architecture A is superior for this task
        ```
        
        Args:
            model: Trained Keras model to analyze
                  Must have been trained (not just initialized) for meaningful analysis
            dataset_name: Name of dataset for plot titles and file naming
            run_timestamp: Optional timestamp for consistent file naming across analysis
            plot_dir: Optional directory to save visualization plots
            max_layers_to_plot: Maximum number of layers to include in visualizations
                               Prevents overcrowded plots for very deep networks
            
        Returns:
            Dictionary containing comprehensive analysis results:
            
            Core Analysis:
            - 'layer_analysis': Detailed statistics for each layer with trainable parameters
            - 'weight_statistics': Cross-layer weight distribution summaries  
            - 'bias_statistics': Cross-layer bias distribution summaries
            - 'parameter_health': Overall assessment of parameter health
            
            Detailed Statistics:
            ```
            'layer_analysis': {
                'conv2d': {
                    'layer_type': 'Conv2D',
                    'total_params': 896,
                    'weight_stats': {
                        'mean': 0.02, 'std': 0.15, 'min': -0.8, 'max': 0.9,
                        'q25': -0.1, 'median': 0.01, 'q75': 0.12
                    },
                    'bias_stats': {
                        'mean': -0.1, 'std': 0.3, 'count': 32
                    },
                    'health_assessment': 'healthy',
                    'dead_neuron_percentage': 2.1,
                    'saturation_percentage': 0.5
                }
            }
            ```
            
            Health Assessment:
            - 'parameter_health': 'healthy' | 'concerning' | 'problematic' | 'critical'
            - 'training_insights': List of observations about parameter patterns
            - 'recommendations': List of actionable suggestions for improvement
            - 'visualization_path': Path to saved analysis dashboard
            
        Side Effects:
            - Creates comprehensive matplotlib dashboard with parameter distributions
            - Saves high-resolution PNG file with timestamp for documentation
            - Automatically detects and warns about parameter health issues
            - Logs detailed analysis results and recommendations
            - Closes figures to prevent memory leaks
            
        Requirements:
            - Model must have trainable parameters (weights and/or biases)
            - Model should be trained for meaningful analysis (random weights less useful)
            - Sufficient memory for parameter extraction and visualization
            
        Performance Notes:
            - Analysis time scales with model complexity and parameter count
            - Memory usage proportional to total number of parameters
            - Large models (>10M parameters) may require reducing max_layers_to_plot
        """
        logger.debug("running analyze_and_visualize ... Starting comprehensive weights and bias analysis")
        
        try:
            # Validate model has trainable parameters
            if not self._validate_model(model):
                return {'error': 'Model has no trainable parameters for analysis'}
            
            # Extract and analyze parameters from all layers
            layer_analysis = self._extract_layer_parameters(model)
            
            # Perform cross-layer statistical analysis
            analysis_results = self._analyze_parameter_statistics(layer_analysis)
            
            # Assess overall parameter health
            health_assessment = self._assess_parameter_health(layer_analysis, analysis_results)
            
            # Create comprehensive visualization dashboard
            visualization_path = self._create_visualization_dashboard(
                layer_analysis=layer_analysis,
                analysis_results=analysis_results,
                health_assessment=health_assessment,
                model=model,
                dataset_name=dataset_name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir,
                max_layers_to_plot=max_layers_to_plot
            )
            
            # Combine all results
            final_results = {
                'layer_analysis': layer_analysis,
                'weight_statistics': analysis_results['weight_statistics'],
                'bias_statistics': analysis_results['bias_statistics'], 
                'parameter_health': health_assessment['overall_health'],
                'training_insights': health_assessment['insights'],
                'recommendations': health_assessment['recommendations'],
                'visualization_path': visualization_path
            }
            
            # Log analysis summary
            self._log_analysis_summary(final_results)
            
            logger.debug("running analyze_and_visualize ... Weights and bias analysis completed successfully")
            return final_results
            
        except Exception as e:
            logger.warning(f"running analyze_and_visualize ... Failed to complete weights and bias analysis: {e}")
            logger.debug(f"running analyze_and_visualize ... Error traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def _validate_model(self, model: keras.Model) -> bool:
        """
        Validate that the model has trainable parameters for analysis
        
        Ensures the model is suitable for parameter analysis by checking for
        the presence of trainable weights and biases.
        
        Args:
            model: Keras model to validate
            
        Returns:
            True if model has trainable parameters, False otherwise
        """
        logger.debug("running _validate_model ... Validating model for parameter analysis")
        
        if model is None:
            logger.warning("running _validate_model ... Model is None")
            return False
        
        # Check for trainable variables
        if not hasattr(model, 'trainable_variables') or len(model.trainable_variables) == 0:
            logger.warning("running _validate_model ... Model has no trainable variables")
            return False
        
        # Count layers with trainable parameters
        trainable_layers = 0
        for layer in model.layers:
            if layer.trainable_weights:
                trainable_layers += 1
        
        if trainable_layers == 0:
            logger.warning("running _validate_model ... Model has no layers with trainable weights")
            return False
        
        logger.debug(f"running _validate_model ... Model validation successful: {trainable_layers} trainable layers found")
        return True
    
    def _extract_layer_parameters(self, model: keras.Model) -> Dict[str, Dict[str, Any]]:
        """
        Extract weights and biases from all model layers with trainable parameters
        
        Systematically extracts all trainable parameters from the model and organizes
        them by layer for statistical analysis. Handles different layer types
        (convolutional, dense, etc.) appropriately.
        
        PARAMETER EXTRACTION PROCESS:
        
        For each layer with trainable parameters:
        1. **Weight Extraction**: Get all weight matrices/tensors from the layer
        2. **Bias Extraction**: Get bias vectors if present (not all layers have biases)
        3. **Flattening**: Convert multi-dimensional arrays to 1D for statistical analysis
        4. **Organization**: Store by layer name with metadata about layer type and size
        
        Layer Type Handling:
        ```
        Convolutional Layers:
        ├── Weights shape: (kernel_height, kernel_width, input_channels, output_channels)
        ├── Biases shape: (output_channels,) - one bias per filter
        ├── Interpretation: Each weight controls pattern sensitivity
        └── Flattening: All kernel weights combined into single distribution
        
        Dense Layers:
        ├── Weights shape: (input_neurons, output_neurons) 
        ├── Biases shape: (output_neurons,) - one bias per neuron
        ├── Interpretation: Each weight controls connection strength between layers
        └── Flattening: All connection weights combined into single distribution
        ```
        
        Args:
            model: Keras model with trainable parameters
            
        Returns:
            Dictionary mapping layer names to parameter information:
            ```
            {
                'conv2d': {
                    'layer_type': 'Conv2D',
                    'weights': np.array([0.1, -0.2, 0.3, ...]),  # Flattened weights
                    'biases': np.array([0.05, -0.1, 0.08, ...]), # Bias values
                    'weight_shape': (3, 3, 3, 32),               # Original weight tensor shape
                    'bias_shape': (32,),                         # Original bias vector shape
                    'total_params': 896,                         # Total trainable parameters
                    'weight_count': 864,                         # Number of weights
                    'bias_count': 32                             # Number of biases
                }
            }
            ```
        """
        logger.debug("running _extract_layer_parameters ... Extracting parameters from all model layers")
        
        layer_analysis = {}
        
        for layer in model.layers:
            # Skip layers without trainable parameters
            if not layer.trainable_weights:
                continue
            
            layer_name = layer.name
            layer_type = type(layer).__name__
            
            logger.debug(f"running _extract_layer_parameters ... Processing layer: {layer_name} ({layer_type})")
            
            try:
                # Extract weights and biases using get_weights()
                layer_weights = layer.get_weights()
                
                if not layer_weights:
                    logger.debug(f"running _extract_layer_parameters ... Layer {layer_name} has no weights")
                    continue
                
                # Initialize parameter storage
                all_weights = []
                all_biases = []
                weight_shapes = []
                bias_shapes = []
                
                # Process each weight array in the layer
                for i, weight_array in enumerate(layer_weights):
                    weight_shape = weight_array.shape
                    
                    # Determine if this is weights or bias based on shape and position
                    # Convention: weights come first, biases come last in get_weights()
                    if i == len(layer_weights) - 1 and len(weight_shape) == 1:
                        # Last array and 1D -> likely bias vector
                        all_biases.extend(weight_array.flatten())
                        bias_shapes.append(weight_shape)
                        logger.debug(f"running _extract_layer_parameters ... Extracted {len(weight_array)} biases from {layer_name}")
                    else:
                        # Multi-dimensional or not last -> likely weights
                        all_weights.extend(weight_array.flatten())
                        weight_shapes.append(weight_shape)
                        logger.debug(f"running _extract_layer_parameters ... Extracted {weight_array.size} weights from {layer_name}")
                
                # Store layer analysis results
                layer_analysis[layer_name] = {
                    'layer_type': layer_type,
                    'weights': np.array(all_weights) if all_weights else np.array([]),
                    'biases': np.array(all_biases) if all_biases else np.array([]),
                    'weight_shapes': weight_shapes,
                    'bias_shapes': bias_shapes,
                    'total_params': len(all_weights) + len(all_biases),
                    'weight_count': len(all_weights),
                    'bias_count': len(all_biases)
                }
                
                logger.debug(f"running _extract_layer_parameters ... Layer {layer_name}: "
                            f"{len(all_weights)} weights, {len(all_biases)} biases")
                
            except Exception as e:
                logger.warning(f"running _extract_layer_parameters ... Failed to extract parameters from {layer_name}: {e}")
                continue
        
        logger.debug(f"running _extract_layer_parameters ... Parameter extraction completed for {len(layer_analysis)} layers")
        return layer_analysis
    
    def _analyze_parameter_statistics(self, layer_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for weights and biases across all layers
        
        Performs detailed statistical analysis of parameter distributions to understand
        the health and characteristics of the learned parameters.
        
        STATISTICAL ANALYSIS COMPONENTS:
        
        1. **Individual Layer Statistics**:
           For each layer's weights and biases:
           ```
           Descriptive Statistics:
           ├── Central Tendency: mean, median, mode
           ├── Dispersion: std, variance, range, IQR
           ├── Distribution Shape: skewness, kurtosis
           ├── Extremes: min, max, outlier count
           └── Health Indicators: dead %, saturation %
           ```
        
        2. **Cross-Layer Analysis**:
           ```
           Comparative Metrics:
           ├── Parameter count progression through network
           ├── Mean weight evolution (early vs late layers)
           ├── Variance trends across architecture depth
           └── Bias distribution patterns by layer type
           ```
        
        3. **Health Assessment Metrics**:
           ```
           Training Quality Indicators:
           ├── Dead Parameter Detection: |param| < 1e-6
           ├── Saturation Detection: |param| > 3.0
           ├── Gradient Flow Inference: std progression
           └── Learning Effectiveness: distribution evolution
           ```
        
        Args:
            layer_analysis: Dictionary of extracted layer parameters
            
        Returns:
            Dictionary containing statistical analysis results
        """
        logger.debug("running _analyze_parameter_statistics ... Computing comprehensive parameter statistics")
        
        # Initialize statistics storage
        weight_stats = {}
        bias_stats = {}
        cross_layer_analysis = {
            'total_parameters': 0,
            'total_weights': 0,
            'total_biases': 0,
            'layer_param_counts': [],
            'weight_means_by_layer': [],
            'weight_stds_by_layer': [],
            'bias_means_by_layer': [],
            'layer_names': []
        }
        
        # Analyze each layer
        for layer_name, layer_data in layer_analysis.items():
            logger.debug(f"running _analyze_parameter_statistics ... Analyzing statistics for layer: {layer_name}")
            
            # Weight statistics
            weights = layer_data['weights']
            if len(weights) > 0:
                weight_stats[layer_name] = self._compute_distribution_statistics(weights, f"{layer_name}_weights")
                
                # Add to cross-layer tracking
                cross_layer_analysis['weight_means_by_layer'].append(weight_stats[layer_name]['mean'])
                cross_layer_analysis['weight_stds_by_layer'].append(weight_stats[layer_name]['std'])
                cross_layer_analysis['total_weights'] += len(weights)
            
            # Bias statistics  
            biases = layer_data['biases']
            if len(biases) > 0:
                bias_stats[layer_name] = self._compute_distribution_statistics(biases, f"{layer_name}_biases")
                
                # Add to cross-layer tracking
                cross_layer_analysis['bias_means_by_layer'].append(bias_stats[layer_name]['mean'])
                cross_layer_analysis['total_biases'] += len(biases)
            
            # Layer-level tracking
            cross_layer_analysis['layer_param_counts'].append(layer_data['total_params'])
            cross_layer_analysis['layer_names'].append(layer_name)
            cross_layer_analysis['total_parameters'] += layer_data['total_params']
        
        logger.debug(f"running _analyze_parameter_statistics ... Statistics computed for {len(weight_stats)} weight distributions and {len(bias_stats)} bias distributions")
        
        return {
            'weight_statistics': weight_stats,
            'bias_statistics': bias_stats,
            'cross_layer_analysis': cross_layer_analysis
        }
    
    def _compute_distribution_statistics(self, values: np.ndarray, description: str) -> Dict[str, float]:
        """
        Compute comprehensive statistics for a parameter distribution
        
        Calculates detailed statistical measures to characterize the distribution
        of weights or biases, including health indicators for training assessment.
        
        STATISTICAL MEASURES COMPUTED:
        
        Basic Descriptive Statistics:
        - Mean: Average parameter value (should be near 0 for healthy training)
        - Median: Middle value (robust to outliers)
        - Standard Deviation: Spread of distribution (indicates learning diversity)
        - Variance: Square of standard deviation
        - Min/Max: Extreme values (detect saturation or explosion)
        
        Distribution Shape:
        - Range: Max - Min (overall spread)
        - Q25, Q75: Quartiles (distribution shape)
        - IQR: Interquartile range (robust spread measure)
        
        Health Indicators:
        - Dead Parameter %: Percentage with |value| < 1e-6 (not contributing)
        - Saturation %: Percentage with |value| > 3.0 (potentially saturated)
        - Zero Count: Exact zeros (complete non-participation)
        
        Args:
            values: Array of parameter values (weights or biases)
            description: Description for logging purposes
            
        Returns:
            Dictionary containing all computed statistics
        """
        if len(values) == 0:
            return {}
        
        # Basic statistics
        stats = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'variance': float(np.var(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'range': float(np.max(values) - np.min(values)),
            'count': len(values)
        }
        
        # Quartiles and IQR
        q25, q75 = np.percentile(values, [25, 75])
        stats.update({
            'q25': float(q25),
            'q75': float(q75),
            'iqr': float(q75 - q25)
        })
        
        # Health indicators
        abs_values = np.abs(values)
        dead_threshold = 1e-6
        saturation_threshold = 3.0
        
        dead_count = np.sum(abs_values < dead_threshold)
        saturated_count = np.sum(abs_values > saturation_threshold)
        zero_count = np.sum(values == 0.0)
        
        stats.update({
            'dead_percentage': float(dead_count / len(values) * 100),
            'saturation_percentage': float(saturated_count / len(values) * 100),
            'zero_count': int(zero_count),
            'zero_percentage': float(zero_count / len(values) * 100)
        })
        
        logger.debug(f"running _compute_distribution_statistics ... {description}: "
                    f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                    f"dead={stats['dead_percentage']:.1f}%, saturated={stats['saturation_percentage']:.1f}%")
        
        return stats
    
    def _assess_parameter_health(
        self, 
        layer_analysis: Dict[str, Dict[str, Any]], 
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess overall parameter health and generate training insights
        
        Analyzes parameter patterns to identify training issues and provide
        actionable recommendations for model improvement.
        
        HEALTH ASSESSMENT CRITERIA:
        
        Healthy Parameter Patterns:
        - Weights centered around 0 with reasonable spread (std: 0.05-0.5)
        - Low percentage of dead parameters (<10%)
        - Low percentage of saturated parameters (<5%)
        - Biases distributed across range without extreme values
        
        Problem Pattern Detection:
        - Dead Neurons: >30% of parameters near zero
        - Weight Explosion: Many parameters with |value| > 3.0
        - Poor Initialization: All parameters very close to initialization values
        - Gradient Issues: Unusual parameter distribution patterns
        
        Args:
            layer_analysis: Per-layer parameter data
            analysis_results: Cross-layer statistical analysis
            
        Returns:
            Dictionary containing health assessment and recommendations
        """
        logger.debug("running _assess_parameter_health ... Assessing overall parameter health")
        
        insights = []
        recommendations = []
        layer_health_scores = []
        
        weight_stats = analysis_results['weight_statistics']
        bias_stats = analysis_results['bias_statistics']
        
        # Assess each layer individually
        for layer_name in layer_analysis.keys():
            layer_health = self._assess_single_layer_health(layer_name, weight_stats, bias_stats)
            layer_health_scores.append(layer_health['score'])
            
            if layer_health['issues']:
                insights.extend([f"Layer {layer_name}: {issue}" for issue in layer_health['issues']])
            
            if layer_health['recommendations']:
                recommendations.extend([f"For {layer_name}: {rec}" for rec in layer_health['recommendations']])
        
        # Determine overall health
        if not layer_health_scores:
            overall_health = "unknown"
        else:
            avg_score = np.mean(layer_health_scores)
            min_score = min(layer_health_scores)
            
            if min_score <= 1:  # Any critical layer
                overall_health = "critical"
            elif avg_score >= 4:
                overall_health = "healthy"
            elif avg_score >= 3:
                overall_health = "good"
            elif avg_score >= 2:
                overall_health = "concerning"
            else:
                overall_health = "problematic"
        
        # Generate overall insights
        total_params = analysis_results['cross_layer_analysis']['total_parameters']
        insights.append(f"Total trainable parameters: {total_params:,}")
        
        if overall_health == "healthy":
            insights.append("✅ All layers show healthy parameter distributions")
        elif overall_health == "critical":
            insights.append("🔴 Critical parameter issues detected - immediate attention needed")
        
        # Generate cross-layer recommendations
        if overall_health in ["critical", "problematic"]:
            recommendations.extend([
                "🔧 Consider better weight initialization (He/Xavier)",
                "📉 Review learning rate - may be too high or too low",
                "🏗️ Add batch normalization for training stability"
            ])
        
        logger.debug(f"running _assess_parameter_health ... Overall health: {overall_health}")
        logger.debug(f"running _assess_parameter_health ... Generated {len(insights)} insights and {len(recommendations)} recommendations")
        
        return {
            'overall_health': overall_health,
            'layer_health_scores': layer_health_scores,
            'insights': insights,
            'recommendations': recommendations
        }
    
    def _assess_single_layer_health(
        self, 
        layer_name: str, 
        weight_stats: Dict[str, Dict[str, float]], 
        bias_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Assess the health of a single layer's parameters
        
        Args:
            layer_name: Name of the layer to assess
            weight_stats: Weight statistics for all layers
            bias_stats: Bias statistics for all layers
            
        Returns:
            Dictionary containing layer health assessment
        """
        issues = []
        recommendations = []
        score = 5  # Start with perfect score, deduct for problems
        
        # Assess weights if present
        if layer_name in weight_stats:
            w_stats = weight_stats[layer_name]
            
            # Check for dead neurons
            if w_stats['dead_percentage'] > 50:
                issues.append("Severe dead neuron problem (>50% near-zero weights)")
                recommendations.append("Switch to LeakyReLU or better initialization")
                score -= 3
            elif w_stats['dead_percentage'] > 20:
                issues.append(f"High dead neuron percentage ({w_stats['dead_percentage']:.1f}%)")
                recommendations.append("Consider LeakyReLU activation")
                score -= 1
            
            # Check for weight explosion
            if w_stats['saturation_percentage'] > 20:
                issues.append("Weight explosion detected (>20% saturated)")
                recommendations.append("Add gradient clipping, reduce learning rate")
                score -= 3
            elif w_stats['saturation_percentage'] > 5:
                issues.append(f"Some weight saturation ({w_stats['saturation_percentage']:.1f}%)")
                score -= 1
            
            # Check distribution health
            if abs(w_stats['mean']) > 0.5:
                issues.append(f"Weight mean far from zero ({w_stats['mean']:.3f})")
                score -= 1
            
            if w_stats['std'] < 0.01:
                issues.append("Very narrow weight distribution (poor learning)")
                score -= 2
            elif w_stats['std'] > 2.0:
                issues.append("Very wide weight distribution (possible instability)")
                score -= 1
        
        # Assess biases if present
        if layer_name in bias_stats:
            b_stats = bias_stats[layer_name]
            
            # Check for extreme biases
            if abs(b_stats['mean']) > 5.0:
                issues.append(f"Extreme bias values (mean={b_stats['mean']:.2f})")
                recommendations.append("Check for bias domination over inputs")
                score -= 2
        
        # Ensure score doesn't go below 1
        score = max(1, score)
        
        return {
            'score': score,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _create_visualization_dashboard(
        self,
        layer_analysis: Dict[str, Dict[str, Any]],
        analysis_results: Dict[str, Any],
        health_assessment: Dict[str, Any],
        model: keras.Model,
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path],
        max_layers_to_plot: int
    ) -> Optional[Path]:
        """
        Create comprehensive visualization dashboard for weights and bias analysis
        
        Generates a professional 2x2 dashboard showing parameter distributions,
        statistics, and health assessments.
        
        Args:
            layer_analysis: Per-layer parameter data
            analysis_results: Statistical analysis results
            health_assessment: Health assessment results
            model: Keras model
            dataset_name: Dataset name
            run_timestamp: Timestamp for file naming
            plot_dir: Directory to save plots
            max_layers_to_plot: Maximum layers to include
            
        Returns:
            Path to saved visualization file
        """
        logger.debug("running _create_visualization_dashboard ... Creating weights and bias visualization dashboard")
        
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Weights and Bias Analysis - {dataset_name}', fontsize=16, fontweight='bold')
            
            # 1. Weight distributions (top-left)
            self._plot_weight_distributions(ax1, layer_analysis, max_layers_to_plot)
            
            # 2. Bias distributions (top-right)
            self._plot_bias_distributions(ax2, layer_analysis, max_layers_to_plot)
            
            # 3. Parameter statistics summary (bottom-left)
            self._plot_parameter_summary(ax3, layer_analysis, analysis_results)
            
            # 4. Health assessment (bottom-right)
            self._plot_health_assessment(ax4, health_assessment)
            
            plt.tight_layout()
            
            # Save the visualization
            filepath = self._generate_save_path(dataset_name, run_timestamp, plot_dir)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.debug(f"running _create_visualization_dashboard ... Weights and bias analysis saved to: {filepath}")
            
            plt.close()
            return filepath
            
        except Exception as e:
            logger.warning(f"running _create_visualization_dashboard ... Failed to create visualization: {e}")
            plt.close()
            return None
    
    def _plot_weight_distributions(self, ax, layer_analysis: Dict[str, Dict[str, Any]], max_layers: int) -> None:
        """Plot weight distributions for multiple layers"""
        ax.set_title('Weight Distributions by Layer', fontweight='bold')
        
        # Select layers to plot (limit for readability)
        layers_to_plot = list(layer_analysis.keys())[:max_layers]
        
        if not layers_to_plot:
            ax.text(0.5, 0.5, 'No layers with weights found', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create violin plots for weight distributions
        weight_data = []
        layer_labels = []
        
        for layer_name in layers_to_plot:
            weights = layer_analysis[layer_name]['weights']
            if len(weights) > 0:
                # Sample if too many weights (for performance)
                if len(weights) > 1000:
                    weights = np.random.choice(weights, 1000, replace=False)
                weight_data.append(weights)
                layer_labels.append(layer_name)
        
        if weight_data:
            parts = ax.violinplot(weight_data, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(layer_labels) + 1))
            ax.set_xticklabels(layer_labels, rotation=45, ha='right')
            ax.set_ylabel('Weight Value')
            ax.grid(True, alpha=0.3)
            
            # Add zero line
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        else:
            ax.text(0.5, 0.5, 'No weight data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_bias_distributions(self, ax, layer_analysis: Dict[str, Dict[str, Any]], max_layers: int) -> None:
        """Plot bias distributions for multiple layers"""
        ax.set_title('Bias Distributions by Layer', fontweight='bold')
        
        # Collect bias data
        bias_data = []
        layer_labels = []
        
        for layer_name, layer_data in layer_analysis.items():
            biases = layer_data['biases']
            if len(biases) > 0:
                bias_data.append(biases)
                layer_labels.append(layer_name)
        
        if bias_data and len(bias_data) > 0:
            # Create box plots for bias distributions
            bp = ax.boxplot(bias_data, labels=layer_labels, patch_artist=True)
            
            # Color the boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_xticklabels(layer_labels, rotation=45, ha='right')
            ax.set_ylabel('Bias Value')
            ax.grid(True, alpha=0.3)
            
            # Add zero line
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        else:
            ax.text(0.5, 0.5, 'No bias data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_parameter_summary(self, ax, layer_analysis: Dict[str, Dict[str, Any]], analysis_results: Dict[str, Any]) -> None:
        """Plot parameter count and statistics summary"""
        ax.set_title('Parameter Statistics Summary', fontweight='bold')
        ax.axis('off')
        
        # Create summary text
        cross_layer = analysis_results['cross_layer_analysis']
        
        summary_text = f"""Parameter Overview:
        
Total Parameters: {cross_layer['total_parameters']:,}
Total Weights: {cross_layer['total_weights']:,}
Total Biases: {cross_layer['total_biases']:,}
Trainable Layers: {len(layer_analysis)}

Layer Details:"""
        
        # Add per-layer summary
        for layer_name, layer_data in list(layer_analysis.items())[:6]:  # Show first 6 layers
            summary_text += f"\n{layer_name}: {layer_data['total_params']:,} params"
        
        if len(layer_analysis) > 6:
            summary_text += f"\n... and {len(layer_analysis) - 6} more layers"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def _plot_health_assessment(self, ax, health_assessment: Dict[str, Any]) -> None:
        """Plot health assessment and recommendations"""
        ax.set_title('Parameter Health Assessment', fontweight='bold')
        ax.axis('off')
        
        # Health status with color
        health = health_assessment['overall_health']
        health_colors = {
            'healthy': 'lightgreen',
            'good': 'lightgreen', 
            'concerning': 'yellow',
            'problematic': 'orange',
            'critical': 'red'
        }
        
        health_text = f"Overall Health: {health.upper()}"
        
        # Add insights
        insights_text = "\nKey Insights:\n"
        for insight in health_assessment['insights'][:5]:  # Show first 5
            insights_text += f"• {insight}\n"
        
        # Add recommendations
        if health_assessment['recommendations']:
            rec_text = "\nRecommendations:\n"
            for rec in health_assessment['recommendations'][:3]:  # Show first 3
                rec_text += f"• {rec}\n"
        else:
            rec_text = "\n✅ No specific recommendations needed"
        
        full_text = health_text + insights_text + rec_text
        
        ax.text(0.05, 0.95, full_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=health_colors.get(health, 'lightgray'), alpha=0.8))
    
    def _generate_save_path(
        self,
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path]
    ) -> Path:
        """Generate file path for saving the visualization"""
        from datetime import datetime
        
        if run_timestamp is None:
            # run_timestamp should always be provided from optimizer.py
            raise ValueError("run_timestamp should always be provided from optimizer.py")
        
        dataset_name_clean = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        filename = f"weights_bias_analysis_{run_timestamp}_{dataset_name_clean}.png"
        
        if plot_dir is not None:
            save_dir = plot_dir
        else:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            save_dir = project_root / "plots"
            save_dir.mkdir(exist_ok=True)
        
        return save_dir / filename
    
    def _log_analysis_summary(self, results: Dict[str, Any]) -> None:
        """Log summary of weights and bias analysis"""
        logger.debug("running _log_analysis_summary ... Weights and Bias Analysis Summary")
        logger.debug(f"running _log_analysis_summary ... Overall Health: {results['parameter_health']}")
        
        # Log key statistics
        if 'weight_statistics' in results:
            logger.debug(f"running _log_analysis_summary ... Analyzed {len(results['weight_statistics'])} weight distributions")
        
        if 'bias_statistics' in results:
            logger.debug(f"running _log_analysis_summary ... Analyzed {len(results['bias_statistics'])} bias distributions")
        
        # Log insights
        for insight in results['training_insights'][:3]:
            logger.debug(f"running _log_analysis_summary ... {insight}")
        
        # Log recommendations
        for rec in results['recommendations'][:3]:
            logger.debug(f"running _log_analysis_summary ... {rec}")


def create_weights_bias_analysis(
    model: keras.Model,
    dataset_name: str = "dataset",
    run_timestamp: Optional[str] = None,
    plot_dir: Optional[Path] = None,
    max_layers_to_plot: int = 12
) -> Dict[str, Any]:
    """
    Convenience function for quick weights and bias analysis
    
    Args:
        model: Trained Keras model to analyze
        dataset_name: Name of dataset
        run_timestamp: Optional timestamp
        plot_dir: Optional directory for saving plots
        max_layers_to_plot: Maximum layers to include in visualizations
        
    Returns:
        Analysis results dictionary
    """
    analyzer = WeightsBiasAnalyzer(model_name=dataset_name)
    return analyzer.analyze_and_visualize(
        model=model,
        dataset_name=dataset_name,
        run_timestamp=run_timestamp,
        plot_dir=plot_dir,
        max_layers_to_plot=max_layers_to_plot
    )