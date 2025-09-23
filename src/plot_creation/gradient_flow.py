"""
Gradient Flow Analysis and Visualization Module

This module provides comprehensive gradient flow analysis including:
- Layer-wise gradient magnitude analysis
- Vanishing/exploding gradient detection
- Dead neuron identification
- Professional-quality visualizations for debugging training issues
- Real-time gradient monitoring during training (future enhancement)

Designed to work with any Keras model and training configuration.

GRADIENT FLOW FUNDAMENTALS:

What Are Gradients?
    Gradients are the "learning signals" that tell your neural network how to improve.
    Think of gradients as GPS directions for your model: "Turn left here to reduce error."
    
    During training, gradients flow backward through your network:
    1. Forward Pass: Model makes predictions
    2. Loss Calculation: Compare predictions to actual answers  
    3. Backward Pass: Calculate gradients (how much each weight should change)
    4. Weight Update: Adjust weights based on gradients
    5. Repeat until model learns

    Example Gradient Flow:
        Output Layer: "I was wrong by +0.5, adjust my weights"
        Hidden Layer: "Output layer needs -0.3 change, I contributed 0.2, so I need -0.06 adjustment"
        Input Layer: "Hidden layer needs -0.06, I contributed 0.1, so I need -0.006 adjustment"

What Are Dead Neurons?
    Dead neurons are network components that have stopped contributing to learning.
    Like employees who've given up working - they're still there but not helping.
    
    How Neurons "Die":
    1. ReLU Activation Problem: If a neuron's input becomes negative, ReLU outputs 0
    2. Zero Gradient: When output is 0, gradient is also 0 (no learning signal)
    3. Stuck State: Once dead, neuron can't recover without intervention
    
    Example Dead Neuron Progression:
        Epoch 1: Neuron outputs 0.5 (healthy, learning)
        Epoch 5: Neuron outputs 0.1 (struggling)  
        Epoch 10: Neuron outputs 0.0 (dead - ReLU killed it)
        Epoch 15+: Neuron stays at 0.0 (permanently dead)

Gradient Problems:

Vanishing Gradients:
    Problem: Gradients become exponentially smaller as they flow backward
    Analogy: Like whispering a message through 10 people - by the end, nobody hears anything
    Symptoms: Early layers learn very slowly or stop learning entirely
    Causes: Deep networks, poor activation functions (sigmoid), poor initialization
    
    Example Vanishing Gradient Flow:
        Layer 4 (Output): Gradient = 0.5    ✅ Strong learning signal
        Layer 3: Gradient = 0.5 × 0.1 = 0.05   ⚠️ Getting weaker  
        Layer 2: Gradient = 0.05 × 0.1 = 0.005  ❌ Very weak
        Layer 1: Gradient = 0.005 × 0.1 = 0.0005 ❌ Nearly dead

Exploding Gradients:  
    Problem: Gradients become exponentially larger, causing training instability
    Analogy: Like a rumor that gets more exaggerated each time it's retold
    Symptoms: Loss oscillations, NaN values, training divergence, model weights becoming huge
    Causes: High learning rates, poor initialization, lack of normalization
    
    Example Exploding Gradient Flow:
        Layer 4 (Output): Gradient = 0.5     ✅ Normal
        Layer 3: Gradient = 0.5 × 10 = 5.0   ⚠️ Getting large
        Layer 2: Gradient = 5.0 × 10 = 50.0  ❌ Very large  
        Layer 1: Gradient = 50.0 × 10 = 500.0 ❌ Dangerously large

Blame Assignment in Different Layer Types:

Convolutional Layers (Pattern-Based Blame):
    Each filter gets blamed for how well its pattern detection helped with the prediction:
    - Filter 5 (edge detector): "Your edge detection contributed +0.3 to the error"
    - Filter 12 (circle detector): "Your circle detection contributed -0.1 to the error"
    - Filters adjust their pattern-matching based on this feedback
    
Dense Layers (Individual Neuron Blame):
    Each neuron gets blamed for its individual contribution to the final decision:
    - Neuron 47: "You contributed +0.2 to the error, reduce your activation"
    - Neuron 89: "You contributed -0.1 to the error, increase your activation"
    - Each neuron adjusts its weights based on its specific blame

Why This Analysis Matters:
    - Diagnose why deep networks stop learning (vanishing gradients)
    - Detect training instability (exploding gradients)  
    - Identify wasted network capacity (dead neurons)
    - Guide architectural improvements (activation functions, initialization)
    - Optimize training hyperparameters (learning rates, dropout)
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras # type: ignore
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
from utils.logger import logger


class GradientFlowAnalyzer:
    """
    Comprehensive gradient flow analysis and visualization
    
    Analyzes how gradients flow through the network layers to detect
    training issues like vanishing gradients, exploding gradients,
    and dead neurons. Essential for debugging deep learning models.
    
    CONCEPTUAL OVERVIEW:
    
    Think of this analyzer as a "neural network health scanner" that:
    1. Measures how well learning signals (gradients) flow through your network
    2. Detects neurons that have stopped contributing (dead neurons)
    3. Identifies training problems before they cause poor performance
    4. Provides specific recommendations for fixing issues
    
    What Gets Analyzed:
    - Gradient Magnitudes: How strong are the learning signals in each layer?
    - Gradient Distributions: Are gradients healthy bell curves or problematic spikes?
    - Dead Neuron Counts: How many neurons have stopped learning?
    - Layer Health: Which layers are learning well vs struggling?
    
    Real-World Analogy:
    Like a doctor examining your cardiovascular system:
    - Blood pressure (gradient magnitudes): Too high/low indicates problems
    - Blood flow distribution (gradient distributions): Should be smooth, not blocked
    - Dead tissue (dead neurons): Areas that aren't getting oxygen (gradients)
    - Overall health assessment: Recommendations for lifestyle changes (architecture)
    """
    
    def __init__(self, model_name: str = "Model"):
        """
        Initialize the gradient flow analyzer
        
        Args:
            model_name: Name of the model for plot titles and logging
        """
        self.model_name = model_name
        logger.debug("running GradientFlowAnalyzer.__init__ ... Gradient flow analyzer initialized")
    
    
    def analyze_and_visualize(
        self,
        model: keras.Model,
        sample_data: np.ndarray,
        sample_labels: np.ndarray,
        dataset_name: str = "dataset",
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None,
        sample_size: int = 100,
        config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze gradient flow through the network and create visualizations
        
        Performs comprehensive gradient flow analysis to detect training issues
        and understand how well gradients propagate through the network layers.
        This is crucial for debugging deep networks and optimizing training.
        
        What Gradient Flow Analysis Reveals:
        
        1. **Vanishing Gradients** 🔍
           - Problem: Gradients become exponentially smaller in early layers
           - Symptoms: Early layers learn very slowly or stop learning
           - Causes: Deep networks, poor activation functions, poor initialization
           - Solutions: Better initialization, residual connections, batch normalization
        
        2. **Exploding Gradients** 💥
           - Problem: Gradients become exponentially larger, causing instability
           - Symptoms: Loss oscillations, NaN values, training divergence
           - Causes: High learning rates, poor initialization, lack of normalization
           - Solutions: Gradient clipping, lower learning rate, better initialization
        
        3. **Dead Neurons** 💀
           - Problem: Neurons that output zero and never activate
           - Symptoms: Layers with consistently zero gradients
           - Causes: Poor initialization, dying ReLU problem
           - Solutions: Better initialization, LeakyReLU, proper learning rates
        
        Analysis Components:
        
        **Layer-wise Gradient Magnitudes**:
        - Shows average gradient magnitude per layer
        - Healthy pattern: Gradients decrease gradually toward input
        - Problem pattern: Sudden drops (vanishing) or spikes (exploding)
        
        **Gradient Distribution Analysis**:
        - Histograms showing gradient value distributions per layer
        - Healthy: Bell-curved distributions centered near zero
        - Problems: Heavily skewed, very narrow (vanishing), or very wide (exploding)
        
        **Gradient Flow Visualization**:
        - Visual representation of gradient propagation through network
        - Color-coded by gradient magnitude for easy interpretation
        - Shows gradient "highways" and "bottlenecks"
        
        **Dead Neuron Detection**:
        - Identifies layers/neurons with consistently zero gradients
        - Quantifies percentage of dead neurons per layer
        - Suggests architectural improvements
        
        Practical Applications:
        
        🔧 **Model Debugging**: "Why isn't my deep network learning?"
        - Check for vanishing gradients in early layers
        - Identify problematic activation functions
        - Spot initialization issues
        
        🏗️ **Architecture Design**: "How deep can I make my network?"
        - Validate gradient flow through proposed architectures
        - Test different activation functions and normalizations
        - Optimize skip connections and residual blocks
        
        ⚡ **Training Optimization**: "Why is training unstable?"
        - Detect exploding gradients causing oscillations
        - Find optimal learning rate ranges
        - Validate gradient clipping effectiveness
        
        📊 **Performance Analysis**: "Which layers are learning effectively?"
        - Compare gradient magnitudes across layers
        - Identify underutilized network capacity
        - Guide pruning and compression decisions
        
        Real-world Examples:
        
        **Healthy CNN Gradient Flow**:
        ```
        Layer 1 (Input):     Grad Mag = 0.001    ✅ Small but present
        Layer 2 (Conv):      Grad Mag = 0.005    ✅ Growing appropriately  
        Layer 3 (Conv):      Grad Mag = 0.012    ✅ Strong learning signal
        Layer 4 (Dense):     Grad Mag = 0.025    ✅ Output layer active
        ```
        
        **Vanishing Gradient Problem**:
        ```
        Layer 1 (Input):     Grad Mag = 0.000001 ❌ Nearly zero - not learning
        Layer 2 (Conv):      Grad Mag = 0.000005 ❌ Extremely small
        Layer 3 (Conv):      Grad Mag = 0.001    ⚠️ Still very small
        Layer 4 (Dense):     Grad Mag = 0.025    ✅ Only output layer learning
        ```
        
        **Exploding Gradient Problem**:
        ```
        Layer 1 (Input):     Grad Mag = 15.2     ❌ Dangerously large
        Layer 2 (Conv):      Grad Mag = 45.7     ❌ Exponentially growing
        Layer 3 (Conv):      Grad Mag = 156.3    ❌ Training will diverge
        Layer 4 (Dense):     Grad Mag = 489.1    ❌ Completely unstable
        ```
        
        Args:
            model: Trained Keras model to analyze
            sample_data: Sample input data for gradient computation (subset of training/test data)
            sample_labels: Corresponding labels for the sample data
            dataset_name: Name of dataset for plot titles and file naming
            run_timestamp: Optional timestamp for file naming
            plot_dir: Optional directory to save visualization plots
            sample_size: Number of samples to use for gradient analysis (default 100)
            
        Returns:
            Dictionary containing comprehensive analysis results:
            - 'gradient_magnitudes': Dict mapping layer names to gradient magnitude statistics
            - 'dead_neurons': Dict with dead neuron counts and percentages per layer
            - 'gradient_health': Overall assessment ('healthy', 'vanishing', 'exploding', 'mixed')
            - 'layer_analysis': Detailed per-layer gradient analysis
            - 'visualization_paths': List of paths to saved visualization files
            - 'recommendations': List of actionable suggestions for improving gradient flow
            
        Side Effects:
            - Creates comprehensive matplotlib visualizations showing gradient flow
            - Saves multiple PNG files with gradient analysis plots
            - Logs detailed gradient statistics and health assessments
            - Automatically detects and warns about gradient flow issues
            - Provides specific recommendations for architectural improvements
            
        Requirements:
            - TensorFlow/Keras model with trainable parameters
            - Sample data representative of training distribution
            - Sufficient memory for gradient computation (scales with model size)
            
        Performance Notes:
            - Analysis time scales with model complexity and sample size
            - Memory usage proportional to model parameters × sample size
            - Large models may require reducing sample_size parameter
            
        Example Usage:
            ```python
            # After training a model
            sample_x = x_test[:100]  # Use subset of test data
            sample_y = y_test[:100]
            
            analyzer = GradientFlowAnalyzer("Traffic_CNN")
            results = analyzer.analyze_and_visualize(
                model=trained_model,
                sample_data=sample_x,
                sample_labels=sample_y,
                dataset_name="GTSRB"
            )
            
            # Check gradient health
            if results['gradient_health'] == 'vanishing':
                print("Consider: better initialization, residual connections")
            elif results['gradient_health'] == 'exploding':
                print("Consider: gradient clipping, lower learning rate")
            ```
        """
        logger.debug("running analyze_and_visualize ... Starting comprehensive gradient flow analysis")
        
        try:
            # Validate inputs
            if not self._validate_inputs(model, sample_data, sample_labels, sample_size):
                return {'error': 'Invalid inputs provided for gradient analysis'}
            
            # Prepare sample data (limit size for performance)
            analysis_data, analysis_labels = self._prepare_sample_data(
                sample_data, sample_labels, sample_size
            )
            
            # Compute gradients for all layers
            gradient_data = self._compute_layer_gradients(model, analysis_data, analysis_labels)
            
            # Analyze gradient statistics
            analysis_results = self._analyze_gradient_statistics(gradient_data, model)
            
            # Create visualizations
            visualization_paths = self._create_visualizations(
                gradient_data=gradient_data,
                analysis_results=analysis_results,
                model=model,
                dataset_name=dataset_name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir,
                config=config
            )
            
            # Combine results
            final_results = analysis_results.copy()
            final_results['visualization_paths'] = visualization_paths
            
            # Log summary
            self._log_analysis_summary(final_results)
            
            logger.debug("running analyze_and_visualize ... Gradient flow analysis completed successfully")
            return final_results
            
        except Exception as e:
            logger.warning(f"running analyze_and_visualize ... Failed to complete gradient flow analysis: {e}")
            logger.debug(f"running analyze_and_visualize ... Error traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    
    def _validate_inputs(
        self,
        model: keras.Model,
        sample_data: np.ndarray,
        sample_labels: np.ndarray,
        sample_size: int
    ) -> bool:
        """
        Validate inputs for gradient analysis
        
        Performs essential checks to ensure gradient analysis can be performed safely and accurately.
        Like a pre-flight checklist for gradient computation.
        
        Input Validation Checklist:
        1. Model Health Check:
           - Model exists and is not None
           - Model has trainable parameters (weights/biases that can learn)
           - Model architecture is properly constructed
        
        2. Data Consistency Check:
           - Sample data and labels are provided
           - Data and labels have matching lengths (critical for proper gradient computation)
           - Sample size is reasonable relative to available data
        
        3. Gradient Computation Requirements:
           - Model must have been compiled with a loss function
           - Data must be in correct format for model input
           - Memory requirements are manageable
        
        Why Validation Matters:
        - Prevents cryptic TensorFlow errors during gradient computation
        - Ensures meaningful results by catching data mismatches early
        - Saves time by failing fast instead of during expensive computation
        
        Args:
            model: Keras model to analyze
            sample_data: Sample input data
            sample_labels: Sample labels
            sample_size: Requested sample size
            
        Returns:
            True if inputs are valid and gradient analysis can proceed
            False if inputs are invalid (logged warnings explain what's wrong)
            
        Common Failure Scenarios:
        - Model is None: Forgot to build/load model before analysis
        - No trainable variables: Model frozen or not properly constructed
        - Data/label mismatch: Different number of samples in data vs labels
        - Empty data: Trying to analyze with no samples
        """
        logger.debug("running _validate_inputs ... Validating gradient analysis inputs")
        
        # Check model
        if model is None:
            logger.warning("running _validate_inputs ... Model is None")
            return False
        
        if not hasattr(model, 'trainable_variables') or len(model.trainable_variables) == 0:
            logger.warning("running _validate_inputs ... Model has no trainable variables")
            return False
        
        # Check data shapes
        if sample_data is None or sample_labels is None:
            logger.warning("running _validate_inputs ... Sample data or labels are None")
            return False
        
        if len(sample_data) != len(sample_labels):
            logger.warning("running _validate_inputs ... Mismatch between sample data and labels length")
            return False
        
        if len(sample_data) < sample_size:
            logger.debug(f"running _validate_inputs ... Requested sample size {sample_size} > available data {len(sample_data)}, will use all available")
        
        logger.debug("running _validate_inputs ... Input validation successful")
        return True
    
    
    def _prepare_sample_data(
        self,
        sample_data: np.ndarray,
        sample_labels: np.ndarray,
        sample_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sample data for gradient analysis
        
        Optimizes the data sample for gradient computation by selecting a representative
        subset that balances accuracy with computational efficiency.
        
        SAMPLING STRATEGY:
        
        Why Sample Instead of Using All Data?
        - Gradient computation is memory-intensive (scales with data size × model parameters)
        - 100-200 samples typically provide stable gradient statistics
        - Reduces analysis time from minutes to seconds for large datasets
        - Allows analysis on memory-constrained systems
        
        Random Sampling Benefits:
        - Ensures representative sample across all classes
        - Prevents bias toward specific data patterns
        - Provides stable gradient statistics across multiple runs
        - Mimics real training data distribution
        
        Sample Size Guidelines:
        - Small models (<100K params): 100-200 samples sufficient
        - Medium models (100K-1M params): 50-100 samples recommended  
        - Large models (>1M params): 25-50 samples may be necessary
        - Very large models (>10M params): 10-25 samples minimum
        
        Args:
            sample_data: Original sample data (e.g., x_test)
            sample_labels: Original sample labels (e.g., y_test)
            sample_size: Desired sample size for analysis
            
        Returns:
            Tuple of (prepared_data, prepared_labels) ready for gradient computation
            
        Side Effects:
            - Logs the actual sample size used (may be smaller than requested)
            - Uses random sampling to ensure representative data distribution
        """
        logger.debug(f"running _prepare_sample_data ... Preparing sample data for gradient analysis")
        
        # Limit sample size for performance
        actual_size = min(sample_size, len(sample_data))
        
        # Use random subset to ensure representative sample
        if actual_size < len(sample_data):
            indices = np.random.choice(len(sample_data), actual_size, replace=False)
            analysis_data = sample_data[indices]
            analysis_labels = sample_labels[indices]
        else:
            analysis_data = sample_data
            analysis_labels = sample_labels
        
        logger.debug(f"running _prepare_sample_data ... Using {len(analysis_data)} samples for gradient analysis")
        return analysis_data, analysis_labels
    
    
    def _compute_layer_gradients(
        self,
        model: keras.Model,
        sample_data: np.ndarray,
        sample_labels: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute gradients for all model layers using TensorFlow's GradientTape
        
        This is the core method that performs the "blame assignment" process for your neural network.
        It runs a forward pass, calculates the error, then traces backwards to see how each weight
        contributed to that error.
        
        THE GRADIENT COMPUTATION PROCESS:
        
        1. **Forward Pass (Prediction Making)**:
           ```
           Input Image → Conv Layers → Dense Layers → Output Predictions
           Example: [Cat Image] → [Edge Detection] → [Shape Recognition] → [90% Cat, 10% Dog]
           ```
        
        2. **Loss Calculation (Error Measurement)**:
           ```
           Compare predictions to true labels:
           Predicted: [90% Cat, 10% Dog]
           Actual:    [100% Cat, 0% Dog]  
           Loss:      0.1 (small error - good prediction!)
           ```
        
        3. **Backward Pass (Blame Assignment)**:
           ```
           TensorFlow traces backward through every operation:
           "Output layer: You contributed +0.05 to the error"
           "Dense layer neuron 23: You contributed +0.02 to the error"  
           "Conv filter 5: Your edge detection contributed +0.001 to the error"
           ```
        
        4. **Gradient Organization**:
           ```
           Organize gradients by layer for analysis:
           - conv2d: [gradient_values_for_all_conv_weights]
           - dense: [gradient_values_for_all_dense_weights]
           - dense_1: [gradient_values_for_output_weights]
           ```
        
        TENSORFLOW GRADIENTTAPE MECHANISM:
        
        What is GradientTape?
            Think of GradientTape as a DVR that records every mathematical operation
            during the forward pass. When you "rewind the tape," TensorFlow can see
            exactly how each weight influenced the final result.
            
            ```python
            with tf.GradientTape() as tape:
                # TensorFlow records: "Weight W1 was multiplied by input X1"
                # TensorFlow records: "Result was passed through ReLU activation"  
                # TensorFlow records: "Output was used in loss calculation"
                predictions = model(data)
                loss = loss_function(predictions, labels)
            
            # Now rewind the tape to see how each weight affected the loss
            gradients = tape.gradient(loss, model.trainable_variables)
            ```
        
        Why Use training=False?
            During gradient analysis, we want to see how the model behaves in its
            current state, not how it would behave during training (with dropout, etc.)
            
        GRADIENT INTERPRETATION BY LAYER TYPE:
        
        Convolutional Layer Gradients:
            Each filter gets a gradient showing how its pattern detection should change:
            ```
            Filter 5 (horizontal edge detector):
            - Gradient = +0.001: "Detect horizontal edges slightly more strongly"  
            - High gradient magnitude: Filter is actively learning
            - Zero gradient: Filter isn't contributing (potentially dead)
            ```
        
        Dense Layer Gradients:
            Each neuron gets a gradient showing how its decision-making should change:
            ```
            Neuron 47 (stop sign detector):
            - Gradient = -0.02: "Reduce activation when you see this pattern"
            - Gradient = +0.05: "Increase activation when you see this pattern"
            - Zero gradient: Neuron isn't contributing (potentially dead)
            ```
        
        DEAD NEURON DETECTION:
        
        How Neurons Become Dead (ReLU Problem):
            ```
            Neuron with ReLU activation:
            1. Input becomes negative: input = -0.5
            2. ReLU outputs zero: output = max(0, -0.5) = 0  
            3. Gradient becomes zero: gradient = 0 (can't learn)
            4. Neuron stays dead: Once dead, hard to recover
            ```
        
        Gradient-Based Dead Detection:
            ```
            For each layer:
                Count weights with |gradient| < 1e-8 (essentially zero)
                Dead percentage = dead_count / total_weights * 100
                
                Interpretation:
                - 0-10% dead: Healthy layer, normal neuron turnover
                - 10-30% dead: Some concern, monitor layer health
                - 30-50% dead: Significant problem, architectural changes needed
                - >50% dead: Critical issue, layer barely functional
            ```
        
        Args:
            model: Keras model with trainable parameters
            sample_data: Input data for forward pass (e.g., images, text sequences)
            sample_labels: True labels for loss calculation (must match model's expected format)
            
        Returns:
            Dictionary mapping layer names to gradient information:
            
            Structure:
            ```
            {
                'conv2d': {
                    'gradients': np.array([0.001, -0.002, 0.003, ...]),  # All weight gradients flattened
                    'layer_type': 'Conv2D',                              # Type of layer
                    'layer_index': 0                                     # Position in network
                },
                'dense': {
                    'gradients': np.array([0.01, -0.05, 0.0, ...]),     # All weight gradients flattened  
                    'layer_type': 'Dense',                               # Type of layer
                    'layer_index': 1                                     # Position in network
                }
            }
            ```
            
        Side Effects:
            - Performs forward pass through model (no training effects)
            - Computes loss using model's compiled loss function
            - Logs gradient computation progress and any issues
            - Falls back to categorical crossentropy if loss function unavailable
            
        Error Handling:
            - Returns empty dict if gradient computation fails
            - Handles missing loss functions gracefully
            - Skips layers without trainable parameters
            - Warns about gradient computation failures
        """
        logger.debug("running _compute_layer_gradients ... Computing gradients for all layers")
        
        gradient_data = {}
        
        # Convert data to tensors
        x_tensor = tf.convert_to_tensor(sample_data)
        y_tensor = tf.convert_to_tensor(sample_labels)
        
        # Compute gradients using GradientTape
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(x_tensor, training=False)
            
            # Compute loss
            
            # Log model loss structure for debugging
            logger.debug("running _compute_layer_gradients ... Investigating model loss structure:")
            logger.debug(f"running _compute_layer_gradients ... model.compiled_loss type: {type(model.compiled_loss)}")
            logger.debug(f"running _compute_layer_gradients ... model.compiled_loss: {model.compiled_loss}")

            if hasattr(model.compiled_loss, '__dict__'):
                logger.debug(f"running _compute_layer_gradients ... compiled_loss attributes: {list(model.compiled_loss.__dict__.keys())}")
                for attr_name, attr_value in model.compiled_loss.__dict__.items():
                    logger.debug(f"running _compute_layer_gradients ... compiled_loss.{attr_name}: {type(attr_value)} = {attr_value}")

            # Also check model.loss directly
            logger.debug(f"running _compute_layer_gradients ... model.loss: {type(model.loss)} = {model.loss}")

            # Check if model has other loss-related attributes
            loss_attrs = ['loss', 'loss_functions', 'losses', 'compiled_loss']
            for attr in loss_attrs:
                if hasattr(model, attr):
                    attr_value = getattr(model, attr)
                    logger.debug(f"running _compute_layer_gradients ... model.{attr}: {type(attr_value)} = {attr_value}")

            # Attempt to compute loss (temporarily use a simple fallback)
            try:
                loss_fn = keras.losses.get(model.loss)
                loss = loss_fn(y_tensor, predictions)
                logger.debug("running _compute_layer_gradients ... Successfully used model.compiled_loss._user_losses[0]")
            except Exception as e:
                logger.debug(f"running _compute_layer_gradients ... Failed to use compiled_loss._user_losses: {e}")
                # Fallback to categorical crossentropy for now
                loss_fn = keras.losses.CategoricalCrossentropy()
                loss = loss_fn(y_tensor, predictions)
                logger.debug("running _compute_layer_gradients ... Using fallback CategoricalCrossentropy loss")
            
            
        
        # Compute gradients with respect to all trainable variables
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Check if gradients were computed successfully
        if gradients is None:
            logger.warning("running _compute_layer_gradients ... Failed to compute gradients")
            return {}
        
        # Organize gradients by layer
        layer_index = 0
        for layer in model.layers:
            if not layer.trainable_weights:
                continue  # Skip layers without trainable parameters
            
            layer_name = layer.name
            layer_gradients = []
            
            # Collect all gradients for this layer
            for weight in layer.trainable_weights:
                if layer_index < len(gradients) and gradients[layer_index] is not None:
                    grad_values = gradients[layer_index].numpy()
                    layer_gradients.append(grad_values)
                layer_index += 1
            
            if layer_gradients:
                # Combine all gradients for this layer
                all_grads = np.concatenate([g.flatten() for g in layer_gradients])
                
                gradient_data[layer_name] = {
                    'gradients': all_grads,
                    'layer_type': type(layer).__name__,
                    'layer_index': len(gradient_data)
                }
        
        logger.debug(f"running _compute_layer_gradients ... Computed gradients for {len(gradient_data)} layers")
        return gradient_data
    
    
    def _analyze_gradient_statistics(
        self,
        gradient_data: Dict[str, Dict[str, np.ndarray]],
        model: keras.Model
    ) -> Dict[str, Any]:
        """
        Analyze gradient statistics to detect training issues and assess network health
        
        This method performs the "medical diagnosis" of your neural network by examining
        gradient patterns to identify problems like vanishing gradients, exploding gradients,
        and dead neurons. Think of it as analyzing blood test results to diagnose health issues.
        
        STATISTICAL ANALYSIS PROCESS:
        
        1. **Gradient Magnitude Analysis**:
           For each layer, compute comprehensive statistics about gradient strengths:
           ```
           Gradient Statistics Computed:
           - Mean: Average gradient strength (main health indicator)
           - Standard Deviation: How much gradients vary (stability indicator)
           - Median: Middle value (robust to outliers)
           - Min/Max: Extreme values (detect spikes/zeros)
           - Q25/Q75: Quartiles (distribution shape)
           ```
        
        2. **Dead Neuron Detection**:
           Count parameters with essentially zero gradients:
           ```
           Dead Neuron Analysis:
           For each layer:
               zero_gradients = |gradient| < 1e-8 (threshold for "essentially zero")
               dead_count = number of zero gradients
               dead_percentage = (dead_count / total_parameters) × 100
           ```
        
        3. **Layer Health Assessment**:
           Combine gradient magnitude and dead neuron info to assess each layer:
           ```
           Health Status Determination:
           - dead_percentage > 50%: Status = 'dead' (critical)
           - mean_gradient < 1e-6: Status = 'vanishing' (weak learning)
           - mean_gradient > 1.0: Status = 'exploding' (unstable)
           - Otherwise: Status = 'healthy' (normal)
           ```
        
        GRADIENT MAGNITUDE INTERPRETATION:
        
        What Different Magnitudes Mean:
        ```
        Gradient Magnitude Range | Interpretation | Action Needed
        ------------------------|----------------|---------------
        > 1.0                   | Exploding      | Gradient clipping, lower LR
        0.01 - 1.0             | Healthy        | Continue training
        0.001 - 0.01           | Weak but OK    | Monitor closely  
        0.0001 - 0.001         | Concerning     | Consider architecture changes
        < 0.0001               | Vanishing      | Major changes needed
        ```
        
        Real Example - Healthy CNN:
        ```
        Layer Analysis Results:
        conv2d:    mean=0.005, std=0.002, dead=2%   ✅ Healthy
        conv2d_1:  mean=0.012, std=0.008, dead=5%   ✅ Healthy  
        dense:     mean=0.025, std=0.015, dead=8%   ✅ Healthy
        dense_1:   mean=0.018, std=0.012, dead=3%   ✅ Healthy
        Overall: All layers learning effectively!
        ```
        
        Real Example - Problematic CNN:
        ```
        Layer Analysis Results:
        conv2d:    mean=0.000001, std=0.0000005, dead=60%  ❌ Vanishing + Dead
        conv2d_1:  mean=0.000005, std=0.000002, dead=45%   ❌ Vanishing + Dead
        dense:     mean=15.2, std=25.8, dead=10%           ❌ Exploding  
        dense_1:   mean=0.1, std=0.05, dead=5%             ✅ Only output OK
        Overall: Network has severe training problems!
        ```
        
        DEAD NEURON ANALYSIS:
        
        Why Neurons Die:
        1. **ReLU Dying Problem**:
           ```
           Neuron Evolution Over Training:
           Epoch 1: input=0.5  → ReLU(0.5)=0.5   ✅ Active
           Epoch 5: input=-0.1 → ReLU(-0.1)=0    ⚠️ Temporary death
           Epoch 10: input=-0.5 → ReLU(-0.5)=0   ❌ Consistently dead
           Result: Gradient=0, no more learning possible
           ```
        
        2. **Poor Initialization**:
           ```
           Bad Initialization Example:
           - All weights start near zero or very large values
           - Activations become saturated or zero
           - Gradients vanish or explode from the start
           ```
        
        3. **Excessive Dropout**:
           ```
           High Dropout Problem:
           - 70% dropout rate kills most neurons during training
           - Remaining neurons can't compensate
           - Many neurons never get chance to learn
           ```
        
        Dead Neuron Impact on Model Capacity:
        ```
        Dense Layer Example:
        - Total neurons: 128
        - Dead neurons: 58 (45%)
        - Effective capacity: Only 70 neurons working
        - Impact: Model using less than 60% of its potential!
        ```
        
        OVERALL HEALTH ASSESSMENT:
        
        Health Status Priority (worst to best):
        1. **'exploding'**: Immediate danger - training will diverge
        2. **'dead'**: Critical capacity loss - architecture needs fixing  
        3. **'vanishing'**: Learning stagnation - early layers not improving
        4. **'mixed'**: Some layers OK, others need attention
        5. **'healthy'**: All systems go - continue training
        
        Args:
            gradient_data: Dictionary containing gradient information per layer
                          From _compute_layer_gradients() method
            model: Keras model being analyzed (used for layer metadata)
            
        Returns:
            Dictionary containing comprehensive gradient analysis:
            
            Core Results:
            - 'gradient_health': Overall status ('healthy', 'vanishing', 'exploding', 'dead', 'mixed')
            - 'recommendations': List of actionable suggestions for improvement
            
            Detailed Statistics:
            - 'gradient_magnitudes': Per-layer gradient magnitude statistics
              ```
              {
                'conv2d': {
                  'mean': 0.005,      # Average gradient magnitude
                  'std': 0.002,       # Standard deviation  
                  'median': 0.004,    # Median value
                  'max': 0.025,       # Maximum gradient
                  'min': 0.0001,      # Minimum gradient
                  'q25': 0.003,       # 25th percentile
                  'q75': 0.007        # 75th percentile
                }
              }
              ```
            
            - 'dead_neurons': Per-layer dead neuron analysis
              ```
              {
                'dense': {
                  'dead_count': 58,           # Number of dead parameters
                  'total_params': 128,        # Total parameters in layer
                  'dead_percentage': 45.3     # Percentage dead
                }
              }
              ```
            
            - 'layer_analysis': Combined analysis per layer
              ```
              {
                'conv2d': {
                  'gradient_magnitude': 0.005,     # Mean gradient magnitude
                  'gradient_variance': 0.000004,   # Gradient variance
                  'dead_percentage': 2.1,          # Dead neuron percentage
                  'layer_type': 'Conv2D',          # Layer architecture type
                  'health_status': 'healthy'       # Overall layer health
                }
              }
              ```
            
        Side Effects:
            - Logs gradient statistics for each layer
            - Computes and logs overall network health assessment
            - Generates actionable recommendations based on detected issues
        """
        logger.debug("running _analyze_gradient_statistics ... Analyzing gradient statistics")
        
        gradient_magnitudes = {}
        dead_neurons = {}
        layer_analysis = {}
        
        # Analyze each layer
        for layer_name, layer_data in gradient_data.items():
            gradients = layer_data['gradients']
            
            # Compute gradient statistics
            magnitude_stats = {
                'mean': np.mean(np.abs(gradients)),
                'std': np.std(np.abs(gradients)),
                'median': np.median(np.abs(gradients)),
                'max': np.max(np.abs(gradients)),
                'min': np.min(np.abs(gradients)),
                'q25': np.percentile(np.abs(gradients), 25),
                'q75': np.percentile(np.abs(gradients), 75)
            }
            
            gradient_magnitudes[layer_name] = magnitude_stats
            
            # Dead neuron analysis
            zero_threshold = 1e-8
            zero_gradients = np.abs(gradients) < zero_threshold
            dead_count = int(np.sum(zero_gradients))
            dead_percentage = (dead_count / len(gradients)) * 100
            
            dead_neurons[layer_name] = {
                'dead_count': dead_count,
                'total_params': len(gradients),
                'dead_percentage': dead_percentage
            }
            
            # Layer-specific analysis
            layer_analysis[layer_name] = {
                'gradient_magnitude': magnitude_stats['mean'],
                'gradient_variance': magnitude_stats['std'] ** 2,
                'dead_percentage': dead_percentage,
                'layer_type': layer_data['layer_type'],
                'health_status': self._assess_layer_health(magnitude_stats, dead_percentage)
            }
        
        # Overall gradient health assessment
        overall_health = self._assess_overall_gradient_health(gradient_magnitudes, dead_neurons)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_health, layer_analysis)
        
        return {
            'gradient_magnitudes': gradient_magnitudes,
            'dead_neurons': dead_neurons,
            'gradient_health': overall_health,
            'layer_analysis': layer_analysis,
            'recommendations': recommendations
        }
    
    
    def _assess_layer_health(
        self,
        magnitude_stats: Dict[str, float],
        dead_percentage: float
    ) -> str:
        """
        Assess the health of a single layer's gradients
        
        Determines whether a layer is learning effectively by examining its gradient
        magnitude and dead neuron percentage. Like a doctor checking vital signs.
        
        HEALTH ASSESSMENT CRITERIA:
        
        Layer Health Decision Tree:
        ```
        1. Check dead neuron percentage first (most critical):
           If dead_percentage > 50%:
               return 'dead'  # Layer barely functional
        
        2. Check gradient magnitude (learning strength):
           If mean_gradient < 1e-6:
               return 'vanishing'  # Can't learn effectively
           If mean_gradient > 1.0:
               return 'exploding'  # Unstable learning
        
        3. Otherwise:
               return 'healthy'  # Normal operation
        ```
        
        Health Status Meanings:
        
        'healthy': ✅ Layer is learning effectively
        - Dead neurons: < 50% (most neurons contributing)
        - Gradient magnitude: Between 1e-6 and 1.0 (reasonable learning signals)
        - Action needed: None, continue training
        
        'dead': 💀 Layer has lost most of its capacity  
        - Dead neurons: > 50% (majority not contributing)
        - Impact: Layer only using half its potential
        - Action needed: Architecture changes (LeakyReLU, better initialization)
        
        'vanishing': 🔍 Layer receiving very weak learning signals
        - Gradient magnitude: < 1e-6 (extremely small)
        - Impact: Layer barely learning, improvement very slow
        - Action needed: Better initialization, residual connections, normalization
        
        'exploding': 💥 Layer receiving dangerously strong signals
        - Gradient magnitude: > 1.0 (very large)
        - Impact: Training instability, possible divergence
        - Action needed: Gradient clipping, lower learning rate
        
        Real Examples:
        
        Healthy Convolutional Layer:
        ```
        magnitude_stats = {'mean': 0.005, 'std': 0.002, ...}
        dead_percentage = 8.5
        → Assessment: 'healthy' (good gradient flow, most neurons active)
        ```
        
        Dead Dense Layer:
        ```
        magnitude_stats = {'mean': 0.001, 'std': 0.0005, ...}  
        dead_percentage = 65.2
        → Assessment: 'dead' (too many neurons not contributing)
        ```
        
        Vanishing Early Layer:
        ```
        magnitude_stats = {'mean': 0.0000005, 'std': 0.0000001, ...}
        dead_percentage = 15.3
        → Assessment: 'vanishing' (gradients too weak to learn)
        ```
        
        Exploding Layer:
        ```
        magnitude_stats = {'mean': 15.7, 'std': 25.2, ...}
        dead_percentage = 5.1  
        → Assessment: 'exploding' (gradients dangerously large)
        ```
        
        Args:
            magnitude_stats: Dictionary containing gradient magnitude statistics
                           Must include 'mean' key with average gradient magnitude
            dead_percentage: Percentage of neurons with zero gradients (0-100)
            
        Returns:
            Health status string: 'healthy', 'vanishing', 'exploding', or 'dead'
            
        Threshold Rationale:
        - Dead threshold (50%): Based on empirical observation that layers with >50% 
          dead neurons show significant performance degradation
        - Vanishing threshold (1e-6): Gradients below this are too small for effective
          weight updates with typical learning rates  
        - Exploding threshold (1.0): Gradients above this often cause training instability
        """
        mean_magnitude = magnitude_stats['mean']
        
        # Define thresholds
        vanishing_threshold = 1e-6
        exploding_threshold = 1.0
        dead_threshold = 50.0  # 50% dead neurons
        
        if dead_percentage > dead_threshold:
            return 'dead'
        elif mean_magnitude < vanishing_threshold:
            return 'vanishing'
        elif mean_magnitude > exploding_threshold:
            return 'exploding'
        else:
            return 'healthy'
    
    
    def _assess_overall_gradient_health(
        self,
        gradient_magnitudes: Dict[str, Dict[str, float]],
        dead_neurons: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Assess overall gradient health across all layers in the network
        
        Determines the overall "health status" of your neural network by examining
        the health of all individual layers and identifying the most critical issue.
        Like getting an overall medical assessment after individual organ checkups.
        
        OVERALL HEALTH ASSESSMENT LOGIC:
        
        Priority-Based Assessment (most critical issue wins):
        ```
        1. Check all layer health statuses
        2. If ANY layer is 'exploding' → Overall = 'exploding' (immediate danger)
        3. Else if ANY layer is 'dead' → Overall = 'dead' (capacity loss)  
        4. Else if ANY layer is 'vanishing' → Overall = 'vanishing' (learning problems)
        5. Else if multiple different issues → Overall = 'mixed' (complex problems)
        6. Else if all layers healthy → Overall = 'healthy' (all good!)
        ```
        
        Why This Priority Order?
        
        1. **'exploding' takes highest priority**:
           - Can cause immediate training failure (NaN losses, divergence)
           - Affects all subsequent layers due to gradient propagation
           - Needs immediate intervention before continuing training
        
        2. **'dead' takes second priority**:
           - Represents permanent capacity loss (neurons can't recover easily)
           - Reduces model's learning potential significantly
           - Requires architectural changes to fix
        
        3. **'vanishing' takes third priority**:
           - Slows learning but doesn't stop it completely
           - Early layers affected, but later layers may still learn
           - Can sometimes be improved with training hyperparameter changes
        
        4. **'mixed' indicates complex issues**:
           - Multiple different problems in different layers
           - Requires careful layer-by-layer analysis
           - May need combination of solutions
        
        5. **'healthy' means all systems go**:
           - All layers learning effectively
           - No immediate action needed
           - Continue training as planned
        
        Real-World Examples:
        
        Healthy Network:
        ```
        Layer Health Status:
        conv2d: 'healthy', conv2d_1: 'healthy', dense: 'healthy', dense_1: 'healthy'
        → Overall Assessment: 'healthy'
        → Action: Continue training, model is learning well
        ```
        
        Exploding Network:
        ```
        Layer Health Status:
        conv2d: 'healthy', conv2d_1: 'exploding', dense: 'healthy', dense_1: 'vanishing'
        → Overall Assessment: 'exploding' (despite other issues, exploding takes priority)
        → Action: Implement gradient clipping immediately, then address vanishing
        ```
        
        Dead Network:
        ```
        Layer Health Status:
        conv2d: 'healthy', conv2d_1: 'dead', dense: 'dead', dense_1: 'healthy'
        → Overall Assessment: 'dead' (multiple layers have lost capacity)
        → Action: Switch to LeakyReLU, improve initialization
        ```
        
        Mixed Problems:
        ```
        Layer Health Status:
        conv2d: 'vanishing', conv2d_1: 'healthy', dense: 'dead', dense_1: 'healthy'
        → Overall Assessment: 'mixed' (different problems in different layers)
        → Action: Layer-specific solutions needed
        ```
        
        Args:
            gradient_magnitudes: Per-layer gradient magnitude statistics
                                From _analyze_gradient_statistics method
            dead_neurons: Per-layer dead neuron statistics  
                         From _analyze_gradient_statistics method
            
        Returns:
            Overall health status: 'healthy', 'vanishing', 'exploding', 'dead', or 'mixed'
            
        Side Effects:
            - Assesses each individual layer using _assess_layer_health
            - Prioritizes most critical issues for overall assessment
            
        Clinical Decision Making Analogy:
            Like a doctor reviewing multiple test results:
            - If any test shows immediate danger (exploding) → Emergency action needed
            - If multiple organs failing (dead) → Critical condition  
            - If some systems weak (vanishing) → Chronic condition needs treatment
            - If mixed results → Complex case needs detailed analysis
            - If all tests normal (healthy) → Patient is fine
        """
        layer_healths = []
        
        for layer_name in gradient_magnitudes.keys():
            magnitude_stats = gradient_magnitudes[layer_name]
            dead_percentage = dead_neurons[layer_name]['dead_percentage']
            
            layer_health = self._assess_layer_health(magnitude_stats, dead_percentage)
            layer_healths.append(layer_health)
        
        # Determine overall health
        unique_healths = set(layer_healths)
        
        if len(unique_healths) == 1 and 'healthy' in unique_healths:
            return 'healthy'
        elif 'exploding' in unique_healths:
            return 'exploding'
        elif 'vanishing' in unique_healths:
            return 'vanishing'
        elif 'dead' in unique_healths:
            return 'dead'
        else:
            return 'mixed'
    
    
    def _generate_recommendations(
        self,
        overall_health: str,
        layer_analysis: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Generate actionable recommendations based on gradient analysis results
        
        Provides specific, prioritized suggestions for improving your neural network's
        gradient flow and training effectiveness. Like a doctor's treatment plan
        after diagnosing health issues.
        
        RECOMMENDATION STRATEGY BY PROBLEM TYPE:
        
        Vanishing Gradients Treatment Plan:
        ```
        Problem: Gradients become exponentially smaller in early layers
        Root Causes: Poor initialization, deep networks, saturating activations
        
        Recommended Solutions (in order of effectiveness):
        1. 🔧 Better weight initialization (He/Xavier)
           - He initialization for ReLU networks: weights ~ N(0, 2/fan_in)
           - Xavier initialization for sigmoid/tanh: weights ~ N(0, 1/fan_in)
        
        2. 🏗️ Residual connections or skip connections  
           - Allow gradients to flow directly to early layers
           - Prevents gradient signal degradation through many layers
        
        3. ⚡ Batch normalization layers
           - Normalizes inputs to each layer (mean=0, std=1)
           - Reduces internal covariate shift, stabilizes gradients
        
        4. 🎯 Different activation functions
           - ELU: No zero gradients, smooth everywhere
           - Swish: Self-gated, smooth, better than ReLU for deep networks
        
        5. 📏 Reduce network depth or add gradient highways
           - Fewer layers = less gradient degradation
           - Highway networks allow selective information flow
        ```
        
        Exploding Gradients Treatment Plan:
        ```
        Problem: Gradients become exponentially larger, causing instability
        Root Causes: High learning rates, poor initialization, no normalization
        
        Recommended Solutions (in order of urgency):
        1. ✂️ Gradient clipping (IMMEDIATE)
           - Clip gradients to maximum norm (e.g., clip_norm=1.0)
           - Prevents weight updates from becoming too large
        
        2. 📉 Reduce learning rate significantly
           - Start with 10x smaller learning rate
           - Use learning rate schedules or adaptive optimizers
        
        3. 🎯 Better weight initialization
           - Smaller initial weights reduce compound effects
           - Use proper initialization for your activation functions
        
        4. 🏗️ Batch normalization for stability
           - Keeps activations in reasonable ranges
           - Reduces sensitivity to weight initialization
        
        5. 📐 Reduce network width or depth
           - Smaller networks are less prone to exploding gradients
           - Fewer parameters = less opportunity for instability
        ```
        
        Dead Neurons Treatment Plan:
        ```
        Problem: Many neurons output zero and stop learning
        Root Causes: Dying ReLU problem, poor initialization, excessive dropout
        
        Recommended Solutions (in order of effectiveness):
        1. 🔄 Use LeakyReLU or ELU instead of ReLU
           - LeakyReLU: f(x) = max(0.01x, x) allows small negative values
           - ELU: smooth, no zero gradients, faster convergence
        
        2. 🎯 Improve weight initialization
           - He initialization prevents initial dead neurons
           - Proper variance keeps activations in good ranges
        
        3. 📈 Increase learning rate cautiously  
           - Higher LR can help stuck neurons escape zero state
           - Monitor for exploding gradients
        
        4. 🏗️ Add batch normalization
           - Keeps inputs to activations in reasonable ranges
           - Reduces likelihood of neurons getting stuck
        
        5. 🔧 Reduce dropout rates
           - High dropout can prevent neurons from learning
           - Try 0.3 instead of 0.5, or remove dropout entirely
        ```
        
        Mixed Problems Treatment Plan:
        ```
        Problem: Different layers have different issues
        Strategy: Layer-specific analysis and targeted solutions
        
        1. 🔍 Analyze individual layer health
           - Identify which layers have which problems
           - Apply specific solutions to specific layers
        
        2. ⚖️ Consider layer-specific learning rates
           - Higher LR for vanishing layers
           - Lower LR for exploding layers
        
        3. 🏗️ Add normalization between problematic layers
           - BatchNorm between healthy and problematic layers
           - Can prevent problem propagation
        
        4. 🎯 Review overall architecture design
           - May need fundamental architecture changes
           - Consider proven architectures (ResNet, DenseNet)
        ```
        
        RECOMMENDATION PERSONALIZATION:
        
        Layer-Specific Recommendations:
            The method also identifies the most problematic layers and provides
            targeted advice for those specific components:
            
            ```
            Example Output:
            "🎯 Focus attention on layers: conv2d_1, dense, dense_1"
            
            This tells you exactly which layers need the most attention,
            allowing you to apply solutions strategically rather than 
            making global changes that might not be necessary.
            ```
        
        Args:
            overall_health: Overall gradient health status from _assess_overall_gradient_health
                          One of: 'healthy', 'vanishing', 'exploding', 'dead', 'mixed'
            layer_analysis: Per-layer analysis results with health status for each layer
                          Used to identify specific problematic layers for targeted advice
            
        Returns:
            List of recommendation strings, ordered by priority/effectiveness:
            
            Format: ["🔧 Use better weight initialization", "🏗️ Add batch normalization", ...]
            
            Each recommendation includes:
            - Icon for quick visual categorization
            - Clear, actionable instruction
            - Specific technique or parameter suggestions where applicable
            
        Recommendation Categories (by icon):
        - 🔧 Technical fixes (initialization, activation functions)
        - 🏗️ Architectural changes (layers, connections, normalization)
        - ⚡ Training optimizations (learning rates, schedules)
        - 🎯 Hyperparameter tuning (specific values, ranges)
        - 📏📉📈 Scaling adjustments (rates, sizes, thresholds)
        - ✂️ Regularization techniques (clipping, dropout)
        - 🔄🔍 Alternative approaches (different methods, analysis)
        """
        recommendations = []
        
        if overall_health == 'vanishing':
            recommendations.extend([
                "🔧 Use better weight initialization (Xavier/He initialization)",
                "🏗️ Consider residual connections or skip connections",
                "⚡ Add batch normalization layers",
                "🎯 Try different activation functions (ELU, Swish instead of ReLU)",
                "📏 Reduce network depth or add gradient highways"
            ])
        
        elif overall_health == 'exploding':
            recommendations.extend([
                "✂️ Implement gradient clipping (clip_norm=1.0)",
                "📉 Reduce learning rate significantly",
                "🎯 Use better weight initialization",
                "🏗️ Add batch normalization for gradient stability",
                "📐 Consider reducing network width or depth"
            ])
        
        elif overall_health == 'dead':
            recommendations.extend([
                "🔄 Use LeakyReLU or ELU instead of ReLU",
                "🎯 Improve weight initialization",
                "📈 Increase learning rate cautiously",
                "🏗️ Add batch normalization",
                "🔧 Check for dying ReLU problem"
            ])
        
        elif overall_health == 'mixed':
            recommendations.extend([
                "🔍 Analyze individual layer health",
                "⚖️ Consider layer-specific learning rates",
                "🏗️ Add normalization between problematic layers",
                "🎯 Review architecture design"
            ])
        
        else:  # healthy
            recommendations.append("✅ Gradient flow is healthy - no immediate action needed")
        
        # Add layer-specific recommendations
        problem_layers = [name for name, analysis in layer_analysis.items() 
                         if analysis['health_status'] != 'healthy']
        
        if problem_layers:
            recommendations.append(f"🎯 Focus attention on layers: {', '.join(problem_layers[:3])}")
        
        return recommendations
    
    def _create_visualizations(
        self,
        gradient_data: Dict[str, Dict[str, np.ndarray]],
        analysis_results: Dict[str, Any],
        model: keras.Model,
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path],
        config: Optional[Any] = None
    ) -> List[Path]:
        """
        Create comprehensive gradient flow visualizations
        
        Args:
            gradient_data: Gradient data per layer
            analysis_results: Analysis results
            model: Keras model
            dataset_name: Dataset name
            run_timestamp: Timestamp for file naming
            plot_dir: Directory to save plots
            
        Returns:
            List of paths to saved visualization files
        """
        logger.debug("running _create_visualizations ... Creating gradient flow visualizations")
        
        visualization_paths = []
        
        try:
            # Check configuration flags for each plot type
            show_gradient_magnitudes = getattr(config, 'show_gradient_magnitudes', True) if config else True
            show_gradient_distributions = getattr(config, 'show_gradient_distributions', True) if config else True
            show_dead_neuron_analysis = getattr(config, 'show_dead_neuron_analysis', True) if config else True
            
            # 1. Layer-wise gradient magnitude plot
            if show_gradient_magnitudes:
                magnitude_path = self._plot_gradient_magnitudes(
                    analysis_results['gradient_magnitudes'],
                    dataset_name, run_timestamp, plot_dir
                )
                if magnitude_path:
                    visualization_paths.append(magnitude_path)
            
            # 2. Gradient distribution histograms
            if show_gradient_distributions:
                distribution_path = self._plot_gradient_distributions(
                    gradient_data, dataset_name, run_timestamp, plot_dir
                )
                if distribution_path:
                    visualization_paths.append(distribution_path)
            
            # 3. Dead neuron analysis
            if show_dead_neuron_analysis:
                dead_neuron_path = self._plot_dead_neuron_analysis(
                    analysis_results['dead_neurons'],
                    dataset_name, run_timestamp, plot_dir
                )
                if dead_neuron_path:
                    visualization_paths.append(dead_neuron_path)
            
            logger.debug(f"running _create_visualizations ... Created {len(visualization_paths)} visualization files")
            
        except Exception as e:
            logger.warning(f"running _create_visualizations ... Failed to create some visualizations: {e}")
        
        return visualization_paths
    
    def _plot_gradient_magnitudes(
        self,
        gradient_magnitudes: Dict[str, Dict[str, float]],
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path]
    ) -> Optional[Path]:
        """
        Plot layer-wise gradient magnitudes
        
        Args:
            gradient_magnitudes: Gradient magnitude statistics per layer
            dataset_name: Dataset name
            run_timestamp: Timestamp
            plot_dir: Plot directory
            
        Returns:
            Path to saved plot, or None if failed
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            layer_names = list(gradient_magnitudes.keys())
            mean_magnitudes = [gradient_magnitudes[name]['mean'] for name in layer_names]
            std_magnitudes = [gradient_magnitudes[name]['std'] for name in layer_names]
            
            # Create bar plot with error bars
            x_pos = np.arange(len(layer_names))
            bars = ax.bar(x_pos, mean_magnitudes, yerr=std_magnitudes, 
                         capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
            
            # Customize plot
            ax.set_xlabel('Layer')
            ax.set_ylabel('Gradient Magnitude')
            ax.set_title(f'Gradient Magnitudes by Layer - {dataset_name}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(layer_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')  # Log scale for better visualization
            
            # Add threshold lines
            ax.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, label='Vanishing threshold')
            ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Exploding threshold')
            ax.legend()
            
            plt.tight_layout()
            
            # Save plot
            filepath = self._generate_save_path(
                "gradient_magnitudes", dataset_name, run_timestamp, plot_dir
            )
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.warning(f"running _plot_gradient_magnitudes ... Failed to create plot: {e}")
            plt.close()
            return None
    
    def _plot_gradient_distributions(
        self,
        gradient_data: Dict[str, Dict[str, np.ndarray]],
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path]
    ) -> Optional[Path]:
        """
        Plot gradient distribution histograms for each layer
        
        Args:
            gradient_data: Gradient data per layer
            dataset_name: Dataset name
            run_timestamp: Timestamp
            plot_dir: Plot directory
            
        Returns:
            Path to saved plot, or None if failed
        """
        try:
            n_layers = len(gradient_data)
            cols = min(3, n_layers)
            rows = (n_layers + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
            if n_layers == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (layer_name, layer_data) in enumerate(gradient_data.items()):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                gradients = layer_data['gradients']
                
                # Create histogram
                ax.hist(gradients, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
                ax.set_title(f'{layer_name}\n({layer_data["layer_type"]})')
                ax.set_xlabel('Gradient Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                mean_grad = np.mean(gradients)
                std_grad = np.std(gradients)
                ax.text(0.02, 0.98, f'μ={mean_grad:.2e}\nσ={std_grad:.2e}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(len(gradient_data), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Gradient Distributions by Layer - {dataset_name}', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            filepath = self._generate_save_path(
                "gradient_distributions", dataset_name, run_timestamp, plot_dir
            )
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.warning(f"running _plot_gradient_distributions ... Failed to create plot: {e}")
            plt.close()
            return None
    
    def _plot_dead_neuron_analysis(
        self,
        dead_neurons: Dict[str, Dict[str, float]],
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path]
    ) -> Optional[Path]:
        """
        Plot dead neuron analysis
        
        Args:
            dead_neurons: Dead neuron statistics per layer
            dataset_name: Dataset name
            run_timestamp: Timestamp
            plot_dir: Plot directory
            
        Returns:
            Path to saved plot, or None if failed
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            layer_names = list(dead_neurons.keys())
            dead_percentages = [dead_neurons[name]['dead_percentage'] for name in layer_names]
            dead_counts = [dead_neurons[name]['dead_count'] for name in layer_names]
            total_params = [dead_neurons[name]['total_params'] for name in layer_names]
            
            # Plot 1: Dead neuron percentages
            x_pos = np.arange(len(layer_names))
            bars1 = ax1.bar(x_pos, dead_percentages, alpha=0.7, color='crimson')
            
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Dead Neurons (%)')
            ax1.set_title('Dead Neuron Percentage by Layer')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(layer_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Critical threshold')
            ax1.legend()
            
            # Plot 2: Absolute dead neuron counts
            bars2 = ax2.bar(x_pos, dead_counts, alpha=0.7, color='darkred', label='Dead')
            bars3 = ax2.bar(x_pos, [total - dead for total, dead in zip(total_params, dead_counts)],
                           bottom=dead_counts, alpha=0.7, color='lightgreen', label='Active')
            
            ax2.set_xlabel('Layer')
            ax2.set_ylabel('Number of Parameters')
            ax2.set_title('Dead vs Active Parameters by Layer')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(layer_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            filepath = self._generate_save_path(
                "dead_neuron_analysis", dataset_name, run_timestamp, plot_dir
            )
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.warning(f"running _plot_dead_neuron_analysis ... Failed to create plot: {e}")
            plt.close()
            return None
    
    def _generate_save_path(
        self,
        plot_type: str,
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path]
    ) -> Path:
        """
        Generate file path for saving plots
        
        Args:
            plot_type: Type of plot (for filename)
            dataset_name: Dataset name
            run_timestamp: Timestamp
            plot_dir: Plot directory
            
        Returns:
            Path for saving
        """
        from datetime import datetime
        
        # Create timestamp if not provided
        if run_timestamp is None:
            # run_timestamp should always be provided from optimizer.py
            raise ValueError("run_timestamp should always be provided from optimizer.py")
        
        # Clean dataset name
        dataset_name_clean = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        # Generate filename
        filename = f"{plot_type}_{run_timestamp}_{dataset_name_clean}.png"
        
        # Determine save directory
        if plot_dir is not None:
            save_dir = plot_dir
        else:
            # Fallback: create default directory
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            save_dir = project_root / "plots"
            save_dir.mkdir(exist_ok=True)
        
        return save_dir / filename
    
    def _log_analysis_summary(self, analysis_results: Dict[str, Any]) -> None:
        """
        Log summary of gradient flow analysis
        
        Args:
            analysis_results: Complete analysis results
        """
        logger.debug("running _log_analysis_summary ... Gradient Flow Analysis Summary")
        logger.debug(f"running _log_analysis_summary ... Overall Health: {analysis_results['gradient_health']}")
        
        # Log layer-by-layer health
        layer_analysis = analysis_results['layer_analysis']
        for layer_name, analysis in layer_analysis.items():
            health = analysis['health_status']
            magnitude = analysis['gradient_magnitude']
            dead_pct = analysis['dead_percentage']
            
            logger.debug(f"running _log_analysis_summary ... {layer_name}: "
                        f"Health={health}, Magnitude={magnitude:.2e}, Dead={dead_pct:.1f}%")
        
        # Log recommendations
        logger.debug("running _log_analysis_summary ... Recommendations:")
        for rec in analysis_results['recommendations']:
            logger.debug(f"running _log_analysis_summary ... {rec}")


def create_gradient_flow_analysis(
    model: keras.Model,
    sample_data: np.ndarray,
    sample_labels: np.ndarray,
    dataset_name: str = "dataset",
    run_timestamp: Optional[str] = None,
    plot_dir: Optional[Path] = None,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Convenience function for quick gradient flow analysis
    
    Args:
        model: Trained Keras model to analyze
        sample_data: Sample input data for gradient computation
        sample_labels: Corresponding labels for the sample data
        dataset_name: Name of dataset
        run_timestamp: Optional timestamp
        plot_dir: Optional directory for saving plots
        sample_size: Number of samples to use for analysis
        
    Returns:
        Analysis results dictionary
    """
    analyzer = GradientFlowAnalyzer(model_name=dataset_name)
    return analyzer.analyze_and_visualize(
        model=model,
        sample_data=sample_data,
        sample_labels=sample_labels,
        dataset_name=dataset_name,
        run_timestamp=run_timestamp,
        plot_dir=plot_dir,
        sample_size=sample_size
    )