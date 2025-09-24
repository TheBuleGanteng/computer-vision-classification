"""
RunPod Service Handler - Thin wrapper around existing optimizer.py orchestration
Implements specialized serverless hyperparameter optimization using existing battle-tested logic.
"""

# Web server wrapper for local testing and container deployment
import json
import numpy as np
import os
from pathlib import Path
from pydantic import BaseModel
import runpod
import sys
import tensorflow as tf
import traceback
from typing import Dict, Any, Optional, TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.data_classes.configs import OptimizationConfig
from datetime import datetime

# Add project root to Python path for imports
current_file = Path(__file__)
project_root = current_file.parent.parent  # Go up 2 levels to project root
sys.path.insert(0, str(project_root / "src"))

# Import existing orchestration layer
from src.optimizer import OptimizationResult
from src.utils.logger import logger
from src.model_builder import create_and_train_model, ModelConfig
from src.dataset_manager import DatasetManager

def adjust_concurrency(current_concurrency):
    return min(current_concurrency + 1, 6)  # Your max workers


def generate_plots(
    model_builder,
    dataset_name: str,
    trial_id: str,
    test_data: Optional[Dict[str, Any]] = None,
    optimization_config: Optional['OptimizationConfig'] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate plots using PlotGenerator and save to local plot directory.

    Args:
        model_builder: ModelBuilder instance with trained model
        dataset_name: Name of the dataset
        trial_id: Trial identifier for plot directory
        optimization_config: Optional OptimizationConfig object containing plot flags and settings

    Returns:
        Dictionary with plot info if successful, None if skipped/failed
    """
    # Check plot generation setting from optimization_config
    if optimization_config and getattr(optimization_config, 'plot_generation', 'all') == 'none':
        logger.debug(f"generate_plots ... Skipping plot generation: plot_generation='none'")
        return None

    try:
        logger.debug(f"generate_plots ... Generating plots for trial {trial_id}")

        # Import here to avoid circular dependencies
        from src.plot_generator import PlotGenerator

        # Create persistent directory for plots in /tmp/plots
        plots_base_dir = Path("/tmp/plots")
        plots_base_dir.mkdir(parents=True, exist_ok=True)

        # Check if trial_id already contains dataset name (for full run names)
        if dataset_name in trial_id:
            plot_dir = plots_base_dir / trial_id
        else:
            plot_dir = plots_base_dir / f"{trial_id}_{dataset_name}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Get dataset config and model config from model_builder
        if not hasattr(model_builder, 'dataset_config') or not hasattr(model_builder, 'model_config'):
            logger.warning(f"generate_plots ... ModelBuilder missing required configs, skipping plot generation")
            return None

        # Create PlotGenerator with provided or fallback optimization_config
        effective_optimization_config = optimization_config or getattr(model_builder, 'optimization_config', None)

        plot_generator = PlotGenerator(
            dataset_config=model_builder.dataset_config,
            model_config=model_builder.model_config,
            optimization_config=effective_optimization_config
        )

        # Use provided test_data or get from model_builder as fallback
        if not test_data:
            test_data = getattr(model_builder, 'test_data', None)
            if test_data:
                logger.debug(f"generate_plots ... Using test_data from model_builder with {len(test_data.get('x_test', []))} samples")
            else:
                logger.warning(f"generate_plots ... No test_data provided and none found in model_builder")
        else:
            logger.debug(f"generate_plots ... Using provided test_data with {len(test_data.get('x_test', []))} samples")

        if not test_data:
            logger.warning(f"generate_plots ... No test data available, skipping plot generation")
            return None

        # Get training metrics
        test_loss = 0.0
        test_accuracy = 0.0
        if hasattr(model_builder, 'model') and model_builder.model:
            try:
                test_loss, test_accuracy = model_builder.model.evaluate(
                    test_data['x_test'], test_data['y_test'], verbose=0
                )
            except Exception as e:
                logger.warning(f"generate_plots ... Could not evaluate model: {e}")

        # Extract timestamp from trial_id (run name) for consistent file naming
        # trial_id format: "2025-09-22-16-59-51_mnist_health" -> extract "2025-09-22-16-59-51"
        if '_' in trial_id and len(trial_id.split('_')[0]) >= 19:  # timestamp format length
            run_timestamp = trial_id.split('_')[0]  # Extract timestamp part
            logger.debug(f"generate_plots ... Using coordinated timestamp from trial_id: {run_timestamp}")
        else:
            # Fallback to current time if trial_id doesn't contain timestamp
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            logger.warning(f"generate_plots ... Could not extract timestamp from trial_id '{trial_id}', using current time: {run_timestamp}")

        analysis_results = plot_generator.generate_comprehensive_plots(
            model=model_builder.model,
            training_history=getattr(model_builder, 'training_history', None),
            data=test_data,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            run_timestamp=run_timestamp,
            plot_dir=plot_dir,
            log_detailed_predictions=True,
            max_predictions_to_show=20,
            progress_callback=None  # No progress callback for plot generation
        )

        # Check if any plots were generated
        generated_plots = []
        available_files = []

        for plot_type, result in analysis_results.items():
            if result and not result.get('error'):
                generated_plots.append(plot_type)

        # Get list of available files
        if plot_dir.exists():
            for file_path in plot_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(plot_dir)
                    available_files.append(str(relative_path))

        if not generated_plots:
            logger.warning(f"generate_plots ... No plots were generated successfully")
            return None

        logger.debug(f"generate_plots ... Generated plots: {', '.join(generated_plots)}")
        logger.info(f"generate_plots ... ✅ Successfully generated plots in {plot_dir}")

        # Set run_name to match the actual directory used
        if dataset_name in trial_id:
            run_name = trial_id
        else:
            run_name = f"{trial_id}_{dataset_name}"

        return {
            "success": True,
            "plot_dir": str(plot_dir),
            "run_name": run_name,
            "generated_plots": generated_plots,
            "available_files": available_files
        }

    except Exception as e:
        logger.error(f"generate_plots ... Plot generation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }




def extract_metrics(optimization_result: OptimizationResult) -> Dict[str, Any]:
    """
    Extract metrics from optimization result for API response.
    
    Args:
        optimization_result: OptimizationResult from optimize_model() call
        
    Returns:
        Structured metrics dictionary for API response
    """
    logger.debug("running extract_metrics ... extracting metrics from optimization result")
    
    try:
        # Extract best trial results from OptimizationResult
        best_params = optimization_result.best_params
        best_value = optimization_result.best_total_score
        
        # Extract core metrics from best_trial_health if available
        test_accuracy = 0.0
        test_loss = 0.0
        val_accuracy = 0.0
        val_loss = 0.0
        
        if optimization_result.best_trial_health:
            test_accuracy = optimization_result.best_trial_health.get('test_accuracy', 0.0)
            test_loss = optimization_result.best_trial_health.get('test_loss', 0.0)
            val_accuracy = optimization_result.best_trial_health.get('val_accuracy', test_accuracy)  # Fallback to test_accuracy
            val_loss = optimization_result.best_trial_health.get('val_loss', test_loss)  # Fallback to test_loss
        
        # If health metrics not available, use best_value as accuracy fallback
        if test_accuracy == 0.0 and best_value > 0.0:
            test_accuracy = best_value
            val_accuracy = best_value
        
        # Extract core metrics
        metrics = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'best_value': best_value,
            'optimize_for': optimization_result.optimization_config.objective.value if optimization_result.optimization_config else 'unknown',
            'training_time_seconds': optimization_result.optimization_time_hours * 3600.0  # Convert hours to seconds
        }
        
        # Extract model information
        epochs_completed = best_params.get('epochs', 0) if best_params else 0
        model_info = {
            'epochs_completed': epochs_completed,
            'total_trials': optimization_result.total_trials,
            'successful_trials': optimization_result.successful_trials
        }
        
        # Extract health metrics if available
        health_metrics = {}
        if optimization_result.best_trial_health:
            health_metrics = {
                'overall_health': optimization_result.best_trial_health.get('overall_health', 0.0),
                'neuron_utilization': optimization_result.best_trial_health.get('neuron_utilization', 0.0),
                'training_stability': optimization_result.best_trial_health.get('training_stability', 0.0),
                'gradient_health': optimization_result.best_trial_health.get('gradient_health', 0.0)
            }
        
        # Extract parameter importance
        parameter_importance = optimization_result.parameter_importance if optimization_result.parameter_importance else {}
        
        logger.debug(f"running extract_metrics ... successfully extracted metrics with best_value: {best_value}")
        
        return {
            'metrics': metrics,
            'model_info': model_info,
            'health_metrics': health_metrics,
            'parameter_importance': parameter_importance,
            'best_params': best_params
        }
        
    except Exception as e:
        logger.error(f"running extract_metrics ... error extracting metrics: {str(e)}")
        # Return default structure on error
        return {
            'metrics': {
                'best_value': 0.0,
                'optimize_for': 'unknown',
                'training_time_seconds': 0.0,
                'test_accuracy': 0.0,
                'test_loss': 0.0,
                'val_accuracy': 0.0,
                'val_loss': 0.0
            },
            'model_info': {
                'epochs_completed': 0,
                'total_trials': 0,
                'successful_trials': 0
            },
            'health_metrics': {},
            'parameter_importance': {},
            'best_params': {}
        }

def extract_model_attributes(model_builder, dataset_name: str, trial_id: str = "unknown") -> Dict[str, Any]:
    """
    Extract model attributes needed for local plot generation
    
    Args:
        model_builder: ModelBuilder instance with trained model
        dataset_name: Name of the dataset
        trial_id: Trial identifier for logging
        
    Returns:
        Dictionary containing serializable model attributes:
        - weights_bias_data: Model weights and biases
        - gradient_flow_data: Gradient flow information  
        - activation_data: Sample activation maps
        - predictions_data: Detailed predictions on test set
    """
    logger.debug(f"running extract_model_attributes ... Extracting model attributes for trial {trial_id}")
    
    try:
        model_attributes = {
            'weights_bias_data': None,
            'gradient_flow_data': None,
            'activation_data': None,
            'predictions_data': None,
            'extraction_success': False
        }
        
        if not model_builder or not hasattr(model_builder, 'model') or not model_builder.model:
            logger.warning(f"running extract_model_attributes ... No model available for attribute extraction")
            return model_attributes
        
        model = model_builder.model
        
        # 1. Extract weights and biases data
        try:
            weights_bias_data = extract_weights_bias_data(model)
            model_attributes['weights_bias_data'] = weights_bias_data
            logger.debug(f"running extract_model_attributes ... Extracted weights/bias data for {len(weights_bias_data['layers'])} layers")
        except Exception as e:
            logger.warning(f"running extract_model_attributes ... Failed to extract weights/bias data: {e}")
        
        # 2. Extract gradient flow data
        try:
            # Get test data for gradient computation
            test_data = getattr(model_builder, 'test_data', None)
            if test_data:
                gradient_data = extract_gradient_flow_data(model, test_data, sample_size=50)
                model_attributes['gradient_flow_data'] = gradient_data
                logger.debug(f"running extract_model_attributes ... Extracted gradient flow data for {len(gradient_data['layer_gradients'])} layers")
            else:
                logger.warning(f"running extract_model_attributes ... No test data available for gradient extraction")
        except Exception as e:
            logger.warning(f"running extract_model_attributes ... Failed to extract gradient data: {e}")
        
        # 3. Extract activation data (for CNN models)
        try:
            # Check if this is a CNN model with convolutional layers
            has_conv_layers = any('conv' in layer.name.lower() for layer in model.layers)
            if has_conv_layers:
                test_data = getattr(model_builder, 'test_data', None)
                if test_data:
                    activation_data = extract_activation_data(model, test_data, sample_size=10)
                    model_attributes['activation_data'] = activation_data
                    logger.debug(f"running extract_model_attributes ... Extracted activation data for {len(activation_data['sample_activations'])} samples")
                else:
                    logger.debug(f"running extract_model_attributes ... No test data available for activation extraction")
            else:
                logger.debug(f"running extract_model_attributes ... No convolutional layers found, skipping activation extraction")
        except Exception as e:
            logger.warning(f"running extract_model_attributes ... Failed to extract activation data: {e}")
        
        # 4. Extract detailed predictions
        try:
            test_data = getattr(model_builder, 'test_data', None)
            if test_data:
                predictions_data = extract_predictions_data(model, test_data, sample_size=100)
                model_attributes['predictions_data'] = predictions_data
                logger.debug(f"running extract_model_attributes ... Extracted predictions for {len(predictions_data['predictions'])} samples")
            else:
                logger.warning(f"running extract_model_attributes ... No test data available for predictions extraction")
        except Exception as e:
            logger.warning(f"running extract_model_attributes ... Failed to extract predictions data: {e}")
        
        # Mark extraction as successful if we got any data
        model_attributes['extraction_success'] = any([
            model_attributes['weights_bias_data'],
            model_attributes['gradient_flow_data'], 
            model_attributes['activation_data'],
            model_attributes['predictions_data']
        ])
        
        if model_attributes['extraction_success']:
            logger.debug(f"running extract_model_attributes ... Model attribute extraction completed successfully")
        else:
            logger.warning(f"running extract_model_attributes ... No model attributes were successfully extracted")
        
        return model_attributes
        
    except Exception as e:
        logger.error(f"running extract_model_attributes ... Model attribute extraction failed: {e}")
        return {
            'weights_bias_data': None,
            'gradient_flow_data': None,
            'activation_data': None,
            'predictions_data': None,
            'extraction_success': False,
            'error': str(e)
        }

def extract_weights_bias_data(model) -> Dict[str, Any]:
    """Extract model weights and biases in serializable format"""
    
    layers_data = []
    for i, layer in enumerate(model.layers):
        if layer.get_weights():  # Only layers with weights
            weights = layer.get_weights()
            layer_info = {
                'layer_name': layer.name,
                'layer_type': type(layer).__name__,
                'layer_index': i,
                'weights': [w.tolist() for w in weights],  # Convert to lists for JSON serialization
                'weight_shapes': [w.shape for w in weights]
            }
            layers_data.append(layer_info)
    
    return {
        'layers': layers_data,
        'total_layers': len(layers_data)
    }

def extract_gradient_flow_data(model, test_data: Dict[str, Any], sample_size: int = 50) -> Dict[str, Any]:
    """Extract gradient flow information in serializable format"""
    
    # Get sample data for gradient computation
    x_test = test_data.get('x_test', [])
    y_test = test_data.get('y_test', [])
    
    if len(x_test) == 0 or len(y_test) == 0:
        return {'layer_gradients': [], 'error': 'No test data available'}
    
    # Use a small sample for gradient computation
    sample_size = min(sample_size, len(x_test))
    sample_indices = np.random.choice(len(x_test), sample_size, replace=False)
    sample_x = x_test[sample_indices]
    sample_y = y_test[sample_indices]
    
    # Convert to tensors
    x_tensor = tf.convert_to_tensor(sample_x)
    y_tensor = tf.convert_to_tensor(sample_y)
    
    layer_gradients = []
    
    # Compute gradients using GradientTape
    with tf.GradientTape() as tape:
        predictions = model(x_tensor, training=False)
        # Use a simple loss for gradient computation
        loss = tf.keras.losses.categorical_crossentropy(y_tensor, predictions) # type: ignore
        loss = tf.reduce_mean(loss)
    
    # Get gradients for all trainable variables
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Organize gradients by layer
    var_index = 0
    for layer in model.layers:
        if layer.trainable_weights:
            layer_grads = []
            for weight in layer.trainable_weights:
                if (gradients is not None and 
                    var_index < len(gradients) and 
                    gradients[var_index] is not None):
                    grad_values = gradients[var_index].numpy()
                    layer_grads.append({
                        'gradient_stats': {
                            'mean': float(np.mean(np.abs(grad_values))),
                            'std': float(np.std(np.abs(grad_values))),
                            'max': float(np.max(np.abs(grad_values))),
                            'min': float(np.min(np.abs(grad_values))),
                            'shape': grad_values.shape
                        }
                    })
                var_index += 1
            
            if layer_grads:
                layer_gradients.append({
                    'layer_name': layer.name,
                    'layer_type': type(layer).__name__,
                    'gradients': layer_grads
                })
    
    return {
        'layer_gradients': layer_gradients,
        'sample_size_used': sample_size
    }

def extract_activation_data(model, test_data: Dict[str, Any], sample_size: int = 10) -> Dict[str, Any]:
    """Extract activation maps for CNN layers in serializable format"""
    
    x_test = test_data.get('x_test', [])
    if len(x_test) == 0:
        return {'sample_activations': [], 'error': 'No test data available'}
    
    # Use small sample for activation extraction
    sample_size = min(sample_size, len(x_test))
    sample_indices = np.random.choice(len(x_test), sample_size, replace=False)
    sample_x = x_test[sample_indices]
    
    # Get convolutional layers
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
    if not conv_layers:
        return {'sample_activations': [], 'error': 'No convolutional layers found'}
    
    # Extract activations for first few conv layers (to keep data manageable)
    target_layers = conv_layers[:3]  # Limit to first 3 conv layers
    
    sample_activations = []
    for i, sample in enumerate(sample_x):
        sample_input = np.expand_dims(sample, axis=0)
        
        activations = {}
        for layer in target_layers:
            # Create model that outputs the activation of this layer
            activation_model = tf.keras.Model(inputs=model.input, outputs=layer.output) # type: ignore
            activation = activation_model(sample_input)
            
            # Store summary statistics instead of full activation maps (to reduce size)
            activations[layer.name] = {
                'shape': activation.shape[1:],  # Exclude batch dimension
                'mean_activation': float(np.mean(activation)),
                'std_activation': float(np.std(activation)),
                'max_activation': float(np.max(activation)),
                'min_activation': float(np.min(activation))
            }
        
        sample_activations.append({
            'sample_index': int(sample_indices[i]),
            'layer_activations': activations
        })
    
    return {
        'sample_activations': sample_activations,
        'conv_layers_analyzed': [layer.name for layer in target_layers]
    }

def extract_predictions_data(model, test_data: Dict[str, Any], sample_size: int = 100) -> Dict[str, Any]:
    """Extract detailed predictions in serializable format"""
    
    x_test = test_data.get('x_test', [])
    y_test = test_data.get('y_test', [])
    
    if len(x_test) == 0 or len(y_test) == 0:
        return {'predictions': [], 'error': 'No test data available'}
    
    # Use sample for predictions
    sample_size = min(sample_size, len(x_test))
    sample_indices = np.random.choice(len(x_test), sample_size, replace=False)
    sample_x = x_test[sample_indices]
    sample_y = y_test[sample_indices]
    
    # Get predictions
    predictions = model.predict(sample_x, verbose=0)
    
    # Convert labels if needed
    if sample_y.ndim > 1 and sample_y.shape[1] > 1:
        true_labels = np.argmax(sample_y, axis=1)
    else:
        true_labels = sample_y.flatten()
    
    predicted_labels = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    # Store detailed prediction data
    predictions_data = []
    for i in range(len(sample_x)):
        predictions_data.append({
            'sample_index': int(sample_indices[i]),
            'true_label': int(true_labels[i]),
            'predicted_label': int(predicted_labels[i]),
            'confidence': float(confidence_scores[i]),
            'prediction_probabilities': predictions[i].tolist(),
            'correct': bool(true_labels[i] == predicted_labels[i])
        })
    
    return {
        'predictions': predictions_data,
        'accuracy': float(np.mean(true_labels == predicted_labels)),
        'avg_confidence': float(np.mean(confidence_scores))
    }

def validate_request(request: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate incoming training request.
    
    Args:
        request: Request dictionary from RunPod
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    logger.debug("running validate_request ... validating incoming request structure")
    
    # Required fields
    required_fields = ['command', 'dataset_name']
    
    for field in required_fields:
        if field not in request:
            error_msg = f"Missing required field: {field}"
            logger.error(f"running validate_request ... request validation failed: {error_msg}")
            return False, error_msg
    
    # Validate command
    if request['command'] != 'start_training':
        error_msg = f"Invalid command: {request['command']}. Expected 'start_training'"
        logger.error(f"running validate_request ... request validation failed: {error_msg}")
        return False, error_msg
    
    # Validate dataset_name (basic check)
    valid_datasets = ['mnist', 'cifar10', 'fashion_mnist']  # Add your supported datasets
    if request['dataset_name'] not in valid_datasets:
        error_msg = f"Unsupported dataset_name: {request['dataset_name']}. Supported: {valid_datasets}"
        logger.error(f"running validate_request ... request validation failed: {error_msg}")
        return False, error_msg
    
    logger.debug("running validate_request ... request validation successful")
    return True, None

def build_optimization_config(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build optimization config AND hyperparameters from request.
    
    Args:
        request: Request dictionary from RunPod
        
    Returns:
        Dict of parameters for optimize_model() function
    """
    logger.debug("running build_optimization_config ... building COMPLETE config from request")
    
    config_data = request.get('config', {})
    hyperparameters = request.get('hyperparameters', {})
    
    # Extract optimization parameters
    mode = config_data.get('mode', 'simple')
    objective = config_data.get('objective', 'val_accuracy')
    trials = config_data.get('trials', 1)  # Single trial per serverless call
    
    # Build configuration parameters for optimize_model()
    config_params = {
        'mode': mode,
        'optimize_for': objective,
        'trials': trials,
        'use_gpu_proxy': False,  # We ARE the GPU proxy
        'plot_generation': 'none',  # Skip plots in optimizer (we generate them in handler)
        'create_optuna_model_plots': config_data.get('create_optuna_model_plots', False)
    }
    
    # Add optional parameters if provided
    if 'validation_split' in config_data:
        config_params['validation_split'] = config_data['validation_split']
    
    if 'max_training_time' in config_data:
        config_params['max_training_time_minutes'] = config_data['max_training_time']
    
    if 'health_weight' in config_data:
        config_params['health_weight'] = config_data['health_weight']
    
    if 'max_epochs_per_trial' in config_data:
        config_params['max_epochs_per_trial'] = config_data['max_epochs_per_trial']
    
    if 'min_epochs_per_trial' in config_data:
        config_params['min_epochs_per_trial'] = config_data['min_epochs_per_trial']
    
    if 'gpu_proxy_sample_percentage' in config_data:
        config_params['gpu_proxy_sample_percentage'] = config_data['gpu_proxy_sample_percentage']
    
    # 🎯 CRITICAL FIX: Apply hyperparameters to config_params
    # These were being lost before, causing the accuracy gap
    logger.debug(f"running build_optimization_config ... APPLYING {len(hyperparameters)} hyperparameters")
    
    for param_name, param_value in hyperparameters.items():
        config_params[param_name] = param_value
        logger.debug(f"running build_optimization_config ... Applied hyperparameter: {param_name} = {param_value}")
    
    logger.debug(f"running build_optimization_config ... COMPLETE config built with {len(config_params)} total parameters")
    logger.debug(f"running build_optimization_config ... mode: {mode}, objective: {objective}")
    logger.debug(f"running build_optimization_config ... hyperparameters applied: {list(hyperparameters.keys())}")
    
    return config_params

async def start_training(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler function for RunPod serverless training requests.
    Uses existing optimizer.py orchestration for consistency.
    
    Args:
        job: RunPod job dictionary containing input request
        
    Returns:
        Structured response dictionary
    """
    logger.debug("running start_training ... starting serverless training request with COMPLETE config")
    
    trial_id = 'unknown_trial'
    
    try:
        request = job.get('input', {})
        trial_id = request.get('trial_id', 'unknown_trial')
        
        logger.debug(f"running start_training ... processing trial: {trial_id}")
        
        # Validate request
        is_valid, error_msg = validate_request(request)
        if not is_valid:
            return {
                "trial_id": trial_id,
                "status": "failed",
                "error": error_msg,
                "success": False
            }
        
        # 🎯 Build configuration (config + hyperparameters)
        all_params = build_optimization_config(request)
        
        # 🎯 Log what we're actually using
        logger.debug(f"running start_training ... VERIFICATION: Complete parameter set:")
        for key, value in all_params.items():
            logger.debug(f"running start_training ... - {key}: {value}")
        
        # 🎯 Call create_and_train_model directly, not optimize_model
        # optimize_model runs its own optimization study, ignoring our hyperparameters
        logger.debug(f"running start_training ... calling create_and_train_model with trial hyperparameters: {trial_id}")    
        
        # Create ModelConfig with hyperparameters
        model_config = ModelConfig()
        
        # Apply hyperparameters to ModelConfig
        hyperparameters = request.get('hyperparameters', {})
        for param_name, param_value in hyperparameters.items():
            if hasattr(model_config, param_name):
                setattr(model_config, param_name, param_value)
                logger.debug(f"running start_training ... Applied to ModelConfig: {param_name} = {param_value}")

        # Apply validation_split from config
        if 'validation_split' in all_params:
            model_config.validation_split = all_params['validation_split']

        # Apply gpu_proxy_sample_percentage from config  
        if 'gpu_proxy_sample_percentage' in all_params:
            model_config.gpu_proxy_sample_percentage = all_params['gpu_proxy_sample_percentage']

        # Extract and apply multi-GPU configuration from request
        config_data = request.get('config', {})
        use_multi_gpu = config_data.get('use_multi_gpu', False)
        target_gpus_per_worker = config_data.get('target_gpus_per_worker', 2)
        auto_detect_gpus = config_data.get('auto_detect_gpus', True)
        multi_gpu_batch_size_scaling = config_data.get('multi_gpu_batch_size_scaling', True)

        logger.debug(f"running start_training ... Multi-GPU configuration received:")
        logger.debug(f"running start_training ... - use_multi_gpu: {use_multi_gpu}")
        logger.debug(f"running start_training ... - target_gpus_per_worker: {target_gpus_per_worker}")
        logger.debug(f"running start_training ... - auto_detect_gpus: {auto_detect_gpus}")
        logger.debug(f"running start_training ... - multi_gpu_batch_size_scaling: {multi_gpu_batch_size_scaling}")

        # Add a default batch_size to ModelConfig for the batch size scaling to work
        if not hasattr(model_config, 'batch_size') or not model_config.batch_size:
            model_config.batch_size = 32  # Set default batch size
            logger.debug(f"running start_training ... Set default batch_size: {model_config.batch_size}")

        # Replace the section around lines 230-260 in your start_training function:

        # Get total epochs from model_config for progress reporting
        total_epochs = model_config.epochs if hasattr(model_config, 'epochs') and model_config.epochs else 10
        
        # Create progress callback to send updates to RunPod (if supported)
        def progress_callback(epoch, epoch_progress):
            """Send progress updates to RunPod during training"""
            logger.info(f"🔥 PROGRESS CALLBACK TRIGGERED: Epoch {epoch}, progress {epoch_progress}")
            try:
                # Send structured progress update to RunPod
                progress_data = {
                    'current_epoch': epoch,
                    'total_epochs': total_epochs,
                    'epoch_progress': epoch_progress,
                    'message': f"Epoch {epoch}/{total_epochs} - {epoch_progress:.1%} complete"
                }
                
                # Only send progress update if we're in RunPod environment
                if os.getenv('RUNPOD_ENDPOINT_ID'):
                    runpod.serverless.progress_update(job, progress_data)
                    logger.info(f"✅ Sent progress update: Epoch {epoch}/{total_epochs}, progress {epoch_progress:.1%}")
                else:
                    logger.info(f"🏠 Local progress: Epoch {epoch}/{total_epochs}, progress {epoch_progress:.1%}")
                
            except Exception as e:
                logger.error(f"❌ Error sending progress update: {e}")

        # Call create_and_train_model - check if it supports progress_callback parameter
        logger.info(f"🚀 CALLING create_and_train_model WITH progress_callback")
        try:
            # First, try calling with progress_callback
            training_result = create_and_train_model(
                dataset_name=request['dataset_name'],
                model_config=model_config,
                test_size=all_params.get('test_size', 0.2),
                use_multi_gpu=use_multi_gpu,
                run_name=trial_id,
                progress_callback=progress_callback
            )
            logger.info(f"✅ create_and_train_model completed successfully with progress_callback")
        except TypeError as e:
            # If progress_callback parameter is not supported, call without it
            if "progress_callback" in str(e):
                logger.debug(f"create_and_train_model doesn't support progress_callback, calling without it")
                training_result = create_and_train_model(
                    dataset_name=request['dataset_name'],
                    model_config=model_config,
                    test_size=all_params.get('test_size', 0.2),
                    use_multi_gpu=use_multi_gpu,
                    run_name=trial_id
                )
            else:
                # Re-raise if it's a different TypeError
                raise
        
        # Extract metrics from training result
        model_builder = training_result['model_builder']
        test_accuracy = training_result['test_accuracy']
        test_loss = training_result['test_loss']
        
        # Get comprehensive health metrics
        health_metrics = model_builder.get_last_health_analysis()
        
        # Generate plots locally for direct download
        config_data = request.get('config', {})

        # Create OptimizationConfig object from config_data for plot generation
        optimization_config = None
        if config_data:
            from src.data_classes.configs import OptimizationConfig
            optimization_config = OptimizationConfig(**config_data)

        plots_direct_info = generate_plots(
            model_builder=model_builder,
            dataset_name=request['dataset_name'],
            trial_id=trial_id,
            test_data=training_result.get('test_data'),
            optimization_config=optimization_config
        )

        # Save trained trial model to plots directory for later copying (similar to local approach)
        if model_builder and hasattr(model_builder, 'model') and model_builder.model:
            try:
                # Create model filename similar to final model training approach
                dataset_name = request['dataset_name']
                model_filename = f"final_model_{dataset_name}_trial_{trial_id}.keras"

                # Get plots directory path (same place where plots are saved)
                plots_dir = Path("/tmp/plots") / trial_id
                model_path_in_plots = plots_dir / model_filename

                # Save model to plots directory so it gets downloaded with plots
                model_builder.model.save(model_path_in_plots)

                logger.info(f"running start_training ... Trial model saved: {model_path_in_plots}")
                logger.info(f"running start_training ... Trial model will be included in plots download")

            except Exception as e:
                logger.error(f"running start_training ... Failed to save trial model: {e}")
                # Don't fail the entire request if model saving fails
        else:
            logger.warning(f"running start_training ... No model found to save for trial {trial_id}")

        # Skip model attributes extraction - RunPod now returns only metrics
        logger.debug(f"running start_training ... Skipping model attributes extraction (metrics-only response)")
        model_attributes = None
        
        # Build simplified response structure
        response = {
            "trial_id": trial_id,
            "status": "completed",
            "success": True,
            "metrics": {
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "val_accuracy": test_accuracy,  # Use test_accuracy as proxy
                "val_loss": test_loss,  # Use test_loss as proxy
                "best_value": test_accuracy,
                "optimize_for": all_params.get('optimize_for', 'val_accuracy'),
                "training_time_seconds": 0.0  # Not available from create_and_train_model
            },
            "health_metrics": health_metrics or {},
            "model_info": {
                "epochs_completed": model_config.epochs,
                "total_trials": 1,
                "successful_trials": 1
            },
            "best_params": hyperparameters,
            "multi_gpu_used": use_multi_gpu,
            "target_gpus": target_gpus_per_worker if use_multi_gpu else 1,
            "model_attributes": model_attributes,  # Model attributes for local plotting
            "plots_direct": plots_direct_info  # Direct plot download information
        }
        
        logger.debug(f"running start_training ... trial {trial_id} completed with COMPLETE config")
        logger.debug(f"running start_training ... best value: {test_accuracy:.4f}")
        
        # DETAILED RESPONSE ANALYSIS LOGGING
        logger.debug("=" * 60)
        logger.debug("RUNPOD RESPONSE ANALYSIS")
        logger.debug("=" * 60)
        
        # Log response structure and types
        logger.debug(f"Response keys: {list(response.keys())}")
        for key, value in response.items():
            logger.debug(f"- {key}: type={type(value).__name__}, length={len(str(value))}")
            if hasattr(value, '__len__') and not isinstance(value, (str, int, float, bool)):
                try:
                    logger.debug(f"  └─ Container length: {len(value)}")
                except:
                    pass
        
        # Detailed analysis of potentially problematic fields
        logger.debug("\nDETAILED FIELD ANALYSIS:")
        
        # Health metrics analysis
        if health_metrics:
            logger.debug(f"health_metrics type: {type(health_metrics)}")
            logger.debug(f"health_metrics keys: {list(health_metrics.keys()) if hasattr(health_metrics, 'keys') else 'No keys'}")
            for hm_key, hm_value in (health_metrics.items() if hasattr(health_metrics, 'items') else []):
                logger.debug(f"  - {hm_key}: {type(hm_value).__name__}")
        
        # Model attributes analysis  
        if model_attributes:
            logger.debug(f"model_attributes type: {type(model_attributes)}")
            logger.debug(f"model_attributes keys: {list(model_attributes.keys()) if hasattr(model_attributes, 'keys') else 'No keys'}")
            for ma_key, ma_value in (model_attributes.items() if hasattr(model_attributes, 'items') else []):
                logger.debug(f"  - {ma_key}: {type(ma_value).__name__}")
                if ma_value is not None:
                    if hasattr(ma_value, 'shape'):
                        logger.debug(f"    └─ Shape: {ma_value.shape}")
                    elif hasattr(ma_value, '__len__'):
                        try:
                            logger.debug(f"    └─ Length: {len(ma_value)}")
                        except:
                            pass
        
        # Best params analysis
        logger.debug(f"best_params type: {type(hyperparameters)}")
        if hasattr(hyperparameters, 'items'):
            for bp_key, bp_value in hyperparameters.items():
                logger.debug(f"  - {bp_key}: {type(bp_value).__name__} = {bp_value}")
        
        # Test JSON serialization
        logger.debug("\nJSON SERIALIZATION TEST:")
        try:
            json_str = json.dumps(response)
            logger.debug(f"✅ JSON serialization: SUCCESS")
            logger.debug(f"✅ JSON size: {len(json_str.encode('utf-8'))} bytes")
        except Exception as e:
            logger.error(f"❌ JSON serialization: FAILED - {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            
            # Try to identify which field is problematic
            logger.debug("TESTING INDIVIDUAL FIELDS:")
            for key, value in response.items():
                try:
                    json.dumps({key: value})
                    logger.debug(f"  ✅ {key}: serializable")
                except Exception as field_error:
                    logger.error(f"  ❌ {key}: NOT serializable - {field_error}")
        
        logger.debug("=" * 60)
        
        return response
        
    except Exception as e:
        logger.error(f"running start_training ... trial {trial_id} failed: {str(e)}")
        logger.error(f"running start_training ... error traceback: {traceback.format_exc()}")
        
        return {
            "trial_id": trial_id,
            "status": "failed",
            "error": str(e),
            "success": False
        }

async def start_final_model_training(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for final model training requests.
    Trains the final model using best hyperparameters from optimization.
    
    Args:
        job: RunPod job dictionary containing final model training request
        
    Returns:
        Structured response dictionary with model path and metrics
    """
    logger.debug("running start_final_model_training ... starting final model training request")
    
    
    try:
        request = job.get('input', {})
        
        # Validate final model training request
        required_fields = ['dataset_name', 'best_params', 'config']
        for field in required_fields:
            if field not in request:
                return {
                    "status": "failed",
                    "error": f"Missing required field: {field}",
                    "success": False
                }
        
        logger.debug(f"running start_final_model_training ... training final model with best params: {request['best_params']}")
        
        # Build configuration for final model
        all_params = build_optimization_config(request)
        
        # Create ModelConfig with best hyperparameters
        model_config = ModelConfig()
        best_params = request.get('best_params', {})
        for param_name, param_value in best_params.items():
            if hasattr(model_config, param_name):
                # Handle kernel_size conversion from int to tuple
                if param_name == 'kernel_size' and isinstance(param_value, int):
                    param_value = (param_value, param_value)
                    logger.debug(f"running start_final_model_training ... Converted kernel_size from int {best_params['kernel_size']} to tuple {param_value}")
                
                setattr(model_config, param_name, param_value)
                logger.debug(f"running start_final_model_training ... Applied to ModelConfig: {param_name} = {param_value}")

        # Extract multi-GPU configuration from request
        config_data = request.get('config', {})
        use_multi_gpu = config_data.get('use_multi_gpu', False)
        
        # Add validation split from config
        if 'validation_split' in all_params:
            model_config.validation_split = all_params['validation_split']
        
        # Progress callback for final model training with RunPod progress updates
        def progress_callback_func(current_epoch: int, epoch_progress: float):
            """Send real-time epoch progress updates to RunPod during final model training"""
            logger.debug(f"running start_final_model_training ... Epoch {current_epoch} progress: {epoch_progress:.1%}")
            
            try:
                # Send structured progress update to RunPod (same format as trial training)
                progress_data = {
                    'current_epoch': current_epoch,
                    'total_epochs': best_params.get('epochs', 5),  # Get epochs from best_params
                    'epoch_progress': epoch_progress,
                    'message': f"Final model - Epoch {current_epoch}/{best_params.get('epochs', 5)} - {epoch_progress:.1%} complete",
                    'final_model': True  # Flag to distinguish from trial progress
                }
                
                # Only send progress update if we're in RunPod environment
                if os.getenv('RUNPOD_ENDPOINT_ID'):
                    runpod.serverless.progress_update(job, progress_data)
                    logger.info(f"🏗️ Final Model Progress: Epoch {current_epoch}/{best_params.get('epochs', 5)}, progress {epoch_progress:.1%}")
                else:
                    logger.info(f"🏠 Local Final Model Progress: Epoch {current_epoch}/{best_params.get('epochs', 5)}, progress {epoch_progress:.1%}")
                
            except Exception as e:
                logger.error(f"❌ Error sending final model progress update: {e}")
                # Continue training even if progress update fails
        
        # Train final model with best hyperparameters
        logger.debug(f"running start_final_model_training ... Calling create_and_train_model for final model")
        logger.debug(f"running start_final_model_training ... Parameters: dataset={request['dataset_name']}, multi_gpu={use_multi_gpu}, test_size={all_params.get('test_size', 0.2)}")
        
        try:
            training_result = create_and_train_model(
                dataset_name=request['dataset_name'],
                model_config=model_config,
                test_size=all_params.get('test_size', 0.2),
                use_multi_gpu=use_multi_gpu,
                run_name=f"final_model_{request['dataset_name']}",
                progress_callback=progress_callback_func
            )
            logger.debug(f"running start_final_model_training ... create_and_train_model completed, result type: {type(training_result)}")
        except Exception as e:
            logger.error(f"running start_final_model_training ... create_and_train_model failed: {e}")
            logger.error(f"running start_final_model_training ... Traceback: {traceback.format_exc()}")
            raise
        
        if training_result and isinstance(training_result, dict) and 'test_accuracy' in training_result:
            logger.info(f"✅ Final model training completed successfully")
            
            # DEBUG: Log training_result structure
            logger.debug(f"running start_final_model_training ... DEBUG training_result keys: {list(training_result.keys())}")
            logger.debug(f"running start_final_model_training ... DEBUG model_builder present: {'model_builder' in training_result}")
            if 'model_builder' in training_result:
                logger.debug(f"running start_final_model_training ... DEBUG model_builder value: {training_result['model_builder']}")
                logger.debug(f"running start_final_model_training ... DEBUG model_builder type: {type(training_result['model_builder'])}")
            
            # Generate plots locally for direct download
            config_data = request.get('config', {})
            plots_direct_info = None

            # Initialize model_attributes to ensure it's always defined
            model_attributes = None

            # Save final model BEFORE generating plots so it's included in batch download
            model_path = None
            plots_direct_info = None

            if 'model_builder' in training_result and training_result['model_builder']:
                model_builder_obj = training_result['model_builder']
                logger.debug(f"running start_final_model_training ... DEBUG: model_builder conditional passed - proceeding to save model")

                try:
                    # Save model to plots directory first, before generating plots
                    plots_dir = Path("/tmp/plots") / request['run_name']
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    # Generate model filename
                    test_accuracy = training_result['test_accuracy']
                    accuracy_str = f"acc_{test_accuracy:.4f}".replace('.', 'p')
                    model_filename = f"optimized_{request['dataset_name']}_{accuracy_str}_model.keras"

                    # Save model directly to plots directory
                    model_path_in_plots = plots_dir / model_filename
                    model_builder_obj.model.save(model_path_in_plots)

                    model_path = str(model_path_in_plots)
                    logger.info(f"running start_final_model_training ... Final model saved to plots directory: {model_path}")
                    logger.info(f"running start_final_model_training ... Model will be included in batch download with plots")

                    # Debug: List what's in the plots directory before generating plots
                    try:
                        files_in_plots_dir = list(plots_dir.iterdir())
                        logger.info(f"running start_final_model_training ... Contents of {plots_dir} before plot generation:")
                        for file_path in files_in_plots_dir:
                            logger.info(f"   - {file_path.name} ({file_path.stat().st_size} bytes)")
                        logger.info(f"running start_final_model_training ... Total files before plot generation: {len(files_in_plots_dir)}")
                    except Exception as e:
                        logger.warning(f"running start_final_model_training ... Failed to list plots directory contents: {e}")

                except Exception as e:
                    logger.error(f"running start_final_model_training ... Failed to save final model to plots directory: {e}")

                # Now generate plots AFTER model is saved
                # Create OptimizationConfig object from config_data for plot generation
                optimization_config = None
                if config_data:
                    from src.data_classes.configs import OptimizationConfig
                    optimization_config = OptimizationConfig(**config_data)

                # DEBUG: Check files BEFORE plot generation
                plots_dir = Path("/tmp/plots") / request['run_name']
                if plots_dir.exists():
                    logger.info(f"running start_final_model_training ... Contents of {plots_dir} BEFORE plot generation:")
                    all_files_before = []
                    for file_path in sorted(plots_dir.rglob('*')):
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            all_files_before.append(file_path.name)
                            logger.info(f"running start_final_model_training ...    - {file_path.name} ({size} bytes)")
                    logger.info(f"running start_final_model_training ... Total files BEFORE plot generation: {len(all_files_before)}")
                    logger.info(f"running start_final_model_training ... File list BEFORE: {all_files_before}")

                plots_direct_info = generate_plots(
                    model_builder=model_builder_obj,
                    dataset_name=request['dataset_name'],
                    trial_id=request['run_name'],  # Use run name to match directory with trials
                    test_data=training_result.get('test_data'),
                    optimization_config=optimization_config
                )

                # DEBUG: Check files AFTER plot generation
                plots_dir = Path("/tmp/plots") / request['run_name']
                if plots_dir.exists():
                    logger.info(f"running start_final_model_training ... Contents of {plots_dir} AFTER plot generation:")
                    all_files = []
                    for file_path in sorted(plots_dir.rglob('*')):
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            all_files.append(file_path.name)
                            logger.info(f"running start_final_model_training ...    - {file_path.name} ({size} bytes)")
                    logger.info(f"running start_final_model_training ... Total files AFTER plot generation: {len(all_files)}")
                    logger.info(f"running start_final_model_training ... File list: {all_files}")

                # run_name is already correct from generate_plots using request['run_name']
                # No override needed since plots and model use same directory

                # Skip model attributes extraction - RunPod now returns only metrics
                logger.debug(f"running start_final_model_training ... Skipping model attributes extraction (metrics-only response)")
            
            return {
                "status": "completed",
                "success": True,
                "final_model_path": model_path,
                "test_accuracy": training_result.get('test_accuracy', 0.0),
                "test_loss": training_result.get('test_loss', 0.0),
                "training_time_seconds": training_result.get('training_time_seconds', 0.0),
                "multi_gpu_used": use_multi_gpu,
                "model_attributes": model_attributes,  # Model attributes for local plotting
                "plots_direct": plots_direct_info  # Direct plot download information
            }
        else:
            logger.error(f"running start_final_model_training ... Final model training failed or returned invalid result")
            return {
                "status": "failed",
                "error": "Final model training failed",
                "success": False
            }
            
    except Exception as e:
        error_msg = f"Final model training failed: {str(e)}"
        logger.error(f"running start_final_model_training ... {error_msg}")
        logger.error(f"running start_final_model_training ... Traceback: {traceback.format_exc()}")
        
        return {
            "status": "failed", 
            "error": error_msg,
            "success": False
        }


async def handle_simple_http_endpoints(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle simple HTTP-like endpoints through RunPod's main handler.
    """
    try:
        command = event.get('input', {}).get('command')
        logger.info(f"Handling simple HTTP endpoint: {command}")

        if command == 'health':
            return {
                "status": "healthy",
                "service": "cv-classification-direct-download",
                "port": "RunPod handler (not FastAPI)",
                "timestamp": datetime.now().isoformat()
            }

        elif command == 'list_files':
            run_name = event.get('input', {}).get('run_name')
            if not run_name:
                return {"error": "run_name parameter required", "status_code": 400}

            plots_dir = Path("/tmp/plots") / run_name
            if not plots_dir.exists():
                return {"error": f"Run {run_name} not found", "status_code": 404}

            files = []
            for file_path in plots_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(plots_dir)
                    files.append(str(relative_path))

            return {"run_name": run_name, "files": files}

        elif command == 'download_file':
            run_name = event.get('input', {}).get('run_name')
            file_path = event.get('input', {}).get('file_path')

            if not run_name or not file_path:
                return {"error": "run_name and file_path parameters required", "status_code": 400}

            plots_dir = Path("/tmp/plots") / run_name
            full_file_path = plots_dir / file_path

            # Security check
            if not str(full_file_path.resolve()).startswith(str(plots_dir.resolve())):
                return {"error": "Invalid file path", "status_code": 400}

            if not full_file_path.exists() or not full_file_path.is_file():
                return {"error": f"File {file_path} not found in run {run_name}", "status_code": 404}

            # Read file and return as base64 (simple approach)
            import base64
            with open(full_file_path, 'rb') as f:
                file_content = base64.b64encode(f.read()).decode('utf-8')

            return {
                "run_name": run_name,
                "file_path": file_path,
                "filename": full_file_path.name,
                "content": file_content,
                "encoding": "base64",
                "size": full_file_path.stat().st_size
            }

        elif command == 'download_directory':
            run_name = event.get('input', {}).get('run_name')
            if not run_name:
                return {"error": "run_name parameter required", "status_code": 400}

            plots_dir = Path("/tmp/plots") / run_name
            logger.info(f"download_directory ... Looking for directory: {plots_dir}")

            # Debug: List what's available in /tmp/plots to help troubleshoot
            try:
                base_plots_dir = Path("/tmp/plots")
                if base_plots_dir.exists():
                    available_dirs = [d.name for d in base_plots_dir.iterdir() if d.is_dir()]
                    logger.info(f"download_directory ... Available directories in /tmp/plots: {available_dirs}")
                else:
                    logger.warning(f"download_directory ... /tmp/plots directory doesn't exist!")
            except Exception as e:
                logger.warning(f"download_directory ... Failed to list /tmp/plots contents: {e}")

            if not plots_dir.exists() or not plots_dir.is_dir():
                logger.error(f"download_directory ... Directory {plots_dir} not found or not a directory")
                return {"error": f"Directory for run {run_name} not found", "status_code": 404}

            # Create temporary zip file
            import tempfile
            import zipfile
            import base64

            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                zip_path = tmp_zip.name

                # Create zip archive of the entire directory
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in plots_dir.rglob('*'):
                        if file_path.is_file():
                            # Add file to zip with relative path
                            arcname = file_path.relative_to(plots_dir)
                            zipf.write(file_path, arcname)
                            logger.debug(f"Added to zip: {arcname}")

                # Read zip file and encode as base64
                with open(zip_path, 'rb') as f:
                    zip_content = base64.b64encode(f.read()).decode('utf-8')

                # Get zip file size and file count (only count files, not directories)
                zip_size = Path(zip_path).stat().st_size
                file_count = len([f for f in plots_dir.rglob('*') if f.is_file()])

                # Clean up temporary file
                os.unlink(zip_path)

                logger.info(f"Created zip archive for {run_name}: {file_count} files, {zip_size} bytes")

                return {
                    "run_name": run_name,
                    "filename": f"{run_name}_plots.zip",
                    "content": zip_content,
                    "encoding": "base64",
                    "size": zip_size,
                    "file_count": file_count,
                    "compression": "zip"
                }

        else:
            return {"error": f"Unknown command: {command}", "status_code": 400}

    except Exception as e:
        logger.error(f"Error handling simple HTTP endpoint: {e}")
        return {"error": str(e), "status_code": 500}


async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler function.
    Processes incoming serverless requests and routes to appropriate handlers.
    Also handles simple HTTP endpoints for file downloads.

    Args:
        event: RunPod event dictionary containing job information

    Returns:
        Response dictionary for RunPod
    """
    logger.debug("running handler ... processing RunPod serverless request")

    # Handle simple HTTP endpoints for file downloads
    if isinstance(event, dict) and event.get('input', {}).get('command') in ['health', 'list_files', 'download_file', 'download_directory']:
        return await handle_simple_http_endpoints(event)

    # 🔍 DEBUG: Log the entire event structure
    logger.debug(f"running handler ... DEBUG: event type: {type(event)}")
    logger.debug(f"running handler ... DEBUG: event keys: {list(event.keys()) if isinstance(event, dict) else 'not a dict'}")
    logger.debug(f"running handler ... DEBUG: full event content: {event}")

    # Initialize trial_id before try block to ensure it's always available
    trial_id = 'unknown_trial'
    
    try:
        # Extract job from event
        job = event.get('job', event)  # Handle both event formats
        
        if not job:
            error_msg = "No job found in event"
            logger.error(f"running handler ... {error_msg}")
            return {"error": error_msg, "success": False}
        else:
            # 🔍 DEBUG: Log what we extracted from event
            logger.debug(f"running handler ... DEBUG: job type: {type(job)}")
            logger.debug(f"running handler ... DEBUG: job keys: {list(job.keys()) if isinstance(job, dict) else 'not a dict'}")
            logger.debug(f"running handler ... DEBUG: job content: {job}")
        
        # Extract input from job
        request = job.get('input', {})
        # 🔍 DEBUG: Log what we extracted from job
        logger.debug(f"running handler ... DEBUG: request type: {type(request)}")
        logger.debug(f"running handler ... DEBUG: request keys: {list(request.keys()) if isinstance(request, dict) else 'not a dict'}")
        logger.debug(f"running handler ... DEBUG: request content: {request}")
        
        command = request.get('command', 'unknown')
        logger.debug(f"running handler ... command: {command}")
        
        # Route to appropriate handler
        if command == 'start_training':
            return await start_training(job)  # ✅ Awaits coroutine to get Dict
        elif command == 'start_final_model_training':
            return await start_final_model_training(job)  # ✅ Final model training
        else:
            error_msg = f"Unknown command: {command}"
            logger.error(f"running handler ... {error_msg}")
            return {"error": error_msg, "success": False}
            
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(f"running handler ... {error_msg}")
        logger.error(f"running handler ... traceback: {traceback.format_exc()}")
        
        return {
            "error": error_msg,
            "success": False
        }

# Simple HTTP endpoints now handled through main RunPod handler using RunPod's standard request/response pattern
logger.info("SIMPLE HTTP ENDPOINTS: Ready to handle health, list_files, and download_file commands")
logger.info("SIMPLE HTTP ENDPOINTS: Access via RunPod API with commands: health, list_files, download_file")

# Ensure handler is accessible for import
logger.info("HANDLER MODULE: handler.py loaded successfully, all functions available")

# RunPod serverless entry point
async def runpod_handler(event):
    """
    RunPod serverless entry point.
    This is the function that RunPod will call for each serverless request.
    """
    # RunPod expects a sync function, so we need to run the async handler
    return await handler(event)


if __name__ == "__main__":
    print("=== HANDLER.PY STARTUP ===")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"RUNPOD_ENDPOINT_ID: {os.getenv('RUNPOD_ENDPOINT_ID')}")
    print("=========================")

    try:
        logger.info("=== HANDLER.PY STARTUP ===")
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"RUNPOD_ENDPOINT_ID: {os.getenv('RUNPOD_ENDPOINT_ID')}")
        logger.info("=========================")

        # Start RunPod serverless handler
        logger.info("Starting RunPod serverless handler...")
        print("Starting RunPod serverless handler...")
        runpod.serverless.start({
            "handler": handler,
            "concurrency_modifier": adjust_concurrency
        })

    except Exception as e:
        print(f"ERROR in handler startup: {e}")
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}")
        raise