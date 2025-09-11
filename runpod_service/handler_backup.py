"""
RunPod Service Handler - Thin wrapper around existing optimizer.py orchestration
Implements specialized serverless hyperparameter optimization using existing battle-tested logic.
"""

# Web server wrapper for local testing and container deployment
from fastapi import FastAPI, HTTPException
import json
import numpy as np
import boto3
import os
from pathlib import Path
from pydantic import BaseModel
import runpod
import sys
import tempfile
import tensorflow as tf
import traceback
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_classes.configs import OptimizationConfig
from datetime import datetime
import uvicorn

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


def generate_and_upload_plots_to_s3(
    model_builder,
    dataset_name: str, 
    trial_id: str,
    optimization_config: Optional['OptimizationConfig'] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate plots using PlotGenerator and upload to S3 (RunPod environment only).
    
    Args:
        model_builder: ModelBuilder instance with trained model
        dataset_name: Name of the dataset
        trial_id: Trial identifier for S3 prefix
        optimization_config: Optional OptimizationConfig object containing plot flags and settings
        
    Returns:
        Dictionary with S3 upload info if successful, None if skipped/failed
    """
    # Only proceed if running in RunPod environment and plots should be generated
    if not os.getenv('RUNPOD_ENDPOINT_ID'):
        logger.debug(f"generate_and_upload_plots_to_s3 ... Skipping plot generation: not in RunPod environment")
        return None
        
    # Check plot generation setting from optimization_config
    if optimization_config and getattr(optimization_config, 'plot_generation', 'all') == 'none':
        logger.debug(f"generate_and_upload_plots_to_s3 ... Skipping plot generation: plot_generation='none'")
        return None
        
    try:
        logger.debug(f"generate_and_upload_plots_to_s3 ... Generating plots for trial {trial_id}")
        
        # Import here to avoid circular dependencies
        from src.plot_generator import PlotGenerator
        from pathlib import Path
        import tempfile
        
        # Create temporary directory for plots
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_dir = Path(temp_dir)
            
            # Get dataset config and model config from model_builder
            if not hasattr(model_builder, 'dataset_config') or not hasattr(model_builder, 'model_config'):
                logger.warning(f"generate_and_upload_plots_to_s3 ... ModelBuilder missing required configs, skipping plot generation")
                return None
            
            # Create PlotGenerator with provided or fallback optimization_config
            effective_optimization_config = optimization_config or getattr(model_builder, 'optimization_config', None)
            
            plot_generator = PlotGenerator(
                dataset_config=model_builder.dataset_config,
                model_config=model_builder.model_config,
                optimization_config=effective_optimization_config
            )
            
            # Get test data from model_builder
            test_data = getattr(model_builder, 'test_data', None)
            if not test_data:
                logger.warning(f"generate_and_upload_plots_to_s3 ... No test data available, skipping plot generation")
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
                    logger.warning(f"generate_and_upload_plots_to_s3 ... Could not evaluate model: {e}")
            
            # Generate comprehensive plots
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            
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
                progress_callback=None  # No progress callback for RunPod plot generation
            )
            
            # Check if any plots were generated
            generated_plots = []
            for plot_type, result in analysis_results.items():
                if result and not result.get('error'):
                    generated_plots.append(plot_type)
            
            if not generated_plots:
                logger.warning(f"generate_and_upload_plots_to_s3 ... No plots were generated successfully")
                return None
                
            logger.debug(f"generate_and_upload_plots_to_s3 ... Generated plots: {', '.join(generated_plots)}")
            
            # Upload plots directory to S3
            s3_prefix = f"trial_plots/{trial_id}_{dataset_name}"
            upload_success = upload_to_runpod_s3(str(plot_dir), s3_prefix)
            
            if upload_success:
                logger.info(f"generate_and_upload_plots_to_s3 ... ✅ Successfully uploaded plots to S3: s3://40ub9vhaa7/{s3_prefix}")
                return {
                    "success": True,
                    "s3_prefix": s3_prefix,
                    "bucket": "40ub9vhaa7",
                    "generated_plots": generated_plots
                }
            else:
                logger.error(f"generate_and_upload_plots_to_s3 ... ❌ Failed to upload plots to S3")
                return {
                    "success": False,
                    "error": "Failed to upload plots to S3"
                }
                
    except Exception as e:
        logger.error(f"generate_and_upload_plots_to_s3 ... Plot generation and upload failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def upload_to_runpod_s3(local_path: str, s3_key: str) -> bool:
    """
    Upload a file or directory to RunPod S3 storage.
    
    Args:
        local_path: Local file or directory path to upload
        s3_key: S3 object key (path) for the uploaded file/directory
        
    Returns:
        True if upload successful, False otherwise
    """
    try:
        # Get S3 credentials from environment
        access_key = os.getenv('RUNPOD_S3_ACCESS_KEY')
        secret_key = os.getenv('RUNPOD_S3_SECRET_ACCESS_KEY')
        
        # DEBUG: Check credential availability (don't log actual values for security)
        logger.debug(f"S3 credentials check - access_key present: {bool(access_key)}, secret_key present: {bool(secret_key)}")
        if access_key:
            logger.debug(f"S3 access_key starts with: {access_key[:8]}...")
        if secret_key:
            logger.debug(f"S3 secret_key starts with: {secret_key[:8]}...")
        
        if not access_key or not secret_key:
            logger.error("RunPod S3 credentials not found in environment")
            return False
        
        # Initialize S3 client for RunPod with correct endpoint and region
        endpoint_url = 'https://s3api-us-ks-2.runpod.io'
        region_name = 'us-ks-2'
        bucket_name = '40ub9vhaa7'
        
        logger.debug(f"Connecting to RunPod S3 - Endpoint: {endpoint_url}, Region: {region_name}, Bucket: {bucket_name}")
        
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        
        # Ensure bucket exists - test access first
        try:
            logger.debug(f"Testing bucket access for: {bucket_name}")
            s3_client.head_bucket(Bucket=bucket_name)
            logger.debug(f"✅ Bucket {bucket_name} exists and is accessible")
        except Exception as e:
            logger.warning(f"Bucket access test failed for {bucket_name}: {e}")
            logger.debug(f"Full bucket access error: {str(e)}")
            
            # Try to list buckets to see what's available
            try:
                logger.debug("Attempting to list available buckets...")
                response = s3_client.list_buckets()
                bucket_names = [b['Name'] for b in response.get('Buckets', [])]
                logger.debug(f"Available buckets: {bucket_names}")
            except Exception as list_error:
                logger.warning(f"Could not list buckets: {list_error}")
            
            # Don't try to create - just continue and let the upload fail with better error info
        
        local_path_obj = Path(local_path)
        
        if local_path_obj.is_file():
            # Upload single file
            logger.debug(f"Uploading file {local_path_obj} to S3 key {s3_key}")
            s3_client.upload_file(str(local_path_obj), bucket_name, s3_key)
            logger.debug(f"Successfully uploaded {local_path_obj} to s3://{bucket_name}/{s3_key}")
            return True
            
        elif local_path_obj.is_dir():
            # Upload directory recursively
            logger.debug(f"Uploading directory {local_path_obj} to S3 prefix {s3_key}")
            
            for file_path in local_path_obj.rglob('*'):
                if file_path.is_file():
                    # Calculate relative path for S3 key
                    relative_path = file_path.relative_to(local_path_obj)
                    file_s3_key = f"{s3_key}/{relative_path}".replace('\\', '/')
                    
                    logger.debug(f"Uploading {file_path} to {file_s3_key}")
                    s3_client.upload_file(str(file_path), bucket_name, file_s3_key)
            
            logger.debug(f"Successfully uploaded directory {local_path_obj} to s3://{bucket_name}/{s3_key}")
            return True
        else:
            logger.error(f"Path {local_path} does not exist")
            return False
            
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to S3: {e}")
        return False


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
                    use_multi_gpu=use_multi_gpu
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
        
        # Generate and upload plots to S3 (RunPod environment only)
        config_data = request.get('config', {})
        
        # Create OptimizationConfig object from config_data for plot generation
        optimization_config = None
        if config_data:
            from src.data_classes.configs import OptimizationConfig
            optimization_config = OptimizationConfig(**config_data)
        
        plots_s3_info = generate_and_upload_plots_to_s3(
            model_builder=model_builder,
            dataset_name=request['dataset_name'],
            trial_id=trial_id,
            optimization_config=optimization_config
        )

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
            "plots_s3": plots_s3_info  # S3 plot upload information
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
                progress_callback=progress_callback_func
            )
            logger.debug(f"running start_final_model_training ... create_and_train_model completed, result type: {type(training_result)}")
        except Exception as e:
            logger.error(f"running start_final_model_training ... create_and_train_model failed: {e}")
            logger.error(f"running start_final_model_training ... Traceback: {traceback.format_exc()}")
            raise
        
        if training_result and isinstance(training_result, dict) and 'test_accuracy' in training_result:
            logger.info(f"✅ Final model training completed successfully")
            
            # DEBUG: Log training_result structure to understand S3 upload issue
            logger.debug(f"running start_final_model_training ... DEBUG training_result keys: {list(training_result.keys())}")
            logger.debug(f"running start_final_model_training ... DEBUG model_builder present: {'model_builder' in training_result}")
            if 'model_builder' in training_result:
                logger.debug(f"running start_final_model_training ... DEBUG model_builder value: {training_result['model_builder']}")
                logger.debug(f"running start_final_model_training ... DEBUG model_builder type: {type(training_result['model_builder'])}")
            
            # Generate and upload plots to S3 (RunPod environment only)
            config_data = request.get('config', {})
            plots_s3_info = None
            
            # Initialize model_attributes to ensure it's always defined
            model_attributes = None
            
            if 'model_builder' in training_result and training_result['model_builder']:
                model_builder_obj = training_result['model_builder']
                
                # Create OptimizationConfig object from config_data for plot generation
                optimization_config = None
                if config_data:
                    from src.data_classes.configs import OptimizationConfig
                    optimization_config = OptimizationConfig(**config_data)
                
                plots_s3_info = generate_and_upload_plots_to_s3(
                    model_builder=model_builder_obj,
                    dataset_name=request['dataset_name'],
                    trial_id="final_model",  # Use fixed identifier for final model
                    optimization_config=optimization_config
                )
                
                # Skip model attributes extraction - RunPod now returns only metrics
                logger.debug(f"running start_final_model_training ... Skipping model attributes extraction (metrics-only response)")
            
            # Save final model
            model_path = None
            s3_upload_info = None
            
            if 'model_builder' in training_result and training_result['model_builder']:
                logger.debug(f"running start_final_model_training ... DEBUG: model_builder conditional passed - proceeding to save model")
                try:
                    model_path = training_result['model_builder'].save_model(
                        test_accuracy=training_result['test_accuracy'],
                        run_name=f"optimized_{request['dataset_name']}"
                    )
                    logger.debug(f"running start_final_model_training ... Final model saved to: {model_path}")
                    
                    # DEBUG: Check S3 upload conditions
                    logger.debug(f"running start_final_model_training ... DEBUG: model_path exists: {bool(model_path)}")
                    logger.debug(f"running start_final_model_training ... DEBUG: RUNPOD_ENDPOINT_ID: {os.getenv('RUNPOD_ENDPOINT_ID')}")
                    
                    # Only upload to S3 if running in RunPod environment
                    if model_path and os.getenv('RUNPOD_ENDPOINT_ID'):
                        logger.debug(f"running start_final_model_training ... Running in RunPod environment, uploading model artifacts to S3")
                        
                        # Create unique S3 prefix for this job
                        job_id = job.get('id', 'unknown_job')
                        dataset_name = request.get('dataset_name', 'unknown_dataset')
                        s3_prefix = f"final_models/{job_id}_{dataset_name}"
                        
                        # Get model directory (parent of the .keras file)
                        model_file_path = Path(model_path)
                        model_dir = model_file_path.parent
                        
                        # Upload entire model directory to S3
                        upload_success = upload_to_runpod_s3(str(model_dir), s3_prefix)
                        
                        if upload_success:
                            s3_upload_info = {
                                "success": True,
                                "s3_prefix": s3_prefix,
                                "bucket": "40ub9vhaa7"
                            }
                            logger.info(f"✅ Successfully uploaded model artifacts to S3: s3://40ub9vhaa7/{s3_prefix}")
                        else:
                            s3_upload_info = {
                                "success": False,
                                "error": "Failed to upload to S3"
                            }
                            logger.error(f"❌ Failed to upload model artifacts to S3")
                    else:
                        logger.debug(f"running start_final_model_training ... Running locally, skipping S3 upload")
                    
                except Exception as e:
                    logger.error(f"running start_final_model_training ... Failed to save final model: {e}")
            
            return {
                "status": "completed",
                "success": True,
                "final_model_path": model_path,
                "test_accuracy": training_result.get('test_accuracy', 0.0),
                "test_loss": training_result.get('test_loss', 0.0),
                "training_time_seconds": training_result.get('training_time_seconds', 0.0),
                "multi_gpu_used": use_multi_gpu,
                "model_attributes": model_attributes,  # Model attributes for local plotting
                "s3_upload": s3_upload_info,  # S3 upload information for local download
                "plots_s3": plots_s3_info  # S3 plot upload information
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


async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler function.
    Processes incoming serverless requests and routes to appropriate handlers.
    
    Args:
        event: RunPod event dictionary containing job information
        
    Returns:
        Response dictionary for RunPod
    """
    logger.debug("running handler ... processing RunPod serverless request")
    
    
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

# RunPod serverless entry point
async def runpod_handler(event):
    """
    RunPod serverless entry point.
    This is the function that RunPod will call for each serverless request.
    """
    # RunPod expects a sync function, so we need to run the async handler
    return await handler(event)


if __name__ == "__main__":
    
    # Check if running in RunPod environment vs local development
    if os.getenv('RUNPOD_ENDPOINT_ID'):
        # Running in RunPod serverless environment
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({
            "handler": handler,
            "concurrency_modifier": adjust_concurrency
        })
    else:
        # Running locally for development/testing
        logger.info("Starting FastAPI web server for local development...")
        app = FastAPI(title="CV Classification Optimizer", version="1.0.0")
        
        class TrainingRequest(BaseModel):
            command: str
            trial_id: str
            dataset_name: str
            hyperparameters: Dict[str, Any]
            config: Dict[str, Any]
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "cv-classification-optimizer"}
        
        @app.post("/")
        async def web_handler(request: TrainingRequest):
            """FastAPI wrapper around RunPod handler"""
            try:
                # Convert Pydantic model to dict
                data = request.model_dump()
                
                # Wrap in RunPod event format
                event = {"input": data}
                
                # Call your existing RunPod handler - need to await the async function
                result = await handler(event)
                
                return result
                
            except Exception as e:
                logger.error(f"Web handler error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        logger.info("FastAPI server starting...")
        logger.info("API docs available at http://localhost:8080/docs")
        uvicorn.run(app, host="0.0.0.0", port=8080)