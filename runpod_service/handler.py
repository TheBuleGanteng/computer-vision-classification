"""
RunPod Service Handler - Thin wrapper around existing optimizer.py orchestration
Implements specialized serverless hyperparameter optimization using existing battle-tested logic.
"""

# Web server wrapper for local testing and container deployment
import base64
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
import json
import numpy as np
import os
from pathlib import Path
from pydantic import BaseModel
import runpod
import sys
import tempfile
import tensorflow as tf
import traceback
from typing import Dict, Any, Optional, TYPE_CHECKING, List
import zipfile
            
from src.data_classes.configs import OptimizationConfig
# Import here to avoid circular dependencies
from src.plot_generator import PlotGenerator            


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


def get_s3_client():
    """
    Create and return boto3 S3 client configured for RunPod S3.

    Follows RunPod's official documentation pattern exactly.
    Requires environment variables:
    - AWS_ACCESS_KEY_ID or S3_ACCESS_KEY_RUNPOD
    - AWS_SECRET_ACCESS_KEY or S3_SECRET_ACCESS_KEY_RUNPOD
    - S3_VOLUME_DATACENTER_RUNPOD

    Returns:
        boto3.client: Configured S3 client
    """
    # Try standard AWS env vars first, then fall back to custom RUNPOD_S3_* names
    s3_access_key_runpod = os.getenv('S3_ACCESS_KEY_RUNPOD') or os.getenv('AWS_ACCESS_KEY_ID')
    s3_secret_key = os.getenv('S3_SECRET_ACCESS_KEY_RUNPOD') or os.getenv('AWS_SECRET_ACCESS_KEY')
    datacenter = os.getenv('S3_VOLUME_DATACENTER_RUNPOD')

    if not s3_access_key_runpod or not s3_secret_key:
        raise ValueError("S3 credentials not found in environment variables (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)")

    if not datacenter:
        raise ValueError("S3_VOLUME_DATACENTER_RUNPOD not found in environment variables")

    # Construct datacenter-specific endpoint URL (lowercase)
    datacenter_lower = datacenter.lower()
    s3_endpoint = f"https://s3api-{datacenter_lower}.runpod.io"

    logger.info(f"get_s3_client ... Configuring S3 client: endpoint={s3_endpoint}, region={datacenter_lower}")
    logger.info(f"get_s3_client ... Access Key ID: {s3_access_key_runpod}")
    logger.info(f"get_s3_client ... Secret Key prefix: {s3_secret_key[:10]}...")

    # Use RunPod's documented pattern - simple boto3 client with no extra config
    return boto3.client(
        's3',
        aws_access_key_id=s3_access_key_runpod,
        aws_secret_access_key=s3_secret_key,
        region_name=datacenter_lower,
        endpoint_url=s3_endpoint
    )


def cleanup_all_s3_models():
    """
    Aggressively delete ALL files from S3 models/ prefix.
    Called at the start of each trial to prevent orphaned files.

    This ensures:
    - No orphaned files from failed uploads
    - No orphaned files from interrupted downloads
    - No orphaned files from previous errors
    - Clean slate for each trial
    """
    try:
        s3_client = get_s3_client()
        bucket = os.getenv('S3_VOLUME_ID_RUNPOD')

        if not bucket:
            logger.warning("cleanup_all_s3_models ... S3_VOLUME_ID_RUNPOD not set, skipping S3 cleanup")
            return

        logger.info("cleanup_all_s3_models ... Starting aggressive S3 cleanup (deleting ALL models/)")

        # List all objects under models/ prefix
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix='models/'
            )

            if 'Contents' not in response or len(response['Contents']) == 0:
                logger.info("cleanup_all_s3_models ... No files found in S3 models/, nothing to clean")
                return

            objects_to_delete = response['Contents']
            logger.info(f"cleanup_all_s3_models ... Found {len(objects_to_delete)} files to delete")

            # Delete objects one at a time (RunPod S3 doesn't support batch delete well)
            deleted_count = 0
            failed_count = 0

            for obj in objects_to_delete:
                try:
                    s3_client.delete_object(
                        Bucket=bucket,
                        Key=obj['Key']
                    )
                    deleted_count += 1
                    logger.debug(f"cleanup_all_s3_models ... Deleted: {obj['Key']}")
                except Exception as delete_error:
                    failed_count += 1
                    logger.warning(f"cleanup_all_s3_models ... Failed to delete {obj['Key']}: {delete_error}")

            if failed_count > 0:
                logger.warning(f"cleanup_all_s3_models ... Deleted {deleted_count}/{len(objects_to_delete)} files ({failed_count} failures)")
            else:
                logger.info(f"cleanup_all_s3_models ... ✅ Successfully deleted {deleted_count} files from S3")

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                logger.warning(f"cleanup_all_s3_models ... S3 bucket '{bucket}' does not exist")
            else:
                raise

    except Exception as e:
        # Log but don't fail - cleanup failure shouldn't block training
        logger.warning(f"cleanup_all_s3_models ... S3 cleanup failed (non-fatal): {e}")
        logger.debug(f"cleanup_all_s3_models ... Traceback: {traceback.format_exc()}")


def upload_model_to_s3(zip_file_path: Path, run_name: str, trial_number: int = 0) -> Optional[Dict[str, Any]]:
    """
    Upload model ZIP file to RunPod S3 and return presigned URL.

    Args:
        zip_file_path: Path to the compressed models ZIP file
        run_name: Run name for organizing S3 storage
        trial_number: Trial number for unique S3 paths (prevents concurrent trial conflicts)

    Returns:
        Dictionary with S3 metadata, or None on failure
    """
    try:
        s3_client = get_s3_client()
        bucket = os.getenv('S3_VOLUME_ID_RUNPOD')

        if not bucket:
            logger.error("upload_model_to_s3 ... S3_VOLUME_ID_RUNPOD not set")
            return None

        # S3 object key - include trial_number to prevent concurrent trials from overwriting files
        s3_key = f"models/{run_name}/trial_{trial_number}/models.zip"
        file_size = zip_file_path.stat().st_size

        logger.info(f"upload_model_to_s3 ... Uploading {file_size} bytes to s3://{bucket}/{s3_key}")
        logger.info(f"upload_model_to_s3 ... Bucket: {bucket}, Key: {s3_key}")
        logger.info(f"upload_model_to_s3 ... Endpoint URL: {s3_client.meta.endpoint_url}")

        # Upload file
        s3_client.upload_file(
            str(zip_file_path),
            bucket,
            s3_key,
            ExtraArgs={
                'ContentType': 'application/zip'
            }
        )

        logger.info("upload_model_to_s3 ... ✅ Successfully uploaded models ZIP to S3")
        logger.debug("running upload_model_to_s3 ... skipping presign (RunPod S3 does not support presigned URLs)")

        # Return S3 metadata with HTTP URL for download
        s3_path = f"s3://{bucket}/{s3_key}"
        endpoint_url = getattr(getattr(s3_client, "meta", None), "endpoint_url", "")

        # Construct HTTP URL for direct download: https://endpoint/bucket/key
        s3_url = f"{endpoint_url}/{bucket}/{s3_key}"

        return {
            "filename": "models.zip",
            "s3_path": s3_path,               # S3 protocol path
            "s3_url": s3_url,                 # HTTP URL for download
            "s3_key": s3_key,
            "s3_bucket": bucket,
            "endpoint_url": endpoint_url,
            "size": file_size,
            "storage_type": "runpod-s3"
        }


    except ClientError as e:
        logger.error(f"upload_model_to_s3 ... S3 upload failed: {e}")
        logger.error(f"upload_model_to_s3 ... Error code: {e.response['Error']['Code']}")
        logger.error(f"upload_model_to_s3 ... Error message: {e.response['Error']['Message']}")
        return None
    except Exception as e:
        logger.error(f"upload_model_to_s3 ... Unexpected error during S3 upload: {e}")
        logger.error(f"upload_model_to_s3 ... Traceback: {traceback.format_exc()}")
        return None


def generate_plots(
    model_builder,
    dataset_name: str,
    trial_id: str,
    trial_number: int = 0,
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
    logger.info(f"generate_plots ... DEBUG: Starting plot generation for trial_id={trial_id}, dataset={dataset_name}")
    logger.info(f"generate_plots ... DEBUG: model_builder provided: {model_builder is not None}")
    logger.info(f"generate_plots ... DEBUG: test_data provided: {test_data is not None}")
    logger.info(f"generate_plots ... DEBUG: optimization_config provided: {optimization_config is not None}")
    # Check plot generation setting from optimization_config
    if optimization_config and getattr(optimization_config, 'plot_generation', 'all') == 'none':
        logger.debug(f"generate_plots ... Skipping plot generation: plot_generation='none'")
        return None

    try:
        logger.debug(f"generate_plots ... Generating plots for trial {trial_id}")

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

        # Create compressed zip files: plots inline (base64), models on S3
        plots_zip_data = None
        models_s3_data = None  # S3 reference instead of inline ZIP
        total_size = 0
        plot_count = 0
        model_count = 0
        zip_size = 0
        s3_data = None

        if plot_dir.exists():
            # Collect all files (plots + models) for single ZIP
            all_files = []
            for file_path in plot_dir.rglob('*'):
                if file_path.is_file():
                    file_extension = file_path.suffix.lower()
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    all_files.append((file_path, file_size))

                    if file_extension in ['.keras', '.h5', '.pkl']:
                        model_count += 1
                        logger.info(f"generate_plots ... Found model: {file_path.name} ({file_size} bytes)")
                    else:
                        plot_count += 1
                        logger.info(f"generate_plots ... Found plot: {file_path.name} ({file_size} bytes)")

            # Create single ZIP with both plots and models
            if all_files:
                with tempfile.NamedTemporaryFile(suffix='_complete.zip', delete=False) as tmp_zip_file:
                    temp_zip_path = tmp_zip_file.name

                try:
                    # Create compressed ZIP with all files
                    with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                        for file_path, file_size in all_files:
                            relative_path = file_path.relative_to(plot_dir)
                            zipf.write(file_path, relative_path)
                            logger.info(f"generate_plots ... Added to ZIP: {relative_path}")

                    zip_size = Path(temp_zip_path).stat().st_size
                    compression_ratio = zip_size / total_size if total_size > 0 else 1.0

                    logger.info(f"generate_plots ... Created complete ZIP: {len(all_files)} files ({plot_count} plots, {model_count} models), {zip_size} bytes ({compression_ratio:.2f} ratio)")

                    # Upload single ZIP to S3
                    logger.info(f"generate_plots ... Uploading complete ZIP to S3")
                    s3_data = upload_model_to_s3(Path(temp_zip_path), run_name, trial_number)

                    if s3_data:
                        # Add metadata
                        s3_data["uncompressed_size"] = total_size
                        s3_data["compression_ratio"] = compression_ratio
                        s3_data["file_count"] = len(all_files)
                        s3_data["plot_count"] = plot_count
                        s3_data["model_count"] = model_count
                        logger.info(f"generate_plots ... ✅ Complete ZIP uploaded to S3: {s3_data['s3_key']}")
                    else:
                        logger.error(f"generate_plots ... ❌ Failed to upload ZIP to S3")
                        return {
                            "success": False,
                            "error": "Failed to upload ZIP to S3",
                            "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown_worker'),
                            "available_files": [],
                            "generated_plots": []
                        }

                finally:
                    # Clean up temporary zip file
                    try:
                        os.unlink(temp_zip_path)
                    except Exception as cleanup_e:
                        logger.warning(f"generate_plots ... Failed to cleanup temp zip: {cleanup_e}")

        logger.info(f"generate_plots ... ✅ Training complete: {plot_count} plots + {model_count} models uploaded to S3")

        return {
            "success": True,
            "plot_dir": str(plot_dir),
            "run_name": run_name,
            "generated_plots": generated_plots,
            "available_files": available_files,
            "s3_zip": s3_data,  # Single S3 ZIP with everything
            "total_files": plot_count + model_count,
            "plot_count": plot_count,
            "model_count": model_count,
            "total_size": total_size,
            "zip_size": zip_size if s3_data else 0,
            "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown_worker')
        }

    except Exception as e:
        logger.error(f"generate_plots ... Plot generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown_worker'),
            "available_files": [],
            "generated_plots": []
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
    valid_datasets = ['mnist', 'cifar10', 'cifar100', 'fashion_mnist', 'imdb', 'reuters', 'gtsrb']  # Add your supported datasets
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

async def start_training(job: Dict[str, Any]):
    """
    Main handler function for RunPod serverless training requests.
    Uses existing optimizer.py orchestration for consistency.
    With S3 storage, responses are always small (just S3 URLs).

    Args:
        job: RunPod job dictionary containing input request

    Returns:
        Response dictionary with S3 URLs for plots and models
    """
    logger.debug("running start_training ... starting serverless training request with COMPLETE config")

    # FIRST THING: Aggressively clean ALL S3 storage before starting trial
    # This prevents orphaned files from failed uploads, interrupted downloads, or previous errors
    logger.info("running start_training ... 🧹 Cleaning S3 storage before starting trial")
    cleanup_all_s3_models()

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

        # Extract trial number for unique S3 paths
        trial_number = request.get('trial_number', 0)
        logger.debug(f"running start_training ... Trial number: {trial_number}")

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
                if os.getenv('ENDPOINT_ID_RUNPOD'):
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

        # Save trained trial model to plots directory BEFORE plot generation (so it gets included in response)
        logger.info(f"running start_training ... DEBUG: Checking model saving conditions for {trial_id}")
        logger.info(f"running start_training ... DEBUG: model_builder exists: {model_builder is not None}")
        if model_builder:
            logger.info(f"running start_training ... DEBUG: model_builder has 'model' attribute: {hasattr(model_builder, 'model')}")
            if hasattr(model_builder, 'model'):
                logger.info(f"running start_training ... DEBUG: model_builder.model is not None: {model_builder.model is not None}")

        if model_builder and hasattr(model_builder, 'model') and model_builder.model:
            try:
                logger.info(f"running start_training ... Attempting to save trial model for {trial_id}")
                # Create model filename similar to final model training approach
                dataset_name = request['dataset_name']
                model_filename = f"final_model_{dataset_name}_trial_{trial_id}.keras"

                # Get plots directory path (same place where plots are saved)
                plots_dir = Path("/tmp/plots") / trial_id
                plots_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                model_path_in_plots = plots_dir / model_filename

                # Save model to plots directory so it gets included in training response
                model_builder.model.save(model_path_in_plots)

                logger.info(f"running start_training ... Trial model saved: {model_path_in_plots}")
                logger.info(f"running start_training ... Trial model will be included in training response (eliminates worker isolation)")

                # Verify model file was actually saved
                if model_path_in_plots.exists():
                    model_size = model_path_in_plots.stat().st_size
                    logger.info(f"running start_training ... ✅ Model file verified: {model_size} bytes")
                else:
                    logger.error(f"running start_training ... ❌ Model file NOT found after save: {model_path_in_plots}")

            except Exception as e:
                logger.error(f"running start_training ... Failed to save trial model for {trial_id}: {e}")
                logger.error(f"running start_training ... Model save error traceback: {traceback.format_exc()}")
                # Don't fail the entire request if model saving fails
        else:
            logger.warning(f"running start_training ... No model found to save for trial {trial_id}")
            if model_builder:
                logger.warning(f"running start_training ... model_builder exists but no model: hasattr(model)={hasattr(model_builder, 'model')}")
                if hasattr(model_builder, 'model'):
                    logger.warning(f"running start_training ... model_builder.model is None: {model_builder.model is None}")

        # Create OptimizationConfig object from config_data for plot generation
        optimization_config = None
        if config_data:
            optimization_config = OptimizationConfig(**config_data)

        logger.info(f"running start_training ... DEBUG: Now generating plots (model should already be saved)")

        plots_direct_info = generate_plots(
            model_builder=model_builder,
            dataset_name=request['dataset_name'],
            trial_id=trial_id,
            trial_number=trial_number,
            test_data=training_result.get('test_data'),
            optimization_config=optimization_config
        )

        logger.info(f"running start_training ... DEBUG: Finished plot generation")

        # CRITICAL FIX: Ensure plots_direct_info is never None to prevent response failures
        if plots_direct_info is None:
            logger.warning(f"running start_training ... plots_direct_info is None, creating fallback")
            plots_direct_info = {
                "success": False,
                "error": "Plot generation returned None",
                "plot_files_zip": None,
                "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown_worker')
            }

        # Initialize worker persistence flag
        worker_persistence_enabled = locals().get('worker_persistence_enabled', False)

        # Skip model attributes extraction - RunPod now returns only metrics
        # This eliminates the massive model weights/gradients/activations data that was causing 3.3MB+ responses
        logger.info(f"running start_training ... Skipping model attributes extraction to prevent response size issues")
        logger.info(f"running start_training ... Model attributes would include weights, gradients, activations - not needed when plots are generated")
        model_attributes = None

        # Verify model_attributes is truly None for debugging
        if model_attributes is not None:
            logger.error(f"❌ CRITICAL: model_attributes should be None but is: {type(model_attributes)}")
        else:
            logger.info(f"✅ Confirmed: model_attributes = None (will not contribute to response size)")

        # Log worker persistence status
        logger.info(f"running start_training ... Worker persistence enabled: {worker_persistence_enabled}")

        # Log worker tracking info for isolation analysis
        worker_id = os.environ.get('RUNPOD_POD_ID', 'unknown_worker')
        logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: Training completed on worker_id={worker_id}, trial_id={trial_id}")

        # Build simplified response structure with worker persistence info
        response = {
            "trial_id": trial_id,
            "status": "completed",
            "success": True,
            "worker_persistence": {
                "enabled": worker_persistence_enabled,
                "reason": "Keeping worker alive for download coordination",
                "downloads_pending": True,
                "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown')
            },
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

        # SIZE DEBUGGING: Log what's actually in the response before serialization
        logger.info(f"🔍 RESPONSE SIZE DEBUG - Response keys: {list(response.keys())}")
        logger.info(f"🔍 RESPONSE SIZE DEBUG - model_attributes type: {type(response.get('model_attributes'))}")
        logger.info(f"🔍 RESPONSE SIZE DEBUG - plots_direct type: {type(response.get('plots_direct'))}")
        if response.get('plots_direct') and isinstance(response.get('plots_direct'), dict):
            plots_info = response['plots_direct']
            logger.info(f"🔍 RESPONSE SIZE DEBUG - plots_direct keys: {list(plots_info.keys())}")
            if 'plot_files_zip' in plots_info:
                zip_info = plots_info['plot_files_zip']
                if zip_info:
                    logger.info(f"🔍 RESPONSE SIZE DEBUG - zip size: {zip_info.get('size', 'unknown')} bytes")
                    logger.info(f"🔍 RESPONSE SIZE DEBUG - zip compression: {zip_info.get('compression_ratio', 'unknown')}")

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
            json_size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
            logger.debug(f"✅ JSON serialization: SUCCESS")
            logger.debug(f"✅ JSON size: {len(json_str.encode('utf-8'))} bytes ({json_size_mb:.2f} MB)")
            logger.info(f"✅ S3-based response is small ({json_size_mb:.2f} MB) - no streaming needed")

        except Exception as e:
            logger.error(f"❌ JSON serialization: FAILED - {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")

            # Try to identify which field is problematic
            logger.debug("TESTING INDIVIDUAL FIELDS:")
            problematic_fields = []
            for key, value in response.items():
                try:
                    json.dumps({key: value})
                    logger.debug(f"  ✅ {key}: serializable")
                except Exception as field_error:
                    logger.error(f"  ❌ {key}: NOT serializable - {field_error}")
                    problematic_fields.append(key)

            # CRITICAL FIX: Create minimal fallback response if serialization fails
            if problematic_fields:
                logger.error(f"🚨 CREATING FALLBACK RESPONSE - Removing problematic fields: {problematic_fields}")

                # Create a minimal guaranteed-serializable response
                fallback_response = {
                    "trial_id": trial_id,
                    "status": "completed",
                    "success": True,
                    "metrics": {
                        "test_accuracy": float(test_accuracy) if test_accuracy is not None else 0.0,
                        "test_loss": float(test_loss) if test_loss is not None else 0.0,
                        "val_accuracy": float(test_accuracy) if test_accuracy is not None else 0.0,
                        "val_loss": float(test_loss) if test_loss is not None else 0.0,
                        "best_value": float(test_accuracy) if test_accuracy is not None else 0.0,
                        "optimize_for": str(all_params.get('optimize_for', 'val_accuracy')),
                        "training_time_seconds": 0.0
                    },
                    "plots_direct": plots_direct_info if plots_direct_info and 'success' in plots_direct_info else {
                        "success": False,
                        "error": "Plot data not serializable"
                    },
                    "serialization_fallback": True,
                    "removed_fields": problematic_fields
                }

                # Test fallback serialization
                try:
                    json.dumps(fallback_response)
                    logger.info("✅ FALLBACK response serialization: SUCCESS")
                    response = fallback_response
                except Exception as fallback_error:
                    logger.error(f"❌ FALLBACK response serialization: FAILED - {fallback_error}")
                    # Last resort: absolute minimal response
                    response = {
                        "trial_id": trial_id,
                        "status": "completed",
                        "success": True,
                        "error": "Response serialization failed",
                        "test_accuracy": float(test_accuracy) if test_accuracy is not None else 0.0
                    }

        logger.debug("=" * 60)

        # With S3 storage, response is always small - just return it
        logger.info(f"running start_training ... Returning S3-based response for trial {trial_id}")
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
            download_type = event.get('input', {}).get('download_type', 'plots')  # 'plots' or 'models'
            trial_id = event.get('input', {}).get('trial_id')
            trial_number = event.get('input', {}).get('trial_number')

            # Format trial information for logs
            trial_info = ""
            if trial_number is not None:
                trial_info = f" (trial_{trial_number}"
                if trial_id:
                    trial_info += f", {trial_id}"
                trial_info += ")"

            if not run_name:
                return {"error": "run_name parameter required", "status_code": 400}

            plots_dir = Path("/tmp/plots") / run_name
            logger.info(f"download_directory{trial_info} ... Looking for directory: {plots_dir}")

            # Debug: List what's available in /tmp/plots to help troubleshoot
            try:
                base_plots_dir = Path("/tmp/plots")
                if base_plots_dir.exists():
                    available_dirs = [d.name for d in base_plots_dir.iterdir() if d.is_dir()]
                    logger.info(f"download_directory{trial_info} ... Available directories in /tmp/plots: {available_dirs}")
                else:
                    logger.warning(f"download_directory{trial_info} ... /tmp/plots directory doesn't exist!")
            except Exception as e:
                logger.warning(f"download_directory{trial_info} ... Failed to list /tmp/plots contents: {e}")

            if not plots_dir.exists() or not plots_dir.is_dir():
                logger.error(f"running download_directory{trial_info} ... Directory {plots_dir} not found or not a directory")
                return {"error": f"Directory for run {run_name} not found", "status_code": 404}

            logger.info(f"running download_directory{trial_info} ... Processing {download_type} download request")

            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                zip_path = tmp_zip.name

                # Create zip archive based on download_type
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in plots_dir.rglob('*'):
                        if file_path.is_file():
                            is_model_file = file_path.suffix in ['.keras', '.h5', '.pb']

                            # Include file based on download_type
                            should_include = False
                            if download_type == 'plots' and not is_model_file:
                                should_include = True
                            elif download_type == 'models' and is_model_file:
                                should_include = True
                            elif download_type == 'all':  # Fallback for compatibility
                                should_include = True

                            if should_include:
                                # Add file to zip with relative path
                                arcname = file_path.relative_to(plots_dir)
                                zipf.write(file_path, arcname)
                                logger.debug(f"Added to {download_type} zip: {arcname}")

                # Read zip file and encode as base64
                with open(zip_path, 'rb') as f:
                    zip_content = base64.b64encode(f.read()).decode('utf-8')

                # Get zip file size and file count (only count files, not directories)
                zip_size = Path(zip_path).stat().st_size
                file_count = 0
                for file_path in plots_dir.rglob('*'):
                    if file_path.is_file():
                        is_model_file = file_path.suffix in ['.keras', '.h5', '.pb']
                        if download_type == 'plots' and not is_model_file:
                            file_count += 1
                        elif download_type == 'models' and is_model_file:
                            file_count += 1
                        elif download_type == 'all':
                            file_count += 1

                # Clean up temporary file
                os.unlink(zip_path)

                logger.info(f"running download_directory{trial_info} ... Created {download_type} zip archive for {run_name}: {file_count} files, {zip_size} bytes")


                logger.info(f"running download_directory{trial_info} ... {download_type} download completed successfully for {run_name}")

                return {
                    "run_name": run_name,
                    "filename": f"{run_name}_{download_type}.zip",
                    "content": zip_content,
                    "encoding": "base64",
                    "size": zip_size,
                    "file_count": file_count,
                    "compression": "zip",
                    "download_type": download_type
                }

        elif command == 'download_directory_multipart':
            """
            Multi-part single response download to solve worker isolation issues.
            Returns all file types in separate parts within a single response.
            """
            run_name = event.get('input', {}).get('run_name')
            trial_id = event.get('input', {}).get('trial_id')
            trial_number = event.get('input', {}).get('trial_number')
            max_part_size_mb = event.get('input', {}).get('max_part_size_mb', 8)  # Default 8MB per part

            # Format trial information for logs
            trial_info = ""
            if trial_number is not None:
                trial_info = f" (trial_{trial_number}"
                if trial_id:
                    trial_info += f", {trial_id}"
                trial_info += ")"

            if not run_name:
                return {"error": "run_name parameter required", "status_code": 400}

            # Log worker tracking info for isolation analysis
            worker_id = os.environ.get('RUNPOD_POD_ID', 'unknown_worker')
            logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: Download request received on worker_id={worker_id}, run_name={run_name}{trial_info}")

            plots_dir = Path("/tmp/plots") / run_name
            logger.info(f"download_directory_multipart{trial_info} ... Looking for directory: {plots_dir}")

            # Enhanced directory existence verification for isolation analysis
            try:
                base_plots_dir = Path("/tmp/plots")
                if base_plots_dir.exists():
                    available_dirs = [d.name for d in base_plots_dir.iterdir() if d.is_dir()]
                    logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: worker_id={worker_id} has directories: {available_dirs}")
                    logger.info(f"download_directory_multipart{trial_info} ... Available directories in /tmp/plots: {available_dirs}")

                    # Check if the requested directory exists
                    if run_name in available_dirs:
                        logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: ✅ worker_id={worker_id} HAS directory {run_name}")
                    else:
                        logger.error(f"🏗️ WORKER_ISOLATION_TRACKING: ❌ worker_id={worker_id} MISSING directory {run_name}")
                        logger.error(f"🏗️ WORKER_ISOLATION_TRACKING: This confirms worker isolation - directory was created on different worker")
                else:
                    logger.warning(f"🏗️ WORKER_ISOLATION_TRACKING: worker_id={worker_id} has no /tmp/plots directory at all")
                    logger.warning(f"download_directory_multipart{trial_info} ... /tmp/plots directory doesn't exist!")
            except Exception as e:
                logger.warning(f"download_directory_multipart{trial_info} ... Failed to list /tmp/plots contents: {e}")

            if not plots_dir.exists() or not plots_dir.is_dir():
                logger.error(f"🏗️ WORKER_ISOLATION_TRACKING: CONFIRMED - worker_id={worker_id} cannot access directory {run_name}")
                logger.error(f"running download_directory_multipart{trial_info} ... Directory {plots_dir} not found or not a directory")
                return {
                    "error": f"Directory for run {run_name} not found",
                    "status_code": 404,
                    "download_worker_id": worker_id
                }

            logger.info(f"running download_directory_multipart{trial_info} ... Processing multi-part download request")

            # Create temporary zip files for each part

            # Define file type categories
            file_types = [
                {
                    "name": "plots",
                    "filter": lambda path: path.suffix.lower() not in ['.keras', '.h5', '.pb', '.pkl', '.pickle'],
                    "priority": 1
                },
                {
                    "name": "models",
                    "filter": lambda path: path.suffix.lower() in ['.keras', '.h5', '.pb'],
                    "priority": 2
                },
                {
                    "name": "other",
                    "filter": lambda path: path.suffix.lower() in ['.pkl', '.pickle'] or path.name.endswith('.json'),
                    "priority": 3
                }
            ]

            parts = []
            max_part_size = max_part_size_mb * 1024 * 1024  # Convert MB to bytes
            total_file_count = 0
            total_size = 0

            for file_type in file_types:
                # Collect files for this type
                type_files = []
                for file_path in plots_dir.rglob('*'):
                    if file_path.is_file() and file_type["filter"](file_path):
                        type_files.append(file_path)

                if not type_files:
                    logger.info(f"download_directory_multipart{trial_info} ... No {file_type['name']} files found")
                    continue

                # Create zip for this file type
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                    zip_path = tmp_zip.name

                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for file_path in type_files:
                            # Add file to zip with relative path
                            arcname = file_path.relative_to(plots_dir)
                            zipf.write(file_path, arcname)
                            logger.debug(f"Added to {file_type['name']} zip: {arcname}")

                    # Check zip size
                    zip_size = Path(zip_path).stat().st_size

                    if zip_size > max_part_size:
                        logger.warning(f"download_directory_multipart{trial_info} ... {file_type['name']} zip size ({zip_size} bytes) exceeds max part size ({max_part_size} bytes)")
                        # For now, include it anyway but log the warning
                        # Future enhancement: could split large file types into sub-parts

                    # Read zip file and encode as base64
                    with open(zip_path, 'rb') as f:
                        zip_content = base64.b64encode(f.read()).decode('utf-8')

                    # Clean up temporary file
                    os.unlink(zip_path)

                    # Add part to response
                    part = {
                        "type": file_type["name"],
                        "filename": f"{run_name}_{file_type['name']}.zip",
                        "content": zip_content,
                        "encoding": "base64",
                        "size": zip_size,
                        "file_count": len(type_files),
                        "compression": "zip",
                        "priority": file_type["priority"]
                    }
                    parts.append(part)
                    total_file_count += len(type_files)
                    total_size += zip_size

                    logger.info(f"download_directory_multipart{trial_info} ... Created {file_type['name']} part: {len(type_files)} files, {zip_size} bytes")

            if not parts:
                logger.warning(f"download_directory_multipart{trial_info} ... No files found for download")
                return {"error": f"No files found for run {run_name}", "status_code": 404}

            logger.info(f"download_directory_multipart{trial_info} ... Multi-part download completed: {len(parts)} parts, {total_file_count} total files, {total_size} total bytes")

            return {
                "run_name": run_name,
                "download_method": "multipart",
                "parts": parts,
                "total_parts": len(parts),
                "total_file_count": total_file_count,
                "total_size": total_size,
                "trial_id": trial_id,
                "trial_number": trial_number,
                "download_worker_id": worker_id
            }

        else:
            return {"error": f"Unknown command: {command}", "status_code": 400}

    except Exception as e:
        logger.error(f"Error handling simple HTTP endpoint: {e}")
        return {"error": str(e), "status_code": 500}


async def handler(event: Dict[str, Any]):
    """
    Main RunPod handler function.
    Processes incoming serverless requests and routes to appropriate handlers.
    Also handles simple HTTP endpoints for file downloads.
    With S3 storage, responses are always small and returned directly.

    Args:
        event: RunPod event dictionary containing job information

    Returns:
        Response dictionary with results or S3 URLs
    """
    # Get worker identification for isolation tracking
    worker_id = os.environ.get('RUNPOD_POD_ID', 'unknown_worker')

    logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: Handler called on worker_id={worker_id}")
    logger.debug("running handler ... processing RunPod serverless request")

    # Handle simple HTTP endpoints for file downloads
    if isinstance(event, dict) and event.get('input', {}).get('command') in ['health', 'list_files', 'download_file', 'download_directory', 'download_directory_multipart']:
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
            # start_training now returns a dict directly (no streaming)
            result = await start_training(job)
            return result
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
    Returns dict responses directly (no streaming with S3).
    """
    # handler now returns a dict directly
    return await handler(event)


if __name__ == "__main__":
    print("=== HANDLER.PY STARTUP ===")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"ENDPOINT_ID_RUNPOD: {os.getenv('ENDPOINT_ID_RUNPOD')}")
    print("=========================")

    try:
        logger.info("=== HANDLER.PY STARTUP ===")
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"ENDPOINT_ID_RUNPOD: {os.getenv('ENDPOINT_ID_RUNPOD')}")
        logger.info("=========================")

        # Start RunPod serverless handler (no streaming needed with S3)
        logger.info("Starting RunPod serverless handler...")
        print("Starting RunPod serverless handler...")
        runpod.serverless.start({
            "handler": handler,
            "concurrency_modifier": adjust_concurrency
        })

    except Exception as e:
        print(f"ERROR in handler startup: {e}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        raise