"""
RunPod Service Handler - Thin wrapper around existing optimizer.py orchestration
Implements specialized serverless hyperparameter optimization using existing battle-tested logic.
"""

# Web server wrapper for local testing and container deployment
import asyncio
from fastapi import FastAPI, HTTPException
import json
from pathlib import Path
from pydantic import BaseModel
import runpod
import sys
import traceback
from typing import Dict, Any, Optional
import uvicorn


# Add project root to Python path for imports
current_file = Path(__file__)
project_root = current_file.parent.parent  # Go up 2 levels to project root
sys.path.insert(0, str(project_root / "src"))

# Import existing orchestration layer
from src.optimizer import optimize_model, OptimizationConfig, OptimizationMode, OptimizationObjective, OptimizationResult
from src.utils.logger import logger


def adjust_concurrency(current_concurrency):
    return min(current_concurrency + 1, 6)  # Your max workers


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
        best_value = optimization_result.best_value
        
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
            'objective_name': optimization_result.optimization_config.objective.value if optimization_result.optimization_config else 'unknown',
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
                'objective_name': 'unknown',
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
    required_fields = ['command', 'dataset']
    
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
    
    # Validate dataset (basic check)
    valid_datasets = ['mnist', 'cifar10', 'fashion_mnist']  # Add your supported datasets
    if request['dataset'] not in valid_datasets:
        error_msg = f"Unsupported dataset: {request['dataset']}. Supported: {valid_datasets}"
        logger.error(f"running validate_request ... request validation failed: {error_msg}")
        return False, error_msg
    
    logger.debug("running validate_request ... request validation successful")
    return True, None

def build_optimization_config(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Build optimization config AND hyperparameters from request.
    
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
        'plot_generation': 'none'  # Skip plots for serverless
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
        
        # 🎯 CRITICAL FIX: Build COMPLETE configuration (config + hyperparameters)
        all_params = build_optimization_config(request)
        
        # 🎯 VERIFICATION: Log what we're actually using
        logger.debug(f"running start_training ... VERIFICATION: Complete parameter set:")
        for key, value in all_params.items():
            logger.debug(f"running start_training ... - {key}: {value}")
        
        # 🎯 CRITICAL FIX: Call create_and_train_model directly, not optimize_model
        # optimize_model runs its own optimization study, ignoring our hyperparameters
        logger.debug(f"running start_training ... calling create_and_train_model with trial hyperparameters: {trial_id}")
        
        # Import create_and_train_model for single trial execution
        from src.model_builder import create_and_train_model, ModelConfig
        
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
        
        # Call create_and_train_model with the configured ModelConfig
        training_result = create_and_train_model(
            dataset_name=request['dataset'],
            model_config=model_config,
            test_size=all_params.get('test_size', 0.2)
        )
        
        # Extract metrics from training result
        model_builder = training_result['model_builder']
        test_accuracy = training_result['test_accuracy']
        test_loss = training_result['test_loss']
        
        # Get comprehensive health metrics
        health_metrics = model_builder.get_last_health_analysis()
        
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
                "objective_name": all_params.get('optimize_for', 'val_accuracy'),
                "training_time_seconds": 0.0  # Not available from create_and_train_model
            },
            "health_metrics": health_metrics or {},
            "model_info": {
                "epochs_completed": model_config.epochs,
                "total_trials": 1,
                "successful_trials": 1
            },
            "best_params": hyperparameters
        }
        
        logger.debug(f"running start_training ... trial {trial_id} completed with COMPLETE config")
        logger.debug(f"running start_training ... best value: {test_accuracy:.4f}")
        
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
    import os
    
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
            dataset: str
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
                data = request.dict()
                
                # Wrap in RunPod event format
                event = {"input": data}
                
                # Call your existing RunPod handler - need to await the async function
                import asyncio
                result = await handler(event)
                
                return result
                
            except Exception as e:
                logger.error(f"Web handler error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        logger.info("FastAPI server starting...")
        logger.info("API docs available at http://localhost:8080/docs")
        uvicorn.run(app, host="0.0.0.0", port=8080)