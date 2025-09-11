"""
RunPod Service Handler - Unified Approach

Simplified handler that uses the existing optimize_model() function,
ensuring identical behavior between local and RunPod execution.
"""

import json
import os
from pathlib import Path
import runpod
import shutil
import sys
import tempfile       
import traceback
from typing import Dict, Any
from datetime import datetime

from src.optimizer import optimize_model
from src.utils.logger import logger
from src.utils.s3_transfer import upload_to_runpod_s3
                    


# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



def validate_request(request: Dict[str, Any]) -> tuple[bool, str | None]:
    """Validate incoming request structure"""
    logger.debug("Validating incoming request structure")
    
    # Required fields
    required_fields = ['command', 'dataset_name']
    
    for field in required_fields:
        if field not in request:
            error_msg = f"Missing required field: {field}"
            logger.error(f"Request validation failed: {error_msg}")
            return False, error_msg
    
    # Validate command
    if request['command'] not in ['start_training', 'start_final_model_training']:
        error_msg = f"Invalid command: {request['command']}"
        logger.error(f"Request validation failed: {error_msg}")
        return False, error_msg
    
    # Validate dataset_name (basic check)
    valid_datasets = ['mnist', 'cifar10', 'fashion_mnist']
    if request['dataset_name'] not in valid_datasets:
        error_msg = f"Unsupported dataset_name: {request['dataset_name']}. Supported: {valid_datasets}"
        logger.error(f"Request validation failed: {error_msg}")
        return False, error_msg
    
    logger.debug("Request validation successful")
    return True, None

async def start_training(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start training using the unified optimize_model() function
    
    This is a thin wrapper that calls the same optimize_model() function
    used locally, ensuring consistent behavior between local and RunPod execution.
    """
    request = job['input']
    trial_id = request.get('trial_id', 'unknown')
    logger.info(f"🚀 RunPod serverless training starting for trial {trial_id}")
    
    try:
        # Validate request
        is_valid, error_msg = validate_request(request)
        if not is_valid:
            return {"error": error_msg, "success": False}
        
        # Import the unified optimization function
        from src.optimizer import optimize_model
        
        # Extract parameters from request
        config_data = request.get('config', {})
        hyperparameters = request.get('hyperparameters', {})
        
        # ========================================
        # DEBUG: LOG RECEIVED CONFIG DATA
        # ========================================
        logger.info(f"🔍 DEBUG: Received config_data keys: {list(config_data.keys())}")
        logger.info(f"🎨 DEBUG: plot_generation from config_data: '{config_data.get('plot_generation', 'NOT_FOUND')}'")
        logger.info(f"📊 DEBUG: create_optuna_model_plots from config_data: {config_data.get('create_optuna_model_plots', 'NOT_FOUND')}")
        logger.info(f"🔍 DEBUG: Full config_data: {config_data}")
        
        # Create progress callback to send updates to RunPod (only in serverless environment)
        def progress_callback(progress_update):
            """Send progress updates to RunPod during optimization"""
            try:
                # Only send updates if we're in actual RunPod serverless environment
                if os.getenv('RUNPOD_ENDPOINT_ID') and hasattr(progress_update, '__dict__'):
                    # Send structured progress update to RunPod
                    progress_data = {
                        'trial_id': getattr(progress_update, 'trial_id', trial_id),
                        'trial_number': getattr(progress_update, 'trial_number', 1),
                        'status': getattr(progress_update, 'status', 'running'),
                        'message': f"Trial {trial_id} - {getattr(progress_update, 'status', 'running')}"
                    }
                    
                    runpod.serverless.progress_update(job, progress_data)
                    logger.debug(f"Sent progress update to RunPod: {progress_data}")
                    
            except Exception as e:
                logger.debug(f"Progress update skipped (test environment): {e}")
        
        logger.info(f"Calling unified optimize_model for trial {trial_id}")
        
        # Extract required parameters without defaults
        if 'mode' not in config_data:
            raise ValueError("Required parameter 'mode' not found in config")
        if 'objective' not in config_data:
            raise ValueError("Required parameter 'objective' not found in config")
        
        # Generate unified run_name using the same function as local optimizer
        # This ensures identical directory structure between local and RunPod execution
        from src.data_classes.configs import generate_unified_run_name
        unified_run_name = generate_unified_run_name(
            dataset_name=request['dataset_name'],
            mode=config_data['mode'],
            optimize_for=config_data['objective']
        )
        
        logger.info(f"🏷️ Using unified run_name: {unified_run_name}")
        logger.info(f"🔍 This replaces the previous runpod_trial_{trial_id} logic to ensure consistent paths")
        
        # Call the unified optimize_model function
        result = optimize_model(
            dataset_name=request['dataset_name'],
            mode=config_data['mode'],
            optimize_for=config_data['objective'],
            trials=1,  # Single trial per RunPod invocation
            run_name=unified_run_name,
            progress_callback=progress_callback,
            use_runpod_service=False,  # We ARE the RunPod service
            **{k: v for k, v in config_data.items() if k not in ['mode', 'objective']},  # Pass remaining config
            **hyperparameters  # Pass all hyperparameters
        )
        
        logger.info(f"✅ optimize_model completed successfully for trial {trial_id}")
        
        # ========================================
        # PLOTS_S3_INFO CREATION ANALYSIS
        # ========================================
        logger.info(f"🔍 ===== RUNPOD HANDLER PLOTS_S3 ANALYSIS =====")
        logger.info(f"🏃 Trial ID: {trial_id}")
        logger.info(f"🌐 RunPod Environment: {bool(os.getenv('RUNPOD_ENDPOINT_ID'))}")
        logger.info(f"✅ Successful Trials: {result.successful_trials}")
        
        plots_s3_info = None
        final_model_s3_info = None
        
        # Create plots_s3_info when plots are generated successfully
        # Since handler.py only runs in RunPod contexts, always create S3 info when appropriate
        if result.successful_trials > 0:
            logger.info(f"✅ Trials completed successfully")
            
            # Check if plot generation was enabled
            plot_generation = config_data.get('plot_generation', 'all')  # Changed default from 'none' to 'all'
            logger.info(f"🎨 Plot generation setting: '{plot_generation}'")
            
            if plot_generation and plot_generation.lower() != 'none':
                logger.info(f"✅ Plot generation is ENABLED")
                
                # Use the actual results_dir path from the optimization result
                # This ensures S3 path matches exactly what was created on the RunPod worker
                if result.results_dir:
                    # Convert Path to relative path from /app root
                    actual_results_dir = str(result.results_dir)
                    logger.info(f"🏷️ Actual results directory created: {actual_results_dir}")
                    
                    # Extract the relative path (remove /app/ prefix if present)
                    if actual_results_dir.startswith('/app/'):
                        relative_path = actual_results_dir[5:]  # Remove '/app/'
                    else:
                        relative_path = actual_results_dir
                    
                    s3_prefix = f"{relative_path}/plots/trial_0"
                    logger.info(f"🏷️ S3 prefix derived from actual path: {s3_prefix}")
                    
                    # Create plots_s3_info using actual directory structure
                    plots_s3_info = {
                        'success': True,
                        's3_prefix': s3_prefix,
                        'bucket': '40ub9vhaa7',
                        'plot_generation_mode': plot_generation
                    }
                else:
                    logger.warning(f"❌ No results_dir in optimization result, cannot create plots_s3_info")
                    plots_s3_info = None
                
                if plots_s3_info:
                    logger.info(f"🎉 PLOTS_S3_INFO CREATED SUCCESSFULLY")
                    logger.info(f"📊 S3 Prefix: {plots_s3_info['s3_prefix']}")
                    logger.info(f"🪣 S3 Bucket: {plots_s3_info['bucket']}")
                    logger.info(f"🎨 Generation Mode: {plots_s3_info['plot_generation_mode']}")
                    logger.info(f"📤 This will be included in the response to local optimizer")
            else:
                logger.warning(f"❌ Plot generation is DISABLED ('{plot_generation}')")
                logger.warning(f"📤 No plots_s3_info will be included in response")
        else:
            logger.warning(f"❌ No successful trials ({result.successful_trials})")
            logger.warning(f"📤 No plots_s3_info will be included in response")
        
        logger.info(f"🔍 ===== RUNPOD HANDLER PLOTS_S3 ANALYSIS END =====")
        logger.info(f"📦 Final plots_s3_info: {plots_s3_info}")
        
        # ========================================
        # FINAL MODEL S3 ANALYSIS
        # ========================================
        logger.info(f"🔍 ===== RUNPOD HANDLER FINAL_MODEL_S3 ANALYSIS =====")
        
        if result.successful_trials > 0 and result.best_model_path:
            logger.info(f"🎯 FINAL MODEL: Model path found: {result.best_model_path}")
            
            # Check if final model file exists
            model_path = Path(result.best_model_path)
            if model_path.exists():
                logger.info(f"🎯 FINAL MODEL: ✅ Model file exists at: {model_path}")
                
                # Extract S3 path info similar to plots
                model_dir_str = str(model_path.parent)
                if "optimization_results" in model_dir_str:
                    opt_results_index = model_dir_str.find("optimization_results")
                    relative_part = model_dir_str[opt_results_index + len("optimization_results"):].lstrip("/")
                    s3_prefix = f"optimization_results/{relative_part}" if relative_part else "optimization_results"
                else:
                    s3_prefix = "optimization_results"
                
                model_filename = model_path.name
                s3_key = f"{s3_prefix}/{model_filename}"
                
                logger.info(f"🎯 FINAL MODEL: S3 key will be: {s3_key}")
                
                # Upload to S3
                try:
                    
                    logger.info(f"🎯 FINAL MODEL: Uploading to S3: s3://40ub9vhaa7/{s3_key}")
                    
                    # Create a temporary directory with the model file to upload
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_model_path = Path(temp_dir) / model_filename
                        shutil.copy2(str(model_path), str(temp_model_path))
                        
                        # Upload using the directory-based function
                        upload_result = upload_to_runpod_s3(
                            local_dir=temp_dir,
                            s3_prefix=s3_prefix
                        )
                        
                        success = upload_result is not None
                    
                    if success:
                        logger.info(f"🎯 FINAL MODEL: ✅ S3 upload successful!")
                        
                        final_model_s3_info = {
                            'success': True,
                            's3_prefix': s3_prefix,
                            's3_key': s3_key,
                            'bucket': '40ub9vhaa7',
                            'model_filename': model_filename
                        }
                        
                        logger.info(f"🎯 FINAL MODEL: Created final_model_s3_info")
                        logger.info(f"🎯 FINAL MODEL: S3 Prefix: {final_model_s3_info['s3_prefix']}")
                        logger.info(f"🎯 FINAL MODEL: S3 Key: {final_model_s3_info['s3_key']}")
                        logger.info(f"🎯 FINAL MODEL: Bucket: {final_model_s3_info['bucket']}")
                    else:
                        logger.error(f"🎯 FINAL MODEL: ❌ S3 upload failed")
                        final_model_s3_info = None
                        
                except Exception as e:
                    logger.error(f"🎯 FINAL MODEL: S3 upload error: {e}")
                    final_model_s3_info = None
                    
            else:
                logger.warning(f"🎯 FINAL MODEL: ❌ Model file not found at: {model_path}")
                final_model_s3_info = None
                
        else:
            if result.successful_trials == 0:
                logger.warning(f"🎯 FINAL MODEL: No successful trials")
            else:
                logger.warning(f"🎯 FINAL MODEL: No model path in result")
            final_model_s3_info = None
            
        logger.info(f"🔍 ===== RUNPOD HANDLER FINAL_MODEL_S3 ANALYSIS END =====")
        logger.info(f"📦 Final final_model_s3_info: {final_model_s3_info}")
        
        # ========================================
        # RESPONSE CONSTRUCTION
        # ========================================
        logger.info(f"🔍 ===== RESPONSE CONSTRUCTION =====")
        
        if result.successful_trials > 0:
            logger.info(f"✅ Building successful response")
            response = {
                "trial_id": trial_id,
                "status": "completed", 
                "success": True,
                "metrics": {
                    "test_accuracy": result.best_total_score,
                    "test_loss": 0.0,  # Will be populated from health data if available
                    "val_accuracy": result.best_total_score,
                    "val_loss": 0.0,
                    "training_time_seconds": result.optimization_time_hours * 3600
                },
                "health_metrics": result.best_trial_health or {},
                "architecture": {
                    "layers": {},
                    "parameters": 0,
                    "successful_trials": result.successful_trials
                },
                "best_params": result.best_params,
                "multi_gpu_used": config_data.get('use_multi_gpu', False),
                "target_gpus": config_data.get('target_gpus_per_worker', 1),
                "model_attributes": None,
                "plots_s3": plots_s3_info,  # S3 plot upload information
                "final_model_s3": final_model_s3_info  # S3 final model upload information
            }
        else:
            # No successful trials
            response = {
                "trial_id": trial_id,
                "status": "failed",
                "success": False,
                "error": "No successful trials completed",
                "metrics": {},
                "health_metrics": {},
                "architecture": {},
                "best_params": {},
                "multi_gpu_used": False,
                "target_gpus": 1,
                "model_attributes": None,
                "plots_s3": None,
                "final_model_s3": None
            }
        
        # ========================================
        # FINAL RESPONSE LOGGING
        # ======================================== 
        logger.info(f"📦 Final response keys: {list(response.keys())}")
        logger.info(f"📤 Response being sent to local optimizer:")
        logger.info(f"   - trial_id: {response['trial_id']}")
        logger.info(f"   - status: {response['status']}")
        logger.info(f"   - success: {response['success']}")
        logger.info(f"   - plots_s3: {'INCLUDED' if response.get('plots_s3') else 'NULL/NONE'}")
        
        if response.get('plots_s3'):
            logger.info(f"📊 plots_s3 details in response: {response['plots_s3']}")
        else:
            logger.warning(f"❌ plots_s3 is None/missing in final response")
        
        logger.info(f"🎉 Returning response for trial {trial_id}")
        return response
        
    except Exception as e:
        logger.error(f"🚨 RunPod training failed for trial {trial_id}: {e}")
        logger.error(f"🚨 RunPod training error traceback: {traceback.format_exc()}")
        return {
            "trial_id": trial_id, 
            "status": "failed",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

async def start_final_model_training(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start final model training using the unified optimize_model() function
    
    This handles final model training by calling optimize_model with final model settings.
    """
    request = job['input']
    logger.info(f"🚀 RunPod serverless final model training starting")
    
    try:
        # Validate request
        is_valid, error_msg = validate_request(request)
        if not is_valid:
            return {"error": error_msg, "success": False}
        
        
        # Extract parameters from request
        config_data = request.get('config', {})
        hyperparameters = request.get('hyperparameters', {})
        
        # For final model training, we want to use the best hyperparameters
        # and potentially different settings
        logger.info(f"Calling unified optimize_model for final model training")
        
        # Extract required parameters without defaults
        if 'mode' not in config_data:
            raise ValueError("Required parameter 'mode' not found in config")
        if 'objective' not in config_data:
            raise ValueError("Required parameter 'objective' not found in config")
        
        # Generate unified run_name for final model using the same function as local optimizer
        from src.data_classes.configs import generate_unified_run_name
        unified_run_name = generate_unified_run_name(
            dataset_name=request['dataset_name'],
            mode=config_data['mode'],
            optimize_for=config_data['objective']
        )
        
        logger.info(f"🏷️ Using unified run_name for final model: {unified_run_name}")
        
        # Call the unified optimize_model function for final model
        result = optimize_model(
            dataset_name=request['dataset_name'],
            mode=config_data['mode'],
            optimize_for=config_data['objective'],
            trials=1,  # Single trial for final model
            run_name=unified_run_name,
            use_runpod_service=False,  # We ARE the RunPod service
            **{k: v for k, v in config_data.items() if k not in ['mode', 'objective']},  # Pass remaining config
            **hyperparameters  # Pass the best hyperparameters
        )
        
        logger.info(f"✅ Final model training completed successfully")
        
        # Extract results
        if result.successful_trials > 0:
            response = {
                "status": "completed",
                "success": True,
                "final_model_path": result.best_model_path,
                "test_accuracy": result.best_total_score,
                "test_loss": 0.0,  # Will be populated from health data if available
                "training_time_seconds": result.optimization_time_hours * 3600,
                "multi_gpu_used": config_data.get('use_multi_gpu', False),
                "model_attributes": None,
                "s3_upload": None,  # Will be populated by optimizer's S3 upload logic
                "plots_s3": None    # Will be populated by optimizer's S3 upload logic
            }
        else:
            response = {
                "status": "failed",
                "success": False,
                "error": "Final model training failed - no successful trials"
            }
        
        return response
        
    except Exception as e:
        logger.error(f"🚨 RunPod final model training failed: {e}")
        logger.error(f"🚨 RunPod final model training error traceback: {traceback.format_exc()}")
        return {
            "status": "failed",
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# RunPod serverless handler
async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod serverless handler
    
    Routes requests to appropriate functions based on command.
    """
    try:
        request = job.get('input', {})
        command = request.get('command', '')
        
        logger.info(f"🎯 RunPod handler received command: {command}")
        
        if command == 'start_training':
            return await start_training(job)
        elif command == 'start_final_model_training':
            return await start_final_model_training(job)
        else:
            return {
                "success": False,
                "error": f"Unknown command: {command}"
            }
            
    except Exception as e:
        logger.error(f"🚨 RunPod handler error: {e}")
        logger.error(f"🚨 RunPod handler error traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Create alias for backward compatibility
runpod_handler = handler

# Start the serverless worker
if __name__ == "__main__":
    # Check if running in RunPod environment vs local development
    if os.getenv('RUNPOD_ENDPOINT_ID'):
        # Running in RunPod serverless environment
        logger.info("🚀 Starting RunPod serverless worker with unified approach")
        runpod.serverless.start({"handler": handler})
    else:
        # Running locally for development/testing
        logger.info("🚀 Starting FastAPI web server for local development...")
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(title="CV Classification Optimizer", version="2.0.0")
        
        class RunPodEvent(BaseModel):
            input: dict
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "cv-classification-optimizer", "version": "unified"}
        
        @app.post("/")
        async def web_handler(event: RunPodEvent):
            """FastAPI wrapper around unified RunPod handler"""
            try:
                # Convert Pydantic model to dict (already in RunPod event format)
                event_dict = event.model_dump()
                
                # Call the unified handler
                result = await handler(event_dict)
                
                return result
                
            except Exception as e:
                logger.error(f"Web handler error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        logger.info("FastAPI server starting...")
        logger.info("API docs available at http://localhost:8080/docs")
        uvicorn.run(app, host="0.0.0.0", port=8080)