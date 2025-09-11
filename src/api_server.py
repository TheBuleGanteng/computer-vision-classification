"""
FastAPI Server for Hyperparameter Optimization

Provides REST API endpoints for managing hyperparameter optimization jobs.
Integrates with existing optimizer system to provide:
- Asynchronous job management
- Progress monitoring
- Result retrieval
- Model download capabilities

Designed for deployment on RunPod with GPU acceleration and local development.
"""

import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import json
import numpy as np
import os
from pathlib import Path
from pydantic import BaseModel, Field
import shutil
from optimizer import ModelOptimizer
from data_classes.configs import OptimizationConfig, OptimizationRequest, OptimizationMode, OptimizationObjective
import subprocess
import tempfile
import traceback
from typing import Dict, Any, List, Optional, Union
import uuid
import uvicorn
import zipfile

# UPDATED IMPORTS - Fixed compatibility
from optimizer import optimize_model, OptimizationResult
from data_classes.callbacks import UnifiedProgress
from dataset_manager import DatasetManager
from utils.logger import logger, setup_logging


class TensorBoardManager:
    """
    Manages TensorBoard server processes for different jobs
    
    Handles starting, stopping, and tracking TensorBoard servers with proper
    process management and port allocation.
    """
    
    def __init__(self):
        self.running_servers = {}  # job_id -> (process, port)
    
    def get_port_for_job(self, job_id: str) -> int:
        """Generate consistent port for job"""
        return 6006 + hash(job_id) % 1000
    
    def start_server(self, job_id: str, log_dir: Path) -> Dict[str, Any]:
        """
        Start TensorBoard server for a job
        
        Args:
            job_id: Unique job identifier
            log_dir: Path to TensorBoard logs directory
            
        Returns:
            Dictionary with server info
        """
        port = self.get_port_for_job(job_id)
        
        # Check if server already running
        if job_id in self.running_servers:
            process, existing_port = self.running_servers[job_id]
            if process.poll() is None:  # Process still running
                return {
                    "status": "already_running",
                    "job_id": job_id,
                    "port": existing_port,
                    "tensorboard_url": f"http://localhost:{existing_port}",
                    "message": "TensorBoard server is already running"
                }
            else:
                # Process died, clean up
                del self.running_servers[job_id]
        
        try:
            # Start TensorBoard server
            cmd = [
                "tensorboard",
                "--logdir", str(log_dir),
                "--port", str(port),
                "--host", "0.0.0.0",
                "--reload_interval", "10"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy()
            )
            
            # Store process reference
            self.running_servers[job_id] = (process, port)
            
            logger.info(f"Started TensorBoard server for job {job_id} on port {port}")
            
            return {
                "status": "started",
                "job_id": job_id,
                "port": port,
                "tensorboard_url": f"http://localhost:{port}",
                "pid": process.pid,
                "message": f"TensorBoard server started successfully on port {port}"
            }
            
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="TensorBoard not installed. Run: pip install tensorboard"
            )
        except Exception as e:
            logger.error(f"Failed to start TensorBoard server for job {job_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start TensorBoard server: {str(e)}"
            )
    
    def stop_server(self, job_id: str) -> Dict[str, Any]:
        """
        Stop TensorBoard server for a job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with stop status
        """
        if job_id not in self.running_servers:
            return {
                "status": "not_running",
                "job_id": job_id,
                "message": "TensorBoard server is not running for this job"
            }
        
        try:
            process, port = self.running_servers[job_id]
            
            if process.poll() is None:  # Process still running
                process.terminate()
                try:
                    process.wait(timeout=5)  # Wait up to 5 seconds
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if it doesn't terminate gracefully
                    process.wait()
            
            del self.running_servers[job_id]
            
            logger.info(f"Stopped TensorBoard server for job {job_id} (port {port})")
            
            return {
                "status": "stopped",
                "job_id": job_id,
                "port": port,
                "message": f"TensorBoard server stopped successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to stop TensorBoard server for job {job_id}: {e}")
            # Clean up entry even if stop failed
            if job_id in self.running_servers:
                del self.running_servers[job_id]
            
            return {
                "status": "error",
                "job_id": job_id,
                "message": f"Error stopping TensorBoard server: {str(e)}"
            }
    
    def get_server_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of TensorBoard server for a job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with server status
        """
        if job_id not in self.running_servers:
            return {
                "status": "stopped",
                "job_id": job_id,
                "running": False,
                "port": self.get_port_for_job(job_id),
                "tensorboard_url": f"http://localhost:{self.get_port_for_job(job_id)}"
            }
        
        process, port = self.running_servers[job_id]
        running = process.poll() is None
        
        if not running:
            # Process died, clean up
            del self.running_servers[job_id]
        
        return {
            "status": "running" if running else "stopped",
            "job_id": job_id,
            "running": running,
            "port": port,
            "tensorboard_url": f"http://localhost:{port}",
            "pid": process.pid if running else None
        }
    
    def cleanup_all(self):
        """Stop all running TensorBoard servers"""
        for job_id in list(self.running_servers.keys()):
            self.stop_server(job_id)


class JobStatus(str, Enum):
    """
    Enumeration of possible job states for optimization tasks
    
    Provides type-safe status tracking throughout the optimization lifecycle.
    Each status represents a distinct phase of the optimization process.
    
    States:
        PENDING: Job created but not yet started
        RUNNING: Optimization currently in progress
        COMPLETED: Job finished successfully with results
        FAILED: Job encountered an error and stopped
        CANCELLED: Job was manually cancelled by user
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# OptimizationRequest and OptimizationConfig are now centralized in configs.py


class JobResponse(BaseModel):
    """
    Response model for job status and information
    
    Provides comprehensive job information including current status,
    progress metrics, and results when available.
    
    Attributes:
        job_id: Unique identifier for the optimization job
        status: Current job status (pending, running, completed, failed, cancelled)
        created_at: ISO timestamp when job was created
        started_at: ISO timestamp when job started (None if not started)
        completed_at: ISO timestamp when job completed (None if not completed)
        progress: Progress information (trial counts, current metrics)
        result: Optimization results when job is completed
        error: Error message if job failed
        
    Example:
        {
            "job_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "running",
            "created_at": "2025-01-08T14:30:22Z",
            "started_at": "2025-01-08T14:30:25Z",
            "completed_at": null,
            "progress": {
                "current_trial": 15,
                "total_trials": 50,
                "best_total_score": 0.8750,
                "elapsed_time": 1800.5
            },
            "result": null,
            "error": null
        }
    """
    job_id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TrialData(BaseModel):
    """
    Comprehensive data structure for individual optimization trials
    
    UPDATED: Aligned with new TrialProgress structure
    
    Attributes:
        trial_id: Unique trial identifier within the job
        trial_number: Sequential trial number (1, 2, 3, ...)
        status: Current trial status (running, completed, failed, pruned)
        started_at: ISO timestamp when trial started
        completed_at: ISO timestamp when trial completed (None if running)
        duration_seconds: Total trial duration in seconds
        
        # Architecture Information
        architecture: Complete model architecture details
        hyperparameters: All hyperparameters used for this trial
        model_size: Model size metrics (parameters, memory, etc.)
        
        # Health Metrics
        health_metrics: Model health assessment data
        training_stability: Training stability indicators
        
        # Performance Data
        performance: Training and validation metrics
        training_history: Complete training history curves
        
        # Pruning Information
        pruning_info: Information about early stopping/pruning
        
    Example:
        {
            "trial_id": "trial_15_uuid",
            "trial_number": 15,
            "status": "completed",
            "started_at": "2025-01-08T14:45:30Z",
            "completed_at": "2025-01-08T14:47:15Z",
            "duration_seconds": 105.5,
            "architecture": {
                "type": "cnn",
                "layers": [...],
                "total_params": 245000,
                "memory_mb": 12.5
            },
            "health_metrics": {
                "overall_health": 0.82,
                "neuron_utilization": 0.85,
                "parameter_efficiency": 0.91,
                "training_stability": 0.88
            },
            "performance": {
                "final_accuracy": 0.875,
                "final_val_accuracy": 0.832,
                "final_loss": 0.234,
                "best_val_accuracy": 0.845
            }
        }
    """
    trial_id: str
    trial_number: int
    status: str
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # Architecture Information
    architecture: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    model_size: Optional[Dict[str, Any]] = None
    
    # Health Metrics
    health_metrics: Optional[Dict[str, Any]] = None
    training_stability: Optional[Dict[str, Any]] = None
    
    # Performance Data
    performance: Optional[Dict[str, Any]] = None
    training_history: Optional[Dict[str, Any]] = None
    
    # Pruning/Early Stopping Information
    pruning_info: Optional[Dict[str, Any]] = None


class OptimizationJob:
    """
    Enhanced job management class for tracking optimization tasks with real-time updates
    
    UPDATED: Integrated with new optimizer.py progress tracking system
    
    Handles the lifecycle of individual optimization jobs including:
    - Job state management
    - Real-time trial tracking via progress callbacks
    - Progress tracking with live updates
    - Result storage and retrieval
    - Error handling and logging
    
    This class bridges the FastAPI async world with the new optimizer system,
    providing clean separation of concerns while capturing comprehensive data 
    for frontend visualization.
    
    Attributes:
        job_id: Unique identifier for this job
        request: Original optimization request
        status: Current job status
        created_at: Job creation timestamp
        started_at: Job start timestamp (None if not started)
        completed_at: Job completion timestamp (None if not completed)
        progress: Current progress information
        result: Final optimization results (None if not completed)
        error: Error message if job failed (None if successful)
        task: Background asyncio task handle
        
        # NEW: Real-time trial tracking
        optimizer: ModelOptimizer instance with progress callback
        trial_progress_history: List of all trial progress updates
        current_trial_progress: Currently running trial data
        best_trial_progress: Best performing trial so far
        
    Example Usage:
        job = OptimizationJob(request)
        await job.start()
        status = job.get_status()
        trials = job.get_trial_history()
        current = job.get_current_trial()
    """
    
    def __init__(self, request: OptimizationRequest, tensorboard_manager=None):
        """
        Initialize a new optimization job with real-time progress tracking
        
        Args:
            request: OptimizationRequest containing job parameters
            tensorboard_manager: TensorBoard manager for automatic server startup
        """
        self.job_id = str(uuid.uuid4())
        self.request = request
        self.status = JobStatus.PENDING
        self.created_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.progress: Optional[Dict[str, Any]] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        
        # Real-time aggregated progress tracking
        self.optimizer: Optional[Any] = None  # Will be ModelOptimizer instance
        self.latest_aggregated_progress: Optional[Dict[str, Any]] = None
        
        # TensorBoard manager for automatic server startup
        self.tensorboard_manager = tensorboard_manager
        self.tensorboard_logs_dir: Optional[Path] = None  # Will be set when optimization starts
        self._current_epoch_info: Dict[str, Any] = {}
        
        logger.debug(f"running OptimizationJob.__init__ ... Created job {self.job_id} for dataset {request.dataset_name}")
        logger.debug(f"running OptimizationJob.__init__ ... Mode: {request.mode}, Objective: {request.optimize_for}")
        logger.debug(f"running OptimizationJob.__init__ ... Real-time trial tracking enabled")
    
    def _progress_callback(self, progress_data: Any) -> None:
        """
        Callback function to receive real-time progress updates
        
        This method is called by the optimizer whenever progress is updated.
        The optimizer can send either TrialProgress or AggregatedProgress objects.
        
        Args:
            progress_data: Either TrialProgress or AggregatedProgress object with current optimization state
        """
        try:
            # Check progress data type - prioritize UnifiedProgress
            if isinstance(progress_data, UnifiedProgress):
                # This is the new UnifiedProgress object (Phase 1: unified progress system)
                logger.debug(f"running OptimizationJob._progress_callback ... Received unified progress update")
                self._handle_unified_progress(progress_data)
            elif hasattr(progress_data, 'trial_number'):
                # Legacy: TrialProgress object (individual trial update) 
                logger.debug(f"running OptimizationJob._progress_callback ... Received individual trial progress update for trial {progress_data.trial_number}")
                self._handle_trial_progress(progress_data)
            else:
                # Legacy: AggregatedProgress object (consolidated update)
                logger.debug(f"running OptimizationJob._progress_callback ... Received aggregated progress update")
                self._handle_aggregated_progress(progress_data)
                
        except Exception as e:
            logger.error(f"running OptimizationJob._progress_callback ... Error processing progress update: {e}")
            # Don't re-raise to avoid disrupting optimization

    def _handle_unified_progress(self, unified_progress: UnifiedProgress) -> None:
        """
        Handle unified progress updates containing both trial statistics and epoch information
        This replaces the dual callback system and eliminates race conditions
        """
        try:
            # Extract data from UnifiedProgress object
            completed_count = len(unified_progress.completed_trials)
            running_count = len(unified_progress.running_trials)
            failed_count = len(unified_progress.failed_trials)
            total_trials = unified_progress.total_trials
            
            # Calculate current trial count for UI display
            current_trial = completed_count
            if running_count > 0:
                current_trial = completed_count + 1  # Show the next trial being processed
            
            # Calculate elapsed time
            elapsed_time = 0
            if self.started_at:
                start_time = datetime.fromisoformat(self.started_at.replace('Z', '+00:00').replace('T', ' '))
                elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # Store latest progress data
            self.latest_aggregated_progress = {
                "completed_trials": unified_progress.completed_trials,
                "running_trials": unified_progress.running_trials,
                "failed_trials": unified_progress.failed_trials,
                "total_trials": total_trials,
                "current_best_total_score": unified_progress.current_best_total_score
            }
            
            # Create progress update with all information
            progress_update = {
                "current_trial": current_trial,
                "total_trials": total_trials,
                "completed_trials": completed_count,
                "success_rate": completed_count / total_trials if total_trials > 0 else 0.0,
                "best_total_score": unified_progress.current_best_total_score,
                "best_accuracy": unified_progress.current_best_accuracy,
                "trials_performed": completed_count,
                "average_duration_per_trial": unified_progress.average_duration_per_trial,
                "elapsed_time": elapsed_time,
                "status_message": unified_progress.status_message or f"Trial {current_trial}/{total_trials} - {completed_count} completed, {running_count} running, {failed_count} failed",
                "is_gpu_mode": bool(self.optimizer and hasattr(self.optimizer, 'config') and getattr(self.optimizer.config, 'use_runpod_service', False))
            }
            
            # Include epoch information directly from unified progress (no more race conditions!)
            if unified_progress.current_epoch is not None:
                progress_update["current_epoch"] = unified_progress.current_epoch
            if unified_progress.total_epochs is not None:
                progress_update["total_epochs"] = unified_progress.total_epochs
            if unified_progress.epoch_progress is not None:
                progress_update["epoch_progress"] = unified_progress.epoch_progress
            
            # Include final model building progress
            if unified_progress.final_model_building is not None:
                progress_update["final_model_building"] = unified_progress.final_model_building
            
            self.progress = progress_update
            
            # 🔍 UNIFIED PROGRESS DEBUG: Log all data being sent to UI
            logger.info(f"🚀 UNIFIED PROGRESS UPDATE:")
            logger.info(f"  📊 Trial Info: {current_trial}/{total_trials} trials (completed: {completed_count}, running: {running_count}, failed: {failed_count})")
            logger.info(f"  📊 Best Score: {progress_update.get('best_total_score', 'None')}")
            logger.info(f"  📊 Elapsed Time: {progress_update.get('elapsed_time', 'None')}s")
            logger.info(f"  📊 Status Message: {progress_update.get('status_message', 'None')}")
            logger.info(f"  ⏱️ Epoch Info:")
            logger.info(f"    - Current Epoch: {progress_update.get('current_epoch', 'None')}")
            logger.info(f"    - Total Epochs: {progress_update.get('total_epochs', 'None')}")
            logger.info(f"    - Epoch Progress: {progress_update.get('epoch_progress', 'None')}")
            logger.info(f"  🏗️ Final Model Building:")
            logger.info(f"    - Status: {progress_update.get('final_model_building', {}).get('status', 'None')}")
            logger.info(f"    - Current Step: {progress_update.get('final_model_building', {}).get('current_step', 'None')}")
            logger.info(f"    - Progress: {progress_update.get('final_model_building', {}).get('progress', 'None')}")
            logger.info(f"  📊 Complete Progress Object: {progress_update}")
            
            # Log best score information
            best_total_score = unified_progress.current_best_total_score
            if best_total_score is not None:
                logger.info(f"📊 BEST SCORE: {best_total_score:.4f} (after {completed_count} completed trials)")
            
            logger.info(f"🔄 UNIFIED PROGRESS: {completed_count}/{total_trials} completed, {running_count} running, {failed_count} failed, best_score={best_total_score}")
            
        except Exception as e:
            logger.error(f"running OptimizationJob._handle_unified_progress ... Error processing unified progress: {e}")
            logger.error(f"running OptimizationJob._handle_unified_progress ... Traceback: {traceback.format_exc()}")
    
    def _handle_trial_progress(self, trial_progress: Any) -> None:
        """Handle individual trial progress updates with epoch information"""
        try:
            # 🔍 COMPREHENSIVE DEBUG: Log all trial progress data received
            logger.info(f"🔍 TRIAL PROGRESS UPDATE DEBUG:")
            logger.info(f"  📊 Trial Number: {getattr(trial_progress, 'trial_number', 'None')}")
            logger.info(f"  📊 Trial Status: {getattr(trial_progress, 'status', 'None')}")
            logger.info(f"  📊 Raw Epoch Data:")
            logger.info(f"    - current_epoch: {getattr(trial_progress, 'current_epoch', 'None')}")
            logger.info(f"    - total_epochs: {getattr(trial_progress, 'total_epochs', 'None')}")
            logger.info(f"    - epoch_progress: {getattr(trial_progress, 'epoch_progress', 'None')}")
            logger.info(f"  📊 All Trial Progress Attributes: {[attr for attr in dir(trial_progress) if not attr.startswith('_')]}")
            
            # Store epoch information from individual trial
            self._current_epoch_info = {
                'current_epoch': getattr(trial_progress, 'current_epoch', None),
                'total_epochs': getattr(trial_progress, 'total_epochs', None),
                'epoch_progress': getattr(trial_progress, 'epoch_progress', None)
            }
            logger.info(f"🔍 STORED EPOCH INFO: {self._current_epoch_info}")
            
        except Exception as e:
            logger.error(f"running OptimizationJob._handle_trial_progress ... Error handling trial progress: {e}")
    
    def _handle_aggregated_progress(self, aggregated_progress: Any) -> None:
        """Handle aggregated progress updates and update UI"""
        try:
            # Extract data from AggregatedProgress object
            completed_count = len(aggregated_progress.completed_trials) if hasattr(aggregated_progress, 'completed_trials') else 0
            running_count = len(aggregated_progress.running_trials) if hasattr(aggregated_progress, 'running_trials') else 0
            failed_count = len(aggregated_progress.failed_trials) if hasattr(aggregated_progress, 'failed_trials') else 0
            total_trials = getattr(aggregated_progress, 'total_trials', 0)
            
            # FIXED: Calculate current trial count for UI display (not trial numbers)
            # The UI expects "X/Y trials" where X is the count of trials processed
            current_trial = completed_count
            if running_count > 0:
                # If there are trials running, show the next trial count
                current_trial = completed_count + 1  # Show the next trial being processed
            
            # Calculate elapsed time
            elapsed_time = 0
            if self.started_at:
                start_time = datetime.fromisoformat(self.started_at.replace('Z', '+00:00').replace('T', ' '))
                elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # Store latest aggregated progress data
            self.latest_aggregated_progress = {
                "completed_trials": aggregated_progress.completed_trials if hasattr(aggregated_progress, 'completed_trials') else [],
                "running_trials": aggregated_progress.running_trials if hasattr(aggregated_progress, 'running_trials') else [],
                "failed_trials": aggregated_progress.failed_trials if hasattr(aggregated_progress, 'failed_trials') else [],
                "total_trials": total_trials,
                "current_best_total_score": getattr(aggregated_progress, 'current_best_total_score', None)
            }
            
            # Extract epoch information from stored trial progress if available
            current_epoch = self._current_epoch_info.get('current_epoch')
            total_epochs = self._current_epoch_info.get('total_epochs')  
            epoch_progress = self._current_epoch_info.get('epoch_progress')
            
            # Update progress directly from aggregated data
            progress_update = {
                "current_trial": current_trial,
                "total_trials": total_trials,
                "completed_trials": completed_count,
                "success_rate": completed_count / total_trials if total_trials > 0 else 0.0,
                "best_total_score": getattr(aggregated_progress, 'current_best_total_score', None),
                "elapsed_time": elapsed_time,
                "status_message": f"Trial {current_trial}/{total_trials} - {completed_count} completed, {running_count} running, {failed_count} failed",
                "is_gpu_mode": bool(self.optimizer and hasattr(self.optimizer, 'config') and getattr(self.optimizer.config, 'use_runpod_service', False))
            }
            
            # Add epoch information if available
            if current_epoch is not None:
                progress_update["current_epoch"] = current_epoch
            if total_epochs is not None:
                progress_update["total_epochs"] = total_epochs
            if epoch_progress is not None:
                progress_update["epoch_progress"] = epoch_progress
                
            self.progress = progress_update
            
            # 🔍 COMPREHENSIVE DEBUG: Log all progress data being sent to UI
            logger.info(f"🔍 FULL PROGRESS UPDATE DEBUG:")
            logger.info(f"  📊 Trial Info: {current_trial}/{total_trials} trials (completed: {completed_count}, running: {running_count}, failed: {failed_count})")
            logger.info(f"  📊 Best Score: {progress_update.get('best_total_score', 'None')}")
            logger.info(f"  📊 Elapsed Time: {progress_update.get('elapsed_time', 'None')}s")
            logger.info(f"  📊 Status Message: {progress_update.get('status_message', 'None')}")
            logger.info(f"  📊 Epoch Info:")
            logger.info(f"    - Current Epoch: {progress_update.get('current_epoch', 'None')}")
            logger.info(f"    - Total Epochs: {progress_update.get('total_epochs', 'None')}")
            logger.info(f"    - Epoch Progress: {progress_update.get('epoch_progress', 'None')}")
            logger.info(f"  📊 Stored Epoch Info: {self._current_epoch_info}")
            logger.info(f"  📊 Complete Progress Object: {progress_update}")
            
            # ENHANCED: Log best score information for UI verification
            best_total_score = getattr(aggregated_progress, 'current_best_total_score', None)
            if best_total_score is not None:
                logger.info(f"📊 BEST SCORE UPDATE: Current best value = {best_total_score:.4f} (after {completed_count} completed trials)")
            
            # Log detailed progress summary for verification
            logger.info(f"🔄 PROGRESS UPDATE: {completed_count}/{total_trials} completed, {running_count} running, {failed_count} failed, best_score={best_total_score}")
            
        except Exception as e:
            logger.error(f"running OptimizationJob._handle_aggregated_progress ... Error handling aggregated progress: {e}")
    
    def _update_job_progress(self) -> None:
        """
        Update job progress based on current trial state
        
        Called whenever trial progress is updated to maintain overall job progress.
        """
        if not self.optimizer:
            return
        
        try:
            # Get progress from optimizer
            opt_progress = self.optimizer.get_optimization_progress()
            
            # Update job progress
            self.progress = {
                "current_trial": opt_progress["current_trial"],
                "total_trials": opt_progress["total_trials"],
                "completed_trials": opt_progress["completed_trials"],
                "success_rate": opt_progress["success_rate"],
                "best_total_score": opt_progress["best_total_score"],
                "elapsed_time": opt_progress["elapsed_time"],
                "status_message": f"Trial {opt_progress['current_trial']}/{opt_progress['total_trials']} running"
            }
            
            logger.debug(f"running OptimizationJob._update_job_progress ... Progress: {opt_progress['current_trial']}/{opt_progress['total_trials']} trials")
            
        except Exception as e:
            logger.warning(f"running OptimizationJob._update_job_progress ... Failed to update progress: {e}")
    
    async def start(self) -> None:
        """
        Start the optimization job as a background task
        
        UPDATED: Uses new optimizer.py with progress callback integration
        
        Launches the optimization process asynchronously while updating
        job status and progress. Handles all exceptions gracefully.
        
        Side Effects:
            - Updates job status to RUNNING
            - Sets started_at timestamp
            - Launches background asyncio task
            - Logs job start event
            
        Raises:
            RuntimeError: If job is already running or completed
        """
        if self.status != JobStatus.PENDING:
            raise RuntimeError(f"Job {self.job_id} is not in pending state")
        
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now().isoformat()
        
        logger.debug(f"running OptimizationJob.start ... Starting optimization job {self.job_id}")
        logger.debug(f"running OptimizationJob.start ... Dataset: {self.request.dataset_name}")
        logger.debug(f"running OptimizationJob.start ... Mode: {self.request.mode}")
        logger.debug(f"running OptimizationJob.start ... Objective: {self.request.optimize_for}")
        logger.debug(f"running OptimizationJob.start ... Trials: {self.request.trials}")
        
        # Start the optimization task
        self.task = asyncio.create_task(self._run_optimization())
        
        # Auto-start TensorBoard server in the background
        asyncio.create_task(self._auto_start_tensorboard())
    
    async def _auto_start_tensorboard(self) -> None:
        """
        Automatically start TensorBoard server after optimization begins
        
        Waits for TensorBoard logs to be created, then starts the server automatically.
        This eliminates the need for manual TensorBoard server startup.
        """
        try:
            if not self.tensorboard_manager:
                logger.info(f"No TensorBoard manager available for job {self.job_id}")
                return
                
            logger.info(f"Auto-starting TensorBoard for job {self.job_id}")
            
            # Wait a bit for optimization to start and create log directories
            await asyncio.sleep(10)  # Give time for first trial to start logging
            
            # Wait for tensorboard logs directory to be available
            max_wait = 30  # Wait up to 30 more seconds
            wait_count = 0
            while not self.tensorboard_logs_dir and wait_count < max_wait:
                await asyncio.sleep(1)
                wait_count += 1
            
            if not self.tensorboard_logs_dir:
                logger.warning(f"running OptimizationJob._auto_start_tensorboard ... TensorBoard logs directory not available after waiting")
                return
            
            if not self.tensorboard_logs_dir.exists():
                logger.info(f"running OptimizationJob._auto_start_tensorboard ... Creating TensorBoard logs directory: {self.tensorboard_logs_dir}")
                try:
                    self.tensorboard_logs_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"running OptimizationJob._auto_start_tensorboard ... Successfully created TensorBoard logs directory")
                except Exception as e:
                    logger.warning(f"running OptimizationJob._auto_start_tensorboard ... Failed to create TensorBoard logs directory: {e}")
                    return
            
            # Start TensorBoard server automatically
            logger.info(f"Starting TensorBoard server with logs from: {self.tensorboard_logs_dir}")
            result = self.tensorboard_manager.start_server(self.job_id, self.tensorboard_logs_dir)
            logger.info(f"running OptimizationJob._auto_start_tensorboard ... TensorBoard auto-started: {result}")
                
        except Exception as e:
            logger.warning(f"running OptimizationJob._auto_start_tensorboard ... Failed to auto-start TensorBoard: {e}")
            # Don't fail the optimization if TensorBoard startup fails
    
    async def _run_optimization(self) -> None:
        """
        Execute the optimization process in background
        
        UPDATED: Uses new optimizer.py with progress callback
        
        Runs the actual hyperparameter optimization using the new unified
        optimizer system. Updates progress and handles results.
        
        This method bridges async FastAPI with the optimizer by running
        the optimization in a thread pool executor.
        
        Side Effects:
            - Updates job progress during execution
            - Sets final result on completion
            - Updates status to COMPLETED or FAILED
            - Sets completed_at timestamp
            - Logs optimization progress and results
            
        Error Handling:
            - Catches all exceptions and stores in job.error
            - Sets status to FAILED on any error
            - Logs detailed error information
        """
        try:
            logger.debug(f"running OptimizationJob._run_optimization ... Starting optimization for job {self.job_id}")
            
            # Initialize progress tracking
            self.progress = {
                "current_trial": 0,
                "total_trials": self.request.trials,
                "completed_trials": 0,
                "success_rate": 0.0,
                "best_total_score": None,
                "elapsed_time": 0,
                "status_message": "Initializing optimization..."
            }
            
            # Run optimization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Update progress: Starting optimization
            self.progress["status_message"] = "Loading dataset and initializing optimizer..."
            logger.debug(f"running OptimizationJob._run_optimization ... Loading dataset {self.request.dataset_name}")
            
            # Execute optimization with cancellation handling
            # Use a dedicated executor so we can control cancellation better
            result = None  # Initialize to avoid "possibly unbound" errors
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(self._execute_optimization)
                    
                    # Check for cancellation periodically while waiting for result
                    while not future.done():
                        if self.status == JobStatus.CANCELLED:
                            logger.info(f"running OptimizationJob._run_optimization ... Cancellation detected, requesting optimizer stop")
                            if hasattr(self, 'optimizer') and self.optimizer:
                                self.optimizer.cancel()
                            
                            # Give the optimization a short time to stop gracefully
                            try:
                                result = await asyncio.wait_for(
                                    asyncio.wrap_future(future), 
                                    timeout=5.0
                                )
                            except asyncio.TimeoutError:
                                logger.warning(f"running OptimizationJob._run_optimization ... Optimization didn't stop gracefully, forcing cancellation")
                                executor.shutdown(wait=False, cancel_futures=True)
                                raise asyncio.CancelledError("Optimization forcefully cancelled")
                            break
                        
                        await asyncio.sleep(0.5)  # Check every 500ms
                    
                    if future.done():
                        result = future.result()
                    
                except asyncio.CancelledError:
                    logger.info(f"running OptimizationJob._run_optimization ... Job {self.job_id} was cancelled")
                    self.status = JobStatus.CANCELLED
                    self.completed_at = datetime.now().isoformat()
                    self.progress["status_message"] = "Optimization cancelled by user"
                    raise
            
            # Ensure we have a valid result before processing
            if result is None:
                raise RuntimeError("Optimization completed but no result was obtained")
            
            # Convert OptimizationResult to API format
            api_result = self._convert_optimization_result(result)
            
            # Store successful result
            self.result = api_result
            self.status = JobStatus.COMPLETED
            self.completed_at = datetime.now().isoformat()
            
            # Update final progress
            self.progress["status_message"] = "Optimization completed successfully"
            self.progress["current_trial"] = self.request.trials
            self.progress["best_total_score"] = result.best_total_score
            
            logger.debug(f"running OptimizationJob._run_optimization ... Job {self.job_id} completed successfully")
            logger.debug(f"running OptimizationJob._run_optimization ... Best value: {result.best_total_score:.4f}")
            
        except Exception as e:
            # Handle any optimization errors
            self.error = str(e)
            self.status = JobStatus.FAILED
            self.completed_at = datetime.now().isoformat()
            
            if self.progress:
                self.progress["status_message"] = f"Optimization failed: {str(e)}"
            
            logger.error(f"running OptimizationJob._run_optimization ... Job {self.job_id} failed: {e}")
            logger.debug(f"running OptimizationJob._run_optimization ... Error traceback: {traceback.format_exc()}")
    
    def _execute_optimization(self) -> OptimizationResult:
        """
        Execute the actual optimization (synchronous)
        
        UPDATED: Uses new optimizer.py with progress callback integration
        FIXED: Generate proper timestamp-based run_name instead of using job_id
        
        Calls the new unified optimize_model function with the job's parameters
        and integrates real-time progress tracking. This runs in a thread pool 
        to avoid blocking the async event loop.
        
        Returns:
            OptimizationResult object from the new optimizer
            
        Raises:
            Exception: Any error from the optimization process
        """
        logger.debug(f"running OptimizationJob._execute_optimization ... Executing optimization for job {self.job_id}")
        
        # CRITICAL FIX: Ensure logging setup is applied before running optimization
        # This guarantees UI-triggered optimizations write to the same log file as command-line runs
        setup_logging()
        logger.info(f"Starting UI-triggered optimization job {self.job_id} - logs will be written to logs/non-cron.log")
        
        # Validate mode and objective
        try:
            opt_mode = OptimizationMode(self.request.mode.lower())
            opt_objective = OptimizationObjective(self.request.optimize_for.lower())
        except ValueError as e:
            raise ValueError(f"Invalid optimization parameters: {e}")
        
        # Early validation for mode-objective compatibility
        if opt_mode == OptimizationMode.SIMPLE and OptimizationObjective.is_health_only(opt_objective):
            universal_objectives = [obj.value for obj in OptimizationObjective.get_universal_objectives()]
            raise ValueError(
                f"Cannot use health-only objective '{self.request.optimize_for}' in simple mode. "
                f"Available objectives for simple mode: {universal_objectives}"
            )
        
        # FIXED: Use enhanced optimize_model function to avoid duplicating run_name generation logic
        # This ensures consistency and eliminates code duplication
        
        # Use clean conversion function - gets all user values + system defaults
        opt_config = self.request  # OptimizationRequest is now the unified config
        
        logger.debug(f"running OptimizationJob._execute_optimization ... Using clean OptimizationConfig conversion")
        logger.debug(f"running OptimizationJob._execute_optimization ... Config created with user values: dataset_name={opt_config.dataset_name}, mode={opt_config.mode}, trials={opt_config.trials}")
        logger.debug(f"running OptimizationJob._execute_optimization ... System defaults: batch_size={opt_config.batch_size}, learning_rate={opt_config.learning_rate}")
        
        # Create optimizer instance and store reference for cancellation
        self.optimizer = ModelOptimizer(
            dataset_name=self.request.dataset_name,
            optimization_config=opt_config,
            run_name=None,
            progress_callback=self._progress_callback,
            activation_override=opt_config.activation_functions[0] if opt_config.activation_functions else None
        )
        
        # Store TensorBoard logs directory for auto-start
        if hasattr(self.optimizer, 'results_dir') and self.optimizer.results_dir:
            self.tensorboard_logs_dir = self.optimizer.results_dir / "tensorboard_logs"
        
        result = self.optimizer.optimize()
        
        logger.debug(f"running OptimizationJob._execute_optimization ... Optimization completed for job {self.job_id}")
        logger.debug(f"running OptimizationJob._execute_optimization ... Results directory: {result.results_dir}")
        logger.debug(f"running OptimizationJob._execute_optimization ... Best value: {result.best_total_score:.4f}")
        
        return result
            
    def _convert_optimization_result(self, result: OptimizationResult) -> Dict[str, Any]:
        """
        Convert OptimizationResult to API-compatible format
        
        UPDATED: Handles new OptimizationResult dataclass format
        
        Args:
            result: OptimizationResult from optimizer
            
        Returns:
            Dictionary in API-expected format
        """
        try:
            # Convert to API format
            api_result = {
                "optimization_result": {
                    "best_total_score": result.best_total_score,
                    "best_params": result.best_params,
                    "total_trials": result.total_trials,
                    "successful_trials": result.successful_trials,
                    "optimization_time_hours": result.optimization_time_hours,
                    "parameter_importance": result.parameter_importance,
                    "dataset_name": result.dataset_name,
                    "optimization_mode": result.optimization_mode,
                    "health_weight": result.health_weight,
                    "objective_history": result.objective_history,
                    "best_trial_health": result.best_trial_health,
                    "average_health_metrics": result.average_health_metrics
                },
                "model_result": {
                    "model_path": result.best_model_path,
                    "test_accuracy": result.best_total_score if (result.optimization_config and "accuracy" in str(result.optimization_config.objective)) else None,
                    "results_dir": str(result.results_dir) if result.results_dir else None
                } if result.best_model_path else None,
                "run_name": self.job_id,
                "best_total_score": result.best_total_score,
                "best_params": result.best_params,
                "health_data": {
                    "best_trial_health": result.best_trial_health,
                    "average_health_metrics": result.average_health_metrics,
                    "health_history": result.health_history
                }
            }
            
            logger.debug(f"running OptimizationJob._convert_optimization_result ... Converted OptimizationResult to API format")
            return api_result
            
        except Exception as e:
            logger.error(f"running OptimizationJob._convert_optimization_result ... Failed to convert result: {e}")
            # Return minimal result on conversion error
            return {
                "optimization_result": {
                    "best_total_score": result.best_total_score,
                    "best_params": result.best_params,
                    "total_trials": result.total_trials,
                    "successful_trials": result.successful_trials,
                    "dataset_name": result.dataset_name,
                    "optimization_mode": result.optimization_mode
                },
                "run_name": self.job_id,
                "best_total_score": result.best_total_score,
                "best_params": result.best_params,
                "conversion_error": str(e)
            }
    
    def get_status(self) -> JobResponse:
        """
        Get current job status and information
        
        Returns comprehensive job information including status,
        progress, and results. Used by API endpoints to provide
        real-time job information to clients.
        
        Returns:
            JobResponse containing all current job information
        """
        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            progress=self.progress,
            result=self.result,
            error=self.error
        )
    
    def get_trial_history(self) -> List[Dict[str, Any]]:
        """
        Get complete trial history for visualization
        
        NEW: Uses real-time trial progress data from optimizer
        
        Returns serialized trial data for frontend consumption including
        architecture details, health metrics, and performance data.
        
        Returns:
            List of trial dictionaries with complete information
        """
        if self.optimizer:
            return self.optimizer.get_trial_history()
        else:
            # Fallback when optimizer is not available
            return []
    
    def get_current_trial(self) -> Optional[Dict[str, Any]]:
        """
        Get currently running trial data
        
        NEW: Uses real-time trial progress data from optimizer
        
        Returns:
            Current trial dictionary or None if no trial running
        """
        if self.optimizer:
            return self.optimizer.get_current_trial()
        else:
            # Fallback: use aggregated progress data if available
            if self.latest_aggregated_progress and self.latest_aggregated_progress.get('running_trials'):
                running_trials = self.latest_aggregated_progress['running_trials']
                if running_trials:
                    return {"trial_id": f"trial_{running_trials[0]}", "trial_number": running_trials[0], "status": "running"}
            return None
    
    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """
        Get best performing trial so far
        
        NEW: Uses real-time trial progress data from optimizer
        
        Returns:
            Best trial dictionary or None if no completed trials
        """
        if self.optimizer:
            return self.optimizer.get_best_trial()
        else:
            # Fallback: use aggregated progress data if available
            if self.latest_aggregated_progress and self.latest_aggregated_progress.get('current_best_total_score') is not None:
                return {
                    "trial_id": "best_trial", 
                    "status": "completed",
                    "best_total_score": self.latest_aggregated_progress['current_best_total_score']
                }
            return None
    
    def get_architecture_trends(self) -> Dict[str, List[float]]:
        """
        Get architecture performance trends for visualization
        
        NEW: Uses real-time data from optimizer
        
        Returns trends showing how different architectural choices
        affect performance over time.
        
        Returns:
            Dictionary mapping architecture features to performance trends
        """
        if self.optimizer:
            return self.optimizer.get_architecture_trends()
        else:
            return {}
    
    def get_health_trends(self) -> Dict[str, List[float]]:
        """
        Get health metrics trends for visualization
        
        NEW: Uses real-time data from optimizer
        
        Returns trends showing how model health metrics evolve
        across trials.
        
        Returns:
            Dictionary mapping health metrics to trend data
        """
        if self.optimizer:
            return self.optimizer.get_health_trends()
        else:
            return {}
    
    async def cancel(self) -> None:
        """
        Cancel a running optimization job
        
        Attempts to gracefully cancel the optimization task.
        Note: Due to the nature of the optimization process,
        cancellation may not be immediate.
        
        Side Effects:
            - Updates status to CANCELLED
            - Cancels background asyncio task
            - Sets completed_at timestamp
            - Logs cancellation event
            
        Raises:
            RuntimeError: If job is not in a cancellable state
        """
        if self.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
            raise RuntimeError(f"Job {self.job_id} cannot be cancelled in {self.status} state")
        
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.now().isoformat()
        
        if self.task:
            self.task.cancel()
        
        # Cancel the optimizer if it exists
        if hasattr(self, 'optimizer') and self.optimizer:
            logger.debug(f"running OptimizationJob.cancel ... Cancelling optimizer for job {self.job_id}")
            self.optimizer.cancel()
        
        if self.progress:
            self.progress["status_message"] = "Job cancelled by user"
        
        logger.debug(f"running OptimizationJob.cancel ... Job {self.job_id} cancelled")


class OptimizationAPI:
    """
    Main FastAPI application class for hyperparameter optimization
    
    UPDATED: Integrated with new optimizer.py system
    
    Provides REST API endpoints for managing optimization jobs and
    integrates with the new unified optimizer system. Handles
    job lifecycle management, progress tracking, and result retrieval.
    
    Key Features:
        - Asynchronous job management
        - Real-time progress monitoring with trial tracking
        - Result persistence and retrieval
        - Model download capabilities
        - Comprehensive error handling
        - Integration with new optimizer.py
    
    Attributes:
        app: FastAPI application instance
        jobs: Dictionary mapping job IDs to OptimizationJob instances
        dataset_manager: DatasetManager instance for dataset validation
        
    Example Usage:
        api = OptimizationAPI()
        app = api.app
        
        # Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
    """
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """
        Lifespan event handler for FastAPI application
        Replaces deprecated on_event handlers
        """
        # Startup - nothing needed currently
        yield
        # Shutdown - cleanup TensorBoard servers
        logger.info("Shutting down API server - cleaning up TensorBoard servers")
        if hasattr(self, 'tensorboard_manager'):
            self.tensorboard_manager.cleanup_all()
    
    def __init__(self):
        """
        Initialize the FastAPI application with all endpoints
        
        Sets up the FastAPI instance, configures CORS, initializes
        job storage, and registers all API endpoints.
        
        FIXED: Ensures consistent logging to files for all optimizations
        """
        # CRITICAL FIX: Ensure logging is configured consistently for both UI and command-line triggers
        # This ensures logs always go to logs/non-cron.log regardless of how optimization is started
        setup_logging()
        logger.info("API server initializing - logging configured to write to logs/non-cron.log")
        self.app = FastAPI(
            title="Hyperparameter Optimization API",
            description="REST API for automated hyperparameter optimization with GPU acceleration",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            lifespan=self.lifespan
        )
        
        # Initialize TensorBoard manager
        self.tensorboard_manager = TensorBoardManager()
        
        # Configure CORS for Next.js development server
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",  # Next.js dev server
                "http://127.0.0.1:3000",
                "http://0.0.0.0:3000"
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Job storage - in production, use Redis or database
        self.jobs: Dict[str, OptimizationJob] = {}
        
        # Initialize dataset manager for validation
        self.dataset_manager = DatasetManager()
        
        # Register API endpoints
        self._register_routes()
        
        logger.debug("running OptimizationAPI.__init__ ... FastAPI application initialized with CORS")
        logger.debug("running OptimizationAPI.__init__ ... Available datasets: " + 
                    ", ".join(self.dataset_manager.get_available_datasets()))
    
    def _register_routes(self) -> None:
        """
        Register all API endpoints with the FastAPI application
        
        Defines the complete REST API interface including:
        - Job management endpoints
        - Status and progress endpoints
        - Result retrieval endpoints
        - Health check endpoints
        - File download endpoints
        - Job control endpoints
        
        Side Effects:
            - Registers all endpoints with self.app
            - Configures request/response models
            - Sets up error handling
        """
        logger.debug("running OptimizationAPI._register_routes ... Registering API endpoints")
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for monitoring"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # Dataset information endpoints
        @self.app.get("/datasets")
        async def list_datasets():
            """Get list of available datasets"""
            return {"datasets": self.dataset_manager.get_available_datasets()}
        
        # Available modes and objectives endpoints
        @self.app.get("/modes")
        async def list_modes():
            """Get available optimization modes"""
            return {
                "modes": [mode.value for mode in OptimizationMode],
                "descriptions": {
                    "simple": "Pure objective optimization (health monitoring only)",
                    "health": "Health-aware optimization with configurable weighting"
                }
            }
        
        @self.app.get("/objectives")
        async def list_objectives():
            """Get available optimization objectives"""
            universal_objectives = [obj.value for obj in OptimizationObjective.get_universal_objectives()]
            health_objectives = [obj.value for obj in OptimizationObjective.get_health_only_objectives()]
            
            return {
                "universal_objectives": universal_objectives,
                "health_only_objectives": health_objectives,
                "descriptions": {
                    "universal": "Work in both simple and health modes",
                    "health_only": "Only work in health mode"
                }
            }
        
        # Job management endpoints
        @self.app.post("/optimize", response_model=JobResponse)
        async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
            """Start a new hyperparameter optimization job"""
            return await self._start_optimization(request, background_tasks)
        
        @self.app.get("/jobs/{job_id}", response_model=JobResponse)
        async def get_job_status(job_id: str):
            """Get status and progress of an optimization job"""
            return await self._get_job_status(job_id)
        
        @self.app.get("/jobs/{job_id}/comprehensive")
        async def get_comprehensive_status(job_id: str):
            """Get comprehensive status combining job progress, trials, and elapsed time"""
            return await self._get_comprehensive_status(job_id)
        
        @self.app.get("/jobs")
        async def list_jobs():
            """List all optimization jobs"""
            return await self._list_jobs()
        
        @self.app.delete("/jobs/{job_id}")
        async def cancel_job(job_id: str):
            """Cancel a running optimization job"""
            return await self._cancel_job(job_id)
        
        # Results endpoints
        @self.app.get("/results/{job_id}")
        async def get_results(job_id: str):
            """Get detailed results from completed optimization job"""
            return await self._get_results(job_id)
        
        @self.app.get("/download/{job_id}")
        async def download_model(job_id: str):
            """Download the trained model from completed optimization job"""
            return await self._download_model(job_id)
        
        # Enhanced endpoints for visualization
        @self.app.get("/jobs/{job_id}/trials")
        async def get_trial_history(job_id: str):
            """Get complete trial history for visualization"""
            return await self._get_trial_history(job_id)
        
        @self.app.get("/jobs/{job_id}/current-trial")
        async def get_current_trial(job_id: str):
            """Get currently running trial data"""
            return await self._get_current_trial(job_id)
        
        @self.app.get("/jobs/{job_id}/best-trial")
        async def get_best_trial(job_id: str):
            """Get best performing trial so far"""
            return await self._get_best_trial(job_id)
        
        # 3D Visualization endpoints
        @self.app.get("/jobs/{job_id}/best-model")
        async def get_best_model_visualization(job_id: str):
            """Get best model data with 3D visualization information"""
            return await self._get_best_model_visualization(job_id)
        
        @self.app.get("/jobs/{job_id}/best-model/download")
        async def download_best_model_visualization(job_id: str):
            """Download 3D visualization data as JSON file"""
            return await self._download_best_model_visualization(job_id)
        
        @self.app.get("/jobs/{job_id}/cytoscape/architecture")
        async def get_cytoscape_architecture(job_id: str, trial_id: Optional[str] = None):
            """Get Cytoscape.js architecture data for best model or specific trial"""
            return await self._get_cytoscape_architecture(job_id, trial_id)
        
        @self.app.get("/jobs/{job_id}/trends")
        async def get_trends(job_id: str):
            """Get architecture and health trends for visualization"""
            return await self._get_trends(job_id)
        
        # File management endpoints
        @self.app.get("/jobs/{job_id}/files")
        async def list_job_files(job_id: str):
            """List all files in a job's results directory"""
            return await self._list_job_files(job_id)
        
        @self.app.get("/jobs/{job_id}/files/{file_path:path}")
        async def download_job_file(job_id: str, file_path: str):
            """Download a specific file from job results"""
            return await self._download_job_file(job_id, file_path)
        
        @self.app.get("/jobs/{job_id}/download")
        async def download_job_results(job_id: str):
            """Download entire job results directory as ZIP"""
            return await self._download_job_results_zip(job_id)
        
        # Job control endpoints
        @self.app.post("/jobs/stop")
        async def stop_all_jobs():
            """Stop all running jobs"""
            return await self._stop_all_jobs()
        
        @self.app.post("/jobs/{job_id}/stop")
        async def stop_job(job_id: str):
            """Stop a specific running job"""
            return await self._stop_job(job_id)
        
        # TensorBoard integration endpoints
        @self.app.get("/jobs/{job_id}/tensorboard/logs")
        async def get_tensorboard_logs(job_id: str):
            """Get available TensorBoard log directories for a job"""
            return await self._get_tensorboard_logs(job_id)
        
        @self.app.get("/jobs/{job_id}/tensorboard/url")
        async def get_tensorboard_url(job_id: str):
            """Get TensorBoard server URL for a job"""
            return await self._get_tensorboard_url(job_id)
        
        @self.app.post("/jobs/{job_id}/tensorboard/start")
        async def start_tensorboard_server(job_id: str):
            """Start TensorBoard server for a job"""
            return await self._start_tensorboard_server(job_id)
        
        @self.app.post("/jobs/{job_id}/tensorboard/stop")
        async def stop_tensorboard_server(job_id: str):
            """Stop TensorBoard server for a job"""
            return await self._stop_tensorboard_server(job_id)
        
        # Plot serving endpoints for embedded visualizations
        @self.app.get("/jobs/{job_id}/plots/{trial_id}/{plot_type}")
        async def get_trial_plot(job_id: str, trial_id: str, plot_type: str):
            """Serve training visualization plots for specific trial"""
            return await self._get_trial_plot(job_id, trial_id, plot_type)
        
        @self.app.get("/jobs/{job_id}/plots/{trial_id}")
        async def list_trial_plots(job_id: str, trial_id: str):
            """List available plots for a specific trial"""
            return await self._list_trial_plots(job_id, trial_id)
        
        @self.app.get("/jobs/{job_id}/plots")
        async def list_job_plots(job_id: str):
            """List available plots for all trials in a job"""
            return await self._list_job_plots(job_id)
        
        logger.debug("running OptimizationAPI._register_routes ... All API endpoints registered")
               
    async def _start_optimization(self, request: OptimizationRequest, background_tasks: BackgroundTasks) -> JobResponse:
        """
        Start a new hyperparameter optimization job
        
        UPDATED: Enhanced validation for new optimizer.py parameters
        
        Validates the request, creates a new job, and starts the optimization
        process in the background. Returns immediately with job information.
        
        Args:
            request: OptimizationRequest containing job parameters
            background_tasks: FastAPI background tasks for async execution
            
        Returns:
            JobResponse with new job information
            
        Raises:
            HTTPException: If request validation fails
        """
        logger.debug(f"running OptimizationAPI._start_optimization ... Starting optimization request")
        logger.debug(f"running OptimizationAPI._start_optimization ... Dataset: {request.dataset_name}")
        logger.debug(f"running OptimizationAPI._start_optimization ... Mode: {request.mode}")
        logger.debug(f"running OptimizationAPI._start_optimization ... Objective: {request.optimize_for}")
        logger.debug(f"running OptimizationAPI._start_optimization ... Trials: {request.trials}")
        logger.debug(f"running OptimizationAPI._start_optimization ... Health weight: {request.health_weight}")
        
        # Validate dataset
        if request.dataset_name not in self.dataset_manager.get_available_datasets():
            available = ", ".join(self.dataset_manager.get_available_datasets())
            logger.error(f"running OptimizationAPI._start_optimization ... Invalid dataset: {request.dataset_name}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset '{request.dataset_name}' not supported. Available: {available}"
            )
        
        # Validate mode
        try:
            opt_mode = OptimizationMode(request.mode.lower())
        except ValueError:
            available_modes = [mode.value for mode in OptimizationMode]
            logger.error(f"running OptimizationAPI._start_optimization ... Invalid mode: {request.mode}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Mode '{request.mode}' not supported. Available: {available_modes}"
            )
        
        # Validate objective
        try:
            opt_objective = OptimizationObjective(request.optimize_for.lower())
        except ValueError:
            available_objectives = [obj.value for obj in OptimizationObjective]
            logger.error(f"running OptimizationAPI._start_optimization ... Invalid objective: {request.optimize_for}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Objective '{request.optimize_for}' not supported. Available: {available_objectives}"
            )
        
        # Validate mode-objective compatibility
        if opt_mode == OptimizationMode.SIMPLE and OptimizationObjective.is_health_only(opt_objective):
            universal_objectives = [obj.value for obj in OptimizationObjective.get_universal_objectives()]
            logger.error(f"running OptimizationAPI._start_optimization ... Invalid mode-objective combination")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot use health-only objective '{request.optimize_for}' in simple mode. Available for simple mode: {universal_objectives}"
            )
        
        # Validate epoch configuration is sane BEFORE creating the job
        max_epochs = request.max_epochs_per_trial
        min_epochs = request.min_epochs_per_trial
        
        # Ensure max_epochs is at least min_epochs
        if max_epochs < min_epochs:
            logger.warning(f"running _start_optimization ... max_epochs_per_trial ({max_epochs}) too low, setting to {min_epochs}")
            # Update the request in place
            request.max_epochs_per_trial = min_epochs
        
        logger.debug(f"running _start_optimization ... Using epoch configuration: min={request.min_epochs_per_trial}, max={request.max_epochs_per_trial}")
        logger.debug(f"running _start_optimization ... User parameters: trials={request.trials}, mode={request.mode}, health_weight={request.health_weight}, use_runpod_service={request.use_runpod_service}")
        
        # Create new job with the comprehensive request (no need to recreate)
        job = OptimizationJob(request, self.tensorboard_manager)
        self.jobs[job.job_id] = job
        
        # Start optimization in background
        background_tasks.add_task(job.start)
        
        logger.debug(f"running _start_optimization ... Created job {job.job_id} with corrected epoch config")
        return job.get_status()
    
    async def _get_job_status(self, job_id: str) -> JobResponse:
        """
        Get current status of an optimization job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            JobResponse with current job status
            
        Raises:
            HTTPException: If job not found
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._get_job_status ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        job_status = job.get_status()
    
        # Enhance job status with actual optimization data when available
        if job_status.status == JobStatus.COMPLETED and job_status.result:
            try:
                # Extract real optimization results for accurate reporting
                optimization_result = job_status.result.get("optimization_result", {})
                
                # FIXED: Update progress with actual results
                if job_status.progress is None:
                    job_status.progress = {}
                
                # Use actual results to populate progress data
                job_status.progress.update({
                    "total_trials": optimization_result.get("total_trials", 0),
                    "completed_trials": optimization_result.get("successful_trials", 0),
                    "best_total_score": optimization_result.get("best_total_score", 0.0),
                    "success_rate": (
                        optimization_result.get("successful_trials", 0) / 
                        max(optimization_result.get("total_trials", 1), 1)
                    ),
                    "status_message": "Optimization completed successfully"
                })
                
                # FIXED: Also update the main result best_total_score for monitor display
                if "best_total_score" in optimization_result:
                    # Ensure the job result reflects the actual best value
                    if isinstance(job_status.result, dict):
                        job_status.result["best_total_score"] = optimization_result["best_total_score"]
                
                logger.debug(f"running OptimizationAPI._get_job_status ... "
                            f"Enhanced completed job status: best_total_score={optimization_result.get('best_total_score', 'N/A')}")
                
            except Exception as e:
                logger.warning(f"running OptimizationAPI._get_job_status ... "
                            f"Failed to enhance job status with optimization results: {e}")
        
        return job_status
    
    async def _get_comprehensive_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get comprehensive status combining job progress, trials, and elapsed time
        
        This unified endpoint eliminates the need for multiple polling requests
        by combining data from job status, trial history, and elapsed time calculation.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary containing:
            - job_status: Current job status and progress
            - trials: Complete trial history
            - elapsed_seconds: Server-calculated elapsed time
            - is_complete: Boolean indicating if optimization finished
            
        Raises:
            HTTPException: If job not found
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._get_comprehensive_status ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        
        # Get job status (reuse existing method logic)
        job_status = job.get_status()
        
        # Calculate elapsed time on server side
        elapsed_seconds = 0
        if job.started_at:
            start_time = datetime.fromisoformat(job.started_at.replace('Z', '+00:00'))
            elapsed_seconds = int((datetime.now() - start_time).total_seconds())
        
        # Get trial history (reuse existing method logic)
        trial_history = []
        if hasattr(job, 'optimizer') and job.optimizer:
            trial_history = job.get_trial_history()
        
        # Determine completion status
        is_complete = job_status.status in ['completed', 'failed', 'cancelled']
        
        return {
            "job_id": job_id,
            "job_status": job_status.dict(),
            "trials": trial_history,
            "elapsed_seconds": elapsed_seconds,
            "is_complete": is_complete,
            "total_trials": len(trial_history),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _list_jobs(self) -> Dict[str, Any]:
        """
        List all optimization jobs with their current status
        
        Returns:
            Dictionary containing all jobs and summary statistics
        """
        logger.debug(f"running OptimizationAPI._list_jobs ... Listing {len(self.jobs)} jobs")
        
        jobs_list = []
        status_counts = {}
        
        for job in self.jobs.values():
            job_status = job.get_status()
            jobs_list.append(job_status.dict())
            
            # Count statuses
            status = job_status.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "jobs": jobs_list,
            "total_jobs": len(self.jobs),
            "status_counts": status_counts
        }
    
    async def _cancel_job(self, job_id: str) -> Dict[str, str]:
        """
        Cancel a running optimization job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Confirmation message
            
        Raises:
            HTTPException: If job not found or not cancellable
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._cancel_job ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        
        try:
            await job.cancel()
            logger.debug(f"running OptimizationAPI._cancel_job ... Job {job_id} cancelled successfully")
            return {"message": f"Job {job_id} cancelled successfully"}
        except RuntimeError as e:
            logger.error(f"running OptimizationAPI._cancel_job ... Cannot cancel job {job_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    
    async def _get_results(self, job_id: str) -> Dict[str, Any]:
        """
        Get detailed results from completed optimization job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Complete optimization results
            
        Raises:
            HTTPException: If job not found or not completed
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._get_results ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        
        if job.status != JobStatus.COMPLETED:
            logger.error(f"running OptimizationAPI._get_results ... Job {job_id} not completed: {job.status}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} is not completed. Status: {job.status}"
            )
        
        logger.debug(f"running OptimizationAPI._get_results ... Returning results for job {job_id}")
        return job.result or {}
    
    async def _download_model(self, job_id: str) -> FileResponse:
        """
        Download the trained model and metadata from completed optimization job
        
        ENHANCED: Creates ZIP archive containing:
        - .keras model file
        - best_hyperparameters.yaml metadata file
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            FileResponse with ZIP archive containing model and metadata
            
        Raises:
            HTTPException: If job not found, not completed, or model not available
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._download_model ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        
        if job.status != JobStatus.COMPLETED:
            logger.error(f"running OptimizationAPI._download_model ... Job {job_id} not completed: {job.status}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} is not completed. Status: {job.status}"
            )
        
        # Get model path from results
        if not job.result or not job.result.get("model_result"):
            logger.error(f"running OptimizationAPI._download_model ... No model available for job {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No trained model available for job {job_id}"
            )
        
        model_path = job.result["model_result"].get("model_path")
        if not model_path or not Path(model_path).exists():
            logger.error(f"running OptimizationAPI._download_model ... Model file not found: {model_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model file not found for job {job_id}"
            )
        
        try:
            # Create ZIP archive with model and metadata
            return await self._create_model_download_archive(job_id, model_path)
            
        except Exception as e:
            logger.error(f"running OptimizationAPI._download_model ... Failed to create download archive: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to prepare model download: {str(e)}"
            )
    
    async def _create_model_download_archive(self, job_id: str, model_path: str) -> FileResponse:
        """
        Create ZIP archive containing model file and metadata
        
        Args:
            job_id: Job identifier for naming and metadata lookup
            model_path: Path to the .keras model file
            
        Returns:
            FileResponse with ZIP archive
        """
        import zipfile
        import tempfile
        from datetime import datetime
        
        model_path_obj = Path(model_path)
        
        # Determine the hyperparameters YAML file path
        # The YAML file is in the same directory as the model
        yaml_path = model_path_obj.parent / "best_hyperparameters.yaml"
        
        logger.debug(f"running _create_model_download_archive ... Model path: {model_path}")
        logger.debug(f"running _create_model_download_archive ... YAML path: {yaml_path}")
        
        # Create temporary ZIP file
        temp_dir = Path(tempfile.gettempdir())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"optimized_model_{job_id}_{timestamp}.zip"
        temp_zip_path = temp_dir / zip_filename
        
        try:
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add the .keras model file
                if model_path_obj.exists():
                    zip_file.write(model_path_obj, model_path_obj.name)
                    logger.debug(f"running _create_model_download_archive ... Added model file: {model_path_obj.name}")
                else:
                    logger.warning(f"running _create_model_download_archive ... Model file not found: {model_path}")
                
                # Add the hyperparameters YAML file
                if yaml_path.exists():
                    zip_file.write(yaml_path, yaml_path.name)
                    logger.debug(f"running _create_model_download_archive ... Added YAML file: {yaml_path.name}")
                else:
                    logger.warning(f"running _create_model_download_archive ... YAML file not found: {yaml_path}")
                    # Create a minimal YAML file if it doesn't exist
                    minimal_yaml_content = f"""# Optimization Metadata
dataset: "unknown"
optimization_mode: "unknown"
job_id: "{job_id}"
note: "Original hyperparameters file not found"
generated_at: "{datetime.now().isoformat()}"
"""
                    yaml_in_zip = "best_hyperparameters.yaml"
                    zip_file.writestr(yaml_in_zip, minimal_yaml_content)
                    logger.debug(f"running _create_model_download_archive ... Created minimal YAML file in archive")
                
                # Add a README file for user guidance
                readme_content = f"""# Optimized Model Package

This archive contains your optimized neural network model and configuration.

## Contents

1. **{model_path_obj.name}** - Your trained TensorFlow/Keras model
   - Load with: `tensorflow.keras.models.load_model('{model_path_obj.name}')`
   - Ready for inference and deployment

2. **best_hyperparameters.yaml** - Optimization metadata
   - Contains the hyperparameters that achieved the best performance
   - Includes dataset info, optimization settings, and performance scores
   - Use these settings to reproduce or fine-tune your model

## Usage Example

```python
import tensorflow as tf
import yaml

# Load the trained model
model = tf.keras.models.load_model('{model_path_obj.name}')

# Load the hyperparameters
with open('best_hyperparameters.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Model loaded successfully!")
print(f"Dataset: {{config['dataset']}}")
print(f"Best score: {{config['best_total_score']:.4f}}")

# Use model for predictions
# predictions = model.predict(your_data)
```

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Job ID: {job_id}
"""
                zip_file.writestr("README.md", readme_content)
                logger.debug(f"running _create_model_download_archive ... Added README.md")
            
            logger.info(f"running _create_model_download_archive ... Created download archive: {temp_zip_path}")
            
            return FileResponse(
                path=str(temp_zip_path),
                media_type="application/zip",
                filename=zip_filename,
                headers={
                    "Content-Disposition": f"attachment; filename=\"{zip_filename}\"",
                    "Content-Type": "application/zip",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
            
        except Exception as e:
            # Clean up temp file if creation failed
            if temp_zip_path.exists():
                temp_zip_path.unlink()
            raise e
    
    async def _get_trial_history(self, job_id: str) -> Dict[str, Any]:
        """
        Get complete trial history for visualization
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with trial history data
            
        Raises:
            HTTPException: If job not found
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._get_trial_history ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        
        # Check if optimizer is available
        if not hasattr(job, 'optimizer') or not job.optimizer:
            logger.warning(f"running OptimizationAPI._get_trial_history ... No optimizer instance for job {job_id}")
            return {
                "job_id": job_id,
                "trials": [],
                "total_trials": 0
            }
        
        trial_history = job.get_trial_history()
        
        return {
            "job_id": job_id,
            "trials": trial_history,
            "total_trials": len(trial_history)
        }
    
    async def _get_current_trial(self, job_id: str) -> Dict[str, Any]:
        """
        Get currently running trial data
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with current trial data
            
        Raises:
            HTTPException: If job not found
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._get_current_trial ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        current_trial = job.get_current_trial()
        
        return {
            "job_id": job_id,
            "current_trial": current_trial
        }
    
    async def _get_best_trial(self, job_id: str) -> Dict[str, Any]:
        """
        Get best performing trial so far
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with best trial data
            
        Raises:
            HTTPException: If job not found
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._get_best_trial ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        best_trial = job.get_best_trial()
        
        return {
            "job_id": job_id,
            "best_trial": best_trial
        }
    
    async def _get_best_model_visualization(self, job_id: str) -> Dict[str, Any]:
        """
        Get best model data with 3D visualization information
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with best model data and 3D visualization information
            
        Raises:
            HTTPException: If job not found or no completed trials
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._get_best_model_visualization ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        
        # Get the 3D visualization data from the optimizer
        if not hasattr(job, 'optimizer') or not job.optimizer:
            logger.error(f"running OptimizationAPI._get_best_model_visualization ... No optimizer instance for job {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No optimizer instance found for job {job_id}"
            )
        
        visualization_data = job.optimizer.get_best_model_visualization_data()
        
        if not visualization_data:
            logger.warning(f"running OptimizationAPI._get_best_model_visualization ... No visualization data available for job {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No completed trials yet for job {job_id}"
            )
        
        logger.debug(f"running OptimizationAPI._get_best_model_visualization ... Returning 3D visualization data for job {job_id}, trial {visualization_data.get('trial_number')}")
        
        return {
            "job_id": job_id,
            **visualization_data
        }
    
    async def _download_best_model_visualization(self, job_id: str) -> FileResponse:
        """
        Download 3D visualization data as JSON file
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            FileResponse with JSON file containing visualization data
            
        Raises:
            HTTPException: If job not found or no completed trials
        """
        # Get the visualization data (reuse existing method)
        visualization_data = await self._get_best_model_visualization(job_id)
        
        # Create temporary file for download
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trial_number = visualization_data.get('trial_number', 'unknown')
        filename = f"best_model_visualization_job_{job_id}_trial_{trial_number}_{timestamp}.json"
        
        # Create temporary directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "model_visualizations"
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / filename
        
        try:
            # Write visualization data to temporary file
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(visualization_data, f, indent=2, default=str)
            
            logger.info(f"running OptimizationAPI._download_best_model_visualization ... Created download file: {temp_file_path}")
            
            return FileResponse(
                path=str(temp_file_path),
                filename=filename,
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Cache-Control": "no-cache"
                }
            )
            
        except Exception as e:
            logger.error(f"running OptimizationAPI._download_best_model_visualization ... Failed to create download file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to prepare visualization download: {str(e)}"
            )
    
    async def _get_cytoscape_architecture(self, job_id: str, trial_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get Cytoscape.js architecture data for visualization
        
        Args:
            job_id: Unique job identifier
            trial_id: Optional specific trial ID, defaults to best trial
            
        Returns:
            Dictionary containing Cytoscape.js nodes and edges data
            
        Raises:
            HTTPException: If job not found or no completed trials
        """
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        try:
            # Get the best trial or specific trial
            if trial_id:
                # Check if optimizer is available
                if not hasattr(job, 'optimizer') or not job.optimizer:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No optimizer instance found for job {job_id}"
                    )
                
                # Find specific trial using trial history
                trial_history = job.optimizer.get_trial_history()
                target_trial = None
                for trial_dict in trial_history:
                    trial_number = trial_dict.get('trial_number')
                    if trial_number is not None and str(trial_number) == trial_id:
                        target_trial = trial_dict
                        break
                
                if not target_trial:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Trial {trial_id} not found in job {job_id}"
                    )
                
                best_trial = target_trial
            else:
                # Check if optimizer is available
                if not hasattr(job, 'optimizer') or not job.optimizer:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No optimizer instance found for job {job_id}"
                    )
                
                # Get best trial
                best_trial_result = job.optimizer.get_best_model_visualization_data()
                if not best_trial_result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No completed trials found for job {job_id}"
                    )
                best_trial = best_trial_result.get('trial_data')
            
            if not best_trial:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No trial data available"
                )
            
            # Check if trial has architecture data
            if isinstance(best_trial, dict):
                architecture_data = best_trial.get('architecture')
                health_metrics = best_trial.get('health_metrics')
                # For dictionary-based trials, performance data is often in the trial itself
                performance_data = best_trial.get('performance') or best_trial
            else:
                # Fallback for object-based trials
                architecture_data = getattr(best_trial, 'architecture', None)
                health_metrics = getattr(best_trial, 'health_metrics', None)
                performance_data = getattr(best_trial, 'performance', None)
                
            if not architecture_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No architecture data available for trial"
                )
            
            # Get performance and health scores
            performance_score = 0.0
            health_score = None
            
            if isinstance(performance_data, dict):
                performance_score = performance_data.get('total_score', 0.0)
            elif performance_data and hasattr(performance_data, 'total_score'):
                performance_score = performance_data.total_score
            
            if isinstance(health_metrics, dict):
                health_score = health_metrics.get('overall_health')
            elif health_metrics and hasattr(health_metrics, 'overall_health'):
                health_score = health_metrics.overall_health
            
            # Import and use ModelVisualizer
            from model_visualizer import create_model_visualizer
            
            visualizer = create_model_visualizer()
            
            # Prepare the architecture visualization
            arch_viz = visualizer.prepare_visualization_data(
                architecture_data, 
                performance_score, 
                health_score
            )
            
            # Convert to Cytoscape format
            cytoscape_data = visualizer.export_cytoscape_architecture(arch_viz)
            
            # Add additional metadata
            if isinstance(best_trial, dict):
                trial_number = best_trial.get('trial_number')
                trial_status = best_trial.get('status', 'unknown')
            else:
                trial_number = getattr(best_trial, 'number', None)
                trial_status = getattr(best_trial, 'status', 'unknown')
                
            result = {
                "job_id": job_id,
                "trial_id": trial_id or trial_number or 'best',
                "cytoscape_data": cytoscape_data,
                "generated_at": datetime.utcnow().isoformat(),
                "trial_info": {
                    "trial_number": trial_number,
                    "status": trial_status,
                    "performance_score": performance_score,
                    "health_score": health_score
                }
            }
            
            logger.info(f"Generated Cytoscape architecture data for job {job_id}, trial {trial_id or 'best'}")
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to generate Cytoscape architecture data for job {job_id}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate Cytoscape data: {str(e)}"
            )
    
    async def _get_trends(self, job_id: str) -> Dict[str, Any]:
        """
        Get architecture and health trends for visualization
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with trend data
            
        Raises:
            HTTPException: If job not found
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._get_trends ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        architecture_trends = job.get_architecture_trends()
        health_trends = job.get_health_trends()
        
        return {
            "job_id": job_id,
            "architecture_trends": architecture_trends,
            "health_trends": health_trends
        }

    async def _list_job_files(self, job_id: str) -> Dict[str, Any]:
        """
        List all files in a job's results directory
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with file listing and metadata
            
        Raises:
            HTTPException: If job not found or results directory doesn't exist
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._list_job_files ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        
        # Find results directory
        results_dir = self._get_job_results_directory(job)
        if not results_dir or not results_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Results directory not found for job {job_id}"
            )
        
        # Collect all files recursively
        files = []
        total_size = 0
        
        for file_path in results_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(results_dir)
                file_size = file_path.stat().st_size
                total_size += file_size
                
                files.append({
                    "path": str(relative_path),
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        # Sort by path for consistent ordering
        files.sort(key=lambda x: x["path"])
        
        logger.debug(f"running OptimizationAPI._list_job_files ... Found {len(files)} files for job {job_id}")
        
        return {
            "job_id": job_id,
            "results_directory": str(results_dir),
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": files
        }  
    
    async def _download_job_file(self, job_id: str, file_path: str) -> FileResponse:
        """
        Download a specific file from job results
        
        Args:
            job_id: Unique job identifier
            file_path: Relative path to file within results directory
            
        Returns:
            FileResponse with the requested file
            
        Raises:
            HTTPException: If job/file not found or path is invalid
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._download_job_file ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        
        # Find results directory
        results_dir = self._get_job_results_directory(job)
        if not results_dir or not results_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Results directory not found for job {job_id}"
            )
        
        # Validate and resolve file path
        try:
            # Prevent path traversal attacks
            safe_file_path = results_dir / file_path
            safe_file_path = safe_file_path.resolve()
            
            # Ensure the resolved path is still within results directory
            if not str(safe_file_path).startswith(str(results_dir.resolve())):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid file path"
                )
            
            if not safe_file_path.exists() or not safe_file_path.is_file():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"File not found: {file_path}"
                )
            
        except Exception as e:
            logger.error(f"running OptimizationAPI._download_job_file ... Path resolution error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file path: {file_path}"
            )
        
        logger.debug(f"running OptimizationAPI._download_job_file ... Serving file: {safe_file_path}")
        
        # Determine appropriate media type
        media_type = "application/octet-stream"
        if safe_file_path.suffix.lower() in ['.json', '.txt', '.md', '.yaml', '.yml']:
            media_type = "text/plain"
        elif safe_file_path.suffix.lower() in ['.html']:
            media_type = "text/html"
        elif safe_file_path.suffix.lower() in ['.csv']:
            media_type = "text/csv"
        elif safe_file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            media_type = f"image/{safe_file_path.suffix[1:]}"
        
        return FileResponse(
            path=str(safe_file_path),
            media_type=media_type,
            filename=safe_file_path.name
        )

    async def _download_job_results_zip(self, job_id: str) -> FileResponse:
        """
        Download entire job results directory as ZIP
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            FileResponse with ZIP file containing all results
            
        Raises:
            HTTPException: If job not found or results directory doesn't exist
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._download_job_results_zip ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        
        # Find results directory
        results_dir = self._get_job_results_directory(job)
        if not results_dir or not results_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Results directory not found for job {job_id}"
            )
        
        # Create temporary ZIP file
        try:
            # Create temporary file for ZIP
            temp_dir = Path(tempfile.gettempdir())
            zip_filename = f"optimization_results_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = temp_dir / zip_filename
            
            logger.debug(f"running OptimizationAPI._download_job_results_zip ... Creating ZIP: {zip_path}")
            
            # Create ZIP file with all results
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in results_dir.rglob("*"):
                    if file_path.is_file():
                        # Use the results directory name as the root in the ZIP
                        arcname = results_dir.name / file_path.relative_to(results_dir)
                        zipf.write(file_path, arcname)
                        logger.debug(f"running OptimizationAPI._download_job_results_zip ... Added to ZIP: {arcname}")
            
            logger.debug(f"running OptimizationAPI._download_job_results_zip ... ZIP created successfully: {zip_path}")
            
            return FileResponse(
                path=str(zip_path),
                media_type="application/zip",
                filename=zip_filename,
                headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
            )
            
        except Exception as e:
            logger.error(f"running OptimizationAPI._download_job_results_zip ... Failed to create ZIP: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create results archive: {str(e)}"
            )

    async def _stop_all_jobs(self) -> Dict[str, Any]:
        """
        Stop all running optimization jobs
        
        Returns:
            Dictionary with operation results
        """
        logger.debug("running OptimizationAPI._stop_all_jobs ... Stopping all running jobs")
        
        running_jobs = [job for job in self.jobs.values() if job.status == JobStatus.RUNNING]
        
        if not running_jobs:
            return {
                "message": "No running jobs to stop",
                "stopped_jobs": [],
                "total_stopped": 0
            }
        
        stopped_jobs = []
        failed_stops = []
        
        for job in running_jobs:
            try:
                await job.cancel()
                stopped_jobs.append({
                    "job_id": job.job_id,
                    "dataset": job.request.dataset_name,
                    "mode": job.request.mode,
                    "stopped_at": datetime.now().isoformat()
                })
                logger.debug(f"running OptimizationAPI._stop_all_jobs ... Stopped job {job.job_id}")
            except Exception as e:
                failed_stops.append({
                    "job_id": job.job_id,
                    "error": str(e)
                })
                logger.error(f"running OptimizationAPI._stop_all_jobs ... Failed to stop job {job.job_id}: {e}")
        
        result = {
            "message": f"Stopped {len(stopped_jobs)} jobs",
            "stopped_jobs": stopped_jobs,
            "total_stopped": len(stopped_jobs),
            "failed_stops": failed_stops
        }
        
        if failed_stops:
            result["message"] += f", {len(failed_stops)} failed to stop"
        
        return result

    async def _stop_job(self, job_id: str) -> Dict[str, str]:
        """
        Stop a specific running optimization job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Confirmation message
            
        Raises:
            HTTPException: If job not found or not running
        """
        if job_id not in self.jobs:
            logger.error(f"running OptimizationAPI._stop_job ... Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = self.jobs[job_id]
        
        if job.status != JobStatus.RUNNING:
            logger.error(f"running OptimizationAPI._stop_job ... Job {job_id} not running: {job.status}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} is not running. Current status: {job.status}"
            )
        
        try:
            await job.cancel()
            logger.debug(f"running OptimizationAPI._stop_job ... Job {job_id} stopped successfully")
            return {
                "message": f"Job {job_id} stopped successfully",
                "job_id": job_id,
                "stopped_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"running OptimizationAPI._stop_job ... Failed to stop job {job_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to stop job {job_id}: {str(e)}"
            )

    def _get_job_results_directory(self, job: 'OptimizationJob') -> Optional[Path]:
        """
        Get the results directory path for a job
        
        Args:
            job: OptimizationJob instance
            
        Returns:
            Path to results directory or None if not available
        """
        try:
            # Check if job has completed results with directory path
            if job.result and job.result.get("model_result"):
                results_dir_str = job.result["model_result"].get("results_dir")
                if results_dir_str:
                    results_dir = Path(results_dir_str)
                    if results_dir.exists():
                        return results_dir
            
            # Fallback: Search for directory by pattern
            optimization_results_dir = Path(os.getenv("OPTIMIZATION_RESULTS_DIR", "/app/optimization_results"))
            if not optimization_results_dir.exists():
                optimization_results_dir = Path(os.getenv("OPTIMIZATION_RESULTS_FALLBACK_DIR", "./optimization_results"))
            
            if optimization_results_dir.exists():
                # Look for directories containing the dataset name and mode
                dataset_name = job.request.dataset_name
                mode = job.request.mode
                
                # Search for matching directory pattern
                for result_dir in optimization_results_dir.iterdir():
                    if (result_dir.is_dir() and 
                        dataset_name in result_dir.name and 
                        mode in result_dir.name):
                        logger.debug(f"running OptimizationAPI._get_job_results_directory ... Found results directory: {result_dir}")
                        return result_dir
            
            logger.warning(f"running OptimizationAPI._get_job_results_directory ... No results directory found for job {job.job_id}")
            return None
            
        except Exception as e:
            logger.error(f"running OptimizationAPI._get_job_results_directory ... Error finding results directory: {e}")
            return None

    async def _get_tensorboard_logs(self, job_id: str) -> Dict[str, Any]:
        """
        Get available TensorBoard log directories for a job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with available log directories and trial information
            
        Note:
            Works with any job_id as long as TensorBoard logs exist
        """
        # Remove job existence check - allow any job_id if logs exist
        
        try:
            tensorboard_dir = Path("tensorboard_logs")
            log_directories = []
            
            if tensorboard_dir.exists():
                # Find the most recent run directory (new namespaced structure only)
                run_dirs = [d for d in tensorboard_dir.iterdir() if d.is_dir()]
                
                if run_dirs:
                    # Sort by modification time to get the most recent run
                    latest_run_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
                    
                    # Find all trial directories within the latest run
                    trial_dirs = [d for d in latest_run_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")]
                else:
                    trial_dirs = []
                
                for trial_dir in trial_dirs:
                    trial_info = {
                        "trial_directory": str(trial_dir),
                        "trial_name": trial_dir.name,
                        "log_files": []
                    }
                    
                    # Check for log files in this trial
                    if trial_dir.exists():
                        for log_file in trial_dir.rglob("*"):
                            if log_file.is_file():
                                trial_info["log_files"].append({
                                    "file_path": str(log_file),
                                    "file_name": log_file.name,
                                    "size_bytes": log_file.stat().st_size,
                                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                                })
                    
                    log_directories.append(trial_info)
            
            return {
                "job_id": job_id,
                "tensorboard_logs": log_directories,
                "total_trials": len(log_directories),
                "base_log_directory": str(tensorboard_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to get TensorBoard logs for job {job_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve TensorBoard logs: {str(e)}"
            )
    
    async def _get_tensorboard_url(self, job_id: str) -> Dict[str, Any]:
        """
        Get TensorBoard server URL for a job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with TensorBoard URL and status
            
        Note:
            Works with any job_id for TensorBoard server management
        """
        # Remove job existence check - allow any job_id for TensorBoard server management
        
        # Get server status from TensorBoard manager
        return self.tensorboard_manager.get_server_status(job_id)
    
    async def _start_tensorboard_server(self, job_id: str) -> Dict[str, Any]:
        """
        Start TensorBoard server for a job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with server status and URL
            
        Raises:
            HTTPException: If TensorBoard logs not found or server cannot be started
        """
        # Remove job existence check - only check if TensorBoard logs exist
        
        try:
            tensorboard_base_dir = Path("tensorboard_logs")
            
            if not tensorboard_base_dir.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No TensorBoard logs found for this job"
                )
            
            # Find the most recent run directory (new namespaced structure only)
            run_dirs = [d for d in tensorboard_base_dir.iterdir() if d.is_dir()]
            
            if not run_dirs:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No run directories found in TensorBoard logs"
                )
            
            # Sort by modification time to get the most recent run
            latest_run_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
            tensorboard_dir = latest_run_dir
            
            # Check if there are any trial directories in the chosen directory
            trial_dirs = [d for d in tensorboard_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")]
            if not trial_dirs:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No trial directories found in TensorBoard logs"
                )
            
            # Use TensorBoard manager to start server pointing to the correct directory
            return self.tensorboard_manager.start_server(job_id, tensorboard_dir)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to start TensorBoard server for job {job_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start TensorBoard server: {str(e)}"
            )
    
    async def _stop_tensorboard_server(self, job_id: str) -> Dict[str, Any]:
        """
        Stop TensorBoard server for a job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with stop status
            
        Note:
            Works with any job_id for TensorBoard server management
        """
        # Remove job existence check - allow any job_id for TensorBoard server management
        
        # Use TensorBoard manager to stop server
        return self.tensorboard_manager.stop_server(job_id)
    
    async def _get_trial_plot(self, job_id: str, trial_id: str, plot_type: str):
        """
        Serve a specific training visualization plot for a trial
        
        Args:
            job_id: Unique job identifier
            trial_id: Trial number (e.g., "0", "1", "2")
            plot_type: Plot type ("training_history", "weights_bias", "gradient_magnitudes", "gradient_distributions", "dead_neuron_analysis", "training_progress", "activation_maps", "activation_summary", "confusion_matrix", "training_animation")
            
        Returns:
            File response with the plot image
        """
        try:
            # Validate job exists
            if job_id not in self.jobs:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job {job_id} not found"
                )
            
            job = self.jobs[job_id]
            
            # Determine plot directory from job's optimizer results
            if not hasattr(job, 'optimizer') or not job.optimizer:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No optimizer instance found for job {job_id}"
                )
            
            results_dir = job.optimizer.results_dir
            plot_dir = results_dir / "plots" / f"trial_{trial_id}"
            
            # Map plot types to filename patterns (scan for actual files)
            plot_patterns = {
                "training_history": ["training_history*", "training_progress*", "training_*dashboard*"],
                "weights_bias": ["weights_bias*", "*weights*bias*"], 
                "gradient_magnitudes": ["gradient_magnitudes*"],
                "gradient_distributions": ["gradient_distributions*"],
                "dead_neuron_analysis": ["*dead_neuron*"],
                "training_progress": ["training_progress*"],
                "activation_maps": ["activation_maps*", "activation_comparison*"],
                "activation_progression": ["activation_progression*"],
                "activation_summary": ["activation_summary*"],
                "confusion_matrix": ["confusion_matrix*"],
                "training_animation": ["training_animation*"]
            }
            
            if plot_type not in plot_patterns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid plot type. Available types: {list(plot_patterns.keys())}"
                )
            
            # Find the first matching file for this plot type
            plot_file = None
            for pattern in plot_patterns[plot_type]:
                matching_files = list(plot_dir.glob(f"{pattern}.png"))
                if matching_files:
                    plot_file = matching_files[0]  # Use the first match
                    break
            
            if not plot_file or not plot_file.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Plot {plot_type} not found for trial {trial_id}"
                )
            
            from fastapi.responses import FileResponse
            return FileResponse(
                path=str(plot_file),
                media_type="image/png",
                filename=f"{job_id}_trial_{trial_id}_{plot_type}.png"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"running _get_trial_plot ... Error serving plot: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to serve plot: {str(e)}"
            )
    
    async def _list_trial_plots(self, job_id: str, trial_id: str) -> Dict[str, Any]:
        """
        List available plots for a specific trial
        
        Args:
            job_id: Unique job identifier
            trial_id: Trial number
            
        Returns:
            Dictionary with available plots and metadata
        """
        try:
            # Validate job exists
            if job_id not in self.jobs:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job {job_id} not found"
                )
            
            job = self.jobs[job_id]
            
            # Get plot directory
            if not hasattr(job, 'optimizer') or not job.optimizer:
                return {"plots": [], "message": "No optimizer instance found"}
            
            results_dir = job.optimizer.results_dir
            plot_dir = results_dir / "plots" / f"trial_{trial_id}"
            
            if not plot_dir.exists():
                return {"plots": [], "message": f"No plots found for trial {trial_id}"}
            
            # Check for actual plot files using pattern matching
            plot_patterns = {
                "training_history": ["training_history*", "training_progress*", "training_*dashboard*"],
                "weights_bias": ["weights_bias*", "*weights*bias*"], 
                "gradient_magnitudes": ["gradient_magnitudes*"],
                "gradient_distributions": ["gradient_distributions*"],
                "dead_neuron_analysis": ["*dead_neuron*"],
                "training_progress": ["training_progress*"],
                "activation_maps": ["activation_maps*", "activation_comparison*"],
                "activation_progression": ["activation_progression*"],
                "activation_summary": ["activation_summary*"],
                "confusion_matrix": ["confusion_matrix*"],
                "training_animation": ["training_animation*"]
            }
            
            available_plots = []
            for plot_type, patterns in plot_patterns.items():
                # Find the first matching file for this plot type
                plot_file = None
                for pattern in patterns:
                    matching_files = list(plot_dir.glob(f"{pattern}.png"))
                    if matching_files:
                        plot_file = matching_files[0]  # Use the first match
                        break
                
                if plot_file and plot_file.exists():
                    available_plots.append({
                        "plot_type": plot_type,
                        "filename": plot_file.name,
                        "url": f"/jobs/{job_id}/plots/{trial_id}/{plot_type}",
                        "size_bytes": plot_file.stat().st_size,
                        "created_time": plot_file.stat().st_mtime
                    })
            
            return {
                "job_id": job_id,
                "trial_id": trial_id,
                "plots": available_plots,
                "total_plots": len(available_plots)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"running _list_trial_plots ... Error listing plots: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list plots: {str(e)}"
            )
    
    async def _list_job_plots(self, job_id: str) -> Dict[str, Any]:
        """
        List available plots for all trials in a job
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Dictionary with plots organized by trial
        """
        try:
            # Validate job exists
            if job_id not in self.jobs:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Job {job_id} not found"
                )
            
            job = self.jobs[job_id]
            
            # Get plots directory
            if not hasattr(job, 'optimizer') or not job.optimizer:
                return {"trials": {}, "message": "No optimizer instance found"}
            
            results_dir = job.optimizer.results_dir
            plots_dir = results_dir / "plots"
            
            if not plots_dir.exists():
                return {"trials": {}, "message": "No plots directory found"}
            
            # Scan for trial directories
            trials_plots = {}
            for trial_dir in plots_dir.iterdir():
                if trial_dir.is_dir() and trial_dir.name.startswith("trial_"):
                    trial_id = trial_dir.name.replace("trial_", "")
                    trial_plots = await self._list_trial_plots(job_id, trial_id)
                    if trial_plots["plots"]:
                        trials_plots[trial_id] = trial_plots["plots"]
            
            return {
                "job_id": job_id,
                "trials": trials_plots,
                "total_trials": len(trials_plots),
                "total_plots": sum(len(plots) for plots in trials_plots.values())
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"running _list_job_plots ... Error listing job plots: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list job plots: {str(e)}"
            )
    


# Initialize the FastAPI application
api = OptimizationAPI()
app = api.app


if __name__ == "__main__":
    """
    Run the FastAPI server for development
    
    For production deployment, use:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
    """
    # Ensure logging is set up before server starts
    setup_logging()
    logger.info("FastAPI server starting - all logs will be written to logs/non-cron.log")
    logger.debug("running api_server.__main__ ... Starting FastAPI development server")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )