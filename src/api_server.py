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
from datetime import datetime
from enum import Enum
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
import json
import numpy as np
import os
from pathlib import Path
from pydantic import BaseModel, Field
import shutil
import tempfile
import traceback
from typing import Dict, Any, List, Optional, Union
import uuid
import uvicorn
import zipfile

# UPDATED IMPORTS - Fixed compatibility
from optimizer import optimize_model, OptimizationResult, OptimizationMode, OptimizationObjective, TrialProgress
from dataset_manager import DatasetManager
from utils.logger import logger


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


class OptimizationRequest(BaseModel):
    """
    Request model for starting a new hyperparameter optimization job
    
    UPDATED: Aligned with new optimizer.py parameters
    
    Attributes:
        dataset_name: Name of dataset to optimize (e.g., 'cifar10', 'imdb')
        mode: Optimization mode ('simple' or 'health')
        optimize_for: Optimization objective ('val_accuracy', 'accuracy', etc.)
        trials: Number of optimization trials to run
        health_weight: Health weighting (0.0-1.0, only used in health mode)
        config_overrides: Additional configuration parameters
        
    Example:
        {
            "dataset_name": "cifar10",
            "mode": "health",
            "optimize_for": "val_accuracy",
            "trials": 50,
            "health_weight": 0.3,
            "config_overrides": {
                "max_epochs_per_trial": 25,
                "n_startup_trials": 15
            }
        }
    """
    dataset_name: str = Field(..., description="Dataset name (e.g., 'cifar10', 'imdb')")
    mode: str = Field(default="simple", description="Optimization mode ('simple' or 'health')")
    optimize_for: str = Field(default="val_accuracy", description="Optimization objective")
    trials: int = Field(default=50, ge=1, le=200, description="Number of optimization trials")
    health_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Health weighting (health mode only)")
    config_overrides: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")


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
                "best_value": 0.8750,
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
    
    def __init__(self, request: OptimizationRequest):
        """
        Initialize a new optimization job with real-time progress tracking
        
        Args:
            request: OptimizationRequest containing job parameters
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
        
        # NEW: Real-time trial tracking
        self.optimizer: Optional[Any] = None  # Will be ModelOptimizer instance
        self.trial_progress_history: List[TrialProgress] = []
        self.current_trial_progress: Optional[TrialProgress] = None
        self.best_trial_progress: Optional[TrialProgress] = None
        
        logger.debug(f"running OptimizationJob.__init__ ... Created job {self.job_id} for dataset {request.dataset_name}")
        logger.debug(f"running OptimizationJob.__init__ ... Mode: {request.mode}, Objective: {request.optimize_for}")
        logger.debug(f"running OptimizationJob.__init__ ... Real-time trial tracking enabled")
    
    def _progress_callback(self, trial_progress: TrialProgress) -> None:
        """
        Callback function to receive real-time trial progress updates
        
        This method is called by the optimizer whenever trial progress is updated.
        It maintains the job's trial tracking state and updates progress metrics.
        
        Args:
            trial_progress: TrialProgress object with current trial state
        """
        logger.debug(f"running OptimizationJob._progress_callback ... Trial {trial_progress.trial_number} update: {trial_progress.status}")
        
        # Store trial progress
        self.trial_progress_history.append(trial_progress)
        
        # Update current trial
        if trial_progress.status == "running":
            self.current_trial_progress = trial_progress
        elif trial_progress.status in ["completed", "failed", "pruned"]:
            # Clear current trial when completed
            if self.current_trial_progress and self.current_trial_progress.trial_id == trial_progress.trial_id:
                self.current_trial_progress = None
            
            # Update best trial if this one performed better
            if (trial_progress.status == "completed" and 
                trial_progress.performance and 
                trial_progress.performance.get('final_val_accuracy')):
                
                current_val_acc = trial_progress.performance['final_val_accuracy']
                best_val_acc = (self.best_trial_progress.performance.get('final_val_accuracy', 0) 
                              if self.best_trial_progress and self.best_trial_progress.performance else 0)
                
                if current_val_acc > best_val_acc:
                    self.best_trial_progress = trial_progress
                    logger.debug(f"running OptimizationJob._progress_callback ... New best trial: {trial_progress.trial_number} with val_acc: {current_val_acc:.4f}")
        
        # Update job progress
        self._update_job_progress()
    
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
                "best_value": opt_progress["best_value"],
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
                "best_value": None,
                "elapsed_time": 0,
                "status_message": "Initializing optimization..."
            }
            
            # Run optimization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Update progress: Starting optimization
            self.progress["status_message"] = "Loading dataset and initializing optimizer..."
            logger.debug(f"running OptimizationJob._run_optimization ... Loading dataset {self.request.dataset_name}")
            
            # Execute optimization
            result = await loop.run_in_executor(
                None,  # Use default thread pool
                self._execute_optimization
            )
            
            # Convert OptimizationResult to API format
            api_result = self._convert_optimization_result(result)
            
            # Store successful result
            self.result = api_result
            self.status = JobStatus.COMPLETED
            self.completed_at = datetime.now().isoformat()
            
            # Update final progress
            self.progress["status_message"] = "Optimization completed successfully"
            self.progress["current_trial"] = self.request.trials
            self.progress["best_value"] = result.best_value
            
            logger.debug(f"running OptimizationJob._run_optimization ... Job {self.job_id} completed successfully")
            logger.debug(f"running OptimizationJob._run_optimization ... Best value: {result.best_value:.4f}")
            
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
        
        # Apply config overrides
        config_overrides = self.request.config_overrides.copy()
        config_overrides['health_weight'] = self.request.health_weight
        
        logger.debug(f"running OptimizationJob._execute_optimization ... Using optimize_model function")
        logger.debug(f"running OptimizationJob._execute_optimization ... Config overrides: {config_overrides}")
        
        # Use the enhanced optimize_model function with progress callback support
        from optimizer import optimize_model
        
        result = optimize_model(
            dataset_name=self.request.dataset_name,
            mode=self.request.mode,
            optimize_for=self.request.optimize_for,
            trials=self.request.trials,
            run_name=None,  # Let optimize_model generate the run_name using its established logic
            progress_callback=self._progress_callback,  # Real-time progress updates
            **config_overrides
        )
        
        # Note: The optimizer instance is not directly accessible when using optimize_model
        # but that's okay since the progress_callback handles real-time updates
        self.optimizer = None  # Will be set by the optimize_model function if needed
        
        logger.debug(f"running OptimizationJob._execute_optimization ... Optimization completed for job {self.job_id}")
        logger.debug(f"running OptimizationJob._execute_optimization ... Results directory: {result.results_dir}")
        logger.debug(f"running OptimizationJob._execute_optimization ... Best value: {result.best_value:.4f}")
        
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
                    "best_value": result.best_value,
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
                    "test_accuracy": result.best_value if (result.optimization_config and "accuracy" in str(result.optimization_config.objective)) else None,
                    "results_dir": str(result.results_dir) if result.results_dir else None
                } if result.best_model_path else None,
                "run_name": self.job_id,
                "best_value": result.best_value,
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
                    "best_value": result.best_value,
                    "best_params": result.best_params,
                    "total_trials": result.total_trials,
                    "successful_trials": result.successful_trials,
                    "dataset_name": result.dataset_name,
                    "optimization_mode": result.optimization_mode
                },
                "run_name": self.job_id,
                "best_value": result.best_value,
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
            # Fallback to stored progress history
            return [trial.to_dict() for trial in self.trial_progress_history]
    
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
            return self.current_trial_progress.to_dict() if self.current_trial_progress else None
    
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
            return self.best_trial_progress.to_dict() if self.best_trial_progress else None
    
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
    
    def __init__(self):
        """
        Initialize the FastAPI application with all endpoints
        
        Sets up the FastAPI instance, configures CORS, initializes
        job storage, and registers all API endpoints.
        """
        self.app = FastAPI(
            title="Hyperparameter Optimization API",
            description="REST API for automated hyperparameter optimization with GPU acceleration",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Job storage - in production, use Redis or database
        self.jobs: Dict[str, OptimizationJob] = {}
        
        # Initialize dataset manager for validation
        self.dataset_manager = DatasetManager()
        
        # Register API endpoints
        self._register_routes()
        
        logger.debug("running OptimizationAPI.__init__ ... FastAPI application initialized")
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
        
        # Ensure epoch configuration is sane BEFORE creating the job
        config_overrides = request.config_overrides.copy()
        
        # Ensure the values for max and min epochs are used configuration issues
        max_epochs = config_overrides.get('max_epochs_per_trial', 20)  # Default 20
        min_epochs = 5  # Always use 3 as minimum for API requests
        
        # Ensure max_epochs is at least min_epochs
        if max_epochs < min_epochs:
            logger.warning(f"running _start_optimization ... max_epochs_per_trial ({max_epochs}) too low, setting to {min_epochs}")
            max_epochs = min_epochs
        
        # Update config overrides with corrected values
        config_overrides['max_epochs_per_trial'] = max_epochs
        config_overrides['min_epochs_per_trial'] = min_epochs
        
        logger.debug(f"running _start_optimization ... Using epoch configuration: min={min_epochs}, max={max_epochs}")
        
        # Create optimization request with fixed config
        fixed_request = OptimizationRequest(
            dataset_name=request.dataset_name,
            mode=request.mode,
            optimize_for=request.optimize_for,
            trials=request.trials,
            health_weight=request.health_weight,
            config_overrides=config_overrides  # Use the fixed config
        )
        
        # Create new job with the fixed request
        job = OptimizationJob(fixed_request)
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
                    "best_value": optimization_result.get("best_value", 0.0),
                    "success_rate": (
                        optimization_result.get("successful_trials", 0) / 
                        max(optimization_result.get("total_trials", 1), 1)
                    ),
                    "status_message": "Optimization completed successfully"
                })
                
                # FIXED: Also update the main result best_value for monitor display
                if "best_value" in optimization_result:
                    # Ensure the job result reflects the actual best value
                    if isinstance(job_status.result, dict):
                        job_status.result["best_value"] = optimization_result["best_value"]
                
                logger.debug(f"running OptimizationAPI._get_job_status ... "
                            f"Enhanced completed job status: best_value={optimization_result.get('best_value', 'N/A')}")
                
            except Exception as e:
                logger.warning(f"running OptimizationAPI._get_job_status ... "
                            f"Failed to enhance job status with optimization results: {e}")
        
        return job_status
    
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
        Download the trained model from completed optimization job
        
        UPDATED: Uses new result format
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            FileResponse with model file
            
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
        
        logger.debug(f"running OptimizationAPI._download_model ... Serving model file for job {job_id}: {model_path}")
        
        return FileResponse(
            path=model_path,
            media_type="application/octet-stream",
            filename=Path(model_path).name
        )
    
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
        
    


# Initialize the FastAPI application
api = OptimizationAPI()
app = api.app


if __name__ == "__main__":
    """
    Run the FastAPI server for development
    
    For production deployment, use:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
    """
    logger.debug("running api_server.__main__ ... Starting FastAPI development server")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )