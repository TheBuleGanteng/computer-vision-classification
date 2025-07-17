"""
FastAPI Server for Hyperparameter Optimization

Provides REST API endpoints for managing hyperparameter optimization jobs.
Integrates with existing model_optimizer system to provide:
- Asynchronous job management
- Progress monitoring
- Result retrieval
- Model download capabilities

Designed for deployment on RunPod with GPU acceleration and local development.
"""

import asyncio
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import traceback
from typing import Dict, Any, List, Optional, Union
import uuid
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import existing optimizer components
from model_optimizer import optimize_model
from model_builder import create_and_train_model
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
    
    Defines the structure and validation for optimization requests.
    All parameters correspond to those accepted by the underlying
    model_optimizer system.
    
    Attributes:
        dataset_name: Name of dataset to optimize (e.g., 'cifar10', 'imdb')
        optimizer: Optimizer type ('simple' or 'health')
        optimize_for: Optimization objective ('accuracy', 'val_accuracy', etc.)
        trials: Number of optimization trials to run
        create_model: Whether to build final model with best parameters
        config_overrides: Additional configuration parameters
        
    Example:
        {
            "dataset_name": "cifar10",
            "optimizer": "health",
            "optimize_for": "val_accuracy",
            "trials": 50,
            "create_model": true,
            "config_overrides": {
                "max_epochs_per_trial": 25,
                "n_startup_trials": 15
            }
        }
    """
    dataset_name: str = Field(..., description="Dataset name (e.g., 'cifar10', 'imdb')")
    optimizer: str = Field(default="health", description="Optimizer type ('simple' or 'health')")
    optimize_for: str = Field(default="accuracy", description="Optimization objective")
    trials: int = Field(default=50, ge=1, le=200, description="Number of optimization trials")
    create_model: bool = Field(default=True, description="Build final model with best parameters")
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
    
    Captures all information needed for rich visualization including
    model architecture, health metrics, performance data, and timing.
    
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
        gradient_health: Gradient flow health metrics
        
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
                "layers": [
                    {"type": "conv2d", "filters": 64, "kernel_size": [3,3], "params": 1792},
                    {"type": "maxpool", "pool_size": [2,2], "params": 0},
                    {"type": "dense", "units": 128, "params": 32896}
                ],
                "total_params": 245000,
                "memory_mb": 12.5
            },
            "health_metrics": {
                "dead_neuron_ratio": 0.15,
                "gradient_health": 0.82,
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
    gradient_health: Optional[Dict[str, Any]] = None
    
    # Performance Data
    performance: Optional[Dict[str, Any]] = None
    training_history: Optional[Dict[str, Any]] = None
    
    # Pruning/Early Stopping Information
    pruning_info: Optional[Dict[str, Any]] = None


class OptimizationJob:
    """
    Enhanced job management class for tracking optimization tasks with rich visualization data
    
    Handles the lifecycle of individual optimization jobs including:
    - Job state management
    - Detailed trial tracking with architecture and health metrics
    - Progress tracking with real-time updates
    - Result storage and retrieval
    - Error handling and logging
    
    This class bridges the FastAPI async world with the synchronous
    optimization code, providing clean separation of concerns while
    capturing comprehensive data for frontend visualization.
    
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
        
        # Enhanced tracking for visualization
        trials: List of all trial data for rich visualization
        current_trial: Currently running trial data
        trial_history: Complete trial history for analysis
        best_trial: Best performing trial so far
        architecture_trends: Architecture performance trends
        health_trends: Health metrics trends over time
        
    Example Usage:
        job = OptimizationJob(request)
        await job.start()
        status = job.get_status()
        trials = job.get_trial_history()
        current = job.get_current_trial()
    """
    
    def __init__(self, request: OptimizationRequest):
        """
        Initialize a new optimization job with enhanced trial tracking
        
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
        
        # Enhanced tracking for visualization
        self.trials: List[TrialData] = []
        self.current_trial: Optional[TrialData] = None
        self.trial_history: Dict[str, TrialData] = {}
        self.best_trial: Optional[TrialData] = None
        self.architecture_trends: Dict[str, List[float]] = {}
        self.health_trends: Dict[str, List[float]] = {}
        
        logger.debug(f"running OptimizationJob.__init__ ... Created job {self.job_id} for dataset {request.dataset_name}")
        logger.debug(f"running OptimizationJob.__init__ ... Enhanced tracking enabled for rich visualization")
    
    async def start(self) -> None:
        """
        Start the optimization job as a background task
        
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
        logger.debug(f"running OptimizationJob.start ... Optimizer: {self.request.optimizer}")
        logger.debug(f"running OptimizationJob.start ... Trials: {self.request.trials}")
        
        # Start the optimization task
        self.task = asyncio.create_task(self._run_optimization())
    
    async def _run_optimization(self) -> None:
        """
        Execute the optimization process in background
        
        Runs the actual hyperparameter optimization using the existing
        model_optimizer system. Updates progress and handles results.
        
        This method bridges async FastAPI with sync optimization code
        by running the optimization in a thread pool executor.
        
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
            
            # Store successful result
            self.result = result
            self.status = JobStatus.COMPLETED
            self.completed_at = datetime.now().isoformat()
            
            # Update final progress
            self.progress["status_message"] = "Optimization completed successfully"
            self.progress["current_trial"] = self.request.trials
            
            logger.debug(f"running OptimizationJob._run_optimization ... Job {self.job_id} completed successfully")
            logger.debug(f"running OptimizationJob._run_optimization ... Best value: {result.get('best_value', 'unknown')}")
            
        except Exception as e:
            # Handle any optimization errors
            self.error = str(e)
            self.status = JobStatus.FAILED
            self.completed_at = datetime.now().isoformat()
            
            if self.progress:
                self.progress["status_message"] = f"Optimization failed: {str(e)}"
            
            logger.error(f"running OptimizationJob._run_optimization ... Job {self.job_id} failed: {e}")
            logger.debug(f"running OptimizationJob._run_optimization ... Error traceback: {traceback.format_exc()}")
    
    def _execute_optimization(self) -> Dict[str, Any]:
        """
        Execute the actual optimization (synchronous)
        
        Calls the existing model_optimizer.optimize_model function with
        the job's parameters. This runs in a thread pool to avoid
        blocking the async event loop.
        
        Returns:
            Dictionary containing optimization results including:
            - optimization_result: OptimizationResult object
            - model_result: Model training results (if create_model=True)
            - run_name: Unique run identifier
            - best_value: Best optimization value achieved
            - best_params: Best hyperparameters found
            
        Raises:
            Exception: Any error from the optimization process
        """
        logger.debug(f"running OptimizationJob._execute_optimization ... Executing optimization for job {self.job_id}")
        
        # Call existing optimizer with job parameters
        result = optimize_model(
            dataset_name=self.request.dataset_name,
            optimizer=self.request.optimizer,
            optimize_for=self.request.optimize_for,
            trials=self.request.trials,
            create_model=self.request.create_model,
            **self.request.config_overrides
        )
        
        logger.debug(f"running OptimizationJob._execute_optimization ... Optimization completed for job {self.job_id}")
        
        # Convert OptimizationResult to serializable format
        serializable_result = {
            "optimization_result": {
                "best_value": result["best_value"],
                "best_params": result["best_params"],
                "total_trials": result["optimization_result"].total_trials,
                "successful_trials": result["optimization_result"].successful_trials,
                "optimization_time_hours": result["optimization_result"].optimization_time_hours,
                "parameter_importance": result["optimization_result"].parameter_importance,
                "dataset_name": result["optimization_result"].dataset_name
            },
            "model_result": result.get("model_result"),
            "run_name": result["run_name"],
            "best_value": result["best_value"],
            "best_params": result["best_params"]
        }
        
        return serializable_result
    
    def _start_trial(self, trial_number: int, hyperparameters: Dict[str, Any]) -> str:
        """
        Start tracking a new optimization trial
        
        Creates a new TrialData object and begins tracking the trial's
        architecture, health metrics, and performance data.
        
        Args:
            trial_number: Sequential trial number (1, 2, 3, ...)
            hyperparameters: Complete hyperparameters for this trial
            
        Returns:
            trial_id: Unique identifier for this trial
            
        Side Effects:
            - Creates new TrialData object
            - Adds to trials list and trial_history
            - Updates current_trial pointer
            - Logs trial start event
        """
        trial_id = f"trial_{trial_number}_{uuid.uuid4().hex[:8]}"
        
        trial_data = TrialData(
            trial_id=trial_id,
            trial_number=trial_number,
            status="running",
            started_at=datetime.now().isoformat(),
            hyperparameters=hyperparameters
        )
        
        self.trials.append(trial_data)
        self.trial_history[trial_id] = trial_data
        self.current_trial = trial_data
        
        logger.debug(f"running OptimizationJob._start_trial ... Started trial {trial_number} (ID: {trial_id})")
        logger.debug(f"running OptimizationJob._start_trial ... Hyperparameters: {self._format_hyperparameters(hyperparameters)}")
        
        return trial_id
    
    def _update_trial_architecture(self, trial_id: str, model: Any, hyperparameters: Dict[str, Any]) -> None:
        """
        Update trial with detailed architecture information
        
        Analyzes the built model and extracts comprehensive architecture
        details for visualization including layer structure, parameters,
        and memory usage.
        
        Args:
            trial_id: Unique trial identifier
            model: Built Keras model
            hyperparameters: Trial hyperparameters
            
        Side Effects:
            - Updates trial_data.architecture with detailed layer info
            - Updates trial_data.model_size with parameter counts
            - Logs architecture analysis
        """
        if trial_id not in self.trial_history:
            logger.warning(f"running OptimizationJob._update_trial_architecture ... Trial {trial_id} not found")
            return
        
        trial_data = self.trial_history[trial_id]
        
        try:
            # Extract architecture details
            architecture = self._analyze_model_architecture(model, hyperparameters)
            model_size = self._calculate_model_size(model)
            
            trial_data.architecture = architecture
            trial_data.model_size = model_size
            
            logger.debug(f"running OptimizationJob._update_trial_architecture ... Updated architecture for trial {trial_id}")
            logger.debug(f"running OptimizationJob._update_trial_architecture ... Total parameters: {model_size.get('total_params', 'unknown')}")
            
        except Exception as e:
            logger.warning(f"running OptimizationJob._update_trial_architecture ... Failed to analyze architecture for trial {trial_id}: {e}")
    
    def _update_trial_health(self, trial_id: str, model: Any, history: Any) -> None:
        """
        Update trial with health metrics and training stability data
        
        Analyzes model health using existing health metric functions
        and training stability indicators for visualization.
        
        Args:
            trial_id: Unique trial identifier
            model: Trained Keras model
            history: Training history object
            
        Side Effects:
            - Updates trial_data.health_metrics
            - Updates trial_data.training_stability
            - Updates trial_data.gradient_health
            - Updates health_trends for visualization
            - Logs health assessment
        """
        if trial_id not in self.trial_history:
            logger.warning(f"running OptimizationJob._update_trial_health ... Trial {trial_id} not found")
            return
        
        trial_data = self.trial_history[trial_id]
        
        try:
            # Calculate health metrics (integrate with existing health system)
            health_metrics = self._calculate_trial_health_metrics(model, history)
            training_stability = self._calculate_trial_training_stability(history)
            gradient_health = self._calculate_trial_gradient_health(model)
            
            trial_data.health_metrics = health_metrics
            trial_data.training_stability = training_stability
            trial_data.gradient_health = gradient_health
            
            # Update trends for visualization
            self._update_health_trends(trial_data.trial_number, health_metrics)
            
            logger.debug(f"running OptimizationJob._update_trial_health ... Updated health metrics for trial {trial_id}")
            logger.debug(f"running OptimizationJob._update_trial_health ... Dead neuron ratio: {health_metrics.get('dead_neuron_ratio', 'unknown')}")
            logger.debug(f"running OptimizationJob._update_trial_health ... Gradient health: {gradient_health.get('overall_health', 'unknown')}")
            
        except Exception as e:
            logger.warning(f"running OptimizationJob._update_trial_health ... Failed to analyze health for trial {trial_id}: {e}")
    
    def _complete_trial(self, trial_id: str, performance: Dict[str, Any], training_history: Dict[str, Any], 
                       status: str = "completed", pruning_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark trial as completed and store final results
        
        Finalizes trial data with performance metrics, training history,
        and pruning information. Updates best trial tracking.
        
        Args:
            trial_id: Unique trial identifier
            performance: Final performance metrics
            training_history: Complete training history
            status: Final trial status (completed, failed, pruned)
            pruning_info: Information about early stopping/pruning
            
        Side Effects:
            - Updates trial completion timestamp and duration
            - Stores final performance and training history
            - Updates best_trial if this trial performed better
            - Updates architecture_trends for visualization
            - Logs trial completion
        """
        if trial_id not in self.trial_history:
            logger.warning(f"running OptimizationJob._complete_trial ... Trial {trial_id} not found")
            return
        
        trial_data = self.trial_history[trial_id]
        
        # Update completion info
        trial_data.completed_at = datetime.now().isoformat()
        trial_data.status = status
        trial_data.performance = performance
        trial_data.training_history = training_history
        trial_data.pruning_info = pruning_info
        
        # Calculate duration
        if trial_data.started_at:
            start_time = datetime.fromisoformat(trial_data.started_at.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(trial_data.completed_at.replace('Z', '+00:00'))
            trial_data.duration_seconds = (end_time - start_time).total_seconds()
        
        # Update best trial tracking
        if status == "completed" and performance.get('final_val_accuracy'):
            if (self.best_trial is None or 
                performance['final_val_accuracy'] > self.best_trial.performance.get('final_val_accuracy', 0)): # type: ignore
                self.best_trial = trial_data
                logger.debug(f"running OptimizationJob._complete_trial ... New best trial: {trial_id} with val_acc: {performance['final_val_accuracy']:.4f}")
        
        # Update architecture trends
        self._update_architecture_trends(trial_data.trial_number, performance, trial_data.architecture)
        
        # Clear current trial if this was it
        if self.current_trial and self.current_trial.trial_id == trial_id:
            self.current_trial = None
        
        logger.debug(f"running OptimizationJob._complete_trial ... Completed trial {trial_id} with status: {status}")
        logger.debug(f"running OptimizationJob._complete_trial ... Duration: {trial_data.duration_seconds:.1f}s")
    
    def _analyze_model_architecture(self, model: Any, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze model architecture for visualization
        
        Extracts detailed layer information, parameter counts, and
        architectural patterns for frontend visualization.
        
        Args:
            model: Built Keras model
            hyperparameters: Trial hyperparameters
            
        Returns:
            Dictionary with detailed architecture information
        """
        try:
            architecture = {
                "type": "cnn" if hyperparameters.get("architecture_type") == "cnn" else "lstm",
                "layers": [],
                "total_params": model.count_params(),
                "trainable_params": sum([layer.count_params() for layer in model.layers if layer.trainable]),
                "input_shape": list(model.input_shape[1:]) if model.input_shape else [],
                "output_shape": list(model.output_shape[1:]) if model.output_shape else []
            }
            
            # Analyze each layer
            for i, layer in enumerate(model.layers):
                layer_info = {
                    "index": i,
                    "name": layer.name,
                    "type": type(layer).__name__,
                    "params": layer.count_params(),
                    "trainable": layer.trainable,
                    "input_shape": list(layer.input_shape[1:]) if hasattr(layer, 'input_shape') and layer.input_shape else [],
                    "output_shape": list(layer.output_shape[1:]) if hasattr(layer, 'output_shape') and layer.output_shape else []
                }
                
                # Add layer-specific details
                if hasattr(layer, 'filters'):
                    layer_info["filters"] = layer.filters
                if hasattr(layer, 'kernel_size'):
                    layer_info["kernel_size"] = list(layer.kernel_size)
                if hasattr(layer, 'units'):
                    layer_info["units"] = layer.units
                if hasattr(layer, 'pool_size'):
                    layer_info["pool_size"] = list(layer.pool_size)
                if hasattr(layer, 'rate'):
                    layer_info["dropout_rate"] = layer.rate
                if hasattr(layer, 'activation'):
                    layer_info["activation"] = str(layer.activation)
                
                architecture["layers"].append(layer_info)
            
            return architecture
            
        except Exception as e:
            logger.warning(f"running OptimizationJob._analyze_model_architecture ... Failed to analyze architecture: {e}")
            return {
                "type": "unknown",
                "layers": [],
                "total_params": 0,
                "error": str(e)
            }
    
    def _calculate_model_size(self, model: Any) -> Dict[str, Any]:
        """
        Calculate model size metrics for visualization
        
        Args:
            model: Built Keras model
            
        Returns:
            Dictionary with model size information
        """
        try:
            total_params = model.count_params()
            trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
            
            # Estimate memory usage (rough approximation)
            memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
            
            return {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "non_trainable_params": total_params - trainable_params,
                "memory_mb": round(memory_mb, 2),
                "size_category": self._categorize_model_size(total_params)
            }
            
        except Exception as e:
            logger.warning(f"running OptimizationJob._calculate_model_size ... Failed to calculate model size: {e}")
            return {"error": str(e)}
    
    def _categorize_model_size(self, total_params: int) -> str:
        """
        Categorize model size for visualization
        
        Args:
            total_params: Total number of parameters
            
        Returns:
            Size category string
        """
        if total_params < 100_000:
            return "small"
        elif total_params < 1_000_000:
            return "medium"
        elif total_params < 10_000_000:
            return "large"
        else:
            return "very_large"
    
    def _calculate_trial_health_metrics(self, model: Any, history: Any) -> Dict[str, Any]:
        """
        Calculate health metrics for a trial (stub for integration)
        
        This method should integrate with your existing health metric
        calculations from model_optimizer_health.py
        
        Args:
            model: Trained Keras model
            history: Training history
            
        Returns:
            Dictionary with health metrics
        """
        try:
            # TODO: Integrate with existing health metric calculations
            # For now, return placeholder values
            health_metrics = {
                "dead_neuron_ratio": 0.0,  # Placeholder
                "parameter_efficiency": 0.0,  # Placeholder
                "training_stability": 0.0,  # Placeholder
                "gradient_health": 0.0,  # Placeholder
                "convergence_quality": 0.0,  # Placeholder
                "overall_health": 0.0  # Placeholder
            }
            
            # Add basic health indicators
            if history and hasattr(history, 'history'):
                # Calculate basic stability from loss curve
                if 'loss' in history.history:
                    losses = history.history['loss']
                    if len(losses) > 1:
                        loss_stability = 1.0 - (np.std(losses) / max(np.mean(losses), 1e-6))
                        health_metrics["training_stability"] = max(0.0, min(1.0, float(loss_stability)))
                
                # Calculate convergence quality from accuracy
                if 'val_accuracy' in history.history:
                    val_acc = history.history['val_accuracy']
                    if val_acc:
                        health_metrics["convergence_quality"] = val_acc[-1]
            
            return health_metrics
            
        except Exception as e:
            logger.warning(f"running OptimizationJob._calculate_trial_health_metrics ... Failed to calculate health metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_trial_training_stability(self, history: Any) -> Dict[str, Any]:
        """
        Calculate training stability metrics
        
        Args:
            history: Training history
            
        Returns:
            Dictionary with training stability data
        """
        try:
            if not history or not hasattr(history, 'history'):
                return {"error": "No training history available"}
            
            stability_metrics = {}
            
            # Analyze loss stability
            if 'loss' in history.history:
                losses = np.array(history.history['loss'])
                stability_metrics["loss_stability"] = {
                    "mean": float(np.mean(losses)),
                    "std": float(np.std(losses)),
                    "trend": "decreasing" if losses[-1] < losses[0] else "increasing",
                    "stability_score": float(1.0 - (np.std(losses) / max(float(np.mean(losses)), 1e-6)))
                }
            
            # Analyze accuracy stability
            if 'accuracy' in history.history:
                acc = np.array(history.history['accuracy'])
                stability_metrics["accuracy_stability"] = {
                    "mean": float(np.mean(acc)),
                    "std": float(np.std(acc)),
                    "final": float(acc[-1]),
                    "best": float(np.max(acc))
                }
            
            return stability_metrics
            
        except Exception as e:
            logger.warning(f"running OptimizationJob._calculate_trial_training_stability ... Failed to calculate training stability: {e}")
            return {"error": str(e)}
    
    def _calculate_trial_gradient_health(self, model: Any) -> Dict[str, Any]:
        """
        Calculate gradient health metrics (stub for integration)
        
        Args:
            model: Trained Keras model
            
        Returns:
            Dictionary with gradient health data
        """
        try:
            # TODO: Integrate with existing gradient health calculations
            gradient_health = {
                "overall_health": 0.5,  # Placeholder
                "gradient_norm": 0.0,  # Placeholder
                "gradient_variance": 0.0,  # Placeholder
                "vanishing_gradients": False,  # Placeholder
                "exploding_gradients": False  # Placeholder
            }
            
            return gradient_health
            
        except Exception as e:
            logger.warning(f"running OptimizationJob._calculate_trial_gradient_health ... Failed to calculate gradient health: {e}")
            return {"error": str(e)}
    
    def _update_health_trends(self, trial_number: int, health_metrics: Dict[str, Any]) -> None:
        """
        Update health trends for visualization
        
        Args:
            trial_number: Sequential trial number
            health_metrics: Health metrics for this trial
        """
        for metric_name, value in health_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if metric_name not in self.health_trends:
                    self.health_trends[metric_name] = []
                self.health_trends[metric_name].append(float(value))
    
    def _update_architecture_trends(self, trial_number: int, performance: Dict[str, Any], 
                                   architecture: Optional[Dict[str, Any]]) -> None:
        """
        Update architecture performance trends for visualization
        
        Args:
            trial_number: Sequential trial number
            performance: Performance metrics for this trial
            architecture: Architecture information for this trial
        """
        if not architecture:
            return
        
        # Track performance by architecture features
        arch_type = architecture.get("type", "unknown")
        total_params = architecture.get("total_params", 0)
        
        # Update parameter count trend
        if "parameter_count" not in self.architecture_trends:
            self.architecture_trends["parameter_count"] = []
        self.architecture_trends["parameter_count"].append(total_params)
        
        # Update performance trend
        val_accuracy = performance.get("final_val_accuracy", 0)
        if "val_accuracy" not in self.architecture_trends:
            self.architecture_trends["val_accuracy"] = []
        self.architecture_trends["val_accuracy"].append(val_accuracy)
        
        # Track architecture-specific trends
        if arch_type == "cnn":
            layer_count = len([layer for layer in architecture.get("layers", []) if "conv" in layer.get("type", "").lower()])
            if "cnn_layer_count" not in self.architecture_trends:
                self.architecture_trends["cnn_layer_count"] = []
            self.architecture_trends["cnn_layer_count"].append(layer_count)
        elif arch_type == "lstm":
            lstm_layers = [layer for layer in architecture.get("layers", []) if "lstm" in layer.get("type", "").lower()]
            if lstm_layers and "units" in lstm_layers[0]:
                if "lstm_units" not in self.architecture_trends:
                    self.architecture_trends["lstm_units"] = []
                self.architecture_trends["lstm_units"].append(lstm_layers[0]["units"])
    
    def _format_hyperparameters(self, hyperparameters: Dict[str, Any]) -> str:
        """
        Format hyperparameters for logging
        
        Args:
            hyperparameters: Dictionary of hyperparameters
            
        Returns:
            Formatted string for logging
        """
        key_params = []
        if 'num_layers_conv' in hyperparameters:
            key_params.append(f"conv_layers={hyperparameters['num_layers_conv']}")
        if 'filters_per_conv_layer' in hyperparameters:
            key_params.append(f"filters={hyperparameters['filters_per_conv_layer']}")
        if 'embedding_dim' in hyperparameters:
            key_params.append(f"embedding_dim={hyperparameters['embedding_dim']}")
        if 'lstm_units' in hyperparameters:
            key_params.append(f"lstm_units={hyperparameters['lstm_units']}")
        if 'epochs' in hyperparameters:
            key_params.append(f"epochs={hyperparameters['epochs']}")
        
        return ', '.join(key_params) if key_params else "hyperparameters logged"
    
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
        
        Returns serialized trial data for frontend consumption including
        architecture details, health metrics, and performance data.
        
        Returns:
            List of trial dictionaries with complete information
        """
        return [trial.dict() for trial in self.trials]
    
    def get_current_trial(self) -> Optional[Dict[str, Any]]:
        """
        Get currently running trial data
        
        Returns:
            Current trial dictionary or None if no trial running
        """
        return self.current_trial.dict() if self.current_trial else None
    
    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """
        Get best performing trial so far
        
        Returns:
            Best trial dictionary or None if no completed trials
        """
        return self.best_trial.dict() if self.best_trial else None
    
    def get_architecture_trends(self) -> Dict[str, List[float]]:
        """
        Get architecture performance trends for visualization
        
        Returns trends showing how different architectural choices
        affect performance over time.
        
        Returns:
            Dictionary mapping architecture features to performance trends
        """
        return self.architecture_trends.copy()
    
    def get_health_trends(self) -> Dict[str, List[float]]:
        """
        Get health metrics trends for visualization
        
        Returns trends showing how model health metrics evolve
        across trials.
        
        Returns:
            Dictionary mapping health metrics to trend data
        """
        return self.health_trends.copy()
    
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
    
    Provides REST API endpoints for managing optimization jobs and
    integrates with the existing model_optimizer system. Handles
    job lifecycle management, progress tracking, and result retrieval.
    
    Key Features:
        - Asynchronous job management
        - Real-time progress monitoring
        - Result persistence and retrieval
        - Model download capabilities
        - Comprehensive error handling
        - Integration with existing codebase
    
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
        
        logger.debug("running OptimizationAPI._register_routes ... All API endpoints registered")
    
    async def _start_optimization(self, request: OptimizationRequest, background_tasks: BackgroundTasks) -> JobResponse:
        """
        Start a new hyperparameter optimization job
        
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
        logger.debug(f"running OptimizationAPI._start_optimization ... Optimizer: {request.optimizer}")
        logger.debug(f"running OptimizationAPI._start_optimization ... Trials: {request.trials}")
        
        # Validate dataset
        if request.dataset_name not in self.dataset_manager.get_available_datasets():
            available = ", ".join(self.dataset_manager.get_available_datasets())
            logger.error(f"running OptimizationAPI._start_optimization ... Invalid dataset: {request.dataset_name}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset '{request.dataset_name}' not supported. Available: {available}"
            )
        
        # Validate optimizer
        if request.optimizer not in ["simple", "health"]:
            logger.error(f"running OptimizationAPI._start_optimization ... Invalid optimizer: {request.optimizer}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Optimizer '{request.optimizer}' not supported. Available: simple, health"
            )
        
        # Create new job
        job = OptimizationJob(request)
        self.jobs[job.job_id] = job
        
        # Start optimization in background
        background_tasks.add_task(job.start)
        
        logger.debug(f"running OptimizationAPI._start_optimization ... Created job {job.job_id}")
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
        return job.get_status()
    
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
        return {
            "job_id": job_id,
            "trials": job.get_trial_history(),
            "total_trials": len(job.trials)
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
        return {
            "job_id": job_id,
            "current_trial": job.get_current_trial()
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
        return {
            "job_id": job_id,
            "best_trial": job.get_best_trial()
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
        return {
            "job_id": job_id,
            "architecture_trends": job.get_architecture_trends(),
            "health_trends": job.get_health_trends()
        }


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