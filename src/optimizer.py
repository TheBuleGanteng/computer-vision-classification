"""
Unified Model Optimizer for Multi-Modal Classification - RUNPOD SERVICE INTEGRATION

RUNPOD SERVICE INTEGRATION:
- Sends JSON commands instead of Python code
- Uses specialized handler.py on RunPod side
- Maintains backward compatibility for local execution
- Uses same optimizer.py orchestration logic on both local and RunPod

Supports two optimization modes:
- "simple": Pure objective optimization (accuracy, efficiency, etc.)
- "health": Health-aware optimization with configurable weighting

Health metrics are always calculated for monitoring and API reporting regardless of mode.

Uses Bayesian optimization (Optuna) for intelligent hyperparameter search.
"""
import base64
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv
import json
import numpy as np
import optuna
from optuna.samplers import TPESampler
import os
from pathlib import Path
import pytz
import random
import re
import requests  # For RunPod service communication
import shutil
import sys
import tempfile
import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
import threading
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import yaml             
import zipfile


# Import existing modules
from data_classes.callbacks import TrialProgress, AggregatedProgress, UnifiedProgress, ConcurrentProgressAggregator, EpochProgressCallback, default_progress_callback

from data_classes.configs import OptimizationConfig, OptimizationMode, OptimizationObjective

from dataset_manager import DatasetManager, DatasetConfig
from health_analyzer import HealthAnalyzer
from hyperparameter_selector import HyperparameterSelector
from model_builder import ModelBuilder, ModelConfig
from model_visualizer import ModelVisualizer
from plot_generator import PlotGenerator
from utils.logger import logger
from utils.run_name import create_run_name
from utils.runpod_direct_download import get_runpod_worker_endpoint
from utils.test_runpod_endpoints import run_comprehensive_endpoint_tests, log_endpoint_test_summary


# download_directory functions no longer needed - plots come directly with training response


# Auto-load .env file from project root
def _load_env_file():
    """Automatically load .env file from project root"""
    current_file = Path(__file__)
    project_root = current_file.parent.parent  # Go up to project root
    env_file = project_root / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
        logger.debug(f"running _load_env_file ... Loaded environment variables from {env_file}")
        return True
    else:
        logger.debug(f"running _load_env_file ... No .env file found at {env_file}")
        return False

# Load environment variables automatically
_load_env_file()


# Progress tracking classes moved to callbacks.py


@dataclass
class OptimizationResult:
    """Results from optimization process"""

    # Best trial results
    best_total_score: float
    best_params: Dict[str, Any]
    best_trial_number: Optional[int] = None
    best_model_path: Optional[str] = None
    
    # Optimization metadata
    total_trials: int = 0
    successful_trials: int = 0
    optimization_time_hours: float = 0.0
    optimization_mode: str = "simple"
    health_weight: float = 0.0
    
    # Performance analysis
    objective_history: List[float] = field(default_factory=list)
    parameter_importance: Dict[str, float] = field(default_factory=dict)
    
    # Health monitoring data
    health_history: List[Dict[str, Any]] = field(default_factory=list)
    best_trial_health: Optional[Dict[str, Any]] = None
    
    # Direct download information (for RunPod execution)
    plots_direct_info: Optional[Dict[str, Any]] = None
    final_model_direct_info: Optional[Dict[str, Any]] = None
    average_health_metrics: Optional[Dict[str, float]] = None
    
    # Dataset and configuration info
    dataset_name: str = ""
    dataset_config: Optional[DatasetConfig] = None
    optimization_config: Optional[OptimizationConfig] = None
    
    # File paths
    results_dir: Optional[Path] = None
    optimization_report_path: Optional[str] = None
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        optimize_for = self.optimization_config.objective.value if self.optimization_config else "unknown"
        mode_name = self.optimization_mode
        
        # Add health summary if available
        health_summary = ""
        if self.best_trial_health:
            overall_health = self.best_trial_health.get('overall_health', 0.0)
            health_summary = f"\nBest Trial Health Score: {overall_health:.3f}"
            
            if mode_name == "health" and self.health_weight > 0:
                obj_weight = 1.0 - self.health_weight
                health_summary += f"\nWeighting: {obj_weight:.1%} objective, {self.health_weight:.1%} health"
        
        return f"""
Optimization Summary for {self.dataset_name}:
===========================================
Optimization Mode: {mode_name}
Best {optimize_for}: {self.best_total_score:.4f}
Successful trials: {self.successful_trials}/{self.total_trials}
Optimization time: {self.optimization_time_hours:.2f} hours{health_summary}

Best Parameters:
{self._format_params(self.best_params)}

Top Parameter Importance:
{self._format_importance(self.parameter_importance)}
"""
    
    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format parameters for display"""
        lines = []
        for key, value in sorted(params.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    
    def _format_importance(self, importance: Dict[str, float]) -> str:
        """Format parameter importance for display"""
        if not importance:
            return "  (Not calculated)"
        
        lines = []
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for key, value in sorted_importance[:5]:  # Top 5
            lines.append(f"  {key}: {value:.3f}")
        return "\n".join(lines)


# Callback and progress tracking classes moved to callbacks.py


class ModelOptimizer:
    """
    Unified optimizer class with RunPod service integration

    Integrates with existing ModelBuilder and DatasetManager to provide
    automated hyperparameter tuning with simple or health-aware optimization.
    """
    
    def __init__(self, dataset_name: str, optimization_config: Optional[OptimizationConfig] = None, 
        datasets_root: Optional[str] = None, run_name: Optional[str] = None,
        health_analyzer: Optional[HealthAnalyzer] = None,
        progress_callback: Optional[Callable[[Union[TrialProgress, AggregatedProgress, UnifiedProgress]], None]] = None,
        activation_override: Optional[str] = None):
        """
        Initialize ModelOptimizer with RunPod service support
        
        Args:
            dataset_name: Name of dataset to optimize for
            optimization_config: Optimization settings (uses defaults if None)
            datasets_root: Optional custom datasets directory
            run_name: Optional unified run name for consistent directory/file naming
            health_analyzer: Optional HealthAnalyzer instance (creates new if None)
            progress_callback: Optional callback function that receives TrialProgress updates
            activation_override: Optional activation function override for all trials
        """
        self.dataset_name = dataset_name
        if not optimization_config:
            raise ValueError("optimization_config is required - no defaults provided to avoid configuration duplication")
        self.config = optimization_config
        self.run_name = run_name
        self.activation_override = activation_override
        
        # Extract timestamp from run_name to ensure consistency between local and RunPod execution
        # run_name format: "YYYY-MM-DD-HH-MM-SS_dataset_mode" or similar
        if self.run_name:
            try:
                # Extract timestamp from the beginning of run_name
                timestamp_part = self.run_name.split('_')[0]
                # Validate timestamp format (should be YYYY-MM-DD-HH-MM-SS)
                if len(timestamp_part) == 19 and timestamp_part.count('-') == 5:
                    self.run_timestamp = timestamp_part
                    logger.debug(f"running ModelOptimizer.__init__ ... Extracted timestamp from run_name: {self.run_timestamp}")
                else:
                    raise ValueError(f"Invalid timestamp format in run_name: {timestamp_part}")
            except (IndexError, ValueError) as e:
                # Fallback: generate new timestamp with Jakarta timezone (should rarely happen)
                logger.warning(f"running ModelOptimizer.__init__ ... Could not extract timestamp from run_name '{self.run_name}': {e}")
                logger.warning(f"running ModelOptimizer.__init__ ... Generating fallback timestamp...")
                
                jakarta_tz = pytz.timezone('Asia/Jakarta')
                jakarta_time = datetime.now(jakarta_tz)
                self.run_timestamp = jakarta_time.strftime("%Y-%m-%d-%H-%M-%S")
                logger.warning(f"running ModelOptimizer.__init__ ... Using fallback timestamp: {self.run_timestamp}")
        else:
            # No run_name provided, generate timestamp with Jakarta timezone
            logger.debug(f"running ModelOptimizer.__init__ ... No run_name provided, generating timestamp...")
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            jakarta_time = datetime.now(jakarta_tz)
            self.run_timestamp = jakarta_time.strftime("%Y-%m-%d-%H-%M-%S")
            logger.debug(f"running ModelOptimizer.__init__ ... Generated timestamp: {self.run_timestamp}")
        if self.activation_override:
            logger.debug(f"running ModelOptimizer.__init__ ... Activation override: {self.activation_override} (will force this activation for all trials)")
        
        # Enhanced plot tracking
        self.trial_plot_data = {}
        #self.best_trial_number = None      
        
        # Log plot generation configuration
        
        # Initialize health analyzer (always available for monitoring)
        self.health_analyzer = health_analyzer or HealthAnalyzer()
        
        # Thread-safe shared state management
        self._state_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._best_trial_lock = threading.Lock()
        
        # Initialize dataset manager and load dataset info
        self.dataset_manager = DatasetManager(datasets_root)
        
        # Validate dataset
        if dataset_name not in self.dataset_manager.get_available_datasets():
            available = ', '.join(self.dataset_manager.get_available_datasets())
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")
        
        self.dataset_config = self.dataset_manager.get_dataset_config(dataset_name)
        
        # Load and prepare data once (for efficiency)
        logger.debug(f"running ModelOptimizer.__init__ ... Loading dataset: {dataset_name}")
        self.data = self.dataset_manager.load_dataset(
            dataset_name, 
            test_size=self.config.test_size
        )
        
        # Initialize HyperparameterSelector
        logger.debug(f"running ModelOptimizer.__init__ ... Initializing HyperparameterSelector")
        self.hyperparameter_selector = HyperparameterSelector(
            dataset_config=self.dataset_config,
            min_epochs=self.config.min_epochs_per_trial,
            max_epochs=self.config.max_epochs_per_trial
        )
        logger.debug(f"running ModelOptimizer.__init__ ... HyperparameterSelector initialized for data type: {self.hyperparameter_selector.data_type}")
        
        # Detect data type for search space selection (kept for backward compatibility)
        self.data_type = self._detect_data_type()
        logger.debug(f"running ModelOptimizer.__init__ ... Detected data type: {self.data_type}")
        
        # Initialize PlotGenerator
        logger.debug(f"running ModelOptimizer.__init__ ... Initializing PlotGenerator")
        default_model_config = ModelConfig()
        self.plot_generator = PlotGenerator(
            dataset_config=self.dataset_config,
            model_config=default_model_config,
            optimization_config=optimization_config
        )
        logger.debug(f"running ModelOptimizer.__init__ ... PlotGenerator initialized for data type: {self.plot_generator.data_type}")
        
        # Initialize ModelVisualizer for 3D architecture visualization
        logger.debug(f"running ModelOptimizer.__init__ ... Initializing ModelVisualizer")
        self.model_visualizer = ModelVisualizer()
        logger.debug(f"running ModelOptimizer.__init__ ... ModelVisualizer initialized for 3D architecture preparation")
        
        # Initialize optimization state
        self.study: Optional[optuna.Study] = None
        self.optimization_start_time: Optional[float] = None
        self.results_dir: Optional[Path] = None
        
        # Create results directory
        self._setup_results_directory()      
                        
        # Health monitoring storage (protected by _state_lock)
        self._trial_health_history: List[Dict[str, Any]] = []
        self._best_trial_health: Optional[Dict[str, Any]] = None
        
        # Thread-safe shared state management
        self._state_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._best_trial_lock = threading.Lock()
        
        # Real-time trial tracking (protected by _progress_lock)
        self.progress_callback = progress_callback
        self._current_trial_progress: Optional[TrialProgress] = None
        self._trial_progress_history: List[TrialProgress] = []
        self._best_trial_progress: Optional[TrialProgress] = None
        
        # Progress aggregation infrastructure
        self._progress_aggregator = ConcurrentProgressAggregator(self.config.trials)
        
        # Cancellation flag for graceful shutdown
        self._cancelled = threading.Event()
        self._trial_start_times: Dict[int, float] = {}
        self._trial_statuses: Dict[int, str] = {}  # "running", "completed", "failed"
        self._current_epoch_info: Dict[int, Dict[str, Any]] = {}  # Trial -> {current_epoch, total_epochs, epoch_progress}
        
        # Architecture and health trends (protected by _state_lock)
        self._architecture_trends: Dict[str, List[float]] = {}
        self._health_trends: Dict[str, List[float]] = {}
        
        # Best trial tracking (protected by _best_trial_lock)
        self._best_trial_number: Optional[int] = None
        self._best_trial_value: Optional[float] = None
        self._best_trial_accuracy: Optional[float] = None
        self._last_trial_accuracy: Optional[float] = None  # Temporary storage for latest trial accuracy
        self._last_trial_health_metrics: Optional[Dict[str, Any]] = None  # Temporary storage for latest trial health metrics
        
        # Current trial progress for unified progress updates
        self._current_trial_progress: Optional[TrialProgress] = None

        # Track trials with plots available for download
        self._trials_with_plots: set = set()

        # Track final model status for status updates
        self._final_model_building: bool = False
        self._final_model_available: bool = False
        
        
        logger.debug(f"running ModelOptimizer.__init__ ... Optimizer initialized for {dataset_name}")
        logger.debug(f"running ModelOptimizer.__init__ ... Mode: {self.config.mode}")
        logger.debug(f"running ModelOptimizer.__init__ ... Objective: {self.config.optimize_for}")
        if self.config.mode == OptimizationMode.HEALTH and not OptimizationObjective.is_health_only(self.config.objective):
            logger.debug(f"running ModelOptimizer.__init__ ... Health weight: {self.config.health_weight} ({(1-self.config.health_weight)*100:.0f}% objective, {self.config.health_weight*100:.0f}% health)")
        logger.debug(f"running ModelOptimizer.__init__ ... Max trials: {self.config.trials}")
        if self.run_name:
            logger.debug(f"running ModelOptimizer.__init__ ... Run name: {self.run_name}")
        
        # Log runpod service configuration
        if self.config.use_runpod_service:
            logger.debug(f"running ModelOptimizer.__init__ ... RunPod service integration ENABLED")
            logger.debug(f"running ModelOptimizer.__init__ ... - Approach: JSON API calls (specialized serverless)")
            logger.debug(f"running ModelOptimizer.__init__ ... - Endpoint: {self.config.runpod_service_endpoint}")
            logger.debug(f"running ModelOptimizer.__init__ ... - Timeout: {self.config.runpod_service_timeout}s")
            logger.debug(f"running ModelOptimizer.__init__ ... - Fallback local: {self.config.runpod_service_fallback_local}")
            logger.debug(f"running ModelOptimizer.__init__ ... - Payload: Tiny JSON commands (<1KB) instead of Python code")

            # Skip FastAPI endpoint testing since we're using simpler RunPod API approach
            logger.info("running ModelOptimizer.__init__ ... Skipping FastAPI endpoint testing - using RunPod API approach")
            logger.info("running ModelOptimizer.__init__ ... File downloads will use RunPod API calls instead of proxy URLs")
        else:
            logger.debug(f"running ModelOptimizer.__init__ ... RunPod service integration: DISABLED (local execution only)")

        logger.debug(f"running ModelOptimizer.__init__ ... GPU proxy code injection → RunPod service JSON API")
    
    
    # Add these methods to ModelOptimizer class
    def add_trial_health(self, trial_health: Dict[str, Any]) -> None:
        """Thread-safe method to add trial health data"""
        with self._state_lock:
            self._trial_health_history.append(trial_health)

    def update_best_trial_health(self, trial_health: Dict[str, Any]) -> None:
        """Thread-safe method to update best trial health"""
        with self._state_lock:
            self._best_trial_health = trial_health

    def get_trial_health_history(self) -> List[Dict[str, Any]]:
        """Thread-safe method to get trial health history"""
        with self._state_lock:
            return self._trial_health_history.copy()

    def get_best_trial_health(self) -> Optional[Dict[str, Any]]:
        """Thread-safe method to get best trial health"""
        with self._state_lock:
            return self._best_trial_health

    def update_best_trial(self, trial_number: int, trial_value: float, trial_accuracy: Optional[float] = None) -> None:
        """Thread-safe method to update best trial tracking"""
        with self._best_trial_lock:
            if self._best_trial_value is None or trial_value > self._best_trial_value:
                self._best_trial_number = trial_number
                self._best_trial_value = trial_value
                self._best_trial_accuracy = trial_accuracy

    def _test_runpod_endpoints(self) -> None:
        """
        Test RunPod endpoints to ensure connectivity before optimization starts.
        """
        try:
            logger.info("running _test_runpod_endpoints ... Testing RunPod endpoint connectivity")

            
            # Get worker endpoint from API URL
            if not self.config.runpod_service_endpoint:
                logger.error("running _test_runpod_endpoints ... RunPod service endpoint not configured")
                return

            worker_endpoint = get_runpod_worker_endpoint(self.config.runpod_service_endpoint)

            # Use run name for testing, fallback to a test name
            test_run_name = self.run_name or "endpoint_test"

            # Run comprehensive tests
            test_results = run_comprehensive_endpoint_tests(worker_endpoint, test_run_name)

            # Log detailed summary
            log_endpoint_test_summary(test_results)

            # Check if critical endpoints are working
            if test_results["summary"]["all_critical_tests_passed"]:
                logger.info("running _test_runpod_endpoints ... All critical RunPod endpoints are working")
            else:
                logger.warning("running _test_runpod_endpoints ... Some critical RunPod endpoints are not working")
                logger.warning("running _test_runpod_endpoints ... This may cause issues with plot downloads")

                # Store test results for later reference
                if not hasattr(self, '_endpoint_test_results'):
                    self._endpoint_test_results = test_results

        except Exception as e:
            logger.error(f"running _test_runpod_endpoints ... Error testing RunPod endpoints: {e}")
            logger.warning("running _test_runpod_endpoints ... Proceeding with optimization, but file downloads may fail")

    def _create_unified_progress(self, aggregated_progress: AggregatedProgress, trial_progress: Optional[TrialProgress] = None) -> UnifiedProgress:
        """
        Create unified progress by combining aggregated progress with current trial progress
        This eliminates the race condition between dual callbacks
        """
        # Use the provided trial_progress or the stored current trial progress
        current_trial = trial_progress or self._current_trial_progress
        
        # Get epoch information - different logic for GPU vs CPU mode
        current_epoch = None
        total_epochs = None
        epoch_progress = None
        
        if self.config.use_runpod_service:
            # GPU mode: Calculate aggregate epoch progress across all trials
            total_epochs_across_trials, completed_epochs = self._calculate_aggregate_epoch_progress()
            if total_epochs_across_trials > 0:
                current_epoch = completed_epochs
                total_epochs = total_epochs_across_trials
                epoch_progress = round(completed_epochs / total_epochs_across_trials, 2) if total_epochs_across_trials > 0 else 0
        else:
            # CPU mode: Use per-trial epoch information (existing behavior)
            if self._current_epoch_info:
                for trial_num in aggregated_progress.running_trials:
                    if trial_num in self._current_epoch_info:
                        epoch_info = self._current_epoch_info[trial_num]
                        current_epoch = epoch_info.get('current_epoch')
                        total_epochs = epoch_info.get('total_epochs')
                        epoch_progress = epoch_info.get('epoch_progress')
                        break  # Use the first running trial's epoch info
        
        current_trial_id = getattr(current_trial, 'trial_id', None) if current_trial else None
        current_trial_status = getattr(current_trial, 'status', None) if current_trial else None
        
        # Calculate average duration per trial for completed trials
        average_duration = None
        if self._trial_progress_history:
            completed_durations = [
                trial.duration_seconds for trial in self._trial_progress_history 
                if trial.status == "completed" and trial.duration_seconds is not None
            ]
            if completed_durations:
                average_duration = round(sum(completed_durations) / len(completed_durations))
        
        # Track completed trials with plots available
        available_plots = []
        if hasattr(self, '_trials_with_plots'):
            available_plots = [f"trial_{trial_num}" for trial_num in sorted(self._trials_with_plots)]

        # Determine Optuna Trials status
        optuna_trials_status = "pending"
        if len(aggregated_progress.running_trials) > 0:
            optuna_trials_status = "running"
        elif len(aggregated_progress.completed_trials) >= aggregated_progress.total_trials:
            optuna_trials_status = "completed"

        # Determine Final Model status based on trials completion and final model availability
        final_model_status = "pending"
        if optuna_trials_status == "completed":
            # Check if final model has been built/is available
            if hasattr(self, '_final_model_available') and self._final_model_available:
                final_model_status = "available"
            elif hasattr(self, '_final_model_building') and self._final_model_building:
                final_model_status = "running"
            else:
                final_model_status = "pending"

        return UnifiedProgress(
            # Copy all aggregated progress data
            total_trials=aggregated_progress.total_trials,
            running_trials=aggregated_progress.running_trials,
            completed_trials=aggregated_progress.completed_trials,
            failed_trials=aggregated_progress.failed_trials,
            current_best_total_score=aggregated_progress.current_best_total_score,
            current_best_total_score_trial_number=aggregated_progress.current_best_total_score_trial_number,
            current_best_accuracy=self._best_trial_accuracy,  # Track raw accuracy separately
            average_duration_per_trial=average_duration,
            estimated_time_remaining=aggregated_progress.estimated_time_remaining,
            # Add epoch information from current trial
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            epoch_progress=epoch_progress,
            current_trial_id=current_trial_id,
            current_trial_status=current_trial_status,
            status_message=self._get_current_status_message(aggregated_progress, current_epoch),  # Preserve GPU init or use default
            # Add plot generation and file download information
            plot_generation=getattr(current_trial, 'plot_generation', None) if current_trial else None,
            downloaded_files=getattr(self, '_recent_downloads', None),
            available_plots=available_plots,
            # Add Optuna Trials and Final Model status sections
            optuna_trials_status=optuna_trials_status,
            final_model_status=final_model_status
        )

    def _get_current_status_message(self, aggregated_progress: 'AggregatedProgress', current_epoch: Optional[int]) -> Optional[str]:
        """Determine the appropriate status message, preserving GPU initialization when needed"""
        # If using RunPod service and no trials have actually started training yet, 
        # preserve the "Initializing GPU" message
        if (self.config.use_runpod_service and 
            len(aggregated_progress.completed_trials) == 0 and
            (current_epoch is None or current_epoch < 1)):
            
            return "Initializing GPU resources..."
        
        return None  # Use default status message

    def _calculate_aggregate_epoch_progress(self) -> Tuple[int, int]:
        """Calculate total epochs across all trials and completed epochs for GPU mode"""
        # Only return epoch information if actual training has started
        # (i.e., we have real epoch info from at least one trial)
        if not self._current_epoch_info:
            return 0, 0
        
        # Check if any trial has started actual training (epoch > 0)
        any_training_started = False
        for trial_id, epoch_info in self._current_epoch_info.items():
            if epoch_info.get('current_epoch', 0) > 0:
                any_training_started = True
                break
        
        if not any_training_started:
            return 0, 0
        
        total_epochs_all_trials = 0
        completed_epochs = 0
        
        # Calculate total epochs across all trials that have started or will start
        for trial_id in range(self.config.trials):
            trial_epochs = 0
            
            # First check if we have current epoch info for this trial
            if self._current_epoch_info and trial_id in self._current_epoch_info:
                epoch_info = self._current_epoch_info[trial_id]
                trial_epochs = epoch_info.get('total_epochs', 0)
            
            # Only use configured epochs if we have started training (not as initial fallback)
            if trial_epochs == 0 and any_training_started:
                trial_epochs = self.config.max_epochs_per_trial
                
            total_epochs_all_trials += trial_epochs
        
        # Calculate progress as total progress across all trials
        total_progress = 0.0
        
        # First, handle completed trials (100% progress each)
        completed_trial_ids = set()
        if self._trial_progress_history:
            for trial in self._trial_progress_history:
                if trial.status == "completed":
                    trial_id = getattr(trial, 'trial_number', None)
                    if trial_id is not None:
                        completed_trial_ids.add(trial_id)
                        total_progress += 1.0  # 100% progress for completed trial
        
        # Then, handle currently running trials
        for trial_id, epoch_info in self._current_epoch_info.items():
            if trial_id not in completed_trial_ids:  # Don't double-count completed trials
                current_epoch = epoch_info.get('current_epoch', 0)
                total_trial_epochs = epoch_info.get('total_epochs', self.config.max_epochs_per_trial)
                epoch_progress = epoch_info.get('epoch_progress', 0.0)
                
                if current_epoch > 0 and total_trial_epochs > 0:
                    # Calculate progress for this trial: (completed_epochs + current_epoch_progress) / total_epochs
                    trial_progress = ((current_epoch - 1) + epoch_progress) / total_trial_epochs
                    total_progress += trial_progress
        
        # Convert total progress back to "completed epochs" for display
        completed_epochs = int(total_progress * total_epochs_all_trials / self.config.trials)
        
        return total_epochs_all_trials, completed_epochs

    def get_best_trial_info(self) -> Tuple[Optional[int], Optional[float]]:
        """Thread-safe method to get best trial info"""
        with self._best_trial_lock:
            return self._best_trial_number, self._best_trial_value
    
    def cancel(self) -> None:
        """Request cancellation of the optimization process"""
        logger.info("running ModelOptimizer.cancel ... Cancellation requested")
        self._cancelled.set()
    
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested"""
        return self._cancelled.is_set()
    
    def get_trial_history(self) -> List[Dict[str, Any]]:
        """
        Get complete trial history for UI visualization
        
        Returns all trials from _trial_progress_history as API-friendly dictionaries
        sorted by trial number (most recent first for UI display)
        
        Returns:
            List of trial dictionaries with trial_number, status, timestamps, etc.
        """
        with self._progress_lock:
            trials = []
            for trial_progress in self._trial_progress_history:
                trial_dict = trial_progress.to_dict()
                trials.append(trial_dict)
            
            # Sort by trial number descending (most recent first)
            trials.sort(key=lambda x: x.get('trial_number', 0), reverse=True)
            return trials
    
    def get_current_trial(self) -> Optional[Dict[str, Any]]:
        """
        Get currently running trial if any exists
        
        Returns:
            Trial dictionary for currently running trial, or None
        """
        with self._progress_lock:
            for trial_progress in self._trial_progress_history:
                if trial_progress.status == "running":
                    return trial_progress.to_dict()
            return None
    
    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """
        Get best completed trial based on _best_trial_number
        
        Returns:
            Trial dictionary for best trial, or None if no trials completed
        """
        with self._best_trial_lock:
            best_number = self._best_trial_number
        
        if best_number is None:
            return None
        
        with self._progress_lock:
            for trial_progress in self._trial_progress_history:
                if trial_progress.trial_number == best_number:
                    return trial_progress.to_dict()
            return None
    
    def get_best_model_visualization_data(self) -> Optional[Dict[str, Any]]:
        """
        Get 3D visualization data for the best performing model
        
        Returns:
            Dictionary containing best model data with 3D visualization information,
            or None if no completed trials exist
        """
        # Get the best trial data
        best_trial = self.get_best_trial()
        if not best_trial:
            logger.debug("No best trial available for visualization")
            return None
        
        # Extract architecture and performance data
        architecture = best_trial.get('architecture')
        if not architecture:
            logger.warning(f"Best trial {best_trial.get('trial_number')} has no architecture data")
            return None
        
        # Get performance scores
        performance = best_trial.get('performance', {})
        health_metrics = best_trial.get('health_metrics', {})
        
        performance_score = performance.get('total_score', 0.0)
        health_score = health_metrics.get('overall_health') if health_metrics else None
        
        logger.debug(f"Preparing 3D visualization for best trial {best_trial.get('trial_number')} "
                    f"(score: {performance_score:.3f}, health: {health_score})")
        
        try:
            # Generate 3D visualization data
            visualization_data = self.model_visualizer.prepare_visualization_data(
                architecture=architecture,
                performance_score=performance_score,
                health_score=health_score
            )
            
            # Get color scheme
            color_scheme = self.model_visualizer.get_performance_color_scheme(
                performance_score, health_score
            )
            
            # Combine all data for API response
            return {
                'trial_id': best_trial.get('trial_id'),
                'trial_number': best_trial.get('trial_number'),
                'total_score': performance_score,
                'accuracy': performance.get('accuracy'),
                'architecture': architecture,
                'health_metrics': health_metrics,
                'training_duration': best_trial.get('duration_seconds'),
                'updated_at': best_trial.get('completed_at'),
                
                # 3D visualization data
                'visualization_data': {
                    'type': visualization_data.architecture_type,
                    'layers': [
                        {
                            'layer_id': layer.layer_id,
                            'layer_type': layer.layer_type,
                            'position_z': layer.position_z,
                            'width': layer.width,
                            'height': layer.height,
                            'depth': layer.depth,
                            'parameters': layer.parameters,
                            'activation': layer.activation,
                            'filters': layer.filters,
                            'kernel_size': layer.kernel_size,
                            'units': layer.units,
                            'color_intensity': layer.color_intensity,
                            'opacity': layer.opacity
                        }
                        for layer in visualization_data.layers
                    ],
                    'total_parameters': visualization_data.total_parameters,
                    'model_depth': visualization_data.model_depth,
                    'max_layer_width': visualization_data.max_layer_width,
                    'max_layer_height': visualization_data.max_layer_height,
                    'performance_score': visualization_data.performance_score,
                    'health_score': visualization_data.health_score
                },
                
                # Color scheme for 3D rendering
                'color_scheme': color_scheme
            }
            
        except Exception as e:
            logger.error(f"Failed to generate 3D visualization data: {e}")
            return None
        
    def _thread_safe_progress_callback(self, trial_progress: TrialProgress) -> None:
        """
        Thread-safe progress callback wrapper as specified in status.md

        Args:
            trial_progress: TrialProgress object with current trial status
        """
        logger.debug(f"running _thread_safe_progress_callback ... processing progress for trial {trial_progress.trial_number}")

        with self._progress_lock:
            # Check if we're trying to report "running" status for a trial that's already completed
            # This prevents race conditions where epoch callbacks report running status after completion
            current_status = self._trial_statuses.get(trial_progress.trial_number)
            if (current_status in ['completed', 'failed', 'pruned'] and
                trial_progress.status == 'running'):
                logger.debug(f"running _thread_safe_progress_callback ... Trial {trial_progress.trial_number} already {current_status}, ignoring 'running' status update")
                return

            # Update trial status tracking
            self._trial_statuses[trial_progress.trial_number] = trial_progress.status
            logger.debug(f"running _thread_safe_progress_callback ... updated trial {trial_progress.trial_number} status to '{trial_progress.status}'")
            
            # Update or add trial progress to history
            # Find existing trial and update it, or append if it's new
            existing_trial_index = None
            for i, existing_trial in enumerate(self._trial_progress_history):
                if existing_trial.trial_number == trial_progress.trial_number:
                    existing_trial_index = i
                    break
            
            if existing_trial_index is not None:
                # Update existing trial with latest status and data
                self._trial_progress_history[existing_trial_index] = trial_progress
                logger.debug(f"running _thread_safe_progress_callback ... updated existing trial {trial_progress.trial_number} in history")
            else:
                # Add new trial to history
                self._trial_progress_history.append(trial_progress)
                logger.debug(f"running _thread_safe_progress_callback ... added new trial {trial_progress.trial_number} to history")
            
            # Call user-provided progress callback if available
            if self.progress_callback:
                try:
                    # Get current best value for aggregation
                    best_trial_number, best_trial_value = self.get_best_trial_info()
                    
                    # Update the progress aggregator with current best value
                    self._progress_aggregator.get_current_best_total_score = lambda: best_trial_value
                    self._progress_aggregator.get_current_best_trial_number = lambda: best_trial_number
                    
                    # Create aggregated progress
                    aggregated_progress = self._progress_aggregator.aggregate_progress(
                        current_trial=trial_progress,
                        all_trial_statuses=self._trial_statuses
                    )
                    
                    # logger.debug(f"running _thread_safe_progress_callback ... calling user progress callback with aggregated data")
                    # logger.debug(f"running _thread_safe_progress_callback ... - Total trials: {aggregated_progress.total_trials}")
                    # logger.debug(f"running _thread_safe_progress_callback ... - Running: {len(aggregated_progress.running_trials)}")
                    # logger.debug(f"running _thread_safe_progress_callback ... - Completed: {len(aggregated_progress.completed_trials)}")
                    # logger.debug(f"running _thread_safe_progress_callback ... - Failed: {len(aggregated_progress.failed_trials)}")
                    
                    # PHASE 1: Unified progress callback system
                    # Store current trial progress for epoch information
                    self._current_trial_progress = trial_progress
                    
                    # Create unified progress combining trial statistics with epoch information
                    unified_progress = self._create_unified_progress(aggregated_progress, trial_progress)
                    
                    # Single call with all information the UI needs
                    logger.debug(f"running _thread_safe_progress_callback ... calling user progress callback with unified progress")
                    logger.debug(f"running _thread_safe_progress_callback ... - Epoch info: {unified_progress.current_epoch}/{unified_progress.total_epochs} (progress: {unified_progress.epoch_progress})")
                    self.progress_callback(unified_progress)
                    
                except Exception as e:
                    logger.error(f"running _thread_safe_progress_callback ... error in progress callback: {e}")
                    # Don't re-raise to avoid disrupting trial execution
            else:
                logger.debug(f"running _thread_safe_progress_callback ... no user progress callback configured")
    
    def _report_final_model_progress(self, step_name: str, overall_progress: float, detailed_info: Optional[str] = None) -> None:
        """
        Report progress during final model building phase
        
        Args:
            step_name: Current step being performed (e.g., "Training epoch 3/5")
            overall_progress: Overall progress (0.0 to 1.0)
            detailed_info: Optional detailed information for the step
        """
        if not self.progress_callback:
            return
            
        try:
            logger.debug(f"running _report_final_model_progress ... {step_name} progress: {overall_progress:.1%}")
            logger.debug(f"running _report_final_model_progress ... progress callback available: {self.progress_callback is not None}")
            
            # Create progress data structure
            final_model_data = {
                "status": "building" if overall_progress < 1.0 else "completed",
                "current_step": step_name,
                "progress": overall_progress,
                "detailed_info": detailed_info
            }
            
            # Create a unified progress update with final model building info
            # Get current aggregated progress
            best_trial_number, best_trial_value = self.get_best_trial_info()
            self._progress_aggregator.get_current_best_total_score = lambda: best_trial_value
            self._progress_aggregator.get_current_best_trial_number = lambda: best_trial_number
            
            # During final model building, there's no current trial, so use the last completed trial
            last_trial_progress = None
            if self._trial_progress_history:
                last_trial_progress = self._trial_progress_history[-1]
            
            aggregated_progress = self._progress_aggregator.aggregate_progress(
                current_trial=last_trial_progress,
                all_trial_statuses=self._trial_statuses
            )
            
            # Create unified progress with final model building info
            unified_progress = self._create_unified_progress(aggregated_progress, last_trial_progress)
            
            # Add final model building progress to unified progress
            unified_progress.final_model_building = final_model_data
            
            logger.debug(f"running _report_final_model_progress ... calling progress callback with final model progress")
            self.progress_callback(unified_progress)
            
        except Exception as e:
            logger.error(f"running _report_final_model_progress ... error reporting progress: {e}")
    
    
    
    def _detect_data_type(self) -> str:
        """Detect whether this is image or text data"""
        if (self.dataset_config.img_height == 1 and 
            self.dataset_config.channels == 1 and 
            self.dataset_config.img_width > 100):
            return "text"
        else:
            return "image"
    
    def _setup_results_directory(self) -> None:
        """Create directory using the unified run_name"""
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        optimization_results_dir = project_root / "optimization_results"
        
        if self.run_name:
            self.results_dir = optimization_results_dir / self.run_name
            logger.debug(f"running _setup_results_directory ... Using provided run_name: {self.run_name}")
        else:
            
            fallback_name = create_run_name(
                dataset_name=self.dataset_name,
                mode=self.config.mode,
                optimize_for=self.config.optimize_for
            )
            self.results_dir = optimization_results_dir / fallback_name
            logger.debug(f"running _setup_results_directory ... No run_name provided, generated: {fallback_name}")
        
        # Ensure results_dir is set (for type checker)
        assert self.results_dir is not None, "results_dir should be set by this point"

        # Create the main results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"running _setup_results_directory ... Results directory: {self.results_dir}")

        # Create plots subdirectory
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Create trial directories (zero-indexed)
        for trial_num in range(self.config.trials):
            trial_dir = plots_dir / f"trial_{trial_num}"
            trial_dir.mkdir(exist_ok=True)
        
        # Create optimized model directory
        optimized_dir = self.results_dir / "optimized_model"
        optimized_dir.mkdir(exist_ok=True)
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Use HyperparameterSelector for hyperparameter suggestion
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        logger.debug(f"running _suggest_hyperparameters ... Trial {trial.number}: activation_override = '{self.activation_override}'")
        
        # Use the modular HyperparameterSelector
        params = self.hyperparameter_selector.suggest_hyperparameters(
            trial=trial,
            activation_override=self.activation_override
        )
        
        logger.debug(f"running _suggest_hyperparameters ... HyperparameterSelector generated {len(params)} parameters")
        logger.debug(f"running _suggest_hyperparameters ... Architecture type: {params.get('architecture_type', 'unknown')}")
        
        return params

    def _extract_architecture_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract architecture information from hyperparameters for display in trial gallery
        
        Args:
            params: Hyperparameter dictionary from _suggest_hyperparameters
            
        Returns:
            Dictionary containing structured architecture information
        """
        architecture_type = params.get('architecture_type', 'unknown')
        
        if architecture_type == 'cnn':
            return {
                'type': 'CNN',
                'conv_layers': params.get('num_layers_conv', 0),
                'filters_per_layer': params.get('filters_per_conv_layer', 0),
                'kernel_size': params.get('kernel_size', (0, 0)),
                'dense_layers': params.get('num_layers_hidden', 0),
                'first_dense_nodes': params.get('first_hidden_layer_nodes', 0),
                'activation': params.get('activation', 'unknown'),
                'batch_normalization': params.get('batch_normalization', False),
                'use_global_pooling': params.get('use_global_pooling', False),
                'kernel_initializer': params.get('kernel_initializer', 'unknown')
            }
        elif architecture_type == 'lstm':
            return {
                'type': 'LSTM',
                'lstm_units': params.get('lstm_units', 0),
                'dense_layers': params.get('num_layers_hidden', 0),
                'first_dense_nodes': params.get('first_hidden_layer_nodes', 0),
                'activation': params.get('activation', 'unknown'),
                'dropout_rate': params.get('dropout_rate', 0.0),
                'recurrent_dropout': params.get('recurrent_dropout', 0.0)
            }
        else:
            return {
                'type': 'Unknown',
                'raw_params': params
            }

    # RunPod Service Methods (replacing GPU proxy)
    
    def _should_use_runpod_service(self) -> bool:
        """
        Determine if RunPod service should be used for this trial
        
        Returns:
            True if RunPod service should be used, False for local execution
        """
        if self.config.use_runpod_service:
            if self.config.runpod_service_endpoint:
                logger.debug(f"running _should_use_runpod_service ... Using RunPod service: {self.config.runpod_service_endpoint}")
                return True
            else:
                logger.warning(f"running _should_use_runpod_service ... RunPod service enabled but no endpoint configured")
                return False
        
        return False
    
    def _wait_for_previous_trial_head_start(self, current_trial_number: int) -> None:
        """
        Wait for the previous trial to complete a head start number of epochs
        to ensure trials complete in order (trial_0, trial_1, trial_2, etc.)
        
        Args:
            current_trial_number: The number of the current trial
        """
        previous_trial_number = current_trial_number - 1
        head_start_epochs = 2  # Number of epochs the previous trial should complete first
        
        logger.debug(f"Staggered start: Trial {current_trial_number} waiting for trial {previous_trial_number} to complete {head_start_epochs} epochs")
        
        # Wait up to 10 minutes for the previous trial to get its head start
        max_wait_time = 600  # 10 minutes
        check_interval = 5   # Check every 5 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            with self._progress_lock:
                # Look for the previous trial's progress in current epoch info
                previous_trial_info = self._current_epoch_info.get(previous_trial_number)
                if previous_trial_info:
                    current_epoch = previous_trial_info.get('current_epoch', 0)
                    if current_epoch >= head_start_epochs:
                        logger.debug(f"Staggered start: Trial {previous_trial_number} has completed {current_epoch} epochs, starting trial {current_trial_number}")
                        return
                
                # Also check if previous trial has completed (no longer needs head start)
                previous_trial_status = self._trial_statuses.get(previous_trial_number)
                if previous_trial_status in ['completed', 'failed', 'pruned']:
                    logger.debug(f"Staggered start: Trial {previous_trial_number} has finished ({previous_trial_status}), starting trial {current_trial_number}")
                    return
            
            time.sleep(check_interval)
        
        # If we've waited too long, proceed anyway
        logger.warning(f"Staggered start: Timeout waiting for trial {previous_trial_number} head start, proceeding with trial {current_trial_number}")
    
    def train(self, trial: optuna.Trial, params: Dict[str, Any]) -> float:
        """
        Unified training function that automatically determines execution mode based on configuration
        
        Args:
            trial: Optuna trial object
            params: Hyperparameters for this trial
            
        Returns:
            Objective value for optimization
        """
        # Automatically determine execution mode based on configuration
        train_locally = not self._should_use_runpod_service()
        
        if train_locally:
            logger.debug(f"running train ... Trial {trial.number}: Using local execution (RunPod disabled)")
            return self._train_locally_for_trial(trial, params)
        else:
            logger.debug(f"running train ... Trial {trial.number}: Using RunPod service execution")
            return self._train_via_runpod_service(trial, params)
    
    def _train_via_runpod_service(self, trial: optuna.Trial, params: Dict[str, Any]) -> float:
        logger.debug(f"running _train_via_runpod_service ... Starting RunPod service training for trial {trial.number}")
        logger.debug(f"running _train_via_runpod_service ... Using JSON API approach (tiny payloads) instead of code injection")
        
        try:
            api_key = os.getenv('RUNPOD_API_KEY')
            if not api_key:
                raise RuntimeError("RUNPOD_API_KEY environment variable not set")
            if self.config.runpod_service_endpoint is None:
                raise RuntimeError("RunPod service endpoint is not configured")

            request_payload = {
                "input": {
                    "command": "start_training",
                    "trial_id": self.run_name,  # Use full run name for consistent file organization
                    "dataset_name": self.dataset_name,
                    "run_name": self.run_name,  # Include coordinated run_name for timestamp consistency
                    "hyperparameters": params,
                    "config": {
                        "validation_split": self.config.validation_split,
                        "max_training_time": self.config.max_training_time_minutes,
                        "mode": self.config.mode,
                        "objective": self.config.optimize_for,
                        "gpu_proxy_sample_percentage": self.config.gpu_proxy_sample_percentage,
                        "use_multi_gpu": self.config.use_multi_gpu,
                        "target_gpus_per_worker": self.config.target_gpus_per_worker,
                        "auto_detect_gpus": self.config.auto_detect_gpus,
                        "multi_gpu_batch_size_scaling": self.config.multi_gpu_batch_size_scaling,
                        "create_optuna_model_plots": self.config.create_optuna_model_plots
                    }
                }
            }
            payload_size = len(json.dumps(request_payload).encode('utf-8'))
            # logger.debug(f"running _train_via_runpod_service ... 🔄 PAYLOAD SIZE: {payload_size} bytes")
            # logger.debug(f"running _train_via_runpod_service ... DEBUG PAYLOAD: {json.dumps(request_payload, indent=2)}")

            # Implement staggered start mechanism for proper trial completion order
            if trial.number > 0:
                self._wait_for_previous_trial_head_start(trial.number)

            # --- per-trial HTTP session (safer under n_jobs > 1) ---
            with requests.Session() as sess:
                sess.headers.update({
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "Connection": "close"
                })

                # Submit
                submit_url = self.config.runpod_service_endpoint
                logger.debug(f"running _train_via_runpod_service ... POST {submit_url}")
                response = sess.post(submit_url, json=request_payload, timeout=self.config.runpod_service_timeout)
                response.raise_for_status()
                result = response.json()
                
                # Log the initial submission response - Commented out b/c optimizer now running locally and thus, these are duplicative of the logs returned by api_server.py
                # logger.info(f"🔍 RUNPOD SUBMIT RESPONSE - Status Code: {response.status_code}")
                # logger.info(f"🔍 RUNPOD SUBMIT RESPONSE - Keys: {list(result.keys())}")
                # logger.info(f"🔍 RUNPOD SUBMIT RESPONSE - Full response: {json.dumps(result, indent=2)}")
                
                job_id = result.get('id')
                if not job_id:
                    raise RuntimeError("No job ID returned from RunPod service")

                logger.debug(f"running _train_via_runpod_service ... Job submitted with ID: {job_id}")
                logger.debug(f"running _train_via_runpod_service ... Polling for completion...")

                # Report initial trial progress to UI (no epoch estimates)
                initial_progress = TrialProgress(
                    trial_id=f"trial_{trial.number}",
                    trial_number=trial.number,
                    status="running",
                    started_at=datetime.now().isoformat()
                )
                self._thread_safe_progress_callback(initial_progress)

                # Poll with faster interval for real-time plot downloads
                max_poll_time = self.config.runpod_service_timeout
                poll_interval = 2  # Reduced from 10 to 2 seconds for faster notification detection
                start_time = time.time()
                status_url = f"{self.config.runpod_service_endpoint.rsplit('/run', 1)[0]}/status/{job_id}"

                while time.time() - start_time < max_poll_time:
                    status_response = sess.get(status_url, timeout=30)
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    job_status = status_data.get('status', 'UNKNOWN')
                    logger.debug(f"running _train_via_runpod_service ... Job {job_id} status: {job_status}")
                    
                    # Log what we're getting back from RunPod for debugging
                    logger.info(f"🔍 RUNPOD POLLING - Trial {trial.number} - Status: {job_status}")
                    logger.info(f"🔍 RUNPOD POLLING - Status response keys: {list(status_data.keys())}")

                    # Extract worker ID for proxy URL construction
                    worker_id = status_data.get('workerId', status_data.get('worker_id'))
                    if worker_id:
                        logger.info(f"🎯 RUNPOD POLLING - Worker ID: {worker_id}")
                        logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: Trial {trial.number} TRAINED on worker_id={worker_id}")
                    else:
                        logger.debug(f"🔍 RUNPOD POLLING - No worker ID found in response")
                        logger.warning(f"🏗️ WORKER_ISOLATION_TRACKING: Trial {trial.number} completed but no worker_id available")

                    # logger.info(f"🔍 RUNPOD POLLING - Full status response: {json.dumps(status_data, indent=2)}")
                    
                    # Check for output data
                    output_data = status_data.get('output')
                    if output_data:
                        logger.info(f"🔍 RUNPOD POLLING - Output data found with keys: {list(output_data.keys())}")
                        # logger.info(f"🔍 RUNPOD POLLING - Full output data: {json.dumps(output_data, indent=2)}")
                    else:
                        logger.info(f"🔍 RUNPOD POLLING - No output data available yet")

                    if job_status == 'COMPLETED':
                        logger.info(f"🎉 RUNPOD COMPLETED - Trial {trial.number} job finished!")
                        logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: ✅ Training COMPLETED for trial {trial.number} on worker_id={worker_id}")

                        output = status_data.get('output', {})
                        #logger.info(f"🔍 RUNPOD COMPLETED - Processing output data: {json.dumps(output, indent=2) if output else 'NO OUTPUT DATA'}")

                        # Retry logic for empty output (transient failures)
                        if not output:
                            logger.warning(f"running _train_via_runpod_service ... No output data in COMPLETED response for trial {trial.number}")
                            logger.warning(f"running _train_via_runpod_service ... Implementing retry logic with exponential backoff")

                            max_retries = 3
                            retry_delays = [5, 10, 15]  # seconds

                            for retry_attempt in range(max_retries):
                                logger.warning(f"running _train_via_runpod_service ... Retry attempt {retry_attempt + 1}/{max_retries} - waiting {retry_delays[retry_attempt]}s before re-checking status")
                                logger.warning(f"running _train_via_runpod_service ... Worker ID: {worker_id}, Job ID: {job_id}")

                                time.sleep(retry_delays[retry_attempt])

                                logger.debug(f"running _train_via_runpod_service ... Re-fetching job status from {status_url}")

                                # Re-fetch the job status using the existing session
                                try:
                                    retry_status_response = sess.get(status_url, timeout=30)
                                    retry_status_response.raise_for_status()
                                    retry_status_data = retry_status_response.json()
                                    output = retry_status_data.get('output', {})

                                    if output:
                                        logger.info(f"running _train_via_runpod_service ... ✅ Output retrieved successfully on retry attempt {retry_attempt + 1}")
                                        logger.info(f"running _train_via_runpod_service ... Output keys: {list(output.keys())}")
                                        logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: ✅ Retry successful for trial {trial.number} on worker_id={worker_id}")
                                        status_data = retry_status_data  # Update status_data with successful retry
                                        break
                                    else:
                                        logger.warning(f"running _train_via_runpod_service ... Still no output on retry attempt {retry_attempt + 1}/{max_retries}")
                                        logger.debug(f"running _train_via_runpod_service ... Status data keys: {list(retry_status_data.keys())}")

                                except Exception as retry_error:
                                    logger.error(f"running _train_via_runpod_service ... Retry attempt {retry_attempt + 1} failed: {retry_error}")
                                    logger.error(f"running _train_via_runpod_service ... Error type: {type(retry_error).__name__}")

                            # If still no output after retries, raise error
                            if not output:
                                logger.error(f"running _train_via_runpod_service ... No output returned after {max_retries} retry attempts")
                                logger.error(f"running _train_via_runpod_service ... Total wait time: {sum(retry_delays)}s")
                                logger.error(f"🏗️ WORKER_ISOLATION_TRACKING: ❌ Training FAILED - no output from trial {trial.number} on worker_id={worker_id}")
                                logger.error(f"running _train_via_runpod_service ... Final status_data: {json.dumps(status_data, indent=2)}")
                                raise RuntimeError(f"No output returned from completed RunPod job after {max_retries} retries")

                        # Verify training completion success
                        training_success = output.get('success', False)
                        if training_success:
                            logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: ✅ Training SUCCESS confirmed for trial {trial.number} on worker_id={worker_id}")
                        else:
                            logger.warning(f"🏗️ WORKER_ISOLATION_TRACKING: ⚠️ Training completed but success=False for trial {trial.number} on worker_id={worker_id}")
                        
                        success_flag = output.get('success', False)
                        logger.info(f"🔍 RUNPOD COMPLETED - Success flag: {success_flag}")
                        
                        if not success_flag:
                            error_msg = output.get('error', 'Unknown error from RunPod service')
                            logger.error(f"🚨 RUNPOD COMPLETED - Job failed with error: {error_msg}")
                            raise RuntimeError(f"RunPod service training failed: {error_msg}")

                        metrics = output.get('metrics', {})
                        logger.info(f"🔍 RUNPOD COMPLETED - Metrics data: {json.dumps(metrics, indent=2) if metrics else 'NO METRICS'}")
                        
                        if not metrics:
                            logger.error(f"🚨 RUNPOD COMPLETED - No metrics returned from RunPod service")
                            logger.error(f"🚨 RUNPOD COMPLETED - Available output keys: {list(output.keys())}")
                            raise RuntimeError("No metrics returned from RunPod service")

                        total_score = self._calculate_total_score_from_service_response(metrics, output, trial)
                        logger.info(f"🎯 RUNPOD COMPLETED - Trial {trial.number} completed successfully!")
                        logger.info(f"🎯 RUNPOD COMPLETED - Total score calculated: {total_score:.4f}")
                        
                        # ========================================
                        # ENHANCED RUNPOD RESPONSE ANALYSIS
                        # ========================================
                        logger.info(f"🔍 RUNPOD RESPONSE ANALYSIS - Trial {trial.number}")
                        logger.info(f"📦 RUNPOD RESPONSE KEYS: {list(output.keys())}")
                        logger.info(f"📦 RUNPOD RESPONSE SIZE: {len(str(output))} characters")
                        
                        # Check for plots_direct information
                        if 'plots_direct' in output:
                            plots_direct_info = output['plots_direct']
                            logger.info(f"✅ PLOTS_DIRECT FOUND in RunPod response")
                            # logger.info(f"📊 PLOTS_DIRECT CONTENT: {plots_direct_info}")

                            # Validate plots_direct structure
                            required_fields = ['success', 'available_files']
                            missing_fields = [field for field in required_fields if field not in plots_direct_info]
                            if missing_fields:
                                logger.error(f"❌ PLOTS_DIRECT MISSING FIELDS: {missing_fields}")
                            else:
                                logger.info(f"✅ PLOTS_DIRECT STRUCTURE VALID - All required fields present")
                                logger.info(f"📂 AVAILABLE_FILES: {len(plots_direct_info.get('available_files', []))} files")
                        else:
                            logger.warning(f"❌ PLOTS_DIRECT NOT FOUND in RunPod response")
                            logger.warning(f"🔍 Available response keys: {list(output.keys())}")
                        
                        # Note: final_model_direct checks removed - final models are only built after all trials complete

                        # Extract plots directly from training response (eliminates worker isolation)
                        plots_direct_info = output.get('plots_direct', {})
                        if plots_direct_info and plots_direct_info.get('success'):
                            # Determine local directory for plots
                            if self.results_dir:
                                local_plots_dir = self.results_dir / "plots" / f"trial_{trial.number}"
                                local_plots_dir.mkdir(parents=True, exist_ok=True)

                                # Extract compressed zip from training response (new compressed format)
                                plot_files_zip = plots_direct_info.get('plot_files_zip')
                                plots_worker_id = plots_direct_info.get('worker_id', 'unknown_plots_worker')

                                logger.info(f"📥 Extracting trial {trial.number} compressed files from training response")
                                logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: ✅ SAME WORKER - Training worker_id={worker_id} == Plots worker_id={plots_worker_id}")

                                if plot_files_zip:
                               
                                    # Log compression details
                                    file_count = plot_files_zip.get('file_count', 0)
                                    plot_count = plot_files_zip.get('plot_count', 0)
                                    model_count = plot_files_zip.get('model_count', 0)
                                    zip_size = plot_files_zip.get('size', 0)
                                    uncompressed_size = plot_files_zip.get('uncompressed_size', 0)
                                    compression_ratio = plot_files_zip.get('compression_ratio', 1.0)

                                    logger.info(f"📦 Found compressed zip: {file_count} files ({plot_count} plots, {model_count} models)")
                                    logger.info(f"📦 Compression: {uncompressed_size} → {zip_size} bytes (ratio: {compression_ratio:.2f})")

                                    try:
                                        # Decode base64 zip content
                                        zip_content = plot_files_zip.get('content')
                                        if not zip_content:
                                            logger.error(f"❌ No zip content found in training response")
                                        else:
                                            decoded_zip = base64.b64decode(zip_content)

                                            # Create temporary zip file
                                            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                                                tmp_zip.write(decoded_zip)
                                                temp_zip_path = tmp_zip.name

                                            try:
                                                # Extract zip to local directory
                                                with zipfile.ZipFile(temp_zip_path, 'r') as zipf:
                                                    zipf.extractall(local_plots_dir)
                                                    extracted_files = zipf.namelist()

                                                    logger.info(f"✅ Trial {trial.number} files extracted successfully: {len(extracted_files)} files")
                                                    for file_name in extracted_files:
                                                        file_path = local_plots_dir / file_name
                                                        if file_path.exists():
                                                            file_size = file_path.stat().st_size
                                                            file_ext = file_path.suffix.lower()
                                                            file_type = 'model' if file_ext in ['.keras', '.h5', '.pkl'] else 'plot'
                                                            logger.info(f"📂 Extracted {file_type}: {file_name} ({file_size} bytes)")

                                                    logger.info(f"🏗️ WORKER_ISOLATION_TRACKING: ✅ COMPRESSED EXTRACTION SUCCESS - no worker isolation possible")
                                                    self._trials_with_plots.add(trial.number)

                                            finally:
                                                # Clean up temporary zip file
                                                try:
                                                    os.unlink(temp_zip_path)
                                                except Exception as cleanup_e:
                                                    logger.warning(f"⚠️ Failed to cleanup temporary zip file: {cleanup_e}")

                                    except Exception as e:
                                        logger.error(f"❌ Failed to extract compressed files for trial {trial.number}: {e}")
                                else:
                                    logger.warning(f"📂 No compressed files found in training response for trial {trial.number}")
                            else:
                                logger.warning(f"❌ No results_dir configured for trial {trial.number} plot extraction")

                        # Store metrics for standard completion flow in _objective_function
                        # Remove manual completion callback to prevent race conditions
                        self._last_trial_accuracy = metrics.get('test_accuracy', 0.0)
                        self._last_trial_health_metrics = output.get('health_metrics', {})
                        
                        # Check for plots created directly on RunPod (new architecture)
                        if self.config.create_optuna_model_plots:
                            try:
                                plots_direct_info = output.get('plots_direct')
                                if plots_direct_info and plots_direct_info.get('success'):
                                    logger.debug(f"running _train_via_runpod_service ... Trial {trial.number} plots generated directly on RunPod: {plots_direct_info.get('run_name')}")
                                    logger.debug(f"running _train_via_runpod_service ... Available plots: {len(plots_direct_info.get('available_files', []))} files")
                                else:
                                    logger.debug(f"running _train_via_runpod_service ... No direct plots generated for trial {trial.number} (normal - plots downloaded separately)")
                            except Exception as e:
                                logger.error(f"running _train_via_runpod_service ... Failed to check plot status for trial {trial.number}: {e}")
                        
                        return total_score

                    if job_status == 'FAILED':
                        error_logs = status_data.get('error', 'Job failed without details')

                        # Let standard exception handling in _objective_function deal with failure progress
                        raise RuntimeError(f"RunPod job failed: {error_logs}")

                    if job_status in ['IN_QUEUE', 'IN_PROGRESS']:
                        # Report basic progress status during RunPod execution
                        self._report_runpod_progress(trial, job_status, status_data)
                        time.sleep(poll_interval)
                        continue

                    logger.warning(f"running _train_via_runpod_service ... Unknown job status: {job_status}")
                    time.sleep(poll_interval)

                raise RuntimeError(f"RunPod job {job_id} did not complete within {max_poll_time} seconds")

        except Exception as e:
            # ========================================
            # RUNPOD FAILURE AND FALLBACK ANALYSIS
            # ========================================
            logger.error(f"💥 RUNPOD SERVICE FAILURE for trial {trial.number}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error message: {str(e)}")
            logger.error(f"🔍 Failure occurred during RunPod request/response cycle")
            
            # Check fallback configuration
            if self.config.runpod_service_fallback_local:
                logger.warning(f"🔄 LOCAL FALLBACK ENABLED - Switching to local execution")
                logger.warning(f"⚠️ This means trial {trial.number} will run on your local machine instead of RunPod")
                logger.warning(f"📍 CRITICAL: You will see continued local optimization logs after this point")
                logger.warning(f"🔄 STARTING LOCAL FALLBACK for trial {trial.number}")
                
                # Call local fallback
                result = self._train_locally_for_trial(trial, params)
                
                logger.info(f"✅ LOCAL FALLBACK COMPLETED for trial {trial.number}")
                logger.info(f"📊 Fallback result: {result}")
                return result
            else:
                logger.error(f"🚫 LOCAL FALLBACK DISABLED - No fallback will occur")
                logger.error(f"💀 Trial {trial.number} will fail completely")
                logger.error("running _train_via_runpod_service ... RunPod service failed and local fallback disabled")
                raise RuntimeError(f"RunPod service training failed for trial {trial.number}: {e}")

    def _report_runpod_progress(self, trial: optuna.Trial, job_status: str, status_data: Dict[str, Any]) -> None:
        """
        Report progress updates during RunPod service polling
        
        Args:
            trial: Optuna trial object
            job_status: Current job status ('IN_QUEUE' or 'IN_PROGRESS')
            status_data: Full status response from RunPod service
        """
        try:
            # Extract progress updates sent via runpod.serverless.progress_update()
            output = status_data.get('output', {})
            
            # Look for progress updates in various possible locations in the response
            progress_updates = []
            
            # Check if output directly contains progress data (this is where RunPod puts it)
            if output and any(key in output for key in ['current_epoch', 'total_epochs', 'epoch_progress', 'message']):
                # Direct progress data in output
                progress_updates = [output]
            # Check for progress updates array
            elif 'progress_updates' in output:
                progress_updates = output['progress_updates']
            # Check for streaming output that might contain progress
            elif 'stream' in output and isinstance(output['stream'], list):
                progress_updates = output['stream']
            # Check root level for progress info
            elif 'progress' in status_data:
                progress_data = status_data['progress']
                if isinstance(progress_data, dict):
                    progress_updates = [progress_data]
            
            # Create trial progress object
            trial_status = "running" if job_status == 'IN_PROGRESS' else "queued"
            
            # Extract epoch information from the latest progress update
            current_epoch = None
            total_epochs = None
            epoch_progress = None
            latest_update = None  # Initialize to avoid variable scope issues

            if progress_updates and isinstance(progress_updates, list) and len(progress_updates) > 0:
                # Get the most recent progress update
                latest_update = progress_updates[-1]
                
                if isinstance(latest_update, dict):
                    # Look for epoch information in the progress update
                    current_epoch = latest_update.get('current_epoch')
                    total_epochs = latest_update.get('total_epochs') 
                    epoch_progress = latest_update.get('epoch_progress')
                    
                    # Also check if it's a string message that we need to parse
                    if current_epoch is None and 'message' in latest_update:
                        message = latest_update['message']
                        current_epoch, total_epochs, epoch_progress = self._parse_progress_message(message)
                elif isinstance(latest_update, str):
                    # If the update is a string message, try to parse epoch info from it
                    current_epoch, total_epochs, epoch_progress = self._parse_progress_message(latest_update)
                
                if current_epoch is not None:
                    logger.debug(f"running _report_runpod_progress ... Found progress from RunPod: epoch {current_epoch}/{total_epochs}, progress {epoch_progress:.1%}")

                # Check for real-time plot downloads (only if latest_update is a dict)
                if isinstance(latest_update, dict):
                    # Downloads are handled automatically by API server
                    pass
            
            # Log what we found (or didn't find) for debugging
            '''
            logger.info(f"🔍 PROGRESS EXTRACT - Trial {trial.number} - Extracted progress: epoch {current_epoch}/{total_epochs}, progress {epoch_progress}")
            logger.info(f"🔍 PROGRESS EXTRACT - Progress updates found: {len(progress_updates) if progress_updates else 0}")
            if progress_updates:
                logger.info(f"🔍 PROGRESS EXTRACT - Latest update: {progress_updates[-1]}")
            else:
                logger.info(f"🔍 PROGRESS EXTRACT - No progress updates found in any expected location")
                logger.info(f"🔍 PROGRESS EXTRACT - Full status_data keys: {list(status_data.keys())}")
                logger.info(f"🔍 PROGRESS EXTRACT - Output keys: {list(output.keys()) if output else 'No output'}")
                logger.info(f"🔍 PROGRESS EXTRACT - Full status_data: {json.dumps(status_data, indent=2)}")
            '''
            
            # Extract plot generation data for real-time updates
            plot_generation_data = None
            if isinstance(latest_update, dict):
                plot_generation = latest_update.get('plot_generation')
                if plot_generation and isinstance(plot_generation, dict):
                    plot_generation_data = plot_generation

            # Create TrialProgress object with available information including plot generation
            trial_progress = TrialProgress(
                trial_id=f"trial_{trial.number}",
                trial_number=trial.number,
                status=trial_status,
                current_epoch=current_epoch,
                total_epochs=total_epochs,
                epoch_progress=epoch_progress,
                started_at=datetime.now().isoformat(),
                plot_generation=plot_generation_data
            )
            
            # Update the _current_epoch_info dictionary that _create_unified_progress relies on
            if current_epoch is not None and total_epochs is not None:
                self._current_epoch_info[trial.number] = {
                    'current_epoch': current_epoch,
                    'total_epochs': total_epochs,
                    'epoch_progress': epoch_progress
                }
                logger.debug(f"running _report_runpod_progress ... Updated epoch info for trial {trial.number}: {current_epoch}/{total_epochs} (progress: {epoch_progress})")

            # Report progress for IN_PROGRESS status to keep UI updated
            if job_status == 'IN_PROGRESS':
                logger.debug(f"running _report_runpod_progress ... Reporting progress for trial {trial.number}: {trial_status}")
                self._thread_safe_progress_callback(trial_progress)
        
        except Exception as e:
            logger.warning(f"running _report_runpod_progress ... Error reporting RunPod progress for trial {trial.number}: {e}")

    def _parse_progress_message(self, message: str) -> tuple:
        """
        Parse progress information from RunPod progress update messages
        
        Extracts epoch and batch progress from Keras training output like:
        "Epoch 2/6: 1374/1400 [============================>.] - ETA: 0s"
        
        Args:
            message: Progress message string from RunPod
            
        Returns:
            Tuple of (current_epoch, total_epochs, epoch_progress)
        """
        try:
            current_epoch = None
            total_epochs = None
            epoch_progress = None
            
            # Pattern to match "Epoch X/Y" 
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', message)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
            
            # Pattern to match batch progress "1374/1400 [====...===>.] - ETA: 0s"
            batch_match = re.search(r'(\d+)/(\d+)\s+\[', message)
            if batch_match:
                current_batch = int(batch_match.group(1))
                total_batches = int(batch_match.group(2))
                epoch_progress = round(min(current_batch / total_batches, 1.0), 2)
            
            # Alternative: look for percentage in message
            percent_match = re.search(r'(\d+(?:\.\d+)?)%', message)
            if percent_match and epoch_progress is None:
                epoch_progress = round(float(percent_match.group(1)) / 100.0, 2)
            
            logger.debug(f"running _parse_progress_message ... Parsed '{message}' -> epoch {current_epoch}/{total_epochs}, progress {epoch_progress}")
            
            return current_epoch, total_epochs, epoch_progress
            
        except Exception as e:
            logger.debug(f"running _parse_progress_message ... Failed to parse message '{message}': {e}")
            return None, None, None

    def _calculate_total_score_from_service_response(
        self, 
        metrics: Dict[str, Any], 
        full_result: Dict[str, Any], 
        trial: optuna.Trial
    ) -> float:
        """
        Calculate total score from RunPod service response
        
        Processes the comprehensive metrics returned by the RunPod service
        (which uses the same HealthAnalyzer logic as local execution).
        
        Args:
            metrics: Metrics dictionary from service response
            full_result: Full response from service (for additional data)
            trial: Optuna trial object
            
        Returns:
            Total score for optimization
        """
        logger.debug(f"running _calculate_total_score_from_service_response ... Calculating total score for trial {trial.number}")
        logger.debug(f"running _calculate_total_score_from_service_response ... Available metrics: {list(metrics.keys())}")
        
        try:
            # Extract basic metrics
            test_accuracy = metrics.get('test_accuracy', 0.0)
            test_loss = metrics.get('test_loss', 0.0)
            training_time_seconds = metrics.get('training_time_seconds', 0.0)
            
            # Extract health metrics if available
            health_metrics = full_result.get('health_metrics', {})
            overall_health = health_metrics.get('overall_health', 0.5)
            
            logger.debug(f"running _calculate_total_score_from_service_response ... Basic metrics: acc={test_accuracy:.4f}, loss={test_loss:.4f}")
            logger.debug(f"running _calculate_total_score_from_service_response ... Health metrics: overall={overall_health:.3f}")
            
            # Store the raw accuracy and health metrics for trial progress
            self._last_trial_accuracy = test_accuracy
            # Combine basic metrics with health metrics for comprehensive view
            comprehensive_service_metrics = health_metrics.copy()
            comprehensive_service_metrics.update({
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'training_time_seconds': training_time_seconds
            })
            self._last_trial_health_metrics = comprehensive_service_metrics
            
            # 🔍 DEBUG: Log all health metrics from service response for frontend display
            logger.info(f"🔍 SERVICE HEALTH METRICS DEBUG - Trial {trial.number}:")
            logger.info(f"  📊 Service comprehensive_service_metrics keys: {list(comprehensive_service_metrics.keys())}")
            logger.info(f"  📊 test_loss: {comprehensive_service_metrics.get('test_loss', 'MISSING')}")
            logger.info(f"  📊 test_accuracy: {comprehensive_service_metrics.get('test_accuracy', 'MISSING')}")
            logger.info(f"  📊 overall_health: {comprehensive_service_metrics.get('overall_health', 'MISSING')}")
            logger.info(f"  📊 neuron_utilization: {comprehensive_service_metrics.get('neuron_utilization', 'MISSING')}")
            logger.info(f"  📊 parameter_efficiency: {comprehensive_service_metrics.get('parameter_efficiency', 'MISSING')}")
            logger.info(f"  📊 training_stability: {comprehensive_service_metrics.get('training_stability', 'MISSING')}")
            logger.info(f"  📊 gradient_health: {comprehensive_service_metrics.get('gradient_health', 'MISSING')}")
            logger.info(f"  📊 convergence_quality: {comprehensive_service_metrics.get('convergence_quality', 'MISSING')}")
            logger.info(f"  📊 accuracy_consistency: {comprehensive_service_metrics.get('accuracy_consistency', 'MISSING')}")
            logger.info(f"  📊 Complete comprehensive_service_metrics: {comprehensive_service_metrics}")
            
            # Calculate total score based on optimization mode and target
            if self.config.objective == OptimizationObjective.VAL_ACCURACY:
                primary_value = test_accuracy
                
                if self.config.mode == OptimizationMode.HEALTH and not OptimizationObjective.is_health_only(self.config.objective):
                    # Weighted combination
                    objective_weight = 1.0 - self.config.health_weight
                    health_weight = self.config.health_weight
                    final_value = objective_weight * primary_value + health_weight * overall_health
                    
                    logger.debug(f"running _calculate_total_score_from_service_response ... HEALTH mode weighted combination:")
                    logger.debug(f"running _calculate_total_score_from_service_response ... - Primary (acc): {primary_value:.4f} * {objective_weight:.1f} = {primary_value * objective_weight:.4f}")
                    logger.debug(f"running _calculate_total_score_from_service_response ... - Health: {overall_health:.3f} * {health_weight:.1f} = {overall_health * health_weight:.4f}")
                    logger.debug(f"running _calculate_total_score_from_service_response ... - Final: {final_value:.4f}")
                else:
                    final_value = primary_value
                    logger.debug(f"running _calculate_total_score_from_service_response ... SIMPLE mode: using primary value {final_value:.4f}")
                
                return float(final_value)
            
            elif self.config.objective == OptimizationObjective.ACCURACY:
                primary_value = test_accuracy
                
                if self.config.mode == OptimizationMode.HEALTH and not OptimizationObjective.is_health_only(self.config.objective):
                    objective_weight = 1.0 - self.config.health_weight
                    health_weight = self.config.health_weight
                    final_value = objective_weight * primary_value + health_weight * overall_health
                else:
                    final_value = primary_value
                
                return float(final_value)
            
            elif self.config.objective == OptimizationObjective.PARAMETER_EFFICIENCY:
                parameter_efficiency = health_metrics.get('parameter_efficiency', 0.0)
                
                if self.config.mode == OptimizationMode.HEALTH and not OptimizationObjective.is_health_only(self.config.objective):
                    objective_weight = 1.0 - self.config.health_weight
                    health_weight = self.config.health_weight
                    final_value = objective_weight * parameter_efficiency + health_weight * overall_health
                else:
                    final_value = parameter_efficiency
                
                return float(final_value)
            
            elif self.config.objective == OptimizationObjective.TRAINING_TIME:
                if training_time_seconds > 0:
                    training_time_minutes = training_time_seconds / 60.0
                    time_efficiency = 1.0 / (1.0 + training_time_minutes / 10.0)
                else:
                    time_efficiency = 1.0
                    
                if self.config.mode == OptimizationMode.HEALTH and not OptimizationObjective.is_health_only(self.config.objective):
                    objective_weight = 1.0 - self.config.health_weight
                    health_weight = self.config.health_weight
                    final_value = objective_weight * time_efficiency + health_weight * overall_health
                else:
                    final_value = time_efficiency
                    
                return float(final_value)
            
            # Health-only objectives
            elif self.config.objective == OptimizationObjective.OVERALL_HEALTH:
                if self.config.mode != OptimizationMode.HEALTH:
                    raise ValueError(f"Health-only objective '{self.config.optimize_for}' requires HEALTH mode")
                return float(overall_health)
            
            elif self.config.objective == OptimizationObjective.NEURON_UTILIZATION:
                if self.config.mode != OptimizationMode.HEALTH:
                    raise ValueError(f"Health-only objective '{self.config.optimize_for}' requires HEALTH mode")
                return float(health_metrics.get('neuron_utilization', 0.5))
            
            elif self.config.objective == OptimizationObjective.TRAINING_STABILITY:
                if self.config.mode != OptimizationMode.HEALTH:
                    raise ValueError(f"Health-only objective '{self.config.optimize_for}' requires HEALTH mode")
                return float(health_metrics.get('training_stability', 0.5))
            
            elif self.config.objective == OptimizationObjective.GRADIENT_HEALTH:
                if self.config.mode != OptimizationMode.HEALTH:
                    raise ValueError(f"Health-only objective '{self.config.optimize_for}' requires HEALTH mode")
                return float(health_metrics.get('gradient_health', 0.5))
            
            else:
                # Fallback to test accuracy
                logger.warning(f"running _calculate_total_score_from_service_response ... Unknown objective {self.config.optimize_for}, using test_accuracy")
                return float(test_accuracy)
            
        except Exception as e:
            logger.error(f"running _calculate_total_score_from_service_response ... Error calculating total score: {e}")
            return float(metrics.get('test_accuracy', 0.0))
    
    def _train_locally_for_trial(self, trial: optuna.Trial, params: Dict[str, Any]) -> float:
        """
        Fallback method to train locally when RunPod service fails
        
        This maintains the original local training logic as a fallback.
        
        Args:
            trial: Optuna trial object
            params: Hyperparameters for this trial
            
        Returns:
            Objective value for optimization
        """
        logger.debug(f"running _train_locally_for_trial ... Starting local training fallback for trial {trial.number}")
        
        try:
            trial_start_time = time.time()
            
            # Create ModelConfig from suggested parameters
            # 
            # This implements the core of the dual-purpose ModelConfig system:
            # 1. Start with ModelConfig() containing sensible defaults 
            # 2. Override defaults with Optuna-suggested parameters
            # 3. Parameters not suggested by Optuna retain their default values
            # 4. Result is a fully-configured ModelConfig ready for ModelBuilder
            #
            # Example flow:
            # - ModelConfig.use_global_pooling = False (default)
            # - params = {'use_global_pooling': True, 'kernel_size': (5,5), ...} (from Optuna)
            # - After setattr loop: ModelConfig.use_global_pooling = True (overridden)
            #                      ModelConfig.activation = 'relu' (default retained)
            model_config = ModelConfig()
            
            # Apply all suggested parameters, overriding defaults
            for key, value in params.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            
            # Configuration parameters from OptimizationConfig to ModelConfig
            config_to_model_params = [
                'gpu_proxy_sample_percentage',
                'validation_split',
                # Add other config params that ModelConfig needs
            ]
            
            # 🎯 STEP 4.1f VERIFICATION: Enhanced parameter transfer logging
            logger.debug(f"running _train_locally_for_trial ... 🔄 PARAMETER TRANSFER VERIFICATION:")
            logger.debug(f"running _train_locally_for_trial ... OptimizationConfig.gpu_proxy_sample_percentage: {self.config.gpu_proxy_sample_percentage}")

            for param_name in config_to_model_params:
                if hasattr(self.config, param_name) and hasattr(model_config, param_name):
                    config_value = getattr(self.config, param_name)
                    setattr(model_config, param_name, config_value)
                    
                    # Verify the transfer worked
                    model_value = getattr(model_config, param_name)
                    logger.debug(f"running _train_locally_for_trial ... ✅ {param_name}: {config_value} → ModelConfig (verified: {model_value})")
                    
                    # Extra verification for gpu_proxy_sample_percentage
                    if param_name == 'gpu_proxy_sample_percentage':
                        logger.debug(f"running _train_locally_for_trial ... 🎯 GPU_PROXY_SAMPLE_PERCENTAGE TRANSFER VERIFICATION:")
                        logger.debug(f"running _train_locally_for_trial ... - Source (OptimizationConfig): {config_value}")
                        logger.debug(f"running _train_locally_for_trial ... - Target (ModelConfig): {model_value}")
                        logger.debug(f"running _train_locally_for_trial ... - Transfer success: {config_value == model_value}")
                else:
                    logger.warning(f"running _train_locally_for_trial ... ⚠️ Parameter transfer failed: {param_name}")
                    if not hasattr(self.config, param_name):
                        logger.warning(f"running _train_locally_for_trial ... - OptimizationConfig missing: {param_name}")
                    if not hasattr(model_config, param_name):
                        logger.warning(f"running _train_locally_for_trial ... - ModelConfig missing: {param_name}")

            # 🎯 FINAL VERIFICATION: Log ModelConfig state before ModelBuilder creation
            logger.debug(f"running _train_locally_for_trial ... 🎯 FINAL MODELCONFIG VERIFICATION:")
            logger.debug(f"running _train_locally_for_trial ... - ModelConfig.gpu_proxy_sample_percentage: {model_config.gpu_proxy_sample_percentage}")
            logger.debug(f"running _train_locally_for_trial ... - ModelConfig.validation_split: {model_config.validation_split}")
            logger.debug(f"running _train_locally_for_trial ... - Creating ModelBuilder with verified config...")
            
            # Create ModelBuilder with trial number, shared timestamp, results directory, and optimization config
            model_builder = ModelBuilder(self.dataset_config, self.run_timestamp, model_config, trial.number, self.results_dir, self.config)

            # Store for later access to plot generation data
            self._current_model_builder = model_builder
            
            # Build model
            model_builder.build_model()
            
            if model_builder.model is None:
                raise RuntimeError("Model building failed - model is None")
            
            # Prepare data
            training_data = {
                'x_train': self.data['x_train'],
                'y_train': self.data['y_train'],
                'x_test': self.data['x_test'],
                'y_test': self.data['y_test']
            }
            
            # Handle epochs
            trial_epochs = params.get('epochs', self.config.max_epochs_per_trial)
            if not isinstance(trial_epochs, int):
                trial_epochs = int(trial_epochs)
            
            min_epochs = self.config.min_epochs_per_trial
            max_epochs = self.config.max_epochs_per_trial
            final_epochs = max(min_epochs, min(trial_epochs, max_epochs))
            
            model_builder.model_config.epochs = final_epochs
            
            # Add epoch progress callback to track training progress
            epoch_callback = EpochProgressCallback(
                trial_number=trial.number,
                total_epochs=final_epochs,
                optimizer_instance=self
            )
            
            # Add the callback to model_builder's callback setup method
            original_setup_callbacks = model_builder._setup_training_callbacks_optimized
            def enhanced_setup_callbacks():
                callbacks_list = original_setup_callbacks()
                callbacks_list.append(epoch_callback)
                return callbacks_list
            model_builder._setup_training_callbacks_optimized = enhanced_setup_callbacks
            
            # Create plot progress callback
            def plot_progress_callback(current_plot_name: str, completed_plots: int, total_plots: int, overall_progress: float):
                """Update trial progress with plot generation status"""
                plot_generation_data = {
                    "status": "generating",
                    "current_plot": current_plot_name,
                    "completed_plots": completed_plots,
                    "total_plots": total_plots,
                    "plot_progress": overall_progress
                }

                # Update current trial progress with plot generation status
                if self._current_trial_progress:
                    self._current_trial_progress.plot_generation = plot_generation_data

                logger.debug(f"running plot_progress_callback ... Trial {trial.number}: Plot progress - {current_plot_name} ({completed_plots}/{total_plots}, {overall_progress*100:.1f}%)")

            # Train the model
            history = model_builder.train(
                data=training_data,
                validation_split=self.config.validation_split,
                plot_progress_callback=plot_progress_callback
            )
            
            # Calculate training time
            training_time_minutes = (time.time() - trial_start_time) / 60
            
            # Mark plot generation as completed and include uploaded files for real-time download
            if self._current_trial_progress:
                plot_generation_data = {
                    "status": "completed",
                    "current_plot": "All plots complete",
                    "completed_plots": self._current_trial_progress.plot_generation.get("total_plots", 0) if self._current_trial_progress.plot_generation else 0,
                    "total_plots": self._current_trial_progress.plot_generation.get("total_plots", 0) if self._current_trial_progress.plot_generation else 0,
                    "plot_progress": 1.0
                }

                # Try to get uploaded files from model_builder if available
                if hasattr(model_builder, '_last_plot_s3_result') and model_builder._last_plot_s3_result:
                    plot_generation_data["uploaded_files"] = model_builder._last_plot_s3_result.get("uploaded_files", [])
                    plot_generation_data["s3_prefix"] = model_builder._last_plot_s3_result.get("s3_prefix", "")
                    plot_generation_data["bucket"] = model_builder._last_plot_s3_result.get("bucket", "")
                    logger.debug(f"running _train_locally_for_trial ... Including {len(plot_generation_data.get('uploaded_files', []))} uploaded files in progress update")

                self._current_trial_progress.plot_generation = plot_generation_data
                
                # Update the trial progress in the history to include the completed plot generation
                logger.debug(f"running _train_locally_for_trial ... Updating trial {trial.number} in history with completed plot generation")
                self._thread_safe_progress_callback(self._current_trial_progress)

            # Get comprehensive metrics using HealthAnalyzer
            comprehensive_metrics = model_builder.health_analyzer.calculate_comprehensive_health(
                model=model_builder.model,
                history=history,
                data=training_data,
                sample_data=training_data['x_test'][:50] if len(training_data['x_test']) > 50 else training_data['x_test'],
                training_time_minutes=training_time_minutes,
                total_params=model_builder.model.count_params()
            )
            
            # Extract metrics
            test_accuracy = comprehensive_metrics.get('test_accuracy', 0.0)
            test_loss = comprehensive_metrics.get('test_loss', 0.0)
            overall_health = comprehensive_metrics.get('overall_health', 0.5)
            
            logger.debug(f"running _train_locally_for_trial ... Trial {trial.number}: Local fallback completed")
            logger.debug(f"running _train_locally_for_trial ... - Test accuracy: {test_accuracy:.4f}")
            logger.debug(f"running _train_locally_for_trial ... - Overall health: {overall_health:.3f}")
            
            # Store the raw accuracy and comprehensive health metrics for trial progress
            self._last_trial_accuracy = test_accuracy
            self._last_trial_health_metrics = comprehensive_metrics

            # Save trial model for potential use in final model building (replicate RunPod approach)
            self._save_trial_model(trial.number, model_builder.model, params, test_accuracy, comprehensive_metrics)

            # 🔍 DEBUG: Log all health metrics being stored for frontend display
            logger.info(f"🔍 HEALTH METRICS DEBUG - Trial {trial.number}:")
            logger.info(f"  📊 Raw comprehensive_metrics keys: {list(comprehensive_metrics.keys())}")
            logger.info(f"  📊 test_loss: {comprehensive_metrics.get('test_loss', 'MISSING')}")
            logger.info(f"  📊 test_accuracy: {comprehensive_metrics.get('test_accuracy', 'MISSING')}")
            logger.info(f"  📊 overall_health: {comprehensive_metrics.get('overall_health', 'MISSING')}")
            logger.info(f"  📊 neuron_utilization: {comprehensive_metrics.get('neuron_utilization', 'MISSING')}")
            logger.info(f"  📊 parameter_efficiency: {comprehensive_metrics.get('parameter_efficiency', 'MISSING')}")
            logger.info(f"  📊 training_stability: {comprehensive_metrics.get('training_stability', 'MISSING')}")
            logger.info(f"  📊 gradient_health: {comprehensive_metrics.get('gradient_health', 'MISSING')}")
            logger.info(f"  📊 convergence_quality: {comprehensive_metrics.get('convergence_quality', 'MISSING')}")
            logger.info(f"  📊 accuracy_consistency: {comprehensive_metrics.get('accuracy_consistency', 'MISSING')}")
            logger.info(f"  📊 Complete comprehensive_metrics: {comprehensive_metrics}")
            
            # Calculate objective (reuse same logic as service response)
            if self.config.objective == OptimizationObjective.VAL_ACCURACY:
                if self.config.mode == OptimizationMode.HEALTH and not OptimizationObjective.is_health_only(self.config.objective):
                    objective_weight = 1.0 - self.config.health_weight
                    health_weight = self.config.health_weight
                    total_score = round(objective_weight * test_accuracy + health_weight * overall_health, 4)
                else:
                    total_score = round(test_accuracy, 4)
            else:
                # For other objectives, use test_accuracy as fallback
                total_score = round(test_accuracy, 4)
            
            logger.debug(f"running _train_locally_for_trial ... Trial {trial.number}: Local fallback total score: {total_score:.4f}")
            
            # Generate plots if enabled
            if self.config.create_optuna_model_plots:
                try:
                    logger.debug(f"running _train_locally_for_trial ... Generating plots for trial {trial.number}")
                    if not self.results_dir:
                        logger.error(f"running _train_locally_for_trial ... Results directory not set, cannot generate plots")
                        return float(total_score)
                    # Plots will be generated by model_builder.py after training completion
                    logger.debug(f"running _objective ... Plots will be generated by model_builder, skipping duplicate generation")
                except Exception as e:
                    logger.error(f"running _train_locally_for_trial ... Failed to generate plots for trial {trial.number}: {e}")
            
            return float(total_score)
            
        except Exception as e:
            logger.error(f"running _train_locally_for_trial ... Local training fallback failed for trial {trial.number}: {e}")
            raise

    def _save_trial_model(self, trial_number: int, model: keras.Model, params: Dict[str, Any], test_accuracy: float, comprehensive_metrics: Dict[str, Any]) -> None:
        """
        Save trial model to disk for later use in final model building.
        This replicates the RunPod approach where trial models are preserved.

        Args:
            trial_number: Optuna trial number
            model: Trained Keras model
            params: Hyperparameters used for this trial
            test_accuracy: Final test accuracy
            comprehensive_metrics: Complete health metrics
        """
        try:
            # Check if results_dir is available
            if self.results_dir is None:
                logger.error(f"running _save_trial_model ... Results directory not set, cannot save trial {trial_number} model")
                return

            # Create plots directory structure to match RunPod approach
            plots_dir = Path(self.results_dir) / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Create trial-specific directory (matching RunPod pattern)
            trial_dir = plots_dir / f"trial_{trial_number}"
            trial_dir.mkdir(exist_ok=True)

            # Save the model in Keras format (matching RunPod filename pattern)
            model_filename = f"final_model_{self.dataset_config.name}_trial_{trial_number}.keras"
            model_path = trial_dir / model_filename
            model.save(str(model_path))

            # Save metadata about this trial
            metadata = {
                "trial_number": trial_number,
                "model_filename": model_filename,
                "test_accuracy": test_accuracy,
                "hyperparameters": params,
                "comprehensive_metrics": comprehensive_metrics,
                "timestamp": datetime.now().isoformat(),
                "run_name": self.run_name
            }

            metadata_path = trial_dir / f"trial_{trial_number}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.debug(f"running _save_trial_model ... Trial {trial_number}: Model saved to {model_path}")
            logger.debug(f"running _save_trial_model ... Trial {trial_number}: Metadata saved to {metadata_path}")

        except Exception as e:
            logger.error(f"running _save_trial_model ... Failed to save trial {trial_number} model: {e}")
            # Don't raise - this should not interrupt optimization

    def optimize(self) -> OptimizationResult:
        """
        Run optimization study to find best hyperparameters
        
        Returns:
            OptimizationResult with best parameters and comprehensive metrics
        """
        logger.debug(f"running ModelOptimizer.optimize ... Starting optimization for {self.dataset_name}")
        logger.debug(f"running ModelOptimizer.optimize ... Mode: {self.config.mode}")
        logger.debug(f"running ModelOptimizer.optimize ... Objective: {self.config.optimize_for}")
        logger.debug(f"running ModelOptimizer.optimize ... Trials: {self.config.trials}")
        
        # Log execution approach
        if self.config.use_runpod_service:
            logger.debug(f"running ModelOptimizer.optimize ... 🔄 EXECUTION: RunPod Service (JSON API, tiny payloads)")
            logger.debug(f"running ModelOptimizer.optimize ... Endpoint: {self.config.runpod_service_endpoint}")
        else:
            logger.debug(f"running ModelOptimizer.optimize ... EXECUTION: Local only")
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(n_startup_trials=self.config.startup_trials, seed=self.config.random_seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.config.startup_trials,
                n_warmup_steps=self.config.warmup_steps
            )
        )
        
        # Record optimization start time
        self.optimization_start_time = time.time()
        
        # Show "Initializing GPU" status for RunPod service before trials begin
        if self.config.use_runpod_service and self.progress_callback:
            gpu_init_progress = UnifiedProgress(
                total_trials=self.config.trials,
                running_trials=[],
                completed_trials=[],
                failed_trials=[],
                current_best_total_score=None,
                current_best_total_score_trial_number=None,
                current_best_accuracy=None,
                average_duration_per_trial=None,
                estimated_time_remaining=None,
                current_epoch=None,
                total_epochs=None,
                epoch_progress=None,
                current_trial_id=None,
                current_trial_status="initializing",
                status_message="Initializing GPU resources..."
            )
            logger.debug(f"running ModelOptimizer.optimize ... Showing 'Initializing GPU' status")
            self.progress_callback(gpu_init_progress)
        
        # Determine execution mode: sequential vs concurrent
        if self.config.use_runpod_service and self.config.concurrent:
            # RunPod mode with concurrent workers
            n_jobs = min(self.config.concurrent_workers, self.config.trials)
            logger.debug(f"running ModelOptimizer.optimize ... RunPod CONCURRENT mode: {n_jobs} workers for {self.config.trials} trials")
        else:
            # Sequential mode (either local execution or RunPod sequential)
            n_jobs = 1
            if self.config.use_runpod_service:
                logger.debug(f"running ModelOptimizer.optimize ... RunPod SEQUENTIAL mode: trials will run one-by-one")
            else:
                logger.debug(f"running ModelOptimizer.optimize ... LOCAL SEQUENTIAL mode: trials will run locally one-by-one")


        # Run optimization
        try:
            self.study.optimize(
                self._objective_function,
                n_trials=self.config.trials,
                n_jobs=n_jobs,
                catch=(Exception,),          # ← don't let a single trial kill the study
                gc_after_trial=True          # ← reclaim memory between concurrent trials
            )
            logger.debug(f"running ModelOptimizer.optimize ... Completed {len(self.study.trials)} trials")
        except KeyboardInterrupt:
            logger.warning("running ModelOptimizer.optimize ... Optimization interrupted by user")
        except Exception as e:
            logger.error(f"running ModelOptimizer.optimize ... Optimization failed: {e}")
            raise

        
        # Compile results
        results = self._compile_results()
        
        # Save optimization results
        self._save_results(results)

        # Build final model automatically (always enabled with copy-based approach)
        logger.debug("running ModelOptimizer.optimize ... Building final model automatically via copy-based approach")
        try:
            final_model_path = self._build_final_model(results)
            if final_model_path:
                logger.info(f"✅ Final model built and saved automatically to: {final_model_path}")
                results.best_model_path = final_model_path
            else:
                logger.warning("⚠️ Final model building completed but no model path returned")
        except Exception as e:
            logger.error(f"❌ Automatic final model building failed: {e}")
            logger.error("💡 Final model can still be built manually via /build_final_model endpoint")
        
        logger.debug(f"running ModelOptimizer.optimize ... Optimization completed successfully")
        
        return results
    
    def _objective_function(self, trial: optuna.Trial) -> float:
        """
        Objective function with RunPod service integration
        
        Now checks execution method:
        1. RunPod service (JSON API approach) - if enabled and configured
        2. Local execution - fallback or when service disabled
        
        Args:
            trial: Optuna trial object for parameter suggestion
            
        Returns:
            Objective value (higher is better for maximization objectives)
        """
        # Check for cancellation at the start of each trial
        if self.is_cancelled():
            logger.info(f"running ModelOptimizer._objective_function ... Trial {trial.number} cancelled")
            raise optuna.TrialPruned()
        
        trial_start_time = time.time()
    
        # Track trial start in progress aggregation (basic info only, epochs added later)
        if self.progress_callback:
            trial_progress = TrialProgress(
                trial_id=f"trial_{trial.number}",
                trial_number=trial.number,
                status="running",
                started_at=datetime.now().isoformat()
            )
            self._thread_safe_progress_callback(trial_progress)
        
        # Record trial start time for ETA calculation
        with self._progress_lock:
            self._trial_start_times[trial.number] = trial_start_time
        
        params = None  # Initialize to avoid "possibly unbound" errors in except block
        try:
            logger.debug(
                f"running _objective_function ... start "
                f"trial={trial.number} "
                f"mode={'runpod' if self.config.use_runpod_service else 'local'} "
                f"concurrent={self.config.concurrent} "
                f"workers={self.config.concurrent_workers}"
            )
            
            # Per-trial deterministic seeding (thread-safe for concurrent trials)
            try:
                base_seed: int = int(self.config.random_seed) if self.config.random_seed is not None else 0
            except Exception:
                base_seed = 0
            seed_value: int = (base_seed + int(trial.number)) & 0x7FFFFFFF  # keep in 32-bit range

            logger.debug(f"running _objective_function ... setting per-trial seed={seed_value} for trial={trial.number}")

            random.seed(seed_value)
            np.random.seed(seed_value)

            try:
                # If TensorFlow is in use, seed it as well
                tf.random.set_seed(seed_value)  # deterministic graph-level seed
            except Exception:
                # Silent if TF not installed/needed in this run context
                pass

                   
            # Use modular hyperparameter suggestion
            params = self._suggest_hyperparameters(trial)
            
            # Update trial progress with epoch information now that we have params
            if self.progress_callback:
                # Get expected epochs from params or config
                trial_epochs = params.get('epochs', self.config.max_epochs_per_trial)
                if not isinstance(trial_epochs, int):
                    trial_epochs = int(trial_epochs)
                
                min_epochs = self.config.min_epochs_per_trial
                max_epochs = self.config.max_epochs_per_trial
                final_epochs = max(min_epochs, min(trial_epochs, max_epochs))
                
                # Extract architecture information for display
                architecture_info = self._extract_architecture_info(params)
                
                trial_progress = TrialProgress(
                    trial_id=f"trial_{trial.number}",
                    trial_number=trial.number,
                    status="running",
                    started_at=datetime.now().isoformat(),
                    current_epoch=0,
                    total_epochs=final_epochs,
                    epoch_progress=0.0,
                    architecture=architecture_info,
                    hyperparameters=params
                )
                self._thread_safe_progress_callback(trial_progress)
            
            # Train using automatically determined execution method
            total_score = self.train(trial, params)

            # Store the last model builder for plot generation data extraction
            # This is needed for RunPod execution to access plot S3 results
            if hasattr(self, '_current_model_builder'):
                self._last_model_builder = self._current_model_builder
            
            # Track trial completion in progress aggregation
            if self.progress_callback:
                trial_end_time = time.time()
                # Extract architecture information for display
                architecture_info = self._extract_architecture_info(params)
                
                # Extract plot generation data if available (for RunPod execution)
                plot_generation_data = None
                if hasattr(self, '_last_model_builder') and self._last_model_builder:
                    model_builder = self._last_model_builder
                    if hasattr(model_builder, '_last_plot_s3_result') and model_builder._last_plot_s3_result:
                        s3_result = model_builder._last_plot_s3_result
                        plot_generation_data = {
                            'status': 'completed',
                            'uploaded_files': s3_result.get('uploaded_files', []),
                            's3_prefix': s3_result.get('s3_prefix', ''),
                            'bucket': s3_result.get('bucket', ''),
                            'total_plots': len(s3_result.get('uploaded_files', [])),
                            'completed_plots': len(s3_result.get('uploaded_files', [])),
                            'current_plot': 'All plots complete',
                            'plot_progress': 1.0
                        }
                        logger.debug(f"🎨 TRIAL COMPLETION - Including plot generation data with {len(plot_generation_data['uploaded_files'])} uploaded files")

                trial_progress = TrialProgress(
                    trial_id=f"trial_{trial.number}",
                    trial_number=trial.number,
                    status="completed",
                    started_at=datetime.fromtimestamp(trial_start_time).isoformat(),
                    completed_at=datetime.now().isoformat(),
                    duration_seconds=round(trial_end_time - trial_start_time),
                    architecture=architecture_info,
                    hyperparameters=params,
                    performance={
                        'total_score': total_score,
                        'accuracy': self._last_trial_accuracy
                    },
                    health_metrics=getattr(self, '_last_trial_health_metrics', None),
                    plot_generation=plot_generation_data
                )
                
                # 🔍 DEBUG: Log what's actually in the TrialProgress being sent to frontend
                logger.info(f"🚀 TRIAL PROGRESS TO FRONTEND - Trial {trial.number}:")
                logger.info(f"  📊 performance: {trial_progress.performance}")
                logger.info(f"  📊 health_metrics: {trial_progress.health_metrics}")
                logger.info(f"  📊 health_metrics type: {type(trial_progress.health_metrics)}")
                if trial_progress.health_metrics:
                    logger.info(f"  📊 health_metrics keys: {list(trial_progress.health_metrics.keys())}")
                    logger.info(f"  📊 convergence_quality in health_metrics: {trial_progress.health_metrics.get('convergence_quality', 'MISSING')}")
                    logger.info(f"  📊 accuracy_consistency in health_metrics: {trial_progress.health_metrics.get('accuracy_consistency', 'MISSING')}")
                logger.info(f"  📊 Complete trial_progress.to_dict(): {trial_progress.to_dict()}")
                self._thread_safe_progress_callback(trial_progress)
            
            # Update best trial tracking with both total score and raw accuracy
            self.update_best_trial(trial.number, total_score, self._last_trial_accuracy)
            
            return total_score
            
        except Exception as e:
            # Track trial failure in progress aggregation
            if self.progress_callback:
                # Extract architecture information if params are available
                architecture_info = None
                hyperparameters = None
                try:
                    if params is not None:
                        architecture_info = self._extract_architecture_info(params)
                        hyperparameters = params
                except:
                    pass  # If architecture extraction fails, proceed without it
                
                trial_progress = TrialProgress(
                    trial_id=f"trial_{trial.number}",
                    trial_number=trial.number,
                    status="failed",
                    started_at=datetime.fromtimestamp(trial_start_time).isoformat(),
                    completed_at=datetime.now().isoformat(),
                    duration_seconds=round(time.time() - trial_start_time),
                    architecture=architecture_info,
                    hyperparameters=hyperparameters
                )
                self._thread_safe_progress_callback(trial_progress)
            
            logger.error(f"running ModelOptimizer._objective_function ... Trial {trial.number} failed: {str(e)}")
            logger.error(f"running ModelOptimizer._objective_function ... Traceback: {traceback.format_exc()}")
            raise

    def _compile_results(self) -> OptimizationResult:
        """Compile optimization results into structured format"""
        if self.study is None:
            raise RuntimeError("No study available - run optimize() first")
        
        optimization_time = time.time() - self.optimization_start_time if self.optimization_start_time else 0.0
        
        # Check if we have any completed trials
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            logger.warning(f"running ModelOptimizer._compile_results ... No trials completed successfully")
            return OptimizationResult(
                best_total_score=0.0,
                best_params={},
                best_trial_number=None,  # No successful trials
                total_trials=len(self.study.trials),
                successful_trials=0,
                optimization_time_hours=optimization_time / 3600,
                optimization_mode=self.config.mode,
                health_weight=self.config.health_weight,
                objective_history=[],
                parameter_importance={},
                health_history=self._trial_health_history,
                best_trial_health=self._best_trial_health,
                dataset_name=self.dataset_name,
                dataset_config=self.dataset_config,
                optimization_config=self.config,
                results_dir=self.results_dir
            )
        
        # Get best_params from Optuna study
        best_params = self.study.best_params.copy()
        
        # Add activation override to best_params if it was used
        if self.activation_override:
            best_params['activation'] = self.activation_override
            logger.debug(f"running _compile_results ... Applied activation override to OptimizationResult.best_params: '{self.activation_override}'")
        
        # Calculate parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
        except:
            importance = {}
        
        return OptimizationResult(
            best_total_score=self.study.best_value,
            best_params=best_params,
            best_trial_number=self.study.best_trial.number,
            total_trials=len(self.study.trials),
            successful_trials=len(completed_trials),
            optimization_time_hours=optimization_time / 3600,
            optimization_mode=self.config.mode,
            health_weight=self.config.health_weight,
            objective_history=[t.value for t in self.study.trials if t.value is not None],
            parameter_importance=importance,
            health_history=self._trial_health_history,
            best_trial_health=self._best_trial_health,
            dataset_name=self.dataset_name,
            dataset_config=self.dataset_config,
            optimization_config=self.config,
            results_dir=self.results_dir
        )

    def _save_results(self, results: OptimizationResult) -> None:
        """Save optimization results to disk"""       
        if self.results_dir is None:
            logger.warning("running ModelOptimizer._save_results ... No results directory available, skipping save")
            return
        
        logger.debug(f"running ModelOptimizer._save_results ... Saving optimization results to {self.results_dir}")
        
        try:
            # Get best_params from results
            best_params = results.best_params.copy()
            
            # Add activation override to best_params if it was used
            if self.activation_override:
                best_params['activation'] = self.activation_override
                logger.debug(f"running _save_results ... Applied activation override to best_params: '{self.activation_override}'")
            
            # Add use_flattening field as inverse of use_global_pooling
            if 'use_global_pooling' in best_params:
                best_params['use_flattening'] = not best_params['use_global_pooling']
            
            # Save best hyperparameters as YAML
            yaml_file = self.results_dir / "optimized_model" / "best_hyperparameters.yaml"
            yaml_data = {
                "dataset": results.dataset_name,
                "optimization_mode": results.optimization_mode,
                "objective": results.optimization_config.objective.value if results.optimization_config else "unknown",
                "health_weight": results.health_weight,
                "best_total_score": float(results.best_total_score),
                "hyperparameters": best_params,
                "execution_method": "runpod_service" if results.optimization_config and results.optimization_config.use_runpod_service else "local"  # Track execution method
            }
            
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            logger.debug(f"running ModelOptimizer._save_results ... Saved best hyperparameters to {yaml_file}")
            
            logger.debug(f"running ModelOptimizer._save_results ... Successfully saved optimization results")
            
        except Exception as e:
            logger.error(f"running ModelOptimizer._save_results ... Failed to save optimization results: {e}")

    def _build_final_model_via_runpod_copy(self, results: OptimizationResult) -> Optional[str]:
        """
        Build final model for RunPod by copying the best trial model and plots.
        This matches the improved local approach - no retraining, just copy the best existing model.

        Args:
            results: Optimization results containing best trial information

        Returns:
            Path to the saved final model, or None if copying failed
        """
        logger.debug("running _build_final_model_via_runpod_copy ... Building final model by copying best trial")

        if not self.results_dir:
            logger.error("running _build_final_model_via_runpod_copy ... Results directory not set")
            raise RuntimeError("Results directory not set")

        best_trial_number = results.best_trial_number
        if best_trial_number is None:
            logger.error("running _build_final_model_via_runpod_copy ... Best trial number not available")
            raise RuntimeError("Best trial number not available in optimization results")

        logger.info(f"running _build_final_model_via_runpod_copy ... Copying model and plots from best trial: {best_trial_number}")

        # Create optimized_model directory
        optimized_model_dir = Path(self.results_dir) / "optimized_model"
        optimized_model_dir.mkdir(parents=True, exist_ok=True)

        # Find best trial's plots directory (where RunPod downloads include both plots and models)
        trial_plots_dir = Path(self.results_dir) / "plots" / f"trial_{best_trial_number}"

        if not trial_plots_dir.exists():
            logger.error(f"running _build_final_model_via_runpod_copy ... Best trial plots directory not found: {trial_plots_dir}")
            raise FileNotFoundError(f"Best trial plots directory not found: {trial_plots_dir}")

        logger.info(f"running _build_final_model_via_runpod_copy ... Found best trial directory: {trial_plots_dir}")

        # Find the Keras model file in the trial directory
        keras_model_files = list(trial_plots_dir.glob("*.keras"))
        if not keras_model_files:
            logger.error(f"running _build_final_model_via_runpod_copy ... No Keras model found in trial directory: {trial_plots_dir}")
            raise FileNotFoundError(f"No Keras model found in best trial directory")

        source_model_path = keras_model_files[0]  # Take the first one
        logger.info(f"running _build_final_model_via_runpod_copy ... Found source model: {source_model_path}")

        # Copy model to optimized_model directory
        model_filename = f"final_model_{self.dataset_config.name}.keras"
        final_model_path = optimized_model_dir / model_filename

        try:
            shutil.copy2(source_model_path, final_model_path)
            logger.info(f"running _build_final_model_via_runpod_copy ... Model copied to: {final_model_path}")

            # Copy all plots from trial directory to optimized_model directory
            plots_copied = 0
            for plot_file in trial_plots_dir.glob("*"):
                if plot_file.is_file() and plot_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
                    dest_file = optimized_model_dir / plot_file.name
                    shutil.copy2(plot_file, dest_file)
                    plots_copied += 1

            logger.info(f"running _build_final_model_via_runpod_copy ... Copied {plots_copied} plot files to optimized_model directory")

            # Create metadata file with final model info
            metadata = {
                "model_path": str(final_model_path),
                "source_trial": best_trial_number,
                "best_total_score": results.best_total_score,
                "dataset_name": self.dataset_config.name,
                "optimization_mode": str(self.config.mode),
                "optimize_for": str(self.config.optimize_for),
                "trials": results.total_trials,
                "training_completed_at": datetime.now().isoformat(),
                "approach": "copy_best_trial_runpod"
            }

            metadata_file = optimized_model_dir / "optimization_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"running _build_final_model_via_runpod_copy ... Created metadata file: {metadata_file}")
            logger.info(f"running _build_final_model_via_runpod_copy ... ✅ Final model built successfully by copying from trial {best_trial_number}")

            return str(final_model_path)

        except Exception as e:
            logger.error(f"running _build_final_model_via_runpod_copy ... Failed to copy model and plots: {e}")
            raise RuntimeError(f"Failed to copy best trial model and plots: {e}")

    def _build_final_model(self, results: OptimizationResult) -> Optional[str]:
        """
        Build and train the final model using the best hyperparameters from optimization.
        Routes to RunPod or local execution based on configuration.

        Args:
            results: Optimization results containing best hyperparameters

        Returns:
            Path to the saved final model, or None if building failed
        """
        if self.config.use_runpod_service:
            logger.debug("running _build_final_model ... Building final model via RunPod copy approach")
            return self._build_final_model_via_runpod_copy(results)
        else:
            logger.debug("running _build_final_model ... Building final model locally")
            return self._build_final_model_locally(results)

    def _build_final_model_locally(self, results: OptimizationResult) -> Optional[str]:
        """
        Build final model locally by copying the best trial model.
        This matches the RunPod approach - no retraining, just copy the best existing model.

        Args:
            results: Optimization results containing best trial information

        Returns:
            Path to the saved final model, or None if building failed
        """
        try:
            logger.debug("running _build_final_model_locally ... Building final model by copying best trial model")

            # Mark final model building as started
            self._final_model_building = True
            self._final_model_available = False

            # Report progress: Starting final model building
            self._report_final_model_progress("Initializing", 0.0)

            # Find the best trial model from the saved trial models
            self._report_final_model_progress("Finding best trial model", 0.10)

            # Check if results_dir is available
            if self.results_dir is None:
                logger.error("running _build_final_model_locally ... Results directory not set")
                raise RuntimeError("Results directory not set")

            plots_dir = Path(self.results_dir) / "plots"
            if not plots_dir.exists():
                logger.error(f"running _build_final_model_locally ... Plots directory not found: {plots_dir}")
                raise FileNotFoundError("No trial plots were created during optimization")

            # Get the best trial number from results
            best_trial_number = getattr(results, 'best_trial_number', None)
            if best_trial_number is None:
                logger.warning("running _build_final_model_locally ... Best trial number not found in results, will search by accuracy")
            else:
                logger.debug(f"running _build_final_model_locally ... from results, best_trial_number is: {best_trial_number}")

            # Find the best trial model directory
            best_trial_dir: Optional[Path] = None
            best_model_path: Optional[Path] = None

            if best_trial_number is not None:
                # Try to find the specific trial directory (new plots pattern: trial_0, trial_1, etc.)
                trial_dir = plots_dir / f"trial_{best_trial_number}"
                if trial_dir.exists():
                    best_trial_dir = trial_dir
                    logger.debug(f"running _build_final_model_locally ... Found best trial directory: {best_trial_dir}")
                else:
                    logger.warning(f"running _build_final_model_locally ... Best trial directory not found: {trial_dir}")

            if best_trial_dir is None:
                # Fallback: find the trial with highest accuracy from all metadata files
                logger.debug("running _build_final_model_locally ... Searching for best trial by accuracy")
                best_accuracy = -1.0

                for trial_dir in plots_dir.iterdir():
                    if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                        continue

                    metadata_files = list(trial_dir.glob("trial_*_metadata.json"))
                    if not metadata_files:
                        continue

                    try:
                        with open(metadata_files[0], 'r') as f:
                            metadata = json.load(f)

                        trial_accuracy = metadata.get('test_accuracy', 0.0)
                        if trial_accuracy > best_accuracy:
                            best_accuracy = trial_accuracy
                            best_trial_dir = trial_dir
                            logger.debug(f"running _build_final_model_locally ... New best trial found: {trial_dir.name} (accuracy: {trial_accuracy:.4f})")
                    except Exception as e:
                        logger.warning(f"running _build_final_model_locally ... Failed to read metadata from {metadata_files[0]}: {e}")

            if best_trial_dir is None:
                logger.error("running _build_final_model_locally ... No valid trial models found")
                raise FileNotFoundError("No valid trial models found")

            # Find the model file in the best trial directory
            self._report_final_model_progress("Locating best model file", 0.20)

            # Type assertion: best_trial_dir is guaranteed to be non-None at this point
            assert best_trial_dir is not None
            model_files = list(best_trial_dir.glob("*.keras"))
            if not model_files:
                logger.error(f"running _build_final_model_locally ... No Keras model files found in {best_trial_dir}")
                raise FileNotFoundError("No Keras model files found in best trial directory")

            best_model_path = model_files[0]  # Take the first (should be only one)
            logger.debug(f"running _build_final_model_locally ... Found best model: {best_model_path}")

            # Load the metadata to get accuracy and other info
            metadata_files = list(best_trial_dir.glob("trial_*_metadata.json"))
            metadata: Dict[str, Any] = {}  # Initialize metadata as empty dict
            if metadata_files:
                with open(metadata_files[0], 'r') as f:
                    metadata = json.load(f)
                test_accuracy = metadata.get('test_accuracy', 0.0)
                logger.debug(f"running _build_final_model_locally ... Best model accuracy: {test_accuracy:.4f}")
            else:
                logger.warning("running _build_final_model_locally ... No metadata found, using default accuracy")
                test_accuracy = 0.0

            # Create optimized_model directory
            self._report_final_model_progress("Preparing final model directory", 0.30)

            # Type assertion: self.results_dir is guaranteed to be non-None at this point
            assert self.results_dir is not None
            optimized_model_dir = Path(self.results_dir) / "optimized_model"
            optimized_model_dir.mkdir(parents=True, exist_ok=True)

            # Generate final model filename
            if self.run_name:
                optimized_run_name = self.run_name
            else:
                # self.results_dir is guaranteed to be non-None at this point
                optimized_run_name = self.results_dir.name

            final_model_filename = f"{optimized_run_name}_acc_{test_accuracy:.4f}_model.keras"
            final_model_path = optimized_model_dir / final_model_filename

            # Copy the best model to optimized_model directory
            self._report_final_model_progress("Copying best model", 0.50)

            shutil.copy2(best_model_path, final_model_path)

            logger.debug(f"running _build_final_model_locally ... Model copied from {best_model_path} to {final_model_path}")

            # Create model metadata for the final model
            self._report_final_model_progress("Creating model metadata", 0.70)

            final_metadata = {
                "filename": final_model_filename,
                "timestamp": self.run_timestamp,
                "run_name": optimized_run_name,
                "test_accuracy": test_accuracy,
                "dataset_name": self.dataset_config.name,
                "model_config": {
                    "epochs": results.best_params.get('epochs', 5),
                    "architecture_type": "image",  # Hardcoded for now, could be dynamic
                    "num_classes": self.dataset_config.num_classes,
                    "input_shape": list(self.dataset_config.input_shape)
                },
                "source_trial": {
                    "trial_number": metadata.get('trial_number', 0),
                    "source_model_path": str(best_model_path)
                }
            }

            metadata_filename = f"{optimized_run_name}_acc_{test_accuracy:.4f}_model_metadata.json"
            metadata_path = optimized_model_dir / metadata_filename

            with open(metadata_path, 'w') as f:
                json.dump(final_metadata, f, indent=2)

            logger.debug(f"running _build_final_model_locally ... Metadata saved to {metadata_path}")

            # Create best_hyperparameters.yaml file for consistency
            self._report_final_model_progress("Creating hyperparameters file", 0.80)

            hyperparams_data = {
                'dataset': self.dataset_name,
                'optimization_mode': self.config.mode,
                'objective': str(self.config.objective),
                'health_weight': getattr(self.config, 'health_weight', None),
                'best_total_score': results.best_total_score,
                'hyperparameters': results.best_params,
                'execution_method': 'local'
            }

            hyperparams_path = optimized_model_dir / "best_hyperparameters.yaml"
            with open(hyperparams_path, 'w') as f:
                yaml.dump(hyperparams_data, f, default_flow_style=False)

            logger.debug(f"running _build_final_model_locally ... Hyperparameters saved to {hyperparams_path}")

            # Copy plots from best trial to optimized_model directory
            self._report_final_model_progress("Copying best trial plots", 0.85)

            if best_trial_number is not None:
                try:
                    # Find trial plots directory
                    trial_plots_dir = Path(self.results_dir) / "plots" / f"trial_{best_trial_number}"

                    if trial_plots_dir.exists():
                        # Copy all plots from trial directory to optimized_model directory
                        plot_files_copied = 0
                        for plot_file in trial_plots_dir.glob("*"):
                            if plot_file.is_file():
                                dest_file = optimized_model_dir / plot_file.name
                                shutil.copy2(plot_file, dest_file)
                                plot_files_copied += 1
                                logger.debug(f"running _build_final_model_locally ... Copied plot: {plot_file.name}")

                        logger.info(f"running _build_final_model_locally ... ✅ Copied {plot_files_copied} plots from trial {best_trial_number}")
                    else:
                        logger.warning(f"running _build_final_model_locally ... Trial plots directory not found: {trial_plots_dir}")

                except Exception as e:
                    logger.warning(f"running _build_final_model_locally ... Failed to copy trial plots: {e}")
                    # Don't fail the entire operation if plot copying fails
            else:
                # Fallback: try to get trial number from metadata if we found trial by accuracy search
                trial_number_from_metadata = metadata.get('trial_number', None)
                if trial_number_from_metadata is not None:
                    try:
                        trial_plots_dir = Path(self.results_dir) / "plots" / f"trial_{trial_number_from_metadata}"
                        if trial_plots_dir.exists():
                            plot_files_copied = 0
                            for plot_file in trial_plots_dir.glob("*"):
                                if plot_file.is_file():
                                    dest_file = optimized_model_dir / plot_file.name
                                    shutil.copy2(plot_file, dest_file)
                                    plot_files_copied += 1
                                    logger.debug(f"running _build_final_model_locally ... Copied plot: {plot_file.name}")

                            logger.info(f"running _build_final_model_locally ... ✅ Copied {plot_files_copied} plots from trial {trial_number_from_metadata} (found by accuracy search)")
                        else:
                            logger.warning(f"running _build_final_model_locally ... Trial plots directory not found: {trial_plots_dir}")
                    except Exception as e:
                        logger.warning(f"running _build_final_model_locally ... Failed to copy trial plots from metadata: {e}")
                else:
                    logger.debug("running _build_final_model_locally ... No trial number available from metadata either")

            # Verify final model file exists and is valid
            self._report_final_model_progress("Verifying final model", 0.90)

            if final_model_path.exists():
                file_size_mb = final_model_path.stat().st_size / (1024*1024)
                logger.debug(f"running _build_final_model_locally ... ✅ Final model confirmed at: {final_model_path} ({file_size_mb:.1f} MB)")
            else:
                logger.error(f"running _build_final_model_locally ... ❌ Final model file does not exist after copying: {final_model_path}")
                raise FileNotFoundError("Final model file does not exist after copying")

            # Report progress: Complete
            self._report_final_model_progress("Completed", 1.0)

            # Mark final model as completed and available
            self._final_model_building = False
            self._final_model_available = True

            logger.info(f"running _build_final_model_locally ... ✅ Final model successfully built by copying best trial model")
            logger.info(f"running _build_final_model_locally ... Final model path: {final_model_path}")

            return str(final_model_path)

        except Exception as e:
            logger.error(f"running _build_final_model_locally ... Failed to build final model: {e}")
            # Reset final model building status on failure
            self._final_model_building = False
            self._final_model_available = False
            return None


# Convenience function for command-line usage with RunPod service support
def optimize_model(
    dataset_name: Optional[str] = None,
    mode: Optional[str] = None,
    optimize_for: Optional[str] = None,
    trials: Optional[int] = None,
    run_name: Optional[str] = None,
    activation: Optional[str] = None,
    progress_callback: Optional[Callable[[Union[TrialProgress, AggregatedProgress, UnifiedProgress]], None]] = None,
    # RunPod service parameters
    use_runpod_service: Optional[bool] = None,
    runpod_service_endpoint: Optional[str] = None,
    runpod_service_timeout: Optional[int] = None,
    runpod_service_fallback_local: Optional[bool] = None,
    gpu_proxy_sample_percentage: Optional[float] = None,
    concurrent: Optional[bool] = None,
    concurrent_workers: Optional[int] = None,
    **config_overrides
) -> OptimizationResult:
    """
    Unified optimization function that uses centralized defaults from configs.py
    
    This function serves as the common entry point for optimization from three sources:
    1. API server (web requests via api_server.py)
    2. Command-line interface (python src/optimizer.py)
    3. Direct function calls (programmatic usage)
    
    All parameters are optional and use defaults from OptimizationConfig in configs.py
    unless explicitly overridden.
    
    Args:
        All parameters are optional and correspond to OptimizationConfig fields:
        dataset_name: Dataset to optimize (default: "mnist")
        mode: Optimization mode "simple" or "health" (default: "health") 
        optimize_for: Optimization objective (default: "val_accuracy")
        trials: Number of optimization trials (default: 2)
        run_name: Optional unified run name for consistent directory/file naming
        progress_callback: Optional callback for real-time progress updates
        use_runpod_service: Enable/disable RunPod service (default: True)
        runpod_service_endpoint: RunPod service endpoint URL (auto-configured from env)
        runpod_service_timeout: Request timeout in seconds (default: 600)
        runpod_service_fallback_local: Fall back to local execution if service fails (default: True)
        **config_overrides: Additional optimization config overrides
        
    Returns:
        OptimizationResult with best parameters and metrics
        
    Usage Examples:
    
    1. API Usage (via web server):
        # Start the API server
        python src/api_server.py
        
        # Make API request with overrides
        curl -X POST http://localhost:8000/optimize \\
             -H "Content-Type: application/json" \\
             -d '{
                 "dataset_name": "cifar10",
                 "mode": "simple", 
                 "trials": 10,
                 "max_epochs_per_trial": 8
             }'
    
    2. Command-line Usage (direct CLI):
        # Call optimizer.py with overrides
        python src/optimizer.py dataset_name=cifar10 mode=simple trials=10 max_epochs_per_trial=8
    
    3. Direct Function Call (programmatic):
        from optimizer import optimize_model
        
        # Call function with overrides
        result = optimize_model(
            dataset_name="cifar10",
            mode="simple", 
            trials=10,
            max_epochs_per_trial=8
        )
        
        # With RunPod service
        result = optimize_model(
            dataset_name="cifar10",
            mode="simple",
            trials=10,
            use_runpod_service=True,
            runpod_service_endpoint="https://api.runpod.ai/v2/your-endpoint/run"
        )
    """
    # Collect all non-None parameters for OptimizationConfig
    override_params = {}
    
    # Get all function parameters (excluding special ones)
    function_locals = locals()
    exclude_params = {'override_params', 'config_overrides', 'run_name', 'activation', 'progress_callback'}
    
    for param_name, param_value in function_locals.items():
        if param_name not in exclude_params and param_value is not None:
            override_params[param_name] = param_value
    
    # Add config_overrides
    override_params.update(config_overrides)
    
    # Create optimization config - Pydantic will use defaults from configs.py for unspecified parameters
    opt_config = OptimizationConfig(**override_params)
    
    # Create unified run name if not provided using centralized function
    if run_name is None:
        run_name = create_run_name(
            dataset_name=opt_config.dataset_name,
            mode=opt_config.mode,
            optimize_for=opt_config.optimize_for
        )
    
    logger.debug(f"running optimize_model ... Using unified run name: {run_name}")
    logger.debug(
        "running optimize_model ... opt_config loaded: concurrent=%s, workers=%s, use_runpod_service=%s",
        opt_config.concurrent, opt_config.concurrent_workers, opt_config.use_runpod_service
    )
    
    # Log execution approach
    if opt_config.use_runpod_service:
        logger.debug(f"running optimize_model ... 🔄 EXECUTION APPROACH: RunPod Service (JSON API)")
        logger.debug(f"running optimize_model ... - Endpoint: {opt_config.runpod_service_endpoint}")
        logger.debug(f"running optimize_model ... - Timeout: {opt_config.runpod_service_timeout}s")
        logger.debug(f"running optimize_model ... - Fallback local: {opt_config.runpod_service_fallback_local}")
        logger.debug(f"running optimize_model ... - Payload approach: Tiny JSON commands (<1KB)")
    else:
        logger.debug(f"running optimize_model ... EXECUTION APPROACH: Local only")
    
    # Run optimization
    optimizer = ModelOptimizer(
        dataset_name=opt_config.dataset_name,
        optimization_config=opt_config,
        run_name=run_name,
        progress_callback=progress_callback,
        activation_override=activation
    )
    return optimizer.optimize()


if __name__ == "__main__":
    # Command-line interface with RunPod service support
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    
    # Extract required arguments
    dataset_name = args.get('dataset_name', 'cifar10')
    mode = args.get('mode', 'simple')
    optimize_for = args.get('optimize_for', 'val_accuracy')
    trials = int(args.get('trials', '50'))
    run_name = args.get('run_name', None)
    activation = args.get('activation', None)
    
    # Extract RunPod service parameters
    use_runpod_service = args.get('use_runpod_service', 'false').lower() in ['true', '1', 'yes', 'on']
    runpod_service_endpoint = args.get('runpod_service_endpoint', None)
    runpod_service_timeout = int(args.get('runpod_service_timeout', '600'))
    runpod_service_fallback_local = args.get('runpod_service_fallback_local', 'true').lower() in ['true', '1', 'yes', 'on']
    
    # Convert parameters
    int_params = [
        'trials', 'startup_trials', 'warmup_steps', 'random_seed',
        'max_epochs_per_trial', 'early_stopping_patience', 'min_epochs_per_trial',
        'stability_window', 'health_analysis_sample_size', 'health_monitoring_frequency',
        'runpod_service_timeout', 'concurrent_workers'
    ]
    for int_param in int_params:
        if int_param in args:
            try:
                args[int_param] = int(args[int_param])
            except ValueError:
                logger.warning(f"running optimizer.py ... Invalid {int_param}: {args[int_param]}, using default")
                del args[int_param]
    
    float_params = [
        'timeout_hours', 'max_training_time_minutes', 'validation_split', 'test_size',
        'max_bias_change_per_epoch', 'health_weight', 'gpu_proxy_sample_percentage'
    ]
    for float_param in float_params:
        if float_param in args:
            try:
                args[float_param] = float(args[float_param])
            except ValueError:
                logger.warning(f"running optimizer.py ... Invalid {float_param}: {args[float_param]}, using default")
                del args[float_param]
    
    bool_params = [
        'save_optimization_history', 'create_comparison_plots',
        'enable_early_stopping', 'enable_stability_checks',
        'use_runpod_service', 'runpod_service_fallback_local',
        'concurrent'
    ]
    for bool_param in bool_params:
        if bool_param in args:
            args[bool_param] = args[bool_param].lower() in ['true', '1', 'yes', 'on']
    
    # Handle string parameters
    string_params = ['activation', 'runpod_service_endpoint']
    for string_param in string_params:
        if string_param in args:
            if args[string_param].strip():
                logger.debug(f"running optimizer.py ... Set {string_param}: {args[string_param]}")
            else:
                logger.warning(f"running optimizer.py ... Empty {string_param}, removing")
                del args[string_param]
    
    logger.debug(f"running optimizer.py ... Starting optimization")
    logger.debug(f"running optimizer.py ... Dataset: {dataset_name}")
    logger.debug(f"running optimizer.py ... Mode: {mode}")
    logger.debug(f"running optimizer.py ... Objective: {optimize_for}")
    logger.debug(f"running optimizer.py ... Trials: {trials}")
    if run_name:
        logger.debug(f"running optimizer.py ... Run name: {run_name}")
    
    # Log execution approach
    if use_runpod_service:
        logger.debug(f"running optimizer.py ... 🔄 EXECUTION: RunPod Service (JSON API approach)")
        logger.debug(f"running optimizer.py ... - Endpoint: {runpod_service_endpoint}")
        logger.debug(f"running optimizer.py ... - Timeout: {runpod_service_timeout}s")
        logger.debug(f"running optimizer.py ... - Fallback local: {runpod_service_fallback_local}")
        logger.debug(f"running optimizer.py ... - Payload: Tiny JSON commands (<1KB) vs old code injection (1.15MB+)")
    else:
        logger.debug(f"running optimizer.py ... EXECUTION: Local only")
    
    logger.debug(f"running optimizer.py ... Parsed arguments: {args}")
    
    try:
        # Run optimization with RunPod service support
        result = optimize_model(
            dataset_name=dataset_name,
            mode=mode,
            optimize_for=optimize_for,
            trials=trials,
            run_name=run_name,
            activation=activation,
            progress_callback=default_progress_callback,
            use_runpod_service=use_runpod_service,
            runpod_service_endpoint=runpod_service_endpoint,
            runpod_service_timeout=runpod_service_timeout,
            runpod_service_fallback_local=runpod_service_fallback_local,
            **{k: v for k, v in args.items() if k not in ['dataset_name', 'mode', 'optimize_for', 'trials', 'run_name', 'activation', 'use_runpod_service', 'runpod_service_endpoint', 'runpod_service_timeout', 'runpod_service_fallback_local']}
        )
        
        # Print results
        print(result.summary())
        
        logger.debug(f"running optimizer.py ... ✅ Optimization completed successfully!")
        
        # Log execution method in results
        if use_runpod_service:
            print(f"\n🚀 RunPod Service: All trials executed via JSON API")
            print(f"   Endpoint: {runpod_service_endpoint}")
            print(f"   Approach: Tiny JSON payloads (<1KB) instead of code injection")
        else:
            print(f"\n💻 Local Execution: All trials executed on local hardware")
        
        if activation:
            print(f"\n🎯 Activation Override: All trials used '{activation}' activation function")
        
    except Exception as e:
        error_msg = str(e)
        if "health-only objective" in error_msg.lower() and "simple mode" in error_msg.lower():
            print(f"\n❌ Configuration Error:")
            print(f"Cannot use health objective '{optimize_for}' in simple mode.")
            print(f"\nTry one of these instead:")
            print(f"1. Use simple mode with universal objective:")
            print(f"   python optimizer.py dataset={dataset_name} mode=simple optimize_for=val_accuracy")
            print(f"2. Use health mode with your desired objective:")
            print(f"   python optimizer.py dataset={dataset_name} mode=health optimize_for={optimize_for}")
        elif "runpod" in error_msg.lower():
            print(f"\n❌ RunPod Service Error:")
            print(f"{error_msg}")
            print(f"\nTry one of these:")
            print(f"1. Check that your RunPod service endpoint is accessible:")
            print(f"   curl {runpod_service_endpoint}")
            print(f"2. Try with local fallback enabled:")
            print(f"   python optimizer.py dataset_name={dataset_name} use_runpod_service=true runpod_service_fallback_local=true")
            print(f"3. Use local execution only:")
            print(f"   python optimizer.py dataset_name={dataset_name} use_runpod_service=false")
        else:
            print(f"\n❌ Error: {error_msg}")
        
        logger.error(f"running optimizer.py ... ❌ Optimization failed: {e}")
        sys.exit(1)