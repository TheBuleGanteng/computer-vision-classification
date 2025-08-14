"""
Unified Model Optimizer for Multi-Modal Classification - RUNPOD SERVICE INTEGRATION

ARCHITECTURAL PIVOT: FROM CODE INJECTION TO SPECIALIZED SERVERLESS
- ❌ REMOVED: GPU proxy code injection approach
- ✅ ADDED: RunPod service JSON API approach  
- 🔄 SAME: Uses same RunPod infrastructure, different approach

PHASE 2 REFACTORING: Transforming into pure orchestrator
- ✅ STEP 2.1a: Added HyperparameterSelector integration
- ✅ STEP 2.1b: Removed embedded hyperparameter suggestion methods
- ✅ STEP 2.1c: Added PlotGenerator integration
- ✅ STEP 2.1: Replaced GPU proxy with RunPod service integration

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

import copy
import csv
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv
from enum import Enum
import json
import numpy as np
import optuna
import optuna.integration.keras as optuna_keras
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import os
from pathlib import Path
import requests  # For RunPod service communication
from sklearn.model_selection import train_test_split
import sys
from tensorflow import keras # type: ignore
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import uuid
import yaml

# Import existing modules
from dataset_manager import DatasetManager, DatasetConfig
from health_analyzer import HealthAnalyzer
from model_builder import ModelBuilder, ModelConfig, create_and_train_model
from utils.logger import logger

# PHASE 2 REFACTORING: Import new modular components
from hyperparameter_selector import HyperparameterSelector
from plot_generator import PlotGenerator


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

@dataclass
class TrialProgress:
    """Real-time trial progress data for API streaming"""
    trial_id: str
    trial_number: int
    status: str  # "running", "completed", "failed", "pruned"
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # Architecture Information
    architecture: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    model_size: Optional[Dict[str, Any]] = None
    
    # Health Metrics (populated during/after training)
    health_metrics: Optional[Dict[str, Any]] = None
    training_stability: Optional[Dict[str, Any]] = None
    
    # Performance Data
    performance: Optional[Dict[str, Any]] = None
    training_history: Optional[Dict[str, Any]] = None
    
    # Pruning Information
    pruning_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization"""
        return {
            'trial_id': self.trial_id,
            'trial_number': self.trial_number,
            'status': self.status,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'duration_seconds': self.duration_seconds,
            'architecture': self.architecture,
            'hyperparameters': self.hyperparameters,
            'model_size': self.model_size,
            'health_metrics': self.health_metrics,
            'training_stability': self.training_stability,
            'performance': self.performance,
            'training_history': self.training_history,
            'pruning_info': self.pruning_info
        }


class OptimizationMode(Enum):
    """
    Optimization mode selection for different optimization strategies
    
    SIMPLE: Pure objective optimization focusing on primary metrics (accuracy, efficiency, etc.)
            Health metrics are calculated for monitoring but don't influence trial scoring
            
    HEALTH: Health-aware optimization that combines objective with health metrics
            Uses configurable health_weight to balance objective vs health importance
    """
    SIMPLE = "simple"
    HEALTH = "health"


class OptimizationObjective(Enum):
    """
    Simplified optimization objectives for hyperparameter tuning
    
    Universal Objectives (work in both SIMPLE and HEALTH modes):
        VAL_ACCURACY: Maximize validation accuracy (preferred for generalization)
        ACCURACY: Maximize final training accuracy (may overfit)
        TRAINING_TIME: Minimize training time (faster models)
        PARAMETER_EFFICIENCY: Maximize accuracy per parameter (compact models)
        MEMORY_EFFICIENCY: Maximize accuracy per memory usage (memory-efficient models)
        INFERENCE_SPEED: Maximize accuracy per inference time (fast prediction models)
    
    Health-Only Objectives (only work in HEALTH mode):
        OVERALL_HEALTH: Maximize overall model health score
        NEURON_UTILIZATION: Maximize active neuron usage (minimize dead neurons)
        TRAINING_STABILITY: Maximize training process stability
        GRADIENT_HEALTH: Maximize gradient flow quality
    """
    # Universal objectives (work in both modes)
    VAL_ACCURACY = "val_accuracy"
    ACCURACY = "accuracy"
    TRAINING_TIME = "training_time"
    PARAMETER_EFFICIENCY = "parameter_efficiency"
    MEMORY_EFFICIENCY = "memory_efficiency"
    INFERENCE_SPEED = "inference_speed"
    
    # Health-only objectives (only work in HEALTH mode)
    OVERALL_HEALTH = "overall_health"
    NEURON_UTILIZATION = "neuron_utilization"
    TRAINING_STABILITY = "training_stability"
    GRADIENT_HEALTH = "gradient_health"
    
    @classmethod
    def get_universal_objectives(cls) -> List['OptimizationObjective']:
        """Get objectives that work in both SIMPLE and HEALTH modes"""
        return [
            cls.VAL_ACCURACY,
            cls.ACCURACY,
            cls.TRAINING_TIME,
            cls.PARAMETER_EFFICIENCY,
            cls.MEMORY_EFFICIENCY,
            cls.INFERENCE_SPEED
        ]
    
    @classmethod
    def get_health_only_objectives(cls) -> List['OptimizationObjective']:
        """Get objectives that only work in HEALTH mode"""
        return [
            cls.OVERALL_HEALTH,
            cls.NEURON_UTILIZATION,
            cls.TRAINING_STABILITY,
            cls.GRADIENT_HEALTH
        ]
    
    @classmethod
    def is_health_only(cls, objective: 'OptimizationObjective') -> bool:
        """Check if objective only works in HEALTH mode"""
        return objective in cls.get_health_only_objectives()


class PlotGenerationMode(Enum):
    """Plot generation modes for optimization"""
    ALL = "all"      # Generate plots for all trials
    BEST = "best"    # Generate plots for best trial only  
    NONE = "none"    # No plot generation


@dataclass
class OptimizationConfig:
    """Configuration for optimization process"""
    
    # Optimization mode and strategy
    mode: OptimizationMode = OptimizationMode.SIMPLE
    objective: OptimizationObjective = OptimizationObjective.VAL_ACCURACY
    n_trials: int = 50
    timeout_hours: Optional[float] = None
    gpu_proxy_sample_percentage: float = 0.5
    
    # Health weighting (only used in HEALTH mode with universal objectives)
    health_weight: float = 0.3
    
    # Pruning and sampling
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
    random_seed: int = 42
    
    # Resource constraints
    max_epochs_per_trial: int = 20
    max_training_time_minutes: float = 60.0
    max_parameters: int = 10_000_000
    min_accuracy_threshold: float = 0.5
    
    # Stability detection parameters
    min_epochs_per_trial: int = 5
    enable_stability_checks: bool = True
    stability_window: int = 3
    max_bias_change_per_epoch: float = 10.0
    
    # Health analysis settings
    health_analysis_sample_size: int = 50
    health_monitoring_frequency: int = 1
    
    # Validation settings
    validation_split: float = 0.2
    test_size: float = 0.2
    
    # Output settings
    save_best_model: bool = True
    save_optimization_history: bool = True
    create_comparison_plots: bool = True
    
    # Advanced options
    enable_early_stopping: bool = True
    early_stopping_patience: int = 5
    
    # 🔄 ARCHITECTURAL PIVOT: RunPod Service Integration (replaces GPU proxy)
    use_runpod_service: bool = False               # Enable/disable RunPod service usage
    runpod_service_endpoint: Optional[str] = None  # RunPod service endpoint URL
    runpod_service_timeout: int = 600              # Request timeout in seconds (10 minutes)
    runpod_service_fallback_local: bool = True     # Fall back to local execution if service fails
    
    # Plot generation configuration
    plot_generation: PlotGenerationMode = PlotGenerationMode.ALL
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 <= self.health_weight <= 1:
            raise ValueError("health_weight must be between 0 and 1")
        
        # Validate mode-objective compatibility
        self._validate_mode_objective_compatibility()
        logger.debug(f"running OptimizationConfig.__post_init__ ... Plot generation mode: {self.plot_generation.value}")
        
        # 🔄 ARCHITECTURAL PIVOT: Auto-configure RunPod endpoint if not provided
        if self.use_runpod_service and not self.runpod_service_endpoint:
            endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
            if endpoint_id:
                self.runpod_service_endpoint = f"https://api.runpod.ai/v2/{endpoint_id}/run"
                logger.debug(f"running OptimizationConfig.__post_init__ ... Auto-configured endpoint from RUNPOD_ENDPOINT_ID: {self.runpod_service_endpoint}")
            else:
                logger.warning(f"running OptimizationConfig.__post_init__ ... RunPod service enabled but RUNPOD_ENDPOINT_ID not found in environment")
        
        # 🔄 ARCHITECTURAL PIVOT: Log RunPod Service configuration
        if self.use_runpod_service:
            logger.debug(f"running OptimizationConfig.__post_init__ ... RunPod service enabled in optimization config")
            logger.debug(f"running OptimizationConfig.__post_init__ ... - Endpoint: {self.runpod_service_endpoint}")
            logger.debug(f"running OptimizationConfig.__post_init__ ... - Timeout: {self.runpod_service_timeout}s")
            logger.debug(f"running OptimizationConfig.__post_init__ ... - Fallback local: {self.runpod_service_fallback_local}")
        else:
            logger.debug(f"running OptimizationConfig.__post_init__ ... RunPod service disabled - using local execution only")
    
    def _validate_mode_objective_compatibility(self) -> None:
        """Validate that the objective is compatible with the selected mode"""
        if self.mode == OptimizationMode.SIMPLE:
            if OptimizationObjective.is_health_only(self.objective):
                universal_objectives = [obj.value for obj in OptimizationObjective.get_universal_objectives()]
                raise ValueError(
                    f"Health-only objective '{self.objective.value}' cannot be used in SIMPLE mode. "
                    f"Available objectives for SIMPLE mode: {universal_objectives}"
                )
        
        logger.debug(f"running OptimizationConfig._validate_mode_objective_compatibility ... "
                    f"Mode '{self.mode.value}' is compatible with objective '{self.objective.value}'")


@dataclass
class OptimizationResult:
    """Results from optimization process"""
    
    # Best trial results
    best_value: float
    best_params: Dict[str, Any]
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
        objective_name = self.optimization_config.objective.value if self.optimization_config else "unknown"
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
Best {objective_name}: {self.best_value:.4f}
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


class ModelOptimizer:
    """
    ARCHITECTURAL PIVOT: Unified optimizer class with RunPod service integration
    
    REFACTORING PROGRESS:
    - ✅ STEP 2.1a: Added HyperparameterSelector integration
    - ✅ STEP 2.1b: Removed embedded hyperparameter suggestion methods
    - ✅ STEP 2.1c: Added PlotGenerator integration
    - ✅ STEP 2.1: Added RunPod service integration (replaces GPU proxy)
    
    ARCHITECTURAL PIVOT CHANGES:
    - ❌ REMOVED: GPU proxy code injection approach
    - ✅ ADDED: RunPod service JSON API approach
    - 🔄 SAME: Uses same RunPod infrastructure with specialized handler
    
    Integrates with existing ModelBuilder and DatasetManager to provide
    automated hyperparameter tuning with simple or health-aware optimization.
    """
    
    def __init__(self, dataset_name: str, optimization_config: Optional[OptimizationConfig] = None, 
        datasets_root: Optional[str] = None, run_name: Optional[str] = None,
        health_analyzer: Optional[HealthAnalyzer] = None,
        progress_callback: Optional[Callable[[TrialProgress], None]] = None,
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
        self.config = optimization_config or OptimizationConfig()
        self.run_name = run_name
        self.activation_override = activation_override
        if self.activation_override:
            logger.debug(f"running ModelOptimizer.__init__ ... Activation override: {self.activation_override} (will force this activation for all trials)")
        
        # Enhanced plot tracking
        self.trial_plot_data = {}
        self.best_trial_number = None      
        
        # Log plot generation configuration
        logger.debug(f"running ModelOptimizer.__init__ ... Plot generation mode: {self.config.plot_generation.value}")
        
        # Initialize health analyzer (always available for monitoring)
        self.health_analyzer = health_analyzer or HealthAnalyzer()
        
        # Health monitoring storage
        self.trial_health_history: List[Dict[str, Any]] = []
        self.best_trial_health: Optional[Dict[str, Any]] = None
        
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
        
        # PHASE 2 REFACTORING: Initialize HyperparameterSelector
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
        
        # PHASE 2 REFACTORING: Initialize PlotGenerator
        logger.debug(f"running ModelOptimizer.__init__ ... Initializing PlotGenerator")
        default_model_config = ModelConfig()
        self.plot_generator = PlotGenerator(
            dataset_config=self.dataset_config,
            model_config=default_model_config
        )
        logger.debug(f"running ModelOptimizer.__init__ ... PlotGenerator initialized for data type: {self.plot_generator.data_type}")
        
        # Initialize optimization state
        self.study: Optional[optuna.Study] = None
        self.optimization_start_time: Optional[float] = None
        self.results_dir: Optional[Path] = None
        
        # Create results directory
        self._setup_results_directory()
        
        # Real-time trial tracking
        self.progress_callback = progress_callback
        self.current_trial_progress: Optional[TrialProgress] = None
        self.trial_progress_history: List[TrialProgress] = []
        self.best_trial_progress: Optional[TrialProgress] = None
        
        # Architecture and health trends for visualization
        self.architecture_trends: Dict[str, List[float]] = {}
        self.health_trends: Dict[str, List[float]] = {}
        
        logger.debug(f"running ModelOptimizer.__init__ ... Optimizer initialized for {dataset_name}")
        logger.debug(f"running ModelOptimizer.__init__ ... Mode: {self.config.mode.value}")
        logger.debug(f"running ModelOptimizer.__init__ ... Objective: {self.config.objective.value}")
        if self.config.mode == OptimizationMode.HEALTH and not OptimizationObjective.is_health_only(self.config.objective):
            logger.debug(f"running ModelOptimizer.__init__ ... Health weight: {self.config.health_weight} ({(1-self.config.health_weight)*100:.0f}% objective, {self.config.health_weight*100:.0f}% health)")
        logger.debug(f"running ModelOptimizer.__init__ ... Max trials: {self.config.n_trials}")
        if self.run_name:
            logger.debug(f"running ModelOptimizer.__init__ ... Run name: {self.run_name}")
        
        # 🔄 ARCHITECTURAL PIVOT: LOG RUNPOD SERVICE CONFIGURATION
        if self.config.use_runpod_service:
            logger.debug(f"running ModelOptimizer.__init__ ... 🔄 ARCHITECTURAL PIVOT: RunPod service integration ENABLED")
            logger.debug(f"running ModelOptimizer.__init__ ... - Approach: JSON API calls (specialized serverless)")
            logger.debug(f"running ModelOptimizer.__init__ ... - Endpoint: {self.config.runpod_service_endpoint}")
            logger.debug(f"running ModelOptimizer.__init__ ... - Timeout: {self.config.runpod_service_timeout}s")
            logger.debug(f"running ModelOptimizer.__init__ ... - Fallback local: {self.config.runpod_service_fallback_local}")
            logger.debug(f"running ModelOptimizer.__init__ ... - Payload: Tiny JSON commands (<1KB) instead of Python code")
        else:
            logger.debug(f"running ModelOptimizer.__init__ ... RunPod service integration: DISABLED (local execution only)")
        
        # PHASE 2 REFACTORING LOG
        logger.debug(f"running ModelOptimizer.__init__ ... PHASE 2 REFACTORING: Pure orchestrator transformation complete")
        logger.debug(f"running ModelOptimizer.__init__ ... ARCHITECTURAL PIVOT: GPU proxy code injection → RunPod service JSON API")
    
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
            timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            dataset_clean = self.dataset_name.replace(" ", "_").lower()
            mode_suffix = self.config.mode.value
            fallback_name = f"{timestamp}_{dataset_clean}_{mode_suffix}_fallback"
            self.results_dir = optimization_results_dir / fallback_name
            logger.debug(f"running _setup_results_directory ... No run_name provided, using fallback: {fallback_name}")
        
        # Create the main results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"running _setup_results_directory ... Results directory: {self.results_dir}")
        
        # Create plots subdirectory
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Create trial directories
        for trial_num in range(self.config.n_trials):
            trial_dir = plots_dir / f"trial_{trial_num + 1}"
            trial_dir.mkdir(exist_ok=True)
        
        # Create optimized model directory
        optimized_dir = self.results_dir / "optimized_model"
        optimized_dir.mkdir(exist_ok=True)
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        PHASE 2 REFACTORING: Use HyperparameterSelector for hyperparameter suggestion
        
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

    # 🔄 ARCHITECTURAL PIVOT: RunPod Service Methods (replacing GPU proxy)
    
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
    
    def _train_via_runpod_service(self, trial: optuna.Trial, params: Dict[str, Any]) -> float:
        """
        🔄 ARCHITECTURAL PIVOT: Train model via RunPod service using JSON API
        
        Sends a small JSON request to the RunPod service instead of large Python code.
        The RunPod service uses the same optimizer.py orchestration logic via handler.py.
        
        Args:
            trial: Optuna trial object
            params: Hyperparameters for this trial
            
        Returns:
            Objective value for optimization
            
        Raises:
            RuntimeError: If RunPod service call fails and fallback is disabled
        """
        logger.debug(f"running _train_via_runpod_service ... 🔄 ARCHITECTURAL PIVOT: Starting RunPod service training for trial {trial.number}")
        logger.debug(f"running _train_via_runpod_service ... Using JSON API approach (tiny payloads) instead of code injection")
        
        try:
            # Get API key from environment
            api_key = os.getenv('RUNPOD_API_KEY')
            if not api_key:
                raise RuntimeError("RUNPOD_API_KEY environment variable not set")
            
            # Prepare JSON request payload (tiny payload <1KB)
            request_payload = {
                "input": {  # ← ADD THIS WRAPPER
                    "command": "start_training",
                    "trial_id": f"trial_{trial.number}",
                    "dataset": self.dataset_name,
                    "hyperparameters": params,
                    "config": {
                        "validation_split": self.config.validation_split,
                        "max_training_time": self.config.max_training_time_minutes,
                        "mode": self.config.mode.value,
                        "objective": self.config.objective.value,
                        "gpu_proxy_sample_percentage": self.config.gpu_proxy_sample_percentage
                    }
                }
            }
            
            # Log payload size for architectural pivot verification
            payload_size = len(json.dumps(request_payload).encode('utf-8'))
            logger.debug(f"running _train_via_runpod_service ... 🔄 PAYLOAD SIZE: {payload_size} bytes (vs old approach: 1.15MB+)")
            logger.debug(f"running _train_via_runpod_service ... Request payload prepared for trial {trial.number}")
            logger.debug(f"running _train_via_runpod_service ... Dataset: {self.dataset_name}")
            logger.debug(f"running _train_via_runpod_service ... Hyperparameters: {params}")
            logger.debug(f"running _train_via_runpod_service ... Endpoint: {self.config.runpod_service_endpoint}")
            
            # Send request to RunPod service
            if self.config.runpod_service_endpoint is None:
                raise RuntimeError("RunPod service endpoint is not configured")

            logger.debug(f"running _train_via_runpod_service ... DEBUG PAYLOAD: {json.dumps(request_payload, indent=2)}")

            response = requests.post(
                self.config.runpod_service_endpoint,
                json=request_payload,
                timeout=self.config.runpod_service_timeout,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
            )
            
            # Check response status
            if response.status_code != 200:
                raise RuntimeError(f"RunPod service returned status {response.status_code}: {response.text}")
            
            # Parse initial response
            result = response.json()
            job_id = result.get('id')
            
            if not job_id:
                raise RuntimeError("No job ID returned from RunPod service")
            
            logger.debug(f"running _train_via_runpod_service ... Job submitted with ID: {job_id}")
            logger.debug(f"running _train_via_runpod_service ... Polling for completion...")
            
            # Poll for job completion
            max_poll_time = self.config.runpod_service_timeout
            poll_interval = 10  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_poll_time:
                # Check job status
                status_url = f"{self.config.runpod_service_endpoint.rsplit('/run', 1)[0]}/status/{job_id}"
                status_response = requests.get(
                    status_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    job_status = status_data.get('status', 'UNKNOWN')
                    
                    logger.debug(f"running _train_via_runpod_service ... Job {job_id} status: {job_status}")
                    
                    if job_status == 'COMPLETED':
                        # Get the output
                        output = status_data.get('output', {})
                        if not output:
                            raise RuntimeError("No output returned from completed RunPod job")
                        
                        # Validate response format
                        if not output.get('success', False):
                            error_msg = output.get('error', 'Unknown error from RunPod service')
                            raise RuntimeError(f"RunPod service training failed: {error_msg}")
                        
                        # Extract metrics from response
                        metrics = output.get('metrics', {})
                        if not metrics:
                            raise RuntimeError("No metrics returned from RunPod service")
                        
                        # Calculate objective value based on configuration
                        objective_value = self._calculate_objective_from_service_response(metrics, output, trial)
                        
                        logger.debug(f"running _train_via_runpod_service ... Trial {trial.number} completed via RunPod service")
                        logger.debug(f"running _train_via_runpod_service ... 🔄 ARCHITECTURAL PIVOT SUCCESS: JSON API approach working")
                        logger.debug(f"running _train_via_runpod_service ... Objective value: {objective_value:.4f}")
                        
                        return objective_value
                        
                    elif job_status == 'FAILED':
                        error_logs = status_data.get('error', 'Job failed without details')
                        raise RuntimeError(f"RunPod job failed: {error_logs}")
                        
                    elif job_status in ['IN_QUEUE', 'IN_PROGRESS']:
                        # Continue polling
                        time.sleep(poll_interval)
                        continue
                    else:
                        logger.warning(f"running _train_via_runpod_service ... Unknown job status: {job_status}")
                        time.sleep(poll_interval)
                        continue
                else:
                    logger.warning(f"running _train_via_runpod_service ... Failed to get job status: {status_response.status_code}")
                    time.sleep(poll_interval)
                    continue
            
            # Timeout reached
            raise RuntimeError(f"RunPod job {job_id} did not complete within {max_poll_time} seconds")
            
        except Exception as e:
            logger.error(f"running _train_via_runpod_service ... RunPod service training failed for trial {trial.number}: {e}")
            
            if self.config.runpod_service_fallback_local:
                logger.warning(f"running _train_via_runpod_service ... Falling back to local execution for trial {trial.number}")
                return self._train_locally_for_trial(trial, params)
            else:
                logger.error(f"running _train_via_runpod_service ... RunPod service failed and local fallback disabled")
                raise RuntimeError(f"RunPod service training failed for trial {trial.number}: {e}")
    
    def _calculate_objective_from_service_response(
        self, 
        metrics: Dict[str, Any], 
        full_result: Dict[str, Any], 
        trial: optuna.Trial
    ) -> float:
        """
        Calculate objective value from RunPod service response
        
        Processes the comprehensive metrics returned by the RunPod service
        (which uses the same HealthAnalyzer logic as local execution).
        
        Args:
            metrics: Metrics dictionary from service response
            full_result: Full response from service (for additional data)
            trial: Optuna trial object
            
        Returns:
            Objective value for optimization
        """
        logger.debug(f"running _calculate_objective_from_service_response ... Calculating objective for trial {trial.number}")
        logger.debug(f"running _calculate_objective_from_service_response ... Available metrics: {list(metrics.keys())}")
        
        try:
            # Extract basic metrics
            test_accuracy = metrics.get('test_accuracy', 0.0)
            test_loss = metrics.get('test_loss', 0.0)
            training_time_seconds = metrics.get('training_time_seconds', 0.0)
            
            # Extract health metrics if available
            health_metrics = full_result.get('health_metrics', {})
            overall_health = health_metrics.get('overall_health', 0.5)
            
            logger.debug(f"running _calculate_objective_from_service_response ... Basic metrics: acc={test_accuracy:.4f}, loss={test_loss:.4f}")
            logger.debug(f"running _calculate_objective_from_service_response ... Health metrics: overall={overall_health:.3f}")
            
            # Calculate objective based on optimization mode and target
            if self.config.objective == OptimizationObjective.VAL_ACCURACY:
                primary_value = test_accuracy
                
                if self.config.mode == OptimizationMode.HEALTH and not OptimizationObjective.is_health_only(self.config.objective):
                    # Weighted combination
                    objective_weight = 1.0 - self.config.health_weight
                    health_weight = self.config.health_weight
                    final_value = objective_weight * primary_value + health_weight * overall_health
                    
                    logger.debug(f"running _calculate_objective_from_service_response ... HEALTH mode weighted combination:")
                    logger.debug(f"running _calculate_objective_from_service_response ... - Primary (acc): {primary_value:.4f} * {objective_weight:.1f} = {primary_value * objective_weight:.4f}")
                    logger.debug(f"running _calculate_objective_from_service_response ... - Health: {overall_health:.3f} * {health_weight:.1f} = {overall_health * health_weight:.4f}")
                    logger.debug(f"running _calculate_objective_from_service_response ... - Final: {final_value:.4f}")
                else:
                    final_value = primary_value
                    logger.debug(f"running _calculate_objective_from_service_response ... SIMPLE mode: using primary value {final_value:.4f}")
                
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
                    raise ValueError(f"Health-only objective '{self.config.objective.value}' requires HEALTH mode")
                return float(overall_health)
            
            elif self.config.objective == OptimizationObjective.NEURON_UTILIZATION:
                if self.config.mode != OptimizationMode.HEALTH:
                    raise ValueError(f"Health-only objective '{self.config.objective.value}' requires HEALTH mode")
                return float(health_metrics.get('neuron_utilization', 0.5))
            
            elif self.config.objective == OptimizationObjective.TRAINING_STABILITY:
                if self.config.mode != OptimizationMode.HEALTH:
                    raise ValueError(f"Health-only objective '{self.config.objective.value}' requires HEALTH mode")
                return float(health_metrics.get('training_stability', 0.5))
            
            elif self.config.objective == OptimizationObjective.GRADIENT_HEALTH:
                if self.config.mode != OptimizationMode.HEALTH:
                    raise ValueError(f"Health-only objective '{self.config.objective.value}' requires HEALTH mode")
                return float(health_metrics.get('gradient_health', 0.5))
            
            else:
                # Fallback to test accuracy
                logger.warning(f"running _calculate_objective_from_service_response ... Unknown objective {self.config.objective.value}, using test_accuracy")
                return float(test_accuracy)
            
        except Exception as e:
            logger.error(f"running _calculate_objective_from_service_response ... Error calculating objective: {e}")
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
            model_config = ModelConfig()
            
            # Apply all suggested parameters
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
            
            # Create ModelBuilder (now without GPU proxy since we're doing pure local)
            model_builder = ModelBuilder(self.dataset_config, model_config)
            
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
            
            # Train the model
            history = model_builder.train(
                data=training_data,
                validation_split=self.config.validation_split
            )
            
            # Calculate training time
            training_time_minutes = (time.time() - trial_start_time) / 60

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
            overall_health = comprehensive_metrics.get('overall_health', 0.5)
            
            logger.debug(f"running _train_locally_for_trial ... Trial {trial.number}: Local fallback completed")
            logger.debug(f"running _train_locally_for_trial ... - Test accuracy: {test_accuracy:.4f}")
            logger.debug(f"running _train_locally_for_trial ... - Overall health: {overall_health:.3f}")
            
            # Calculate objective (reuse same logic as service response)
            if self.config.objective == OptimizationObjective.VAL_ACCURACY:
                if self.config.mode == OptimizationMode.HEALTH and not OptimizationObjective.is_health_only(self.config.objective):
                    objective_weight = 1.0 - self.config.health_weight
                    health_weight = self.config.health_weight
                    objective_value = objective_weight * test_accuracy + health_weight * overall_health
                else:
                    objective_value = test_accuracy
            else:
                # For other objectives, use test_accuracy as fallback
                objective_value = test_accuracy
            
            logger.debug(f"running _train_locally_for_trial ... Trial {trial.number}: Local fallback objective value: {objective_value:.4f}")
            
            return float(objective_value)
            
        except Exception as e:
            logger.error(f"running _train_locally_for_trial ... Local training fallback failed for trial {trial.number}: {e}")
            raise

    def optimize(self) -> OptimizationResult:
        """
        Run optimization study to find best hyperparameters
        
        Returns:
            OptimizationResult with best parameters and comprehensive metrics
        """
        logger.debug(f"running ModelOptimizer.optimize ... Starting optimization for {self.dataset_name}")
        logger.debug(f"running ModelOptimizer.optimize ... Mode: {self.config.mode.value}")
        logger.debug(f"running ModelOptimizer.optimize ... Objective: {self.config.objective.value}")
        logger.debug(f"running ModelOptimizer.optimize ... Trials: {self.config.n_trials}")
        
        # 🔄 ARCHITECTURAL PIVOT: Log execution approach
        if self.config.use_runpod_service:
            logger.debug(f"running ModelOptimizer.optimize ... 🔄 EXECUTION: RunPod Service (JSON API, tiny payloads)")
            logger.debug(f"running ModelOptimizer.optimize ... Endpoint: {self.config.runpod_service_endpoint}")
        else:
            logger.debug(f"running ModelOptimizer.optimize ... EXECUTION: Local only")
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(n_startup_trials=self.config.n_startup_trials, seed=self.config.random_seed),
            pruner=MedianPruner(
                n_startup_trials=self.config.n_startup_trials,
                n_warmup_steps=self.config.n_warmup_steps
            )
        )
        
        # Record optimization start time
        self.optimization_start_time = time.time()
        
        # Run optimization
        try:
            self.study.optimize(self._objective_function, n_trials=self.config.n_trials)
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
        
        logger.debug(f"running ModelOptimizer.optimize ... Optimization completed successfully")
        
        return results
    
    def _objective_function(self, trial: optuna.Trial) -> float:
        """
        🔄 ARCHITECTURAL PIVOT: Objective function with RunPod service integration
        
        Now checks execution method:
        1. RunPod service (JSON API approach) - if enabled and configured
        2. Local execution - fallback or when service disabled
        
        Args:
            trial: Optuna trial object for parameter suggestion
            
        Returns:
            Objective value (higher is better for maximization objectives)
        """
        try:
            # PHASE 2 REFACTORING: Use modular hyperparameter suggestion
            params = self._suggest_hyperparameters(trial)
            
            # 🔄 ARCHITECTURAL PIVOT: Check execution method
            if self._should_use_runpod_service():
                logger.debug(f"running _objective_function ... Trial {trial.number}: 🔄 Using RunPod service (JSON API)")
                return self._train_via_runpod_service(trial, params)
            else:
                logger.debug(f"running _objective_function ... Trial {trial.number}: Using local execution")
                return self._train_locally_for_trial(trial, params)
            
        except Exception as e:
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
                best_value=0.0,
                best_params={},
                total_trials=len(self.study.trials),
                successful_trials=0,
                optimization_time_hours=optimization_time / 3600,
                optimization_mode=self.config.mode.value,
                health_weight=self.config.health_weight,
                objective_history=[],
                parameter_importance={},
                health_history=self.trial_health_history,
                best_trial_health=self.best_trial_health,
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
            best_value=self.study.best_value,
            best_params=best_params,
            total_trials=len(self.study.trials),
            successful_trials=len(completed_trials),
            optimization_time_hours=optimization_time / 3600,
            optimization_mode=self.config.mode.value,
            health_weight=self.config.health_weight,
            objective_history=[t.value for t in self.study.trials if t.value is not None],
            parameter_importance=importance,
            health_history=self.trial_health_history,
            best_trial_health=self.best_trial_health,
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
            
            # Save best hyperparameters as YAML
            yaml_file = self.results_dir / "best_hyperparameters.yaml"
            yaml_data = {
                "dataset": results.dataset_name,
                "optimization_mode": results.optimization_mode,
                "objective": results.optimization_config.objective.value if results.optimization_config else "unknown",
                "health_weight": results.health_weight,
                "best_value": float(results.best_value),
                "hyperparameters": best_params,
                "execution_method": "runpod_service" if results.optimization_config and results.optimization_config.use_runpod_service else "local"  # 🔄 ARCHITECTURAL PIVOT: Track execution method
            }
            
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            logger.debug(f"running ModelOptimizer._save_results ... Saved best hyperparameters to {yaml_file}")
            
            logger.debug(f"running ModelOptimizer._save_results ... Successfully saved optimization results")
            
        except Exception as e:
            logger.error(f"running ModelOptimizer._save_results ... Failed to save optimization results: {e}")


# Convenience function for command-line usage with RunPod service support
def optimize_model(
    dataset_name: str,
    mode: str = "simple",
    optimize_for: str = "val_accuracy",
    trials: int = 50,
    run_name: Optional[str] = None,
    activation: Optional[str] = None,
    progress_callback: Optional[Callable[[TrialProgress], None]] = None,
    # 🔄 ARCHITECTURAL PIVOT: RunPod service parameters (replacing GPU proxy)
    use_runpod_service: bool = False,
    runpod_service_endpoint: Optional[str] = None,
    runpod_service_timeout: int = 600,
    runpod_service_fallback_local: bool = True,
    gpu_proxy_sample_percentage: float = 0.5,
    **config_overrides
) -> OptimizationResult:
    """
    🔄 ARCHITECTURAL PIVOT: Convenience function with RunPod service support
    
    Args:
        dataset_name: Name of dataset to optimize
        mode: Optimization mode ("simple" or "health")
        optimize_for: Optimization objective
        trials: Number of trials to run
        run_name: Optional unified run name for consistent directory/file naming
        progress_callback: Optional callback for real-time progress updates
        # 🔄 ARCHITECTURAL PIVOT: RunPod service parameters
        use_runpod_service: Enable/disable RunPod service usage
        runpod_service_endpoint: RunPod service endpoint URL
        runpod_service_timeout: Request timeout in seconds
        runpod_service_fallback_local: Fall back to local execution if service fails
        **config_overrides: Additional optimization config overrides
        
    Returns:
        OptimizationResult with best parameters and metrics
        
    Examples:
        # Local execution
        result = optimize_model('cifar10', mode='simple', optimize_for='val_accuracy', trials=20)
        
        # 🔄 ARCHITECTURAL PIVOT: RunPod service execution (JSON API)
        result = optimize_model('cifar10', mode='simple', optimize_for='val_accuracy', 
                               trials=20, use_runpod_service=True, 
                               runpod_service_endpoint='https://your-runpod-endpoint.com')
    """
    # Create unified run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        dataset_clean = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        if mode == 'health':
            run_name = f"{timestamp}_{dataset_clean}_health"
        elif mode == 'simple':
            run_name = f"{timestamp}_{dataset_clean}_simple-{optimize_for}"
        else:
            run_name = f"{timestamp}_{dataset_clean}_{mode}"
    
    logger.debug(f"running optimize_model ... Using unified run name: {run_name}")
    
    # Convert string parameters to enums
    try:
        opt_mode = OptimizationMode(mode.lower())
    except ValueError:
        available_modes = [m.value for m in OptimizationMode]
        raise ValueError(f"Unknown optimization mode '{mode}'. Available: {available_modes}")
    
    try:
        objective = OptimizationObjective(optimize_for.lower())
    except ValueError:
        available_objectives = [obj.value for obj in OptimizationObjective]
        raise ValueError(f"Unknown objective '{optimize_for}'. Available: {available_objectives}")
    
    # Early validation
    if opt_mode == OptimizationMode.SIMPLE and OptimizationObjective.is_health_only(objective):
        universal_objectives = [obj.value for obj in OptimizationObjective.get_universal_objectives()]
        health_objectives = [obj.value for obj in OptimizationObjective.get_health_only_objectives()]
        raise ValueError(
            f"Cannot use health-only objective '{optimize_for}' in SIMPLE mode.\n"
            f"Available objectives for SIMPLE mode: {universal_objectives}\n"
            f"Health-only objectives (require HEALTH mode): {health_objectives}\n"
            f"To use '{optimize_for}', change mode to 'health'"
        )
    
    # Create optimization config
    opt_config = OptimizationConfig(
        mode=opt_mode,
        objective=objective,
        n_trials=trials,
        # 🔄 ARCHITECTURAL PIVOT: RunPod service configuration
        use_runpod_service=use_runpod_service,
        runpod_service_endpoint=runpod_service_endpoint,
        runpod_service_timeout=runpod_service_timeout,
        runpod_service_fallback_local=runpod_service_fallback_local,
        gpu_proxy_sample_percentage=gpu_proxy_sample_percentage
    )
    
    # Apply config overrides
    for key, value in config_overrides.items():
        if hasattr(opt_config, key):
            if key == 'plot_generation' and isinstance(value, str):
                try:
                    if value.lower() == 'all':
                        value = PlotGenerationMode.ALL
                    elif value.lower() == 'best':
                        value = PlotGenerationMode.BEST
                    elif value.lower() == 'none':
                        value = PlotGenerationMode.NONE
                    else:
                        logger.warning(f"running optimize_model ... Invalid plot_generation value: '{value}', defaulting to ALL")
                        value = PlotGenerationMode.ALL
                except Exception as e:
                    logger.warning(f"running optimize_model ... Error converting plot_generation: {e}, defaulting to ALL")
                    value = PlotGenerationMode.ALL
            
            setattr(opt_config, key, value)
            logger.debug(f"running optimize_model ... Set {key} = {value}")
        else:
            logger.warning(f"running optimize_model ... Unknown config parameter: {key}")
    
    # 🔄 ARCHITECTURAL PIVOT: LOG EXECUTION APPROACH
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
        dataset_name=dataset_name,
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
    dataset_name = args.get('dataset', 'cifar10')
    mode = args.get('mode', 'simple')
    optimize_for = args.get('optimize_for', 'val_accuracy')
    trials = int(args.get('trials', '50'))
    run_name = args.get('run_name', None)
    activation = args.get('activation', None)
    
    # 🔄 ARCHITECTURAL PIVOT: Extract RunPod service parameters
    use_runpod_service = args.get('use_runpod_service', 'false').lower() in ['true', '1', 'yes', 'on']
    runpod_service_endpoint = args.get('runpod_service_endpoint', None)
    runpod_service_timeout = int(args.get('runpod_service_timeout', '600'))
    runpod_service_fallback_local = args.get('runpod_service_fallback_local', 'true').lower() in ['true', '1', 'yes', 'on']
    
    # Convert parameters
    int_params = [
        'n_trials', 'n_startup_trials', 'n_warmup_steps', 'random_seed',
        'max_epochs_per_trial', 'early_stopping_patience', 'min_epochs_per_trial',
        'stability_window', 'health_analysis_sample_size', 'health_monitoring_frequency',
        'runpod_service_timeout'
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
        'save_best_model', 'save_optimization_history', 'create_comparison_plots',
        'enable_early_stopping', 'enable_stability_checks',
        'use_runpod_service', 'runpod_service_fallback_local'  # 🔄 ARCHITECTURAL PIVOT
    ]
    for bool_param in bool_params:
        if bool_param in args:
            args[bool_param] = args[bool_param].lower() in ['true', '1', 'yes', 'on']
    
    # Handle string parameters
    string_params = ['plot_generation', 'activation', 'runpod_service_endpoint']  # 🔄 ARCHITECTURAL PIVOT
    for string_param in string_params:
        if string_param in args:
            if args[string_param].strip():
                logger.debug(f"running optimizer.py ... Set {string_param}: {args[string_param]}")
            else:
                logger.warning(f"running optimizer.py ... Empty {string_param}, removing")
                del args[string_param]
    
    # Convert plot_generation string to enum
    if 'plot_generation' in args:
        plot_gen_str = args['plot_generation']
        if isinstance(plot_gen_str, str):
            plot_gen_str = plot_gen_str.lower()
            try:
                if plot_gen_str == 'all':
                    args['plot_generation'] = PlotGenerationMode.ALL
                elif plot_gen_str == 'best':
                    args['plot_generation'] = PlotGenerationMode.BEST
                elif plot_gen_str == 'none':
                    args['plot_generation'] = PlotGenerationMode.NONE
                else:
                    logger.warning(f"running optimizer.py ... Invalid plot_generation value: '{args['plot_generation']}'")
                    logger.warning(f"running optimizer.py ... Valid options: all, best, none. Using default 'all'")
                    args['plot_generation'] = PlotGenerationMode.ALL
                
                logger.debug(f"running optimizer.py ... Converted plot_generation to enum: {args['plot_generation']}")
            except Exception as e:
                logger.warning(f"running optimizer.py ... Error converting plot_generation: {e}")
                args['plot_generation'] = PlotGenerationMode.ALL
    
    logger.debug(f"running optimizer.py ... Starting optimization")
    logger.debug(f"running optimizer.py ... Dataset: {dataset_name}")
    logger.debug(f"running optimizer.py ... Mode: {mode}")
    logger.debug(f"running optimizer.py ... Objective: {optimize_for}")
    logger.debug(f"running optimizer.py ... Trials: {trials}")
    if run_name:
        logger.debug(f"running optimizer.py ... Run name: {run_name}")
    
    # 🔄 ARCHITECTURAL PIVOT: LOG EXECUTION APPROACH
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
            use_runpod_service=use_runpod_service,  # 🔄 ARCHITECTURAL PIVOT
            runpod_service_endpoint=runpod_service_endpoint,  # 🔄 ARCHITECTURAL PIVOT
            runpod_service_timeout=runpod_service_timeout,  # 🔄 ARCHITECTURAL PIVOT
            runpod_service_fallback_local=runpod_service_fallback_local,  # 🔄 ARCHITECTURAL PIVOT
            **{k: v for k, v in args.items() if k not in ['dataset', 'mode', 'optimize_for', 'trials', 'run_name', 'activation', 'use_runpod_service', 'runpod_service_endpoint', 'runpod_service_timeout', 'runpod_service_fallback_local']}
        )
        
        # Print results
        print(result.summary())
        
        logger.debug(f"running optimizer.py ... ✅ Optimization completed successfully!")
        
        # 🔄 ARCHITECTURAL PIVOT: LOG EXECUTION METHOD IN RESULTS
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
            print(f"   python optimizer.py dataset={dataset_name} use_runpod_service=true runpod_service_fallback_local=true")
            print(f"3. Use local execution only:")
            print(f"   python optimizer.py dataset={dataset_name} use_runpod_service=false")
        else:
            print(f"\n❌ Error: {error_msg}")
        
        logger.error(f"running optimizer.py ... ❌ Optimization failed: {e}")
        sys.exit(1)