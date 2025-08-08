"""
Unified Model Optimizer for Multi-Modal Classification - PHASE 2 REFACTORING

Automated hyperparameter optimization system that integrates with ModelBuilder
and DatasetManager to find optimal configurations for any supported dataset.

PHASE 2 REFACTORING: Transforming into pure orchestrator
- ✅ STEP 2.1a: Added HyperparameterSelector integration
- ✅ STEP 2.1b: Removed embedded hyperparameter suggestion methods
- ✅ STEP 2.1c: Added PlotGenerator integration

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
from enum import Enum
import json
import numpy as np
import optuna
import optuna.integration.keras as optuna_keras
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pathlib import Path
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
from plot_generator import PlotGenerator  # ✅ STEP 2.1c: Added PlotGenerator


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
    
    Mode Behavior:
        SIMPLE mode: Uses pure objective values, health metrics for monitoring only
        HEALTH mode: 
            - Universal objectives: weighted combination (1-health_weight)*objective + health_weight*health
            - Health-only objectives: direct health optimization
    
    Usage Examples:
        # Simple mode - pure objectives only
        config = OptimizationConfig(mode=SIMPLE, objective=VAL_ACCURACY)
        
        # Health mode - default 30% health weighting
        config = OptimizationConfig(mode=HEALTH, objective=VAL_ACCURACY)
        
        # Health mode - balanced 50/50
        config = OptimizationConfig(mode=HEALTH, objective=VAL_ACCURACY, health_weight=0.5)
        
        # Health mode - direct health optimization
        config = OptimizationConfig(mode=HEALTH, objective=OVERALL_HEALTH)
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
    n_trials: int = 50 # Total length of optimization is n epochs per model build * n trials
    timeout_hours: Optional[float] = None  # None = no timeout
    
    # Health weighting (only used in HEALTH mode with universal objectives)
    health_weight: float = 0.3  # Default: 70% objective, 30% health
    
    # Pruning and sampling
    n_startup_trials: int = 10  # Trials before pruning starts. Done to allow initial exploration and develop baseline.
    n_warmup_steps: int = 5     # Steps before pruning evaluation. Prevents pruning too early in training.
    random_seed: int = 42
    
    # Resource constraints
    max_epochs_per_trial: int = 20
    max_training_time_minutes: float = 60.0
    max_parameters: int = 10_000_000
    min_accuracy_threshold: float = 0.5
    
    # Stability detection parameters
    min_epochs_per_trial: int = 5      # Force longer observation - updated from 15
    enable_stability_checks: bool = True # Monitor for instabilities
    stability_window: int = 3           # Check stability over last N epochs
    max_bias_change_per_epoch: float = 10.0  # Flag rapid bias changes
    
    # Health analysis settings (always enabled for monitoring)
    health_analysis_sample_size: int = 50  # Sample size for activation-based health analysis
    health_monitoring_frequency: int = 1   # Monitor health every N trials (1 = every trial)
    
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
    
    # GPU Proxy Integration parameters (NEW)
    use_gpu_proxy: bool = False                    # Enable/disable GPU proxy usage
    gpu_proxy_auto_clone: bool = True              # Automatically clone GPU proxy repo if not found
    gpu_proxy_endpoint: Optional[str] = None       # Optional specific endpoint override
    gpu_proxy_fallback_local: bool = True          # Fall back to local execution if GPU proxy fails
    
    # Enhanced GPU proxy sampling parameters (NEWEST)
    gpu_proxy_sample_percentage: float = 0.50      # Percentage of training data to use
    gpu_proxy_use_stratified_sampling: bool = True # Use stratified sampling to maintain class balance
    gpu_proxy_adaptive_batch_size: bool = True     # Adapt batch size to sample count
    gpu_proxy_optimize_data_types: bool = True     # Optimize data types for transfer efficiency
    gpu_proxy_compression_level: int = 6           # Compression level for large payloads
    
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
        
        # Log GPU proxy configuration
        if self.use_gpu_proxy:
            logger.debug(f"running OptimizationConfig.__post_init__ ... GPU proxy enabled in optimization config")
            logger.debug(f"running OptimizationConfig.__post_init__ ... - Auto-clone: {self.gpu_proxy_auto_clone}")
            logger.debug(f"running OptimizationConfig.__post_init__ ... - Fallback local: {self.gpu_proxy_fallback_local}")
            logger.debug(f"running OptimizationConfig.__post_init__ ... - Sample percentage: {self.gpu_proxy_sample_percentage:.1%}")
            logger.debug(f"running OptimizationConfig.__post_init__ ... - Stratified sampling: {self.gpu_proxy_use_stratified_sampling}")
            if self.gpu_proxy_endpoint:
                logger.debug(f"running OptimizationConfig.__post_init__ ... - Custom endpoint: {self.gpu_proxy_endpoint}")
        else:
            logger.debug(f"running OptimizationConfig.__post_init__ ... GPU proxy disabled in optimization config")
    
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
    PHASE 2 REFACTORING: Unified optimizer class transforming into pure orchestrator
    
    REFACTORING PROGRESS:
    - ✅ STEP 2.1a: Added HyperparameterSelector integration
    - ✅ STEP 2.1b: Removed embedded hyperparameter suggestion methods
    - ✅ STEP 2.1c: Added PlotGenerator integration
    
    Integrates with existing ModelBuilder and DatasetManager to provide
    automated hyperparameter tuning with simple or health-aware optimization.
    
    Uses configurable health_weight to balance objective vs health importance.
    """
    
    def __init__(self, dataset_name: str, optimization_config: Optional[OptimizationConfig] = None, 
        datasets_root: Optional[str] = None, run_name: Optional[str] = None,
        health_analyzer: Optional[HealthAnalyzer] = None,
        progress_callback: Optional[Callable[[TrialProgress], None]] = None,
        activation_override: Optional[str] = None):
        """
        Initialize ModelOptimizer with optional progress callback for real-time updates
        
        Args:
            dataset_name: Name of dataset to optimize for
            optimization_config: Optimization settings (uses defaults if None)
            datasets_root: Optional custom datasets directory
            run_name: Optional unified run name for consistent directory/file naming
            health_analyzer: Optional HealthAnalyzer instance (creates new if None)
            progress_callback: Optional callback function that receives TrialProgress updates
                            Used by API endpoints to stream real-time progress
        """
        self.dataset_name = dataset_name
        self.config = optimization_config or OptimizationConfig()
        self.run_name = run_name
        self.activation_override = activation_override
        if self.activation_override:
            logger.debug(f"running ModelOptimizer.__init__ ... Activation override: {self.activation_override} (will force this activation for all trials)")
        
        # Enhanced plot tracking
        self.trial_plot_data = {}  # Store plot data for each trial
        self.best_trial_number = None      
        
        # Log plot generation configuration
        logger.debug(f"running ModelOptimizer.__init__ ... Plot generation mode: {self.config.plot_generation.value}")
        if self.config.plot_generation == PlotGenerationMode.ALL:
            logger.debug(f"running ModelOptimizer.__init__ ... Plots will be generated for ALL trials")
        elif self.config.plot_generation == PlotGenerationMode.BEST:
            logger.debug(f"running ModelOptimizer.__init__ ... Plots will be generated for BEST trial only")
        else:
            logger.debug(f"running ModelOptimizer.__init__ ... Plot generation DISABLED")
                    
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
        
        # PHASE 2 REFACTORING: Initialize PlotGenerator (Step 2.1c)
        logger.debug(f"running ModelOptimizer.__init__ ... Initializing PlotGenerator")
        # Note: PlotGenerator needs a ModelConfig, but we don't have trial-specific config yet
        # We'll create a default ModelConfig for PlotGenerator initialization
        default_model_config = ModelConfig()  # Default config for PlotGenerator
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
        logger.debug(f"running ModelOptimizer.__init__ ... Health monitoring: enabled (all modes)")
        if self.run_name:
            logger.debug(f"running ModelOptimizer.__init__ ... Run name: {self.run_name}")
        logger.debug(f"running ModelOptimizer.__init__ ... Real-time trial tracking enabled: {progress_callback is not None}")
        
        # LOG GPU PROXY CONFIGURATION STATUS (NEW)
        if self.config.use_gpu_proxy:
            logger.debug(f"running ModelOptimizer.__init__ ... GPU proxy integration: ENABLED")
            logger.debug(f"running ModelOptimizer.__init__ ... - Auto-clone: {self.config.gpu_proxy_auto_clone}")
            logger.debug(f"running ModelOptimizer.__init__ ... - Fallback local: {self.config.gpu_proxy_fallback_local}")
            logger.debug(f"running ModelOptimizer.__init__ ... - Sample percentage: {self.config.gpu_proxy_sample_percentage:.1%}")
            logger.debug(f"running ModelOptimizer.__init__ ... - Stratified sampling: {self.config.gpu_proxy_use_stratified_sampling}")
            if self.config.gpu_proxy_endpoint:
                logger.debug(f"running ModelOptimizer.__init__ ... - Custom endpoint: {self.config.gpu_proxy_endpoint}")
            logger.debug(f"running ModelOptimizer.__init__ ... GPU proxy will be passed to all ModelBuilder instances")
        else:
            logger.debug(f"running ModelOptimizer.__init__ ... GPU proxy integration: DISABLED (local execution only)")
        
        # PHASE 2 REFACTORING LOG
        logger.debug(f"running ModelOptimizer.__init__ ... PHASE 2 REFACTORING: HyperparameterSelector integrated")
        logger.debug(f"running ModelOptimizer.__init__ ... PHASE 2 STEP 2.1b: Embedded hyperparameter methods removed")
        logger.debug(f"running ModelOptimizer.__init__ ... PHASE 2 STEP 2.1c: PlotGenerator integrated")
        logger.debug(f"running ModelOptimizer.__init__ ... PHASE 2 STATUS: Pure orchestrator transformation complete")
    
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
            # Use the provided run_name directly
            self.results_dir = optimization_results_dir / self.run_name
            logger.debug(f"running _setup_results_directory ... Using provided run_name: {self.run_name}")
        else:
            # Fallback to old naming if no run_name provided
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
        logger.debug(f"running _setup_results_directory ... Plots directory: {plots_dir}")

        # Create a subdirectory within the plots directory for each trial in the run
        for trial_num in range(self.config.n_trials):
            trial_dir = plots_dir / f"trial_{trial_num + 1}"
            trial_dir.mkdir(exist_ok=True)
            logger.debug(f"running _setup_results_directory ... Created trial directory: {trial_dir}")
        
        # Create a subdirectory for the optimized model
        optimized_dir = self.results_dir / "optimized_model"
        optimized_dir.mkdir(exist_ok=True)
        logger.debug(f"running _setup_results_directory ... Optimized model directory: {optimized_dir}")
        
    
    def _create_model_config(self, params: Dict[str, Any]) -> ModelConfig:
        """
        Create ModelConfig from suggested parameters
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            ModelConfig object with suggested parameters and GPU proxy configuration
        """
        logger.debug(f"running _create_model_config ... Creating ModelConfig from params: {params}")
        
        # Create base config
        config = ModelConfig()
        
        # Apply all suggested parameters
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.debug(f"running _create_model_config ... Set {key} = {value} (type: {type(value)})")
            else:
                logger.warning(f"running _create_model_config ... ModelConfig has no attribute '{key}', skipping")
        
        # Apply GPU proxy config from OptimizationConfig to ModelConfig
        config.use_gpu_proxy = self.config.use_gpu_proxy
        config.gpu_proxy_auto_clone = self.config.gpu_proxy_auto_clone
        config.gpu_proxy_endpoint = self.config.gpu_proxy_endpoint
        config.gpu_proxy_fallback_local = self.config.gpu_proxy_fallback_local
        
        # Apply enhanced GPU proxy sampling parameters
        config.gpu_proxy_sample_percentage = self.config.gpu_proxy_sample_percentage
        config.gpu_proxy_use_stratified_sampling = self.config.gpu_proxy_use_stratified_sampling
        config.gpu_proxy_adaptive_batch_size = self.config.gpu_proxy_adaptive_batch_size
        config.gpu_proxy_optimize_data_types = self.config.gpu_proxy_optimize_data_types
        config.gpu_proxy_compression_level = self.config.gpu_proxy_compression_level
        
        # Log GPU proxy configuration transfer
        if self.config.use_gpu_proxy:
            logger.debug(f"running _create_model_config ... GPU proxy configuration applied to ModelConfig")
            logger.debug(f"running _create_model_config ... - use_gpu_proxy: {config.use_gpu_proxy}")
            logger.debug(f"running _create_model_config ... - gpu_proxy_auto_clone: {config.gpu_proxy_auto_clone}")
            logger.debug(f"running _create_model_config ... - gpu_proxy_fallback_local: {config.gpu_proxy_fallback_local}")
            logger.debug(f"running _create_model_config ... - gpu_proxy_sample_percentage: {config.gpu_proxy_sample_percentage:.1%}")
            logger.debug(f"running _create_model_config ... - gpu_proxy_use_stratified_sampling: {config.gpu_proxy_use_stratified_sampling}")
            if config.gpu_proxy_endpoint:
                logger.debug(f"running _create_model_config ... - gpu_proxy_endpoint: {config.gpu_proxy_endpoint}")
        else:
            logger.debug(f"running _create_model_config ... GPU proxy disabled, using local execution only")
        
        logger.debug(f"running _create_model_config ... ModelConfig created successfully")
        return config

    # PHASE 2 REFACTORING: Updated _suggest_hyperparameters to use HyperparameterSelector
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
        
        suggested_activation = params.get('activation', 'NOT_FOUND')
        logger.debug(f"running _suggest_hyperparameters ... HyperparameterSelector returned activation: '{suggested_activation}'")
        logger.debug(f"running _suggest_hyperparameters ... Full params returned: {params}")
        
        return params

    # REST OF THE CLASS METHODS REMAIN THE SAME FOR NOW
    # These will be updated in subsequent steps...
    
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
        logger.debug(f"running ModelOptimizer.optimize ... Health weight: {self.config.health_weight}")
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='maximize',  # We always maximize (convert minimize objectives in _objective_function)
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
        
        # Check if we have any completed trials before trying to build final model
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Handle best trial artifacts based on plot generation mode
        if completed_trials and self.config.plot_generation == PlotGenerationMode.ALL:
            logger.debug("running ModelOptimizer.optimize ... Plot generation is 'all', copying best trial artifacts instead of rebuilding")
            self._copy_best_trial_artifacts()
        elif completed_trials and self.config.plot_generation == PlotGenerationMode.BEST:
            logger.debug("running ModelOptimizer.optimize ... Plot generation is 'best', generating plots for best trial")
            self._generate_best_trial_plots()
        else:
            logger.debug("running ModelOptimizer.optimize ... Plot generation is not 'all' or 'best', proceeding with current final model build logic")
        
        logger.debug(f"running ModelOptimizer.optimize ... Optimization completed successfully")
        
        return results
    
    
    def _copy_best_trial_artifacts(self) -> None:
        """
        Copy the best trial's model and plots to the optimized_model directory
        Only called when plot_generation=ALL since plots already exist
        """
        from utils.logger import logger
        import shutil
        
        logger.debug("running _copy_best_trial_artifacts ... Starting best trial artifact copy")
        
        if self.results_dir is None:
            logger.error("running _copy_best_trial_artifacts ... Results directory not set, cannot copy artifacts")
            return
        
        if not hasattr(self, 'study') or self.study is None or not self.study.trials:
            logger.warning("running _copy_best_trial_artifacts ... No study or trials found, cannot copy artifacts")
            return
        
        # Get best trial
        best_trial = self.study.best_trial
        best_trial_number = best_trial.number
        
        logger.debug(f"running _copy_best_trial_artifacts ... Best trial is #{best_trial_number} with value: {best_trial.value:.4f}")
        
        # Define source and destination paths
        source_trial_dir = self.results_dir / "plots" / f"trial_{best_trial_number + 1}"
        dest_optimized_dir = self.results_dir / "optimized_model"
        
        # Check if source directory exists and has content
        if not source_trial_dir.exists():
            logger.error(f"running _copy_best_trial_artifacts ... Source trial directory not found: {source_trial_dir}")
            logger.error("running _copy_best_trial_artifacts ... This suggests plots were not generated for the best trial")
            return
        
        # Check if directory has any files
        source_files = list(source_trial_dir.iterdir())
        if not source_files:
            logger.warning(f"running _copy_best_trial_artifacts ... Source trial directory is empty: {source_trial_dir}")
            return
        
        # Create optimized_model directory
        dest_optimized_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"running _copy_best_trial_artifacts ... Created optimized_model directory: {dest_optimized_dir}")
        
        try:
            # Copy all contents from best trial directory to optimized_model directory
            files_copied = 0
            dirs_copied = 0
            
            for item in source_trial_dir.iterdir():
                if item.is_file():
                    dest_file = dest_optimized_dir / item.name
                    shutil.copy2(item, dest_file)
                    files_copied += 1
                    logger.debug(f"running _copy_best_trial_artifacts ... Copied file: {item.name}")
                elif item.is_dir():
                    dest_subdir = dest_optimized_dir / item.name
                    if dest_subdir.exists():
                        shutil.rmtree(dest_subdir)  # Remove existing directory
                    shutil.copytree(item, dest_subdir)
                    dirs_copied += 1
                    logger.debug(f"running _copy_best_trial_artifacts ... Copied directory: {item.name}")
            
            logger.debug(f"running _copy_best_trial_artifacts ... Successfully copied {files_copied} files and {dirs_copied} directories from trial_{best_trial_number + 1}")
            logger.debug(f"running _copy_best_trial_artifacts ... Best trial artifacts available in: {dest_optimized_dir}")
            
        except Exception as e:
            logger.error(f"running _copy_best_trial_artifacts ... Failed to copy artifacts: {e}")
            # Fallback: log what we attempted to copy
            logger.debug(f"running _copy_best_trial_artifacts ... Attempted to copy from: {source_trial_dir}")
            logger.debug(f"running _copy_best_trial_artifacts ... Available items: {[item.name for item in source_trial_dir.iterdir()]}")
    
    def _objective_function(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna for each trial
        
        Args:
            trial: Optuna trial object for parameter suggestion
            
        Returns:
            Objective value (higher is better for maximization objectives)
        """
        try:
            trial_start = time.time()
            
            # PHASE 2 REFACTORING: Use modular hyperparameter suggestion
            params = self._suggest_hyperparameters(trial)
            
            # Create ModelConfig from suggested parameters
            model_config = self._create_model_config(params)
            
            # Record trial start time
            trial_start_time = time.time()
            
            # Create ModelBuilder with GPU proxy integration
            model_builder = ModelBuilder(self.dataset_config, model_config)
            
            # Build model
            model_builder.build_model()
            
            # Check that model was built successfully
            if model_builder.model is None:
                raise RuntimeError("Model building failed - model is None")
            
            # Prepare data dictionary for ModelBuilder.train()
            training_data = {
                'x_train': self.data['x_train'],
                'y_train': self.data['y_train'],
                'x_test': self.data['x_test'],  # Not used in training but required by train method
                'y_test': self.data['y_test']   # Not used in training but required by train method
            }
            
            # Safely get epochs value and ensure it's an integer
            trial_epochs = params.get('epochs', self.config.max_epochs_per_trial)
            if not isinstance(trial_epochs, int):
                logger.warning(f"running ModelOptimizer._objective_function ... epochs is not int: {trial_epochs} (type: {type(trial_epochs)}), converting")
                trial_epochs = int(trial_epochs)
            
            # Respect the validated configuration
            min_epochs = self.config.min_epochs_per_trial
            max_epochs = self.config.max_epochs_per_trial
            
            # Ensure trial epochs are within the validated range
            final_epochs = max(min_epochs, min(trial_epochs, max_epochs))

            if final_epochs != trial_epochs:
                logger.debug(f"running _objective_function ... Trial {trial.number}: "
                            f"Adjusted epochs from {trial_epochs} to {final_epochs} "
                            f"(range: {min_epochs}-{max_epochs})")
        
            logger.debug(f"running _objective_function ... Trial {trial.number}: "
                        f"Training for {final_epochs} epochs (min={min_epochs}, max={max_epochs})")
            
            # Update model config with final epochs
            model_builder.model_config.epochs = final_epochs
            
            # Use ModelBuilder.train() method which includes GPU proxy integration
            history = model_builder.train(
                data=training_data,
                validation_split=self.config.validation_split
            )
            
            # Calculate training time
            training_time_minutes = (time.time() - trial_start_time) / 60

            # Calculate objective value based on optimization mode and target
            objective_value = self._calculate_objective_value(
                history=history,
                model=model_builder.model,
                training_time_minutes=training_time_minutes,
                total_params=model_builder.model.count_params(),
                trial=trial
            )
            
            # Plot generation: Just call ModelBuilder.evaluate() when needed
            should_generate_plots = self._should_generate_plots_for_trial(trial.number, objective_value)
            
            if should_generate_plots:
                logger.debug(f"running _objective_function ... Generating plots for trial {trial.number}")
                
                # Create plot directory for this trial
                if self.results_dir is None:
                    logger.warning(f"running _objective_function ... Results directory not set, skipping plot generation for trial {trial.number}")
                    return objective_value
                trial_plot_dir = self.results_dir / "plots" / f"trial_{trial.number + 1}"
                trial_plot_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate timestamp
                run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                
                try:
                    # PHASE 2 REFACTORING: Use PlotGenerator for comprehensive plot generation
                    logger.debug(f"running _objective_function ... Using PlotGenerator for trial {trial.number}")
                    
                    # Get basic evaluation metrics first
                    test_loss, test_accuracy = model_builder.evaluate(data=training_data)
                    
                    # Use PlotGenerator for comprehensive plot generation
                    plot_analysis_results = self.plot_generator.generate_comprehensive_plots(
                        model=model_builder.model,
                        training_history=history,
                        data=training_data,
                        test_loss=test_loss,
                        test_accuracy=test_accuracy,
                        run_timestamp=run_timestamp,
                        plot_dir=trial_plot_dir,
                        log_detailed_predictions=True,
                        max_predictions_to_show=10
                    )
                    
                    logger.debug(f"running _objective_function ... Trial {trial.number} plots generated: "
                                f"test_accuracy={test_accuracy:.4f}, plots_dir={trial_plot_dir}")
                    
                    # Store comprehensive plot information
                    self.trial_plot_data[trial.number] = {
                        'objective_value': objective_value,
                        'test_accuracy': test_accuracy,
                        'test_loss': test_loss,
                        'plot_dir': str(trial_plot_dir),
                        'plots_generated': True,
                        'plot_analysis_results': plot_analysis_results  # Store detailed analysis results
                    }
                    
                    # Save the trained model alongside plots when plot_generation=ALL
                    if self.config.plot_generation == PlotGenerationMode.ALL:
                        try:
                            # Save model to the same trial directory as the plots
                            model_save_path = model_builder.save_model(
                                test_accuracy=test_accuracy,
                                run_timestamp=run_timestamp,
                                run_name=None  # Don't use run_name here, we want it in trial directory
                            )
                            
                            # Move the saved model to the trial directory
                            import shutil
                            from pathlib import Path
                            
                            saved_model_file = Path(model_save_path)
                            trial_model_path = trial_plot_dir / saved_model_file.name
                            
                            if saved_model_file.exists():
                                shutil.move(str(saved_model_file), str(trial_model_path))
                                logger.debug(f"running _objective_function ... Moved model to trial directory: {trial_model_path}")
                                
                                # Update trial_plot_data to include the trial-specific model path
                                self.trial_plot_data[trial.number]['model_path'] = str(trial_model_path)
                            else:
                                logger.warning(f"running _objective_function ... Saved model file not found: {model_save_path}")
                            
                        except Exception as model_save_error:
                            logger.warning(f"running _objective_function ... Failed to save model for trial {trial.number}: {model_save_error}")
                    
                except Exception as e:
                    logger.warning(f"running _objective_function ... Plot generation failed for trial {trial.number}: {e}")
                    # Fallback to basic evaluation if PlotGenerator fails
                    try:
                        test_loss, test_accuracy = model_builder.evaluate(data=training_data)
                        self.trial_plot_data[trial.number] = {
                            'objective_value': objective_value,
                            'test_accuracy': test_accuracy,
                            'test_loss': test_loss,
                            'plots_generated': False,
                            'error': str(e)
                        }
                    except Exception as eval_error:
                        logger.error(f"running _objective_function ... Both plot generation and evaluation failed for trial {trial.number}: {eval_error}")
                        self.trial_plot_data[trial.number] = {
                            'objective_value': objective_value,
                            'plots_generated': False,
                            'error': f"Plot generation failed: {e}, Evaluation failed: {eval_error}"
                        }
            
            # Log trial success
            final_accuracy = history.history.get('val_accuracy', [0])[-1] if history.history.get('val_accuracy') else 0
            logger.debug(f"running ModelOptimizer._objective_function ... Trial {trial.number} completed: val_accuracy={final_accuracy:.4f}")
            logger.debug(f"running ModelOptimizer._objective_function ... Trial stats: accuracy={final_accuracy:.3f}, params={model_builder.model.count_params():,}, time={training_time_minutes:.1f}min")
            
            return objective_value
            
        except Exception as e:
            logger.error(f"running ModelOptimizer._objective_function ... Trial {trial.number} failed: {str(e)}")
            logger.error(f"running ModelOptimizer._objective_function ... Traceback: {traceback.format_exc()}")
            raise
           
    def _calculate_objective_value(
        self,
        history: Any,
        model: Any,
        training_time_minutes: float,
        total_params: int,
        trial: optuna.Trial
    ) -> float:
        """
        Calculate objective value with robust error handling for empty val_accuracy
        """
        logger.debug(f"running _calculate_objective_value ... Calculating objective for trial {trial.number}")
        logger.debug(f"running _calculate_objective_value ... history.history is: {history.history}")
        
        try:
            # Validate history object
            if not history or not hasattr(history, 'history'):
                logger.error(f"running _calculate_objective_value ... Invalid history object")
                return 0.0
            
            history_dict = history.history
            if not history_dict:
                logger.error(f"running _calculate_objective_value ... Empty history dictionary")
                return 0.0
            
            # Try validation accuracy first
            # Try validation accuracy first (check both names for compatibility)
            val_accuracy = history_dict.get('val_categorical_accuracy', history_dict.get('val_accuracy', []))
            if val_accuracy and len(val_accuracy) > 0:
                final_val_accuracy = val_accuracy[-1]
                # Check if validation accuracy is meaningful (> 0)
                if final_val_accuracy > 0.0:
                    logger.debug(f"running _calculate_objective_value ... Using val_accuracy: {final_val_accuracy:.4f}")
                    return float(final_val_accuracy)
                else:
                    logger.warning(f"running _calculate_objective_value ... val_accuracy is {final_val_accuracy} (all zeros), falling back to training accuracy")
            
            # Fallback to training accuracy
            accuracy = history_dict.get('categorical_accuracy', history_dict.get('accuracy', []))
            if accuracy and len(accuracy) > 0:
                final_accuracy = accuracy[-1]
                logger.debug(f"running _calculate_objective_value ... Using training accuracy: {final_accuracy:.4f}")
                return float(final_accuracy)
            
            # No accuracy found
            logger.error(f"running _calculate_objective_value ... No accuracy metrics found")
            return 0.0
            
        except Exception as e:
            logger.error(f"running _calculate_objective_value ... Error: {e}")
            return 0.0
        
    # PHASE 2 REFACTORING: ✅ Step 2.1b COMPLETE - Embedded domain logic removed
    # Domain-specific hyperparameter suggestion now handled by HyperparameterSelector module
    
    def _generate_best_trial_plots(self) -> None:
        """
        PHASE 2 REFACTORING: Generate plots for the best trial only (BEST mode)
        
        This method rebuilds the best model and generates comprehensive plots
        for the optimized_model directory when plot_generation=BEST.
        """
        logger.debug("running _generate_best_trial_plots ... Starting best trial plot generation")
        
        if self.study is None or not self.study.trials:
            logger.warning("running _generate_best_trial_plots ... No study or trials found, cannot generate plots")
            return
        
        if self.results_dir is None:
            logger.error("running _generate_best_trial_plots ... Results directory not set, cannot generate plots")
            return
        
        try:
            # Get best trial parameters
            best_trial = self.study.best_trial
            best_params = best_trial.params
            
            logger.debug(f"running _generate_best_trial_plots ... Best trial #{best_trial.number} with value: {best_trial.value:.4f}")
            
            # Create ModelConfig from best parameters
            model_config = self._create_model_config(best_params)
            
            # Create ModelBuilder with best configuration
            model_builder = ModelBuilder(self.dataset_config, model_config)
            
            # Build and train the best model
            logger.debug("running _generate_best_trial_plots ... Building best model")
            model_builder.build_model()
            
            # Prepare training data
            training_data = {
                'x_train': self.data['x_train'],
                'y_train': self.data['y_train'],
                'x_test': self.data['x_test'],
                'y_test': self.data['y_test']
            }
            
            # Train the best model
            logger.debug("running _generate_best_trial_plots ... Training best model")
            history = model_builder.train(
                data=training_data,
                validation_split=self.config.validation_split
            )
            
            # Get evaluation metrics
            test_loss, test_accuracy = model_builder.evaluate(data=training_data)
            
            # Generate plots using PlotGenerator
            optimized_dir = self.results_dir / "optimized_model"
            optimized_dir.mkdir(parents=True, exist_ok=True)
            
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            
            logger.debug("running _generate_best_trial_plots ... Generating comprehensive plots using PlotGenerator")
            plot_analysis_results = self.plot_generator.generate_comprehensive_plots(
                model=model_builder.model,
                training_history=history,
                data=training_data,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
                run_timestamp=run_timestamp,
                plot_dir=optimized_dir,
                log_detailed_predictions=True,
                max_predictions_to_show=20  # More detailed analysis for best model
            )
            
            # Save the best model
            logger.debug("running _generate_best_trial_plots ... Saving best model")
            model_save_path = model_builder.save_model(
                test_accuracy=test_accuracy,
                run_timestamp=run_timestamp,
                run_name=f"optimized_{self.dataset_name}"
            )
            
            # Move model to optimized_model directory
            import shutil
            from pathlib import Path
            
            saved_model_file = Path(model_save_path)
            optimized_model_path = optimized_dir / saved_model_file.name
            
            if saved_model_file.exists():
                if optimized_model_path.exists():
                    optimized_model_path.unlink()  # Remove existing file
                shutil.move(str(saved_model_file), str(optimized_model_path))
                logger.debug(f"running _generate_best_trial_plots ... Best model saved to: {optimized_model_path}")
            
            logger.debug(f"running _generate_best_trial_plots ... Best trial plots and model generated successfully")
            logger.debug(f"running _generate_best_trial_plots ... Final accuracy: {test_accuracy:.4f}")
            logger.debug(f"running _generate_best_trial_plots ... All artifacts saved to: {optimized_dir}")
            
        except Exception as e:
            logger.error(f"running _generate_best_trial_plots ... Failed to generate best trial plots: {e}")
            logger.error(f"running _generate_best_trial_plots ... Error details: {traceback.format_exc()}")
    
    
    """
    Fix for optimizer.py _compile_results method to include activation override
    in the OptimizationResult.best_params

    The issue: The test validation accesses result.best_params from the OptimizationResult
    object, but our previous fix only applied to the saved YAML file. The OptimizationResult
    object itself still contains the original best_params from Optuna which doesn't include
    activation override.

    Solution: Add activation override to best_params in _compile_results() method.
    """

    def _compile_results(self) -> OptimizationResult:
        """Compile optimization results into structured format"""
        if self.study is None:
            raise RuntimeError("No study available - run optimize() first")
        
        optimization_time = time.time() - self.optimization_start_time if self.optimization_start_time else 0.0
        
        # Check if we have any completed trials
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            logger.warning(f"running ModelOptimizer._compile_results ... No trials completed successfully")
            # Return a result object with default values when no trials completed
            return OptimizationResult(
                best_value=0.0,  # Default value
                best_params={},  # Empty params
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
        best_params = self.study.best_params.copy()  # Make a copy to avoid modifying original
        
        # DEBUG LOGGING: Check best_params before fix
        logger.debug(f"running _compile_results ... best_params from Optuna (before fix): {best_params}")
        logger.debug(f"running _compile_results ... activation override: {self.activation_override}")
        
        # FIX: Add activation override to best_params if it was used
        if self.activation_override:
            best_params['activation'] = self.activation_override
            logger.debug(f"running _compile_results ... Applied activation override to OptimizationResult.best_params: '{self.activation_override}'")
        
        # DEBUG LOGGING: Check best_params after fix
        logger.debug(f"running _compile_results ... best_params for OptimizationResult (after fix): {best_params}")
        
        # Calculate parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
        except:
            importance = {}
        
        return OptimizationResult(
            best_value=self.study.best_value,
            best_params=best_params,  # Now includes activation override
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
    
    """
Fix for optimizer.py _save_results method to include activation override
in best_params when saving results.

The issue: Optuna's study.best_params only contains parameters that were
suggested by the trial. Activation overrides are applied externally and
don't appear in study.best_params.

Solution: Add activation override to best_params before saving.
"""

    def _save_results(self, results: OptimizationResult) -> None:
        """Save optimization results to disk"""       
        if self.results_dir is None:
            logger.warning("running ModelOptimizer._save_results ... No results directory available, skipping save")
            return
        
        logger.debug(f"running ModelOptimizer._save_results ... Saving optimization results to {self.results_dir}")
        
        try:
            # Get best_params from results
            best_params = results.best_params.copy()  # Make a copy to avoid modifying original
            
            # DEBUG LOGGING: Check what we're about to save (BEFORE fix)
            logger.debug(f"running _save_results ... About to save best_params (before fix): {best_params}")
            logger.debug(f"running _save_results ... best_params activation value (before fix): '{best_params.get('activation', 'NOT_FOUND')}'")
            
            # FIX: Add activation override to best_params if it was used
            if self.activation_override:
                best_params['activation'] = self.activation_override
                logger.debug(f"running _save_results ... Applied activation override to best_params: '{self.activation_override}'")
            
            # DEBUG LOGGING: Check what we're about to save (AFTER fix)
            logger.debug(f"running _save_results ... About to save best_params (after fix): {best_params}")
            logger.debug(f"running _save_results ... best_params activation value (after fix): '{best_params.get('activation', 'NOT_FOUND')}'")
            
            # Additional debug: Show all keys in best_params
            if best_params:
                param_keys = list(best_params.keys())
                logger.debug(f"running _save_results ... All best_params keys: {param_keys}")
                
                # Log a few key parameters for comparison
                for key in ['activation', 'epochs', 'batch_size', 'learning_rate']:
                    if key in best_params:
                        value = best_params[key]
                        logger.debug(f"running _save_results ... {key}: '{value}' (type: {type(value)})")
            else:
                logger.warning(f"running _save_results ... best_params is empty or None!")
            
            # Save best hyperparameters as YAML (human-readable, copy-paste ready)
            yaml_file = self.results_dir / "best_hyperparameters.yaml"
            yaml_data = {
                "dataset": results.dataset_name,
                "optimization_mode": results.optimization_mode,
                "objective": results.optimization_config.objective.value if results.optimization_config else "unknown",
                "health_weight": results.health_weight,
                "best_value": float(results.best_value),
                "hyperparameters": best_params  # Now includes activation override
            }
            
            # DEBUG LOGGING: Check what we're writing to YAML
            logger.debug(f"running _save_results ... yaml_data hyperparameters section: {yaml_data['hyperparameters']}")
            if yaml_data['hyperparameters']:
                yaml_activation = yaml_data['hyperparameters'].get('activation', 'NOT_FOUND')
                logger.debug(f"running _save_results ... yaml_data activation: '{yaml_activation}'")
            
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            logger.debug(f"running ModelOptimizer._save_results ... Saved best hyperparameters to {yaml_file}")
            
            # DEBUG LOGGING: Verify what was actually written to file
            try:
                with open(yaml_file, 'r') as f:
                    saved_yaml = yaml.safe_load(f)
                saved_activation = saved_yaml.get('hyperparameters', {}).get('activation', 'NOT_FOUND')
                logger.debug(f"running _save_results ... VERIFICATION - File contains activation: '{saved_activation}'")
            except Exception as verify_error:
                logger.warning(f"running _save_results ... Could not verify saved file: {verify_error}")
            
            logger.debug(f"running ModelOptimizer._save_results ... Successfully saved optimization results")
            
        except Exception as e:
            logger.error(f"running ModelOptimizer._save_results ... Failed to save optimization results: {e}")
            # Additional debug on error
            logger.error(f"running _save_results ... Error occurred while processing best_params: {results.best_params}")
            logger.error(f"running _save_results ... Results object type: {type(results)}")
            logger.error(f"running _save_results ... Results best_value: {results.best_value}")
            logger.error(f"running _save_results ... Results dataset_name: {results.dataset_name}")

    def _should_generate_plots_for_trial(self, trial_number: int, objective_value: float) -> bool:
        """
        Determine if plots should be generated for this trial
        
        Args:
            trial_number: Current trial number
            objective_value: Objective value achieved by this trial
            
        Returns:
            bool: True if plots should be generated for this trial
        """
        if self.config.plot_generation == PlotGenerationMode.NONE:
            return False
        elif self.config.plot_generation == PlotGenerationMode.ALL:
            return True
        elif self.config.plot_generation == PlotGenerationMode.BEST:
            # For "best" mode, we'll generate plots in post-processing
            # Store trial info for later evaluation of best trial
            self.trial_plot_data[trial_number] = {
                'objective_value': objective_value,
                'needs_plots': False  # Will be set to True for best trial later
            }
            return False
        
        return False

    def _generate_plots_for_trial(
        self, 
        trial: optuna.Trial, 
        model_builder: ModelBuilder, 
        training_data: Dict[str, Any],
        objective_value: float
    ) -> None:
        """
        PHASE 2 REFACTORING: Generate plots for a specific trial using PlotGenerator
        
        Args:
            trial: Optuna trial object
            model_builder: ModelBuilder instance with trained model
            training_data: Training data dictionary
            objective_value: Objective value achieved by this trial
        """
        logger.debug(f"running _generate_plots_for_trial ... Generating plots for trial {trial.number}")
        
        try:
            # Create plot directory for this trial
            if self.results_dir is None:
                logger.error("running _generate_plots_for_trial ... Results directory not set, cannot generate plots")
                return
            trial_plot_dir = self.results_dir / "plots" / f"trial_{trial.number + 1}"
            trial_plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for this evaluation
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            
            # PHASE 2 REFACTORING: Use PlotGenerator for comprehensive plot generation
            logger.debug(f"running _generate_plots_for_trial ... Using PlotGenerator for trial {trial.number}")
            
            # Get basic evaluation metrics first
            test_loss, test_accuracy = model_builder.evaluate(data=training_data)
            
            # Use PlotGenerator for comprehensive plot generation
            plot_analysis_results = self.plot_generator.generate_comprehensive_plots(
                model=model_builder.model,
                training_history=model_builder.training_history,
                data=training_data,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
                run_timestamp=run_timestamp,
                plot_dir=trial_plot_dir,
                log_detailed_predictions=True,
                max_predictions_to_show=10
            )
            
            logger.debug(f"running _generate_plots_for_trial ... Trial {trial.number} plots generated: "
                        f"test_accuracy={test_accuracy:.4f}, plots_dir={trial_plot_dir}")
            
            # Store comprehensive plot information
            self.trial_plot_data[trial.number] = {
                'objective_value': objective_value,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'plot_dir': str(trial_plot_dir),
                'plots_generated': True,
                'plot_analysis_results': plot_analysis_results  # Store detailed analysis results
            }
            
        except Exception as e:
            logger.warning(f"running _generate_plots_for_trial ... Plot generation failed for trial {trial.number}: {e}")
            # Fallback to basic evaluation if PlotGenerator fails
            try:
                test_loss, test_accuracy = model_builder.evaluate(data=training_data)
                self.trial_plot_data[trial.number] = {
                    'objective_value': objective_value,
                    'test_accuracy': test_accuracy,
                    'test_loss': test_loss,
                    'plots_generated': False,
                    'error': str(e)
                }
            except Exception as eval_error:
                logger.error(f"running _generate_plots_for_trial ... Both plot generation and evaluation failed for trial {trial.number}: {eval_error}")
                self.trial_plot_data[trial.number] = {
                    'objective_value': objective_value,
                    'plots_generated': False,
                    'error': f"Plot generation failed: {e}, Evaluation failed: {eval_error}"
                }

# Convenience function for command-line usage with mode selection and GPU proxy support
def optimize_model(
    dataset_name: str,
    mode: str = "simple",
    optimize_for: str = "val_accuracy",
    trials: int = 50,
    run_name: Optional[str] = None,
    activation: Optional[str] = None,
    progress_callback: Optional[Callable[[TrialProgress], None]] = None,
    # GPU proxy parameters (NEW)
    use_gpu_proxy: bool = True,
    gpu_proxy_auto_clone: bool = True,
    gpu_proxy_endpoint: Optional[str] = None,
    gpu_proxy_fallback_local: bool = True,
    # Enhanced GPU proxy sampling parameters (NEWEST)
    gpu_proxy_sample_percentage: float = 0.50,
    gpu_proxy_use_stratified_sampling: bool = True,
    gpu_proxy_adaptive_batch_size: bool = True,
    gpu_proxy_optimize_data_types: bool = True,
    gpu_proxy_compression_level: int = 6,
    **config_overrides
) -> OptimizationResult:
    """
    Convenience function for unified model optimization with enhanced GPU proxy support
    
    Args:
        dataset_name: Name of dataset to optimize
        mode: Optimization mode ("simple" or "health")
        optimize_for: Optimization objective
        trials: Number of trials to run
        run_name: Optional unified run name for consistent directory/file naming
        progress_callback: Optional callback for real-time progress updates
        # GPU proxy parameters (NEW)
        use_gpu_proxy: Enable/disable GPU proxy usage
        gpu_proxy_auto_clone: Automatically clone GPU proxy repo if not found
        gpu_proxy_endpoint: Optional specific endpoint override
        gpu_proxy_fallback_local: Fall back to local execution if GPU proxy fails
        # Enhanced GPU proxy sampling parameters (NEWEST)
        gpu_proxy_sample_percentage: Percentage of training data to use (0.01-1.0)
        gpu_proxy_use_stratified_sampling: Use stratified sampling to maintain class balance
        gpu_proxy_adaptive_batch_size: Adapt batch size to sample count
        gpu_proxy_optimize_data_types: Optimize data types for transfer efficiency
        gpu_proxy_compression_level: Compression level for large payloads (1-9)
        **config_overrides: Additional optimization config overrides
        
    Returns:
        OptimizationResult with best parameters and metrics
        
    Raises:
        ValueError: If mode-objective combination is invalid
        
    Examples:
        # Local execution (default)
        result = optimize_model('cifar10', mode='simple', optimize_for='val_accuracy', trials=20)
        
        # GPU proxy with default settings (50% of data)
        result = optimize_model('cifar10', mode='health', optimize_for='val_accuracy', 
                               trials=20, use_gpu_proxy=True)
        
        # GPU proxy with smaller sample for faster testing (5% of data)
        result = optimize_model('cifar10', mode='health', optimize_for='val_accuracy', 
                               trials=20, use_gpu_proxy=True, gpu_proxy_sample_percentage=0.05)
        
        # GPU proxy with large sample for better accuracy (90% of data)
        result = optimize_model('cifar10', mode='health', optimize_for='val_accuracy', 
                               trials=20, use_gpu_proxy=True, gpu_proxy_sample_percentage=0.90)
    """
    # CREATE UNIFIED RUN NAME HERE if not provided
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
    
    # Convert string mode to enum
    try:
        opt_mode = OptimizationMode(mode.lower())
    except ValueError:
        available_modes = [m.value for m in OptimizationMode]
        raise ValueError(f"Unknown optimization mode '{mode}'. Available: {available_modes}")
    
    # Convert string objective to enum
    try:
        objective = OptimizationObjective(optimize_for.lower())
    except ValueError:
        available_objectives = [obj.value for obj in OptimizationObjective]
        raise ValueError(f"Unknown objective '{optimize_for}'. Available: {available_objectives}")
    
    # Early validation with helpful error messages
    if opt_mode == OptimizationMode.SIMPLE and OptimizationObjective.is_health_only(objective):
        universal_objectives = [obj.value for obj in OptimizationObjective.get_universal_objectives()]
        health_objectives = [obj.value for obj in OptimizationObjective.get_health_only_objectives()]
        raise ValueError(
            f"Cannot use health-only objective '{optimize_for}' in SIMPLE mode.\n"
            f"Available objectives for SIMPLE mode: {universal_objectives}\n"
            f"Health-only objectives (require HEALTH mode): {health_objectives}\n"
            f"To use '{optimize_for}', change mode to 'health'"
        )
    
    # Create optimization config with GPU proxy parameters
    opt_config = OptimizationConfig(
        mode=opt_mode,
        objective=objective,
        n_trials=trials,
        # GPU proxy configuration (NEW)
        use_gpu_proxy=use_gpu_proxy,
        gpu_proxy_auto_clone=gpu_proxy_auto_clone,
        gpu_proxy_endpoint=gpu_proxy_endpoint,
        gpu_proxy_fallback_local=gpu_proxy_fallback_local,
        # Enhanced GPU proxy sampling parameters (NEWEST)
        gpu_proxy_sample_percentage=gpu_proxy_sample_percentage,
        gpu_proxy_use_stratified_sampling=gpu_proxy_use_stratified_sampling,
        gpu_proxy_adaptive_batch_size=gpu_proxy_adaptive_batch_size,
        gpu_proxy_optimize_data_types=gpu_proxy_optimize_data_types,
        gpu_proxy_compression_level=gpu_proxy_compression_level
    )
    
    # Apply any config overrides with proper type handling
    for key, value in config_overrides.items():
        if hasattr(opt_config, key):
            # Special handling for plot_generation enum conversion
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
                    logger.debug(f"running optimize_model ... Converted plot_generation string '{config_overrides[key]}' to enum: {value}")
                except Exception as e:
                    logger.warning(f"running optimize_model ... Error converting plot_generation: {e}, defaulting to ALL")
                    value = PlotGenerationMode.ALL
            
            setattr(opt_config, key, value)
            logger.debug(f"running optimize_model ... Set {key} = {value}")
        else:
            logger.warning(f"running optimize_model ... Unknown config parameter: {key}")
    
    # LOG GPU PROXY CONFIGURATION
    if opt_config.use_gpu_proxy:
        logger.debug(f"running optimize_model ... GPU proxy integration: ENABLED")
        logger.debug(f"running optimize_model ... - Sample percentage: {opt_config.gpu_proxy_sample_percentage:.1%}")
        logger.debug(f"running optimize_model ... - Stratified sampling: {opt_config.gpu_proxy_use_stratified_sampling}")
    else:
        logger.debug(f"running optimize_model ... GPU proxy integration: DISABLED")
    
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
    # Simple command-line interface with mode selection and GPU proxy support
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
    if activation:
        logger.debug(f"running optimizer.py ... Activation override: {activation}")
    
    # Convert integer parameters for OptimizationConfig
    int_params = [
        'n_trials', 'n_startup_trials', 'n_warmup_steps', 'random_seed',
        'max_epochs_per_trial', 'early_stopping_patience', 'min_epochs_per_trial',
        'stability_window', 'health_analysis_sample_size', 'health_monitoring_frequency',
        'gpu_proxy_compression_level'  # Added GPU proxy compression level
    ]
    for int_param in int_params:
        if int_param in args:
            try:
                args[int_param] = int(args[int_param])
                logger.debug(f"running optimizer.py ... Converted {int_param} to int: {args[int_param]}")
            except ValueError:
                logger.warning(f"running optimizer.py ... Invalid {int_param}: {args[int_param]}, using default")
                del args[int_param]
    
    # Convert float parameters for OptimizationConfig
    float_params = [
        'timeout_hours', 'max_training_time_minutes', 'validation_split', 'test_size',
        'max_bias_change_per_epoch', 'health_weight',
        'gpu_proxy_sample_percentage'  # Added GPU proxy sampling percentage
    ]
    for float_param in float_params:
        if float_param in args:
            try:
                value = float(args[float_param])
                
                # SPECIAL HANDLING for gpu_proxy_sample_percentage
                if float_param == 'gpu_proxy_sample_percentage':
                    # If value is > 1.0, assume it's a percentage (e.g., 25 = 25%)
                    # Convert to decimal (e.g., 25 -> 0.25)
                    if value > 1.0:
                        if value <= 100.0:
                            value = value / 100.0
                            logger.debug(f"running optimizer.py ... Converted {float_param} from percentage to decimal: {args[float_param]} -> {value}")
                        else:
                            logger.warning(f"running optimizer.py ... {float_param} value {value} > 100, clamping to 100%")
                            value = 1.0
                    elif value < 0.0:
                        logger.warning(f"running optimizer.py ... {float_param} value {value} < 0, clamping to 0%")
                        value = 0.0
                    # If value is between 0.0 and 1.0, assume it's already a decimal
                    logger.debug(f"running optimizer.py ... Final {float_param} value: {value:.3f} ({value*100:.1f}%)")
                
                args[float_param] = value
                logger.debug(f"running optimizer.py ... Converted {float_param} to float: {args[float_param]}")
            except ValueError:
                logger.warning(f"running optimizer.py ... Invalid {float_param}: {args[float_param]}, using default")
                del args[float_param]
    
    # Convert boolean parameters for OptimizationConfig (INCLUDING GPU PROXY PARAMS)
    bool_params = [
        'save_best_model', 'save_optimization_history', 'create_comparison_plots',
        'enable_early_stopping', 'enable_stability_checks',
        # GPU proxy parameters (NEW)
        'use_gpu_proxy', 'gpu_proxy_auto_clone', 'gpu_proxy_fallback_local',
        'gpu_proxy_use_stratified_sampling', 'gpu_proxy_adaptive_batch_size', 
        'gpu_proxy_optimize_data_types'  # Added GPU proxy sampling options
    ]
    for bool_param in bool_params:
        if bool_param in args:
            args[bool_param] = args[bool_param].lower() in ['true', '1', 'yes', 'on']
            logger.debug(f"running optimizer.py ... Converted {bool_param} to bool: {args[bool_param]}")
    
    # Handle string parameters (including GPU proxy endpoint)
    string_params = ['gpu_proxy_endpoint', 'plot_generation', 'activation']
    for string_param in string_params:
        if string_param in args:
            # Keep as string, but validate it's not empty
            if args[string_param].strip():
                logger.debug(f"running optimizer.py ... Set {string_param}: {args[string_param]}")
            else:
                logger.warning(f"running optimizer.py ... Empty {string_param}, removing")
                del args[string_param]
    
    # Convert plot_generation string to enum (after string_params processing)
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
        elif isinstance(plot_gen_str, PlotGenerationMode):
            # Already an enum, keep as is
            logger.debug(f"running optimizer.py ... plot_generation already enum: {plot_gen_str}")
        else:
            logger.warning(f"running optimizer.py ... Unknown plot_generation type: {type(plot_gen_str)}")
            args['plot_generation'] = PlotGenerationMode.ALL
    
    logger.debug(f"running optimizer.py ... Starting optimization")
    logger.debug(f"running optimizer.py ... Dataset: {dataset_name}")
    logger.debug(f"running optimizer.py ... Mode: {mode}")
    logger.debug(f"running optimizer.py ... Objective: {optimize_for}")
    logger.debug(f"running optimizer.py ... Trials: {trials}")
    if run_name:
        logger.debug(f"running optimizer.py ... Run name: {run_name}")
    
    # LOG GPU PROXY CONFIGURATION
    if args.get('use_gpu_proxy', False):
        logger.debug(f"running optimizer.py ... GPU proxy: ENABLED")
        logger.debug(f"running optimizer.py ... - Sample percentage: {args.get('gpu_proxy_sample_percentage', 0.50):.1%}")
        logger.debug(f"running optimizer.py ... - Stratified sampling: {args.get('gpu_proxy_use_stratified_sampling', True)}")
    else:
        logger.debug(f"running optimizer.py ... GPU proxy: DISABLED (local execution)")
    
    logger.debug(f"running optimizer.py ... Parsed arguments: {args}")
    
    try:
        # Run optimization
        result = optimize_model(
            dataset_name=dataset_name,
            mode=mode,
            optimize_for=optimize_for,
            trials=trials,
            run_name=run_name,
            activation=activation,
            **{k: v for k, v in args.items() if k not in ['dataset', 'mode', 'optimize_for', 'trials', 'run_name', 'activation']}
        )
        
        # Print results
        print(result.summary())
        
        logger.debug(f"running optimizer.py ... ✅ Optimization completed successfully!")
        
        # LOG GPU PROXY USAGE IN RESULTS
        gpu_proxy_used = args.get('use_gpu_proxy', False)
        if gpu_proxy_used:
            sample_pct = args.get('gpu_proxy_sample_percentage', 0.50)
            print(f"\n🚀 GPU Proxy: Used {sample_pct:.1%} of training data per trial")
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
        else:
            print(f"\n❌ Error: {error_msg}")
        
        logger.error(f"running optimizer.py ... ❌ Optimization failed: {e}")
        sys.exit(1)