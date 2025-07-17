"""
Unified Model Optimizer for Multi-Modal Classification

Automated hyperparameter optimization system that integrates with ModelBuilder
and DatasetManager to find optimal configurations for any supported dataset.

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
import yaml

# Import existing modules
from dataset_manager import DatasetManager, DatasetConfig
from health_analyzer import HealthAnalyzer
from model_builder import ModelBuilder, ModelConfig, create_and_train_model
from utils.logger import logger
from model_builder import create_and_train_model


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
    '''
    Pruning allows early stopping of unpromising trials to save resources. If # epochs is 20 and we know the current trial is worse than 
    those that have already been run, we can stop it early, instead of wasting time on the remaining epochs in that bad trial.
    '''
    n_startup_trials: int = 10  # Trials before pruning starts. Done to allow initial exploration and develop baseline.
    n_warmup_steps: int = 5     # Steps before pruning evaluation. Prevents pruning too early in training.
    random_seed: int = 42
    
    # Resource constraints
    max_epochs_per_trial: int = 20
    max_training_time_minutes: float = 60.0
    max_parameters: int = 10_000_000
    min_accuracy_threshold: float = 0.5
    
    # Stability detection parameters
    min_epochs_per_trial: int = 10      # Force longer observation - updated from 15
    enable_stability_checks: bool = True # Monitor for instabilities
    stability_window: int = 5           # Check stability over last N epochs
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
    
    def _validate_mode_objective_compatibility(self) -> None:
        """Validate that the objective is compatible with the selected mode"""
        if self.mode == OptimizationMode.SIMPLE:
            if OptimizationObjective.is_health_only(self.objective):
                universal_objectives = [obj.value for obj in OptimizationObjective.get_universal_objectives()]
                raise ValueError(
                    f"Health-only objective '{self.objective.value}' cannot be used in SIMPLE mode. "
                    f"Available objectives for SIMPLE mode: {universal_objectives}"
                )
        
        # HEALTH mode can use any objective, so no validation needed
        logger.debug(f"running _validate_mode_objective_compatibility ... Mode '{self.mode.value}' is compatible with objective '{self.objective.value}'")


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
    Unified optimizer class that coordinates hyperparameter optimization
    
    Integrates with existing ModelBuilder and DatasetManager to provide
    automated hyperparameter tuning with simple or health-aware optimization.
    
    Uses configurable health_weight to balance objective vs health importance.
    """
    
    def __init__(
        self, 
        dataset_name: str,
        optimization_config: Optional[OptimizationConfig] = None,
        datasets_root: Optional[str] = None,
        run_name: Optional[str] = None,
        health_analyzer: Optional[HealthAnalyzer] = None
    ):
        """
        Initialize ModelOptimizer
        
        Args:
            dataset_name: Name of dataset to optimize for
            optimization_config: Optimization settings (uses defaults if None)
            datasets_root: Optional custom datasets directory
            run_name: Optional unified run name for consistent directory/file naming
            health_analyzer: Optional HealthAnalyzer instance (creates new if None)
        """
        self.dataset_name = dataset_name
        self.config = optimization_config or OptimizationConfig()
        self.run_name = run_name
        self.summary_plots_dir: Optional[Path] = None
        
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
        
        # Detect data type for search space selection
        self.data_type = self._detect_data_type()
        logger.debug(f"running ModelOptimizer.__init__ ... Detected data type: {self.data_type}")
        
        # Initialize optimization state
        self.study: Optional[optuna.Study] = None
        self.optimization_start_time: Optional[float] = None
        self.results_dir: Optional[Path] = None
        
        # Create results directory
        self._setup_results_directory()
        
        logger.debug(f"running ModelOptimizer.__init__ ... Optimizer initialized for {dataset_name}")
        logger.debug(f"running ModelOptimizer.__init__ ... Mode: {self.config.mode.value}")
        logger.debug(f"running ModelOptimizer.__init__ ... Objective: {self.config.objective.value}")
        if self.config.mode == OptimizationMode.HEALTH and not OptimizationObjective.is_health_only(self.config.objective):
            logger.debug(f"running ModelOptimizer.__init__ ... Health weight: {self.config.health_weight} ({(1-self.config.health_weight)*100:.0f}% objective, {self.config.health_weight*100:.0f}% health)")
        logger.debug(f"running ModelOptimizer.__init__ ... Max trials: {self.config.n_trials}")
        logger.debug(f"running ModelOptimizer.__init__ ... Health monitoring: enabled (all modes)")
        if self.run_name:
            logger.debug(f"running ModelOptimizer.__init__ ... Run name: {self.run_name}")
    
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
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"running _setup_results_directory ... Results directory: {self.results_dir}")

        # Create a subdirectory within the plots directory for each trial in the run
        for trial_num in range(self.config.n_trials):
            trial_dir = plots_dir / f"trial_{trial_num + 1}"
            trial_dir.mkdir(exist_ok=True)
            logger.debug(f"running _setup_results_directory ... Created trial directory: {trial_dir}")
    
        # Create the summary plots directory
        self.summary_plots_dir = self.results_dir / "plots" / "summary_plots"
        self.summary_plots_dir.mkdir(exist_ok=True)
        logger.debug(f"running _setup_results_directory ... Created summary plots directory: {self.summary_plots_dir}")


    def _detect_training_instabilities(
        self, 
        history: Any, 
        trial: optuna.Trial,
        model: Any
    ) -> Tuple[bool, List[str]]:
        """
        Detect training instabilities using the configured stability parameters
        
        Args:
            history: Keras training history
            trial: Optuna trial object
            model: Trained model
            
        Returns:
            Tuple of (is_stable, list_of_issues)
        """
        logger.debug(f"running _detect_training_instabilities ... Checking stability for trial {trial.number}")
        
        issues = []
        is_stable = True
        
        if not self.config.enable_stability_checks:
            logger.debug(f"running _detect_training_instabilities ... Stability checks disabled, skipping")
            return True, []
        
        # Check if we have enough epochs for stability analysis
        epochs_completed = len(history.history.get('loss', []))
        if epochs_completed < self.config.stability_window:
            logger.debug(f"running _detect_training_instabilities ... Only {epochs_completed} epochs completed, need {self.config.stability_window} for stability check")
            return True, []  # Not enough data to assess stability
        
        # 1. Check for bias explosion by examining model weights
        try:
            for layer in model.layers:
                if hasattr(layer, 'bias') and layer.bias is not None:
                    bias_values = layer.bias.numpy()
                    max_bias = np.max(np.abs(bias_values))
                    mean_bias = np.mean(np.abs(bias_values))
                    
                    # Flag extreme bias values
                    if max_bias > 100.0:  # Absolute threshold for bias explosion
                        issues.append(f"BIAS EXPLOSION in {layer.name}: max_bias={max_bias:.2f}")
                        is_stable = False
                        logger.debug(f"running _detect_training_instabilities ... Detected bias explosion in {layer.name}")
                    
                    if mean_bias > 50.0:  # Mean bias threshold
                        issues.append(f"High mean bias in {layer.name}: mean_bias={mean_bias:.2f}")
                        is_stable = False
                        logger.debug(f"running _detect_training_instabilities ... Detected high mean bias in {layer.name}")
                        
        except Exception as e:
            logger.warning(f"running _detect_training_instabilities ... Failed to check bias values: {e}")
        
        # 2. Check for rapid changes in loss over the stability window
        if 'loss' in history.history:
            loss_values = history.history['loss']
            if len(loss_values) >= self.config.stability_window:
                recent_losses = loss_values[-self.config.stability_window:]
                
                # Calculate loss change rate
                for i in range(1, len(recent_losses)):
                    loss_change = abs(recent_losses[i] - recent_losses[i-1])
                    if loss_change > self.config.max_bias_change_per_epoch:
                        issues.append(f"Rapid loss change at epoch {epochs_completed - len(recent_losses) + i + 1}: change={loss_change:.3f}")
                        is_stable = False
                        logger.debug(f"running _detect_training_instabilities ... Detected rapid loss change: {loss_change:.3f}")
        
        # 3. Check for NaN or infinity in metrics
        for metric_name, values in history.history.items():
            if values:  # Check if list is not empty
                latest_value = values[-1]
                if np.isnan(latest_value) or np.isinf(latest_value):
                    issues.append(f"NaN/Inf detected in {metric_name}: {latest_value}")
                    is_stable = False
                    logger.debug(f"running _detect_training_instabilities ... Detected NaN/Inf in {metric_name}")
        
        # 4. Check for gradient explosion indicators (accuracy oscillation)
        if 'accuracy' in history.history:
            acc_values = history.history['accuracy']
            if len(acc_values) >= self.config.stability_window:
                recent_acc = acc_values[-self.config.stability_window:]
                
                # Calculate accuracy volatility
                acc_std = np.std(recent_acc)
                acc_mean = np.mean(recent_acc)
                
                # Flag high volatility in recent epochs
                if acc_std > 0.1 and acc_mean > 0.1:  # 10% standard deviation threshold
                    issues.append(f"High accuracy volatility: std={acc_std:.3f}, mean={acc_mean:.3f}")
                    is_stable = False
                    logger.debug(f"running _detect_training_instabilities ... Detected accuracy volatility: {acc_std:.3f}")
        
        logger.debug(f"running _detect_training_instabilities ... Stability check complete: stable={is_stable}, issues={len(issues)}")
        return is_stable, issues

    
    def _build_and_save_best_model(self, results: OptimizationResult) -> str:
        """
        Build and save the final model using the best hyperparameters found during optimization
        
        Args:
            results: OptimizationResult containing the best hyperparameters
            
        Returns:
            str: Path to the saved model file
        """
        logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Building final model with best params")
        
        try:            
            # Convert best_params to the format expected by create_and_train_model
            best_params = results.best_params.copy()
            
            logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Best hyperparameters: {best_params}")
            
            # Create ModelConfig and ModelBuilder directly so we can control the plot directory
            model_config = self._create_model_config(best_params)
            model_builder = ModelBuilder(self.dataset_config, model_config)
            
            # IMPORTANT: Set the plot directory to summary_plots BEFORE building/training
            # This ensures all realtime plots go to the correct location
            model_builder.plot_dir = self.summary_plots_dir
            logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Set plot directory to: {self.summary_plots_dir}")
            
            # Build and train the model (this is where realtime plots are generated)
            model_builder.build_model()
            training_history = model_builder.train(self.data)
            
            # CRITICAL FIX: Ensure training_history is properly stored
            # The train() method returns the history, but we need to make sure it's accessible
            if training_history is not None:
                model_builder.training_history = training_history
                logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Training history properly stored")
            else:
                logger.warning(f"running ModelOptimizer._build_and_save_best_model ... Training returned None history")
            
            # Double-check that training_history is available
            if model_builder.training_history is None:
                logger.warning(f"running ModelOptimizer._build_and_save_best_model ... ModelBuilder.training_history is None, evaluation plots may be limited")
            else:
                logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Training history available with keys: {list(model_builder.training_history.history.keys())}")
            
            # Evaluate the model 
            test_loss, test_accuracy = model_builder.evaluate(
                data=self.data,
                log_detailed_predictions=True,
                max_predictions_to_show=20,
                run_timestamp=datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
                plot_dir=self.summary_plots_dir
            )
            
            # Save the model
            final_model_path = model_builder.save_model(
                test_accuracy=test_accuracy,
                run_timestamp=datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
                run_name=self.run_name
            )
            
            final_accuracy = test_accuracy
            
            logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Final model built successfully")
            logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Final model accuracy: {final_accuracy:.4f}")
            logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Final model saved to: {final_model_path}")
            
            # Generate comprehensive plots for the best model and save to summary_plots directory
            try:
                if self.summary_plots_dir is None:
                    logger.warning("running ModelOptimizer._build_and_save_best_model ... Summary plots directory not available, skipping plot generation")
                else:
                    logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Generating summary plots for best model")
                    logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Using summary plots directory: {self.summary_plots_dir}")
                    
                    # Generate timestamp for the best model plots
                    best_model_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                    
                    # Note: Additional comprehensive plots were already generated during evaluate() call above
                    # which used self.summary_plots_dir as the plot_dir
                    
                    # Create a summary file for the best model performance
                    best_model_summary_file = self.summary_plots_dir / "best_model_summary.txt"
                    with open(best_model_summary_file, 'w') as f:
                        f.write(f"Best Model Summary\n")
                        f.write(f"==================\n\n")
                        f.write(f"Dataset: {self.dataset_name}\n")
                        f.write(f"Optimization Mode: {self.config.mode.value}\n")
                        f.write(f"Optimization Objective: {self.config.objective.value}\n")
                        f.write(f"Health Weight: {self.config.health_weight}\n\n")
                        f.write(f"Best Model Performance:\n")
                        f.write(f"- Test Accuracy: {final_accuracy:.4f}\n")
                        f.write(f"- Test Loss: {test_loss:.4f}\n")
                        f.write(f"- Model Path: {final_model_path}\n\n")
                        f.write(f"Best Hyperparameters:\n")
                        for param, value in sorted(results.best_params.items()):
                            f.write(f"- {param}: {value}\n")
                        f.write(f"\nOptimization Results:\n")
                        f.write(f"- Best Optimization Value: {results.best_value:.4f}\n")
                        f.write(f"- Total Trials: {results.total_trials}\n")
                        f.write(f"- Successful Trials: {results.successful_trials}\n")
                        f.write(f"- Optimization Time: {results.optimization_time_hours:.2f} hours\n")
                        
                        # Add health information if available
                        if results.best_trial_health:
                            f.write(f"\nBest Trial Health Metrics:\n")
                            for metric, value in sorted(results.best_trial_health.items()):
                                if isinstance(value, (int, float)):
                                    f.write(f"- {metric.replace('_', ' ').title()}: {value:.4f}\n")
                        
                        # Add training history summary if available
                        if model_builder.training_history is not None:
                            f.write(f"\nTraining History Summary:\n")
                            history_dict = model_builder.training_history.history
                            for metric, values in history_dict.items():
                                if values:
                                    f.write(f"- Final {metric}: {values[-1]:.4f}\n")
                        
                        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n")
                    
                    logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Best model summary saved to: {best_model_summary_file}")
                    logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Summary plots generated successfully")
                    logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Summary plots location: {self.summary_plots_dir}")
                    
            except Exception as plot_error:
                logger.warning(f"running ModelOptimizer._build_and_save_best_model ... Failed to generate summary plots: {plot_error}")
                logger.debug(f"running ModelOptimizer._build_and_save_best_model ... Plot error details: {traceback.format_exc()}")
                # Don't fail the entire process if plot generation fails
            
            return final_model_path
            
        except Exception as e:
            logger.error(f"running ModelOptimizer._build_and_save_best_model ... Failed to build final model: {e}")
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
        
        # NEW: Build and save the final best model if save_best_model is enabled
        if self.config.save_best_model and self.study.best_params:
            logger.debug(f"running ModelOptimizer.optimize ... Building final model with best hyperparameters")
            try:
                final_model_path = self._build_and_save_best_model(results)
                results.best_model_path = final_model_path
                logger.debug(f"running ModelOptimizer.optimize ... Final model saved to: {final_model_path}")
            except Exception as e:
                logger.error(f"running ModelOptimizer.optimize ... Failed to build final model: {e}")
                logger.debug(f"running ModelOptimizer.optimize ... Final model error traceback: {traceback.format_exc()}")
        
        logger.debug(f"running ModelOptimizer.optimize ... Optimization completed successfully")
        
        return results
    
    
    def _is_maximization_objective(self) -> bool:
        """Determine if objective should be maximized or minimized"""
        maximization_objectives = {
            OptimizationObjective.ACCURACY,
            OptimizationObjective.VAL_ACCURACY,
            OptimizationObjective.PARAMETER_EFFICIENCY,
            OptimizationObjective.MEMORY_EFFICIENCY,
            OptimizationObjective.INFERENCE_SPEED,
            OptimizationObjective.OVERALL_HEALTH,
            OptimizationObjective.NEURON_UTILIZATION,
            OptimizationObjective.TRAINING_STABILITY,
            OptimizationObjective.GRADIENT_HEALTH
        }
        return self.config.objective in maximization_objectives
    
    
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
            
            # Generate hyperparameters based on data type
            params = self._suggest_hyperparameters(trial)
            
            # Log trial start
            logger.debug(f"running _objective_function ... Trial {trial.number} config: "
                f"conv_layers={params.get('num_layers_conv')}, "
                f"kernel={params.get('kernel_size')}, "
                f"padding={params.get('padding')}, "
                f"filters={params.get('filters_per_conv_layer')}")
            
            # Create ModelConfig from suggested parameters
            model_config = self._create_model_config(params)
            
            # Record trial start time
            trial_start_time = time.time()
            
            # Create and train model
            model_builder = ModelBuilder(self.dataset_config, model_config)
            
            # Build and train model
            model_builder.build_model()
            
            # Check that model was built successfully
            if model_builder.model is None:
                raise RuntimeError("Model building failed - model is None")
            
            # Split training data for validation            
            x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
                self.data['x_train'], 
                self.data['y_train'], 
                test_size=self.config.validation_split, 
                random_state=42
            )
            
            # Prepare validation data for training
            validation_data = (x_val_split, y_val_split)
            
            # Create callbacks for early stopping and pruning
            callbacks = []
            
            if self.config.enable_early_stopping:
                
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_accuracy' if 'accuracy' in self.config.objective.value else 'val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1
                )
                callbacks.append(early_stopping)
            
            # Add Optuna pruning callback            
            pruning_callback = optuna_keras.KerasPruningCallback(
                trial, 
                'val_accuracy' if 'accuracy' in self.config.objective.value else 'val_loss'
            )
            callbacks.append(pruning_callback)
            
            # Safely get epochs value and ensure it's an integer
            trial_epochs = params.get('epochs', self.config.max_epochs_per_trial)
            if not isinstance(trial_epochs, int):
                logger.warning(f"running ModelOptimizer._objective_function ... epochs is not int: {trial_epochs} (type: {type(trial_epochs)}), converting")
                trial_epochs = int(trial_epochs)
            
            max_epochs = self.config.max_epochs_per_trial
            if not isinstance(max_epochs, int):
                logger.warning(f"running ModelOptimizer._objective_function ... max_epochs_per_trial is not int: {max_epochs} (type: {type(max_epochs)}), converting")
                max_epochs = int(max_epochs)
            
            final_epochs = min(trial_epochs, max_epochs)
            logger.debug(f"running ModelOptimizer._objective_function ... Using {final_epochs} epochs (trial={trial_epochs}, max={max_epochs})")
            
            # Train model with validation data
            history = model_builder.model.fit(
                x_train_split, 
                y_train_split,
                epochs=final_epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1  # Show training output
            )
            
            # Store the training history in the ModelBuilder instance
            # This ensures that when evaluate() is called later, it has access to the training history
            model_builder.training_history = history
            logger.debug(f"running ModelOptimizer._objective_function ... Stored training history in ModelBuilder")

            # Calculate training time
            training_time_minutes = (time.time() - trial_start_time) / 60

            # Save plots for this trial after successful calculation - PASS THE TRAINING HISTORY
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            self._save_trial_plots_after_training(trial, model_builder.model, run_timestamp, training_history=history)
            
            # Perform stability detection if enabled
            if self.config.enable_stability_checks:
                logger.debug(f"running ModelOptimizer._objective_function ... Performing stability checks for trial {trial.number}")
                
                # Check if we trained for minimum required epochs
                epochs_completed = len(history.history.get('loss', []))
                if epochs_completed < self.config.min_epochs_per_trial:
                    logger.debug(f"running ModelOptimizer._objective_function ... Trial {trial.number} insufficient epochs: {epochs_completed} < {self.config.min_epochs_per_trial}")
                    raise optuna.TrialPruned(f"Trial completed only {epochs_completed} epochs, minimum required: {self.config.min_epochs_per_trial}")
                
                # Detect training instabilities
                is_stable, stability_issues = self._detect_training_instabilities(
                    history=history,
                    trial=trial,
                    model=model_builder.model
                )
                
                if not is_stable:
                    logger.debug(f"running ModelOptimizer._objective_function ... Trial {trial.number} unstable training detected")
                    for issue in stability_issues:
                        logger.debug(f"running ModelOptimizer._objective_function ... - Stability issue: {issue}")
                    
                    # Prune trial due to instability
                    issue_summary = "; ".join(stability_issues[:3])  # Limit to first 3 issues for readability
                    raise optuna.TrialPruned(f"Training instability detected: {issue_summary}")
                else:
                    logger.debug(f"running ModelOptimizer._objective_function ... Trial {trial.number} passed stability checks")
            
            
            # Check resource constraints
            if training_time_minutes > self.config.max_training_time_minutes:
                logger.debug(f"running ModelOptimizer._objective_function ... Trial {trial.number} exceeded time limit")
                raise optuna.TrialPruned(f"Training time {training_time_minutes:.1f}min exceeded limit")
            
            # Count parameters
            total_params = model_builder.model.count_params()
            if total_params > self.config.max_parameters:
                logger.debug(f"running ModelOptimizer._objective_function ... Trial {trial.number} exceeded parameter limit")
                raise optuna.TrialPruned(f"Parameters {total_params:,} exceeded limit")
            
            # Calculate objective value based on optimization mode and target
            objective_value = self._calculate_objective_value(
                history=history,
                model=model_builder.model,
                validation_data=(x_val_split, y_val_split),
                training_time_minutes=training_time_minutes,
                total_params=total_params,
                trial=trial
            )
            
            # Check minimum accuracy threshold
            final_accuracy = history.history.get('val_accuracy', [0])[-1] if history.history.get('val_accuracy') else 0
            if final_accuracy < self.config.min_accuracy_threshold:
                logger.debug(f"running ModelOptimizer._objective_function ... Trial {trial.number} below accuracy threshold")
                raise optuna.TrialPruned(f"Accuracy {final_accuracy:.3f} below threshold")
            
            # Log trial success
            logger.debug(f"running ModelOptimizer._objective_function ... Trial {trial.number} completed: {self.config.objective.value}={objective_value:.4f}")
            logger.debug(f"running ModelOptimizer._objective_function ... Trial stats: accuracy={final_accuracy:.3f}, params={total_params:,}, time={training_time_minutes:.1f}min")
            
            return objective_value
            
        except optuna.TrialPruned:
            # Re-raise pruned trials
            raise
        except Exception as e:
            logger.warning(f"running ModelOptimizer._objective_function ... Trial {trial.number} failed: {e}")
            logger.debug(f"running ModelOptimizer._objective_function ... Error details: {traceback.format_exc()}")
            # Return worst possible value for failed trials
            return -1000.0 if self._is_maximization_objective() else 1000.0
    
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on data type (CNN vs LSTM)
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        if self.data_type == "text":
            return self._suggest_lstm_hyperparameters(trial)
        else:
            return self._suggest_cnn_hyperparameters(trial)
           
    def _calculate_objective_value(
        self,
        history: Any,
        model: Any,
        validation_data: Tuple[Any, Any],
        training_time_minutes: float,
        total_params: int,
        trial: optuna.Trial
    ) -> float:
        """
        Calculate objective value based on optimization mode and target
        
        Simplified approach:
        - SIMPLE mode: Pure objective only (health for monitoring)
        - HEALTH mode: Weighted combination using health_weight parameter
        
        Args:
            history: Keras training history
            model: Trained model
            validation_data: Validation data tuple (x, y)
            training_time_minutes: Training time in minutes
            total_params: Total model parameters
            trial: Optuna trial object
            
        Returns:
            Objective value
        """
        # ALWAYS calculate health metrics for monitoring
        health_metrics = self._calculate_trial_health_metrics(
            history, model, validation_data, training_time_minutes, total_params, trial
        )
        
        # Store health metrics for trial history
        trial_health_data = {
            'trial_number': trial.number,
            'health_metrics': health_metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.trial_health_history.append(trial_health_data)
        
        if self.config.mode == OptimizationMode.SIMPLE:
            # SIMPLE mode: Pure objectives only
            objective_value = self._calculate_pure_objective_value(
                history, model, validation_data, training_time_minutes, total_params, trial
            )
            
            logger.debug(f"running _calculate_objective_value ... Trial {trial.number} (SIMPLE mode):")
            logger.debug(f"running _calculate_objective_value ... - Pure objective: {objective_value:.4f}")
            logger.debug(f"running _calculate_objective_value ... - Health (monitoring): {health_metrics.get('overall_health', 0.0):.4f}")
            
            trial_health_data['base_objective'] = objective_value
            trial_health_data['final_objective'] = objective_value
                        
            return objective_value
            
        else:  # HEALTH mode
            if OptimizationObjective.is_health_only(self.config.objective):
                # Direct health optimization
                objective_value = self._calculate_health_objective_value(health_metrics, trial)
                
                logger.debug(f"running _calculate_objective_value ... Trial {trial.number} (HEALTH mode - direct health):")
                logger.debug(f"running _calculate_objective_value ... - Health objective: {objective_value:.4f}")
                
                trial_health_data['final_objective'] = objective_value
                
            else:
                # Universal objectives with health weighting
                base_objective = self._calculate_pure_objective_value(
                    history, model, validation_data, training_time_minutes, total_params, trial
                )
                overall_health = health_metrics.get('overall_health', 0.0)
                
                # Simple weighted combination
                objective_weight = 1.0 - self.config.health_weight
                objective_value = (objective_weight * base_objective) + (self.config.health_weight * overall_health)
                
                logger.debug(f"running _calculate_objective_value ... Trial {trial.number} (HEALTH mode - weighted):")
                logger.debug(f"running _calculate_objective_value ... - Base objective: {base_objective:.4f}")
                logger.debug(f"running _calculate_objective_value ... - Health score: {overall_health:.4f}")
                logger.debug(f"running _calculate_objective_value ... - Weights: {objective_weight:.1f} objective, {self.config.health_weight:.1f} health")
                logger.debug(f"running _calculate_objective_value ... - Final weighted: {objective_value:.4f}")
                
                trial_health_data['base_objective'] = base_objective
                trial_health_data['final_objective'] = objective_value
                
            return objective_value
    
    
    def _calculate_pure_objective_value(
        self,
        history: Any,
        model: Any,
        validation_data: Tuple[Any, Any],
        training_time_minutes: float,
        total_params: int,
        trial: optuna.Trial
    ) -> float:
        """
        Calculate pure objective value without any health considerations
        
        Args:
            history: Keras training history
            model: Trained model
            validation_data: Validation data tuple (x, y)
            training_time_minutes: Training time in minutes
            total_params: Total model parameters
            trial: Optuna trial object
            
        Returns:
            Pure objective value
        """
        if self.config.objective == OptimizationObjective.ACCURACY:
            # Use final training accuracy
            return history.history.get('accuracy', [0])[-1]
            
        elif self.config.objective == OptimizationObjective.VAL_ACCURACY:
            # Use final validation accuracy
            return history.history.get('val_accuracy', [0])[-1]
            
        elif self.config.objective == OptimizationObjective.TRAINING_TIME:
            # Minimize training time (return negative for maximization)
            return -training_time_minutes
            
        elif self.config.objective == OptimizationObjective.PARAMETER_EFFICIENCY:
            # Maximize accuracy per parameter (accuracy/log(params))
            accuracy = history.history.get('val_accuracy', [0])[-1]
            efficiency = accuracy / (np.log10(max(total_params, 1)))
            return efficiency
            
        elif self.config.objective == OptimizationObjective.MEMORY_EFFICIENCY:
            # Maximize accuracy per MB of memory (rough estimate)
            accuracy = history.history.get('val_accuracy', [0])[-1]
            memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
            return accuracy / max(memory_mb, 0.1)
            
        elif self.config.objective == OptimizationObjective.INFERENCE_SPEED:
            # Measure inference time and maximize accuracy/time
            x_val, y_val = validation_data
            
            # Time inference on small batch
            sample_size = min(32, len(x_val))
            sample_x = x_val[:sample_size]
            
            inference_times = []
            for _ in range(3):  # Average over 3 runs
                start_time = time.time()
                model.predict(sample_x, verbose=0)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            avg_inference_time = np.mean(inference_times)
            accuracy = history.history.get('val_accuracy', [0])[-1]
            
            # Return accuracy per second (higher is better)
            return accuracy / max(float(avg_inference_time), 0.001)
        
        else:
            raise ValueError(f"Unknown pure objective: {self.config.objective}")

    def _calculate_health_objective_value(
        self,
        health_metrics: Dict[str, Any],
        trial: optuna.Trial
    ) -> float:
        """
        Calculate objective value for health-only objectives
        
        Args:
            health_metrics: Health metrics dictionary
            trial: Optuna trial object
            
        Returns:
            Health-based objective value
        """
        if self.config.objective == OptimizationObjective.OVERALL_HEALTH:
            return health_metrics.get('overall_health', 0.0)
            
        elif self.config.objective == OptimizationObjective.NEURON_UTILIZATION:
            return health_metrics.get('neuron_utilization', 0.0)
            
        elif self.config.objective == OptimizationObjective.TRAINING_STABILITY:
            return health_metrics.get('training_stability', 0.0)
            
        elif self.config.objective == OptimizationObjective.GRADIENT_HEALTH:
            return health_metrics.get('gradient_health', 0.0)
        
        else:
            raise ValueError(f"Unknown health objective: {self.config.objective}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    def _suggest_cnn_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for CNN image classification architecture
        
        ARCHITECTURAL CONSTRAINT IMPROVEMENTS:
        1. Prevent spatial dimension over-reduction that causes negative dimension errors
        2. Apply intelligent constraints based on input image size and layer depth
        3. Ensure valid kernel/pool size combinations for deep networks
        """
        logger.debug("running _suggest_cnn_hyperparameters ... Suggesting CNN hyperparameters")
        
        # Get input image dimensions for constraint calculations
        input_height = self.dataset_config.img_height
        input_width = self.dataset_config.img_width
        logger.debug(f"running _suggest_cnn_hyperparameters ... Input dimensions: {input_height}x{input_width}")
        
        # Suggest number of conv layers first (this drives other constraints)
        num_layers_conv = trial.suggest_int('num_layers_conv', 1, 4)
        
        # IMPROVEMENT: Calculate safe pool sizes based on input dimensions and layer depth
        max_safe_pool_size = min(input_height, input_width) // (2 ** (num_layers_conv - 1))
        safe_pool_sizes = [2]  # Always include 2x2 as safe option
        if max_safe_pool_size >= 3:
            safe_pool_sizes.append(3)
        
        logger.debug(f"running _suggest_cnn_hyperparameters ... Network depth: {num_layers_conv} layers")
        logger.debug(f"running _suggest_cnn_hyperparameters ... Max safe pool size: {max_safe_pool_size}")
        logger.debug(f"running _suggest_cnn_hyperparameters ... Available pool sizes: {safe_pool_sizes}")
        
        # Suggest kernel and pool sizes with architectural constraints
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
        pool_size = trial.suggest_categorical('pool_size', safe_pool_sizes)
        padding = trial.suggest_categorical('padding', ['same', 'valid'])
        
        # Convert to tuples
        kernel_size = (kernel_size, kernel_size)
        pool_size = (pool_size, pool_size)
        
        # IMPROVEMENT: Apply intelligent architectural constraints
        constraints_applied = []
        
        # Constraint 1: Deep networks should use smaller kernels to preserve spatial information
        if num_layers_conv >= 3 and kernel_size == 5:
            kernel_size = (3, 3)
            constraints_applied.append(f"Deep network ({num_layers_conv} layers): forced 3x3 kernels")
        
        # Constraint 2: 'valid' padding with large kernels and many layers is dangerous
        if padding == 'valid' and kernel_size == 5 and num_layers_conv > 2:
            kernel_size = (3, 3)
            constraints_applied.append("Valid padding + large kernels + deep network: forced 3x3 kernels")
        
        # Constraint 3: 'valid' padding with many layers should use smaller pools
        if padding == 'valid' and num_layers_conv >= 3 and pool_size == 3:
            if 2 in safe_pool_sizes:
                pool_size = (2, 2)
                constraints_applied.append("Valid padding + deep network: forced 2x2 pooling")
        
        # Constraint 4: Very deep networks (4+ layers) should be conservative
        if num_layers_conv >= 4:
            kernel_size = (3, 3)
            pool_size = (2, 2)
            padding = 'same'
            constraints_applied.append("Very deep network (4+ layers): forced conservative settings")
        
        # Constraint 5: Small input images (≤16x16) should use minimal pooling
        if min(input_height, input_width) <= 16 and pool_size == 3:
            pool_size = (2, 2)
            constraints_applied.append("Small input image: forced 2x2 pooling")
        
        # Log applied constraints
        if constraints_applied:
            logger.debug(f"running _suggest_cnn_hyperparameters ... Applied architectural constraints:")
            for constraint in constraints_applied:
                logger.debug(f"running _suggest_cnn_hyperparameters ... - {constraint}")
        
        # IMPROVEMENT: Calculate expected spatial dimensions after all layers
        expected_height, expected_width = input_height, input_width
        for layer_idx in range(num_layers_conv):
            # Calculate conv output size
            if padding == 'same':
                # 'same' padding preserves spatial dimensions
                conv_height, conv_width = expected_height, expected_width
            else:  # 'valid' padding
                # 'valid' padding reduces dimensions by (kernel_size - 1)
                conv_height = expected_height - (kernel_size[0] - 1)
                conv_width = expected_width - (kernel_size[1] - 1)
            
            # Calculate pool output size
            pool_height = conv_height // pool_size[0]
            pool_width = conv_width // pool_size[1]
            
            logger.debug(f"running _suggest_cnn_hyperparameters ... Layer {layer_idx + 1}: {expected_height}x{expected_width} → conv({padding}) → {conv_height}x{conv_width} → pool({pool_size}) → {pool_height}x{pool_width}")
            
            expected_height, expected_width = pool_height, pool_width
            
            # SAFETY CHECK: Ensure we don't reduce to 0 or negative dimensions
            if expected_height <= 0 or expected_width <= 0:
                logger.warning(f"running _suggest_cnn_hyperparameters ... Architecture would create invalid dimensions at layer {layer_idx + 1}")
                logger.warning(f"running _suggest_cnn_hyperparameters ... Falling back to conservative settings")
                # Emergency fallback to safe settings
                kernel_size = (3, 3)
                pool_size = (2, 2)
                padding = 'same'
                num_layers_conv = min(num_layers_conv, 2)  # Reduce depth
                break
        
        logger.debug(f"running _suggest_cnn_hyperparameters ... Final spatial dimensions: {expected_height}x{expected_width}")
        
        # Call suggest_categorical ONCE and store the result for gradient clipping
        enable_gradient_clipping = trial.suggest_categorical('enable_gradient_clipping', [True, False])
        
        # Suggest other parameters normally
        filters_per_conv_layer = trial.suggest_categorical('filters_per_conv_layer', [16, 32, 64, 128, 256])
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'swish'])
        kernel_initializer = trial.suggest_categorical('kernel_initializer', ['he_normal', 'glorot_uniform'])
        batch_normalization = trial.suggest_categorical('batch_normalization', [True, False])
        use_global_pooling = trial.suggest_categorical('use_global_pooling', [True, False])
        
        # Hidden layer parameters
        num_layers_hidden = trial.suggest_int('num_layers_hidden', 1, 4)
        first_hidden_layer_nodes = trial.suggest_categorical('first_hidden_layer_nodes', [64, 128, 256, 512, 1024])
        subsequent_hidden_layer_nodes_decrease = trial.suggest_float('subsequent_hidden_layer_nodes_decrease', 0.25, 0.75)
        hidden_layer_activation_algo = trial.suggest_categorical('hidden_layer_activation_algo', ['relu', 'leaky_relu', 'sigmoid'])
        first_hidden_layer_dropout = trial.suggest_float('first_hidden_layer_dropout', 0.2, 0.7)
        subsequent_hidden_layer_dropout_decrease = trial.suggest_float('subsequent_hidden_layer_dropout_decrease', 0.1, 0.3)
        
        # Ensure epochs respects minimum requirement
        min_epochs = self.config.min_epochs_per_trial
        max_epochs = self.config.max_epochs_per_trial
        if min_epochs > max_epochs:
            logger.warning(f"running _suggest_cnn_hyperparameters ... min_epochs_per_trial ({self.config.min_epochs_per_trial}) > max_epochs_per_trial ({self.config.max_epochs_per_trial}), using min as both min and max")
            min_epochs = max_epochs = self.config.min_epochs_per_trial
        epochs = trial.suggest_int('epochs', min_epochs, max_epochs)
        logger.debug(f"running _suggest_cnn_hyperparameters ... Epochs range: {min_epochs}-{max_epochs}, suggested: {epochs}")

        # Remaining training parameters
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        gradient_clip_norm = trial.suggest_float('gradient_clip_norm', 0.5, 2.0) if enable_gradient_clipping else 1.0
        
        # Log final suggested parameters
        logger.debug(f"running _suggest_cnn_hyperparameters ... FINAL CNN configuration:")
        logger.debug(f"running _suggest_cnn_hyperparameters ... - conv_layers: {num_layers_conv}")
        logger.debug(f"running _suggest_cnn_hyperparameters ... - filters: {filters_per_conv_layer}")
        logger.debug(f"running _suggest_cnn_hyperparameters ... - kernel_size: {kernel_size} (suggested: {kernel_size})")
        logger.debug(f"running _suggest_cnn_hyperparameters ... - pool_size: {pool_size} (suggested: {pool_size})")
        logger.debug(f"running _suggest_cnn_hyperparameters ... - padding: {padding}")
        logger.debug(f"running _suggest_cnn_hyperparameters ... - expected_final_size: {expected_height}x{expected_width}")
        logger.debug(f"running _suggest_cnn_hyperparameters ... - constraints_applied: {len(constraints_applied)}")
        
        return {
            # Architecture selection
            'architecture_type': 'cnn',
            'use_global_pooling': use_global_pooling,
            
            # Convolutional layers (with applied constraints)
            'num_layers_conv': num_layers_conv,
            'filters_per_conv_layer': filters_per_conv_layer,
            'kernel_size': kernel_size,
            'pool_size': pool_size,
            'activation': activation,
            'kernel_initializer': kernel_initializer,
            'batch_normalization': batch_normalization,
            'padding': padding,
            
            # Hidden layers
            'num_layers_hidden': num_layers_hidden,
            'first_hidden_layer_nodes': first_hidden_layer_nodes,
            'subsequent_hidden_layer_nodes_decrease': subsequent_hidden_layer_nodes_decrease,
            'hidden_layer_activation_algo': hidden_layer_activation_algo,
            'first_hidden_layer_dropout': first_hidden_layer_dropout,
            'subsequent_hidden_layer_dropout_decrease': subsequent_hidden_layer_dropout_decrease,
            
            # Training parameters
            'epochs': epochs,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy'],
            
            # Regularization
            'enable_gradient_clipping': enable_gradient_clipping,
            'gradient_clip_norm': gradient_clip_norm,
                        
            # Disable real-time monitoring for optimization (performance)
            'enable_realtime_plots': False,
            'enable_gradient_flow_monitoring': False,
            'enable_realtime_weights_bias': False,
            'show_confusion_matrix': False,
            'show_training_history': False,
            'show_gradient_flow': False,
            'show_weights_bias_analysis': False,
            'show_activation_maps': False
        }
    
    
    def _suggest_lstm_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for LSTM text classification architecture"""
        logger.debug("running _suggest_lstm_hyperparameters ... Suggesting LSTM hyperparameters")
    
        # Text-specific parameters
        embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256, 512])
        lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128, 256])
        vocab_size = trial.suggest_categorical('vocab_size', [5000, 10000, 20000])
        use_bidirectional = trial.suggest_categorical('use_bidirectional', [True, False])
        text_dropout = trial.suggest_float('text_dropout', 0.2, 0.6)
        
        # Hidden layer parameters
        num_layers_hidden = trial.suggest_int('num_layers_hidden', 1, 3)
        first_hidden_layer_nodes = trial.suggest_categorical('first_hidden_layer_nodes', [64, 128, 256, 512])
        subsequent_hidden_layer_nodes_decrease = trial.suggest_float('subsequent_hidden_layer_nodes_decrease', 0.25, 0.75)
        hidden_layer_activation_algo = trial.suggest_categorical('hidden_layer_activation_algo', ['relu', 'leaky_relu', 'tanh'])
        first_hidden_layer_dropout = trial.suggest_float('first_hidden_layer_dropout', 0.2, 0.6)
        subsequent_hidden_layer_dropout_decrease = trial.suggest_float('subsequent_hidden_layer_dropout_decrease', 0.1, 0.3)
        
        # Training parameters with epoch constraint logic
        min_epochs = max(5, self.config.min_epochs_per_trial)  # Ensure at least 5, but respect config minimum
        max_epochs = min(25, self.config.max_epochs_per_trial)  # Respect max constraint (25 for text vs 30 for images)
        
        # Ensure we have a valid range
        if min_epochs > max_epochs:
            logger.warning(f"running _suggest_lstm_hyperparameters ... min_epochs_per_trial ({self.config.min_epochs_per_trial}) > max_epochs_per_trial ({self.config.max_epochs_per_trial}), using min as both min and max")
            min_epochs = max_epochs = self.config.min_epochs_per_trial
        
        epochs = trial.suggest_int('epochs', min_epochs, max_epochs)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        # Regularization parameters
        enable_gradient_clipping = trial.suggest_categorical('enable_gradient_clipping', [True, False])
        gradient_clip_norm = trial.suggest_float('gradient_clip_norm', 0.5, 2.0) if enable_gradient_clipping else 1.0
        
        # Fixed parameters
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        
        # Log suggested parameters
        logger.debug(f"running _suggest_lstm_hyperparameters ... Suggested LSTM params:")
        logger.debug(f"running _suggest_lstm_hyperparameters ... - embedding_dim: {embedding_dim}")
        logger.debug(f"running _suggest_lstm_hyperparameters ... - lstm_units: {lstm_units}")
        logger.debug(f"running _suggest_lstm_hyperparameters ... - bidirectional: {use_bidirectional}")
        logger.debug(f"running _suggest_lstm_hyperparameters ... - vocab_size: {vocab_size}")
        logger.debug(f"running _suggest_lstm_hyperparameters ... - text_dropout: {text_dropout}")
        logger.debug(f"running _suggest_lstm_hyperparameters ... - gradient_clipping: {enable_gradient_clipping}")
        logger.debug(f"running _suggest_lstm_hyperparameters ... - epochs: {epochs} (range: {min_epochs}-{max_epochs})")

        return {
            # Architecture selection
            'architecture_type': 'text',
            
            # Text-specific parameters
            'embedding_dim': embedding_dim,
            'lstm_units': lstm_units,
            'vocab_size': vocab_size,
            'use_bidirectional': use_bidirectional,
            'text_dropout': text_dropout,
            
            # Hidden layers
            'num_layers_hidden': num_layers_hidden,
            'first_hidden_layer_nodes': first_hidden_layer_nodes,
            'subsequent_hidden_layer_nodes_decrease': subsequent_hidden_layer_nodes_decrease,
            'hidden_layer_activation_algo': hidden_layer_activation_algo,
            'first_hidden_layer_dropout': first_hidden_layer_dropout,
            'subsequent_hidden_layer_dropout_decrease': subsequent_hidden_layer_dropout_decrease,
            
            # Training parameters
            'epochs': epochs,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'loss': loss,
            'metrics': metrics,
            
            # Regularization
            'enable_gradient_clipping': enable_gradient_clipping,
            'gradient_clip_norm': gradient_clip_norm,

            # Disable real-time monitoring for optimization (performance)
            'enable_realtime_plots': False,
            'enable_gradient_flow_monitoring': False,
            'enable_realtime_weights_bias': False,
            'show_confusion_matrix': False,
            'show_training_history': False,
            'show_gradient_flow': False,
            'show_weights_bias_analysis': False,
            'show_activation_maps': False
        }
    
    
    def _create_model_config(self, params: Dict[str, Any]) -> ModelConfig:
        """
        Create ModelConfig from suggested parameters
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            ModelConfig object with suggested parameters
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
        
        logger.debug(f"running _create_model_config ... ModelConfig created successfully")
        return config



    def _save_trial_plots_after_training(self, trial: optuna.Trial, model: Any, run_timestamp: str, training_history: Any = None) -> None:
        """
        Save plots for the current trial to its designated subdirectory
        
        Args:
            trial: Optuna trial object  
            model: The trained Keras model
            run_timestamp: Timestamp for consistent naming
            training_history: The training history from model.fit() (CRITICAL: pass this from caller)
        """
        from utils.logger import logger
        from pathlib import Path
        
        logger.debug(f"running _save_trial_plots_after_training ... Saving plots for trial {trial.number + 1}")
        
        if self.results_dir is None:
            logger.warning("running _save_trial_plots_after_training ... No results directory available, skipping plot save")
            return
        
        # Determine the trial-specific plot directory
        current_file = Path(__file__)
        project_root = current_file.parent.parent  # Go up 2 levels to project root
        trial_plot_dir = self.results_dir / "plots" / f"trial_{trial.number + 1}"
        
        # Ensure trial directory exists
        trial_plot_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"running _save_trial_plots_after_training ... Trial plot directory: {trial_plot_dir}")
        
        try:
            # Create a temporary ModelBuilder to use its evaluate method for plot generation
            # We need to recreate the model config from the trial parameters
            trial_params = trial.params.copy()
            model_config = self._create_model_config(trial_params)
            
            # Create ModelBuilder instance
            temp_model_builder = ModelBuilder(self.dataset_config, model_config)
            temp_model_builder.model = model  # Assign the trained model
            
            # CRITICAL FIX: Assign the training history if provided
            if training_history is not None:
                temp_model_builder.training_history = training_history
                logger.debug(f"running _save_trial_plots_after_training ... Training history assigned to temp ModelBuilder")
            else:
                logger.warning(f"running _save_trial_plots_after_training ... No training history provided, some plots may be unavailable")
                temp_model_builder.training_history = None
            
            # Use the existing ModelBuilder evaluate method to generate all plots
            # This will create: confusion matrix, training history, training animation, etc.
            test_loss, test_accuracy = temp_model_builder.evaluate(
                data=self.data,
                log_detailed_predictions=True,  # Enable detailed analysis
                max_predictions_to_show=10,     # Limit for trial plots
                run_timestamp=run_timestamp,
                plot_dir=trial_plot_dir         # This directs all plots to trial directory
            )
            
            logger.debug(f"running _save_trial_plots_after_training ... Trial {trial.number + 1} plots saved successfully")
            logger.debug(f"running _save_trial_plots_after_training ... Trial performance: accuracy={test_accuracy:.4f}, loss={test_loss:.4f}")
            
        except Exception as e:
            logger.warning(f"running _save_trial_plots_after_training ... Failed to save plots for trial {trial.number + 1}: {e}")
            logger.debug(f"running _save_trial_plots_after_training ... Error details: {traceback.format_exc()}")
    
    
    
    
    
    
    def _calculate_trial_health_metrics(
        self,
        history: Any,
        model: Any,
        validation_data: Tuple[Any, Any],
        training_time_minutes: float,
        total_params: int,
        trial: optuna.Trial
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive health metrics for monitoring and API reporting
        
        IMPROVEMENT: Enhanced error handling and model state validation
        
        Args:
            history: Keras training history
            model: Trained model
            validation_data: Validation data tuple (x, y)
            training_time_minutes: Training time in minutes
            total_params: Total model parameters
            trial: Optuna trial object
            
        Returns:
            Dictionary with health metrics
        """
        try:
            # Prepare sample data for health analysis
            x_val, y_val = validation_data
            sample_size = min(self.config.health_analysis_sample_size, len(x_val))
            sample_data = x_val[:sample_size] if len(x_val) >= sample_size else x_val
            
            # IMPROVEMENT: Validate model state before health analysis
            if model is None:
                logger.warning(f"running ModelOptimizer._calculate_trial_health_metrics ... Trial {trial.number}: Model is None")
                return self._get_default_health_metrics_with_error("Model is None")
            
            # IMPROVEMENT: Ensure model is callable and built
            try:
                # Test that model can process data
                test_input = sample_data[:1] if len(sample_data) > 0 else x_val[:1]
                test_output = model(test_input)
                logger.debug(f"running ModelOptimizer._calculate_trial_health_metrics ... Trial {trial.number}: Model validation successful, output shape: {test_output.shape}")
            except Exception as model_test_error:
                logger.warning(f"running ModelOptimizer._calculate_trial_health_metrics ... Trial {trial.number}: Model validation failed: {model_test_error}")
                return self._get_default_health_metrics_with_error(f"Model validation failed: {model_test_error}")
            
            # IMPROVEMENT: Validate training history
            if history is None or not hasattr(history, 'history'):
                logger.warning(f"running ModelOptimizer._calculate_trial_health_metrics ... Trial {trial.number}: Invalid training history")
                return self._get_default_health_metrics_with_error("Invalid training history")
            
            # Calculate comprehensive health metrics using shared HealthAnalyzer
            health_metrics = self.health_analyzer.calculate_comprehensive_health(
                model=model,
                history=history,
                sample_data=sample_data,
                training_time_minutes=training_time_minutes,
                total_params=total_params
            )
            
            # IMPROVEMENT: Validate health metrics result
            if not isinstance(health_metrics, dict) or 'overall_health' not in health_metrics:
                logger.warning(f"running ModelOptimizer._calculate_trial_health_metrics ... Trial {trial.number}: Invalid health metrics returned")
                return self._get_default_health_metrics_with_error("Invalid health metrics format")
            
            # Log health summary
            overall_health = health_metrics.get('overall_health', 0.0)
            logger.debug(f"running ModelOptimizer._calculate_trial_health_metrics ... Trial {trial.number} health: {overall_health:.3f}")
            
            # Track best trial health for reporting
            if (self.best_trial_health is None or 
                overall_health > self.best_trial_health.get('overall_health', 0.0)):
                self.best_trial_health = health_metrics.copy()
                self.best_trial_health['trial_number'] = trial.number
                logger.debug(f"running ModelOptimizer._calculate_trial_health_metrics ... Trial {trial.number}: New best health score: {overall_health:.3f}")
            
            return health_metrics
            
        except Exception as e:
            logger.warning(f"running ModelOptimizer._calculate_trial_health_metrics ... Health calculation failed for trial {trial.number}: {e}")
            logger.debug(f"running ModelOptimizer._calculate_trial_health_metrics ... Health calculation error traceback: {traceback.format_exc()}")
            # Return default health metrics on error
            return self._get_default_health_metrics_with_error(str(e))

    def _get_default_health_metrics_with_error(self, error_message: str) -> Dict[str, Any]:
        """
        IMPROVEMENT: Enhanced default health metrics with error context
        
        Args:
            error_message: Specific error message for debugging
            
        Returns:
            Dictionary with default health metrics and error information
        """
        return {
            'overall_health': 0.5,
            'neuron_utilization': 0.5,
            'parameter_efficiency': 0.5,
            'training_stability': 0.5,
            'gradient_health': 0.5,
            'convergence_quality': 0.5,
            'accuracy_consistency': 0.5,
            'health_breakdown': {
                'neuron_utilization': {'score': 0.5, 'weight': 0.25, 'status': 'error'},
                'parameter_efficiency': {'score': 0.5, 'weight': 0.15, 'status': 'error'},
                'training_stability': {'score': 0.5, 'weight': 0.20, 'status': 'error'},
                'gradient_health': {'score': 0.5, 'weight': 0.15, 'status': 'error'},
                'convergence_quality': {'score': 0.5, 'weight': 0.15, 'status': 'error'},
                'accuracy_consistency': {'score': 0.5, 'weight': 0.10, 'status': 'error'}
            },
            'recommendations': [f"Health analysis failed: {error_message}"],
            'analysis_mode': 'error',
            'error': error_message,
            'error_timestamp': datetime.now().isoformat()
        }
    
    def _compile_results(self) -> OptimizationResult:
        """Compile optimization results into structured format"""
        if self.study is None:
            raise RuntimeError("No study available - run optimize() first")
        
        optimization_time = time.time() - self.optimization_start_time if self.optimization_start_time else 0.0
        
        # Calculate parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
        except:
            importance = {}
        
        # Calculate average health metrics
        average_health = self._calculate_average_health_metrics()
        
        return OptimizationResult(
            best_value=self.study.best_value,
            best_params=self.study.best_params,
            total_trials=len(self.study.trials),
            successful_trials=len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            optimization_time_hours=optimization_time / 3600,
            optimization_mode=self.config.mode.value,
            health_weight=self.config.health_weight,
            objective_history=[t.value for t in self.study.trials if t.value is not None],
            parameter_importance=importance,
            health_history=self.trial_health_history,
            best_trial_health=self.best_trial_health,
            average_health_metrics=average_health,
            dataset_name=self.dataset_name,
            dataset_config=self.dataset_config,
            optimization_config=self.config,
            results_dir=self.results_dir
        )
    
    def _calculate_average_health_metrics(self) -> Optional[Dict[str, float]]:
        """Calculate average health metrics across all trials"""
        if not self.trial_health_history:
            return None
        
        # Collect all health metrics
        health_keys = set()
        for trial_data in self.trial_health_history:
            health_metrics = trial_data.get('health_metrics', {})
            if isinstance(health_metrics, dict):
                health_keys.update(health_metrics.keys())
        
        # Calculate averages
        averages = {}
        for key in health_keys:
            if key == 'error':  # Skip error entries
                continue
            values = []
            for trial_data in self.trial_health_history:
                health_metrics = trial_data.get('health_metrics', {})
                if key in health_metrics and isinstance(health_metrics[key], (int, float)):
                    values.append(health_metrics[key])
            
            if values:
                averages[key] = np.mean(values)
        
        return averages if averages else None
    
    def _save_results(self, results: OptimizationResult) -> None:
        """
        Save comprehensive optimization results to disk
        
        Enhanced to include health monitoring data for API/visualization.
        """       
        if self.results_dir is None:
            logger.warning("running ModelOptimizer._save_results ... No results directory available, skipping save")
            return
        
        logger.debug(f"running ModelOptimizer._save_results ... Saving optimization results to {self.results_dir}")
        
        try:
            # 1. Save complete optimization summary as JSON (machine-readable)
            summary_data = {
                "optimization_metadata": {
                    "dataset_name": results.dataset_name,
                    "optimization_mode": results.optimization_mode,
                    "optimization_objective": results.optimization_config.objective.value if results.optimization_config else "unknown",
                    "health_weight": results.health_weight,
                    "total_trials": results.total_trials,
                    "successful_trials": results.successful_trials,
                    "optimization_time_hours": results.optimization_time_hours,
                    "timestamp": datetime.now().isoformat()
                },
                "best_results": {
                    "best_value": results.best_value,
                    "best_params": results.best_params,
                    "best_trial_health": results.best_trial_health
                },
                "analysis": {
                    "parameter_importance": results.parameter_importance,
                    "objective_history": results.objective_history,
                    "average_health_metrics": results.average_health_metrics
                },
                "health_monitoring": {
                    "trial_health_history": results.health_history,
                    "health_analysis_enabled": True,
                    "health_weighting_applied": results.optimization_mode == "health" and results.health_weight > 0
                },
                "configuration": {
                    "mode": results.optimization_mode,
                    "health_weight": results.health_weight,
                    "n_trials": results.optimization_config.n_trials if results.optimization_config else None,
                    "max_epochs_per_trial": results.optimization_config.max_epochs_per_trial if results.optimization_config else None,
                    "max_training_time_minutes": results.optimization_config.max_training_time_minutes if results.optimization_config else None,
                    "optimization_objective": results.optimization_config.objective.value if results.optimization_config else None
                }
            }
            
            summary_file = self.results_dir / "optimization_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            logger.debug(f"running ModelOptimizer._save_results ... Saved optimization summary to {summary_file}")
            
            # 2. Save best hyperparameters as YAML (human-readable, copy-paste ready)
            yaml_file = self.results_dir / "best_hyperparameters.yaml"
            yaml_data = {
                "# Best hyperparameters found by optimization": None,
                "# Copy these values for reproducing the best model": None,
                "dataset": results.dataset_name,
                "optimization_mode": results.optimization_mode,
                "objective": results.optimization_config.objective.value if results.optimization_config else "unknown",
                "health_weight": results.health_weight,
                "best_value": float(results.best_value),
                "hyperparameters": results.best_params
            }
            
            if results.best_trial_health:
                yaml_data["best_trial_health_score"] = results.best_trial_health.get('overall_health', 0.0)
            
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            logger.debug(f"running ModelOptimizer._save_results ... Saved best hyperparameters to {yaml_file}")
            
            # 3. Save parameter importance (for analysis)
            if results.parameter_importance:
                importance_file = self.results_dir / "parameter_importance.json"
                with open(importance_file, 'w') as f:
                    json.dump(results.parameter_importance, f, indent=2)
                logger.debug(f"running ModelOptimizer._save_results ... Saved parameter importance to {importance_file}")
            
            # 4. Save health monitoring data
            if results.health_history:
                health_file = self.results_dir / "health_monitoring.json"
                health_data = {
                    "trial_health_history": results.health_history,
                    "best_trial_health": results.best_trial_health,
                    "average_health_metrics": results.average_health_metrics,
                    "optimization_mode": results.optimization_mode,
                    "health_weight": results.health_weight,
                    "health_weighting_applied": results.optimization_mode == "health" and results.health_weight > 0
                }
                with open(health_file, 'w') as f:
                    json.dump(health_data, f, indent=2)
                logger.debug(f"running ModelOptimizer._save_results ... Saved health monitoring data to {health_file}")
            
            # 5. Save trial history as CSV (for detailed analysis)
            if self.study is not None:
                csv_file = self.results_dir / "trial_history.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header (including health metrics)
                    header = ['trial_number', 'objective_value', 'state', 'duration_seconds', 'overall_health']
                    first_trial = None
                    if self.study.trials:
                        # Add parameter columns based on first completed trial
                        first_trial = next((t for t in self.study.trials if t.params), None)
                        if first_trial:
                            header.extend(sorted(first_trial.params.keys()))
                    writer.writerow(header)
                    
                    # Write trial data (including health data)
                    for trial in self.study.trials:
                        # Find corresponding health data
                        trial_health = None
                        for health_data in self.trial_health_history:
                            if health_data.get('trial_number') == trial.number:
                                trial_health = health_data.get('health_metrics', {})
                                break
                        
                        overall_health = trial_health.get('overall_health', 'N/A') if trial_health else 'N/A'
                        
                        row = [
                            trial.number,
                            trial.value if trial.value is not None else 'Failed',
                            trial.state.name,
                            (trial.duration.total_seconds() if trial.duration else 0),
                            overall_health
                        ]
                        # Add parameter values
                        if first_trial:
                            for param_name in sorted(first_trial.params.keys()):
                                row.append(trial.params.get(param_name, 'N/A'))
                        writer.writerow(row)
                
                logger.debug(f"running ModelOptimizer._save_results ... Saved trial history to {csv_file}")
            
            # 6. Create enhanced HTML report (visual summary with health data)
            html_file = self.results_dir / "optimization_report.html"
            html_content = self._generate_html_report(results)
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            # Update results object with report path
            results.optimization_report_path = str(html_file)
            logger.debug(f"running ModelOptimizer._save_results ... Saved HTML report to {html_file}")
            
            # 7. Save a quick README for users
            readme_file = self.results_dir / "README.md"
            readme_content = f"""# Optimization Results for {results.dataset_name}

## Quick Summary
- **Optimization Mode**: {results.optimization_mode}
- **Best {results.optimization_config.objective.value if results.optimization_config else 'unknown'}**: {results.best_value:.4f}
- **Health Weight**: {results.health_weight} ({(1-results.health_weight)*100:.0f}% objective, {results.health_weight*100:.0f}% health)
- **Successful Trials**: {results.successful_trials}/{results.total_trials}
- **Optimization Time**: {results.optimization_time_hours:.2f} hours

## Health Monitoring
- **Health Analysis**: Always enabled for monitoring and API reporting
- **Health Weighting**: {'Applied' if results.optimization_mode == 'health' and results.health_weight > 0 else 'Not applied'}
- **Best Trial Health Score**: {results.best_trial_health.get('overall_health', 'N/A') if results.best_trial_health else 'N/A'}

## Files in this directory:
- `optimization_summary.json`: Complete machine-readable results including health data
- `best_hyperparameters.yaml`: Copy-paste ready hyperparameters
- `optimization_report.html`: Visual summary (open in browser)
- `trial_history.csv`: Detailed trial-by-trial results with health metrics
- `parameter_importance.json`: Which hyperparameters matter most
- `health_monitoring.json`: Comprehensive health analysis data for API/visualization

## To reproduce the best model:
```python
from model_builder import create_and_train_model

# Load the best hyperparameters and train
result = create_and_train_model(
    dataset_name='{results.dataset_name}',
    # Copy parameters from best_hyperparameters.yaml
)
```

## Usage Examples
```bash
# Pure accuracy optimization (health monitoring only)
python optimizer.py dataset={results.dataset_name} mode=simple optimize_for=val_accuracy trials=20

# Health-weighted accuracy with default 30% health weight
python optimizer.py dataset={results.dataset_name} mode=health optimize_for=val_accuracy trials=20

# Balanced accuracy and health (50/50)
python optimizer.py dataset={results.dataset_name} mode=health optimize_for=val_accuracy health_weight=0.5 trials=20

# Strong health bias (20% objective, 80% health)
python optimizer.py dataset={results.dataset_name} mode=health optimize_for=val_accuracy health_weight=0.8 trials=20

# Direct health optimization
python optimizer.py dataset={results.dataset_name} mode=health optimize_for=overall_health trials=20
```
"""
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            logger.debug(f"running ModelOptimizer._save_results ... Saved README to {readme_file}")
            
            logger.debug(f"running ModelOptimizer._save_results ... Successfully saved all optimization results")
            
        except Exception as e:
            logger.error(f"running ModelOptimizer._save_results ... Failed to save optimization results: {e}")
            # Don't raise exception - optimization completed successfully, saving is just a bonus


    def _generate_html_report(self, results: OptimizationResult) -> str:
        """Generate enhanced HTML report with health monitoring data"""
        
        # Format parameter importance for display
        importance_html = ""
        if results.parameter_importance:
            importance_items = sorted(results.parameter_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for param, importance in importance_items:
                importance_html += f"<li><strong>{param}</strong>: {importance:.3f}</li>\n"
        else:
            importance_html = "<li>Parameter importance not calculated</li>"
        
        # Format best parameters for display
        params_html = ""
        for key, value in sorted(results.best_params.items()):
            params_html += f"<li><strong>{key}</strong>: {value}</li>\n"
        
        # Format health metrics for display
        health_html = ""
        if results.best_trial_health:
            health_metrics = results.best_trial_health
            health_html = f"""
            <p><strong>Overall Health Score:</strong> {health_metrics.get('overall_health', 'N/A'):.3f}</p>
            <ul>
                <li><strong>Neuron Utilization:</strong> {health_metrics.get('neuron_utilization', 'N/A'):.3f}</li>
                <li><strong>Parameter Efficiency:</strong> {health_metrics.get('parameter_efficiency', 'N/A'):.3f}</li>
                <li><strong>Training Stability:</strong> {health_metrics.get('training_stability', 'N/A'):.3f}</li>
                <li><strong>Gradient Health:</strong> {health_metrics.get('gradient_health', 'N/A'):.3f}</li>
                <li><strong>Convergence Quality:</strong> {health_metrics.get('convergence_quality', 'N/A'):.3f}</li>
                <li><strong>Accuracy Consistency:</strong> {health_metrics.get('accuracy_consistency', 'N/A'):.3f}</li>
            </ul>
            """
        else:
            health_html = "<p>Health metrics not available</p>"
        
        # Format average health metrics
        avg_health_html = ""
        if results.average_health_metrics:
            avg_health_html = "<h3>Average Health Metrics Across All Trials</h3><ul>"
            for metric, value in sorted(results.average_health_metrics.items()):
                avg_health_html += f"<li><strong>{metric.replace('_', ' ').title()}:</strong> {value:.3f}</li>\n"
            avg_health_html += "</ul>"
        
        # Format weighting information
        weighting_html = ""
        if results.optimization_mode == "health" and results.health_weight > 0:
            obj_weight = 1.0 - results.health_weight
            weighting_html = f"""
            <div class="weighting-info">
                <h3>Health Weighting Configuration</h3>
                <p><strong>Objective Weight:</strong> {obj_weight:.1%}</p>
                <p><strong>Health Weight:</strong> {results.health_weight:.1%}</p>
                <div class="weight-bar">
                    <div class="obj-portion" style="width: {obj_weight*100:.0f}%; background-color: #2196F3;">Objective ({obj_weight:.1%})</div>
                    <div class="health-portion" style="width: {results.health_weight*100:.0f}%; background-color: #4CAF50;">Health ({results.health_weight:.1%})</div>
                </div>
            </div>
            """
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Optimization Results - {results.dataset_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ font-size: 24px; color: #2e7d32; font-weight: bold; }}
        .mode-badge {{ 
            display: inline-block; 
            padding: 5px 10px; 
            border-radius: 15px; 
            color: white; 
            font-weight: bold;
            background-color: {'#2196F3' if results.optimization_mode == 'simple' else '#4CAF50'};
        }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ margin: 5px 0; padding: 5px; background-color: #f9f9f9; }}
        .health-section {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; }}
        .weighting-info {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; }}
        .weight-bar {{ 
            display: flex; 
            height: 30px; 
            border-radius: 15px; 
            overflow: hidden; 
            margin-top: 10px;
        }}
        .obj-portion, .health-portion {{ 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            color: white; 
            font-weight: bold;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Hyperparameter Optimization Results</h1>
        <h2>Dataset: {results.dataset_name}</h2>
        <span class="mode-badge">{results.optimization_mode.upper()} MODE</span>
        <p>Optimization completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h3>Performance Summary</h3>
        <p class="metric">Best {results.optimization_config.objective.value if results.optimization_config else 'unknown'}: {results.best_value:.4f}</p>
        <p>Successful trials: {results.successful_trials}/{results.total_trials}</p>
        <p>Optimization time: {results.optimization_time_hours:.2f} hours</p>
        <p><strong>Mode:</strong> {results.optimization_mode}</p>
    </div>
    
    {weighting_html}
    
    <div class="section health-section">
        <h3>Best Trial Health Analysis</h3>
        {health_html}
    </div>
    
    <div class="section">
        <h3>Best Hyperparameters</h3>
        <ul>
{params_html}
        </ul>
    </div>
    
    <div class="section">
        <h3>Parameter Importance (Top 10)</h3>
        <ul>
{importance_html}
        </ul>
    </div>
    
    <div class="section">
        {avg_health_html}
    </div>
    
    <div class="section">
        <h3>Files Generated</h3>
        <ul>
            <li><strong>optimization_summary.json</strong>: Complete machine-readable results with health data</li>
            <li><strong>best_hyperparameters.yaml</strong>: Copy-paste ready configuration</li>
            <li><strong>trial_history.csv</strong>: Trial-by-trial detailed results with health metrics</li>
            <li><strong>parameter_importance.json</strong>: Parameter importance analysis</li>
            <li><strong>health_monitoring.json</strong>: Comprehensive health analysis data</li>
        </ul>
    </div>
    
    <div class="section">
        <h3>Health Monitoring Information</h3>
        <p><strong>Health Analysis:</strong> Always enabled for all optimization modes</p>
        <p><strong>Health Weighting:</strong> {'Applied with weight ' + str(results.health_weight) if results.optimization_mode == 'health' and results.health_weight > 0 else 'Not applied (monitoring only)'}</p>
        <p><strong>API Integration:</strong> All health data available for real-time monitoring and comparison</p>
    </div>
</body>
</html>"""

    def get_health_history(self) -> List[Dict[str, Any]]:
        """
        Get complete health monitoring history for API/visualization
        
        Returns:
            List of health data for all trials
        """
        return self.trial_health_history.copy()
    
    def get_best_trial_health(self) -> Optional[Dict[str, Any]]:
        """
        Get health metrics for the best performing trial
        
        Returns:
            Best trial health metrics or None if not available
        """
        return self.best_trial_health.copy() if self.best_trial_health else None
    
    def get_average_health_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get average health metrics across all trials
        
        Returns:
            Average health metrics or None if not available
        """
        return self._calculate_average_health_metrics()


# Convenience function for command-line usage with mode selection
def optimize_model(
    dataset_name: str,
    mode: str = "simple",
    optimize_for: str = "val_accuracy",
    trials: int = 50,
    run_name: Optional[str] = None,
    **config_overrides
) -> OptimizationResult:
    """
    Convenience function for unified model optimization
    
    Args:
        dataset_name: Name of dataset to optimize
        mode: Optimization mode ("simple" or "health")
        optimize_for: Optimization objective
        trials: Number of trials to run
        run_name: Optional unified run name for consistent directory/file naming
        **config_overrides: Additional optimization config overrides
        
    Returns:
        OptimizationResult with best parameters and metrics
        
    Raises:
        ValueError: If mode-objective combination is invalid
    """
    # CREATE UNIFIED RUN NAME HERE if not provided (same logic as model_optimizer.py)
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
    
    # Create optimization config
    opt_config = OptimizationConfig(
        mode=opt_mode,
        objective=objective,
        n_trials=trials
    )
    
    # Apply any config overrides
    for key, value in config_overrides.items():
        if hasattr(opt_config, key):
            setattr(opt_config, key, value)
            logger.debug(f"running optimize_model ... Set {key} = {value}")
        else:
            logger.warning(f"running optimize_model ... Unknown config parameter: {key}")
    
    # Run optimization
    optimizer = ModelOptimizer(dataset_name, opt_config, run_name=run_name)
    return optimizer.optimize()


if __name__ == "__main__":
    # Simple command-line interface with mode selection
    '''
    Example usage:
        python src/optimizer.py dataset=mnist mode=health optimize_for=val_accuracy trials=10 max_epochs_per_trial=15 # Default 30% health weight
        python src/optimizer.py dataset=mnist mode=health optimize_for=val_accuracy health_weight=0.5 trials=10 max_epochs_per_trial=15 # Balanced 50/50
        python src/optimizer.py dataset=mnist mode=health optimize_for=overall_health trials=10 max_epochs_per_trial=15 # Direct health optimization
        python src/optimizer.py dataset=mnist mode=simple optimize_for=val_accuracy trials=10 max_epochs_per_trial=15 # Pure accuracy optimization
    '''
    
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    
    # Extract required arguments
    dataset_name = args.get('dataset', 'cifar10')
    mode = args.get('mode', 'simple')  # mode selection
    optimize_for = args.get('optimize_for', 'val_accuracy')
    trials = int(args.get('trials', '50'))
    run_name = args.get('run_name', None)
    
    # Convert integer parameters for OptimizationConfig
    int_params = [
        'n_trials', 'n_startup_trials', 'n_warmup_steps', 'random_seed',
        'max_epochs_per_trial', 'early_stopping_patience', 'min_epochs_per_trial',
        'stability_window', 'health_analysis_sample_size', 'health_monitoring_frequency'
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
        'max_bias_change_per_epoch', 'health_weight'  # Added health_weight
    ]
    for float_param in float_params:
        if float_param in args:
            try:
                args[float_param] = float(args[float_param])
                logger.debug(f"running optimizer.py ... Converted {float_param} to float: {args[float_param]}")
            except ValueError:
                logger.warning(f"running optimizer.py ... Invalid {float_param}: {args[float_param]}, using default")
                del args[float_param]
    
    # Convert boolean parameters for OptimizationConfig
    bool_params = [
        'save_best_model', 'save_optimization_history', 'create_comparison_plots',
        'enable_early_stopping', 'enable_stability_checks'
    ]
    for bool_param in bool_params:
        if bool_param in args:
            args[bool_param] = args[bool_param].lower() in ['true', '1', 'yes', 'on']
            logger.debug(f"running optimizer.py ... Converted {bool_param} to bool: {args[bool_param]}")
    
    logger.debug(f"running optimizer.py ... Starting optimization")
    logger.debug(f"running optimizer.py ... Dataset: {dataset_name}")
    logger.debug(f"running optimizer.py ... Mode: {mode}")
    logger.debug(f"running optimizer.py ... Objective: {optimize_for}")
    logger.debug(f"running optimizer.py ... Trials: {trials}")
    if run_name:
        logger.debug(f"running optimizer.py ... Run name: {run_name}")
    logger.debug(f"running optimizer.py ... Parsed arguments: {args}")
    
    try:
        # Run optimization
        result = optimize_model(
            dataset_name=dataset_name,
            mode=mode,
            optimize_for=optimize_for,
            trials=trials,
            run_name=run_name,
            **{k: v for k, v in args.items() if k not in ['dataset', 'mode', 'optimize_for', 'trials', 'run_name']}
        )
        
        # Print results
        print(result.summary())
        
        logger.debug(f"running optimizer.py ... ✅ Optimization completed successfully!")
        logger.debug(f"running optimizer.py ... Mode: {result.optimization_mode}")
        logger.debug(f"running optimizer.py ... Health monitoring: enabled")
        
        # Print mode-specific information
        if result.optimization_mode == "health":
            if OptimizationObjective.is_health_only(OptimizationObjective(optimize_for)):
                print(f"\n🏥 Health-Only Optimization:")
                print(f"Optimized directly for model health metric: {optimize_for}")
            else:
                obj_weight = 1.0 - result.health_weight
                print(f"\n⚖️ Health-Weighted Optimization:")
                print(f"Balanced {optimize_for} ({obj_weight:.1%}) and health ({result.health_weight:.1%})")
        else:
            print(f"\n📊 Simple Optimization:")
            print(f"Pure {optimize_for} optimization (health monitoring only)")
        
        if result.best_trial_health:
            health_score = result.best_trial_health.get('overall_health', 0.0)
            print(f"Best trial health score: {health_score:.3f}")
        
    except Exception as e:
        # Provide helpful error messages for common mistakes
        error_msg = str(e)
        if "health-only objective" in error_msg.lower() and "simple mode" in error_msg.lower():
            print(f"\n❌ Configuration Error:")
            print(f"Cannot use health objective '{optimize_for}' in simple mode.")
            print(f"\nTry one of these instead:")
            print(f"1. Use simple mode with universal objective:")
            print(f"   python optimizer.py dataset={dataset_name} mode=simple optimize_for=val_accuracy")
            print(f"2. Use health mode with your desired objective:")
            print(f"   python optimizer.py dataset={dataset_name} mode=health optimize_for={optimize_for}")
            print(f"\nAvailable objectives by mode:")
            universal_objs = [obj.value for obj in OptimizationObjective.get_universal_objectives()]
            health_objs = [obj.value for obj in OptimizationObjective.get_health_only_objectives()]
            print(f"Universal (both modes): {universal_objs}")
            print(f"Health-only (health mode): {health_objs}")
        else:
            print(f"\n❌ Error: {error_msg}")
        
        logger.error(f"running optimizer.py ... ❌ Optimization failed: {e}")
        sys.exit(1)