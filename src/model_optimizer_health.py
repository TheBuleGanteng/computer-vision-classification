"""
Model Optimizer for Multi-Modal Classification

Automated hyperparameter optimization system that integrates with ModelBuilder
and DatasetManager to find optimal configurations for any supported dataset.

Supports multiple optimization objectives:
- Accuracy maximization
- Training time minimization  
- Parameter efficiency optimization
- Multi-objective optimization

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


class OptimizationObjective(Enum):
    """
    Enumeration of supported optimization objectives for hyperparameter tuning
    
    Inheriting from Enum provides type safety, prevents invalid values, and enables
    IDE auto-completion for optimization objectives. Each enum member maps to a
    string value that identifies the optimization target.
    
    Available Objectives:
        ACCURACY: Maximize final training accuracy (may overfit)
        VAL_ACCURACY: Maximize validation accuracy (preferred for generalization)
        TRAINING_TIME: Minimize training time (faster models)
        PARAMETER_EFFICIENCY: Maximize accuracy per parameter (compact models)
        MEMORY_EFFICIENCY: Maximize accuracy per memory usage (memory-efficient models)
        INFERENCE_SPEED: Maximize accuracy per inference time (fast prediction models)
    
    Usage Examples:
        # Type-safe objective selection
        config = OptimizationConfig(objective=OptimizationObjective.VAL_ACCURACY)
        
        # String conversion for logging/display
        print(f"Optimizing for: {config.objective.value}")  # "val_accuracy"
        
        # Conditional logic based on objective
        if objective == OptimizationObjective.TRAINING_TIME:
            # Use time-based optimization strategy
    
    Benefits of Using Enum:
        - Prevents typos in objective names
        - IDE auto-completion shows all valid options
        - Type checking catches invalid assignments at development time
        - Centralized definition of all supported objectives
    """
    ACCURACY = "accuracy"
    VAL_ACCURACY = "val_accuracy"
    TRAINING_TIME = "training_time"
    PARAMETER_EFFICIENCY = "parameter_efficiency"
    MEMORY_EFFICIENCY = "memory_efficiency"
    INFERENCE_SPEED = "inference_speed"


@dataclass
class OptimizationConfig:
    """Configuration for optimization process"""
    
    # Optimization settings
    objective: OptimizationObjective = OptimizationObjective.ACCURACY
    n_trials: int = 50 # Total length of optimization is n epochs per model build * n trials
    timeout_hours: Optional[float] = None  # None = no timeout
    
    # Pruning and sampling
    '''
    Pruning allows early stopping of unpromising trials to save resources. If # epochs is 20 and we know the current tial is worse than 
    those that have already been run, we can stop it early, instead of wasting time on the remaining epochs in that bad trial.
    '''
    n_startup_trials: int = 10  # Trials before pruning starts. Done to allow initial exploration and develp baseline.
    n_warmup_steps: int = 5     # Steps before pruning evaluation. Prevents pruning too early in training.
    random_seed: int = 42
    
    # Resource constraints
    max_epochs_per_trial: int = 20
    max_training_time_minutes: float = 60.0
    max_parameters: int = 10_000_000
    min_accuracy_threshold: float = 0.5
    
    # Stability detection parameters
    min_epochs_per_trial: int = 15      # Force longer observation
    enable_stability_checks: bool = True # Monitor for instabilities
    stability_window: int = 5           # Check stability over last N epochs
    max_bias_change_per_epoch: float = 10.0  # Flag rapid bias changes
    
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
    
    # Performance analysis
    objective_history: List[float] = field(default_factory=list)
    parameter_importance: Dict[str, float] = field(default_factory=dict)
    
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
        return f"""
Optimization Summary for {self.dataset_name}:
===========================================
Best {objective_name}: {self.best_value:.4f}
Successful trials: {self.successful_trials}/{self.total_trials}
Optimization time: {self.optimization_time_hours:.2f} hours

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
    Main optimizer class that coordinates hyperparameter optimization
    
    Integrates with existing ModelBuilder and DatasetManager to provide
    automated hyperparameter tuning with multiple optimization objectives.
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
        
        # Initialize dataset manager and load dataset info
        self.health_analyzer = health_analyzer or HealthAnalyzer()
        
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
        logger.debug(f"running ModelOptimizer.__init__ ... Objective: {self.config.objective.value}")
        logger.debug(f"running ModelOptimizer.__init__ ... Max trials: {self.config.n_trials}")
    
    
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
        project_root = Path(__file__).resolve().parent.parent
        optimization_dir = project_root / "saved_models" / "optimization_results"
        
        if self.run_name:
            # Use the provided run_name directly
            self.results_dir = optimization_dir / self.run_name
            logger.debug(f"running _setup_results_directory ... Using provided run_name: {self.run_name}")
        else:
            # Fallback to old naming if no run_name provided
            timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            dataset_clean = self.dataset_name.replace(" ", "_").lower()
            fallback_name = f"{timestamp}_{dataset_clean}_fallback"
            self.results_dir = optimization_dir / fallback_name
            logger.debug(f"running _setup_results_directory ... No run_name provided, using fallback: {fallback_name}")
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"running _setup_results_directory ... Results directory: {self.results_dir}")

    
    
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

    
    def optimize(self) -> OptimizationResult:
        """
        Run hyperparameter optimization
        
        Returns:
            OptimizationResult with best parameters and performance metrics
        """
        logger.debug(f"running ModelOptimizer.optimize ... Starting optimization for {self.dataset_name}")
        logger.debug(f"running ModelOptimizer.optimize ... Objective: {self.config.objective.value}")
        logger.debug(f"running ModelOptimizer.optimize ... Trials: {self.config.n_trials}")
        
        # Record start time
        self.optimization_start_time = time.time()
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='maximize' if self._is_maximization_objective() else 'minimize',
            sampler=TPESampler(seed=self.config.random_seed),
            pruner=MedianPruner(
                n_startup_trials=self.config.n_startup_trials,
                n_warmup_steps=self.config.n_warmup_steps
            ),
            study_name=f"optimize_{self.dataset_name}_{self.config.objective.value}"
        )
        
        # Run optimization
        try:
            logger.debug(f"running ModelOptimizer.optimize ... Beginning {self.config.n_trials} trials...")
            
            self.study.optimize(
                self._objective_function,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_hours * 3600 if self.config.timeout_hours else None
            )
            
            logger.debug(f"running ModelOptimizer.optimize ... Optimization completed!")
            
        except KeyboardInterrupt:
            logger.debug(f"running ModelOptimizer.optimize ... Optimization interrupted by user")
        except Exception as e:
            logger.error(f"running ModelOptimizer.optimize ... Optimization failed: {e}")
            raise
        
        # Compile results
        results = self._compile_results()
        
        # Save optimization results
        self._save_results(results)
        
        logger.debug(f"running ModelOptimizer.optimize ... Best {self.config.objective.value}: {results.best_value:.4f}")
        return results
    
    def _is_maximization_objective(self) -> bool:
        """Determine if objective should be maximized or minimized"""
        maximization_objectives = {
            OptimizationObjective.ACCURACY,
            OptimizationObjective.VAL_ACCURACY,
            OptimizationObjective.PARAMETER_EFFICIENCY,
            OptimizationObjective.MEMORY_EFFICIENCY,
            OptimizationObjective.INFERENCE_SPEED
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
            
            # Store plot directory for this trial
            trial_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            
            '''
            if self.results_dir is None:
                self._setup_results_directory()
            trial_plot_dir = self.results_dir / f"trial_{trial.number:03d}_{trial_timestamp}" # type: ignore
            trial_plot_dir.mkdir(exist_ok=True)
            '''
            
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
                verbose=1  # Suppress training output
            )
            
            # Calculate training time
            training_time_minutes = (time.time() - trial_start_time) / 60
            
            
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
            
            # Calculate objective value based on optimization target
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
           
    
    def _suggest_cnn_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for CNN image classification architecture
        
        Uses Optuna's Bayesian optimization to intelligently suggest hyperparameters
        specifically tuned for image classification tasks (CIFAR-10, GTSRB, etc.).
        Optuna learns from previous trials to focus on promising parameter combinations.
        
        CNN Architecture Overview:
            Input (images) → Conv+Pool → Conv+Pool → Flatten/GlobalPool → Dense → Output
            
            Key Image-Specific Considerations:
            - Convolutional layers detect spatial features (edges, shapes, patterns)
            - Pooling reduces dimensions while preserving important features
            - Modern architectures use GlobalAveragePooling for parameter efficiency
            - Batch normalization stabilizes training of deeper networks
            - Kernel size affects receptive field and computational cost
            
        Hyperparameter Search Spaces:
            
            Architecture Strategy:
            - use_global_pooling: True/False (modern vs traditional parameter efficiency)
            - batch_normalization: True/False (training stability vs simplicity)
            
            Convolutional Feature Extraction:
            - num_layers_conv: 1-4 (depth vs training complexity)
            - filters_per_conv_layer: 16-256 (feature detection capacity)
            - kernel_size: 3x3/5x5 (local vs broader pattern detection)
            - pool_size: 2x2/3x3 (dimensionality reduction aggressiveness)
            - activation: relu/leaky_relu/swish (nonlinearity choice)
            - kernel_initializer: he_normal/glorot_uniform (weight initialization strategy)
            
            Classification Layers:
            - num_layers_hidden: 1-4 (decision-making complexity)
            - first_hidden_layer_nodes: 64-1024 (feature combination capacity)
            - subsequent_layer_decrease: 0.25-0.75 (funnel effect strength)
            - dropout rates: 0.2-0.7 (prevent overfitting to training images)
            
            Training Configuration:
            - epochs: 5-30 (images often need more training than text)
            - optimizer: adam/rmsprop/sgd (full optimizer search space)
            - learning_rate: 1e-4 to 1e-2 (logarithmic search)
            - gradient_clipping: True/False (less critical than RNNs but still useful)
        
        Optuna Intelligence Examples:
            Trial 1: filters=32, global_pooling=False → accuracy=0.78, params=1.2M
            Trial 2: filters=64, global_pooling=False → accuracy=0.82, params=4.8M (learn: more filters help)
            Trial 3: filters=64, global_pooling=True → accuracy=0.81, params=800K (learn: efficiency gain)
            Trial 4: filters=128, global_pooling=True → accuracy=0.85, params=1.5M (learn: sweet spot)
            → Optuna focuses future trials around filters=64-128, global_pooling=True
        
        Modern vs Traditional Architecture Comparison:
            Traditional: Conv→Pool→Conv→Pool→Flatten→Dense(1024)→Output
            - Parameter-heavy: Flatten creates massive parameter count
            - Memory intensive: Large dense layers
            
            Modern: Conv→Pool→Conv→Pool→GlobalAveragePooling→Output  
            - Parameter-efficient: Direct connection from features to output
            - Faster training: Fewer parameters to optimize
            - Better generalization: Less prone to overfitting
        
        Args:
            trial: Optuna trial object that provides intelligent parameter suggestion
                methods based on Bayesian optimization of previous trial results
        
        Returns:
            Dictionary containing all suggested hyperparameters for CNN model:
            {
                'architecture_type': 'cnn',
                'use_global_pooling': True,      # Optuna's intelligent suggestion
                'num_layers_conv': 3,            # Based on previous trial performance
                'filters_per_conv_layer': 128,   # Learned from trial history
                'kernel_size': (3, 3),           # Converted from '3x3' string
                'pool_size': (2, 2),             # Converted from '2x2' string
                'activation': 'relu',
                'kernel_initializer': 'he_normal',
                'batch_normalization': True,
                'num_layers_hidden': 2,
                'first_hidden_layer_nodes': 256,
                'epochs': 20,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'enable_gradient_clipping': False,
                'gradient_clip_norm': 1.0,
                # ... plus monitoring flags (disabled for optimization speed)
            }
        
        Performance Optimizations:
            - Real-time monitoring disabled to maximize trial throughput
            - String representations for tuples (kernel_size, pool_size) converted later
            - Wider search ranges than LSTM (CNNs more parameter-sensitive)
            - Includes SGD optimizer (can work well for image classification)
        
        Search Strategy:
            Optuna's TPE sampler identifies patterns like "larger filters + global pooling = 
            better accuracy/parameter ratio" and guides future suggestions toward these
            promising regions while maintaining exploration of untested combinations.
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
        # FIXED: Use different parameter names to avoid conflicts with ModelConfig attributes
        kernel_size_raw = trial.suggest_categorical('kernel_size_raw', [3, 5])
        pool_size_raw = trial.suggest_categorical('pool_size_raw', safe_pool_sizes)
        padding = trial.suggest_categorical('padding', ['same', 'valid'])
        
        # Convert to tuples
        kernel_size = (kernel_size_raw, kernel_size_raw)
        pool_size = (pool_size_raw, pool_size_raw)
        
        # IMPROVEMENT: Apply intelligent architectural constraints
        constraints_applied = []
        
        # Constraint 1: Deep networks should use smaller kernels to preserve spatial information
        if num_layers_conv >= 3 and kernel_size_raw == 5:
            kernel_size = (3, 3)
            constraints_applied.append(f"Deep network ({num_layers_conv} layers): forced 3x3 kernels")
        
        # Constraint 2: 'valid' padding with large kernels and many layers is dangerous
        if padding == 'valid' and kernel_size_raw == 5 and num_layers_conv > 2:
            kernel_size = (3, 3)
            constraints_applied.append("Valid padding + large kernels + deep network: forced 3x3 kernels")
        
        # Constraint 3: 'valid' padding with many layers should use smaller pools
        if padding == 'valid' and num_layers_conv >= 3 and pool_size_raw == 3:
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
        if min(input_height, input_width) <= 16 and pool_size_raw == 3:
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
        min_epochs = max(5, self.config.min_epochs_per_trial)
        max_epochs = min(30, self.config.max_epochs_per_trial)
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
        logger.debug(f"running _suggest_cnn_hyperparameters ... - kernel_size: {kernel_size} (suggested: {kernel_size_raw})")
        logger.debug(f"running _suggest_cnn_hyperparameters ... - pool_size: {pool_size} (suggested: {pool_size_raw})")
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
            'kernel_size': kernel_size,  # Final tuple version
            'pool_size': pool_size,      # Final tuple version
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
            
            # NOTE: kernel_size_raw and pool_size_raw are automatically tracked by Optuna
            # but not included in this return dictionary since ModelConfig doesn't need them
        }
    
    
    def _suggest_lstm_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for LSTM text classification architecture
        
        Uses Optuna's Bayesian optimization to intelligently suggest hyperparameters
        specifically tuned for text classification tasks (IMDB, Reuters, etc.).
        Optuna learns from previous trials to focus on promising parameter combinations.
        
        LSTM Architecture Overview:
            Input (word indices) → Embedding → LSTM/Bidirectional → Dense → Output
            
            Key Text-Specific Considerations:
            - Embedding converts sparse word indices to dense semantic vectors
            - LSTM processes sequences while maintaining context memory
            - Bidirectional processing reads text forward AND backward
            - Vocabulary size controls memory usage vs language coverage
            
        Hyperparameter Search Spaces:
            
            Text Processing:
            - embedding_dim: 64-512 (word vector dimensions, larger = more expressive)
            - lstm_units: 32-256 (memory cells, more = better context but slower)
            - vocab_size: 5K-20K (vocabulary coverage vs memory tradeoff)
            - use_bidirectional: True/False (forward+backward vs forward-only)
            - text_dropout: 0.2-0.6 (LSTM-specific regularization)
            
            Classification Layers:
            - num_layers_hidden: 1-3 (fewer than CNN due to LSTM complexity)
            - first_hidden_layer_nodes: 64-512 (feature combination capacity)
            - subsequent_layer_decrease: 0.25-0.75 (funnel effect strength)
            - dropout rates: 0.2-0.6 (prevent overfitting to text patterns)
            
            Training Configuration:
            - epochs: 5-25 (text often converges faster than images)
            - optimizer: adam/rmsprop (SGD excluded, poor for RNNs)
            - learning_rate: 1e-4 to 1e-2 (logarithmic search)
            - gradient_clipping: True/False (RNNs prone to exploding gradients)
        
        Optuna Intelligence Examples:
            Trial 1: embedding_dim=128, lstm_units=64 → accuracy=0.82
            Trial 2: embedding_dim=256, lstm_units=64 → accuracy=0.85 (learn: larger embedding helps)
            Trial 3: embedding_dim=256, lstm_units=128 → accuracy=0.87 (learn: more LSTM units helps)
            Trial 4: embedding_dim=512, lstm_units=128 → accuracy=0.86 (learn: diminishing returns)
            → Optuna focuses future trials around embedding_dim=256, lstm_units=128
        
        Args:
            trial: Optuna trial object that provides intelligent parameter suggestion
                methods based on Bayesian optimization of previous trial results
        
        Returns:
            Dictionary containing all suggested hyperparameters for LSTM model:
            {
                'architecture_type': 'text',
                'embedding_dim': 256,           # Optuna's intelligent suggestion
                'lstm_units': 128,              # Based on previous trial performance
                'use_bidirectional': True,      # Learned from trial history
                'vocab_size': 10000,
                'text_dropout': 0.4,
                'num_layers_hidden': 2,
                'first_hidden_layer_nodes': 256,
                'epochs': 15,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'enable_gradient_clipping': True,
                'gradient_clip_norm': 1.0,
                # ... plus monitoring flags (disabled for optimization speed)
            }
        
        Performance Optimizations:
            - Real-time monitoring disabled to maximize trial throughput
            - Gradient clipping intelligently suggested (RNNs need this)
            - Epoch range reduced vs CNN (text typically converges faster)
            - Bidirectional option balances performance vs training time
        
        Search Strategy:
            Optuna's TPE (Tree-structured Parzen Estimator) sampler builds probabilistic
            models of which parameter combinations work well, then suggests parameters
            with high probability of improvement. Early trials are more exploratory,
            later trials focus on promising regions of hyperparameter space.
        """
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
                # No longer need string-to-tuple conversion since we store tuples directly
                setattr(config, key, value)
                logger.debug(f"running _create_model_config ... Set {key} = {value} (type: {type(value)})")
            else:
                logger.warning(f"running _create_model_config ... ModelConfig has no attribute '{key}', skipping")
        
        logger.debug(f"running _create_model_config ... ModelConfig created successfully")
        return config   
    

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
        Calculate health-constrained objective value using shared HealthAnalyzer
        """
        # Get base accuracy (same as simple optimizer)
        if self.config.objective == OptimizationObjective.ACCURACY:
            base_accuracy = history.history.get('accuracy', [0])[-1]
        elif self.config.objective == OptimizationObjective.VAL_ACCURACY:
            base_accuracy = history.history.get('val_accuracy', [0])[-1]
        else:
            # For other objectives, fall back to simple optimizer behavior
            return self._calculate_simple_objective_value(
                history, model, validation_data, training_time_minutes, total_params, trial
            )
        
        # Use shared HealthAnalyzer for comprehensive health assessment
        x_val, y_val = validation_data
        sample_data = x_val[:100] if len(x_val) >= 100 else x_val  # Use up to 100 samples
        
        # Calculate comprehensive health metrics
        health_metrics = self.health_analyzer.calculate_comprehensive_health(
            model=model,
            history=history,
            sample_data=sample_data,
            training_time_minutes=training_time_minutes,
            total_params=total_params
        )
        
        # Extract key health indicators
        overall_health = health_metrics.get('overall_health', 0.5)
        dead_neuron_ratio = 1.0 - health_metrics.get('neuron_utilization', 0.5)
        
        # Apply health-based penalties using overall health score
        if overall_health < 0.3:  # Severely unhealthy
            penalty_factor = 0.2  # 80% penalty
            logger.debug(f"running _calculate_objective_value (health) ... Trial {trial.number}: SEVERE penalty - overall health: {overall_health:.3f}")
        elif overall_health < 0.5:  # Moderately unhealthy
            penalty_factor = 0.5  # 50% penalty
            logger.debug(f"running _calculate_objective_value (health) ... Trial {trial.number}: MODERATE penalty - overall health: {overall_health:.3f}")
        elif overall_health < 0.7:  # Slightly unhealthy
            penalty_factor = 0.8  # 20% penalty
            logger.debug(f"running _calculate_objective_value (health) ... Trial {trial.number}: LIGHT penalty - overall health: {overall_health:.3f}")
        else:  # Healthy model
            penalty_factor = 1.0  # No penalty
            logger.debug(f"running _calculate_objective_value (health) ... Trial {trial.number}: HEALTHY - overall health: {overall_health:.3f}")
        
        final_objective = base_accuracy * penalty_factor
        
        # Log comprehensive breakdown
        logger.debug(f"running _calculate_objective_value (health) ... Trial {trial.number} breakdown:")
        logger.debug(f"running _calculate_objective_value (health) ... - Base accuracy: {base_accuracy:.4f}")
        logger.debug(f"running _calculate_objective_value (health) ... - Overall health: {overall_health:.4f}")
        logger.debug(f"running _calculate_objective_value (health) ... - Dead neuron ratio: {dead_neuron_ratio:.4f}")
        logger.debug(f"running _calculate_objective_value (health) ... - Penalty factor: {penalty_factor:.4f}")
        logger.debug(f"running _calculate_objective_value (health) ... - Final objective: {final_objective:.4f}")
        
        # Log health recommendations for debugging
        recommendations = health_metrics.get('recommendations', [])
        if recommendations:
            logger.debug(f"running _calculate_objective_value (health) ... Health recommendations:")
            for rec in recommendations[:3]:  # Show first 3 recommendations
                logger.debug(f"running _calculate_objective_value (health) ... - {rec}")
        
        return final_objective


    def _calculate_simple_objective_value(
    self,
    history: Any,
    model: Any,
    validation_data: Tuple[Any, Any],
    training_time_minutes: float,
    total_params: int,
    trial: optuna.Trial
    ) -> float:
        """
        Fallback to simple optimizer behavior for non-accuracy objectives
        
        This ensures fair comparison for objectives like PARAMETER_EFFICIENCY
        """
        if self.config.objective == OptimizationObjective.TRAINING_TIME:
            return -training_time_minutes
            
        elif self.config.objective == OptimizationObjective.PARAMETER_EFFICIENCY:
            accuracy = history.history.get('val_accuracy', [0])[-1]
            efficiency = accuracy / (np.log10(max(total_params, 1)))
            return efficiency
            
        elif self.config.objective == OptimizationObjective.MEMORY_EFFICIENCY:
            accuracy = history.history.get('val_accuracy', [0])[-1]
            memory_mb = total_params * 4 / (1024 * 1024)
            return accuracy / max(memory_mb, 0.1)
            
        elif self.config.objective == OptimizationObjective.INFERENCE_SPEED:
            x_val, y_val = validation_data
            sample_size = min(32, len(x_val))
            sample_x = x_val[:sample_size]
            
            inference_times = []
            for _ in range(3):
                start_time = time.time()
                model.predict(sample_x, verbose=0)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            avg_inference_time = np.mean(inference_times)
            accuracy = history.history.get('val_accuracy', [0])[-1]
            return accuracy / max(float(avg_inference_time), 0.001)
        
        else:
            raise ValueError(f"Unknown optimization objective: {self.config.objective}")

    
    def _format_trial_params(self, params: Dict[str, Any]) -> str:
        """Format trial parameters for logging"""
        key_params = {}
        
        if self.data_type == "image":
            key_params = {
                'conv_layers': params.get('num_layers_conv'),
                'filters': params.get('filters_per_conv_layer'),
                'hidden_layers': params.get('num_layers_hidden'),
                'hidden_nodes': params.get('first_hidden_layer_nodes'),
                'global_pooling': params.get('use_global_pooling'),
                'epochs': params.get('epochs')
            }
        else:
            key_params = {
                'embedding_dim': params.get('embedding_dim'),
                'lstm_units': params.get('lstm_units'),
                'bidirectional': params.get('use_bidirectional'),
                'hidden_layers': params.get('num_layers_hidden'),
                'hidden_nodes': params.get('first_hidden_layer_nodes'),
                'epochs': params.get('epochs')
            }
        
        return ', '.join([f"{k}={v}" for k, v in key_params.items()])
     
    
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
        
        return OptimizationResult(
            best_value=self.study.best_value,
            best_params=self.study.best_params,
            total_trials=len(self.study.trials),
            successful_trials=len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            optimization_time_hours=optimization_time / 3600,
            objective_history=[t.value for t in self.study.trials if t.value is not None],
            parameter_importance=importance,
            dataset_name=self.dataset_name,
            dataset_config=self.dataset_config,
            optimization_config=self.config,
            results_dir=self.results_dir
        )
    
    def _save_results(self, results: OptimizationResult) -> None:
        """
        Save comprehensive optimization results to disk
        
        Creates multiple output formats for different use cases:
        - JSON file: Machine-readable optimization metadata and results
        - YAML file: Human-readable best hyperparameters for easy copying
        - HTML report: Visual summary with parameter importance and trial history
        - CSV file: Trial-by-trial results for further analysis
        
        Saves optimization metadata (hyperparameters, performance metrics, trial history)
        but NOT the actual trained models (those are saved separately if needed).
        
        File Structure Created:
            optimization_results/TIMESTAMP_optimize_DATASET_OBJECTIVE/
            ├── optimization_summary.json          # Complete results metadata
            ├── best_hyperparameters.yaml         # Copy-paste ready config
            ├── optimization_report.html          # Visual summary report  
            ├── trial_history.csv                 # All trial results
            └── parameter_importance.json         # Which params matter most
        
        Args:
            results: OptimizationResult containing all optimization metadata,
                    best parameters, trial history, and performance metrics
        
        Side Effects:
            - Creates multiple files in self.results_dir
            - Updates results.optimization_report_path with HTML report location
            - Logs file creation locations for user reference
            
        Error Handling:
            - Continues saving other files even if one format fails
            - Logs warnings for any save failures but doesn't raise exceptions
            - Ensures at least the JSON summary is saved for recovery
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
                    "optimization_objective": results.optimization_config.objective.value if results.optimization_config else "unknown",
                    "total_trials": results.total_trials,
                    "successful_trials": results.successful_trials,
                    "optimization_time_hours": results.optimization_time_hours,
                    "timestamp": datetime.now().isoformat()
                },
                "best_results": {
                    "best_value": results.best_value,
                    "best_params": results.best_params
                },
                "analysis": {
                    "parameter_importance": results.parameter_importance,
                    "objective_history": results.objective_history
                },
                "configuration": {
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
                "objective": results.optimization_config.objective.value if results.optimization_config else "unknown",
                "best_value": float(results.best_value),
                "hyperparameters": results.best_params
            }
            
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            logger.debug(f"running ModelOptimizer._save_results ... Saved best hyperparameters to {yaml_file}")
            
            # 3. Save parameter importance (for analysis)
            if results.parameter_importance:
                importance_file = self.results_dir / "parameter_importance.json"
                with open(importance_file, 'w') as f:
                    json.dump(results.parameter_importance, f, indent=2)
                logger.debug(f"running ModelOptimizer._save_results ... Saved parameter importance to {importance_file}")
            
            # 4. Save trial history as CSV (for detailed analysis)
            if self.study is not None:
                csv_file = self.results_dir / "trial_history.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    header = ['trial_number', 'objective_value', 'state', 'duration_seconds']
                    first_trial = None
                    if self.study.trials:
                        # Add parameter columns based on first completed trial
                        first_trial = next((t for t in self.study.trials if t.params), None)
                        if first_trial:
                            header.extend(sorted(first_trial.params.keys()))
                    writer.writerow(header)
                    
                    # Write trial data
                    for trial in self.study.trials:
                        row = [
                            trial.number,
                            trial.value if trial.value is not None else 'Failed',
                            trial.state.name,
                            (trial.duration.total_seconds() if trial.duration else 0)
                        ]
                        # Add parameter values
                        if first_trial:
                            for param_name in sorted(first_trial.params.keys()):
                                row.append(trial.params.get(param_name, 'N/A'))
                        writer.writerow(row)
                
                logger.debug(f"running ModelOptimizer._save_results ... Saved trial history to {csv_file}")
            
            # 5. Create simple HTML report (visual summary)
            html_file = self.results_dir / "optimization_report.html"
            html_content = self._generate_html_report(results)
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            # Update results object with report path
            results.optimization_report_path = str(html_file)
            logger.debug(f"running ModelOptimizer._save_results ... Saved HTML report to {html_file}")
            
            # 6. Save a quick README for users
            readme_file = self.results_dir / "README.md"
            readme_content = f"""# Optimization Results for {results.dataset_name}

    ## Quick Summary
    - **Best {results.optimization_config.objective.value if results.optimization_config else 'unknown'}**: {results.best_value:.4f}
    - **Successful Trials**: {results.successful_trials}/{results.total_trials}
    - **Optimization Time**: {results.optimization_time_hours:.2f} hours

    ## Files in this directory:
    - `optimization_summary.json`: Complete machine-readable results
    - `best_hyperparameters.yaml`: Copy-paste ready hyperparameters
    - `optimization_report.html`: Visual summary (open in browser)
    - `trial_history.csv`: Detailed trial-by-trial results
    - `parameter_importance.json`: Which hyperparameters matter most

    ## To reproduce the best model:
    ```python
    from model_builder import create_and_train_model

    # Load the best hyperparameters and train
    result = create_and_train_model(
        dataset_name='{results.dataset_name}',
        # Copy parameters from best_hyperparameters.yaml
    )
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
        """Generate simple HTML report for optimization results"""
        
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
        
        return f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Optimization Results - {results.dataset_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ font-size: 24px; color: #2e7d32; font-weight: bold; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin: 5px 0; padding: 5px; background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Hyperparameter Optimization Results</h1>
            <h2>Dataset: {results.dataset_name}</h2>
            <p>Optimization completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h3>Performance Summary</h3>
            <p class="metric">Best {results.optimization_config.objective.value if results.optimization_config else 'unknown'}: {results.best_value:.4f}</p>
            <p>Successful trials: {results.successful_trials}/{results.total_trials}</p>
            <p>Optimization time: {results.optimization_time_hours:.2f} hours</p>
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
            <h3>Files Generated</h3>
            <ul>
                <li><strong>optimization_summary.json</strong>: Complete machine-readable results</li>
                <li><strong>best_hyperparameters.yaml</strong>: Copy-paste ready configuration</li>
                <li><strong>trial_history.csv</strong>: Trial-by-trial detailed results</li>
                <li><strong>parameter_importance.json</strong>: Parameter importance analysis</li>
            </ul>
        </div>
    </body>
    </html>"""


# Convenience function for command-line usage
def optimize_model(
    dataset_name: str,
    optimize_for: str = "accuracy",
    trials: int = 50,
    run_name: Optional[str] = None,
    **config_overrides
) -> OptimizationResult:
    """
    Convenience function for model optimization
    
    Args:
        dataset_name: Name of dataset to optimize
        optimize_for: Optimization objective
        trials: Number of trials to run
        **config_overrides: Additional optimization config overrides
        
    Returns:
        OptimizationResult with best parameters and metrics
    """
    # Convert string objective to enum
    try:
        objective = OptimizationObjective(optimize_for.lower())
    except ValueError:
        available_objectives = [obj.value for obj in OptimizationObjective]
        raise ValueError(f"Unknown objective '{optimize_for}'. Available: {available_objectives}")
    
    # Create optimization config
    opt_config = OptimizationConfig(
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
    # Simple command-line interface
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    
    # Extract required arguments
    dataset_name = args.get('dataset', 'cifar10')
    optimize_for = args.get('optimize_for', 'accuracy')
    trials = int(args.get('trials', '50'))
    
    # Convert integer parameters for OptimizationConfig
    int_params = [
        'n_trials', 'n_startup_trials', 'n_warmup_steps', 'random_seed',
        'max_epochs_per_trial', 'early_stopping_patience'
    ]
    for int_param in int_params:
        if int_param in args:
            try:
                args[int_param] = int(args[int_param])
                logger.debug(f"running model_optimizer.py ... Converted {int_param} to int: {args[int_param]}")
            except ValueError:
                logger.warning(f"running model_optimizer.py ... Invalid {int_param}: {args[int_param]}, using default")
                del args[int_param]
    
    # Convert float parameters for OptimizationConfig
    float_params = [
        'timeout_hours', 'max_training_time_minutes', 'validation_split', 'test_size'
    ]
    for float_param in float_params:
        if float_param in args:
            try:
                args[float_param] = float(args[float_param])
                logger.debug(f"running model_optimizer.py ... Converted {float_param} to float: {args[float_param]}")
            except ValueError:
                logger.warning(f"running model_optimizer.py ... Invalid {float_param}: {args[float_param]}, using default")
                del args[float_param]
    
    # Convert boolean parameters for OptimizationConfig
    bool_params = [
        'save_best_model', 'save_optimization_history', 'create_comparison_plots',
        'enable_early_stopping'
    ]
    for bool_param in bool_params:
        if bool_param in args:
            args[bool_param] = args[bool_param].lower() in ['true', '1', 'yes', 'on']
            logger.debug(f"running model_optimizer.py ... Converted {bool_param} to bool: {args[bool_param]}")
    
    logger.debug(f"running model_optimizer.py ... Starting optimization")
    logger.debug(f"running model_optimizer.py ... Dataset: {dataset_name}")
    logger.debug(f"running model_optimizer.py ... Objective: {optimize_for}")
    logger.debug(f"running model_optimizer.py ... Trials: {trials}")
    logger.debug(f"running model_optimizer.py ... Parsed arguments: {args}")
    
    try:
        # Run optimization
        result = optimize_model(
            dataset_name=dataset_name,
            optimize_for=optimize_for,
            trials=trials,
            **{k: v for k, v in args.items() if k not in ['dataset', 'optimize_for', 'trials']}
        )
        
        # Print results
        print(result.summary())
        
        logger.debug(f"running model_optimizer.py ... ✅ Optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"running model_optimizer.py ... ❌ Optimization failed: {e}")
        sys.exit(1)