"""
Centralized Configuration Module

Contains all configuration classes and enums to eliminate duplication between
api_server.py and optimizer.py. This provides a single source of truth for
all configuration-related definitions.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os
from utils.logger import logger


def _get_default_runpod_endpoint() -> Optional[str]:
    """Generate default RunPod endpoint URL from environment variable"""
    # Ensure .env file is loaded
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available, rely on system environment
    
    endpoint_id = os.getenv('ENDPOINT_ID_RUNPOD')
    if endpoint_id:
        return f"https://api.runpod.ai/v2/{endpoint_id}/run"
    return None


class OptimizationMode(Enum):
    """Optimization modes available"""
    SIMPLE = "simple" 
    HEALTH = "health"


class OptimizationObjective(Enum):
    """Optimization objectives available"""
    VAL_ACCURACY = "val_accuracy"
    ACCURACY = "accuracy"
    TRAINING_TIME = "training_time"
    PARAMETER_EFFICIENCY = "parameter_efficiency"
    MEMORY_EFFICIENCY = "memory_efficiency"
    INFERENCE_SPEED = "inference_speed"
    OVERALL_HEALTH = "overall_health"
    NEURON_UTILIZATION = "neuron_utilization"
    TRAINING_STABILITY = "training_stability"
    GRADIENT_HEALTH = "gradient_health"

    @classmethod
    def get_universal_objectives(cls):
        """Get objectives that work in both simple and health modes"""
        return [cls.VAL_ACCURACY, cls.ACCURACY, cls.TRAINING_TIME,
                cls.PARAMETER_EFFICIENCY, cls.MEMORY_EFFICIENCY, cls.INFERENCE_SPEED]
    
    @classmethod
    def get_health_only_objectives(cls):
        """Get objectives that require health mode"""
        return [cls.OVERALL_HEALTH, cls.NEURON_UTILIZATION,
               cls.TRAINING_STABILITY, cls.GRADIENT_HEALTH]
    
    @classmethod
    def is_health_only(cls, objective):
        """Check if an objective requires health mode"""
        health_only = cls.get_health_only_objectives()
        return objective in health_only



class OptimizationConfig(BaseModel):
    """
    Unified configuration for the entire optimization system.
    
    Replaces both OptimizationRequest (API) and OptimizationConfig (business logic)
    to provide a single source of truth for all configuration settings.
    """
    
    
    # User-controlled variables with defaults
    dataset_name: str = Field("mnist", description="Dataset name (e.g., 'cifar10', `'mnist', 'imdb')")
    mode: str = Field("health", pattern="^(simple|health)$", description="Optimization mode")
    optimize_for: str = Field("val_accuracy", description="Optimization objective")
    trials: int = Field(2, ge=1, le=500, description="Number of optimization trials")
    max_epochs_per_trial: int = Field(7, ge=1, le=100, description="Maximum epochs per trial")
    min_epochs_per_trial: int = Field(5, ge=1, le=50, description="Minimum epochs per trial")
    health_weight: float = Field(0.3, ge=0.0, le=1.0, description="Health weighting (DEPRECATED - use accuracy_weight instead)")

    # AUTHORITATIVE SOURCE for default scoring weights
    # Frontend fetches these via /api/default-scoring-weights
    # Users can customize per-optimization-run via UI sliders
    accuracy_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Accuracy weight in final score (0.0-1.0). Auto-set based on mode if not provided.")
    health_overall_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Health overall weight in final score (auto-calculated: 1.0 - accuracy_weight)")
    health_component_proportions: Optional[Dict[str, float]] = Field(
        None,
        description="Health sub-component proportions (must sum to 1.0). Multiplied by health_overall_weight for absolute weights."
    )
    
    # Execution control
    use_runpod_service: bool = Field(True, description="Use RunPod cloud service for trials and final model building")
    
    # System configuration
    timeout_hours: Optional[float] = Field(5, description="Optimization timeout in hours")
    health_monitoring_frequency: int = Field(1, description="Health monitoring frequency")
    max_bias_change_per_epoch: float = Field(10.0, description="Maximum bias change per epoch")
    runpod_service_endpoint: Optional[str] = Field(default_factory=_get_default_runpod_endpoint, description="RunPod service endpoint URL (auto-configured from ENDPOINT_ID_RUNPOD)")
    runpod_service_timeout: int = Field(1800, description="RunPod service timeout in seconds")  # 30 minutes for health mode
    runpod_service_fallback_local: bool = Field(True, description="Fallback to local execution if RunPod fails")
    concurrent: bool = Field(False, description="Enable concurrent execution")
    
    # Training parameters
    batch_size: int = Field(32, description="Training batch size")
    learning_rate: float = Field(0.001, description="Learning rate")
    optimizer_name: str = Field("adam", description="Optimizer name")
    validation_split: float = Field(0.2, ge=0.0, le=1.0, description="Validation split ratio")
    test_size: float = Field(0.2, ge=0.0, le=1.0, description="Test size ratio")
    
    # Optimization parameters
    activation_functions: List[str] = Field(default_factory=lambda: ["relu", "tanh", "sigmoid"], description="Available activation functions")
    startup_trials: int = Field(10, description="Number of startup trials")
    warmup_steps: int = Field(5, description="Warmup steps")
    random_seed: int = Field(42, description="Random seed")
    gpu_proxy_sample_percentage: float = Field(0.5, description="GPU proxy sample percentage")
    
    # Resource constraints
    max_training_time_minutes: float = Field(60.0, description="Maximum training time per trial in minutes")
    max_parameters: int = Field(10_000_000, description="Maximum model parameters")
    min_accuracy_threshold: float = Field(0.5, description="Minimum accuracy threshold")
    
    # Stability parameters
    enable_stability_checks: bool = Field(True, description="Enable stability checks")
    stability_window: int = Field(3, description="Stability window size")
    health_analysis_sample_size: int = Field(50, description="Health analysis sample size")
    
    # Advanced options
    enable_early_stopping: bool = Field(True, description="Enable early stopping")
    early_stopping_patience: int = Field(5, description="Early stopping patience")
    
    # Output settings
    save_optimization_history: bool = Field(True, description="Save optimization history")
    create_comparison_plots: bool = Field(True, description="Create comparison plots")
    create_optuna_model_plots: bool = Field(True, description="Create Optuna model plots")
    create_final_model_plots: bool = Field(True, description="Create final model plots")
    
    # Individual plot flags - LOCAL MODEL REQUIRED
    show_activation_maps: bool = Field(True, description="Show activation maps")
    show_weights_bias: bool = Field(True, description="Show weights and bias")
    show_gradient_magnitudes: bool = Field(True, description="Show gradient magnitudes")
    show_gradient_distributions: bool = Field(True, description="Show gradient distributions")
    show_dead_neuron_analysis: bool = Field(True, description="Show dead neuron analysis")
    show_detailed_predictions: bool = Field(True, description="Show detailed predictions")
    show_activation_progression: bool = Field(True, description="Show activation progression")
    show_activation_comparison: bool = Field(True, description="Show activation comparison")
    show_activation_summary: bool = Field(True, description="Show activation summary")
    
    # Individual plot flags - RUNPOD COMPATIBLE
    show_training_history: bool = Field(True, description="Show training history")
    show_confusion_matrix: bool = Field(True, description="Show confusion matrix")
    show_training_animation: bool = Field(False, description="Show training animation")
    show_training_progress: bool = Field(True, description="Show training progress")
    
    # Concurrency settings
    concurrent_workers: int = Field(2, description="Number of concurrent workers")
    use_multi_gpu: bool = Field(True, description="Use multiple GPUs")
    target_gpus_per_worker: int = Field(2, description="Target GPUs per worker")
    auto_detect_gpus: bool = Field(True, description="Auto-detect GPUs")
    multi_gpu_batch_size_scaling: bool = Field(True, description="Scale batch size with multiple GPUs")
    max_gpus_per_worker: int = Field(4, description="Maximum GPUs per worker")
    
    # Legacy compatibility
    config_overrides: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration overrides")
    
    # Computed properties for backward compatibility
    @property
    def objective(self) -> OptimizationObjective:
        """Convert optimize_for string to objective enum"""
        objective_mapping = {
            "val_accuracy": OptimizationObjective.VAL_ACCURACY,
            "accuracy": OptimizationObjective.ACCURACY,
            "training_time": OptimizationObjective.TRAINING_TIME,
            "parameter_efficiency": OptimizationObjective.PARAMETER_EFFICIENCY,
            "memory_efficiency": OptimizationObjective.MEMORY_EFFICIENCY,
            "inference_speed": OptimizationObjective.INFERENCE_SPEED,
            "overall_health": OptimizationObjective.OVERALL_HEALTH,
            "neuron_utilization": OptimizationObjective.NEURON_UTILIZATION,
            "training_stability": OptimizationObjective.TRAINING_STABILITY,
            "gradient_health": OptimizationObjective.GRADIENT_HEALTH
        }
        return objective_mapping.get(self.optimize_for, OptimizationObjective.VAL_ACCURACY)
    
    @property
    def mode_enum(self) -> OptimizationMode:
        """Convert mode string to enum"""
        return OptimizationMode.SIMPLE if self.mode == "simple" else OptimizationMode.HEALTH
        
    # Backward compatibility aliases
    @property
    def n_trials(self) -> int:
        return self.trials
        
    @property 
    def n_startup_trials(self) -> int:
        return self.startup_trials
        
    @property
    def n_warmup_steps(self) -> int:
        return self.warmup_steps

    def model_post_init(self, __context) -> None:
        """Validation and system setup after model creation"""

        # Initialize scoring weights based on mode
        self._initialize_scoring_weights()

        # Validate mode-objective compatibility
        self._validate_mode_objective_compatibility()

        # Log RunPod endpoint configuration
        if self.use_runpod_service:
            if self.runpod_service_endpoint:
                logger.debug(f"running OptimizationConfig.model_post_init ... RunPod endpoint configured: {self.runpod_service_endpoint}")
            else:
                logger.warning(f"running OptimizationConfig.model_post_init ... RunPod service enabled but ENDPOINT_ID_RUNPOD not found in environment")
        
        # Log RunPod Service configuration
        if self.use_runpod_service:
            logger.debug(f"running OptimizationConfig.model_post_init ... RunPod service enabled in optimization config")
            logger.debug(f"running OptimizationConfig.model_post_init ... - Endpoint: {self.runpod_service_endpoint}")
            logger.debug(f"running OptimizationConfig.model_post_init ... - Timeout: {self.runpod_service_timeout}s")
            logger.debug(f"running OptimizationConfig.model_post_init ... - Fallback local: {self.runpod_service_fallback_local}")
            logger.debug(f"running OptimizationConfig.model_post_init ... concurrent is: {self.concurrent}")
            logger.debug(f"running OptimizationConfig.model_post_init ... - concurrent_workers is: {self.concurrent_workers}")
        else:
            logger.debug(f"running OptimizationConfig.model_post_init ... RunPod service disabled - using local execution only")
        
        # Enforce: local execution must not use concurrent workers
        if not self.use_runpod_service:
            if self.concurrent or self.concurrent_workers != 1:
                logger.debug("running OptimizationConfig.model_post_init ... local execution detected; "
                            "forcing concurrent=False and concurrent_workers=1")
            self.concurrent = False
            self.concurrent_workers = 1

    def _initialize_scoring_weights(self) -> None:
        """Initialize scoring weights based on optimization mode and user input"""

        # Default health component proportions (sum to 1.0)
        DEFAULT_HEALTH_PROPORTIONS = {
            'neuron_utilization': 0.25,
            'parameter_efficiency': 0.15,
            'training_stability': 0.20,
            'gradient_health': 0.15,
            'convergence_quality': 0.15,
            'accuracy_consistency': 0.10
        }

        # Set defaults based on mode if not provided by user
        if self.mode_enum == OptimizationMode.SIMPLE:
            # Simple mode: 100% accuracy, 0% health
            if self.accuracy_weight is None:
                self.accuracy_weight = 1.0
            self.health_overall_weight = 0.0
            # Health component proportions don't matter in simple mode
            if self.health_component_proportions is None:
                self.health_component_proportions = DEFAULT_HEALTH_PROPORTIONS.copy()
        else:
            # Health-aware mode: Default 70% accuracy, 30% health
            if self.accuracy_weight is None:
                self.accuracy_weight = 0.70

            # Health overall weight is always 1.0 - accuracy_weight
            self.health_overall_weight = 1.0 - self.accuracy_weight

            # Set default health component proportions if not provided
            if self.health_component_proportions is None:
                self.health_component_proportions = DEFAULT_HEALTH_PROPORTIONS.copy()

        # Validate weights
        self._validate_scoring_weights()

        logger.debug(f"running OptimizationConfig._initialize_scoring_weights ... "
                    f"Mode: {self.mode}, Accuracy weight: {self.accuracy_weight:.2f}, "
                    f"Health overall weight: {self.health_overall_weight:.2f}")

    def _validate_scoring_weights(self) -> None:
        """Validate that scoring weights are mathematically correct"""

        # Validate accuracy + health = 1.0 (allow small floating point errors)
        if self.accuracy_weight is not None and self.health_overall_weight is not None:
            total_weight = self.accuracy_weight + self.health_overall_weight
            if not (0.99 <= total_weight <= 1.01):
                raise ValueError(
                    f"Accuracy weight ({self.accuracy_weight}) + Health overall weight "
                    f"({self.health_overall_weight}) must sum to 1.0, got {total_weight:.4f}"
                )

        # Validate health component proportions sum to 1.0
        if self.health_component_proportions:
            component_sum = sum(self.health_component_proportions.values())
            if not (0.99 <= component_sum <= 1.01):
                raise ValueError(
                    f"Health component proportions must sum to 1.0, got {component_sum:.4f}. "
                    f"Proportions: {self.health_component_proportions}"
                )

        logger.debug(f"running OptimizationConfig._validate_scoring_weights ... Weight validation passed")

    def _validate_mode_objective_compatibility(self) -> None:
        """Validate that the objective is compatible with the selected mode"""
        if self.mode_enum == OptimizationMode.SIMPLE:
            if OptimizationObjective.is_health_only(self.objective):
                universal_objectives = [obj.value for obj in OptimizationObjective.get_universal_objectives()]
                raise ValueError(
                    f"Health-only objective '{self.objective.value}' cannot be used in SIMPLE mode. "
                    f"Available objectives for SIMPLE mode: {universal_objectives}"
                )
        
        logger.debug(f"running OptimizationConfig._validate_mode_objective_compatibility ... "
                    f"Mode '{self.mode}' is compatible with objective '{self.objective.value}'")


# For backward compatibility, create aliases
OptimizationRequest = OptimizationConfig