"""
Model Builder for Multi-Modal Classification

Creates and trains neural networks for both image classification (CNN) and 
text classification (LSTM). Automatically detects data type and builds 
appropriate architecture. Designed to work with any dataset configuration 
from DatasetManager.

REFACTORED: Plot generation logic moved to separate PlotGenerator module
for clean separation of concerns. This module now focuses purely on:
- Model building and compilation
- Training execution (local and GPU proxy)
- Basic evaluation (metrics only)
- Model saving

Key Optimizations:
- Enhanced GPU proxy integration with intelligent sampling
- Improved memory management and data type optimization
- Better error handling and fallback mechanisms
- Streamlined code structure with plot logic separated
- Enhanced logging and monitoring capabilities

Supported Architectures:
- CNN: For image data (GTSRB, CIFAR-10, MNIST, etc.)
- LSTM: For text data (IMDB, Reuters, etc.)
"""
import copy
from dataset_manager import DatasetConfig, DatasetManager
import datetime
import os
from plot_creation.realtime_gradient_flow import RealTimeGradientFlowCallback, RealTimeGradientFlowMonitor
from plot_creation.realtime_training_visualization import RealTimeTrainingVisualizer, RealTimeTrainingCallback
from plot_creation.realtime_weights_bias import create_realtime_weights_bias_monitor, RealTimeWeightsBiasMonitor, RealTimeWeightsBiasCallback
from plot_creation.training_history import TrainingHistoryAnalyzer
from plot_creation.weights_bias import WeightsBiasAnalyzer
from plot_creation.gradient_flow import GradientFlowAnalyzer
#from gpu_proxy_code import get_gpu_proxy_training_code

from dataclasses import dataclass, field
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
import subprocess
import sys
import tensorflow as tf
from tensorflow import keras # type: ignore
import traceback
from typing import Dict, Any, List, Tuple, Optional, Union, TYPE_CHECKING, Callable
from utils.logger import logger, PerformanceLogger, TimedOperation

if TYPE_CHECKING:
    from optimizer import OptimizationConfig

from dataclasses import dataclass, field
from enum import Enum

# Move PaddingOption outside the class to avoid issues
class PaddingOption(Enum):
    SAME = "same"
    VALID = "valid"

@dataclass
class ModelConfig:
    """
    Model architecture configuration with dual-purpose default system.
    
    This class serves two distinct roles:
    
    **1. Hyperparameter Optimization (Primary Use)**
    During optimization, ModelOptimizer creates an empty ModelConfig() and dynamically 
    populates it with Optuna-suggested parameters:
    
        model_config = ModelConfig()  # Uses defaults initially
        for key, value in optuna_params.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)  # Overrides defaults
    
    **2. Standalone/Testing/Development Usage**
    When used without optimization, the dataclass defaults provide sensible 
    fallback values that create working models:
    
        model_config = ModelConfig()  # Uses all defaults
        # Creates a basic CNN: 2 conv layers, 32 filters, 3x3 kernels, no global pooling
    
    **Default Override Pattern:**
    - num_layers_conv: int = 2              → Default used in testing/standalone
    - kernel_size: Tuple[int, int] = (3, 3) → Overridden by Optuna during optimization
    - use_global_pooling: bool = False      → Randomly True/False via Optuna trials
    
    **Key Design Principles:**
    - Defaults are conservative and stable (work for most architectures)
    - All defaults can be overridden by Optuna suggestions
    - Parameters not suggested by Optuna retain their default values
    - Supports both CNN and LSTM architectures with appropriate defaults
    
    This dual-purpose design enables the same class to serve both optimized 
    hyperparameter tuning and standalone model development workflows.
    """
    
    # Architecture selection
    architecture_type: str = "auto"  # "auto", "cnn", "text"
    use_global_pooling: bool = False # Use global average pooling instead of flattening
    
    # Convolutional layer parameters
    num_layers_conv: int = 2
    filters_per_conv_layer: int = 32
    kernel_size: Tuple[int, int] = (3, 3)
    activation: str = "relu"
    kernel_initializer: str = "he_normal"
    pool_size: Tuple[int, int] = (2, 2)
    batch_normalization: bool = False
    
    # Use the external PaddingOption and provide proper default
    padding: Union[PaddingOption, str] = PaddingOption.SAME
    
    # Text-specific parameters
    embedding_dim: int = 128
    lstm_units: int = 64
    vocab_size: int = 10000
    use_bidirectional: bool = True
    text_dropout: float = 0.5
    
    # Hidden layer parameters
    num_layers_hidden: int = 1
    first_hidden_layer_nodes: int = 128
    subsequent_hidden_layer_nodes_decrease: float = 0.50
    hidden_layer_activation_algo: str = "relu"
    first_hidden_layer_dropout: float = 0.5
    subsequent_hidden_layer_dropout_decrease: float = 0.20
    
    # Training parameters
    epochs: int = 10
    optimizer: str = "adam"
    learning_rate: float = 0.001
    loss: str = "categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["categorical_accuracy"])
    validation_split: float = 0.2
    batch_size: int = 32
    
    # Real-time visualization parameters (kept for training callbacks)
    enable_realtime_plots: bool = True
    save_realtime_plots: bool = True
    save_intermediate_plots: bool = True
    save_plot_every_n_epochs: int = 1
    
    # Activation map analysis parameters
    activation_layer_frequency: int = 1  # Analyze every layer by default
    activation_max_layers_to_analyze: int = 8  # Maximum number of layers to analyze
    activation_sample_selection_strategy: str = "representative"  # Sample selection strategy
    activation_max_total_samples: int = 50  # Maximum total samples to analyze
    activation_num_samples_per_class: int = 3  # Number of samples per class for representative sampling
    activation_max_filters_per_layer: int = 16  # Maximum number of filters to visualize per layer
    activation_filters_per_row: int = 4  # Number of filters per row in grid visualization
    activation_figsize_individual: Tuple[int, int] = (16, 12)  # Figure size for individual layer plots
    activation_cmap: str = "viridis"  # Colormap for activation visualizations
    activation_cmap_original: str = "gray"  # Colormap for original images
    activation_dead_filter_threshold: float = 0.1  # Threshold for detecting dead filters
    activation_saturated_filter_threshold: float = 0.8  # Threshold for detecting saturated filters
    show_activation_maps: bool = True  # Whether to generate activation map visualizations
    
    # Gradient flow analysis parameters (kept for training validation)
    gradient_flow_sample_size: int = 100
    enable_gradient_clipping: bool = False
    gradient_clip_norm: float = 1.0
    
    # Real-time gradient flow monitoring parameters
    enable_gradient_flow_monitoring: bool = True
    gradient_monitoring_frequency: int = 1
    gradient_history_length: int = 50
    gradient_sample_size: int = 32
    save_gradient_flow_plots: bool = True
    
    # Weights and bias analysis parameters
    enable_realtime_weights_bias: bool = True
    weights_bias_monitoring_frequency: int = 1
    weights_bias_sample_percentage: float = 0.1
    
    # Enhanced GPU Proxy Integration parameters
    use_gpu_proxy: bool = False
    gpu_proxy_auto_clone: bool = True
    gpu_proxy_endpoint: Optional[str] = None
    gpu_proxy_fallback_local: bool = True
    
    # Enhanced GPU proxy sampling parameters
    gpu_proxy_sample_percentage: float = 1.0  # Use 100% of training data by default
    gpu_proxy_use_stratified_sampling: bool = True      # Use stratified sampling
    gpu_proxy_adaptive_batch_size: bool = True          # Adapt batch size to sample count
    gpu_proxy_optimize_data_types: bool = True          # Optimize data types for transfer
    gpu_proxy_compression_level: int = 6                # Compression level for large payloads
    
    def __post_init__(self) -> None:
        if not self.metrics:
            self.metrics = ["categorical_accuracy"]
        
        # Handle padding conversion if it comes in as a string
        if isinstance(self.padding, str):
            try:
                self.padding = PaddingOption(self.padding)
            except ValueError:
                # If the string value doesn't match enum values, default to SAME
                from utils.logger import logger
                logger.warning(f"running ModelConfig.__post_init__ ... Invalid padding value '{self.padding}', defaulting to 'same'")
                self.padding = PaddingOption.SAME
    
    def get_padding_value(self) -> str:
        """
        Get the string value of padding, handling both enum and string cases
        """
        if isinstance(self.padding, PaddingOption):
            return self.padding.value
        elif isinstance(self.padding, str):
            return self.padding
        else:
            return "same"  # Default fallback

class ModelBuilder:
    """
    Main class for building and training neural network models
    
    NOW FOCUSES ON:
    - Model building and compilation
    - Training execution (local and GPU proxy)
    - Basic evaluation (metrics only)
    - Model saving
    
    REMOVED:
    - Plot generation logic (moved to PlotGenerator)
    - Analysis pipelines (moved to PlotGenerator)
    - Visualization coordination (moved to PlotGenerator)
    
    Key optimizations:
    - Enhanced GPU proxy integration with intelligent sampling
    - Improved memory management and error handling
    - Streamlined architecture detection and model building
    - Better logging and performance monitoring
    """
    
    def __init__(self, dataset_config: DatasetConfig, model_config: Optional[ModelConfig] = None, trial_number: Optional[int] = None, run_timestamp: Optional[str] = None, results_dir: Optional[Path] = None, optimization_config: Optional['OptimizationConfig'] = None) -> None:
        """
        Initialize ModelBuilder with enhanced configuration and GPU proxy setup
        
        Args:
            dataset_config: Configuration for dataset handling
            model_config: Configuration for model parameters
            trial_number: Optional trial number for TensorBoard logging
            run_timestamp: Optional timestamp for namespacing TensorBoard logs
            results_dir: Optional path to optimization results directory for TensorBoard logs
            optimization_config: Optional optimization configuration for plot generation flags
        """
        self.dataset_config: DatasetConfig = dataset_config
        self.model_config: ModelConfig = model_config or ModelConfig()
        self.trial_number: Optional[int] = trial_number
        self.run_timestamp: Optional[str] = run_timestamp
        self.results_dir: Optional[Path] = results_dir
        self.optimization_config: Optional['OptimizationConfig'] = optimization_config
        self.model: Optional[keras.Model] = None
        self.training_history: Optional[keras.callbacks.History] = None
        
        
        # Initialize performance logger
        self.perf_logger: PerformanceLogger = PerformanceLogger("model_builder")
        
        # Initialize HealthAnalyzer for evaluation consolidation
        from health_analyzer import HealthAnalyzer  # Import at method level to avoid circular imports
        self.health_analyzer = HealthAnalyzer()
        logger.debug(f"running ModelBuilder.__init__ ... HealthAnalyzer initialized for evaluation consolidation")        
        
        # Enhanced GPU Proxy Integration state
        self.gpu_proxy_available: bool = False
        self.gpu_proxy_path: Optional[str] = None
        self.runpod_client: Optional[Any] = None
        self._gpu_proxy_setup_attempted: bool = False
        
        # Detect and set up GPU proxy if requested
        if self.model_config.use_gpu_proxy:
            self._setup_gpu_proxy_with_retry()
        
        logger.debug(f"running ModelBuilder.__init__ ... Initialized for dataset: {dataset_config.name}")
        logger.debug(f"running ModelBuilder.__init__ ... Input shape: {dataset_config.input_shape}")
        logger.debug(f"running ModelBuilder.__init__ ... Number of classes: {dataset_config.num_classes}")
    
    
    def _setup_gpu_proxy_with_retry(self) -> None:
        """
        SIMPLIFIED: Setup GPU proxy with simple import-and-use pattern
        
        All infrastructure logic (detection, cloning, path management) is handled 
        by GPUProxyClient.auto_setup(). This method only does minimal path setup
        to make the import possible.
        """
        if self._gpu_proxy_setup_attempted:
            return
            
        self._gpu_proxy_setup_attempted = True
        
        try:
            logger.debug("running _setup_gpu_proxy_with_retry ... attempting GPU proxy auto-setup")
            
            # Minimal path setup just to make import possible
            current_dir = Path.cwd()
            project_root = Path(__file__).parent.parent  # Go up to project root
            logger.debug(f"running _setup_gpu_proxy_with_retry ... project_root is: {project_root}")
            gpu_proxy_found = False

            # Check multiple possible locations including sibling directories
            gpu_proxy_locations = [
                current_dir / "gpu-proxy",
                current_dir.parent / "gpu-proxy",
                project_root / "gpu-proxy", 
                project_root.parent / "gpu-proxy",  # Sibling to project root
            ]

            logger.debug(f"running _setup_gpu_proxy_with_retry ... Checking these locations:")
            for i, location in enumerate(gpu_proxy_locations):
                logger.debug(f"running _setup_gpu_proxy_with_retry ... [{i}] {location}")

            for gpu_proxy_path in gpu_proxy_locations:
                logger.debug(f"running _setup_gpu_proxy_with_retry ... Testing: {gpu_proxy_path}")
                logger.debug(f"running _setup_gpu_proxy_with_retry ... - exists(): {gpu_proxy_path.exists()}")
                logger.debug(f"running _setup_gpu_proxy_with_retry ... - is_dir(): {gpu_proxy_path.is_dir() if gpu_proxy_path.exists() else 'N/A'}")
                
                if gpu_proxy_path.exists() and gpu_proxy_path.is_dir():
                    client_file = gpu_proxy_path / "src" / "runpod" / "client.py"
                    logger.debug(f"running _setup_gpu_proxy_with_retry ... - client.py exists: {client_file.exists()}")
                    if client_file.exists():
                        logger.debug(f"running _setup_gpu_proxy_with_retry ... ✅ Found GPU proxy with client.py: {gpu_proxy_path}")
                        sys.path.insert(0, str(gpu_proxy_path))
                        gpu_proxy_found = True
                        break
                    else:
                        logger.debug(f"running _setup_gpu_proxy_with_retry ... Directory exists but missing client.py: {gpu_proxy_path}")
                else:
                    logger.debug(f"running _setup_gpu_proxy_with_retry ... Directory not found: {gpu_proxy_path}")
            
            # Import and let auto_setup handle all infrastructure
            try:
                logger.debug("running _setup_gpu_proxy_with_retry ... Attempting to import GPUProxyClient")
                from src.runpod.client import GPUProxyClient # type: ignore
                logger.debug("running _setup_gpu_proxy_with_retry ... GPUProxyClient import successful")
                
                logger.debug("running _setup_gpu_proxy_with_retry ... Calling GPUProxyClient.auto_setup")
                gpu_proxy_client = GPUProxyClient.auto_setup(
                    endpoint_id=self.model_config.gpu_proxy_endpoint,
                    auto_clone=self.model_config.gpu_proxy_auto_clone
                )
                logger.debug("running _setup_gpu_proxy_with_retry ... GPUProxyClient.auto_setup completed")
                
                # Extract for compatibility with existing code
                self.runpod_client = gpu_proxy_client.runpod_client
                self.gpu_proxy_available = True
                self.gpu_proxy_path = "managed_by_gpu_proxy"
                
                logger.debug("running _setup_gpu_proxy_with_retry ... GPU proxy integration enabled")
                
            except ImportError as import_error:
                logger.debug(f"running _setup_gpu_proxy_with_retry ... ImportError during GPU proxy import: {import_error}")
                logger.debug(f"running _setup_gpu_proxy_with_retry ... Current sys.path includes: {[p for p in sys.path if 'gpu' in p.lower()]}")
                raise
            
            except Exception as setup_error:
                logger.debug(f"running _setup_gpu_proxy_with_retry ... Exception during GPU proxy auto_setup: {setup_error}")
                logger.debug(f"running _setup_gpu_proxy_with_retry ... Exception type: {type(setup_error)}")
                raise
            
        except (ImportError, ModuleNotFoundError):
            logger.debug("running _setup_gpu_proxy_with_retry ... GPU proxy not available, using local execution")
            self.gpu_proxy_available = False
            self.runpod_client = None
            self.gpu_proxy_path = None
            
        except Exception as e:
            logger.warning(f"running _setup_gpu_proxy_with_retry ... GPU proxy setup failed: {e}")
            self.gpu_proxy_available = False
            self.runpod_client = None
            self.gpu_proxy_path = None
            
            if not self.model_config.gpu_proxy_fallback_local:
                raise RuntimeError(f"GPU proxy setup failed and local fallback disabled: {e}")
    
    def build_model(self) -> keras.Model:
        """
        Build the appropriate model with optimized architecture detection
        """
        logger.debug("running build_model ... Building model...")
        
        with TimedOperation("model building", "model_builder"):
            # Enhanced data type detection
            data_type = self._detect_data_type_enhanced()
            
            # Build appropriate model architecture
            if data_type == "text":
                logger.debug("running build_model ... Building TEXT model architecture")
                self.model = self._build_text_model_optimized()
            else:
                logger.debug("running build_model ... Building CNN model architecture")
                self.model = self._build_cnn_model_optimized()
            
            # Compile model with enhanced optimizer configuration
            self._compile_model_optimized()
            
            # Log model summary
            self._log_model_summary()
        
        return self.model
    
    def _detect_data_type_enhanced(self) -> str:
        """
        Enhanced data type detection with better heuristics
        """
        if self.model_config.architecture_type != "auto":
            return self.model_config.architecture_type
        
        # Enhanced detection logic
        height, width, channels = self.dataset_config.img_height, self.dataset_config.img_width, self.dataset_config.channels
        
        # Text indicators: flat sequence structure
        if height == 1 and channels == 1 and width > 50:
            logger.debug(f"running _detect_data_type_enhanced ... Detected TEXT data: sequence_length={width}")
            return "text"
        
        # Image indicators: spatial structure
        if height > 1 and width > 1:
            logger.debug(f"running _detect_data_type_enhanced ... Detected IMAGE data: shape=({height}, {width}, {channels})")
            return "image"
        
        # Fallback to image for ambiguous cases
        logger.debug(f"running _detect_data_type_enhanced ... Ambiguous data shape {(height, width, channels)}, defaulting to IMAGE")
        return "image"

    def _build_cnn_model_optimized(self) -> keras.Model:
        """
        Build optimized CNN model with proper LeakyReLU activation handling
        
        This fixes the TensorFlow Remapper Error by ensuring proper graph construction
        for LeakyReLU activations, preventing the BiasAdd node mismatch issue.
        """
        project_root = Path(__file__).parent.parent.parent  # Go up 3 levels to project root
        
        logger.debug("running _build_cnn_model_optimized ... Building CNN model with fixed activation handling")
        logger.debug(f"running _build_cnn_model_optimized ... ModelConfig activation: '{self.model_config.activation}'")
        logger.debug(f"running _build_cnn_model_optimized ... ModelConfig activation type: {type(self.model_config.activation)}")

        layers = []
        
        # Input layer
        layers.append(keras.layers.Input(shape=self.dataset_config.input_shape))
        logger.debug(f"running _build_cnn_model_optimized ... Input shape: {self.dataset_config.input_shape}")
        
        # Convolutional layers with proper activation handling
        for i in range(self.model_config.num_layers_conv):
            logger.debug(f"running _build_cnn_model_optimized ... Building conv layer {i+1}/{self.model_config.num_layers_conv}")
            
            # Separate activation handling to prevent graph construction issues
            if self.model_config.activation == 'leaky_relu':
                # For LeakyReLU: Create Conv2D without activation, then add separate LeakyReLU layer
                conv_layer = keras.layers.Conv2D(
                    filters=self.model_config.filters_per_conv_layer,
                    kernel_size=self.model_config.kernel_size,
                    activation=None,  # No activation in Conv2D layer
                    kernel_initializer=self.model_config.kernel_initializer,
                    padding=self.model_config.get_padding_value(),
                    use_bias=True,  # Ensure bias is used for proper graph construction
                    name=f'conv2d_{i}'  # Explicit naming for debugging
                )
                layers.append(conv_layer)
                
                # Add separate LeakyReLU activation layer with explicit naming
                leaky_relu_layer = keras.layers.LeakyReLU(
                    alpha=0.01, 
                    name=f'leaky_relu_{i}'
                )
                layers.append(leaky_relu_layer)
                
                logger.debug(f"running _build_cnn_model_optimized ... Added Conv2D + separate LeakyReLU layer {i}")
                
            else:
                # For other activations: Use standard approach
                conv_layer = keras.layers.Conv2D(
                    filters=self.model_config.filters_per_conv_layer,
                    kernel_size=self.model_config.kernel_size,
                    activation=self.model_config.activation,
                    kernel_initializer=self.model_config.kernel_initializer,
                    padding=self.model_config.get_padding_value(),
                    use_bias=True,
                    name=f'conv2d_{i}'
                )
                layers.append(conv_layer)
                
                logger.debug(f"running _build_cnn_model_optimized ... Added Conv2D with {self.model_config.activation} activation {i}")
            
            # Optional batch normalization (add after activation for proper normalization)
            if self.model_config.batch_normalization:
                batch_norm_layer = keras.layers.BatchNormalization(name=f'batch_norm_{i}')
                layers.append(batch_norm_layer)
                logger.debug(f"running _build_cnn_model_optimized ... Added BatchNormalization layer {i}")
            
            # Pooling layer
            pooling_layer = keras.layers.MaxPooling2D(
                pool_size=self.model_config.pool_size,
                name=f'max_pooling_{i}'
            )
            layers.append(pooling_layer)
            logger.debug(f"running _build_cnn_model_optimized ... Added MaxPooling2D layer {i}")
        
        # Pooling strategy
        if self.model_config.use_global_pooling:
            global_pool_layer = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')
            layers.append(global_pool_layer)
            logger.debug("running _build_cnn_model_optimized ... Using GlobalAveragePooling2D")
        else:
            flatten_layer = keras.layers.Flatten(name='flatten')
            layers.append(flatten_layer)
            logger.debug("running _build_cnn_model_optimized ... Using Flatten")
        
        # Hidden layers
        hidden_layers = self._build_hidden_layers_optimized()
        layers.extend(hidden_layers)
        logger.debug(f"running _build_cnn_model_optimized ... Added {len(hidden_layers)} hidden layers")
        
        # Output layer
        output_layer = keras.layers.Dense(
            self.dataset_config.num_classes,
            activation="softmax",
            name="output"
        )
        layers.append(output_layer)
        logger.debug(f"running _build_cnn_model_optimized ... Added output layer with {self.dataset_config.num_classes} classes")
        
        # Build final model
        model = keras.models.Sequential(layers, name="cnn_model_optimized")
        
        logger.debug(f"running _build_cnn_model_optimized ... CNN model built successfully with {len(layers)} layers")
        logger.debug(f"running _build_cnn_model_optimized ... Total parameters: {model.count_params():,}")
        
        return model
    
    def _build_text_model_optimized(self) -> keras.Model:
        """
        Build optimized text model with enhanced LSTM configuration
        """
        sequence_length = self.dataset_config.img_width
        
        layers = [
            keras.layers.Input(shape=(sequence_length,)),
            
            # Enhanced embedding layer
            keras.layers.Embedding(
                input_dim=self.model_config.vocab_size,
                output_dim=self.model_config.embedding_dim,
                input_length=sequence_length,
                mask_zero=True
            ),
            
            # Optimized LSTM layer
            self._create_lstm_layer_optimized(),
            
            # Dense layer
            keras.layers.Dense(
                self.model_config.first_hidden_layer_nodes,
                activation=self.model_config.hidden_layer_activation_algo
            ),
            
            # Dropout
            keras.layers.Dropout(self.model_config.first_hidden_layer_dropout),
            
            # Output layer
            keras.layers.Dense(
                self.dataset_config.num_classes,
                activation="softmax" if self.dataset_config.num_classes > 1 else "sigmoid"
            )
        ]
        
        logger.debug(f"running _build_text_model_optimized ... Text model: seq_len={sequence_length}, "
                    f"vocab={self.model_config.vocab_size}, embed_dim={self.model_config.embedding_dim}")
        
        return keras.models.Sequential(layers)
    
    def _create_lstm_layer_optimized(self) -> keras.layers.Layer:
        """Create optimized LSTM layer with bidirectional option"""
        lstm_layer = keras.layers.LSTM(
            units=self.model_config.lstm_units,
            dropout=self.model_config.text_dropout,
            recurrent_dropout=self.model_config.text_dropout / 2,
            return_sequences=False
        )
        
        if self.model_config.use_bidirectional:
            return keras.layers.Bidirectional(lstm_layer)
        
        return lstm_layer
    
    def _build_hidden_layers_optimized(self) -> List[keras.layers.Layer]:
        """
        Build optimized hidden layers with dynamic sizing
        """
        layers = []
        current_nodes = float(self.model_config.first_hidden_layer_nodes)
        current_dropout = self.model_config.first_hidden_layer_dropout
        
        for layer_num in range(self.model_config.num_layers_hidden):
            # Dense layer
            layers.append(keras.layers.Dense(
                int(current_nodes),
                activation=self.model_config.hidden_layer_activation_algo
            ))
            
            # Dropout layer
            layers.append(keras.layers.Dropout(current_dropout))
            
            logger.debug(f"running _build_hidden_layers_optimized ... Layer {layer_num + 1}: "
                        f"{int(current_nodes)} nodes, {current_dropout:.2f} dropout")
            
            # Update for next layer
            current_nodes = max(8.0, current_nodes * self.model_config.subsequent_hidden_layer_nodes_decrease)
            current_dropout = max(0.1, current_dropout - self.model_config.subsequent_hidden_layer_dropout_decrease)
        
        return layers
    
    def _compile_model_optimized(self) -> None:
        """
        Compile model with optimized settings and gradient clipping
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Create optimizer with optional gradient clipping
        if self.model_config.enable_gradient_clipping:
            logger.debug(f"running _compile_model_optimized ... Enabling gradient clipping: {self.model_config.gradient_clip_norm}")
            optimizer = self._create_optimizer_with_clipping()
        else:
            optimizer = self.model_config.optimizer
        
        # Compile model
        # Ensure consistent metric names
        result_metrics = ['categorical_accuracy' if m == 'accuracy' else m for m in self.model_config.metrics]
        self.model.compile(
            optimizer=optimizer,
            loss=self.model_config.loss,
            metrics=result_metrics
        )
        logger.debug(f"running _compile_model_optimized ... Using result_metrics: {result_metrics}")
        
        logger.debug("running _compile_model_optimized ... Model compiled successfully")
    
    def _create_optimizer_with_clipping(self) -> keras.optimizers.Optimizer:
        """Create optimizer with gradient clipping"""
        optimizer_map = {
            "adam": keras.optimizers.Adam,
            "sgd": keras.optimizers.SGD,
            "rmsprop": keras.optimizers.RMSprop
        }
        
        optimizer_class = optimizer_map.get(
            self.model_config.optimizer.lower(),
            keras.optimizers.Adam
        )
        
        return optimizer_class(clipnorm=self.model_config.gradient_clip_norm)
    
    def train(
        self, 
        data: Dict[str, Any], 
        validation_split: Optional[float] = None,
        use_multi_gpu: bool = False,
        plot_progress_callback: Optional[Callable[[str, int, int, float], None]] = None,
        epoch_progress_callback: Optional[Callable[[int, float], None]] = None,
        create_plots: bool = True
    ) -> keras.callbacks.History:
        """
        Enhanced training with optimized GPU proxy execution
        """
        # For multi-GPU, model building happens inside _train_locally_optimized
        # For single-GPU, build model here if not already built
        if not use_multi_gpu and self.model is None:
            logger.debug("running train ... No model found, building model first...")
            self.build_model()
        
        logger.debug("running train ... Starting model training...")
        
        # Execute local training
        logger.debug("running train ... Using local training execution")
        return self._train_locally_optimized(data, validation_split, use_multi_gpu, plot_progress_callback, epoch_progress_callback, create_plots)
    
    
    def _should_use_gpu_proxy(self) -> bool:
        """Determine if GPU proxy should be used"""
        return (
            self.gpu_proxy_available and 
            self.runpod_client is not None and 
            self.model_config.use_gpu_proxy
        )
        
    
    def _estimate_payload_size(self, x_train: np.ndarray, y_train: np.ndarray) -> float:
        """Estimate payload size in MB"""
        # Rough estimation based on array sizes and JSON overhead
        x_size = x_train.nbytes
        y_size = y_train.nbytes
        
        # JSON overhead (roughly 2-3x for nested arrays)
        json_overhead_factor = 2.5
        
        total_size_mb = (x_size + y_size) * json_overhead_factor / (1024 * 1024)
        
        return total_size_mb
    
    
    def _compress_context_data(
        self,
        model_config_dict: Dict[str, Any],
        dataset_config_dict: Dict[str, Any], 
        x_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Compress training data for large payload transmission with improved error handling"""
        import gzip
        import base64
        
        logger.debug("running _compress_context_data ... compressing training data with improved handling")
        
        try:
            # Convert arrays to JSON with optimized serialization
            logger.debug(f"running _compress_context_data ... x_train shape: {x_train.shape}, dtype: {x_train.dtype}")
            logger.debug(f"running _compress_context_data ... y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
            
            # Optimize data types before compression
            if x_train.dtype == np.float64:
                x_train = x_train.astype(np.float32)
                logger.debug("running _compress_context_data ... converted x_train from float64 to float32")
            
            if y_train.dtype == np.float64:
                y_train = y_train.astype(np.float32)
                logger.debug("running _compress_context_data ... converted y_train from float64 to float32")
            
            # Convert to JSON strings
            logger.debug("running _compress_context_data ... converting arrays to JSON...")
            x_train_json = json.dumps(x_train.tolist())
            y_train_json = json.dumps(y_train.tolist())
            
            logger.debug(f"running _compress_context_data ... JSON lengths: x={len(x_train_json)}, y={len(y_train_json)}")
            
            # Compress with high compression level
            logger.debug("running _compress_context_data ... compressing JSON data...")
            x_train_compressed = gzip.compress(x_train_json.encode('utf-8'), compresslevel=9)
            y_train_compressed = gzip.compress(y_train_json.encode('utf-8'), compresslevel=9)
            
            logger.debug(f"running _compress_context_data ... compressed lengths: x={len(x_train_compressed)}, y={len(y_train_compressed)}")
            
            # Encode as base64 for JSON transmission
            logger.debug("running _compress_context_data ... encoding to base64...")
            x_train_b64 = base64.b64encode(x_train_compressed).decode('utf-8')
            y_train_b64 = base64.b64encode(y_train_compressed).decode('utf-8')
            
            # Calculate compression statistics
            original_size = len(x_train_json) + len(y_train_json)
            compressed_size = len(x_train_b64) + len(y_train_b64)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            original_size_mb = original_size / (1024 * 1024)
            compressed_size_mb = compressed_size / (1024 * 1024)
            
            logger.debug(f"running _compress_context_data ... compression statistics:")
            logger.debug(f"running _compress_context_data ... - original size: {original_size_mb:.2f} MB")
            logger.debug(f"running _compress_context_data ... - compressed size: {compressed_size_mb:.2f} MB")
            logger.debug(f"running _compress_context_data ... - compression ratio: {compression_ratio:.3f}")
            logger.debug(f"running _compress_context_data ... - space saved: {(1-compression_ratio)*100:.1f}%")
            
            # Check if compression is effective
            if compression_ratio > 0.9:
                logger.warning(f"running _compress_context_data ... poor compression ratio: {compression_ratio:.3f}")
                logger.warning("running _compress_context_data ... consider reducing sample size further")
            
            return {
                'model_config': model_config_dict,
                'dataset_config': dataset_config_dict,
                'x_train_compressed': x_train_b64,
                'y_train_compressed': y_train_b64,
                'compressed': True,
                'original_size_mb': round(original_size_mb, 2),
                'compressed_size_mb': round(compressed_size_mb, 2),
                'compression_ratio': round(compression_ratio, 3),
                'space_saved_percent': round((1-compression_ratio)*100, 1)
            }
            
        except Exception as e:
            logger.error(f"running _compress_context_data ... compression failed: {e}")
            logger.error("running _compress_context_data ... falling back to uncompressed format")
            
            # Fallback to uncompressed format
            return {
                'model_config': model_config_dict,
                'dataset_config': dataset_config_dict,
                'x_train': x_train.tolist(),
                'y_train': y_train.tolist(),
                'compressed': False,
                'compression_error': str(e)
            }
    
    
    def _prepare_gpu_proxy_context_enhanced(
        self, 
        data: Dict[str, Any], 
        validation_split: Optional[float] = None
    ) -> Dict[str, Any]:
        """Prepare context data for GPU proxy execution with adaptive payload management"""
        logger.debug("running _prepare_gpu_proxy_context_enhanced ... preparing context with adaptive payload management")
        
        # Apply intelligent sampling to reduce payload size
        x_train, y_train = self._apply_intelligent_sampling(
            data['x_train'], 
            data['y_train']
        )
        
        # Detect data type
        data_type = self._detect_data_type_enhanced()
        
        # Prepare model configuration
        model_config_dict = {
            'data_type': data_type,
            'epochs': self.model_config.epochs,
            'optimizer': self.model_config.optimizer,
            'loss': self.model_config.loss,
            'metrics': self.model_config.metrics,
            'first_hidden_layer_nodes': self.model_config.first_hidden_layer_nodes,
            'first_hidden_layer_dropout': self.model_config.first_hidden_layer_dropout,
        }
        
        # Add architecture-specific parameters
        if data_type == 'text':
            model_config_dict.update({
                'sequence_length': self.dataset_config.img_width,
                'vocab_size': self.model_config.vocab_size,
                'embedding_dim': self.model_config.embedding_dim,
                'lstm_units': self.model_config.lstm_units
            })
        else:
            model_config_dict.update({
                'num_layers_conv': self.model_config.num_layers_conv,
                'filters_per_conv_layer': self.model_config.filters_per_conv_layer,
                'kernel_size': list(self.model_config.kernel_size) if isinstance(self.model_config.kernel_size, tuple) else self.model_config.kernel_size,
                'activation': self.model_config.activation
            })
        
        # Define dataset_config_dict
        dataset_config_dict = {
            'name': self.dataset_config.name,
            'num_classes': self.dataset_config.num_classes,
            'input_shape': list(self.dataset_config.input_shape) if hasattr(self.dataset_config, 'input_shape') else [self.dataset_config.img_height, self.dataset_config.img_width, self.dataset_config.channels]
        }
        
        # ENHANCED: Adaptive payload size management
        sample_percentage = self.model_config.gpu_proxy_sample_percentage
        
        # Estimate uncompressed payload size
        estimated_size_mb = self._estimate_payload_size(x_train, y_train)
        logger.debug(f"running _prepare_gpu_proxy_context_enhanced ... estimated payload size: {estimated_size_mb:.2f} MB")
        
        # Adaptive strategy based on size
        if estimated_size_mb > 50:  # Very large payload
            logger.warning(f"running _prepare_gpu_proxy_context_enhanced ... payload too large ({estimated_size_mb:.1f} MB), reducing sample size")
            # Reduce sample size automatically
            target_size_mb = 30
            reduction_factor = target_size_mb / estimated_size_mb
            new_sample_count = max(100, int(len(x_train) * reduction_factor))
            
            indices = np.random.choice(len(x_train), new_sample_count, replace=False)
            x_train = x_train[indices]
            y_train = y_train[indices]
            
            logger.debug(f"running _prepare_gpu_proxy_context_enhanced ... reduced to {len(x_train)} samples")
            estimated_size_mb = self._estimate_payload_size(x_train, y_train)
        
        # Use compression for payloads > 5MB or > 15% sample size
        if estimated_size_mb > 5.0 or sample_percentage > 0.15:
            logger.debug(f"running _prepare_gpu_proxy_context_enhanced ... using compression (size: {estimated_size_mb:.2f} MB, sample: {sample_percentage*100:.1f}%)")
            context_data = self._compress_context_data(
                model_config_dict, dataset_config_dict, x_train, y_train
            )
            
            # Check if compression was effective
            if context_data.get('compressed', False):
                compressed_size = context_data.get('compressed_size_mb', estimated_size_mb)
                if compressed_size > 25:  # Still too large even after compression
                    logger.error(f"running _prepare_gpu_proxy_context_enhanced ... payload still too large after compression: {compressed_size:.1f} MB")
                    raise ValueError(f"Payload too large even with compression: {compressed_size:.1f} MB. Try reducing gpu_proxy_sample_percentage below {sample_percentage*100:.0f}%")
            else:
                logger.warning("running _prepare_gpu_proxy_context_enhanced ... compression failed, using uncompressed format")
                
        else:
            logger.debug(f"running _prepare_gpu_proxy_context_enhanced ... using uncompressed format (size: {estimated_size_mb:.2f} MB)")
            # Original uncompressed format for small payloads
            context_data = {
                'model_config': model_config_dict,
                'dataset_config': dataset_config_dict,
                'x_train': x_train.tolist(),
                'y_train': y_train.tolist(),
                'compressed': False,
                'payload_size_mb': round(estimated_size_mb, 2)
            }
        
        final_size = context_data.get('compressed_size_mb', context_data.get('payload_size_mb', estimated_size_mb))
        logger.debug(f"running _prepare_gpu_proxy_context_enhanced ... context prepared with {len(x_train)} samples, final size: {final_size:.2f} MB")
        
        return context_data
    
    
    # Add this enhanced logging to the _apply_intelligent_sampling method in model_builder.py
    def _apply_intelligent_sampling(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply intelligent sampling based on configuration"""
        logger.debug("running _apply_intelligent_sampling ... applying sampling strategy")
        
        # 🎯 STEP 4.1f VERIFICATION: Log the sampling configuration
        logger.debug(f"running _apply_intelligent_sampling ... ModelConfig.gpu_proxy_sample_percentage: {self.model_config.gpu_proxy_sample_percentage}")
        logger.debug(f"running _apply_intelligent_sampling ... ModelConfig.gpu_proxy_use_stratified_sampling: {self.model_config.gpu_proxy_use_stratified_sampling}")
        
        total_samples = len(x_train)
        target_samples = int(total_samples * self.model_config.gpu_proxy_sample_percentage)
        
        # 🎯 DETAILED SAMPLING CALCULATION LOGS
        logger.debug(f"running _apply_intelligent_sampling ... SAMPLING CALCULATION:")
        logger.debug(f"running _apply_intelligent_sampling ... - Total available samples: {total_samples}")
        logger.debug(f"running _apply_intelligent_sampling ... - Sample percentage: {self.model_config.gpu_proxy_sample_percentage} ({self.model_config.gpu_proxy_sample_percentage*100:.1f}%)")
        logger.debug(f"running _apply_intelligent_sampling ... - Target sample count: {target_samples}")
        
        if target_samples >= total_samples:
            logger.debug("running _apply_intelligent_sampling ... ✅ NO SAMPLING: target_samples >= total_samples (using full dataset)")
            logger.debug(f"running _apply_intelligent_sampling ... - Using all {total_samples} samples (no reduction needed)")
            return x_train, y_train
        
        # 🎯 LOG SAMPLING STRATEGY SELECTION
        logger.debug(f"running _apply_intelligent_sampling ... 🔄 APPLYING SAMPLING: reducing from {total_samples} to {target_samples} samples")
        
        if self.model_config.gpu_proxy_use_stratified_sampling:
            logger.debug("running _apply_intelligent_sampling ... Using STRATIFIED sampling (maintains class balance)")
            
            # Stratified sampling to maintain class balance
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                # One-hot encoded labels
                labels = np.argmax(y_train, axis=1)
                logger.debug(f"running _apply_intelligent_sampling ... Detected one-hot encoded labels, shape: {y_train.shape}")
            else:
                labels = y_train.flatten()
                logger.debug(f"running _apply_intelligent_sampling ... Using direct labels, shape: {y_train.shape}")
            
            unique_classes = np.unique(labels)
            samples_per_class = max(1, target_samples // len(unique_classes))
            
            logger.debug(f"running _apply_intelligent_sampling ... STRATIFIED SAMPLING DETAILS:")
            logger.debug(f"running _apply_intelligent_sampling ... - Unique classes: {len(unique_classes)} classes")
            logger.debug(f"running _apply_intelligent_sampling ... - Target samples per class: {samples_per_class}")
            logger.debug(f"running _apply_intelligent_sampling ... - Class distribution: {[(cls, np.sum(labels == cls)) for cls in unique_classes]}")
            
            selected_indices = []
            for class_id in unique_classes:
                class_indices = np.where(labels == class_id)[0]
                if len(class_indices) > 0:
                    n_samples = min(samples_per_class, len(class_indices))
                    sampled = np.random.choice(class_indices, n_samples, replace=False)
                    selected_indices.extend(sampled)
                    logger.debug(f"running _apply_intelligent_sampling ... - Class {class_id}: sampled {n_samples} from {len(class_indices)} available")
            
            # Fill remaining slots if needed
            if len(selected_indices) < target_samples:
                remaining_needed = target_samples - len(selected_indices)
                all_indices = set(range(total_samples))
                unused_indices = list(all_indices - set(selected_indices))
                
                if unused_indices:
                    additional = np.random.choice(
                        unused_indices, 
                        min(remaining_needed, len(unused_indices)), 
                        replace=False
                    )
                    selected_indices.extend(additional)
                    logger.debug(f"running _apply_intelligent_sampling ... - Added {len(additional)} additional samples to reach target")
            
            selected_indices = np.array(selected_indices[:target_samples])
            
        else:
            logger.debug("running _apply_intelligent_sampling ... Using RANDOM sampling")
            # Random sampling
            selected_indices = np.random.choice(total_samples, target_samples, replace=False)
        
        sampled_x = x_train[selected_indices]
        sampled_y = y_train[selected_indices]
        
        # 🎯 VERIFICATION LOGS
        logger.debug(f"running _apply_intelligent_sampling ... SAMPLING RESULTS:")
        logger.debug(f"running _apply_intelligent_sampling ... - Original dataset: {x_train.shape}")
        logger.debug(f"running _apply_intelligent_sampling ... - Sampled dataset: {sampled_x.shape}")
        logger.debug(f"running _apply_intelligent_sampling ... - Reduction ratio: {len(sampled_x)/len(x_train):.3f} ({len(sampled_x)/len(x_train)*100:.1f}%)")
        logger.debug(f"running _apply_intelligent_sampling ... - Expected ratio: {self.model_config.gpu_proxy_sample_percentage:.3f} ({self.model_config.gpu_proxy_sample_percentage*100:.1f}%)")
        
        # Verify class distribution if using stratified sampling
        if self.model_config.gpu_proxy_use_stratified_sampling:
            if sampled_y.ndim > 1 and sampled_y.shape[1] > 1:
                sampled_labels = np.argmax(sampled_y, axis=1)
            else:
                sampled_labels = sampled_y.flatten()
            
            unique_sampled_classes, sampled_counts = np.unique(sampled_labels, return_counts=True)
            logger.debug(f"running _apply_intelligent_sampling ... - Sampled class distribution: {list(zip(unique_sampled_classes, sampled_counts))}")
        
        # Optimize data types if requested
        if self.model_config.gpu_proxy_optimize_data_types:
            logger.debug("running _apply_intelligent_sampling ... Optimizing data types...")
            if sampled_x.dtype == np.float64:
                sampled_x = sampled_x.astype(np.float32)
                logger.debug("running _apply_intelligent_sampling ... - Converted x from float64 to float32")
            if sampled_x.max() <= 1.0 and sampled_x.min() >= 0.0:
                # Convert to uint8 if data is normalized
                sampled_x = (sampled_x * 255).astype(np.uint8)
                logger.debug("running _apply_intelligent_sampling ... - Converted normalized data to uint8")
        
        logger.debug(f"running _apply_intelligent_sampling ... ✅ SAMPLING COMPLETE: {len(sampled_x)} samples selected from {total_samples} total")
        return sampled_x, sampled_y
    
    def _calculate_optimal_timeout(self, context_data: Dict[str, Any]) -> int:
        """Calculate optimal timeout based on context size and epochs"""
        base_timeout = 120  # Base timeout in seconds
        
        # Add time based on epochs
        epoch_time = self.model_config.epochs * 30  # 30 seconds per epoch estimate
        
        # Add time based on data size
        num_samples = len(context_data.get('x_train', []))
        data_time = max(60, num_samples // 100)  # Scale with data size
        
        total_timeout = base_timeout + epoch_time + data_time
        
        # Cap at reasonable maximum
        return min(total_timeout, 1800)  # Max 30 minutes
    
    
    def _convert_gpu_results_to_history_enhanced(self, execution_result: Dict[str, Any]) -> keras.callbacks.History:
        """Convert GPU proxy results back to Keras History object"""
        logger.debug("running _convert_gpu_results_to_history_enhanced ... converting results")
        
        # Create a mock History object
        history = keras.callbacks.History()
        history.history = execution_result.get('history', {})
        
        # Ensure all required keys exist
        for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
            if key not in history.history:
                history.history[key] = []
        
        logger.debug("running _convert_gpu_results_to_history_enhanced ... conversion completed")
        return history
    
    
    def _log_gpu_proxy_error(self, error: Exception, context: str) -> None:
        """Log GPU proxy errors with context"""
        logger.error(f"running _log_gpu_proxy_error ... GPU proxy error in {context}: {error}")
        logger.debug(f"running _log_gpu_proxy_error ... Error type: {type(error).__name__}")
        
        # Log additional context for debugging
        if hasattr(self, 'runpod_client') and self.runpod_client:
            try:
                endpoint = getattr(self.runpod_client, 'endpoint_id', 'unknown')
                logger.debug(f"running _log_gpu_proxy_error ... Endpoint: {endpoint}")
            except Exception:
                pass
    
    
    """
    Updated model_builder.py methods to handle minimal GPU proxy responses
    """

    def _convert_minimal_gpu_results_to_history(self, minimal_result: Dict[str, Any]) -> keras.callbacks.History:
        """
        Convert minimal GPU proxy results back to Keras History object
        """
        logger.debug("running _convert_minimal_gpu_results_to_history ... converting minimal results")
        
        # Create a mock History object
        history = keras.callbacks.History()
        
        # Extract metrics from minimal response
        epochs = minimal_result.get('epochs', 1)
        final_loss = minimal_result.get('loss', 0.0)
        final_acc = minimal_result.get('acc', 0.0)
        final_val_loss = minimal_result.get('val_loss', 0.0)
        final_val_acc = minimal_result.get('val_acc', 0.0)
        
        logger.debug(f"running _convert_minimal_gpu_results_to_history ... epochs: {epochs}")
        logger.debug(f"running _convert_minimal_gpu_results_to_history ... final metrics: loss={final_loss}, acc={final_acc}")
        
        # Reconstruct plausible training curves
        history.history = {}
        
        if final_loss > 0:
            # Create a decreasing loss curve
            start_loss = max(final_loss * 3, 2.0)
            history.history['loss'] = [
                start_loss - (start_loss - final_loss) * (i / max(1, epochs - 1)) 
                for i in range(epochs)
            ]
        
        if final_acc > 0:
            # Create an increasing accuracy curve
            start_acc = max(final_acc * 0.3, 0.1)
            history.history['accuracy'] = [
                start_acc + (final_acc - start_acc) * (i / max(1, epochs - 1)) 
                for i in range(epochs)
            ]
        
        if final_val_loss > 0:
            # Create validation loss curve
            start_val_loss = max(final_val_loss * 2.5, 1.5)
            history.history['val_loss'] = [
                start_val_loss - (start_val_loss - final_val_loss) * (i / max(1, epochs - 1)) 
                for i in range(epochs)
            ]
        
        if final_val_acc > 0:
            # Create validation accuracy curve
            start_val_acc = max(final_val_acc * 0.4, 0.1)
            history.history['val_accuracy'] = [
                start_val_acc + (final_val_acc - start_val_acc) * (i / max(1, epochs - 1)) 
                for i in range(epochs)
            ]
        
        # Ensure all required keys exist with defaults
        for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
            if key not in history.history:
                history.history[key] = [0.0] * epochs
        
        logger.debug(f"running _convert_minimal_gpu_results_to_history ... reconstructed {epochs} epochs")
        return history
  
        
    def _process_training_results(self, execution_result: Dict[str, Any]) -> keras.callbacks.History:
        """
        Process complete execution result from GPU proxy with comprehensive environment logging
        ENHANCED: Log complete environment comparison and validation metric analysis
        """
        logger.debug("running _process_training_results ... Processing complete execution result with comprehensive environment analysis")
        logger.debug(f"running _process_training_results ... execution_result is: {execution_result}")
        
        # Validate execution success
        if not execution_result.get('success', False):
            error_msg = execution_result.get('error', 'Unknown execution error')
            stderr = execution_result.get('stderr', '')
            stdout = execution_result.get('stdout', '')
            
            logger.error(f"running _process_training_results ... GPU execution failed: {error_msg}")
            if stderr:
                logger.error(f"running _process_training_results ... Stderr: {stderr}")
            if stdout:
                logger.debug(f"running _process_training_results ... Stdout: {stdout}")
            
            raise RuntimeError(f"GPU execution failed: {error_msg}")
        
        # Extract raw training result from complete execution
        raw_result = execution_result.get('result')
        logger.debug(f"running _process_training_results ... FULL RAW RESULT: {raw_result}")
        if raw_result is None:
            logger.error("running _process_training_results ... No result found in execution_result")
            raise RuntimeError("No training result found in execution result")
        
        logger.debug(f"running _process_training_results ... Raw result type: {type(raw_result)}")
        logger.debug(f"running _process_training_results ... Raw result keys: {list(raw_result.keys()) if isinstance(raw_result, dict) else 'Not a dict'}")
        
        # ENHANCED: Log comprehensive environment information
        if 'environment_info' in raw_result:
            env_info = raw_result['environment_info']
            logger.debug("running _process_training_results ... ========================================")
            logger.debug("running _process_training_results ... 🔬 COMPREHENSIVE GPU ENVIRONMENT ANALYSIS")
            logger.debug("running _process_training_results ... ========================================")
            
            # Python and TensorFlow versions
            logger.debug(f"running _process_training_results ... Python version: {env_info.get('python_version', 'Unknown')}")
            logger.debug(f"running _process_training_results ... TensorFlow version: {env_info.get('tensorflow_version', 'Unknown')}")
            logger.debug(f"running _process_training_results ... TensorFlow Keras version: {env_info.get('tensorflow_keras_version', 'Unknown')}")
            
            # Keras standalone analysis
            keras_standalone = env_info.get('keras_standalone_available', False)
            if keras_standalone:
                standalone_version = env_info.get('keras_standalone_version', 'Unknown')
                keras_version = env_info.get('tensorflow_keras_version', 'Unknown')
                logger.debug(f"running _process_training_results ... Keras standalone: Available (v{standalone_version})")
                logger.debug(f"running _process_training_results ... Keras version match: {standalone_version == keras_version}")
                if standalone_version != keras_version:
                    logger.warning(f"running _process_training_results ... ⚠️  KERAS VERSION MISMATCH DETECTED!")
                    logger.warning(f"running _process_training_results ... - TF Keras: {keras_version}")
                    logger.warning(f"running _process_training_results ... - Standalone: {standalone_version}")
            else:
                logger.debug("running _process_training_results ... Keras standalone: Not available")
            
            # CUDA and GPU information
            logger.debug(f"running _process_training_results ... TensorFlow built with CUDA: {env_info.get('tensorflow_cuda_built', 'Unknown')}")
            logger.debug(f"running _process_training_results ... CUDA version: {env_info.get('cuda_version', 'Unknown')}")
            logger.debug(f"running _process_training_results ... cuDNN version: {env_info.get('cudnn_version', 'Unknown')}")
            logger.debug(f"running _process_training_results ... GPU devices: {env_info.get('gpu_devices', 'Unknown')}")
            
            # Metric information
            logger.debug(f"running _process_training_results ... Metric used for training: {env_info.get('metric_used', 'Unknown')}")
            logger.debug(f"running _process_training_results ... Model metric names: {env_info.get('metric_names', 'Unknown')}")
            logger.debug(f"running _process_training_results ... Validation split: {env_info.get('validation_split_used', 'Unknown')}")
            
            # TensorFlow build info (selective - most important ones)
            build_info = env_info.get('tf_build_info', {})
            if build_info:
                logger.debug("running _process_training_results ... TensorFlow build details:")
                important_keys = ['cuda_version', 'cudnn_version', 'is_cuda_build', 'cuda_compute_capabilities']
                for key in important_keys:
                    if key in build_info:
                        logger.debug(f"running _process_training_results ... - {key}: {build_info[key]}")
        
        # ENHANCED: Log validation metric analysis
        if 'validation_analysis' in raw_result:
            val_analysis = raw_result['validation_analysis']
            logger.debug("running _process_training_results ... ========================================")
            logger.debug("running _process_training_results ... 🎯 VALIDATION METRIC ANALYSIS")
            logger.debug("running _process_training_results ... ========================================")
            
            val_metrics = val_analysis.get('val_metrics_found', [])
            val_acc_keys = val_analysis.get('val_accuracy_keys', [])
            val_working = val_analysis.get('val_accuracy_working', False)
            
            logger.debug(f"running _process_training_results ... Validation metrics found: {val_metrics}")
            logger.debug(f"running _process_training_results ... Validation accuracy keys: {val_acc_keys}")
            logger.debug(f"running _process_training_results ... Validation accuracy working: {val_working}")
            
            # Minimal test results
            minimal_test = val_analysis.get('minimal_test_results', {})
            if minimal_test:
                minimal_working = minimal_test.get('val_acc_working', False)
                minimal_values = minimal_test.get('val_acc_values', [])
                logger.debug(f"running _process_training_results ... Minimal test validation working: {minimal_working}")
                logger.debug(f"running _process_training_results ... Minimal test values: {minimal_values}")
            
            # CRITICAL: Flag the validation issue if detected
            if not val_working:
                logger.error("running _process_training_results ... 🚨 VALIDATION ACCURACY ISSUE CONFIRMED!")
                logger.error("running _process_training_results ... - Validation accuracy metrics are returning zero")
                logger.error("running _process_training_results ... - This confirms the environment-specific bug")
            else:
                logger.debug("running _process_training_results ... ✅ Validation accuracy working correctly")
        
        # Process the training history
        if isinstance(raw_result, dict) and 'history' in raw_result:
            logger.debug("running _process_training_results ... Found training history in raw result")
            
            # Extract training history
            history_data = raw_result['history']
            logger.debug(f"running _process_training_results ... History keys: {list(history_data.keys()) if isinstance(history_data, dict) else 'Not a dict'}")
            
            # ENHANCED: Log detailed validation accuracy analysis
            for key, values in history_data.items():
                if 'val' in key and ('acc' in key or 'accuracy' in key):
                    logger.debug(f"running _process_training_results ... 🔍 VALIDATION METRIC DETAILED ANALYSIS:")
                    logger.debug(f"running _process_training_results ... - Key: {key}")
                    logger.debug(f"running _process_training_results ... - Values: {values}")
                    logger.debug(f"running _process_training_results ... - Length: {len(values) if values else 0}")
                    logger.debug(f"running _process_training_results ... - All zeros: {all(v == 0 for v in values) if values else 'Empty'}")
                    logger.debug(f"running _process_training_results ... - Any non-zero: {any(v != 0 for v in values) if values else 'Empty'}")
                    
                    if values and all(v == 0 for v in values):
                        logger.error(f"running _process_training_results ... 🚨 CONFIRMED: {key} is all zeros!")
                        logger.error("running _process_training_results ... This is the validation accuracy bug")
            
            # Create Keras History object
            history = keras.callbacks.History()
            logger.debug("running _process_training_results ... Creating Keras History object")
            
            normalized_history = self._normalize_metric_names(history_data)
            history.history = normalized_history
                        
            # Validate training history
            self._validate_training_history(history)
            
            # Log additional training metadata if available
            if 'model_params' in raw_result:
                logger.debug(f"running _process_training_results ... Model parameters: {raw_result['model_params']:,}")
            if 'training_time' in raw_result:
                logger.debug(f"running _process_training_results ... Training time: {raw_result['training_time']:.2f}s")
            
            # SUMMARY LOG: Environment comparison for Phase 3 analysis
            logger.debug("running _process_training_results ... ========================================")
            logger.debug("running _process_training_results ... 📊 PHASE 3 ENVIRONMENT SUMMARY")
            logger.debug("running _process_training_results ... ========================================")
            
            # Compare with local environment (if available in logs)
            if 'environment_info' in raw_result:
                env_info = raw_result['environment_info']
                logger.debug("running _process_training_results ... GPU Environment:")
                logger.debug(f"running _process_training_results ... - TensorFlow: {env_info.get('tensorflow_version', 'Unknown')}")
                logger.debug(f"running _process_training_results ... - Keras (TF): {env_info.get('tensorflow_keras_version', 'Unknown')}")
                logger.debug(f"running _process_training_results ... - Keras (standalone): {env_info.get('keras_standalone_version', 'Not available')}")
                logger.debug(f"running _process_training_results ... - CUDA: {env_info.get('cuda_version', 'Unknown')}")
                logger.debug(f"running _process_training_results ... - cuDNN: {env_info.get('cudnn_version', 'Unknown')}")
                
            # Local environment for comparison
            import tensorflow as tf
            logger.debug("running _process_training_results ... Local Environment (for comparison):")
            logger.debug(f"running _process_training_results ... - TensorFlow: {tf.__version__}")
            logger.debug(f"running _process_training_results ... - Keras (TF): {tf.keras.__version__}")             # type: ignore
            logger.debug(f"running _process_training_results ... - CUDA built: {tf.test.is_built_with_cuda()}")
            
            logger.debug("running _process_training_results ... Training history processed successfully with comprehensive analysis")
            return history
        
        else:
            logger.error("running _process_training_results ... No training history found in raw result")
            logger.error(f"running _process_training_results ... Available keys: {list(raw_result.keys()) if isinstance(raw_result, dict) else 'Not a dict'}")
            raise RuntimeError("No training history found in execution result")

    def _validate_training_history(self, history: keras.callbacks.History) -> None:
        """
        Validate training history has required metrics
        
        Args:
            history: Keras History object to validate
            
        Raises:
            RuntimeError: If history is invalid or missing required metrics
        """
        logger.debug("running _validate_training_history ... Validating training history")
        logger.debug(f"running _validate_training_history ... history is: {history}")
        logger.debug(f"running _validate_training_history ... history.history is: {history.history}")
        
        if not hasattr(history, 'history') or not isinstance(history.history, dict):
            raise RuntimeError("Invalid training history: missing or invalid history attribute")
        
        if not history.history:
            raise RuntimeError("Invalid training history: empty history dictionary")
        
        # Check for at least one metric
        # 🎯 PHASE 3 FIX: Check for both old and new metric names
        required_metrics = ['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy']
        found_metrics = [metric for metric in required_metrics if metric in history.history]

        # Also check legacy names for backward compatibility
        legacy_metrics = ['accuracy', 'val_accuracy']
        found_legacy = [metric for metric in legacy_metrics if metric in history.history]

        if found_legacy:
            logger.debug(f"running _validate_training_history ... Found legacy metrics: {found_legacy}")
            found_metrics.extend(found_legacy)
        
        if not found_metrics:
            logger.warning("running _validate_training_history ... No standard metrics found, checking alternative names")
            # Check alternative metric names
            alt_metrics = ['acc', 'val_acc']
            found_alt_metrics = [metric for metric in alt_metrics if metric in history.history]
            
            if not found_alt_metrics:
                available_keys = list(history.history.keys())
                logger.error(f"running _validate_training_history ... Available metrics: {available_keys}")
                raise RuntimeError(f"No valid training metrics found. Available: {available_keys}")
            else:
                logger.debug(f"running _validate_training_history ... Found alternative metrics: {found_alt_metrics}")
        else:
            logger.debug(f"running _validate_training_history ... Found standard metrics: {found_metrics}")
        
        # Validate metric arrays have consistent lengths
        metric_lengths = {key: len(values) for key, values in history.history.items() if isinstance(values, list)}
        if metric_lengths:
            unique_lengths = set(metric_lengths.values())
            if len(unique_lengths) > 1:
                logger.warning(f"running _validate_training_history ... Inconsistent metric lengths: {metric_lengths}")
            else:
                epochs = list(unique_lengths)[0]
                logger.debug(f"running _validate_training_history ... Training history validated: {epochs} epochs")
        
        logger.debug("running _validate_training_history ... Training history validation completed")
    
    
    def _normalize_metric_names(self, history_data: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        🎯 PHASE 3 FIX: Normalize metric names for consistent processing
        Maps both 'val_accuracy' and 'val_categorical_accuracy' to expected names
        """
        logger.debug("running _normalize_metric_names ... normalizing metric key names")
        
        normalized = dict(history_data)  # Copy original data
        
        # Map categorical_accuracy to accuracy for backward compatibility if needed
        if 'categorical_accuracy' in normalized and 'accuracy' not in normalized:
            normalized['accuracy'] = normalized['categorical_accuracy']
            logger.debug("running _normalize_metric_names ... mapped categorical_accuracy to accuracy")
        
        # Map val_categorical_accuracy to val_accuracy for backward compatibility if needed  
        if 'val_categorical_accuracy' in normalized and 'val_accuracy' not in normalized:
            normalized['val_accuracy'] = normalized['val_categorical_accuracy']
            logger.debug("running _normalize_metric_names ... mapped val_categorical_accuracy to val_accuracy")
        
        logger.debug(f"running _normalize_metric_names ... final metric keys: {list(normalized.keys())}")
        return normalized

    
    def _is_model_suitable_for_multi_gpu(self) -> bool:
        """
        Determine if the current model configuration is suitable for multi-GPU training
        
        Returns:
            True if model is complex enough to benefit from multi-GPU
        """
        # Check if model has sufficient complexity for multi-GPU benefits
        min_params_for_multi_gpu = 100_000  # Minimum parameters to benefit from multi-GPU
        min_epochs_for_multi_gpu = 15      # Minimum epochs to amortize multi-GPU overhead
        
        # Estimate parameter count based on configuration
        estimated_params = self._estimate_model_parameters()
        
        is_suitable = (
            estimated_params >= min_params_for_multi_gpu and
            self.model_config.epochs >= min_epochs_for_multi_gpu
        )
        
        logger.debug(f"running _is_model_suitable_for_multi_gpu ... Estimated parameters: {estimated_params:,}")
        logger.debug(f"running _is_model_suitable_for_multi_gpu ... Epochs: {self.model_config.epochs}")
        logger.debug(f"running _is_model_suitable_for_multi_gpu ... Suitable for multi-GPU: {is_suitable}")
        
        return is_suitable

    def _estimate_model_parameters(self) -> int:
        """
        Estimate the number of parameters in the model based on configuration
        
        Returns:
            Estimated parameter count
        """
        try:
            if self.model is not None:
                return self.model.count_params()
        except:
            pass
        
        # Rough estimation based on configuration
        if self._detect_data_type_enhanced() == "text":
            # LSTM model estimation
            vocab_size = self.model_config.vocab_size
            embedding_dim = self.model_config.embedding_dim
            lstm_units = self.model_config.lstm_units
            
            # Embedding layer
            embedding_params = vocab_size * embedding_dim
            
            # LSTM layer (rough approximation)
            lstm_params = 4 * lstm_units * (embedding_dim + lstm_units + 1)
            
            # Dense layers
            dense_params = lstm_units * self.model_config.first_hidden_layer_nodes
            output_params = self.model_config.first_hidden_layer_nodes * self.dataset_config.num_classes
            
            total_params = embedding_params + lstm_params + dense_params + output_params
        else:
            # CNN model estimation
            input_size = self.dataset_config.img_height * self.dataset_config.img_width * self.dataset_config.channels
            
            # Convolutional layers (rough approximation)
            conv_params = 0
            for i in range(self.model_config.num_layers_conv):
                kernel_params = (
                    self.model_config.kernel_size[0] * 
                    self.model_config.kernel_size[1] * 
                    self.model_config.filters_per_conv_layer * 
                    (self.dataset_config.channels if i == 0 else self.model_config.filters_per_conv_layer)
                )
                conv_params += kernel_params
            
            # Dense layers (rough approximation)
            flattened_size = input_size // (4 ** self.model_config.num_layers_conv)  # Rough pooling reduction
            dense_params = flattened_size * self.model_config.first_hidden_layer_nodes
            output_params = self.model_config.first_hidden_layer_nodes * self.dataset_config.num_classes
            
            total_params = conv_params + dense_params + output_params
        
        return max(1000, int(total_params))  # Minimum of 1000 parameters
    
    def _train_locally_optimized(
        self, 
        data: Dict[str, Any], 
        validation_split: Optional[float] = None,
        use_multi_gpu: bool = False,
        plot_progress_callback: Optional[Callable[[str, int, int, float], None]] = None,
        epoch_progress_callback: Optional[Callable[[int, float], None]] = None,
        create_plots: bool = True
    ) -> keras.callbacks.History:
        """
        Optimized local training execution with PROPER multi-GPU implementation
        """
        logger.debug("running _train_locally_optimized ... Starting optimized local training")
        
        # Log performance information
        self.perf_logger.log_data_info(
            total_images=len(data['x_train']) + len(data['x_test']),
            train_size=len(data['x_train']),
            test_size=len(data['x_test']),
            num_categories=self.dataset_config.num_classes
        )
        
        # Enhanced callback setup
        callbacks_list = self._setup_training_callbacks_optimized()
        
        # Add epoch progress callback if provided (for final model building)
        if epoch_progress_callback:
            from keras.callbacks import Callback
            
            class FinalModelEpochCallback(Callback):
                def __init__(self, progress_callback):
                    super().__init__()
                    self.progress_callback = progress_callback
                    self.current_epoch = 0
                    self.total_epochs = 0
                    
                def on_train_begin(self, logs=None):
                    self.total_epochs = self.params.get('epochs', 0) if self.params else 0
                    
                def on_epoch_begin(self, epoch, logs=None):
                    self.current_epoch = epoch + 1  # Convert to 1-based
                    self.progress_callback(self.current_epoch, 0.0)  # Start of epoch
                    
                def on_batch_end(self, batch, logs=None):
                    if hasattr(self, 'params') and self.params:
                        total_batches = self.params.get('steps', 1)
                        batch_progress = (batch + 1) / total_batches
                        self.progress_callback(self.current_epoch, batch_progress)
                        
                def on_epoch_end(self, epoch, logs=None):
                    self.progress_callback(self.current_epoch, 1.0)  # End of epoch
            
            callbacks_list.append(FinalModelEpochCallback(epoch_progress_callback))
        
        # Check multi-GPU availability and model requirements
        available_gpus = tf.config.list_physical_devices('GPU')
        should_use_multi_gpu = (
            use_multi_gpu and 
            len(available_gpus) > 1 and
            self._is_model_suitable_for_multi_gpu()
        )
        
        logger.debug(f"running _train_locally_optimized ... Available GPUs: {len(available_gpus)}")
        logger.debug(f"running _train_locally_optimized ... Multi-GPU requested: {use_multi_gpu}")
        logger.debug(f"running _train_locally_optimized ... Will use multi-GPU: {should_use_multi_gpu}")
        
        # Apply manual validation split for consistency
        validation_split_value = validation_split or self.model_config.validation_split
        logger.debug(f"running _train_locally_optimized ... validation_split_value: {validation_split_value}")
        
        if validation_split_value > 0:
            logger.debug("running _train_locally_optimized ... Applying manual validation split")
            
            x_train = data['x_train']
            y_train = data['y_train']
            split_idx = int(len(x_train) * (1 - validation_split_value))
            
            x_train_manual = x_train[:split_idx]
            y_train_manual = y_train[:split_idx]
            x_val_manual = x_train[split_idx:]
            y_val_manual = y_train[split_idx:]
            
            logger.debug(f"running _train_locally_optimized ... Training samples: {len(x_train_manual)}")
            logger.debug(f"running _train_locally_optimized ... Validation samples: {len(x_val_manual)}")
            
            training_data = (x_train_manual, y_train_manual)
            validation_data = (x_val_manual, y_val_manual)
        else:
            logger.debug("running _train_locally_optimized ... No validation split")
            training_data = (data['x_train'], data['y_train'])
            validation_data = None
        
        # Proper multi-GPU model building and training
        if should_use_multi_gpu:
            logger.debug("running _train_locally_optimized ... Using MirroredStrategy for multi-GPU training")
            strategy = tf.distribute.MirroredStrategy()
            logger.debug(f"running _train_locally_optimized ... Strategy devices: {strategy.extended.worker_devices}")
            logger.debug(f"running _train_locally_optimized ... Number of replicas: {strategy.num_replicas_in_sync}")
            
            with strategy.scope():
                # CRITICAL FIX: Build model components directly inside strategy scope
                # Do NOT call self.build_model() which may have its own strategy logic
                self.model = None  # Reset any existing model
                
                logger.debug("running _train_locally_optimized ... Building model components inside strategy scope")
                
                # Detect data type
                data_type = self._detect_data_type_enhanced()
                
                # Build appropriate model architecture directly
                if data_type == "text":
                    logger.debug("running _train_locally_optimized ... Building TEXT model inside strategy")
                    self.model = self._build_text_model_optimized()
                else:
                    logger.debug("running _train_locally_optimized ... Building CNN model inside strategy")
                    self.model = self._build_cnn_model_optimized()
                
                # Compile model directly inside strategy scope
                self._compile_model_optimized()
                
                # TYPE SAFETY: Ensure model was built successfully
                if self.model is None:
                    raise RuntimeError("Failed to build model inside strategy scope")
                
                logger.debug("running _train_locally_optimized ... Model built and compiled inside strategy scope")
                
                # Scale batch size for multi-GPU (optional optimization)
                # ENHANCED: Ensure minimum effective batch size for multi-GPU
                original_batch_size = getattr(self.model_config, 'batch_size', 32)
                min_batch_size_per_gpu = 8  # Minimum batch size per GPU for effective distribution
                min_total_batch_size = min_batch_size_per_gpu * strategy.num_replicas_in_sync

                # Use larger of scaled original or minimum required
                scaled_batch_size = max(
                    original_batch_size * strategy.num_replicas_in_sync,
                    min_total_batch_size
                )

                logger.debug(f"running _train_locally_optimized ... Multi-GPU batch size optimization:")
                logger.debug(f"running _train_locally_optimized ... - Original batch size: {original_batch_size}")
                logger.debug(f"running _train_locally_optimized ... - Min per GPU: {min_batch_size_per_gpu}")
                logger.debug(f"running _train_locally_optimized ... - GPUs: {strategy.num_replicas_in_sync}")
                logger.debug(f"running _train_locally_optimized ... - Final batch size: {scaled_batch_size}")
                logger.debug(f"running _train_locally_optimized ... - Per GPU batch size: {scaled_batch_size // strategy.num_replicas_in_sync}")
                logger.debug(f"running _train_locally_optimized ... About to start training with scaled_batch_size: {scaled_batch_size}")
                logger.debug(f"running _train_locally_optimized ... Training data shapes: {training_data[0].shape}, {training_data[1].shape}")

                with TimedOperation("multi-GPU model training", "model_builder"):
                    if validation_data:
                        self.training_history = self.model.fit(
                            training_data[0], 
                            training_data[1],
                            epochs=self.model_config.epochs,
                            batch_size=scaled_batch_size,
                            validation_data=validation_data,
                            verbose=1,
                            callbacks=callbacks_list
                        )
                    else:
                        self.training_history = self.model.fit(
                            training_data[0], 
                            training_data[1],
                            epochs=self.model_config.epochs,
                            batch_size=scaled_batch_size,
                            verbose=1,
                            callbacks=callbacks_list
                        )
        else:
            logger.debug("running _train_locally_optimized ... Using single-GPU training")
            
            # Build model normally for single GPU
            if self.model is None:
                self.build_model()
            
            # TYPE SAFETY: Ensure model was built successfully
            if self.model is None:
                raise RuntimeError("Failed to build model")
            
            with TimedOperation("single-GPU model training", "model_builder"):
                if validation_data:
                    self.training_history = self.model.fit(
                        training_data[0], 
                        training_data[1],
                        epochs=self.model_config.epochs,
                        validation_data=validation_data,
                        verbose=1,
                        callbacks=callbacks_list
                    )
                else:
                    self.training_history = self.model.fit(
                        training_data[0], 
                        training_data[1],
                        epochs=self.model_config.epochs,
                        verbose=1,
                        callbacks=callbacks_list
                    )
        
        logger.debug("running _train_locally_optimized ... Training completed")
        
        # TYPE SAFETY: Ensure training history was created
        if self.training_history is None:
            raise RuntimeError("Training failed - no history returned")
        
        # Generate training visualization plots after successful training (if enabled)
        if create_plots:
            self._generate_training_plots(training_data, validation_data, plot_progress_callback)
        else:
            logger.debug("running _train_locally_optimized ... Skipping plot generation (create_plots=False)")
        
        return self.training_history

    def _setup_training_callbacks_optimized(self) -> List[keras.callbacks.Callback]:
        """Setup optimized training callbacks including TensorBoard"""
        callbacks_list = []
        
        # TensorBoard logging for trial metrics and visualization
        if self.trial_number is not None:
            # Use results_dir if provided (optimizer integration), otherwise fall back to project root
            if self.results_dir:
                # Place TensorBoard logs inside the optimization results directory
                log_dir = self.results_dir / "tensorboard_logs" / f"trial_{self.trial_number}"
            else:
                # Fallback to project root with namespaced directory structure
                project_root = Path(__file__).resolve().parent.parent
                if self.run_timestamp:
                    # Clean dataset name for directory naming
                    dataset_name_clean = self.dataset_config.name.replace('-', '_').replace(' ', '_').lower()
                    run_dir = f"{self.run_timestamp}_{dataset_name_clean}"
                    log_dir = project_root / f"tensorboard_logs/{run_dir}/trial_{self.trial_number}"
                else:
                    # Fallback to old naming if no timestamp provided
                    log_dir = project_root / f"tensorboard_logs/trial_{self.trial_number}"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Add TensorBoard callback for comprehensive logging
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,  # Log weight histograms every epoch
                write_graph=True,  # Log model architecture
                write_images=True,  # Log model weights as images
                update_freq='epoch',  # Update logs every epoch
                profile_batch=0,  # Disable profiling for performance
                embeddings_freq=0  # Disable embeddings logging
            )
            callbacks_list.append(tensorboard_callback)
            
            # Add custom health metrics callback for TensorBoard integration
            health_callback = self._create_health_metrics_callback(log_dir)
            callbacks_list.append(health_callback)
            
            logger.debug(f"TensorBoard logging enabled for trial {self.trial_number} at {log_dir}")
        
        # REMOVED: Real-time visualization callbacks setup
        # These would require plot_dir which is no longer maintained by ModelBuilder
        # Real-time callbacks can be added by the orchestrator if needed
        
        return callbacks_list

    def _create_health_metrics_callback(self, log_dir: Path) -> keras.callbacks.Callback:
        """Create custom callback for logging health metrics to TensorBoard"""
        
        class HealthMetricsCallback(keras.callbacks.Callback):
            def __init__(self, log_dir: Path, health_analyzer):
                super().__init__()
                self.log_dir = log_dir
                self.health_analyzer = health_analyzer
                self.writer = tf.summary.create_file_writer(str(log_dir / "health_metrics"))
                
            def on_epoch_end(self, epoch, logs=None):
                """Log health metrics to TensorBoard at the end of each epoch"""
                logs = logs or {}
                
                # Extract basic metrics from training logs
                train_loss = logs.get('loss', 0.0)
                train_acc = logs.get('accuracy', 0.0)
                val_loss = logs.get('val_loss', 0.0)
                val_acc = logs.get('val_accuracy', 0.0)
                
                # Log custom health metrics
                with self.writer.as_default():
                    # Training stability metrics
                    tf.summary.scalar('health/training_stability', 
                                    1.0 - min(abs(train_loss - val_loss), 1.0), step=epoch)
                    
                    # Accuracy consistency
                    tf.summary.scalar('health/accuracy_consistency', 
                                    1.0 - min(abs(train_acc - val_acc), 1.0), step=epoch)
                    
                    # Overfitting indicator (higher is worse)
                    overfitting_score = max(0.0, (train_acc - val_acc) / max(train_acc, 0.01))
                    tf.summary.scalar('health/overfitting_risk', overfitting_score, step=epoch)
                    
                    # Learning progress (improvement rate)
                    if epoch > 0:
                        prev_loss = getattr(self, '_prev_val_loss', val_loss)
                        improvement = max(0.0, (prev_loss - val_loss) / max(prev_loss, 0.01))
                        tf.summary.scalar('health/learning_progress', improvement, step=epoch)
                    
                    self._prev_val_loss = val_loss
                    self.writer.flush()
        
        return HealthMetricsCallback(log_dir, self.health_analyzer)

    def _generate_training_plots(self, training_data: Tuple[np.ndarray, np.ndarray], validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, plot_progress_callback: Optional[Callable[[str, int, int, float], None]] = None) -> None:
        """
        Generate comprehensive training visualization plots after training completion
        
        Uses PlotGenerator to create all available plot types based on optimization flags:
        - Training history: Loss/accuracy curves with overfitting detection
        - Activation maps: CNN layer activations and filter visualizations (image data only)
        - Confusion matrix: Classification accuracy and error analysis
        - Training animation: Animated training progress visualization
        - Weights & biases: Parameter health and distribution analysis
        - Gradient flow: Dead neuron detection and gradient health
        
        Args:
            training_data: Tuple of (X_train, y_train) data
            validation_data: Optional tuple of (X_val, y_val) data
        """
        if not self.model or not self.training_history:
            logger.warning("running _generate_training_plots ... No model or training history available, skipping plot generation")
            return
            
        try:
            # Determine plot output directory
            if self.results_dir:
                # Use results directory if provided (optimizer integration)
                plot_dir = self.results_dir / "plots" / f"trial_{self.trial_number or 0}"
            else:
                # Fallback to project root structure
                project_root = Path(__file__).resolve().parent.parent
                if self.run_timestamp:
                    dataset_name_clean = self.dataset_config.name.replace('-', '_').replace(' ', '_').lower()
                    run_dir = f"{self.run_timestamp}_{dataset_name_clean}"
                    plot_dir = project_root / f"plots/{run_dir}/trial_{self.trial_number or 0}"
                else:
                    plot_dir = project_root / f"plots/trial_{self.trial_number or 0}"
            
            plot_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"running _generate_training_plots ... Created plot directory: {plot_dir}")
            
            # Prepare test data for comprehensive plot generation
            if validation_data is not None:
                test_data = {'x_test': validation_data[0], 'y_test': validation_data[1]}
                test_loss, test_accuracy = self.model.evaluate(validation_data[0], validation_data[1], verbose=0)
            else:
                # Use a sample from training data as test data
                sample_size = min(1000, len(training_data[0]))
                test_data = {'x_test': training_data[0][:sample_size], 'y_test': training_data[1][:sample_size]}
                test_loss, test_accuracy = self.model.evaluate(test_data['x_test'], test_data['y_test'], verbose=0)
            
            # Use comprehensive PlotGenerator (delayed import to avoid circular dependency)
            from plot_generator import PlotGenerator
            
            # Create optimization config with plot flags (if not available, all plots enabled by default)
            optimization_config = None
            if hasattr(self, 'optimization_config'):
                optimization_config = self.optimization_config
            
            plot_generator = PlotGenerator(
                dataset_config=self.dataset_config,
                model_config=self.model_config,
                optimization_config=optimization_config
            )
            
            logger.debug("running _generate_training_plots ... Generating comprehensive plots using PlotGenerator...")
            analysis_results = plot_generator.generate_comprehensive_plots(
                model=self.model,
                training_history=self.training_history,
                data=test_data,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
                run_timestamp=self.run_timestamp or datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                plot_dir=plot_dir,
                log_detailed_predictions=True,
                max_predictions_to_show=20,
                progress_callback=plot_progress_callback
            )
            
            # Log which plot types were generated
            generated_plots = []
            for plot_type, result in analysis_results.items():
                if result and not result.get('error'):
                    generated_plots.append(plot_type)
            
            if generated_plots:
                logger.info(f"running _generate_training_plots ... Generated plots: {', '.join(generated_plots)}")
            else:
                logger.warning("running _generate_training_plots ... No plots were generated successfully")
            
            logger.info(f"running _generate_training_plots ... Successfully generated training plots in: {plot_dir}")
            
        except Exception as e:
            logger.error(f"running _generate_training_plots ... Failed to generate plots: {str(e)}")
            logger.debug(f"running _generate_training_plots ... Plot generation traceback: {traceback.format_exc()}")

    def evaluate(
        self, 
        data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        ✅ EVALUATION CONSOLIDATION: Simplified model evaluation using HealthAnalyzer delegation
        
        This method now acts as a thin wrapper that:
        1. Calls HealthAnalyzer.calculate_comprehensive_health() with data parameter
        2. Extracts test_loss and test_accuracy from the comprehensive results  
        3. Maintains the same return signature for backward compatibility
        
        Args:
            data: Dictionary containing test data
            
        Returns:
            Tuple of (test_loss, test_accuracy) - same as before for compatibility
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluation")
        
        logger.debug("running evaluate ... Starting CONSOLIDATED model evaluation via HealthAnalyzer...")
        
        # ✅ Use HealthAnalyzer as single source of truth for evaluation
        comprehensive_metrics = self.health_analyzer.calculate_comprehensive_health(
            model=self.model,
            history=self.training_history,
            data=data,  # ✅ KEY: Pass data to get basic metrics
            sample_data=data['x_test'][:50] if len(data['x_test']) > 50 else data['x_test'],
            training_time_minutes=getattr(self, 'training_time_minutes', None),
            total_params=self.model.count_params()
        )
        
        # ✅ Extract basic metrics that ModelBuilder users expect (backward compatibility)
        test_loss = comprehensive_metrics.get('test_loss', 0.0)
        test_accuracy = comprehensive_metrics.get('test_accuracy', 0.0)
        
        # Log evaluation results with health context
        overall_health = comprehensive_metrics.get('overall_health', 0.5)
        logger.debug(f"running evaluate ... CONSOLIDATED evaluation completed:")
        logger.debug(f"running evaluate ... - Test accuracy: {test_accuracy:.4f}")
        logger.debug(f"running evaluate ... - Test loss: {test_loss:.4f}")
        logger.debug(f"running evaluate ... - Overall model health: {overall_health:.3f}")
        
        # ✅ Store comprehensive metrics for potential later access
        self.last_comprehensive_evaluation = comprehensive_metrics
        logger.debug("running evaluate ... Comprehensive metrics stored for later access")
        
        return test_loss, test_accuracy
    
    def _extract_evaluation_metrics(self, evaluation_results: Union[float, List[float]]) -> Tuple[float, float]:
        """Extract loss and accuracy from evaluation results"""
        if isinstance(evaluation_results, list):
            test_loss = float(evaluation_results[0])
            test_accuracy = float(evaluation_results[1]) if len(evaluation_results) > 1 else 0.0
        else:
            test_loss = float(evaluation_results)
            test_accuracy = 0.0
        
        return test_loss, test_accuracy
    
    
    def get_last_health_analysis(self) -> Optional[Dict[str, Any]]:
        """
        ✅ NEW: Get comprehensive health metrics from last evaluation
        
        Provides access to the full health analysis results from the most recent
        evaluate() call, enabling users to access detailed health metrics beyond
        just test_loss and test_accuracy.
        
        Returns:
            Dictionary with comprehensive health metrics or None if no evaluation performed
            
        Example:
            # Get basic metrics
            test_loss, test_accuracy = model_builder.evaluate(data)
            
            # Get detailed health analysis from same evaluation
            health_metrics = model_builder.get_last_health_analysis()
            if health_metrics:
                overall_health = health_metrics['overall_health']
                recommendations = health_metrics['recommendations']
                neuron_utilization = health_metrics['neuron_utilization']
        """
        return getattr(self, 'last_comprehensive_evaluation', None)
    
    
    # Model saving methods
    def _generate_optimized_filename(
        self, 
        run_timestamp: str,
        test_accuracy: Optional[float] = None,
        run_name: Optional[str] = None
    ) -> str:
        """
        Generate optimized filename for model saving
        
        Args:
            test_accuracy: Test accuracy value
            run_timestamp: Timestamp for file naming
            run_name: Optional run name for consistency
            
        Returns:
            Generated filename string
        """
        logger.debug("running _generate_optimized_filename ... generating optimized filename")
        
        # Base filename components
        dataset_name_clean = self.dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        # Include accuracy if available
        if test_accuracy is not None:
            accuracy_str = f"acc_{test_accuracy:.4f}".replace(".", "p")
        else:
            accuracy_str = "acc_unknown"
        
        # Generate filename with .keras extension (TensorFlow's recommended format)
        if run_name:
            # Use run_name as base for consistency
            filename = f"{run_name}_{accuracy_str}_model.keras"
        else:
            # Create the run_name based on the timestamp + dataset used + optimization mode
            run_name = f"{run_timestamp}_{dataset_name_clean}_{accuracy_str}"
            
            # Fallback to timestamp-based naming
            filename = f"{run_name}_model.keras"
        
        logger.debug(f"running _generate_optimized_filename ... run_name is: {run_name}")
        logger.debug(f"running _generate_optimized_filename ... generated filename: {filename}")
        return filename
    
    
    def _determine_save_directory(self, run_name: Optional[str] = None) -> Path:
        """
        Determine the appropriate directory for saving models
        
        Args:
            run_name: Optional run name for directory selection
            
        Returns:
            Path object for the save directory
        """
        project_root = Path(__file__).parent.parent.parent  # Go up 3 levels to project root
        
        # Check if we're in RunPod environment - use optimization_results structure
        if os.getenv('RUNPOD_ENDPOINT_ID'):
            # Use /app/optimization_results structure for RunPod to match local behavior
            if run_name:
                save_dir = Path("/app/optimization_results") / run_name / "models"
            else:
                save_dir = Path("/app/optimization_results") / "default_run" / "models"
        else:
            # Local execution - use legacy paths
            if run_name and run_name.startswith("optimized_"):
                # For optimized runs, save to optimized_model directory
                save_dir = project_root / "optimized_model"
            else:
                # For regular runs, save to trained_models directory
                save_dir = project_root / "trained_models"
        
        # Create directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"running _determine_save_directory ... Using save directory: {save_dir}")
        return save_dir
    
    
    def _save_model_metadata(
        self, 
        save_dir: Path, 
        filename: str, 
        test_accuracy: Optional[float], 
        run_timestamp: str, 
        run_name: Optional[str]
    ) -> None:
        """
        Save model metadata to accompanying JSON file
        """
        metadata = {
            'filename': filename,
            'timestamp': run_timestamp,
            'run_name': run_name,
            'test_accuracy': test_accuracy,
            'dataset_name': self.dataset_config.name,
            'model_config': {
                'epochs': self.model_config.epochs,
                'architecture_type': self._detect_data_type_enhanced(),
                'num_classes': self.dataset_config.num_classes,
                'input_shape': list(self.dataset_config.input_shape)
            }
        }
        
        metadata_file = save_dir / f"{filename.replace('.tf', '').replace('.keras', '')}_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"running _save_model_metadata ... Metadata saved to {metadata_file}")
    
    
    def save_model(
        self, 
        test_accuracy: Optional[float] = None,
        run_timestamp: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> str:
        """
        Optimized model saving with enhanced metadata and compression
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train a model first.")
        
        logger.debug("running save_model ... Starting optimized model saving...")
        
        # Create timestamp if not provided
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        
        filename = self._generate_optimized_filename(
            run_timestamp=run_timestamp, 
            test_accuracy=test_accuracy, 
            run_name=run_name
            )

        # Determine save directory
        save_dir = self._determine_save_directory(run_name)
        
        # Create final filepath
        final_filepath = save_dir / filename
        
        # Save model with optimization
        logger.debug(f"running save_model ... Saving optimized model to {final_filepath}")
        
        try:
            # Save with optimized settings
            self.model.save(final_filepath)
            
            # Save additional metadata
            self._save_model_metadata(save_dir, filename, test_accuracy, run_timestamp, run_name)
            
            logger.debug(f"running save_model ... Model saved successfully")
            # Inline model save summary logging
            logger.debug(f"running save_model ... Model save summary:")
            logger.debug(f"running save_model ... - Path: {final_filepath}")
            if test_accuracy is not None:
                logger.debug(f"running save_model ... - Accuracy: {test_accuracy:.4f}")
            else:
                logger.debug(f"running save_model ... - Accuracy: Unknown")
            logger.debug(f"running save_model ... - Run name: {run_name or 'None'}")
            
            # Log file size if available
            if final_filepath.exists():
                file_size_mb = final_filepath.stat().st_size / (1024*1024)
                logger.debug(f"running save_model ... - File size: {file_size_mb:.1f} MB")
            else:
                logger.debug(f"running save_model ... - File size: Unknown")
            
            # Upload to S3 if running on RunPod
            if os.getenv('RUNPOD_ENDPOINT_ID'):
                logger.debug(f"running save_model ... Uploading model to S3 (RunPod environment detected)")
                try:
                    from utils.s3_transfer import upload_to_runpod_s3
                    from pathlib import Path
                    
                    # Upload the model directory to S3 using same structure as local
                    # Extract the relative path from optimization_results onward  
                    # Handle both RunPod container paths (/app/...) and local test paths
                    save_dir_str = str(save_dir)
                    if "optimization_results" in save_dir_str:
                        # Find the optimization_results part and extract relative path from there
                        opt_results_index = save_dir_str.find("optimization_results")
                        relative_part = save_dir_str[opt_results_index + len("optimization_results"):].lstrip("/")
                        s3_prefix = f"optimization_results/{relative_part}" if relative_part else "optimization_results"
                    else:
                        # Fallback: use the directory name structure
                        s3_prefix = f"optimization_results/{save_dir.name}"
                    s3_result = upload_to_runpod_s3(
                        local_dir=str(save_dir),
                        s3_prefix=s3_prefix
                    )
                    
                    if s3_result:
                        logger.info(f"✅ Model uploaded to S3: s3://40ub9vhaa7/{s3_prefix}")
                        # Store S3 info for later retrieval
                        if not hasattr(self, '_s3_uploads'):
                            self._s3_uploads = {}
                        self._s3_uploads['model'] = s3_result
                    else:
                        logger.warning(f"⚠️ Failed to upload model to S3")
                        
                except Exception as e:
                    logger.error(f"Failed to upload model to S3: {e}")
            
            return str(final_filepath)
            
        except Exception as e:
            logger.error(f"running save_model ... Failed to save model: {e}")
            # Fallback to standard .keras format
            keras_filepath = save_dir / filename.replace('.tf', '.keras')
            logger.debug(f"running save_model ... Falling back to .keras format: {keras_filepath}")
            
            self.model.save(keras_filepath)
            final_filepath = keras_filepath
        
        return str(final_filepath)
    
    # [ALL OTHER METHODS REMAIN UNCHANGED - GPU PROXY, MODEL SAVING, LOGGING, ETC.]
    # ... (keeping all other methods but removing for brevity)
    
    def _log_model_summary(self) -> None:
        """Enhanced model summary logging with performance metrics"""
        if self.model is None:
            return
        
        # Get model summary
        summary_lines: List[str] = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        
        logger.debug("running _log_model_summary ... Enhanced Model Architecture:")
        for line in summary_lines:
            logger.debug(f"running _log_model_summary ... {line}")
        
        # Enhanced layer explanations with performance insights
        self._log_enhanced_layer_explanations()
        
        # Performance metrics
        total_params = self.model.count_params()
        trainable_params = sum([keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        logger.debug(f"running _log_model_summary ... Enhanced Model Metrics:")
        logger.debug(f"running _log_model_summary ... - Total parameters: {total_params:,}")
        logger.debug(f"running _log_model_summary ... - Trainable parameters: {trainable_params:,}")
        logger.debug(f"running _log_model_summary ... - Non-trainable parameters: {non_trainable_params:,}")
        
        # Memory estimation
        estimated_memory_mb = self._estimate_model_memory()
        logger.debug(f"running _log_model_summary ... - Estimated memory usage: {estimated_memory_mb:.1f} MB")
    
    def _log_enhanced_layer_explanations(self) -> None:
        """Enhanced layer explanations with performance insights"""
        
        logger.debug("running _log_enhanced_layer_explanations ... Enhanced Layer Analysis:")
        
        if self.model is None:
            return
        for i, layer in enumerate(self.model.layers):
            layer_type = type(layer).__name__
            layer_info = self._analyze_layer_performance(layer, i)
            
            logger.debug(f"running _log_enhanced_layer_explanations ... Layer {i}: {layer.name} ({layer_type})")
            logger.debug(f"running _log_enhanced_layer_explanations ... - {layer_info}")
    
    def _analyze_layer_performance(self, layer: keras.layers.Layer, layer_index: int) -> str:
        """Analyze individual layer performance characteristics"""
        
        layer_type = type(layer).__name__
        
        try:
            output_shape = layer.output.shape if hasattr(layer, 'output') else "unknown"
            param_count = layer.count_params() if hasattr(layer, 'count_params') else 0
            
            if layer_type == "Conv2D":
                filters = getattr(layer, 'filters', 'unknown')
                kernel_size = getattr(layer, 'kernel_size', 'unknown')
                return f"Feature extraction: {filters} filters, kernel {kernel_size}, {param_count:,} params, output {output_shape}"
            
            elif layer_type == "Dense":
                units = getattr(layer, 'units', 'unknown')
                if self.model is not None and layer_index == len(self.model.layers) - 1:
                    return f"Output classification: {units} classes, {param_count:,} params"
                else:
                    return f"Feature combination: {units} neurons, {param_count:,} params"
            
            elif layer_type == "LSTM":
                units = getattr(layer, 'units', 'unknown')
                return f"Sequence processing: {units} memory cells, {param_count:,} params"
            
            elif layer_type == "Embedding":
                input_dim = getattr(layer, 'input_dim', 'unknown')
                output_dim = getattr(layer, 'output_dim', 'unknown')
                return f"Word embeddings: {input_dim} vocab → {output_dim}D vectors, {param_count:,} params"
            
            else:
                return f"Shape: {output_shape}, {param_count:,} params"
                
        except Exception:
            return f"Layer analysis unavailable"
    
    def _estimate_model_memory(self) -> float:
        """Estimate model memory usage in MB"""
        try:
            # Rough estimation based on parameters and layer types
            if self.model is None:
                return 0.0
            total_params = self.model.count_params()
            
            # Base memory for parameters (4 bytes per float32 parameter)
            param_memory_mb = (total_params * 4) / (1024 * 1024)
            
            # Additional memory for activations (rough estimate)
            activation_memory_mb = param_memory_mb * 0.5  # Conservative estimate
            
            # Overhead for optimization states, gradients, etc.
            overhead_memory_mb = param_memory_mb * 0.3
            
            total_memory_mb = param_memory_mb + activation_memory_mb + overhead_memory_mb
            
            return total_memory_mb
            
        except Exception:
            return 0.0


# Enhanced convenience function WITHOUT embedded plot generation
def create_and_train_model(
    data: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    model_config: Optional[ModelConfig] = None,
    load_model_path: Optional[str] = None,
    test_size: float = 0.4,
    run_name: Optional[str] = None,
    enable_performance_monitoring: bool = True,
    use_multi_gpu: bool = False,
    progress_callback: Optional[Callable] = None,
    **config_overrides
) -> Dict[str, Any]:
    """
    Enhanced convenience function focused on model training
    
    Plot generation has been moved to separate PlotGenerator module.
    This function now focuses purely on model building, training, and evaluation.
    """
    
    # Enhanced timestamp generation
    if run_name:
        run_timestamp = run_name.split('_')[0]
        logger.debug(f"running create_and_train_model ... Using timestamp from run_name: {run_timestamp}")
    else:
        run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        logger.debug(f"running create_and_train_model ... Generated new timestamp: {run_timestamp}")
    
    # Enhanced validation
    if data is None and dataset_name is None:
        raise ValueError("Must provide either 'data' or 'dataset_name'")
    
    if data is not None and dataset_name is not None:
        raise ValueError("Provide either 'data' OR 'dataset_name', not both")
    
    # Enhanced data loading with performance monitoring
    if dataset_name is not None:
        logger.debug(f"running create_and_train_model ... Loading dataset with enhanced monitoring: {dataset_name}")
        
        with TimedOperation(f"dataset loading: {dataset_name}", "enhanced_model_builder"):
            manager = DatasetManager()
            
            if dataset_name not in manager.get_available_datasets():
                available = ', '.join(manager.get_available_datasets())
                raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")
            
            data = manager.load_dataset(dataset_name, test_size=test_size)
            logger.debug(f"running create_and_train_model ... Dataset {dataset_name} loaded successfully")
    
    assert data is not None
    
    # Enhanced configuration setup
    dataset_config = data['config']
    
    # Determine architecture type
    if (dataset_config.img_height == 1 and 
        dataset_config.channels == 1 and 
        dataset_config.img_width > 100):
        architecture_type = "LSTM"
    else:
        architecture_type = "CNN"
    
    dataset_name_clean = dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    
    # Handle model loading vs training
    if load_model_path:
        return _handle_model_loading_refactored(
            load_model_path, dataset_config, data, 
            run_timestamp, architecture_type, dataset_name_clean
        )
    else:
        return _handle_model_training_refactored(
            dataset_config, data, model_config, config_overrides,
            run_timestamp, run_name, 
            architecture_type, dataset_name_clean, enable_performance_monitoring,
            use_multi_gpu, progress_callback
        )


def _handle_model_loading_refactored(
    load_model_path: str,
    dataset_config: DatasetConfig,
    data: Dict[str, Any],
    run_timestamp: str,
    architecture_type: str,
    dataset_name_clean: str
) -> Dict[str, Any]:
    """Model loading handler without plot generation"""
    
    logger.debug(f"running _handle_model_loading_refactored ... Loading existing model from: {load_model_path}")
    
    model_file = Path(load_model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {load_model_path}")
    
    # Create ModelBuilder with timestamp for TensorBoard logging
    builder = ModelBuilder(dataset_config, None, None, run_timestamp)
    
    with TimedOperation("model loading", "refactored_model_builder"):
        builder.model = keras.models.load_model(load_model_path)
        logger.debug("running _handle_model_loading_refactored ... Model loaded successfully!")
    
    # Simple evaluation without plots
    logger.debug("running _handle_model_loading_refactored ... Starting simple evaluation...")
    
    with TimedOperation("model evaluation", "refactored_model_builder"):
        test_loss, test_accuracy = builder.evaluate(data=data)
    
    logger.debug(f"running _handle_model_loading_refactored ... Evaluation completed:")
    logger.debug(f"running _handle_model_loading_refactored ... - Test accuracy: {test_accuracy:.4f}")
    logger.debug(f"running _handle_model_loading_refactored ... - Test loss: {test_loss:.4f}")
    
    return {
        'model_builder': builder,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'model_path': None,
        'run_timestamp': run_timestamp,
        'architecture_type': architecture_type,
        'dataset_name': dataset_name_clean,
        'refactored': True  # Indicate this is the refactored version
    }


def _handle_model_training_refactored(
    dataset_config: DatasetConfig,
    data: Dict[str, Any],
    model_config: Optional[ModelConfig],
    config_overrides: Dict[str, Any],
    run_timestamp: str,
    run_name: Optional[str],
    architecture_type: str,
    dataset_name_clean: str,
    enable_performance_monitoring: bool,
    use_multi_gpu: bool = False,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Model training handler without embedded plot generation"""
    
    logger.debug("running _handle_model_training_refactored ... Starting refactored model training")
    
    # Enhanced model configuration
    if model_config is None:
        model_config = ModelConfig()
    else:
        model_config = copy.deepcopy(model_config)
    
    # Apply enhanced config overrides
    if config_overrides:
        logger.debug(f"running _handle_model_training_refactored ... Applying config overrides: {config_overrides}")
        for key, value in config_overrides.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
                logger.debug(f"running _handle_model_training_refactored ... Override: {key} = {value}")
            else:
                logger.warning(f"running _handle_model_training_refactored ... Unknown config parameter: {key}")
    
    # Create ModelBuilder with timestamp for TensorBoard logging
    builder = ModelBuilder(dataset_config, model_config, None, run_timestamp)
    
    # Training pipeline without plot generation
    with TimedOperation("refactored model training pipeline", "refactored_model_builder"):
        # Don't build model here if using multi-GPU
        if not use_multi_gpu:
            logger.debug("running _handle_model_training_refactored ... Building model for single-GPU...")
            builder.build_model()
        else:
            logger.debug("running _handle_model_training_refactored ... Skipping model building - will be done inside MirroredStrategy scope")
        
        logger.debug("running _handle_model_training_refactored ... Training model without plot generation...")
        builder.train(data, use_multi_gpu=use_multi_gpu, epoch_progress_callback=progress_callback, create_plots=False)
        
        logger.debug("running _handle_model_training_refactored ... Evaluating model...")
        test_loss, test_accuracy = builder.evaluate(data=data)
        
        logger.debug("running _handle_model_training_refactored ... Saving model...")
        model_path = builder.save_model(
            test_accuracy=test_accuracy,
            run_timestamp=run_timestamp,
            run_name=run_name
        )
    
    logger.debug(f"running _handle_model_training_refactored ... Training completed:")
    logger.debug(f"running _handle_model_training_refactored ... - Accuracy: {test_accuracy:.4f}")
    logger.debug(f"running _handle_model_training_refactored ... - Model saved: {model_path}")
    
    return {
        'model_builder': builder,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'model_path': model_path,
        'run_timestamp': run_timestamp,
        'architecture_type': architecture_type,
        'dataset_name': dataset_name_clean,
        'refactored': True,  # Indicate this is the refactored version
        'gpu_proxy_used': builder.gpu_proxy_available and model_config.use_gpu_proxy,
        'performance_monitoring': enable_performance_monitoring
    }


if __name__ == "__main__":
    logger.debug("running refactored_model_builder.py ... Testing Refactored ModelBuilder...")
    
    # Same command-line interface as before, but without plot-related parameters
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    
    # Extract dataset name (required)
    dataset_name = args.get('dataset_name', 'cifar10')
    
    # REMOVED: plot-related parameter parsing
    # All plot generation now handled by separate PlotGenerator module
    
    logger.debug(f"running refactored_model_builder.py ... Parsed arguments: {args}")
    
    try:
        # Function call
        result = create_and_train_model(**args)
        builder = result['model_builder']
        test_accuracy = result['test_accuracy']
        
        # Success logging
        load_path = args.get('load_model_path')
        workflow_msg = f"loaded existing model from {load_path}" if load_path else "trained new refactored model"
        
        logger.debug(f"running refactored_model_builder.py ... ✅ REFACTORED SUCCESS!")
        logger.debug(f"running refactored_model_builder.py ... Successfully {workflow_msg}")
        logger.debug(f"running refactored_model_builder.py ... Final accuracy: {test_accuracy:.4f}")
        logger.debug(f"running refactored_model_builder.py ... Refactored version: {result.get('refactored', False)}")
        logger.debug(f"running refactored_model_builder.py ... GPU proxy used: {result.get('gpu_proxy_used', False)}")
        
        logger.debug("running refactored_model_builder.py ... NOTE: Plot generation now handled by separate PlotGenerator module")
        
    except Exception as e:
        logger.error(f"running refactored_model_builder.py ... ❌ REFACTORED ERROR: {e}")
        logger.error(f"running refactored_model_builder.py ... Traceback: {traceback.format_exc()}")
        sys.exit(1)