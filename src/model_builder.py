"""
Model Builder for Multi-Modal Classification - OPTIMIZED VERSION

Creates and trains neural networks for both image classification (CNN) and 
text classification (LSTM). Automatically detects data type and builds 
appropriate architecture. Designed to work with any dataset configuration 
from DatasetManager.

Key Optimizations:
- Enhanced GPU proxy integration with intelligent sampling
- Improved memory management and data type optimization
- Better error handling and fallback mechanisms
- Streamlined code structure and reduced redundancy
- Enhanced logging and monitoring capabilities

Supported Architectures:
- CNN: For image data (GTSRB, CIFAR-10, MNIST, etc.)
- LSTM: For text data (IMDB, Reuters, etc.)
"""
import copy
from dataset_manager import DatasetConfig, DatasetManager
import datetime
from plot_creation.realtime_gradient_flow import RealTimeGradientFlowCallback, RealTimeGradientFlowMonitor
from plot_creation.realtime_training_visualization import RealTimeTrainingVisualizer, RealTimeTrainingCallback
from plot_creation.realtime_weights_bias import create_realtime_weights_bias_monitor, RealTimeWeightsBiasMonitor, RealTimeWeightsBiasCallback
from plot_creation.confusion_matrix import ConfusionMatrixAnalyzer
from plot_creation.training_history import TrainingHistoryAnalyzer
from plot_creation.training_animation import TrainingAnimationAnalyzer
from plot_creation.gradient_flow import GradientFlowAnalyzer
from plot_creation.weights_bias import WeightsBiasAnalyzer
from plot_creation.activation_map import ActivationMapAnalyzer

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
from typing import Dict, Any, List, Tuple, Optional, Union
from utils.logger import logger, PerformanceLogger, TimedOperation


from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union

# Move PaddingOption outside the class to avoid issues
class PaddingOption(Enum):
    SAME = "same"
    VALID = "valid"

@dataclass
class ModelConfig:
    """
    Optimized configuration for model architecture with enhanced GPU proxy support
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
    
    # Fix: Use the external PaddingOption and provide proper default
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
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    validation_split: float = 0.2
    
    # Evaluation and analysis parameters
    show_confusion_matrix: bool = True
    show_training_history: bool = True
    
    # Real-time visualization parameters
    enable_realtime_plots: bool = True
    save_realtime_plots: bool = True
    save_intermediate_plots: bool = True
    save_plot_every_n_epochs: int = 1
    
    # Gradient flow analysis parameters
    show_gradient_flow: bool = True
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
    show_weights_bias_analysis: bool = True
    enable_realtime_weights_bias: bool = True
    weights_bias_monitoring_frequency: int = 1
    weights_bias_sample_percentage: float = 0.1
    
    # Activation map analysis parameters
    show_activation_maps: bool = True
    activation_layer_frequency: int = 1
    activation_max_layers_to_analyze: int = 10
    activation_num_samples_per_class: int = 1
    activation_max_total_samples: int = 10
    activation_sample_selection_strategy: str = "mixed"
    activation_filters_per_row: int = 8
    activation_max_filters_per_layer: int = 32
    activation_dead_filter_threshold: float = 0.1
    activation_saturated_filter_threshold: float = 0.8
    activation_figsize_individual: Tuple[int, int] = (15, 10)
    activation_figsize_overview: Tuple[int, int] = (20, 12)
    activation_cmap: str = "viridis"
    activation_cmap_original: str = "gray"
    enable_realtime_activation_maps: bool = False
    activation_monitoring_frequency: int = 5
    activation_save_frequency: int = 10
    
    # ENHANCED GPU Proxy Integration parameters
    use_gpu_proxy: bool = False
    gpu_proxy_auto_clone: bool = True
    gpu_proxy_endpoint: Optional[str] = None
    gpu_proxy_fallback_local: bool = True
    
    # NEW: Enhanced GPU proxy sampling parameters
    gpu_proxy_sample_percentage: float = 0.50  # Use 1% of training data by default
    #gpu_proxy_min_samples_per_class: int = 50           # Minimum samples per class
    gpu_proxy_use_stratified_sampling: bool = True      # Use stratified sampling
    gpu_proxy_adaptive_batch_size: bool = True          # Adapt batch size to sample count
    gpu_proxy_optimize_data_types: bool = True          # Optimize data types for transfer
    gpu_proxy_compression_level: int = 6                # Compression level for large payloads
    
    def __post_init__(self) -> None:
        if not self.metrics:
            self.metrics = ["accuracy"]
        
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
    Optimized main class for building and training neural network models
    
    Key optimizations:
    - Enhanced GPU proxy integration with intelligent sampling
    - Improved memory management and error handling
    - Streamlined architecture detection and model building
    - Better logging and performance monitoring
    """
    
    def __init__(self, dataset_config: DatasetConfig, model_config: Optional[ModelConfig] = None) -> None:
        """
        Initialize ModelBuilder with enhanced configuration and GPU proxy setup
        """
        self.dataset_config: DatasetConfig = dataset_config
        self.model_config: ModelConfig = model_config or ModelConfig()
        self.model: Optional[keras.Model] = None
        self.training_history: Optional[keras.callbacks.History] = None
        self.plot_dir: Optional[Path] = None
        
        # Initialize performance logger
        self.perf_logger: PerformanceLogger = PerformanceLogger("model_builder")
        
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
        """Setup GPU proxy with retry logic and detailed error reporting"""
        if self._gpu_proxy_setup_attempted:
            return
            
        self._gpu_proxy_setup_attempted = True
        
        try:
            self.gpu_proxy_available, self.gpu_proxy_path, self.runpod_client = self._detect_and_setup_gpu_proxy()
            if self.gpu_proxy_available:
                logger.debug("running _setup_gpu_proxy_with_retry ... GPU proxy integration enabled")
            elif self.model_config.gpu_proxy_fallback_local:
                logger.debug("running _setup_gpu_proxy_with_retry ... GPU proxy unavailable, will use local execution")
            else:
                logger.warning("running _setup_gpu_proxy_with_retry ... GPU proxy unavailable and fallback disabled")
        except Exception as e:
            logger.error(f"running _setup_gpu_proxy_with_retry ... GPU proxy setup failed: {e}")
            if not self.model_config.gpu_proxy_fallback_local:
                raise
    
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
        
        layers = []
        
        # Input layer
        layers.append(keras.layers.Input(shape=self.dataset_config.input_shape))
        logger.debug(f"running _build_cnn_model_optimized ... Input shape: {self.dataset_config.input_shape}")
        
        # Convolutional layers with proper activation handling
        for i in range(self.model_config.num_layers_conv):
            logger.debug(f"running _build_cnn_model_optimized ... Building conv layer {i+1}/{self.model_config.num_layers_conv}")
            
            # FIXED: Separate activation handling to prevent graph construction issues
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
        self.model.compile(
            optimizer=optimizer,
            loss=self.model_config.loss,
            metrics=self.model_config.metrics
        )
        
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
        validation_split: Optional[float] = None
    ) -> keras.callbacks.History:
        """
        Enhanced training with optimized GPU proxy execution
        """
        if self.model is None:
            logger.debug("running train ... No model found, building model first...")
            self.build_model()
        
        assert self.model is not None
        logger.debug("running train ... Starting model training...")
        
        # Enhanced GPU proxy execution
        if self._should_use_gpu_proxy():
            logger.debug("running train ... Using GPU proxy for training execution")
            
            try:
                training_result = self._train_on_gpu_proxy_enhanced(data, validation_split)
                
                if training_result is not None:
                    logger.debug("running train ... GPU proxy training completed successfully")
                    return training_result
                else:
                    logger.warning("running train ... GPU proxy training failed")
                    
                    if self.model_config.gpu_proxy_fallback_local:
                        logger.warning("running train ... Falling back to local training")
                    else:
                        raise RuntimeError("GPU proxy training failed and local fallback disabled")
                        
            except Exception as gpu_error:
                logger.warning(f"running train ... GPU proxy training error: {gpu_error}")
                
                if self.model_config.gpu_proxy_fallback_local:
                    logger.warning("running train ... Falling back to local training due to GPU proxy error")
                else:
                    raise RuntimeError(f"GPU proxy training failed and local fallback disabled: {gpu_error}")
        
        # Execute local training
        logger.debug("running train ... Using local training execution")
        return self._train_locally_optimized(data, validation_split)
    
    def _should_use_gpu_proxy(self) -> bool:
        """Determine if GPU proxy should be used"""
        return (
            self.gpu_proxy_available and 
            self.runpod_client is not None and 
            self.model_config.use_gpu_proxy
        )
    
    def _train_on_gpu_proxy_enhanced(
        self, 
        data: Dict[str, Any], 
        validation_split: Optional[float] = None
    ) -> Optional[keras.callbacks.History]:
        """
        Enhanced GPU proxy training with comprehensive error handling
        """
        try:
            logger.debug("running _train_on_gpu_proxy_enhanced ... Starting enhanced GPU proxy training")
            
            # Generate optimized training code
            training_code = self._generate_gpu_proxy_training_code_enhanced(validation_split)
            logger.debug("running _train_on_gpu_proxy_enhanced ... Generated training code")
            
            # Prepare enhanced context data with intelligent sampling
            context_data = self._prepare_gpu_proxy_context_enhanced(data, validation_split)
            context_size_mb = len(str(context_data).encode()) / (1024 * 1024)
            logger.debug(f"running _train_on_gpu_proxy_enhanced ... Context size: {context_size_mb:.1f} MB")
            
            # Calculate timeout
            timeout_seconds = self._calculate_optimal_timeout(context_data)
            logger.debug(f"running _train_on_gpu_proxy_enhanced ... Executing on remote GPU (timeout: {timeout_seconds}s)")
            
            # Log pre-execution details
            logger.debug(f"running _train_on_gpu_proxy_enhanced ... Model params: {self.model.count_params() if hasattr(self, 'model') and self.model is not None else 'Unknown'}")
            logger.debug(f"running _train_on_gpu_proxy_enhanced ... Epochs: {self.model_config.epochs}")
            
            # Execute on GPU proxy with proper error handling
            if self.runpod_client is None:
                raise RuntimeError("GPU proxy client is not available")
            
            # Health check
            logger.debug("running _train_on_gpu_proxy_enhanced ... Testing worker health before execution")
            try:
                health_check = self.runpod_client.health()
                logger.debug(f"running _train_on_gpu_proxy_enhanced ... Worker health: {health_check}")
            except Exception as health_error:
                logger.error(f"running _train_on_gpu_proxy_enhanced ... Worker health check failed: {health_error}")
            
            result = self.runpod_client.execute_code_sync(
                code=training_code,
                context=context_data,
                timeout_seconds=timeout_seconds
            )
            
            # Enhanced result debugging and RunPod error detection
            logger.debug(f"running _train_on_gpu_proxy_enhanced ... Raw result type: {type(result)}")
            # logger.debug(f"running _train_on_gpu_proxy_enhanced ... Full result content: {result}")

            if isinstance(result, dict):
                logger.debug(f"running _train_on_gpu_proxy_enhanced ... Result keys: {list(result.keys())}")
                
                # Log all key-value pairs for debugging
                for key, value in result.items():
                    if key == 'execution_result':
                        logger.debug(f"running _train_on_gpu_proxy_enhanced ... - {key}: <execution result skipped>")
                    else:
                        logger.debug(f"running _train_on_gpu_proxy_enhanced ... - {key}: {value}")
                
                # Enhanced RunPod service issue detection
                if not result:  # Empty result
                    # Get fresh health check to see current status
                    try:
                        current_health = self.runpod_client.health()
                        failed_jobs = current_health.get('jobs', {}).get('failed', 0)
                        ready_workers = current_health.get('workers', {}).get('ready', 0)
                        in_progress = current_health.get('jobs', {}).get('inProgress', 0)
                        
                        logger.error(f"running _train_on_gpu_proxy_enhanced ... RunPod service issues detected:")
                        logger.error(f"running _train_on_gpu_proxy_enhanced ... - Failed jobs: {failed_jobs}")
                        logger.error(f"running _train_on_gpu_proxy_enhanced ... - Ready workers: {ready_workers}")
                        logger.error(f"running _train_on_gpu_proxy_enhanced ... - Jobs in progress: {in_progress}")
                        
                        if failed_jobs > 5:
                            raise Exception(f"RunPod endpoint unstable - {failed_jobs} failed jobs detected")
                        if ready_workers == 0:
                            raise Exception("No RunPod workers available - all busy or occupied")
                        else:
                            raise Exception("RunPod job completed but returned empty result - possible service issue")
                            
                    except Exception as health_error:
                        logger.error(f"running _train_on_gpu_proxy_enhanced ... Could not check RunPod health: {health_error}")
                        raise Exception("RunPod worker returned empty result and health check failed")
                
                # Check for RunPod-specific error indicators
                if 'error' in result:
                    logger.error(f"running _train_on_gpu_proxy_enhanced ... RunPod error in result: {result['error']}")
                    raise Exception(f"RunPod execution error: {result['error']}")
                
                if 'output' in result and result['output'] is None:
                    logger.error("running _train_on_gpu_proxy_enhanced ... RunPod output is None - execution may have failed")
                    logger.error(f"running _train_on_gpu_proxy_enhanced ... Full result: {result}")
                    raise Exception("RunPod execution returned no output")
                
                if 'status' in result:
                    logger.debug(f"running _train_on_gpu_proxy_enhanced ... RunPod status: {result['status']}")
            
            logger.debug("running _train_on_gpu_proxy_enhanced ... GPU proxy training completed successfully")
            
            # Convert results using existing method
            if result and result.get('execution_result', {}).get('success', False):
                execution_result = result['execution_result']['result']
                history = self._convert_gpu_results_to_history_enhanced(execution_result)
                self.training_history = history
                return history
            else:
                # Use the enhanced error logging but maintain existing flow
                self._log_gpu_proxy_error(Exception("GPU proxy execution failed - see previous logs"), "training execution")
                return None
            
        except Exception as e:
            # Enhanced error logging with context
            self._log_gpu_proxy_error(e, "training execution")
            
            # Log additional context about the failure
            logger.error("running _train_on_gpu_proxy_enhanced ... Training context when error occurred:")
            if hasattr(self, 'model') and self.model is not None:
                logger.error(f"running _train_on_gpu_proxy_enhanced ... - Model parameters: {self.model.count_params()}")
            else:
                logger.error(f"running _train_on_gpu_proxy_enhanced ... - Model parameters: Not available (model is None)")
            logger.error(f"running _train_on_gpu_proxy_enhanced ... - Training epochs: {self.model_config.epochs}")
            if hasattr(self, 'runpod_client') and self.runpod_client is not None:
                logger.error(f"running _train_on_gpu_proxy_enhanced ... - GPU proxy endpoint: {getattr(self.runpod_client, 'endpoint_id', 'Unknown')}")
            
            # Re-raise for fallback handling (maintaining existing behavior)
            return None
    
    def _prepare_gpu_proxy_context_enhanced(
        self, 
        data: Dict[str, Any], 
        validation_split: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Enhanced context preparation with intelligent stratified sampling
        """
        logger.debug("running _prepare_gpu_proxy_context_enhanced ... Preparing enhanced context with stratified sampling")
        
        x_train = data['x_train']
        y_train = data['y_train']
        
        # Calculate optimal sample sizes
        sample_config = self._calculate_optimal_sampling(x_train, y_train)
        
        # Perform stratified sampling
        x_sampled, y_sampled = self._perform_stratified_sampling(
            x_train, y_train, sample_config
        )
        
        # Optimize data types for transfer
        x_optimized, y_optimized = self._optimize_data_types_for_transfer(x_sampled, y_sampled)
        
        # Create enhanced context
        context = {
            'x_train': x_optimized.tolist(),
            'y_train': y_optimized.tolist(),
            'config': {
                'input_shape': self.dataset_config.input_shape,
                'num_classes': self.dataset_config.num_classes,
                'epochs': self.model_config.epochs,
                'batch_size': self._calculate_optimal_batch_size(len(x_optimized)),
                'validation_split': validation_split or self.model_config.validation_split,
                'original_dataset_size': len(x_train),
                'sampled_size': len(x_optimized),
                'architecture_type': self._detect_data_type_enhanced()
            },
            'sampling_info': sample_config,
            'model_config': self._extract_essential_model_config()
        }
        
        # Log context information
        size_mb = self._calculate_context_size(context)
        logger.debug(f"running _prepare_gpu_proxy_context_enhanced ... Context size: {size_mb:.1f} MB")
        logger.debug(f"running _prepare_gpu_proxy_context_enhanced ... Sample reduction: {len(x_train)} → {len(x_optimized)}")
        
        return context
    
    def _calculate_optimal_sampling(self, x_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Calculate optimal sampling parameters based on percentage of total dataset"""
        
        # Convert one-hot to class indices if needed
        if y_train.ndim > 1 and y_train.shape[1] > 1:
            class_indices = np.argmax(y_train, axis=1)
        else:
            class_indices = y_train.flatten()
        
        num_classes = self.dataset_config.num_classes
        total_samples = len(x_train)
        
        # Calculate target samples based on percentage
        target_total_samples = int(total_samples * self.model_config.gpu_proxy_sample_percentage)
        target_total_samples = max(num_classes, target_total_samples)  # At least 1 sample per class
        
        # Distribute evenly across classes for stratified sampling
        samples_per_class = target_total_samples // num_classes
        actual_total_samples = samples_per_class * num_classes
        reduction_ratio = actual_total_samples / total_samples
        
        # Calculate class distribution for logging
        class_counts = np.bincount(class_indices, minlength=num_classes)
        min_class_count = np.min(class_counts[class_counts > 0])
        
        # Warn if we're asking for more samples than available in smallest class
        if samples_per_class > min_class_count:
            logger.warning(f"running _calculate_optimal_sampling ... Requested {samples_per_class} samples per class, but smallest class only has {min_class_count}")
            samples_per_class = min_class_count
            actual_total_samples = samples_per_class * num_classes
            reduction_ratio = actual_total_samples / total_samples
        
        sample_config = {
            'strategy': 'percentage_stratified',
            'total_classes': num_classes,
            'samples_per_class': samples_per_class,
            'total_target_samples': actual_total_samples,
            'reduction_ratio': reduction_ratio,
            'percentage_requested': self.model_config.gpu_proxy_sample_percentage,
            'percentage_actual': reduction_ratio,
            'class_distribution': class_counts.tolist(),
            'min_class_count': int(min_class_count)
        }
        
        logger.debug(f"running _calculate_optimal_sampling ... Percentage stratified sampling: {self.model_config.gpu_proxy_sample_percentage:.1%} requested")
        logger.debug(f"running _calculate_optimal_sampling ... Target: {samples_per_class} samples × {num_classes} classes = {actual_total_samples} total ({reduction_ratio:.1%} of dataset)")
        
        return sample_config
    
    def _perform_stratified_sampling(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        sample_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform stratified sampling to maintain class distribution"""
        
        if not self.model_config.gpu_proxy_use_stratified_sampling:
            # Simple random sampling fallback
            n_samples = min(sample_config['total_target_samples'], len(x_train))
            indices = np.random.choice(len(x_train), n_samples, replace=False)
            return x_train[indices], y_train[indices]
        
        # Convert one-hot to class indices if needed
        if y_train.ndim > 1 and y_train.shape[1] > 1:
            class_indices = np.argmax(y_train, axis=1)
        else:
            class_indices = y_train.flatten()
        
        selected_indices = []
        samples_per_class = sample_config['samples_per_class']
        num_classes = sample_config['total_classes']
        
        # Sample from each class
        for class_id in range(num_classes):
            class_mask = class_indices == class_id
            class_indices_list = np.where(class_mask)[0]
            
            if len(class_indices_list) > 0:
                n_samples = min(samples_per_class, len(class_indices_list))
                if n_samples > 0:
                    sampled_indices = np.random.choice(class_indices_list, n_samples, replace=False)
                    selected_indices.extend(sampled_indices)
                    logger.debug(f"running _perform_stratified_sampling ... Class {class_id}: selected {n_samples}/{len(class_indices_list)} samples")
        
        # Shuffle and return
        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)
        
        return x_train[selected_indices], y_train[selected_indices]
    
    def _optimize_data_types_for_transfer(
        self, 
        x_data: np.ndarray, 
        y_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize data types to reduce transfer size"""
        
        if not self.model_config.gpu_proxy_optimize_data_types:
            return x_data, y_data
        
        x_optimized = x_data.copy()
        y_optimized = y_data.copy()
        
        # Optimize x_data (features)
        if x_optimized.dtype == np.float64:
            x_optimized = x_optimized.astype(np.float32)
            logger.debug("running _optimize_data_types_for_transfer ... Converted features from float64 to float32")
        
        # For image data in [0,1] range, convert to uint8
        if (len(x_optimized.shape) > 2 and 
            x_optimized.max() <= 1.0 and 
            x_optimized.min() >= 0.0):
            x_optimized = (x_optimized * 255).astype(np.uint8)
            logger.debug("running _optimize_data_types_for_transfer ... Converted image data to uint8 format")
        
        # Optimize y_data (labels)
        if y_optimized.dtype == np.float64:
            y_optimized = y_optimized.astype(np.float32)
            logger.debug("running _optimize_data_types_for_transfer ... Converted labels from float64 to float32")
        
        return x_optimized, y_optimized
    
    def _calculate_optimal_batch_size(self, sample_count: int) -> int:
        """Calculate optimal batch size based on sample count"""
        if not self.model_config.gpu_proxy_adaptive_batch_size:
            return 32  # Default batch size
        
        # Adaptive batch size calculation
        if sample_count < 100:
            return max(4, sample_count // 4)
        elif sample_count < 500:
            return 16
        elif sample_count < 1000:
            return 32
        else:
            return 64
    
    def _extract_essential_model_config(self) -> Dict[str, Any]:
        """Extract essential model configuration for GPU proxy"""
        return {
            'architecture_type': self.model_config.architecture_type,
            'num_layers_conv': self.model_config.num_layers_conv,
            'filters_per_conv_layer': self.model_config.filters_per_conv_layer,
            'kernel_size': self.model_config.kernel_size,
            'activation': self.model_config.activation,
            'use_global_pooling': self.model_config.use_global_pooling,
            'num_layers_hidden': self.model_config.num_layers_hidden,
            'first_hidden_layer_nodes': self.model_config.first_hidden_layer_nodes,
            'first_hidden_layer_dropout': self.model_config.first_hidden_layer_dropout,
            'optimizer': self.model_config.optimizer,
            'loss': self.model_config.loss,
            'metrics': self.model_config.metrics
        }
    
    def _calculate_context_size(self, context: Dict[str, Any]) -> float:
        """Calculate approximate context size in MB"""
        try:
            context_json = json.dumps(context)
            size_bytes = len(context_json.encode('utf-8'))
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _calculate_optimal_timeout(self, context_data: Dict[str, Any]) -> int:
        """Calculate optimal timeout based on data size and model complexity"""
        base_timeout = 300  # 5 minutes base
        
        # Add time based on sample count
        sample_count = context_data.get('config', {}).get('sampled_size', 1000)
        epochs = context_data.get('config', {}).get('epochs', 10)
        
        # Estimate additional time needed
        additional_time = (sample_count // 100) * epochs * 2  # 2 seconds per 100 samples per epoch
        
        return min(base_timeout + additional_time, 1800)  # Cap at 30 minutes
    
    def _log_gpu_proxy_error(self, error: Exception, context: str = "") -> None:
        """Enhanced GPU proxy error logging with full error details."""
        logger.debug("running _log_gpu_proxy_error ... Capturing detailed GPU proxy error")
        
        # Log the full error details instead of just "Unknown error"
        error_type = type(error).__name__
        error_message = str(error)
        
        logger.error(f"running _log_gpu_proxy_error ... GPU proxy {context} failed:")
        logger.error(f"running _log_gpu_proxy_error ... - Error type: {error_type}")
        logger.error(f"running _log_gpu_proxy_error ... - Error message: {error_message}")
        
        # If it's a requests error, get HTTP details
        if hasattr(error, 'response'):
            try:
                response = getattr(error, 'response', None)
                if response is not None:
                    status_code = getattr(response, 'status_code', 'Unknown')
                    response_text = getattr(response, 'text', 'No response text')[:500]
                    logger.error(f"running _log_gpu_proxy_error ... - HTTP status: {status_code}")
                    logger.error(f"running _log_gpu_proxy_error ... - Response text: {response_text}")
            except Exception as e:
                logger.error(f"running _log_gpu_proxy_error ... - Could not extract response details: {e}")
        
        # Log any exception chaining for root cause
        if error.__cause__:
            logger.error(f"running _log_gpu_proxy_error ... - Caused by: {type(error.__cause__).__name__}: {error.__cause__}")
        if error.__context__:
            logger.error(f"running _log_gpu_proxy_error ... - Context: {type(error.__context__).__name__}: {error.__context__}")
    
    def _generate_gpu_proxy_training_code_enhanced(self, validation_split: Optional[float] = None) -> str:
        """
        Generate enhanced training code for GPU proxy with better architecture support
        """
        logger.debug("running _generate_gpu_proxy_training_code_enhanced ... Generating enhanced GPU proxy code")
        
        val_split = validation_split or self.model_config.validation_split
        data_type = self._detect_data_type_enhanced()
        
        if data_type == "text":
            return self._generate_text_model_gpu_code(val_split)
        else:
            return self._generate_cnn_model_gpu_code(val_split)
    
    def _generate_cnn_model_gpu_code(self, val_split: float) -> str:
        """Generate optimized CNN model code for GPU proxy with enhanced CUDA verification"""
        
        return f"""
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

print("=== ENHANCED GPU AVAILABILITY CHECK ===")
print(f"CUDA_HOME: {{os.environ.get('CUDA_HOME', 'Not set')}}")  
print(f"NVIDIA_VISIBLE_DEVICES: {{os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')}}")

print("Starting enhanced CNN training on GPU proxy...")
print(f"TensorFlow version: {{tf.__version__}}")
print(f"CUDA built with TF: {{tf.test.is_built_with_cuda()}}")
print(f"GPU devices available: {{tf.config.list_physical_devices('GPU')}}")

# Critical GPU verification - fail fast if no GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {{len(gpus)}} devices")
        print(f"GPU details: {{[gpu.name for gpu in gpus]}}")
    except RuntimeError as e:
        print(f"GPU memory growth setup failed: {{e}}")
else:
    print("CRITICAL ERROR: No GPU devices detected!")
    print("This indicates CUDA libraries are not properly installed.")
    raise RuntimeError("No GPU devices available - CUDA libraries missing or corrupted")

print("=== GPU CHECK COMPLETE - PROCEEDING WITH TRAINING ===")

# Get data and config from context
x_train_raw = np.array(context['x_train'])
y_train = np.array(context['y_train'])
config = context['config']
model_config = context['model_config']
sampling_info = context['sampling_info']

print(f"Training data shape: {{x_train_raw.shape}}")
print(f"Training labels shape: {{y_train.shape}}")
print(f"Sampling strategy: {{sampling_info['strategy']}}")
print(f"Reduction ratio: {{sampling_info['reduction_ratio']:.2%}}")

# Enhanced data preprocessing
if len(x_train_raw.shape) > 2 and x_train_raw.dtype == np.uint8:
    x_train = x_train_raw.astype(np.float32) / 255.0
    print("Converted uint8 to normalized float32")
elif x_train_raw.max() > 1.0:
    x_train = x_train_raw.astype(np.float32) / 255.0
    print("Normalized image data to 0-1 range")
else:
    x_train = x_train_raw.astype(np.float32)
    print("Using data as-is")

# Build enhanced CNN model with GPU placement verification
print("Building enhanced CNN model on GPU...")
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Input(shape=config['input_shape']))

    # Convolutional layers
    num_conv_layers = model_config.get('num_layers_conv', 2)
    filters = model_config.get('filters_per_conv_layer', 32)
    kernel_size = model_config.get('kernel_size', [3, 3])
    activation = model_config.get('activation', 'relu')

    for i in range(num_conv_layers):
        model.add(keras.layers.Conv2D(
            filters=filters,
            kernel_size=tuple(kernel_size),
            activation=activation,
            padding='same'
        ))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        print(f"Added Conv2D layer {{i+1}}: {{filters}} filters")

    # Pooling strategy
    if model_config.get('use_global_pooling', False):
        model.add(keras.layers.GlobalAveragePooling2D())
        print("Using GlobalAveragePooling2D")
    else:
        model.add(keras.layers.Flatten())
        print("Using Flatten")

    # Hidden layers
    num_hidden = model_config.get('num_layers_hidden', 1)
    hidden_nodes = model_config.get('first_hidden_layer_nodes', 128)
    dropout_rate = model_config.get('first_hidden_layer_dropout', 0.5)

    for i in range(num_hidden):
        model.add(keras.layers.Dense(hidden_nodes, activation=activation))
        model.add(keras.layers.Dropout(dropout_rate))
        print(f"Added Dense layer {{i+1}}: {{hidden_nodes}} nodes, {{dropout_rate}} dropout")

    # Output layer
    model.add(keras.layers.Dense(config['num_classes'], activation='softmax'))

    # Compile model
    optimizer = model_config.get('optimizer', 'adam')
    loss = model_config.get('loss', 'categorical_crossentropy')
    metrics = model_config.get('metrics', ['accuracy'])

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print(f"Enhanced CNN model created with {{model.count_params():,}} parameters")
print(f"Model device placement: GPU" if gpus else "Model device placement: CPU (ERROR!)")

# Train the model
epochs = config.get('epochs', 10)
batch_size = config.get('batch_size', 32)

print(f"Training for {{epochs}} epochs with batch size {{batch_size}} on {{'GPU' if gpus else 'CPU'}}...")

try:
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split={val_split},
        batch_size=batch_size,
        verbose=1
    )
    
    print("Enhanced CNN training completed successfully!")
    
    # Extract and return results
    final_loss = float(history.history['loss'][-1])
    final_accuracy = float(history.history.get('accuracy', [0])[-1])
    final_val_loss = float(history.history.get('val_loss', [999])[-1])
    final_val_accuracy = float(history.history.get('val_accuracy', [0])[-1])
    
    print(f"Final training accuracy: {{final_accuracy:.4f}}")
    print(f"Final validation accuracy: {{final_val_accuracy:.4f}}")
    
    result = {{
        'training_history': {{k: [float(v) for v in values] for k, values in history.history.items()}},
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy,
        'epochs_completed': len(history.history['loss']),
        'model_params': int(model.count_params()),
        'gpu_used': len(tf.config.list_physical_devices('GPU')) > 0,
        'gpu_count': len(tf.config.list_physical_devices('GPU')),
        'execution_location': 'gpu_proxy_enhanced_cnn',
        'sampling_info': sampling_info,
        'cuda_available': tf.test.is_built_with_cuda(),
        'device_used': 'GPU' if gpus else 'CPU'
    }}
    
    print("Enhanced CNN training completed successfully")
    
except Exception as e:
    print(f"Training failed: {{str(e)}}")
    import traceback
    traceback.print_exc()
    
    result = {{
        'training_history': {{}},
        'final_loss': 999.0,
        'final_accuracy': 0.0,
        'final_val_loss': 999.0,  
        'final_val_accuracy': 0.0,
        'epochs_completed': 0,
        'model_params': 0,
        'gpu_used': len(tf.config.list_physical_devices('GPU')) > 0,
        'gpu_count': len(tf.config.list_physical_devices('GPU')),
        'execution_location': 'gpu_proxy_enhanced_cnn',
        'error': str(e),
        'sampling_info': sampling_info,
        'cuda_available': tf.test.is_built_with_cuda(),
        'device_used': 'GPU' if gpus else 'CPU'
    }}

print("Returning enhanced CNN results...")
"""
    
    def _generate_text_model_gpu_code(self, val_split: float) -> str:
        """Generate optimized text model code for GPU proxy with enhanced CUDA verification"""
        
        return f"""
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

print("=== ENHANCED GPU AVAILABILITY CHECK ===")
print(f"CUDA_HOME: {{os.environ.get('CUDA_HOME', 'Not set')}}")  
print(f"NVIDIA_VISIBLE_DEVICES: {{os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')}}")

print("Starting enhanced text/LSTM training on GPU proxy...")
print(f"TensorFlow version: {{tf.__version__}}")
print(f"CUDA built with TF: {{tf.test.is_built_with_cuda()}}")
print(f"GPU devices available: {{tf.config.list_physical_devices('GPU')}}")

# Critical GPU verification - fail fast if no GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {{len(gpus)}} devices")
        print(f"GPU details: {{[gpu.name for gpu in gpus]}}")
    except RuntimeError as e:
        print(f"GPU memory growth setup failed: {{e}}")
else:
    print("CRITICAL ERROR: No GPU devices detected!")
    print("This indicates CUDA libraries are not properly installed.")
    raise RuntimeError("No GPU devices available - CUDA libraries missing or corrupted")

print("=== GPU CHECK COMPLETE - PROCEEDING WITH TRAINING ===")

# Get data and config from context
x_train = np.array(context['x_train'])
y_train = np.array(context['y_train'])
config = context['config']
model_config = context['model_config']
sampling_info = context['sampling_info']

print(f"Text data shape: {{x_train.shape}}")
print(f"Labels shape: {{y_train.shape}}")
print(f"Sampling strategy: {{sampling_info['strategy']}}")

# Build enhanced text model with GPU placement verification
sequence_length = config['input_shape'][0] if isinstance(config['input_shape'], (list, tuple)) else config['input_shape']

print("Building enhanced LSTM model on GPU...")
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length,)),
        
        # Embedding layer
        keras.layers.Embedding(
            input_dim=10000,  # vocab_size
            output_dim=128,   # embedding_dim
            input_length=sequence_length,
            mask_zero=True
        ),
        
        # LSTM layer
        keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.25),
        
        # Dense layers
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(config['num_classes'], activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

print(f"Enhanced LSTM model created with {{model.count_params():,}} parameters")
print(f"Model device placement: GPU" if gpus else "Model device placement: CPU (ERROR!)")

# Train the model
epochs = config.get('epochs', 10)
batch_size = config.get('batch_size', 32)

print(f"Training LSTM for {{epochs}} epochs with batch size {{batch_size}} on {{'GPU' if gpus else 'CPU'}}...")

try:
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split={val_split},
        batch_size=batch_size,
        verbose=1
    )
    
    print("Enhanced LSTM training completed successfully!")
    
    # Extract results
    final_loss = float(history.history['loss'][-1])
    final_accuracy = float(history.history.get('accuracy', [0])[-1])
    final_val_loss = float(history.history.get('val_loss', [999])[-1])
    final_val_accuracy = float(history.history.get('val_accuracy', [0])[-1])
    
    print(f"Final LSTM training accuracy: {{final_accuracy:.4f}}")
    print(f"Final LSTM validation accuracy: {{final_val_accuracy:.4f}}")
    
    result = {{
        'training_history': {{k: [float(v) for v in values] for k, values in history.history.items()}},
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy,
        'epochs_completed': len(history.history['loss']),
        'model_params': int(model.count_params()),
        'gpu_used': len(tf.config.list_physical_devices('GPU')) > 0,
        'gpu_count': len(tf.config.list_physical_devices('GPU')),
        'execution_location': 'gpu_proxy_enhanced_lstm',
        'sampling_info': sampling_info,
        'cuda_available': tf.test.is_built_with_cuda(),
        'device_used': 'GPU' if gpus else 'CPU'
    }}
    
    print("Enhanced LSTM training completed successfully")
    
except Exception as e:
    print(f"LSTM training failed: {{str(e)}}")
    import traceback
    traceback.print_exc()
    
    result = {{
        'training_history': {{}},
        'final_loss': 999.0,
        'final_accuracy': 0.0,
        'final_val_loss': 999.0,
        'final_val_accuracy': 0.0,
        'epochs_completed': 0,
        'model_params': 0,
        'gpu_used': len(tf.config.list_physical_devices('GPU')) > 0,
        'gpu_count': len(tf.config.list_physical_devices('GPU')),
        'execution_location': 'gpu_proxy_enhanced_lstm',
        'error': str(e),
        'sampling_info': sampling_info,
        'cuda_available': tf.test.is_built_with_cuda(),
        'device_used': 'GPU' if gpus else 'CPU'
    }}

print("Returning enhanced LSTM results...")
"""
    
    def _convert_gpu_results_to_history_enhanced(self, execution_result: Dict[str, Any]) -> keras.callbacks.History:
        """
        Enhanced conversion of GPU proxy results to Keras History format
        """
        logger.debug("running _convert_gpu_results_to_history_enhanced ... Converting enhanced GPU proxy results")
        
        # Create enhanced History object
        history = keras.callbacks.History()
        history.history = execution_result.get('training_history', {})
        
        # Log enhanced training completion details
        epochs_completed = execution_result.get('epochs_completed', 0)
        final_val_accuracy = execution_result.get('final_val_accuracy', 0.0)
        model_params = execution_result.get('model_params', 0)
        execution_location = execution_result.get('execution_location', 'unknown')
        sampling_info = execution_result.get('sampling_info', {})
        
        logger.debug(f"running _convert_gpu_results_to_history_enhanced ... Enhanced GPU training completed:")
        logger.debug(f"running _convert_gpu_results_to_history_enhanced ... - Epochs: {epochs_completed}")
        logger.debug(f"running _convert_gpu_results_to_history_enhanced ... - Final val accuracy: {final_val_accuracy:.4f}")
        logger.debug(f"running _convert_gpu_results_to_history_enhanced ... - Model parameters: {model_params:,}")
        logger.debug(f"running _convert_gpu_results_to_history_enhanced ... - Execution location: {execution_location}")
        
        if sampling_info:
            reduction_ratio = sampling_info.get('reduction_ratio', 1.0)
            strategy = sampling_info.get('strategy', 'unknown')
            logger.debug(f"running _convert_gpu_results_to_history_enhanced ... - Sampling strategy: {strategy}")
            logger.debug(f"running _convert_gpu_results_to_history_enhanced ... - Data reduction: {reduction_ratio:.1%}")
        
        return history

    def _train_locally_optimized(
        self, 
        data: Dict[str, Any], 
        validation_split: Optional[float] = None
    ) -> keras.callbacks.History:
        """
        Optimized local training execution
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
        
        # Train model with enhanced monitoring
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        with TimedOperation("optimized model training", "model_builder"):
            self.training_history = self.model.fit(
                data['x_train'], 
                data['y_train'],
                epochs=self.model_config.epochs,
                validation_split=validation_split or self.model_config.validation_split,
                verbose=1,
                callbacks=callbacks_list
            )
        
        logger.debug("running _train_locally_optimized ... Optimized local training completed")
        return self.training_history

    def _setup_training_callbacks_optimized(self) -> List[keras.callbacks.Callback]:
        """Setup optimized training callbacks"""
        callbacks_list = []
        
        # Add real-time visualization callbacks if enabled
        if self.model_config.enable_realtime_plots and self.plot_dir:
            try:
                # Real-time training visualization
                realtime_visualizer = RealTimeTrainingVisualizer(
                    model_builder=self,
                    plot_dir=self.plot_dir
                )
                
                realtime_callback = RealTimeTrainingCallback(
                    visualizer=realtime_visualizer
                )
                callbacks_list.append(realtime_callback)
                
                logger.debug("running _setup_training_callbacks_optimized ... Added real-time training visualization")
                
            except Exception as viz_error:
                logger.warning(f"running _setup_training_callbacks_optimized ... Failed to setup real-time visualization: {viz_error}")
        
        # Add other optimized callbacks as needed
        # (gradient flow monitoring, weights/bias monitoring, etc.)
        
        return callbacks_list

    def evaluate(
        self, 
        data: Dict[str, Any], 
        log_detailed_predictions: bool, 
        max_predictions_to_show: int,
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None
    ) -> Tuple[float, float]:
        """
        Optimized model evaluation with enhanced analysis and performance monitoring
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluation")
        
        logger.debug("running evaluate ... Starting optimized model evaluation...")
        
        # Create timestamp if not provided
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        
        # Use provided plot directory or create default
        if plot_dir is None:
            plot_dir = Path("plots") / run_timestamp
            plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced evaluation with performance monitoring
        with TimedOperation("optimized model evaluation", "model_builder"):
            evaluation_results = self.model.evaluate(
                data['x_test'], 
                data['y_test'],
                verbose=1
            )
            
            # Handle both single and multiple metrics
            test_loss, test_accuracy = self._extract_evaluation_metrics(evaluation_results)
        
        # Enhanced analysis pipeline
        analysis_results = self._run_enhanced_analysis_pipeline(
            data, test_loss, test_accuracy, run_timestamp, plot_dir
        )
        
        # Log detailed predictions if requested
        if log_detailed_predictions and max_predictions_to_show > 0:
            self._log_detailed_predictions_optimized(
                data, max_predictions_to_show, run_timestamp, plot_dir
            )
        
        # Performance summary
        self._log_evaluation_summary(test_loss, test_accuracy, analysis_results)
        
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
    
    def _run_enhanced_analysis_pipeline(
        self, 
        data: Dict[str, Any], 
        test_loss: float, 
        test_accuracy: float,
        run_timestamp: str, 
        plot_dir: Path
    ) -> Dict[str, Any]:
        """
        Run enhanced analysis pipeline with optimized error handling
        """
        analysis_results: Dict[str, Optional[Dict[str, Any]]] = {
            'confusion_matrix': None,
            'training_history': None,
            'training_animation': None,
            'gradient_flow': None,
            'weights_bias': None,
            'activation_maps': None
        }
        
        # Confusion Matrix Analysis
        if self.model_config.show_confusion_matrix:
            analysis_results['confusion_matrix'] = self._run_confusion_matrix_analysis_optimized(
                data, run_timestamp, plot_dir
            )
        
        # Training History Analysis
        if self.model_config.show_training_history and self.training_history:
            analysis_results['training_history'] = self._run_training_history_analysis_optimized(
                run_timestamp, plot_dir
            )
        
        # Training Animation
        if self.training_history:
            analysis_results['training_animation'] = self._run_training_animation_optimized(
                run_timestamp, plot_dir
            )
        
        # Gradient Flow Analysis
        if self.model_config.show_gradient_flow:
            analysis_results['gradient_flow'] = self._run_gradient_flow_analysis_optimized(
                data, run_timestamp, plot_dir
            )
        
        # Weights and Bias Analysis
        if self.model_config.show_weights_bias_analysis:
            analysis_results['weights_bias'] = self._run_weights_bias_analysis_optimized(
                run_timestamp, plot_dir
            )
        
        # Activation Maps Analysis (CNN only)
        if self.model_config.show_activation_maps and self._detect_data_type_enhanced() == "image":
            analysis_results['activation_maps'] = self._run_activation_maps_analysis_optimized(
                data, run_timestamp, plot_dir
            )
        
        return analysis_results
    
    def _run_confusion_matrix_analysis_optimized(
        self, 
        data: Dict[str, Any], 
        run_timestamp: str, 
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Optimized confusion matrix analysis"""
        try:
            logger.debug("running _run_confusion_matrix_analysis_optimized ... Generating confusion matrix analysis...")
            
            # Get predictions efficiently
            # Get predictions efficiently
            if self.model is None:
                raise ValueError("Model must be built and trained before generating predictions")
            predictions = self.model.predict(data['x_test'], verbose=0, batch_size=64)
            
            # Convert labels efficiently
            true_labels, predicted_labels = self._convert_labels_for_analysis(data['y_test'], predictions)
            
            # Create analyzer
            class_names = self.dataset_config.class_names or [f"Class_{i}" for i in range(self.dataset_config.num_classes)]
            cm_analyzer = ConfusionMatrixAnalyzer(class_names=class_names)
            
            # Perform analysis
            results = cm_analyzer.analyze_and_visualize(
                true_labels=true_labels,
                predicted_labels=predicted_labels,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir
            )
            
            if 'error' not in results:
                logger.debug("running _run_confusion_matrix_analysis_optimized ... Confusion matrix analysis completed successfully")
                overall_accuracy = results.get('overall_accuracy', 0.0)
                logger.debug(f"running _run_confusion_matrix_analysis_optimized ... Confusion matrix accuracy: {overall_accuracy:.4f}")
            
            return results
            
        except Exception as e:
            logger.warning(f"running _run_confusion_matrix_analysis_optimized ... Analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_training_history_analysis_optimized(
        self, 
        run_timestamp: str, 
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Optimized training history analysis"""
        try:
            logger.debug("running _run_training_history_analysis_optimized ... Generating training history analysis...")
            
            history_analyzer = TrainingHistoryAnalyzer(model_name=self.dataset_config.name)
            
            if self.training_history is None:
                raise ValueError("Training history must be available for analysis")
            results = history_analyzer.analyze_and_visualize(
                training_history=self.training_history.history,
                model=self.model,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir
            )
            
            if 'error' not in results:
                logger.debug("running _run_training_history_analysis_optimized ... Training history analysis completed")
                
                # Log key insights
                insights = results.get('training_insights', [])
                for insight in insights[:3]:
                    logger.debug(f"running _run_training_history_analysis_optimized ... Insight: {insight}")
            
            return results
            
        except Exception as e:
            logger.warning(f"running _run_training_history_analysis_optimized ... Analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_training_animation_optimized(
        self, 
        run_timestamp: str, 
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Optimized training animation creation"""
        try:
            logger.debug("running _run_training_animation_optimized ... Generating training animation...")
            
            animation_analyzer = TrainingAnimationAnalyzer(model_name=self.dataset_config.name)
            
            if self.training_history is None:
                raise ValueError("Training history must be available for analysis")
            results = animation_analyzer.analyze_and_animate(
                training_history=self.training_history.history,
                model=self.model,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir,
                animation_duration=8.0,  # Optimized duration
                fps=12  # Higher FPS for smoother animation
            )
            
            if 'error' not in results:
                frame_count = results.get('frame_count', 0)
                logger.debug(f"running _run_training_animation_optimized ... Animation created with {frame_count} frames")
            
            return results
            
        except Exception as e:
            logger.warning(f"running _run_training_animation_optimized ... Animation creation failed: {e}")
            return {'error': str(e)}
    
    def _run_gradient_flow_analysis_optimized(
        self, 
        data: Dict[str, Any], 
        run_timestamp: str, 
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Optimized gradient flow analysis with intelligent sampling"""
        try:
            logger.debug("running _run_gradient_flow_analysis_optimized ... Generating gradient flow analysis...")
            
            # Intelligent sample selection for gradient analysis
            sample_size = min(self.model_config.gradient_flow_sample_size, len(data['x_test']))
            sample_indices = self._select_representative_samples(data['x_test'], data['y_test'], sample_size)
            
            sample_x = data['x_test'][sample_indices]
            sample_y = data['y_test'][sample_indices]
            
            gradient_analyzer = GradientFlowAnalyzer(model_name=self.dataset_config.name)
            
            results = gradient_analyzer.analyze_and_visualize(
                model=self.model,
                sample_data=sample_x,
                sample_labels=sample_y,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir
            )
            
            if 'error' not in results:
                gradient_health = results.get('gradient_health', 'unknown')
                logger.debug(f"running _run_gradient_flow_analysis_optimized ... Gradient health: {gradient_health}")
            
            return results
            
        except Exception as e:
            logger.warning(f"running _run_gradient_flow_analysis_optimized ... Analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_weights_bias_analysis_optimized(
        self, 
        run_timestamp: str, 
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Optimized weights and bias analysis"""
        try:
            logger.debug("running _run_weights_bias_analysis_optimized ... Generating weights and bias analysis...")
            
            weights_bias_analyzer = WeightsBiasAnalyzer(model_name=self.dataset_config.name)
            
            results = weights_bias_analyzer.analyze_and_visualize(
                model=self.model,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir,
                max_layers_to_plot=8  # Optimized for performance
            )
            
            if 'error' not in results:
                parameter_health = results.get('parameter_health', 'unknown')
                logger.debug(f"running _run_weights_bias_analysis_optimized ... Parameter health: {parameter_health}")
            
            return results
            
        except Exception as e:
            logger.warning(f"running _run_weights_bias_analysis_optimized ... Analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_activation_maps_analysis_optimized(
        self, 
        data: Dict[str, Any], 
        run_timestamp: str, 
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Optimized activation maps analysis for CNN models"""
        try:
            logger.debug("running _run_activation_maps_analysis_optimized ... Generating activation maps analysis...")
            
            # Intelligent sample selection for activation analysis
            sample_size = min(self.model_config.activation_max_total_samples, len(data['x_test']))
            sample_indices = self._select_diverse_samples_for_activation(data['x_test'], data['y_test'], sample_size)
            
            sample_x = data['x_test'][sample_indices]
            sample_y = data['y_test'][sample_indices]
            
            # Convert labels
            if sample_y.ndim > 1 and sample_y.shape[1] > 1:
                sample_labels = np.argmax(sample_y, axis=1)
            else:
                sample_labels = sample_y.flatten()
            
            class_names = self.dataset_config.class_names or [f"Class_{i}" for i in range(self.dataset_config.num_classes)]
            
            activation_analyzer = ActivationMapAnalyzer(model_name=self.dataset_config.name)
            
            results = activation_analyzer.analyze_and_visualize(
                model=self.model,
                sample_images=sample_x,
                sample_labels=sample_labels,
                class_names=class_names,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir,
                model_config=self.model_config
            )
            
            if 'error' not in results:
                filter_health = results.get('filter_health', {})
                health_status = filter_health.get('overall_status', 'unknown')
                logger.debug(f"running _run_activation_maps_analysis_optimized ... Filter health: {health_status}")
            
            return results
            
        except Exception as e:
            logger.warning(f"running _run_activation_maps_analysis_optimized ... Analysis failed: {e}")
            return {'error': str(e)}
    
    def _convert_labels_for_analysis(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert one-hot encoded labels to class indices efficiently"""
        
        # Convert true labels
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            true_labels = np.argmax(y_true, axis=1)
        else:
            true_labels = y_true.flatten()
        
        # Convert predicted labels
        predicted_labels = np.argmax(y_pred, axis=1)
        
        return true_labels, predicted_labels
    
    def _select_representative_samples(
        self, 
        x_data: np.ndarray, 
        y_data: np.ndarray, 
        sample_size: int
    ) -> np.ndarray:
        """Select representative samples for analysis using stratified sampling"""
        
        if sample_size >= len(x_data):
            return np.arange(len(x_data))
        
        # Convert labels if needed
        if y_data.ndim > 1 and y_data.shape[1] > 1:
            labels = np.argmax(y_data, axis=1)
        else:
            labels = y_data.flatten()
        
        # Stratified sampling
        selected_indices = []
        unique_classes = np.unique(labels)
        samples_per_class = max(1, sample_size // len(unique_classes))
        
        for class_id in unique_classes:
            class_indices = np.where(labels == class_id)[0]
            if len(class_indices) > 0:
                n_samples = min(samples_per_class, len(class_indices))
                sampled = np.random.choice(class_indices, n_samples, replace=False)
                selected_indices.extend(sampled)
        
        # Fill remaining slots if needed
        if len(selected_indices) < sample_size:
            remaining_needed = sample_size - len(selected_indices)
            all_indices = set(range(len(x_data)))
            unused_indices = list(all_indices - set(selected_indices))
            
            if unused_indices:
                additional = np.random.choice(
                    unused_indices, 
                    min(remaining_needed, len(unused_indices)), 
                    replace=False
                )
                selected_indices.extend(additional)
        
        return np.array(selected_indices[:sample_size])
    
    def _select_diverse_samples_for_activation(
        self, 
        x_data: np.ndarray, 
        y_data: np.ndarray, 
        sample_size: int
    ) -> np.ndarray:
        """Select diverse samples for activation analysis"""
        
        strategy = self.model_config.activation_sample_selection_strategy
        
        if strategy == "random":
            return np.random.choice(len(x_data), sample_size, replace=False)
        
        elif strategy == "representative":
            return self._select_representative_samples(x_data, y_data, sample_size)
        
        elif strategy == "mixed":
            # Mix of representative and random samples
            half_size = sample_size // 2
            representative = self._select_representative_samples(x_data, y_data, half_size)
            
            # Select random samples not in representative set
            remaining_indices = list(set(range(len(x_data))) - set(representative))
            random_needed = sample_size - len(representative)
            
            if remaining_indices and random_needed > 0:
                random_samples = np.random.choice(
                    remaining_indices, 
                    min(random_needed, len(remaining_indices)), 
                    replace=False
                )
                return np.concatenate([representative, random_samples])
            
            return representative
        
        else:
            # Default to representative
            return self._select_representative_samples(x_data, y_data, sample_size)
    
    def _log_detailed_predictions_optimized(
        self, 
        data: Dict[str, Any], 
        max_predictions_to_show: int,
        run_timestamp: str,
        plot_dir: Path
    ) -> None:
        """Optimized detailed prediction logging with performance improvements"""
        
        logger.debug("running _log_detailed_predictions_optimized ... Generating optimized prediction analysis...")
        
        # Efficient prediction generation
        if self.model is None:
            raise ValueError("Model must be built and trained before generating predictions")
        predictions = self.model.predict(data['x_test'], verbose=0, batch_size=128)
        
        # Convert labels efficiently
        true_labels, predicted_labels = self._convert_labels_for_analysis(data['y_test'], predictions)
        confidence_scores = np.max(predictions, axis=1)
        
        # Get class names
        class_names = self.dataset_config.class_names or [f"Class_{i}" for i in range(self.dataset_config.num_classes)]
        
        # Efficient correct/incorrect separation
        correct_mask = true_labels == predicted_labels
        correct_indices = np.where(correct_mask)[0]
        incorrect_indices = np.where(~correct_mask)[0]
        
        # Optimized logging of samples
        max_correct = min(3, len(correct_indices), max_predictions_to_show // 3)
        max_incorrect = min(max_predictions_to_show - max_correct, len(incorrect_indices))
        
        if max_correct > 0:
            logger.debug("running _log_detailed_predictions_optimized ... Sample correct predictions:")
            sample_correct = np.random.choice(correct_indices, max_correct, replace=False)
            
            for idx in sample_correct:
                confidence = confidence_scores[idx]
                true_class = class_names[true_labels[idx]]
                logger.debug(f"running _log_detailed_predictions_optimized ... ✅ Correct: {true_class}, confidence: {confidence:.3f}")
        
        if max_incorrect > 0:
            logger.debug("running _log_detailed_predictions_optimized ... Sample incorrect predictions:")
            sample_incorrect = np.random.choice(incorrect_indices, max_incorrect, replace=False)
            
            for idx in sample_incorrect:
                confidence = confidence_scores[idx]
                true_class = class_names[true_labels[idx]]
                pred_class = class_names[predicted_labels[idx]]
                logger.debug(f"running _log_detailed_predictions_optimized ... ❌ Incorrect: predicted {pred_class}, actual {true_class}, confidence: {confidence:.3f}")
        
        # Performance summary
        total_predictions = len(true_labels)
        correct_count = len(correct_indices)
        accuracy = correct_count / total_predictions
        
        logger.debug(f"running _log_detailed_predictions_optimized ... Prediction summary:")
        logger.debug(f"running _log_detailed_predictions_optimized ... - Total: {total_predictions}, Correct: {correct_count} ({accuracy:.1%})")
        logger.debug(f"running _log_detailed_predictions_optimized ... - Average confidence: {np.mean(confidence_scores):.3f}")
    
    def _log_evaluation_summary(
        self, 
        test_loss: float, 
        test_accuracy: float, 
        analysis_results: Dict[str, Any]
    ) -> None:
        """Log comprehensive evaluation summary"""
        
        logger.debug("running _log_evaluation_summary ... Evaluation Summary:")
        logger.debug(f"running _log_evaluation_summary ... - Test accuracy: {test_accuracy:.4f}")
        logger.debug(f"running _log_evaluation_summary ... - Test loss: {test_loss:.4f}")
        
        # Log analysis completion status
        completed_analyses = []
        failed_analyses = []
        
        for analysis_name, result in analysis_results.items():
            if result is None:
                continue
            elif isinstance(result, dict) and 'error' in result:
                failed_analyses.append(analysis_name)
            else:
                completed_analyses.append(analysis_name)
        
        if completed_analyses:
            logger.debug(f"running _log_evaluation_summary ... - Completed analyses: {', '.join(completed_analyses)}")
        
        if failed_analyses:
            logger.warning(f"running _log_evaluation_summary ... - Failed analyses: {', '.join(failed_analyses)}")
    
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
        
        # Generate optimized filename
        filename = self._generate_optimized_filename(test_accuracy, run_timestamp, run_name)
        
        # Determine save directory
        save_dir = self._determine_save_directory(run_name)
        
        # Create final filepath
        final_filepath = save_dir / filename
        
        # Save model with optimization
        logger.debug(f"running save_model ... Saving optimized model to {final_filepath}")
        
        try:
            # Save with optimized settings
            self.model.save(
                final_filepath,
                save_format='tf',  # Use TensorFlow SavedModel format for better compression
                save_traces=False  # Skip saving function traces for smaller size
            )
            
            # Save additional metadata
            self._save_model_metadata(save_dir, filename, test_accuracy, run_timestamp, run_name)
            
            logger.debug(f"running save_model ... Model saved successfully")
            self._log_model_save_summary(final_filepath, test_accuracy, run_name)
            
        except Exception as e:
            logger.error(f"running save_model ... Failed to save model: {e}")
            # Fallback to standard .keras format
            keras_filepath = save_dir / filename.replace('.tf', '.keras')
            logger.debug(f"running save_model ... Falling back to .keras format: {keras_filepath}")
            
            self.model.save(keras_filepath)
            final_filepath = keras_filepath
        
        return str(final_filepath)
    
    def _generate_optimized_filename(
        self, 
        test_accuracy: Optional[float], 
        run_timestamp: str, 
        run_name: Optional[str]
    ) -> str:
        """Generate optimized filename with metadata"""
        
        # Determine architecture type
        data_type = self._detect_data_type_enhanced()
        architecture_name = "CNN" if data_type == "image" else "LSTM"
        
        # Clean dataset name
        dataset_name_clean = self.dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        dataset_name_clean = dataset_name_clean.replace("_dataset", "")
        
        # Format accuracy
        accuracy_str = f"{test_accuracy:.1f}".replace(".", "_") if test_accuracy is not None else "unknown"
        
        if run_name:
            # Use run_name with additional metadata
            filename = f"model_{run_name}_acc_{accuracy_str}_{architecture_name}.tf"
        else:
            # Fallback naming
            filename = f"model_{run_timestamp}_{architecture_name}_{dataset_name_clean}_acc_{accuracy_str}.tf"
        
        return filename
    
    def _determine_save_directory(self, run_name: Optional[str]) -> Path:
        """Determine optimal save directory"""
        
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        
        if run_name:
            # Save in optimization_results structure
            optimization_results_dir = project_root / "optimization_results"
            save_dir = optimization_results_dir / run_name
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"running _determine_save_directory ... Using optimization results directory: {save_dir}")
        else:
            # Fallback to saved_models
            save_dir = project_root / "saved_models"
            save_dir.mkdir(exist_ok=True)
            logger.debug(f"running _determine_save_directory ... Using saved_models directory: {save_dir}")
        
        return save_dir
    
    def _save_model_metadata(
        self, 
        save_dir: Path, 
        filename: str, 
        test_accuracy: Optional[float], 
        run_timestamp: str, 
        run_name: Optional[str]
    ) -> None:
        """Save additional model metadata for tracking and reproducibility"""
        
        try:
            metadata = {
                'model_info': {
                    'filename': filename,
                    'save_timestamp': datetime.now().isoformat(),
                    'run_timestamp': run_timestamp,
                    'run_name': run_name,
                    'architecture_type': self._detect_data_type_enhanced(),
                    'dataset_name': self.dataset_config.name,
                    'input_shape': self.dataset_config.input_shape,
                    'num_classes': self.dataset_config.num_classes,
                    'total_parameters': self.model.count_params() if self.model else 0
                },
                'performance': {
                    'test_accuracy': test_accuracy,
                    'training_epochs': self.model_config.epochs,
                    'validation_split': self.model_config.validation_split
                },
                'model_config': {
                    'architecture_type': self.model_config.architecture_type,
                    'num_layers_conv': self.model_config.num_layers_conv,
                    'filters_per_conv_layer': self.model_config.filters_per_conv_layer,
                    'num_layers_hidden': self.model_config.num_layers_hidden,
                    'first_hidden_layer_nodes': self.model_config.first_hidden_layer_nodes,
                    'optimizer': self.model_config.optimizer,
                    'loss': self.model_config.loss,
                    'use_gpu_proxy': self.model_config.use_gpu_proxy,
                    'gpu_proxy_used': self.gpu_proxy_available and self.model_config.use_gpu_proxy
                },
                'training_history': self.training_history.history if self.training_history else None
            }
            
            # Save metadata as JSON
            metadata_file = save_dir / f"{filename.split('.')[0]}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.debug(f"running _save_model_metadata ... Metadata saved to: {metadata_file}")
            
        except Exception as e:
            logger.warning(f"running _save_model_metadata ... Failed to save metadata: {e}")
    
    def _log_model_save_summary(
        self, 
        filepath: Path, 
        test_accuracy: Optional[float], 
        run_name: Optional[str]
    ) -> None:
        """Log comprehensive model save summary"""
        
        logger.debug(f"running _log_model_save_summary ... Model Save Summary:")
        logger.debug(f"running _log_model_save_summary ... - File path: {filepath}")
        logger.debug(f"running _log_model_save_summary ... - Dataset: {self.dataset_config.name}")
        logger.debug(f"running _log_model_save_summary ... - Architecture: {self._detect_data_type_enhanced().upper()}")
        logger.debug(f"running _log_model_save_summary ... - Input shape: {self.dataset_config.input_shape}")
        logger.debug(f"running _log_model_save_summary ... - Classes: {self.dataset_config.num_classes}")
        if self.model is not None:
            logger.debug(f"running _log_model_save_summary ... - Parameters: {self.model.count_params():,}")
        
        if test_accuracy is not None:
            logger.debug(f"running _log_model_save_summary ... - Test accuracy: {test_accuracy:.4f}")
        
        if run_name:
            logger.debug(f"running _log_model_save_summary ... - Run name: {run_name}")
        
        # Log file size
        try:
            if filepath.exists():
                if filepath.is_dir():  # TensorFlow SavedModel format
                    total_size = sum(f.stat().st_size for f in filepath.rglob('*') if f.is_file())
                else:  # Single file format
                    total_size = filepath.stat().st_size
                
                size_mb = total_size / (1024 * 1024)
                logger.debug(f"running _log_model_save_summary ... - File size: {size_mb:.1f} MB")
        except Exception:
            pass
    
    def _detect_and_setup_gpu_proxy(self) -> Tuple[bool, Optional[str], Optional[Any]]:
        """
        Enhanced GPU proxy detection and setup with advanced error handling
        """
        if not self.model_config.use_gpu_proxy:
            logger.debug("running _detect_and_setup_gpu_proxy ... GPU proxy disabled in config")
            return False, None, None
        
        logger.debug("running _detect_and_setup_gpu_proxy ... Starting enhanced GPU proxy detection...")
        
        # Check dependencies with detailed reporting
        missing_deps = self._check_gpu_proxy_dependencies()
        if missing_deps:
            return self._handle_missing_dependencies(missing_deps)
        
        # Enhanced path detection
        gpu_proxy_path = self._detect_gpu_proxy_path_enhanced()
        
        if gpu_proxy_path is None and self.model_config.gpu_proxy_auto_clone:
            logger.debug("running _detect_and_setup_gpu_proxy ... GPU proxy not found, attempting enhanced auto-clone...")
            gpu_proxy_path = self._clone_gpu_proxy_repo_enhanced()
        
        if gpu_proxy_path is None:
            return self._handle_gpu_proxy_not_found()
        
        # Setup client with enhanced configuration
        try:
            runpod_client = self._setup_gpu_proxy_client_enhanced(gpu_proxy_path)
            if runpod_client is not None:
                logger.debug(f"running _detect_and_setup_gpu_proxy ... Enhanced GPU proxy setup successful: {gpu_proxy_path}")
                return True, str(gpu_proxy_path), runpod_client
            else:
                return self._handle_client_setup_failure(gpu_proxy_path)
                
        except Exception as e:
            return self._handle_setup_exception(e, gpu_proxy_path)
    
    def _check_gpu_proxy_dependencies(self) -> List[str]:
        """Check for required GPU proxy dependencies"""
        missing_deps = []
        
        dependencies = {
            'dotenv': 'python-dotenv',
            'requests': 'requests'
        }
        
        for module_name, package_name in dependencies.items():
            try:
                __import__(module_name)
                logger.debug(f"running _check_gpu_proxy_dependencies ... {package_name} dependency available")
            except ImportError:
                missing_deps.append(package_name)
        
        return missing_deps
    
    def _handle_missing_dependencies(self, missing_deps: List[str]) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Handle missing GPU proxy dependencies"""
        logger.error(f"running _handle_missing_dependencies ... Missing GPU proxy dependencies: {', '.join(missing_deps)}")
        logger.error(f"running _handle_missing_dependencies ... Install with: pip install {' '.join(missing_deps)}")
        
        if self.model_config.gpu_proxy_fallback_local:
            logger.warning("running _handle_missing_dependencies ... Falling back to local execution")
            return False, None, None
        else:
            raise RuntimeError(f"GPU proxy dependencies missing: {', '.join(missing_deps)}")
    
    def _detect_gpu_proxy_path_enhanced(self) -> Optional[Path]:
        """Enhanced GPU proxy path detection with multiple strategies"""
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        
        # Enhanced detection paths with priority order
        detection_paths = [
            project_root / "gpu-proxy",                    # Subdirectory
            project_root.parent / "gpu-proxy",             # Sibling directory
            Path.home() / "gpu-proxy",                     # User home directory
            Path("/opt/gpu-proxy"),                        # System-wide installation
        ]
        
        for path in detection_paths:
            logger.debug(f"running _detect_gpu_proxy_path_enhanced ... Checking path: {path}")
            
            if self._validate_gpu_proxy_installation(path):
                logger.debug(f"running _detect_gpu_proxy_path_enhanced ... Found valid GPU proxy at: {path}")
                return path
        
        return None
    
    def _validate_gpu_proxy_installation(self, path: Path) -> bool:
        """Validate that the path contains a proper GPU proxy installation"""
        if not path.exists() or not path.is_dir():
            return False
        
        # Check for key files that indicate a proper installation
        required_files = [
            "src/runpod/client.py",
            "requirements.txt"
        ]
        
        for required_file in required_files:
            if not (path / required_file).exists():
                logger.debug(f"running _validate_gpu_proxy_installation ... Missing required file: {required_file}")
                return False
        
        logger.debug(f"running _validate_gpu_proxy_installation ... Valid GPU proxy installation found")
        return True
    
    def _clone_gpu_proxy_repo_enhanced(self) -> Optional[Path]:
        """Enhanced GPU proxy repository cloning with better error handling"""
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        target_path = project_root / "gpu-proxy"
        
        try:
            logger.debug(f"running _clone_gpu_proxy_repo_enhanced ... Enhanced cloning to: {target_path}")
            
            # Ensure parent directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Enhanced clone command with progress
            clone_command = [
                "git", "clone", 
                "--progress",
                "--depth", "1",  # Shallow clone for faster download
                "https://github.com/TheBuleGanteng/gpu-proxy.git",
                str(target_path)
            ]
            
            result = subprocess.run(
                clone_command,
                capture_output=True,
                text=True,
                timeout=120  # Increased timeout for large repos
            )
            
            if result.returncode == 0:
                logger.debug(f"running _clone_gpu_proxy_repo_enhanced ... Successfully cloned GPU proxy")
                
                # Validate the cloned installation
                if self._validate_gpu_proxy_installation(target_path):
                    return target_path
                else:
                    logger.error("running _clone_gpu_proxy_repo_enhanced ... Cloned repository appears invalid")
                    return None
            else:
                logger.error(f"running _clone_gpu_proxy_repo_enhanced ... Git clone failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("running _clone_gpu_proxy_repo_enhanced ... Git clone timed out")
            return None
        except FileNotFoundError:
            logger.error("running _clone_gpu_proxy_repo_enhanced ... Git not found - install git or manually clone GPU proxy")
            return None
        except Exception as e:
            logger.error(f"running _clone_gpu_proxy_repo_enhanced ... Enhanced clone failed: {e}")
            return None
    
    def _handle_gpu_proxy_not_found(self) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Handle case where GPU proxy cannot be found"""
        logger.warning("running _handle_gpu_proxy_not_found ... GPU proxy repository not found")
        
        if self.model_config.gpu_proxy_fallback_local:
            logger.warning("running _handle_gpu_proxy_not_found ... Falling back to local execution")
            return False, None, None
        else:
            raise RuntimeError("GPU proxy repository not found and fallback disabled")
    
    def _handle_client_setup_failure(self, gpu_proxy_path: Path) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Handle GPU proxy client setup failure"""
        logger.warning("running _handle_client_setup_failure ... GPU proxy found but client setup failed")
        logger.warning("running _handle_client_setup_failure ... Common issues:")
        logger.warning("running _handle_client_setup_failure ... - Missing .env file in GPU proxy directory")
        logger.warning("running _handle_client_setup_failure ... - Invalid RunPod API key or endpoint")
        logger.warning("running _handle_client_setup_failure ... - Network connectivity issues")
        
        if self.model_config.gpu_proxy_fallback_local:
            logger.warning("running _handle_client_setup_failure ... Falling back to local execution")
            return False, str(gpu_proxy_path), None
        else:
            raise RuntimeError("GPU proxy client setup failed and fallback disabled")
    
    def _handle_setup_exception(self, e: Exception, gpu_proxy_path: Optional[Path]) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Handle GPU proxy setup exceptions"""
        logger.warning(f"running _handle_setup_exception ... GPU proxy setup failed: {e}")
        logger.debug(f"running _handle_setup_exception ... Setup error details: {traceback.format_exc()}")
        
        if self.model_config.gpu_proxy_fallback_local:
            logger.warning("running _handle_setup_exception ... Falling back to local execution")
            return False, str(gpu_proxy_path) if gpu_proxy_path else None, None
        else:
            raise RuntimeError(f"GPU proxy setup failed and fallback disabled: {e}")
    
    def _setup_gpu_proxy_client_enhanced(self, gpu_proxy_path: Path) -> Optional[Any]:
        """
        Enhanced GPU proxy client setup with better configuration management
        """
        try:
            # Add GPU proxy to Python path
            gpu_proxy_str = str(gpu_proxy_path)
            if gpu_proxy_str not in sys.path:
                sys.path.insert(0, gpu_proxy_str)
                logger.debug(f"running _setup_gpu_proxy_client_enhanced ... Added to Python path: {gpu_proxy_str}")
            
            # Enhanced .env validation
            if not self._validate_env_file_enhanced(gpu_proxy_path):
                return None
            
            # Dynamic module import with enhanced error handling
            client = self._import_and_initialize_client_enhanced(gpu_proxy_path)
            
            if client is None:
                return None
            
            # Enhanced health check
            if self._perform_enhanced_health_check(client):
                logger.debug("running _setup_gpu_proxy_client_enhanced ... Enhanced GPU proxy client setup successful")
                return client
            else:
                logger.warning("running _setup_gpu_proxy_client_enhanced ... Enhanced health check failed")
                return None
                
        except Exception as e:
            logger.warning(f"running _setup_gpu_proxy_client_enhanced ... Enhanced client setup failed: {e}")
            logger.debug(f"running _setup_gpu_proxy_client_enhanced ... Error details: {traceback.format_exc()}")
            return None
    
    def _validate_env_file_enhanced(self, gpu_proxy_path: Path) -> bool:
        """Enhanced .env file validation"""
        env_file_path = gpu_proxy_path / ".env"
        
        if not env_file_path.exists():
            logger.error(f"running _validate_env_file_enhanced ... .env file not found at: {env_file_path}")
            logger.error("running _validate_env_file_enhanced ... Create .env file with RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")
            return False
        
        try:
            with open(env_file_path, 'r') as f:
                env_content = f.read()
            
            required_keys = ['RUNPOD_API_KEY', 'RUNPOD_ENDPOINT_ID']
            missing_keys = []
            
            for key in required_keys:
                if key not in env_content:
                    missing_keys.append(key)
            
            if missing_keys:
                logger.error(f"running _validate_env_file_enhanced ... Missing required keys in .env: {', '.join(missing_keys)}")
                return False
            
            # Additional validation - check for empty values
            import dotenv
            env_vars = dotenv.dotenv_values(env_file_path)
            
            for key in required_keys:
                value = env_vars.get(key, '')
                if not value or not value.strip():
                    logger.error(f"running _validate_env_file_enhanced ... Empty value for {key} in .env file")
                    return False
                        
            logger.debug(f"running _validate_env_file_enhanced ... .env file validated successfully")
            return True
            
        except Exception as env_error:
            logger.error(f"running _validate_env_file_enhanced ... Failed to validate .env file: {env_error}")
            return False
    
    def _import_and_initialize_client_enhanced(self, gpu_proxy_path: Path) -> Optional[Any]:
        """Enhanced client import and initialization"""
        
        import importlib.util
        import os
        
        client_module_path = gpu_proxy_path / "src" / "runpod" / "client.py"
        
        if not client_module_path.exists():
            logger.error(f"running _import_and_initialize_client_enhanced ... Client module not found: {client_module_path}")
            return None
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location("runpod_client", client_module_path)
        if spec is None or spec.loader is None:
            logger.error("running _import_and_initialize_client_enhanced ... Failed to create module spec")
            return None
        
        client_module = importlib.util.module_from_spec(spec)
        
        # Change working directory temporarily for initialization
        original_cwd = os.getcwd()
        try:
            os.chdir(gpu_proxy_path)
            logger.debug(f"running _import_and_initialize_client_enhanced ... Changed working directory to: {gpu_proxy_path}")
            
            spec.loader.exec_module(client_module)
            
            # Get and initialize client
            RunPodClient = getattr(client_module, 'RunPodClient', None)
            if RunPodClient is None:
                logger.error("running _import_and_initialize_client_enhanced ... RunPodClient class not found")
                return None
            
            # Initialize with enhanced error handling
            try:
                client = RunPodClient()
                logger.debug("running _import_and_initialize_client_enhanced ... RunPodClient initialized successfully")
                return client
                
            except Exception as init_error:
                logger.error(f"running _import_and_initialize_client_enhanced ... Client initialization failed: {init_error}")
                return None
        
        finally:
            os.chdir(original_cwd)
            logger.debug(f"running _import_and_initialize_client_enhanced ... Restored working directory")
    
    def _perform_enhanced_health_check(self, client: Any) -> bool:
        """Enhanced health check with detailed validation"""
        
        try:
            logger.debug("running _perform_enhanced_health_check ... Performing enhanced health check...")
            
            # Perform health check with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Health check timed out")
            
            # Set timeout for health check
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            try:
                health_status = client.health()
                signal.alarm(0)  # Cancel timeout
                
                logger.debug(f"running _perform_enhanced_health_check ... Health check response: {health_status}")
                
                # Enhanced health validation
                if self._validate_health_response(health_status):
                    logger.debug("running _perform_enhanced_health_check ... Enhanced health check passed")
                    return True
                else:
                    logger.warning("running _perform_enhanced_health_check ... Enhanced health check failed validation")
                    return False
                    
            except TimeoutError:
                signal.alarm(0)
                logger.warning("running _perform_enhanced_health_check ... Health check timed out")
                return False
                
        except Exception as health_error:
            logger.warning(f"running _perform_enhanced_health_check ... Health check failed: {health_error}")
            logger.debug(f"running _perform_enhanced_health_check ... Health check error details: {traceback.format_exc()}")
            return False
    
    def _validate_health_response(self, health_status: Any) -> bool:
        """Validate health check response with multiple criteria"""
        
        if not health_status:
            return False
        
        if isinstance(health_status, dict):
            # Check for success indicators
            success_indicators = ['status', 'health', 'ok', 'success', 'ready']
            
            for indicator in success_indicators:
                value = health_status.get(indicator)
                if value:
                    # Check for positive values
                    if isinstance(value, str) and value.lower() in ['ok', 'healthy', 'ready', 'success', 'true']:
                        return True
                    elif isinstance(value, bool) and value:
                        return True
                    elif isinstance(value, (int, float)) and value > 0:
                        return True
            
            # If no explicit success indicator, check for absence of error indicators
            error_indicators = ['error', 'failed', 'down', 'unavailable']
            has_errors = any(health_status.get(indicator) for indicator in error_indicators)
            
            if not has_errors and health_status:
                logger.debug("running _validate_health_response ... No errors found in health response, assuming healthy")
                return True
        
        elif health_status:  # Non-dict truthy response
            logger.debug("running _validate_health_response ... Non-dict truthy health response, assuming healthy")
            return True
        
        return False
    
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


# Enhanced convenience function with advanced features
def create_and_train_model(
    data: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    model_config: Optional[ModelConfig] = None,
    load_model_path: Optional[str] = None,
    test_size: float = 0.4,
    log_detailed_predictions: bool = True, 
    max_predictions_to_show: int = 20,
    run_name: Optional[str] = None,
    enable_performance_monitoring: bool = True,
    **config_overrides
) -> Dict[str, Any]:
    """
    Enhanced convenience function with advanced monitoring and optimization features
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
    
    # Enhanced plot directory setup
    if run_name:
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        optimization_results_dir = project_root / "optimization_results"
        plot_dir = optimization_results_dir / run_name / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"running create_and_train_model ... Enhanced plot directory: {run_name}")
    else:
        plot_dir = _create_plot_directory(dataset_config, run_timestamp, None)
        logger.debug(f"running create_and_train_model ... Created fallback plot directory")
    
    # Handle model loading vs training
    if load_model_path:
        return _handle_model_loading_enhanced(
            load_model_path, dataset_config, data, 
            log_detailed_predictions, max_predictions_to_show,
            run_timestamp, plot_dir, architecture_type, dataset_name_clean
        )
    else:
        return _handle_model_training_enhanced(
            dataset_config, data, model_config, config_overrides,
            log_detailed_predictions, max_predictions_to_show,
            run_timestamp, plot_dir, run_name, 
            architecture_type, dataset_name_clean, enable_performance_monitoring
        )


def _handle_model_loading_enhanced(
    load_model_path: str,
    dataset_config: DatasetConfig,
    data: Dict[str, Any],
    log_detailed_predictions: bool,
    max_predictions_to_show: int,
    run_timestamp: str,
    plot_dir: Path,
    architecture_type: str,
    dataset_name_clean: str
) -> Dict[str, Any]:
    """Enhanced model loading handler"""
    
    logger.debug(f"running _handle_model_loading_enhanced ... Loading existing model from: {load_model_path}")
    
    model_file = Path(load_model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {load_model_path}")
    
    # Create enhanced ModelBuilder
    builder = ModelBuilder(dataset_config)
    
    with TimedOperation("enhanced model loading", "enhanced_model_builder"):
        builder.model = keras.models.load_model(load_model_path)
        logger.debug("running _handle_model_loading_enhanced ... Model loaded successfully!")
    
    # Enhanced evaluation
    logger.debug("running _handle_model_loading_enhanced ... Starting enhanced evaluation...")
    
    with TimedOperation("enhanced model evaluation", "enhanced_model_builder"):
        test_loss, test_accuracy = builder.evaluate(
            data=data,
            log_detailed_predictions=log_detailed_predictions,
            max_predictions_to_show=max_predictions_to_show,
            run_timestamp=run_timestamp,
            plot_dir=plot_dir
        )
    
    logger.debug(f"running _handle_model_loading_enhanced ... Enhanced evaluation completed:")
    logger.debug(f"running _handle_model_loading_enhanced ... - Test accuracy: {test_accuracy:.4f}")
    logger.debug(f"running _handle_model_loading_enhanced ... - Test loss: {test_loss:.4f}")
    
    return {
        'model_builder': builder,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'model_path': None,
        'plot_dir': str(plot_dir),
        'run_timestamp': run_timestamp,
        'architecture_type': architecture_type,
        'dataset_name': dataset_name_clean,
        'enhanced_features': True
    }


def _handle_model_training_enhanced(
    dataset_config: DatasetConfig,
    data: Dict[str, Any],
    model_config: Optional[ModelConfig],
    config_overrides: Dict[str, Any],
    log_detailed_predictions: bool,
    max_predictions_to_show: int,
    run_timestamp: str,
    plot_dir: Path,
    run_name: Optional[str],
    architecture_type: str,
    dataset_name_clean: str,
    enable_performance_monitoring: bool
) -> Dict[str, Any]:
    """Enhanced model training handler"""
    
    logger.debug("running _handle_model_training_enhanced ... Starting enhanced model training")
    
    # Enhanced model configuration
    if model_config is None:
        model_config = ModelConfig()
    else:
        model_config = copy.deepcopy(model_config)
    
    # Apply enhanced config overrides
    if config_overrides:
        logger.debug(f"running _handle_model_training_enhanced ... Applying enhanced config overrides: {config_overrides}")
        for key, value in config_overrides.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
                logger.debug(f"running _handle_model_training_enhanced ... Enhanced override: {key} = {value}")
            else:
                logger.warning(f"running _handle_model_training_enhanced ... Unknown config parameter: {key}")
    
    # Enhanced feature logging
    _log_enhanced_features(model_config)
    
    # Create enhanced ModelBuilder
    builder = ModelBuilder(dataset_config, model_config)
    builder.plot_dir = plot_dir
    
    # Enhanced training pipeline
    with TimedOperation("enhanced model training pipeline", "enhanced_model_builder"):
        logger.debug("running _handle_model_training_enhanced ... Building enhanced model...")
        builder.build_model()
        
        logger.debug("running _handle_model_training_enhanced ... Training enhanced model...")
        builder.train(data)
        
        logger.debug("running _handle_model_training_enhanced ... Evaluating enhanced model...")
        test_loss, test_accuracy = builder.evaluate(
            data=data,
            log_detailed_predictions=log_detailed_predictions,
            max_predictions_to_show=max_predictions_to_show,
            run_timestamp=run_timestamp,
            plot_dir=plot_dir
        )
        
        logger.debug("running _handle_model_training_enhanced ... Saving enhanced model...")
        model_path = builder.save_model(
            test_accuracy=test_accuracy,
            run_timestamp=run_timestamp,
            run_name=run_name
        )
    
    logger.debug(f"running _handle_model_training_enhanced ... Enhanced training completed:")
    logger.debug(f"running _handle_model_training_enhanced ... - Accuracy: {test_accuracy:.4f}")
    logger.debug(f"running _handle_model_training_enhanced ... - Model saved: {model_path}")
    
    return {
        'model_builder': builder,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'model_path': model_path,
        'plot_dir': str(plot_dir),
        'run_timestamp': run_timestamp,
        'architecture_type': architecture_type,
        'dataset_name': dataset_name_clean,
        'enhanced_features': True,
        'gpu_proxy_used': builder.gpu_proxy_available and model_config.use_gpu_proxy,
        'performance_monitoring': enable_performance_monitoring
    }


def _log_enhanced_features(model_config: ModelConfig) -> None:
    """Log enhanced feature status"""
    
    logger.debug("running _log_enhanced_features ... Enhanced Features Status:")
    
    if model_config.use_gpu_proxy:
        logger.debug("running _log_enhanced_features ... - GPU Proxy: ENABLED")
        logger.debug(f"running _log_enhanced_features ... - gpu_proxy_use_stratified_sampling is: {model_config.gpu_proxy_use_stratified_sampling}")
        logger.debug(f"running _log_enhanced_features ... - gpu_proxy_sample_percentage is: {model_config.gpu_proxy_sample_percentage}")
    else:
        logger.debug("running _log_enhanced_features ... - GPU Proxy: DISABLED")
    
    if model_config.enable_realtime_plots:
        logger.debug("running _log_enhanced_features ... - Real-time Visualization: ENABLED")
    else:
        logger.debug("running _log_enhanced_features ... - Real-time Visualization: DISABLED")
    
    if model_config.enable_gradient_flow_monitoring:
        logger.debug("running _log_enhanced_features ... - Gradient Flow Monitoring: ENABLED")
        logger.debug(f"running _log_enhanced_features ... - Monitoring Frequency: every {model_config.gradient_monitoring_frequency} epochs")
    else:
        logger.debug("running _log_enhanced_features ... - Gradient Flow Monitoring: DISABLED")
    
    if model_config.enable_realtime_weights_bias:
        logger.debug("running _log_enhanced_features ... - Weights/Bias Monitoring: ENABLED")
    else:
        logger.debug("running _log_enhanced_features ... - Weights/Bias Monitoring: DISABLED")


def _create_plot_directory(
    dataset_config: DatasetConfig, 
    run_timestamp: str, 
    run_name: Optional[str] = None
) -> Path:
    """
    Create plot directory with enhanced run_name approach and better organization
    """
    project_root = Path(__file__).resolve().parent.parent
    plots_dir = project_root / "plots"
    
    if run_name:
        plot_dir = plots_dir / run_name
        logger.debug(f"running _create_plot_directory ... Using enhanced run_name: {run_name}")
    else:
        # Enhanced fallback naming
        dataset_name_clean = dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        # Detect architecture type
        if (dataset_config.img_height == 1 and 
            dataset_config.channels == 1 and 
            dataset_config.img_width > 100):
            architecture_name = "LSTM"
        else:
            architecture_name = "CNN"
        
        dir_name = f"{run_timestamp}_{architecture_name}_{dataset_name_clean}_enhanced"
        plot_dir = plots_dir / dir_name
        logger.debug(f"running _create_plot_directory ... Using enhanced fallback: {dir_name}")
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"running _create_plot_directory ... Enhanced plot directory created: {plot_dir}")
    return plot_dir


# Enhanced uncertainty-based sampling for advanced optimization
class UncertaintyBasedSampler:
    """
    Advanced sampling strategy that selects samples based on model uncertainty
    """
    
    def __init__(self, model: keras.Model, threshold: float = 0.1):
        self.model = model
        self.threshold = threshold
        logger.debug(f"running UncertaintyBasedSampler.__init__ ... Initialized with threshold: {threshold}")
    
    def select_uncertain_samples(
        self, 
        x_data: np.ndarray, 
        y_data: np.ndarray, 
        n_samples: int,
        strategy: str = "entropy"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select samples based on prediction uncertainty
        
        Args:
            x_data: Input data
            y_data: True labels
            n_samples: Number of samples to select
            strategy: Uncertainty strategy ('entropy', 'margin', 'variance')
            
        Returns:
            Tuple of (selected_x, selected_y, uncertainty_scores)
        """
        logger.debug(f"running select_uncertain_samples ... Selecting {n_samples} uncertain samples using {strategy} strategy")
        
        # Get model predictions
        predictions = self.model.predict(x_data, verbose=0, batch_size=64)
        
        # Calculate uncertainty scores
        if strategy == "entropy":
            uncertainty_scores = self._calculate_entropy(predictions)
        elif strategy == "margin":
            uncertainty_scores = self._calculate_margin(predictions)
        elif strategy == "variance":
            uncertainty_scores = self._calculate_variance(predictions)
        else:
            logger.warning(f"running select_uncertain_samples ... Unknown strategy {strategy}, using entropy")
            uncertainty_scores = self._calculate_entropy(predictions)
        
        # Select most uncertain samples
        uncertain_indices = np.argsort(uncertainty_scores)[-n_samples:]
        
        logger.debug(f"running select_uncertain_samples ... Selected samples with uncertainty range: "
                    f"{uncertainty_scores[uncertain_indices].min():.3f} - {uncertainty_scores[uncertain_indices].max():.3f}")
        
        return x_data[uncertain_indices], y_data[uncertain_indices], uncertainty_scores[uncertain_indices]
    
    def _calculate_entropy(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate prediction entropy as uncertainty measure"""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        entropy = -np.sum(predictions * np.log(predictions), axis=1)
        return entropy
    
    def _calculate_margin(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate margin between top two predictions"""
        sorted_preds = np.sort(predictions, axis=1)
        margin = sorted_preds[:, -1] - sorted_preds[:, -2]
        
        # Lower margin = higher uncertainty
        return 1 - margin
    
    def _calculate_variance(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate prediction variance"""
        return np.var(predictions, axis=1)


# Enhanced payload compression for large datasets
class PayloadCompressor:
    """
    Advanced payload compression for GPU proxy data transfer
    """
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        logger.debug(f"running PayloadCompressor.__init__ ... Initialized with compression level: {compression_level}")
    
    def compress_data(
        self, 
        data: Dict[str, Any], 
        use_compression: bool = True
    ) -> Dict[str, Any]:
        """
        Compress payload data for efficient transfer
        
        Args:
            data: Data dictionary to compress
            use_compression: Whether to apply compression
            
        Returns:
            Compressed data dictionary with metadata
        """
        if not use_compression:
            logger.debug("running compress_data ... Compression disabled, returning original data")
            return data
        
        logger.debug("running compress_data ... Starting advanced payload compression...")
        
        try:
            import gzip
            import pickle
            import base64
            
            # Separate compressible and non-compressible data
            compressible_data = {
                'x_train': data.get('x_train'),
                'y_train': data.get('y_train')
            }
            
            non_compressible_data = {
                key: value for key, value in data.items() 
                if key not in ['x_train', 'y_train']
            }
            
            # Serialize and compress the large arrays
            serialized = pickle.dumps(compressible_data)
            compressed = gzip.compress(serialized, compresslevel=self.compression_level)
            encoded = base64.b64encode(compressed).decode('utf-8')
            
            # Calculate compression ratio
            original_size = len(serialized)
            compressed_size = len(compressed)
            compression_ratio = compressed_size / original_size
            
            logger.debug(f"running compress_data ... Compression results:")
            logger.debug(f"running compress_data ... - Original size: {original_size / (1024*1024):.1f} MB")
            logger.debug(f"running compress_data ... - Compressed size: {compressed_size / (1024*1024):.1f} MB")
            logger.debug(f"running compress_data ... - Compression ratio: {compression_ratio:.2%}")
            
            # Return compressed payload
            return {
                **non_compressible_data,
                'compressed_data': encoded,
                'compression_metadata': {
                    'compressed': True,
                    'compression_level': self.compression_level,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio
                }
            }
            
        except Exception as e:
            logger.warning(f"running compress_data ... Compression failed: {e}, returning original data")
            return data
    
    def decompress_data(self, compressed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompress payload data
        
        Args:
            compressed_data: Compressed data dictionary
            
        Returns:
            Decompressed data dictionary
        """
        if not compressed_data.get('compression_metadata', {}).get('compressed', False):
            logger.debug("running decompress_data ... Data not compressed, returning as-is")
            return compressed_data
        
        logger.debug("running decompress_data ... Decompressing payload data...")
        
        try:
            import gzip
            import pickle
            import base64
            
            # Extract compressed data
            encoded_data = compressed_data['compressed_data']
            metadata = compressed_data['compression_metadata']
            
            # Decode and decompress
            compressed_bytes = base64.b64decode(encoded_data.encode('utf-8'))
            decompressed_bytes = gzip.decompress(compressed_bytes)
            decompressed_data = pickle.loads(decompressed_bytes)
            
            # Combine with non-compressed data
            result = {
                key: value for key, value in compressed_data.items() 
                if key not in ['compressed_data', 'compression_metadata']
            }
            result.update(decompressed_data)
            
            logger.debug(f"running decompress_data ... Decompression successful, ratio was {metadata['compression_ratio']:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"running decompress_data ... Decompression failed: {e}")
            raise RuntimeError(f"Failed to decompress payload data: {e}")


# Enhanced performance monitoring and metrics collection
class AdvancedPerformanceMonitor:
    """
    Advanced performance monitoring for model training and evaluation
    """
    
    def __init__(self, monitor_name: str = "advanced_monitor"):
        self.monitor_name = monitor_name
        self.metrics = {}
        self.start_times = {}
        logger.debug(f"running AdvancedPerformanceMonitor.__init__ ... Initialized monitor: {monitor_name}")
    
    def start_monitoring(self, operation_name: str) -> None:
        """Start monitoring an operation"""
        import time
        self.start_times[operation_name] = time.time()
        logger.debug(f"running start_monitoring ... Started monitoring: {operation_name}")
    
    def end_monitoring(self, operation_name: str) -> float:
        """End monitoring and return duration"""
        import time
        
        if operation_name not in self.start_times:
            logger.warning(f"running end_monitoring ... Operation {operation_name} was not started")
            return 0.0
        
        duration = time.time() - self.start_times[operation_name]
        self.metrics[operation_name] = duration
        
        logger.debug(f"running end_monitoring ... {operation_name} completed in {duration:.2f} seconds")
        return duration
    
    def add_metric(self, metric_name: str, value: float) -> None:
        """Add a custom metric"""
        self.metrics[metric_name] = value
        logger.debug(f"running add_metric ... Added metric {metric_name}: {value}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        total_time = sum(v for k, v in self.metrics.items() if k.endswith('_time') or 'duration' in k)
        
        summary = {
            'total_execution_time': total_time,
            'individual_metrics': self.metrics.copy(),
            'performance_insights': self._generate_performance_insights()
        }
        
        return summary
    
    def _generate_performance_insights(self) -> List[str]:
        """Generate performance insights based on collected metrics"""
        insights = []
        
        # Identify bottlenecks
        time_metrics = {k: v for k, v in self.metrics.items() if 'time' in k or 'duration' in k}
        
        if time_metrics:
            slowest_operation = max(time_metrics.items(), key=lambda x: x[1])
            insights.append(f"Slowest operation: {slowest_operation[0]} ({slowest_operation[1]:.2f}s)")
            
            # Check for potential GPU proxy benefits
            training_time = time_metrics.get('training_time', 0)
            if training_time > 300:  # 5 minutes
                insights.append("Consider enabling GPU proxy for faster training")
        
        return insights
    
    def log_performance_summary(self) -> None:
        """Log comprehensive performance summary"""
        summary = self.get_performance_summary()
        
        logger.debug(f"running log_performance_summary ... Performance Summary for {self.monitor_name}:")
        logger.debug(f"running log_performance_summary ... - Total execution time: {summary['total_execution_time']:.2f}s")
        
        for metric_name, value in summary['individual_metrics'].items():
            if isinstance(value, float):
                logger.debug(f"running log_performance_summary ... - {metric_name}: {value:.2f}")
            else:
                logger.debug(f"running log_performance_summary ... - {metric_name}: {value}")
        
        for insight in summary['performance_insights']:
            logger.debug(f"running log_performance_summary ... - Insight: {insight}")


# Enhanced GPU proxy training code generation with compression support
def generate_enhanced_gpu_training_code(
    model_config: ModelConfig,
    validation_split: float,
    data_type: str,
    supports_compression: bool = True
) -> str:
    """
    Generate enhanced GPU proxy training code with advanced features
    """
    
    logger.debug(f"running generate_enhanced_gpu_training_code ... Generating {data_type} model code with compression: {supports_compression}")
    
    # Common header with enhanced features
    header_code = f"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json

print("Starting ENHANCED GPU proxy training...")
print(f"TensorFlow version: {{tf.__version__}}")
print(f"GPU devices available: {{tf.config.list_physical_devices('GPU')}}")

# Enhanced memory management
if tf.config.list_physical_devices('GPU'):
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU memory growth enabled")

# Get data and config from context
context_data = context

# Check for compression
if context_data.get('compression_metadata', {{}}).get('compressed', False):
    print("Decompressing payload data...")
    try:
        import gzip
        import pickle
        import base64
        
        # Decompress the data
        encoded_data = context_data['compressed_data']
        compressed_bytes = base64.b64decode(encoded_data.encode('utf-8'))
        decompressed_bytes = gzip.decompress(compressed_bytes)
        decompressed_data = pickle.loads(decompressed_bytes)
        
        # Update context with decompressed data
        context_data.update(decompressed_data)
        
        compression_ratio = context_data['compression_metadata']['compression_ratio']
        print(f"Data decompressed successfully (ratio: {{compression_ratio:.2%}})")
        
    except Exception as decomp_error:
        print(f"Decompression failed: {{decomp_error}}")
        raise

x_train_raw = np.array(context_data['x_train'])
y_train = np.array(context_data['y_train'])
config = context_data['config']
model_config = context_data.get('model_config', {{}})
sampling_info = context_data.get('sampling_info', {{}})

print(f"Enhanced training data shape: {{x_train_raw.shape}}")
print(f"Enhanced labels shape: {{y_train.shape}}")
print(f"Sampling strategy: {{sampling_info.get('strategy', 'unknown')}}")
print(f"Reduction ratio: {{sampling_info.get('reduction_ratio', 1.0):.2%}}")
"""
    
    if data_type == "text":
        return header_code + _generate_enhanced_lstm_code(model_config, validation_split)
    else:
        return header_code + _generate_enhanced_cnn_code(model_config, validation_split)


def _generate_enhanced_cnn_code(model_config: ModelConfig, validation_split: float) -> str:
    """Generate enhanced CNN training code"""
    
    return f"""
# Enhanced CNN data preprocessing
if len(x_train_raw.shape) > 2 and x_train_raw.dtype == np.uint8:
    x_train = x_train_raw.astype(np.float32) / 255.0
    print("Enhanced: Converted uint8 to normalized float32")
elif x_train_raw.max() > 1.0:
    x_train = x_train_raw.astype(np.float32) / 255.0
    print("Enhanced: Normalized image data to 0-1 range")
else:
    x_train = x_train_raw.astype(np.float32)
    print("Enhanced: Using data as-is")

# Build enhanced CNN model with dynamic architecture
print("Building enhanced CNN model...")

model = keras.Sequential()
model.add(keras.layers.Input(shape=config['input_shape']))

# Dynamic convolutional layers
num_conv_layers = model_config.get('num_layers_conv', 2)
filters = model_config.get('filters_per_conv_layer', 32)
kernel_size = tuple(model_config.get('kernel_size', [3, 3]))
activation = model_config.get('activation', 'relu')
use_batch_norm = model_config.get('batch_normalization', False)

print(f"Building {{num_conv_layers}} convolutional layers...")

for i in range(num_conv_layers):
    print(f"Layer {{i+1}}: Conv2D with {{filters}} filters, kernel {{kernel_size}}")
    
    # FIXED: Proper LeakyReLU handling to prevent Remapper Error
    if activation == 'leaky_relu':
        # Create Conv2D without activation, then add separate LeakyReLU layer
        model.add(keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=None,  # No activation in Conv2D layer
            padding='same',
            kernel_initializer='he_normal',
            use_bias=True,  # Ensure bias is used for proper graph construction
            name=f'conv2d_{{i}}'  # Explicit naming for debugging
        ))
        
        # Add separate LeakyReLU activation layer
        model.add(keras.layers.LeakyReLU(
            alpha=0.01, 
            name=f'leaky_relu_{{i}}'
        ))
        print(f"Added Conv2D + separate LeakyReLU layer {{i+1}}")
        
    else:
        # For other activations: Use standard approach
        model.add(keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding='same',
            kernel_initializer='he_normal',
            use_bias=True,
            name=f'conv2d_{{i}}'
        ))
        print(f"Added Conv2D with {{activation}} activation {{i+1}}")
    
    # Optional batch normalization (after activation for proper normalization)
    if use_batch_norm:
        model.add(keras.layers.BatchNormalization(name=f'batch_norm_{{i}}'))
        print(f"Added BatchNormalization layer {{i+1}}")
    
    # Pooling layer
    model.add(keras.layers.MaxPooling2D((2, 2), name=f'max_pooling_{{i}}'))
    print(f"Added MaxPooling2D layer {{i+1}}")

# Enhanced pooling strategy
use_global_pooling = model_config.get('use_global_pooling', False)
if use_global_pooling:
    model.add(keras.layers.GlobalAveragePooling2D())
    print("Using GlobalAveragePooling2D (modern approach)")
else:
    model.add(keras.layers.Flatten())
    print("Using Flatten (traditional approach)")

# Dynamic hidden layers
num_hidden = model_config.get('num_layers_hidden', 1)
hidden_nodes = model_config.get('first_hidden_layer_nodes', 128)
dropout_rate = model_config.get('first_hidden_layer_dropout', 0.5)

print(f"Building {{num_hidden}} hidden layers...")

for i in range(num_hidden):
    model.add(keras.layers.Dense(hidden_nodes, activation=activation))
    model.add(keras.layers.Dropout(dropout_rate))
    print(f"Hidden layer {{i+1}}: {{hidden_nodes}} nodes, {{dropout_rate}} dropout")
    
    # Reduce size for subsequent layers (if any)
    hidden_nodes = max(32, int(hidden_nodes * 0.5))
    dropout_rate = max(0.1, dropout_rate - 0.1)

# Output layer
num_classes = config['num_classes']
model.add(keras.layers.Dense(num_classes, activation='softmax'))
print(f"Output layer: {{num_classes}} classes")

# Enhanced compilation
optimizer = model_config.get('optimizer', 'adam')
loss = model_config.get('loss', 'categorical_crossentropy')
metrics = model_config.get('metrics', ['accuracy'])

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print(f"Enhanced CNN model compiled:")
print(f"- Total parameters: {{model.count_params():,}}")
print(f"- Optimizer: {{optimizer}}")
print(f"- Loss: {{loss}}")

# Enhanced training with adaptive batch size
epochs = config.get('epochs', 10)
batch_size = config.get('batch_size', 32)
validation_split = {validation_split}

print(f"Starting enhanced training:")
print(f"- Epochs: {{epochs}}")
print(f"- Batch size: {{batch_size}}")
print(f"- Validation split: {{validation_split}}")

try:
    # Train with enhanced monitoring
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split=validation_split,
        batch_size=batch_size,
        verbose=1,
        shuffle=True
    )
    
    print("Enhanced CNN training completed successfully!")
    
    # Enhanced result extraction
    final_loss = float(history.history['loss'][-1])
    final_accuracy = float(history.history.get('accuracy', [0])[-1])
    final_val_loss = float(history.history.get('val_loss', [999])[-1])
    final_val_accuracy = float(history.history.get('val_accuracy', [0])[-1])
    
    # Calculate additional metrics
    best_val_accuracy = float(max(history.history.get('val_accuracy', [0])))
    best_val_epoch = int(np.argmax(history.history.get('val_accuracy', [0]))) + 1
    
    print(f"Enhanced results:")
    print(f"- Final training accuracy: {{final_accuracy:.4f}}")
    print(f"- Final validation accuracy: {{final_val_accuracy:.4f}}")
    print(f"- Best validation accuracy: {{best_val_accuracy:.4f}} (epoch {{best_val_epoch}})")
    
    # Enhanced result dictionary
    result = {{
        'training_history': {{k: [float(v) for v in values] for k, values in history.history.items()}},
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy,
        'best_val_accuracy': best_val_accuracy,
        'best_val_epoch': best_val_epoch,
        'epochs_completed': len(history.history['loss']),
        'model_params': int(model.count_params()),
        'gpu_used': len(tf.config.list_physical_devices('GPU')) > 0,
        'execution_location': 'gpu_proxy_enhanced_cnn',
        'sampling_info': sampling_info,
        'model_architecture': 'CNN',
        'enhanced_features': True
    }}
    
    print("Enhanced CNN training completed successfully!")
    
except Exception as e:
    print(f"Enhanced training failed: {{str(e)}}")
    import traceback
    traceback.print_exc()
    
    result = {{
        'training_history': {{}},
        'final_loss': 999.0,
        'final_accuracy': 0.0,
        'final_val_loss': 999.0,  
        'final_val_accuracy': 0.0,
        'best_val_accuracy': 0.0,
        'best_val_epoch': 0,
        'epochs_completed': 0,
        'model_params': 0,
        'gpu_used': len(tf.config.list_physical_devices('GPU')) > 0,
        'execution_location': 'gpu_proxy_enhanced_cnn',
        'error': str(e),
        'sampling_info': sampling_info,
        'model_architecture': 'CNN',
        'enhanced_features': True
    }}

print("Returning enhanced CNN results...")
"""


def _generate_enhanced_lstm_code(model_config: ModelConfig, validation_split: float) -> str:
    """Generate enhanced LSTM training code"""
    
    return f"""
# Enhanced text data preprocessing
print("Processing enhanced text data...")

x_train = x_train_raw.astype(np.int32)
print(f"Text sequences shape: {{x_train.shape}}")

# Build enhanced LSTM model
print("Building enhanced LSTM model...")

sequence_length = config['input_shape'][0] if isinstance(config['input_shape'], (list, tuple)) else config['input_shape']

# Enhanced embedding parameters
vocab_size = model_config.get('vocab_size', 10000)
embedding_dim = model_config.get('embedding_dim', 128)
lstm_units = model_config.get('lstm_units', 64)
use_bidirectional = model_config.get('use_bidirectional', True)
text_dropout = model_config.get('text_dropout', 0.5)

print(f"Enhanced LSTM configuration:")
print(f"- Sequence length: {{sequence_length}}")
print(f"- Vocabulary size: {{vocab_size}}")
print(f"- Embedding dimension: {{embedding_dim}}")
print(f"- LSTM units: {{lstm_units}}")
print(f"- Bidirectional: {{use_bidirectional}}")

model = keras.Sequential()

# Input layer
model.add(keras.layers.Input(shape=(sequence_length,)))

# Enhanced embedding layer
model.add(keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    input_length=sequence_length,
    mask_zero=True
))
print("Added enhanced embedding layer")

# Enhanced LSTM layer
if use_bidirectional:
    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units, 
            dropout=text_dropout, 
            recurrent_dropout=text_dropout/2,
            return_sequences=False
        )
    ))
    print(f"Added bidirectional LSTM: {{lstm_units}} units each direction")
else:
    model.add(keras.layers.LSTM(
        lstm_units, 
        dropout=text_dropout, 
        recurrent_dropout=text_dropout/2,
        return_sequences=False
    ))
    print(f"Added LSTM: {{lstm_units}} units")

# Enhanced dense layers
hidden_nodes = model_config.get('first_hidden_layer_nodes', 128)
hidden_dropout = model_config.get('first_hidden_layer_dropout', 0.5)
activation = model_config.get('hidden_layer_activation_algo', 'relu')

model.add(keras.layers.Dense(hidden_nodes, activation=activation))
model.add(keras.layers.Dropout(hidden_dropout))
print(f"Added dense layer: {{hidden_nodes}} nodes, {{hidden_dropout}} dropout")

# Output layer
num_classes = config['num_classes']
output_activation = 'softmax' if num_classes > 1 else 'sigmoid'
model.add(keras.layers.Dense(num_classes, activation=output_activation))
print(f"Added output layer: {{num_classes}} classes, {{output_activation}} activation")

# Enhanced compilation
optimizer = model_config.get('optimizer', 'adam')
loss = 'categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
metrics = model_config.get('metrics', ['accuracy'])

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print(f"Enhanced LSTM model compiled:")
print(f"- Total parameters: {{model.count_params():,}}")
print(f"- Optimizer: {{optimizer}}")
print(f"- Loss: {{loss}}")

# Enhanced training
epochs = config.get('epochs', 10)
batch_size = config.get('batch_size', 32)
validation_split = {validation_split}

print(f"Starting enhanced LSTM training:")
print(f"- Epochs: {{epochs}}")
print(f"- Batch size: {{batch_size}}")
print(f"- Validation split: {{validation_split}}")

try:
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split=validation_split,
        batch_size=batch_size,
        verbose=1,
        shuffle=True
    )
    
    print("Enhanced LSTM training completed successfully!")
    
    # Enhanced result extraction
    final_loss = float(history.history['loss'][-1])
    final_accuracy = float(history.history.get('accuracy', [0])[-1])
    final_val_loss = float(history.history.get('val_loss', [999])[-1])
    final_val_accuracy = float(history.history.get('val_accuracy', [0])[-1])
    
    # Calculate additional metrics
    best_val_accuracy = float(max(history.history.get('val_accuracy', [0])))
    best_val_epoch = int(np.argmax(history.history.get('val_accuracy', [0]))) + 1
    
    print(f"Enhanced LSTM results:")
    print(f"- Final training accuracy: {{final_accuracy:.4f}}")
    print(f"- Final validation accuracy: {{final_val_accuracy:.4f}}")
    print(f"- Best validation accuracy: {{best_val_accuracy:.4f}} (epoch {{best_val_epoch}})")
    
    result = {{
        'training_history': {{k: [float(v) for v in values] for k, values in history.history.items()}},
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy,
        'best_val_accuracy': best_val_accuracy,
        'best_val_epoch': best_val_epoch,
        'epochs_completed': len(history.history['loss']),
        'model_params': int(model.count_params()),
        'gpu_used': len(tf.config.list_physical_devices('GPU')) > 0,
        'execution_location': 'gpu_proxy_enhanced_lstm',
        'sampling_info': sampling_info,
        'model_architecture': 'LSTM',
        'enhanced_features': True
    }}
    
    print("Enhanced LSTM training completed successfully!")
    
except Exception as e:
    print(f"Enhanced LSTM training failed: {{str(e)}}")
    import traceback
    traceback.print_exc()
    
    result = {{
        'training_history': {{}},
        'final_loss': 999.0,
        'final_accuracy': 0.0,
        'final_val_loss': 999.0,
        'final_val_accuracy': 0.0,
        'best_val_accuracy': 0.0,
        'best_val_epoch': 0,
        'epochs_completed': 0,
        'model_params': 0,
        'gpu_used': len(tf.config.list_physical_devices('GPU')) > 0,
        'execution_location': 'gpu_proxy_enhanced_lstm',
        'error': str(e),
        'sampling_info': sampling_info,
        'model_architecture': 'LSTM',
        'enhanced_features': True
    }}

print("Returning enhanced LSTM results...")
"""


# Main execution for enhanced testing
if __name__ == "__main__":
    logger.debug("running optimized_model_builder.py ... Testing Enhanced ModelBuilder...")
    
    # Enhanced argument parsing with new features
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    
    # Extract dataset name (required)
    dataset_name = args.get('dataset_name', 'cifar10')
    
    # Enhanced type conversion with new parameters
    if 'test_size' in args:
        try:
            args['test_size'] = float(args['test_size'])
        except ValueError:
            logger.warning(f"Invalid test_size: {args['test_size']}")
            del args['test_size']
    
    # Enhanced boolean parameters
    bool_params = [
        'use_global_pooling', 'use_bidirectional', 'log_detailed_predictions', 
        'enable_realtime_plots', 'save_realtime_plots', 'enable_gradient_flow_monitoring',  
        'save_gradient_flow_plots', 'enable_gradient_clipping', 'show_weights_bias_analysis', 
        'enable_realtime_weights_bias', 'batch_normalization', 'use_gpu_proxy',
        'gpu_proxy_auto_clone', 'gpu_proxy_fallback_local', 'gpu_proxy_use_stratified_sampling',
        'gpu_proxy_adaptive_batch_size', 'gpu_proxy_optimize_data_types', 'enable_performance_monitoring'
    ]
    
    for bool_param in bool_params:
        if bool_param in args:
            args[bool_param] = args[bool_param].lower() in ['true', '1', 'yes', 'on']
    
    # Enhanced integer parameters
    int_params = [
        'epochs', 'num_layers_conv', 'filters_per_conv_layer', 'num_layers_hidden', 
        'first_hidden_layer_nodes', 'embedding_dim', 'lstm_units', 'vocab_size', 
        'max_predictions_to_show', 'gradient_monitoring_frequency', 'gradient_history_length', 
        'gradient_sample_size', 'weights_bias_monitoring_frequency', 'gpu_proxy_max_samples',
        'gpu_proxy_min_samples_per_class', 'gpu_proxy_compression_level'
    ]
    
    for int_param in int_params:
        if int_param in args:
            try:
                args[int_param] = int(args[int_param])
            except ValueError:
                logger.warning(f"Invalid {int_param}: {args[int_param]}")
                del args[int_param]
    
    # Enhanced float parameters
    float_params = [
        'first_hidden_layer_dropout', 'subsequent_hidden_layer_dropout_decrease', 
        'subsequent_hidden_layer_nodes_decrease', 'text_dropout', 'gradient_clip_norm', 
        'weights_bias_sample_percentage', 'learning_rate', 'validation_split'
    ]
    
    for float_param in float_params:
        if float_param in args:
            try:
                args[float_param] = float(args[float_param])
            except ValueError:
                logger.warning(f"Invalid {float_param}: {args[float_param]}")
                del args[float_param]
    
    # Enhanced tuple parameters
    for tuple_param in ['kernel_size', 'pool_size']:
        if tuple_param in args:
            try:
                value = args[tuple_param]
                if ',' in value:
                    parts = [int(x.strip()) for x in value.split(',')]
                    if len(parts) == 2:
                        args[tuple_param] = tuple(parts)
                    else:
                        del args[tuple_param]
                else:
                    size = int(value)
                    args[tuple_param] = (size, size)
            except ValueError:
                logger.warning(f"Invalid {tuple_param}: {args[tuple_param]}")
                del args[tuple_param]
    
    # Normalize dataset name
    if 'dataset_name' in args:
        args['dataset_name'] = args['dataset_name'].lower()
    
    logger.debug(f"running optimized_model_builder.py ... Enhanced arguments parsed: {args}")
    
    try:
        # Initialize performance monitor
        perf_monitor = None
        if args.get('enable_performance_monitoring', True):
            perf_monitor = AdvancedPerformanceMonitor("enhanced_model_builder_test")
            perf_monitor.start_monitoring("total_execution_time")
        
        # Enhanced function call with all features
        result = create_and_train_model(**args)
        builder = result['model_builder']
        test_accuracy = result['test_accuracy']
        
        # End performance monitoring
        if perf_monitor is not None:
            perf_monitor.end_monitoring("total_execution_time")
            perf_monitor.add_metric("final_accuracy", test_accuracy)
            perf_monitor.add_metric("gpu_proxy_used", result.get('gpu_proxy_used', False))
            perf_monitor.log_performance_summary()
        
        # Enhanced success logging
        load_path = args.get('load_model_path')
        workflow_msg = f"loaded existing model from {load_path}" if load_path else "trained new enhanced model"
        
        logger.debug(f"running optimized_model_builder.py ... ✅ ENHANCED SUCCESS!")
        logger.debug(f"running optimized_model_builder.py ... Successfully {workflow_msg}")
        logger.debug(f"running optimized_model_builder.py ... Final accuracy: {test_accuracy:.4f}")
        logger.debug(f"running optimized_model_builder.py ... Enhanced features: {result.get('enhanced_features', False)}")
        logger.debug(f"running optimized_model_builder.py ... GPU proxy used: {result.get('gpu_proxy_used', False)}")
        
        # Enhanced feature status logging
        if args.get('enable_gradient_flow_monitoring', False):
            logger.debug("running optimized_model_builder.py ... Enhanced gradient flow monitoring was active")
        
        if args.get('use_gpu_proxy', False):
            logger.debug("running optimized_model_builder.py ... Enhanced GPU proxy integration was enabled")
        
    except Exception as e:
        logger.error(f"running optimized_model_builder.py ... ❌ ENHANCED ERROR: {e}")
        logger.error(f"running optimized_model_builder.py ... Traceback: {traceback.format_exc()}")
        sys.exit(1)