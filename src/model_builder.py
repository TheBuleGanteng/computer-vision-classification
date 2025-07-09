"""
Model Builder for Multi-Modal Classification

Creates and trains neural networks for both image classification (CNN) and 
text classification (LSTM). Automatically detects data type and builds 
appropriate architecture. Designed to work with any dataset configuration 
from DatasetManager.

Supported Architectures:
- CNN: For image data (GTSRB, CIFAR-10, MNIST, etc.)
- LSTM: For text data (IMDB, Reuters, etc.)
"""
import copy
from dataset_manager import DatasetConfig, DatasetManager
import datetime
from plot_creation.realtime_gradient_flow import RealTimeGradientFlowCallback, RealTimeGradientFlowMonitor
from plot_creation.realtime_training_visualization import RealTimeTrainingVisualizer, RealTimeTrainingCallback
from plot_creation.confusion_matrix import ConfusionMatrixAnalyzer
from plot_creation.training_history import TrainingHistoryAnalyzer
from plot_creation.training_animation import TrainingAnimationAnalyzer
from plot_creation.gradient_flow import GradientFlowAnalyzer

from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import tensorflow as tf
from tensorflow import keras # type: ignore
import traceback
from typing import Dict, Any, List, Tuple, Optional, Union
from utils.logger import logger, PerformanceLogger, TimedOperation

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    """
    Illustration of model architecture:
    Raw Image (30x30x3 pixels)
        ↓
    CONVOLUTIONAL LAYERS (Feature Detection)
    ├── Conv Layer 1: 32 filters looking for basic features (edges, colors)
    ├── Pool Layer 1: Reduce size, keep important features
    ├── Conv Layer 2: 32 filters combining basic features into shapes
    └── Pool Layer 2: Further reduction
            ↓
        Flatten: Convert 2D feature maps to 1D list
            ↓
    HIDDEN LAYERS (Decision Making)  
    ├── Dense Layer 1: 128 neurons combining all features
    ├── Dropout: Prevent overfitting
    └── (Optional additional layers with decreasing neurons)
            ↓
        Output Layer: 43 neurons (one per traffic sign class)
        
        
    # Complete architecture (alternative view):
        Input Layer         # Position: Input
        Conv2D             # Type: Convolutional, Position: Hidden  
        MaxPooling2D       # Type: Pooling, Position: Hidden
        Conv2D             # Type: Convolutional, Position: Hidden
        MaxPooling2D       # Type: Pooling, Position: Hidden
        Flatten            # Type: Reshape, Position: Hidden
        Dense              # Type: Dense, Position: Hidden ← THIS IS YOUR "HIDDEN LAYER"
        Dropout            # Type: Regularization, Position: Hidden
        Dense              # Type: Dense, Position: Output ← THIS IS YOUR "OUTPUT LAYER"
    """
    """
    Conceptual example: Stop Sign Recognition
    Convolutional Layers:
        Filter 1: "I detect red color patches here and here"
        Filter 5: "I see octagonal edges forming a shape" 
        Filter 12: "There's white text-like patterns in the center"
        Filter 23: "This has the characteristic red border pattern"
        Hidden Layers:
        Neuron 47: "Filters 1, 5, 12, and 23 are all active → probably a stop sign"
        Neuron 89: "The size and position suggest it's a regulatory sign"
        Neuron 156: "Confidence level: very high this is a stop sign"
    
    This separation allows the network to:
        - Learn reusable features (convolutional layers can detect "red octagon" in any traffic sign dataset)
        - Make dataset-specific decisions (hidden layers learn "red octagon means stop sign in this particular dataset")
    """
    
    # Architecture selection: Determines which model type to build
    """
    Architecture Selection: Determines Model Type
    Purpose: Control whether to build CNN (images) or LSTM (text) architecture
    
    Options:
        - "auto": Automatically detect based on dataset characteristics
        - "cnn": Force CNN architecture (for images)
        - "text": Force text/LSTM architecture (for sequences)
    
    Auto-detection logic:
        - Text indicators: img_height=1, channels=1, img_width>100 (sequence length)
        - Image indicators: img_height>1, channels=3, typical image dimensions
    
    Examples:
        - IMDB: (500, 1, 1) → detects as "text" → builds LSTM
        - CIFAR-10: (32, 32, 3) → detects as "image" → builds CNN
    """
    architecture_type: str = "auto"  # "auto", "cnn", "text"
    
    
    
    """
    Flattening (traditional approach): Heavy computational bottleneck
        Conv2D → Pool → Conv2D → Pool → Flatten → Dense(128) → Output
        Example: (6×6×32 = 1152) × 128 = 147,456 parameters just for first dense layer!
    Gloabal pooling (modern approach): Efficient parameter reduction  
        Conv2D → Pool → Conv2D → Pool → GlobalAveragePooling2D → Dense(43)
        Example: 32 → 43 = only 1,376 parameters (99% reduction!)
    """
    use_global_pooling: bool = False # Use global average pooling instead of flattening (modern CNNs)
    
    # Convolutional layer parameters
    """
    The convolutional layers are used for feature extraction from images
        - Layer 1: Edge detectors - "I see horizontal lines here, vertical lines there"
        - Layer 2: Shape combiners - "Those edges form a circle, those form a triangle"
    Each filter acts like a specialized detective looking for one specific feature, eg. a "Sliding Pattern Detector"
        - One filter scans the entire image using a small window (e.g., 3x3 pixels)
        - Slides systematically across every position: top-left, one pixel right, one pixel right again, etc.
        - Same weights used at every position (parameter sharing)
        - Produces one feature map showing "where did I find my pattern?"
    Relevance for images:
        - Spatial awareness: They understand that nearby pixels are related
        - Translation invariant: Can find a stop sign whether it's top-left or bottom-right
        - Parameter sharing: Same filter scans the entire image (efficient!)
    """
    num_layers_conv: int = 2
    filters_per_conv_layer: int = 32
    kernel_size: Tuple[int, int] = (3, 3) # Each filter looks at 3x3 pixel neighborhoods. Larger kernels (5x5, 7x7) detect bigger patterns but need more computation
    activation: str = "relu" # Activation function applied after each conv layer. Options: "relu" (most common, outputs max(0,x), handles negatives well), "sigmoid" (outputs 0-1, good for probabilities but can cause vanishing gradients), "tanh" (outputs -1 to 1, centered around zero), "leaky_relu" (like relu but allows small negative values), "swish" (smooth, modern alternative to relu)
    pool_size: Tuple[int, int] = (2, 2) # Pooling: Takes max value from 2x2 areas, reduces spatial dimensions, making features more robust and reduces computation
           
    # Text-specific parameters
    """
    Text Model Architecture = "Sequential Understanding"
    Purpose: Process word sequences to understand meaning and context
    Analogy: Like reading a sentence word-by-word while remembering what came before
    
    Text vs Image Fundamental Differences:
        - Images: Spatial relationships (nearby pixels are related)
        - Text: Temporal relationships (word order matters, context builds over time)
        - Images: Fixed 2D grid structure
        - Text: Variable-length sequences of discrete tokens
    
    Text Model Pipeline:
        Raw Text: "The movie was fantastic and well-acted"
            ↓ Tokenization
        Integers: [2, 45, 89, 234, 12, 567]  (word → index mapping)
            ↓ Embedding Layer
        Dense Vectors: Each word becomes 128-dimensional vector, wherein similar words have similar vectors
            ↓ LSTM/Bidirectional LSTM
        Context Understanding: Process sequence, build understanding
            ↓ Dense Layers
        Final Classification: Positive/Negative sentiment
    
    Key Text Components:
        - Embedding: Converts sparse word indices to dense, learnable representations
        - LSTM: Processes sequences while maintaining "memory" of previous words
        - Bidirectional: Reads both forward and backward for better context
    """
    embedding_dim: int = 128           # Size of word embeddings: How many dimensions to represent each word. Common: 50-300. Larger = more expressive but slower. IMDB example: word "fantastic" becomes 128-dim vector like [0.1, -0.4, 0.8, ...]
    lstm_units: int = 64              # Number of LSTM units: LSTM "memory cells" that track sequence context. More units = more memory capacity but slower. Range: 32-512 typical
    vocab_size: int = 10000           # Vocabulary size for text: Number of unique words to track. IMDB uses top 10k most frequent words. Larger vocab = more precise but more parameters
    use_bidirectional: bool = True    # Use bidirectional LSTM: Processes sequences forward AND backward. "I love this movie" vs "This movie I love" - bidirectional catches both patterns
    text_dropout: float = 0.3         # Dropout for text layers: Randomly disable LSTM connections during training. Prevents overfitting to specific word patterns. Range: 0.2-0.6 typical
    
    # Hidden layer parameters
    """
    Hidden Layers = "Decision Makers"
    Purpose: Combine features to make final classifications
    Analogy: Like a judge who listens to all the specialists' reports and makes the final decision
    What they do:
        - Take all detected features and combine them logically (no sliding window here (key difference vs. a filter in a conv. layer), just a full view)
        - Make abstract connections: "Red + octagonal + white text = STOP sign"
        - Final reasoning: "Based on all evidence, this is 95% likely a speed limit sign"
    """
    num_layers_hidden: int = 1
    first_hidden_layer_nodes: int = 128
    subsequent_hidden_layer_nodes_decrease: float = 0.50 # Layer 1: 128 nodes, Layer 2: 128 * 0.50 = 64 nodes, Layer 3: 64 * 0.50 = 32 nodes. Creates a "funnel" effect - broad feature combination → specific decisions
    hidden_layer_activation_algo: str = "leaky_relu" # Activation function applied after each conv layer. Options: "relu" (most common, outputs max(0,x), handles negatives well), "sigmoid" (outputs 0-1, good for probabilities but can cause vanishing gradients), "tanh" (outputs -1 to 1, centered around zero), "leaky_relu" (like relu but allows small negative values), "swish" (smooth, modern alternative to relu)
    first_hidden_layer_dropout: float = 0.3 # Dropout rate for first hidden layer. Randomly sets this fraction of neurons to 0 during training to prevent overfitting. Options: 0.0 (no dropout), 0.1-0.3 (light), 0.4-0.6 (moderate, most common), 0.7-0.9 (heavy, can hurt learning). Higher values = more regularization but slower learning
    subsequent_hidden_layer_dropout_decrease: float = 0.10 # How much to reduce dropout in each subsequent layer. Layer 1: 0.5 dropout, Layer 2: 0.5-0.2=0.3 dropout, Layer 3: 0.3-0.2=0.1 dropout. Rationale: deeper layers need less regularization as they're making more specific decisions
    
    
    # Training parameters
    """
    Training Parameters = "Learning Process Configuration"
    Purpose: Control how the neural network learns from the data
    Analogy: Like setting the rules for how a student studies - how many times to review material, 
            how fast to learn, and how to measure progress

    Training Process Overview:
        1. Forward Pass: Input flows through conv layers → hidden layers → output predictions
        2. Loss Calculation: Compare predictions to actual labels using loss function
        3. Backward Pass: Calculate gradients (how to adjust weights to reduce error)
        4. Weight Update: Optimizer adjusts weights based on gradients
        5. Repeat for specified number of epochs

    Key Training Concepts:
        - Epoch: One complete pass through entire training dataset
        - Batch: Subset of training data processed together (for efficiency)
        - Gradient: Direction and magnitude of weight adjustments needed
        - Learning Rate: How big steps to take when adjusting weights
    """
    epochs: int = 10 # Number of complete passes through training data. More epochs = more learning but risk overfitting. Range: 5-50 typical.
    optimizer: str = "adam" # Algorithm for adjusting weights during training. Options: "adam" (adaptive, most popular, good default), "sgd" (simple, requires learning rate tuning), "rmsprop" (good for RNNs), "adagrad" (adapts to sparse data). Adam combines best of multiple approaches
    loss: str = "categorical_crossentropy" # Function measuring prediction error. For multi-class classification (traffic signs): "categorical_crossentropy" (standard choice). Other options: "sparse_categorical_crossentropy" (if labels are integers not one-hot), "binary_crossentropy" (for binary classification), "mse" (for regression)
    metrics: List[str] = field(default_factory=lambda: ["accuracy"]) # What to track during training beyond loss. "accuracy" = percentage of correct predictions. Other options: "precision", "recall", "f1_score", "top_5_accuracy" (useful for large datasets)
    
    # Evaluation and analysis parameters
    show_confusion_matrix: bool = True    # Generate confusion matrix analysis during evaluation
    show_training_history: bool = True    # Generate training history plots during evaluation
    
    # Real-time visualization parameters
    enable_realtime_plots: bool = True  # Enable/disable real-time training visualization
    save_realtime_plots: bool = True    # Save final real-time plot to disk
    save_intermediate_plots: bool = True  # Save plots during training
    save_plot_every_n_epochs: int = 1    # Save frequency
    
    # Gradient flow analysis parameters
    show_gradient_flow: bool = True           # Enable/disable gradient flow analysis during evaluation
    gradient_flow_sample_size: int = 100      # Number of samples to use for gradient analysis
    enable_gradient_clipping: bool = True     # Enable/disable gradient clipping
    gradient_clip_norm: float = 1.0          # Maximum gradient norm (typical: 0.5-2.0)
    
    # Real-time gradient flow monitoring parameters (ADD THESE)
    enable_gradient_flow_monitoring: bool = False    # Enable/disable gradient flow monitoring
    gradient_monitoring_frequency: int = 1           # Monitor every N epochs (1 = every epoch)
    gradient_history_length: int = 50               # Number of epochs to keep in gradient history
    gradient_sample_size: int = 32                  # Samples used for gradient computation
    save_gradient_flow_plots: bool = True           # Save gradient flow plots to disk
    
    def __post_init__(self) -> None:
        if not self.metrics:
            self.metrics = ["accuracy"]


class ModelBuilder:
    """
    Main class for building and training neural network models

    Supports both CNN (image) and LSTM (text) architectures with automatic
    data type detection and architecture selection.
    """
    
    def __init__(self, dataset_config: DatasetConfig, model_config: Optional[ModelConfig] = None) -> None:
        """
        Initialize ModelBuilder with dataset and model configurations
        
        1. Configuration Storage
        Purpose: Store the "blueprint" for both the data and the model
            - dataset_config: Comes from DatasetManager - tells us image size, number of classes, etc.
            - model_config: Architecture settings - uses your detailed config or defaults
        2. Model State Initialization
            Purpose: Set up placeholders for the actual model and training results
                - model = None: No neural network built yet (that happens in build_model())
                - training_history = None: No training completed yet (that happens in train())
       
        Args:
            dataset_config: Configuration from DatasetManager specifying input shape, classes etc.
            model_config: Optional model architecture configuration (uses defaults if None)
        """
        self.dataset_config: DatasetConfig = dataset_config
        self.model_config: ModelConfig = model_config or ModelConfig()
        self.model: Optional[keras.Model] = None
        self.training_history: Optional[keras.callbacks.History] = None
        self.plot_dir: Optional[Path] = None
        
        
        # Initialize performance logger
        self.perf_logger: PerformanceLogger = PerformanceLogger("model_builder")
        
        logger.debug(f"running class ModelBuilder ... Initialized for dataset: {dataset_config.name}")
        logger.debug(f"running class ModelBuilder ... Input shape: {dataset_config.input_shape}")
        logger.debug(f"running class ModelBuilder ... Number of classes: {dataset_config.num_classes}")
    
    
    def build_model(self) -> keras.Model:
        """
        Build the appropriate model based on dataset and model configurations
        
        Automatically detects data type and builds either:
        - CNN architecture for image data (GTSRB, CIFAR, etc.)
        - LSTM architecture for text data (IMDB, Reuters, etc.)
        
        Returns:
            Compiled Keras model ready for training
        """
        logger.debug("running build_model ... Building model...")
        
        with TimedOperation("model building", "model_builder"): # Tracks how long model construction takes using your logging system
            
            # Detect data type
            if self.model_config.architecture_type == "auto":
                data_type = self._detect_data_type()
            else:
                data_type = self.model_config.architecture_type
            
            # Build appropriate model architecture
            if data_type == "text":
                logger.debug("running build_model ... Building TEXT model architecture")
                self.model = self._build_text_model()
            else:
                logger.debug("running build_model ... Building CNN model architecture")
                self.model = self._build_cnn_model()
                
            
            # Compile model
            assert self.model is not None
            
            
            if self.model_config.enable_gradient_clipping:
                logger.debug(f"running build_model ... Enabling gradient clipping with norm={self.model_config.gradient_clip_norm}")
                
                # Create optimizer with gradient clipping
                if self.model_config.optimizer.lower() == "adam":
                    optimizer = keras.optimizers.Adam(clipnorm=self.model_config.gradient_clip_norm)
                elif self.model_config.optimizer.lower() == "sgd":
                    optimizer = keras.optimizers.SGD(clipnorm=self.model_config.gradient_clip_norm)
                elif self.model_config.optimizer.lower() == "rmsprop":
                    optimizer = keras.optimizers.RMSprop(clipnorm=self.model_config.gradient_clip_norm)
                else:
                    # Fallback: use Adam with clipping for unknown optimizers
                    logger.warning(f"running build_model ... Unknown optimizer '{self.model_config.optimizer}', using Adam with clipping")
                    optimizer = keras.optimizers.Adam(clipnorm=self.model_config.gradient_clip_norm)
            else:
                logger.debug("running build_model ... Gradient clipping disabled, using standard optimizer")
                # Use standard optimizer without clipping
                optimizer = self.model_config.optimizer
            
            # Compile model with the configured optimizer
            self.model.compile(
                optimizer=optimizer,  # Use the configured optimizer (with or without clipping)
                loss=self.model_config.loss,
                metrics=self.model_config.metrics
            )
            
            # Log model summary
            logger.debug("running build_model ... Model architecture created")
            self._log_model_summary()
        
        return self.model
            
        
    def _detect_data_type(self) -> str:
        """
        Detect whether this is image or text data based on dataset configuration
        
        Returns:
            "image" or "text"
        """
        # Check if this looks like text data
        if (self.dataset_config.img_height == 1 and 
            self.dataset_config.channels == 1 and 
            self.dataset_config.img_width > 100):  # Sequence length > 100
            logger.debug(f"running _detect_data_type ... Detected TEXT data: sequence_length={self.dataset_config.img_width}")
            return "text"
        else:
            logger.debug(f"running _detect_data_type ... Detected IMAGE data: shape={self.dataset_config.input_shape}")
            return "image"   
    
    
    
    def _build_cnn_model(self) -> keras.Model:
        """
        Build CNN model for image data (your existing architecture)
        
        Returns:
            CNN model
        """
        # Build convolutional layers
        conv_layers: List[keras.layers.Layer] = self._build_conv_pooling_layers()
        
        # Build hidden layers
        hidden_layers: List[keras.layers.Layer] = self._build_hidden_layers()
        
        # Choose pooling strategy based on configuration
        if self.model_config.use_global_pooling:
            pooling_layer = keras.layers.GlobalAveragePooling2D()
            logger.debug("running _build_cnn_model ... Using GlobalAveragePooling2D (modern architecture)")
        else:
            pooling_layer = keras.layers.Flatten()
            logger.debug("running _build_cnn_model ... Using Flatten (traditional architecture)")
        
        # Create complete CNN model
        model_layers: List[Union[keras.layers.Layer, keras.layers.InputLayer]] = [
            # Input layer with dataset-specific shape
            keras.layers.Input(shape=self.dataset_config.input_shape),
            
            # Convolutional feature extraction layers
            *conv_layers,
            
            # Pooling strategy (flatten or global average pooling, as determined above)
            pooling_layer,
            
            # Hidden layers for classification
            *hidden_layers,
            
            # Output layer
            keras.layers.Dense(
                self.dataset_config.num_classes, 
                activation="softmax"
            )
        ]
        
        """
        Sequential: Creates the actual neural network by connecting all the individual layers into one complete model.
            - Example: Input → Conv1 → Pool1 → Conv2 → Pool2 → Flatten → Dense1 → Dropout → Output
        Before this line, we built a list:
            model_layers = [
                keras.layers.Input(shape=(30, 30, 3)),           # Layer 0: Input
                keras.layers.Conv2D(32, (3, 3), activation="relu"),  # Layer 1: Conv
                keras.layers.MaxPooling2D(pool_size=(2, 2)),         # Layer 2: Pool  
                keras.layers.Conv2D(32, (3, 3), activation="relu"),  # Layer 3: Conv
                keras.layers.MaxPooling2D(pool_size=(2, 2)),         # Layer 4: Pool
                keras.layers.Flatten(),                               # Layer 5: Flatten
                keras.layers.Dense(128, activation="relu"),           # Layer 6: Hidden
                keras.layers.Dropout(0.5),                           # Layer 7: Dropout
                keras.layers.Dense(43, activation="softmax")          # Layer 8: Output
            ]
        Sequential automatically connects them:
            Layer 0 output → Layer 1 input
            Layer 1 output → Layer 2 input  
            Layer 2 output → Layer 3 input
            # ... and so on
        """
        return keras.models.Sequential(model_layers)
    
    
    def _build_text_model(self) -> keras.Model:
        """
        Build text model for sequence data (IMDB, Reuters, etc.)
        
        Architecture:
        Input (sequence_length,) → Embedding → LSTM → Dense → Output
        
        Returns:
            Text model
        """
        sequence_length = self.dataset_config.img_width  # We store sequence length in img_width
        
        model_layers: List[keras.layers.Layer] = [
            # Input layer for integer sequences
            keras.layers.Input(shape=(sequence_length,)),
            
            # Embedding layer: converts integers to dense vectors
            keras.layers.Embedding(
                input_dim=self.model_config.vocab_size,
                output_dim=self.model_config.embedding_dim,
                input_length=sequence_length,
                mask_zero=True  # Handle padding tokens
            ),
            
            # LSTM layer for sequence processing
            keras.layers.LSTM(
                units=self.model_config.lstm_units,
                dropout=self.model_config.text_dropout,
                recurrent_dropout=self.model_config.text_dropout / 2,
                return_sequences=False  # Only return final output
            ) if not self.model_config.use_bidirectional else keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=self.model_config.lstm_units,
                    dropout=self.model_config.text_dropout,
                    recurrent_dropout=self.model_config.text_dropout / 2,
                    return_sequences=False
                )
            ),
            
            # Dense layer for feature combination
            keras.layers.Dense(
                self.model_config.first_hidden_layer_nodes,
                activation=self.model_config.hidden_layer_activation_algo
            ),
            
            # Dropout for regularization
            keras.layers.Dropout(self.model_config.first_hidden_layer_dropout),
            
            # Output layer
            keras.layers.Dense(
                self.dataset_config.num_classes,
                activation="softmax" if self.dataset_config.num_classes > 1 else "sigmoid"
            )
        ]
        
        logger.debug(f"running _build_text_model ... Text model architecture:")
        logger.debug(f"running _build_text_model ... - Sequence length: {sequence_length}")
        logger.debug(f"running _build_text_model ... - Vocab size: {self.model_config.vocab_size}")
        logger.debug(f"running _build_text_model ... - Embedding dim: {self.model_config.embedding_dim}")
        logger.debug(f"running _build_text_model ... - LSTM units: {self.model_config.lstm_units}")
        logger.debug(f"running _build_text_model ... - Bidirectional: {self.model_config.use_bidirectional}")
        
        return keras.models.Sequential(model_layers)    
    
    
    def _build_conv_pooling_layers(self) -> List[keras.layers.Layer]:
        """
        Build convolutional and pooling layers for feature extraction
    
        Creates pairs of (Conv2D + MaxPooling2D) layers based on model_config
        Each pair: Conv detects features → Pool reduces dimensions
        
        Returns:
            List of Keras layers for feature extraction
        """
        
        # Create container to hold all the convolutional and pooling layers
        conv_layers: List[keras.layers.Layer] = []
        
        # Loop to create the specified number of convolutional layers
        logger.debug("running _build_conv_pooling_layers ... Starting to build convolutional and pooling layers...")
        for layer_num in range(self.model_config.num_layers_conv):
            
            # Add convolutional layer
            conv_layer: keras.layers.Conv2D = keras.layers.Conv2D(
                self.model_config.filters_per_conv_layer,
                self.model_config.kernel_size,
                activation=self.model_config.activation
            )
            conv_layers.append(conv_layer)
            
            # Add pooling layer
            """
            # Each filter in Conv2D produces one "feature map"
                Filter 1 → Feature Map 1 (shows where horizontal edges were found)
                Filter 2 → Feature Map 2 (shows where vertical edges were found)  
                Filter 3 → Feature Map 3 (shows where red patches were found)
                ...
                Filter 32 → Feature Map 32 (shows where some pattern was found)
            After convolution, you have lots of detailed spatial information:
                - Input: (30, 30, 3) - Original traffic sign
                - After Conv2D: (28, 28, 32) - 32 feature maps, each 28x28 pixels
            MaxPooling: Keep Only the Strongest Signals
                - Benefits:
                    -  Translation Invariance: Can recognize features regardless of their exact position
                    -  Dimensionality Reduction: Reduces size of feature maps, making them easier to process
                    -  Robustness: Makes model less sensitive to small changes in input (e.g. slight rotations, noise)
            
            Example of MaxPooling:
                Before pooling - Feature Map showing "horizontal edge detections":
                    Original 4x4 feature map:
                    [8  3  1  9]
                    [2  7  4  2] 
                    [5  1  8  3]
                    [6  4  2  7]

                    MaxPooling with (2,2) - take max from each 2x2 area:
                    Top-left 2x2:     Top-right 2x2:
                    [8  3]  → 8       [1  9]  → 9
                    [2  7]            [4  2]

                    Bottom-left 2x2:  Bottom-right 2x2:
                    [5  1]  → 5       [8  3]  → 8  
                    [6  4]            [2  7]

                    Result after pooling:
                    [8  9]
                    [5  8]
            """
            pool_layer: keras.layers.MaxPooling2D = keras.layers.MaxPooling2D(
                pool_size=self.model_config.pool_size
            )
            conv_layers.append(pool_layer)
            
            logger.debug(f"running _build_conv_pooling_layers ... Layer {layer_num + 1}: "
                        f"{self.model_config.filters_per_conv_layer} filters, "
                        f"kernel {self.model_config.kernel_size}, "
                        f"pool {self.model_config.pool_size}")
        
        return conv_layers
    
    
    def _build_hidden_layers(self) -> List[keras.layers.Layer]:
        """
        Build dense hidden layers with dropout
        
        Each hidden layer contains:
            1. a dense layer (to make final classification decisions by combining all detected features from either convolutional layers (images) or LSTM layers (text))
            2. a dropout layer (to prevent overfitting) 
            
        Returns:
            List of Keras layers for classification
        """
        
        # Create container to hold all the hidden layers
        hidden_layers: List[keras.layers.Layer] = []
        
        current_nodes: float = float(self.model_config.first_hidden_layer_nodes)
        current_dropout: float = self.model_config.first_hidden_layer_dropout
        
        # Repeat for the specified number of hidden layers
        """
        Each layer will have decreasing nodes and dropout
            - Example: Layer 1: 128 nodes, 0.5 dropout → Layer 2: 64 nodes, 0.3 dropout → Layer 3: 32 nodes
        This creates a "funnel" effect where each layer narrows down the features and reduces overfitting risk by applying dropout
        """
        logger.debug("running _build_hidden_layers ... Starting to build hidden layers...")
        for layer_num in range(self.model_config.num_layers_hidden):
            
            # Add dense layer
            """
            Dense Layer Connections:
            For CNNs: 128 neurons connect to all flattened feature values from convolutional layers (e.g., if final dimension is 6x6 and 32 filters, then 6 x 6 x 32 = 1152 inputs × 128 neurons = 147,456 weights)
            For LSTMs: 128 neurons connect to LSTM output features (e.g., 128 LSTM units × 128 neurons = 16,384 weights)

            Architecture Examples:
            CNN: Conv → Pool → Conv → Pool → Flatten → Dense(128) → Output
            LSTM: Input → Embedding → LSTM → Dense(128) → Output
            Modern CNN: Conv → Pool → Conv → Pool → GlobalAveragePooling → Output
            """
            dense_layer: keras.layers.Dense = keras.layers.Dense(
                int(current_nodes), 
                activation=self.model_config.hidden_layer_activation_algo
            )
            hidden_layers.append(dense_layer)
            
            # Add dropout layer
            """
            Dropout Layer: Randomly disables neurons
            Conceptual illustration:
                - Network might learn:
                    Neuron 23: "If I see red + circular patterns, it's ALWAYS a stop sign"
                    Neuron 47: "I only activate for stop signs, nothing else matters"
                    Neuron 89: "I depend entirely on Neuron 23's output"
                    Problem: Too specialized, doesn't generalize to new images
                - With dropout:
                    Training Step 1: Neurons 23, 47 randomly disabled
                    Network forced to learn: "Other neurons must also detect stop signs"
                    Training Step 2: Neuron 89 randomly disabled  
                    Network forced to learn: "Can't rely on just one neuron for decisions"
                    Result: Multiple neurons learn to detect each pattern (redundancy)
            Dropout decreases with each layer because the deeper layers are making more specific decisions and thus require less regularization (i.e. they don't need to be as well-rounded as prior layers)
            """
            dropout_layer: keras.layers.Dropout = keras.layers.Dropout(current_dropout)
            hidden_layers.append(dropout_layer)
            
            logger.debug(f"running _build_hidden_layers ... Layer {layer_num + 1}: "
                        f"{int(current_nodes)} nodes, {current_dropout:.2f} dropout")
            
            # Calculate next layer's parameters
            current_nodes = max(8.0, current_nodes * self.model_config.subsequent_hidden_layer_nodes_decrease)
            current_dropout = max(0.1, current_dropout - self.model_config.subsequent_hidden_layer_dropout_decrease)
        
        return hidden_layers
    
    
    def train(
        self, 
        data: Dict[str, Any], 
        validation_split: float = 0.2
        ) -> keras.callbacks.History:
        """
        Train the model on provided data
        
        Args:
            data: Dataset dictionary from DatasetManager with x_train, y_train, x_test, y_test
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Training history object
        """
        if self.model is None:
            logger.debug("running train ... No model found, building model first...")
            self.build_model()
        
        # Type guard: ensure model is not None after build_model
        assert self.model is not None        
        logger.debug("running train ... Starting model training...")
        
        # Log dataset information
        self.perf_logger.log_data_info(
            total_images=len(data['x_train']) + len(data['x_test']),
            train_size=len(data['x_train']),
            test_size=len(data['x_test']),
            num_categories=self.dataset_config.num_classes
        )
        
        # Log model parameters
        model_params: Dict[str, Union[str, Tuple[int, int, int], int]] = {
            'dataset': self.dataset_config.name,
            'input_shape': self.dataset_config.input_shape,
            'num_classes': self.dataset_config.num_classes,
            'conv_layers': self.model_config.num_layers_conv,
            'conv_filters': self.model_config.filters_per_conv_layer,
            'hidden_layers': self.model_config.num_layers_hidden,
            'hidden_nodes': self.model_config.first_hidden_layer_nodes,
            'epochs': self.model_config.epochs
        }
        self.perf_logger.log_model_params(model_params)
        
        # Train model with timing
        """
        # This is what Keras does INSIDE model.fit():
            def fit(self, x_train, y_train, epochs, validation_split, verbose):
                
                # Split data for validation
                train_data, val_data = split_data(x_train, y_train, validation_split)
                
                # 5-step learning cycle:
                for epoch in range(epochs):  # ← This loop is INSIDE Keras
                    for batch in create_batches(train_data):  # ← This loop is INSIDE Keras
                        # 1. Forward pass - INSIDE Keras
                        predictions = self.predict(batch_x)
                        
                        # 2. Loss calculation - INSIDE Keras  
                        loss = self.loss_function(predictions, batch_y)
                        
                        # 3. Backward pass - INSIDE Keras
                        gradients = calculate_gradients(loss)
                        
                        # 4. Weight updates - INSIDE Keras
                        self.optimizer.apply_gradients(gradients)
                    
                    # 5. Validation - INSIDE Keras
                    val_loss, val_accuracy = self.evaluate(val_data)
                    
                    if verbose:
                        print(f"Epoch {epoch}: loss={loss}, val_loss={val_loss}")
                
                return training_history
        
        training_history contains:
            {
                'loss': [2.1, 1.8, 1.4, 1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2],
                'accuracy': [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.90, 0.92],
                'val_loss': [2.3, 1.9, 1.5, 1.1, 0.9, 0.7, 0.5, 0.4, 0.35, 0.3],
                'val_accuracy': [0.35, 0.48, 0.58, 0.68, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89]
            }
            
        Gradient Descent Analogy:
            Think of training like learning to ride a bike:

            Forward pass: Try to ride (make predictions)
            Loss calculation: Measure how much you wobbled (how wrong you were)
            Backward pass: Analyze what went wrong (calculate gradients)
            Weight update: Adjust your technique (update neural network weights)
            Repeat: Try again with slight improvements
        """
        # Set up callbacks list
        callbacks_list = []
        
        # Add real-time visualization if enabled
        realtime_visualizer = None
        if self.model_config.enable_realtime_plots:
            logger.debug("running train ... Setting up real-time training visualization...")
            
            # Use the provided plot_dir instead of creating a new one
            if self.plot_dir is None:
                # Fallback: create directory if not provided (shouldn't happen in normal flow)
                dataset_name_clean = self.dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
                data_type = self._detect_data_type()
                architecture_name = "CNN" if data_type == "image" else "LSTM"
                run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                project_root = Path(__file__).resolve().parent.parent
                plots_dir = project_root / "plots"
                plot_dir = plots_dir / f"{run_timestamp}_{architecture_name}_{dataset_name_clean}"
                plot_dir.mkdir(parents=True, exist_ok=True)
                logger.debug("running train ... Created fallback plot directory")
            else:
                plot_dir = self.plot_dir
                logger.debug(f"running train ... Using provided plot directory: {plot_dir}")

            realtime_visualizer = RealTimeTrainingVisualizer(self, plot_dir)           
            
            # Configure intermediate saving
            realtime_visualizer.save_intermediate_plots = self.model_config.save_intermediate_plots
            realtime_visualizer.save_every_n_epochs = self.model_config.save_plot_every_n_epochs
            
            realtime_callback = RealTimeTrainingCallback(realtime_visualizer)
            callbacks_list.append(realtime_callback)
            logger.debug("running train ... Real-time visualization enabled")
        
        # Add gradient flow monitoring if enabled (NEW SECTION)
        gradient_flow_monitor = None
        if self.model_config.enable_gradient_flow_monitoring:
            logger.debug("running train ... Setting up real-time gradient flow monitoring...")
            
            # Use the same plot_dir as training visualization
            if self.plot_dir is None:
                # Fallback: create directory if not provided (shouldn't happen in normal flow)
                dataset_name_clean = self.dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
                data_type = self._detect_data_type()
                architecture_name = "CNN" if data_type == "image" else "LSTM"
                run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                project_root = Path(__file__).resolve().parent.parent
                plots_dir = project_root / "plots"
                plot_dir = plots_dir / f"{run_timestamp}_{architecture_name}_{dataset_name_clean}"
                plot_dir.mkdir(parents=True, exist_ok=True)
                logger.debug("running train ... Created fallback plot directory for gradient monitoring")
            else:
                plot_dir = self.plot_dir
                logger.debug(f"running train ... Using provided plot directory for gradient monitoring: {plot_dir}")
            
            # Create gradient flow monitor
            gradient_flow_monitor = RealTimeGradientFlowMonitor(
                model_builder=self,
                plot_dir=plot_dir,  # This should be the main run directory
                monitoring_frequency=self.model_config.gradient_monitoring_frequency,
                history_length=self.model_config.gradient_history_length,
                sample_size=self.model_config.gradient_sample_size
            )
            
            # Setup monitoring with training data - ENSURE THIS WORKS
            training_data = (data['x_train'], data['y_train'])
            try:
                gradient_flow_monitor.setup_monitoring(training_data)
                logger.debug(f"running train ... Gradient monitoring setup successful: is_monitoring={gradient_flow_monitor.is_monitoring}")
            except Exception as setup_error:
                logger.error(f"running train ... Gradient monitoring setup failed: {setup_error}")
                gradient_flow_monitor = None  # Disable monitoring if setup fails
            
            # Only create callback if setup was successful
            if gradient_flow_monitor is not None and gradient_flow_monitor.is_monitoring:
                gradient_flow_callback = RealTimeGradientFlowCallback(gradient_flow_monitor)
                callbacks_list.append(gradient_flow_callback)  # <-- FIX: This was missing!
                logger.debug("running train ... Real-time gradient flow monitoring enabled")
            else:
                logger.warning("running train ... Gradient flow monitoring disabled due to setup failure")
                gradient_flow_monitor = None
            
            # Log gradient monitoring configuration
            logger.debug(f"running train ... Gradient monitoring frequency: every {self.model_config.gradient_monitoring_frequency} epochs")
            logger.debug(f"running train ... Gradient history length: {self.model_config.gradient_history_length} epochs")
            logger.debug(f"running train ... Gradient sample size: {self.model_config.gradient_sample_size}")   
            
        # Train model with timing and callbacks
        with TimedOperation("model training", "model_builder"):
            self.training_history = self.model.fit(
                data['x_train'], 
                data['y_train'],
                epochs=self.model_config.epochs,
                validation_split=validation_split,
                verbose=1,  # Show progress bars
                callbacks=callbacks_list  # Include real-time visualization
            )
        
        logger.debug("running train ... Training completed")
        return self.training_history
    
    
    def evaluate(
        self, 
        data: Dict[str, Any], 
        log_detailed_predictions: bool, 
        max_predictions_to_show: int,
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None
        ) -> Tuple[float, float]:
        """
        Evaluate model performance on test data
        Evaluation is read-only - the model doesn't learn or change during evaluation.
        
        Args:
            data: Dataset dictionary from DatasetManager
            log_detailed_predictions: Whether to show individual prediction results
            max_predictions_to_show: Maximum number of individual predictions to log
            
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluation")
        
        logger.debug("running evaluate ... Evaluating model on test data...")
        
        # Type guard: we know model is not None here due to the check above
        model: keras.Model = self.model
        
        # Create timestamp if not provided
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        
        
        # If no plot directory provided, use default
        if plot_dir is None:
            plot_dir = Path("plots") / run_timestamp
            plot_dir.mkdir(parents=True, exist_ok=True)        
        
        """
        # What Keras does internally in model.evaluate():
            def evaluate(self, x_test, y_test, verbose):
                total_loss = 0
                total_correct = 0
                total_samples = 0
                
                # Process test data in batches (no training, just prediction)
                for batch_x, batch_y in create_batches(x_test, y_test):
                    # 1. Forward pass only (no backward pass!)
                    predictions = self.predict(batch_x)
                    
                    # 2. Calculate loss (for monitoring, not training)
                    batch_loss = self.loss_function(predictions, batch_y)
                    
                    # 3. Calculate accuracy
                    batch_accuracy = calculate_accuracy(predictions, batch_y)
                    
                    # 4. Accumulate results
                    total_loss += batch_loss
                    total_correct += batch_accuracy * len(batch_x)
                    total_samples += len(batch_x)
                
                # 5. Return average metrics
                avg_loss = total_loss / num_batches
                avg_accuracy = total_correct / total_samples
                
                return [avg_loss, avg_accuracy]  # List format
                
        test_loss interpretation ("How Confident Am I?", lower is better):
            0.0 = Perfect predictions (impossible in practice)
            0.1-0.3 = Very good model
            0.5-1.0 = Decent model, room for improvement
            2.0+ = Poor model, needs work
            Loss = -log(0.99) = 0.010
                Note: The use of -log is meant to apply exponentially increasing penality for increasing uncertainty
            

        # accuracy interpretation ("Am I Right or Wrong?", higher is better):
            # 95%+ = Excellent
            # 90-95% = Very good
            # 85-90% = Good
            # <85% = Needs improvement for safety-critical applications
        """
        with TimedOperation("model evaluation", "model_builder"):
            evaluation_results = model.evaluate(
                data['x_test'], 
                data['y_test'],
                verbose=1
            )
            
            # Handle both single metric and multiple metrics cases
            if isinstance(evaluation_results, list):
                test_loss: float = float(evaluation_results[0])
                test_accuracy: float = float(evaluation_results[1]) if len(evaluation_results) > 1 else 0.0
            else:
                test_loss = float(evaluation_results)
                test_accuracy = 0.0
        
        # Generate confusion matrix analysis if enabled
        if self.model_config.show_confusion_matrix:
            logger.debug("running evaluate ... Generating confusion matrix analysis...")
            try:
                # Get predictions for confusion matrix
                predictions = model.predict(data['x_test'], verbose=0)
                
                # Convert one-hot encoded labels back to class indices
                if data['y_test'].ndim > 1 and data['y_test'].shape[1] > 1:
                    true_labels = np.argmax(data['y_test'], axis=1)
                else:
                    true_labels = data['y_test'].flatten()
                
                predicted_labels = np.argmax(predictions, axis=1)
                
                # Create confusion matrix analyzer
                class_names = self.dataset_config.class_names or [f"Class_{i}" for i in range(self.dataset_config.num_classes)]
                cm_analyzer = ConfusionMatrixAnalyzer(class_names=class_names)
                
                # Perform comprehensive analysis and visualization
                analysis_results = cm_analyzer.analyze_and_visualize(
                    true_labels=true_labels,
                    predicted_labels=predicted_labels,
                    dataset_name=self.dataset_config.name,
                    run_timestamp=run_timestamp,
                    plot_dir=plot_dir
                )
                
                # Log results
                if 'error' in analysis_results:
                    logger.warning(f"running evaluate ... Confusion matrix analysis failed: {analysis_results['error']}")
                else:
                    logger.debug("running evaluate ... Confusion matrix analysis completed successfully")
                    
                    # Log key results
                    overall_accuracy = analysis_results.get('overall_accuracy', 0.0)
                    logger.debug(f"running evaluate ... Confusion matrix accuracy: {overall_accuracy:.4f}")
                    
                    viz_path = analysis_results.get('visualization_path')
                    if viz_path:
                        logger.debug(f"running evaluate ... Confusion matrix saved to: {viz_path}")
                    
            except Exception as cm_error:
                logger.warning(f"running evaluate ... Failed to create confusion matrix: {cm_error}")
                logger.debug(f"running evaluate ... Confusion matrix error traceback: {traceback.format_exc()}")
        
        # Generate training history plots if enabled using the new modular approach
        if self.model_config.show_training_history:
            try:
                logger.debug("running evaluate ... Generating training history analysis...")
                
                # Check if training history is available
                if self.training_history is not None:
                    # Create training history analyzer
                    history_analyzer = TrainingHistoryAnalyzer(model_name=self.dataset_config.name)
                    
                    # Perform comprehensive analysis and visualization
                    history_results = history_analyzer.analyze_and_visualize(
                        training_history=self.training_history.history,
                        model=model,
                        dataset_name=self.dataset_config.name,
                        run_timestamp=run_timestamp,
                        plot_dir=plot_dir
                    )
                    
                    # Log results
                    if 'error' in history_results:
                        logger.warning(f"running evaluate ... Training history analysis failed: {history_results['error']}")
                    else:
                        logger.debug("running evaluate ... Training history analysis completed successfully")
                        
                        # Log key insights
                        insights = history_results.get('training_insights', [])
                        for insight in insights:
                            logger.debug(f"running evaluate ... Training insight: {insight}")
                        
                        # Log overfitting detection
                        if history_results.get('overfitting_detected', False):
                            logger.warning("running evaluate ... Overfitting detected in training history")
                        
                        viz_path = history_results.get('visualization_path')
                        if viz_path:
                            logger.debug(f"running evaluate ... Training history saved to: {viz_path}")
                else:
                    logger.warning("running evaluate ... No training history available for analysis")
                    
            except Exception as plot_error:
                logger.warning(f"running evaluate ... Failed to create training history analysis: {plot_error}")
                logger.debug(f"running evaluate ... Training history error traceback: {traceback.format_exc()}")
        
        
        # Generate training animation if enabled and training history is available
        if self.training_history is not None:
            try:
                logger.debug("running evaluate ... Generating training animation...")
                
                # Create training animation analyzer
                animation_analyzer = TrainingAnimationAnalyzer(model_name=self.dataset_config.name)
                
                # Perform animation creation
                animation_results = animation_analyzer.analyze_and_animate(
                    training_history=self.training_history.history,
                    model=model,
                    dataset_name=self.dataset_config.name,
                    run_timestamp=run_timestamp,
                    plot_dir=plot_dir,
                    animation_duration=10.0,  # 10 second animation
                    fps=10  # 10 frames per second
                )
                
                # Log results
                if 'error' in animation_results:
                    logger.warning(f"running evaluate ... Training animation creation failed: {animation_results['error']}")
                else:
                    logger.debug("running evaluate ... Training animation creation completed successfully")
                    
                    # Log animation insights
                    insights = animation_results.get('animation_insights', [])
                    for insight in insights:
                        logger.debug(f"running evaluate ... Animation insight: {insight}")
                    
                    # Log file paths
                    gif_path = animation_results.get('gif_path')
                    mp4_path = animation_results.get('mp4_path')
                    frame_count = animation_results.get('frame_count', 0)
                    
                    if gif_path:
                        logger.debug(f"running evaluate ... Training animation GIF saved to: {gif_path}")
                    if mp4_path:
                        logger.debug(f"running evaluate ... Training animation MP4 saved to: {mp4_path}")
                        
                    logger.debug(f"running evaluate ... Animation generated with {frame_count} frames")
                    
            except Exception as animation_error:
                logger.warning(f"running evaluate ... Failed to create training animation: {animation_error}")
                logger.debug(f"running evaluate ... Animation error traceback: {traceback.format_exc()}")
        else:
            logger.debug("running evaluate ... No training history available for animation creation")
        
        
        # Generate gradient flow analysis if enabled and model is available
        if self.model_config.show_gradient_flow and self.model is not None:
            try:
                logger.debug("running evaluate ... Generating gradient flow analysis...")
                
                # Prepare sample data for gradient analysis
                # Use a subset of test data for performance
                sample_size = min(self.model_config.gradient_flow_sample_size, len(data['x_test']))
                sample_indices = np.random.choice(len(data['x_test']), sample_size, replace=False)
                sample_x = data['x_test'][sample_indices]
                sample_y = data['y_test'][sample_indices]
                
                # Create gradient flow analyzer
                gradient_analyzer = GradientFlowAnalyzer(model_name=self.dataset_config.name)
                
                # Perform gradient flow analysis
                gradient_results = gradient_analyzer.analyze_and_visualize(
                    model=model,
                    sample_data=sample_x,
                    sample_labels=sample_y,
                    dataset_name=self.dataset_config.name,
                    run_timestamp=run_timestamp,
                    plot_dir=plot_dir
                )
                
                # Log results
                if 'error' in gradient_results:
                    logger.warning(f"running evaluate ... Gradient flow analysis failed: {gradient_results['error']}")
                else:
                    logger.debug("running evaluate ... Gradient flow analysis completed successfully")
                    
                    # Log key insights
                    gradient_health = gradient_results.get('gradient_health', 'unknown')
                    logger.debug(f"running evaluate ... Gradient health assessment: {gradient_health}")
                    
                    # Log recommendations
                    recommendations = gradient_results.get('recommendations', [])
                    if recommendations:
                        logger.debug("running evaluate ... Gradient flow recommendations:")
                        for rec in recommendations[:3]:  # Show first 3 recommendations
                            logger.debug(f"running evaluate ... - {rec}")
                    
                    # Log visualization paths
                    viz_paths = gradient_results.get('visualization_paths', [])
                    for path in viz_paths:
                        logger.debug(f"running evaluate ... Gradient flow visualization saved to: {path}")
                        
            except Exception as gradient_error:
                logger.warning(f"running evaluate ... Failed to create gradient flow analysis: {gradient_error}")
                logger.debug(f"running evaluate ... Gradient flow error traceback: {traceback.format_exc()}")
        
        
        # Show detailed individual predictions if requested
        if log_detailed_predictions and max_predictions_to_show > 0:
            self._log_detailed_predictions(
                data=data, 
                max_predictions_to_show=max_predictions_to_show,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir
            )        
        
        logger.debug(f"running evaluate ... Test accuracy: {test_accuracy:.4f}")
        logger.debug(f"running evaluate ... Test loss: {test_loss:.4f}")        
        return test_loss, test_accuracy

                
    def _log_detailed_predictions(
        self, 
        data: Dict[str, Any], 
        max_predictions_to_show: int,
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None
        ) -> None:
        """
        Log detailed per-prediction results with visual feedback and confusion matrix
        
        Args:
            data: Dataset dictionary from DatasetManager
            max_predictions_to_show: Maximum number of predictions to show
        """
        if self.model is None:
            return
            
        logger.debug("running _log_detailed_predictions ... Generating detailed prediction analysis...")
        
        # If no timestamp provided, create one
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            
        # If no plot directory provided, use default
        if plot_dir is None:
            plot_dir = Path("plots") / run_timestamp
            plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model predictions
        predictions = self.model.predict(data['x_test'], verbose=0)
        
        # Convert one-hot encoded labels back to class indices
        if data['y_test'].ndim > 1 and data['y_test'].shape[1] > 1:
            true_labels = np.argmax(data['y_test'], axis=1)
        else:
            true_labels = data['y_test'].flatten()
        
        predicted_labels = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        # Get class names
        class_names = self.dataset_config.class_names or [f"Class_{i}" for i in range(self.dataset_config.num_classes)]
        
        # Separate correct and incorrect predictions
        correct_indices = []
        incorrect_indices = []
        
        for i in range(len(true_labels)):
            if true_labels[i] == predicted_labels[i]:
                correct_indices.append(i)
            else:
                incorrect_indices.append(i)
        
        # Show some correct predictions first
        max_correct_to_show = min(5, len(correct_indices), max_predictions_to_show // 2)
        if max_correct_to_show > 0:
            logger.debug("running _log_detailed_predictions ... Correct predictions (sample):")
            for i in range(max_correct_to_show):
                idx = correct_indices[i]
                confidence = confidence_scores[idx]
                true_class = class_names[true_labels[idx]]
                predicted_class = class_names[predicted_labels[idx]]
                
                logger.debug(f"running _log_detailed_predictions ... ✅ Correct: predicted: {predicted_class}, actual: {true_class}, confidence: {confidence:.3f}")
        
        # Show incorrect predictions with more detail
        max_incorrect_to_show = min(max_predictions_to_show - max_correct_to_show, len(incorrect_indices))
        if max_incorrect_to_show > 0:
            logger.debug("running _log_detailed_predictions ... Incorrect predictions:")
            for i in range(max_incorrect_to_show):
                idx = incorrect_indices[i]
                confidence = confidence_scores[idx]
                true_class = class_names[true_labels[idx]]
                predicted_class = class_names[predicted_labels[idx]]
                
                logger.debug(f"running _log_detailed_predictions ... ❌ Incorrect: predicted: {predicted_class}, actual: {true_class}, confidence: {confidence:.3f}")
        
        # Summary statistics
        total_predictions = len(true_labels)
        correct_count = len(correct_indices)
        incorrect_count = len(incorrect_indices)
        
        logger.debug(f"running _log_detailed_predictions ... Prediction summary:")
        logger.debug(f"running _log_detailed_predictions ... - Total predictions: {total_predictions}")
        logger.debug(f"running _log_detailed_predictions ... - Correct: {correct_count} ({correct_count/total_predictions*100:.1f}%)")
        logger.debug(f"running _log_detailed_predictions ... - Incorrect: {incorrect_count} ({incorrect_count/total_predictions*100:.1f}%)")           
       
        logger.debug("running _log_detailed_predictions ... Detailed prediction analysis completed")
        return None


    def create_confusion_matrix(
        self, 
        true_labels: np.ndarray, 
        predicted_labels: np.ndarray,
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None
        ) -> None:
        """
        Create, analyze, and visualize confusion matrix with detailed statistics
        
        A confusion matrix is a comprehensive performance evaluation tool that shows:
        - How often each class was correctly predicted (diagonal elements)
        - Which classes are most commonly confused with each other (off-diagonal elements)
        - Per-class performance metrics (precision and recall)
        
        Think of it as a "mistake analysis report" that helps you understand:
        1. Which traffic signs your model recognizes well
        2. Which signs it confuses (e.g., does it mix up speed limit signs?)
        3. Whether certain classes are harder to detect than others
        
        Matrix Structure Example (3-class problem):
                        PREDICTED
                    Stop  Yield  Speed
        ACTUAL  Stop  [95    2     3  ]  ← 95 stops correctly identified, 2 confused as yield, 3 as speed
            Yield  [ 5   88     7  ]  ← 5 yields confused as stop, 88 correct, 7 as speed  
            Speed  [ 1    4    89  ]  ← 1 speed sign confused as stop, 4 as yield, 89 correct
        
        Key Metrics Calculated:
        - Precision: Of all times model said "stop sign", how often was it actually a stop sign?
        Formula: True Positives / (True Positives + False Positives)
        Example: Stop precision = 95 / (95 + 5 + 1) = 94.1%
        
        - Recall: Of all actual stop signs, how many did the model correctly identify?
        Formula: True Positives / (True Positives + False Negatives)  
        Example: Stop recall = 95 / (95 + 2 + 3) = 95.0%
        
        What Good vs Bad Results Look Like:
        ✅ Good: High numbers on diagonal, low numbers off diagonal
        ❌ Bad: Scattered values, indicating frequent misclassifications
        
        Common Patterns to Watch For:
        - Systematic confusion: Speed limit signs consistently confused with each other
        - Class imbalance effects: Rare signs performing poorly due to limited training data
        - Similar-looking signs: Yield vs warning signs having cross-confusion
        
        The method automatically identifies:
        1. Top 5 most common misclassifications (helps focus improvement efforts)
        2. Best and worst performing classes (guides data collection priorities)
        3. Overall accuracy from matrix diagonal (validation of model performance)
        
        Output Logging Examples:
        "Most common misclassifications:"
        "1. Speed_Limit_30 → Speed_Limit_50: 12 times"
        "2. Yield → Warning_General: 8 times"
        
        "Top performing classes:"
        "1. Stop_Sign: recall=0.987, precision=0.995 (450 samples)"
        "2. No_Entry: recall=0.978, precision=0.989 (234 samples)"
        
        Troubleshooting Interpretation:
        - Low precision: Model is "trigger-happy" - says this class too often
        - Low recall: Model is "conservative" - misses many instances of this class
        - Both low: Class is genuinely difficult or needs more training data
        
        Args:
            true_labels: True class labels (1D array of integers representing actual traffic sign classes)
                        Example: [0, 1, 2, 0, 1] where 0=stop, 1=yield, 2=speed_limit
            predicted_labels: Predicted class labels (1D array of integers from model output)
                            Example: [0, 1, 1, 0, 1] showing one misclassification (2→1)
            run_timestamp: Optional timestamp for when this analysis was run (default is current time)
                            
        Side Effects:
            - Logs detailed confusion matrix analysis to console
            - Calls plot_confusion_matrix() for visual representation (if ≤20 classes)
            - Shows top misclassifications and per-class performance statistics
            
        Example Usage Context:
            After model evaluation, this method helps answer questions like:
            - "Why did my traffic sign classifier get 85% accuracy but fail in real testing?"
            - "Which speed limit signs are most commonly confused?"
            - "Should I collect more data for warning signs vs stop signs?"
        
        Note: 
            For datasets with >20 classes, visual matrix is skipped to maintain readability,
            but statistical analysis is still performed and logged.
        """
        logger.debug("running create_confusion_matrix ... Generating confusion matrix analysis...")

        # Create timestamp if not provided
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                
        try:
            # Create confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels)
            
            # Basic confusion matrix statistics
            total_correct_cm = np.trace(cm)  # Sum of diagonal elements
            total_predictions_cm = np.sum(cm)
            accuracy_cm = total_correct_cm / total_predictions_cm
            
            logger.debug(f"running create_confusion_matrix ... Confusion matrix statistics:")
            logger.debug(f"running create_confusion_matrix ... - Matrix shape: {cm.shape}")
            logger.debug(f"running create_confusion_matrix ... - Total correct (diagonal sum): {total_correct_cm}")
            logger.debug(f"running create_confusion_matrix ... - Total predictions: {total_predictions_cm}")
            logger.debug(f"running create_confusion_matrix ... - Accuracy from CM: {accuracy_cm:.4f}")
            
            # Find most confused classes (highest off-diagonal values)
            logger.debug("running create_confusion_matrix ... Most common misclassifications:")
            
            # Create a copy of confusion matrix with diagonal set to 0 to find off-diagonal maxima
            cm_off_diagonal = cm.copy()
            np.fill_diagonal(cm_off_diagonal, 0)
            
            # Get class names
            class_names = self.dataset_config.class_names or [f"Class_{i}" for i in range(self.dataset_config.num_classes)]
            
            # Find top 5 misclassifications
            top_misclassifications = []
            for i in range(min(5, cm.shape[0] * cm.shape[1])):
                max_idx = np.unravel_index(np.argmax(cm_off_diagonal), cm_off_diagonal.shape)
                true_idx, pred_idx = max_idx
                count = cm_off_diagonal[true_idx, pred_idx]
                
                if count > 0:  # Only show if there are misclassifications
                    true_class = class_names[true_idx] if true_idx < len(class_names) else f"Class_{true_idx}"
                    pred_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
                    top_misclassifications.append((true_class, pred_class, count))
                    cm_off_diagonal[true_idx, pred_idx] = 0  # Remove this entry for next iteration
                else:
                    break
            
            for i, (true_class, pred_class, count) in enumerate(top_misclassifications, 1):
                logger.debug(f"running create_confusion_matrix ... {i}. {true_class} → {pred_class}: {count} times")
            
            # Per-class accuracy (precision for each class)
            logger.debug("running create_confusion_matrix ... Per-class performance (top 10 and bottom 10):")
            
            class_accuracies = []
            for i in range(cm.shape[0]):
                true_positives = cm[i, i]
                total_actual = np.sum(cm[i, :])  # Total actual instances of this class
                total_predicted = np.sum(cm[:, i])  # Total predicted instances of this class
                
                # Precision: TP / (TP + FP) = TP / total_predicted
                precision = true_positives / total_predicted if total_predicted > 0 else 0.0
                
                # Recall: TP / (TP + FN) = TP / total_actual  
                recall = true_positives / total_actual if total_actual > 0 else 0.0
                
                class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
                class_accuracies.append((class_name, recall, precision, total_actual))
            
            # Sort by recall (descending) and show top 10
            class_accuracies_sorted = sorted(class_accuracies, key=lambda x: x[1], reverse=True)
            
            logger.debug("running create_confusion_matrix ... Top 10 performing classes (by recall):")
            for i, (class_name, recall, precision, count) in enumerate(class_accuracies_sorted[:10], 1):
                logger.debug(f"running create_confusion_matrix ... {i:2d}. {class_name:20s}: recall={recall:.3f}, precision={precision:.3f} ({count} samples)")
            
            logger.debug("running create_confusion_matrix ... Bottom 10 performing classes (by recall):")
            for i, (class_name, recall, precision, count) in enumerate(class_accuracies_sorted[-10:], 1):
                logger.debug(f"running create_confusion_matrix ... {i:2d}. {class_name:20s}: recall={recall:.3f}, precision={precision:.3f} ({count} samples)")
            
            # Decide whether to display visual confusion matrix
            num_classes = len(class_names)
            if num_classes <= 20:
                logger.debug("running create_confusion_matrix ... Displaying confusion matrix visualization...")
                try:
                    self.plot_confusion_matrix(
                        cm=cm, 
                        class_names=class_names,
                        run_timestamp=run_timestamp,
                        plot_dir=plot_dir
                        )
                except Exception as plot_error:
                    logger.warning(f"running create_confusion_matrix ... Failed to plot confusion matrix: {plot_error}")
            else:
                logger.debug(f"running create_confusion_matrix ... Skipping confusion matrix visualization ({num_classes} classes too many for readable display)")
                
        except Exception as e:
            logger.warning(f"running create_confusion_matrix ... Failed to generate confusion matrix analysis: {e}")
            logger.debug(f"running create_confusion_matrix ... Confusion matrix error traceback: {traceback.format_exc()}")
        
        logger.debug("running create_confusion_matrix ... Confusion matrix analysis completed")


    def plot_confusion_matrix(
        self, cm: np.ndarray, 
        class_names: List[str],
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None
        ) -> None:
        """
        Visualize confusion matrix as heatmap and save to file for detailed analysis
        
        Creates a professional-quality visual confusion matrix that makes patterns 
        immediately apparent through color coding and annotations. This is the visual
        companion to create_confusion_matrix()'s statistical analysis.
        
        Visual Design Elements:
        - Heatmap colors: Darker blue = higher values (more predictions)
        - Diagonal: Shows correct predictions (ideally the darkest cells)
        - Off-diagonal: Shows mistakes (ideally lighter/white)
        - Numbers in cells: Exact count of predictions for that true/predicted combination
        - Axis labels: Class names for easy interpretation
        
        How to Read the Visualization:
        
        Perfect Model Example:
        ```
                Predicted
                Stop Yield Speed
        Actual Stop [100   0    0  ]  ← Dark blue diagonal
            Yield [  0  95    0  ]  ← Light/white off-diagonal  
            Speed [  0   0   88  ]  ← Perfect classification
        ```
        
        Problematic Model Example:
        ```
                Predicted  
                Stop Yield Speed
        Actual Stop [ 60  25   15 ]  ← Many stop signs misclassified
            Yield [ 30  50   15 ]  ← Yield signs confused with stop
            Speed [ 10  20   45 ]  ← Speed signs performing poorly
        ```
        
        Key Visual Patterns to Identify:
        
        1. **Strong Diagonal**: Dark blue line from top-left to bottom-right
        → Indicates good overall performance
        
        2. **Scattered Heat**: Colors spread throughout matrix
        → Indicates poor performance, model is guessing randomly
        
        3. **Cluster Patterns**: Groups of confusion between similar classes
        → Example: All speed limit signs confused with each other
        → Solution: More diverse training data or better feature extraction
        
        4. **Row/Column Dominance**: One row very light, one column very dark
        → Row dominance: Model never predicts this class (conservative)
        → Column dominance: Model over-predicts this class (trigger-happy)
        
        File Output Details:
        - Saves high-resolution PNG for presentations/papers
        - Filename format: "confusion_matrix_YYYYMMDD_HHMMSS_dataset_name.png"
        - For >10 classes: Also saves ultra-high-res version for detailed examination
        - Files saved to: project_root/plots/ directory
        
        Professional Usage:
        - Include in research papers to show model performance
        - Use in presentations to explain model behavior to stakeholders
        - Compare matrices before/after model improvements
        - Share with domain experts to validate misclassifications make sense
        
        Troubleshooting Visual Patterns:
        - **Checkered pattern**: Class imbalance - some classes have much more data
        - **Vertical/horizontal stripes**: Systematic bias toward certain predictions
        - **Block patterns**: Model learned to distinguish groups but not individuals
        
        Args:
            cm: Confusion matrix as 2D numpy array where cm[i,j] represents 
                the number of samples of true class i predicted as class j
                Shape: (n_classes, n_classes)
                
            class_names: List of human-readable class names for axis labels
                        Example: ['Stop_Sign', 'Yield_Sign', 'Speed_Limit_30']
                        Length must match cm.shape[0] and cm.shape[1]
        
        Side Effects:
            - Creates matplotlib figure with confusion matrix heatmap
            - Saves PNG file(s) to plots/ directory with timestamp
            - Closes matplotlib figure to free memory
            - Logs save location and any errors to console
            
        Technical Details:
            - Uses seaborn heatmap for professional appearance
            - Colormap: 'Blues' (white=0, dark blue=maximum)
            - Annotations: Integer counts in each cell
            - Figure size: 10x8 inches for readability
            - DPI: 300 for publication quality, 600 for high-res version
            
        Example File Outputs:
            - "confusion_matrix_20250708_143022_cifar10.png" (standard resolution)
            - "confusion_matrix_20250708_143022_cifar10_highres.png" (detailed view)
        """
        
        # Create timestamp if not provided
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label') 
        plt.title('Confusion Matrix')
        
        # Generate filename with timestamp and dataset name
        dataset_name_clean = self.dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        filename = f"confusion_matrix_{run_timestamp}_{dataset_name_clean}.png"
        
        
        # Use provided plot_dir or create fallback
        if plot_dir is not None:
            # Use the provided directory (should be the run-specific directory)
            save_dir = plot_dir
            logger.debug(f"running plot_confusion_matrix ... Using provided plot directory: {save_dir}")
        else:
            # Fallback: create our own directory (shouldn't happen in normal flow)
            data_type = self._detect_data_type()
            architecture_name = "CNN" if data_type == "image" else "LSTM"
            
            project_root: Path = Path(__file__).resolve().parent.parent
            plots_dir = project_root / "plots"
            save_dir = plots_dir / f"{run_timestamp}_{architecture_name}_{dataset_name_clean}"
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"running plot_confusion_matrix ... Created fallback plot directory: {save_dir}")
        
        # Generate final filepath
        filepath = save_dir / filename
        logger.debug(f"running plot_confusion_matrix ... Saving confusion matrix to filepath: {filepath}")      
        
        try:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.debug(f"running plot_confusion_matrix ... Confusion matrix saved to: {filepath}")
            
            # Also save a high-res version for detailed viewing
            if len(class_names) > 10:
                high_res_filename = f"confusion_matrix_{run_timestamp}_{dataset_name_clean}_highres.png"
                high_res_filepath = save_dir / high_res_filename
                plt.savefig(high_res_filepath, dpi=600, bbox_inches='tight')
                logger.debug(f"running plot_confusion_matrix ... High-res version saved to: {high_res_filepath}")
                
        except Exception as save_error:
            logger.warning(f"running plot_confusion_matrix ... Failed to save confusion matrix: {save_error}")
        
        finally:
            plt.close()  # Clean up memory
   


    def plot_weight_distributions(self) -> None:
        """
        Show histograms of weights in each layer over training
        Helps detect dead neurons or weight saturation
        """
        pass

    def plot_activation_maps(
        self, 
        sample_images: np.ndarray
        ) -> None:
        """
        Visualize what each layer 'sees' for sample inputs
        Shows feature maps from convolutional layers
        """
        pass
    
        
    
    def save_model(
        self, 
        test_accuracy: Optional[float] = None,
        run_timestamp: Optional[str] = None
        ) -> None:
        """
        Save the trained model to disk
        
        Model formats:
            - Modern .keras format benefits:
                - Single file (easier to manage)
                - Better compression (smaller file size)
                - Improved security (safer loading)
                - Future-proof (TensorFlow's standard)

            - Legacy .h5 format:
                - Older HDF5 format
                - Still supported but not recommended for new projects
                - May have compatibility issues in future versions
                
        What gets saved: Complete model state
            1. Architecture: All layer definitions and connections
            2. Weights: All learned parameters (millions of numbers)
            3. Optimizer state: Adam optimizer's internal variables
            4. Compilation info: Loss function, metrics, optimizer settings
            5. Training config: How the model was configured for training
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train a model first.")
        
        # Type guard: we know model is not None here
        model: keras.Model = self.model
        
        # Create timestamp if not provided
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        
        # Determine model architecture type for filename
        data_type = self._detect_data_type()
        architecture_name = "CNN" if data_type == "image" else "LSTM"
        
        # Create run-specific subdirectory
        dataset_name_clean = self.dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        project_root: Path = Path(__file__).resolve().parent.parent
        plots_dir = project_root / "plots"
        run_subdir = plots_dir / f"{run_timestamp}_{architecture_name}_{dataset_name_clean}"
        run_subdir.mkdir(parents=True, exist_ok=True)
        
        # Clean dataset name for filename (remove spaces, special chars)
        dataset_name_clean = self.dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        # Convert to lowercase and remove common words for shorter names
        dataset_name_clean = dataset_name_clean.lower()
        if "dataset" in dataset_name_clean:
            dataset_name_clean = dataset_name_clean.replace("_dataset", "")
        
        
        # Auto-generate filename based on results:
        accuracy_str = f"{test_accuracy:.1f}".replace(".", "_")
        filename = f"model_{run_timestamp}_{architecture_name}_{dataset_name_clean}_acc_{accuracy_str}.keras" # e.g., "model_20250107_143022_acc_94_2.keras"
        
        # Determine filepath
        project_root: Path = Path(__file__).resolve().parent.parent
        models_dir = project_root / "saved_models"
        
        # Create directory if it doesn't exist
        models_dir.mkdir(exist_ok=True)
        
        final_filepath = models_dir / filename
                
        logger.debug(f"running save_model ... Saving model to {final_filepath}")
        self.model.save(final_filepath)
        logger.debug(f"running save_model ... Model saved successfully")
        
        logger.debug(f"running save_model ... Model details:")
        logger.debug(f"running save_model ... - Dataset: {self.dataset_config.name}")
        logger.debug(f"running save_model ... - Architecture: {architecture_name}")
        logger.debug(f"running save_model ... - Input shape: {self.dataset_config.input_shape}")
        logger.debug(f"running save_model ... - Classes: {self.dataset_config.num_classes}")
        logger.debug(f"running save_model ... - Test accuracy: {test_accuracy:.4f}" if test_accuracy else "running save_model ... - Test accuracy: Not provided")
        
    
    def load_model(self, filepath: str) -> keras.Model:
        """
        Load a saved model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded Keras model
        """
        logger.debug(f"running load_model ... Loading model from {filepath}")
        self.model = keras.models.load_model(filepath)
        logger.debug(f"running load_model ... Model loaded successfully")
        return self.model
    
    
    def _log_model_summary(self) -> None:
        """Log model architecture summary with explanatory information"""
        if self.model is None:
            return
        
        # Get model summary as string
        summary_lines: List[str] = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        
        logger.debug("running _log_model_summary ... Model Architecture:")
        for line in summary_lines:
            logger.debug(f"running _log_model_summary ... {line}")
        
        # Add explanatory information for each layer
        logger.debug("running _log_model_summary ... Layer Explanations:")
        
        for i, layer in enumerate(self.model.layers):
            layer_type = type(layer).__name__
            # Use the model's layer output shape, which is available after model is built
            try:
                # Get output shape from the model's layer configuration
                output_shape = self.model.layers[i].output.shape
            except:
                # Fallback if output shape is not available
                output_shape = "unknown"
            
            if layer_type == "Conv2D":
                # output_shape example: (None, 30, 30, 16)
                # None = batch size (dynamic), 30x30 = spatial dimensions, 16 = number of filters
                if output_shape != "unknown" and len(output_shape) > 3:
                    filters = output_shape[-1]
                    height = output_shape[1]
                    width = output_shape[2]
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Conv2D) - "
                                f"Output: (batch_size={output_shape[0]}, height={height}, width={width}, filters={filters}) - "
                                f"Detects {filters} different feature patterns in {height}x{width} spatial dimensions")
                else:
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Conv2D) - "
                                f"Feature detection layer with {getattr(layer, 'filters', 'unknown')} filters")
            
            elif layer_type == "MaxPooling2D":
                # output_shape example: (None, 15, 15, 16) 
                # Reduces spatial dimensions while keeping same number of channels
                if output_shape != "unknown" and len(output_shape) > 3:
                    height = output_shape[1]
                    width = output_shape[2]
                    channels = output_shape[-1]
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (MaxPooling2D) - "
                                f"Output: (batch_size={output_shape[0]}, height={height}, width={width}, channels={channels}) - "
                                f"Reduces spatial size to {height}x{width}, keeps {channels} feature maps, retains strongest signals")
                else:
                    pool_size = getattr(layer, 'pool_size', 'unknown')
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (MaxPooling2D) - "
                                f"Downsampling layer with pool_size={pool_size}, retains strongest signals")
            
            elif layer_type == "Flatten":
                # output_shape example: (None, 3600)
                # Converts 2D feature maps to 1D vector for dense layers
                if output_shape != "unknown" and len(output_shape) > 1:
                    total_features = output_shape[-1]
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Flatten) - "
                                f"Output: (batch_size={output_shape[0]}, features={total_features}) - "
                                f"Converts 2D feature maps into 1D vector of {total_features} values for dense layers")
                else:
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Flatten) - "
                                f"Converts 2D feature maps into 1D vector for dense layers")
            
            elif layer_type == "Dense":
                # output_shape example: (None, 32) or (None, 10)
                # Fully connected layer with specified number of neurons
                if output_shape != "unknown" and len(output_shape) > 1:
                    neurons = output_shape[-1]
                    if i == len(self.model.layers) - 1:  # Last layer (output)
                        logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Dense/Output) - "
                                    f"Output: (batch_size={output_shape[0]}, classes={neurons}) - "
                                    f"Final classification layer with {neurons} neurons (one per class)")
                    else:  # Hidden layer
                        logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Dense/Hidden) - "
                                    f"Output: (batch_size={output_shape[0]}, neurons={neurons}) - "
                                    f"Hidden layer with {neurons} neurons for feature combination and decision making")
                else:
                    units = getattr(layer, 'units', 'unknown')
                    if i == len(self.model.layers) - 1:
                        logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Dense/Output) - "
                                    f"Final classification layer with {units} neurons")
                    else:
                        logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Dense/Hidden) - "
                                    f"Hidden layer with {units} neurons for feature combination")
            
            elif layer_type == "Dropout":
                # output_shape example: (None, 32)
                # Same shape as input, but randomly zeros some values during training
                dropout_rate = getattr(layer, 'rate', 'unknown')
                if output_shape != "unknown" and len(output_shape) > 1:
                    neurons = output_shape[-1]
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Dropout) - "
                                f"Output: (batch_size={output_shape[0]}, neurons={neurons}) - "
                                f"Regularization layer, randomly disables {dropout_rate*100 if dropout_rate != 'unknown' else 'unknown'}% of {neurons} neurons during training to prevent overfitting")
                else:
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Dropout) - "
                                f"Regularization layer, randomly disables {dropout_rate*100 if dropout_rate != 'unknown' else 'unknown'}% of neurons during training to prevent overfitting")
            
            elif layer_type == "InputLayer" or layer_type == "Input":
                # Handle both image and text input shapes
                if output_shape != "unknown":
                    if len(output_shape) > 3:  # Image data: (batch, height, width, channels)
                        height = output_shape[1]
                        width = output_shape[2]
                        channels = output_shape[-1]
                        logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Input) - "
                                    f"Output: (batch_size={output_shape[0]}, height={height}, width={width}, channels={channels}) - "
                                    f"Input layer expecting {height}x{width} images with {channels} color channels")
                    elif len(output_shape) == 2:  # Text data: (batch, sequence_length)
                        seq_length = output_shape[1]
                        logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Input) - "
                                    f"Output: (batch_size={output_shape[0]}, sequence_length={seq_length}) - "
                                    f"Input layer expecting text sequences of {seq_length} word indices")
                    else:
                        logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Input) - "
                                    f"Output: {output_shape} - Input layer")
                else:
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Input) - "
                                f"Input layer for data")
                       
            elif layer_type == "Embedding":
                # output_shape example: (None, 500, 128)
                # None = batch size, 500 = sequence length, 128 = embedding dimension
                if output_shape != "unknown" and len(output_shape) > 2:
                    seq_length = output_shape[1]
                    embed_dim = output_shape[2]
                    vocab_size = getattr(layer, 'input_dim', 'unknown')
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Embedding) - "
                                f"Output: (batch_size={output_shape[0]}, seq_length={seq_length}, embedding_dim={embed_dim}) - "
                                f"Converts {vocab_size} word indices to {embed_dim}-dimensional dense vectors, enabling semantic understanding")
                else:
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Embedding) - "
                                f"Word-to-vector conversion layer for text processing")

            elif layer_type == "LSTM":
                # output_shape example: (None, 64) when return_sequences=False
                if output_shape != "unknown" and len(output_shape) > 1:
                    units = output_shape[-1]
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (LSTM) - "
                                f"Output: (batch_size={output_shape[0]}, units={units}) - "
                                f"Sequential processor with {units} memory cells, reads text left-to-right while maintaining context")
                else:
                    units = getattr(layer, 'units', 'unknown')
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (LSTM) - "
                                f"Sequential text processor with {units} memory cells for context understanding")

            elif layer_type == "Bidirectional":
                # output_shape example: (None, 128) - double the LSTM units due to forward + backward
                if output_shape != "unknown" and len(output_shape) > 1:
                    total_units = output_shape[-1]
                    lstm_units = total_units // 2  # Bidirectional doubles the output size
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Bidirectional) - "
                                f"Output: (batch_size={output_shape[0]}, units={total_units}) - "
                                f"Bidirectional LSTM with {lstm_units} units each direction, reads text both forward and backward for enhanced context")
                else:
                    logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} (Bidirectional) - "
                                f"Bidirectional LSTM layer processing sequences in both directions")
            
            
            
            
            
            
            
            
            
            
            else:
                # Generic layer information for any other layer types
                logger.debug(f"running _log_model_summary ... Layer {i}: {layer.name} ({layer_type}) - "
                            f"Output: {output_shape} - "
                            f"Layer type: {layer_type}")
        
        # Log key metrics
        total_params: int = self.model.count_params()
        logger.debug(f"running _log_model_summary ... Total parameters: {total_params:,} - "
                    f"These are the weights and biases the model learns during training")


# Convenience function for easy usage
# Convenience function for easy usage
def create_and_train_model(
    data: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    model_config: Optional[ModelConfig] = None,
    load_model_path: Optional[str] = None,
    test_size: float = 0.4,
    log_detailed_predictions: bool = True, 
    max_predictions_to_show: int = 20,
    **config_overrides
) -> Tuple[ModelBuilder, float]:
    """
    Convenience function to create, train, and evaluate a model in one call
    
    Args:
        data: Dataset dictionary from DatasetManager (optional if dataset_name provided)
        dataset_name: Name of dataset to load (optional if data provided)
        model_config: Optional model configuration
        load_model_path: Optional path to existing model to load (NEW)
        test_size: Fraction of data to use for testing (only used if dataset_name provided)
        log_detailed_predictions: Whether to show individual prediction results
        max_predictions_to_show: Maximum number of individual predictions to log
        **config_overrides: Any ModelConfig parameters to override (ignored if loading existing model)
        
    Returns:
        Tuple of (ModelBuilder instance, test_accuracy)
        
    Examples:
        - Use via command line arguments to specify parameters
            - Example (trains new model): python model_builder.py dataset_name=cifar10 use_global_pooling=true epochs=15
            - Example (loads existing model): python src/model_builder.py load_model_path=/home/thebuleganteng/01_Repos/06_personal_work/computer-vision-classification/saved_models/model_20250708_122719_acc_0_3.keras dataset_name=cifar100 test_size=0.1
    
        - Option 1: Use dataset name with config overrides (most convenient)
        builder, accuracy = create_and_train_model(
            dataset_name='cifar10',
            use_global_pooling=True,
            epochs=15,
            filters_per_conv_layer=64
        )
        
        - Option 2: Use dataset name with ModelConfig object
        config = ModelConfig(use_global_pooling=True, epochs=20)
        builder, accuracy = create_and_train_model(
            dataset_name='cifar10',
            model_config=config
        )
        
        - Option 3: Mix ModelConfig with overrides (overrides take precedence)
        base_config = ModelConfig(epochs=10)
        builder, accuracy = create_and_train_model(
            dataset_name='imdb',
            model_config=base_config,
            epochs=5,  # This overrides the epochs=10 in base_config
            embedding_dim=256
        )
        
        - Option 4: Pre-loaded data
        manager = DatasetManager()
        data = manager.load_dataset('cifar10')
        builder, accuracy = create_and_train_model(
            data=data,
            use_global_pooling=True
        )
        
        - Option 5: Quick experiments
        for pooling in [True, False]:
            builder, acc = create_and_train_model(
                dataset_name='fashion_mnist',
                use_global_pooling=pooling,
                epochs=5
            )
            print(f"Global pooling {pooling}: {acc:.4f}")
    """
    # Obtain start of run timestamp to be used for saving model and plots
    run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    
    # Validate arguments
    if data is None and dataset_name is None:
        raise ValueError("Must provide either 'data' or 'dataset_name'")
    
    if data is not None and dataset_name is not None:
        raise ValueError("Provide either 'data' OR 'dataset_name', not both")
    
    # Load data if dataset_name provided
    if dataset_name is not None:
        logger.debug(f"running create_and_train_model ... Loading dataset: {dataset_name}")
        manager = DatasetManager()
        
        # Check if dataset is available
        if dataset_name not in manager.get_available_datasets():
            available = ', '.join(manager.get_available_datasets())
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")
        
        try:
            data = manager.load_dataset(dataset_name, test_size=test_size)
            logger.debug(f"running create_and_train_model ... Dataset {dataset_name} loaded successfully")
        except Exception as e:
            logger.error(f"running create_and_train_model ... Failed to load dataset {dataset_name}: {e}")
            raise
    
    # Type guard: at this point data is guaranteed to not be None
    assert data is not None
    
    # Get dataset config from data
    dataset_config = data['config']
    
    # Handling for loading existing model
    if load_model_path:
        logger.debug(f"running create_and_train_model ... Loading existing model from: {load_model_path}")
        
        # Validate model file exists        
        model_file = Path(load_model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {load_model_path}")
        
        # Create ModelBuilder and load existing model
        builder = ModelBuilder(dataset_config)
        model = builder.load_model(load_model_path)
        logger.debug("running create_and_train_model ... Existing model loaded successfully!")
        
        # Create the plot directory once
        dataset_name_clean = dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        data_type = builder._detect_data_type()
        architecture_name = "CNN" if data_type == "image" else "LSTM"
        project_root = Path(__file__).resolve().parent.parent
        plots_dir = project_root / "plots"
        plot_dir = plots_dir / f"{run_timestamp}_{architecture_name}_{dataset_name_clean}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate the loaded model
        logger.debug("running create_and_train_model ... Evaluating loaded model...")
        test_loss, test_accuracy = builder.evaluate(
            data=data,
            log_detailed_predictions=log_detailed_predictions,
            max_predictions_to_show=max_predictions_to_show,
            plot_dir=plot_dir
        )
        
        logger.debug(f"running create_and_train_model ... Loaded model performance:")
        logger.debug(f"running create_and_train_model ... - Test accuracy: {test_accuracy:.4f}")
        logger.debug(f"running create_and_train_model ... - Test loss: {test_loss:.4f}")
        
        return builder, test_accuracy     
        
        
    # Handling for loading data and creating a new model
    else:
        logger.debug("running create_and_train_model ... No existing model to load, creating new model")
        # Create or modify model configuration
        if model_config is None:
            model_config = ModelConfig()
        else:
            # Create a copy to avoid modifying the original            
            model_config = copy.deepcopy(model_config)
        
        ## Apply any config overrides (now includes real-time visualization options)
        if config_overrides:
            logger.debug(f"running create_and_train_model ... Applying config overrides: {config_overrides}")
            for key, value in config_overrides.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
                    logger.debug(f"running create_and_train_model ... Set {key} = {value}")
                else:
                    logger.warning(f"running create_and_train_model ... Unknown config parameter: {key}")
        
        # Log real-time visualization status
        if model_config.enable_realtime_plots:
            logger.debug("running create_and_train_model ... Real-time training visualization ENABLED")
        else:
            logger.debug("running create_and_train_model ... Real-time training visualization DISABLED")
        
        # Log gradient flow monitoring status (NEW)
        if model_config.enable_gradient_flow_monitoring:
            logger.debug("running create_and_train_model ... Real-time gradient flow monitoring ENABLED")
            logger.debug(f"running create_and_train_model ... Gradient monitoring frequency: every {model_config.gradient_monitoring_frequency} epochs")
            logger.debug(f"running create_and_train_model ... Gradient history length: {model_config.gradient_history_length} epochs")
            logger.debug(f"running create_and_train_model ... Gradient sample size: {model_config.gradient_sample_size}")
        else:
            logger.debug("running create_and_train_model ... Real-time gradient flow monitoring DISABLED")
        
        # Create model builder
        builder = ModelBuilder(dataset_config, model_config)
        
        # Create the plot directory BEFORE building/training model
        dataset_name_clean = dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        data_type = builder._detect_data_type()
        architecture_name = "CNN" if data_type == "image" else "LSTM"
        project_root = Path(__file__).resolve().parent.parent
        plots_dir = project_root / "plots"
        plot_dir = plots_dir / f"{run_timestamp}_{architecture_name}_{dataset_name_clean}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # PASS the plot_dir to the ModelBuilder so it can use it for real-time plots
        builder.plot_dir = plot_dir  # Add this attribute to store the plot directory
        
        # Build and train model
        logger.debug("running create_and_train_model ... Building and training model...")
        builder.build_model()
        builder.train(data)
        
        # Create the plot directory once
        dataset_name_clean = dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        data_type = builder._detect_data_type()
        architecture_name = "CNN" if data_type == "image" else "LSTM"
        project_root = Path(__file__).resolve().parent.parent
        plots_dir = project_root / "plots"
        plot_dir = plots_dir / f"{run_timestamp}_{architecture_name}_{dataset_name_clean}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Evaluate model
        logger.debug("running create_and_train_model ... Evaluating model...")
        test_loss, test_accuracy = builder.evaluate(
            data=data, 
            log_detailed_predictions=log_detailed_predictions, 
            max_predictions_to_show=max_predictions_to_show,
            plot_dir=plot_dir
            )
        
        # Save the model
        builder.save_model(
            test_accuracy=test_accuracy,
            run_timestamp=run_timestamp
            )
        
        logger.debug(f"running create_and_train_model ... Completed with accuracy: {test_accuracy:.4f}")
        return builder, test_accuracy


if __name__ == "__main__":
    logger.debug("running model_builder.py ... Testing ModelBuilder...")
    
    # Simple argument parsing - convert command line to dictionary
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    
    # Extract dataset name (required)
    dataset_name = args.get('dataset_name', 'cifar10')
    
    # Convert string values to appropriate types for specific parameters
    if 'test_size' in args:
        try:
            args['test_size'] = float(args['test_size'])
        except ValueError:
            logger.warning(f"Invalid test_size: {args['test_size']}")
            del args['test_size']
    
    # Convert boolean parameters (INCLUDING NEW GRADIENT FLOW PARAMETERS)
    for bool_param in ['use_global_pooling', 'use_bidirectional',
                       'log_detailed_predictions', 'enable_realtime_plots', 
                       'save_realtime_plots', 'enable_gradient_flow_monitoring',  
                       'save_gradient_flow_plots', 'enable_gradient_clipping']:
        if bool_param in args:
            args[bool_param] = args[bool_param].lower() in ['true', '1', 'yes', 'on']
    
    # Convert integer parameters (INCLUDING NEW GRADIENT FLOW PARAMETERS)
    int_params = ['epochs', 'num_layers_conv', 'filters_per_conv_layer', 'num_layers_hidden', 
                  'first_hidden_layer_nodes', 'embedding_dim', 'lstm_units', 'vocab_size', 
                  'max_predictions_to_show', 'gradient_monitoring_frequency',
                  'gradient_history_length', 'gradient_sample_size']
    for int_param in int_params:
        if int_param in args:
            try:
                args[int_param] = int(args[int_param])
            except ValueError:
                logger.warning(f"Invalid {int_param}: {args[int_param]}")
                del args[int_param]
    
    # Convert float parameters  
    float_params = ['first_hidden_layer_dropout', 'subsequent_hidden_layer_dropout_decrease', 
                    'subsequent_hidden_layer_nodes_decrease', 'text_dropout',
                    'gradient_clip_norm']
    for float_param in float_params:
        if float_param in args:
            try:
                args[float_param] = float(args[float_param])
            except ValueError:
                logger.warning(f"Invalid {float_param}: {args[float_param]}")
                del args[float_param]
    
    # Convert tuple parameters
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
    
    logger.debug(f"running model_builder.py ... Parsed arguments: {args}")
    
    try:
        # One function call with all arguments
        builder, test_accuracy = create_and_train_model(**args)
        
        # Success
        load_path = args.get('load_model_path')
        workflow_msg = f"loaded existing model from {load_path}" if load_path else "trained new model"
        
        logger.debug(f"running model_builder.py ... ✅ SUCCESS!")
        logger.debug(f"running model_builder.py ... Successfully {workflow_msg}")
        logger.debug(f"running model_builder.py ... Final accuracy: {test_accuracy:.4f}")
        
        # Log gradient flow monitoring status if it was enabled
        if args.get('enable_gradient_flow_monitoring', False):
            logger.debug("running model_builder.py ... Real-time gradient flow monitoring was active during training")
        
    except Exception as e:
        logger.error(f"running model_builder.py ... ❌ ERROR: {e}")
        logger.error(f"running model_builder.py ... Traceback: {traceback.format_exc()}")
        sys.exit(1)