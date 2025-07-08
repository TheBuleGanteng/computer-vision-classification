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
from dataset_manager import DatasetManager
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from utils.logger import logger, PerformanceLogger, TimedOperation
from dataset_manager import DatasetConfig
from datetime import datetime

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
    text_dropout: float = 0.5         # Dropout for text layers: Randomly disable LSTM connections during training. Prevents overfitting to specific word patterns. Range: 0.2-0.6 typical
    
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
    hidden_layer_activation_algo: str = "relu" # Activation function applied after each conv layer. Options: "relu" (most common, outputs max(0,x), handles negatives well), "sigmoid" (outputs 0-1, good for probabilities but can cause vanishing gradients), "tanh" (outputs -1 to 1, centered around zero), "leaky_relu" (like relu but allows small negative values), "swish" (smooth, modern alternative to relu)
    first_hidden_layer_dropout: float = 0.5 # Dropout rate for first hidden layer. Randomly sets this fraction of neurons to 0 during training to prevent overfitting. Options: 0.0 (no dropout), 0.1-0.3 (light), 0.4-0.6 (moderate, most common), 0.7-0.9 (heavy, can hurt learning). Higher values = more regularization but slower learning
    subsequent_hidden_layer_dropout_decrease: float = 0.20 # How much to reduce dropout in each subsequent layer. Layer 1: 0.5 dropout, Layer 2: 0.5-0.2=0.3 dropout, Layer 3: 0.3-0.2=0.1 dropout. Rationale: deeper layers need less regularization as they're making more specific decisions
    
    
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
            self.model.compile(
                optimizer=self.model_config.optimizer,
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
    
    
    def train(self, data: Dict[str, Any], validation_split: float = 0.2) -> keras.callbacks.History:
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
        with TimedOperation("model training", "model_builder"):
            self.training_history = self.model.fit(
                data['x_train'], 
                data['y_train'],
                epochs=self.model_config.epochs,
                validation_split=validation_split,
                verbose=1  # Show progress bars
            )
        
        logger.debug("running train ... Training completed")
        return self.training_history
    
    
    def evaluate(self, data: Dict[str, Any]) -> Tuple[float, float]:
        """
        Evaluate model performance on test data
        Evaluation is read-only - the model doesn't learn or change during evaluation.
        
        Args:
            data: Dataset dictionary from DatasetManager
            
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluation")
        
        logger.debug("running evaluate ... Evaluating model on test data...")
        
        # Type guard: we know model is not None here due to the check above
        model: keras.Model = self.model
        
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
        
        logger.debug(f"running evaluate ... Test accuracy: {test_accuracy:.4f}")
        logger.debug(f"running evaluate ... Test loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    
    def save_model(self, test_accuracy: Optional[float] = None) -> None:
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
        
        # Auto-generate filename based on results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        accuracy_str = f"{test_accuracy:.1f}".replace(".", "_")
        filename = f"model_{timestamp}_acc_{accuracy_str}.keras" # e.g., "model_20250107_143022_acc_94_2.keras"
        
        # Determine filepath
        project_root: Path = Path(__file__).resolve().parent.parent
        models_dir = project_root / "saved_models"
        
        # Create directory if it doesn't exist
        models_dir.mkdir(exist_ok=True)
        
        final_filepath = models_dir / filename
                
        logger.debug(f"running save_model ... Saving model to {final_filepath}")
        self.model.save(final_filepath)
        logger.debug(f"running save_model ... Model saved successfully")
    
    
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
    save_path: Optional[str] = None,
    test_size: float = 0.4,
    **config_overrides
) -> Tuple[ModelBuilder, float]:
    """
    Convenience function to create, train, and evaluate a model in one call
    
    Args:
        data: Dataset dictionary from DatasetManager (optional if dataset_name provided)
        dataset_name: Name of dataset to load (optional if data provided)
        model_config: Optional model configuration
        save_path: Optional path to save the trained model
        test_size: Fraction of data to use for testing (only used if dataset_name provided)
        **config_overrides: Any ModelConfig parameters to override
        
    Returns:
        Tuple of (ModelBuilder instance, test_accuracy)
        
    Examples:
        - Use via command line arguments to specify parameters
            - Example (trains new model): python model_builder.py dataset_name=cifar10 use_global_pooling=true epochs=15
            - xample (loads existing model): python src/model_builder.py load_model=/home/thebuleganteng/01_Repos/06_personal_work/computer-vision-classification/saved_models/model_20250708_122719_acc_0_3.keras dataset_name=cifar100 test_size=0.1
    
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
    
    # Create or modify model configuration
    if model_config is None:
        model_config = ModelConfig()
    else:
        # Create a copy to avoid modifying the original
        import copy
        model_config = copy.deepcopy(model_config)
    
    # Apply any config overrides
    if config_overrides:
        logger.debug(f"running create_and_train_model ... Applying config overrides: {config_overrides}")
        for key, value in config_overrides.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
                logger.debug(f"running create_and_train_model ... Set {key} = {value}")
            else:
                logger.warning(f"running create_and_train_model ... Unknown config parameter: {key}")
    
    logger.debug(f"running create_and_train_model ... Creating model for {dataset_config.name}")
    
    # Create model builder
    builder = ModelBuilder(dataset_config, model_config)
    
    # Build and train model
    logger.debug("running create_and_train_model ... Building and training model...")
    builder.build_model()
    builder.train(data)
    
    # Evaluate model
    logger.debug("running create_and_train_model ... Evaluating model...")
    test_loss, test_accuracy = builder.evaluate(data)
    
    # Save if requested
    if save_path:
        logger.debug(f"running create_and_train_model ... Saving model to {save_path}")
        builder.save_model(test_accuracy=test_accuracy)
    
    logger.debug(f"running create_and_train_model ... Completed with accuracy: {test_accuracy:.4f}")
    return builder, test_accuracy


# Example usage and testing
if __name__ == "__main__":
    from utils.logger import logger
    logger.debug("running model_builder.py ... Testing ModelBuilder...")
    
    # Parse command line arguments in key=value format
    # Expected formats:
    # NEW: python model_builder.py load_model=/path/to/model.keras dataset_name=cifar100
    # OLD: python model_builder.py dataset_name=cifar100 use_global_pooling=true epochs=20
    
    # Default values - comprehensive set
    config_params = {
        'dataset_name': 'cifar10',  # Default dataset
        'test_size': 0.2,           # Default test split
        'save_path': 'test_model.keras',  # Default save path
        'load_model': None,         # NEW: Path to existing model to load
        
        # ModelConfig parameters with defaults
        'use_global_pooling': False,
        'epochs': 10,
        'num_layers_conv': 2,
        'filters_per_conv_layer': 32,
        'kernel_size': (3, 3),
        'pool_size': (2, 2),
        'num_layers_hidden': 1,
        'first_hidden_layer_nodes': 128,
        'first_hidden_layer_dropout': 0.5,
        'subsequent_hidden_layer_nodes_decrease': 0.50,
        'subsequent_hidden_layer_dropout_decrease': 0.20,
        
        # Text-specific parameters
        'embedding_dim': 128,
        'lstm_units': 64,
        'vocab_size': 10000,
        'use_bidirectional': True,
        'text_dropout': 0.5,
        
        # Training parameters
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'activation': 'relu',
        'hidden_layer_activation_algo': 'relu'
    }
    
    # Parse all command line arguments
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)  # Split only on first '='
            
            # Handle boolean parameters
            if key in ['use_global_pooling', 'use_bidirectional']:
                config_params[key] = value.lower() in ['true', '1', 'yes', 'on']
                logger.debug(f"running model_builder.py ... Parsed {key} = {config_params[key]} (boolean)")
                
            # Handle integer parameters
            elif key in ['epochs', 'num_layers_conv', 'filters_per_conv_layer', 'num_layers_hidden', 
                        'first_hidden_layer_nodes', 'embedding_dim', 'lstm_units', 'vocab_size']:
                try:
                    config_params[key] = int(value)
                    logger.debug(f"running model_builder.py ... Parsed {key} = {config_params[key]} (integer)")
                except ValueError:
                    logger.warning(f"running model_builder.py ... Invalid integer value for {key}: {value}")
                    
            # Handle float parameters
            elif key in ['first_hidden_layer_dropout', 'subsequent_hidden_layer_dropout_decrease', 
                        'subsequent_hidden_layer_nodes_decrease', 'text_dropout', 'test_size']:
                try:
                    config_params[key] = float(value)
                    logger.debug(f"running model_builder.py ... Parsed {key} = {config_params[key]} (float)")
                except ValueError:
                    logger.warning(f"running model_builder.py ... Invalid float value for {key}: {value}")
                    
            # Handle tuple parameters (kernel_size, pool_size)
            elif key in ['kernel_size', 'pool_size']:
                try:
                    if ',' in value:
                        # Handle "3,3" format
                        parts = [int(x.strip()) for x in value.split(',')]
                        if len(parts) == 2:
                            config_params[key] = tuple(parts)
                        else:
                            logger.warning(f"running model_builder.py ... Invalid tuple format for {key}: {value}")
                    else:
                        # Handle "3" format (becomes (3,3))
                        size = int(value)
                        config_params[key] = (size, size)
                    logger.debug(f"running model_builder.py ... Parsed {key} = {config_params[key]} (tuple)")
                except ValueError:
                    logger.warning(f"running model_builder.py ... Invalid tuple value for {key}: {value}")
                    
            # Handle string parameters (including NEW load_model parameter)
            elif key in ['dataset_name', 'save_path', 'load_model', 'optimizer', 'loss', 'activation', 'hidden_layer_activation_algo']:
                if key == 'dataset_name':
                    config_params[key] = value.lower()  # Normalize dataset names
                else:
                    config_params[key] = value
                logger.debug(f"running model_builder.py ... Parsed {key} = {config_params[key]} (string)")
                
            # Unknown parameter
            else:
                logger.warning(f"running model_builder.py ... Unknown parameter: {key}={value}")
                
        else:
            logger.warning(f"running model_builder.py ... Invalid argument format (expected key=value): {arg}")
    
    # Extract special parameters that need different handling
    dataset_name = config_params.pop('dataset_name')
    test_size = config_params.pop('test_size', 0.2)
    save_path = config_params.pop('save_path', 'test_model.keras')
    load_model_path = config_params.pop('load_model', None)  # NEW
    
    # Log final configuration
    logger.debug(f"running model_builder.py ... \n"
                 f"Final Configuration:\n"
                 f"- Dataset: {dataset_name.upper()}\n"
                 f"- Test size: {test_size}\n"
                 f"- Save path: {save_path}\n"
                 f"- Load existing model: {load_model_path or 'None (create new)'}\n"
                 f"- Model config overrides: {len(config_params)} parameters")
    
    # Check if dataset is available
    manager: DatasetManager = DatasetManager()
    if dataset_name not in manager.get_available_datasets():
        available = ', '.join(manager.get_available_datasets())
        print(f"Error: Unsupported dataset '{dataset_name}'. Available: {available}")
        sys.exit(1)
    
    try:
        # Load dataset
        logger.debug(f"running model_builder.py ... Loading dataset: {dataset_name}")
        data = manager.load_dataset(dataset_name, test_size=test_size)
        
        # NEW: Handle model loading vs creating
        if load_model_path:
            # Load existing model workflow
            logger.debug(f"running model_builder.py ... Loading existing model from: {load_model_path}")
            
            # Validate model file exists
            from pathlib import Path
            model_file = Path(load_model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {load_model_path}")
            
            # Create ModelBuilder and load existing model
            builder = ModelBuilder(data['config'])
            model = builder.load_model(load_model_path)
            
            logger.debug("running model_builder.py ... Existing model loaded successfully!")
            
            # Evaluate the loaded model
            logger.debug("running model_builder.py ... Evaluating loaded model...")
            test_loss, test_accuracy = builder.evaluate(data)
            
            logger.debug(f"running model_builder.py ... Loaded model performance:")
            logger.debug(f"running model_builder.py ... - Test accuracy: {test_accuracy:.4f}")
            logger.debug(f"running model_builder.py ... - Test loss: {test_loss:.4f}")
            
        else:
            # Create new model workflow (existing code)
            logger.debug("running model_builder.py ... Creating new model...")
            
            builder, test_accuracy = create_and_train_model(
                dataset_name=dataset_name,
                test_size=test_size,
                save_path=save_path,
                **config_params  # Pass all remaining parameters as keyword arguments
            )
            
            logger.debug(f"running model_builder.py ... New model trained with accuracy: {test_accuracy:.4f}")
        
        logger.debug(f"running model_builder.py ... ✅ SUCCESS!")
        logger.debug(f"running model_builder.py ... Final accuracy: {test_accuracy:.4f}")
        logger.debug("running model_builder.py ... ModelBuilder is ready for integration!")
        
    except Exception as e:
        logger.error(f"running model_builder.py ... ❌ ERROR: {e}")
        import traceback
        logger.error(f"running model_builder.py ... Traceback: {traceback.format_exc()}")
        sys.exit(1)