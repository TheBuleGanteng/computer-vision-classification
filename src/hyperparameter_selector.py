"""
Hyperparameter Selector for Multi-Modal Classification

Extracted from ModelOptimizer to provide clean separation of concerns.
Handles hyperparameter suggestion logic for both CNN and LSTM architectures
with parameter validation and search space definitions.

This module contains domain-specific knowledge about what hyperparameters
work well for different types of neural networks and datasets.
"""

import optuna
from typing import Dict, Any, List, Tuple, Optional
from utils.logger import logger
from dataset_manager import DatasetConfig


class HyperparameterSelector:
    """
    Handles hyperparameter suggestion and validation for different model architectures
    
    Provides clean interface for suggesting hyperparameters based on data type
    and validates parameter combinations for feasibility.
    """
    
    def __init__(self, dataset_config: DatasetConfig, min_epochs: int = 5, max_epochs: int = 20):
        """
        Initialize HyperparameterSelector with dataset configuration
        
        Args:
            dataset_config: Configuration object containing dataset metadata
            min_epochs: Minimum number of training epochs
            max_epochs: Maximum number of training epochs
        """
        self.dataset_config = dataset_config
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        
        # Detect data type for architecture selection
        self.data_type = self._detect_data_type()
        
        logger.debug(f"running HyperparameterSelector.__init__ ... Initialized for dataset: {dataset_config.name}")
        logger.debug(f"running HyperparameterSelector.__init__ ... Data type: {self.data_type}")
        logger.debug(f"running HyperparameterSelector.__init__ ... Epoch range: {min_epochs}-{max_epochs}")
    
    def suggest_hyperparameters(
        self, 
        trial: optuna.Trial, 
        activation_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main interface for suggesting hyperparameters based on data type
        
        Args:
            trial: Optuna trial object for parameter suggestion
            activation_override: Optional activation function override
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        logger.debug(f"running suggest_hyperparameters ... Starting hyperparameter suggestion for trial {trial.number}")
        logger.debug(f"running suggest_hyperparameters ... Data type: {self.data_type}")
        logger.debug(f"running suggest_hyperparameters ... Activation override: {activation_override}")
        
        if self.data_type == "text":
            # Pass activation_override to LSTM suggestion method for consistency
            params = self.suggest_lstm_hyperparameters(trial, activation_override)
            logger.debug(f"running suggest_hyperparameters ... Generated LSTM hyperparameters")
        else:
            # Pass activation_override to CNN suggestion method
            params = self.suggest_cnn_hyperparameters(trial, activation_override)
            logger.debug(f"running suggest_hyperparameters ... Generated CNN hyperparameters")
        
        # Activation override is now handled inside suggest_cnn_hyperparameters
        # No need to apply it here anymore
        
        # Validate parameter combination
        if self._validate_parameters(params):
            logger.debug(f"running suggest_hyperparameters ... Parameter validation passed")
        else:
            logger.warning(f"running suggest_hyperparameters ... Parameter validation failed, using fallback")
            params = self._get_fallback_parameters()
        
        return params
    
    def suggest_cnn_hyperparameters(self, trial: optuna.Trial, activation_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Suggest hyperparameters for CNN image classification architecture
        
        Args:
            trial: Optuna trial object
            activation_override: Optional activation function override
            
        Returns:
            Dictionary of CNN-specific hyperparameters
        """
        logger.debug(f"running suggest_cnn_hyperparameters ... Suggesting CNN hyperparameters")
        
        # Get input image dimensions for constraint calculations
        input_height = self.dataset_config.img_height
        input_width = self.dataset_config.img_width
        logger.debug(f"running suggest_cnn_hyperparameters ... Input dimensions: {input_height}x{input_width}")
        
        # Core architecture parameters
        num_layers_conv = trial.suggest_int('num_layers_conv', 1, 4)
        filters_per_conv_layer = trial.suggest_categorical('filters_per_conv_layer', [16, 32, 64, 128, 256])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
        
        # Handle activation parameter - only suggest if no override is provided
        if activation_override is not None:
            activation = activation_override
            logger.debug(f"running suggest_cnn_hyperparameters ... Using activation override: {activation}")
        else:
            activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'swish'])
            logger.debug(f"running suggest_cnn_hyperparameters ... Suggested activation from trial: {activation}")
        
        kernel_initializer = trial.suggest_categorical('kernel_initializer', ['he_normal', 'glorot_uniform'])
        batch_normalization = trial.suggest_categorical('batch_normalization', [True, False])
        use_global_pooling = trial.suggest_categorical('use_global_pooling', [True, False])
        
        # Hidden layer parameters
        num_layers_hidden = trial.suggest_int('num_layers_hidden', 1, 4)
        first_hidden_layer_nodes = trial.suggest_categorical('first_hidden_layer_nodes', [64, 128, 256, 512, 1024])
        
        # Training parameters
        epochs = trial.suggest_int('epochs', self.min_epochs, self.max_epochs)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
        
        params = {
            # Architecture selection
            'architecture_type': 'cnn',
            'use_global_pooling': use_global_pooling,
            
            # Convolutional layers
            'num_layers_conv': num_layers_conv,
            'filters_per_conv_layer': filters_per_conv_layer,
            'kernel_size': (kernel_size, kernel_size),
            'activation': activation,
            'kernel_initializer': kernel_initializer,
            'batch_normalization': batch_normalization,
            'padding': 'same',
            
            # Hidden layers
            'num_layers_hidden': num_layers_hidden,
            'first_hidden_layer_nodes': first_hidden_layer_nodes,
            
            # Training parameters
            'epochs': epochs,
            'optimizer': optimizer,
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy'],
        }
        
        logger.debug(f"running suggest_cnn_hyperparameters ... Generated {len(params)} CNN parameters")
        logger.debug(f"running suggest_cnn_hyperparameters ... Final activation in params: {params['activation']}")
        return params
    
    def suggest_lstm_hyperparameters(self, trial: optuna.Trial, activation_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Suggest hyperparameters for LSTM text classification architecture
        
        Args:
            trial: Optuna trial object
            activation_override: Optional activation function override (for consistency, 
                            though LSTM architectures may not use this parameter)
            
        Returns:
            Dictionary of LSTM-specific hyperparameters
        """
        logger.debug(f"running suggest_lstm_hyperparameters ... Suggesting LSTM hyperparameters")
        
        # Text-specific parameters
        embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256, 512])
        lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128, 256])
        vocab_size = trial.suggest_categorical('vocab_size', [5000, 10000, 20000])
        use_bidirectional = trial.suggest_categorical('use_bidirectional', [True, False])
        
        # Hidden layer parameters
        num_layers_hidden = trial.suggest_int('num_layers_hidden', 1, 3)
        first_hidden_layer_nodes = trial.suggest_categorical('first_hidden_layer_nodes', [64, 128, 256, 512])
        
        # Training parameters
        epochs = trial.suggest_int('epochs', self.min_epochs, self.max_epochs)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
        
        params = {
            # Architecture selection
            'architecture_type': 'text',
            
            # Text-specific parameters
            'embedding_dim': embedding_dim,
            'lstm_units': lstm_units,
            'vocab_size': vocab_size,
            'use_bidirectional': use_bidirectional,
            
            # Hidden layers
            'num_layers_hidden': num_layers_hidden,
            'first_hidden_layer_nodes': first_hidden_layer_nodes,
            
            # Training parameters
            'epochs': epochs,
            'optimizer': optimizer,
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy']
        }
        
        # Add activation parameter if override is provided (for consistency)
        if activation_override is not None:
            params['activation'] = activation_override
            logger.debug(f"running suggest_lstm_hyperparameters ... Applied activation override: {activation_override}")
        
        logger.debug(f"running suggest_lstm_hyperparameters ... Generated {len(params)} LSTM parameters")
        return params
    
    def _detect_data_type(self) -> str:
        """
        Detect whether this is image or text data based on dataset configuration
        
        Returns:
            String indicating data type: "image" or "text"
        """
        # Text indicators: flat sequence structure
        if (self.dataset_config.img_height == 1 and 
            self.dataset_config.channels == 1 and 
            self.dataset_config.img_width > 100):
            logger.debug(f"running _detect_data_type ... Detected TEXT data: sequence_length={self.dataset_config.img_width}")
            return "text"
        
        # Image indicators: spatial structure
        if (self.dataset_config.img_height > 1 and 
            self.dataset_config.img_width > 1):
            logger.debug(f"running _detect_data_type ... Detected IMAGE data: shape=({self.dataset_config.img_height}, {self.dataset_config.img_width}, {self.dataset_config.channels})")
            return "image"
        
        # Fallback to image for ambiguous cases
        logger.debug(f"running _detect_data_type ... Ambiguous data shape, defaulting to IMAGE")
        return "image"
    
    def _validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameter combinations for feasibility
        
        Args:
            params: Dictionary of hyperparameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # Basic validation checks
            if params.get('epochs', 0) < self.min_epochs:
                logger.warning(f"running _validate_parameters ... Epochs {params['epochs']} below minimum {self.min_epochs}")
                return False
            
            if params.get('epochs', 0) > self.max_epochs:
                logger.warning(f"running _validate_parameters ... Epochs {params['epochs']} above maximum {self.max_epochs}")
                return False
            
            # Architecture-specific validation
            if params.get('architecture_type') == 'cnn':
                return self._validate_cnn_parameters(params)
            elif params.get('architecture_type') == 'text':
                return self._validate_lstm_parameters(params)
            
            return True
            
        except Exception as e:
            logger.warning(f"running _validate_parameters ... Validation error: {e}")
            return False
    
    def _validate_cnn_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate CNN-specific parameter combinations
        
        Args:
            params: Dictionary of CNN hyperparameters
            
        Returns:
            True if valid, False otherwise
        """
        # Check for reasonable parameter ranges
        num_conv_layers = params.get('num_layers_conv', 0)
        if not 1 <= num_conv_layers <= 4:
            logger.warning(f"running _validate_cnn_parameters ... Invalid num_layers_conv: {num_conv_layers}")
            return False
        
        filters = params.get('filters_per_conv_layer', 0)
        if filters not in [16, 32, 64, 128, 256]:
            logger.warning(f"running _validate_cnn_parameters ... Invalid filters_per_conv_layer: {filters}")
            return False
        
        return True
    
    def _validate_lstm_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate LSTM-specific parameter combinations
        
        Args:
            params: Dictionary of LSTM hyperparameters
            
        Returns:
            True if valid, False otherwise
        """
        # Check embedding dimension
        embedding_dim = params.get('embedding_dim', 0)
        if embedding_dim not in [64, 128, 256, 512]:
            logger.warning(f"running _validate_lstm_parameters ... Invalid embedding_dim: {embedding_dim}")
            return False
        
        # Check LSTM units
        lstm_units = params.get('lstm_units', 0)
        if lstm_units not in [32, 64, 128, 256]:
            logger.warning(f"running _validate_lstm_parameters ... Invalid lstm_units: {lstm_units}")
            return False
        
        return True
    
    def _get_fallback_parameters(self) -> Dict[str, Any]:
        """
        Get safe fallback parameters when validation fails
        
        Returns:
            Dictionary of safe default parameters
        """
        logger.debug(f"running _get_fallback_parameters ... Generating fallback parameters for {self.data_type}")
        
        if self.data_type == "text":
            return {
                'architecture_type': 'text',
                'embedding_dim': 128,
                'lstm_units': 64,
                'vocab_size': 10000,
                'use_bidirectional': True,
                'num_layers_hidden': 1,
                'first_hidden_layer_nodes': 128,
                'epochs': self.min_epochs,
                'optimizer': 'adam',
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy']
            }
        else:
            return {
                'architecture_type': 'cnn',
                'use_global_pooling': False,
                'num_layers_conv': 2,
                'filters_per_conv_layer': 32,
                'kernel_size': (3, 3),
                'activation': 'relu',
                'kernel_initializer': 'he_normal',
                'batch_normalization': True,
                'padding': 'same',
                'num_layers_hidden': 1,
                'first_hidden_layer_nodes': 128,
                'epochs': self.min_epochs,
                'optimizer': 'adam',
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy']
            }
    
    def get_search_space_info(self) -> Dict[str, Any]:
        """
        Get information about the search space for this dataset
        
        Returns:
            Dictionary containing search space metadata
        """
        return {
            'data_type': self.data_type,
            'dataset_name': self.dataset_config.name,
            'input_shape': self.dataset_config.input_shape,
            'num_classes': self.dataset_config.num_classes,
            'epoch_range': (self.min_epochs, self.max_epochs),
            'supported_architectures': ['cnn'] if self.data_type == 'image' else ['lstm'],
            'parameter_ranges': self._get_parameter_ranges()
        }
    
    def _get_parameter_ranges(self) -> Dict[str, Any]:
        """
        Get the ranges/options for each parameter type
        
        Returns:
            Dictionary mapping parameter names to their possible values
        """
        if self.data_type == "text":
            return {
                'embedding_dim': [64, 128, 256, 512],
                'lstm_units': [32, 64, 128, 256],
                'vocab_size': [5000, 10000, 20000],
                'use_bidirectional': [True, False],
                'num_layers_hidden': (1, 3),
                'first_hidden_layer_nodes': [64, 128, 256, 512],
                'optimizer': ['adam', 'rmsprop']
            }
        else:
            return {
                'num_layers_conv': (1, 4),
                'filters_per_conv_layer': [16, 32, 64, 128, 256],
                'kernel_size': [3, 5],
                'activation': ['relu', 'leaky_relu', 'swish'],
                'kernel_initializer': ['he_normal', 'glorot_uniform'],
                'batch_normalization': [True, False],
                'use_global_pooling': [True, False],
                'num_layers_hidden': (1, 4),
                'first_hidden_layer_nodes': [64, 128, 256, 512, 1024],
                'optimizer': ['adam', 'rmsprop', 'sgd']
            }


# Convenience function for standalone testing
def test_hyperparameter_selector(dataset_name: str = "cifar10") -> None:
    """
    Test function for HyperparameterSelector
    
    Args:
        dataset_name: Name of dataset to test with
    """
    from dataset_manager import DatasetManager
    
    logger.debug(f"running test_hyperparameter_selector ... Testing with dataset: {dataset_name}")
    
    # Load dataset config
    dataset_manager = DatasetManager()
    dataset_config = dataset_manager.get_dataset_config(dataset_name)
    
    # Create selector
    selector = HyperparameterSelector(dataset_config)
    
    # Create a mock trial for testing
    study = optuna.create_study()
    trial = study.ask()
    
    try:
        # Test hyperparameter suggestion
        params = selector.suggest_hyperparameters(trial)
        
        logger.debug(f"running test_hyperparameter_selector ... Generated parameters:")
        for key, value in params.items():
            logger.debug(f"running test_hyperparameter_selector ... - {key}: {value}")
        
        # Test search space info
        search_space = selector.get_search_space_info()
        logger.debug(f"running test_hyperparameter_selector ... Search space info: {search_space}")
        
        logger.debug(f"running test_hyperparameter_selector ... ✅ Test completed successfully")
        
    except Exception as e:
        logger.error(f"running test_hyperparameter_selector ... ❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    # Simple test when run directly
    import sys
    
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    test_hyperparameter_selector(dataset_name)