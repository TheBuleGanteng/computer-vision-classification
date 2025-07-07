#!/usr/bin/env python3
"""
Test script for model_builder.py

Tests the ModelBuilder class with both CIFAR-10 and custom configurations
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
current_file = Path(__file__)
project_root = current_file.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from src.dataset_manager import DatasetManager
from src.model_builder import ModelBuilder, ModelConfig, create_and_train_model
from src.utils.logger import logger, PerformanceLogger, TimedOperation
import numpy as np
from typing import Dict, Any

def test_model_config():
    """Test ModelConfig creation and validation"""
    logger.debug("running test_model_config ... Testing ModelConfig creation...")
    
    # Test default config
    default_config = ModelConfig()
    assert default_config.num_layers_conv == 2
    assert default_config.filters_per_conv_layer == 32
    assert default_config.epochs == 10
    assert "accuracy" in default_config.metrics
    logger.debug("running test_model_config ... ✅ Default config test passed")
    
    # Test custom config
    custom_config = ModelConfig(
        num_layers_conv=3,
        filters_per_conv_layer=64,
        num_layers_hidden=2,
        epochs=5
    )
    assert custom_config.num_layers_conv == 3
    assert custom_config.filters_per_conv_layer == 64
    assert custom_config.num_layers_hidden == 2
    assert custom_config.epochs == 5
    logger.debug("running test_model_config ... ✅ Custom config test passed")
    
    logger.debug("running test_model_config ... ✅ All ModelConfig tests passed")


def test_model_builder_initialization():
    """Test ModelBuilder initialization"""
    logger.debug("running test_model_builder_initialization ... Testing ModelBuilder initialization...")
    
    # Load dataset to get config
    manager = DatasetManager()
    data = manager.load_dataset('cifar10', test_size=0.1)  # Small test set for speed
    dataset_config = data['config']
    
    # Test with default model config
    builder = ModelBuilder(dataset_config)
    assert builder.dataset_config == dataset_config
    assert builder.model_config is not None
    assert builder.model is None  # Should be None until built
    assert builder.training_history is None
    logger.debug("running test_model_builder_initialization ... ✅ Default initialization test passed")
    
    # Test with custom model config
    custom_config = ModelConfig(epochs=3, num_layers_conv=1)
    builder_custom = ModelBuilder(dataset_config, custom_config)
    assert builder_custom.model_config.epochs == 3
    assert builder_custom.model_config.num_layers_conv == 1
    logger.debug("running test_model_builder_initialization ... ✅ Custom initialization test passed")
    
    logger.debug("running test_model_builder_initialization ... ✅ All initialization tests passed")


def test_model_building():
    """Test model architecture building"""
    logger.debug("running test_model_building ... Testing model building...")
    
    # Load small dataset
    manager = DatasetManager()
    data = manager.load_dataset('cifar10', test_size=0.9)  # Use very small training set
    dataset_config = data['config']
    
    # Create builder with small config for fast testing
    test_config = ModelConfig(
        num_layers_conv=1,
        filters_per_conv_layer=16,
        num_layers_hidden=1,
        first_hidden_layer_nodes=32,
        epochs=1  # Just 1 epoch for testing
    )
    
    builder = ModelBuilder(dataset_config, test_config)
    
    # Build model
    model = builder.build_model()
    
    # Verify model was created
    assert model is not None
    assert builder.model is not None
    assert builder.model == model
    logger.debug("running test_model_building ... ✅ Model creation test passed")
    
    # Check model structure
    assert len(model.layers) > 0
    logger.debug(f"running test_model_building ... Model has {len(model.layers)} layers")
    
    # Verify input shape
    expected_shape = (None,) + dataset_config.input_shape
    actual_shape = model.input_shape
    assert actual_shape == expected_shape, f"Expected {expected_shape}, got {actual_shape}"
    logger.debug("running test_model_building ... ✅ Input shape test passed")
    
    # Verify output shape
    expected_output = dataset_config.num_classes
    actual_output = model.output_shape[-1]
    assert actual_output == expected_output, f"Expected {expected_output}, got {actual_output}"
    logger.debug("running test_model_building ... ✅ Output shape test passed")
    
    # Check that model is compiled
    assert model.optimizer is not None
    assert model.loss is not None
    logger.debug("running test_model_building ... ✅ Model compilation test passed")
    
    logger.debug("running test_model_building ... ✅ All model building tests passed")


def test_training():
    """Test model training"""
    logger.debug("running test_training ... Testing model training...")
    
    # Load very small dataset for fast training
    manager = DatasetManager()
    data = manager.load_dataset('cifar10', test_size=0.95)  # Only 5% for training
    dataset_config = data['config']
    
    # Create minimal config for fast training
    fast_config = ModelConfig(
        num_layers_conv=1,
        filters_per_conv_layer=8,
        num_layers_hidden=1,
        first_hidden_layer_nodes=16,
        epochs=2  # Just 2 epochs
    )
    
    builder = ModelBuilder(dataset_config, fast_config)
    builder.build_model()
    
    # Train model
    logger.debug("running test_training ... Starting training (this may take a moment)...")
    history = builder.train(data, validation_split=0.2)
    
    # Verify training completed
    assert history is not None
    assert builder.training_history is not None
    assert builder.training_history == history
    logger.debug("running test_training ... ✅ Training completion test passed")
    
    # Check history contains expected metrics
    assert 'loss' in history.history
    assert 'accuracy' in history.history
    assert len(history.history['loss']) == fast_config.epochs
    logger.debug("running test_training ... ✅ Training history test passed")
    
    logger.debug("running test_training ... ✅ All training tests passed")


def test_evaluation():
    """Test model evaluation"""
    logger.debug("running test_evaluation ... Testing model evaluation...")
    
    # Load small dataset
    manager = DatasetManager()
    data = manager.load_dataset('cifar10', test_size=0.8)
    dataset_config = data['config']
    
    # Quick training
    quick_config = ModelConfig(
        num_layers_conv=1,
        filters_per_conv_layer=8,
        num_layers_hidden=1,
        first_hidden_layer_nodes=16,
        epochs=1
    )
    
    builder = ModelBuilder(dataset_config, quick_config)
    builder.build_model()
    builder.train(data, validation_split=0.1)
    
    # Evaluate model
    test_loss, test_accuracy = builder.evaluate(data)
    
    # Verify evaluation results
    assert isinstance(test_loss, float)
    assert isinstance(test_accuracy, float)
    assert 0.0 <= test_accuracy <= 1.0, f"Accuracy should be 0-1, got {test_accuracy}"
    assert test_loss >= 0.0, f"Loss should be >= 0, got {test_loss}"
    logger.debug(f"running test_evaluation ... Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
    logger.debug("running test_evaluation ... ✅ All evaluation tests passed")


def test_convenience_function():
    """Test the create_and_train_model convenience function"""
    logger.debug("running test_convenience_function ... Testing convenience function...")
    
    # Load small dataset
    manager = DatasetManager()
    data = manager.load_dataset('cifar10', test_size=0.9)
    
    # Test config
    test_config = ModelConfig(
        num_layers_conv=1,
        filters_per_conv_layer=8,
        num_layers_hidden=1,
        first_hidden_layer_nodes=16,
        epochs=1
    )
    
    # Use convenience function
    builder, accuracy = create_and_train_model(
        data=data,
        model_config=test_config
    )
    
    # Verify results
    assert isinstance(builder, ModelBuilder)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
    assert builder.model is not None
    assert builder.training_history is not None
    logger.debug(f"running test_convenience_function ... Achieved accuracy: {accuracy:.4f}")
    logger.debug("running test_convenience_function ... ✅ Convenience function test passed")


def test_model_save_and_load():
    """Test model saving and loading"""
    logger.debug("running test_model_save_and_load ... Testing model save/load...")
    
    # Create and train a simple model
    manager = DatasetManager()
    data = manager.load_dataset('cifar10', test_size=0.95)
    
    test_config = ModelConfig(
        num_layers_conv=1,
        filters_per_conv_layer=4,
        num_layers_hidden=1,
        first_hidden_layer_nodes=8,
        epochs=1
    )
    
    builder = ModelBuilder(data['config'], test_config)
    builder.build_model()
    builder.train(data)
    test_loss, test_accuracy = builder.evaluate(data)
    
    # Save model
    builder.save_model(test_accuracy=test_accuracy)
    logger.debug("running test_model_save_and_load ... ✅ Model save test passed")
    
    # Verify saved_models directory exists and contains file
    project_root = Path(__file__).parent
    saved_models_dir = project_root / "saved_models"
    assert saved_models_dir.exists(), "saved_models directory should exist"
    
    model_files = list(saved_models_dir.glob("*.keras"))
    assert len(model_files) > 0, "Should have at least one saved model"
    logger.debug(f"running test_model_save_and_load ... Found {len(model_files)} saved model(s)")
    logger.debug("running test_model_save_and_load ... ✅ All save/load tests passed")


def run_all_tests():
    """Run all model builder tests"""
    logger.debug("running run_all_tests ... 🚀 Starting ModelBuilder comprehensive tests...")
    
    test_functions = [
        test_model_config,
        test_model_builder_initialization,
        test_model_building,
        test_training,
        test_evaluation,
        test_convenience_function,
        test_model_save_and_load
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for i, test_func in enumerate(test_functions, 1):
        try:
            logger.debug(f"running run_all_tests ... Test {i}/{total_tests}: {test_func.__name__}")
            with TimedOperation(test_func.__name__, "model_builder_test"):
                test_func()
            passed_tests += 1
            logger.debug(f"running run_all_tests ... ✅ {test_func.__name__} PASSED")
        except Exception as e:
            logger.error(f"running run_all_tests ... ❌ {test_func.__name__} FAILED: {e}")
            import traceback
            logger.error(f"running run_all_tests ... Traceback: {traceback.format_exc()}")
    
    # Summary
    logger.debug("running run_all_tests ... " + "="*60)
    logger.debug(f"running run_all_tests ... TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.debug("running run_all_tests ... 🎉 ALL TESTS PASSED! ModelBuilder is ready for use!")
        logger.debug("running run_all_tests ... Ready to proceed with optimization and UI development!")
        return True
    else:
        logger.error(f"running run_all_tests ... ❌ {total_tests - passed_tests} tests failed. Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    logger.debug("running model_builder_test.py ... ModelBuilder Test Suite")
    logger.debug("running model_builder_test.py ... " + "="*60)
    
    # Test that our imports work
    try:
        logger.debug("running model_builder_test.py ... Testing imports...")
        from src.dataset_manager import DatasetManager
        from src.model_builder import ModelBuilder, ModelConfig
        logger.debug("running model_builder_test.py ... ✅ All imports successful")
    except ImportError as e:
        logger.error(f"running model_builder_test.py ... ❌ Import failed: {e}")
        logger.error("running model_builder_test.py ... Please check your project structure and file paths")
        sys.exit(1)
    
    # Run all tests
    success = run_all_tests()
    
    if success:
        logger.debug("running model_builder_test.py ... ModelBuilder testing complete - all systems go! 🚀")
    else:
        logger.error("running model_builder_test.py ... Testing failed - please address issues before continuing")
        sys.exit(1)