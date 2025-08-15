#!/usr/bin/env python3
"""
Local testing for RunPod service handler.
Tests handler logic by calling optimize_model() directly without RunPod.

This validates that the handler can successfully orchestrate the existing
optimizer.py logic before deploying to RunPod.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Set up paths for local testing
current_file = Path(__file__)
project_root = current_file.parent.parent  # Go up 2 levels to project root
sys.path.append(str(project_root))

# Import handler functions
from handler import start_training, validate_request, build_optimization_config

# Import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_basic_mnist_training() -> None:
    """Test basic MNIST training with minimal parameters."""
    logger.debug("running test_basic_mnist_training ... creating mock job request")
    
    mock_job = {
        'input': {
            'command': 'start_training',
            'trial_id': 'test_basic_001',
            'dataset': 'mnist',
            'hyperparameters': {
                'epochs': 2,
                'batch_size': 32,
                'learning_rate': 0.001,
                'activation': 'relu',
                'optimizer': 'adam'
            },
            'config': {
                'mode': 'simple',
                'objective': 'val_accuracy',
                'validation_split': 0.2,
                'max_training_time': 300
            }
        }
    }
    
    logger.debug("running test_basic_mnist_training ... calling start_training handler")
    result = start_training(mock_job)
    
    # Validate response structure
    logger.debug("running test_basic_mnist_training ... validating response structure")
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result.get('trial_id') == 'test_basic_001', f"Wrong trial_id: {result.get('trial_id')}"
    assert result.get('success') is True, f"Training failed: {result.get('error', 'Unknown error')}"
    assert result.get('status') == 'completed', f"Wrong status: {result.get('status')}"
    
    # Validate metrics presence
    metrics = result.get('metrics', {})
    assert 'test_accuracy' in metrics, "Missing test_accuracy in metrics"
    assert 'val_accuracy' in metrics, "Missing val_accuracy in metrics"
    assert 'best_value' in metrics, "Missing best_value in metrics"
    assert 'training_time_seconds' in metrics, "Missing training_time_seconds in metrics"
    
    # Validate health metrics presence
    assert 'health_metrics' in result, "Missing health_metrics in response"
    health_metrics = result['health_metrics']
    assert 'overall_health' in health_metrics, "Missing overall_health"
    
    # Validate parameter importance presence
    assert 'parameter_importance' in result, "Missing parameter_importance in response"
    
    # Validate best params presence
    assert 'best_params' in result, "Missing best_params in response"
    
    logger.debug("running test_basic_mnist_training ... test passed successfully")
    print("✅ Basic MNIST training test passed")
    print(f"   Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
    print(f"   Val Accuracy: {metrics.get('val_accuracy', 'N/A'):.4f}")
    print(f"   Training Time: {metrics.get('training_time_seconds', 'N/A'):.2f}s")
    print(f"   Overall Health: {health_metrics.get('overall_health', 'N/A'):.3f}")


def test_request_validation() -> None:
    """Test request validation logic."""
    logger.debug("running test_request_validation ... testing valid request")
    
    # Test valid request
    valid_request = {
        'command': 'start_training',
        'trial_id': 'test_validation_001',
        'dataset': 'mnist',
        'hyperparameters': {'epochs': 1},
        'config': {'mode': 'simple', 'objective': 'val_accuracy'}
    }
    
    is_valid, error_msg = validate_request(valid_request)
    assert is_valid is True, f"Valid request failed validation: {error_msg}"
    
    logger.debug("running test_request_validation ... testing invalid requests")
    
    # Test missing command
    invalid_request_1 = {
        'trial_id': 'test_validation_002',
        'dataset': 'mnist',
        'hyperparameters': {'epochs': 1},
        'config': {'mode': 'simple', 'objective': 'val_accuracy'}
    }
    
    is_valid, error_msg = validate_request(invalid_request_1)
    assert is_valid is False, "Missing command should fail validation"
    assert error_msg is not None, "Error message should not be None"
    assert 'command' in error_msg, f"Error message should mention command: {error_msg}"
    
    # Test invalid dataset
    invalid_request_2 = {
        'command': 'start_training',
        'trial_id': 'test_validation_003',
        'dataset': 'invalid_dataset',
        'hyperparameters': {'epochs': 1},
        'config': {'mode': 'simple', 'objective': 'val_accuracy'}
    }
    
    is_valid, error_msg = validate_request(invalid_request_2)
    assert is_valid is False, "Invalid dataset should fail validation"
    assert error_msg is not None and 'dataset' in error_msg, f"Error message should mention dataset: {error_msg}"
    
    logger.debug("running test_request_validation ... validation tests passed")
    print("✅ Request validation tests passed")


def test_config_building() -> None:
    """Test configuration building logic."""
    logger.debug("running test_config_building ... testing config parameter extraction")
    
    request = {
        'command': 'start_training',
        'trial_id': 'test_config_001',
        'dataset': 'mnist',
        'hyperparameters': {
            'epochs': 5,
            'batch_size': 64,
            'learning_rate': 0.01
        },
        'config': {
            'mode': 'comprehensive',
            'objective': 'test_accuracy',
            'validation_split': 0.3,
            'max_training_time': 600
        }
    }
    
    config_params = build_optimization_config(request)
    
    # Validate extracted parameters
    assert config_params['mode'] == 'comprehensive', f"Wrong mode: {config_params['mode']}"
    assert config_params['optimize_for'] == 'test_accuracy', f"Wrong optimize_for: {config_params['optimize_for']}"
    assert config_params['validation_split'] == 0.3, f"Wrong validation_split: {config_params['validation_split']}"
    assert config_params['max_training_time_minutes'] == 600, f"Wrong max_training_time_minutes: {config_params['max_training_time_minutes']}"
    
    logger.debug("running test_config_building ... config building test passed")
    print("✅ Configuration building test passed")


def test_error_handling() -> None:
    """Test error handling for malformed requests."""
    logger.debug("running test_error_handling ... testing malformed request handling")
    
    # Test completely malformed request
    malformed_job = {
        'input': {
            'invalid_field': 'invalid_value'
        }
    }
    
    result = start_training(malformed_job)
    
    # Should return error response with proper structure
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result.get('success') is False, "Malformed request should fail"
    assert result.get('status') == 'failed', f"Wrong status: {result.get('status')}"
    assert 'error' in result, "Error response should contain error message"
    assert result.get('trial_id') == 'unknown_trial', f"Should default trial_id: {result.get('trial_id')}"
    
    logger.debug("running test_error_handling ... error handling test passed")
    print("✅ Error handling test passed")


def run_comprehensive_test() -> None:
    """Run comprehensive test with more realistic parameters."""
    logger.debug("running run_comprehensive_test ... creating comprehensive test scenario")
    
    comprehensive_job = {
        'input': {
            'command': 'start_training',
            'trial_id': 'test_comprehensive_001',
            'dataset': 'mnist',
            'hyperparameters': {
                'epochs': 3,
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_layers_conv': 2,
                'filters_per_conv_layer': 32,
                'activation': 'relu',
                'optimizer': 'adam'
            },
            'config': {
                'mode': 'comprehensive',
                'objective': 'val_accuracy',
                'validation_split': 0.2,
                'max_training_time': 600
            }
        }
    }
    
    logger.debug("running run_comprehensive_test ... executing comprehensive training")
    result = start_training(comprehensive_job)
    
    # Validate comprehensive response
    assert result.get('success') is True, f"Comprehensive training failed: {result.get('error', 'Unknown error')}"
    
    # Check for comprehensive mode specific features
    metrics = result.get('metrics', {})
    health_metrics = result.get('health_metrics', {})
    param_importance = result.get('parameter_importance', {})
    
    # Validate rich data availability
    assert len(param_importance) > 0, "Parameter importance should be populated"
    assert health_metrics.get('overall_health', 0) > 0, "Health metrics should be calculated"
    
    logger.debug("running run_comprehensive_test ... comprehensive test passed")
    print("✅ Comprehensive test passed")
    print(f"   Parameter Importance Keys: {list(param_importance.keys())}")
    print(f"   Health Score: {health_metrics.get('overall_health', 'N/A'):.3f}")
    print(f"   Model Info: {result.get('model_info', {})}")


def main() -> None:
    """Run all local tests."""
    print("🧪 Starting RunPod Service Local Tests")
    print("=" * 50)
    
    try:
        # Run individual test components
        test_request_validation()
        test_config_building()
        test_error_handling()
        
        # Run training tests
        test_basic_mnist_training()
        run_comprehensive_test()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Handler is ready for RunPod deployment.")
        print("\nNext steps:")
        print("1. Deploy to RunPod using: ./deploy.sh")
        print("2. Test RunPod endpoint with integration tests")
        print("3. Add serverless mode to src/optimizer.py")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        print(f"\n❌ Test failed: {str(e)}")
        print("\nDebugging information:")
        print(f"Project root: {project_root}")
        print(f"Python path: {sys.path}")
        raise


if __name__ == "__main__":
    main()