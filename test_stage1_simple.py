#!/usr/bin/env python3
"""
Simple Stage 1 Functional Test

Tests the basic functionality of the unified train() method
without complex configuration requirements.
"""

import os
import sys
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_config(use_runpod=False):
    """Create a minimal test configuration"""
    from src.data_classes.configs import OptimizationConfig, OptimizationMode, OptimizationObjective
    
    return OptimizationConfig(
        dataset_name="mnist",
        optimize_for="val_accuracy", 
        max_epochs_per_trial=5,
        min_epochs_per_trial=1,
        health_weight=0.3,
        mode=OptimizationMode.SIMPLE,
        objective=OptimizationObjective.ACCURACY,
        trials=1,
        use_runpod_service=use_runpod
    )

def test_train_method_exists():
    """Test that the train method exists and is callable"""
    from optimizer import ModelOptimizer
    
    config = create_test_config()
    
    optimizer = ModelOptimizer(dataset_name="mnist", optimization_config=config)
    
    # Test that train method exists
    assert hasattr(optimizer, 'train'), "train() method should exist"
    assert callable(getattr(optimizer, 'train')), "train() method should be callable"
    
    print("✅ train() method exists and is callable")

def test_train_method_routing():
    """Test that train method automatically routes to correct underlying methods based on configuration"""
    from optimizer import ModelOptimizer
    
    # Mock trial and params
    mock_trial = Mock()
    mock_trial.number = 0
    mock_params = {'learning_rate': 0.001}
    
    # Test local routing (use_runpod_service=False)
    config_local = create_test_config(use_runpod=False)
    optimizer_local = ModelOptimizer(dataset_name="mnist", optimization_config=config_local)
    
    with patch.object(optimizer_local, '_train_locally_for_trial', return_value=0.95) as mock_local:
        result = optimizer_local.train(mock_trial, mock_params)
        mock_local.assert_called_once_with(mock_trial, mock_params)
        assert result == 0.95, f"Expected 0.95, got {result}"
        print("✅ Local routing works correctly")
    
    # Test RunPod routing (use_runpod_service=True)
    config_runpod = create_test_config(use_runpod=True)
    optimizer_runpod = ModelOptimizer(dataset_name="mnist", optimization_config=config_runpod)
    
    with patch.object(optimizer_runpod, '_train_via_runpod_service', return_value=0.93) as mock_runpod:
        result = optimizer_runpod.train(mock_trial, mock_params)
        mock_runpod.assert_called_once_with(mock_trial, mock_params)
        assert result == 0.93, f"Expected 0.93, got {result}"
        print("✅ RunPod routing works correctly")

def test_train_method_defaults():
    """Test that train method automatically determines execution mode"""
    from optimizer import ModelOptimizer
    
    # Test with RunPod enabled
    config_runpod = create_test_config(use_runpod=True)
    optimizer_runpod = ModelOptimizer(dataset_name="mnist", optimization_config=config_runpod)
    
    # Mock trial and params
    mock_trial = Mock()
    mock_trial.number = 0 
    mock_params = {'learning_rate': 0.001}
    
    # Test RunPod execution (when use_runpod_service=True)
    with patch.object(optimizer_runpod, '_train_via_runpod_service', return_value=0.91) as mock_runpod:
        result = optimizer_runpod.train(mock_trial, mock_params)
        mock_runpod.assert_called_once_with(mock_trial, mock_params)
        assert result == 0.91, f"Expected 0.91, got {result}"
        print("✅ RunPod execution works correctly")
    
    # Test with local execution (use_runpod_service=False)
    config_local = create_test_config(use_runpod=False)
    optimizer_local = ModelOptimizer(dataset_name="mnist", optimization_config=config_local)
    
    # Test local execution (when use_runpod_service=False)
    with patch.object(optimizer_local, '_train_locally_for_trial', return_value=0.92) as mock_local:
        result = optimizer_local.train(mock_trial, mock_params)
        mock_local.assert_called_once_with(mock_trial, mock_params)
        assert result == 0.92, f"Expected 0.92, got {result}"
        print("✅ Local execution works correctly")

def test_objective_function_integration():
    """Test that _objective_function uses the new train method"""
    from optimizer import ModelOptimizer
    
    config = create_test_config(use_runpod=False)  # Force local execution
    
    optimizer = ModelOptimizer(dataset_name="mnist", optimization_config=config)
    
    # Mock trial
    mock_trial = Mock()
    mock_trial.number = 0
    mock_params = {'learning_rate': 0.001}
    
    # Mock the train method and its dependencies
    with patch.object(optimizer, 'train', return_value=0.89) as mock_train:
        with patch.object(optimizer, '_thread_safe_progress_callback'):
            with patch.object(optimizer, 'hyperparameter_selector') as mock_selector:
                mock_selector.suggest_hyperparameters.return_value = mock_params
                
                # Call objective function
                result = optimizer._objective_function(mock_trial)
                
                # Verify train was called correctly (no train_locally parameter)
                mock_train.assert_called_once_with(mock_trial, mock_params)
                print("✅ _objective_function integration works correctly")

def test_backward_compatibility():
    """Test that original methods still exist"""
    from optimizer import ModelOptimizer
    
    config = create_test_config()
    
    optimizer = ModelOptimizer(dataset_name="mnist", optimization_config=config)
    
    # Check that original methods still exist
    assert hasattr(optimizer, '_train_locally_for_trial'), "_train_locally_for_trial should still exist"
    assert hasattr(optimizer, '_train_via_runpod_service'), "_train_via_runpod_service should still exist"
    assert callable(getattr(optimizer, '_train_locally_for_trial')), "_train_locally_for_trial should be callable"
    assert callable(getattr(optimizer, '_train_via_runpod_service')), "_train_via_runpod_service should be callable"
    
    print("✅ Backward compatibility maintained")

def run_all_tests():
    """Run all Stage 1 simple tests"""
    print("=" * 60)
    print("STAGE 1 SIMPLE FUNCTIONAL TESTS")
    print("=" * 60)
    
    tests = [
        test_train_method_exists,
        test_train_method_routing,
        test_train_method_defaults,
        test_objective_function_integration,
        test_backward_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("STAGE 1 SIMPLE TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    
    if failed == 0:
        print("🎉 ALL STAGE 1 TESTS PASSED!")
        print("\n✅ Stage 1 implementation is ready for manual UI testing")
        return True
    else:
        print("❌ SOME TESTS FAILED - Fix issues before proceeding")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)