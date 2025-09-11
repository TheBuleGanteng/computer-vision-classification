#!/usr/bin/env python3
"""
Stage 1 Testing: Training Function Unification

Tests the unified train() function that combines local and RunPod execution paths.
Validates backward compatibility and proper routing of execution methods.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import optuna

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from optimizer import ModelOptimizer
from src.data_classes.configs import OptimizationConfig, OptimizationMode, OptimizationObjective


class TestStage1TrainingUnification(unittest.TestCase):
    """Test suite for Stage 1: Unified training function"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = OptimizationConfig(
            dataset_name="MNIST",
            optimize_for="val_accuracy",
            mode=OptimizationMode.SIMPLE,
            objective=OptimizationObjective.ACCURACY,
            trials=2,
            concurrent_workers=1,
            runpod_service_endpoint="test-endpoint",
            use_runpod_service=True,
            runpod_service_fallback_local=True
        )
        
        self.optimizer = ModelOptimizer(
            dataset_name="MNIST",
            config=self.config
        )
        
        # Mock trial and params
        self.mock_trial = Mock()
        self.mock_trial.number = 0
        self.mock_params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 5
        }

    def test_train_function_exists(self):
        """Test that the unified train() function exists"""
        self.assertTrue(hasattr(self.optimizer, 'train'))
        self.assertTrue(callable(getattr(self.optimizer, 'train')))

    @patch.object(ModelOptimizer, '_train_locally_for_trial')
    def test_train_with_local_execution(self, mock_local_train):
        """Test train() function routes correctly to local execution"""
        mock_local_train.return_value = 0.95
        
        result = self.optimizer.train(
            trial=self.mock_trial,
            params=self.mock_params,
            train_locally=True
        )
        
        # Verify correct routing
        mock_local_train.assert_called_once_with(self.mock_trial, self.mock_params)
        self.assertEqual(result, 0.95)

    @patch.object(ModelOptimizer, '_train_via_runpod_service')
    def test_train_with_runpod_execution(self, mock_runpod_train):
        """Test train() function routes correctly to RunPod execution"""
        mock_runpod_train.return_value = 0.93
        
        result = self.optimizer.train(
            trial=self.mock_trial,
            params=self.mock_params,
            train_locally=False
        )
        
        # Verify correct routing
        mock_runpod_train.assert_called_once_with(self.mock_trial, self.mock_params)
        self.assertEqual(result, 0.93)

    @patch.object(ModelOptimizer, '_train_via_runpod_service')
    def test_train_default_behavior(self, mock_runpod_train):
        """Test train() function defaults to RunPod execution (train_locally=False)"""
        mock_runpod_train.return_value = 0.91
        
        result = self.optimizer.train(
            trial=self.mock_trial,
            params=self.mock_params
            # train_locally not specified, should default to False
        )
        
        # Verify defaults to RunPod
        mock_runpod_train.assert_called_once_with(self.mock_trial, self.mock_params)
        self.assertEqual(result, 0.91)

    @patch.object(ModelOptimizer, '_should_use_runpod_service')
    @patch.object(ModelOptimizer, 'train')
    def test_objective_function_integration(self, mock_train, mock_should_use_runpod):
        """Test that _objective_function correctly uses the new train() function"""
        # Setup mocks
        mock_should_use_runpod.return_value = True
        mock_train.return_value = 0.89
        
        # Mock other dependencies
        with patch.object(self.optimizer, '_setup_trial_directory'):
            with patch.object(self.optimizer, '_staggered_start'):
                with patch.object(self.optimizer, '_thread_safe_progress_callback'):
                    with patch.object(self.optimizer, 'hyperparameter_selector') as mock_selector:
                        mock_selector.suggest_hyperparameters.return_value = self.mock_params
                        
                        # Call objective function
                        result = self.optimizer._objective_function(self.mock_trial)
                        
                        # Verify train() was called with correct parameters
                        mock_train.assert_called_once_with(
                            self.mock_trial, 
                            self.mock_params, 
                            train_locally=False  # Should be False when using RunPod
                        )

    @patch.object(ModelOptimizer, '_should_use_runpod_service')  
    @patch.object(ModelOptimizer, 'train')
    def test_objective_function_local_execution(self, mock_train, mock_should_use_runpod):
        """Test that _objective_function uses local execution when RunPod disabled"""
        # Setup mocks
        mock_should_use_runpod.return_value = False
        mock_train.return_value = 0.92
        
        # Mock other dependencies
        with patch.object(self.optimizer, '_setup_trial_directory'):
            with patch.object(self.optimizer, '_staggered_start'):
                with patch.object(self.optimizer, '_thread_safe_progress_callback'):
                    with patch.object(self.optimizer, 'hyperparameter_selector') as mock_selector:
                        mock_selector.suggest_hyperparameters.return_value = self.mock_params
                        
                        # Call objective function  
                        result = self.optimizer._objective_function(self.mock_trial)
                        
                        # Verify train() was called with local execution
                        mock_train.assert_called_once_with(
                            self.mock_trial,
                            self.mock_params,
                            train_locally=True  # Should be True when not using RunPod
                        )

    def test_backward_compatibility_method_signatures(self):
        """Test that original methods still exist for backward compatibility"""
        # Original methods should still exist
        self.assertTrue(hasattr(self.optimizer, '_train_locally_for_trial'))
        self.assertTrue(hasattr(self.optimizer, '_train_via_runpod_service'))
        
        # And should be callable
        self.assertTrue(callable(getattr(self.optimizer, '_train_locally_for_trial')))
        self.assertTrue(callable(getattr(self.optimizer, '_train_via_runpod_service')))

    @patch.object(ModelOptimizer, '_train_locally_for_trial')
    @patch.object(ModelOptimizer, '_train_via_runpod_service')
    def test_parameter_passing_integrity(self, mock_runpod, mock_local):
        """Test that parameters are passed correctly to underlying methods"""
        mock_local.return_value = 0.88
        mock_runpod.return_value = 0.87
        
        # Test local execution parameter passing
        self.optimizer.train(self.mock_trial, self.mock_params, train_locally=True)
        mock_local.assert_called_once_with(self.mock_trial, self.mock_params)
        
        # Reset mocks
        mock_local.reset_mock()
        mock_runpod.reset_mock()
        
        # Test RunPod execution parameter passing
        self.optimizer.train(self.mock_trial, self.mock_params, train_locally=False)
        mock_runpod.assert_called_once_with(self.mock_trial, self.mock_params)

    def test_function_signature_compatibility(self):
        """Test that the train() function signature is correct"""
        import inspect
        sig = inspect.signature(self.optimizer.train)
        
        # Should have trial, params, and train_locally parameters
        params = list(sig.parameters.keys())
        self.assertIn('trial', params)
        self.assertIn('params', params) 
        self.assertIn('train_locally', params)
        
        # train_locally should have default value of False
        train_locally_param = sig.parameters['train_locally']
        self.assertEqual(train_locally_param.default, False)


class TestStage1RegressionPrevention(unittest.TestCase):
    """Tests to ensure no regressions in existing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = OptimizationConfig(
            dataset_name="MNIST",
            optimize_for="val_accuracy",
            mode=OptimizationMode.SIMPLE,
            objective=OptimizationObjective.ACCURACY,
            trials=1,
            use_runpod_service=False  # Test local execution
        )
        
        self.optimizer = ModelOptimizer(
            dataset_name="MNIST",
            config=self.config
        )

    @patch('src.optimizer.create_and_train_model')
    def test_local_execution_path_unchanged(self, mock_create_train):
        """Test that local execution path works identically to before"""
        # Mock the create_and_train_model function
        mock_create_train.return_value = {
            'model_builder': Mock(),
            'test_accuracy': 0.94,
            'test_loss': 0.15
        }
        
        # Mock trial
        mock_trial = Mock()
        mock_trial.number = 0
        mock_params = {'learning_rate': 0.001}
        
        # This should work exactly as before
        with patch.object(self.optimizer, '_setup_trial_directory'):
            with patch.object(self.optimizer, '_staggered_start'):
                result = self.optimizer._train_locally_for_trial(mock_trial, mock_params)
                
                # Should return a valid score
                self.assertIsInstance(result, (int, float))
                self.assertGreaterEqual(result, 0)

    def test_configuration_unchanged(self):
        """Test that configuration handling remains unchanged"""
        # Original configuration should work
        self.assertIsInstance(self.optimizer.config, OptimizationConfig)
        self.assertEqual(self.optimizer.dataset_name, "MNIST")
        self.assertFalse(self.optimizer.config.use_runpod_service)


def run_stage1_tests():
    """Run all Stage 1 tests"""
    print("=" * 70)
    print("STAGE 1 AUTOMATED TESTING: Training Function Unification")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestStage1TrainingUnification))
    suite.addTest(unittest.makeSuite(TestStage1RegressionPrevention))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("STAGE 1 TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nSTAGE 1 AUTOMATED TESTING: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == '__main__':
    success = run_stage1_tests()
    sys.exit(0 if success else 1)