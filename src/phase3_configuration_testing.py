#!/usr/bin/env python3
"""
Phase 3 Step 3.1: Configuration Testing Plan
Hyperparameter Optimization System Refactoring

Comprehensive testing framework to verify all configuration combinations
work correctly with the new modular architecture.

Testing Matrix:
- Plot Generation Modes: ALL, BEST, NONE
- Optimization Modes: SIMPLE, HEALTH  
- Dataset Types: Image (CIFAR-10), Text (simulated)
- GPU Proxy: Enabled/Disabled
- Edge Cases: Various error scenarios

Usage:
    python phase3_configuration_testing.py
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import tempfile
import shutil

# Add project root to path for imports
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from optimizer import (
    ModelOptimizer, OptimizationConfig, OptimizationMode, 
    OptimizationObjective, PlotGenerationMode, optimize_model
)
from utils.logger import logger


class ConfigurationTester:
    """
    Comprehensive testing framework for the refactored modular architecture
    
    Tests all major configuration combinations to ensure Phase 2 refactoring
    maintains full functionality while adding new capabilities.
    """
    
    def __init__(self):
        """Initialize the configuration tester"""
        self.test_results: List[Dict[str, Any]] = []
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        
        # Create temporary directory for test runs
        self.temp_dir = Path(tempfile.mkdtemp(prefix="hyperopt_test_"))
        logger.debug(f"running ConfigurationTester.__init__ ... Test temp directory: {self.temp_dir}")
        
        # Test configuration matrix
        self.test_matrix = self._build_test_matrix()
        logger.debug(f"running ConfigurationTester.__init__ ... Built test matrix with {len(self.test_matrix)} test cases")
    
    def _build_test_matrix(self) -> List[Dict[str, Any]]:
        """Build comprehensive test matrix for all configuration combinations"""
        
        test_cases = []
        
        # Basic functionality tests (quick validation)
        basic_tests = [
            {
                'name': 'basic_simple_all_plots',
                'description': 'Simple mode with all plots generation',
                'config': {
                    'mode': 'simple',
                    'optimize_for': 'val_accuracy',
                    'plot_generation': 'all',
                    'trials': 2,  # Minimal for speed
                    'use_gpu_proxy': False,
                    # TESTING OPTIMIZATION: Much shorter training
                    'max_epochs_per_trial': 3,
                    'min_epochs_per_trial': 2
                },
                'dataset': 'cifar10',
                'expected_plots': True,
                'expected_model': True,
                'priority': 'high'
            },
            {
                'name': 'basic_simple_best_plots',
                'description': 'Simple mode with best trial plots only (CRITICAL TEST)',
                'config': {
                    'mode': 'simple',
                    'optimize_for': 'val_accuracy', 
                    'plot_generation': 'best',
                    'trials': 2,
                    'use_gpu_proxy': False,
                    # TESTING OPTIMIZATION: Much shorter training
                    'max_epochs_per_trial': 3,
                    'min_epochs_per_trial': 2
                },
                'dataset': 'cifar10',
                'expected_plots': True,
                'expected_model': True,
                'priority': 'high' 
            },
            {
                'name': 'basic_simple_no_plots',
                'description': 'Simple mode with no plots generation',
                'config': {
                    'mode': 'simple',
                    'optimize_for': 'val_accuracy',
                    'plot_generation': 'none',
                    'trials': 2,
                    'use_gpu_proxy': False,
                    # TESTING OPTIMIZATION: Much shorter training
                    'max_epochs_per_trial': 3,
                    'min_epochs_per_trial': 2
                },
                'dataset': 'cifar10',
                'expected_plots': False,
                'expected_model': False,  # No model saving in NONE mode
                'priority': 'high'
            },
            {
                'name': 'health_mode_test',
                'description': 'Health mode with universal objective',
                'config': {
                    'mode': 'health',
                    'optimize_for': 'val_accuracy',
                    'plot_generation': 'best',
                    'trials': 2,
                    'use_gpu_proxy': False,
                    'health_weight': 0.3,
                    # TESTING OPTIMIZATION: Much shorter training
                    'max_epochs_per_trial': 3,
                    'min_epochs_per_trial': 2
                },
                'dataset': 'cifar10',
                'expected_plots': True,
                'expected_model': True,
                'priority': 'high'
            }
        ]
        
        # Extended functionality tests
        extended_tests = [
            {
                'name': 'hyperparameter_selector_integration',
                'description': 'Test HyperparameterSelector module integration',
                'config': {
                    'mode': 'simple',
                    'optimize_for': 'val_accuracy',
                    'plot_generation': 'none',  # Focus on hyperparameter logic
                    'trials': 1,
                    'use_gpu_proxy': False,
                    # TESTING OPTIMIZATION: Very short training
                    'max_epochs_per_trial': 2,
                    'min_epochs_per_trial': 1
                },
                'dataset': 'cifar10',
                'expected_plots': False,
                'expected_model': False,
                'priority': 'medium',
                'validation_focus': 'hyperparameters'
            },
            {
                'name': 'plot_generator_integration',
                'description': 'Test PlotGenerator module integration',
                'config': {
                    'mode': 'simple',
                    'optimize_for': 'val_accuracy',
                    'plot_generation': 'all',
                    'trials': 1,
                    'use_gpu_proxy': False,
                    # TESTING OPTIMIZATION: Very short training
                    'max_epochs_per_trial': 2,
                    'min_epochs_per_trial': 1
                },
                'dataset': 'cifar10',
                'expected_plots': True,
                'expected_model': True,
                'priority': 'medium',
                'validation_focus': 'plots'
            },
            {
                'name': 'activation_override_test',
                'description': 'Test activation function override functionality',
                'config': {
                    'mode': 'simple',
                    'optimize_for': 'val_accuracy',
                    'plot_generation': 'none',
                    'trials': 1,
                    'use_gpu_proxy': False,
                    'activation': 'swish',  # Override activation
                    # TESTING OPTIMIZATION: Very short training
                    'max_epochs_per_trial': 2,
                    'min_epochs_per_trial': 1
                },
                'dataset': 'cifar10',
                'expected_plots': False,
                'expected_model': False,
                'priority': 'high',
                'validation_focus': 'activation_override'
            }
        ]
        
        # GPU proxy tests (conditional based on availability)
        gpu_proxy_tests = [
            {
                'name': 'gpu_proxy_disabled',
                'description': 'GPU proxy explicitly disabled',
                'config': {
                    'mode': 'simple',
                    'optimize_for': 'val_accuracy',
                    'plot_generation': 'none',
                    'trials': 1,
                    'use_gpu_proxy': False,
                    # TESTING OPTIMIZATION: Very short training
                    'max_epochs_per_trial': 2,
                    'min_epochs_per_trial': 1
                },
                'dataset': 'cifar10',
                'expected_plots': False,
                'expected_model': False,
                'priority': 'medium',
                'validation_focus': 'gpu_proxy'
            }
            # Note: GPU proxy enabled tests would require actual GPU proxy setup
            # These should be added when GPU proxy is available in test environment
        ]
        
        # Edge case tests
        edge_case_tests = [
            {
                'name': 'minimal_trials',
                'description': 'Test with minimal number of trials',
                'config': {
                    'mode': 'simple',
                    'optimize_for': 'val_accuracy',
                    'plot_generation': 'best',
                    'trials': 1,  # Absolute minimum
                    'use_gpu_proxy': False,
                    # TESTING OPTIMIZATION: Minimal training time
                    'max_epochs_per_trial': 2,
                    'min_epochs_per_trial': 1
                },
                'dataset': 'cifar10',
                'expected_plots': True,
                'expected_model': True,
                'priority': 'low',
                'validation_focus': 'edge_cases'
            }
        ]
        
        # Combine all test categories
        test_cases.extend(basic_tests)
        test_cases.extend(extended_tests) 
        test_cases.extend(gpu_proxy_tests)
        test_cases.extend(edge_case_tests)
        
        return test_cases
    
    def run_all_tests(self, quick_mode: bool = True) -> Dict[str, Any]:
        """
        Run all configuration tests
        
        Args:
            quick_mode: If True, runs only critical and high priority tests
            
        Returns:
            Dictionary with test results summary
        """
        logger.debug(f"running ConfigurationTester.run_all_tests ... Starting comprehensive configuration testing")
        logger.debug(f"running ConfigurationTester.run_all_tests ... Quick mode: {quick_mode}")
        
        start_time = datetime.now()
        
        # Filter tests based on mode
        if quick_mode:
            tests_to_run = [t for t in self.test_matrix if t.get('priority') in ['critical', 'high']]
            logger.debug(f"running ConfigurationTester.run_all_tests ... Quick mode: running {len(tests_to_run)} high-priority tests")
        else:
            tests_to_run = self.test_matrix
            logger.debug(f"running ConfigurationTester.run_all_tests ... Full mode: running all {len(tests_to_run)} tests")
        
        # Run each test
        for i, test_case in enumerate(tests_to_run, 1):
            logger.debug(f"running ConfigurationTester.run_all_tests ... Running test {i}/{len(tests_to_run)}: {test_case['name']}")
            
            try:
                result = self._run_single_test(test_case)
                self.test_results.append(result)
                
                if result['status'] == 'PASSED':
                    self.passed_tests += 1
                    logger.debug(f"running ConfigurationTester.run_all_tests ... ✅ PASSED: {test_case['name']}")
                elif result['status'] == 'FAILED':
                    self.failed_tests += 1
                    logger.warning(f"running ConfigurationTester.run_all_tests ... ❌ FAILED: {test_case['name']} - {result.get('error', 'Unknown error')}")
                else:
                    self.skipped_tests += 1
                    logger.debug(f"running ConfigurationTester.run_all_tests ... ⏭️ SKIPPED: {test_case['name']} - {result.get('reason', 'Unknown reason')}")
                    
            except Exception as e:
                logger.error(f"running ConfigurationTester.run_all_tests ... Test execution failed for {test_case['name']}: {e}")
                self.failed_tests += 1
                self.test_results.append({
                    'name': test_case['name'],
                    'status': 'FAILED',
                    'error': f"Test execution failed: {str(e)}",
                    'duration': 0
                })
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Compile results
        results_summary = {
            'total_tests': len(tests_to_run),
            'passed': self.passed_tests,
            'failed': self.failed_tests,
            'skipped': self.skipped_tests,
            'duration_seconds': total_duration,
            'success_rate': (self.passed_tests / len(tests_to_run) * 100) if tests_to_run else 0,
            'critical_issues': self._identify_critical_issues(),
            'test_results': self.test_results
        }
        
        logger.debug(f"running ConfigurationTester.run_all_tests ... Testing completed in {total_duration:.1f}s")
        logger.debug(f"running ConfigurationTester.run_all_tests ... Results: {self.passed_tests} passed, {self.failed_tests} failed, {self.skipped_tests} skipped")
        
        return results_summary
    
    def _run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single configuration test
        
        Args:
            test_case: Test case configuration
            
        Returns:
            Test result dictionary
        """
        test_name = test_case['name']
        start_time = datetime.now()
        
        try:
            logger.debug(f"running _run_single_test ... Starting test: {test_name}")
            logger.debug(f"running _run_single_test ... Description: {test_case['description']}")
            
            # Create unique run name for this test
            test_run_name = f"test_{test_name}_{datetime.now().strftime('%H%M%S')}"
            
            # Extract configuration
            config = test_case['config'].copy()
            dataset_name = test_case['dataset']
            
            # Add test-specific parameters
            config['run_name'] = test_run_name
            
            # Ensure plot_generation is passed as string (will be converted by optimize_model)
            if 'plot_generation' in config:
                # Convert to string if it's not already
                if hasattr(config['plot_generation'], 'value'):
                    config['plot_generation'] = config['plot_generation'].value
                logger.debug(f"running _run_single_test ... plot_generation parameter: {config['plot_generation']} (type: {type(config['plot_generation'])})")
            
            # Run the optimization with the test configuration
            logger.debug(f"running _run_single_test ... Executing optimization with config: {config}")
            
            result = optimize_model(
                dataset_name=dataset_name,
                **config
            )
            
            # Validate results based on test expectations
            validation_result = self._validate_test_result(test_case, result, test_run_name)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Error capture for validation failures
            test_status = 'PASSED' if validation_result['valid'] else 'FAILED'
            test_error = None
            
            if not validation_result['valid']:
                # Create detailed error message from validation issues
                validation_issues = validation_result.get('issues', ['Unknown validation error'])
                test_error = f"Validation failed: {'; '.join(validation_issues)}"
                logger.warning(f"running _run_single_test ... {test_name} validation failed: {test_error}")            
            
            return {
                'name': test_name,
                'status': test_status,
                'duration': duration,
                'config': config,
                'validation': validation_result,
                'error': test_error,
                'optimization_result': {
                    'best_value': result.best_value,
                    'successful_trials': result.successful_trials,
                    'total_trials': result.total_trials,
                    'best_params': getattr(result, 'best_params', {})
                }
            }
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                'name': test_name,
                'status': 'FAILED',
                'duration': duration,
                'config': test_case.get('config', {}),
                'error': f"Test execution failed: {str(e)}",
                'traceback': traceback.format_exc()
            }
    
    def _validate_test_result(
        self, 
        test_case: Dict[str, Any], 
        optimization_result: Any,
        test_run_name: str
    ) -> Dict[str, Any]:
        """
        Validate that test results match expectations
        
        Args:
            test_case: Original test case configuration
            optimization_result: Result from optimize_model()
            test_run_name: Unique test run name
            
        Returns:
            Validation result dictionary
        """
        validation_issues = []
        
        try:
            # Basic result validation
            if optimization_result.successful_trials == 0:
                validation_issues.append("No successful trials completed")
            
            if optimization_result.best_value <= 0:
                validation_issues.append(f"Invalid best value: {optimization_result.best_value}")
            
            # Check if results directory exists
            if optimization_result.results_dir and not optimization_result.results_dir.exists():
                validation_issues.append(f"Results directory not created: {optimization_result.results_dir}")
            
            # Validate plot generation expectations
            expected_plots = test_case.get('expected_plots', False)
            if expected_plots:
                plot_validation = self._validate_plot_generation(optimization_result, test_case)
                if not plot_validation['valid']:
                    validation_issues.extend(plot_validation['issues'])
            
            # Validate model saving expectations  
            expected_model = test_case.get('expected_model', False)
            if expected_model:
                model_validation = self._validate_model_saving(optimization_result, test_case)
                if not model_validation['valid']:
                    validation_issues.extend(model_validation['issues'])
            
            # Special validation based on focus area
            validation_focus = test_case.get('validation_focus')
            if validation_focus:
                focus_validation = self._validate_focus_area(optimization_result, test_case, validation_focus)
                if not focus_validation['valid']:
                    validation_issues.extend(focus_validation['issues'])
            
            return {
                'valid': len(validation_issues) == 0,
                'issues': validation_issues,
                'checks_performed': self._get_validation_checks(test_case)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f"Validation failed with error: {str(e)}"],
                'checks_performed': ['basic_validation']
            }
    
    def _validate_plot_generation(self, result: Any, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Validate plot generation based on configuration"""
        issues = []
        
        if not result.results_dir:
            issues.append("No results directory available for plot validation")
            return {'valid': False, 'issues': issues}
        
        plot_mode = test_case['config'].get('plot_generation', 'all')
        
        if plot_mode == 'all':
            # Should have plots in trial directories
            plots_dir = result.results_dir / "plots"
            if not plots_dir.exists():
                issues.append("Plots directory not created for 'all' mode")
            else:
                trial_dirs = list(plots_dir.glob("trial_*"))
                if not trial_dirs:
                    issues.append("No trial directories found for 'all' mode")
        
        elif plot_mode == 'best':
            # Should have plots in optimized_model directory
            optimized_dir = result.results_dir / "optimized_model"
            if not optimized_dir.exists():
                issues.append("Optimized model directory not created for 'best' mode")
            else:
                # Look for plot files (any image files)
                plot_files = list(optimized_dir.glob("*.png")) + list(optimized_dir.glob("*.jpg")) + list(optimized_dir.glob("*.svg"))
                if not plot_files:
                    issues.append("No plot files found in optimized_model directory for 'best' mode")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _validate_model_saving(self, result: Any, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model saving based on configuration"""
        issues = []
        
        if not result.results_dir:
            issues.append("No results directory available for model validation")
            return {'valid': False, 'issues': issues}
        
        plot_mode = test_case['config'].get('plot_generation', 'all')
        
        if plot_mode in ['all', 'best']:
            # Look for model files
            if plot_mode == 'all':
                # Models should be in trial directories
                plots_dir = result.results_dir / "plots"
                if plots_dir.exists():
                    model_files = list(plots_dir.glob("**/*.tf")) + list(plots_dir.glob("**/*.keras"))
                    if not model_files:
                        issues.append(f"No model files found in trial directories for '{plot_mode}' mode")
                else:
                    issues.append("Plots directory not found for model validation")
            
            elif plot_mode == 'best':
                # Models should be in optimized_model directory
                optimized_dir = result.results_dir / "optimized_model"
                if optimized_dir.exists():
                    model_files = list(optimized_dir.glob("*.tf")) + list(optimized_dir.glob("*.keras"))
                    if not model_files:
                        issues.append("No model files found in optimized_model directory for 'best' mode")
                else:
                    issues.append("Optimized model directory not found")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    
    def _validate_focus_area(self, result: Any, test_case: Dict[str, Any], focus: str) -> Dict[str, Any]:
        """Validate specific focus areas for targeted tests"""
        issues = []
        
        if focus == 'hyperparameters':
            # Validate that hyperparameters were generated correctly
            if not result.best_params:
                issues.append("No best parameters found")
            else:
                # Check for expected hyperparameter types
                expected_params = ['epochs', 'optimizer']
                for param in expected_params:
                    if param not in result.best_params:
                        issues.append(f"Expected parameter '{param}' not found in best_params")
        
        elif focus == 'plots':
            # Enhanced plot validation
            plot_validation = self._validate_plot_generation(result, test_case)
            if not plot_validation['valid']:
                issues.extend(plot_validation['issues'])
        
        elif focus == 'activation_override':
            # Activation override validation with detailed logging
            activation = test_case['config'].get('activation')
            if activation:
                # Check if result has best_params
                if not hasattr(result, 'best_params') or not result.best_params:
                    issues.append("No best_params found in optimization result - cannot validate activation override")
                else:
                    actual_activation = result.best_params.get('activation', '')
                    
                    # Log for debugging
                    logger.debug(f"running _validate_focus_area ... Expected activation: '{activation}'")
                    logger.debug(f"running _validate_focus_area ... Actual activation: '{actual_activation}'")
                    logger.debug(f"running _validate_focus_area ... All best_params keys: {list(result.best_params.keys())}")
                    
                    if actual_activation == activation:
                        logger.debug(f"running _validate_focus_area ... ✅ Activation override validation PASSED")
                        pass  # Good, activation was applied correctly
                    else:
                        issues.append(f"Activation override failed: expected '{activation}', got '{actual_activation}'")
                        logger.warning(f"running _validate_focus_area ... ❌ Activation override validation FAILED: expected '{activation}', got '{actual_activation}'")
            else:
                issues.append("No activation override specified in test config")
        
        elif focus == 'gpu_proxy':
            # Validate GPU proxy configuration
            use_gpu_proxy = test_case['config'].get('use_gpu_proxy', False)
            # For now, just validate that the test completed (GPU proxy availability varies)
            if result.successful_trials == 0 and use_gpu_proxy:
                issues.append("GPU proxy test failed - no successful trials")
        
        elif focus == 'edge_cases':
            # Validate edge case handling
            if result.total_trials != test_case['config'].get('trials', 1):
                issues.append(f"Expected {test_case['config'].get('trials', 1)} trials, got {result.total_trials}")
        
        return {'valid': len(issues) == 0, 'issues': issues}
    
    def _get_validation_checks(self, test_case: Dict[str, Any]) -> List[str]:
        """Get list of validation checks performed for a test case"""
        checks = ['basic_validation']
        
        if test_case.get('expected_plots'):
            checks.append('plot_generation')
        
        if test_case.get('expected_model'):
            checks.append('model_saving')
        
        if test_case.get('validation_focus'):
            checks.append(f"focus_{test_case['validation_focus']}")
        
        return checks
    
    def _identify_critical_issues(self) -> List[str]:
        """Identify critical issues from test results"""
        critical_issues = []
        
        # Look for critical test failures
        for result in self.test_results:
            if result.get('status') == 'FAILED':
                test_name = result.get('name', 'unknown')
                
                # Critical tests that must pass
                if 'critical' in test_name or 'basic_simple_best_plots' in test_name:
                    error = result.get('error', 'Unknown error')
                    critical_issues.append(f"CRITICAL: {test_name} failed - {error}")
                
                # Look for specific critical failure patterns
                error_msg = str(result.get('error', '')).lower()
                if 'plot_generation=best' in error_msg:
                    critical_issues.append(f"CRITICAL: BEST mode functionality failure in {test_name}")
                
                if 'hyperparameter' in error_msg:
                    critical_issues.append(f"CRITICAL: Hyperparameter module failure in {test_name}")
        
        return critical_issues
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report"""
        
        report_lines = [
            "=" * 80,
            "PHASE 3 STEP 3.1: CONFIGURATION TESTING REPORT",
            "=" * 80,
            f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Tests: {results['total_tests']}",
            f"Passed: {results['passed']} ({results['success_rate']:.1f}%)",
            f"Failed: {results['failed']}",
            f"Skipped: {results['skipped']}",
            f"Duration: {results['duration_seconds']:.1f} seconds",
            "",
        ]
        
        # Critical issues section
        if results['critical_issues']:
            report_lines.extend([
                "🚨 CRITICAL ISSUES:",
                "-" * 40
            ])
            for issue in results['critical_issues']:
                report_lines.append(f"  ❌ {issue}")
            report_lines.append("")
        else:
            report_lines.extend([
                "✅ NO CRITICAL ISSUES FOUND",
                ""
            ])
        
        # Detailed test results
        report_lines.extend([
            "DETAILED TEST RESULTS:",
            "-" * 40
        ])
        
        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'PASSED' else "❌" if result['status'] == 'FAILED' else "⏭️"
            duration = result.get('duration', 0)
            
            report_lines.append(f"{status_icon} {result['name']} ({duration:.1f}s)")
            
            if result['status'] == 'FAILED':
                error = result.get('error', 'Unknown error')
                report_lines.append(f"    Error: {error}")
            
            # Show validation details for passed tests
            elif result['status'] == 'PASSED' and 'validation' in result:
                validation = result['validation']
                checks = validation.get('checks_performed', [])
                report_lines.append(f"    Validation: {', '.join(checks)}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "PHASE 3 STEP 3.1 COMPLETE" if results['failed'] == 0 else "PHASE 3 STEP 3.1 NEEDS ATTENTION",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def cleanup(self):
        """Clean up temporary test files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.debug(f"running ConfigurationTester.cleanup ... Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"running ConfigurationTester.cleanup ... Failed to cleanup temp directory: {e}")


def run_phase3_step1_testing(quick_mode: bool = True) -> Dict[str, Any]:
    """
    Main function to run Phase 3 Step 3.1 configuration testing
    
    Args:
        quick_mode: If True, runs only critical and high priority tests
        
    Returns:
        Test results summary
    """
    logger.debug("running run_phase3_step1_testing ... Starting Phase 3 Step 3.1: Configuration Testing")
    
    tester = ConfigurationTester()
    
    try:
        # Run all tests
        results = tester.run_all_tests(quick_mode=quick_mode)
        
        # Generate and display report
        report = tester.generate_report(results)
        print(report)
        
        # Log summary
        if results['failed'] == 0:
            logger.debug("running run_phase3_step1_testing ... ✅ All tests passed! Phase 3 Step 3.1 COMPLETE")
        else:
            logger.warning(f"running run_phase3_step1_testing ... ❌ {results['failed']} tests failed. Phase 3 Step 3.1 needs attention")
        
        return results
        
    except Exception as e:
        logger.error(f"running run_phase3_step1_testing ... Phase 3 Step 3.1 testing failed: {e}")
        raise
    
    finally:
        # Cleanup
        tester.cleanup()


if __name__ == "__main__":
    # Command line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 3 Step 3.1: Configuration Testing")
    parser.add_argument(
        '--full', 
        action='store_true', 
        help='Run full test suite (default: quick mode with critical/high priority tests only)'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_phase3_step1_testing(quick_mode=not args.full)
        
        # Exit with appropriate code
        if results['failed'] > 0:
            print(f"\n❌ Phase 3 Step 3.1 FAILED: {results['failed']} test(s) failed")
            sys.exit(1)
        else:
            print(f"\n✅ Phase 3 Step 3.1 PASSED: All {results['passed']} test(s) successful")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n💥 Phase 3 Step 3.1 ERROR: {e}")
        sys.exit(1)