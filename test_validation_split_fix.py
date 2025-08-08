#!/usr/bin/env python3
"""
Phase 4 Verification: Test Manual Validation Split Implementation

This script tests the manual validation split fix across different validation split values
to verify that GPU proxy no longer returns all-zero validation accuracy.

Based on status.md findings:
- Local environment: validation_split works correctly 
- GPU environment: validation_split was broken (all zeros or constant near-zero)
- Fix: Manual validation split using validation_data parameter

This test verifies the fix is working properly.

Place this file in project root directory (same level as status.md)
"""

import sys
from pathlib import Path
import traceback
from typing import Dict, Any, Optional

# Project structure: project_root/test_validation_split_fix.py
# Source files are in: project_root/src/
current_file = Path(__file__)
project_root = current_file.parent  # Script is in project root
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from src.utils.logger import logger
from src.optimizer import optimize_model


def test_validation_split_values(
    dataset_name: str = "cifar10",
    use_gpu_proxy: bool = True,
    test_local_comparison: bool = True
) -> Dict[str, Any]:
    """
    Test manual validation split implementation across multiple validation split values
    
    Args:
        dataset_name: Dataset to test with
        use_gpu_proxy: Whether to use GPU proxy (True) or local execution (False)
        test_local_comparison: Whether to also test local execution for comparison
        
    Returns:
        Dictionary with test results for each validation split value
    """
    logger.debug("running test_validation_split_values ... Starting Phase 4 verification testing")
    
    # Test different validation split values (same as in status.md)
    validation_splits = [0.1, 0.2, 0.3]
    
    results = {
        'validation_splits_tested': validation_splits,
        'gpu_proxy_used': use_gpu_proxy,
        'test_results': {},
        'local_comparison': {} if test_local_comparison else None,
        'summary': {}
    }
    
    for val_split in validation_splits:
        logger.debug(f"running test_validation_split_values ... Testing validation_split={val_split}")
        
        try:
            # Test with GPU proxy (or local if gpu_proxy=False)
            execution_mode = "GPU_PROXY" if use_gpu_proxy else "LOCAL"
            logger.debug(f"running test_validation_split_values ... Testing {execution_mode} execution with validation_split={val_split}")
            
            result = optimize_model(
                dataset_name=dataset_name,
                mode="simple",
                optimize_for="val_accuracy", 
                trials=2,  # Just 2 trials for quick testing
                use_gpu_proxy=use_gpu_proxy,
                gpu_proxy_sample_percentage=0.10,  # Use small sample for faster testing
                validation_split=val_split,
                plot_generation="none"  # Skip plots for faster testing
            )
            
            # Extract validation accuracy information from the optimization
            # The key test is whether we get non-zero validation accuracy values
            test_result = {
                'validation_split': val_split,
                'execution_mode': execution_mode,
                'best_value': result.best_value,
                'successful_trials': result.successful_trials,
                'total_trials': result.total_trials,
                'optimization_successful': result.successful_trials > 0,
                'non_zero_validation': result.best_value > 0.0,  # Key test: is validation accuracy non-zero?
                'manual_split_fix_working': result.best_value > 0.0  # Manual split fix working if we get non-zero values
            }
            
            results['test_results'][val_split] = test_result
            
            logger.debug(f"running test_validation_split_values ... {execution_mode} test completed for validation_split={val_split}")
            logger.debug(f"running test_validation_split_values ... - Best value: {result.best_value:.4f}")
            logger.debug(f"running test_validation_split_values ... - Successful trials: {result.successful_trials}/{result.total_trials}")
            logger.debug(f"running test_validation_split_values ... - Non-zero validation: {test_result['non_zero_validation']}")
            logger.debug(f"running test_validation_split_values ... - Manual split fix working: {test_result['manual_split_fix_working']}")
            
        except Exception as e:
            logger.error(f"running test_validation_split_values ... {execution_mode} test failed for validation_split={val_split}: {e}")
            results['test_results'][val_split] = {
                'validation_split': val_split,
                'execution_mode': execution_mode,
                'error': str(e),
                'optimization_successful': False,
                'non_zero_validation': False,
                'manual_split_fix_working': False
            }
    
    # Test local execution for comparison (only if we tested GPU proxy above)
    if test_local_comparison and use_gpu_proxy:
        logger.debug(f"running test_validation_split_values ... Running local execution comparison tests")
        
        for val_split in validation_splits:
            try:
                logger.debug(f"running test_validation_split_values ... Testing LOCAL execution with validation_split={val_split} (comparison)")
                
                result_local = optimize_model(
                    dataset_name=dataset_name,
                    mode="simple",
                    optimize_for="val_accuracy",
                    trials=2,  # Just 2 trials for quick testing
                    use_gpu_proxy=False,  # Force local execution
                    validation_split=val_split,
                    plot_generation="none"  # Skip plots for faster testing
                )
                
                local_result = {
                    'validation_split': val_split,
                    'execution_mode': 'LOCAL_COMPARISON',
                    'best_value': result_local.best_value,
                    'successful_trials': result_local.successful_trials,
                    'total_trials': result_local.total_trials,
                    'optimization_successful': result_local.successful_trials > 0,
                    'non_zero_validation': result_local.best_value > 0.0,
                    'manual_split_working': result_local.best_value > 0.0
                }
                
                results['local_comparison'][val_split] = local_result
                
                logger.debug(f"running test_validation_split_values ... LOCAL comparison completed for validation_split={val_split}")
                logger.debug(f"running test_validation_split_values ... - Best value: {result_local.best_value:.4f}")
                logger.debug(f"running test_validation_split_values ... - Non-zero validation: {local_result['non_zero_validation']}")
                
            except Exception as e:
                logger.error(f"running test_validation_split_values ... LOCAL comparison failed for validation_split={val_split}: {e}")
                results['local_comparison'][val_split] = {
                    'validation_split': val_split,
                    'execution_mode': 'LOCAL_COMPARISON',
                    'error': str(e),
                    'optimization_successful': False,
                    'non_zero_validation': False,
                    'manual_split_working': False
                }
    
    # Generate summary
    results['summary'] = _generate_test_summary(results)
    
    return results


def _generate_test_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of test results"""
    
    test_results = results['test_results']
    local_comparison = results.get('local_comparison', {})
    
    # Count successful tests
    total_tests = len(test_results)
    successful_tests = sum(1 for r in test_results.values() if r.get('optimization_successful', False))
    non_zero_validation_tests = sum(1 for r in test_results.values() if r.get('non_zero_validation', False))
    manual_split_working_tests = sum(1 for r in test_results.values() if r.get('manual_split_fix_working', False))
    
    # Local comparison stats
    local_stats = {}
    if local_comparison:
        local_successful = sum(1 for r in local_comparison.values() if r.get('optimization_successful', False))
        local_non_zero = sum(1 for r in local_comparison.values() if r.get('non_zero_validation', False))
        local_stats = {
            'local_tests': len(local_comparison),
            'local_successful': local_successful,
            'local_non_zero_validation': local_non_zero,
            'local_success_rate': local_successful / len(local_comparison) if local_comparison else 0.0,
            'local_non_zero_rate': local_non_zero / len(local_comparison) if local_comparison else 0.0
        }
    
    summary = {
        'total_validation_splits_tested': results['validation_splits_tested'],
        'gpu_proxy_used': results['gpu_proxy_used'],
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'non_zero_validation_tests': non_zero_validation_tests,
        'manual_split_working_tests': manual_split_working_tests,
        'success_rate': successful_tests / total_tests if total_tests > 0 else 0.0,
        'non_zero_validation_rate': non_zero_validation_tests / total_tests if total_tests > 0 else 0.0,
        'manual_split_fix_success_rate': manual_split_working_tests / total_tests if total_tests > 0 else 0.0,
        **local_stats,
        'phase_4_verification_status': 'PASSED' if manual_split_working_tests == total_tests else 'FAILED',
        'conclusion': _generate_conclusion(results)
    }
    
    return summary


def _generate_conclusion(results: Dict[str, Any]) -> str:
    """Generate conclusion about Phase 4 verification"""
    
    test_results = results['test_results']
    total_tests = len(test_results)
    manual_split_working = sum(1 for r in test_results.values() if r.get('manual_split_fix_working', False))
    
    if manual_split_working == total_tests:
        return "✅ PHASE 4 VERIFICATION PASSED: Manual validation split fix is working correctly across all tested validation split values. GPU proxy no longer returns all-zero validation accuracy."
    elif manual_split_working > 0:
        return f"⚠️ PHASE 4 VERIFICATION PARTIAL: Manual validation split fix working for {manual_split_working}/{total_tests} validation split values. Some issues remain."
    else:
        return "❌ PHASE 4 VERIFICATION FAILED: Manual validation split fix is not working. GPU proxy still returning zero/invalid validation accuracy."


def print_test_results(results: Dict[str, Any]) -> None:
    """Print formatted test results"""
    
    print("\n" + "="*80)
    print("PHASE 4 VERIFICATION: Manual Validation Split Testing Results")
    print("="*80)
    
    # Test configuration
    print(f"\nTest Configuration:")
    print(f"- GPU Proxy Used: {results['gpu_proxy_used']}")
    print(f"- Validation Splits Tested: {results['validation_splits_tested']}")
    print(f"- Local Comparison: {'Yes' if results.get('local_comparison') else 'No'}")
    
    # Main test results
    print(f"\nMain Test Results:")
    print("-" * 40)
    
    for val_split, result in results['test_results'].items():
        status = "✅ PASS" if result.get('manual_split_fix_working', False) else "❌ FAIL"
        best_value = result.get('best_value', 0.0)
        execution_mode = result.get('execution_mode', 'UNKNOWN')
        
        print(f"validation_split={val_split:3.1f} ({execution_mode:>9}): {status} - Best Value: {best_value:.4f}")
        
        if 'error' in result:
            print(f"                           Error: {result['error']}")
    
    # Local comparison results
    if results.get('local_comparison'):
        print(f"\nLocal Comparison Results:")
        print("-" * 40)
        
        for val_split, result in results['local_comparison'].items():
            status = "✅ PASS" if result.get('manual_split_working', False) else "❌ FAIL"
            best_value = result.get('best_value', 0.0)
            
            print(f"validation_split={val_split:3.1f} (     LOCAL): {status} - Best Value: {best_value:.4f}")
            
            if 'error' in result:
                print(f"                           Error: {result['error']}")
    
    # Summary
    summary = results['summary']
    print(f"\nTest Summary:")
    print("-" * 40)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful Tests: {summary['successful_tests']}")
    print(f"Non-Zero Validation Tests: {summary['non_zero_validation_tests']}")
    print(f"Manual Split Fix Working: {summary['manual_split_working_tests']}")
    print(f"Fix Success Rate: {summary['manual_split_fix_success_rate']:.1%}")
    
    if results.get('local_comparison'):
        print(f"Local Success Rate: {summary.get('local_success_rate', 0.0):.1%}")
        print(f"Local Non-Zero Rate: {summary.get('local_non_zero_rate', 0.0):.1%}")
    
    # Final status
    print(f"\nPhase 4 Verification Status: {summary['phase_4_verification_status']}")
    print(f"\nConclusion:")
    print(summary['conclusion'])
    print("\n" + "="*80)


def main():
    """Main testing function"""
    
    logger.debug("running main ... Starting Phase 4 verification testing")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test manual validation split fix")
    parser.add_argument('--dataset', default='cifar10', help='Dataset to test with')
    parser.add_argument('--gpu-proxy', action='store_true', default=True, help='Use GPU proxy (default: True)')
    parser.add_argument('--no-gpu-proxy', action='store_true', help='Disable GPU proxy (force local)')
    parser.add_argument('--local-comparison', action='store_true', default=True, help='Include local comparison (default: True)')
    parser.add_argument('--no-local-comparison', action='store_true', help='Skip local comparison')
    
    args = parser.parse_args()
    
    # Handle GPU proxy setting
    use_gpu_proxy = args.gpu_proxy and not args.no_gpu_proxy
    test_local_comparison = args.local_comparison and not args.no_local_comparison
    
    logger.debug(f"running main ... Test configuration:")
    logger.debug(f"running main ... - Dataset: {args.dataset}")
    logger.debug(f"running main ... - GPU Proxy: {use_gpu_proxy}")
    logger.debug(f"running main ... - Local Comparison: {test_local_comparison}")
    
    try:
        # Run the test
        results = test_validation_split_values(
            dataset_name=args.dataset,
            use_gpu_proxy=use_gpu_proxy,
            test_local_comparison=test_local_comparison
        )
        
        # Print results
        print_test_results(results)
        
        # Exit with appropriate code
        if results['summary']['phase_4_verification_status'] == 'PASSED':
            logger.debug("running main ... ✅ Phase 4 verification testing completed successfully")
            sys.exit(0)
        else:
            logger.debug("running main ... ❌ Phase 4 verification testing failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"running main ... Testing failed: {e}")
        logger.error(f"running main ... Traceback: {traceback.format_exc()}")
        print(f"\n❌ Testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()