#!/usr/bin/env python3
"""
Simple Phase 4 Verification Test

Quick test to verify manual validation split fix is working.
Place this file in project root (same level as status.md) and run it.

Project structure: 
- project_root/simple_validation_test.py (this file)
- project_root/src/optimizer.py
- project_root/src/utils/logger.py
"""

import sys
from pathlib import Path

# Project structure: project_root/simple_validation_test.py
# Source files are in: project_root/src/
current_file = Path(__file__)
project_root = current_file.parent  # Script is in project root
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from src.optimizer import optimize_model
from src.utils.logger import logger

def quick_validation_test():
    """Quick test of manual validation split implementation"""
    
    print("="*60)
    print("PHASE 4 VERIFICATION: Quick Validation Split Test")
    print("="*60)
    
    # Test parameters
    dataset_name = "cifar10"
    validation_splits = [0.1, 0.2, 0.3]
    
    print(f"Testing dataset: {dataset_name}")
    print(f"Testing validation splits: {validation_splits}")
    print(f"GPU Proxy: Enabled")
    print(f"Trials per test: 2 (quick test)")
    print("-"*60)
    
    results = {}
    
    for val_split in validation_splits:
        print(f"\n🧪 Testing validation_split = {val_split}")
        
        try:
            # Run optimization with GPU proxy - FIXED PARAMETERS
            result = optimize_model(
                dataset_name=dataset_name,
                mode="simple", 
                optimize_for="val_accuracy",
                trials=2,  # Quick test with just 2 trials
                use_gpu_proxy=True,
                gpu_proxy_sample_percentage=0.05,  # Use only 5% of data for speed
                validation_split=val_split,
                plot_generation="none",  # Skip plots for speed
                min_epochs_per_trial=2,  # FIXED: Set min epochs
                max_epochs_per_trial=3,  # FIXED: Set max epochs (must be >= min)
                n_startup_trials=1,
                n_warmup_steps=1
            )
            
            # Check if we got non-zero validation accuracy
            best_value = result.best_value
            success = result.successful_trials > 0
            non_zero_val = best_value > 0.0
            
            results[val_split] = {
                'success': success,
                'best_value': best_value,
                'non_zero_validation': non_zero_val,
                'trials': f"{result.successful_trials}/{result.total_trials}"
            }
            
            status = "✅ PASS" if non_zero_val else "❌ FAIL"
            print(f"   Result: {status}")
            print(f"   Best validation accuracy: {best_value:.4f}")
            print(f"   Successful trials: {result.successful_trials}/{result.total_trials}")
            
            if non_zero_val:
                print(f"   ✅ Manual validation split fix working!")
            else:
                print(f"   ❌ Still getting zero validation accuracy")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[val_split] = {
                'success': False,
                'error': str(e),
                'non_zero_validation': False
            }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = len(validation_splits)
    successful_tests = sum(1 for r in results.values() if r.get('non_zero_validation', False))
    
    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests:.1%}")
    
    if successful_tests == total_tests:
        print("\n🎉 PHASE 4 VERIFICATION: PASSED")
        print("✅ Manual validation split fix is working correctly!")
        print("✅ GPU proxy no longer returns all-zero validation accuracy")
        return True
    elif successful_tests > 0:
        print(f"\n⚠️  PHASE 4 VERIFICATION: PARTIAL SUCCESS")
        print(f"✅ Working for {successful_tests}/{total_tests} validation split values")
        print("❌ Some issues remain - check failed tests above")
        return False
    else:
        print(f"\n❌ PHASE 4 VERIFICATION: FAILED")
        print("❌ Manual validation split fix is not working")
        print("❌ GPU proxy still returning zero validation accuracy")
        return False

if __name__ == "__main__":
    try:
        success = quick_validation_test()
        exit_code = 0 if success else 1
        print(f"\nExiting with code: {exit_code}")
        exit(exit_code)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        exit(1)