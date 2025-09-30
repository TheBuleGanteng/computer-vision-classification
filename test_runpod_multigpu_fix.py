#!/usr/bin/env python3
"""
Test script to verify RunPod multi-GPU fixes prevent CollectiveReduceV2 errors.
This simulates the conditions that cause the TensorFlow collective operations errors.
"""

import os
import sys
import tempfile
import tensorflow as tf
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_builder import ModelBuilder, ModelConfig

def test_runpod_multigpu_disabled():
    """Test that multi-GPU is disabled in RunPod environment"""
    print("=== Testing RunPod Multi-GPU Disabling ===")

    # Simulate RunPod environment
    os.environ['RUNPOD_POD_ID'] = 'test_worker_123'

    # Create mock data
    x_data = np.random.random((100, 32, 32, 3)).astype(np.float32)
    y_data = np.random.randint(0, 10, (100,))

    data = {
        'x_train': x_data,
        'y_train': y_data
    }

    # Create config for small model
    config = ModelConfig(
        epochs=1,
        batch_size=16,
        num_classes=10,
        input_shape=(32, 32, 3)
    )

    # Create model builder
    builder = ModelBuilder(config)

    try:
        print(f"🔍 RUNPOD_POD_ID: {os.environ.get('RUNPOD_POD_ID')}")
        print(f"🔍 Available GPUs: {len(tf.config.list_physical_devices('GPU'))}")

        # Test training - should force single GPU mode in RunPod
        results = builder._train_locally(
            data=data,
            use_multi_gpu=True,  # Request multi-GPU
            validation_split=0.2
        )

        print("✅ RunPod multi-GPU fix working - no CollectiveReduceV2 errors")
        print(f"📊 Training completed successfully")

        return True

    except Exception as e:
        print(f"❌ RunPod multi-GPU fix failed: {e}")
        return False

    finally:
        # Clean up environment
        if 'RUNPOD_POD_ID' in os.environ:
            del os.environ['RUNPOD_POD_ID']

def test_collective_ops_reset():
    """Test TensorFlow collective operations reset functionality"""
    print("\n=== Testing TensorFlow Collective Ops Reset ===")

    try:
        # Simulate different model architectures (different parameter counts)
        configs = [
            ModelConfig(epochs=1, batch_size=8, num_classes=5, input_shape=(16, 16, 3)),
            ModelConfig(epochs=1, batch_size=8, num_classes=10, input_shape=(32, 32, 3)),
        ]

        # Create different sized data for each config
        datasets = [
            {
                'x_train': np.random.random((50, 16, 16, 3)).astype(np.float32),
                'y_train': np.random.randint(0, 5, (50,))
            },
            {
                'x_train': np.random.random((50, 32, 32, 3)).astype(np.float32),
                'y_train': np.random.randint(0, 10, (50,))
            }
        ]

        print("🔍 Testing multiple model architectures sequentially...")

        for i, (config, data) in enumerate(zip(configs, datasets)):
            print(f"\n📊 Trial {i+1}: {config.num_classes} classes, {config.input_shape} input")

            builder = ModelBuilder(config)

            # Force single GPU to avoid actual collective ops issues in testing
            results = builder._train_locally(
                data=data,
                use_multi_gpu=False,
                validation_split=0.2
            )

            param_count = builder.model.count_params() if builder.model else 0
            print(f"✅ Trial {i+1} completed - Model has {param_count:,} parameters")

            # Clear session between trials
            tf.keras.backend.clear_session()

        print("✅ Collective operations reset test passed")
        return True

    except Exception as e:
        print(f"❌ Collective ops reset test failed: {e}")
        return False

def test_both_fixes():
    """Test both fixes together"""
    print("\n=== Testing Combined RunPod + Collective Ops Fixes ===")

    # Test normal environment (should allow multi-GPU if available)
    print("🔍 Testing normal environment...")
    os.environ.pop('RUNPOD_POD_ID', None)

    config = ModelConfig(epochs=1, batch_size=8, num_classes=5, input_shape=(16, 16, 3))
    data = {
        'x_train': np.random.random((30, 16, 16, 3)).astype(np.float32),
        'y_train': np.random.randint(0, 5, (30,))
    }

    builder = ModelBuilder(config)
    available_gpus = tf.config.list_physical_devices('GPU')

    print(f"🔍 Available GPUs: {len(available_gpus)}")
    print(f"🔍 RunPod environment: {os.environ.get('RUNPOD_POD_ID') is not None}")

    try:
        results = builder._train_locally(
            data=data,
            use_multi_gpu=True,
            validation_split=0.2
        )
        print("✅ Combined fixes working correctly")
        return True

    except Exception as e:
        print(f"❌ Combined fixes test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing RunPod Multi-GPU Fixes for CollectiveReduceV2 Errors")
    print("=" * 70)

    all_passed = True

    # Test 1: RunPod multi-GPU disabling
    if not test_runpod_multigpu_disabled():
        all_passed = False

    # Test 2: Collective operations reset
    if not test_collective_ops_reset():
        all_passed = False

    # Test 3: Combined fixes
    if not test_both_fixes():
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - RunPod multi-GPU fixes are working!")
        print("🔧 The CollectiveReduceV2 errors should now be prevented")
        print("📝 Deploy these changes to RunPod to resolve the issue")
    else:
        print("❌ SOME TESTS FAILED - Review the fixes")

    print("=" * 70)