#!/usr/bin/env python3
"""
Test script for dataset_manager.py

Tests both CIFAR-10 loading and the dataset manager interface
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.dataset_manager import DatasetManager
import numpy as np

def test_dataset_manager():
    """Test the dataset manager functionality"""
    print("🧪 Testing Dataset Manager")
    print("=" * 50)
    
    # Initialize manager
    manager = DatasetManager()
    
    # Test 1: Available datasets
    print("\n1️⃣ Testing available datasets...")
    datasets = manager.get_available_datasets()
    print(f"Available datasets: {datasets}")
    assert 'cifar10' in datasets, "CIFAR-10 should be available"
    assert 'gtsrb' in datasets, "GTSRB should be available"
    print("✅ Dataset availability test passed")
    
    # Test 2: Dataset configurations
    print("\n2️⃣ Testing dataset configurations...")
    for dataset_name in datasets:
        config = manager.get_dataset_config(dataset_name)
        print(f"  {dataset_name}:")
        print(f"    - Name: {config.name}")
        print(f"    - Classes: {config.num_classes}")
        print(f"    - Input shape: {config.input_shape}")
        print(f"    - Class names: {len(config.class_names) if config.class_names else 'None'}")
    print("✅ Configuration test passed")
    
    # Test 3: CIFAR-10 loading
    print("\n3️⃣ Testing CIFAR-10 loading...")
    try:
        data = manager.load_dataset('cifar10', test_size=0.2)  # Use 20% for testing
        
        # Verify data structure
        assert 'x_train' in data, "Missing x_train"
        assert 'x_test' in data, "Missing x_test" 
        assert 'y_train' in data, "Missing y_train"
        assert 'y_test' in data, "Missing y_test"
        assert 'config' in data, "Missing config"
        
        # Verify shapes
        x_train, x_test = data['x_train'], data['x_test']
        y_train, y_test = data['y_train'], data['y_test']
        
        print(f"  Training images: {x_train.shape}")
        print(f"  Test images: {x_test.shape}")
        print(f"  Training labels: {y_train.shape}")
        print(f"  Test labels: {y_test.shape}")
        
        # Verify data types and ranges
        assert x_train.dtype == np.float32, f"Expected float32, got {x_train.dtype}"
        assert 0 <= x_train.min() <= x_train.max() <= 1, f"Pixel values not normalized: {x_train.min()}-{x_train.max()}"
        
        # Verify one-hot encoding
        assert y_train.shape[1] == 10, f"Expected 10 classes, got {y_train.shape[1]}"
        assert np.allclose(y_train.sum(axis=1), 1), "Labels not properly one-hot encoded"
        
        print("✅ CIFAR-10 loading test passed")
        
    except Exception as e:
        print(f"❌ CIFAR-10 loading failed: {e}")
        return False
    
    # Test 4: Error handling
    print("\n4️⃣ Testing error handling...")
    try:
        manager.load_dataset('nonexistent_dataset')
        print("❌ Should have raised an error for nonexistent dataset")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    # Test 5: Data consistency
    print("\n5️⃣ Testing data consistency...")
    
    # Load same dataset twice with same random state
    data1 = manager.load_dataset('cifar10', test_size=0.3)
    data2 = manager.load_dataset('cifar10', test_size=0.3)
    
    # Should get same split due to random_state=42
    assert np.array_equal(data1['x_train'], data2['x_train']), "Random state not working"
    print("✅ Data consistency test passed")
    
    print("\n🎉 All tests passed!")
    print("Dataset Manager is working correctly!")
    return True

def test_sample_data_visualization():
    """Test that we can access and understand the loaded data"""
    print("\n📊 Sample Data Analysis")
    print("=" * 50)
    
    manager = DatasetManager()
    data = manager.load_dataset('cifar10', test_size=0.1)  # Small test set for speed
    
    x_train = data['x_train']
    y_train = data['y_train']
    config = data['config']
    
    print(f"Dataset: {config.name}")
    print(f"Total training samples: {len(x_train)}")
    print(f"Image shape: {x_train[0].shape}")
    print(f"Pixel value range: {x_train.min():.3f} to {x_train.max():.3f}")
    
    # Show class distribution
    class_counts = y_train.sum(axis=0)
    print(f"\nClass distribution:")
    for i, (name, count) in enumerate(zip(config.class_names, class_counts)):
        print(f"  {i}: {name:12s} - {int(count):,} samples")
    
    print(f"\nSample label (one-hot): {y_train[0]}")
    print(f"Corresponds to class: {config.class_names[np.argmax(y_train[0])]}")

if __name__ == "__main__":
    print("🚀 Starting Dataset Manager Tests")
    
    success = test_dataset_manager()
    
    if success:
        test_sample_data_visualization()
        print("\n✅ All tests completed successfully!")
        print("Ready to proceed with model builder implementation!")
    else:
        print("\n❌ Tests failed. Please check the issues above.")
        sys.exit(1)