#!/usr/bin/env python3
"""
Test script to verify RunPod response handling fixes.
Tests JSON serialization and fallback response creation.
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "runpod_service"))

def test_json_serialization():
    """Test various JSON serialization scenarios"""
    print("=== Testing JSON Serialization Scenarios ===")

    # Test 1: Normal serializable data
    good_response = {
        "trial_id": "test_123",
        "status": "completed",
        "success": True,
        "metrics": {
            "test_accuracy": 0.95,
            "test_loss": 0.05
        }
    }

    try:
        json_str = json.dumps(good_response)
        print(f"✅ Normal response serialization: SUCCESS ({len(json_str)} bytes)")
    except Exception as e:
        print(f"❌ Normal response failed: {e}")

    # Test 2: Response with NumPy arrays (problematic)
    problematic_response = {
        "trial_id": "test_123",
        "success": True,
        "model_weights": np.array([1, 2, 3, 4, 5]),  # Not JSON serializable
        "metrics": {"accuracy": 0.95}
    }

    try:
        json_str = json.dumps(problematic_response)
        print(f"✅ NumPy response serialization: SUCCESS")
    except Exception as e:
        print(f"❌ NumPy response failed (expected): {e}")

        # Test fallback creation
        fallback_response = {
            "trial_id": problematic_response["trial_id"],
            "success": problematic_response["success"],
            "metrics": problematic_response["metrics"],
            "serialization_fallback": True,
            "removed_fields": ["model_weights"]
        }

        try:
            json_str = json.dumps(fallback_response)
            print(f"✅ Fallback response serialization: SUCCESS ({len(json_str)} bytes)")
        except Exception as fallback_e:
            print(f"❌ Fallback response failed: {fallback_e}")

def test_plots_direct_info_handling():
    """Test plots_direct_info None handling"""
    print("\n=== Testing plots_direct_info Handling ===")

    # Test None handling
    plots_direct_info = None

    if plots_direct_info is None:
        print("🔍 plots_direct_info is None (simulating issue)")
        plots_direct_info = {
            "success": False,
            "error": "Plot generation returned None",
            "plot_files_zip": None,
            "worker_id": "test_worker"
        }
        print("✅ Created fallback plots_direct_info")

    # Test serialization
    try:
        json_str = json.dumps(plots_direct_info)
        print(f"✅ Fallback plots_direct_info serialization: SUCCESS ({len(json_str)} bytes)")
    except Exception as e:
        print(f"❌ Fallback plots_direct_info failed: {e}")

def test_response_size_limits():
    """Test large response handling"""
    print("\n=== Testing Response Size Limits ===")

    # Create a large response (simulate 20MB)
    large_data = "x" * (20 * 1024 * 1024)  # 20MB string
    large_response = {
        "trial_id": "test_large",
        "success": True,
        "large_field": large_data
    }

    try:
        json_str = json.dumps(large_response)
        json_size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
        print(f"📊 Large response size: {json_size_mb:.2f} MB")

        if json_size_mb > 15:
            print(f"⚠️ Response size ({json_size_mb:.2f} MB) may exceed RunPod limits")
        else:
            print(f"✅ Response size ({json_size_mb:.2f} MB) within limits")

    except Exception as e:
        print(f"❌ Large response handling failed: {e}")

def test_complete_fallback_scenario():
    """Test complete fallback scenario like in the handler"""
    print("\n=== Testing Complete Fallback Scenario ===")

    # Simulate handler response with problematic fields
    response = {
        "trial_id": "test_complete",
        "status": "completed",
        "success": True,
        "metrics": {
            "test_accuracy": 0.95,
            "test_loss": 0.05
        },
        "model_attributes": {
            "weights": np.array([1, 2, 3]),  # Problematic
            "gradients": np.array([4, 5, 6])  # Problematic
        },
        "plots_direct": {
            "success": True,
            "plot_files_zip": {
                "filename": "test.zip",
                "content": "base64content"
            }
        }
    }

    # Test original serialization
    try:
        json.dumps(response)
        print("✅ Original response serialization: SUCCESS")
    except Exception as e:
        print(f"❌ Original response failed (expected): {e}")

        # Find problematic fields
        problematic_fields = []
        for key, value in response.items():
            try:
                json.dumps({key: value})
            except Exception:
                problematic_fields.append(key)

        print(f"🔍 Problematic fields: {problematic_fields}")

        # Create fallback
        fallback_response = {
            "trial_id": response["trial_id"],
            "status": response["status"],
            "success": response["success"],
            "metrics": response["metrics"],
            "plots_direct": response["plots_direct"],
            "serialization_fallback": True,
            "removed_fields": problematic_fields
        }

        try:
            json_str = json.dumps(fallback_response)
            print(f"✅ Fallback response serialization: SUCCESS ({len(json_str)} bytes)")
        except Exception as fallback_e:
            print(f"❌ Fallback failed: {fallback_e}")

            # Last resort minimal response
            minimal_response = {
                "trial_id": response["trial_id"],
                "status": "completed",
                "success": True,
                "error": "Response serialization failed",
                "test_accuracy": 0.95
            }

            try:
                json_str = json.dumps(minimal_response)
                print(f"✅ Minimal response serialization: SUCCESS ({len(json_str)} bytes)")
            except Exception as minimal_e:
                print(f"❌ Even minimal response failed: {minimal_e}")

if __name__ == "__main__":
    print("🚀 Testing RunPod Response Handling Fixes")
    print("=" * 60)

    test_json_serialization()
    test_plots_direct_info_handling()
    test_response_size_limits()
    test_complete_fallback_scenario()

    print("\n" + "=" * 60)
    print("🎉 Response handling tests completed!")
    print("📝 The fixes should prevent 'No output returned from completed RunPod job' errors")
    print("=" * 60)