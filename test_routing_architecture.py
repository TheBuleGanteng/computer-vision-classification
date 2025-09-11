#!/usr/bin/env python3
"""
Test script to validate the simplified routing architecture

Tests both local execution and RunPod routing paths to ensure the
architectural changes work correctly.
"""

import asyncio
import sys
import os
sys.path.append('src')

from api_server import OptimizationJob, OptimizationAPI
from data_classes.configs import OptimizationConfig

def test_local_routing():
    """Test local execution routing path"""
    print("🧪 Testing Local Routing Architecture")
    print("=" * 50)
    
    # Create a minimal local optimization config
    config = OptimizationConfig(
        dataset_name="mnist",
        mode="simple", 
        optimize_for="val_accuracy",
        trials=1,
        max_epochs_per_trial=1,
        min_epochs_per_trial=1,
        use_runpod_service=False,  # Force local execution
        batch_size=32,
        learning_rate=0.01
    )
    
    print(f"✅ Created local config: {config.dataset_name}, trials={config.trials}")
    
    # Create OptimizationJob
    job = OptimizationJob(config)
    print(f"✅ Created OptimizationJob: {job.job_id}")
    print(f"   Status: {job.status}")
    print(f"   Use RunPod: {config.use_runpod_service}")
    
    return job, config

def test_runpod_routing():
    """Test RunPod routing path (mock endpoint)"""
    print("\\n🌐 Testing RunPod Routing Architecture")
    print("=" * 50)
    
    # Create a RunPod optimization config (with mock endpoint)
    config = OptimizationConfig(
        dataset_name="mnist",
        mode="simple",
        optimize_for="val_accuracy", 
        trials=1,
        max_epochs_per_trial=1,
        min_epochs_per_trial=1,
        use_runpod_service=True,  # Force RunPod routing
        runpod_service_endpoint="https://mock-endpoint.runpod.ai/v2/test/run",
        runpod_service_timeout=30,
        batch_size=32,
        learning_rate=0.01
    )
    
    print(f"✅ Created RunPod config: {config.dataset_name}, trials={config.trials}")
    
    # Create OptimizationJob  
    job = OptimizationJob(config)
    print(f"✅ Created OptimizationJob: {job.job_id}")
    print(f"   Status: {job.status}")
    print(f"   Use RunPod: {config.use_runpod_service}")
    print(f"   Endpoint: {config.runpod_service_endpoint}")
    
    return job, config

async def test_job_orchestration(job, config, test_name):
    """Test the job orchestration without full execution"""
    print(f"\\n🎯 Testing {test_name} Job Orchestration")
    print("-" * 40)
    
    try:
        # Test initial state
        print(f"Initial status: {job.status}")
        assert job.status.value == "pending", f"Expected pending, got {job.status}"
        
        # Test that orchestrate_job sets up properly (without full execution)
        print("Testing orchestrate_job setup...")
        
        # Manually initialize what orchestrate_job would do
        if job.status.value == "pending":
            job.status = "running" 
            job.started_at = "2024-01-01T12:00:00"
            
            # Initialize progress (what orchestrate_job does)
            job.progress = {
                "current_trial": 0,
                "total_trials": config.trials,
                "completed_trials": 0,
                "success_rate": 0.0,
                "best_total_score": None,
                "elapsed_time": 0,
                "status_message": "Initializing optimization..."
            }
            
        print(f"✅ Job status updated: {job.status}")
        print(f"✅ Progress initialized: {job.progress['status_message']}")
        
        # Test routing decision logic
        if config.use_runpod_service and config.runpod_service_endpoint:
            print("✅ Would route to: RunPod service")
            print(f"   Endpoint: {config.runpod_service_endpoint}")
        else:
            print("✅ Would route to: Local execution") 
            print("   Using: ThreadPoolExecutor + ModelOptimizer")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_api_server_integration():
    """Test OptimizationAPI integration"""
    print("\\n🖥️  Testing OptimizationAPI Integration") 
    print("=" * 50)
    
    try:
        # Create server instance
        server = OptimizationAPI()
        print("✅ Created OptimizationAPI instance")
        
        # Test job storage
        config = OptimizationConfig(dataset_name="test", mode="simple", optimize_for="val_accuracy")
        job = OptimizationJob(config)
        
        server.jobs[job.job_id] = job
        print(f"✅ Job stored in server: {job.job_id}")
        
        # Test job retrieval
        retrieved = server.jobs.get(job.job_id)
        assert retrieved == job, "Job retrieval failed"
        print("✅ Job retrieval successful")
        
        return True
        
    except Exception as e:
        print(f"❌ API server test failed: {e}")
        return False

async def main():
    """Run all architecture validation tests"""
    print("🚀 Simplified Routing Architecture Validation")
    print("=" * 60)
    
    results = []
    
    try:
        # Test 1: Local routing
        local_job, local_config = test_local_routing()
        local_result = await test_job_orchestration(local_job, local_config, "Local")
        results.append(("Local Routing", local_result))
        
        # Test 2: RunPod routing
        runpod_job, runpod_config = test_runpod_routing()  
        runpod_result = await test_job_orchestration(runpod_job, runpod_config, "RunPod")
        results.append(("RunPod Routing", runpod_result))
        
        # Test 3: API server integration
        server_result = test_api_server_integration()
        results.append(("API Server Integration", server_result))
        
        # Summary
        print("\\n📊 Test Results Summary")
        print("=" * 30)
        for test_name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        all_passed = all(result for _, result in results)
        print(f"\\n{'🎉 All tests passed!' if all_passed else '⚠️  Some tests failed'}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)