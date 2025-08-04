"""
Simple GPU Proxy Plot Check

A simplified script to quickly check if plots exist on the GPU proxy server.
Run this from your project root directory.
"""

import sys
import os
from pathlib import Path

def check_gpu_proxy_plots():
    """Simple check for GPU proxy plots"""
    
    print("🔍 Checking GPU Proxy Plot Status...")
    print("=" * 50)
    
    # Step 1: Check if GPU proxy directory exists
    current_dir = Path.cwd()
    gpu_proxy_dir = current_dir.parent / "gpu-proxy"  # Sibling directory
    
    print(f"1. GPU Proxy Directory: {gpu_proxy_dir}")
    if gpu_proxy_dir.exists():
        print("   ✅ GPU proxy directory found")
    else:
        print("   ❌ GPU proxy directory not found")
        return
    
    # Step 2: Check local optimization results
    opt_results_dir = current_dir / "optimization_results"
    print(f"\n2. Local Optimization Results: {opt_results_dir}")
    
    if opt_results_dir.exists():
        print("   ✅ Optimization results directory exists")
        
        # Find all run directories
        run_dirs = [d for d in opt_results_dir.iterdir() if d.is_dir()]
        print(f"   📁 Found {len(run_dirs)} run directories:")
        
        for run_dir in sorted(run_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
            plots_dir = run_dir / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.rglob("*.png"))
                trial_dirs = [d for d in plots_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")]
                print(f"      📂 {run_dir.name}:")
                print(f"         - Plot files: {len(plot_files)}")
                print(f"         - Trial directories: {len(trial_dirs)}")
                
                # Check trial directories for plots
                for trial_dir in trial_dirs[:2]:  # Check first 2 trials
                    trial_plots = list(trial_dir.rglob("*.png"))
                    print(f"         - {trial_dir.name}: {len(trial_plots)} plots")
            else:
                print(f"      📂 {run_dir.name}: No plots directory")
    else:
        print("   ❌ No optimization results directory")
    
    # Step 3: Try to connect to GPU proxy
    print(f"\n3. GPU Proxy Connection Test:")
    try:
        # Add GPU proxy to Python path
        sys.path.insert(0, str(gpu_proxy_dir))
        
        # Based on your logs, we need to dynamically import the RunPodClient
        import importlib.util
        
        client_path = gpu_proxy_dir / "src" / "runpod" / "client.py"
        if not client_path.exists():
            print(f"   ❌ RunPod client not found at: {client_path}")
            return
            
        spec = importlib.util.spec_from_file_location("runpod_client", client_path)
        client_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client_module)
        
        RunPodClient = client_module.RunPodClient
        client = RunPodClient()
        print("   ✅ RunPodClient imported successfully")
        
        if hasattr(client, 'endpoint'):
            print(f"   🌐 Endpoint: {getattr(client, 'endpoint', 'Not set')}")
        
        # Try a simple health check
        try:
            # Based on your logs, the client has a health() method
            test_response = client.health()
            if test_response:
                print("   ✅ RunPod server responding")
                print(f"   📋 Health status: {test_response}")
            else:
                print("   ⚠️  RunPod server not responding or no response")
        except Exception as ping_error:
            print(f"   ⚠️  RunPod health check failed: {ping_error}")
        
    except ImportError as e:
        print(f"   ❌ Cannot import RunPod client: {e}")
    except Exception as e:
        print(f"   ❌ GPU proxy connection error: {e}")
    
    # Step 4: Check environment file
    print(f"\n4. Environment Configuration:")
    env_file = gpu_proxy_dir / ".env"
    if env_file.exists():
        print("   ✅ .env file found")
        try:
            with open(env_file, 'r') as f:
                env_content = f.read()
                # Don't print sensitive info, just check if it has content
                lines = [line for line in env_content.split('\n') if line.strip() and not line.startswith('#')]
                print(f"   📝 Environment variables: {len(lines)} configured")
        except Exception as e:
            print(f"   ⚠️  Could not read .env file: {e}")
    else:
        print("   ❌ .env file not found")
    
    print("\n" + "=" * 50)
    print("💡 RECOMMENDATIONS:")
    print("=" * 50)
    
    # Provide recommendations based on findings
    if not opt_results_dir.exists():
        print("1. No optimization results found locally")
        print("   → Run an optimization first to generate directories")
    else:
        run_dirs = [d for d in opt_results_dir.iterdir() if d.is_dir()]
        if run_dirs:
            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
            plots_dir = latest_run / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.rglob("*.png"))
                if len(plot_files) == 0:
                    print("1. Plot directories exist but no plot files found")
                    print("   → This confirms plots are not being synchronized from GPU proxy")
                    print("   → Need to fix GPU proxy plot return functionality")
                else:
                    print("1. Some plots found locally")
                    print("   → Check if all expected plots are present")
            else:
                print("1. Latest run has no plots directory")
                print("   → Plot creation/synchronization is failing")
    
    print("\n2. Next steps to investigate:")
    print("   a) Check if GPU proxy server has plots in its filesystem")
    print("   b) Verify GPU proxy is configured to return plot data")
    print("   c) Check for any errors in GPU proxy communication")
    print("   d) Test with local execution (use_gpu_proxy=false) to confirm plot generation works")

if __name__ == "__main__":
    check_gpu_proxy_plots()