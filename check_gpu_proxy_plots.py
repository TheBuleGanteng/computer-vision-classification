"""
RunPod Plot Diagnostic Script

This script specifically checks for plot generation and synchronization 
issues with your RunPod GPU proxy setup.
"""

import sys
import os
from pathlib import Path
from src.utils.logger import logger

def check_runpod_plot_capabilities():
    """Check if RunPod server can generate and return plots"""
    
    logger.debug(f"running check_runpod_plot_capabilities ... Starting RunPod plot diagnostic")
    
    print("🔍 RunPod Plot Diagnostic")
    print("=" * 50)
    
    # Step 1: Setup RunPod client
    try:
        current_dir = Path.cwd()
        gpu_proxy_dir = current_dir.parent / "gpu-proxy"  # Sibling directory
        
        if not gpu_proxy_dir.exists():
            print("❌ GPU proxy directory not found")
            return
        
        # Add to Python path and change directory
        sys.path.insert(0, str(gpu_proxy_dir))
        original_cwd = os.getcwd()
        os.chdir(gpu_proxy_dir)
        
        try:
            # Based on your logs, the correct path is gpu-proxy/src/runpod/client.py
            # But we need to import it as a module after adding gpu-proxy to path
            import importlib.util
            
            client_path = gpu_proxy_dir / "src" / "runpod" / "client.py"
            spec = importlib.util.spec_from_file_location("runpod_client", client_path)
            client_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(client_module)
            
            RunPodClient = client_module.RunPodClient
            client = RunPodClient()
            print("✅ RunPod client initialized")
            
            # Check health
            health_response = client.health()
            print(f"✅ RunPod server health: {health_response}")
            
        finally:
            os.chdir(original_cwd)
    
    except Exception as e:
        print(f"❌ RunPod client setup failed: {e}")
        return
    
    # Step 2: Check for recent optimization results locally
    print("\n📁 Local Results Check:")
    opt_results_dir = current_dir / "optimization_results"
    
    if opt_results_dir.exists():
        run_dirs = sorted([d for d in opt_results_dir.iterdir() if d.is_dir()], 
                         key=lambda x: x.stat().st_mtime, reverse=True)
        
        if run_dirs:
            latest_run = run_dirs[0]
            print(f"📂 Latest run: {latest_run.name}")
            
            # Check plot directories
            plots_dir = latest_run / "plots"
            if plots_dir.exists():
                print(f"📁 Plots directory exists: {plots_dir}")
                
                # Check trial directories
                trial_dirs = [d for d in plots_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")]
                print(f"📊 Trial directories: {len(trial_dirs)}")
                
                total_plots = 0
                for trial_dir in trial_dirs:
                    plot_files = list(trial_dir.rglob("*.png"))
                    total_plots += len(plot_files)
                    print(f"   {trial_dir.name}: {len(plot_files)} plots")
                
                if total_plots == 0:
                    print("⚠️  NO PLOTS FOUND - This confirms the synchronization issue")
                else:
                    print(f"✅ Found {total_plots} total plots")
            else:
                print("❌ No plots directory found")
        else:
            print("❌ No optimization runs found")
    else:
        print("❌ No optimization_results directory")
    
    # Step 3: Test plot generation capability
    print(f"\n🧪 Testing RunPod Plot Generation:")
    
    try:
        # Create a simple test request to check if the server can generate plots
        test_request = {
            "action": "test_plot_generation",
            "dataset": "mnist",
            "epochs": 1,
            "return_plots": True
        }
        
        print("📤 Sending test plot generation request...")
        
        # Use the RunPod client to send the test request
        os.chdir(gpu_proxy_dir)
        try:
            # This might not work if the server doesn't have this endpoint
            # but it's worth testing
            response = client.run(test_request, timeout=60)
            
            if response and "plots" in response:
                print(f"✅ RunPod server CAN generate plots: {len(response['plots'])} plots returned")
                print("📋 Plot files returned:")
                for plot_name in response["plots"].keys():
                    print(f"   📊 {plot_name}")
            else:
                print("⚠️  RunPod server response doesn't contain plot data")
                print(f"📋 Response keys: {list(response.keys()) if response else 'No response'}")
                
        except Exception as test_error:
            print(f"⚠️  Test plot generation failed: {test_error}")
            print("   This might be expected if test endpoint doesn't exist")
        
        finally:
            os.chdir(original_cwd)
    
    except Exception as e:
        print(f"❌ Plot generation test failed: {e}")
    
    # Step 4: Check the logs for clues
    print(f"\n📜 Log Analysis:")
    log_file = current_dir / "logs" / "non-cron.log"
    
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Look for plot-related log entries
            plot_keywords = [
                "plot", "matplotlib", "figure", "savefig", 
                "analysis", "confusion_matrix", "training_history",
                "gpu_proxy", "runpod", "sync"
            ]
            
            print("🔍 Searching logs for plot-related entries...")
            lines = log_content.split('\n')
            relevant_lines = []
            
            for line in lines:
                if any(keyword.lower() in line.lower() for keyword in plot_keywords):
                    relevant_lines.append(line)
            
            if relevant_lines:
                print(f"📝 Found {len(relevant_lines)} relevant log entries")
                print("📋 Recent plot-related log entries:")
                # Show last 10 relevant entries
                for line in relevant_lines[-10:]:
                    print(f"   {line}")
            else:
                print("⚠️  No plot-related log entries found")
        
        except Exception as log_error:
            print(f"❌ Could not read log file: {log_error}")
    else:
        print("❌ Log file not found")
    
    # Step 5: Provide specific recommendations
    print(f"\n💡 DIAGNOSIS & RECOMMENDATIONS:")
    print("=" * 50)
    
    print("Based on your logs, the issue is clear:")
    print("1. ✅ GPU proxy (RunPod) connection is working")
    print("2. ✅ Training is happening on RunPod server")
    print("3. ✅ Local directories are created correctly")
    print("4. ❌ Plots are NOT being synchronized back to local machine")
    print()
    print("🔧 ROOT CAUSE: Plot synchronization issue")
    print("   The RunPod server is likely generating plots, but they're not")
    print("   being returned/downloaded to your local machine.")
    print()
    print("📋 NEXT STEPS:")
    print("1. Check if RunPod server has plot return functionality enabled")
    print("2. Verify plot data is included in RunPod response payload") 
    print("3. Test with local execution to confirm plots work locally:")
    print("   python src/optimizer.py dataset=mnist trials=1 use_gpu_proxy=false")
    print("4. Check RunPod server logs for plot generation success")

if __name__ == "__main__":
    check_runpod_plot_capabilities()