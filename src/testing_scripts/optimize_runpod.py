#!/usr/bin/env python3
"""
Complete Optimization API Client & Monitor
Provides access to all endpoints with monitoring, management, and analysis features
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import requests
import shutil
from tabulate import tabulate
import time
from typing import Dict, Any, Optional, List
import zipfile



class OptimizationAPIClient:
    def __init__(self, runpod_url: str, poll_interval: int = 10):
        self.base_url = f"https://{runpod_url}"
        self.poll_interval = poll_interval
        self.iteration = 0
        self.job_id = None
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # ============================================================================
    # BASIC API ENDPOINTS
    # ============================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_datasets(self) -> Dict[str, Any]:
        """Get list of available datasets"""
        try:
            response = requests.get(f"{self.base_url}/datasets", timeout=10)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_modes(self) -> Dict[str, Any]:
        """Get list of available optimization modes"""
        try:
            response = requests.get(f"{self.base_url}/modes", timeout=10)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_objectives(self) -> Dict[str, Any]:
        """Get list of available optimization objectives"""
        try:
            response = requests.get(f"{self.base_url}/objectives", timeout=10)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ============================================================================
    # JOB MANAGEMENT ENDPOINTS
    # ============================================================================
    
    def start_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new optimization job"""
        try:
            response = requests.post(
                f"{self.base_url}/optimize",
                json=config,
                timeout=30
            )
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_all_jobs(self) -> Dict[str, Any]:
        """Get list of all optimization jobs"""
        try:
            response = requests.get(f"{self.base_url}/jobs", timeout=10)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get specific job status"""
        try:
            response = requests.get(f"{self.base_url}/jobs/{job_id}", timeout=10)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_current_trial(self, job_id: str) -> Dict[str, Any]:
        """Get current trial details"""
        try:
            response = requests.get(f"{self.base_url}/jobs/{job_id}/current-trial", timeout=10)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_best_trial(self, job_id: str) -> Dict[str, Any]:
        """Get best trial so far"""
        try:
            response = requests.get(f"{self.base_url}/jobs/{job_id}/best-trial", timeout=10)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_trial_history(self, job_id: str) -> Dict[str, Any]:
        """Get trial history"""
        try:
            response = requests.get(f"{self.base_url}/jobs/{job_id}/trials", timeout=10)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_trends(self, job_id: str) -> Dict[str, Any]:
        """Get architecture and health trends"""
        try:
            response = requests.get(f"{self.base_url}/jobs/{job_id}/trends", timeout=10)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_job_files(self, job_id: str) -> Dict[str, Any]:
        """List all files in a job's results directory"""
        try:
            response = requests.get(f"{self.base_url}/jobs/{job_id}/files", timeout=30)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def download_job_file(self, job_id: str, file_path: str, local_path: Path) -> bool:
        """Download a specific file from job results"""
        try:
            response = requests.get(f"{self.base_url}/jobs/{job_id}/files/{file_path}", timeout=120)
            if response.status_code == 200:
                # Ensure parent directory exists
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file content
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"📥 Downloaded: {file_path}")
                return True
            else:
                print(f"❌ Failed to download {file_path}: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error downloading {file_path}: {e}")
            return False

    def download_job_results_zip(self, job_id: str, extract_to: Path) -> bool:
        """Download entire job results as ZIP and extract to proper directory structure"""
        try:
            print(f"📦 Downloading results archive for job {job_id}...")
            
            response = requests.get(f"{self.base_url}/jobs/{job_id}/download", timeout=300)
            if response.status_code != 200:
                print(f"❌ Failed to download results: {response.status_code}")
                return False
            
            # Save ZIP to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_zip_path = temp_file.name
            
            try:
                # Create temporary extraction directory
                with tempfile.TemporaryDirectory() as temp_extract_dir:
                    # Extract ZIP to temporary directory first
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract_dir)
                    
                    # Find the actual results directory in the extracted content
                    temp_path = Path(temp_extract_dir)
                    extracted_dirs = [d for d in temp_path.iterdir() if d.is_dir()]
                    
                    if not extracted_dirs:
                        print("❌ No directories found in extracted ZIP")
                        return False
                    
                    # Usually there's one main directory (the results directory)
                    source_dir = extracted_dirs[0]
                    
                    # Ensure target directory exists
                    extract_to.mkdir(parents=True, exist_ok=True)
                    
                    # Move the extracted directory to the target location
                    target_dir = extract_to / source_dir.name
                    
                    # If target already exists, remove it first
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    
                    # Move the directory
                    shutil.move(str(source_dir), str(target_dir))
                    
                    print(f"✅ Results extracted to: {target_dir}")
                    
                    # Show summary of extracted content
                    files = list(target_dir.rglob("*"))
                    file_count = len([f for f in files if f.is_file()])
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    
                    print(f"📊 Extracted {file_count} files ({size_mb:.1f} MB)")
                    
                    return True
                
            finally:
                # Clean up temporary ZIP file
                Path(temp_zip_path).unlink(missing_ok=True)
                
        except Exception as e:
            print(f"❌ Error downloading results: {e}")
            return False

    def download_all_job_files(self, job_id: str, local_dir: Path) -> bool:
        """Download all files from a job individually"""
        try:
            # Get file listing
            files_result = self.list_job_files(job_id)
            if not files_result["success"]:
                print(f"❌ Failed to get file listing: {files_result.get('error', 'Unknown error')}")
                return False
            
            files_data = files_result["data"]
            files = files_data.get("files", [])
            
            if not files:
                print("ℹ️  No files to download")
                return True
            
            print(f"📦 Downloading {len(files)} files...")
            print(f"📁 Total size: {files_data.get('total_size_mb', 0):.1f} MB")
            
            # Download each file
            success_count = 0
            for file_info in files:
                file_path = file_info["path"]
                local_file_path = local_dir / file_path
                
                if self.download_job_file(job_id, file_path, local_file_path):
                    success_count += 1
            
            print(f"✅ Downloaded {success_count}/{len(files)} files successfully")
            return success_count == len(files)
            
        except Exception as e:
            print(f"❌ Error downloading files: {e}")
            return False

    def auto_download_results(self, job_id: str, use_zip: bool = True) -> bool:
        """
        Automatically download job results to local optimization_results directory
        
        Args:
            job_id: Job identifier
            use_zip: If True, download as ZIP (faster). If False, download files individually.
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            print(f"\n🔄 Starting automatic download for job {job_id}...")
            
            # Determine local results directory (project root)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent  # optimize_runpod.py is in project root
            local_results_dir = project_root / "optimization_results"
            
            print(f"📁 Target directory: {local_results_dir}")
            
            if use_zip:
                # Method 1: Download as ZIP (faster for many files)
                success = self.download_job_results_zip(job_id, local_results_dir)
                
                if success:
                    # List what was downloaded
                    extracted_dirs = [d for d in local_results_dir.iterdir() if d.is_dir()]
                    if extracted_dirs:
                        latest_dir = max(extracted_dirs, key=lambda d: d.stat().st_mtime)
                        print(f"🎉 Results successfully downloaded to: {latest_dir}")
                        
                        # Show key files
                        key_files = []
                        for pattern in ["*.json", "*.yaml", "*.html", "*.keras", "*.md"]:
                            key_files.extend(latest_dir.glob(pattern))
                        
                        if key_files:
                            print("🔑 Key files:")
                            for file_path in sorted(key_files)[:10]:  # Show first 10
                                print(f"   • {file_path.name}")
                        
                        # Show plots directories
                        plots_dir = latest_dir / "plots"
                        if plots_dir.exists():
                            trial_dirs = [d for d in plots_dir.iterdir() if d.is_dir()]
                            if trial_dirs:
                                print(f"📊 Plot directories: {len(trial_dirs)} trial folders")
                
            else:
                # Method 2: Download files individually (more granular control)
                # Create a specific directory for this job
                job_status = self.get_job_status(job_id)
                if job_status["success"]:
                    job_data = job_status["data"]
                    dataset_name = job_data.get("dataset_name", "unknown")
                    mode = job_data.get("mode", "unknown")
                    
                    # Create timestamped directory name
                    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                    results_folder_name = f"{timestamp}_{dataset_name}_{mode}-{job_data.get('objective', 'val_accuracy')}"
                    target_dir = local_results_dir / results_folder_name
                    
                    success = self.download_all_job_files(job_id, target_dir)
                    
                    if success:
                        print(f"🎉 Results successfully downloaded to: {target_dir}")
                else:
                    print("❌ Failed to get job status for directory naming")
                    return False
            
            return success
            
        except Exception as e:
            print(f"❌ Error in auto-download: {e}")
            return False

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """Stop a specific running job"""
        try:
            response = requests.post(f"{self.base_url}/jobs/{job_id}/stop", timeout=30)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_all_jobs(self) -> Dict[str, Any]:
        """Stop all running jobs"""
        try:
            response = requests.post(f"{self.base_url}/jobs/stop", timeout=30)
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "success": response.status_code == 200
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def format_number(self, num: int) -> str:
        """Format large numbers with units"""
        if num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        else:
            return str(num)
    
    def format_duration(self, seconds: Optional[float]) -> str:
        """Format duration in human readable format"""
        if not seconds:
            return "N/A"
        
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def draw_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """Draw a progress bar"""
        if total == 0:
            return "█" * width
        
        filled = int(width * current / total)
        empty = width - filled
        percentage = (current / total) * 100
        
        bar = "█" * filled + "░" * empty
        return f"[{bar}] {percentage:.1f}% ({current}/{total})"
    
    # ============================================================================
    # INTERACTIVE COMMANDS
    # ============================================================================
    
    def show_system_info(self):
        """Display system information and available resources"""
        print("🌐 SYSTEM INFORMATION")
        print("=" * 60)
        
        # Health check
        health = self.health_check()
        if health["success"]:
            print("✅ API Health: OK")
        else:
            print(f"❌ API Health: {health.get('error', 'Failed')}")
        
        # Available datasets
        datasets = self.get_datasets()
        if datasets["success"]:
            dataset_list = datasets["data"].get("datasets", [])
            print(f"📊 Available Datasets ({len(dataset_list)}): {', '.join(dataset_list)}")
        else:
            print("❌ Failed to get datasets")
        
        # Available modes
        modes = self.get_modes()
        if modes["success"]:
            mode_list = modes["data"].get("modes", [])
            print(f"🎯 Available Modes: {', '.join(mode_list)}")
        else:
            print("❌ Failed to get modes")
        
        # Available objectives
        objectives = self.get_objectives()
        if objectives["success"]:
            obj_list = objectives["data"].get("objectives", [])
            print(f"🏆 Available Objectives: {', '.join(obj_list)}")
        else:
            print("❌ Failed to get objectives")
        
        print(f"🔗 API Base URL: {self.base_url}")
        print()
    
    def show_all_jobs(self):
        """Display all optimization jobs"""
        print("📋 ALL OPTIMIZATION JOBS")
        print("=" * 60)
        
        jobs = self.get_all_jobs()
        if not jobs["success"]:
            print(f"❌ Failed to get jobs: {jobs.get('error', 'Unknown error')}")
            return
        
        jobs_data = jobs["data"].get("jobs", [])
        if not jobs_data:
            print("No jobs found")
            return
        
        # Prepare table data
        table_data = []
        for job in jobs_data:
            job_id = job.get("job_id", "N/A")[:8] + "..."  # Truncate for display
            dataset = job.get("dataset_name", "N/A")
            mode = job.get("mode", "N/A")
            status = job.get("status", "N/A")
            trials = f"{job.get('current_trial', 0)}/{job.get('total_trials', 0)}"
            best_value = job.get("best_value", "N/A")
            if isinstance(best_value, float):
                best_value = f"{best_value:.3f}"
            
            table_data.append([job_id, dataset, mode, status, trials, best_value])
        
        headers = ["Job ID", "Dataset", "Mode", "Status", "Trials", "Best Value"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
    
    def show_job_details(self, job_id: str):
        """Display detailed information about a specific job"""
        print(f"🔍 JOB DETAILS: {job_id}")
        print("=" * 60)
        
        # Get main job status
        job_status = self.get_job_status(job_id)
        if not job_status["success"]:
            print(f"❌ Failed to get job status: {job_status.get('error', 'Unknown error')}")
            return
        
        job_data = job_status["data"]
        
        # Get progress data (this contains the correct trial info)
        progress = job_data.get('progress', {})
        
        # Basic info from various sources
        print(f"📊 Dataset: {job_data.get('dataset_name', 'N/A')}")
        print(f"🎯 Mode: {job_data.get('mode', 'N/A')}")  
        print(f"🏆 Objective: {job_data.get('objective', 'N/A')}")
        print(f"📈 Status: {job_data.get('status', 'N/A')}")
        
        # Use progress data for accurate trial counts
        current_trial = progress.get('current_trial', 0)
        total_trials = progress.get('total_trials', 0)
        completed_trials = progress.get('completed_trials', 0)
        
        print(f"🔄 Current Trial: {current_trial}/{total_trials}")
        print(f"✅ Completed Trials: {completed_trials}")
        
        # Show status message
        status_message = progress.get('status_message', 'No status message')
        print(f"💬 Status: {status_message}")
        
        # Best value if available
        best_value = progress.get('best_value') or job_data.get('best_value')
        if best_value:
            print(f"🏅 Best Value: {best_value:.4f}")
        
        # Timing info
        created_at = job_data.get('created_at')
        started_at = job_data.get('started_at')
        elapsed_time = progress.get('elapsed_time', 0)
        
        if created_at:
            print(f"📅 Created: {created_at}")
        if started_at:
            print(f"🚀 Started: {started_at}")
        if elapsed_time > 0:
            print(f"⏱️  Elapsed: {self.format_duration(elapsed_time)}")
        
        # Current trial info
        current_trial_resp = self.get_current_trial(job_id)
        if current_trial_resp["success"] and current_trial_resp["data"]:
            trial_data = current_trial_resp["data"]
            if "current_trial" in trial_data:
                trial = trial_data["current_trial"]
                print(f"\n🔧 Current Trial: {trial.get('trial_id', 'N/A')}")
                print(f"   Status: {trial.get('status', 'N/A')}")
                
                if "model_size" in trial:
                    model_size = trial["model_size"]
                    params = model_size.get("total_params", 0)
                    memory = model_size.get("memory_mb", 0)
                    print(f"   Model: {self.format_number(params)} params, {memory:.1f} MB")
        
        # Best trial info
        best_trial = self.get_best_trial(job_id)
        if best_trial["success"] and best_trial["data"]:
            print(f"\n🏆 Best Trial Available")
        
        print()
    
    def show_trial_analysis(self, job_id: str):
        """Show detailed trial analysis"""
        print(f"📊 TRIAL ANALYSIS: {job_id}")
        print("=" * 60)
        
        # Get trial history
        history = self.get_trial_history(job_id)
        if not history["success"]:
            print(f"❌ Failed to get trial history: {history.get('error', 'Unknown error')}")
            return
        
        trials = history["data"].get("trials", [])
        if not trials:
            print("No trials found")
            return
        
        # Prepare trial table
        table_data = []
        for trial in trials:
            trial_num = trial.get("trial_number", "N/A")
            status = trial.get("status", "N/A")
            duration = self.format_duration(trial.get("duration_seconds"))
            
            # Performance data
            perf = trial.get("performance", {})
            val_acc = perf.get("final_val_accuracy", "N/A")
            if isinstance(val_acc, float):
                val_acc = f"{val_acc:.3f}"
            
            # Model size
            model_size = trial.get("model_size", {})
            params = model_size.get("total_params", 0)
            params_fmt = self.format_number(params) if params else "N/A"
            
            table_data.append([trial_num, status, duration, val_acc, params_fmt])
        
        headers = ["Trial", "Status", "Duration", "Val Acc", "Params"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Summary statistics
        completed_trials = [t for t in trials if t.get("status") == "completed"]
        if completed_trials:
            print(f"\n📈 Summary:")
            print(f"   Completed Trials: {len(completed_trials)}/{len(trials)}")
            
            # Best performance
            best_trial = max(completed_trials, key=lambda x: x.get("performance", {}).get("final_val_accuracy", 0))
            best_acc = best_trial.get("performance", {}).get("final_val_accuracy", 0)
            print(f"   Best Accuracy: {best_acc:.3f} (Trial {best_trial.get('trial_number', '?')})")
            
            # Average duration
            durations = [t.get("duration_seconds", 0) for t in completed_trials if t.get("duration_seconds")]
            if durations:
                avg_duration = sum(durations) / len(durations)
                print(f"   Average Duration: {self.format_duration(avg_duration)}")
        
        print()
    
    def interactive_mode(self):
        """Run interactive command interface"""
        print("🚀 INTERACTIVE API CLIENT")
        print("=" * 60)
        print("Available commands:")
        print("  info              - Show system information")
        print("  jobs              - List all jobs")
        print("  job <job_id>      - Show job details")
        print("  trials <job_id>   - Show trial analysis")
        print("  monitor <job_id>  - Monitor job progress")
        print("  download <job_id> - Download job results")
        print("  files <job_id>    - List job files")  
        print("  stop <job_id>     - Stop specific job")
        print("  stop-all          - Stop all running jobs")
        print("  health            - Check API health")
        print("  datasets          - List datasets")
        print("  modes             - List optimization modes")
        print("  objectives        - List optimization objectives")
        print("  trends <job_id>   - Show performance trends")
        print("  start             - Start new optimization")
        print("  help              - Show this help")
        print("  quit              - Exit")
        print()
        
        while True:
            try:
                command = input("🔧 Enter command: ").strip().split()
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif cmd == 'download' and len(command) > 1:
                    job_id = command[1]
                    print(f"📦 Downloading results for job {job_id}...")
                    
                    # First test if we can get job status
                    status_result = self.get_job_status(job_id)
                    print(f"Debug - Job status result: {status_result}")
                    
                    success = self.auto_download_results(job_id)
                    if not success:
                        print("❌ Download failed")
                elif cmd == 'files' and len(command) > 1:
                    job_id = command[1]
                    result = self.list_job_files(job_id)
                    if result["success"]:
                        files_data = result["data"]
                        print(f"📁 Files for job {job_id}:")
                        print(f"   Total files: {files_data.get('total_files', 0)}")
                        print(f"   Total size: {files_data.get('total_size_mb', 0):.1f} MB")
                        for file_info in files_data.get("files", [])[:10]:  # Show first 10
                            print(f"   • {file_info['path']} ({file_info['size_mb']:.1f} MB)")
                    else:
                        print(f"❌ Failed to list files: {result.get('error', 'Unknown error')}")
                elif cmd == 'stop' and len(command) > 1:
                    job_id = command[1]
                    result = self.stop_job(job_id)
                    if result["success"]:
                        print(f"✅ Job {job_id} stopped successfully")
                    else:
                        print(f"❌ Failed to stop job: {result.get('error', 'Unknown error')}")
                elif cmd == 'stop-all':
                    result = self.stop_all_jobs()
                    print(f"Debug - Full result: {result}")  # Add this line
                    print(f"Debug - Status code: {result.get('status_code')}")  # Add this line
                    print(f"Debug - Response data: {result.get('data')}")  # Add this line
                    if result["success"]:
                        data = result["data"]
                        print(f"✅ {data.get('message', 'Operation completed')}")
                        if data.get("stopped_jobs"):
                            print(f"   Stopped jobs: {len(data['stopped_jobs'])}")
                    else:
                        print(f"❌ Failed to stop jobs: {result.get('error', 'Unknown error')}")
                elif cmd == 'help':
                    self.interactive_mode()  # Show help again
                    break
                elif cmd == 'info':
                    self.show_system_info()
                elif cmd == 'jobs':
                    self.show_all_jobs()
                elif cmd == 'job' and len(command) > 1:
                    self.show_job_details(command[1])
                elif cmd == 'trials' and len(command) > 1:
                    self.show_trial_analysis(command[1])
                elif cmd == 'monitor' and len(command) > 1:
                    self.monitor_job(command[1])
                    break  # Exit interactive mode after monitoring
                elif cmd == 'health':
                    result = self.health_check()
                    print(json.dumps(result, indent=2))
                elif cmd == 'datasets':
                    result = self.get_datasets()
                    if result["success"]:
                        datasets = result["data"].get("datasets", [])
                        print(f"Available datasets: {', '.join(datasets)}")
                    else:
                        print(f"Error: {result.get('error', 'Failed to get datasets')}")
                elif cmd == 'modes':
                    result = self.get_modes()
                    if result["success"]:
                        modes = result["data"].get("modes", [])
                        print(f"Available modes: {', '.join(modes)}")
                    else:
                        print(f"Error: {result.get('error', 'Failed to get modes')}")
                elif cmd == 'objectives':
                    result = self.get_objectives()
                    if result["success"]:
                        objectives = result["data"].get("objectives", [])
                        print(f"Available objectives: {', '.join(objectives)}")
                    else:
                        print(f"Error: {result.get('error', 'Failed to get objectives')}")
                elif cmd == 'trends' and len(command) > 1:
                    result = self.get_trends(command[1])
                    print(json.dumps(result, indent=2))
                elif cmd == 'start':
                    self.interactive_start_optimization()
                else:
                    print(f"❌ Unknown command: {' '.join(command)}")
                    print("Type 'help' for available commands")
                
                print()  # Add spacing
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def interactive_start_optimization(self):
        """Interactive optimization starter"""
        print("🚀 START NEW OPTIMIZATION")
        print("=" * 40)
        
        # Get available options
        datasets_result = self.get_datasets()
        modes_result = self.get_modes()
        objectives_result = self.get_objectives()
        
        try:
            # Dataset selection
            if datasets_result["success"]:
                datasets = datasets_result["data"].get("datasets", [])
                print(f"Available datasets: {', '.join(datasets)}")
                dataset = input("Dataset [mnist]: ").strip() or "mnist"
            else:
                dataset = input("Dataset [mnist]: ").strip() or "mnist"
            
            # Mode selection
            if modes_result["success"]:
                modes = modes_result["data"].get("modes", [])
                print(f"Available modes: {', '.join(modes)}")
                mode = input("Mode [simple]: ").strip() or "simple"
            else:
                mode = input("Mode [simple]: ").strip() or "simple"
            
            # Objective selection
            if objectives_result["success"]:
                objectives = objectives_result["data"].get("objectives", [])
                print(f"Available objectives: {', '.join(objectives)}")
                objective = input("Objective [val_accuracy]: ").strip() or "val_accuracy"
            else:
                objective = input("Objective [val_accuracy]: ").strip() or "val_accuracy"
            
            # Configuration
            trials = int(input("Trials [10]: ").strip() or "10")
            max_epochs = int(input("Max epochs per trial [20]: ").strip() or "20")
            
            # Build config
            config = {
                "dataset_name": dataset,
                "mode": mode,
                "optimize_for": objective,
                "trials": trials,
                "config_overrides": {
                    "max_epochs_per_trial": max_epochs
                }
            }
            
            # Add health weight for health mode
            if mode == "health":
                health_weight = float(input("Health weight [0.3]: ").strip() or "0.3")
                config["config_overrides"]["health_weight"] = health_weight
            
            print(f"\n🔧 Configuration:")
            print(json.dumps(config, indent=2))
            confirm = input("\nStart optimization? [y/N]: ").strip().lower()
            
            if confirm in ['y', 'yes']:
                result = self.start_optimization(config)
                if result["success"]:
                    job_id = result["data"].get("job_id")
                    print(f"✅ Optimization started! Job ID: {job_id}")
                    
                    monitor = input("Monitor progress? [Y/n]: ").strip().lower()
                    if monitor not in ['n', 'no']:
                        self.monitor_job(job_id)
                else:
                    print(f"❌ Failed to start optimization: {result.get('error', 'Unknown error')}")
            else:
                print("❌ Optimization cancelled")
                
        except KeyboardInterrupt:
            print("\n❌ Cancelled")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def monitor_job(self, job_id: str):
        """Monitor a specific job with live updates - FIXED VERSION"""
        print(f"🔍 Monitoring job: {job_id}")
        self.job_id = job_id
        
        try:
            while True:
                self.iteration += 1
                
                if self.iteration > 1:
                    self.clear_screen()
                
                self.display_monitoring_header(job_id)
                
                # Get data with error handling
                job_data = self.get_job_status(job_id)
                trial_data = self.get_current_trial(job_id)
                trial_history = self.get_trial_history(job_id)
                
                if not job_data["success"]:
                    print("❌ Failed to get job status - job may not exist")
                    break
                
                # FIX: Add error handling for the display function
                try:
                    self.display_monitoring_status(job_data["data"], trial_data, trial_history.get("data"))
                except Exception as e:
                    print(f"⚠️  Display error (continuing monitoring): {str(e)}")
                    
                    # Still show basic status even if display fails
                    status = job_data["data"].get('status', 'unknown')
                    print(f"📈 Job Status: {status}")
                
                # Check if completed or failed
                status = job_data["data"].get('status', 'unknown')
                if status in ['completed', 'failed']:
                    print("\n🏁 Monitoring complete!")
                    self.save_results_summary(job_data["data"], trial_history.get("data"))
                    break
                
                # Wait for next iteration
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Monitoring stopped by user")
    
    def display_monitoring_header(self, job_id: str):
        """Display monitoring header"""
        print("=" * 70)
        print("    🧠 HYPERPARAMETER OPTIMIZATION MONITOR")
        print("=" * 70)
        print(f"Job ID: {job_id}")
        print(f"Update Interval: {self.poll_interval}s")
        print(f"RunPod URL: {self.base_url}")
        print()
    
    def display_monitoring_status(self, job_data: Dict[str, Any], trial_data: Optional[Dict[str, Any]], 
                                trial_history: Optional[Dict[str, Any]]):
        """Display monitoring status"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status = job_data.get('status', 'unknown')
        
        print(f"🕒 Last Update: {timestamp}")
        print("-" * 60)
        
        if status == "completed":
            self.display_completed_status(job_data, trial_history)
        elif status == "failed":
            self.display_failed_status(job_data)
        elif status == "running":
            self.display_running_status(job_data, trial_data, trial_history)
        else:
            print(f"Status: {status}")
        
        print(f"\n⏱️  Next update in {self.poll_interval}s... (Ctrl+C to stop)")
    
    def display_completed_status(self, job_data: Dict[str, Any], trial_history: Optional[Dict[str, Any]]):
        """FIXED: Display completion status with proper data extraction"""
        print("🎉 OPTIMIZATION COMPLETED!")
        print()
        
        # Try multiple sources for best_value
        best_value = None
        
        # Source 1: Direct from job_data
        if 'best_value' in job_data:
            best_value = job_data['best_value']
        
        # Source 2: From result -> optimization_result
        elif 'result' in job_data and job_data['result']:
            result = job_data['result']
            if isinstance(result, dict):
                if 'optimization_result' in result:
                    opt_result = result['optimization_result']
                    if isinstance(opt_result, dict) and 'best_value' in opt_result:
                        best_value = opt_result['best_value']
                elif 'best_value' in result:
                    best_value = result['best_value']
        
        # Source 3: From progress data
        elif 'progress' in job_data and job_data['progress']:
            progress = job_data['progress']
            if isinstance(progress, dict) and 'best_value' in progress:
                best_value = progress['best_value']
        
        # Source 4: Calculate from trial history
        if best_value is None and trial_history and 'trials' in trial_history:
            trials = trial_history['trials']
            if trials:
                completed_trials = [t for t in trials if t.get('status') == 'completed']
                if completed_trials:
                    # Find trial with highest val_accuracy
                    best_trial = max(completed_trials, 
                                key=lambda t: t.get('performance', {}).get('final_val_accuracy', 0))
                    best_value = best_trial.get('performance', {}).get('final_val_accuracy', 0)
        
        # FIXED: Get trial counts from multiple sources
        total_trials = 0
        successful_trials = 0
        
        # Try progress first
        if 'progress' in job_data and job_data['progress']:
            progress = job_data['progress']
            total_trials = progress.get('total_trials', 0)
            successful_trials = progress.get('completed_trials', 0)
        
        # Fallback to result data
        if total_trials == 0 and 'result' in job_data and job_data['result']:
            result = job_data['result']
            if isinstance(result, dict) and 'optimization_result' in result:
                opt_result = result['optimization_result']
                if isinstance(opt_result, dict):
                    total_trials = opt_result.get('total_trials', 0)
                    successful_trials = opt_result.get('successful_trials', 0)
        
        # Fallback to trial history
        if total_trials == 0 and trial_history and 'trials' in trial_history:
            trials = trial_history['trials']
            total_trials = len(trials) if trials else 0
            successful_trials = len([t for t in trials if t.get('status') == 'completed']) if trials else 0
        
        # Display results
        print(f"✅ Status: COMPLETED")
        if best_value is not None:
            print(f"🏆 Best Value: {best_value:.4f}")
        else:
            print(f"🏆 Best Value: N/A (check results files)")
        
        print(f"📊 Trials: {successful_trials}/{total_trials} successful")
        
        # Try to show best parameters if available
        best_params = None
        if 'result' in job_data and job_data['result']:
            result = job_data['result']
            if isinstance(result, dict):
                if 'optimization_result' in result:
                    opt_result = result['optimization_result']
                    if isinstance(opt_result, dict):
                        best_params = opt_result.get('best_params', {})
                elif 'best_params' in result:
                    best_params = result['best_params']
        
        if best_params and isinstance(best_params, dict) and best_params:
            print(f"\n🔧 Best Hyperparameters (top 5):")
            # Show top 5 most important parameters
            items = list(best_params.items())[:5]
            for key, value in items:
                print(f"   {key}: {value}")
            if len(best_params) > 5:
                print(f"   ... and {len(best_params) - 5} more parameters")
        
        print(f"\n💡 Full results available in downloaded files")

    
    def display_failed_status(self, job_data: Dict[str, Any]):
        """Display failure status"""
        print("❌ OPTIMIZATION FAILED")
        print()
        error = job_data.get('error', 'Unknown error')
        print(f"Error: {error}")
    
    def display_running_status(self, job_data: Dict[str, Any], trial_data: Optional[Dict[str, Any]], 
                         trial_history: Optional[Dict[str, Any]]):
        """Display running status - FIXED VERSION"""
        print("🚀 Status: RUNNING")
        
        # Get progress data from job_data (authoritative source for total_trials)
        progress = job_data.get('progress', {})
        api_total_trials = progress.get('total_trials', 10)
        
        # Get current trial number from trial data (more up-to-date)
        current_trial_num = 0
        trial_info = None
        
        # FIX: Safely handle trial_data that might be None or empty
        if trial_data and trial_data.get("success", False):
            trial_data_content = trial_data.get("data", {})
            if trial_data_content:  # Check if data exists and is not empty
                if 'current_trial' in trial_data_content:
                    trial_info = trial_data_content['current_trial']
                    # FIX: Add 1 to make trial numbers 1-indexed for display
                    current_trial_num = trial_info.get('trial_number', 0) + 1
                elif 'trial_id' in trial_data_content:
                    trial_info = trial_data_content
                    # FIX: Add 1 to make trial numbers 1-indexed for display
                    current_trial_num = trial_info.get('trial_number', 0) + 1
        
        # If no current trial found, check if job is actually completed
        if current_trial_num == 0:
            # Check if all trials are actually completed
            completed_trials = progress.get('completed_trials', 0)
            if completed_trials >= api_total_trials:
                current_trial_num = api_total_trials  # Show as completed
        
        # Use the correct total from API progress data
        total_trials = api_total_trials
        
        print(f"📋 Current Trial: {current_trial_num}/{total_trials}")
        
        # FIX: Only display trial details if trial_info exists
        if trial_info:
            trial_id = trial_info.get('trial_id', 'Unknown')
            trial_status = trial_info.get('status', 'unknown')
            
            print(f"🔑 Trial ID: {trial_id}")
            print(f"🎯 Trial Status: {trial_status}")
            
            # Model architecture info
            if 'model_size' in trial_info:
                model_size = trial_info['model_size']
                total_params = model_size.get('total_params', 0)
                memory_mb = model_size.get('memory_mb', 0)
                size_category = model_size.get('size_category', 'unknown')
                
                print(f"🧠 Model Size: {self.format_number(total_params)} parameters ({size_category})")
                print(f"💾 Memory Usage: {memory_mb:.1f} MB")
        else:
            # FIX: Better handling when no current trial (job completing)
            completed_trials = progress.get('completed_trials', 0)
            if completed_trials >= total_trials:
                print("🔑 Trial ID: All trials completed")
                print("🎯 Trial Status: Finalizing results...")
            else:
                print("📋 Current Trial: Fetching trial information...")
        
        # Progress bar with CORRECT total
        progress_bar = self.draw_progress_bar(current_trial_num, total_trials)
        print(f"📈 Overall Progress: {progress_bar}")
    
    def save_results_summary(self, job_data: Dict[str, Any], trial_history: Optional[Dict[str, Any]]):
        """Save a local summary of results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_summary_{timestamp}.json"
            
            summary = {
                "job_data": job_data,
                "trial_history": trial_history,
                "saved_at": datetime.now().isoformat(),
                "runpod_url": self.base_url
            }
            
            #with open(filename, 'w') as f:
            #    json.dump(summary, f, indent=2)
            
            #print(f"💾 Results summary saved to: {filename}")
            
            # AUTO-DOWNLOAD: Download all result files
            job_id = job_data.get("job_id")
            if job_id:
                print(f"\n🚀 Starting automatic file download...")
                success = self.auto_download_results(job_id, use_zip=True)
                if success:
                    print(f"✅ All optimization results downloaded successfully!")
                else:
                    print(f"⚠️  File download failed, but summary was saved")
            
        except Exception as e:
            print(f"⚠️  Failed to save results summary: {e}")


def create_optimization_config(args) -> Dict[str, Any]:
    """Create optimization configuration from command line arguments"""
    config = {
        "dataset_name": args.dataset,
        "mode": args.mode,
        "optimize_for": args.objective,
        "trials": args.trials
    }
    
    # Add config overrides
    config_overrides = {}
    
    if args.max_epochs:
        config_overrides["max_epochs_per_trial"] = args.max_epochs
    
    if args.health_weight is not None:
        config_overrides["health_weight"] = args.health_weight
    
    if args.timeout:
        config_overrides["timeout_hours"] = args.timeout
    
    if config_overrides:
        config["config_overrides"] = config_overrides
    
    return config


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Complete Optimization API Client")
    
    # Connection settings
    parser.add_argument("--url", default="vsm8hcyp2wi33o-8000.proxy.runpod.net",
                       help="RunPod URL (without https://)")
    parser.add_argument("--poll-interval", type=int, default=10,
                       help="Polling interval in seconds")
    
    # Operation modes
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--monitor-only", type=str, metavar="JOB_ID",
                       help="Only monitor existing job (provide job ID)")
    parser.add_argument("--info", action="store_true",
                       help="Show system information")
    parser.add_argument("--jobs", action="store_true",
                       help="List all jobs")
    parser.add_argument("--job-details", type=str, metavar="JOB_ID",
                       help="Show details for specific job")
    parser.add_argument("--trial-analysis", type=str, metavar="JOB_ID",
                       help="Show trial analysis for specific job")
    
    # Optimization parameters
    parser.add_argument("--dataset", default="mnist",
                       choices=["mnist", "cifar10", "cifar100", "fashion_mnist", "gtsrb", "imdb", "reuters"],
                       help="Dataset to optimize on")
    parser.add_argument("--mode", default="simple",
                       choices=["simple", "health"],
                       help="Optimization mode")
    parser.add_argument("--objective", default="val_accuracy",
                       help="Optimization objective")
    parser.add_argument("--trials", type=int, default=10,
                       help="Number of trials to run")
    
    # Config overrides
    parser.add_argument("--max-epochs", type=int,
                       help="Maximum epochs per trial")
    parser.add_argument("--health-weight", type=float,
                       help="Health weight (0.0-1.0) for health mode")
    parser.add_argument("--timeout", type=float,
                       help="Timeout in hours")
    
    args = parser.parse_args()
    
    # Create client
    client = OptimizationAPIClient(args.url, args.poll_interval)
    
    if args.interactive:
        # Interactive mode
        client.interactive_mode()
    elif args.monitor_only:
        # Monitor existing job
        client.monitor_job(args.monitor_only)
    elif args.info:
        # Show system info
        client.show_system_info()
    elif args.jobs:
        # List all jobs
        client.show_all_jobs()
    elif args.job_details:
        # Show job details
        client.show_job_details(args.job_details)
    elif args.trial_analysis:
        # Show trial analysis
        client.show_trial_analysis(args.trial_analysis)
    else:
        # Start new optimization and monitor
        config = create_optimization_config(args)
        result = client.start_optimization(config)
        
        if result["success"]:
            job_id = result["data"].get("job_id")
            print(f"✅ Optimization started! Job ID: {job_id}")
            print("⏳ Starting monitoring...")
            time.sleep(2)
            client.monitor_job(job_id)
        else:
            print(f"❌ Failed to start optimization: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()