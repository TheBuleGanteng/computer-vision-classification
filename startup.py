#!/usr/bin/env python3
"""
Development Server Startup Script for Computer Vision Classification Project

This script manages the startup and shutdown of both the frontend (Next.js) and 
backend (FastAPI) development servers for the hyperparameter optimization dashboard.

Usage:
    python startup.py

Features:
- Starts both frontend and backend servers simultaneously
- Kills existing servers before starting new ones (if already running)
- Graceful shutdown with Ctrl+C (kills both servers)
- Real-time output from both servers with color coding
- Automatic port conflict detection and resolution

Author: Computer Vision Classification Project
"""

import os
import sys
import signal
import subprocess
import threading
import time
import psutil
from pathlib import Path
from typing import Optional, List

# ANSI color codes for output formatting
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

class DevelopmentServerManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.frontend_dir = self.project_root / "web-ui"
        self.backend_dir = self.project_root / "src"
        
        # Process tracking
        self.frontend_process: Optional[subprocess.Popen] = None
        self.backend_process: Optional[subprocess.Popen] = None
        self.shutdown_requested = False
        
        # Port configuration
        self.frontend_port = 3000
        self.backend_port = 8000
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C and other shutdown signals"""
        print(f"\n{Colors.YELLOW}📡 Shutdown signal received. Stopping servers...{Colors.RESET}")
        self.shutdown_requested = True
        self._stop_all_servers()
        print(f"{Colors.GREEN}✅ All servers stopped successfully{Colors.RESET}")
        sys.exit(0)
        
    def _log(self, message: str, color: str = Colors.RESET, prefix: str = ""):
        """Print formatted log message"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"{color}[{timestamp}] {prefix}{message}{Colors.RESET}")
        
    def _find_processes_on_port(self, port: int) -> List[psutil.Process]:
        """Find processes running on specified port"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        processes.append(proc)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return processes
        
    def _kill_processes_on_port(self, port: int, service_name: str):
        """Kill any existing processes on the specified port"""
        processes = self._find_processes_on_port(port)
        if processes:
            self._log(f"Found {len(processes)} existing {service_name} process(es) on port {port}", Colors.YELLOW, "🔄 ")
            for proc in processes:
                try:
                    self._log(f"Killing {service_name} process (PID: {proc.pid})", Colors.RED, "💀 ")
                    proc.terminate()
                    # Wait for graceful termination
                    proc.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    # Force kill if graceful termination fails
                    try:
                        proc.kill()
                    except psutil.NoSuchProcess:
                        pass
                except Exception as e:
                    self._log(f"Error killing {service_name} process: {e}", Colors.RED, "❌ ")
            
            # Verify processes are gone
            remaining = self._find_processes_on_port(port)
            if remaining:
                self._log(f"Warning: {len(remaining)} {service_name} process(es) still running", Colors.YELLOW, "⚠️ ")
            else:
                self._log(f"Successfully killed existing {service_name} processes", Colors.GREEN, "✅ ")
                
    def _check_dependencies(self) -> bool:
        """Check if required dependencies and directories exist"""
        # Check directories
        if not self.frontend_dir.exists():
            self._log(f"Frontend directory not found: {self.frontend_dir}", Colors.RED, "❌ ")
            return False
            
        if not self.backend_dir.exists():
            self._log(f"Backend directory not found: {self.backend_dir}", Colors.RED, "❌ ")
            return False
            
        # Check package.json for frontend
        package_json = self.frontend_dir / "package.json"
        if not package_json.exists():
            self._log(f"package.json not found in {self.frontend_dir}", Colors.RED, "❌ ")
            return False
            
        # Check api_server.py for backend
        api_server = self.backend_dir / "api_server.py"
        if not api_server.exists():
            self._log(f"api_server.py not found in {self.backend_dir}", Colors.RED, "❌ ")
            return False
            
        self._log("All dependencies and directories verified", Colors.GREEN, "✅ ")
        return True
        
    def _stream_output(self, process: subprocess.Popen, service_name: str, color: str):
        """Stream output from subprocess in real-time with color coding"""
        while not self.shutdown_requested and process.poll() is None:
            try:
                line = process.stdout.readline()
                if line:
                    decoded_line = line.decode('utf-8', errors='replace').strip()
                    if decoded_line:
                        self._log(decoded_line, color, f"{service_name}: ")
            except Exception as e:
                if not self.shutdown_requested:
                    self._log(f"Error reading {service_name} output: {e}", Colors.RED, "❌ ")
                break
                
    def _start_backend_server(self):
        """Start the FastAPI backend server"""
        self._log("Starting backend server (FastAPI)...", Colors.BLUE, "🚀 ")
        
        try:
            # Kill any existing backend processes
            self._kill_processes_on_port(self.backend_port, "backend")
            
            # Start backend server
            cmd = [sys.executable, "api_server.py"]
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=self.backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=False,
                bufsize=1
            )
            
            # Start output streaming in separate thread
            backend_thread = threading.Thread(
                target=self._stream_output,
                args=(self.backend_process, "BACKEND", Colors.BLUE),
                daemon=True
            )
            backend_thread.start()
            
            # Give backend time to start
            time.sleep(2)
            
            if self.backend_process.poll() is None:
                self._log(f"Backend server started successfully (PID: {self.backend_process.pid})", Colors.GREEN, "✅ ")
                self._log(f"Backend API: http://localhost:{self.backend_port}", Colors.CYAN, "🌐 ")
            else:
                self._log("Backend server failed to start", Colors.RED, "❌ ")
                return False
                
        except Exception as e:
            self._log(f"Error starting backend server: {e}", Colors.RED, "❌ ")
            return False
            
        return True
        
    def _start_frontend_server(self):
        """Start the Next.js frontend development server"""
        self._log("Starting frontend server (Next.js)...", Colors.MAGENTA, "🚀 ")
        
        try:
            # Kill any existing frontend processes
            self._kill_processes_on_port(self.frontend_port, "frontend")
            
            # Start frontend server
            cmd = ["npm", "run", "dev"]
            self.frontend_process = subprocess.Popen(
                cmd,
                cwd=self.frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=False,
                bufsize=1
            )
            
            # Start output streaming in separate thread
            frontend_thread = threading.Thread(
                target=self._stream_output,
                args=(self.frontend_process, "FRONTEND", Colors.MAGENTA),
                daemon=True
            )
            frontend_thread.start()
            
            # Give frontend time to start
            time.sleep(3)
            
            if self.frontend_process.poll() is None:
                self._log(f"Frontend server started successfully (PID: {self.frontend_process.pid})", Colors.GREEN, "✅ ")
                self._log(f"Frontend UI: http://localhost:{self.frontend_port}", Colors.CYAN, "🌐 ")
            else:
                self._log("Frontend server failed to start", Colors.RED, "❌ ")
                return False
                
        except Exception as e:
            self._log(f"Error starting frontend server: {e}", Colors.RED, "❌ ")
            return False
            
        return True
        
    def _stop_all_servers(self):
        """Stop both frontend and backend servers"""
        if self.backend_process:
            try:
                self._log("Stopping backend server...", Colors.YELLOW, "🛑 ")
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                self._log("Backend server stopped", Colors.GREEN, "✅ ")
            except subprocess.TimeoutExpired:
                self._log("Force killing backend server...", Colors.RED, "💀 ")
                self.backend_process.kill()
            except Exception as e:
                self._log(f"Error stopping backend: {e}", Colors.RED, "❌ ")
                
        if self.frontend_process:
            try:
                self._log("Stopping frontend server...", Colors.YELLOW, "🛑 ")
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                self._log("Frontend server stopped", Colors.GREEN, "✅ ")
            except subprocess.TimeoutExpired:
                self._log("Force killing frontend server...", Colors.RED, "💀 ")
                self.frontend_process.kill()
            except Exception as e:
                self._log(f"Error stopping frontend: {e}", Colors.RED, "❌ ")
                
        # Clean up any remaining processes on ports
        self._kill_processes_on_port(self.backend_port, "backend")
        self._kill_processes_on_port(self.frontend_port, "frontend")
        
    def start_development_servers(self):
        """Main method to start both development servers"""
        self._log("🚀 Computer Vision Classification - Development Server Startup", Colors.BOLD + Colors.CYAN)
        self._log("=" * 70, Colors.CYAN)
        
        # Check dependencies
        if not self._check_dependencies():
            self._log("Dependency check failed. Exiting.", Colors.RED, "❌ ")
            return False
            
        try:
            # Start backend server first
            if not self._start_backend_server():
                self._log("Failed to start backend server. Exiting.", Colors.RED, "❌ ")
                return False
                
            # Start frontend server
            if not self._start_frontend_server():
                self._log("Failed to start frontend server. Stopping backend.", Colors.RED, "❌ ")
                self._stop_all_servers()
                return False
                
            # Print success message and instructions
            self._log("=" * 70, Colors.GREEN)
            self._log("🎉 Both servers started successfully!", Colors.BOLD + Colors.GREEN)
            self._log(f"🌐 Frontend Dashboard: http://localhost:{self.frontend_port}", Colors.CYAN)
            self._log(f"🔧 Backend API: http://localhost:{self.backend_port}", Colors.CYAN)
            self._log("", Colors.RESET)
            self._log("📝 Instructions:", Colors.BOLD + Colors.YELLOW)
            self._log("   • Open your browser to http://localhost:3000", Colors.YELLOW)
            self._log("   • Use the dashboard to start hyperparameter optimization", Colors.YELLOW)
            self._log("   • Press Ctrl+C to stop both servers", Colors.YELLOW)
            self._log("=" * 70, Colors.GREEN)
            
            # Keep script running and monitor processes
            while not self.shutdown_requested:
                # Check if processes are still running
                if self.backend_process and self.backend_process.poll() is not None:
                    self._log("Backend server stopped unexpectedly", Colors.RED, "❌ ")
                    break
                    
                if self.frontend_process and self.frontend_process.poll() is not None:
                    self._log("Frontend server stopped unexpectedly", Colors.RED, "❌ ")
                    break
                    
                time.sleep(1)
                
        except KeyboardInterrupt:
            # This should be handled by signal handler, but just in case
            self._log("KeyboardInterrupt received", Colors.YELLOW, "⚠️ ")
            
        finally:
            if not self.shutdown_requested:
                self._stop_all_servers()
                
        return True

def main():
    """Main entry point"""
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Create and run server manager
    manager = DevelopmentServerManager()
    success = manager.start_development_servers()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()