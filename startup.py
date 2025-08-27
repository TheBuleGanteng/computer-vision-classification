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
import logging
from pathlib import Path
from typing import Optional, List

# Setup logging
current_file = Path(__file__)
project_root = current_file.parent  # Adjust based on your project structure
log_file_path_resolved = project_root / "logs" / "non-cron.log"
log_file_path_resolved.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path_resolved),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        logger.debug("running _signal_handler ... Shutdown signal received")
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
        logger.debug(f"running _find_processes_on_port ... Searching for processes on port {port}")
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for conn in proc.net_connections():
                    if conn.laddr.port == port:
                        processes.append(proc)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        logger.debug(f"running _find_processes_on_port ... Found {len(processes)} processes on port {port}")
        return processes
        
    def _kill_processes_on_port(self, port: int, service_name: str):
        """Kill any existing processes on the specified port"""
        logger.debug(f"running _kill_processes_on_port ... Killing {service_name} processes on port {port}")
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

    def _cleanup_orphaned_processes(self):
        """Precisely clean up development processes while preserving VSCode functionality"""
        logger.debug("running _cleanup_orphaned_processes ... Starting precise cleanup of development processes")
        
        killed_count = 0
        
        try:
            logger.debug("running _cleanup_orphaned_processes ... Scanning for development processes to clean")
            self._log("🔍 Scanning for development processes to clean up...", Colors.CYAN)
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if not proc.cmdline():
                        continue
                        
                    cmdline = ' '.join(proc.cmdline())
                    proc_name = proc.name()
                    
                    # CRITICAL: Skip our own process
                    if proc.pid == os.getpid() or 'startup.py' in cmdline:
                        continue
                    
                    # CRITICAL: Comprehensive VSCode protection
                    if self._is_vscode_process(cmdline, proc_name):
                        continue
                        
                    # CRITICAL: Preserve system processes
                    if self._is_system_process(cmdline, proc_name):
                        continue
                    
                    should_kill = False
                    reason = ""
                    
                    # Target 1: Development Python servers
                    if self._is_development_python_process(proc_name, cmdline):
                        should_kill = True
                        reason = "Python development server"
                    
                    # Target 2: Node.js development processes (non-VSCode)
                    elif self._is_development_node_process(proc_name, cmdline):
                        should_kill = True
                        reason = "Node.js development process"
                    
                    # Target 3: Development tools
                    elif self._is_development_tool(proc_name, cmdline):
                        should_kill = True
                        reason = f"development tool ({proc_name})"
                    
                    # Target 4: Processes on development ports (most precise)
                    elif self._uses_development_ports(proc):
                        should_kill = True
                        port_info = self._get_process_port_info(proc)
                        reason = f"process using development ports {port_info}"
                    
                    if should_kill:
                        try:
                            port_details = self._get_process_port_info(proc)
                            logger.debug(f"running _cleanup_orphaned_processes ... Killing {reason} (PID: {proc.pid}){port_details}: {cmdline[:80]}...")
                            self._log(f"🔥 Killing {reason} (PID: {proc.pid}){port_details}: {cmdline[:60]}...", Colors.RED, "💀 ")
                            
                            proc.terminate()
                            try:
                                proc.wait(timeout=3)
                                killed_count += 1
                            except psutil.TimeoutExpired:
                                logger.debug(f"running _cleanup_orphaned_processes ... Force killing stubborn process (PID: {proc.pid})")
                                self._log(f"🔥 Force killing stubborn process (PID: {proc.pid})", Colors.RED, "💥 ")
                                proc.kill()
                                killed_count += 1
                                
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        except Exception as e:
                            logger.debug(f"running _cleanup_orphaned_processes ... Error killing {reason}: {e}")
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
                except Exception:
                    pass  # Suppress individual process errors
        
        except Exception as e:
            logger.debug(f"running _cleanup_orphaned_processes ... Error during cleanup: {e}")
        
        if killed_count > 0:
            logger.debug(f"running _cleanup_orphaned_processes ... Successfully cleaned up {killed_count} development processes")
            self._log(f"🧹 Cleaned up {killed_count} development processes", Colors.GREEN, "✅ ")
            time.sleep(2)  # Brief wait for process cleanup
        else:
            logger.debug("running _cleanup_orphaned_processes ... No development processes found to clean")
            self._log("🤷 No development processes found to clean (already clean)", Colors.GREEN, "✅ ")

    def _is_vscode_process(self, cmdline: str, proc_name: str) -> bool:
        """Comprehensive VSCode process detection"""
        
        # VSCode server processes (critical for remote development)
        vscode_server_indicators = [
            '.vscode-server',
            'vscode-server',
            'code-server',
            '/server-main.js',
            'bootstrap-fork',
            'extensionHost',
            'reh-linux',  # Remote Extension Host
            'tunnelService',
            'portForwarding'
        ]
        
        # VSCode client processes
        vscode_client_indicators = [
            'code.exe',
            'Code.exe', 
            '/code',
            'visual-studio-code',
            'vscode'
        ]
        
        # Check for any VSCode indicators
        cmdline_lower = cmdline.lower()
        proc_name_lower = proc_name.lower()
        
        for indicator in vscode_server_indicators + vscode_client_indicators:
            if indicator.lower() in cmdline_lower or indicator.lower() in proc_name_lower:
                return True
        
        # Additional check for VSCode-related paths
        if any(path in cmdline for path in [
            '/.vscode-server/',
            '/vscode-server/',
            'Microsoft VS Code',
            'Code Helper'
        ]):
            return True
        
        return False

    def _is_system_process(self, cmdline: str, proc_name: str) -> bool:
        """Identify critical system processes that should never be killed"""
        system_indicators = [
            'systemd', 'kernel', 'dbus', 'NetworkManager', 'cups', 'ssh',
            'wslhost', 'wsl', 'microsoft', 'windows', 'init', 'kthread',
            'explorer.exe', 'winlogon', 'csrss', 'lsass', 'services',
            'bash', 'zsh', 'fish', 'sh',  # Shell processes
            'tmux', 'screen',  # Terminal multiplexers
            'docker', 'containerd'  # Container processes
        ]
        
        cmdline_lower = cmdline.lower()
        proc_name_lower = proc_name.lower()
        
        return any(indicator in cmdline_lower or indicator in proc_name_lower 
                  for indicator in system_indicators)

    def _is_development_python_process(self, proc_name: str, cmdline: str) -> bool:
        """Identify Python development server processes"""
        if proc_name not in ['python', 'python3'] and 'python' not in cmdline.lower():
            return False
        
        development_keywords = [
            'uvicorn', 'fastapi', 'flask', 'django', 'tornado', 'bottle',
            'streamlit', 'jupyter', 'notebook', 'gunicorn', 'celery',
            'manage.py', 'runserver', 'wsgi', 'asgi', 'http.server',
            'api_server', 'dev', 'serve', 'server', 'cherrypy'
        ]
        
        cmdline_lower = cmdline.lower()
        return any(keyword in cmdline_lower for keyword in development_keywords)

    def _is_development_node_process(self, proc_name: str, cmdline: str) -> bool:
        """Identify Node.js development processes (excluding VSCode)"""
        if proc_name != 'node':
            return False
        
        # Already filtered out VSCode processes, so any remaining node processes
        # are likely development servers
        development_keywords = [
            'next', 'nuxt', 'webpack', 'vite', 'rollup', 'esbuild',
            'serve', 'dev', 'start', 'build', 'watch', 'hot',
            'live-server', 'http-server', 'express', 'koa'
        ]
        
        cmdline_lower = cmdline.lower()
        return any(keyword in cmdline_lower for keyword in development_keywords)

    def _is_development_tool(self, proc_name: str, cmdline: str) -> bool:
        """Identify development tool processes"""
        development_tools = [
            'npm', 'yarn', 'pnpm', 'webpack', 'rollup', 'esbuild', 'parcel',
            'gulp', 'grunt', 'nodemon', 'tsx', 'ts-node', 'next', 'nuxt',
            'vue-cli-service', 'ng', 'serve', 'live-server', 'http-server'
        ]
        
        return proc_name in development_tools

    def _uses_development_ports(self, proc) -> bool:
        """Check if process uses development ports"""
        try:
            for conn in proc.net_connections():
                if conn.status == psutil.CONN_LISTEN:
                    port = conn.laddr.port
                    
                    # Common development ports
                    development_port_ranges = [
                        (3000, 3010),  # React, Next.js
                        (4000, 4010),  # Various dev servers
                        (5000, 5010),  # Flask, various
                        (8000, 8010),  # Django, FastAPI, various
                        (8080, 8090),  # Common alt HTTP
                        (9000, 9010),  # Various dev tools
                    ]
                    
                    specific_dev_ports = [5173, 5174, 1337, 8888, 9999, 10000]
                    
                    # Check port ranges
                    for start, end in development_port_ranges:
                        if start <= port <= end:
                            return True
                    
                    # Check specific ports
                    if port in specific_dev_ports:
                        return True
                            
        except (psutil.AccessDenied, AttributeError, psutil.NoSuchProcess):
            pass
        
        return False

    def _get_process_port_info(self, proc) -> str:
        """Get formatted port information for a process"""
        try:
            listening_ports = []
            for conn in proc.net_connections():
                if conn.status == psutil.CONN_LISTEN:
                    listening_ports.append(str(conn.laddr.port))
            
            if listening_ports:
                return f" (ports: {', '.join(listening_ports)})"
            
        except (psutil.AccessDenied, AttributeError, psutil.NoSuchProcess):
            pass
        
        return ""
                
    def _check_dependencies(self) -> bool:
        """Check if required dependencies and directories exist"""
        logger.debug("running _check_dependencies ... Verifying project structure and dependencies")
        
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
            
        logger.debug("running _check_dependencies ... All dependencies verified successfully")
        self._log("All dependencies and directories verified", Colors.GREEN, "✅ ")
        return True
        
    def _stream_output(self, process: subprocess.Popen, service_name: str, color: str):
        """Stream output from subprocess in real-time with color coding"""
        logger.debug(f"running _stream_output ... Starting output streaming for {service_name}")
        
        # Check if stdout is available
        if process.stdout is None:
            self._log(f"No stdout available for {service_name}", Colors.YELLOW, "⚠️ ")
            return
            
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
        logger.debug("running _start_backend_server ... Starting FastAPI backend server")
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
                logger.debug(f"running _start_backend_server ... Backend server started successfully (PID: {self.backend_process.pid})")
                self._log(f"Backend server started successfully (PID: {self.backend_process.pid})", Colors.GREEN, "✅ ")
                self._log(f"Backend API: http://localhost:{self.backend_port}", Colors.CYAN, "🌐 ")
            else:
                logger.debug("running _start_backend_server ... Backend server failed to start")
                self._log("Backend server failed to start", Colors.RED, "❌ ")
                return False
                
        except Exception as e:
            logger.debug(f"running _start_backend_server ... Error starting backend server: {e}")
            self._log(f"Error starting backend server: {e}", Colors.RED, "❌ ")
            return False
            
        return True
        
    def _start_frontend_server(self):
        """Start the Next.js frontend development server"""
        logger.debug("running _start_frontend_server ... Starting Next.js frontend server")
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
                logger.debug(f"running _start_frontend_server ... Frontend server started successfully (PID: {self.frontend_process.pid})")
                self._log(f"Frontend server started successfully (PID: {self.frontend_process.pid})", Colors.GREEN, "✅ ")
                self._log(f"Frontend UI: http://localhost:{self.frontend_port}", Colors.CYAN, "🌐 ")
            else:
                logger.debug("running _start_frontend_server ... Frontend server failed to start")
                self._log("Frontend server failed to start", Colors.RED, "❌ ")
                return False
                
        except Exception as e:
            logger.debug(f"running _start_frontend_server ... Error starting frontend server: {e}")
            self._log(f"Error starting frontend server: {e}", Colors.RED, "❌ ")
            return False
            
        return True
        
    def _stop_all_servers(self):
        """Stop both frontend and backend servers"""
        logger.debug("running _stop_all_servers ... Stopping all development servers")
        
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
        logger.debug("running start_development_servers ... Starting development server startup process")
        self._log("🚀 Computer Vision Classification - Development Server Startup", Colors.BOLD + Colors.CYAN)
        self._log("=" * 70, Colors.CYAN)
        
        # Clean up orphaned processes first
        self._cleanup_orphaned_processes()
        
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
            self._log("📋 Instructions:", Colors.BOLD + Colors.YELLOW)
            self._log("   • Open your browser to http://localhost:3000", Colors.YELLOW)
            self._log("   • Use the dashboard to start hyperparameter optimization", Colors.YELLOW)
            self._log("   • Press Ctrl+C to stop both servers", Colors.YELLOW)
            self._log("=" * 70, Colors.GREEN)
            
            logger.debug("running start_development_servers ... Both servers started successfully, entering monitoring loop")
            
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
            logger.debug("running start_development_servers ... KeyboardInterrupt received")
            self._log("KeyboardInterrupt received", Colors.YELLOW, "⚠️ ")
            
        finally:
            if not self.shutdown_requested:
                self._stop_all_servers()
                
        return True

def main():
    """Main entry point"""
    logger.debug("running main ... Starting development server manager")
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Create and run server manager
    manager = DevelopmentServerManager()
    success = manager.start_development_servers()
    
    if not success:
        logger.debug("running main ... Development server startup failed")
        sys.exit(1)
    else:
        logger.debug("running main ... Development server startup completed successfully")

if __name__ == "__main__":
    main()