#!/usr/bin/env python3
"""
Development Server Startup Script for Computer Vision Classification Project

This script manages the startup and shutdown of both the frontend (Next.js) and
backend (FastAPI) development servers for the hyperparameter optimization dashboard.

Usage:
    python startup.py                    # Start local development servers
    python startup.py --containerized    # Start Docker containers
    python startup.py --containerized --clear_containers   # Delete existing containers before starting Docker containers

Features:
- Deletes existing containers, images, and build cache when --clear_containers is specified
- Starts both frontend and backend servers simultaneously
- Supports both local development and containerized deployment
- Kills existing servers before starting new ones (if already running)
- Graceful shutdown with Ctrl+C (kills both servers)
- Real-time output from both servers with color coding
- Automatic port conflict detection and resolution

"""

import os
import sys
import signal
import subprocess
import threading
import time
import psutil
import logging
import argparse
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
    def __init__(self, containerized: bool = False, clear_containers: bool = False):
        self.project_root = Path(__file__).parent
        self.frontend_dir = self.project_root / "web-ui"
        self.backend_dir = self.project_root / "src"

        # Deployment mode
        self.containerized = containerized
        self.clear_containers = clear_containers

        # Process tracking
        self.frontend_process: Optional[subprocess.Popen] = None
        self.backend_process: Optional[subprocess.Popen] = None
        self.docker_compose_process: Optional[subprocess.Popen] = None
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
        """Stop both frontend and backend servers (local or containerized)"""
        logger.debug("running _stop_all_servers ... Stopping all development servers")

        if self.containerized:
            # Stop Docker containers
            if self.docker_compose_process:
                try:
                    self._log("Stopping Docker containers...", Colors.YELLOW, "🛑 ")
                    subprocess.run(["docker", "compose", "down"], cwd=self.project_root, check=False)
                    self._log("Docker containers stopped", Colors.GREEN, "✅ ")
                except Exception as e:
                    self._log(f"Error stopping Docker containers: {e}", Colors.RED, "❌ ")
        else:
            # Stop local development servers
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

    def _clear_docker_resources(self):
        """Clear Docker containers, images, and build cache"""
        logger.debug("running _clear_docker_resources ... Clearing Docker resources")
        self._log("Clearing Docker resources...", Colors.YELLOW, "🧹 ")

        try:
            # Stop and remove all containers for this project
            self._log("Stopping and removing project containers...", Colors.YELLOW, "🛑 ")
            subprocess.run(
                ["docker", "compose", "down", "--volumes", "--remove-orphans"],
                cwd=self.project_root,
                check=False,
                capture_output=True
            )

            # Remove project-specific images
            self._log(" Removing project Docker images...", Colors.YELLOW, "🗑️ ")
            subprocess.run(
                ["docker", "compose", "rm", "-f"],
                cwd=self.project_root,
                check=False,
                capture_output=True
            )

            # Get list of images to remove
            image_result = subprocess.run(
                ["docker", "images", "-q", "computer-vision-classification-backend", "computer-vision-classification-frontend"],
                capture_output=True,
                text=True
            )

            if image_result.stdout.strip():
                image_ids = image_result.stdout.strip().split('\n')
                self._log(f"Removing {len(image_ids)} project images...", Colors.YELLOW, "🗑️ ")
                subprocess.run(
                    ["docker", "rmi", "-f"] + image_ids,
                    check=False,
                    capture_output=True
                )

            # Prune build cache
            self._log("Pruning Docker build cache...", Colors.YELLOW, "🧹 ")
            subprocess.run(
                ["docker", "builder", "prune", "-f"],
                check=False,
                capture_output=True,
                timeout=30  # 30 second timeout
            )

            self._log("Docker resources cleared successfully", Colors.GREEN, "✅ ")
            return True

        except Exception as e:
            logger.debug(f"running _clear_docker_resources ... Error clearing Docker resources: {e}")
            self._log(f"Warning: Error clearing Docker resources: {e}", Colors.YELLOW, "⚠️ ")
            return False

    def _start_containerized_servers(self):
        """Start both servers using Docker Compose"""
        logger.debug("running _start_containerized_servers ... Starting Docker containers")
        self._log("Starting containerized servers (Docker Compose)...", Colors.BLUE, "🐳 ")

        try:
            # Clear Docker resources if requested
            if self.clear_containers:
                if not self._clear_docker_resources():
                    self._log("Warning: Failed to clear all Docker resources, continuing anyway...", Colors.YELLOW, "⚠️ ")

            # Build containers first with real-time output
            self._log("Building Docker images (this may take several minutes)...", Colors.CYAN, "🔨 ")
            build_process = subprocess.Popen(
                ["docker", "compose", "build", "--progress=plain"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=False,
                bufsize=1
            )

            # Stream build output in real-time
            if build_process.stdout:
                while True:
                    line = build_process.stdout.readline()
                    if not line and build_process.poll() is not None:
                        break
                    if line:
                        decoded_line = line.decode('utf-8', errors='replace').strip()
                        if decoded_line:
                            self._log(decoded_line, Colors.CYAN, "🔨 ")

            build_process.wait()

            if build_process.returncode != 0:
                self._log("Docker build failed", Colors.RED, "❌ ")
                return False

            self._log("", Colors.GREEN, "")
            self._log("=" * 70, Colors.GREEN, "")
            self._log("✅ BUILD COMPLETE - Both frontend and backend images built successfully", Colors.GREEN, "")
            self._log("=" * 70, Colors.GREEN, "")
            self._log("", Colors.GREEN, "")

            # Start containers
            self._log("Starting Docker containers...", Colors.CYAN, "🚀 ")
            self.docker_compose_process = subprocess.Popen(
                ["docker", "compose", "up"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=False,
                bufsize=1
            )

            # Start output streaming
            docker_thread = threading.Thread(
                target=self._stream_output,
                args=(self.docker_compose_process, "DOCKER", Colors.CYAN),
                daemon=True
            )
            docker_thread.start()

            # Wait for containers to be healthy
            self._log("Waiting for containers to be healthy...", Colors.YELLOW, "⏳ ")
            time.sleep(10)

            # Check container health
            health_check = subprocess.run(
                ["docker", "compose", "ps"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if "cv-classification-backend" in health_check.stdout and "cv-classification-frontend" in health_check.stdout:
                logger.debug("running _start_containerized_servers ... Docker containers started successfully")
                self._log("", Colors.GREEN, "")
                self._log("=" * 70, Colors.GREEN, "")
                self._log("🎉 CONTAINERS READY - Frontend and backend are running!", Colors.GREEN, "")
                self._log("=" * 70, Colors.GREEN, "")
                self._log("🌐 Frontend: http://localhost:3000", Colors.CYAN, "")
                self._log("🔧 Backend: Running internally (accessed via frontend proxy)", Colors.CYAN, "")
                self._log("📊 View logs: docker compose logs -f", Colors.CYAN, "")
                self._log("=" * 70, Colors.GREEN, "")
                self._log("", Colors.GREEN, "")
                return True
            else:
                self._log("Docker containers failed health check", Colors.RED, "❌ ")
                self._log(health_check.stdout, Colors.YELLOW, "")
                return False

        except Exception as e:
            logger.debug(f"running _start_containerized_servers ... Error starting Docker containers: {e}")
            self._log(f"Error starting Docker containers: {e}", Colors.RED, "❌ ")
            return False
        
    def start_development_servers(self):
        """Main method to start both development servers"""
        logger.debug("running start_development_servers ... Starting development server startup process")

        mode_text = "Containerized (Docker)" if self.containerized else "Local Development"
        self._log(f"🚀 Computer Vision Classification - {mode_text} Server Startup", Colors.BOLD + Colors.CYAN)
        self._log("=" * 70, Colors.CYAN)

        if self.containerized:
            # Start containerized servers
            try:
                if not self._start_containerized_servers():
                    self._log("Failed to start Docker containers. Exiting.", Colors.RED, "❌ ")
                    return False
            except Exception as e:
                self._log(f"Error starting containers: {e}", Colors.RED, "❌ ")
                return False
        else:
            # Start local development servers
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
            except Exception as e:
                self._log(f"Error starting local servers: {e}", Colors.RED, "❌ ")
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
        if self.containerized:
            self._log("   • View container logs: docker compose logs -f", Colors.YELLOW)
        self._log("=" * 70, Colors.GREEN)

        logger.debug("running start_development_servers ... Both servers started successfully, entering monitoring loop")

        try:
            # Keep script running and monitor processes
            while not self.shutdown_requested:
                if self.containerized:
                    # Check if Docker containers are still running
                    if self.docker_compose_process and self.docker_compose_process.poll() is not None:
                        self._log("Docker Compose stopped unexpectedly", Colors.RED, "❌ ")
                        break
                else:
                    # Check if local processes are still running
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Start Computer Vision Classification development servers"
    )
    parser.add_argument(
        "--containerized",
        action="store_true",
        help="Run servers in Docker containers instead of locally"
    )
    parser.add_argument(
        "--clear_containers",
        action="store_true",
        help="Clear all Docker containers, images, and build cache before starting (only with --containerized)"
    )
    args = parser.parse_args()

    # Validate arguments
    if args.clear_containers and not args.containerized:
        parser.error("--clear_containers can only be used with --containerized")

    logger.debug(f"running main ... Starting development server manager (containerized={args.containerized}, clear_containers={args.clear_containers})")

    # Change to script directory
    os.chdir(Path(__file__).parent)

    # Create and run server manager
    manager = DevelopmentServerManager(
        containerized=args.containerized,
        clear_containers=args.clear_containers
    )
    success = manager.start_development_servers()

    if not success:
        logger.debug("running main ... Development server startup failed")
        sys.exit(1)
    else:
        logger.debug("running main ... Development server startup completed successfully")

if __name__ == "__main__":
    main()