# Hyperparameter Optimization System

## Project Summary

This project implements a **comprehensive hyperparameter optimization system** for machine learning models with GPU acceleration via RunPod. The system automatically optimizes neural network architectures for both image classification (CNN) and text classification (LSTM) tasks using Bayesian optimization with health-aware model evaluation.

**Key Features:**
- **Multi-modal support**: Automatic CNN/LSTM architecture selection based on data type
- **Dual optimization modes**: Simple (pure performance) vs Health-aware (balanced performance + model health)
- **Cloud GPU acceleration**: Containerized deployment on RunPod for fast optimization
- **Real-time monitoring**: Live visualization of training progress, gradient flow, and model health
- **Automatic result synchronization**: All plots and data automatically downloaded to local machine
- **REST API**: FastAPI backend with comprehensive endpoints for job management

**Supported Datasets:**
- **Images**: MNIST, CIFAR-10, CIFAR-100, Fashion-MNIST, GTSRB (German Traffic Signs)
- **Text**: IMDB (sentiment), Reuters (topic classification)

## Quick Start Guide

### Prerequisites
- Docker installed and running
- RunPod account
- Docker Hub account

### Step 1: Build and Deploy Container

**First time setup or production deployment:**
```bash
./docker_build_script.sh true
```

**After local code updates (faster):**
```bash
./docker_build_script.sh quick
```

### Step 2: Set Up RunPod Instance

1. **Create RunPod Template:**
   - Container Image: `thebuleganteng/hyperparameter-optimizer:latest`
   - Container Start Command: `python3 src/api_server.py`
   - Expose HTTP Port: `8000`
   - Environment Variables: `PYTHONPATH=/app`, `TZ=Asia/Bangkok`
   - Volume Mounts: `/app/optimization_results` (for result persistence)

2. **Deploy Pod:**
   - Choose any GPU instance (system is lightweight)
   - Note the RunPod URL (format: `abc123-8000.proxy.runpod.net`)

### Step 3: Run Optimization

**Basic optimization (simple mode):**
```bash
python optimize_runpod.py --url your-runpod-url dataset=cifar10 mode=simple trials=15
```

**Health-aware optimization:**
```bash
python optimize_runpod.py --url your-runpod-url dataset=mnist mode=health optimize_for=val_accuracy trials=15 health_weight=0.3
```

**Interactive monitoring:**
```bash
python optimize_runpod.py --url your-runpod-url --interactive
```

### Step 4: Results

- **Automatic Download**: Results automatically sync to `./optimization_results/`
- **Comprehensive Output**: Plots, model files, hyperparameters, and analysis reports
- **Performance**: ~30 minutes for MNIST (15 trials, 15 epochs/trial on GPU)

## Detailed Program Review

### Optimization Modes

#### Simple Mode
**Purpose**: Pure performance optimization focusing solely on metrics like accuracy
- **Health Monitoring**: Always enabled for API reporting and analysis
- **Health Weighting**: Not applied to optimization decisions
- **Use Case**: Maximum performance when model health is not a primary concern
- **Example**: `mode=simple optimize_for=val_accuracy`

#### Health Mode
**Purpose**: Balanced optimization considering both performance and model health
- **Health Metrics**: Neuron utilization, parameter efficiency, training stability, gradient health
- **Weighting System**: Configurable balance between objective and health (default: 70% objective, 30% health)
- **Use Case**: Robust models for production deployment
- **Examples**:
  - Default: `mode=health optimize_for=val_accuracy` (30% health weight)
  - Balanced: `mode=health optimize_for=val_accuracy health_weight=0.5` (50/50)
  - Health-focused: `mode=health optimize_for=overall_health` (direct health optimization)

#### Available Objectives

**Universal Objectives** (work in both modes):
- `val_accuracy`: Validation accuracy (recommended)
- `accuracy`: Training accuracy
- `training_time`: Minimize training time
- `parameter_efficiency`: Accuracy per parameter
- `memory_efficiency`: Accuracy per memory usage

**Health-Only Objectives** (health mode only):
- `overall_health`: Overall model health score
- `neuron_utilization`: Active neuron usage
- `training_stability`: Training process stability

### Containerization and RunPod Integration

#### Multi-Stage Docker Architecture
```
Stage 1: Base Environment (CUDA 12.3 + cuDNN)
├── System dependencies
├── Python 3.10 installation
└── GPU drivers and libraries

Stage 2: Dependencies 
├── TensorFlow 2.19.0 with GPU support
├── FastAPI, Optuna, OpenCV
└── Scientific computing stack

Stage 3: Code (Lightweight, frequent updates)
├── Application source code
├── API endpoints
└── Optimization algorithms
```

#### RunPod Benefits
- **GPU Acceleration**: 10-20x faster than CPU-only optimization
- **Scalability**: Choose GPU tier based on workload
- **Cost Efficiency**: Pay-per-use GPU resources
- **Isolation**: Clean environment for each optimization run

#### API Endpoints
```
Health & Info:
GET  /health              # API health check
GET  /datasets            # Available datasets
GET  /modes               # Optimization modes
GET  /objectives          # Available objectives

Job Management:
POST /optimize            # Start optimization
GET  /jobs/{job_id}       # Job status and progress
GET  /jobs                # List all jobs
POST /jobs/{job_id}/stop  # Stop specific job

Results & Monitoring:
GET  /jobs/{job_id}/trials          # Trial history
GET  /jobs/{job_id}/current-trial   # Current trial data
GET  /jobs/{job_id}/best-trial      # Best trial so far
GET  /jobs/{job_id}/trends          # Performance trends
GET  /jobs/{job_id}/download        # Download results ZIP
```

### Real-Time Monitoring Features

#### Training Visualization
- **Progress Tracking**: Live loss/accuracy curves with overfitting detection
- **Architecture Analysis**: Real-time model size and complexity metrics
- **Performance Indicators**: Training speed and resource utilization

#### Health Monitoring
- **Gradient Flow**: Detection of vanishing/exploding gradients
- **Parameter Evolution**: Weight and bias distribution tracking
- **Neuron Utilization**: Dead neuron detection and efficiency analysis
- **Training Stability**: Convergence quality and consistency metrics

#### Result Synchronization
- **Automatic Download**: Complete results package synchronized to local machine
- **Comprehensive Output**:
  - Trial-by-trial plots and analysis
  - Summary visualizations and reports
  - Best model files and hyperparameters
  - Health monitoring data for all trials

### User Interface (Current: CLI + API)

#### Command Line Interface
```bash
# Basic optimization
python optimize_runpod.py --url runpod-url dataset=cifar10 trials=20

# Interactive mode with real-time monitoring
python optimize_runpod.py --url runpod-url --interactive

# Custom configuration
python optimize_runpod.py --url runpod-url dataset=mnist mode=health \
  optimize_for=val_accuracy health_weight=0.4 trials=25 max_epochs=20
```

#### API Integration
- **RESTful Endpoints**: Complete programmatic access
- **Real-time Updates**: WebSocket-style progress streaming via callbacks
- **Comprehensive Data**: All monitoring and result data available via API

## Next Steps

### 4a. RunPod Serverless Transition

**Current State**: Using RunPod Pods (persistent instances)
**Target State**: RunPod Serverless (on-demand GPU functions)

**Benefits of Serverless Transition:**
- **Cost Optimization**: Pay only for GPU compute time (not idle time)
- **Local Development**: Run optimization logic locally, use RunPod only for GPU-intensive training
- **Automatic Scaling**: Serverless functions scale based on demand
- **Result Persistence**: All results saved locally as if running on local CPU

**Implementation Plan:**
1. **Serverless Function Design**: Package training logic as serverless endpoint
2. **Local Orchestration**: Optimization runner stays on local machine
3. **GPU Delegation**: Send individual trial training to serverless functions
4. **Result Aggregation**: Collect and analyze results locally
5. **Hybrid Architecture**: Best of both worlds - local control + cloud GPU power

**Expected Workflow:**
```bash
# Run locally, use RunPod serverless for GPU training
python optimize_local.py dataset=cifar10 trials=20 --use-serverless
# Results saved to ./optimization_results/ as usual
```

### 4b. Frontend Development

**Vision**: Comprehensive web interface for hyperparameter optimization

#### Core Features

**1. Hyperparameter Configuration Interface**
- **Dataset Selection**: Visual dataset browser with preview and statistics
- **Mode Selection**: Simple vs Health with clear explanations and use cases
- **Parameter Sliders**: Interactive controls for all optimization parameters
- **Configuration Validation**: Real-time validation with helpful error messages
- **Preset Templates**: Common configurations for different use cases

**2. Real-Time Architecture Visualization**
- **Model Builder**: Interactive 3D visualization of CNN/LSTM architectures as they're constructed
- **Layer-by-Layer Animation**: Watch the network build with parameter counts and connections
- **Architecture Comparison**: Side-by-side comparison of different trial architectures
- **Performance Overlay**: Real-time performance metrics overlaid on architecture diagrams

**3. Optimization Process Monitoring**
- **Trial Dashboard**: Live grid showing all trials with status and performance
- **Progress Visualization**: Real-time charts of optimization progress and trends
- **Health Monitoring**: Live model health metrics with alerting for issues
- **Resource Utilization**: GPU usage, memory consumption, and timing statistics

**4. Results Analysis Interface**
- **Interactive Result Explorer**: Drill down into trial results with filtering and sorting
- **Performance Comparison**: Compare trials across multiple dimensions
- **Hyperparameter Importance**: Visual analysis of which parameters matter most
- **Export Capabilities**: Download results, configurations, and reports

#### Technical Implementation

**Frontend Stack:**
- **Framework**: React with TypeScript for type safety
- **Visualization**: D3.js for custom charts, Three.js for 3D architecture views
- **Real-time Updates**: WebSocket connection to FastAPI backend
- **State Management**: Redux Toolkit for complex optimization state
- **UI Components**: Modern design system with responsive mobile support

**Backend Integration:**
- **API Enhancement**: Expand FastAPI endpoints for frontend needs
- **Real-time Streaming**: WebSocket endpoints for live updates
- **Result Management**: Enhanced file management and download APIs
- **User Sessions**: Support for multiple concurrent optimizations

**Expected User Journey:**
1. **Setup**: Select dataset, configure optimization parameters via intuitive interface
2. **Launch**: Start optimization with real-time architecture visualization
3. **Monitor**: Watch progress with live charts, health monitoring, and trial results
4. **Analyze**: Explore results with interactive tools and comparison features
5. **Export**: Download optimized models and comprehensive analysis reports

This frontend will make the sophisticated optimization capabilities accessible to users without command-line experience while providing powerful visualization and analysis tools for advanced users.