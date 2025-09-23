# Hyperparameter Optimization System with 3D Neural Network Visualization

## I. Overview

### Project Purpose

This project implements a **comprehensive hyperparameter optimization system** for machine learning models with **cloud GPU acceleration** and **interactive 3D neural network visualization**. The system automatically optimizes neural network architectures for both image classification (CNN) and text classification (LSTM) tasks using Bayesian optimization with health-aware model evaluation.

**Educational Objectives:**

• **Educating users about primary hyperparameters** available in model architecture and how those parameters affect model performance including:
  - Layer depth and width configurations
  - Activation function selection and impact
  - Regularization techniques (dropout, batch normalization)
  - Optimizer selection and learning rate schedules
  - Architecture-specific parameters (conv filters, LSTM units, dense connections)

• **Illustrating the importance of model health** as opposed to pure test accuracy when constructing models useful in real-world applications:
  - Neuron utilization patterns and dead neuron detection
  - Parameter efficiency vs. model bloat trade-offs
  - Training stability and convergence quality metrics
  - Gradient health and vanishing/exploding gradient prevention
  - Accuracy consistency across validation splits
  - Overfitting detection through health-aware evaluation

### Project Structure

```
computer-vision-classification/
├── src/                              # Core backend implementation
│   ├── api_server.py                 # FastAPI server with visualization endpoints
│   ├── optimizer.py                  # Bayesian optimization with health metrics
│   ├── model_visualizer.py           # 3D visualization data preparation ✅ NEW
│   ├── model_builder.py              # Dynamic architecture generation
│   ├── health_analyzer.py            # Model health evaluation system
│   ├── dataset_manager.py            # Multi-modal dataset handling
│   ├── handlers/
│   │   └── runpod_handler.py         # Cloud GPU service integration
│   └── utils/
│       └── logger.py                 # Enhanced logging system
├── web-ui/                           # Next.js frontend application
│   ├── src/
│   │   ├── components/
│   │   │   ├── dashboard/            # Optimization dashboard
│   │   │   ├── optimization/         # Parameter configuration
│   │   │   └── visualization/        # 3D model viewer components 🔄 IN PROGRESS
│   │   ├── lib/
│   │   │   └── api/                  # Backend integration client
│   │   └── hooks/                    # React data fetching hooks
├── datasets/                         # Local dataset storage
├── logs/                            # Unified logging output
├── test_*.py                        # Backend testing suite ✅ COMPREHENSIVE
└── start_servers.py                 # Development environment manager
```

---

### **Understanding Configuration Data Flow**

This project uses a dual-path configuration architecture with a sophisticated hyperparameter management system:

#### **Path 1: API-Driven Flow (Web UI) - Distributed Architecture**
```
🌐 User Input (Web UI)
       ↓ HTTP POST /optimize
💻 OptimizationRequest (api_server.py) [LOCAL MACHINE]
  • API validation layer
  • User-friendly field names
  • HTTP request parsing
  • User-controlled defaults
       ↓
💻 create_optimization_config() [LOCAL MACHINE]
  • Conversion function
  • Type transformations (string → enum)
  • Pass-through all user values
       ↓
💻 OptimizationConfig (optimizer.py) [LOCAL MACHINE]
  • Business logic configuration
  • Fail-fast validation
  • System-controlled defaults
  • Enum types for internal use
       ↓
💻 ModelOptimizer → HyperparameterSelector → Optuna [LOCAL COORDINATION]
  • Optuna study orchestration
  • Hyperparameter selection
  • Trial coordination
  • Progress aggregation
       ↓ For each trial:
       ↓ HTTP POST https://api.runpod.ai/v2/{endpoint}/run
☁️  RunPod API [CLOUD SERVICE]
  • Receives trial parameters
  • Invokes serverless handler
       ↓ Serverless invocation
🔥 handler.py [RUNPOD GPU WORKER]
  • ModelConfig creation
  • ModelBuilder execution  
  • GPU-accelerated training
  • Progress updates via runpod.serverless.progress_update()
       ↓ Training results
☁️  RunPod API [CLOUD SERVICE]
  • Returns trial results
       ↓ HTTP response
💻 ModelOptimizer [LOCAL COORDINATION]
  • Receives trial results
  • Updates Optuna study
  • Continues optimization
       ↓ Progress updates
💻 api_server.py [LOCAL MACHINE]
  • WebSocket/polling updates
       ↓
🌐 Web UI [FRONTEND]
  • Real-time progress display
  • Final results presentation
```

**Key Architecture Points:**
- **Local Coordination**: Optuna study and optimization logic runs on your local machine
- **Remote Execution**: Individual trials execute on RunPod GPU workers  
- **Cost Efficiency**: You only pay for GPU time during actual model training
- **Scalability**: Multiple trials can run in parallel on different RunPod workers

#### **Path 2: Programmatic Flow (Direct Usage)**
```
Python Code
       ↓
OptimizationConfig (optimizer.py)
  • Direct instantiation
  • Business logic configuration
  • Fail-fast validation
  • System-controlled defaults
       ↓
ModelOptimizer → HyperparameterSelector → Optuna → ModelConfig → ModelBuilder
```

#### **Path 3: Hyperparameter Configuration Flow**
```
HyperparameterSelector.suggest_hyperparameters()
  • Uses Optuna to suggest architecture parameters
  • Randomly selects: use_global_pooling, kernel_size, num_layers_conv, etc.
       ↓
ModelOptimizer._train_locally_for_trial()
  • Creates empty ModelConfig()
  • Dynamically populates with Optuna-suggested parameters
  • Uses ModelConfig defaults for non-suggested parameters
       ↓
ModelBuilder(model_config)
  • Receives fully-configured ModelConfig
  • Uses all parameters for model construction
```

#### **ModelConfig Default vs Override Pattern**
```
Scenario 1: Hyperparameter Optimization (Normal Flow)
ModelConfig() defaults → Overridden by Optuna suggestions → Used by ModelBuilder

Scenario 2: Testing/Development/Standalone
ModelConfig() defaults → Used directly by ModelBuilder → No optimization

Scenario 3: Fallback/Error Recovery  
ModelConfig() defaults → Used when Optuna fails → Safe fallback values
```

### **Key Architecture Principles**

**Variable Ownership:**
- **OptimizationRequest**: Owns user-controlled defaults (trials=50, batch_size=32, etc.)
- **OptimizationConfig**: Owns system-controlled defaults (timeout_hours=None, health_monitoring_frequency=1, etc.)
- **ModelConfig**: Owns model architecture defaults (num_layers_conv=2, kernel_size=(3,3), use_global_pooling=False, etc.)
- **HyperparameterSelector**: Manages Optuna parameter suggestion and fallback logic

**Data Flow Rules:**
1. **API Path**: User → OptimizationRequest → create_optimization_config() → OptimizationConfig
2. **Programmatic Path**: Developer → OptimizationConfig directly
3. **Hyperparameter Path**: HyperparameterSelector → Optuna suggestions → ModelConfig population → ModelBuilder
4. **No Conflicting Defaults**: Each variable has defaults in only ONE class
5. **Fail-Fast**: OptimizationConfig validates all required values immediately
6. **Smart Defaults**: ModelConfig provides sensible defaults that work when Optuna is bypassed

**Benefits:**
- ✅ **Clear Separation**: API concerns vs business logic
- ✅ **No Duplication**: Single source of truth for each variable type  
- ✅ **Type Safety**: String validation in API, enum validation in business logic
- ✅ **Flexibility**: Supports both UI and programmatic usage patterns
- ✅ **Maintainability**: Easy to understand which class controls which variables

---

## II. Key Functionality and Features

### Core Optimization Features
- ✅ **Multi-modal Dataset Support**: MNIST, CIFAR-10/100, Fashion-MNIST, GTSRB, IMDB, Reuters
- ✅ **Dual Architecture Support**: Automatic CNN/LSTM selection based on data type
- ✅ **Bayesian Optimization**: Intelligent hyperparameter search with Optuna
- ✅ **Health-Aware Evaluation**: 6-metric model health assessment system
- ✅ **Dual Optimization Modes**: Simple (performance-only) vs Health-aware (balanced)

### Cloud Infrastructure
- ✅ **RunPod Service Integration**: Seamless cloud GPU execution with JSON API
- ✅ **Simultaneous Workers**: 2-6 concurrent workers with 5.5x speedup at maximum concurrency
- ✅ **Multi-GPU per Worker**: TensorFlow MirroredStrategy with 3.07x speedup
- ✅ **Real-time Progress Aggregation**: Thread-safe concurrent training progress visualization
- ✅ **Local Fallback**: Automatic local execution when cloud service unavailable
- ✅ **Accuracy Synchronization**: <0.5% gap between cloud and local execution
- ✅ **RunPod S3 Storage Integration**: Automatic model and artifact transfer via RunPod Network Volumes
- ✅ **Simplified Architecture**: Deprecated train_locally/build_model_locally flags in favor of unified S3-based transfer system

### Backend API & Data Processing
- ✅ **FastAPI REST API**: Comprehensive endpoints for job management and data retrieval
- ✅ **Real-time WebSocket Support**: Live optimization progress streaming
- ✅ **3D Visualization Data Pipeline**: Model architecture to 3D coordinates transformation
- ✅ **Configuration Architecture Consolidation**: Eliminated ~70% field overlap between OptimizationRequest and OptimizationConfig with clean separation of user vs system variables
- ✅ **JSON Serialization**: Complete export functionality for visualization data
- ✅ **Health Metrics Integration**: Performance-based color coding and visual indicators

### Frontend Interface
- ✅ **Next.js Modern UI**: Responsive dashboard with real-time updates
- ✅ **Trial Gallery**: Interactive display of optimization results with best model highlighting
- ✅ **Summary Statistics**: Live aggregated performance metrics
- ✅ **Parameter Configuration**: Intuitive hyperparameter selection interface
- ✅ **Cytoscape.js + TensorBoard Educational Visualization**: Complete interactive neural network architecture exploration with comprehensive training metrics
- ✅ **Embedded Training Plot System**: Immediate visualization of training progress, gradient flow, and model health metrics
- ✅ **Optimized Model Download**: Smart download button that activates when final model is built with best hyperparameters, includes automatic model availability detection
- ✅ **Mobile-Responsive Design**: Touch-friendly controls and optimized mobile experience

### Visualization & Export
- ✅ **Best Model Tracking**: Automatic identification and highlighting of optimal architectures  
- ✅ **Performance Color Coding**: Visual indicators based on accuracy and health metrics
- ✅ **Architecture Data Export**: JSON download with complete model structure and metadata
- ✅ **Dynamic Model Architecture Legend**: Model-specific legends showing only layer types present in current architecture with visual consistency
- ✅ **Optimized Model Download**: Automatic final model building with best hyperparameters and .keras format download for deployment
- 🔄 **Interactive Cytoscape.js Architecture Diagrams**: Layer-by-layer DAG exploration with forward propagation animations and TensorBoard metrics integration
- 🔄 **Educational Export Options**: Vector architecture diagrams (SVG/PDF), training metric charts, animated data flow sequences

### Testing & Quality Assurance
- ✅ **Comprehensive Backend Testing**: Unit, integration, and end-to-end test suites
- ✅ **API Endpoint Validation**: Complete testing of visualization data pipeline
- ✅ **JSON Serialization Testing**: Download functionality and data integrity verification
- ✅ **Multi-architecture Support Testing**: CNN, LSTM, and mixed architecture validation
- 🔄 **Frontend Component Testing**: Cytoscape.js visualization components, TensorBoard integration, and educational user interactions
- 🔄 **Cross-platform Compatibility**: Desktop, tablet, and mobile device testing

---

## III. Completed Implementation

### Data Structures and Core Systems

#### **ModelVisualizer Class** (`src/model_visualizer.py`)
Complete 3D visualization data preparation system with the following data structures:

```python
@dataclass
class LayerVisualization:
    layer_id: str                    # Unique identifier for 3D rendering
    layer_type: str                  # Layer classification (dense, conv, lstm, etc.)
    position_z: float                # Z-axis positioning in 3D space
    width: float                     # Layer width for 3D scaling
    height: float                    # Layer height for 3D scaling  
    depth: float                     # Layer depth for 3D scaling
    parameters: int                  # Parameter count for sizing
    filters: Optional[int]           # CNN-specific filter information
    kernel_size: Optional[Tuple[int, int]]  # CNN kernel dimensions
    units: Optional[int]             # Dense/LSTM unit count
    color_intensity: float           # Performance-based coloring
    opacity: float                   # Visual transparency level

@dataclass
class ArchitectureVisualization:
    architecture_type: str           # CNN, LSTM, or Generic
    layers: List[LayerVisualization] # Complete layer sequence
    total_parameters: int            # Full model parameter count
    model_depth: int                 # Number of layers for 3D depth
    max_layer_width: float          # Scaling reference for proportional sizing
    max_layer_height: float         # Scaling reference for proportional sizing
    performance_score: float        # Model accuracy (0.0-1.0)
    health_score: Optional[float]   # Composite health metric (0.0-1.0)
    
    @property
    def performance_color(self) -> str:  # Dynamic color based on performance
    @property  
    def health_color(self) -> str:       # Dynamic color based on health
```

#### **Health Metrics System** (`src/health_analyzer.py`)
Six-metric model evaluation providing comprehensive model assessment:

```python
# Health Metrics Calculations:
- neuron_utilization: float      # Active neuron percentage (0.0-1.0)
- parameter_efficiency: float    # Performance per parameter ratio
- training_stability: float      # Loss variance during training
- gradient_health: float         # Gradient flow quality assessment
- convergence_quality: float     # Training convergence smoothness
- accuracy_consistency: float    # Cross-validation stability
```

#### **API Endpoint Structure** (`src/api_server.py`)
RESTful endpoints for frontend integration:

```python
# 3D Visualization Endpoints:
GET  /jobs/{job_id}/best-model           # Fetch best model with 3D data
GET  /jobs/{job_id}/best-model/download  # Download visualization JSON
GET  /jobs/{job_id}/status               # Job progress and completion status
POST /optimize                           # Start optimization job
GET  /health                            # Server health check
```

#### **Optimization Integration** (`src/optimizer.py`)
Enhanced ModelOptimizer class with visualization support:

```python
class ModelOptimizer:
    def __init__(self):
        self.model_visualizer = ModelVisualizer()  # Integrated 3D preparation
    
    def get_best_model_visualization_data(self) -> Optional[Dict[str, Any]]:
        # Returns complete 3D visualization data for best performing model
        # Includes trial metadata, architecture data, and performance metrics
```

### Backend Systems Integration

#### **Thread-Safe Concurrent Processing**
- **Multi-worker Support**: 2-6 simultaneous RunPod workers with linear performance scaling
- **Progress Aggregation**: Real-time updates from multiple concurrent optimization trials
- **Shared State Management**: Thread-safe best model tracking across concurrent executions
- **Error Handling**: Graceful degradation and automatic retry mechanisms

#### **Cloud-Local Hybrid Architecture**  
- **RunPod Service Integration**: JSON-based communication with cloud GPU workers
- **Accuracy Synchronization**: Verified <0.5% performance gap between cloud and local execution
- **Automatic Fallback**: Seamless transition to local processing when cloud unavailable
- **Multi-GPU Optimization**: TensorFlow MirroredStrategy with 3.07x speedup validation

#### **Data Pipeline Architecture**
- **Multi-modal Dataset Support**: Unified handling of image and text classification datasets
- **Dynamic Architecture Generation**: Automatic CNN/LSTM selection based on data characteristics
- **Real-time Metrics Collection**: Live health and performance data aggregation during training
- **Comprehensive Logging**: Unified logging system with rotation and error tracking

### Frontend Implementation Status

#### **Dashboard Infrastructure** (`web-ui/src/components/dashboard/`)
- ✅ **Optimization Dashboard**: Complete real-time optimization monitoring interface
- ✅ **Trial Gallery**: Interactive grid display with best model visual highlighting
- ✅ **Summary Statistics**: Live performance metrics with automatic updates
- ✅ **Parameter Configuration**: Intuitive hyperparameter selection with validation

#### **API Integration** (`web-ui/src/lib/api/`)
- ✅ **TypeScript Client**: Complete backend integration with error handling
- ✅ **Real-time Updates**: WebSocket integration for live progress monitoring  
- ✅ **Data Fetching Hooks**: React Query integration for efficient data management
- ✅ **Error Boundary System**: Comprehensive error handling and user feedback

#### **User Interface Components**
- ✅ **Responsive Design**: Mobile-first approach with touch-friendly interactions
- ✅ **Loading States**: Professional loading indicators and skeleton screens
- ✅ **Error Handling**: User-friendly error messages and recovery options
- ✅ **Accessibility**: Screen reader support and keyboard navigation

### Testing Infrastructure

#### **Backend Test Suite** 
- ✅ **`test_model_visualizer.py`**: Unit tests for 3D visualization data generation
- ✅ **`test_simple_integration.py`**: Integration tests for optimizer and API connections  
- ✅ **`test_api_endpoints.py`**: End-to-end API testing with real optimization jobs
- ✅ **`test_json_download.py`**: Comprehensive JSON serialization and download workflow testing

#### **Test Coverage Results**
- ✅ **ModelVisualizer Module**: CNN, LSTM, and mixed architecture visualization tested
- ✅ **API Integration**: Endpoint functionality and data flow validation complete
- ✅ **JSON Serialization**: Download functionality and file integrity verification complete
- ✅ **Architecture Support**: Layer positioning, parameter calculation, and color coding validated

### Bug Fixes and Optimizations

#### **Performance Optimizations**
- ✅ **Accuracy Gap Resolution**: Eliminated 6% performance discrepancy to <0.5%
- ✅ **Multi-worker Scaling**: Achieved 5.5x speedup with 6 concurrent workers
- ✅ **Multi-GPU Integration**: Implemented 3.07x speedup with TensorFlow MirroredStrategy
- ✅ **Memory Optimization**: Efficient handling of large model architectures

#### **Critical Bug Fixes**
- ✅ **Health Metrics Calculation**: Fixed convergence and consistency metrics showing 50% fallback values
- ✅ **Parameter Efficiency**: Corrected negative efficiency calculations for small models
- ✅ **Keras Metric Naming**: Fixed compatibility with 'categorical_accuracy' vs 'accuracy' naming conventions
- ✅ **JSON Serialization**: Resolved LayerVisualization object serialization for downloads
- ✅ **Thread Safety**: Eliminated race conditions in concurrent optimization execution

### Cytoscape.js + TensorBoard Educational Visualization System ✅ **COMPLETED**

#### **Phase 2A: Educational Visualization Implementation** 
**Backend Progress:**
- ✅ **ModelVisualizer Module**: Complete architecture data preparation for CNN/LSTM architectures with Cytoscape.js conversion
- ✅ **Optimizer Integration**: `get_best_model_visualization_data()` method implemented
- ✅ **API Endpoints**: `/jobs/{job_id}/best-model` and `/jobs/{job_id}/best-model/download` endpoints
- ✅ **JSON Serialization**: Full pipeline for frontend consumption tested and working
- ✅ **Performance Integration**: Color coding based on health metrics and performance scores
- ✅ **Architecture Support**: CNN, LSTM, and mixed architectures with layer positioning
- ✅ **Cytoscape.js Data Format**: Convert existing layer data to Cytoscape nodes/edges format
- ✅ **TensorBoard Integration**: Add `tf.keras.callbacks.TensorBoard` to training pipeline  
- ✅ **Architecture JSON Export**: Generate Cytoscape-compatible architecture JSON per trial
- ✅ **TensorBoard Server Setup**: Integrate TensorBoard server with FastAPI backend

#### **Phase 2B: Model & Visualization Download System** 
**Backend Implementation:**
- ✅ **JSON Download API**: `/jobs/{job_id}/best-model/download` endpoint implemented
- ✅ **Data Serialization**: Complete visualization data with metadata in downloadable JSON format
- ✅ **File Generation**: Proper content-type and attachment headers for browser downloads
- ✅ **Model Download API**: `/download/{job_id}` endpoint for optimized .keras model download
- ✅ **Final Model Building**: Automatic rebuilding of best model with optimized hyperparameters after optimization completes
- ✅ **Plot Generation & Serving**: Comprehensive training plots automatically generated and served via API endpoints
- ✅ **Testing**: Comprehensive testing of download functionality and file integrity

**Frontend Implementation:**
- ✅ **Smart Download Button**: Integrated next to optimization controls, activates when final model is available
- ✅ **Model Availability Detection**: Automatic checking via API for when optimized model is ready for download
- ✅ **Training Plot Visualization**: Embedded plot system showing training progress, gradient flow, and model health metrics
- ✅ **TensorBoard Integration**: Full TensorBoard access for deep analysis with embedded plot previews for immediate insights
- ✅ **Plot Download Capability**: Individual plot downloads via API endpoints for training history, gradient analysis, and weight distributions
- ✅ **User Experience**: Seamless workflow from optimization completion to model download with clear availability indicators

---

## IV. Detailed Implementation Roadmap

### **Phase 1: Reversion to Local Orchestration with GPU Plot Generation** ✅ **COMPLETED**
**Status**: Successfully implemented - major architectural issues resolved

**Objective:**
Fix critical architectural problems in current RunPod integration by reverting to proven local Optuna orchestration while maintaining GPU-accelerated plot generation capabilities. This addresses multi-worker coordination failures, debugging difficulties, and resource utilization inefficiencies in the current "everything-on-RunPod" approach.

#### **Current vs Target Architecture Analysis**

**Current Architecture (Problematic):**
```
🌐 User Request → 💻 api_server.py → ☁️ Single RunPod Worker
                                    ↓
                               🔄 Complete optimization (all trials)
                               📊 Optuna study runs on worker
                               🎯 No multi-worker coordination
                               ❌ Poor debugging visibility
```

**Target Architecture (Proven + Enhanced):**
```
🌐 User Request → 💻 api_server.py → 💻 Local Optuna Study
                                    ↓
                               🔄 Individual trial generation
                               ↓
                    ☁️ RunPod Worker 1    ☁️ RunPod Worker 2    ☁️ RunPod Worker N
                    🎯 Single trial        🎯 Single trial        🎯 Single trial
                    📊 Plot generation     📊 Plot generation     📊 Plot generation
                    📤 S3 upload          📤 S3 upload          📤 S3 upload
                               ↓                 ↓                 ↓
                    💻 Local aggregation ← 📥 S3 download ← 🔄 Result coordination
```

#### **Critical Problems Being Solved**

**1. Multi-Worker Coordination Failure**
- **Current Issue**: Each RunPod worker runs isolated optimization, no coordination
- **Solution**: Local Optuna study coordinates trial distribution across workers
- **Evidence**: RunPod logs show `trial_number: 0` for all trials (no proper sequencing)

**2. Debugging and Diagnostic Challenges**
- **Current Issue**: All optimization logic hidden on remote workers
- **Solution**: Local logs show Optuna decisions, trial parameters, progress aggregation
- **Benefit**: Easy identification of hyperparameter trends, convergence issues

**3. Resource Utilization Inefficiency**
- **Current Issue**: Pay for GPU time during Optuna overhead and coordination delays
- **Solution**: Workers only run during actual training + plot generation
- **Savings**: Eliminate idle GPU time during study management

**4. Plot Generation Performance**
- **Current Issue**: Local plot generation causes significant slowdowns
- **Solution**: GPU-accelerated plots on RunPod with S3 transfer
- **Benefit**: Fast local orchestration + fast remote plot generation

#### **Detailed Implementation Plan**

**Stage 1: Local Optuna Orchestration Restoration** ✅ **COMPLETED**

*Step 1.1: Modify api_server.py routing logic* ✅ **COMPLETED**
```python
# IMPLEMENTED: Unified routing for both execution modes
# UNIFIED ROUTING: Always use local Optuna orchestration
# RunPod workers handle individual trials when use_runpod_service=True
if self.request.use_runpod_service and self.request.runpod_service_endpoint:
    logger.info(f"🚀 Using LOCAL orchestration with RunPod workers: {self.request.runpod_service_endpoint}")
    logger.info(f"📊 Optuna study will run locally, individual trials dispatched to RunPod workers")
else:
    logger.info(f"🏠 Using LOCAL orchestration with local execution")
```

*Step 1.2: Restore optimizer.py for individual trial dispatch* ✅ **COMPLETED**
- ✅ Reverted `optimize_model()` to local execution with `use_runpod_service=True`
- ✅ Modified `_objective_function()` to send individual trials to RunPod
- ✅ Updated progress aggregation to handle RunPod worker responses
- ✅ Implemented concurrent trial dispatch for multi-worker coordination

*Step 1.3: Update RunPod handler for single-trial execution* ✅ **COMPLETED**
- ✅ Modified `handler.py` to expect single trial parameters (not complete jobs)
- ✅ Ensured `optimize_model(..., trials=1, ...)` matches actual request
- ✅ Added trial-specific result formatting for local aggregation

**Stage 2: GPU Plot Generation with S3 Transfer** ✅ **COMPLETED**

*Step 2.1: Enable plot generation on RunPod workers* ✅ **COMPLETED**
```python
# IMPLEMENTED: RunPod handler creates plots and uploads to S3
# Plot generation infrastructure already exists and works correctly
plots_s3_info = {
    "bucket": "40ub9vhaa7",
    "s3_prefix": f"optimization_results/{run_name}/plots/trial_{trial_number}",
    "success": True
}
```

*Step 2.2: Implement S3 plot download in local optimizer* ✅ **COMPLETED**
```python
# IMPLEMENTED: S3 download with timestamp synchronization fix
request_payload = {
    "input": {
        "command": "start_training",
        "trial_id": f"trial_{trial.number}",
        "dataset_name": self.dataset_name,
        "run_name": self.run_name,  # Critical fix for timestamp consistency
        "hyperparameters": params,
        # ... rest of payload
    }
}
```

*Step 2.3: Update plot generation configuration* ✅ **COMPLETED**
- ✅ Plot generation enabled on RunPod workers by default
- ✅ Implemented S3 timestamp synchronization for consistent directory structure
- ✅ Maintained backward compatibility for local-only execution mode

**Stage 3: Multi-Worker Coordination Enhancement** (Priority: Medium - 2-3 days)

*Step 3.1: Implement concurrent trial dispatch*
```python
# Concurrent worker management
async def dispatch_trials_concurrently(trials_params, max_workers=4):
    tasks = [send_trial_to_runpod(trial) for trial in trials_params]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return aggregate_trial_results(results)
```

*Step 3.2: Add worker health monitoring*
- Track worker response times and failure rates
- Implement automatic retry logic for failed trials
- Add worker load balancing for optimal resource utilization

*Step 3.3: Enhanced progress tracking*
- Real-time progress aggregation across multiple workers
- Epoch-level progress updates from individual workers
- Unified progress display in UI showing all concurrent trials

**Stage 4: Configuration and Backward Compatibility** (Priority: Medium - 1-2 days)

*Step 4.1: Update optimization configuration*
```python
@dataclass
class OptimizationConfig:
    use_runpod_service: bool = False           # Local vs RunPod execution
    runpod_generate_plots: bool = True         # Plot generation location
    concurrent_runpod_workers: int = 2         # Multi-worker coordination
    runpod_plot_s3_transfer: bool = True       # S3 plot transfer
```

*Step 4.2: Maintain execution mode flexibility*
- `use_runpod_service=False`: Complete local execution (existing)
- `use_runpod_service=True`: Local orchestration + RunPod workers (new)
- Ensure UI configuration supports both modes seamlessly

*Step 4.3: Add comprehensive logging*
- Local Optuna study decisions and trial parameters
- RunPod worker dispatch and response tracking
- S3 plot transfer success/failure monitoring
- Multi-worker coordination and load balancing

#### **Testing and Validation Strategy**

**Unit Testing:**
- Individual trial dispatch to RunPod workers
- S3 plot upload/download functionality
- Local Optuna study state management
- Progress aggregation accuracy

**Integration Testing:**
- Multi-worker concurrent execution (2-6 workers)
- Plot generation and S3 transfer pipeline
- UI real-time progress updates during concurrent trials
- Fallback to local execution when RunPod unavailable

**Performance Validation:**
- Measure speedup with concurrent workers vs current sequential approach
- Validate plot generation performance: GPU vs local timing
- Monitor S3 transfer bandwidth and success rates
- Confirm cost optimization: GPU utilization vs idle time

#### **Success Criteria**

**Functional Requirements:**
- ✅ **COMPLETED**: Local Optuna study coordinates multiple RunPod workers successfully
- ✅ **COMPLETED**: Individual trials execute concurrently on separate workers
- ✅ **COMPLETED**: Plot generation occurs on GPU with automatic S3 transfer
- ✅ **COMPLETED**: Real-time progress tracking across all concurrent workers
- ✅ **COMPLETED**: Backward compatibility maintained for local-only execution

**Performance Requirements:**
- ✅ **VERIFIED**: 2-3x speedup with concurrent workers (2 workers tested)
- ✅ **VERIFIED**: Plot generation on GPU (infrastructure confirmed working)
- ✅ **VERIFIED**: S3 timestamp synchronization fixed for consistent paths
- ✅ **VERIFIED**: GPU utilization during training phases only
- ✅ **VERIFIED**: Real-time UI updates with epoch-level progress

**Debugging and Maintenance:**
- ✅ **COMPLETED**: Clear local logs showing Optuna decisions and trial parameters
- ✅ **COMPLETED**: Detailed worker dispatch and response tracking
- ✅ **COMPLETED**: S3 transfer path monitoring and timestamp coordination
- ✅ **COMPLETED**: Easy identification of worker status and progress

## **Comprehensive Testing Plan for Implemented Features**

### **Test Suite 1: Basic Functionality Verification**

#### **Test 1.1: Local Execution Mode (Baseline)**
**Purpose**: Verify local-only execution still works correctly
**Steps**:
1. Start API server: `python src/api_server.py`
2. Modify `test_curl.sh` to set `"use_runpod_service": false`
3. Run test: `./test_curl.sh`
4. Monitor job status: `curl -X GET "http://localhost:8000/jobs/{job_id}"`
5. Verify completion and results directory creation

**Expected Results**:
- Job completes successfully with `use_runpod_service=false`
- Local directory created in `optimization_results/`
- Plots generated locally in `plots/` subdirectory
- Model saved in `optimized_model/` subdirectory

#### **Test 1.2: RunPod Integration Mode**
**Purpose**: Verify local orchestration with RunPod workers
**Steps**:
1. Ensure `test_curl.sh` has `"use_runpod_service": true`
2. Run test: `./test_curl.sh`
3. Monitor logs: `tail -f api_server.log | grep -E "(RUNPOD|DEBUG PAYLOAD|S3)"`
4. Check job progress: `curl -X GET "http://localhost:8000/jobs/{job_id}"`
5. Verify final results

**Expected Results**:
- Local Optuna orchestration logs visible
- Individual trial dispatch to RunPod workers
- RunPod payload includes coordinated `run_name`
- Timestamp synchronization works correctly
- Real-time progress updates from GPU workers

### **Test Suite 2: Architecture Verification**

#### **Test 2.1: Local Orchestration Confirmation**
**Purpose**: Confirm Optuna runs locally, not on RunPod
**Verification Steps**:
1. Start optimization with RunPod enabled
2. Check local logs for Optuna study creation
3. Verify trial parameters generated locally
4. Confirm only individual trials sent to RunPod

**Log Patterns to Verify**:
```bash
# Look for these patterns in api_server.log:
grep "🚀 Using LOCAL orchestration with RunPod workers" api_server.log
grep "📊 Optuna study will run locally" api_server.log
grep "🔍 SELF.RUN_NAME VALUE:" api_server.log
grep "DEBUG PAYLOAD" api_server.log
```

#### **Test 2.2: Individual Trial Dispatch Verification**
**Purpose**: Confirm trials are sent individually to workers
**Verification Steps**:
1. Run optimization with `trials=3`
2. Monitor RunPod submissions in logs
3. Verify each trial gets separate RunPod job
4. Check trial numbering sequence

**Expected Behavior**:
- 3 separate RunPod API calls
- Each payload contains single trial parameters
- Sequential trial numbers (0, 1, 2)
- Concurrent execution visible in progress updates

### **Test Suite 3: S3 Integration and Timestamp Synchronization**

#### **Test 3.1: Timestamp Coordination Test**
**Purpose**: Verify local and RunPod use same timestamps
**Steps**:
1. Run optimization and capture start time
2. Monitor S3 paths in logs: `grep "S3.*optimization_results" api_server.log`
3. Check local directory creation
4. Compare timestamps between local and S3 paths

**Verification Commands**:
```bash
# Check S3 paths in logs
grep "📂 FROM: s3://.*optimization_results" api_server.log

# Check local directory timestamps
ls -la optimization_results/ | tail -5

# Verify they match the coordinated run_name pattern
grep "🔍 SELF.RUN_NAME VALUE:" api_server.log
```

#### **Test 3.2: S3 Path Structure Verification**
**Purpose**: Confirm consistent directory structure
**Expected S3 Paths**:
```
s3://bucket/optimization_results/{run_name}/plots/trial_0/
s3://bucket/optimization_results/{run_name}/plots/trial_1/
s3://bucket/optimization_results/{run_name}/optimized_model/
```

**Local Directory Structure**:
```
optimization_results/{run_name}/
├── plots/
│   ├── trial_0/
│   ├── trial_1/
│   └── trial_2/
└── optimized_model/
```

### **Test Suite 4: Progress Tracking and Real-time Updates**

#### **Test 4.1: Real-time Progress Monitoring**
**Purpose**: Verify epoch-level progress updates work
**Steps**:
1. Start optimization with longer epochs (modify test to use more epochs)
2. Monitor progress API: `watch -n 2 "curl -s http://localhost:8000/jobs/{job_id} | jq '.progress'"`
3. Verify epoch progress updates in real-time

**Expected Progress Fields**:
```json
{
  "current_trial": 1,
  "total_trials": 2,
  "completed_trials": 0,
  "is_gpu_mode": true,
  "current_epoch": 3,
  "total_epochs": 12,
  "epoch_progress": 0.25,
  "status_message": "Trial 1/2 - 0 completed, 2 running, 0 failed"
}
```

#### **Test 4.2: Multi-Worker Coordination Test**
**Purpose**: Verify concurrent worker management
**Test Scenario**:
1. Run optimization with `trials=4` to trigger multiple workers
2. Monitor concurrent execution in logs
3. Verify progress aggregation across workers
4. Check for proper trial completion sequencing

### **Test Suite 5: Error Handling and Edge Cases**

#### **Test 5.1: S3 Download Failure Handling**
**Purpose**: Test graceful handling when S3 objects don't exist
**Steps**:
1. Run optimization and let it complete
2. Check logs for S3 download attempts
3. Verify graceful handling of missing S3 objects
4. Ensure optimization still completes successfully

**Expected Log Patterns**:
```bash
grep "No objects found with prefix.*optimization_results" api_server.log
grep "S3 DOWNLOAD FAILED" api_server.log
grep "Check S3 credentials, connectivity" api_server.log
```

#### **Test 5.2: RunPod Worker Failure Handling**
**Purpose**: Test resilience to worker failures
**Manual Test**: Temporarily break RunPod endpoint in configuration
**Expected**: Graceful error handling and informative error messages

### **Test Suite 6: Performance and Resource Utilization**

#### **Test 6.1: GPU Utilization Verification**
**Purpose**: Confirm GPU is only used during training
**Metrics to Monitor**:
- RunPod job duration vs total optimization time
- GPU billing time vs idle time
- Training efficiency on GPU workers

#### **Test 6.2: Concurrent Worker Performance**
**Purpose**: Measure speedup with multiple workers
**Test Plan**:
1. Run baseline: 6 trials sequential (local execution)
2. Run comparison: 6 trials on 2-3 RunPod workers
3. Measure total optimization time
4. Calculate speedup ratio

### **Automated Test Script**

Create `test_complete_integration.sh`:
```bash
#!/bin/bash
echo "=== Phase 1 & 2 Implementation Testing ==="

# Test 1: Local execution baseline
echo "Testing local execution..."
sed -i 's/"use_runpod_service": true/"use_runpod_service": false/' test_curl.sh
./test_curl.sh
echo "Local test job started"

# Test 2: RunPod integration
echo "Testing RunPod integration..."
sed -i 's/"use_runpod_service": false/"use_runpod_service": true/' test_curl.sh
./test_curl.sh
JOB_ID=$(curl -s -X POST "http://localhost:8000/optimize" \
  -H "Content-Type: application/json" \
  -d @test_curl.sh | jq -r '.job_id')

echo "RunPod test job: $JOB_ID"

# Monitor progress
echo "Monitoring progress (will show updates for 60 seconds)..."
for i in {1..30}; do
  STATUS=$(curl -s "http://localhost:8000/jobs/$JOB_ID" | jq -r '.status')
  PROGRESS=$(curl -s "http://localhost:8000/jobs/$JOB_ID" | jq '.progress.current_epoch // 0')
  echo "[$i/30] Status: $STATUS, Current Epoch: $PROGRESS"
  sleep 2
  if [ "$STATUS" = "completed" ]; then
    echo "✅ Job completed successfully!"
    break
  fi
done

# Verify results
echo "Checking results directory..."
LATEST_DIR=$(ls -t optimization_results/ | head -1)
echo "Latest results: optimization_results/$LATEST_DIR"
ls -la "optimization_results/$LATEST_DIR/"

echo "=== Testing Complete ==="
```

### **Manual Verification Checklist**

**Architecture Verification**:
- [ ] Local Optuna study logs visible during RunPod execution
- [ ] Individual trial dispatch (not complete job dispatch)
- [ ] Coordinated timestamp usage between local and RunPod
- [ ] Real-time progress aggregation from multiple workers

**S3 Integration Verification**:
- [ ] S3 paths use coordinated `run_name` timestamps
- [ ] Local directories match S3 path structure
- [ ] Graceful handling of S3 download failures
- [ ] Plot generation infrastructure working on RunPod

**Performance Verification**:
- [ ] GPU utilization only during training phases
- [ ] Concurrent worker execution
- [ ] Faster completion with multiple workers
- [ ] Real-time UI progress updates

#### **Risk Mitigation**

**Technical Risks:**
- **S3 Transfer Failures**: Implement retry logic and fallback to local plot generation
- **Worker Coordination Issues**: Add timeout handling and failed trial recovery
- **Progress Tracking Complexity**: Use proven concurrent progress aggregation patterns

**Implementation Risks:**
- **Breaking Existing Functionality**: Maintain strict backward compatibility testing
- **Performance Regression**: Validate speedup metrics at each implementation stage
- **UI Integration Issues**: Test real-time progress updates throughout development

#### **Key Deliverables**

**Week 1: Core Architecture**
- 🔄 **Local Optuna Restoration**: api_server.py and optimizer.py modifications
- 🔄 **Single Trial Handler**: RunPod handler.py updates for individual trial execution
- 🔄 **Basic Multi-Worker**: Concurrent trial dispatch implementation

**Week 2: GPU Plot Integration**
- 🔄 **Plot Generation on RunPod**: Enable GPU-accelerated plot creation
- 🔄 **S3 Plot Transfer**: Upload/download pipeline for trial plots
- 🔄 **UI Plot Integration**: Display downloaded plots in existing UI

**Week 3: Enhancement and Testing**
- 🔄 **Multi-Worker Optimization**: Load balancing and health monitoring
- 🔄 **Configuration Updates**: Flexible execution mode parameters
- 🔄 **Comprehensive Testing**: Unit, integration, and performance validation

This implementation restores the proven local orchestration architecture while adding GPU plot generation capabilities, directly addressing the critical coordination and debugging issues identified in the current RunPod integration.

---

### **Phase 2: Miscellaneous UI Improvements** 🔧
**Status**: Underway

**Key Deliverables:**
- ✅ Show both forward pass and backward pass in the model archatecture visualization animation.
- ✅ Add "output" as an item shown in the model archatecture visualization. Additionally, show "flattening layer" as an item in the model archatcture visualization if flattening is indeed used in that model. Be sure to preserve the correct tensor labels for each edge in the graph.
- ✅ Add the "use_flattening: <true/false>" field following the "use_global_pooling: <true/false>" field in the "best_hyperparameters.yaml" file created at the end of each optimization.
- ✅ Compare the fields listed in "best_hyperparameters.yaml" to those listed in the "Architecture layers" and "Performance & Health" sections of the UI and add any missing params to those sections of the UI.
- ✅ Add activation_progression (already being created and saved to disk in the same directory as the other plots) to the plots available for view and for download in the "Training Metrics & Diagnostics" section of the UI. This will involve adding an additional tab (titled "Activation Progression") to the tabs already present. 
- ✅ Enable download each model archatecture visualization to disk as a .png
- ✅ Add an additional tab to the "Training Metrics & Diagnostics" section called "Model Archatecture" that, when clicked, displays the same model archatecture visualization currently provided in the "Model Architecture" section, including the legend, animate button, and download button, effectively duplicating the model visualization contained in the "Model Architecture" section of the UI.
- ✅ Eliminate the "Model Architecture" section of the UI, since the model archatecture is visible via that tab within the "Training Metrics & Diagnostics" section of the UI.
- ✅ The "Download" button for the plots should use the "Save as" mechanism that propmpts the user to decide where to save the downloaded file, as opposed to the current behavior of automatically saving to the dowloads folder.
- ✅ Remove persistant "Unable to add filesystem: <illegal path>" error in the JS console
- ✅ Vastly improved responsiveness of the UI, so as to render better on mobile.
- ✅ Improve rendering shown when the "View Details" tab is clicked on a tile in the gallery. Current behavior: When that "View Details" button is clicked, it triggers a popup (correct behavior), but that popup has the following incorrect behaviors: (a) Clicking the close icon ("x") in the popup does not close the popup, as intended (b) the "View Details" button for a given trial's tile in the gallery section should be disabled until that trial is complete (c) Currently, the entire trial tial in the gallery triggers a popup. The popup should instead be triggered by clicking the "View Details" button only (d) Currently, the "View Details" button floats below the preceeding text. This should instead be fixed to the bottom inside of the tial (e) Currently, the bottom edge of the tials overlaps with the border of the broader container containing those tials. Add some padding to remove this overlap (f) the popup does not have its own vertical scroll bar, resulting in the bottom section of that popup being cut off by the bottom of the screen (g) in the popup that appears after clicking the 
- ✅ Currently, at the end of each trial, there is a delay as the plots for the trial are completed. That currently shows in the UI as the progress bar for the last epoch being complete, but yet nothing seems to be happening. Create a visualization that indicates to the user that the plot creation is in progress, ideally showing the progress of that plot creation (e.g. via a status bar similar to what is already in place to communicate epoch progress).
- ✅ Currently, after the final trial is complete, there is a delay during which time the best model is being built. That currently shows in the UI via the status bar for the last epoch in the last trial showing 100% completion, without any communication as to the reason for the delay caused by the final model creation. Create a visualization that indicates to the user that the final model creation is in progress, ideally showing the progress of that final model creation (e.g. via a status bar similar to what is already in place to communicate epoch progress).
- ✅ Currently, the "model health vs. accuracy" tooltip does not render properly on mobile- its left half is cut off by the left edge of the screen. Update that tooltip to be more mobile-friendly.
- ✅ Currently, the "model health vs. accuracy" tooltip background is transparent on mobile (but is solid on large screens). This should not happen.
- ✅ Currently, the "model health vs. accuracy" tooltip is located after the drop down in which the user selects the objective (e.g. simple vs. health-aware). Instead, place it before that drop-down, replacing the current target icon.
- ✅ Create a new tooltip for the dropdown in which the user selects the dataet (MNIST, etc.) to be used for hyperparameter tuning. This tooltip should contain the following:
"To learn more about each dataset, see the links below: (items below a bulleted list)
  - MNIST<https://keras.io/api/datasets/mnist/>: Handwritten digits 0-9
  - CIFAR-10 <https://keras.io/api/datasets/cifar10/>: Color images
  - CIFAR-100<https://keras.io/api/datasets/cifar100/>: Color images
  - Fashion-MNIST<https://keras.io/api/datasets/fashion_mnist/>: Greyscale images
  - GTSRB<https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>: Color images
  - IMDB<https://keras.io/api/datasets/imdb/>: Text-based classification
  - Reuters<https://keras.io/api/datasets/reuters/>:  Text-based classification"
- ✅ Create a new tooltip for the "Best Total Score" tial that explains how that best total score is calculated. If the mode is accuracy, then the tooltip should just explain it is pure catagorical accuracy and include a link to tensorflow's documentation re categorical accuracy (https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy). If the model is health-aware, it should show the weights placed on accuracy + health score, including the sub-values for each element of the health score. These values should be dynamic such that if the weights change on the back-end, they are automatically reflected in the tooltip. Before implementing, suggest links (ideally from tensorflow or keras) that explain this topic that I can include as a link. 
- ✅ Currently, the following items from the navbar link to other, non-existant pages: "Dashboard", "Archatcture Explorer", "Trial Comparison", "Performance Analysis", "System Health" and the "Settings" icon.  Remove these items.
- ✅ Currently, the "Onegin: A Neural Archatecture Explorer" text in the navbar disappears on mobile. It should persist on all screen sizes.
- ✅ Currently, the "Onegin: A Neural Archatecture Explorer" text in the navbar is aligned to the left side of the screen without any left padding or margin. Add left padding so that the "Onegin: A Neural Archatecture Explorer" is horizonally aligned with the borders for the elements below it, such as the box that contains the drop-downs for dataset and accuracy/health.
- ✅ Use the Keytech logo in the navbar and as the favicon
- ✅ Implement a footer with the following elements: (a) Link to personal page, KeyTech page, MIT license page, and GitHub repo


---

### **Phase 3: RunPod S3 Storage Integration** ✅ **COMPLETED**
**Status**: Successfully completed with automatic model transfer

**Objectives:**
- ✅ Enable automatic transfer of final models from RunPod GPU workers to local machine
- ✅ Implement S3-compatible storage integration with RunPod Network Volumes  
- ✅ Simplify architecture by eliminating need for train_locally/build_model_locally configuration flags
- ✅ Ensure seamless model availability regardless of where training occurred

**Implementation Completed:**
The RunPod S3 storage integration has been successfully implemented, providing automatic model transfer:
- ✅ **S3 Upload in RunPod Handler**: Models automatically uploaded to RunPod S3 storage after training
- ✅ **S3 Download in Optimizer**: Models automatically downloaded from S3 to local optimization_results directory
- ✅ **Credential Management**: Secure S3 authentication via RunPod Network Volume credentials
- ✅ **Storage Volume Connection**: RunPod serverless instances connected to persistent storage volumes
- ✅ **Automatic Cleanup**: S3 files cleaned up after successful download to save storage space

**Architecture Achievement:**
With S3 storage working, the complex train_locally/build_model_locally flag system is now deprecated:
- **Old Approach**: Required separate code paths and configuration flags to control execution location
- **New Approach**: Unified execution with automatic S3 transfer - models trained anywhere become available locally
- **Benefits**: Simplified codebase, automatic model availability, no configuration complexity

**Remaining Work for Future Phases:**

While the S3 storage integration has simplified model transfer, there are still opportunities for future optimization:

**Potential Future Enhancement - Plot Transfer via S3:**
- **Current Limitation**: Trial plots are still generated locally, requiring local model building for each trial
- **Future Opportunity**: Extend S3 integration to transfer plots from RunPod workers
- **Benefit**: Could enable ALL plot mode while keeping models trained entirely on RunPod
- **Implementation**: Upload trial plots to S3 alongside models, download for UI display 

---

### **Phase 4: Plot Transfer via S3 Extension** 🔧
**Status**: Ready for implementation

**Objectives:**
- Extend S3 storage integration to include trial plots for complete remote execution capability
- Enable ALL plot generation mode while maintaining full GPU acceleration
- Implement plot upload/download mechanism similar to model transfer
- Ensure local operation remains unaffected by S3 plot transfer

**Current State Analysis:**
With S3 model transfer working successfully, the system currently handles:
- ✅ **Final Model Transfer**: Models automatically uploaded/downloaded via S3
- ✅ **RunPod Storage Integration**: Credentials and bucket access working perfectly
- ✅ **Automatic Cleanup**: S3 files cleaned up after successful downloads
- 🔄 **Plot Transfer**: Trial plots still require local model building for generation

**Implementation Plan:**

**Stage 1: Plot Upload Integration in RunPod Handler**
- Extend `handler.py` to generate and upload trial plots to S3 after each trial
- Use existing S3 upload mechanism with plot-specific S3 prefix
- Include all plot types: training_history, gradient_analysis, weight_distributions, activation_progression
- Maintain existing plot generation quality and format

**Stage 2: Plot Download Integration in Optimizer**
- Extend `optimizer.py` S3 download functionality to retrieve trial plots
- Download plots to appropriate trial directories in `optimization_results/`
- Ensure plot availability for UI visualization system
- Implement plot-specific cleanup after successful download

**Stage 3: Conditional Plot Generation Logic**
- Add `generate_plots_on_runpod` configuration parameter
- Enable local model building bypass when plots available via S3
- Maintain backward compatibility with existing local plot generation
- Implement fallback to local generation if S3 plots unavailable

**Testing Strategy:**
- **S3 Plot Upload**: Verify plots correctly uploaded to RunPod S3 storage
- **S3 Plot Download**: Confirm plots downloaded to correct local directories
- **UI Integration**: Test plot visualization with S3-transferred plots
- **Local Fallback**: Ensure local operation unaffected when S3 unavailable

**Benefits:**
- **Full GPU Acceleration**: Enable ALL plot mode with complete RunPod execution
- **Reduced Local Overhead**: Eliminate need for local model building during trials
- **Performance Gain**: Faster trial completion by removing local model building bottleneck
- **Unified Architecture**: Complete S3-based transfer system for all artifacts

**Key Deliverables:**
- 🔄 **Plot Upload**: Extend RunPod handler to generate and upload trial plots to S3
- 🔄 **Plot Download**: Extend optimizer to download and organize plots locally  
- 🔄 **Configuration Options**: Add plot generation location control parameters
- 🔄 **Testing**: Comprehensive validation of plot transfer functionality
- 🔄 **Documentation**: Updated configuration and usage documentation

---

### **Phase 5: System Polish & Testing** 🔧
**Status**: Ready for testing validation

**Objectives:**
- Ensure robust operation with RunPod GPU acceleration and S3 storage
- Comprehensive testing of S3 transfer system under various conditions
- System stability validation and performance benchmarking
- User experience optimization for cloud-based workflows

**Key Areas for Testing:**
- **S3 Integration Robustness**: Test S3 transfer with network interruptions, credential changes
- **Final Model GPU Training**: Validate GPU-based final model training with progress updates
- **UI State Management**: Ensure proper UI state updates during job cancellation and completion
- **Error Handling**: Test graceful degradation when cloud services unavailable
- **Performance Benchmarking**: Measure end-to-end optimization speed improvements

**Key Deliverables:**
- ✅ Fix trial numbering bug (first completed = trial_0, second = trial_1, etc.)
- ✅ Ensure multiple GPU usage via concurrent_workers configuration
- ✅ Migrate final model training to GPU with proper status updates
- 🔄 Implement proper job cancellation with RunPod cleanup
- 🔄 Fix UI polling continuation after job cancellation
- 🔄 Add "Job cancelled" status display when optimization stopped
- 🔄 Add UI checkbox for RunPod GPU resource toggling
- 🔄 Comprehensive logging for UI-triggered optimizations

---


### **Phase 6: Deployment & Container Integration** 🚀
**Status**: Production readiness

**Objectives:**
- Optimize container deployment configurations
- Production environment setup and testing
- Performance tuning for production workloads
- Documentation for deployment procedures

**Key Deliverables:**
- Production-ready container configurations
- Deployment automation scripts
- Performance benchmarking in production environment
- Deployment documentation and maintenance guides

---

### **Phase 7: Website Integration** 🌐
**Status**: Not started

**Objectives:**
- Integration with personal portfolio (mattmcdonnell.net)
- Integration with company website (kebayorantechnologies.com)
- Showcase optimization capabilities and results
- Professional presentation of project achievements

**Key Deliverables:**
  style: {
    'curve-style': 'bezier',
    'control-point-step-size': 40,
    'edge-distance': 'intersection',  // Route to node edges
    'text-margin-x': 10,             // Offset labels from edges
    'text-margin-y': -15,            // Position above edges
    'text-background-opacity': 0.9,   // Better label visibility
    'text-border-width': 1
  }
}
```

#### **Phase 1.3: Responsive Layout System** (1-2 hours)
- Container-aware spacing that adapts to screen size
- Layout re-computation on viewport changes
- Configuration constants for maintainable spacing parameters

#### **Phase 1.4: ELK Layout Alternative** (Backup Solution - 2 hours)
For complex models requiring sophisticated edge routing:
```javascript
const elkLayout = {
  name: 'elk',
  elk: {
    'algorithm': 'layered',
    'direction': 'RIGHT',
    'spacing.nodeNodeBetweenLayers': 120,
    'spacing.nodeNode': 60,
    'layered.edgeRouting.strategy': 'ORTHOGONAL'
  }
};
```
