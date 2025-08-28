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

## II. Configuration Flow Architecture 🏗️

### **Understanding Configuration Data Flow**

This project uses a dual-path configuration architecture to support both API-driven (UI) and programmatic usage patterns:

#### **Path 1: API-Driven Flow (Web UI)**
```
User Input (Web UI)
       ↓
OptimizationRequest (api_server.py)
  • API validation layer
  • User-friendly field names
  • HTTP request parsing
  • User-controlled defaults
       ↓
create_optimization_config()
  • Conversion function
  • Type transformations (string → enum)
  • Pass-through all user values
       ↓
OptimizationConfig (optimizer.py)
  • Business logic configuration
  • Fail-fast validation
  • System-controlled defaults
  • Enum types for internal use
       ↓
ModelOptimizer → ModelBuilder → Training Pipeline
```

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
ModelOptimizer → ModelBuilder → Training Pipeline
```

### **Key Architecture Principles**

**Variable Ownership:**
- **OptimizationRequest**: Owns user-controlled defaults (trials=50, batch_size=32, etc.)
- **OptimizationConfig**: Owns system-controlled defaults (timeout_hours=None, health_monitoring_frequency=1, etc.)

**Data Flow Rules:**
1. **API Path**: User → OptimizationRequest → create_optimization_config() → OptimizationConfig
2. **Programmatic Path**: Developer → OptimizationConfig directly
3. **No Conflicting Defaults**: Each variable has defaults in only ONE class
4. **Fail-Fast**: OptimizationConfig validates all required values immediately

**Benefits:**
- ✅ **Clear Separation**: API concerns vs business logic
- ✅ **No Duplication**: Single source of truth for each variable type  
- ✅ **Type Safety**: String validation in API, enum validation in business logic
- ✅ **Flexibility**: Supports both UI and programmatic usage patterns
- ✅ **Maintainability**: Easy to understand which class controls which variables

---

## III. Key Functionality and Features

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

## IV. Completed Implementation

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

## VI. Detailed Implementation Roadmap

### **Phase 1: Advanced Model Export & Educational Features** 💾
**Status**: After Cytoscape.js + TensorBoard visualization completion

**Enhanced Objectives with Modern Educational Stack:**
- ✅ **Download best performing model in .keras format for deployment**

- **TensorBoard Log Export**: Complete training logs and metrics for external analysis
- Export complete model architecture, weights, and training configuration  
- **Interactive Prediction Interfaces**: Allow users to input data and see predictions with Cytoscape animations
- Model serialization with comprehensive metadata inclusion
- **Educational Data Flow Visualization**: Forward/backward pass animations through model architecture

**Key Deliverables:**
- ✅ **.keras file download for best model**
- **TensorBoard log directories**: Complete training logs for offline analysis and sharing
- Model metadata export (JSON format with architecture details) 
- Training configuration export for reproducibility
- **Interactive prediction demos**: Users input data → animated flow through Cytoscape architecture → prediction results
- **Educational animations**: Forward pass data flow visualization with tensor shape transformations
- Automated file naming with timestamps and performance metrics
- **Comprehensive educational package**: Architecture diagrams + training metrics + model files for complete understanding

---

### **Phase 2: Logging Consolidation & System Polish** 🔧
**Status**: Critical system improvement

**Objectives:**
- **CRITICAL FIX**: Ensure UI-triggered optimizations write logs to `logs/non-cron.log`
- System stability and performance optimizations

**Key Deliverables:**
- Unified logging across all optimization trigger methods
- Enhanced error messages and debugging capability
- Improved system stability and reliability
- Consistent log formatting and rotation

---

### **Phase 3: Deployment & Container Integration** 🚀
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

### **Phase 4: Website Integration** 🌐
**Status**: Business integration

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

---


**Status**: After Cytoscape.js + TensorBoard visualization completion

**Enhanced Objectives with Modern Educational Stack:**
- ✅ **Download best performing model in .keras format for deployment**
- **TensorBoard Log Export**: Complete training logs and metrics for external analysis
- Export complete model architecture, weights, and training configuration  
- **Interactive Prediction Interfaces**: Allow users to input data and see predictions with Cytoscape animations
- Model serialization with comprehensive metadata inclusion
- **Educational Data Flow Visualization**: Forward/backward pass animations through model architecture

**Key Deliverables:**
- ✅ **.keras file download for best model**
- **TensorBoard log directories**: Complete training logs for offline analysis and sharing
- Model metadata export (JSON format with architecture details) 
- Training configuration export for reproducibility
- **Interactive prediction demos**: Users input data → animated flow through Cytoscape architecture → prediction results
- **Educational animations**: Forward pass data flow visualization with tensor shape transformations
- Automated file naming with timestamps and performance metrics
- **Comprehensive educational package**: Architecture diagrams + training metrics + model files for complete understanding

---

### **Phase 3: Logging Consolidation & System Polish** 🔧
**Status**: Critical system improvement

**Objectives:**
- **CRITICAL FIX**: Ensure UI-triggered optimizations write logs to `logs/non-cron.log`
- Consolidate all optimization logs into unified logging system
- Improve error handling and user feedback
- System stability and performance optimizations

**Key Deliverables:**
- Unified logging across all optimization trigger methods
- Enhanced error messages and debugging capability
- Improved system stability and reliability
- Consistent log formatting and rotation

---

### **Phase 4: Deployment & Container Integration** 🚀
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

### **Phase 5: Website Integration** 🌐
**Status**: Business integration

**Objectives:**
- Integration with personal portfolio (mattmcdonnell.net)
- Integration with company website (kebayorantechnologies.com)
- Showcase optimization capabilities and results
- Professional presentation of project achievements

**Key Deliverables:**
- Portfolio integration with live demo capabilities
- Case study documentation with performance results
- Professional presentation materials
- SEO optimization and analytics integration

---

### **Phase 6: Code Path Consolidation & Embedded Training Visualizations** 🔧📊
**Status**: **NEXT PRIORITY - CRITICAL SYSTEM IMPROVEMENT**

**Problem Statement:**
The current system has two independent code paths causing confusion and technical issues:
- **Path 1 (Intended)**: API → ModelOptimizer → Optuna trials → Proper namespaced TensorBoard logs
- **Path 2 (Interfering)**: API → create_and_train_model → Single training → Different log behavior

Additionally, TensorBoard embedding encountered significant technical hurdles (iframe CORS issues, proxy complexity, mobile compatibility). The solution is to leverage existing visualization capabilities in `src/model_visualizer.py` and `src/plot_creation/` to provide immediate, focused training insights directly in the UI.

**Strategic Objectives:**
1. **Consolidate all UI-triggered optimizations** to use the Optuna-based ModelOptimizer path (even for single trials)
2. **Replace complex TensorBoard embedding** with immediate plot-based visualizations
3. **Provide progressive enhancement**: Embedded plots for immediate insights + optional TensorBoard for deep analysis
4. **Ensure consistent logging behavior** across all optimization triggers

---

## **Implementation Plan**

### **Phase 6A: Code Path Consolidation** 🔧
**Status**: Foundation cleanup for reliable system behavior

#### **Step 6A.1: Eliminate Dual Code Paths**
**Objective**: Ensure ALL UI optimizations use ModelOptimizer (Optuna-based) path

**Implementation Details:**
```python
# Remove from api_server.py:
# - Any direct calls to create_and_train_model 
# - Standalone model training paths
# - Inconsistent logging behavior

# Ensure all optimization requests use:
self.optimizer = ModelOptimizer(
    dataset_name=self.request.dataset_name,
    optimization_config=opt_config,
    run_name=self.job_id  # Use job_id for consistency
)

# Add minimum trial configuration:
class OptimizationConfig:
    min_trials: int = 1  # Ensure at least 1 trial always runs
    max_trials: int = 20  # Default maximum
```

**Files Modified:**
- `src/api_server.py` - Remove create_and_train_model usage
- `src/optimizer.py` - Ensure single-trial mode works properly
- `src/model_builder.py` - Verify consistent timestamp usage

#### **Step 6A.2: Fix TensorBoard Log Directory Path**
**Objective**: Resolve logs being created in `src/tensorboard_logs/` instead of project root

**Implementation Details:**
```python
# In model_builder.py - Fix working directory issue
def _setup_training_callbacks_optimized(self):
    # Current (incorrect): Creates logs in src/tensorboard_logs/
    # Fixed: Creates logs in project_root/tensorboard_logs/
    
    project_root = Path(__file__).resolve().parent.parent  # Go up from src/
    log_dir = project_root / f"tensorboard_logs/{run_dir}/trial_{trial_num}"
    log_dir.mkdir(parents=True, exist_ok=True)
```

**Files Modified:**
- `src/model_builder.py` - Fix log directory resolution
- `src/api_server.py` - Update TensorBoard server to look in correct location

#### **Step 6A.3: Verification & Testing**

**Automated Testing:**
```python
# New test: test_unified_code_path.py
def test_ui_optimization_uses_model_optimizer():
    """Verify all UI optimizations go through ModelOptimizer"""
    
def test_tensorboard_logs_correct_location():
    """Verify logs created in project root, not src/"""
    
def test_single_trial_optimization():
    """Verify single trial mode works with Optuna"""
    
def test_consistent_timestamps():
    """Verify same timestamp used across logs, plots, models"""
```

**Manual Testing:**
1. **Trigger optimization via UI** → Verify goes through ModelOptimizer path
2. **Check log location** → Confirm `tensorboard_logs/` at project root (not `src/`)  
3. **Verify single trial** → Set max_trials=1, confirm proper Optuna behavior
4. **Check timestamp consistency** → Same timestamp in logs, plots, model files

**Success Criteria:**
- ✅ All UI optimizations use ModelOptimizer path
- ✅ TensorBoard logs appear in project root `tensorboard_logs/`
- ✅ Single trial mode works properly
- ✅ Consistent timestamps across all generated files
- ✅ No more empty `tensorboard_logs` directories

---

### **Phase 6B: Embedded Training Visualizations** 📊
**Status**: Replace TensorBoard embedding with immediate plot-based insights

#### **Step 6B.1: Integrate Plot Generation in ModelBuilder**
**Objective**: Generate comprehensive training plots immediately after each trial completes

**Implementation Details:**
```python
# In model_builder.py - Add after model training completes
def _generate_training_plots(self, history, model, run_timestamp):
    """Generate comprehensive training visualizations after training"""
    
    # Create plot directory
    plot_dir = Path(f"plots/{run_timestamp}_{self.dataset_config.name}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plot_results = {}
    
    # 1. Training history analysis (loss/accuracy curves, overfitting detection)
    from plot_creation.training_history import create_training_history_analysis
    plot_results['training_history'] = create_training_history_analysis(
        training_history=history.history,
        model=model,
        dataset_name=self.dataset_config.name,
        run_timestamp=run_timestamp,
        plot_dir=plot_dir
    )
    
    # 2. Gradient flow analysis (dead neurons, vanishing/exploding gradients)
    from plot_creation.gradient_flow import analyze_gradient_flow
    plot_results['gradient_analysis'] = analyze_gradient_flow(
        model=model,
        run_timestamp=run_timestamp,
        save_dir=plot_dir
    )
    
    # 3. Weight distribution analysis (learned parameter patterns)
    from plot_creation.weights_bias import analyze_weights_distribution
    plot_results['weight_analysis'] = analyze_weights_distribution(
        model=model,
        run_timestamp=run_timestamp,
        save_dir=plot_dir
    )
    
    return plot_results

# Modify train() method to call plot generation
def train(self, data, validation_split=0.2):
    # ... existing training code ...
    
    # Generate plots after training completes
    if self.training_history is not None:
        training_plots = self._generate_training_plots(
            history=self.training_history,
            model=self.model, 
            run_timestamp=self.run_timestamp
        )
        
        # Store plot paths for API retrieval
        self.training_plots = training_plots
```

**Files Modified:**
- `src/model_builder.py` - Add plot generation integration
- `src/plot_creation/training_history.py` - Ensure consistent API
- `src/plot_creation/gradient_flow.py` - Add missing functions if needed
- `src/plot_creation/weights_bias.py` - Add missing functions if needed

#### **Step 6B.2: Add Plot Serving API Endpoints**
**Objective**: Serve generated training plots via REST API

**Implementation Details:**
```python
# In api_server.py - Add new plot serving endpoints

@app.get("/jobs/{job_id}/plots/{trial_id}/{plot_type}")
async def get_trial_plot(job_id: str, trial_id: str, plot_type: str):
    """
    Serve training visualization plots for specific trial
    
    Args:
        job_id: Optimization job identifier
        trial_id: Trial number (e.g., "0", "1", "2")
        plot_type: Type of plot ("training_history", "gradient_analysis", "weight_analysis")
        
    Returns:
        PNG image file containing the requested visualization
    """
    
    try:
        # Find corresponding run directory based on job_id
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        
        # Locate plot file using job timestamp and trial number
        plot_path = self._find_plot_file(job, trial_id, plot_type)
        
        if plot_path and plot_path.exists():
            return FileResponse(
                plot_path,
                media_type='image/png',
                filename=f"{job_id}_trial_{trial_id}_{plot_type}.png"
            )
        else:
            raise HTTPException(404, f"Plot not found: {plot_type} for trial {trial_id}")
            
    except Exception as e:
        logger.error(f"Error serving plot: {e}")
        raise HTTPException(500, f"Failed to serve plot: {str(e)}")

@app.get("/jobs/{job_id}/plots/{trial_id}/insights")
async def get_trial_insights(job_id: str, trial_id: str):
    """
    Get automated training insights for specific trial
    
    Returns:
        JSON containing analysis insights, overfitting detection, performance metrics
    """
    
    try:
        # Load insights from plot generation results
        insights_data = self._load_trial_insights(job_id, trial_id)
        
        return {
            "trial_id": trial_id,
            "insights": insights_data.get('training_insights', []),
            "overfitting_detected": insights_data.get('overfitting_detected', False),
            "final_metrics": insights_data.get('final_metrics', {}),
            "health_summary": self._generate_health_summary(insights_data)
        }
        
    except Exception as e:
        logger.error(f"Error loading trial insights: {e}")
        return {"insights": ["Analysis data not available"], "error": str(e)}

# Helper methods
def _find_plot_file(self, job, trial_id, plot_type):
    """Find plot file based on job timestamp and trial"""
    # Implementation to locate plot files in organized directory structure
    
def _load_trial_insights(self, job_id, trial_id):
    """Load analysis insights from plot generation results"""
    # Implementation to retrieve insights from training analysis
    
def _generate_health_summary(self, insights_data):
    """Generate high-level health summary for UI display"""
    # Implementation to create concise health overview
```

**Files Modified:**
- `src/api_server.py` - Add plot serving endpoints
- `src/api_server.py` - Add helper methods for plot file management

#### **Step 6B.3: Update Frontend with Embedded Plot Display**
**Objective**: Replace TensorBoard iframe with immediate plot visualization

**Implementation Details:**
```jsx
// Update web-ui/src/components/visualization/tensorboard-panel.tsx
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ExternalLink, BarChart3, Activity, Brain } from 'lucide-react';

interface TensorBoardPanelProps {
  jobId: string;
  trialId?: string;
  height?: number;
}

export const TensorBoardPanel: React.FC<TensorBoardPanelProps> = ({ 
  jobId, 
  trialId = "0",
  height = 600 
}) => {
  const [plotType, setPlotType] = useState('training_history');
  
  // Query for plot image
  const plotUrl = `/api/jobs/${jobId}/plots/${trialId}/${plotType}`;
  
  // Query for TensorBoard URL (for deep dive option)
  const { data: tbConfig } = useQuery({
    queryKey: ['tensorboard-config', jobId],
    queryFn: async () => {
      const response = await fetch(`/api/jobs/${jobId}/tensorboard/url`);
      return response.json();
    },
    staleTime: 30000
  });
  
  // Query for automated insights
  const { data: insights } = useQuery({
    queryKey: ['trial-insights', jobId, trialId],
    queryFn: async () => {
      const response = await fetch(`/api/jobs/${jobId}/plots/${trialId}/insights`);
      return response.json();
    },
    staleTime: 60000
  });
  
  const tensorboardUrl = tbConfig?.tensorboard_url 
    ? `${tbConfig.tensorboard_url}#runs=trial_${trialId}`
    : null;

  const plotOptions = [
    { value: 'training_history', label: 'Loss & Accuracy Curves', icon: <BarChart3 className="w-4 h-4" /> },
    { value: 'gradient_analysis', label: 'Gradient Flow & Dead Neurons', icon: <Activity className="w-4 h-4" /> },
    { value: 'weight_analysis', label: 'Weight Distributions', icon: <Brain className="w-4 h-4" /> }
  ];

  return (
    <Card className="w-full h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-lg">Training Metrics & Diagnostics</CardTitle>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Plot type selector */}
            <Select value={plotType} onValueChange={setPlotType}>
              <SelectTrigger className="w-64">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {plotOptions.map(option => (
                  <SelectItem key={option.value} value={option.value}>
                    <div className="flex items-center gap-2">
                      {option.icon}
                      {option.label}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            {/* TensorBoard deep dive button */}
            {tensorboardUrl && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => window.open(tensorboardUrl, '_blank')}
                className="flex items-center gap-1"
              >
                <ExternalLink className="w-3 h-3" />
                Full TensorBoard Analysis
              </Button>
            )}
          </div>
        </div>
        
        <div className="text-sm text-muted-foreground">
          Job: <code className="font-mono">{jobId}</code>
          {trialId && <span> • Trial: <code className="font-mono">{trialId}</code></span>}
        </div>
      </CardHeader>

      <CardContent className="p-0">
        <div className="relative">
          {/* Main plot display */}
          <div className="border border-border rounded-lg overflow-hidden">
            <img 
              src={plotUrl}
              alt={`Training ${plotType.replace('_', ' ')}`}
              className="w-full h-auto max-h-96 object-contain bg-white"
              onError={(e) => {
                // Fallback display when plot not available
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                target.nextElementSibling?.classList.remove('hidden');
              }}
            />
            
            {/* Fallback display */}
            <div className="hidden w-full h-64 flex items-center justify-center bg-gray-50 text-gray-500">
              <div className="text-center">
                <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Training visualization will appear after trial completes</p>
              </div>
            </div>
          </div>
          
          {/* Insights overlay */}
          {insights && insights.insights && (
            <TrainingInsights 
              insights={insights}
              plotType={plotType}
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// New insights overlay component
interface TrainingInsightsProps {
  insights: {
    insights: string[];
    overfitting_detected: boolean;
    final_metrics: Record<string, number>;
    health_summary?: string;
  };
  plotType: string;
}

const TrainingInsights: React.FC<TrainingInsightsProps> = ({ insights, plotType }) => {
  const relevantInsights = insights.insights
    .filter(insight => {
      // Show relevant insights based on current plot type
      if (plotType === 'training_history') {
        return insight.includes('loss') || insight.includes('accuracy') || insight.includes('overfitting');
      } else if (plotType === 'gradient_analysis') {
        return insight.includes('gradient') || insight.includes('neuron') || insight.includes('dead');
      } else if (plotType === 'weight_analysis') {
        return insight.includes('weight') || insight.includes('parameter') || insight.includes('distribution');
      }
      return true;
    })
    .slice(0, 3); // Limit to 3 most relevant insights

  return (
    <div className="absolute bottom-4 right-4 bg-black bg-opacity-85 text-white p-3 rounded-lg max-w-sm">
      <h4 className="font-semibold mb-2 flex items-center gap-2">
        <Activity className="w-4 h-4" />
        Key Insights
      </h4>
      
      <div className="space-y-2">
        {relevantInsights.map((insight, i) => (
          <div key={i} className="flex items-start gap-2 text-sm">
            <span className="text-xs mt-1">•</span>
            <span>{insight}</span>
          </div>
        ))}
        
        {insights.overfitting_detected && (
          <div className="mt-2 p-2 bg-yellow-600 bg-opacity-50 rounded text-xs">
            ⚠️ Overfitting detected - consider regularization
          </div>
        )}
        
        {insights.final_metrics.final_accuracy && (
          <div className="mt-2 text-xs text-gray-300">
            Final Accuracy: {(insights.final_metrics.final_accuracy * 100).toFixed(1)}%
          </div>
        )}
      </div>
    </div>
  );
};

export default TensorBoardPanel;
```

**Files Modified:**
- `web-ui/src/components/visualization/tensorboard-panel.tsx` - Complete rewrite with plot display
- `web-ui/src/components/ui/select.tsx` - Add if not exists (for plot type selector)

#### **Step 6B.4: Testing & Verification**

**Automated Testing:**
```python
# New test: test_embedded_visualizations.py

def test_plot_generation_during_training():
    """Verify plots are generated after each trial completes"""
    
def test_plot_api_endpoints():
    """Test plot serving API endpoints return correct images"""
    
def test_insights_api_endpoint():
    """Test insights API returns proper analysis data"""
    
def test_plot_file_organization():
    """Verify plots are saved in organized directory structure"""
    
def test_multiple_plot_types():
    """Test all plot types are generated correctly"""
    
# Integration test: test_ui_plot_integration.py
def test_frontend_plot_display():
    """Test complete flow: training → plot generation → API serving → UI display"""
```

**Manual Testing Protocol:**
1. **Start new optimization via UI**
   - Select dataset (e.g., MNIST)
   - Choose optimization mode (e.g., "Accuracy + model health")  
   - Trigger optimization

2. **Monitor trial completion**
   - Wait for first trial to complete
   - Verify plots appear immediately in "Training Metrics & Diagnostics" panel
   - Test plot type selector (Loss & Accuracy, Gradient Flow, Weight Distributions)

3. **Verify insights overlay**
   - Confirm automated insights appear on plot
   - Check overfitting detection if applicable
   - Verify final metrics display

4. **Test TensorBoard deep dive**
   - Click "Full TensorBoard Analysis" button
   - Verify opens in new tab with correct trial data
   - Confirm more detailed metrics available

5. **Test multiple trials**
   - Run optimization with multiple trials
   - Verify each trial generates separate plots
   - Test trial selector functionality

**Success Criteria:**
- ✅ Plots appear immediately after trial completion (no server startup delay)
- ✅ All three plot types generate correctly (training_history, gradient_analysis, weight_analysis)
- ✅ Insights overlay provides meaningful, context-aware information
- ✅ Plot type selector works smoothly
- ✅ TensorBoard "deep dive" option functions properly
- ✅ Mobile responsiveness maintained (images scale properly)
- ✅ Error handling works when plots not available

---

### **Phase 6C: System Integration & File Organization** 🗂️
**Status**: Ensure consistent behavior and proper file management

#### **Step 6C.1: Unified Directory Structure**
**Objective**: Organize all generated files consistently

**Implementation Details:**
```bash
# New organized structure:
computer-vision-classification/
├── plots/                                    # Generated plot images (NEW)
│   ├── 2025-08-27-10-15-32_mnist/           # Timestamped optimization run
│   │   ├── trial_0_training_history.png     # Loss/accuracy curves  
│   │   ├── trial_0_gradient_analysis.png    # Dead neurons, gradient flow
│   │   ├── trial_0_weight_analysis.png      # Weight distributions
│   │   ├── trial_1_training_history.png     # Additional trials...
│   │   └── metadata.json                    # Run metadata and insights
│   └── 2025-08-27-11-30-45_cifar10/         # Different optimization run
├── tensorboard_logs/                         # TensorBoard logs (FIXED PATH)
│   ├── 2025-08-27-10-15-32_mnist/           # Same timestamp as plots
│   │   ├── trial_0/                         # TensorBoard trial logs
│   │   │   ├── train/
│   │   │   ├── validation/
│   │   │   └── health_metrics/
│   │   └── trial_1/
├── optimization_results/                     # Model files and configs
│   ├── 2025-08-27-10-15-32_mnist_health/    # Best model from run
│   │   ├── best_model.keras
│   │   ├── best_hyperparameters.yaml
│   │   └── optimization_summary.json
```

**Implementation:**
```python
# Add to api_server.py - File management utilities
class FileManager:
    """Utility class for organizing generated files consistently"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.plots_dir = base_dir / "plots"
        self.tensorboard_dir = base_dir / "tensorboard_logs" 
        self.results_dir = base_dir / "optimization_results"
    
    def create_run_directories(self, run_timestamp: str, dataset_name: str):
        """Create organized directories for a new optimization run"""
        run_id = f"{run_timestamp}_{dataset_name.lower()}"
        
        directories = {
            'plots': self.plots_dir / run_id,
            'tensorboard': self.tensorboard_dir / run_id,
            'results': self.results_dir / f"{run_id}_health"  # Match existing naming
        }
        
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return directories
    
    def find_plot_file(self, job_id: str, trial_id: str, plot_type: str) -> Path:
        """Locate plot file based on job and trial information"""
        # Implementation to find correct plot file
        
    def cleanup_old_files(self, days_old: int = 30):
        """Clean up files older than specified days"""
        # Implementation for maintenance
```

#### **Step 6C.2: Enhanced Error Handling**
**Objective**: Graceful degradation when plots or TensorBoard unavailable

**Implementation Details:**
```python
# In api_server.py - Enhanced error handling
@app.get("/jobs/{job_id}/plots/{trial_id}/{plot_type}")
async def get_trial_plot(job_id: str, trial_id: str, plot_type: str):
    try:
        plot_path = self._find_plot_file(job_id, trial_id, plot_type)
        
        if plot_path and plot_path.exists():
            return FileResponse(plot_path, media_type='image/png')
        else:
            # Return placeholder image instead of 404
            placeholder_path = self._generate_placeholder_plot(plot_type)
            return FileResponse(
                placeholder_path, 
                media_type='image/png',
                headers={"X-Plot-Status": "placeholder"}
            )
            
    except Exception as e:
        logger.error(f"Error serving plot {plot_type} for job {job_id}, trial {trial_id}: {e}")
        # Return error visualization instead of HTTP error
        error_plot_path = self._generate_error_plot(str(e))
        return FileResponse(error_plot_path, media_type='image/png')

def _generate_placeholder_plot(self, plot_type: str) -> Path:
    """Generate placeholder image when plot not yet available"""
    # Implementation to create "Training in progress..." image
    
def _generate_error_plot(self, error_message: str) -> Path:
    """Generate error visualization with helpful message"""
    # Implementation to create informative error image
```

#### **Step 6C.3: Performance Optimization**
**Objective**: Ensure system performs well with multiple trials and large plots

**Implementation Details:**
```python
# In model_builder.py - Optimize plot generation
def _generate_training_plots(self, history, model, run_timestamp):
    """Generate plots with performance optimizations"""
    
    # Use ThreadPoolExecutor for parallel plot generation
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all plot generation tasks
        future_training = executor.submit(self._generate_training_history_plot, history, run_timestamp)
        future_gradient = executor.submit(self._generate_gradient_plot, model, run_timestamp)  
        future_weights = executor.submit(self._generate_weights_plot, model, run_timestamp)
        
        # Collect results
        plot_results = {
            'training_history': future_training.result(),
            'gradient_analysis': future_gradient.result(),
            'weight_analysis': future_weights.result()
        }
    
    return plot_results

# Optimize plot file sizes
def _optimize_plot_for_web(self, plot_path: Path):
    """Optimize plot file size for web display"""
    # Compress PNG files, reduce DPI if necessary for web display
```

---

## **Comprehensive Testing Plan**

### **Phase 6A Testing: Code Path Consolidation**

#### **Automated Tests**
```python
# tests/test_code_path_consolidation.py

class TestCodePathConsolidation:
    def test_all_ui_optimizations_use_model_optimizer(self):
        """Verify UI optimizations go through ModelOptimizer, not create_and_train_model"""
        # Test API endpoint behavior
        # Verify ModelOptimizer instance creation
        # Check no direct create_and_train_model calls
        
    def test_tensorboard_logs_project_root(self):
        """Verify TensorBoard logs created in project root, not src/"""
        # Run optimization
        # Check log file locations
        # Verify directory structure
        
    def test_single_trial_optuna_mode(self):
        """Test single trial works properly with Optuna framework"""
        # Configure max_trials=1
        # Run optimization
        # Verify proper Optuna behavior
        # Check trial numbering
        
    def test_timestamp_consistency(self):
        """Verify same timestamp used across all generated files"""
        # Run optimization
        # Check timestamps in: logs, plots, model files
        # Verify consistency
        
    def test_job_id_to_run_name_mapping(self):
        """Test job_id properly used as run_name"""
        # Create optimization with known job_id
        # Verify run_name matches job_id
        # Check file naming consistency
```

#### **Manual Testing Protocol**

**Test 7A.1: UI Optimization Path Verification**
1. Open web UI at `http://localhost:3000`
2. Select dataset: "MNIST"
3. Select mode: "Accuracy + model health"
4. Click "Start Optimization"
5. **Expected**: Logs show ModelOptimizer instantiation, not create_and_train_model
6. **Verify**: Check `logs/non-cron.log` for correct code path

**Test 7A.2: TensorBoard Log Location**
1. After optimization starts, monitor file system
2. **Expected**: New directory appears at project root: `tensorboard_logs/YYYY-MM-DD-HH-MM-SS_mnist/`
3. **Verify**: NOT in `src/tensorboard_logs/`
4. **Check**: Trial subdirectories created: `trial_0/train/`, `trial_0/validation/`, `trial_0/health_metrics/`

**Test 7A.3: Single Trial Mode**
1. Modify optimization config: set `max_trials: 1`
2. Run optimization via UI
3. **Expected**: Exactly one trial runs
4. **Verify**: Proper trial numbering (trial_0)
5. **Check**: Optimization completes successfully with single trial

**Test 7A.4: Timestamp Consistency Check**
1. Start optimization and note timestamp from UI
2. After completion, check file locations:
   - `tensorboard_logs/YYYY-MM-DD-HH-MM-SS_mnist/`
   - `optimization_results/YYYY-MM-DD-HH-MM-SS_mnist_health/`
   - Model file timestamps
3. **Expected**: All files use same timestamp
4. **Verify**: No mixing of timestamps from different runs

**Success Criteria for Phase 7A:**
- ✅ All UI optimizations use ModelOptimizer (no create_and_train_model calls)
- ✅ TensorBoard logs appear in project root `tensorboard_logs/` directory
- ✅ Single trial mode works properly with Optuna framework  
- ✅ Consistent timestamps across all generated files
- ✅ No empty or incorrectly placed log directories

---

### **Phase 6B Testing: Embedded Training Visualizations**

#### **Automated Tests**
```python
# tests/test_embedded_visualizations.py

class TestEmbeddedVisualizations:
    def test_plot_generation_integration(self):
        """Test plots generated during ModelBuilder training"""
        # Run model training
        # Verify plot files created
        # Check plot content validity
        
    def test_plot_api_endpoints(self):
        """Test plot serving API functionality"""
        # Test /jobs/{job_id}/plots/{trial_id}/{plot_type}
        # Verify correct image returned
        # Test error handling
        
    def test_insights_api_endpoint(self):
        """Test insights API returns proper data structure"""
        # Test /jobs/{job_id}/plots/{trial_id}/insights
        # Verify JSON response format
        # Check insight content quality
        
    def test_multiple_plot_types(self):
        """Verify all plot types generate correctly"""
        # Test training_history plots
        # Test gradient_analysis plots  
        # Test weight_analysis plots
        # Check file formats and sizes
        
    def test_plot_error_handling(self):
        """Test graceful degradation when plots unavailable"""
        # Request non-existent plot
        # Verify placeholder or error image returned
        # Check no system crashes
        
# tests/test_plot_integration.py
class TestPlotIntegration:
    def test_end_to_end_plot_flow(self):
        """Test complete flow: training → plot generation → API → UI display"""
        # Run full optimization
        # Verify plot generation
        # Test API serving  
        # Check UI display capabilities
```

#### **Manual Testing Protocol**

**Test 7B.1: Plot Generation During Training**
1. Start optimization via UI
2. Wait for first trial to complete
3. **Expected**: Plot files appear in `plots/YYYY-MM-DD-HH-MM-SS_mnist/`
4. **Check Files**:
   - `trial_0_training_history.png` (loss/accuracy curves)
   - `trial_0_gradient_analysis.png` (gradient flow, dead neurons)
   - `trial_0_weight_analysis.png` (weight distributions)
5. **Verify**: Files are valid PNG images with reasonable file sizes

**Test 7B.2: Plot API Endpoint Testing**
1. Use browser or curl to test API endpoints:
   ```bash
   curl -I http://localhost:8000/api/jobs/{job_id}/plots/0/training_history
   curl -I http://localhost:8000/api/jobs/{job_id}/plots/0/gradient_analysis  
   curl -I http://localhost:8000/api/jobs/{job_id}/plots/0/weight_analysis
   ```
2. **Expected**: HTTP 200 responses with `Content-Type: image/png`
3. **Verify**: Images display correctly in browser

**Test 7B.3: Insights API Testing**
1. Test insights endpoint:
   ```bash
   curl http://localhost:8000/api/jobs/{job_id}/plots/0/insights
   ```
2. **Expected**: JSON response with structure:
   ```json
   {
     "trial_id": "0",
     "insights": ["✅ Training loss showing healthy decrease", ...],
     "overfitting_detected": false,
     "final_metrics": {"final_accuracy": 0.95, ...},
     "health_summary": "Training completed successfully"
   }
   ```

**Test 7B.4: Frontend Plot Display**
1. Open UI during/after optimization
2. Navigate to "Training Metrics & Diagnostics" section
3. **Expected**: Plot appears immediately (no loading spinner)
4. **Test Plot Type Selector**:
   - Select "Loss & Accuracy Curves" → training_history plot appears
   - Select "Gradient Flow & Dead Neurons" → gradient_analysis plot appears
   - Select "Weight Distributions" → weight_analysis plot appears
5. **Verify**: Smooth transitions between plot types

**Test 7B.5: Insights Overlay Functionality**
1. After plots load, check for insights overlay (bottom-right corner)
2. **Expected**: 
   - "Key Insights" header with activity icon
   - 2-3 relevant insights displayed  
   - Overfitting warning if detected
   - Final accuracy percentage
3. **Test Context-Awareness**: 
   - Switch plot types, verify insights change relevance
   - Training history → loss/accuracy insights
   - Gradient analysis → dead neuron insights

**Test 7B.6: TensorBoard Deep Dive**
1. Click "Full TensorBoard Analysis" button
2. **Expected**: New tab opens with TensorBoard interface
3. **Verify**: Correct trial data displayed (`#runs=trial_0`)
4. **Check**: All TensorBoard features functional (scalars, histograms, graphs)

**Test 7B.7: Multiple Trial Handling**
1. Run optimization with 3 trials
2. **Expected**: Plot files for each trial:
   - `trial_0_training_history.png`
   - `trial_1_training_history.png`  
   - `trial_2_training_history.png`
3. **Test Trial Selection**: UI should allow switching between trial plots
4. **Verify**: Each trial shows different results

**Test 7B.8: Error Handling & Graceful Degradation**
1. **Test Plot Not Available**:
   - Request plot for non-existent trial
   - **Expected**: Placeholder image or helpful error message
2. **Test Network Issues**:
   - Simulate API unavailable
   - **Expected**: Fallback display, no UI crash
3. **Test Corrupt Plot Files**:
   - Manually corrupt a plot file
   - **Expected**: Error message, offer to regenerate

**Test 7B.9: Mobile Responsiveness**
1. Test on mobile device/browser dev tools
2. **Expected**: 
   - Images scale properly
   - Insights overlay remains readable
   - Plot selector works with touch
   - No horizontal scrolling issues

**Test 7B.10: Performance Testing**  
1. Run optimization with 5+ trials
2. Monitor plot generation time
3. Test API response time for plot serving
4. **Expected**: 
   - Plots generate within 30 seconds of trial completion
   - API serves images in <500ms
   - UI remains responsive during plot loading

**Success Criteria for Phase 7B:**
- ✅ Training plots generate automatically after each trial completion
- ✅ All three plot types (training_history, gradient_analysis, weight_analysis) work correctly
- ✅ Plot API endpoints serve images reliably and quickly
- ✅ Insights API provides meaningful, context-aware analysis
- ✅ Frontend displays plots immediately without loading delays
- ✅ Plot type selector works smoothly with proper transitions
- ✅ Insights overlay provides relevant information for each plot type
- ✅ TensorBoard "deep dive" option functions properly for detailed analysis
- ✅ Multiple trial handling works correctly
- ✅ Error handling provides graceful degradation
- ✅ Mobile responsiveness maintained throughout
- ✅ Performance remains good with multiple trials

---

## **Final Integration Testing**

### **End-to-End System Testing**

**Test E2E.1: Complete Optimization Flow**
1. **Start Fresh**: Clear all previous runs (`rm -rf plots/ tensorboard_logs/ optimization_results/`)
2. **Trigger Optimization**: Via UI, select MNIST, "Accuracy + model health", max_trials=3
3. **Monitor Throughout**:
   - Verify logs go to project root locations
   - Check plot generation after each trial
   - Test real-time UI updates
4. **Verify Final State**:
   - 3 trials completed
   - All plot files generated
   - TensorBoard logs properly organized
   - Best model identified and available for download
   - UI shows comprehensive results

**Test E2E.2: Cross-Dataset Compatibility**
1. Test with multiple datasets: MNIST, CIFAR-10, Fashion-MNIST
2. Verify consistent behavior across all datasets
3. Check plot quality and relevance for different data types

**Test E2E.3: Edge Cases**
1. **Single Trial Optimization**: max_trials=1
2. **Failed Trial Handling**: Simulate training failure
3. **Resource Constraints**: Test with limited disk space
4. **Concurrent Optimizations**: Multiple simultaneous jobs

**Test E2E.4: User Experience Flow**
1. **First-Time User**: Complete workflow documentation
2. **Power User**: Advanced features and TensorBoard integration
3. **Mobile User**: Complete mobile experience testing
4. **Accessibility**: Screen reader compatibility, keyboard navigation

---

## **Post-Implementation Validation**

### **Performance Benchmarks**
- **Plot Generation Time**: <30 seconds per trial
- **API Response Time**: <500ms for image serving
- **UI Responsiveness**: No blocking during plot loads
- **Memory Usage**: Reasonable memory consumption for plot generation
- **Disk Usage**: Efficient storage of plots and logs

### **Quality Assurance Checklist**
- [ ] All previous functionality preserved
- [ ] No regression in optimization accuracy
- [ ] TensorBoard deep dive remains fully functional
- [ ] Mobile experience improved (no iframe issues)
- [ ] Error handling comprehensive and user-friendly
- [ ] Code path consolidation eliminates confusion
- [ ] Documentation updated to reflect changes
- [ ] User feedback incorporated and addressed

---

## **Future Enhancement Opportunities**

### **Phase 6D: Advanced Visualization Features** (Future)
- **Interactive Plots**: Plotly.js integration for zoom/pan capabilities
- **Animated Training Progress**: Time-lapse visualization of training evolution
- **Comparative Analysis**: Side-by-side trial comparison plots
- **Export Functionality**: PDF report generation with all visualizations

### **Phase 6E: Real-Time Training Visualization** (Future)
- **Live Plot Updates**: Real-time plot updates during training
- **WebSocket Integration**: Sub-second latency for training progress
- **Interactive Training Control**: Pause/resume training based on visualizations

---

**Key Benefits of Phase 7 Implementation:**

### **Technical Benefits:**
- ✅ **Unified Code Path**: Eliminates confusion and maintenance overhead
- ✅ **Immediate Feedback**: Plots appear instantly without server complexity
- ✅ **Mobile Compatibility**: Images work perfectly on all devices
- ✅ **Simplified Architecture**: No iframe, proxy, or port management issues
- ✅ **Better Error Handling**: Graceful degradation with helpful feedback

### **User Experience Benefits:**
- ✅ **Instant Gratification**: See training results immediately after trial completion
- ✅ **Focused Insights**: Relevant visualizations without TensorBoard complexity
- ✅ **Progressive Enhancement**: Basic plots + optional deep dive
- ✅ **Context-Aware Help**: Automated insights explain what users are seeing
- ✅ **Professional Quality**: Publication-ready visualizations

### **Maintenance Benefits:**
- ✅ **Single Optimization Path**: Only one code path to maintain and debug
- ✅ **Easier Testing**: Consistent behavior to validate
- ✅ **Better Debugging**: Clear file organization and logging
- ✅ **Future Extensibility**: Easy to add new plot types or features

This comprehensive plan addresses both the technical debt (dual code paths) and user experience issues (complex TensorBoard embedding) while providing immediate value through embedded visualizations that work reliably across all devices and scenarios.

---

### **Phase 7: Enhanced Layer Visualization** 🎯
**Status**: **PLANNED** (Future enhancement after Phase 6)

**Objectives:**
- Add flattening layer visualization as distinct stage in model flow
- Implement filter and node count visualization within layers
- Enhanced educational value through detailed layer information
- Improved model architecture understanding

**Key Deliverables:**
- Flattening layer type with appropriate visual styling
- Filter count visualization for convolutional layers
- Node count visualization for dense layers
- Enhanced layer information display
- Hover tooltips for detailed layer descriptions

**Future Roadmap Items:**
- Interactive layer filtering and highlighting
- Animation improvements for data flow visualization
- Performance metrics integration with layer visualization
- Export functionality for model architecture diagrams

---

### **Phase 8: WebSocket Migration** ⚡
**Status**: Performance enhancement (after Phase 7 complete)

**Objectives:**
- Replace HTTP polling with WebSocket real-time updates for sub-second latency
- Enhanced real-time visualization updates during optimization
- Improved user experience with instant feedback
- Reduced server load through efficient communication protocols

**Key Deliverables:**
- WebSocket-based real-time communication system
- Sub-second latency for optimization progress updates
- Enhanced user experience with instant visual feedback
- Optimized server resource utilization