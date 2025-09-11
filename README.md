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

### **Phase 1: Miscellanious UI improvements** 🔧
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

### **Phase 2: RunPod S3 Storage Integration** ✅ **COMPLETED**
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

### **Phase 3: Plot Transfer via S3 Extension** 🔧
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

### **Phase 4: System Polish & Testing** 🔧
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


### **Phase 5: Deployment & Container Integration** 🚀
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

### **Phase 6: Website Integration** 🌐
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
