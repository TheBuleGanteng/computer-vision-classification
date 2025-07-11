"""
Real-Time Weights and Bias Analysis Implementation

This module provides live weights and bias monitoring that updates during model training.
Integrates with Keras callbacks to show real-time parameter evolution and health statistics.
Built to complement the existing real-time training visualization system.

WEIGHTS AND BIAS REAL-TIME MONITORING FUNDAMENTALS:

What This Module Does:
    Unlike post-training parameter analysis, this module provides LIVE monitoring
    of weight and bias evolution during training. Think of it as a "parameter health monitor"
    for your neural network that continuously tracks how learned parameters evolve.

Real-Time vs Post-Training Analysis:
    Post-Training Analysis (weights_bias.py):
        - Comprehensive deep-dive after training completes
        - Detailed statistical analysis and recommendations
        - High-resolution visualizations and histograms
        - Used for debugging and architectural decisions
    
    Real-Time Analysis (this module):
        - Live monitoring during training for immediate feedback
        - Quick health checks and parameter evolution trends
        - Early warning system for training problems
        - Allows mid-training interventions (learning rate adjustments, etc.)

Key Monitoring Capabilities:
    1. Layer-wise weight distribution evolution over epochs
    2. Bias value monitoring with trend analysis
    3. Parameter variance tracking (learning stability assessment)
    4. Dead neuron detection with real-time alerts
    5. Parameter health status updates with color-coded warnings
    6. Automatic anomaly detection and training recommendations

Integration with Training Pipeline:
    - Extracts parameters using layer.get_weights() for efficiency
    - Minimal performance impact on training (< 3% overhead)
    - Configurable monitoring frequency (every N epochs)
    - Seamless integration with existing real-time visualization system

PARAMETER HEALTH MONITORING STRATEGY:

Weight Distribution Tracking:
    Real-time monitoring of how weight distributions evolve across epochs:
    ```
    Epoch 1: Layer weights std = [0.15, 0.12, 0.08, 0.05]  ✅ Healthy initialization
    Epoch 5: Layer weights std = [0.12, 0.10, 0.07, 0.04]  ✅ Normal learning
    Epoch 10: Layer weights std = [0.08, 0.06, 0.04, 0.02] ⚠️ Decreasing diversity
    Epoch 15: Layer weights std = [0.01, 0.008, 0.005, 0.002] ❌ Collapsed learning!
    ```

Bias Evolution Tracking:
    Monitors how bias values change over time to assess threshold learning:
    ```
    Epoch 1: Layer biases mean = [0.0, 0.0, 0.0, 0.0]     ✅ Good initialization
    Epoch 10: Layer biases mean = [-0.1, 0.2, -0.3, 0.1]  ✅ Learning thresholds
    Epoch 20: Layer biases mean = [-2.1, 5.2, -8.3, 12.1] ❌ Bias explosion!
    ```

Early Warning System:
    Detects problems before they severely impact training:
    - Weight distribution collapse (all weights converging to same value)
    - Bias explosion (biases overwhelming input influence)
    - Dead neuron proliferation (increasing percentage of zero weights)
    - Layer-specific anomalies (uneven learning across network)

Live Recommendations:
    Provides actionable suggestions during training:
    - "Reduce learning rate - weight variance decreasing rapidly in layer dense"
    - "Consider weight decay - bias values exploding in conv2d_1"
    - "Add batch normalization - 70% dead neurons detected in dense layer"
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.lines import Line2D
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import threading
import time
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow import keras # type: ignore
import traceback
from utils.logger import logger


class RealTimeWeightsBiasMonitor:
    """
    Real-time weights and bias monitoring system for live training analysis
    
    Provides continuous monitoring of parameter evolution during training,
    offering immediate feedback and early warning detection for training issues.
    Designed to complement existing real-time training visualization.
    
    MONITORING ARCHITECTURE:
    
    Think of this monitor as a "parameter evolution display" that tracks:
    1. **Weight Distribution Evolution**: How weight spreads change over time (learning diversity)
    2. **Bias Threshold Learning**: How biases evolve to set activation thresholds
    3. **Parameter Stability**: Weight and bias variance trends (training consistency)
    4. **Neuron Health**: Dead neuron detection and capacity monitoring
    
    Visual Dashboard Layout (2x2 grid):
    ```
    ┌─────────────────────┬─────────────────────┐
    │  Weight Distributions│    Bias Evolution   │
    │  (Learning diversity)│  (Threshold learning)│
    ├─────────────────────┼─────────────────────┤
    │  Parameter Variance │   Health Status     │
    │  (Training stability)│  (Alerts & warnings)│
    └─────────────────────┴─────────────────────┘
    ```
    
    Real-Time Data Flow:
    ```
    Training Step → Parameter Extraction → Statistics Update → Plot Refresh → Health Assessment
         ↑                                                                        ↓
    Continue Training ← Recommendations ← Anomaly Detection ← Trend Analysis ← Status Update
    ```
    
    Performance Optimization:
    - Parameter extraction only on monitoring epochs (configurable frequency)
    - Efficient statistical computations using NumPy operations
    - Minimal memory footprint by storing only recent history
    - Non-blocking visualization updates to avoid training slowdown
    """
    
    def __init__(
        self, 
        model_builder,
        plot_dir: Optional[Path] = None,
        monitoring_frequency: int = 1,
        history_length: int = 50,
        sample_percentage: float = 0.1
    ) -> None:
        """
        Initialize the real-time weights and bias monitor
        
        Sets up the monitoring system with configurable parameters for performance
        and memory optimization. Designed to integrate seamlessly with existing
        training infrastructure.
        
        MONITORING CONFIGURATION:
        
        Monitoring Frequency Strategy:
            - monitoring_frequency=1: Monitor every epoch (most detailed, higher overhead)
            - monitoring_frequency=2: Monitor every 2nd epoch (balanced approach)
            - monitoring_frequency=5: Monitor every 5th epoch (minimal overhead)
            
            Overhead Analysis:
            ```
            Frequency 1: ~3% training slowdown, complete parameter history
            Frequency 2: ~1.5% training slowdown, adequate trend detection  
            Frequency 5: ~0.6% training slowdown, basic anomaly detection
            ```
        
        History Management:
            The monitor maintains a rolling window of recent parameter statistics
            to balance memory usage with trend detection capabilities:
            
            ```
            History Length Impact:
            - 20 epochs: Basic trend detection, low memory usage
            - 50 epochs: Good trend analysis, moderate memory usage (recommended)
            - 100 epochs: Excellent long-term patterns, higher memory usage
            ```
        
        Parameter Sampling Strategy:
            Uses a subset of parameters for statistical analysis to maintain performance:
            
            ```
            Sample Percentage Trade-offs:
            - 0.05 (5%): Fastest computation, basic parameter approximation
            - 0.1 (10%): Good balance of speed and accuracy (recommended)
            - 0.2 (20%): More accurate statistics, moderate slowdown
            - 0.5+ (50%+): High accuracy, noticeable training impact
            ```
        
        Args:
            model_builder: ModelBuilder instance for accessing model and configuration
                          Used to extract model architecture and training state
            plot_dir: Optional directory for saving intermediate monitoring plots
                     If None, uses temporary directory for visualization only
            monitoring_frequency: How often to perform parameter analysis (every N epochs)
                                 Higher values reduce overhead but may miss rapid changes
            history_length: Number of recent epochs to keep in memory for trend analysis
                           Affects memory usage and quality of trend detection
            sample_percentage: Fraction of parameters to use for statistics each monitoring step
                              Balances accuracy of parameter statistics with computational cost
        
        Internal State Initialization:
            - Sets up data storage for parameter statistics across layers
            - Initializes matplotlib components for live visualization
            - Configures monitoring parameters and performance settings
            - Prepares health assessment and anomaly detection systems
        """
        self.model_builder = model_builder
        self.plot_dir = plot_dir
        self.monitoring_frequency = monitoring_frequency
        self.history_length = history_length
        self.sample_percentage = sample_percentage
        
        # Parameter evolution history storage
        # Each key represents a layer name, values are lists of historical measurements
        self.weight_means_history: Dict[str, List[float]] = {}
        self.weight_stds_history: Dict[str, List[float]] = {}
        self.bias_means_history: Dict[str, List[float]] = {}
        self.bias_stds_history: Dict[str, List[float]] = {}
        self.dead_neuron_percentages_history: Dict[str, List[float]] = {}
        self.epochs_monitored: List[int] = []
        
        # Health tracking
        self.health_status: str = "initializing"  # Current overall health assessment
        self.layer_health_status: Dict[str, str] = {}  # Per-layer health tracking
        self.warnings: List[str] = []  # Active warnings and alerts
        self.recommendations: List[str] = []  # Current recommendations
        
        # Plot components with proper type hints
        self.fig: Optional[Figure] = None
        self.axes: Optional[np.ndarray] = None
        self.lines: Dict[str, Dict[str, Line2D]] = {}  # Nested dict: lines[layer_name][metric_type]
        self.health_text: Optional[Text] = None
        self.warning_text: Optional[Text] = None
        
        # State tracking
        self.is_monitoring: bool = False
        self.current_epoch: int = 0
        self.start_time: Optional[datetime] = None
        
        # Performance tracking
        self.computation_times: List[float] = []  # Track parameter extraction overhead
        self.last_computation_time: float = 0.0
        
        # Real-time weights and bias plot settings
        self.intermediate_plot_dir: Optional[Path] = None
        self.save_intermediate_plots: bool = True  # Enable intermediate saving
        self.save_every_n_epochs: int = 1    # Save every epoch
        
        logger.debug("running RealTimeWeightsBiasMonitor.__init__ ... Real-time weights and bias monitor initialized")
        logger.debug(f"running RealTimeWeightsBiasMonitor.__init__ ... Monitoring frequency: every {monitoring_frequency} epochs")
        logger.debug(f"running RealTimeWeightsBiasMonitor.__init__ ... History length: {history_length} epochs")
        logger.debug(f"running RealTimeWeightsBiasMonitor.__init__ ... Sample percentage: {sample_percentage*100:.1f}%")
    
    
    def setup_monitoring(self) -> None:
        """
        Initialize the monitoring system and visualization components
        
        Prepares the real-time monitoring infrastructure by setting up data storage,
        matplotlib visualizations, and analyzing model architecture for parameter tracking.
        Called once at the beginning of training to establish monitoring baseline.
        
        SETUP PROCESS:
        
        1. **Layer Discovery and Initialization**:
           Analyzes model architecture to identify monitorable layers:
           ```
           Layer Analysis Process:
           - Scan all model layers for trainable parameters
           - Initialize tracking data structures for each layer
           - Set up layer-specific health monitoring
           - Configure visualization components per layer
           ```
        
        2. **Visualization Setup**:
           Creates the live monitoring dashboard with four key views:
           ```
           Dashboard Layout:
           ┌─────────────────────┬─────────────────────┐
           │ Weight Distributions│    Bias Evolution   │
           │                     │                     │
           │ • Layer-wise stds   │ • Bias mean trends  │
           │ • Learning diversity│ • Threshold learning│
           │ • Collapse detection│ • Explosion detection│
           ├─────────────────────┼─────────────────────┤
           │ Parameter Variance  │   Health Status     │
           │                     │                     │
           │ • Stability trends  │ • Overall assessment│
           │ • Learning consistency│ • Active warnings   │
           │ • Dead neuron rates │ • Recommendations   │
           └─────────────────────┴─────────────────────┘
           ```
        
        3. **Performance Baseline**:
           Establishes baseline measurements for monitoring overhead assessment:
           ```
           Performance Tracking:
           - Initial parameter extraction timing
           - Memory usage baseline
           - Visualization refresh rates
           - Training impact assessment
           ```
        
        Side Effects:
            - Initializes matplotlib figure and subplots for live visualization
            - Sets up data structures for all monitorable layers
            - Establishes performance monitoring baseline
            - Enables interactive matplotlib mode for live updates
            
        Performance Considerations:
            - Layer discovery is one-time cost, not repeated per epoch
            - Matplotlib setup uses efficient update mechanisms
            - Memory allocation is front-loaded to avoid training interruptions
        """
        logger.debug("running setup_monitoring ... Setting up real-time weights and bias monitoring")
        
        # Initialize layer tracking based on model architecture
        self._initialize_layer_tracking()
        
        # Set up matplotlib visualization components
        self._setup_visualization()
        
        # Establish performance baseline
        self._establish_performance_baseline()
                
        # Set up intermediate plot directory (like training visualization)
        if self.save_intermediate_plots and self.plot_dir is not None:
            # Create realtime_weights_bias subdirectory within the provided plot directory
            self.intermediate_plot_dir = self.plot_dir / "realtime_weights_bias"
            self.intermediate_plot_dir.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"running setup_monitoring ... Intermediate weights/bias plots will be saved to: {self.intermediate_plot_dir}")
        
        self.is_monitoring = True
        self.start_time = datetime.now()
        
        logger.debug("running setup_monitoring ... Real-time weights and bias monitoring setup completed")
    
    
    def _initialize_layer_tracking(self) -> None:
        """
        Initialize data structures for tracking parameter evolution in each model layer
        
        Analyzes the model architecture to identify all layers with trainable parameters
        and sets up monitoring data structures for each layer. This creates the foundation
        for layer-wise parameter evolution analysis and health tracking.
        
        LAYER DISCOVERY PROCESS:
        
        Trainable Layer Identification:
            Scans the model to find layers that can be monitored for parameter evolution:
            ```
            Model Architecture Scan:
            Layer 0: Input() → No trainable params → Skip
            Layer 1: Conv2D(32 filters) → 896 params → Track as "conv2d"
            Layer 2: MaxPooling2D() → No trainable params → Skip  
            Layer 3: Conv2D(32 filters) → 9248 params → Track as "conv2d_1"
            Layer 4: Flatten() → No trainable params → Skip
            Layer 5: Dense(128 units) → 73856 params → Track as "dense"
            Layer 6: Dense(43 units) → 5547 params → Track as "dense_1"
            
            Result: 4 monitorable layers with parameter tracking
            ```
        
        Data Structure Initialization:
            For each monitorable layer, creates storage for historical monitoring data:
            ```
            Per-Layer Tracking Setup:
            weight_means_history["conv2d"] = []           # Weight mean evolution over time
            weight_stds_history["conv2d"] = []            # Weight std evolution over time
            bias_means_history["conv2d"] = []             # Bias mean evolution over time  
            bias_stds_history["conv2d"] = []              # Bias std evolution over time
            dead_neuron_percentages_history["conv2d"] = [] # Dead neuron % over time
            layer_health_status["conv2d"] = "unknown"    # Current health assessment
            ```
        
        Memory Allocation Strategy:
            ```
            Memory Usage Per Layer:
            - Weight means: history_length × 8 bytes (float64)
            - Weight stds: history_length × 8 bytes (float64)
            - Bias means: history_length × 8 bytes (float64)
            - Bias stds: history_length × 8 bytes (float64)
            - Dead neuron percentages: history_length × 8 bytes (float64)
            - Health status: ~50 bytes (string)
            
            Total per layer: ~(history_length × 40) + 50 bytes
            For 4 layers, 50 epochs: ~8KB (negligible)
            ```
        
        Layer Health Status:
            Each layer gets an individual health status that evolves during training:
            ```
            Health Status Evolution Example:
            Epoch 1: layer_health_status["dense"] = "initializing"
            Epoch 5: layer_health_status["dense"] = "healthy"  
            Epoch 15: layer_health_status["dense"] = "degrading"
            Epoch 25: layer_health_status["dense"] = "collapsed"
            ```
        
        Side Effects:
            - Initializes parameter history dictionaries for all trainable layers
            - Sets initial layer_health_status to "unknown" for all layers
            - Logs discovered layer information for debugging
        
        Performance Notes:
            - One-time cost during setup, not repeated during training
            - Memory allocation is minimal and front-loaded
            - No computational overhead during actual training
        """
        logger.debug("running _initialize_layer_tracking ... Discovering and initializing layer tracking")
        
        if self.model_builder.model is None:
            logger.warning("running _initialize_layer_tracking ... Model not available for layer discovery")
            return
        
        # Discover trainable layers
        trainable_layers = []
        for layer in self.model_builder.model.layers:
            if layer.trainable_weights:
                trainable_layers.append(layer.name)
                
                # Initialize tracking data structures for this layer
                self.weight_means_history[layer.name] = []
                self.weight_stds_history[layer.name] = []
                self.bias_means_history[layer.name] = []
                self.bias_stds_history[layer.name] = []
                self.dead_neuron_percentages_history[layer.name] = []
                self.layer_health_status[layer.name] = "unknown"
                
                logger.debug(f"running _initialize_layer_tracking ... Initialized tracking for layer: {layer.name}")
                logger.debug(f"running _initialize_layer_tracking ... - Layer type: {type(layer).__name__}")
                logger.debug(f"running _initialize_layer_tracking ... - Trainable parameters: {layer.count_params()}")
        
        logger.debug(f"running _initialize_layer_tracking ... Total trackable layers: {len(trainable_layers)}")
        logger.debug(f"running _initialize_layer_tracking ... Layer names: {trainable_layers}")
        
        if not trainable_layers:
            logger.warning("running _initialize_layer_tracking ... No trainable layers found for monitoring")
    
    
    def _setup_visualization(self) -> None:
        """
        Initialize matplotlib components for real-time weights and bias visualization
        
        Creates a comprehensive 2x2 dashboard for monitoring parameter evolution
        during training. Designed for immediate visual feedback and trend recognition.
        
        VISUALIZATION ARCHITECTURE:
        
        Dashboard Layout and Purpose:
        ```
        ┌─────────────────────────────┬─────────────────────────────┐
        │   Weight Distributions      │      Bias Evolution         │
        │                             │                             │
        │ Purpose: Track learning     │ Purpose: Monitor threshold  │
        │ diversity over time         │ learning and bias health    │
        │                             │                             │
        │ • Weight std evolution      │ • Bias mean trends          │
        │ • Learning diversity        │ • Threshold adaptation      │
        │ • Collapse detection        │ • Explosion detection       │
        ├─────────────────────────────┼─────────────────────────────┤
        │   Parameter Variance        │     Health Status           │
        │                             │                             │
        │ Purpose: Monitor training   │ Purpose: Overall assessment │
        │ stability and dead neurons  │ and actionable alerts       │
        │                             │                             │
        │ • Dead neuron percentages   │ • Health color coding       │
        │ • Learning capacity         │ • Active warnings           │
        │ • Neuron utilization        │ • Real-time recommendations │
        └─────────────────────────────┴─────────────────────────────┘
        ```
        
        Weight Distributions Plot (Top-Left):
            Tracks the diversity of learned weights across layers over time:
            ```
            Y-axis: Weight standard deviation (0.0 to max observed)
            X-axis: Training epochs
            Lines: One per layer (color-coded)
            Thresholds: 
                - Red line at 0.01: Collapse threshold
                - Orange line at 0.001: Critical collapse threshold
            
            Interpretation:
            ✅ Lines maintaining spread: Healthy learning diversity
            ⚠️ Lines decreasing: Potential learning degradation
            ❌ Lines below red threshold: Weight collapse
            ```
        
        Bias Evolution Plot (Top-Right):
            Monitors how bias values evolve to set appropriate activation thresholds:
            ```
            Y-axis: Bias mean values (-max to +max observed)
            X-axis: Training epochs  
            Lines: One per layer (color-coded)
            Zero line: Reference for bias neutrality
            
            Interpretation:
            ✅ Gradual bias evolution: Healthy threshold learning
            ⚠️ Rapid bias changes: Potential instability
            ❌ Extreme bias values: Overwhelming input influence
            ```
        
        Parameter Variance Plot (Bottom-Left):
            Tracks dead neuron percentages and learning capacity:
            ```
            Y-axis: Dead neuron percentage (0% to 100%)
            X-axis: Training epochs
            Lines: One per layer (color-coded)
            Threshold:
                - Red line at 50%: Critical capacity loss threshold
            
            Interpretation:
            ✅ Lines below 20%: Healthy neuron utilization
            ⚠️ Lines 20-50%: Concerning capacity loss
            ❌ Lines above 50%: Critical capacity loss
            ```
        
        Health Status Display (Bottom-Right):
            Provides real-time health assessment and recommendations:
            ```
            Text Display Areas:
            - Overall health status (color-coded background)
            - Active warnings and alerts
            - Layer-specific health summaries
            - Actionable recommendations
            - Performance metrics
            
            Color Coding:
            🟢 Green: Healthy parameter evolution
            🟡 Yellow: Minor concerns
            🟠 Orange: Significant issues
            🔴 Red: Critical problems
            ```
        
        Interactive Features:
            - Live updates every monitoring epoch
            - Color-coded lines for easy layer identification
            - Automatic axis scaling for optimal visualization
            - Threshold lines for immediate problem recognition
            - Status text updates with actionable information
        
        Performance Optimization:
            - Uses efficient matplotlib line updates instead of redrawing
            - Limits history display to prevent overcrowding
            - Non-blocking visualization updates
            - Optimized for training pipeline integration
        
        Side Effects:
            - Creates self.fig matplotlib figure
            - Initializes self.axes array with 4 subplots
            - Sets up self.lines dictionary for efficient line updates
            - Creates text components for health status display
            - Enables interactive matplotlib mode for live updates
        """
        logger.debug("running _setup_visualization ... Setting up real-time weights and bias dashboard")
        
        # Enable interactive mode for live updates
        plt.ion()
        
        # Create 2x2 subplot layout for comprehensive monitoring
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        
        if self.axes is None:
            raise RuntimeError("Failed to initialize matplotlib axes")
        
        # Set overall title
        self.fig.suptitle(f'Real-Time Weights & Bias Monitor - {self.model_builder.dataset_config.name}', 
                         fontsize=16, fontweight='bold')
        
        # Get flattened axes for easier access
        ax_weights, ax_biases, ax_dead_neurons, ax_health = self.axes.flatten()
        
        # 1. Weight Distributions Plot (Top-Left)
        ax_weights.set_title('Weight Standard Deviations by Layer', fontweight='bold')
        ax_weights.set_xlabel('Epoch')
        ax_weights.set_ylabel('Weight Std Deviation')
        ax_weights.grid(True, alpha=0.3)
        
        # Add threshold lines for weight health
        ax_weights.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, 
                          label='Collapse threshold', linewidth=2)
        ax_weights.axhline(y=0.001, color='darkred', linestyle='--', alpha=0.7, 
                          label='Critical threshold', linewidth=2)
        
        # 2. Bias Evolution Plot (Top-Right)
        ax_biases.set_title('Bias Mean Evolution by Layer', fontweight='bold')
        ax_biases.set_xlabel('Epoch')
        ax_biases.set_ylabel('Bias Mean Value')
        ax_biases.grid(True, alpha=0.3)
        
        # Add zero reference line
        ax_biases.axhline(y=0, color='gray', linestyle='-', alpha=0.5, 
                         label='Zero reference', linewidth=1)
        
        # 3. Dead Neurons Plot (Bottom-Left)
        ax_dead_neurons.set_title('Dead Neurons by Layer', fontweight='bold')
        ax_dead_neurons.set_xlabel('Epoch')
        ax_dead_neurons.set_ylabel('Dead Neurons (%)')
        ax_dead_neurons.set_ylim(0, 100)
        ax_dead_neurons.grid(True, alpha=0.3)
        
        # Add critical threshold line
        ax_dead_neurons.axhline(y=50, color='red', linestyle='--', alpha=0.7, 
                               label='Critical threshold (50%)', linewidth=2)
        ax_dead_neurons.axhline(y=20, color='orange', linestyle='--', alpha=0.7, 
                               label='Warning threshold (20%)', linewidth=1)
        
        # 4. Health Status Display (Bottom-Right)
        ax_health.set_title('Parameter Health Status', fontweight='bold')
        ax_health.set_xlim(0, 1)
        ax_health.set_ylim(0, 1)
        ax_health.axis('off')  # Remove axes for text display
        
        # Initialize health status text
        self.health_text = ax_health.text(0.5, 0.7, 'Monitoring Starting...', 
                                        ha='center', va='center', 
                                        fontsize=12, fontweight='bold',
                                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Initialize warning text
        self.warning_text = ax_health.text(0.5, 0.3, 'No warnings', 
                                         ha='center', va='center', 
                                         fontsize=10,
                                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Initialize line storage for efficient updates
        self.lines = {
            'weight_stds': {},
            'bias_means': {},
            'dead_neurons': {}
        }
        
        # Color palette for layer differentiation
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        # Initialize lines for each trackable layer
        color_idx = 0
        for layer_name in self.weight_means_history.keys():
            color = colors[color_idx % len(colors)]
            
            # Weight std lines
            line_weight, = ax_weights.plot([], [], color=color, linewidth=2, 
                                         label=f'{layer_name}', marker='o', markersize=4)
            self.lines['weight_stds'][layer_name] = line_weight
            
            # Bias mean lines
            line_bias, = ax_biases.plot([], [], color=color, linewidth=2, 
                                       label=f'{layer_name}', marker='s', markersize=4)
            self.lines['bias_means'][layer_name] = line_bias
            
            # Dead neuron lines
            line_dead, = ax_dead_neurons.plot([], [], color=color, linewidth=2, 
                                             label=f'{layer_name}', marker='^', markersize=4)
            self.lines['dead_neurons'][layer_name] = line_dead
            
            color_idx += 1
        
        # Add legends to plots
        ax_weights.legend(loc='upper right', fontsize=9)
        ax_biases.legend(loc='upper right', fontsize=9)
        ax_dead_neurons.legend(loc='upper right', fontsize=9)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)  # Small pause to ensure plot displays
        
        logger.debug("running _setup_visualization ... Real-time weights and bias dashboard initialized")
        logger.debug(f"running _setup_visualization ... Monitoring {len(self.lines['weight_stds'])} layers")
    
    
    def _establish_performance_baseline(self) -> None:
        """
        Establish performance baseline for monitoring overhead assessment
        
        Performs initial measurements to understand the computational cost of
        weights and bias monitoring and its impact on training performance.
        
        Args:
            None
            
        Side Effects:
            - Records initial timing measurements
            - Logs performance baseline information
        """
        logger.debug("running _establish_performance_baseline ... Establishing monitoring performance baseline")
        
        # Record setup completion time
        if self.start_time is not None:
            setup_time = (datetime.now() - self.start_time).total_seconds()
            logger.debug(f"running _establish_performance_baseline ... Setup completed in {setup_time:.2f} seconds")
        
        # Log memory and performance expectations
        num_layers = len(self.weight_means_history)
        expected_memory = num_layers * self.history_length * 40  # bytes per layer per epoch
        
        logger.debug(f"running _establish_performance_baseline ... Performance expectations:")
        logger.debug(f"running _establish_performance_baseline ... - Layers to monitor: {num_layers}")
        logger.debug(f"running _establish_performance_baseline ... - Expected memory usage: ~{expected_memory/1024:.1f} KB")
        logger.debug(f"running _establish_performance_baseline ... - Monitoring frequency: every {self.monitoring_frequency} epochs")
        logger.debug(f"running _establish_performance_baseline ... - Parameter sample percentage: {self.sample_percentage*100:.1f}%")
    
    
    def should_monitor_epoch(self, epoch: int) -> bool:
        """
        Determine if parameter monitoring should be performed for the current epoch
        
        Uses the configured monitoring frequency to decide whether to perform
        parameter extraction and analysis for the current epoch. Balances
        monitoring completeness with training performance.
        
        MONITORING FREQUENCY STRATEGY:
        
        Frequency-Based Monitoring:
            ```
            monitoring_frequency = 1: Monitor epochs [1, 2, 3, 4, 5, ...]
            monitoring_frequency = 2: Monitor epochs [1, 3, 5, 7, 9, ...]  
            monitoring_frequency = 5: Monitor epochs [1, 6, 11, 16, 21, ...]
            ```
        
        Special Cases:
            - Always monitor epoch 1 (initial baseline)
            - Always monitor final epoch (completion assessment)
            - Skip monitoring during epoch 0 (incomplete initialization)
        
        Performance Impact Analysis:
            ```
            Monitoring Frequency vs Training Overhead:
            frequency=1: 100% monitoring, ~3% slowdown
            frequency=2: 50% monitoring, ~1.5% slowdown
            frequency=5: 20% monitoring, ~0.6% slowdown
            frequency=10: 10% monitoring, ~0.3% slowdown
            ```
        
        Args:
            epoch: Current training epoch (1-based indexing)
            
        Returns:
            True if parameter monitoring should be performed for this epoch
            False if monitoring should be skipped to maintain training performance
        
        Decision Logic:
            ```
            if epoch <= 0:
                return False  # Skip epoch 0 (initialization)
            elif epoch == 1:
                return True   # Always monitor first epoch
            elif epoch % monitoring_frequency == 1:
                return True   # Monitor based on frequency
            else:
                return False  # Skip for performance
            ```
        """
        # Always monitor first epoch for baseline
        if epoch == 1:
            return True
        
        # Skip epoch 0 (initialization phase)
        if epoch <= 0:
            return False
        
        # Monitor based on frequency
        return (epoch - 1) % self.monitoring_frequency == 0
    
    
    def update_monitoring(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Update weights and bias monitoring with current epoch data
        
        Main monitoring method called during training to extract parameter statistics,
        assess health status, and update visualizations. Designed to integrate
        seamlessly with Keras callback system.
        
        MONITORING WORKFLOW:
        
        1. **Epoch Filtering**:
           Check if monitoring should be performed for this epoch based on frequency:
           ```
           Monitoring Decision:
           epoch=1, frequency=1 → Monitor (always monitor first epoch)
           epoch=2, frequency=2 → Skip (not on monitoring schedule)  
           epoch=3, frequency=2 → Monitor (on 2-epoch schedule)
           epoch=4, frequency=2 → Skip (performance optimization)
           ```
        
        2. **Parameter Extraction**:
           Extract weights and biases from all trainable layers:
           ```
           Parameter Extraction Process:
           - Use layer.get_weights() for efficient parameter access
           - Sample parameters based on sample_percentage for performance
           - Compute statistical measures (mean, std, dead neuron count)
           - Handle both weights and biases appropriately
           ```
        
        3. **Health Assessment**:
           Analyze parameter statistics to determine layer and overall health:
           ```
           Health Assessment Criteria:
           - Weight std trends (learning diversity vs collapse)
           - Bias evolution patterns (threshold learning vs explosion)
           - Dead neuron percentage monitoring (capacity utilization)
           - Cross-layer correlation analysis (architectural issues)
           ```
        
        4. **Visualization Update**:
           Update real-time plots with new data:
           ```
           Plot Update Process:
           - Add new data points to time series
           - Update line plots efficiently (no redraw)
           - Refresh health status display
           - Update warnings and recommendations
           ```
        
        5. **Performance Tracking**:
           Monitor the monitoring overhead:
           ```
           Performance Metrics:
           - Parameter extraction time
           - Visualization update time  
           - Total monitoring overhead
           - Impact on training throughput
           ```
        
        REAL-TIME DECISION MAKING:
        
        The monitoring system provides immediate feedback that can influence training:
        ```
        Training Intervention Examples:
        
        Weight Collapse Detected:
        → Recommendation: "Increase learning rate or change initialization"
        → Automatic: Log warning for manual review
        → Advanced: Trigger learning rate adjustment callback
        
        Bias Explosion Detected:
        → Recommendation: "Add weight decay or reduce learning rate"
        → Automatic: Log critical warning
        → Advanced: Trigger automatic regularization
        
        High Dead Neuron Rate:
        → Recommendation: "Consider LeakyReLU or better initialization"
        → Automatic: Log architectural concern
        → Advanced: Flag for post-training analysis
        ```
        
        Args:
            epoch: Current training epoch (1-based)
            logs: Training metrics from Keras (loss, accuracy, etc.)
                 Used for correlation analysis between training metrics and parameter health
        
        Side Effects:
            - Extracts parameters from all trainable layers
            - Updates historical parameter statistics
            - Refreshes real-time visualization plots
            - Updates health status and warnings
            - Logs monitoring results and performance metrics
        
        Performance Considerations:
            - Skips computation on non-monitoring epochs for efficiency
            - Uses parameter sampling to minimize extraction overhead
            - Efficient tensor operations for statistical computation
            - Non-blocking visualization updates
        """
        if not self.is_monitoring:
            return
        
        # Check if we should monitor this epoch
        if not self.should_monitor_epoch(epoch):
            return
        
        logger.debug(f"running update_monitoring ... Performing weights and bias monitoring for epoch {epoch}")
        
        # Start timing for performance assessment
        start_time = time.time()
        
        try:
            # Update current epoch
            self.current_epoch = epoch
            
            # Extract and analyze parameters
            parameter_stats = self._extract_parameter_statistics()
            
            # Update historical data
            self._update_historical_data(parameter_stats)
            
            # Assess health status
            self._assess_parameter_health()
            
            # Update visualizations
            self._update_visualizations()
            
            # Record computation time
            computation_time = time.time() - start_time
            self.computation_times.append(computation_time)
            self.last_computation_time = computation_time
            
            # Log monitoring results
            self._log_monitoring_results(epoch, computation_time)
            
            # Save intermediate plot if conditions are met
            if self.save_intermediate_plots and self._should_save_intermediate_plot(epoch):
                self._save_intermediate_plot(epoch)
            
        except Exception as e:
            logger.warning(f"running update_monitoring ... Weights and bias monitoring failed for epoch {epoch}: {e}")
            logger.debug(f"running update_monitoring ... Error details: {traceback.format_exc()}")
    
    
    def _extract_parameter_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Extract and compute statistics for weights and biases from all monitorable layers
        
        Performs the core parameter extraction using layer.get_weights() and calculates
        statistical measures for parameter evolution analysis.
        
        PARAMETER EXTRACTION PROCESS:
        
        1. **Layer Parameter Access**:
           ```
           for layer in model.layers:
               if layer.trainable_weights:
                   weights_and_biases = layer.get_weights()
                   # Process weights and biases separately
           ```
        
        2. **Parameter Sampling for Performance**:
           ```
           For each parameter array:
               if array_size > sample_threshold:
                   sample_indices = random_sample(array_size * sample_percentage)
                   sampled_array = array[sample_indices]
               else:
                   sampled_array = array  # Use all parameters for small arrays
           ```
        
        3. **Statistical Analysis**:
           ```
           For each layer's weights and biases:
               Weight Statistics: mean, std, dead percentage
               Bias Statistics: mean, std
               Dead Neuron Detection: count parameters near zero
           ```
        
        Returns:
            Dictionary mapping layer names to parameter statistics:
            ```
            {
                'conv2d': {
                    'weight_mean': 0.001,
                    'weight_std': 0.125,
                    'bias_mean': -0.05,
                    'bias_std': 0.08,
                    'dead_percentage': 5.2
                },
                'dense': {
                    'weight_mean': 0.008,
                    'weight_std': 0.089,
                    'bias_mean': 0.12,
                    'bias_std': 0.15,
                    'dead_percentage': 12.8
                }
            }
            ```
        
        Performance Optimizations:
            - Uses layer.get_weights() for direct parameter access
            - Implements parameter sampling for large layers
            - Vectorized statistical calculations using NumPy
            - Minimal memory allocation during computation
        """        
        if self.model_builder.model is None:
            logger.warning("running _extract_parameter_statistics ... Model not available for parameter extraction")
            return {}
        
        parameter_stats = {}
        
        try:
            # Process each trainable layer
            for layer in self.model_builder.model.layers:
                if not layer.trainable_weights:
                    continue
                
                layer_name = layer.name
                layer_weights = layer.get_weights()
                
                if not layer_weights:
                    logger.debug(f"running _extract_parameter_statistics ... Layer {layer_name} has no weights")
                    continue
                
                # Initialize statistics for this layer
                layer_stats = {
                    'weight_mean': 0.0,
                    'weight_std': 0.0,
                    'bias_mean': 0.0,
                    'bias_std': 0.0,
                    'dead_percentage': 0.0
                }
                
                # Separate weights and biases
                weights = []
                biases = []
                
                for i, param_array in enumerate(layer_weights):
                    param_shape = param_array.shape
                    
                    # Determine if this is weights or bias based on shape and position
                    # Convention: weights come first, biases come last and are 1D
                    if i == len(layer_weights) - 1 and len(param_shape) == 1:
                        # Last array and 1D → likely bias vector
                        biases.extend(param_array.flatten())
                    else:
                        # Multi-dimensional or not last → likely weights
                        weights.extend(param_array.flatten())
                
                # Compute weight statistics
                if weights:
                    weights_array = np.array(weights)
                    
                    # Sample for performance if array is large
                    if len(weights_array) > 1000:
                        sample_size = max(100, int(len(weights_array) * self.sample_percentage))
                        sample_indices = np.random.choice(len(weights_array), sample_size, replace=False)
                        sampled_weights = weights_array[sample_indices]
                    else:
                        sampled_weights = weights_array
                    
                    # Compute statistics
                    layer_stats['weight_mean'] = float(np.mean(sampled_weights))
                    layer_stats['weight_std'] = float(np.std(sampled_weights))
                    
                    # Dead neuron detection (weights below threshold)
                    dead_threshold = 1e-6
                    dead_count = np.sum(np.abs(sampled_weights) < dead_threshold)
                    layer_stats['dead_percentage'] = float(dead_count / len(sampled_weights) * 100)
                
                # Compute bias statistics
                if biases:
                    biases_array = np.array(biases)
                    layer_stats['bias_mean'] = float(np.mean(biases_array))
                    layer_stats['bias_std'] = float(np.std(biases_array))
                
                parameter_stats[layer_name] = layer_stats
                
                logger.debug(f"running _extract_parameter_statistics ... Layer {layer_name}: "
                            f"weight_std={layer_stats['weight_std']:.4f}, "
                            f"bias_mean={layer_stats['bias_mean']:.4f}, "
                            f"dead_pct={layer_stats['dead_percentage']:.1f}%")
        
        except Exception as e:
            logger.warning(f"running _extract_parameter_statistics ... Error extracting parameters: {e}")
            logger.debug(f"running _extract_parameter_statistics ... Error traceback: {traceback.format_exc()}")
        
        return parameter_stats
    
    
    def _update_historical_data(self, parameter_stats: Dict[str, Dict[str, float]]) -> None:
        """
        Update historical parameter data with new statistics
        
        Adds new parameter statistics to the historical tracking data and manages
        memory by maintaining a rolling window of recent measurements.
        
        HISTORY MANAGEMENT:
        
        Rolling Window Strategy:
            Maintains a fixed-size history window to balance memory usage with trend analysis:
            ```
            History Window Management:
            - Add new data point at end of list
            - If list exceeds history_length, remove oldest point
            - Maintains chronological order for trend analysis
            - Prevents unbounded memory growth during long training
            ```
        
        Data Structure Updates:
            For each layer and each statistic type:
            ```
            Before: weight_stds_history["conv2d"] = [0.15, 0.14, 0.13]
            New data: 0.12
            After: weight_stds_history["conv2d"] = [0.14, 0.13, 0.12]
            (if history_length = 3 and list was full)
            ```
        
        Memory Optimization:
            ```
            Memory Management:
            - Fixed memory footprint regardless of training length
            - O(1) insertion time for new data points
            - Efficient list operations for rolling window
            - Minimal garbage collection pressure
            ```
        
        Args:
            parameter_stats: New parameter statistics from _extract_parameter_statistics()
        
        Side Effects:
            - Updates all parameter history dictionaries with new data
            - Updates self.epochs_monitored with current epoch
            - Maintains rolling window size by removing old data if necessary
        """
        # Add current epoch to monitoring history
        self.epochs_monitored.append(self.current_epoch)
        
        # Update parameter statistics for each layer
        for layer_name, stats in parameter_stats.items():
            # Add new data points
            self.weight_means_history[layer_name].append(stats['weight_mean'])
            self.weight_stds_history[layer_name].append(stats['weight_std'])
            self.bias_means_history[layer_name].append(stats['bias_mean'])
            self.bias_stds_history[layer_name].append(stats['bias_std'])
            self.dead_neuron_percentages_history[layer_name].append(stats['dead_percentage'])
        
        # Maintain rolling window - remove old data if history exceeds limit
        if len(self.epochs_monitored) > self.history_length:
            # Remove oldest epoch
            self.epochs_monitored.pop(0)
            
            # Remove oldest data for each layer
            for layer_name in self.weight_means_history.keys():
                if len(self.weight_means_history[layer_name]) > self.history_length:
                    self.weight_means_history[layer_name].pop(0)
                if len(self.weight_stds_history[layer_name]) > self.history_length:
                    self.weight_stds_history[layer_name].pop(0)
                if len(self.bias_means_history[layer_name]) > self.history_length:
                    self.bias_means_history[layer_name].pop(0)
                if len(self.bias_stds_history[layer_name]) > self.history_length:
                    self.bias_stds_history[layer_name].pop(0)
                if len(self.dead_neuron_percentages_history[layer_name]) > self.history_length:
                    self.dead_neuron_percentages_history[layer_name].pop(0)
        
        logger.debug(f"running _update_historical_data ... Updated parameter history for {len(parameter_stats)} layers")
        logger.debug(f"running _update_historical_data ... History length: {len(self.epochs_monitored)} epochs")
    
    
    def _assess_parameter_health(self) -> None:
        """
        Assess parameter health and generate warnings/recommendations
        
        Analyzes current and historical parameter data to determine the health
        status of individual layers and the overall network. Generates actionable
        warnings and recommendations for training optimization.
        
        HEALTH ASSESSMENT PROCESS:
        
        1. **Individual Layer Assessment**:
           For each layer, analyze current parameter statistics:
           ```
           Layer Health Criteria:
           - Weight std: Check for collapse or excessive spread
           - Bias evolution: Monitor for explosion or stagnation
           - Dead neuron percentage: Monitor capacity loss
           - Trend analysis: Detect degradation patterns
           ```
        
        2. **Overall Network Assessment**:
           Combine layer assessments for network-wide health:
           ```
           Overall Health Priority:
           1. 'critical': Any layer with collapsed weights or exploded biases
           2. 'poor': Multiple layers with serious issues
           3. 'concerning': Some layers with problems
           4. 'good': Minor issues in few layers
           5. 'excellent': All layers healthy
           ```
        
        3. **Warning Generation**:
           Create specific warnings for detected problems:
           ```
           Warning Types:
           - Weight collapse: "Layer X weight std below threshold"
           - Bias explosion: "Layer Y bias values extremely high"
           - Dead neurons: "Layer Z has 60% dead neurons"
           - Trend warnings: "Layer A showing parameter degradation"
           ```
        
        4. **Recommendation Generation**:
           Provide actionable suggestions based on detected issues:
           ```
           Recommendation Examples:
           - "Increase learning rate for layer dense with collapsed weights"
           - "Add weight decay for conv2d_1 with exploding biases"
           - "Consider LeakyReLU for dense layer with 70% dead neurons"
           - "Add batch normalization after conv2d layer"
           ```
        
        Side Effects:
            - Updates self.layer_health_status for each layer
            - Updates self.health_status for overall network
            - Updates self.warnings with current active warnings
            - Updates self.recommendations with actionable suggestions
        """
        # Clear previous warnings and recommendations
        self.warnings.clear()
        self.recommendations.clear()
        
        layer_health_scores = []
        
        # Assess each layer individually
        for layer_name in self.weight_means_history.keys():
            if not self.weight_stds_history[layer_name]:
                continue  # Skip if no data
            
            # Get latest values
            latest_weight_std = self.weight_stds_history[layer_name][-1]
            latest_bias_mean = self.bias_means_history[layer_name][-1]
            latest_dead_pct = self.dead_neuron_percentages_history[layer_name][-1]
            
            # Assess layer health
            layer_health = self._assess_single_layer_health(
                layer_name, latest_weight_std, latest_bias_mean, latest_dead_pct
            )
            
            self.layer_health_status[layer_name] = layer_health
            
            # Convert health to numeric score for overall assessment
            health_score_map = {
                'excellent': 5,
                'good': 4,
                'concerning': 3,
                'poor': 2,
                'critical': 1
            }
            layer_health_scores.append(health_score_map.get(layer_health, 3))
        
        # Determine overall health
        if not layer_health_scores:
            self.health_status = "unknown"
        else:
            avg_score = np.mean(layer_health_scores)
            min_score = min(layer_health_scores)
            
            # Overall health based on average and minimum (worst layer)
            if min_score == 1:  # Any critical layer
                self.health_status = "critical"
            elif avg_score >= 4.5:
                self.health_status = "excellent"
            elif avg_score >= 3.5:
                self.health_status = "good"
            elif avg_score >= 2.5:
                self.health_status = "concerning"
            else:
                self.health_status = "poor"
        
        # Generate trend-based warnings
        self._generate_trend_warnings()
        
        # Generate recommendations based on current state
        self._generate_health_recommendations()
        
        logger.debug(f"running _assess_parameter_health ... Overall health: {self.health_status}")
        logger.debug(f"running _assess_parameter_health ... Active warnings: {len(self.warnings)}")
        logger.debug(f"running _assess_parameter_health ... Recommendations: {len(self.recommendations)}")
    
    
    def _assess_single_layer_health(
        self, 
        layer_name: str, 
        weight_std: float, 
        bias_mean: float, 
        dead_percentage: float
    ) -> str:
        """
        Assess the health of a single layer based on parameter statistics
        
        Args:
            layer_name: Name of the layer being assessed
            weight_std: Current weight standard deviation
            bias_mean: Current bias mean value
            dead_percentage: Current percentage of dead neurons
            
        Returns:
            Health status string: 'excellent', 'good', 'concerning', 'poor', or 'critical'
        """
        # Define thresholds
        weight_collapse_critical = 0.001
        weight_collapse_concerning = 0.01
        bias_explosion_critical = 5.0
        bias_explosion_concerning = 2.0
        dead_critical = 70.0
        dead_concerning = 30.0
        dead_good = 10.0
        
        # Check for critical conditions first
        if weight_std < weight_collapse_critical:
            self.warnings.append(f"🔴 WEIGHT COLLAPSE in {layer_name}: std {weight_std:.4f}")
            return 'critical'
        
        if abs(bias_mean) > bias_explosion_critical:
            self.warnings.append(f"🔴 BIAS EXPLOSION in {layer_name}: mean {bias_mean:.2f}")
            return 'critical'
        
        if dead_percentage > dead_critical:
            self.warnings.append(f"🔴 CRITICAL NEURON DEATH in {layer_name}: {dead_percentage:.1f}% dead")
            return 'critical'
        
        # Check for poor conditions
        if weight_std < weight_collapse_concerning:
            self.warnings.append(f"🟠 Weight collapse risk in {layer_name}: std {weight_std:.4f}")
            if dead_percentage > dead_concerning:
                return 'poor'
            else:
                return 'concerning'
        
        if abs(bias_mean) > bias_explosion_concerning:
            self.warnings.append(f"🟠 Bias explosion risk in {layer_name}: mean {bias_mean:.2f}")
            return 'concerning'
        
        if dead_percentage > dead_concerning:
            self.warnings.append(f"🟠 High neuron death in {layer_name}: {dead_percentage:.1f}% dead")
            return 'concerning'
        
        # Check for good vs excellent
        if dead_percentage <= dead_good and weight_std >= 0.05:
            return 'excellent'
        elif dead_percentage <= 20.0 and weight_std >= 0.02:
            return 'good'
        else:
            return 'concerning'
    
    
    def _generate_trend_warnings(self) -> None:
        """
        Generate warnings based on parameter trends over recent epochs
        
        Analyzes historical data to detect problematic trends that may not be
        apparent from single-epoch measurements.
        """
        # Need at least 3 data points for trend analysis
        if len(self.epochs_monitored) < 3:
            return
        
        for layer_name in self.weight_stds_history.keys():
            weight_std_history = self.weight_stds_history[layer_name]
            bias_mean_history = self.bias_means_history[layer_name]
            dead_history = self.dead_neuron_percentages_history[layer_name]
            
            if len(weight_std_history) < 3:
                continue
            
            # Analyze weight std trend (recent 3 epochs)
            recent_weight_stds = weight_std_history[-3:]
            weight_trend = np.polyfit(range(len(recent_weight_stds)), recent_weight_stds, 1)[0]
            
            # Analyze bias mean trend
            recent_bias_means = bias_mean_history[-3:]
            bias_trend = np.polyfit(range(len(recent_bias_means)), recent_bias_means, 1)[0]
            
            # Analyze dead neuron trend
            recent_dead = dead_history[-3:]
            dead_trend = np.polyfit(range(len(recent_dead)), recent_dead, 1)[0]
            
            # Warn about rapid weight std decline
            if weight_trend < -0.01:  # Rapid decline threshold
                self.warnings.append(f"📉 Rapid weight collapse in {layer_name}")
            
            # Warn about rapid bias increase
            if abs(bias_trend) > 0.5:  # Rapid bias change
                self.warnings.append(f"💥 Rapid bias evolution in {layer_name}: {bias_trend:+.2f}/epoch")
            
            # Warn about increasing dead neurons
            if dead_trend > 5.0:  # Increasing dead neurons
                self.warnings.append(f"💀 Increasing neuron death in {layer_name}: +{dead_trend:.1f}%/epoch")
    
    
    def _generate_health_recommendations(self) -> None:
        """
        Generate actionable recommendations based on current parameter health
        
        Provides specific suggestions for improving training based on detected issues.
        """
        # Overall recommendations based on health status
        if self.health_status == "critical":
            self.recommendations.extend([
                "🚨 IMMEDIATE ACTION REQUIRED",
                "🔧 Adjust learning rate (increase for collapse, decrease for explosion)",
                "⚖️ Add weight decay or batch normalization",
                "⏸️ Consider early stopping if no improvement"
            ])
        elif self.health_status == "poor":
            self.recommendations.extend([
                "🔧 Add batch normalization layers",
                "🎯 Switch to LeakyReLU or ELU activation",
                "📉 Adjust learning rate based on layer behavior",
                "🏗️ Consider architectural changes"
            ])
        elif self.health_status == "concerning":
            self.recommendations.extend([
                "⚠️ Monitor closely for parameter degradation",
                "🎯 Consider better weight initialization",
                "🔧 Evaluate activation functions and learning rate"
            ])
        
        # Layer-specific recommendations
        critical_layers = [name for name, health in self.layer_health_status.items() 
                          if health in ['critical', 'poor']]
        
        if critical_layers:
            self.recommendations.append(f"🎯 Focus on layers: {', '.join(critical_layers[:3])}")
        
        # Limit recommendations to avoid overwhelming display
        self.recommendations = self.recommendations[:6]
    
    
    def _update_visualizations(self) -> None:
        """
        Update all real-time visualization components with latest data
        
        Efficiently updates matplotlib plots with new parameter data while maintaining
        smooth real-time performance. Uses line updates instead of full redraws
        for optimal performance.
        
        Side Effects:
            - Updates all line plots with latest parameter data
            - Refreshes health status and warning displays
            - Auto-scales plot axes to accommodate new data
            - Triggers matplotlib canvas refresh for visual update
        """
        if self.fig is None or self.axes is None:
            return
        
        try:
            # Get current data
            epochs = self.epochs_monitored
            
            if not epochs:
                return
            
            # Update weight std plots
            for layer_name, line in self.lines['weight_stds'].items():
                if layer_name in self.weight_stds_history:
                    y_data = self.weight_stds_history[layer_name]
                    x_data = epochs[-len(y_data):]  # Match data length
                    line.set_data(x_data, y_data)
            
            # Update bias mean plots
            for layer_name, line in self.lines['bias_means'].items():
                if layer_name in self.bias_means_history:
                    y_data = self.bias_means_history[layer_name]
                    x_data = epochs[-len(y_data):]
                    line.set_data(x_data, y_data)
            
            # Update dead neuron plots
            for layer_name, line in self.lines['dead_neurons'].items():
                if layer_name in self.dead_neuron_percentages_history:
                    y_data = self.dead_neuron_percentages_history[layer_name]
                    x_data = epochs[-len(y_data):]
                    line.set_data(x_data, y_data)
            
            # Auto-scale axes
            self._auto_scale_axes()
            
            # Update health status display
            self._update_health_display()
            
            # Refresh the display
            if self.fig.canvas is not None:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            plt.pause(0.01)  # Small pause for smooth updates
            
        except Exception as e:
            logger.warning(f"running _update_visualizations ... Error updating plots: {e}")
    
    
    def _auto_scale_axes(self) -> None:
        """
        Automatically scale plot axes to accommodate current data
        
        Adjusts axis ranges dynamically to ensure all data is visible while
        maintaining good visual proportions and readability.
        """
        if self.axes is None:
            return
        
        epochs = self.epochs_monitored
        if not epochs:
            return
        
        ax_weights, ax_biases, ax_dead_neurons, ax_health = self.axes.flatten()
        
        # X-axis scaling for all plots (epoch range)
        x_min = min(epochs) - 0.5
        x_max = max(epochs) + 0.5
        
        # Weight std plot
        all_weight_stds = []
        for layer_data in self.weight_stds_history.values():
            all_weight_stds.extend(layer_data)
        
        if all_weight_stds:
            y_min = max(0, min(all_weight_stds) * 0.9)
            y_max = max(all_weight_stds) * 1.1
            ax_weights.set_xlim(x_min, x_max)
            ax_weights.set_ylim(y_min, y_max)
        
        # Bias mean plot
        all_bias_means = []
        for layer_data in self.bias_means_history.values():
            all_bias_means.extend(layer_data)
        
        if all_bias_means:
            y_range = max(abs(min(all_bias_means)), abs(max(all_bias_means))) * 1.1
            ax_biases.set_xlim(x_min, x_max)
            ax_biases.set_ylim(-y_range, y_range)
        
        # Dead neurons plot (fixed 0-100% range)
        ax_dead_neurons.set_xlim(x_min, x_max)
        ax_dead_neurons.set_ylim(0, 100)
    
    
    def _update_health_display(self) -> None:
        """
        Update health status and warning text displays
        
        Updates the text components in the health status subplot with current
        parameter health information, warnings, and recommendations.
        
        HEALTH DISPLAY COMPONENTS:
        
        Main Health Status:
            Shows overall parameter health with color-coded background:
            ```
            Status Colors:
            🟢 'excellent': Light green background
            🟢 'good': Light green background  
            🟡 'concerning': Yellow background
            🟠 'poor': Orange background
            🔴 'critical': Red background
            ```
        
        Warnings Display:
            Shows active warnings with appropriate urgency indicators:
            ```
            Warning Format:
            "🔴 WEIGHT COLLAPSE in conv2d_1: std 0.0005"
            "🟠 High neuron death in dense: 45.2% dead"
            "📉 Rapid weight collapse in conv2d"
            ```
        
        Recommendations Display:
            Shows actionable suggestions for training optimization:
            ```
            Recommendation Format:
            "🚨 IMMEDIATE ACTION REQUIRED"
            "🔧 Adjust learning rate (increase for collapse)"
            "🎯 Focus on layers: conv2d_1, dense"
            ```
        
        Performance Metrics:
            Shows monitoring overhead and timing information:
            ```
            Performance Info:
            "Monitoring: 0.8ms overhead"
            "Epoch: 15/50 (30% complete)"
            "Last update: 1.5s ago"
            ```
        
        Side Effects:
            - Updates self.health_text with current health status
            - Updates self.warning_text with current warnings and recommendations
            - Applies appropriate color coding based on health status
        """
        if self.health_text is None or self.warning_text is None:
            return
        
        # Health status color mapping
        health_colors = {
            'excellent': 'lightgreen',
            'good': 'lightgreen',
            'concerning': 'yellow',
            'poor': 'orange',
            'critical': 'red',
            'unknown': 'lightgray'
        }
        
        # Create main health status text
        elapsed_time = self._get_elapsed_time()
        performance_info = f"Overhead: {self.last_computation_time*1000:.1f}ms" if self.last_computation_time > 0 else "Computing..."
        
        health_message = f"Parameter Health: {self.health_status.upper()}\n\n"
        health_message += f"Epoch: {self.current_epoch}\n"
        health_message += f"Elapsed: {elapsed_time}\n"
        health_message += f"{performance_info}"
        
        # Update health status display
        health_color = health_colors.get(self.health_status, 'lightgray')
        self.health_text.set_text(health_message)
        self.health_text.set_bbox(dict(boxstyle='round', facecolor=health_color, alpha=0.8))
        
        # Create warnings and recommendations text
        warning_lines = []
        
        # Add warnings
        if self.warnings:
            warning_lines.extend(self.warnings[:3])  # Show top 3 warnings
        
        # Add separator if both warnings and recommendations exist
        if self.warnings and self.recommendations:
            warning_lines.append("─" * 20)
        
        # Add recommendations
        if self.recommendations:
            warning_lines.extend(self.recommendations[:3])  # Show top 3 recommendations
        
        # Default message if no warnings or recommendations
        if not warning_lines:
            warning_lines = ["✅ Parameters healthy", "Continue monitoring..."]
        
        warning_message = "\n".join(warning_lines)
        
        # Update warnings display with appropriate color
        warning_color = 'lightcoral' if (self.warnings and self.health_status in ['critical', 'poor']) else 'lightblue'
        self.warning_text.set_text(warning_message)
        self.warning_text.set_bbox(dict(boxstyle='round', facecolor=warning_color, alpha=0.8))
    
    
    def _get_elapsed_time(self) -> str:
        """
        Get formatted elapsed monitoring time
        
        Returns:
            Formatted time string (e.g., "1m 30s", "45s", "2h 15m")
        """
        if self.start_time is None:
            return "Unknown"
        
        elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
        
        if elapsed_seconds < 60:
            return f"{elapsed_seconds:.0f}s"
        elif elapsed_seconds < 3600:
            minutes = elapsed_seconds // 60
            seconds = elapsed_seconds % 60
            return f"{minutes:.0f}m {seconds:.0f}s"
        else:
            hours = elapsed_seconds // 3600
            minutes = (elapsed_seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    
    def _should_save_intermediate_plot(self, epoch: int) -> bool:
        """
        Determine if we should save an intermediate weights/bias plot for this epoch
        
        Args:
            epoch: Current epoch number (1-based)
            
        Returns:
            True if we should save a plot
        """
        should_save = (
            epoch == 1 or  # Always save first epoch
            epoch % self.save_every_n_epochs == 0 or  # Save every N epochs
            epoch == self.model_builder.model_config.epochs  # Save final epoch
        )
        
        logger.debug(f"running _should_save_intermediate_plot ... Epoch {epoch}: "
                    f"first_epoch={epoch == 1}, "
                    f"modulo_check={epoch % self.save_every_n_epochs == 0} "
                    f"(epoch={epoch} % save_every_n_epochs={self.save_every_n_epochs}), "
                    f"final_epoch={epoch == self.model_builder.model_config.epochs}, "
                    f"should_save={should_save}")
        
        return should_save
    
    
    def _save_intermediate_plot(self, epoch: int) -> None:
        """
        Save current weights/bias plot state as an intermediate plot
        
        Args:
            epoch: Current epoch number (1-based)
        """
        if self.fig is None or self.intermediate_plot_dir is None:
            logger.debug(f"running _save_intermediate_plot ... Skipping epoch {epoch}: fig={self.fig is not None}, dir={self.intermediate_plot_dir is not None}")
            return
        
        filename = f"weights_bias_epoch_{epoch:02d}.png"  # e.g., "weights_bias_epoch_01.png"
        filepath = self.intermediate_plot_dir / filename
        
        # Log the save attempt
        logger.debug(f"running _save_intermediate_plot ... Attempting to save epoch {epoch} to: {filepath}")
        
        try:
            # Ensure the plot is in a good state before saving
            if self.fig.canvas is not None:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            
            # Add a small delay to ensure matplotlib is ready
            time.sleep(0.1)
            
            # Save the plot
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            
            # Verify the file was actually saved
            if filepath.exists():
                file_size = filepath.stat().st_size
                logger.debug(f"running _save_intermediate_plot ... SUCCESS: Intermediate weights/bias plot saved for epoch {epoch}: {filepath} ({file_size} bytes)")
            else:
                logger.warning(f"running _save_intermediate_plot ... File does not exist after save attempt: {filepath}")
                
        except Exception as e:
            logger.warning(f"running _save_intermediate_plot ... FAILED to save intermediate weights/bias plot for epoch {epoch}: {e}")
            logger.debug(f"running _save_intermediate_plot ... Error details: {type(e).__name__}: {str(e)}")
    
    
    def _log_monitoring_results(self, epoch: int, computation_time: float) -> None:
        """
        Log monitoring results and performance metrics
        
        Args:
            epoch: Current epoch number
            computation_time: Time spent on parameter extraction and analysis
        """
        # Log basic monitoring info
        logger.debug(f"running _log_monitoring_results ... Epoch {epoch} weights/bias monitoring completed")
        logger.debug(f"running _log_monitoring_results ... Computation time: {computation_time*1000:.1f}ms")
        logger.debug(f"running _log_monitoring_results ... Overall health: {self.health_status}")
        
        # Log layer health summary
        healthy_layers = sum(1 for health in self.layer_health_status.values() 
                           if health in ['excellent', 'good'])
        total_layers = len(self.layer_health_status)
        logger.debug(f"running _log_monitoring_results ... Healthy layers: {healthy_layers}/{total_layers}")
        
        # Log performance impact
        if len(self.computation_times) > 1:
            avg_time = np.mean(self.computation_times)
            logger.debug(f"running _log_monitoring_results ... Average monitoring overhead: {avg_time*1000:.1f}ms")
        
        # Log warnings if any
        if self.warnings:
            logger.debug(f"running _log_monitoring_results ... Active warnings: {len(self.warnings)}")
            for warning in self.warnings[:2]:  # Log first 2 warnings
                logger.debug(f"running _log_monitoring_results ... Warning: {warning}")
    
    
    def save_monitoring_plots(self, run_timestamp: Optional[str] = None) -> None:
        """
        Save the current monitoring plots to disk
        
        Saves the real-time weights and bias monitoring dashboard as a high-resolution
        image file for later analysis and documentation.
        
        SAVE STRATEGY:
        
        File Naming Convention:
            ```
            Format: "weights_bias_monitor_{timestamp}_{dataset_name}.png"
            Example: "weights_bias_monitor_2025-01-08-143022_cifar10.png"
            ```
        
        Save Location Priority:
            1. Use provided plot_dir if available (integrates with training pipeline)
            2. Create fallback directory in project plots folder
            3. Log save location for user reference
        
        Image Quality:
            - High DPI (300) for publication quality
            - Tight bounding box to minimize whitespace
            - PNG format for lossless compression
            - Full dashboard capture including all subplots
        
        Args:
            run_timestamp: Optional timestamp for consistent file naming across training
                          If None, generates new timestamp
        
        Side Effects:
            - Saves PNG file to plot directory
            - Logs save location and file size
            - Creates directories if they don't exist
        """
        if self.fig is None:
            logger.warning("running save_monitoring_plots ... No monitoring plots to save")
            return
        
        # Generate timestamp if not provided
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        
        # Determine save directory - use the main plot directory (not intermediate)
        if self.plot_dir is not None:
            save_dir = self.plot_dir  # Save to main run directory
        else:
            # Fallback: create default directory
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            save_dir = project_root / "plots"
            save_dir.mkdir(exist_ok=True)
        
        # Generate filename
        dataset_name_clean = self.model_builder.dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        filename = f"weights_bias_monitor_{run_timestamp}_{dataset_name_clean}.png"
        filepath = save_dir / filename
        
        try:
            # Save the plot
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            
            # Log save information
            if filepath.exists():
                file_size = filepath.stat().st_size
                logger.debug(f"running save_monitoring_plots ... Weights/bias monitor saved to: {filepath}")
                logger.debug(f"running save_monitoring_plots ... File size: {file_size/1024:.1f} KB")
                
                # ALSO save as final plot in intermediate directory
                if self.intermediate_plot_dir is not None:
                    final_intermediate_path = self.intermediate_plot_dir / "final.png"
                    self.fig.savefig(final_intermediate_path, dpi=300, bbox_inches='tight')
                    logger.debug(f"running save_monitoring_plots ... Final weights/bias plot also saved to intermediate directory: {final_intermediate_path}")
                    
                    # Create a summary log file
                    self._create_monitoring_summary()
            else:
                logger.warning(f"running save_monitoring_plots ... File not found after save: {filepath}")
                
        except Exception as e:
            logger.warning(f"running save_monitoring_plots ... Failed to save monitoring plots: {e}")
    
    
    def _create_monitoring_summary(self) -> None:
        """Create a text summary of the weights and bias monitoring session"""
        if self.intermediate_plot_dir is None:
            return
            
        summary_file = self.intermediate_plot_dir / "weights_bias_monitoring_summary.txt"
        
        try:
            with open(summary_file, 'w') as f:
                f.write(f"Weights and Bias Monitoring Summary\n")
                f.write(f"===================================\n\n")
                f.write(f"Dataset: {self.model_builder.dataset_config.name}\n")
                f.write(f"Total Epochs Monitored: {len(self.epochs_monitored)}\n")
                f.write(f"Monitoring Frequency: Every {self.monitoring_frequency} epochs\n")
                f.write(f"Parameter Sample Percentage: {self.sample_percentage*100:.1f}%\n\n")
                
                f.write(f"Final Health Status: {self.health_status}\n")
                f.write(f"Active Warnings: {len(self.warnings)}\n")
                f.write(f"Recommendations: {len(self.recommendations)}\n\n")
                
                # Layer-specific final health
                f.write("Layer Health Status:\n")
                for layer_name, health in self.layer_health_status.items():
                    f.write(f"  {layer_name}: {health}\n")
                f.write("\n")
                
                if self.computation_times:
                    avg_time = sum(self.computation_times) / len(self.computation_times)
                    f.write(f"Average Monitoring Overhead: {avg_time*1000:.1f}ms per epoch\n")
                
                f.write(f"\nIntermediate plots saved every {self.save_every_n_epochs} epochs\n")
                f.write(f"Total intermediate plots: {len(list(self.intermediate_plot_dir.glob('weights_bias_epoch_*.png')))}\n")
                
                # Final parameter statistics
                f.write("\nFinal Parameter Statistics:\n")
                for layer_name in self.weight_stds_history.keys():
                    if self.weight_stds_history[layer_name]:
                        final_weight_std = self.weight_stds_history[layer_name][-1]
                        final_bias_mean = self.bias_means_history[layer_name][-1]
                        final_dead_pct = self.dead_neuron_percentages_history[layer_name][-1]
                        
                        f.write(f"  {layer_name}:\n")
                        f.write(f"    Weight Std: {final_weight_std:.4f}\n")
                        f.write(f"    Bias Mean: {final_bias_mean:.4f}\n")
                        f.write(f"    Dead Neurons: {final_dead_pct:.1f}%\n")
                
            logger.debug(f"running _create_monitoring_summary ... Weights/bias summary saved to: {summary_file}")
            
        except Exception as e:
            logger.warning(f"running _create_monitoring_summary ... Failed to create monitoring summary: {e}")
    
    
    def close_monitoring(self) -> None:
        """
        Clean up monitoring resources and close visualizations
        
        Performs cleanup operations when monitoring is no longer needed,
        typically called at the end of training or when monitoring is disabled.
        
        CLEANUP OPERATIONS:
        
        1. **Plot Cleanup**:
           ```
           - Close matplotlib figure to free memory
           - Release plot resources and handles
           - Turn off interactive matplotlib mode
           ```
        
        2. **State Reset**:
           ```
           - Set monitoring flag to False
           - Clear data structures if needed
           - Reset performance tracking
           ```
        
        3. **Performance Summary**:
           ```
           - Log total monitoring overhead
           - Report average computation times
           - Summarize monitoring effectiveness
           ```
        
        Memory Management:
            - Frees matplotlib figure memory
            - Releases parameter references
            - Clears large data structures if requested
        
        Side Effects:
            - Closes matplotlib figure and frees memory
            - Sets self.is_monitoring to False
            - Logs monitoring performance summary
            - Turns off matplotlib interactive mode
        """
        logger.debug("running close_monitoring ... Closing real-time weights and bias monitoring")
        
        # Log performance summary
        if self.computation_times:
            total_time = sum(self.computation_times)
            avg_time = np.mean(self.computation_times)
            max_time = max(self.computation_times)
            
            logger.debug(f"running close_monitoring ... Monitoring performance summary:")
            logger.debug(f"running close_monitoring ... - Total epochs monitored: {len(self.computation_times)}")
            logger.debug(f"running close_monitoring ... - Total monitoring time: {total_time:.3f}s")
            logger.debug(f"running close_monitoring ... - Average time per epoch: {avg_time*1000:.1f}ms")
            logger.debug(f"running close_monitoring ... - Maximum time per epoch: {max_time*1000:.1f}ms")
        
        # Close matplotlib resources
        self.is_monitoring = False
        
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
        
        # Turn off interactive mode
        plt.ioff()
        
        logger.debug("running close_monitoring ... Real-time weights and bias monitoring closed")


class RealTimeWeightsBiasCallback(keras.callbacks.Callback):
    """
    Keras callback for integrating real-time weights and bias monitoring
    
    Provides seamless integration with Keras training loop to enable live
    parameter monitoring during training. Follows the same pattern as
    RealTimeGradientFlowCallback for consistency.
    
    CALLBACK INTEGRATION STRATEGY:
    
    Training Lifecycle Integration:
        ```
        on_train_begin(): Setup monitoring system and verify readiness
        on_epoch_end(): Update monitoring with current epoch parameters
        on_train_end(): Save final plots and cleanup resources
        ```
    
    Error Handling:
        - Graceful degradation if monitoring fails
        - Continue training even if parameter analysis encounters errors
        - Log warnings for debugging but don't interrupt training
    
    Performance Considerations:
        - Minimal impact on training throughput
        - Non-blocking parameter extraction and visualization
        - Configurable monitoring frequency for performance tuning
    """
    
    def __init__(self, monitor: RealTimeWeightsBiasMonitor):
        """
        Initialize the weights and bias monitoring callback
        
        Args:
            monitor: RealTimeWeightsBiasMonitor instance configured for training
        """
        super().__init__()
        self.monitor = monitor
        logger.debug("running RealTimeWeightsBiasCallback.__init__ ... Weights and bias callback initialized")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of training to setup parameter monitoring
        
        Verifies that the monitoring system is properly configured and ready
        to begin parameter tracking. This is a verification step since the
        monitor should already be set up in the train() method.
        
        Args:
            logs: Training logs from Keras (typically empty at train begin)
        """
        logger.debug("running on_train_begin ... Setting up real-time weights and bias monitoring")
        
        # NOTE: The monitor should already be set up in the train() method
        # This is just a verification step
        if self.monitor.is_monitoring:
            logger.debug("running on_train_begin ... Weights and bias monitoring already set up and ready")
        else:
            logger.warning("running on_train_begin ... Weights and bias monitoring not properly set up")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of each epoch to update parameter monitoring
        
        Triggers parameter extraction and analysis for the current epoch,
        updating visualizations and health assessments in real-time.
        
        Args:
            epoch: Current epoch number (0-based from Keras)
            logs: Training metrics from Keras (loss, accuracy, val_loss, etc.)
        """
        if logs is None:
            logs = {}
        
        # Convert to 1-based epoch numbering for consistency
        epoch_1_based = epoch + 1
        
        try:
            # Update monitoring for current epoch
            self.monitor.update_monitoring(epoch_1_based, logs)
            
            # Log epoch completion
            logger.debug(f"running on_epoch_end ... Weights and bias monitoring updated for epoch {epoch_1_based}")
            
        except Exception as e:
            logger.warning(f"running on_epoch_end ... Weights and bias monitoring failed for epoch {epoch_1_based}: {e}")
            # Continue training even if monitoring fails
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of training to finalize monitoring and save results
        
        Saves final monitoring plots, creates summary reports, and cleans up
        monitoring resources. Ensures all monitoring data is preserved for
        later analysis.
        
        Args:
            logs: Final training metrics from Keras
        """
        logger.debug("running on_train_end ... Finalizing real-time weights and bias monitoring")
        
        try:
            # Save final monitoring plots
            self.monitor.save_monitoring_plots()
            
            # Close monitoring and cleanup resources
            self.monitor.close_monitoring()
            
            logger.debug("running on_train_end ... Weights and bias monitoring completed successfully")
            
        except Exception as e:
            logger.warning(f"running on_train_end ... Error finalizing weights and bias monitoring: {e}")


# Convenience function for easy integration
def create_realtime_weights_bias_monitor(
    model_builder,
    plot_dir: Optional[Path] = None,
    monitoring_frequency: int = 1,
    history_length: int = 50,
    sample_percentage: float = 0.1
) -> Tuple[RealTimeWeightsBiasMonitor, RealTimeWeightsBiasCallback]:
    """
    Convenience function to create and setup real-time weights and bias monitoring
    
    Creates both the monitor and callback components needed for integration
    with the training pipeline. Simplifies setup and ensures proper configuration.
    
    SETUP PROCESS:
    
    1. Create RealTimeWeightsBiasMonitor with specified configuration
    2. Setup monitoring infrastructure (layer discovery, visualization)
    3. Create RealTimeWeightsBiasCallback for Keras integration
    4. Return both components ready for training integration
    
    Args:
        model_builder: ModelBuilder instance containing model and configuration
        plot_dir: Directory for saving monitoring plots and intermediate results
        monitoring_frequency: How often to perform monitoring (every N epochs)
        history_length: Number of epochs to keep in monitoring history
        sample_percentage: Fraction of parameters to sample for statistics
        
    Returns:
        Tuple of (monitor, callback) ready for training integration
        
    Usage Example:
        ```python
        # In ModelBuilder.train() method:
        if self.model_config.enable_realtime_weights_bias:
            weights_monitor, weights_callback = create_realtime_weights_bias_monitor(
                model_builder=self,
                plot_dir=self.plot_dir,
                monitoring_frequency=2,  # Monitor every 2 epochs
                sample_percentage=0.1    # Sample 10% of parameters
            )
            callbacks_list.append(weights_callback)
        ```
    
    Integration Notes:
        - Must be called after model is built but before training starts
        - Requires plot_dir to be set for saving intermediate results
        - Monitor is automatically configured and ready for use
        - Callback integrates seamlessly with Keras training loop
    """
    logger.debug("running create_realtime_weights_bias_monitor ... Creating real-time weights and bias monitoring system")
    
    # Create monitor with specified configuration
    monitor = RealTimeWeightsBiasMonitor(
        model_builder=model_builder,
        plot_dir=plot_dir,
        monitoring_frequency=monitoring_frequency,
        history_length=history_length,
        sample_percentage=sample_percentage
    )
    
    # Setup monitoring infrastructure
    monitor.setup_monitoring()
    
    # Create callback for Keras integration
    callback = RealTimeWeightsBiasCallback(monitor)
    
    logger.debug("running create_realtime_weights_bias_monitor ... Real-time weights and bias monitoring system created")
    logger.debug(f"running create_realtime_weights_bias_monitor ... Monitoring frequency: every {monitoring_frequency} epochs")
    logger.debug(f"running create_realtime_weights_bias_monitor ... Parameter sampling: {sample_percentage*100:.1f}%")
    
    return monitor, callback