"use client"

import React, { useEffect, useRef, useCallback } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';

// Register Dagre layout (always available)
cytoscape.use(dagre);

// Dynamically load and register ELK with error handling
let elkAvailable = false;
let elkLoadPromise: Promise<boolean> | null = null;

const loadELK = async (): Promise<boolean> => {
  if (elkLoadPromise) {
    return elkLoadPromise;
  }
  
  elkLoadPromise = (async () => {
    try {
      const elk = await import('cytoscape-elk');
      cytoscape.use(elk.default || elk);
      elkAvailable = true;
      console.log('ELK layout loaded successfully');
      return true;
    } catch (error) {
      console.warn('ELK layout failed to load, using Dagre fallback:', error);
      elkAvailable = false;
      return false;
    }
  })();
  
  return elkLoadPromise;
};

interface LayerData {
  id: string;
  type: string;
  label: string;
  parameters?: number;
  color_intensity?: number;
  opacity?: number;
  units?: number;
  filters?: number;
}

interface EdgeData {
  source: string;
  target: string;
  tensor_transform: string;
}

interface CytoscapeData {
  nodes: Array<{ data: LayerData }>;
  edges: Array<{ data: EdgeData }>;
  metadata?: {
    architecture_type: string;
    total_parameters: number;
  };
}

interface LayerTypeDefinition {
  color: string;
  shape: 'rounded-full' | 'rounded' | 'rounded-lg';
  label: string;
  order: number;
}

const LAYER_TYPE_DEFINITIONS: Record<string, LayerTypeDefinition> = {
  'input': {
    color: 'bg-green-600', // Matches #059669 from model-graph.tsx
    shape: 'rounded-full',
    label: 'Input',
    order: 1
  },
  'conv2d': {
    color: 'bg-blue-500', // Matches #3B82F6 from model-graph.tsx
    shape: 'rounded',
    label: 'Conv2D',
    order: 2
  },
  'conv': {
    color: 'bg-blue-500', // Matches #3B82F6 from model-graph.tsx
    shape: 'rounded',
    label: 'Conv2D',
    order: 2
  },
  'pooling': {
    color: 'bg-cyan-500', // Changed from emerald to cyan for better distinction from input
    shape: 'rounded-full',
    label: 'Pooling',
    order: 3
  },
  'maxpooling2d': {
    color: 'bg-cyan-500',
    shape: 'rounded-full',
    label: 'Pooling',
    order: 3
  },
  'lstm': {
    color: 'bg-purple-500', // Matches #8B5CF6 from model-graph.tsx
    shape: 'rounded-lg',
    label: 'LSTM',
    order: 4
  },
  'dense': {
    color: 'bg-orange-500', // Matches #F97316 from model-graph.tsx
    shape: 'rounded',
    label: 'Dense',
    order: 5
  },
  'linear': {
    color: 'bg-orange-500', // Same as dense
    shape: 'rounded',
    label: 'Dense',
    order: 5
  },
  'dropout': {
    color: 'bg-yellow-500', // Matches #EAB308 from model-graph.tsx
    shape: 'rounded',
    label: 'Dropout',
    order: 6
  },
  'output': {
    color: 'bg-red-500', // Matches #EF4444 from model-graph.tsx
    shape: 'rounded-full',
    label: 'Dense / output',
    order: 7
  }
};

const DEFAULT_LAYER_TYPE: LayerTypeDefinition = {
  color: 'bg-gray-400',
  shape: 'rounded',
  label: 'Unknown',
  order: 99
};

// Function to generate dynamic legend based on actual layers present
const generateDynamicLegend = (architectureData: CytoscapeData | null): LayerTypeDefinition[] => {
  if (!architectureData?.nodes || architectureData.nodes.length === 0) {
    return [];
  }
  
  // Extract unique layer types from the architecture data
  const presentTypes = new Set<string>();
  architectureData.nodes.forEach(node => {
    const layerType = node.data.type?.toLowerCase() || 'unknown';
    presentTypes.add(layerType);
    
    // Special case: check for output layer by ID (since it has type='dense' but id='output')
    if (node.data.id === 'output') {
      presentTypes.add('output');
    }
  });
  
  // Map to legend definitions
  const legendItems: LayerTypeDefinition[] = [];
  presentTypes.forEach(type => {
    const definition = LAYER_TYPE_DEFINITIONS[type] || {
      ...DEFAULT_LAYER_TYPE,
      label: type.charAt(0).toUpperCase() + type.slice(1)
    };
    legendItems.push(definition);
  });
  
  // Sort by flow order (input → conv → pool → lstm → dense → dropout → output)
  // then alphabetically by label for consistency
  return legendItems.sort((a, b) => {
    if (a.order !== b.order) return a.order - b.order;
    return a.label.localeCompare(b.label);
  });
};

// Legend Component
const ModelLegend: React.FC<{ architectureData: CytoscapeData | null }> = React.memo(({ architectureData }) => {
  const legendItems = generateDynamicLegend(architectureData);
  
  if (legendItems.length === 0) {
    return null;
  }
  
  return (
    <div className="absolute bottom-4 right-4 z-10 bg-gray-800 bg-opacity-90 rounded-lg p-3 max-w-xs">
      <div className="text-xs text-gray-300 mb-2 font-semibold">Legend</div>
      <div className="text-xs text-gray-400 mb-2">Layer Types</div>
      <div className="flex flex-col gap-1.5 text-xs">
        {legendItems.map((item) => (
          <div key={item.label} className="flex items-center gap-2">
            <div className={`w-3 h-3 ${item.color} ${item.shape} flex-shrink-0`}></div>
            <span className="text-gray-300">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
});

ModelLegend.displayName = 'ModelLegend';

interface ModelGraphProps {
  architectureData: CytoscapeData;
  onNodeClick?: (nodeData: LayerData) => void;
  className?: string;
  onPngExportRequest?: () => Promise<Blob | null>;
  showLegend?: boolean;
}

export const ModelGraph = React.forwardRef<
  { exportToPNG: () => Promise<Blob | null> },
  ModelGraphProps
>(({ architectureData, onNodeClick, className = "", onPngExportRequest, showLegend = false }, ref) => {
  const cyRef = useRef<cytoscape.Core | null>(null);
  
  // Type definitions for experimental web APIs
  interface SchedulerPostTaskOptions {
    priority: 'background' | 'user-visible' | 'user-blocking'
    delay?: number
  }
  
  interface WindowWithScheduler extends Window {
    scheduler?: {
      postTask: (callback: () => void, options: SchedulerPostTaskOptions) => void
    }
  }

  // Non-blocking scheduler to prevent React setTimeout violations
  const scheduleNonUrgent = useCallback((callback: () => void, delay: number = 0) => {
    const windowWithScheduler = window as WindowWithScheduler
    if (windowWithScheduler.scheduler?.postTask) {
      // Use modern Scheduler API if available
      windowWithScheduler.scheduler.postTask(callback, { priority: 'background', delay });
    } else if ('requestIdleCallback' in window) {
      // Fallback to requestIdleCallback
      requestIdleCallback(callback, { timeout: delay + 50 });
    } else {
      // Final fallback to setTimeout (but try to avoid React's main thread)
      setTimeout(callback, Math.max(delay, 0));
    }
  }, []);

  // Cytoscape.js stylesheet for educational neural network visualization
  const cytoscapeStylesheet = [
    // Default node styling
    {
      selector: 'node',
      style: {
        'background-color': '#374151',
        'label': 'data(label)',
        'text-valign': 'center',
        'text-halign': 'center',
        'color': 'white',
        'font-size': '10px',
        'font-weight': 'bold',
        'text-wrap': 'wrap',
        'text-max-width': '80px',
        'border-width': 2,
        'border-color': '#6B7280',
        'width': 60,  // Reduced from 80
        'height': 35, // Reduced from 40
        'shape': 'round-rectangle'
      }
    },
    // Input layer
    {
      selector: 'node[type="input"]',
      style: {
        'background-color': '#059669',
        'shape': 'ellipse',
        'width': 60,  // Reduced from 80
        'height': 35, // Reduced from 40
        'label': 'data(label)',
        'text-valign': 'center',
        'color': 'white',
        'font-weight': 'bold'
      }
    },
    // Convolutional layers
    {
      selector: 'node[type*="conv"]',
      style: {
        'background-color': '#3B82F6',
        'shape': 'rectangle',
        'width': 'mapData(filters, 8, 512, 50, 90)',  // Reduced max width
        'height': 40,  // Reduced from 50
        'label': 'data(label)',
        'text-valign': 'center',
        'color': 'white',
        'font-weight': 'bold',
        'background-opacity': 'mapData(color_intensity, 0, 1, 0.3, 1.0)'
      }
    },
    // Dense layers
    {
      selector: 'node[type="dense"]',
      style: {
        'background-color': '#F59E0B',
        'shape': 'round-rectangle',
        'width': 'mapData(units, 10, 2048, 45, 80)',  // Reduced sizes
        'height': 40,  // Reduced from 50
        'label': 'data(label)',
        'text-valign': 'center',
        'color': 'white',
        'font-weight': 'bold'
      }
    },
    // LSTM layers
    {
      selector: 'node[type="lstm"]',
      style: {
        'background-color': '#8B5CF6',
        'shape': 'round-rectangle',
        'width': 'mapData(units, 10, 512, 60, 110)',
        'height': 45,
        'label': 'data(label)',
        'text-valign': 'center',
        'color': 'white',
        'font-weight': 'bold'
      }
    },
    // Pooling layers
    {
      selector: 'node[type*="pool"]',
      style: {
        'background-color': '#06B6D4', // Changed from #10B981 (emerald) to #06B6D4 (cyan) for better distinction from input
        'shape': 'ellipse',
        'width': 60,
        'height': 40,
        'label': 'data(label)',
        'text-valign': 'center',
        'color': 'white',
        'font-size': '10px'
      }
    },
    // Flatten layers
    {
      selector: 'node[type="flatten"]',
      style: {
        'background-color': '#8B5CF6', // Purple for flatten layers
        'shape': 'rectangle',
        'width': 55,
        'height': 35,
        'label': 'data(label)',
        'text-valign': 'center',
        'color': 'white',
        'font-size': '10px'
      }
    },
    // Dropout layers
    {
      selector: 'node[type="dropout"]',
      style: {
        'background-color': '#EF4444',
        'shape': 'diamond',
        'width': 50,
        'height': 35,
        'label': 'data(label)',
        'text-valign': 'center',
        'color': 'white',
        'font-size': '9px'
      }
    },
    // Activation layers
    {
      selector: 'node[type="activation"]',
      style: {
        'background-color': '#84CC16',
        'shape': 'octagon',
        'width': 45,
        'height': 35,
        'label': 'data(label)',
        'text-valign': 'center',
        'color': 'white',
        'font-size': '9px'
      }
    },
    // Output layer (identified by ID, not type since it's still a dense layer)
    {
      selector: 'node[id="output"]',
      style: {
        'background-color': '#DC2626',
        'shape': 'ellipse',
        'width': 70,
        'height': 40,
        'label': 'data(label)',
        'text-valign': 'center',
        'color': 'white',
        'font-weight': 'bold'
      }
    },
    // Edges with tensor information - Enhanced for collision avoidance
    {
      selector: 'edge',
      style: {
        'curve-style': 'bezier',
        'control-point-step-size': 40,    // Better curve control for avoiding nodes
        
        // Arrow styling
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#64748B',
        'arrow-scale': 1.2,              // Slightly larger arrows for better visibility
        
        // Line styling  
        'line-color': '#64748B',
        'width': 2,
        
        // Show tensor transform labels by default
        'label': 'data(tensor_transform)',
        'font-size': '8px',
        'text-rotation': 'autorotate',
        'text-margin-x': 10,             // Horizontal offset from edge
        'text-margin-y': -15,            // Position above the edge (increased)
        'color': '#9CA3AF',
        
        // Improved label background for better visibility (when shown)
        'text-background-color': '#1F2937',
        'text-background-opacity': 0.9,  // Increased opacity
        'text-background-padding': '3px', // Increased padding
        'text-border-width': 1,          // Add border for better separation
        'text-border-color': '#374151'
      }
    },
    // Highlighted node (selected)
    {
      selector: 'node.highlighted',
      style: {
        'border-width': 4,
        'border-color': '#F59E0B',
        'border-opacity': 1,
        // Note: Cytoscape.js doesn't support box-shadow, using border styling instead
        'overlay-color': '#F59E0B',
        'overlay-opacity': 0.2,
        'overlay-padding': 4
      }
    },
    // Animation classes for simultaneous forward/backward flow
    {
      selector: '.flowing-data',
      style: {
        'line-color': '#10B981',  // Green for data flow
        'target-arrow-color': '#10B981',
        'target-arrow-shape': 'triangle',
        'width': 4,
        'opacity': 1
      }
    },
    {
      selector: '.flowing-errors',
      style: {
        'line-color': '#EF4444',  // Red for error/gradient flow
        'source-arrow-color': '#EF4444',
        'source-arrow-shape': 'triangle',
        'width': 4,
        'opacity': 1
      }
    },
    {
      selector: 'node.processing',
      style: {
        // Keep original colors, just add heavier border
        'border-color': '#FBBF24',  // Golden border for active state
        'border-width': 4,
        'border-opacity': 1
      }
    },
    // Enhanced tensor shape display
    {
      selector: '.tensor-info',
      style: {
        'font-size': '9px',
        'color': '#D1D5DB',
        'text-background-color': '#1F2937',
        'text-background-opacity': 0.95,
        'text-background-padding': '4px',
        'text-border-width': 1,
        'text-border-color': '#374151'
      }
    }
  ];

  // Multi-row layout algorithm for better space utilization
  const calculateOptimalLayout = (nodeCount: number) => {
    // Determine if we should use multi-row layout
    const shouldWrapRows = nodeCount > 6; // Wrap for models with >6 nodes
    
    if (shouldWrapRows) {
      // Calculate optimal row breaks for better readability
      const maxNodesPerRow = Math.max(3, Math.ceil(Math.sqrt(nodeCount * 1.5))); // 3-6 nodes per row typically
      const rowCount = Math.ceil(nodeCount / maxNodesPerRow);
      
      return {
        useMultiRow: true,
        maxNodesPerRow,
        rowCount,
        // Optimized spacing for multi-row layout
        rankSep: 80,  // Reduced horizontal spacing since we wrap
        nodeSep: 50,  // Good vertical spacing between rows
        edgeSep: 15,
        rowSep: 120   // Space between rows
      };
    } else {
      // Simple single-row layout for small models
      return {
        useMultiRow: false,
        rankSep: 100,  // Comfortable spacing for small models
        nodeSep: 60,
        edgeSep: 20
      };
    }
  };

  // Dynamic spacing calculation based on model complexity (kept for future hybrid layouts)
  // const calculateLayoutSpacing = (nodeCount: number, hasMultipleDense: boolean) => {
  //   const baseRankSep = 120;  // Increased base horizontal spacing
  //   const baseNodeSep = 60;   // Increased base vertical spacing
  //   const baseEdgeSep = 20;   // Increased base edge spacing
  //   
  //   // Increase spacing for complex models
  //   const complexityFactor = nodeCount > 8 ? 1.4 : 1.0;
  //   const denseFactor = hasMultipleDense ? 1.3 : 1.0;
  //   
  //   return {
  //     rankSep: Math.round(baseRankSep * complexityFactor),
  //     nodeSep: Math.round(baseNodeSep * denseFactor),
  //     edgeSep: Math.round(baseEdgeSep * complexityFactor)
  //   };
  // };

  // Analyze architecture to determine layout needs
  const nodeCount = architectureData?.nodes?.length || 0;
  const hasMultipleDense = architectureData?.nodes ? 
    architectureData.nodes.filter(node => node.data.type?.toLowerCase() === 'dense').length > 1 : false;
  
  const optimalLayout = calculateOptimalLayout(nodeCount);
  // Keeping fallbackSpacing for potential future use in hybrid layouts
  // const fallbackSpacing = calculateLayoutSpacing(nodeCount, hasMultipleDense);

  // Debug logging for layout calculation
  React.useEffect(() => {
    if (architectureData?.nodes) {
      console.log('🔧 Layout calculation:', {
        nodeCount,
        hasMultipleDense,
        optimalLayout,
        nodeTypes: architectureData.nodes.map(n => n.data.type)
      });
    }
  }, [nodeCount, hasMultipleDense, optimalLayout, architectureData?.nodes]);

  // Choose layout configuration based on model complexity and ELK availability
  const layout = (optimalLayout.useMultiRow && elkAvailable) ? {
    name: 'elk',
    elk: {
      'algorithm': 'layered',
      'direction': 'RIGHT',
      
      // Multi-row layout configuration
      'spacing.nodeNodeBetweenLayers': optimalLayout.rankSep,
      'spacing.nodeNode': optimalLayout.nodeSep,
      'spacing.edgeNodeBetweenLayers': optimalLayout.edgeSep,
      
      // Row wrapping settings
      'layered.wrapping.strategy': 'MULTI_EDGE',
      'layered.wrapping.additionalEdgeSpacing': 20,
      'layered.wrapping.correctionFactor': 1.2,
      
      // Node placement for better readability
      'layered.nodePlacement.strategy': 'NETWORK_SIMPLEX', 
      'layered.spacing.nodeNodeBetweenLayers': optimalLayout.rowSep,
      
      // Edge routing for cleaner connections
      'layered.edgeRouting.strategy': 'ORTHOGONAL',
      'layered.unnecessaryBendpoints': 'true',
      
      // Port constraints for better edge flow
      'portConstraints': 'FIXED_ORDER'
    },
    fit: true,
    padding: 40  // More padding for multi-row layout
  } : {
    name: 'dagre',
    rankDir: 'LR',  // Left to right flow
    
    // Single-row optimized spacing
    rankSep: optimalLayout.rankSep,
    nodeSep: optimalLayout.nodeSep,
    edgeSep: optimalLayout.edgeSep,
    
    // Collision avoidance and proper sizing
    avoidOverlap: true,              // Prevent node overlap
    nodeDimensionsIncludeLabels: true, // Account for label size in spacing
    spacingFactor: 1.0,              // No additional spacing factor needed
    
    // Viewport management
    fit: true,                       // Ensure layout fits in viewport
    padding: 30                      // Standard padding for single-row
  };

  const handleNodeClick = (event: { target: { addClass: (className: string) => void; data: (key?: string) => unknown } }) => {
    const node = event.target;
    
    // Remove previous highlights
    if (cyRef.current) {
      cyRef.current.nodes().removeClass('highlighted');
      // Add highlight to clicked node
      node.addClass('highlighted');
    }
    
    // Call parent handler
    onNodeClick?.(node.data() as LayerData);
  };

  const handleNodeHover = () => {
    // Cursor styling is handled by CSS instead of Cytoscape style
    const container = cyRef.current?.container();
    if (container) {
      container.style.cursor = 'pointer';
    }
  };

  const handleNodeMouseLeave = () => {
    const container = cyRef.current?.container();
    if (container) {
      container.style.cursor = 'default';
    }
  };

  // Load ELK dynamically when component mounts (if needed for multi-row layouts)
  useEffect(() => {
    if (optimalLayout.useMultiRow && !elkAvailable) {
      loadELK().then((loaded) => {
        if (loaded && cyRef.current && architectureData) {
          // Trigger layout refresh if ELK just became available
          const event = new CustomEvent('elk-loaded');
          window.dispatchEvent(event);
        }
      });
    }
  }, [optimalLayout.useMultiRow, architectureData]);

  // Cleanup effect for component unmounting
  useEffect(() => {
    return () => {
      // Clean up event listeners when component unmounts
      if (cyRef.current) {
        cyRef.current.removeListener('tap');
        cyRef.current.removeListener('mouseover');
        cyRef.current.removeListener('mouseout');
      }
    };
  }, []);

  // Resize handler to refit when container size changes or data updates
  useEffect(() => {
    if (cyRef.current && architectureData) {
      // Trigger refit when architecture data changes with optimized timing
      const refitGraph = () => {
        if (cyRef.current) {
          cyRef.current.fit(undefined, 15);
          cyRef.current.center();
          
          // Second pass for better fit after DOM settles
          scheduleNonUrgent(() => {
            if (cyRef.current) {
              cyRef.current.fit(undefined, 10);
              cyRef.current.center();
            }
          }, 100);
        }
      };
      
      // Initial refit using non-blocking scheduler
      scheduleNonUrgent(refitGraph, 100);
      
      // Setup resize observer with debouncing to handle container size changes  
      const resizeObserver = new ResizeObserver(() => {
        // Use non-blocking scheduler for resize debouncing
        scheduleNonUrgent(refitGraph, 150);
      });
      
      const container = cyRef.current.container();
      if (container) {
        resizeObserver.observe(container);
      }
      
      return () => {
        resizeObserver.disconnect();
      };
    }
  }, [architectureData, scheduleNonUrgent]);

  // Toggle-based looping animation system with viewport optimization
  const [isAnimating, setIsAnimating] = React.useState(false);
  const animationIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isAnimatingRef = useRef(false);
  const [isMobileView, setIsMobileView] = React.useState(false);
  
  // Detect mobile viewport
  React.useEffect(() => {
    const checkMobileView = () => setIsMobileView(window.innerWidth <= 768);
    checkMobileView();
    const handleResize = () => checkMobileView();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  const resetAnimations = () => {
    if (!cyRef.current) return;
    
    const nodes = cyRef.current.nodes();
    const edges = cyRef.current.edges();
    
    // Reset all animation classes
    nodes.removeClass(['processing']);
    edges.removeClass(['flowing-data', 'flowing-errors']);
    
    // Restore original tensor_transform labels and default styling
    edges.each((edge) => {
      const transform = edge.data('tensor_transform');
      if (transform) {
        edge.style('label', transform);
        // Restore default label styling
        edge.style('color', '#9CA3AF');
        edge.style('text-background-color', '#1F2937');
        edge.style('font-weight', 'normal');
      }
    });
    
    console.log('Reset animations and restored tensor transform labels'); // Debug log
  };

  const runSingleAnimationCycle = () => {
    if (!cyRef.current || !isAnimatingRef.current) return;
    
    const nodes = cyRef.current.nodes();
    const trainableNodes = nodes.toArray().filter(node => {
      const nodeType = node.data('type');
      return nodeType !== 'input';
    });
    
    // Reduce animation frequency on mobile to improve performance
    const animationSpeed = isMobileView ? 0.5 : 1;
    const forwardDelay = Math.max(300, 600 * animationSpeed);
    const backwardDelay = Math.max(200, 400 * animationSpeed);
    
    // Use requestAnimationFrame for smoother animations
    let currentNodeIndex = 0;
    
    const animateForwardPass = () => {
      if (!isAnimatingRef.current || currentNodeIndex >= nodes.length) {
        scheduleNonUrgent(animateBackwardPass, forwardDelay);
        return;
      }
      
      const node = nodes[currentNodeIndex];
      requestAnimationFrame(() => {
        if (!isAnimatingRef.current) return;
        
        node.addClass('processing');
        const outgoingEdges = node.outgoers().edges();
        
        // Batch DOM updates to reduce reflows
        requestAnimationFrame(() => {
          const rafStart = performance.now();
          if (!isAnimatingRef.current) return;
          
          try {
            outgoingEdges.addClass('flowing-data');
            outgoingEdges.style({
              'label': 'DATA',
              'color': '#10B981',
              'text-background-color': '#064E3B',
              'font-weight': 'bold'
            });
          } finally {
            const rafDuration = performance.now() - rafStart;
            if (rafDuration > 16.67) {
              console.warn(`🎨 Cytoscape RAF slow: ${rafDuration.toFixed(2)}ms in forward pass animation`);
            }
          }
          
          // Cleanup after animation duration
          scheduleNonUrgent(() => {
            if (outgoingEdges.length > 0) {
              outgoingEdges.removeClass('flowing-data');
            }
          }, 600);
        });
      });
      
      currentNodeIndex++;
      scheduleNonUrgent(animateForwardPass, forwardDelay);
    };
    
    let currentBackwardIndex = 0;
    const reversedTrainableNodes = [...trainableNodes].reverse();
    
    const animateBackwardPass = () => {
      if (!isAnimatingRef.current || currentBackwardIndex >= reversedTrainableNodes.length) {
        scheduleNonUrgent(cleanupAndLoop, backwardDelay);
        return;
      }
      
      const node = reversedTrainableNodes[currentBackwardIndex];
      requestAnimationFrame(() => {
        if (!isAnimatingRef.current) return;
        
        const incomingEdges = node.incomers().edges();
        
        requestAnimationFrame(() => {
          const rafStart = performance.now();
          if (!isAnimatingRef.current) return;
          
          try {
            incomingEdges.addClass('flowing-errors');
            incomingEdges.style({
              'label': 'ERRORS',
              'color': '#EF4444',
              'text-background-color': '#7F1D1D',
              'font-weight': 'bold'
            });
          } finally {
            const rafDuration = performance.now() - rafStart;
            if (rafDuration > 16.67) {
              console.warn(`🎨 Cytoscape RAF slow: ${rafDuration.toFixed(2)}ms in backward pass animation`);
            }
          }
          
          scheduleNonUrgent(() => {
            if (node) node.removeClass('processing');
            if (incomingEdges.length > 0) {
              incomingEdges.removeClass('flowing-errors');
            }
          }, 600);
        });
      });
      
      currentBackwardIndex++;
      scheduleNonUrgent(animateBackwardPass, backwardDelay);
    };
    
    const cleanupAndLoop = () => {
      if (!isAnimatingRef.current) return;
      
      requestAnimationFrame(() => {
        if (!cyRef.current || !isAnimatingRef.current) return;
        
        // Reset all animations
        nodes.removeClass('processing');
        cyRef.current.edges().style({
          'label': 'data(label)',
          'color': '#64748B',
          'text-background-color': '#1E293B',
          'font-weight': 'normal'
        });
        
        // Schedule next cycle with longer pause on mobile
        const pauseDuration = isMobileView ? 1000 : 500;
        scheduleNonUrgent(() => {
          if (isAnimatingRef.current) {
            runSingleAnimationCycle();
          }
        }, pauseDuration);
      });
    };
    
    // Start the animation cycle
    currentNodeIndex = 0;
    currentBackwardIndex = 0;
    animateForwardPass();
  };

  const toggleAnimation = () => {
    console.log('Toggle animation clicked, current state:', isAnimating); // Debug log
    
    if (isAnimating) {
      // Stop animation
      console.log('Stopping animation'); // Debug log
      setIsAnimating(false);
      isAnimatingRef.current = false;
      if (animationIntervalRef.current) {
        clearTimeout(animationIntervalRef.current);
        animationIntervalRef.current = null;
      }
      // Clean up any active animations
      resetAnimations();
    } else {
      // Start animation loop
      console.log('Starting animation'); // Debug log
      setIsAnimating(true);
      isAnimatingRef.current = true;
      resetAnimations();
      
      // Start immediately since we're using ref
      console.log('Starting animation cycle immediately'); // Debug log
      runSingleAnimationCycle();
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationIntervalRef.current) {
        clearTimeout(animationIntervalRef.current);
      }
    };
  }, []);

  // PNG Export function
  const exportToPNG = React.useCallback(async (): Promise<Blob | null> => {
    if (!cyRef.current) {
      console.warn('Cytoscape instance not available for PNG export');
      return null;
    }

    try {
      // Ensure animations are stopped and cleaned up for clean export
      const wasAnimating = isAnimatingRef.current;
      if (wasAnimating) {
        isAnimatingRef.current = false;
        setIsAnimating(false);
        resetAnimations();
      }

      // Wait a moment for animations to clear with non-blocking scheduler
      await new Promise(resolve => scheduleNonUrgent(() => resolve(undefined), 100));

      // Generate PNG with high quality settings
      const pngBlob = cyRef.current.png({
        output: 'blob',
        bg: '#111827', // Match the UI dark background
        full: true, // Export full graph
        scale: 3, // High resolution (3x)
        maxWidth: 2400, // Max width for very large graphs
        maxHeight: 1600, // Max height
      });

      // Resume animation if it was running
      if (wasAnimating) {
        setIsAnimating(true);
        isAnimatingRef.current = true;
        runSingleAnimationCycle();
      }

      console.log('PNG export successful');
      return pngBlob;
    } catch (error) {
      console.error('PNG export failed:', error);
      return null;
    }
  }, [setIsAnimating]); // eslint-disable-line react-hooks/exhaustive-deps

  // Expose export function to parent via ref and callback
  React.useImperativeHandle(ref, () => ({
    exportToPNG
  }), [exportToPNG]);
  
  React.useEffect(() => {
    if (onPngExportRequest) {
      // Store the export function reference so parent can call it
      (window as unknown as Record<string, unknown>).exportArchitecturePNG = exportToPNG;
    }
    
    return () => {
      const windowObj = window as unknown as Record<string, unknown>;
      if (windowObj.exportArchitecturePNG) {
        delete windowObj.exportArchitecturePNG;
      }
    };
  }, [exportToPNG, onPngExportRequest]);


  if (!architectureData?.nodes || architectureData.nodes.length === 0) {
    return (
      <div className={`flex items-center justify-center h-96 bg-gray-900 rounded-lg ${className}`}>
        <div className="text-center">
          <div className="text-gray-400 text-lg mb-2">No Architecture Data</div>
          <div className="text-gray-500 text-sm">
            Architecture visualization will appear here when trial data is available
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>    

      {/* Toggle Animation Controls */}
      <div className="absolute top-2 left-2 z-10">
        <button
          onClick={toggleAnimation}
          className={`text-white text-sm px-2 py-1 rounded-md transition-colors flex items-center gap-2 ${
            isAnimating 
              ? 'bg-red-600 hover:bg-red-700' 
              : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          <span className="text-xs">
            {isAnimating ? '⏸' : '▶'}
          </span>
          {isAnimating ? 'Pause' : 'Animate'}
        </button>
      </div>

      {/* Cytoscape Component */}
      <CytoscapeComponent
        elements={[...architectureData.nodes, ...architectureData.edges]}
        style={{ width: '100%', height: '100%' }}
        stylesheet={cytoscapeStylesheet}
        layout={layout}
        cy={(cy: cytoscape.Core) => {
          cyRef.current = cy;
          
          // Set up event listeners when Cytoscape is ready
          cy.ready(() => {
            // Remove any existing listeners first
            cy.removeListener('tap');
            cy.removeListener('mouseover');
            cy.removeListener('mouseout');
            
            // Add event listeners
            cy.on('tap', 'node', handleNodeClick);
            cy.on('mouseover', 'node', handleNodeHover);
            cy.on('mouseout', 'node', handleNodeMouseLeave);
            
            // Initial fit and center - be more aggressive
            cy.fit(undefined, 15); // Reduced padding for tighter fit
            cy.center();
            
            // Multiple passes to ensure proper fitting with progressive adjustments
            scheduleNonUrgent(() => {
              if (cyRef.current) {
                cyRef.current.fit(undefined, 15);
                cyRef.current.center();
                
                // Final adjustment after layout settles
                scheduleNonUrgent(() => {
                  if (cyRef.current) {
                    cyRef.current.fit(undefined, 10); // Very tight fit for maximum use of space
                    cyRef.current.center();
                  }
                }, 100);
              }
            }, 200);
          });
        }}
        className="bg-gray-900 rounded-lg"
      />

      {/* Legend */}
      {showLegend && <ModelLegend architectureData={architectureData} />}
      
    </div>
  );
});

ModelGraph.displayName = 'ModelGraph';

export default ModelGraph;