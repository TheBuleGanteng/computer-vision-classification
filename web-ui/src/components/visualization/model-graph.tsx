"use client"

import React, { useEffect, useRef } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import elk from 'cytoscape-elk';

// Register layouts
cytoscape.use(dagre);
cytoscape.use(elk);

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

interface ModelGraphProps {
  architectureData: CytoscapeData;
  onNodeClick?: (nodeData: LayerData) => void;
  className?: string;
}

export const ModelGraph: React.FC<ModelGraphProps> = ({
  architectureData,
  onNodeClick,
  className = ""
}) => {
  const cyRef = useRef<cytoscape.Core | null>(null);

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
    // Output layer
    {
      selector: 'node[type="output"]',
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

  // Choose layout configuration based on model complexity
  const layout = optimalLayout.useMultiRow ? {
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
          setTimeout(() => {
            if (cyRef.current) {
              cyRef.current.fit(undefined, 10);
              cyRef.current.center();
            }
          }, 100);
        }
      };
      
      // Initial refit
      setTimeout(refitGraph, 100);
      
      // Setup resize observer to handle container size changes
      const resizeObserver = new ResizeObserver(() => {
        setTimeout(refitGraph, 50);
      });
      
      const container = cyRef.current.container();
      if (container) {
        resizeObserver.observe(container);
      }
      
      return () => {
        resizeObserver.disconnect();
      };
    }
  }, [architectureData]);

  // Toggle-based looping animation system
  const [isAnimating, setIsAnimating] = React.useState(false);
  const animationIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isAnimatingRef = useRef(false);
  
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
    if (!cyRef.current) return;
    
    console.log('Running animation cycle, isAnimatingRef:', isAnimatingRef.current); // Debug log
    
    const nodes = cyRef.current.nodes();
    
    // Get all trainable nodes (excluding input)
    const trainableNodes = nodes.toArray().filter(node => {
      const nodeType = node.data('type');
      return nodeType !== 'input';
    });
    
    console.log('Found nodes:', nodes.length, 'trainable nodes:', trainableNodes.length); // Debug log
    
    // PHASE 1: Forward pass - data flows through all layers
    nodes.forEach((node, index) => {
      const timeoutId = setTimeout(() => {
        if (!isAnimatingRef.current) {
          console.log('Animation stopped during forward pass'); // Debug log
          return;
        }
        
        console.log('Animating forward node', index); // Debug log
        node.addClass('processing');
        
        // Show data flow on outgoing edges
        const outgoingEdges = node.outgoers().edges();
        outgoingEdges.addClass('flowing-data');
        
        // Replace labels with "DATA" during forward pass
        outgoingEdges.style('label', 'DATA');
        outgoingEdges.style('color', '#10B981');
        outgoingEdges.style('text-background-color', '#064E3B');
        outgoingEdges.style('font-weight', 'bold');
        
        // Remove forward animation but keep node processing  
        setTimeout(() => {
          if (outgoingEdges && outgoingEdges.length > 0) {
            outgoingEdges.removeClass('flowing-data');
          }
        }, 800);
      }, index * 600);
      
      // Store timeout for cleanup
      if (animationIntervalRef.current === null) {
        animationIntervalRef.current = timeoutId;
      }
    });
    
    // PHASE 2: Backward pass - errors flow backward through trainable layers only
    const forwardDuration = nodes.length * 600 + 800;
    
    setTimeout(() => {
      if (!isAnimatingRef.current) {
        console.log('Animation stopped before backward pass'); // Debug log
        return;
      }
      
      console.log('Starting backward pass'); // Debug log
      trainableNodes.reverse().forEach((node, index) => {
        setTimeout(() => {
          if (!isAnimatingRef.current) return;
          
          console.log('Animating backward node', index); // Debug log
          
          // Show error flow on incoming edges (for trainable layers only)
          const incomingEdges = node.incomers().edges();
          incomingEdges.addClass('flowing-errors');
          
          // Replace labels with "ERRORS" during backward pass
          incomingEdges.style('label', 'ERRORS');
          incomingEdges.style('color', '#EF4444');
          incomingEdges.style('text-background-color', '#7F1D1D');
          incomingEdges.style('font-weight', 'bold');
          
          // Remove error animation and node processing
          setTimeout(() => {
            if (node) {
              node.removeClass('processing');
            }
            if (incomingEdges && incomingEdges.length > 0) {
              incomingEdges.removeClass('flowing-errors');
            }
          }, 800);
        }, index * 400); // Faster backward pass
      });
    }, forwardDuration);
    
    // PHASE 3: Clean up and schedule next cycle
    const backwardDuration = trainableNodes.length * 400 + 800;
    const totalCycleDuration = forwardDuration + backwardDuration;
    
    setTimeout(() => {
      if (!isAnimatingRef.current) {
        console.log('Animation stopped during cleanup'); // Debug log
        return;
      }
      
      // Remove processing class from input node
      const inputNodes = nodes.filter(node => node.data('type') === 'input');
      inputNodes.removeClass('processing');
      
      console.log('Cycle complete, scheduling next cycle'); // Debug log
      
      // Schedule next cycle if still animating
      setTimeout(() => {
        if (isAnimatingRef.current) {
          runSingleAnimationCycle(); // Loop!
        }
      }, 500); // Small pause between cycles
    }, totalCycleDuration);
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
            setTimeout(() => {
              if (cyRef.current) {
                cyRef.current.fit(undefined, 15);
                cyRef.current.center();
                
                // Final adjustment after layout settles
                setTimeout(() => {
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

      
    </div>
  );
};

export default ModelGraph;