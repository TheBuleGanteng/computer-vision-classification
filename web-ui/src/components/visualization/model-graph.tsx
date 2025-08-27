"use client"

import React, { useEffect, useRef } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';

// Register layout
cytoscape.use(dagre);

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
        'background-color': '#10B981',
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
    // Edges with tensor information
    {
      selector: 'edge',
      style: {
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#64748B',  // Fixed: use target-arrow-color instead of arrow-color
        'line-color': '#64748B',
        'width': 2,
        'label': 'data(tensor_transform)',
        'font-size': '8px',
        'text-rotation': 'autorotate',
        'text-margin-y': -10,
        'color': '#9CA3AF',
        'text-background-color': '#1F2937',
        'text-background-opacity': 0.8,
        'text-background-padding': '2px'
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
    // Animation classes for forward propagation
    {
      selector: '.flowing',
      style: {
        'line-color': '#EF4444',
        'target-arrow-color': '#EF4444',
        'width': 4,
        'opacity': 1
      }
    },
    {
      selector: 'node.active',
      style: {
        'background-color': '#F59E0B',
        'border-color': '#F59E0B',
        'border-width': 3
      }
    }
  ];

  const layout = {
    name: 'dagre',
    rankDir: 'LR',  // Left to right
    spacingFactor: 1.0, // Tighter layout for better fit
    nodeDimensionsIncludeLabels: true,
    rankSep: 50,    // Reduced spacing between ranks
    nodeSep: 25,    // Reduced spacing between nodes  
    edgeSep: 10,    // Reduced edge spacing
    fit: true,      // Ensure layout fits in viewport
    padding: 15     // Minimal padding for better use of space
  };

  const handleNodeClick = (event: any) => {
    const node = event.target;
    
    // Remove previous highlights
    if (cyRef.current) {
      cyRef.current.nodes().removeClass('highlighted');
      // Add highlight to clicked node
      node.addClass('highlighted');
    }
    
    // Call parent handler
    onNodeClick?.(node.data());
  };

  const handleNodeHover = (event: any) => {
    // Cursor styling is handled by CSS instead of Cytoscape style
    const container = cyRef.current?.container();
    if (container) {
      container.style.cursor = 'pointer';
    }
  };

  const handleNodeMouseLeave = (event: any) => {
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

  // Animation helper functions
  const animateForwardPass = () => {
    if (!cyRef.current) return;
    
    const nodes = cyRef.current.nodes();
    const edges = cyRef.current.edges();
    
    // Reset any previous animations
    nodes.removeClass('active');
    edges.removeClass('flowing');
    
    // Animate nodes in sequence
    nodes.forEach((node, index) => {
      setTimeout(() => {
        node.addClass('active');
        
        // Animate connected edges
        const outgoingEdges = node.outgoers().edges();
        outgoingEdges.addClass('flowing');
        
        // Remove animation after delay
        setTimeout(() => {
          node.removeClass('active');
          outgoingEdges.removeClass('flowing');
        }, 800);
      }, index * 600);
    });
  };

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

      {/* Animation Controls */}
      <div className="absolute top-2 left-2 z-10">
        <button
          onClick={animateForwardPass}
          className="bg-blue-600 hover:bg-blue-700 text-white text-xs px-3 py-1.5 rounded-md transition-colors"
        >
          ▶ Animate Forward Pass
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
              cy.fit(undefined, 15);
              cy.center();
              
              // Final adjustment after layout settles
              setTimeout(() => {
                cy.fit(undefined, 10); // Very tight fit for maximum use of space
                cy.center();
              }, 100);
            }, 200);
          });
        }}
        className="bg-gray-900 rounded-lg"
      />

      
    </div>
  );
};

export default ModelGraph;