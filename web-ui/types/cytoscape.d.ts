// Type declarations for cytoscape-related modules

declare module 'react-cytoscapejs' {
  import { Component } from 'react';
  import { Core, ElementDefinition, Stylesheet } from 'cytoscape';

  interface CytoscapeComponentProps {
    elements: ElementDefinition[];
    style?: React.CSSProperties;
    stylesheet?: Stylesheet[];
    layout?: any;
    cy?: (cy: Core) => void;
    className?: string;
    zoom?: number;
    pan?: { x: number; y: number };
    minZoom?: number;
    maxZoom?: number;
    zoomingEnabled?: boolean;
    userZoomingEnabled?: boolean;
    panningEnabled?: boolean;
    userPanningEnabled?: boolean;
    boxSelectionEnabled?: boolean;
    selectionType?: string;
    touchTapThreshold?: number;
    desktopTapThreshold?: number;
    autolock?: boolean;
    autoungrabify?: boolean;
    autounselectify?: boolean;
    headless?: boolean;
    styleEnabled?: boolean;
    hideEdgesOnViewport?: boolean;
    textureOnViewport?: boolean;
    wheelSensitivity?: number;
    motionBlur?: boolean;
    motionBlurOpacity?: number;
    pixelRatio?: number | string;
  }

  export default class CytoscapeComponent extends Component<CytoscapeComponentProps> {}
}

declare module 'cytoscape-dagre' {
  import { Core } from 'cytoscape';
  
  interface DagreLayout {
    name: 'dagre';
    nodeSep?: number;
    edgeSep?: number;
    rankSep?: number;
    rankDir?: 'TB' | 'BT' | 'LR' | 'RL';
    align?: 'UL' | 'UR' | 'DL' | 'DR';
    acyclicer?: 'greedy' | undefined;
    ranker?: 'network-simplex' | 'tight-tree' | 'longest-path';
    minLen?: (edge: any) => number;
    edgeWeight?: (edge: any) => number;
    fit?: boolean;
    padding?: number;
    spacingFactor?: number;
    nodeDimensionsIncludeLabels?: boolean;
    animate?: boolean;
    animateFilter?: (node: any, i: number) => boolean;
    animationDuration?: number;
    animationEasing?: string;
    transform?: (node: any, position: any) => any;
    ready?: () => void;
    stop?: () => void;
    randomize?: boolean;
    maxSimulationTime?: number;
  }

  interface DagreExtension {
    (cytoscape: any): void;
  }

  const dagre: DagreExtension;
  export = dagre;
}