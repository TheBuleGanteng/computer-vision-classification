// Type declarations for cytoscape-related modules

declare module 'react-cytoscapejs' {
  import { Component } from 'react';
  import { Core, ElementDefinition, Stylesheet } from 'cytoscape';

  interface CytoscapeComponentProps {
    elements: ElementDefinition[];
    style?: React.CSSProperties;
    stylesheet?: Stylesheet[];
    layout?: Record<string, unknown>;
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
  interface DagreExtension {
    (cytoscape: unknown): void;
  }

  const dagre: DagreExtension;
  export = dagre;
}

declare module 'cytoscape-elk' {
  interface ElkExtension {
    (cytoscape: unknown): void;
  }

  const elk: ElkExtension;
  export = elk;
}