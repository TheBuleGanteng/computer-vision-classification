// Modern Educational Visualization Components (Cytoscape.js + TensorBoard)
export { default as ModelGraph } from './model-graph';
export { default as LayerInfoPanel } from './layer-info-panel-new';
export { default as MetricsTabs } from './metrics-tabs';
export { default as UnifiedEducationalInterface } from './unified-educational-interface';

// Legacy Components (Still Active) - Components that don't use React Three Fiber
export { LayerInfoPanel as LayerInfoPanelLegacy } from './layer-info-panel';
export { VisualizationErrorBoundary } from './visualization-error-boundary';

// Note: All React Three Fiber components and their dependencies have been renamed to .legacy extensions 
// to prevent compilation errors during migration to Cytoscape.js + TensorBoard educational stack