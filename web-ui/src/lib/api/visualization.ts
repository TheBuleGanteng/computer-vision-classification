// API service functions for 3D Model Visualization
// Connects to backend endpoints created in api_server.py

import { BestModelResponse } from '@/types/visualization';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class VisualizationAPIError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = 'VisualizationAPIError';
  }
}

/**
 * Fetch 3D visualization data for the best model of a job
 */
export async function getBestModelVisualization(jobId: string): Promise<BestModelResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/best-model`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      if (response.status === 404) {
        throw new VisualizationAPIError(
          'No completed trials found for this job',
          404
        );
      }
      
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new VisualizationAPIError(
        errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
        response.status
      );
    }

    const data: BestModelResponse = await response.json();
    
    // Validate response structure
    if (!data.visualization_data || !data.visualization_data.layers) {
      throw new VisualizationAPIError('Invalid visualization data structure received');
    }

    return data;
  } catch (error) {
    if (error instanceof VisualizationAPIError) {
      throw error;
    }
    
    // Network or other errors
    throw new VisualizationAPIError(
      `Failed to fetch visualization data: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Download 3D visualization data as JSON file
 */
export async function downloadModelVisualization(jobId: string): Promise<Blob> {
  try {
    const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/best-model/download`, {
      method: 'GET',
    });

    if (!response.ok) {
      if (response.status === 404) {
        throw new VisualizationAPIError(
          'No visualization data available for download',
          404
        );
      }
      
      throw new VisualizationAPIError(
        `Download failed: HTTP ${response.status}`,
        response.status
      );
    }

    const blob = await response.blob();
    
    // Verify it's actually JSON
    if (!response.headers.get('content-type')?.includes('application/json')) {
      throw new VisualizationAPIError('Downloaded file is not JSON format');
    }

    return blob;
  } catch (error) {
    if (error instanceof VisualizationAPIError) {
      throw error;
    }
    
    throw new VisualizationAPIError(
      `Download failed: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Download and trigger browser download of visualization file using Save As dialog
 */
export async function downloadVisualizationFile(
  jobId: string, 
  filename?: string
): Promise<void> {
  try {
    const blob = await downloadModelVisualization(jobId);
    const defaultFilename = filename || `model-visualization-${jobId}.json`;
    
    // Use File System Access API for modern browsers with Save As dialog
    if ('showSaveFilePicker' in window) {
      try {
        const fileHandle = await (window as unknown as { showSaveFilePicker: (options: {
          suggestedName: string;
          types: Array<{ description: string; accept: Record<string, string[]> }>;
        }) => Promise<{ createWritable: () => Promise<{ write: (data: Blob) => Promise<void>; close: () => Promise<void> }> }> }).showSaveFilePicker({
          suggestedName: defaultFilename,
          types: [{
            description: 'JSON files',
            accept: { 'application/json': ['.json'] },
          }],
        });
        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
        console.log('Visualization file saved successfully');
        return;
      } catch (saveError: unknown) {
        if ((saveError as Error).name === 'AbortError') {
          console.log('Save dialog was cancelled by user');
          return;
        }
        console.warn('Save As dialog failed, falling back to download:', saveError);
      }
    }

    // Fallback for browsers that don't support File System Access API
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = defaultFilename;
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Cleanup
    window.URL.revokeObjectURL(url);
    console.log('Visualization file downloaded successfully (fallback method)');
  } catch (error) {
    if (error instanceof VisualizationAPIError) {
      throw error;
    }
    
    throw new VisualizationAPIError(
      `File download failed: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Check if job has completed trials (and thus visualization data)
 */
export async function hasVisualizationData(jobId: string): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/status`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      return false;
    }

    const statusData = await response.json();
    return (statusData.completed_trials || 0) > 0;
  } catch {
    return false;
  }
}

/**
 * Utility function to format download filename with timestamp
 */
export function generateVisualizationFilename(
  jobId: string, 
  architectureType?: string,
  performanceScore?: number
): string {
  const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
  const arch = architectureType ? `-${architectureType.toLowerCase()}` : '';
  const perf = performanceScore ? `-${(performanceScore * 100).toFixed(1)}pct` : '';
  
  return `model-viz-${jobId}${arch}${perf}-${timestamp}.json`;
}