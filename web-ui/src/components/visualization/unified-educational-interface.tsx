"use client"

import React from 'react';
import MetricsTabs from './metrics-tabs';
import TensorBoardFullscreen from './tensorboard-fullscreen';
import FullscreenPopup from './fullscreen-popup';


interface UnifiedEducationalInterfaceProps {
  jobId: string;
  trialId?: string;
  className?: string;
}


const UnifiedEducationalInterface: React.FC<UnifiedEducationalInterfaceProps> = React.memo(({
  jobId,
  trialId,
  className = ""
}) => {
  const [showMetricsPopup, setShowMetricsPopup] = React.useState(false);






  return (
    <div className={`min-h-[600px] flex flex-col bg-gray-900 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700">
        <div>
          <h2 className="text-lg font-semibold text-white">Training Metrics & Diagnostics</h2>
          <p className="text-sm text-gray-400">
            Job: <code className="font-mono">{jobId}</code>
            {trialId && <span> • Trial: <code className="font-mono">{trialId}</code></span>}
          </p>
        </div>
      </div>

      {/* Main Content - Full Width */}
      <div className="flex-1">
        <MetricsTabs 
          jobId={jobId} 
          trialId={trialId} 
          onExpandClick={() => setShowMetricsPopup(true)}
        />
      </div>

      {/* Metrics Fullscreen Popup */}
      <FullscreenPopup
        isOpen={showMetricsPopup}
        onClose={() => setShowMetricsPopup(false)}
        title=""
      >
        <div className="h-[85vh]">
          <TensorBoardFullscreen 
            jobId={jobId} 
            trialId={trialId}
            onClose={() => setShowMetricsPopup(false)}
          />
        </div>
      </FullscreenPopup>
    </div>
  );
});

UnifiedEducationalInterface.displayName = 'UnifiedEducationalInterface'

export { UnifiedEducationalInterface }
export default UnifiedEducationalInterface;