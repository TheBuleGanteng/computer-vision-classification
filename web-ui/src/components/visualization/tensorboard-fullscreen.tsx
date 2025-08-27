"use client"

import React from 'react';
import TensorBoardPanel from './tensorboard-panel';

interface TensorBoardFullscreenProps {
  jobId: string;
  trialId?: string;
  onClose?: () => void;
}

export const TensorBoardFullscreen: React.FC<TensorBoardFullscreenProps> = ({
  jobId,
  trialId,
  onClose
}) => {
  return (
    <div className="h-full">
      <TensorBoardPanel 
        jobId={jobId} 
        trialId={trialId} 
        height={700}
        isFullscreen={true}
        onExpandClick={onClose}
      />
    </div>
  );
};

export default TensorBoardFullscreen;