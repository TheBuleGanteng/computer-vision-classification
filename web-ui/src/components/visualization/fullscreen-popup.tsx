"use client"

import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Maximize2, X } from 'lucide-react';

interface FullscreenPopupProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  maxWidth?: string;
}

export const FullscreenPopup: React.FC<FullscreenPopupProps> = ({
  isOpen,
  onClose,
  title,
  children,
  maxWidth = "max-w-[95vw]"
}) => {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className={`${maxWidth} max-h-[95vh] w-[95vw] p-0 overflow-hidden`}>
        {title && (
          <DialogHeader className="p-4 pb-2 border-b border-border">
            <div className="flex items-center justify-between">
              <DialogTitle className="flex items-center gap-2">
                <Maximize2 className="w-5 h-5" />
                {title}
              </DialogTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="h-8 w-8 p-0"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </DialogHeader>
        )}
        
        <div className="flex-1 overflow-auto">
          {children}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default FullscreenPopup;