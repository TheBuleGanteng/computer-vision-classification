"""
Progress Tracking and Callback Components

Contains all progress tracking classes and TensorFlow callbacks used by the optimization system.
Separated from optimizer.py for better code organization and maintainability.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import tensorflow as tf
from tensorflow import keras

from utils.logger import logger


@dataclass
class TrialProgress:
    """Real-time trial progress data for API streaming"""
    trial_id: str
    trial_number: int
    status: str  # "running", "completed", "failed", "pruned"
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # Epoch-level progress tracking
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    epoch_progress: Optional[float] = None  # 0.0 to 1.0 within current epoch
    
    # Architecture Information
    architecture: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    model_size: Optional[Dict[str, Any]] = None
    
    # Health Metrics (populated during/after training)
    health_metrics: Optional[Dict[str, Any]] = None
    training_stability: Optional[Dict[str, Any]] = None
    
    # Performance Data
    performance: Optional[Dict[str, Any]] = None
    training_history: Optional[Dict[str, Any]] = None
    
    # Pruning Information
    pruning_info: Optional[Dict[str, Any]] = None
    
    # Plot Generation Progress
    plot_generation: Optional[Dict[str, Any]] = None
    
    # Final Model Building Progress
    final_model_building: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization"""
        return {
            'trial_id': self.trial_id,
            'trial_number': self.trial_number,
            'status': self.status,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'duration_seconds': self.duration_seconds,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'epoch_progress': self.epoch_progress,
            'architecture': self.architecture,
            'hyperparameters': self.hyperparameters,
            'model_size': self.model_size,
            'health_metrics': self.health_metrics,
            'training_stability': self.training_stability,
            'performance': self.performance,
            'training_history': self.training_history,
            'pruning_info': self.pruning_info,
            'plot_generation': self.plot_generation,
            'final_model_building': self.final_model_building
        }


@dataclass 
class AggregatedProgress:
    """Aggregated progress data across multiple concurrent trials"""
    total_trials: int
    running_trials: List[int]
    completed_trials: List[int]
    failed_trials: List[int]
    current_best_total_score: Optional[float]
    estimated_time_remaining: Optional[float]


@dataclass
class UnifiedProgress:
    """
    Unified progress data combining trial statistics with epoch information
    This replaces the dual callback system to eliminate race conditions
    """
    # Trial statistics (from AggregatedProgress)
    total_trials: int
    running_trials: List[int]
    completed_trials: List[int]
    failed_trials: List[int]
    current_best_total_score: Optional[float]  # Optimization objective (accuracy or weighted score)
    current_best_accuracy: Optional[float]     # Raw accuracy for comparison
    average_duration_per_trial: Optional[float]  # Average duration in seconds
    estimated_time_remaining: Optional[float]
    
    # Current epoch information (from most recent TrialProgress)
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    epoch_progress: Optional[float] = None
    current_trial_id: Optional[str] = None
    current_trial_status: Optional[str] = None
    
    # Status message for UI display
    status_message: Optional[str] = None
    
    # Final model building progress
    final_model_building: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization"""
        return {
            'total_trials': self.total_trials,
            'running_trials': self.running_trials,
            'completed_trials': self.completed_trials,
            'failed_trials': self.failed_trials,
            'current_best_total_score': self.current_best_total_score,
            'current_best_accuracy': self.current_best_accuracy,
            'estimated_time_remaining': self.estimated_time_remaining,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'epoch_progress': self.epoch_progress,
            'current_trial_id': self.current_trial_id,
            'current_trial_status': self.current_trial_status,
            'status_message': self.status_message,
            'final_model_building': self.final_model_building
        }


class ConcurrentProgressAggregator:
    """Aggregates progress across multiple concurrent trials"""
    
    def __init__(self, total_trials: int):
        self.total_trials = total_trials
    
    def aggregate_progress(self, current_trial: Optional[TrialProgress], all_trial_statuses: Dict[int, str]) -> AggregatedProgress:
        """
        Aggregate progress from multiple concurrent trials
        
        Args:
            current_trial: Current trial progress data
            all_trial_statuses: Dictionary mapping trial numbers to status strings
            
        Returns:
            AggregatedProgress with consolidated status
        """
        # logger.debug(f"running aggregate_progress ... aggregating progress for {len(all_trial_statuses)} trials")
        
        # Categorize trials by status
        running_trials = [t for t, s in all_trial_statuses.items() if s == "running"]
        completed_trials = [t for t, s in all_trial_statuses.items() if s == "completed"]
        failed_trials = [t for t, s in all_trial_statuses.items() if s == "failed"]
        
        # Calculate ETA using the current trial statuses
        estimated_time_remaining = self.calculate_eta(all_trial_statuses)
        
        # Get current best value (this will be implemented in the callback)
        current_best_value = self.get_current_best_total_score()
        
        return AggregatedProgress(
            total_trials=self.total_trials,
            running_trials=running_trials,
            completed_trials=completed_trials,
            failed_trials=failed_trials,
            current_best_total_score=current_best_value,
            estimated_time_remaining=estimated_time_remaining
        )
    
    def calculate_eta(self, all_trial_statuses: Dict[int, str]) -> Optional[float]:
        """Calculate estimated time remaining based on trial statuses"""
        # Simple implementation - can be enhanced later
        completed_count = len([s for s in all_trial_statuses.values() if s == "completed"])
        
        if completed_count == 0:
            return None
        
        # Rough estimate based on completion rate
        remaining_trials = self.total_trials - completed_count
        avg_time_per_trial = 120.0  # Assume 2 minutes per trial as baseline
        
        return remaining_trials * avg_time_per_trial
    
    def get_current_best_total_score(self) -> Optional[float]:
        """Get current best value - placeholder for now"""
        return None  # Will be populated by the ModelOptimizer instance


class EpochProgressCallback(keras.callbacks.Callback):
    """
    Real-time epoch progress callback that tracks progress within epochs
    Updates progress during batch training for live progress updates
    """
    
    def __init__(self, trial_number: int, total_epochs: int, optimizer_instance=None):
        super().__init__()
        self.trial_number = trial_number
        self.total_epochs = total_epochs
        self.optimizer_instance = optimizer_instance
        self.current_epoch = 0
        self.total_batches = 0
        self.current_batch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        # Check for cancellation first
        if self.optimizer_instance and self.optimizer_instance.is_cancelled():
            logger.info(f"EpochProgressCallback.on_epoch_begin ... Cancellation detected, stopping training")
            self.model.stop_training = True
            return
        
        self.current_epoch = epoch + 1  # Convert 0-based to 1-based
        self.current_batch = 0
        
        # Try to get total batches from params
        if hasattr(self, 'params') and self.params:
            self.total_batches = self.params.get('steps', 0)
        
        self._update_progress(0.0)
        logger.debug(f"🔍 EPOCH PROGRESS: Trial {self.trial_number}, Epoch {self.current_epoch}/{self.total_epochs} started")
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch - update progress within epoch"""
        # Check for cancellation first
        if self.optimizer_instance and self.optimizer_instance.is_cancelled():
            logger.info(f"EpochProgressCallback.on_batch_end ... Cancellation detected, stopping training")
            self.model.stop_training = True
            return
        
        self.current_batch = batch + 1  # Convert 0-based to 1-based
        
        if self.total_batches > 0:
            batch_progress = min(self.current_batch / self.total_batches, 1.0)
            self._update_progress(batch_progress)
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        self._update_progress(1.0)
        logger.debug(f"🔍 EPOCH PROGRESS: Trial {self.trial_number}, Epoch {self.current_epoch}/{self.total_epochs} completed")
    
    def _update_progress(self, epoch_progress: float):
        """Update epoch progress and trigger unified progress update"""
        # Update epoch info in the optimizer
        if self.optimizer_instance and hasattr(self.optimizer_instance, '_current_epoch_info'):
            self.optimizer_instance._current_epoch_info[self.trial_number] = {
                'current_epoch': self.current_epoch,
                'total_epochs': self.total_epochs,
                'epoch_progress': epoch_progress
            }
            
            # Trigger unified progress update every 10 batches or at epoch boundaries
            if (epoch_progress == 0.0 or epoch_progress == 1.0 or 
                (self.current_batch > 0 and self.current_batch % 10 == 0)):
                self._trigger_unified_progress_update()
    
    def _trigger_unified_progress_update(self):
        """Trigger a unified progress update with current epoch information"""
        if self.optimizer_instance and hasattr(self.optimizer_instance, 'progress_callback') and self.optimizer_instance.progress_callback:
            try:
                # Create a mock trial progress for aggregation
                trial_progress = TrialProgress(
                    trial_id=f"trial_{self.trial_number}",
                    trial_number=self.trial_number,
                    status="running",
                    started_at=datetime.now().isoformat(),
                    current_epoch=self.current_epoch,
                    total_epochs=self.total_epochs,
                    epoch_progress=self.optimizer_instance._current_epoch_info.get(self.trial_number, {}).get('epoch_progress', 0.0)
                )
                
                # Get best trial info for aggregation
                best_trial_number, best_trial_value = self.optimizer_instance.get_best_trial_info()
                self.optimizer_instance._progress_aggregator.get_current_best_total_score = lambda: best_trial_value
                
                # Create aggregated progress using the progress aggregator
                aggregated_progress = self.optimizer_instance._progress_aggregator.aggregate_progress(
                    current_trial=trial_progress,
                    all_trial_statuses=self.optimizer_instance._trial_statuses
                )
                
                # Create unified progress and send update
                unified_progress = self.optimizer_instance._create_unified_progress(aggregated_progress)
                self.optimizer_instance.progress_callback(unified_progress)
                
            except Exception as e:
                logger.warning(f"EpochProgressCallback._trigger_unified_progress_update error: {e}")


def default_progress_callback(progress: Union[TrialProgress, AggregatedProgress, UnifiedProgress]) -> None:
    """Default progress callback that prints progress updates to console"""
    if isinstance(progress, UnifiedProgress):
        # New unified progress system
        print(f"📊 Progress: {len(progress.completed_trials)}/{progress.total_trials} trials completed, "
              f"{len(progress.running_trials)} trials running, {len(progress.failed_trials)} trials failed")
        if progress.current_best_total_score is not None:
            print(f"📈 Best value so far: {progress.current_best_total_score:.4f}")
        if progress.current_epoch is not None and progress.total_epochs is not None:
            print(f"⏱️ Current epoch: {progress.current_epoch}/{progress.total_epochs}")
        if progress.estimated_time_remaining is not None:
            eta_minutes = progress.estimated_time_remaining / 60
            print(f"   ETA: {eta_minutes:.1f} minutes")
    elif isinstance(progress, AggregatedProgress):
        # Legacy aggregated progress (deprecated)
        print(f"📊 Progress: {len(progress.completed_trials)}/{progress.total_trials} trials completed, "
              f"{len(progress.running_trials)} trials running, {len(progress.failed_trials)} trials failed")
        if progress.current_best_total_score is not None:
            print(f"📈 Best value so far: {progress.current_best_total_score:.4f}")
        if progress.estimated_time_remaining is not None:
            eta_minutes = progress.estimated_time_remaining / 60
            print(f"   ETA: {eta_minutes:.1f} minutes")
    else:
        # TrialProgress (legacy - should not be used anymore)
        print(f"🔄 Trial {progress.trial_number} ({progress.status})")