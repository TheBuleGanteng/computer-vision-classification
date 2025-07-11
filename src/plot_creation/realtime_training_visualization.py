"""
Real-Time Training Visualization Implementation

This module provides live training progress visualization that updates during model training.
Integrates with Keras callbacks to show real-time metrics and training insights.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.lines import Line2D
import numpy as np
from typing import Dict, List, Optional, Any, Union
import threading
import time
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow import keras # type: ignore
from utils.logger import logger


class RealTimeTrainingVisualizer:
    """
    Real-time training visualization that updates live during model training
    
    Creates a 2x2 dashboard showing:
    1. Loss curves (training and validation)
    2. Accuracy curves (training and validation) 
    3. Learning rate schedule
    4. Overfitting warning system
    """
    
    def __init__(self, model_builder, plot_dir: Optional[Path] = None) -> None:
        """
        Initialize the real-time visualizer
        
        Args:
            model_builder: ModelBuilder instance for accessing config and model info
        """
        self.model_builder = model_builder
        self.plot_dir = plot_dir
        
        # Training history storage
        self.epochs: List[int] = []
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_accuracy: List[float] = []
        self.val_accuracy: List[float] = []
        self.learning_rates: List[float] = []
        
        # Plot components with proper type hints
        self.fig: Optional[Figure] = None
        self.axes: Optional[np.ndarray] = None
        self.lines: Dict[str, Line2D] = {}
        self.warning_text: Optional[Text] = None
        
        # State tracking
        self.is_plotting: bool = False
        self.plot_thread: Optional[threading.Thread] = None
        self.current_epoch: int = 0
        self.start_time: Optional[datetime] = None
        
        # Configuration for intermediate saving
        self.save_intermediate_plots: bool = self.model_builder.model_config.save_intermediate_plots  # Get from ModelConfig
        self.save_every_n_epochs: int = self.model_builder.model_config.save_plot_every_n_epochs  # Get from ModelConfig
        self.intermediate_plot_dir: Optional[Path] = None  # Will be set in setup_plots
        
        logger.debug("running RealTimeTrainingVisualizer.__init__ ... Real-time visualizer initialized")
    
    def setup_plots(self) -> None:
        """
        Initialize the matplotlib figure and subplots for real-time updates
        """
        logger.debug("running setup_plots ... Setting up real-time training plots")
        
        # Set up intermediate plot directory if saving is enabled
        if self.save_intermediate_plots and self.plot_dir is not None:
            # Create intermediate plots subdirectory within the provided plot directory
            self.intermediate_plot_dir = self.plot_dir / "realtime_intermediate"
            self.intermediate_plot_dir.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"running setup_plots ... Intermediate plots will be saved to: {self.intermediate_plot_dir}")
        
        # Enable interactive mode for live updates
        plt.ion()
        
        # Create figure and subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(f'Real-Time Training Progress - {self.model_builder.dataset_config.name}', 
                         fontsize=16, fontweight='bold')
        
        # Flatten axes for easier access
        if self.axes is None:
            raise RuntimeError("Axes not initialized")
        ax_loss, ax_acc, ax_lr, ax_warning = self.axes.flatten()
        
        # 1. Loss curves subplot
        ax_loss.set_title('Training Progress - Loss', fontweight='bold')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend()
        
        # Initialize empty lines for loss
        self.lines['train_loss'], = ax_loss.plot([], [], 'b-', linewidth=2, label='Training Loss')
        self.lines['val_loss'], = ax_loss.plot([], [], 'r-', linewidth=2, label='Validation Loss')
        ax_loss.legend()
        
        # 2. Accuracy curves subplot  
        ax_acc.set_title('Training Progress - Accuracy', fontweight='bold')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim(0, 1)  # Accuracy is always 0-1
        
        # Initialize empty lines for accuracy
        self.lines['train_acc'], = ax_acc.plot([], [], 'b-', linewidth=2, label='Training Accuracy')
        self.lines['val_acc'], = ax_acc.plot([], [], 'r-', linewidth=2, label='Validation Accuracy')
        ax_acc.legend()
        
        # 3. Learning rate subplot
        ax_lr.set_title('Learning Rate Schedule', fontweight='bold')
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.grid(True, alpha=0.3)
        ax_lr.set_yscale('log')  # Log scale for learning rate
        
        # Initialize learning rate line
        self.lines['learning_rate'], = ax_lr.plot([], [], 'g-', linewidth=2, label='Learning Rate')
        ax_lr.legend()
        
        # 4. Warning/status subplot
        ax_warning.set_title('Training Status & Warnings', fontweight='bold')
        ax_warning.set_xlim(0, 1)
        ax_warning.set_ylim(0, 1)
        ax_warning.axis('off')  # Remove axes for text display
        
        # Initialize warning text
        self.warning_text = ax_warning.text(0.5, 0.5, 'Training Starting...', 
                                          ha='center', va='center', 
                                          fontsize=12, fontweight='bold',
                                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Adjust layout and show
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)  # Small pause to ensure plot displays
        
        self.is_plotting = True
        logger.debug("running setup_plots ... Real-time plots initialized and displayed")
    
    def update_plots(self, epoch: int, logs: Dict[str, float]) -> None:
        """
        Update all plots with new training data
        
        Args:
            epoch: Current epoch number
            logs: Dictionary containing training metrics from Keras
        """
        if not self.is_plotting:
            return
            
        logger.debug(f"running update_plots ... Updating plots for epoch {epoch + 1}")
        
        # Store data
        current_epoch = epoch + 1  # Keras uses 0-based indexing, we want 1-based
        self.epochs.append(current_epoch)
        self.current_epoch = current_epoch
        
        # Extract metrics from logs
        train_loss = logs.get('loss', 0.0)
        val_loss = logs.get('val_loss', None)
        train_acc = logs.get('accuracy', 0.0)
        val_acc = logs.get('val_accuracy', None)
        
        # Store training metrics
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_acc)
        
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_acc is not None:
            self.val_accuracy.append(val_acc)
        
        # Get learning rate from model
        lr = self._get_current_learning_rate()
        if lr is not None:
            self.learning_rates.append(lr)
        
        try:
            # Update loss plot
            self._update_loss_plot()
            
            # Update accuracy plot  
            self._update_accuracy_plot()
            
            # Update learning rate plot
            self._update_learning_rate_plot()
            
            # Update warning/status display
            self._update_warning_display(train_loss, val_loss, train_acc, val_acc)
            
            # Refresh the display (with type guard)
            if self.fig is not None:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            plt.pause(0.01)  # Small pause for smooth updates
            
            # Save intermediate plot if conditions are met
            if self.save_intermediate_plots and self._should_save_intermediate_plot(epoch):
                self._save_intermediate_plot(epoch)
            
        except Exception as e:
            logger.warning(f"running update_plots ... Error updating plots: {e}")
    
    def _update_loss_plot(self) -> None:
        """Update the loss curves subplot"""
        if self.axes is None:
            return
            
        ax_loss = self.axes[0, 0]
        
        # Update training loss line
        self.lines['train_loss'].set_data(self.epochs, self.train_loss)
        
        # Update validation loss line if data exists
        if self.val_loss:
            self.lines['val_loss'].set_data(self.epochs, self.val_loss)
        
        # Auto-scale axes
        if self.epochs:
            ax_loss.set_xlim(1, max(self.epochs) + 1)
            
            all_losses = self.train_loss + self.val_loss
            if all_losses:
                min_loss = min(all_losses)
                max_loss = max(all_losses)
                margin = (max_loss - min_loss) * 0.1
                ax_loss.set_ylim(max(0, min_loss - margin), max_loss + margin)
    
    def _update_accuracy_plot(self) -> None:
        """Update the accuracy curves subplot"""
        if self.axes is None:
            return
            
        ax_acc = self.axes[0, 1]
        
        # Update training accuracy line
        self.lines['train_acc'].set_data(self.epochs, self.train_accuracy)
        
        # Update validation accuracy line if data exists
        if self.val_accuracy:
            self.lines['val_acc'].set_data(self.epochs, self.val_accuracy)
        
        # Auto-scale x-axis, keep y-axis at 0-1
        if self.epochs:
            ax_acc.set_xlim(1, max(self.epochs) + 1)
    
    def _update_learning_rate_plot(self) -> None:
        """Update the learning rate subplot"""
        if self.axes is None:
            return
            
        ax_lr = self.axes[1, 0]
        
        if self.learning_rates and self.epochs:
            # Update learning rate line
            epochs_with_lr = self.epochs[:len(self.learning_rates)]
            self.lines['learning_rate'].set_data(epochs_with_lr, self.learning_rates)
            
            # Auto-scale axes
            ax_lr.set_xlim(1, max(self.epochs) + 1)
            if len(self.learning_rates) > 1:
                min_lr = min(self.learning_rates)
                max_lr = max(self.learning_rates)
                ax_lr.set_ylim(min_lr * 0.5, max_lr * 2)
    
    def _update_warning_display(self, train_loss: float, val_loss: Optional[float], 
                              train_acc: float, val_acc: Optional[float]) -> None:
        """
        Update the warning/status display with current training insights
        
        Args:
            train_loss: Current training loss
            val_loss: Current validation loss (if available)
            train_acc: Current training accuracy  
            val_acc: Current validation accuracy (if available)
        """
        warnings = []
        status_color = 'lightgreen'  # Default: good status
        
        # Check for overfitting
        if val_loss is not None and len(self.val_loss) >= 3:
            # Check if validation loss is increasing while training loss decreases
            recent_val_losses = self.val_loss[-3:]
            recent_train_losses = self.train_loss[-3:]
            
            val_trend = recent_val_losses[-1] - recent_val_losses[0]  # Positive = increasing
            train_trend = recent_train_losses[-1] - recent_train_losses[0]  # Negative = decreasing
            
            if val_trend > 0.1 and train_trend < -0.05:
                warnings.append("⚠️ OVERFITTING DETECTED")
                status_color = 'orange'
        
        # Check for poor performance
        if self.current_epoch >= 3:
            if train_acc < 0.5:
                warnings.append("📉 Low training accuracy")
                status_color = 'yellow'
            
            if val_acc is not None and val_acc < 0.4:
                warnings.append("📉 Low validation accuracy")
                status_color = 'yellow'
        
        # Check for learning rate issues
        if len(self.train_loss) >= 2:
            recent_loss_change = abs(self.train_loss[-1] - self.train_loss[-2])
            if recent_loss_change > 1.0:  # Large loss fluctuations
                warnings.append("🔄 Loss oscillating - LR too high?")
                status_color = 'yellow'
            elif recent_loss_change < 0.001 and self.current_epoch > 2:  # Very slow progress
                warnings.append("🐌 Slow progress - LR too low?")
                status_color = 'yellow'
        
        # Create status message
        if not warnings:
            if self.current_epoch == 1:
                message = "🚀 Training Started"
            elif train_acc > 0.9:
                message = "🎯 Excellent Progress!"
            elif train_acc > 0.7:
                message = "✅ Good Progress"
            else:
                message = "📈 Training in Progress"
        else:
            message = "\n".join(warnings)
        
        # Add epoch and time info
        elapsed = self._get_elapsed_time()
        epoch_info = f"Epoch: {self.current_epoch}\nElapsed: {elapsed}"
        
        # Add current metrics
        metrics_info = f"Loss: {train_loss:.4f}\nAccuracy: {train_acc:.3f}"
        if val_loss is not None:
            metrics_info += f"\nVal Loss: {val_loss:.4f}"
        if val_acc is not None:
            metrics_info += f"\nVal Acc: {val_acc:.3f}"
        
        full_message = f"{message}\n\n{epoch_info}\n\n{metrics_info}"
        
        # Update warning text
        if self.warning_text is not None:
            self.warning_text.set_text(full_message)
            self.warning_text.set_bbox(dict(boxstyle='round', facecolor=status_color, alpha=0.8))
    
    def _get_current_learning_rate(self) -> Optional[float]:
        """
        Extract current learning rate from the model
        
        Returns:
            Current learning rate as float, or None if unavailable
        """
        try:
            if (self.model_builder.model is not None and 
                hasattr(self.model_builder.model, 'optimizer') and
                self.model_builder.model.optimizer is not None):
                
                optimizer = self.model_builder.model.optimizer
                lr_value = optimizer.learning_rate
                
                # Handle different learning rate types
                if hasattr(lr_value, 'numpy'):
                    return float(lr_value.numpy())
                elif callable(lr_value):
                    # Learning rate schedule
                    step = len(self.epochs)
                    lr_result = lr_value(step)
                    # Handle different learning rate types
                    if hasattr(lr_value, 'numpy'):
                        numpy_method = getattr(lr_value, 'numpy')
                        return float(numpy_method())
                    else:
                        return float(lr_result) # type: ignore
                else:
                    return float(lr_value)
        except Exception as e:
            logger.debug(f"running _get_current_learning_rate ... Could not extract learning rate: {e}")
            return None
    
    def _get_elapsed_time(self) -> str:
        """Get formatted elapsed training time"""
        if self.start_time is None:
            return "Unknown"
        
        elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
        
        if elapsed_seconds < 60:
            return f"{elapsed_seconds:.0f}s"
        elif elapsed_seconds < 3600:
            minutes = elapsed_seconds // 60
            seconds = elapsed_seconds % 60
            return f"{minutes:.0f}m {seconds:.0f}s"
        else:
            hours = elapsed_seconds // 3600
            minutes = (elapsed_seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def start_training(self) -> None:
        """Mark the start of training"""
        self.start_time = datetime.now()
        logger.debug("running start_training ... Training start time recorded")
    
    def close_plots(self) -> None:
        """Clean up and close the real-time plots"""
        logger.debug("running close_plots ... Closing real-time training plots")
        
        self.is_plotting = False
        
        if self.fig is not None:
            plt.close(self.fig)
        
        # Turn off interactive mode
        plt.ioff()
        
        logger.debug("running close_plots ... Real-time plots closed")
    
    
    
    
    def _save_intermediate_plot(self, epoch: int) -> None:
        """
        Save current plot state as an intermediate plot
        
        Args:
            epoch: Current epoch number (0-based)
        """
        if self.fig is None or self.intermediate_plot_dir is None:
            logger.debug(f"running _save_intermediate_plot ... Skipping epoch {epoch + 1}: fig={self.fig is not None}, dir={self.intermediate_plot_dir is not None}")
            return
        
        current_epoch = epoch + 1  # Convert to 1-based
        filename = f"epoch_{current_epoch:02d}.png"  # e.g., "epoch_01.png", "epoch_05.png"
        filepath = self.intermediate_plot_dir / filename
        
        # Log the save attempt
        logger.debug(f"running _save_intermediate_plot ... Attempting to save epoch {current_epoch} to: {filepath}")
        
        try:
            # Ensure the plot is in a good state before saving
            if self.fig.canvas is not None:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            
            # Add a small delay to ensure matplotlib is ready
            import time
            time.sleep(0.1)
            
            # Save the plot
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            
            # Verify the file was actually saved
            if filepath.exists():
                file_size = filepath.stat().st_size
                logger.debug(f"running _save_intermediate_plot ... SUCCESS: Intermediate plot saved for epoch {current_epoch}: {filepath} ({file_size} bytes)")
            else:
                logger.warning(f"running _save_intermediate_plot ... File does not exist after save attempt: {filepath}")
                
        except Exception as e:
            logger.warning(f"running _save_intermediate_plot ... FAILED to save intermediate plot for epoch {current_epoch}: {e}")
            logger.debug(f"running _save_intermediate_plot ... Error details: {type(e).__name__}: {str(e)}")
            
            # Try alternative save method as fallback
            try:
                logger.debug(f"running _save_intermediate_plot ... Attempting fallback save method for epoch {current_epoch}")
                fallback_filename = f"epoch_{current_epoch:02d}_fallback.png"
                fallback_filepath = self.intermediate_plot_dir / fallback_filename
                
                plt.figure(self.fig.number)  # Ensure we're working with the right figure
                plt.savefig(fallback_filepath, dpi=300, bbox_inches='tight')
                
                if fallback_filepath.exists():
                    logger.debug(f"running _save_intermediate_plot ... Fallback save successful: {fallback_filepath}")
                else:
                    logger.warning(f"running _save_intermediate_plot ... Fallback save also failed for epoch {current_epoch}")
                    
            except Exception as fallback_error:
                logger.warning(f"running _save_intermediate_plot ... Fallback save method also failed: {fallback_error}")


    def _should_save_intermediate_plot(self, epoch: int) -> bool:
        """
        Determine if we should save an intermediate plot for this epoch
        
        Args:
            epoch: Current epoch number (0-based)
            
        Returns:
            True if we should save a plot
        """
        current_epoch = epoch + 1  # Convert to 1-based
        
        # Log the decision process for debugging
        should_save = (
            current_epoch == 1 or  # Always save first epoch
            current_epoch % self.save_every_n_epochs == 0 or  # Save every N epochs
            current_epoch == self.model_builder.model_config.epochs  # Save final epoch
        )
        
        logger.debug(f"running _should_save_intermediate_plot ... Epoch {current_epoch}: "
                    f"first_epoch={current_epoch == 1}, "
                    f"modulo_check={current_epoch % self.save_every_n_epochs == 0} "
                    f"(current_epoch={current_epoch} % save_every_n_epochs={self.save_every_n_epochs}), "
                    f"final_epoch={current_epoch == self.model_builder.model_config.epochs}, "
                    f"should_save={should_save}")
        
        return should_save
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def save_final_plot(self, run_timestamp: Optional[str] = None) -> None:
        if self.fig is None:
            return
        
        # Use provided plot directory or create default
        if self.plot_dir is not None:
            save_dir = self.plot_dir
        else:
            # Fallback to creating our own directory
            if run_timestamp is None:
                run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            
            dataset_name_clean = self.model_builder.dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            data_type = self.model_builder._detect_data_type()
            architecture_name = "CNN" if data_type == "image" else "LSTM"
            
            project_root = Path(__file__).resolve().parent.parent.parent
            plots_dir = project_root / "plots"
            save_dir = plots_dir / f"{run_timestamp}_{architecture_name}_{dataset_name_clean}"
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        dataset_name_clean = self.model_builder.dataset_config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        filename = f"realtime_training_{run_timestamp}_{dataset_name_clean}.png"
        filepath = save_dir / filename
        
        try:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.debug(f"running save_final_plot ... Final real-time training plot saved to: {filepath}")
            
            # ALSO save as final plot in intermediate directory
            if self.intermediate_plot_dir is not None:
                final_intermediate_path = self.intermediate_plot_dir / "final.png"
                self.fig.savefig(final_intermediate_path, dpi=300, bbox_inches='tight')
                logger.debug(f"running save_final_plot ... Final plot also saved to intermediate directory: {final_intermediate_path}")
                
                # Create a summary log file
                self._create_training_summary()
                
        except Exception as e:
            logger.warning(f"running save_final_plot ... Failed to save final real-time plot: {e}")


    def _create_training_summary(self) -> None:
        """
        Create a text summary of the training session
        """
        if self.intermediate_plot_dir is None:
            return
            
        summary_file = self.intermediate_plot_dir / "training_summary.txt"
        
        try:
            with open(summary_file, 'w') as f:
                f.write(f"Training Summary\n")
                f.write(f"================\n\n")
                f.write(f"Dataset: {self.model_builder.dataset_config.name}\n")
                f.write(f"Total Epochs: {len(self.epochs)}\n")
                f.write(f"Training Duration: {self._get_elapsed_time()}\n\n")
                
                if self.train_loss:
                    f.write(f"Final Training Loss: {self.train_loss[-1]:.4f}\n")
                    f.write(f"Final Training Accuracy: {self.train_accuracy[-1]:.4f}\n")
                
                if self.val_loss:
                    f.write(f"Final Validation Loss: {self.val_loss[-1]:.4f}\n")
                    f.write(f"Final Validation Accuracy: {self.val_accuracy[-1]:.4f}\n")
                
                f.write(f"\nPlot files saved every {self.save_every_n_epochs} epochs\n")
                f.write(f"Total intermediate plots: {len(list(self.intermediate_plot_dir.glob('epoch_*.png')))}\n")
                
            logger.debug(f"running _create_training_summary ... Training summary saved to: {summary_file}")
            
        except Exception as e:
            logger.warning(f"running _create_training_summary ... Failed to create training summary: {e}")


class RealTimeTrainingCallback(keras.callbacks.Callback):
    """
    Keras callback that integrates with RealTimeTrainingVisualizer
    
    This callback is passed to model.fit() and calls the visualizer's
    update_plots method after each epoch.
    """
    
    def __init__(self, visualizer: RealTimeTrainingVisualizer):
        """
        Initialize the callback
        
        Args:
            visualizer: RealTimeTrainingVisualizer instance
        """
        super().__init__()
        self.visualizer = visualizer
        logger.debug("running RealTimeTrainingCallback.__init__ ... Real-time training callback initialized")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training"""
        logger.debug("running on_train_begin ... Setting up real-time visualization")
        self.visualizer.setup_plots()
        self.visualizer.start_training()
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of each epoch
        
        Args:
            epoch: Current epoch number (0-based)
            logs: Dictionary containing training metrics
        """
        if logs is None:
            logs = {}
        
        # Update the real-time plots
        self.visualizer.update_plots(epoch, logs)
        
        # Log epoch completion
        epoch_num = epoch + 1
        train_loss = logs.get('loss', 'N/A')
        train_acc = logs.get('accuracy', 'N/A')
        val_loss = logs.get('val_loss', 'N/A')
        val_acc = logs.get('val_accuracy', 'N/A')
        
        logger.debug(f"running on_epoch_end ... Epoch {epoch_num}: "
                    f"loss={train_loss}, acc={train_acc}, "
                    f"val_loss={val_loss}, val_acc={val_acc}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training"""
        logger.debug("running on_train_end ... Training completed, saving final real-time plot")
        
        # Save final plot
        self.visualizer.save_final_plot()
        
        # Keep plots open briefly for final review
        time.sleep(2)
        
        # Close plots
        self.visualizer.close_plots()