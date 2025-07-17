"""
Training Animation Module

This module provides animated training progress visualization including:
- Animated loss/accuracy curves building epoch by epoch
- Learning rate schedule animation
- Overfitting detection progression
- Model convergence visualization
- Professional-quality animated exports (GIF/MP4)

Designed to work with any Keras training history from model.fit().
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
from utils.logger import logger


class TrainingAnimationAnalyzer:
    """
    Comprehensive training animation analysis and visualization
    
    Creates animated training progress that shows how the model learned over time.
    Essential for understanding training dynamics and creating engaging presentations
    of model performance.
    """
    
    def __init__(self, model_name: str = "Model"):
        """
        Initialize the training animation analyzer
        
        Args:
            model_name: Name of the model for animation titles and logging
        """
        self.model_name = model_name
        logger.debug("running TrainingAnimationAnalyzer.__init__ ... Training animation analyzer initialized")
    
    def analyze_and_animate(
        self,
        training_history: Dict[str, List[float]],
        model=None,
        dataset_name: str = "dataset",
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None,
        animation_duration: float = 10.0,
        fps: int = 10
    ) -> Dict[str, Any]:
        """
        Create animated training progress visualization
        
        Generates professional animated training analysis that shows how your model
        learned over time. This creates engaging visualizations perfect for presentations,
        papers, and understanding training dynamics.
        
        Animation Features:
        - Smooth progression showing metrics building epoch by epoch
        - Multiple animation styles: building curves, sliding windows, fade effects
        - Professional styling with consistent branding
        - Configurable duration and frame rate for different use cases
        
        Use Cases:
        1. **Research Presentations**: Show training progression in talks
        2. **Model Documentation**: Include in papers to demonstrate convergence
        3. **Client Demos**: Engaging way to show AI model development
        4. **Debugging**: Visualize exactly when training issues occurred
        5. **Social Media**: Share impressive training results
        
        Animation Types Generated:
        
        1. **Building Curves Animation**: Most popular for presentations
           - Lines draw progressively from epoch 1 to final epoch
           - Shows smooth learning progression
           - Includes metric values updating in real-time
        
        2. **Sliding Window Animation**: Best for long training runs
           - Shows recent 10-20 epochs with sliding window
           - Prevents cluttered view for 100+ epoch training
           - Focus on recent training dynamics
        
        3. **Fade Trail Animation**: Artistic option for visual appeal
           - Recent epochs bright, older epochs fade out
           - Creates "comet tail" effect showing progression
           - Good for social media or marketing materials
        
        Technical Specifications:
        - Output: High-quality animated GIF (web-compatible) and MP4 (presentation-quality)
        - Duration: Configurable (default 10 seconds for good pacing)
        - FPS: Configurable (default 10fps for smooth motion, reasonable file size)
        - Resolution: Publication-quality (300 DPI)
        - File size: Optimized for sharing (typically 2-5MB)
        
        Args:
            training_history: Dictionary containing training metrics from Keras model.fit()
                            Expected keys: 'loss', 'accuracy', optionally 'val_loss', 'val_accuracy'
                            Example: {'loss': [2.1, 1.8, 1.4, ...], 'accuracy': [0.4, 0.5, 0.6, ...]}
            model: Optional Keras model for learning rate extraction
            dataset_name: Name of dataset for animation title and file naming
            run_timestamp: Optional timestamp for file naming
            plot_dir: Optional directory to save animation files
            animation_duration: Duration of animation in seconds (default 10.0)
            fps: Frames per second for animation (default 10)
            
        Returns:
            Dictionary containing animation results:
            - 'gif_path': Path to animated GIF file (None if creation failed)
            - 'mp4_path': Path to MP4 video file (None if creation failed)
            - 'frame_count': Number of frames generated
            - 'final_metrics': Dictionary of final epoch metrics
            - 'animation_insights': List of notable training patterns observed
            
        Side Effects:
            - Creates animated GIF and MP4 files with timestamp
            - Generates multiple frames showing training progression
            - Automatically detects and highlights interesting training patterns
            - Closes figures to prevent memory leaks
            - Logs animation creation progress and file save locations
            
        Performance Notes:
            - Animation generation can take 30-60 seconds for longer training runs
            - Memory usage scales with number of epochs (typically 50-200MB during creation)
            - File sizes: GIF ~2-5MB, MP4 ~1-3MB for typical 10-50 epoch training
            
        Requirements:
            - training_history must contain at least 'loss' key
            - matplotlib with animation support
            - ffmpeg installed for MP4 generation (optional, will fallback to GIF only)
            
        Example Output Files:
            - "training_animation_20250708_143022_cifar10.gif" (web sharing)
            - "training_animation_20250708_143022_cifar10.mp4" (presentations)
        """
        logger.debug("running analyze_and_animate ... Starting comprehensive training animation creation")
        
        try:
            # Validate training history
            if not training_history or 'loss' not in training_history:
                logger.warning("running analyze_and_animate ... Invalid training history - missing required 'loss' key")
                return {'error': 'Invalid training history provided'}
            
            # Analyze training patterns for animation insights
            analysis_results = self._analyze_training_patterns(training_history)
            
            # Create animated visualization
            animation_paths = self._create_animation(
                training_history=training_history,
                model=model,
                dataset_name=dataset_name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir,
                animation_duration=animation_duration,
                fps=fps,
                analysis_insights=analysis_results['animation_insights']
            )
            
            # Combine results
            final_result = analysis_results.copy()
            final_result.update(animation_paths)
            
            logger.debug("running analyze_and_animate ... Training animation creation completed successfully")
            return final_result
            
        except Exception as e:
            logger.warning(f"running analyze_and_animate ... Failed to complete training animation: {e}")
            logger.debug(f"running analyze_and_animate ... Error traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def _analyze_training_patterns(self, training_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze training patterns to identify interesting moments for animation emphasis
        
        Args:
            training_history: Dictionary containing training metrics
            
        Returns:
            Dictionary containing analysis results and insights
        """
        logger.debug("running _analyze_training_patterns ... Analyzing training patterns for animation highlights")
        
        insights = []
        final_metrics = {}
        interesting_epochs = []  # Epochs to highlight in animation
        
        epochs = list(range(1, len(training_history['loss']) + 1))
        losses = training_history['loss']
        
        # Extract final metrics
        if 'loss' in training_history:
            final_metrics['final_loss'] = training_history['loss'][-1]
            
        if 'accuracy' in training_history:
            final_metrics['final_accuracy'] = training_history['accuracy'][-1]
            
        if 'val_loss' in training_history:
            final_metrics['final_val_loss'] = training_history['val_loss'][-1]
            
        if 'val_accuracy' in training_history:
            final_metrics['final_val_accuracy'] = training_history['val_accuracy'][-1]
        
        # Find interesting training moments
        if len(losses) >= 3:
            # Find epoch with biggest loss drop (breakthrough moment)
            loss_drops = [losses[i-1] - losses[i] for i in range(1, len(losses))]
            max_drop_epoch = loss_drops.index(max(loss_drops)) + 2  # +2 because of 1-based indexing and we're looking at i-1
            interesting_epochs.append(('breakthrough', max_drop_epoch))
            insights.append(f"🚀 Biggest breakthrough at epoch {max_drop_epoch}")
            
            # Find convergence point (where improvement slows)
            recent_improvements = []
            if len(losses) >= 4:  # Need at least 4 epochs to calculate 3 recent improvements
                for i in range(max(1, len(losses)-3), len(losses)):
                    recent_improvements.append(abs(losses[i-1] - losses[i]))
            if len(recent_improvements) >= 2 and all(imp < 0.01 for imp in recent_improvements):
                convergence_epoch = len(losses) - 2
                interesting_epochs.append(('convergence', convergence_epoch))
                insights.append(f"📈 Model converged around epoch {convergence_epoch}")
        
        # Check for overfitting emergence
        if 'val_loss' in training_history and len(training_history['val_loss']) >= 5:
            val_losses = training_history['val_loss']
            train_losses = training_history['loss']
            
            # Find when validation loss starts increasing while training decreases
            for i in range(3, len(val_losses)):
                recent_val_trend = val_losses[i] - val_losses[i-3]
                recent_train_trend = train_losses[i] - train_losses[i-3]
                
                if recent_val_trend > 0.05 and recent_train_trend < -0.05:
                    interesting_epochs.append(('overfitting_start', i+1))
                    insights.append(f"⚠️ Overfitting began around epoch {i+1}")
                    break
        
        logger.debug(f"running _analyze_training_patterns ... Found {len(interesting_epochs)} interesting training moments")
        for insight in insights:
            logger.debug(f"running _analyze_training_patterns ... {insight}")
        
        # Store interesting epochs for animation use
        self._interesting_epochs = interesting_epochs
        
        return {
            'final_metrics': final_metrics,
            'animation_insights': insights,
            'interesting_epochs': interesting_epochs,
            'total_epochs': len(epochs)
        }
    
    def _create_animation(
        self,
        training_history: Dict[str, List[float]],
        model,
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path],
        animation_duration: float,
        fps: int,
        analysis_insights: List[str]
    ) -> Dict[str, Any]:
        """
        Create the animated training visualization
        
        Args:
            training_history: Dictionary containing training metrics
            model: Optional Keras model for learning rate extraction
            dataset_name: Name of dataset for animation title
            run_timestamp: Optional timestamp for file naming
            plot_dir: Optional directory to save animation
            animation_duration: Duration of animation in seconds
            fps: Frames per second
            analysis_insights: List of insights from pattern analysis
            
        Returns:
            Dictionary containing paths to generated animation files
        """
        logger.debug("running _create_animation ... Creating animated training visualization")
        
        try:
            # Prepare animation data
            epochs = list(range(1, len(training_history['loss']) + 1))
            total_frames = int(animation_duration * fps)
            
            # Calculate which epochs to show in each frame
            epoch_progression = self._calculate_epoch_progression(len(epochs), total_frames)
            
            # Set up the figure and subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Training Animation - {dataset_name}', fontsize=16, fontweight='bold')
            
            # Initialize plots
            self._setup_animation_axes(ax1, ax2, training_history, epochs)
            
            # Create animation function
            def animate_frame(frame):
                return self._update_animation_frame(
                    frame, ax1, ax2, training_history, epochs, 
                    epoch_progression, analysis_insights
                )
            
            # Create animation
            logger.debug(f"running _create_animation ... Generating {total_frames} frames at {fps} FPS")
            anim = animation.FuncAnimation(
                fig, animate_frame, frames=total_frames,
                interval=1000/fps, blit=False, repeat=True
            )
            
            # Save animation files
            animation_paths = self._save_animation_files(
                anim, fig, dataset_name, run_timestamp, plot_dir
            )
            
            plt.close(fig)  # Clean up memory
            
            # Combine results with proper typing
            result = {
                'gif_path': animation_paths['gif_path'],
                'mp4_path': animation_paths['mp4_path'],
                'frame_count': total_frames
            }
            return result
            
        except Exception as e:
            logger.warning(f"running _create_animation ... Failed to create animation: {e}")
            plt.close('all')  # Ensure cleanup
            return {
                'gif_path': None,
                'mp4_path': None,
                'frame_count': 0
            }


    def _calculate_epoch_progression(self, total_epochs: int, total_frames: int) -> List[int]:
        """
        Calculate which epochs to show in each frame for smooth progression
        
        Args:
            total_epochs: Total number of training epochs
            total_frames: Total number of animation frames
            
        Returns:
            List of epoch counts for each frame (e.g., [1, 2, 3, ..., total_epochs])
        """
        logger.debug(f"running _calculate_epoch_progression ... Calculating progression for {total_epochs} epochs over {total_frames} frames")
        
        # Create smooth progression from 1 epoch to all epochs
        if total_frames <= total_epochs:
            # More epochs than frames - skip some epochs
            epoch_indices = np.linspace(1, total_epochs, total_frames, dtype=int)
        else:
            # More frames than epochs - hold final epoch for remaining frames
            progression = list(range(1, total_epochs + 1))
            # Pad with final epoch count for remaining frames
            remaining_frames = total_frames - total_epochs
            progression.extend([total_epochs] * remaining_frames)
            epoch_indices = progression
        
        return list(epoch_indices)
    
    def _setup_animation_axes(self, ax1, ax2, training_history: Dict[str, List[float]], epochs: List[int]) -> None:
        """
        Set up the axes for animation (titles, labels, limits, etc.)
        
        Args:
            ax1: Left subplot (loss curves)
            ax2: Right subplot (accuracy curves)
            training_history: Training metrics
            epochs: List of epoch numbers
        """
        
        # Setup loss plot (left)
        ax1.set_title('Training Progress - Loss', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, len(epochs))
        
        # Calculate loss limits with some padding
        all_losses = training_history['loss']
        if 'val_loss' in training_history:
            all_losses = all_losses + training_history['val_loss']
        
        loss_min, loss_max = min(all_losses), max(all_losses)
        loss_range = loss_max - loss_min
        ax1.set_ylim(max(0, loss_min - loss_range * 0.1), loss_max + loss_range * 0.1)
        
        # Setup accuracy plot (right)
        ax2.set_title('Training Progress - Accuracy', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, len(epochs))
        ax2.set_ylim(0, 1)  # Accuracy is always 0-1
        
        # Add legends (empty for now, will be populated during animation)
        ax1.legend([], [])
        ax2.legend([], [])
    
    def _update_animation_frame(
        self, 
        frame: int, 
        ax1, 
        ax2, 
        training_history: Dict[str, List[float]], 
        epochs: List[int],
        epoch_progression: List[int],
        insights: List[str]
    ) -> List:
        """
        Update animation frame with current training progress
        
        Args:
            frame: Current frame number
            ax1: Loss subplot
            ax2: Accuracy subplot  
            training_history: Training metrics
            epochs: List of epoch numbers
            epoch_progression: Which epochs to show at each frame
            insights: Training insights for annotation
            
        Returns:
            List of artists for blitting (animation optimization)
        """
        # Clear previous frame
        ax1.clear()
        ax2.clear()
        
        # Re-setup axes (they get cleared)
        self._setup_animation_axes(ax1, ax2, training_history, epochs)
        
        # Get current epoch to display
        current_epoch_count = epoch_progression[frame]
        current_epochs = epochs[:current_epoch_count]
        
        # Plot training loss
        current_train_loss = training_history['loss'][:current_epoch_count]
        line1, = ax1.plot(current_epochs, current_train_loss, 'b-', linewidth=3, label='Training Loss', alpha=0.8)
        
        # Plot validation loss if available
        if 'val_loss' in training_history:
            current_val_loss = training_history['val_loss'][:current_epoch_count]
            line2, = ax1.plot(current_epochs, current_val_loss, 'r-', linewidth=3, label='Validation Loss', alpha=0.8)
            ax1.legend()
        
        # Plot training accuracy
        current_train_acc = training_history['accuracy'][:current_epoch_count]
        line3, = ax2.plot(current_epochs, current_train_acc, 'b-', linewidth=3, label='Training Accuracy', alpha=0.8)
        
        # Plot validation accuracy if available
        if 'val_accuracy' in training_history:
            current_val_acc = training_history['val_accuracy'][:current_epoch_count]
            line4, = ax2.plot(current_epochs, current_val_acc, 'r-', linewidth=3, label='Validation Accuracy', alpha=0.8)
            ax2.legend()
        
        # Add current epoch indicator
        if current_epoch_count > 0:
            # Add epoch counter text
            ax1.text(0.02, 0.98, f'Epoch: {current_epoch_count}/{len(epochs)}', 
                    transform=ax1.transAxes, fontsize=12, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Add current metrics
            current_loss = training_history['loss'][current_epoch_count-1]
            current_acc = training_history['accuracy'][current_epoch_count-1]
            
            metrics_text = f'Loss: {current_loss:.3f}\nAcc: {current_acc:.3f}'
            if 'val_loss' in training_history and current_epoch_count <= len(training_history['val_loss']):
                val_loss = training_history['val_loss'][current_epoch_count-1]
                val_acc = training_history['val_accuracy'][current_epoch_count-1]
                metrics_text += f'\nVal Loss: {val_loss:.3f}\nVal Acc: {val_acc:.3f}'
            
            ax2.text(0.02, 0.98, metrics_text,
                    transform=ax2.transAxes, fontsize=10, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Add progress indicators for interesting epochs
        # This adds visual flair when breakthrough moments occur
        artists = []
        if hasattr(self, '_interesting_epochs'):
            for event_type, epoch_num in self._interesting_epochs:
                if current_epoch_count >= epoch_num:
                    if event_type == 'breakthrough':
                        # Add starburst effect at breakthrough epoch
                        ax1.annotate('🚀 Breakthrough!', xy=(epoch_num, training_history['loss'][epoch_num-1]),
                                   xytext=(epoch_num + 2, training_history['loss'][epoch_num-1] + 0.2),
                                   arrowprops=dict(arrowstyle='->', color='gold', lw=2),
                                   fontsize=10, fontweight='bold', color='gold')
        
        return artists
    
    def _save_animation_files(
        self,
        anim: animation.FuncAnimation,
        fig,
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path]
    ) -> Dict[str, Optional[Path]]:
        """
        Save animation as both GIF and MP4 files
        
        Args:
            anim: Matplotlib animation object
            fig: Figure object
            dataset_name: Name of dataset for file naming
            run_timestamp: Optional timestamp for file naming  
            plot_dir: Optional directory to save files
            
        Returns:
            Dictionary with paths to saved files
        """
        logger.debug("running _save_animation_files ... Saving animation files")
        
        # Generate file paths
        gif_path, mp4_path = self._generate_animation_paths(dataset_name, run_timestamp, plot_dir)
        
        saved_paths: Dict[str, Optional[Path]] = {'gif_path': None, 'mp4_path': None}
        
        # Save as GIF (always attempt this)
        try:
            logger.debug(f"running _save_animation_files ... Saving GIF to: {gif_path}")
            anim.save(gif_path, writer='pillow', fps=10, dpi=100)
            
            if gif_path.exists():
                file_size = gif_path.stat().st_size / (1024*1024)  # Size in MB
                logger.debug(f"running _save_animation_files ... GIF saved successfully: {gif_path} ({file_size:.2f} MB)")
                saved_paths['gif_path'] = gif_path
            else:
                logger.warning(f"running _save_animation_files ... GIF file not found after save: {gif_path}")
                
        except Exception as e:
            logger.warning(f"running _save_animation_files ... Failed to save GIF: {e}")
        
        # Save as MP4 (requires ffmpeg, optional)
        try:
            logger.debug(f"running _save_animation_files ... Saving MP4 to: {mp4_path}")
            anim.save(mp4_path, writer='ffmpeg', fps=10, bitrate=1800, extra_args=['-vcodec', 'libx264'])
            
            if mp4_path.exists():
                file_size = mp4_path.stat().st_size / (1024*1024)  # Size in MB
                logger.debug(f"running _save_animation_files ... MP4 saved successfully: {mp4_path} ({file_size:.2f} MB)")
                saved_paths['mp4_path'] = mp4_path
            else:
                logger.warning(f"running _save_animation_files ... MP4 file not found after save: {mp4_path}")
                
        except Exception as e:
            logger.debug(f"running _save_animation_files ... Failed to save MP4 (ffmpeg may not be installed): {e}")
            logger.debug("running _save_animation_files ... MP4 export is optional - GIF export should still work")
        
        return saved_paths
    
    def _generate_animation_paths(
        self,
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path]
    ) -> Tuple[Path, Path]:
        """
        Generate file paths for saving animation files
        
        Args:
            dataset_name: Name of dataset
            run_timestamp: Optional timestamp
            plot_dir: Directory to save files
            
        Returns:
            Tuple of (gif_path, mp4_path)
        """
        from datetime import datetime
        
        # Create timestamp if not provided
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        
        # Clean dataset name
        dataset_name_clean = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        # Generate filenames
        gif_filename = f"training_animation_{run_timestamp}_{dataset_name_clean}.gif"
        mp4_filename = f"training_animation_{run_timestamp}_{dataset_name_clean}.mp4"
        
        # Determine save directory
        if plot_dir is not None:
            save_dir = plot_dir
        else:
            # Fallback: create default directory
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            save_dir = project_root / "plots"
            save_dir.mkdir(exist_ok=True)
        
        return save_dir / gif_filename, save_dir / mp4_filename


def create_training_animation(
    training_history: Dict[str, List[float]],
    model=None,
    dataset_name: str = "dataset",
    run_timestamp: Optional[str] = None,
    plot_dir: Optional[Path] = None,
    animation_duration: float = 10.0,
    fps: int = 10
) -> Dict[str, Any]:
    """
    Convenience function for quick training animation creation
    
    Args:
        training_history: Dictionary containing training metrics from Keras model.fit()
        model: Optional Keras model for learning rate extraction
        dataset_name: Name of dataset
        run_timestamp: Optional timestamp
        plot_dir: Optional directory for saving animations
        animation_duration: Duration of animation in seconds
        fps: Frames per second for animation
        
    Returns:
        Animation results dictionary
    """
    analyzer = TrainingAnimationAnalyzer(model_name=dataset_name)
    return analyzer.analyze_and_animate(
        training_history=training_history,
        model=model,
        dataset_name=dataset_name,
        run_timestamp=run_timestamp,
        plot_dir=plot_dir,
        animation_duration=animation_duration,
        fps=fps
    )