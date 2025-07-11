"""
Training History Visualization Module

This module provides comprehensive training progress analysis including:
- Loss and accuracy curves over time
- Learning rate visualization 
- Overfitting detection and warnings
- Professional-quality plots for presentations and papers
- Training dynamics analysis and interpretation

Designed to work with any Keras training history from model.fit().
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
from utils.logger import logger


class TrainingHistoryAnalyzer:
    """
    Comprehensive training history analysis and visualization
    
    Creates professional training analysis dashboard that shows how the model
    learned over time. Essential for understanding training dynamics,
    diagnosing problems, and optimizing future training runs.
    """
    
    def __init__(self, model_name: str = "Model"):
        """
        Initialize the training history analyzer
        
        Args:
            model_name: Name of the model for plot titles and logging
        """
        self.model_name = model_name
        logger.debug("running TrainingHistoryAnalyzer.__init__ ... Training history analyzer initialized")
    
    def analyze_and_visualize(
        self,
        training_history: Dict[str, List[float]],
        model=None,
        dataset_name: str = "dataset",
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive training progress visualizations in a 2x2 dashboard
        
        Generates a professional training analysis dashboard that shows how your model
        learned over time. This is essential for understanding training dynamics,
        diagnosing problems, and optimizing future training runs.
        
        Dashboard Layout (2x2 Grid):
        
        ┌─────────────────┬─────────────────┐
        │ 1. Loss Curves  │ 2. Accuracy     │  ← Most important plots
        │                 │    Curves       │
        ├─────────────────┼─────────────────┤
        │ 3. Learning     │ 4. Overfitting  │  ← Advanced diagnostics
        │    Rate         │    Indicators   │
        └─────────────────┴─────────────────┘
        
        Plot 1: Loss Curves (TOP PRIORITY)
        Purpose: Shows how well the model is learning to minimize errors
        
        What to Look For:
        ✅ Good Training:
            - Both lines decreasing steadily
            - Training loss slightly below validation loss  
            - Converging toward a low value (< 0.5 for most tasks)
            
        ❌ Problem Patterns:
            - Loss increasing: Learning rate too high or data problems
            - Validation loss increasing while training decreases: OVERFITTING
            - Both plateauing early: Learning rate too low or model too simple
            - Spiky/erratic: Batch size too small or data quality issues
        
        Example Interpretation:
        ```
        Epoch 1-3: Loss drops rapidly (model learning basic patterns)
        Epoch 4-7: Steady improvement (refining understanding)  
        Epoch 8-10: Plateauing (diminishing returns, consider stopping)
        ```
        
        Plot 2: Accuracy Curves
        Purpose: Shows classification performance improving over time
        
        Ideal Pattern:
            - Smooth increase from ~10% (random) to 85%+ (good performance)
            - Training accuracy slightly above validation accuracy
            - Both curves stabilizing at high values
            
        Warning Signs:
            - Large gap between training/validation: Overfitting
            - Accuracy decreasing: Model is getting confused, reduce learning rate
            - Oscillating: Training is unstable
        
        Plot 3: Learning Rate Display
        Purpose: Shows the learning rate used (constant or scheduled)
        
        Information Provided:
            - Constant rate: Displays single value (e.g., 0.001000)
            - Learning rate schedule: Shows how rate changed over time
            - Helps explain training behavior patterns
            
        Interpretation:
            - Higher rates (0.01+): Fast learning but potentially unstable
            - Lower rates (0.0001): Stable but slow learning
            - Scheduled rates: May explain sudden changes in loss curves
        
        Plot 4: Overfitting Analysis (CRITICAL FOR MODEL RELIABILITY)
        Purpose: Detects if model memorized training data vs learning generalizable patterns
        
        Key Metrics:
            - Loss Gap: Training Loss - Validation Loss (should be near 0)
            - Accuracy Gap: Validation Accuracy - Training Accuracy (should be near 0)
            
        Overfitting Warnings:
            🟨 Caution: Loss gap > 0.1 or accuracy gap < -0.05
            🟥 Danger: Loss gap > 0.3 or accuracy gap < -0.10
            
        Real-World Implications:
            - Overfitted model: Works great on training data, fails on new traffic signs
            - Well-generalized model: Consistent performance on new, unseen images
        
        Dashboard Usage Scenarios:
        
        1. **During Training**: Monitor for early stopping
        - Stop if validation loss starts increasing consistently
        - Stop if overfitting indicators appear
        
        2. **After Training**: Diagnose issues
        - Low accuracy: Need more data, better architecture, or longer training
        - Overfitting: Add dropout, get more data, or reduce model complexity
        
        3. **Model Comparison**: Compare different architectures
        - Which converges faster?
        - Which achieves better final performance?
        - Which generalizes better?
        
        Professional Applications:
        - Include in research papers to show training stability
        - Present to stakeholders to justify model reliability  
        - Use for debugging when model performance is unexpected
        - Guide hyperparameter tuning decisions
        
        Troubleshooting Common Patterns:
        
        Problem: "My loss goes down but accuracy stays low"
        Solution: Model is learning but not well enough for correct classifications
        Action: Train longer, get more data, or improve architecture
        
        Problem: "Training accuracy is 99% but validation is 70%"  
        Solution: Severe overfitting
        Action: Add dropout, reduce model complexity, get more training data
        
        Problem: "Loss is spiky and erratic"
        Solution: Training is unstable
        Action: Reduce learning rate, increase batch size, check data quality
        
        Args:
            training_history: Dictionary containing training metrics from Keras model.fit()
                            Expected keys: 'loss', 'accuracy', optionally 'val_loss', 'val_accuracy'
                            Example: {'loss': [2.1, 1.8, 1.4, ...], 'accuracy': [0.4, 0.5, 0.6, ...]}
            model: Optional Keras model for learning rate extraction
            dataset_name: Name of dataset for plot title and file naming
            run_timestamp: Optional timestamp for file naming
            plot_dir: Optional directory to save plots
            
        Returns:
            Dictionary containing analysis results:
            - 'visualization_path': Path to saved plot file (None if save failed)
            - 'overfitting_detected': Boolean indicating if overfitting was detected
            - 'final_metrics': Dictionary of final epoch metrics
            - 'training_insights': List of diagnostic messages about training quality
            
        Side Effects:
            - Creates comprehensive matplotlib figure with 4 subplots
            - Saves high-resolution PNG file with timestamp
            - Automatically detects and warns about overfitting patterns
            - Closes figure to prevent memory leaks
            - Logs analysis insights and file save location
            
        Requirements:
            - training_history must contain at least 'loss' and 'accuracy' keys
            - Each metric should be a list of floats (one per epoch)
            - Matplotlib and numpy for visualization
            
        Note: If validation data wasn't used during training, validation plots will show 
            "Not Available" placeholders, but training metrics will still be displayed.
        """
        logger.debug("running analyze_and_visualize ... Starting comprehensive training history analysis")
        
        try:
            # Validate training history
            if not training_history or 'loss' not in training_history:
                logger.warning("running analyze_and_visualize ... Invalid training history - missing required 'loss' key")
                return {'error': 'Invalid training history provided'}
            
            # Analyze training metrics
            analysis_results = self._analyze_training_metrics(training_history)
            
            # Create visualization
            visualization_path = self._create_visualization(
                training_history=training_history,
                model=model,
                dataset_name=dataset_name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir,
                analysis_insights=analysis_results['training_insights']
            )
            
            analysis_results['visualization_path'] = visualization_path
            
            logger.debug("running analyze_and_visualize ... Training history analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.warning(f"running analyze_and_visualize ... Failed to complete training history analysis: {e}")
            logger.debug(f"running analyze_and_visualize ... Error traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def _analyze_training_metrics(self, training_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze training metrics to detect patterns and issues
        
        Args:
            training_history: Dictionary containing training metrics
            
        Returns:
            Dictionary containing analysis results
        """
        logger.debug("running _analyze_training_metrics ... Analyzing training patterns")
        
        insights = []
        final_metrics = {}
        overfitting_detected = False
        
        # Extract final metrics
        if 'loss' in training_history:
            final_metrics['final_loss'] = training_history['loss'][-1]
            logger.debug(f"running _analyze_training_metrics ... Final training loss: {final_metrics['final_loss']:.4f}")
        
        if 'accuracy' in training_history:
            final_metrics['final_accuracy'] = training_history['accuracy'][-1]
            logger.debug(f"running _analyze_training_metrics ... Final training accuracy: {final_metrics['final_accuracy']:.4f}")
        
        if 'val_loss' in training_history:
            final_metrics['final_val_loss'] = training_history['val_loss'][-1]
            logger.debug(f"running _analyze_training_metrics ... Final validation loss: {final_metrics['final_val_loss']:.4f}")
        
        if 'val_accuracy' in training_history:
            final_metrics['final_val_accuracy'] = training_history['val_accuracy'][-1]
            logger.debug(f"running _analyze_training_metrics ... Final validation accuracy: {final_metrics['final_val_accuracy']:.4f}")
        
        # Analyze training quality
        if len(training_history['loss']) >= 3:
            # Check for loss trends
            recent_losses = training_history['loss'][-3:]
            if recent_losses[0] < recent_losses[-1]:
                insights.append("⚠️ Training loss increased in recent epochs - possible learning rate too high")
            elif all(abs(recent_losses[i] - recent_losses[i+1]) < 0.001 for i in range(len(recent_losses)-1)):
                insights.append("📈 Training loss plateaued - consider early stopping or learning rate adjustment")
            else:
                insights.append("✅ Training loss showing healthy decrease")
        
        # Check for overfitting
        if 'val_loss' in training_history and 'val_accuracy' in training_history:
            # Calculate gaps
            loss_gap = final_metrics.get('final_loss', 0) - final_metrics.get('final_val_loss', 0)
            acc_gap = final_metrics.get('final_val_accuracy', 0) - final_metrics.get('final_accuracy', 0)
            
            if loss_gap > 0.3 or acc_gap < -0.10:
                overfitting_detected = True
                insights.append("🟥 SEVERE OVERFITTING: Large gap between training and validation metrics")
            elif loss_gap > 0.1 or acc_gap < -0.05:
                overfitting_detected = True
                insights.append("🟨 MILD OVERFITTING: Consider adding regularization")
            else:
                insights.append("✅ Good generalization - training and validation metrics aligned")
        
        # Check final performance
        final_acc = final_metrics.get('final_accuracy', 0)
        if final_acc > 0.95:
            insights.append("🎯 Excellent training performance achieved")
        elif final_acc > 0.85:
            insights.append("✅ Good training performance")
        elif final_acc > 0.70:
            insights.append("📊 Moderate training performance - room for improvement")
        else:
            insights.append("📉 Low training performance - consider architecture changes")
        
        # Log insights
        logger.debug("running _analyze_training_metrics ... Training insights:")
        for insight in insights:
            logger.debug(f"running _analyze_training_metrics ... {insight}")
        
        return {
            'final_metrics': final_metrics,
            'overfitting_detected': overfitting_detected,
            'training_insights': insights
        }
    
    def _create_visualization(
        self,
        training_history: Dict[str, List[float]],
        model,
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path],
        analysis_insights: List[str]
    ) -> Optional[Path]:
        """
        Create the comprehensive training history visualization dashboard
        
        Args:
            training_history: Dictionary containing training metrics
            model: Optional Keras model for learning rate extraction
            dataset_name: Name of dataset for plot title
            run_timestamp: Optional timestamp for file naming
            plot_dir: Optional directory to save plot
            analysis_insights: List of insights from analysis
            
        Returns:
            Path to saved plot file, or None if save failed
        """
        logger.debug("running _create_visualization ... Creating training history dashboard")
        
        try:
            # Prepare data
            epochs = range(1, len(training_history['loss']) + 1)
            
            # Create subplot layout: 2x2 grid
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - {dataset_name}', fontsize=16, fontweight='bold')
            
            # 1. Loss curves (most important!)
            ax1.plot(epochs, training_history['loss'], 'b-', label='Training Loss', linewidth=2)
            if 'val_loss' in training_history:
                ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title('Model Loss Over Time', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Accuracy curves
            ax2.plot(epochs, training_history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
            if 'val_accuracy' in training_history:
                ax2.plot(epochs, training_history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_title('Model Accuracy Over Time', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Learning rate (if available)
            lr_float = self._extract_learning_rate(model)
            if lr_float is not None:
                lr_values = [lr_float] * len(epochs)
                ax3.plot(epochs, lr_values, 'g-', linewidth=2)
                ax3.set_title(f'Learning Rate: {lr_float:.6f}', fontweight='bold')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Learning Rate')
                ax3.grid(True, alpha=0.3)
                logger.debug(f"running _create_visualization ... Learning rate plotted: {lr_float}")
            else:
                ax3.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                        ha='center', va='center', transform=ax3.transAxes,
                        fontsize=12, fontweight='bold')
                ax3.set_title('Learning Rate (N/A)', fontweight='bold')
            
            # 4. Overfitting analysis
            if 'val_loss' in training_history and 'val_accuracy' in training_history:
                train_val_loss_gap = [train - val for train, val in zip(training_history['loss'], training_history['val_loss'])]
                train_val_acc_gap = [val - train for train, val in zip(training_history['accuracy'], training_history['val_accuracy'])]
                
                ax4.plot(epochs, train_val_loss_gap, 'r-', label='Loss Gap (Train-Val)', linewidth=2)
                ax4_twin = ax4.twinx()
                ax4_twin.plot(epochs, train_val_acc_gap, 'b-', label='Accuracy Gap (Val-Train)', linewidth=2)
                
                ax4.set_title('Overfitting Indicators', fontweight='bold')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Loss Gap', color='r')
                ax4_twin.set_ylabel('Accuracy Gap', color='b')
                ax4.grid(True, alpha=0.3)
                
                # Add overfitting warning if needed
                if len(train_val_loss_gap) > 3 and train_val_loss_gap[-1] > 0.1:
                    ax4.text(0.5, 0.9, '⚠️ Possible Overfitting', 
                            ha='center', va='top', transform=ax4.transAxes, 
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                            fontweight='bold')
            else:
                # Show insights text when validation data not available
                insights_text = '\n'.join(analysis_insights[:3])  # Show first 3 insights
                ax4.text(0.5, 0.5, f'Training Insights:\n\n{insights_text}', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                ax4.set_title('Training Analysis', fontweight='bold')
            
            plt.tight_layout()
            
            # Save the plot
            filepath = self._generate_save_path(dataset_name, run_timestamp, plot_dir)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.debug(f"running _create_visualization ... Training history saved to: {filepath}")
            
            plt.close()  # Clean up memory
            return filepath
            
        except Exception as e:
            logger.warning(f"running _create_visualization ... Failed to create visualization: {e}")
            plt.close()  # Ensure cleanup even on error
            return None
    
    def _extract_learning_rate(self, model) -> Optional[float]:
        """
        Extract current learning rate from the model
        
        Args:
            model: Keras model with optimizer
            
        Returns:
            Current learning rate as float, or None if unavailable
        """
        try:
            if (model is not None and 
                hasattr(model, 'optimizer') and
                model.optimizer is not None and
                hasattr(model.optimizer, 'learning_rate')):
                
                lr_value = model.optimizer.learning_rate
                
                # Handle different learning rate types
                if hasattr(lr_value, 'numpy'):
                    # TensorFlow tensor - convert to float
                    numpy_value = getattr(lr_value, 'numpy')()
                    return float(numpy_value)
                elif callable(lr_value):
                    # Learning rate schedule - get initial value
                    initial_step = 0
                    lr_result = lr_value(initial_step)
                    
                    if hasattr(lr_result, 'numpy'):
                        numpy_method = getattr(lr_result, 'numpy')
                        return float(numpy_method())
                    else:
                        try:
                            return float(lr_result)  # type: ignore
                        except (TypeError, ValueError):
                            return float(str(lr_result))
                else:
                    # Already a scalar
                    try:
                        return float(lr_value)
                    except (TypeError, ValueError):
                        return float(str(lr_value))
        except Exception as e:
            logger.debug(f"running _extract_learning_rate ... Could not extract learning rate: {e}")
            return None
    
    def _generate_save_path(
        self,
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path]
    ) -> Path:
        """
        Generate the file path for saving training history plot
        
        Args:
            dataset_name: Name of dataset
            run_timestamp: Optional timestamp
            plot_dir: Directory to save plot
            
        Returns:
            Path object for saving
        """
        from datetime import datetime
        
        # Create timestamp if not provided
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        
        # Clean dataset name
        dataset_name_clean = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        # Generate filename
        filename = f"training_progress_{run_timestamp}_{dataset_name_clean}.png"
        
        # Determine save directory
        if plot_dir is not None:
            save_dir = plot_dir
        else:
            # Fallback: create default directory
            project_root = Path(__file__).resolve().parent.parent.parent
            save_dir = project_root / "plots"
            save_dir.mkdir(exist_ok=True)
        
        return save_dir / filename


def create_training_history_analysis(
    training_history: Dict[str, List[float]],
    model=None,
    dataset_name: str = "dataset",
    run_timestamp: Optional[str] = None,
    plot_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function for quick training history analysis
    
    Args:
        training_history: Dictionary containing training metrics from Keras model.fit()
        model: Optional Keras model for learning rate extraction
        dataset_name: Name of dataset
        run_timestamp: Optional timestamp
        plot_dir: Optional directory for saving plots
        
    Returns:
        Analysis results dictionary
    """
    analyzer = TrainingHistoryAnalyzer(model_name=dataset_name)
    return analyzer.analyze_and_visualize(
        training_history=training_history,
        model=model,
        dataset_name=dataset_name,
        run_timestamp=run_timestamp,
        plot_dir=plot_dir
    )