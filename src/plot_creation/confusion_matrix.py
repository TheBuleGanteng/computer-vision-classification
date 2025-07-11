"""
Confusion Matrix Analysis and Visualization Module

This module provides comprehensive confusion matrix analysis including:
- Statistical analysis of classification performance
- Visual heatmap generation
- Per-class performance metrics (precision, recall)
- Top misclassifications identification
- Professional-quality plots for presentations and papers

Designed to work with any classification model and dataset configuration.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
import traceback
from typing import Dict, List, Optional, Tuple, Any
from utils.logger import logger


class ConfusionMatrixAnalyzer:
    """
    Comprehensive confusion matrix analysis and visualization
    
    Creates statistical analysis and professional visualizations of model
    classification performance. Identifies common misclassifications and
    per-class performance metrics.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the confusion matrix analyzer
        
        Args:
            class_names: Optional list of human-readable class names for labels
        """
        self.class_names = class_names or []
        logger.debug("running ConfusionMatrixAnalyzer.__init__ ... Confusion matrix analyzer initialized")
    
    def analyze_and_visualize(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        dataset_name: str,
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None,
        max_visual_classes: int = 20
    ) -> Dict[str, Any]:
        """
        Create, analyze, and visualize confusion matrix with detailed statistics
        
        A confusion matrix is a comprehensive performance evaluation tool that shows:
        - How often each class was correctly predicted (diagonal elements)
        - Which classes are most commonly confused with each other (off-diagonal elements)
        - Per-class performance metrics (precision and recall)
        
        Think of it as a "mistake analysis report" that helps you understand:
        1. Which traffic signs your model recognizes well
        2. Which signs it confuses (e.g., does it mix up speed limit signs?)
        3. Whether certain classes are harder to detect than others
        
        Matrix Structure Example (3-class problem):
                        PREDICTED
                    Stop  Yield  Speed
        ACTUAL  Stop  [95    2     3  ]  ← 95 stops correctly identified, 2 confused as yield, 3 as speed
            Yield  [ 5   88     7  ]  ← 5 yields confused as stop, 88 correct, 7 as speed  
            Speed  [ 1    4    89  ]  ← 1 speed sign confused as stop, 4 as yield, 89 correct
        
        Key Metrics Calculated:
        - Precision: Of all times model said "stop sign", how often was it actually a stop sign?
        Formula: True Positives / (True Positives + False Positives)
        Example: Stop precision = 95 / (95 + 5 + 1) = 94.1%
        
        - Recall: Of all actual stop signs, how many did the model correctly identify?
        Formula: True Positives / (True Positives + False Negatives)  
        Example: Stop recall = 95 / (95 + 2 + 3) = 95.0%
        
        What Good vs Bad Results Look Like:
        ✅ Good: High numbers on diagonal, low numbers off diagonal
        ❌ Bad: Scattered values, indicating frequent misclassifications
        
        Common Patterns to Watch For:
        - Systematic confusion: Speed limit signs consistently confused with each other
        - Class imbalance effects: Rare signs performing poorly due to limited training data
        - Similar-looking signs: Yield vs warning signs having cross-confusion
        
        The method automatically identifies:
        1. Top 5 most common misclassifications (helps focus improvement efforts)
        2. Best and worst performing classes (guides data collection priorities)
        3. Overall accuracy from matrix diagonal (validation of model performance)
        
        Args:
            true_labels: True class labels (1D array of integers representing actual traffic sign classes)
                        Example: [0, 1, 2, 0, 1] where 0=stop, 1=yield, 2=speed_limit
            predicted_labels: Predicted class labels (1D array of integers from model output)
                            Example: [0, 1, 1, 0, 1] showing one misclassification (2→1)
            dataset_name: Name of dataset for file naming and logging
            run_timestamp: Optional timestamp for when this analysis was run (default is current time)
            plot_dir: Optional directory to save visualization plots
            max_visual_classes: Maximum number of classes to show in visual matrix (default 20)
                               For datasets with >20 classes, visual matrix is skipped to maintain readability
            
        Returns:
            Dictionary containing comprehensive analysis results:
            - 'confusion_matrix': The raw confusion matrix as numpy array
            - 'overall_accuracy': Overall classification accuracy as float
            - 'total_correct': Number of correct predictions (int)
            - 'total_predictions': Total number of predictions (int)
            - 'top_misclassifications': List of most common mistakes as tuples (true_class, pred_class, count)
            - 'class_metrics': List of per-class precision/recall metrics
            - 'visualization_path': Path to saved plot file (None if not created)
            
        Side Effects:
            - Logs detailed confusion matrix analysis to console
            - Creates and saves visualization plot (if ≤20 classes)
            - Shows top misclassifications and per-class performance statistics
            
        Example Usage Context:
            After model evaluation, this method helps answer questions like:
            - "Why did my traffic sign classifier get 85% accuracy but fail in real testing?"
            - "Which speed limit signs are most commonly confused?"
            - "Should I collect more data for warning signs vs stop signs?"
        
        Troubleshooting Interpretation:
        - Low precision: Model is "trigger-happy" - says this class too often
        - Low recall: Model is "conservative" - misses many instances of this class
        - Both low: Class is genuinely difficult or needs more training data
        """
        logger.debug("running analyze_and_visualize ... Starting comprehensive confusion matrix analysis")
        
        try:
            # Generate confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels)
            
            # Perform statistical analysis
            analysis_results = self._analyze_confusion_matrix(cm, true_labels, predicted_labels)
            
            # Create visualization if not too many classes
            num_classes = len(self.class_names) if self.class_names else cm.shape[0]
            if num_classes <= max_visual_classes:
                logger.debug(f"running analyze_and_visualize ... Creating visual confusion matrix ({num_classes} classes)")
                visualization_path = self._create_visualization(
                    cm=cm,
                    dataset_name=dataset_name,
                    run_timestamp=run_timestamp,
                    plot_dir=plot_dir
                )
                analysis_results['visualization_path'] = visualization_path
            else:
                logger.debug(f"running analyze_and_visualize ... Skipping visualization ({num_classes} classes > {max_visual_classes})")
                analysis_results['visualization_path'] = None
            
            logger.debug("running analyze_and_visualize ... Confusion matrix analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.warning(f"running analyze_and_visualize ... Failed to complete confusion matrix analysis: {e}")
            logger.debug(f"running analyze_and_visualize ... Error traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def _analyze_confusion_matrix(
        self,
        cm: np.ndarray,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform detailed statistical analysis of confusion matrix
        
        Args:
            cm: Confusion matrix
            true_labels: True class labels
            predicted_labels: Predicted class labels
            
        Returns:
            Dictionary containing analysis results
        """
        logger.debug("running _analyze_confusion_matrix ... Performing statistical analysis")
        
        # Basic statistics
        total_correct = np.trace(cm)  # Sum of diagonal elements
        total_predictions = np.sum(cm)
        overall_accuracy = total_correct / total_predictions
        
        logger.debug(f"running _analyze_confusion_matrix ... Matrix shape: {cm.shape}")
        logger.debug(f"running _analyze_confusion_matrix ... Total correct: {total_correct}")
        logger.debug(f"running _analyze_confusion_matrix ... Total predictions: {total_predictions}")
        logger.debug(f"running _analyze_confusion_matrix ... Overall accuracy: {overall_accuracy:.4f}")
        
        # Find top misclassifications
        top_misclassifications = self._find_top_misclassifications(cm)
        
        # Calculate per-class metrics
        class_metrics = self._calculate_class_metrics(cm)
        
        # Log key findings
        self._log_analysis_results(top_misclassifications, class_metrics)
        
        return {
            'confusion_matrix': cm,
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_predictions': total_predictions,
            'top_misclassifications': top_misclassifications,
            'class_metrics': class_metrics
        }
    
    def _find_top_misclassifications(self, cm: np.ndarray, top_k: int = 5) -> List[Tuple[str, str, int]]:
        """
        Identify the most common misclassifications
        
        Args:
            cm: Confusion matrix
            top_k: Number of top misclassifications to return
            
        Returns:
            List of tuples (true_class, predicted_class, count)
        """
        logger.debug("running _find_top_misclassifications ... Finding most common misclassifications")
        
        # Create copy with diagonal set to 0 to find off-diagonal maxima
        cm_off_diagonal = cm.copy()
        np.fill_diagonal(cm_off_diagonal, 0)
        
        # Generate class names if not provided
        if not self.class_names:
            class_names = [f"Class_{i}" for i in range(cm.shape[0])]
        else:
            class_names = self.class_names
        
        top_misclassifications = []
        for i in range(min(top_k, cm.shape[0] * cm.shape[1])):
            max_idx = np.unravel_index(np.argmax(cm_off_diagonal), cm_off_diagonal.shape)
            true_idx, pred_idx = max_idx
            count = cm_off_diagonal[true_idx, pred_idx]
            
            if count > 0:  # Only include actual misclassifications
                true_class = class_names[true_idx] if true_idx < len(class_names) else f"Class_{true_idx}"
                pred_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
                top_misclassifications.append((true_class, pred_class, count))
                cm_off_diagonal[true_idx, pred_idx] = 0  # Remove for next iteration
            else:
                break
        
        return top_misclassifications
    
    def _calculate_class_metrics(self, cm: np.ndarray) -> List[Dict[str, Any]]:
        """
        Calculate precision and recall for each class
        
        Args:
            cm: Confusion matrix
            
        Returns:
            List of dictionaries containing class metrics
        """
        logger.debug("running _calculate_class_metrics ... Calculating per-class performance metrics")
        
        # Generate class names if not provided
        if not self.class_names:
            class_names = [f"Class_{i}" for i in range(cm.shape[0])]
        else:
            class_names = self.class_names
        
        class_metrics = []
        for i in range(cm.shape[0]):
            true_positives = cm[i, i]
            total_actual = np.sum(cm[i, :])  # Total actual instances of this class
            total_predicted = np.sum(cm[:, i])  # Total predicted instances of this class
            
            # Calculate precision and recall
            precision = true_positives / total_predicted if total_predicted > 0 else 0.0
            recall = true_positives / total_actual if total_actual > 0 else 0.0
            
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            
            class_metrics.append({
                'class_name': class_name,
                'class_index': i,
                'precision': precision,
                'recall': recall,
                'true_positives': true_positives,
                'total_actual': total_actual,
                'total_predicted': total_predicted
            })
        
        return class_metrics
    
    def _log_analysis_results(
        self,
        top_misclassifications: List[Tuple[str, str, int]],
        class_metrics: List[Dict[str, Any]]
    ) -> None:
        """
        Log the analysis results to console
        
        Args:
            top_misclassifications: List of top misclassifications
            class_metrics: List of per-class metrics
        """
        # Log top misclassifications
        if top_misclassifications:
            logger.debug("running _log_analysis_results ... Most common misclassifications:")
            for i, (true_class, pred_class, count) in enumerate(top_misclassifications, 1):
                logger.debug(f"running _log_analysis_results ... {i}. {true_class} → {pred_class}: {count} times")
        
        # Sort by recall and show top/bottom performers
        class_metrics_sorted = sorted(class_metrics, key=lambda x: x['recall'], reverse=True)
        
        logger.debug("running _log_analysis_results ... Top 10 performing classes (by recall):")
        for i, metrics in enumerate(class_metrics_sorted[:10], 1):
            logger.debug(f"running _log_analysis_results ... {i:2d}. {metrics['class_name']:20s}: "
                        f"recall={metrics['recall']:.3f}, precision={metrics['precision']:.3f} "
                        f"({metrics['total_actual']} samples)")
        
        logger.debug("running _log_analysis_results ... Bottom 10 performing classes (by recall):")
        for i, metrics in enumerate(class_metrics_sorted[-10:], 1):
            logger.debug(f"running _log_analysis_results ... {i:2d}. {metrics['class_name']:20s}: "
                        f"recall={metrics['recall']:.3f}, precision={metrics['precision']:.3f} "
                        f"({metrics['total_actual']} samples)")
    
    def _create_visualization(
        self,
        cm: np.ndarray,
        dataset_name: str,
        run_timestamp: Optional[str] = None,
        plot_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Visualize confusion matrix as heatmap and save to file for detailed analysis
        
        Creates a professional-quality visual confusion matrix that makes patterns 
        immediately apparent through color coding and annotations. This is the visual
        companion to the statistical analysis.
        
        Visual Design Elements:
        - Heatmap colors: Darker blue = higher values (more predictions)
        - Diagonal: Shows correct predictions (ideally the darkest cells)
        - Off-diagonal: Shows mistakes (ideally lighter/white)
        - Numbers in cells: Exact count of predictions for that true/predicted combination
        - Axis labels: Class names for easy interpretation
        
        How to Read the Visualization:
        
        Perfect Model Example:
        ```
                Predicted
                Stop Yield Speed
        Actual Stop [100   0    0  ]  ← Dark blue diagonal
            Yield [  0  95    0  ]  ← Light/white off-diagonal  
            Speed [  0   0   88  ]  ← Perfect classification
        ```
        
        Problematic Model Example:
        ```
                Predicted  
                Stop Yield Speed
        Actual Stop [ 60  25   15 ]  ← Many stop signs misclassified
            Yield [ 30  50   15 ]  ← Yield signs confused with stop
            Speed [ 10  20   45 ]  ← Speed signs performing poorly
        ```
        
        Key Visual Patterns to Identify:
        
        1. **Strong Diagonal**: Dark blue line from top-left to bottom-right
        → Indicates good overall performance
        
        2. **Scattered Heat**: Colors spread throughout matrix
        → Indicates poor performance, model is guessing randomly
        
        3. **Cluster Patterns**: Groups of confusion between similar classes
        → Example: All speed limit signs confused with each other
        → Solution: More diverse training data or better feature extraction
        
        4. **Row/Column Dominance**: One row very light, one column very dark
        → Row dominance: Model never predicts this class (conservative)
        → Column dominance: Model over-predicts this class (trigger-happy)
        
        File Output Details:
        - Saves high-resolution PNG for presentations/papers
        - Filename format: "confusion_matrix_YYYYMMDD_HHMMSS_dataset_name.png"
        - For >10 classes: Also saves ultra-high-res version for detailed examination
        
        Professional Usage:
        - Include in research papers to show model performance
        - Use in presentations to explain model behavior to stakeholders
        - Compare matrices before/after model improvements
        - Share with domain experts to validate misclassifications make sense
        
        Troubleshooting Visual Patterns:
        - **Checkered pattern**: Class imbalance - some classes have much more data
        - **Vertical/horizontal stripes**: Systematic bias toward certain predictions
        - **Block patterns**: Model learned to distinguish groups but not individuals
        
        Args:
            cm: Confusion matrix as 2D numpy array where cm[i,j] represents 
                the number of samples of true class i predicted as class j
                Shape: (n_classes, n_classes)
            dataset_name: Name of dataset for file naming
            run_timestamp: Optional timestamp for file naming
            plot_dir: Optional directory to save plot
            
        Returns:
            Path to saved plot file, or None if save failed
            
        Side Effects:
            - Creates matplotlib figure with confusion matrix heatmap
            - Saves PNG file(s) with timestamp
            - Closes matplotlib figure to free memory
            - Logs save location and any errors to console
            
        Technical Details:
            - Uses seaborn heatmap for professional appearance
            - Colormap: 'Blues' (white=0, dark blue=maximum)
            - Annotations: Integer counts in each cell
            - Figure size: 10x8 inches for readability
            - DPI: 300 for publication quality, 600 for high-res version
        """
        logger.debug("running _create_visualization ... Creating confusion matrix heatmap")
        
        try:
            # Generate class names for axes
            if not self.class_names:
                display_names = [f"Class_{i}" for i in range(cm.shape[0])]
            else:
                display_names = self.class_names
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=display_names, yticklabels=display_names)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            
            # Determine save path
            filepath = self._generate_save_path(dataset_name, run_timestamp, plot_dir)
            
            # Save the plot
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.debug(f"running _create_visualization ... Confusion matrix saved to: {filepath}")
            
            # Save high-res version for detailed viewing if many classes
            if len(display_names) > 10:
                high_res_filepath = self._generate_save_path(
                    dataset_name, run_timestamp, plot_dir, suffix="_highres"
                )
                plt.savefig(high_res_filepath, dpi=600, bbox_inches='tight')
                logger.debug(f"running _create_visualization ... High-res version saved to: {high_res_filepath}")
            
            plt.close()  # Clean up memory
            return filepath
            
        except Exception as e:
            logger.warning(f"running _create_visualization ... Failed to create visualization: {e}")
            plt.close()  # Ensure cleanup even on error
            return None
    
    def _generate_save_path(
        self,
        dataset_name: str,
        run_timestamp: Optional[str],
        plot_dir: Optional[Path],
        suffix: str = ""
    ) -> Path:
        """
        Generate the file path for saving confusion matrix plot
        
        Args:
            dataset_name: Name of dataset
            run_timestamp: Optional timestamp
            plot_dir: Directory to save plot
            suffix: Optional suffix for filename
            
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
        filename = f"confusion_matrix_{run_timestamp}_{dataset_name_clean}{suffix}.png"
        
        # Determine save directory
        if plot_dir is not None:
            save_dir = plot_dir
        else:
            # Fallback: create default directory
            project_root = Path(__file__).resolve().parent.parent.parent
            save_dir = project_root / "plots"
            save_dir.mkdir(exist_ok=True)
        
        return save_dir / filename


def create_confusion_matrix_analysis(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    dataset_name: str = "dataset",
    run_timestamp: Optional[str] = None,
    plot_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function for quick confusion matrix analysis
    
    Args:
        true_labels: True class labels
        predicted_labels: Predicted class labels
        class_names: Optional list of class names
        dataset_name: Name of dataset
        run_timestamp: Optional timestamp
        plot_dir: Optional directory for saving plots
        
    Returns:
        Analysis results dictionary
    """
    analyzer = ConfusionMatrixAnalyzer(class_names)
    return analyzer.analyze_and_visualize(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        dataset_name=dataset_name,
        run_timestamp=run_timestamp,
        plot_dir=plot_dir
    )