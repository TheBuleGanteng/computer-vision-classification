"""
Plot Generator for Multi-Modal Classification

Extracted from ModelBuilder to provide clean separation of concerns.
Handles all plot generation and visualization logic for model training analysis.

This module contains visualization domain knowledge and coordinates between
different plot analysis modules to create comprehensive training reports.
"""

from datetime import datetime
import numpy as np
import os
from pathlib import Path
from tensorflow import keras # type: ignore
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from utils.logger import logger

# Import all the plot analysis modules
from plot_creation.confusion_matrix import ConfusionMatrixAnalyzer
from plot_creation.training_history import TrainingHistoryAnalyzer
from plot_creation.training_animation import TrainingAnimationAnalyzer
from plot_creation.gradient_flow import GradientFlowAnalyzer
from plot_creation.weights_bias import WeightsBiasAnalyzer
from plot_creation.activation_map import ActivationMapAnalyzer

from dataset_manager import DatasetConfig
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from model_builder import ModelConfig
    from optimizer import OptimizationConfig


class PlotGenerator:
    """
    Handles all plot generation and visualization for model training analysis
    
    Coordinates between different analysis modules to create comprehensive
    visualization reports for model training results.
    """
    
    def __init__(self, dataset_config: DatasetConfig, model_config: 'ModelConfig', optimization_config: Optional['OptimizationConfig'] = None):
        """
        Initialize PlotGenerator with dataset and model configurations
        
        Args:
            dataset_config: Configuration object containing dataset metadata
            model_config: Configuration object containing model parameters
            optimization_config: Optional optimization configuration containing plot flags
        """
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.optimization_config = optimization_config
        
        # Detect data type for visualization selection
        self.data_type = self._detect_data_type()
        
        logger.debug(f"running PlotGenerator.__init__ ... Initialized for dataset: {dataset_config.name}")
        logger.debug(f"running PlotGenerator.__init__ ... Data type: {self.data_type}")
        logger.debug(f"running PlotGenerator.__init__ ... Model architecture: {model_config.architecture_type}")
    
    def generate_comprehensive_plots(
        self,
        model: keras.Model,
        training_history: Optional[keras.callbacks.History],
        data: Dict[str, Any],
        test_loss: float,
        test_accuracy: float,
        run_timestamp: str,
        plot_dir: Path,
        log_detailed_predictions: bool = True,
        max_predictions_to_show: int = 20,
        progress_callback: Optional[Callable[[str, int, int, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive plots and analysis for model training results
        
        Args:
            model: Trained Keras model
            training_history: Training history from model.fit()
            data: Dictionary containing training and test data
            test_loss: Final test loss value
            test_accuracy: Final test accuracy value
            run_timestamp: Timestamp string for file naming
            plot_dir: Directory to save plots
            log_detailed_predictions: Whether to log detailed predictions
            max_predictions_to_show: Maximum number of predictions to analyze
            progress_callback: Optional callback function for progress updates
                              Signature: (current_plot_name, completed_plots, total_plots, overall_progress)
            
        Returns:
            Dictionary containing analysis results from all plot generators
        """
        logger.debug(f"running generate_comprehensive_plots ... Starting comprehensive plot generation")
        logger.debug(f"running generate_comprehensive_plots ... Plot directory: {plot_dir}")
        
        # Ensure plot directory exists
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results dictionary
        analysis_results: Dict[str, Optional[Dict[str, Any]]] = {
            'confusion_matrix': None,
            'training_history': None,
            'training_animation': None,
            'gradient_flow': None,
            'weights_bias': None,
            'activation_maps': None,
            'detailed_predictions': None
        }
        
        # Determine which config to use for plot flags (prefer optimization_config)
        config_source = self.optimization_config if self.optimization_config else self.model_config
        
        # Progress tracking setup
        plot_tasks = []
        completed_plots = 0
        
        # Build list of plots to generate
        if getattr(config_source, 'show_confusion_matrix', True):
            plot_tasks.append(('confusion_matrix', 'Confusion Matrix'))
        if training_history and getattr(config_source, 'show_training_history', True):
            plot_tasks.append(('training_history', 'Training History'))
        if training_history and getattr(config_source, 'show_training_animation', True):
            plot_tasks.append(('training_animation', 'Training Animation'))
        
        # Check gradient flow sub-components individually
        show_any_gradient_flow = (getattr(config_source, 'show_gradient_magnitudes', True) or 
                                 getattr(config_source, 'show_gradient_distributions', True) or 
                                 getattr(config_source, 'show_dead_neuron_analysis', True))
        if show_any_gradient_flow:
            plot_tasks.append(('gradient_flow', 'Gradient Flow'))
        
        if getattr(config_source, 'show_weights_bias', True):
            plot_tasks.append(('weights_bias', 'Weights & Bias'))
        if self.data_type == "image" and getattr(config_source, 'show_activation_maps', True):
            plot_tasks.append(('activation_maps', 'Activation Maps'))
        if log_detailed_predictions and max_predictions_to_show > 0 and getattr(config_source, 'show_detailed_predictions', True):
            plot_tasks.append(('detailed_predictions', 'Detailed Predictions'))
        
        total_plots = len(plot_tasks)
        
        # Generate plots with progress tracking
        for plot_key, plot_name in plot_tasks:
            # Report progress before starting current plot
            if progress_callback:
                overall_progress = completed_plots / total_plots if total_plots > 0 else 0.0
                progress_callback(plot_name, completed_plots, total_plots, overall_progress)
            
            # Generate the plot
            if plot_key == 'confusion_matrix':
                analysis_results['confusion_matrix'] = self.generate_confusion_matrix(
                    model, data, run_timestamp, plot_dir
                )
            elif plot_key == 'training_history':
                analysis_results['training_history'] = self.generate_training_history(
                    training_history, model, run_timestamp, plot_dir
                )
            elif plot_key == 'training_animation':
                analysis_results['training_animation'] = self.generate_training_animation(
                    training_history, model, run_timestamp, plot_dir
                )
            elif plot_key == 'gradient_flow':
                analysis_results['gradient_flow'] = self.generate_gradient_flow(
                    model, data, run_timestamp, plot_dir
                )
            elif plot_key == 'weights_bias':
                analysis_results['weights_bias'] = self.generate_weights_bias(
                    model, run_timestamp, plot_dir
                )
            elif plot_key == 'activation_maps':
                analysis_results['activation_maps'] = self.generate_activation_maps(
                    model, data, run_timestamp, plot_dir
                )
            elif plot_key == 'detailed_predictions':
                analysis_results['detailed_predictions'] = self.generate_detailed_predictions(
                    model, data, max_predictions_to_show, run_timestamp, plot_dir
                )
            
            completed_plots += 1
        
        # Report final completion
        if progress_callback:
            progress_callback("Plots Complete", total_plots, total_plots, 1.0)
        
        # Log summary
        self._log_generation_summary(analysis_results, test_loss, test_accuracy)
        
        # Upload plots to S3 if running on RunPod
        if os.getenv('RUNPOD_ENDPOINT_ID'):
            logger.debug(f"Uploading plots to S3 (RunPod environment detected)")
            try:
                from utils.s3_transfer import upload_to_runpod_s3
                
                # Upload the plot directory to S3 using same structure as local
                # Extract the relative path from optimization_results onward
                # Handle both RunPod container paths (/app/...) and local test paths
                plot_dir_str = str(plot_dir)
                if "optimization_results" in plot_dir_str:
                    # Find the optimization_results part and extract relative path from there
                    opt_results_index = plot_dir_str.find("optimization_results")
                    relative_part = plot_dir_str[opt_results_index + len("optimization_results"):].lstrip("/")
                    s3_prefix = f"optimization_results/{relative_part}" if relative_part else "optimization_results"
                else:
                    # Fallback: use the directory name structure
                    s3_prefix = f"optimization_results/{plot_dir.name}"
                s3_result = upload_to_runpod_s3(
                    local_dir=str(plot_dir),
                    s3_prefix=s3_prefix
                )
                
                if s3_result:
                    logger.info(f"✅ Plots uploaded to S3: s3://40ub9vhaa7/{s3_prefix}")
                    # Store S3 info in analysis results
                    analysis_results['plots_s3'] = s3_result
                else:
                    logger.warning(f"⚠️ Failed to upload plots to S3")
                    
            except Exception as e:
                logger.error(f"Failed to upload plots to S3: {e}")
        
        return analysis_results
    
    def generate_confusion_matrix(
        self,
        model: keras.Model,
        data: Dict[str, Any],
        run_timestamp: str,
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Generate confusion matrix analysis
        
        Args:
            model: Trained Keras model
            data: Dictionary containing test data
            run_timestamp: Timestamp for file naming
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing confusion matrix analysis results
        """
        try:
            logger.debug(f"running generate_confusion_matrix ... Generating confusion matrix analysis")
            
            # Get predictions efficiently
            predictions = model.predict(data['x_test'], verbose=0, batch_size=64)
            
            # Convert labels efficiently
            true_labels, predicted_labels = self._convert_labels_for_analysis(data['y_test'], predictions)
            
            # Create analyzer
            class_names = self.dataset_config.class_names or [f"Class_{i}" for i in range(self.dataset_config.num_classes)]
            cm_analyzer = ConfusionMatrixAnalyzer(class_names=class_names)
            
            # Perform analysis
            results = cm_analyzer.analyze_and_visualize(
                true_labels=true_labels,
                predicted_labels=predicted_labels,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir
            )
            
            if 'error' not in results:
                logger.debug(f"running generate_confusion_matrix ... Confusion matrix analysis completed successfully")
                overall_accuracy = results.get('overall_accuracy', 0.0)
                logger.debug(f"running generate_confusion_matrix ... Confusion matrix accuracy: {overall_accuracy:.4f}")
            
            return results
            
        except Exception as e:
            logger.warning(f"running generate_confusion_matrix ... Analysis failed: {e}")
            return {'error': str(e)}
    
    def generate_training_history(
        self,
        training_history: keras.callbacks.History,
        model: keras.Model,
        run_timestamp: str,
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Generate training history analysis
        
        Args:
            training_history: Training history from model.fit()
            model: Trained Keras model
            run_timestamp: Timestamp for file naming
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing training history analysis results
        """
        try:
            logger.debug(f"running generate_training_history ... Generating training history analysis")
            
            history_analyzer = TrainingHistoryAnalyzer(model_name=self.dataset_config.name)
            
            results = history_analyzer.analyze_and_visualize(
                training_history=training_history.history,
                model=model,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir
            )
            
            if 'error' not in results:
                logger.debug(f"running generate_training_history ... Training history analysis completed")
                
                # Log key insights
                insights = results.get('training_insights', [])
                for insight in insights[:3]:
                    logger.debug(f"running generate_training_history ... Insight: {insight}")
            
            return results
            
        except Exception as e:
            logger.warning(f"running generate_training_history ... Analysis failed: {e}")
            return {'error': str(e)}
    
    def generate_training_animation(
        self,
        training_history: keras.callbacks.History,
        model: keras.Model,
        run_timestamp: str,
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Generate training animation
        
        Args:
            training_history: Training history from model.fit()
            model: Trained Keras model
            run_timestamp: Timestamp for file naming
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing animation generation results
        """
        try:
            logger.debug(f"running generate_training_animation ... Generating training animation")
            
            animation_analyzer = TrainingAnimationAnalyzer(model_name=self.dataset_config.name)
            
            results = animation_analyzer.analyze_and_animate(
                training_history=training_history.history,
                model=model,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir,
                animation_duration=8.0,  # Optimized duration
                fps=12  # Higher FPS for smoother animation
            )
            
            if 'error' not in results:
                frame_count = results.get('frame_count', 0)
                logger.debug(f"running generate_training_animation ... Animation created with {frame_count} frames")
            
            return results
            
        except Exception as e:
            logger.warning(f"running generate_training_animation ... Animation creation failed: {e}")
            return {'error': str(e)}
    
    def generate_gradient_flow(
        self,
        model: keras.Model,
        data: Dict[str, Any],
        run_timestamp: str,
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Generate gradient flow analysis with individual sub-component control
        
        Args:
            model: Trained Keras model
            data: Dictionary containing test data
            run_timestamp: Timestamp for file naming
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing gradient flow analysis results
        """
        try:
            logger.debug(f"running generate_gradient_flow ... Generating gradient flow analysis")
            
            # Get individual gradient flow flags
            config_source = self.optimization_config if self.optimization_config else self.model_config
            show_gradient_magnitudes = getattr(config_source, 'show_gradient_magnitudes', True)
            show_gradient_distributions = getattr(config_source, 'show_gradient_distributions', True) 
            show_dead_neuron_analysis = getattr(config_source, 'show_dead_neuron_analysis', True)
            
            # Skip if all sub-components are disabled
            if not (show_gradient_magnitudes or show_gradient_distributions or show_dead_neuron_analysis):
                logger.debug(f"running generate_gradient_flow ... All gradient flow sub-components disabled, skipping")
                return {'skipped': True, 'reason': 'All gradient flow sub-components disabled'}
            
            # Intelligent sample selection for gradient analysis
            default_gradient_sample_size = 100  # Default for gradient analysis
            sample_size = min(default_gradient_sample_size, len(data['x_test']))
            sample_indices = self._select_representative_samples(data['x_test'], data['y_test'], sample_size)
            
            sample_x = data['x_test'][sample_indices]
            sample_y = data['y_test'][sample_indices]
            
            gradient_analyzer = GradientFlowAnalyzer(model_name=self.dataset_config.name)
            
            # Pass configuration to analyzer for selective plotting
            results = gradient_analyzer.analyze_and_visualize(
                model=model,
                sample_data=sample_x,
                sample_labels=sample_y,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir,
                config=config_source  # Pass the configuration object
            )
            
            if 'error' not in results:
                gradient_health = results.get('gradient_health', 'unknown')
                logger.debug(f"running generate_gradient_flow ... Gradient health: {gradient_health}")
            
            return results
            
        except Exception as e:
            logger.warning(f"running generate_gradient_flow ... Analysis failed: {e}")
            return {'error': str(e)}
    
    def generate_weights_bias(
        self,
        model: keras.Model,
        run_timestamp: str,
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Generate weights and bias analysis
        
        Args:
            model: Trained Keras model
            run_timestamp: Timestamp for file naming
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing weights and bias analysis results
        """
        try:
            logger.debug(f"running generate_weights_bias ... Generating weights and bias analysis")
            
            weights_bias_analyzer = WeightsBiasAnalyzer(model_name=self.dataset_config.name)
            
            results = weights_bias_analyzer.analyze_and_visualize(
                model=model,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir,
                max_layers_to_plot=8  # Optimized for performance
            )
            
            if 'error' not in results:
                parameter_health = results.get('parameter_health', 'unknown')
                logger.debug(f"running generate_weights_bias ... Parameter health: {parameter_health}")
            
            return results
            
        except Exception as e:
            logger.warning(f"running generate_weights_bias ... Analysis failed: {e}")
            return {'error': str(e)}
    
    def generate_activation_maps(
        self,
        model: keras.Model,
        data: Dict[str, Any],
        run_timestamp: str,
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Generate activation maps analysis for CNN models
        
        Args:
            model: Trained Keras model
            data: Dictionary containing test data
            run_timestamp: Timestamp for file naming
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing activation maps analysis results
        """
        try:
            logger.debug(f"running generate_activation_maps ... Generating activation maps analysis")
            
            # Intelligent sample selection for activation analysis
            max_activation_samples = 20  # Default for activation analysis
            sample_size = min(max_activation_samples, len(data['x_test']))
            sample_indices = self._select_diverse_samples_for_activation(data['x_test'], data['y_test'], sample_size)
            
            sample_x = data['x_test'][sample_indices]
            sample_y = data['y_test'][sample_indices]
            
            # Convert labels
            if sample_y.ndim > 1 and sample_y.shape[1] > 1:
                sample_labels = np.argmax(sample_y, axis=1)
            else:
                sample_labels = sample_y.flatten()
            
            class_names = self.dataset_config.class_names or [f"Class_{i}" for i in range(self.dataset_config.num_classes)]
            
            activation_analyzer = ActivationMapAnalyzer(model_name=self.dataset_config.name)
            
            results = activation_analyzer.analyze_and_visualize(
                model=model,
                sample_images=sample_x,
                sample_labels=sample_labels,
                class_names=class_names,
                dataset_name=self.dataset_config.name,
                run_timestamp=run_timestamp,
                plot_dir=plot_dir,
                model_config=self.model_config
            )
            
            if 'error' not in results:
                filter_health = results.get('filter_health', {})
                health_status = filter_health.get('overall_status', 'unknown')
                logger.debug(f"running generate_activation_maps ... Filter health: {health_status}")
            
            return results
            
        except Exception as e:
            logger.warning(f"running generate_activation_maps ... Analysis failed: {e}")
            return {'error': str(e)}
    
    def generate_detailed_predictions(
        self,
        model: keras.Model,
        data: Dict[str, Any],
        max_predictions_to_show: int,
        run_timestamp: str,
        plot_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Generate detailed predictions analysis
        
        Args:
            model: Trained Keras model
            data: Dictionary containing test data
            max_predictions_to_show: Maximum number of predictions to analyze
            run_timestamp: Timestamp for file naming
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing detailed predictions analysis results
        """
        try:
            logger.debug(f"running generate_detailed_predictions ... Generating detailed prediction analysis")
            
            # Efficient prediction generation
            predictions = model.predict(data['x_test'], verbose=0, batch_size=128)
            
            # Convert labels efficiently
            true_labels, predicted_labels = self._convert_labels_for_analysis(data['y_test'], predictions)
            confidence_scores = np.max(predictions, axis=1)
            
            # Get class names
            class_names = self.dataset_config.class_names or [f"Class_{i}" for i in range(self.dataset_config.num_classes)]
            
            # Efficient correct/incorrect separation
            correct_mask = true_labels == predicted_labels
            correct_indices = np.where(correct_mask)[0]
            incorrect_indices = np.where(~correct_mask)[0]
            
            # Optimized logging of samples
            max_correct = min(3, len(correct_indices), max_predictions_to_show // 3)
            max_incorrect = min(max_predictions_to_show - max_correct, len(incorrect_indices))
            
            results = {
                'total_predictions': len(true_labels),
                'correct_predictions': len(correct_indices),
                'incorrect_predictions': len(incorrect_indices),
                'accuracy': len(correct_indices) / len(true_labels),
                'average_confidence': float(np.mean(confidence_scores)),
                'sample_correct': [],
                'sample_incorrect': []
            }
            
            if max_correct > 0:
                logger.debug(f"running generate_detailed_predictions ... Sample correct predictions:")
                sample_correct = np.random.choice(correct_indices, max_correct, replace=False)
                
                for idx in sample_correct:
                    confidence = confidence_scores[idx]
                    true_class = class_names[true_labels[idx]]
                    results['sample_correct'].append({
                        'index': int(idx),
                        'true_class': true_class,
                        'predicted_class': true_class,
                        'confidence': float(confidence)
                    })
                    logger.debug(f"running generate_detailed_predictions ... ✅ Correct: {true_class}, confidence: {confidence:.3f}")
            
            if max_incorrect > 0:
                logger.debug(f"running generate_detailed_predictions ... Sample incorrect predictions:")
                sample_incorrect = np.random.choice(incorrect_indices, max_incorrect, replace=False)
                
                for idx in sample_incorrect:
                    confidence = confidence_scores[idx]
                    true_class = class_names[true_labels[idx]]
                    pred_class = class_names[predicted_labels[idx]]
                    results['sample_incorrect'].append({
                        'index': int(idx),
                        'true_class': true_class,
                        'predicted_class': pred_class,
                        'confidence': float(confidence)
                    })
                    logger.debug(f"running generate_detailed_predictions ... ❌ Incorrect: predicted {pred_class}, actual {true_class}, confidence: {confidence:.3f}")
            
            # Performance summary
            logger.debug(f"running generate_detailed_predictions ... Prediction summary:")
            logger.debug(f"running generate_detailed_predictions ... - Total: {results['total_predictions']}, Correct: {results['correct_predictions']} ({results['accuracy']:.1%})")
            logger.debug(f"running generate_detailed_predictions ... - Average confidence: {results['average_confidence']:.3f}")
            
            return results
            
        except Exception as e:
            logger.warning(f"running generate_detailed_predictions ... Analysis failed: {e}")
            return {'error': str(e)}
    
    def _detect_data_type(self) -> str:
        """
        Detect whether this is image or text data based on dataset configuration
        
        Returns:
            String indicating data type: "image" or "text"
        """
        # Text indicators: flat sequence structure
        if (self.dataset_config.img_height == 1 and 
            self.dataset_config.channels == 1 and 
            self.dataset_config.img_width > 100):
            return "text"
        
        # Image indicators: spatial structure
        if (self.dataset_config.img_height > 1 and 
            self.dataset_config.img_width > 1):
            return "image"
        
        # Fallback to image for ambiguous cases
        return "image"
    
    def _convert_labels_for_analysis(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert one-hot encoded labels to class indices efficiently"""
        
        # Convert true labels
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            true_labels = np.argmax(y_true, axis=1)
        else:
            true_labels = y_true.flatten()
        
        # Convert predicted labels
        predicted_labels = np.argmax(y_pred, axis=1)
        
        return true_labels, predicted_labels
    
    def _select_representative_samples(
        self, 
        x_data: np.ndarray, 
        y_data: np.ndarray, 
        sample_size: int
    ) -> np.ndarray:
        """Select representative samples for analysis using stratified sampling"""
        
        if sample_size >= len(x_data):
            return np.arange(len(x_data))
        
        # Convert labels if needed
        if y_data.ndim > 1 and y_data.shape[1] > 1:
            labels = np.argmax(y_data, axis=1)
        else:
            labels = y_data.flatten()
        
        # Stratified sampling
        selected_indices = []
        unique_classes = np.unique(labels)
        samples_per_class = max(1, sample_size // len(unique_classes))
        
        for class_id in unique_classes:
            class_indices = np.where(labels == class_id)[0]
            if len(class_indices) > 0:
                n_samples = min(samples_per_class, len(class_indices))
                sampled = np.random.choice(class_indices, n_samples, replace=False)
                selected_indices.extend(sampled)
        
        # Fill remaining slots if needed
        if len(selected_indices) < sample_size:
            remaining_needed = sample_size - len(selected_indices)
            all_indices = set(range(len(x_data)))
            unused_indices = list(all_indices - set(selected_indices))
            
            if unused_indices:
                additional = np.random.choice(
                    unused_indices, 
                    min(remaining_needed, len(unused_indices)), 
                    replace=False
                )
                selected_indices.extend(additional)
        
        return np.array(selected_indices[:sample_size])
    
    def _select_diverse_samples_for_activation(
        self, 
        x_data: np.ndarray, 
        y_data: np.ndarray, 
        sample_size: int
    ) -> np.ndarray:
        """Select diverse samples for activation analysis"""
        
        # The default "representative" strategy uses stratified sampling which gives better activation analysis results than pure random sampling
        strategy = "representative"
        
        if strategy == "random":
            return np.random.choice(len(x_data), sample_size, replace=False)
        
        elif strategy == "representative":
            return self._select_representative_samples(x_data, y_data, sample_size)
        
        elif strategy == "mixed":
            # Mix of representative and random samples
            half_size = sample_size // 2
            representative = self._select_representative_samples(x_data, y_data, half_size)
            
            # Select random samples not in representative set
            remaining_indices = list(set(range(len(x_data))) - set(representative))
            random_needed = sample_size - len(representative)
            
            if remaining_indices and random_needed > 0:
                random_samples = np.random.choice(
                    remaining_indices, 
                    min(random_needed, len(remaining_indices)), 
                    replace=False
                )
                return np.concatenate([representative, random_samples])
            
            return representative
        
        else:
            # Default to representative
            return self._select_representative_samples(x_data, y_data, sample_size)
    
    def _log_generation_summary(
        self, 
        analysis_results: Dict[str, Any], 
        test_loss: float, 
        test_accuracy: float
    ) -> None:
        """Log comprehensive plot generation summary"""
        
        logger.debug(f"running _log_generation_summary ... Plot Generation Summary:")
        logger.debug(f"running _log_generation_summary ... - Test accuracy: {test_accuracy:.4f}")
        logger.debug(f"running _log_generation_summary ... - Test loss: {test_loss:.4f}")
        
        # Log analysis completion status
        completed_analyses = []
        failed_analyses = []
        
        for analysis_name, result in analysis_results.items():
            if result is None:
                continue
            elif isinstance(result, dict) and 'error' in result:
                failed_analyses.append(analysis_name)
            else:
                completed_analyses.append(analysis_name)
        
        if completed_analyses:
            logger.debug(f"running _log_generation_summary ... - Completed analyses: {', '.join(completed_analyses)}")
        
        if failed_analyses:
            logger.warning(f"running _log_generation_summary ... - Failed analyses: {', '.join(failed_analyses)}")


# Convenience function for standalone testing
def test_plot_generator(dataset_name: str = "cifar10") -> None:
    """
    Test function for PlotGenerator
    
    Args:
        dataset_name: Name of dataset to test with
    """
    from dataset_manager import DatasetManager
    
    logger.debug(f"running test_plot_generator ... Testing with dataset: {dataset_name}")
    
    # Load dataset config
    dataset_manager = DatasetManager()
    dataset_config = dataset_manager.get_dataset_config(dataset_name)
    
    # Create model config (delayed import to avoid circular dependency)
    from model_builder import ModelConfig
    model_config = ModelConfig()
    
    # Create plot generator
    plot_generator = PlotGenerator(dataset_config, model_config)
    
    logger.debug(f"running test_plot_generator ... Plot generator created for {dataset_name}")
    logger.debug(f"running test_plot_generator ... Data type: {plot_generator.data_type}")
    logger.debug(f"running test_plot_generator ... ✅ Test completed successfully")


if __name__ == "__main__":
    # Simple test when run directly
    import sys
    
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    test_plot_generator(dataset_name)