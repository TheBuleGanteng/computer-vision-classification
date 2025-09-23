"""
Enhanced logging setup for traffic sign recognition project.
Provides colored console output and detailed file logging.

Usage:
    from utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Training started")
    logger.warning("Low accuracy detected")
    logger.error("Model failed to load")
"""

import logging
import os
from datetime import datetime
from typing import Optional, Union, Dict, Any
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset to default
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to the level name for console output
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.colored_levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message with the parent class
        formatted = super().format(record)
        return formatted


def setup_logging(
    log_file_path: Optional[str] = None,
    console_level: str = "DEBUG",
    file_level: str = "DEBUG",
    max_file_size_mb: int = 10,
    backup_count: int = 5,
    silence_matplotlib: bool = True  # NEW PARAMETER
) -> logging.Logger:
    """
    Set up logging configuration with colored console output and file logging.
    
    Args:
        log_file_path: Path to the log file (if None, uses relative path from project root)
        console_level: Minimum level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_level: Minimum level for file output
        max_file_size_mb: Maximum size of log file before rotation (in MB)
        backup_count: Number of backup files to keep
        silence_matplotlib: Whether to silence verbose matplotlib debug logs (default: True)
    """
    
    # Use relative path if no specific path provided
    if log_file_path is None:
        # Get the project root (assuming logger.py is in src/utils/)
        current_file = Path(__file__)  # src/utils/logger.py
        project_root = current_file.parent.parent.parent  # Go up 3 levels to project root
        log_file_path_resolved = project_root / "logs" / "non-cron.log"
    else:
        log_file_path_resolved = Path(log_file_path)
    
    # Ensure log directory exists
    log_dir = log_file_path_resolved.parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Log directory created/verified: {log_dir}")
    except Exception as e:
        print(f"Failed to create log directory {log_dir}: {e}")
        # Fallback to /tmp
        log_file_path_resolved = Path("/tmp/non-cron.log")
        print(f"Falling back to: {log_file_path_resolved}")
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper()))
    
    console_format = ColoredFormatter(
        fmt='%(asctime)s | %(colored_levelname)-8s | %(name)-20s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler with rotation and error handling
    file_handler = None
    try:
        from logging.handlers import RotatingFileHandler
        
        # Test if we can write to the file first
        log_file_path_resolved.touch()  # Create file if it doesn't exist
        
        file_handler = RotatingFileHandler(
            str(log_file_path_resolved),
            maxBytes=max_file_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, file_level.upper()))
        
        file_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        print(f"File handler created for: {log_file_path_resolved}")
        
    except Exception as e:
        print(f"Failed to create file handler for {log_file_path_resolved}: {e}")
        print("File logging will be disabled, only console logging available")
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    
    # NEW: Configure third-party library logging levels
    if silence_matplotlib:
        # Silence matplotlib's verbose logging while keeping errors/warnings
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.backends').setLevel(logging.WARNING)
        
        # Also silence other commonly verbose libraries
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('fontTools').setLevel(logging.WARNING)

        # Silence boto3/botocore verbose debug logs
        logging.getLogger('boto3').setLevel(logging.WARNING)
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('botocore.httpsession').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        # Optionally silence TensorFlow verbose logs (uncomment if needed)
        # logging.getLogger('tensorflow').setLevel(logging.ERROR)
        # logging.getLogger('tensorflow.python.platform').setLevel(logging.ERROR)
        
        setup_logger = logging.getLogger(__name__)
        setup_logger.info("Third-party library verbose logging silenced: matplotlib, PIL, boto3/botocore (WARNING level and above will still show)")
    
    # Log the setup completion
    setup_logger = logging.getLogger(__name__)
    setup_logger.info(f"Logging initialized - Console: {console_level}, File: {file_level}")
    setup_logger.info(f"Log file: {log_file_path_resolved}")
    
    if silence_matplotlib:
        setup_logger.info("Third-party library verbose logging silenced (matplotlib, PIL, boto3/botocore)")
    
    if file_handler:
        # Force a test write to ensure file logging works
        setup_logger.info("Test file write - if you see this in the file, logging is working!")
        file_handler.flush()  # Force immediate write
    else:
        setup_logger.warning("File logging not available - check permissions and path")
    
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the calling module.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("This is an info message")
    """
    if name is None:
        name = __name__
    
    # If logging hasn't been set up yet, do it now
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)


# NEW: Convenience function to configure third-party logging after initial setup
def configure_third_party_logging(
    matplotlib_level: str = "WARNING",
    tensorflow_level: str = "ERROR",
    pil_level: str = "WARNING"
) -> None:
    """
    Configure logging levels for common third-party libraries.
    Call this after setup_logging() if you want different levels.
    
    Args:
        matplotlib_level: Logging level for matplotlib (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        tensorflow_level: Logging level for TensorFlow
        pil_level: Logging level for PIL/Pillow
    """
    logging.getLogger('matplotlib').setLevel(getattr(logging, matplotlib_level.upper()))
    logging.getLogger('matplotlib.font_manager').setLevel(getattr(logging, matplotlib_level.upper()))
    logging.getLogger('matplotlib.pyplot').setLevel(getattr(logging, matplotlib_level.upper()))
    
    logging.getLogger('tensorflow').setLevel(getattr(logging, tensorflow_level.upper()))
    logging.getLogger('tensorflow.python.platform').setLevel(getattr(logging, tensorflow_level.upper()))
    
    logging.getLogger('PIL').setLevel(getattr(logging, pil_level.upper()))
    logging.getLogger('fontTools').setLevel(getattr(logging, pil_level.upper()))
    
    logger = get_logger(__name__)
    logger.info(f"Third-party logging configured - matplotlib: {matplotlib_level}, "
               f"tensorflow: {tensorflow_level}, PIL: {pil_level}")


# Performance logging utilities
class PerformanceLogger:
    """Utility class for logging performance metrics and training progress."""
    
    def __init__(self, logger_name: str = "performance") -> None:
        self.logger = get_logger(logger_name)
    
    def log_epoch(self, epoch: int, total_epochs: int, loss: float, accuracy: float, 
                  val_loss: Optional[float] = None, val_accuracy: Optional[float] = None, 
                  duration: Optional[float] = None) -> None:
        """Log training epoch results."""
        msg = f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
        
        if val_loss is not None and val_accuracy is not None:
            msg += f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        
        if duration is not None:
            msg += f", Duration: {duration:.2f}s"
        
        self.logger.info(msg)
    
    def log_model_params(self, params: Dict[str, Any]) -> None:
        """Log model parameters and hyperparameters."""
        self.logger.info("Model Configuration:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_data_info(self, total_images: int, train_size: int, test_size: int, 
                      num_categories: int) -> None:
        """Log dataset information."""
        self.logger.info(f"Dataset loaded - Total: {total_images}, Train: {train_size}, "
                        f"Test: {test_size}, Categories: {num_categories}")
    
    def log_optimization_trial(self, trial_num: int, params: Dict[str, Any], accuracy: float) -> None:
        """Log Bayesian optimization trial results."""
        self.logger.info(f"Trial {trial_num}: Accuracy = {accuracy:.4f}")
        for key, value in params.items():
            self.logger.debug(f"  {key}: {value}")


# Context manager for timing operations
class TimedOperation:
    """Context manager for timing and logging operations."""
    
    def __init__(self, operation_name: str, logger_name: Optional[str] = None) -> None:
        self.operation_name = operation_name
        self.logger = get_logger(logger_name or __name__)
        self.start_time: Optional[datetime] = None
    
    def __enter__(self) -> 'TimedOperation':
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        if self.start_time is not None:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            if exc_type is None:
                self.logger.info(f"Completed {self.operation_name} in {duration:.2f}s")
            else:
                self.logger.error(f"Failed {self.operation_name} after {duration:.2f}s: {exc_val}")
        else:
            self.logger.error(f"Timer was not properly initialized for {self.operation_name}")


# Automatic logger setup - initialize logging when module is imported
# This ensures logging is set up once when the module is first imported
if not logging.getLogger().handlers:
    setup_logging()  # Will use silence_matplotlib=True by default

# Create a default logger for this module
logger = get_logger(__name__)


# Example usage and testing
if __name__ == "__main__":
    # Test different log levels
    test_logger = get_logger(__name__)
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    # Test performance logger
    perf_logger = PerformanceLogger()
    perf_logger.log_epoch(1, 10, 0.5234, 0.8567, 0.4123, 0.8901, 45.67)
    
    # Test timed operation
    with TimedOperation("data loading"):
        import time
        time.sleep(1)  # Simulate work
    
    # Test matplotlib logging (should be silenced)
    import matplotlib.pyplot as plt
    import matplotlib.font_manager
    print("Testing matplotlib - verbose logs should be silenced...")
    plt.figure()
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.close()
    
    print("\nLogging setup complete! Check your console output and log file.")
    print("Matplotlib font debug messages should now be silenced.")