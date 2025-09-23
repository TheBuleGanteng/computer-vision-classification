"""
Run Name Generation Utility

Centralized utility for generating consistent run names across the entire codebase.
This ensures that local execution, RunPod execution, and API server all use
identical naming conventions.
"""

from datetime import datetime
import pytz


def create_run_name(dataset_name: str, mode: str, optimize_for: str) -> str:
    """
    Create run_name using Jakarta timezone - single source of truth for the entire codebase.

    This function ensures that local execution, RunPod execution, and API server
    all generate identical run names, preventing path mismatches in S3 transfers.

    Args:
        dataset_name: Name of the dataset (e.g., 'mnist', 'cifar10')
        mode: Optimization mode ('simple' or 'health')
        optimize_for: Optimization objective (e.g., 'val_accuracy')

    Returns:
        Unified run name string (e.g., '2025-09-17-15-52-57_mnist_health')
    """
    # Use Jakarta timezone for consistent timestamps across local and RunPod execution
    jakarta_tz = pytz.timezone('Asia/Jakarta')
    jakarta_time = datetime.now(jakarta_tz)
    timestamp = jakarta_time.strftime("%Y-%m-%d-%H-%M-%S")

    # Clean dataset name for file system compatibility
    dataset_clean = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").lower()

    # Generate run name based on mode
    if mode == 'health':
        run_name = f"{timestamp}_{dataset_clean}_health"
    elif mode == 'simple':
        run_name = f"{timestamp}_{dataset_clean}_simple-{optimize_for}"
    else:
        run_name = f"{timestamp}_{dataset_clean}_{mode}"

    return run_name