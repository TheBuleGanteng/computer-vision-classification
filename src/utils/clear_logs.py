"""
Utility for clearing logs directory on server startup.
"""

import shutil
from pathlib import Path


def clear_logs_directory(log_dir_path: str | None = None) -> bool:
    """
    Clear all contents of the logs directory.

    Args:
        log_dir_path: Path to the logs directory. If None, uses default logs/ directory.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if log_dir_path is None:
            # Get the project root (assuming this file is in src/utils/)
            current_file = Path(__file__)  # src/utils/clear_logs.py
            project_root = current_file.parent.parent.parent  # Go up 3 levels to project root
            log_dir = project_root / "logs"
        else:
            log_dir = Path(log_dir_path)

        if not log_dir.exists():
            print(f"Logs directory does not exist: {log_dir}")
            return True  # Nothing to clear

        # Remove all contents of the logs directory
        for item in log_dir.iterdir():
            if item.is_file():
                item.unlink()
                print(f"Removed log file: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"Removed log directory: {item}")

        print(f"Successfully cleared logs directory: {log_dir}")
        return True

    except Exception as e:
        print(f"Failed to clear logs directory: {e}")
        return False