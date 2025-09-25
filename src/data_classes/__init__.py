"""
Data Classes Package

Contains centralized configuration classes and progress tracking components
for better code organization and maintainability.

Modules:
- configs: Configuration classes and enums (OptimizationConfig, OptimizationMode, etc.)
- callbacks: Progress tracking and callback components (TrialProgress, UnifiedProgress, etc.)
"""

# Import commonly used classes for convenience
from .configs import OptimizationConfig, OptimizationMode, OptimizationObjective
from .callbacks import TrialProgress, AggregatedProgress, UnifiedProgress

__all__ = [
    'OptimizationConfig',
    'OptimizationMode',
    'OptimizationObjective',
    'TrialProgress',
    'AggregatedProgress',
    'UnifiedProgress'
]