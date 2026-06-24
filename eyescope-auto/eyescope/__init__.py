"""EyeScope Auto: high-speed interface eye-diagram validation helpers."""

from .batch import BatchConfig, run_batch_validation
from .mask import Thresholds, evaluate_eye
from .metrics import calculate_eye_metrics
from .parser import load_eye_csv

__all__ = [
    "BatchConfig",
    "Thresholds",
    "calculate_eye_metrics",
    "evaluate_eye",
    "load_eye_csv",
    "run_batch_validation",
]
