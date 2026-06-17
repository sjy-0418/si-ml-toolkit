"""EyeScope Auto: high-speed interface eye-diagram validation helpers."""

from .mask import Thresholds, evaluate_eye
from .metrics import calculate_eye_metrics
from .parser import load_eye_csv
from .report import generate_markdown_report

__all__ = [
    "Thresholds",
    "calculate_eye_metrics",
    "evaluate_eye",
    "generate_markdown_report",
    "load_eye_csv",
]
