"""Threshold-based pass/fail and rule diagnosis for validation results."""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import EyeMetrics


@dataclass(frozen=True)
class Thresholds:
    min_eye_height_v: float = 0.45
    min_eye_width_ui: float = 0.55
    max_jitter_ui: float = 0.035


def evaluate_eye(metrics: EyeMetrics, thresholds: Thresholds) -> dict[str, object]:
    checks = {
        "eye_height": metrics.eye_height_v >= thresholds.min_eye_height_v,
        "eye_width": metrics.eye_width_ui >= thresholds.min_eye_width_ui,
        "jitter": metrics.jitter_ui <= thresholds.max_jitter_ui,
    }

    diagnosis: list[str] = []
    if not checks["eye_height"]:
        diagnosis.append(
            "Low eye height may indicate noise margin degradation, attenuation, or a power integrity issue."
        )
    if not checks["eye_width"]:
        diagnosis.append(
            "Narrow eye width may indicate a timing margin issue, inter-symbol interference, or channel loss."
        )
    if not checks["jitter"]:
        diagnosis.append(
            "High jitter may indicate clock instability, crosstalk, or a signal integrity issue."
        )
    if not diagnosis:
        diagnosis.append("All configured eye-mask checks passed for this capture.")

    return {
        "overall": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "diagnosis": diagnosis,
    }
