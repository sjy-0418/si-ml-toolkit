"""Metric extraction for eye-diagram validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EyeMetrics:
    eye_height_v: float
    eye_width_ui: float
    jitter_ui: float
    threshold_v: float
    left_crossing_ui: float
    right_crossing_ui: float


def calculate_eye_metrics(
    frame: pd.DataFrame,
    center_ui: float = 0.5,
    vertical_window_ui: float = 0.08,
) -> EyeMetrics:
    """Calculate eye height, eye width, and jitter from sampled eye data."""
    required = {"trace_id", "time_ui", "voltage_v"}
    missing = required.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s): {missing_list}")

    threshold_v = _estimate_decision_threshold(frame["voltage_v"].to_numpy())
    eye_height_v = _calculate_eye_height(frame, center_ui, vertical_window_ui, threshold_v)
    crossings = _find_threshold_crossings(frame, threshold_v)

    left_crossings = np.array([value for value in crossings if value < center_ui], dtype=float)
    right_crossings = np.array([value for value in crossings if value > center_ui], dtype=float)

    if left_crossings.size == 0 or right_crossings.size == 0:
        raise ValueError("Could not identify both left and right eye crossings")

    left_edge = float(np.percentile(left_crossings, 95))
    right_edge = float(np.percentile(right_crossings, 5))
    eye_width_ui = max(0.0, right_edge - left_edge)

    all_crossings = np.concatenate([left_crossings, right_crossings])
    edge_centers = np.where(all_crossings < center_ui, np.median(left_crossings), np.median(right_crossings))
    jitter_ui = float(np.std(all_crossings - edge_centers, ddof=0))

    return EyeMetrics(
        eye_height_v=float(eye_height_v),
        eye_width_ui=float(eye_width_ui),
        jitter_ui=jitter_ui,
        threshold_v=float(threshold_v),
        left_crossing_ui=left_edge,
        right_crossing_ui=right_edge,
    )


def _estimate_decision_threshold(values: np.ndarray) -> float:
    low_level = np.percentile(values, 10)
    high_level = np.percentile(values, 90)
    return float((low_level + high_level) / 2.0)


def _calculate_eye_height(
    frame: pd.DataFrame,
    center_ui: float,
    vertical_window_ui: float,
    threshold_v: float,
) -> float:
    center_samples = frame[
        (frame["time_ui"] >= center_ui - vertical_window_ui)
        & (frame["time_ui"] <= center_ui + vertical_window_ui)
    ]
    if center_samples.empty:
        raise ValueError("No samples found in the eye-center measurement window")

    high_samples = center_samples[center_samples["voltage_v"] >= threshold_v]["voltage_v"].to_numpy()
    low_samples = center_samples[center_samples["voltage_v"] < threshold_v]["voltage_v"].to_numpy()

    if high_samples.size == 0 or low_samples.size == 0:
        raise ValueError("Eye-center window must contain both high and low logic levels")

    high_inner = np.percentile(high_samples, 5)
    low_inner = np.percentile(low_samples, 95)
    return max(0.0, float(high_inner - low_inner))


def _find_threshold_crossings(frame: pd.DataFrame, threshold_v: float) -> list[float]:
    crossings: list[float] = []

    for _, trace in frame.groupby("trace_id"):
        ordered = trace.sort_values("time_ui")
        time = ordered["time_ui"].to_numpy(dtype=float)
        voltage = ordered["voltage_v"].to_numpy(dtype=float)

        for index in range(len(time) - 1):
            v0 = voltage[index] - threshold_v
            v1 = voltage[index + 1] - threshold_v

            if v0 == 0:
                crossings.append(float(time[index]))
                continue

            if v0 * v1 < 0:
                fraction = abs(v0) / (abs(v0) + abs(v1))
                crossings.append(float(time[index] + fraction * (time[index + 1] - time[index])))

    if not crossings:
        raise ValueError("Could not find threshold crossings in the eye data")

    return crossings
