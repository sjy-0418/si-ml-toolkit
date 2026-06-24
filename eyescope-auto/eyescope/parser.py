"""CSV loading utilities for oscilloscope-style eye-diagram captures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"time_ui", "voltage_v"}
OPTIONAL_COLUMNS = {"trace_id"}


def load_eye_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate eye-diagram sample data.

    Expected columns:
    - trace_id: acquisition or waveform identifier, optional
    - time_ui: time normalized to one unit interval
    - voltage_v: measured voltage in volts
    """
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    frame.columns = [column.strip().lower() for column in frame.columns]

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s): {missing_list}")

    if "trace_id" not in frame.columns:
        frame["trace_id"] = 0

    frame = frame[["trace_id", "time_ui", "voltage_v"]].copy()
    frame["time_ui"] = pd.to_numeric(frame["time_ui"], errors="raise")
    frame["voltage_v"] = pd.to_numeric(frame["voltage_v"], errors="raise")

    if frame.empty:
        raise ValueError("Eye-diagram CSV contains no samples")

    return frame.sort_values(["trace_id", "time_ui"]).reset_index(drop=True)
