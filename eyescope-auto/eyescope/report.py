"""Markdown report generation for validation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .mask import Thresholds
from .metrics import EyeMetrics


def generate_eye_plot(frame: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(frame["time_ui"], frame["voltage_v"], s=14, alpha=0.65)
    ax.set_title("Eye Diagram Capture")
    ax.set_xlabel("Time (UI)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def generate_markdown_report(
    metrics: EyeMetrics,
    evaluation: dict[str, object],
    thresholds: Thresholds,
    output_path: str | Path,
    plot_path: str | Path | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    checks = evaluation["checks"]
    diagnosis = evaluation["diagnosis"]
    plot_section = f"\n![Eye diagram capture]({Path(plot_path).name})\n" if plot_path else ""

    content = f"""# EyeScope Auto Validation Report

## Summary

Overall result: **{evaluation["overall"]}**

{plot_section}
## Measured Metrics

| Metric | Measured | Limit | Result |
| --- | ---: | ---: | :---: |
| Eye height | {metrics.eye_height_v:.3f} V | >= {thresholds.min_eye_height_v:.3f} V | {_status(checks["eye_height"])} |
| Eye width | {metrics.eye_width_ui:.3f} UI | >= {thresholds.min_eye_width_ui:.3f} UI | {_status(checks["eye_width"])} |
| Jitter estimate | {metrics.jitter_ui:.4f} UI | <= {thresholds.max_jitter_ui:.4f} UI | {_status(checks["jitter"])} |

## Measurement Context

- Decision threshold: {metrics.threshold_v:.3f} V
- Left eye crossing: {metrics.left_crossing_ui:.3f} UI
- Right eye crossing: {metrics.right_crossing_ui:.3f} UI

## Rule-Based Diagnosis

{_format_diagnosis(diagnosis)}
"""
    output.write_text(content, encoding="utf-8")
    return output


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _format_diagnosis(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)
