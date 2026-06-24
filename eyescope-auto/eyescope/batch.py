"""Batch validation across multiple eye-diagram captures and corners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .mask import Thresholds, evaluate_eye
from .metrics import calculate_eye_metrics
from .parser import load_eye_csv


@dataclass(frozen=True)
class BatchConfig:
    manifest_path: Path
    output_csv_path: Path
    output_report_path: Path


def run_batch_validation(config: BatchConfig) -> pd.DataFrame:
    """Run validation for every row in a batch manifest."""
    manifest = pd.read_csv(config.manifest_path)
    _validate_manifest(manifest)

    rows: list[dict[str, object]] = []
    manifest_root = config.manifest_path.parent.parent

    for item in manifest.to_dict(orient="records"):
        input_path = manifest_root / str(item["input"])
        thresholds = Thresholds(
            min_eye_height_v=float(item["min_eye_height_v"]),
            min_eye_width_ui=float(item["min_eye_width_ui"]),
            max_jitter_ui=float(item["max_jitter_ui"]),
        )

        frame = load_eye_csv(input_path)
        metrics = calculate_eye_metrics(frame)
        evaluation = evaluate_eye(metrics, thresholds)
        checks = evaluation["checks"]

        rows.append(
            {
                "capture_id": item["capture_id"],
                "corner": item["corner"],
                "input": item["input"],
                "temperature_c": float(item["temperature_c"]),
                "voltage_v": float(item["voltage_v"]),
                "overall": evaluation["overall"],
                "eye_height_v": metrics.eye_height_v,
                "eye_width_ui": metrics.eye_width_ui,
                "jitter_ui": metrics.jitter_ui,
                "eye_height_check": checks["eye_height"],
                "eye_width_check": checks["eye_width"],
                "jitter_check": checks["jitter"],
                "diagnosis": " ".join(evaluation["diagnosis"]),
            }
        )

    result = pd.DataFrame(rows)
    config.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.output_csv_path, index=False)
    write_batch_report(result, config.output_report_path)
    return result


def write_batch_report(result: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a Markdown summary for a batch validation run."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = len(result)
    passed = int((result["overall"] == "PASS").sum())
    failed = total - passed
    pass_rate = passed / total if total else 0.0

    table_rows = []
    for row in result.itertuples(index=False):
        table_rows.append(
            "| "
            + " | ".join(
                [
                    str(row.capture_id),
                    str(row.corner),
                    str(row.overall),
                    f"{row.eye_height_v:.3f}",
                    f"{row.eye_width_ui:.3f}",
                    f"{row.jitter_ui:.4f}",
                    _failed_checks(row),
                ]
            )
            + " |"
        )

    failed_rows = result[result["overall"] == "FAIL"]
    if failed_rows.empty:
        debug_section = "- No failing corners in this batch."
    else:
        debug_items = []
        for row in failed_rows.itertuples(index=False):
            debug_items.append(
                f"- `{row.capture_id}` ({row.corner}): {_failed_checks(row)}. {row.diagnosis}"
            )
        debug_section = "\n".join(debug_items)

    content = "\n".join(
        [
            "# EyeScope Auto Batch Validation Report",
            "",
            "## Summary",
            "",
            f"- Captures evaluated: {total}",
            f"- Passed: {passed}",
            f"- Failed: {failed}",
            f"- Pass rate: {pass_rate:.1%}",
            "",
            "## Corner Matrix",
            "",
            "| Capture ID | Corner | Overall | Eye height (V) | Eye width (UI) | Jitter (UI) | Failed checks |",
            "| --- | --- | :---: | ---: | ---: | ---: | --- |",
            *table_rows,
            "",
            "## Debug Focus",
            "",
            debug_section,
            "",
            "## Validation Interpretation",
            "",
            "A batch report is useful for validation teams because it turns repeated lab captures into a reviewable regression matrix. The goal is not only to mark pass/fail, but to identify which corners deserve debug attention first.",
            "",
        ]
    )

    output.write_text(content, encoding="utf-8")
    return output


def _validate_manifest(manifest: pd.DataFrame) -> None:
    required = {
        "capture_id",
        "corner",
        "input",
        "temperature_c",
        "voltage_v",
        "min_eye_height_v",
        "min_eye_width_ui",
        "max_jitter_ui",
    }
    missing = required.difference(manifest.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing manifest column(s): {missing_list}")

    if manifest.empty:
        raise ValueError("Batch manifest contains no captures")


def _failed_checks(row: object) -> str:
    failed = []
    if not row.eye_height_check:
        failed.append("eye_height")
    if not row.eye_width_check:
        failed.append("eye_width")
    if not row.jitter_check:
        failed.append("jitter")
    return ", ".join(failed) if failed else "-"

