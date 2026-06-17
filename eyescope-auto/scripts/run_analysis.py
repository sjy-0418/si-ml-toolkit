"""Run eye-diagram validation on a CSV capture."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eyescope.mask import Thresholds, evaluate_eye
from eyescope.metrics import calculate_eye_metrics
from eyescope.parser import load_eye_csv
from eyescope.report import generate_eye_plot, generate_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EyeScope Auto validation")
    parser.add_argument("--input", default="data/sample_eye.csv", help="Input oscilloscope CSV")
    parser.add_argument("--report", default="reports/sample_report.md", help="Output Markdown report")
    parser.add_argument("--plot", default="reports/sample_eye.png", help="Output eye plot")
    parser.add_argument("--min-eye-height", type=float, default=0.45)
    parser.add_argument("--min-eye-width", type=float, default=0.55)
    parser.add_argument("--max-jitter", type=float, default=0.035)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = Thresholds(
        min_eye_height_v=args.min_eye_height,
        min_eye_width_ui=args.min_eye_width,
        max_jitter_ui=args.max_jitter,
    )

    frame = load_eye_csv(args.input)
    metrics = calculate_eye_metrics(frame)
    evaluation = evaluate_eye(metrics, thresholds)
    plot_path = generate_eye_plot(frame, args.plot)
    report_path = generate_markdown_report(metrics, evaluation, thresholds, args.report, plot_path)

    print(f"Overall result: {evaluation['overall']}")
    print(f"Eye height: {metrics.eye_height_v:.3f} V")
    print(f"Eye width: {metrics.eye_width_ui:.3f} UI")
    print(f"Jitter estimate: {metrics.jitter_ui:.4f} UI")
    print(f"Report written to: {Path(report_path)}")
    return 0 if evaluation["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
