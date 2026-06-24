"""Run batch eye-diagram validation from a manifest file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eyescope.batch import BatchConfig, run_batch_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EyeScope Auto batch validation")
    parser.add_argument("--manifest", default="data/batch_manifest.csv", help="Batch manifest CSV")
    parser.add_argument("--output-csv", default="reports/batch_summary.csv", help="Output CSV summary")
    parser.add_argument(
        "--report",
        default="reports/batch_report.md",
        help="Output Markdown batch report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_batch_validation(
        BatchConfig(
            manifest_path=Path(args.manifest),
            output_csv_path=Path(args.output_csv),
            output_report_path=Path(args.report),
        )
    )

    passed = int((result["overall"] == "PASS").sum())
    failed = len(result) - passed
    print(f"Batch captures: {len(result)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"CSV summary written to: {args.output_csv}")
    print(f"Markdown report written to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
