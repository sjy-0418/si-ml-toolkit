from pathlib import Path

import pandas as pd

from eyescope.batch import BatchConfig, run_batch_validation


def test_run_batch_validation_writes_summary(tmp_path):
    fixture_root = Path(__file__).resolve().parents[1]
    manifest_path = fixture_root / "data" / "batch_manifest.csv"
    output_csv = tmp_path / "batch_summary.csv"
    output_report = tmp_path / "batch_report.md"

    result = run_batch_validation(
        BatchConfig(
            manifest_path=manifest_path,
            output_csv_path=output_csv,
            output_report_path=output_report,
        )
    )

    assert len(result) == 4
    assert {"PASS", "FAIL"}.issubset(set(result["overall"]))
    assert output_csv.exists()
    assert output_report.exists()

    persisted = pd.read_csv(output_csv)
    assert list(persisted["capture_id"]) == list(result["capture_id"])

