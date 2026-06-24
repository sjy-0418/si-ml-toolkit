import pandas as pd

from eyescope.mask import Thresholds, evaluate_eye
from eyescope.metrics import EyeMetrics, calculate_eye_metrics


def test_calculate_eye_metrics_for_clean_capture():
    frame = pd.DataFrame(
        {
            "trace_id": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "time_ui": [
                0.1,
                0.2,
                0.5,
                0.8,
                0.9,
                0.1,
                0.22,
                0.5,
                0.78,
                0.9,
                0.1,
                0.21,
                0.5,
                0.79,
                0.9,
            ],
            "voltage_v": [0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 0.98, 0.5, 0.0, 1.0, 0.5, 0.02, 0.5, 1.0],
        }
    )

    metrics = calculate_eye_metrics(frame)

    assert metrics.eye_height_v > 0.95
    assert 0.55 <= metrics.eye_width_ui <= 0.60
    assert metrics.jitter_ui < 0.02


def test_evaluate_eye_passes_when_metrics_meet_thresholds():
    metrics = EyeMetrics(
        eye_height_v=0.72,
        eye_width_ui=0.62,
        jitter_ui=0.014,
        threshold_v=0.5,
        left_crossing_ui=0.2,
        right_crossing_ui=0.8,
    )

    result = evaluate_eye(metrics, Thresholds())

    assert result["overall"] == "PASS"
    assert all(result["checks"].values())


def test_evaluate_eye_fails_and_reports_diagnosis():
    metrics = EyeMetrics(
        eye_height_v=0.30,
        eye_width_ui=0.40,
        jitter_ui=0.060,
        threshold_v=0.5,
        left_crossing_ui=0.3,
        right_crossing_ui=0.7,
    )

    result = evaluate_eye(metrics, Thresholds())

    assert result["overall"] == "FAIL"
    assert result["checks"] == {"eye_height": False, "eye_width": False, "jitter": False}
    assert len(result["diagnosis"]) == 3
