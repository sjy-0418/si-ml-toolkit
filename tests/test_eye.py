import numpy as np
import pytest

from siml.channel.eye import make_eye_diagram


class TestMakeEyeDiagramOutput:
    def test_returns_ndarray(self) -> None:
        """Return value must be a numpy ndarray."""
        wf = np.zeros(32 * 10, dtype=np.float32)
        result = make_eye_diagram(wf, samples_per_bit=32)
        assert isinstance(result, np.ndarray)

    def test_shape_is_amp_bins_by_time_bins(self) -> None:
        """Return shape must be (n_amp_bins, n_time_bins)."""
        wf = np.zeros(32 * 10, dtype=np.float32)
        result = make_eye_diagram(
            wf, samples_per_bit=32, n_time_bins=64, n_amp_bins=48
        )
        assert result.shape == (48, 64)

    def test_dtype_is_float32(self) -> None:
        """Return dtype must be np.float32."""
        wf = np.zeros(32 * 10, dtype=np.float32)
        result = make_eye_diagram(wf, samples_per_bit=32)
        assert result.dtype == np.float32


class TestMakeEyeDiagramCounting:
    def test_total_counts_equal_usable_samples(self) -> None:
        """Every in-range sample must be counted exactly once.

        With no overshoot, total counts == n_windows * window_len, where the
        usable length is the waveform minus the half-UI offset, truncated to a
        whole number of windows.
        """
        samples_per_bit, ui_per_window = 32, 2
        window_len = ui_per_window * samples_per_bit
        offset = samples_per_bit // 2
        # 10 windows worth of samples plus the offset and a stray remainder.
        wf = np.full(offset + window_len * 10 + 5, 0.5, dtype=np.float32)

        eye = make_eye_diagram(
            wf, samples_per_bit=samples_per_bit, ui_per_window=ui_per_window
        )

        assert eye.sum() == window_len * 10

    def test_samples_outside_amp_range_are_dropped(self) -> None:
        """Amplitudes beyond amp_range must not be counted."""
        # +5.0 is far outside the default (-1.2, 1.2) range.
        wf = np.full(32 * 10, 5.0, dtype=np.float32)
        eye = make_eye_diagram(wf, samples_per_bit=32)
        assert eye.sum() == 0


class TestMakeEyeDiagramOrientation:
    def test_positive_dc_lands_in_upper_rows(self) -> None:
        """A constant +1.0 waveform must accumulate near the top of the axis.

        Row 0 is the lowest amplitude, so a positive level belongs in the
        upper half of the array.
        """
        wf = np.full(32 * 10, 1.0, dtype=np.float32)
        eye = make_eye_diagram(wf, samples_per_bit=32, n_amp_bins=100)
        occupied_rows = np.flatnonzero(eye.sum(axis=1))
        assert occupied_rows.min() >= 50

    def test_negative_dc_lands_in_lower_rows(self) -> None:
        """A constant -1.0 waveform must accumulate near the bottom."""
        wf = np.full(32 * 10, -1.0, dtype=np.float32)
        eye = make_eye_diagram(wf, samples_per_bit=32, n_amp_bins=100)
        occupied_rows = np.flatnonzero(eye.sum(axis=1))
        assert occupied_rows.max() < 50


class TestMakeEyeDiagramErrors:
    def test_rejects_non_1d_waveform(self) -> None:
        """A 2-D waveform must raise ValueError."""
        wf = np.zeros((32 * 10, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            make_eye_diagram(wf, samples_per_bit=32)

    def test_rejects_waveform_shorter_than_one_window(self) -> None:
        """A waveform shorter than offset + one window must raise ValueError."""
        wf = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="too short"):
            make_eye_diagram(wf, samples_per_bit=32)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"samples_per_bit": 0},
            {"ui_per_window": 0},
            {"n_time_bins": 0},
            {"n_amp_bins": 0},
        ],
    )
    def test_rejects_non_positive_parameters(self, kwargs: dict) -> None:
        """Any sizing parameter below 1 must raise ValueError."""
        wf = np.zeros(32 * 10, dtype=np.float32)
        base = {"samples_per_bit": 32}
        base.update(kwargs)
        with pytest.raises(ValueError, match=">= 1"):
            make_eye_diagram(wf, **base)
