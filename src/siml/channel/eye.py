import numpy as np


def make_eye_diagram(
    waveform: np.ndarray,
    samples_per_bit: int = 32,
    ui_per_window: int = 2,
    n_time_bins: int = 100,
    n_amp_bins: int = 100,
    amp_range: tuple[float, float] = (-1.2, 1.2),
) -> np.ndarray:
    """Fold a time-domain waveform into an eye-diagram 2-D histogram.

    The waveform is sliced into fixed-length windows (each spanning
    ``ui_per_window`` unit intervals) and the windows are overlaid in a
    common time/amplitude grid. Each output pixel counts how many times the
    waveform passed through that (time, amplitude) location, which is exactly
    the density an eye diagram visualises. The result is intended as a CNN
    input image in a later phase.

    Args:
        waveform: 1-D voltage waveform, typically the output of
            ``bits_to_waveform``.
        samples_per_bit: Samples per unit interval. Must match the value used
            to build ``waveform``. Must be >= 1.
        ui_per_window: Number of unit intervals captured per window. Must be
            >= 1.
        n_time_bins: Horizontal (time-axis) resolution of the output image.
            Must be >= 1.
        n_amp_bins: Vertical (amplitude-axis) resolution of the output image.
            Must be >= 1.
        amp_range: ``(low, high)`` amplitude limits for the vertical axis.
            Samples outside this range are dropped from the histogram.

    Returns:
        np.ndarray of shape ``(n_amp_bins, n_time_bins)`` and dtype
        ``np.float32``. Row 0 is the bottom of the amplitude axis; values
        increase upward. Column 0 is time 0; values increase rightward over
        ``ui_per_window`` UI. Each entry is an un-normalised crossing count.

    Raises:
        ValueError: If ``waveform`` is not 1-D.
        ValueError: If ``waveform`` is shorter than one window.
        ValueError: If any of ``samples_per_bit``, ``ui_per_window``,
            ``n_time_bins`` or ``n_amp_bins`` is < 1.
    """
    if waveform.ndim != 1:
        raise ValueError(f"waveform must be 1-D, got shape {waveform.shape}.")
    if samples_per_bit < 1:
        raise ValueError(f"samples_per_bit must be >= 1, got {samples_per_bit}.")
    if ui_per_window < 1:
        raise ValueError(f"ui_per_window must be >= 1, got {ui_per_window}.")
    if n_time_bins < 1:
        raise ValueError(f"n_time_bins must be >= 1, got {n_time_bins}.")
    if n_amp_bins < 1:
        raise ValueError(f"n_amp_bins must be >= 1, got {n_amp_bins}.")

    # One window spans ui_per_window UI; this is the fold period in samples.
    window_len = ui_per_window * samples_per_bit

    # We discard the first half-UI so that bit edges land at the window borders
    # and the eye opening sits in the centre. After that offset there must be
    # at least one full window left, otherwise no eye can be formed.
    offset = samples_per_bit // 2
    if waveform.size - offset < window_len:
        raise ValueError(
            f"waveform too short: need at least offset + one window "
            f"({offset} + {window_len} = {offset + window_len}) samples, "
            f"got {waveform.size}."
        )

    # Drop the leading half-UI so the eye centres in the window.
    shifted = waveform[offset:]

    # Keep only a whole number of windows; the trailing remainder is discarded.
    # Integer division gives how many full windows fit after the offset.
    n_windows = shifted.size // window_len
    usable = shifted[: n_windows * window_len]

    # Reshape into (n_windows, window_len): each row is one overlaid window,
    # and column j is the same time position across every window. This is the
    # "folding" that stacks all UIs on top of each other.
    windows = usable.reshape(n_windows, window_len)

    # Flatten into paired 1-D vectors of (time index, amplitude). ravel() goes
    # row by row, so amp_coords[k] and time_coords[k] describe the same sample.
    amp_coords = windows.ravel()
    # Each window contributes the same column indices 0..window_len-1, repeated
    # once per window — this matches ravel()'s row-major order exactly.
    time_coords = np.tile(np.arange(window_len), n_windows)

    # 2-D histogram. First arg -> first output axis (amplitude rows), second arg
    # -> second output axis (time columns), giving shape (n_amp_bins, n_time_bins).
    # Samples outside amp_range fall outside the bin edges and are dropped.
    counts, _, _ = np.histogram2d(
        amp_coords,
        time_coords,
        bins=(n_amp_bins, n_time_bins),
        range=(amp_range, (0, window_len)),
    )

    # Row 0 is the lowest amplitude (counts increases upward); float32 keeps the
    # array light and ready as a CNN input tensor.
    return counts.astype(np.float32)
