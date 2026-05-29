import numpy as np
from scipy.ndimage import gaussian_filter1d


def bits_to_waveform(
    bits: np.ndarray,
    samples_per_bit: int = 32,
    rise_time_ui: float = 0.3,
) -> np.ndarray:
    """Convert an NRZ bit sequence into a time-domain voltage waveform.

    Args:
        bits: 1-D array of 0/1 values (dtype uint8). Typically the output
            of ``generate_prbs``.
        samples_per_bit: Number of samples per unit interval (UI). Must be >= 1.
        rise_time_ui: 10-90 % rise time expressed in UI. 0 returns an ideal
            square wave with no filtering.

    Returns:
        np.ndarray of shape ``(len(bits) * samples_per_bit,)`` and dtype
        ``np.float32``. Values nominally in [-1, +1].

    Raises:
        ValueError: If ``bits`` is not 1-D.
        ValueError: If ``bits`` contains values other than 0 or 1.
        ValueError: If ``samples_per_bit < 1``.
        ValueError: If ``rise_time_ui < 0``.
    """
    if bits.ndim != 1:
        raise ValueError(f"bits must be 1-D, got shape {bits.shape}.")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0 and 1 values.")
    if samples_per_bit < 1:
        raise ValueError(f"samples_per_bit must be >= 1, got {samples_per_bit}.")
    if rise_time_ui < 0:
        raise ValueError(f"rise_time_ui must be >= 0, got {rise_time_ui}.")

    # Map 0 → -1.0 and 1 → +1.0.
    # Subtraction is cheaper than a conditional: (0*2 - 1) = -1, (1*2 - 1) = +1.
    nrz = bits.astype(np.float32) * 2.0 - 1.0

    # Repeat each sample samples_per_bit times to create an ideal step waveform.
    # np.repeat([−1, +1, +1], 4) → [−1,−1,−1,−1, +1,+1,+1,+1, +1,+1,+1,+1]
    waveform = np.repeat(nrz, samples_per_bit)

    if rise_time_ui == 0.0:
        return waveform

    # 10-90% rise time of a Gaussian CDF relates to sigma by t_r ≈ 2.197 * sigma.
    # Rearranging: sigma_samples = (rise_time_ui * samples_per_bit) / 2.197
    sigma_samples = (rise_time_ui * samples_per_bit) / 2.197

    # mode='nearest' repeats the edge value instead of wrapping or zero-padding,
    # which prevents artificial transitions at the waveform boundaries.
    smoothed = gaussian_filter1d(waveform, sigma=sigma_samples, mode="nearest")

    return smoothed.astype(np.float32)
