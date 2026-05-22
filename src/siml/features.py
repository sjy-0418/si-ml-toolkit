import numpy as np
import skrf


def insertion_loss_db(network: skrf.Network, freq_hz: float) -> float:
    """Extract insertion loss (|S21|) at a given frequency.

    Args:
        network: A 2-port (or more) skrf.Network object.
        freq_hz: Target frequency in Hz (e.g. 1.6e9 for DDR4 Nyquist).

    Returns:
        Insertion loss at freq_hz in dB (20 * log10(|S21|)).
        Negative values indicate attenuation (typical for passive channels).

    Raises:
        ValueError: If network has fewer than 2 ports.
        ValueError: If freq_hz is outside the network's measured frequency range.
    """
    if network.nports < 2:
        raise ValueError(
            f"Network must have at least 2 ports to extract S21, "
            f"but got {network.nports} port(s)."
        )

    freqs = network.f  # 1-D float array of frequency points [Hz]
    f_min, f_max = freqs[0], freqs[-1]

    if not (f_min <= freq_hz <= f_max):
        raise ValueError(
            f"freq_hz={freq_hz:.4e} Hz is outside the measured range "
            f"[{f_min:.4e}, {f_max:.4e}] Hz."
        )

    # S21: complex voltage wave ratio (port 2 out / port 1 in)
    # s[:, 1, 0] → all freq points, row=1 (port 2), col=0 (port 1)
    s21_complex = network.s[:, 1, 0]

    # Interpolate in linear magnitude domain, NOT in dB.
    # Reason: dB is logarithmic, so linear interpolation between dB values
    # gives the wrong physical result. E.g. midpoint of -3 dB and -10 dB
    # is NOT -6.5 dB — it should be computed from linear magnitudes first.
    mag_linear = np.abs(s21_complex)
    mag_at_freq = np.interp(freq_hz, freqs, mag_linear)

    return 20.0 * np.log10(mag_at_freq)
