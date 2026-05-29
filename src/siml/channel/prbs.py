import numpy as np

# Tap polynomials from IEEE 802.3 / ITU-T O.150 standard.
# Key   = PRBS order (n)
# Value = (n, k) for polynomial x^n + x^k + 1
_TAPS: dict[int, tuple[int, int]] = {
    7:  (7,  6),
    15: (15, 14),
    23: (23, 18),
    31: (31, 28),
}


def generate_prbs(
    order: int,
    length: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a PRBS bit sequence using a Fibonacci LFSR.

    Args:
        order: PRBS order. Supported values: 7, 15, 23, 31.
        length: Number of bits to generate. Must be >= 1.
        seed: Initial LFSR state (1 to 2**order - 1 inclusive).
            Defaults to all-ones (2**order - 1) if not provided.
            Zero is invalid — an all-zero LFSR produces only zeros forever.

    Returns:
        np.ndarray of shape (length,) and dtype np.uint8 containing 0/1 values.

    Raises:
        ValueError: If order is not in [7, 15, 23, 31].
        ValueError: If length < 1.
        ValueError: If seed == 0 or seed is outside [1, 2**order - 1].
    """
    if order not in _TAPS:
        supported = sorted(_TAPS.keys())
        raise ValueError(
            f"order={order} is not supported. Supported orders: {supported}."
        )

    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}.")

    max_seed = (1 << order) - 1  # e.g. 127 for order=7

    if seed is None:
        seed = max_seed  # all-ones: standard PRBS initialisation
    elif seed == 0:
        raise ValueError(
            "seed=0 is invalid: an all-zero LFSR state produces only zeros forever."
        )
    elif not (1 <= seed <= max_seed):
        raise ValueError(
            f"seed={seed} is out of range for order={order}. "
            f"Valid range: [1, {max_seed}]."
        )

    n, k = _TAPS[order]

    bits = np.empty(length, dtype=np.uint8)
    state = seed

    for i in range(length):
        # Output the LSB before shifting
        bits[i] = state & 1

        # Recurrence for x^n + x^k + 1: s_{t+n} = s_{t+k} XOR s_t
        # s_{t+k} = bit at 0-indexed position k; s_t = LSB (output bit just emitted)
        feedback = ((state >> k) & 1) ^ (state & 1)

        # Shift right by 1, insert feedback at the MSB position
        state = (feedback << (n - 1)) | (state >> 1)

    return bits
