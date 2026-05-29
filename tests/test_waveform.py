import numpy as np
import pytest

from siml.channel.waveform import bits_to_waveform


class TestBitsToWaveformOutput:
    def test_returns_ndarray(self) -> None:
        """Return value must be a numpy ndarray."""
        bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        result = bits_to_waveform(bits)
        assert isinstance(result, np.ndarray)

    def test_shape_is_bits_times_samples_per_bit(self) -> None:
        """Return shape must be (len(bits) * samples_per_bit,)."""
        bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=16)
        assert result.shape == (4 * 16,)

    def test_dtype_is_float32(self) -> None:
        """Return dtype must be np.float32."""
        bits = np.array([0, 1], dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=8, rise_time_ui=0.3)
        assert result.dtype == np.float32


class TestBitsToWaveformNrzMapping:
    def test_all_zeros_map_to_minus_one(self) -> None:
        """If all bits are 0, the entire output must be -1.0."""
        bits = np.zeros(8, dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=4, rise_time_ui=0.0)
        np.testing.assert_array_equal(result, np.full(32, -1.0, dtype=np.float32))

    def test_all_ones_map_to_plus_one(self) -> None:
        """If all bits are 1, the entire output must be +1.0."""
        bits = np.ones(8, dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=4, rise_time_ui=0.0)
        np.testing.assert_array_equal(result, np.full(32, 1.0, dtype=np.float32))

    def test_each_bit_fills_exactly_samples_per_bit_slots(self) -> None:
        """Each bit must fill exactly samples_per_bit slots.

        With [0, 1, 0] and samples_per_bit=4:
        - Samples 0-3: -1.0 (bit 0)
        - Samples 4-7: +1.0 (bit 1)
        - Samples 8-11: -1.0 (bit 2)
        """
        bits = np.array([0, 1, 0], dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=4, rise_time_ui=0.0)
        np.testing.assert_array_equal(result[0:4], [-1.0, -1.0, -1.0, -1.0])
        np.testing.assert_array_equal(result[4:8], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result[8:12], [-1.0, -1.0, -1.0, -1.0])


class TestBitsToWaveformRiseTime:
    def test_zero_rise_time_returns_exact_square_wave(self) -> None:
        """With rise_time_ui=0, must return an exact ±1.0 square wave without Gaussian filtering."""
        bits = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=10, rise_time_ui=0.0)
        unique_values = np.unique(result)
        np.testing.assert_array_equal(unique_values, [-1.0, 1.0])

    def test_nonzero_rise_time_creates_intermediate_values_at_transition(self) -> None:
        """With rise_time_ui > 0, intermediate values other than ±1.0 must appear at transition edges.

        The most direct verification that the Gaussian filter actually operates.
        """
        bits = np.array([0] * 5 + [1] * 5, dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=20, rise_time_ui=0.3)
        # samples just before/after the transition must be intermediate values between -1.0 and +1.0
        transition_idx = 5 * 20  # start of bit 5 (first +1 bit)
        assert -1.0 < result[transition_idx - 1] < 0.0
        assert 0.0 < result[transition_idx] < 1.0

    def test_filtered_values_stay_within_minus_one_to_plus_one(self) -> None:
        """After Gaussian filtering, values must not exceed the [-1, +1] range.

        Gaussian is a weighted average of input values, so it cannot expand the range.
        """
        bits = np.array([0, 1] * 20, dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=32, rise_time_ui=0.3)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_transition_midpoint_is_near_zero(self) -> None:
        """The transition midpoint (average of samples n-1 and n) must be close to 0.

        Since Gaussian is a symmetric kernel, w[n-1] + w[n] ≈ 0 holds mathematically
        at the midpoint of a step transition.
        With sufficient flat regions on both sides, boundary effects become negligible.
        """
        bits = np.array([0] * 10 + [1] * 10, dtype=np.uint8)
        spb = 100
        result = bits_to_waveform(bits, samples_per_bit=spb, rise_time_ui=0.3)
        transition_idx = 10 * spb  # start sample of the first +1 bit
        midpoint = (result[transition_idx - 1] + result[transition_idx]) / 2
        assert abs(midpoint) < 0.05


class TestBitsToWaveformErrors:
    def test_raises_for_2d_bits(self) -> None:
        """Must raise ValueError if bits is a 2-D array."""
        bits_2d = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        with pytest.raises(ValueError, match="1-D"):
            bits_to_waveform(bits_2d)

    def test_raises_for_invalid_bit_values(self) -> None:
        """Must raise ValueError if bits contains values other than 0 or 1."""
        bits = np.array([0, 1, 2], dtype=np.uint8)
        with pytest.raises(ValueError, match="only 0 and 1"):
            bits_to_waveform(bits)

    def test_raises_for_zero_samples_per_bit(self) -> None:
        """Must raise ValueError if samples_per_bit=0."""
        bits = np.array([0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="samples_per_bit must be >= 1"):
            bits_to_waveform(bits, samples_per_bit=0)

    def test_raises_for_negative_samples_per_bit(self) -> None:
        """Must raise ValueError if samples_per_bit is negative."""
        bits = np.array([0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="samples_per_bit must be >= 1"):
            bits_to_waveform(bits, samples_per_bit=-4)

    def test_raises_for_negative_rise_time(self) -> None:
        """Must raise ValueError if rise_time_ui is negative."""
        bits = np.array([0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="rise_time_ui must be >= 0"):
            bits_to_waveform(bits, rise_time_ui=-0.1)
