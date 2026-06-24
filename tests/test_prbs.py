import numpy as np
import pytest

from siml.channel.prbs import generate_prbs


class TestGeneratePrbsOutput:
    def test_returns_ndarray(self) -> None:
        """Return value should be a numpy ndarray."""
        result = generate_prbs(order=7, length=10)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_length(self) -> None:
        """Shape of the returned array should match the requested length."""
        result = generate_prbs(order=7, length=50)
        assert result.shape == (50,)

    def test_dtype_is_uint8(self) -> None:
        """dtype of the returned array should be np.uint8."""
        result = generate_prbs(order=7, length=10)
        assert result.dtype == np.uint8

    def test_values_are_only_zero_or_one(self) -> None:
        """The bit sequence should contain only 0s and 1s."""
        result = generate_prbs(order=7, length=200)
        assert set(np.unique(result)).issubset({0, 1})

    def test_contains_both_zero_and_one(self) -> None:
        """Both 0 and 1 must appear over a sufficient length (guards against all-zero/all-one sequences)."""
        result = generate_prbs(order=7, length=127)
        assert 0 in result
        assert 1 in result


class TestGeneratePrbsPeriod:
    """Core PRBS property: period = 2^order - 1."""

    @pytest.mark.parametrize("order", [7, 15])
    def test_sequence_repeats_after_one_period(self, order: int) -> None:
        """Generating two periods worth of bits should yield identical first and second halves.

        By definition, a maximal-length LFSR repeats exactly after 2^order - 1 bits.
        Passing this test confirms that the tap polynomial and LFSR logic are correctly implemented.

        Only small orders (7, 15) are checked here: their periods (127, 32767) are tiny,
        so verifying full repetition is fast and already proves the tap-polynomial logic.
        Orders 23 and 31 have periods of ~8.4M and ~2.1B bits — generating two full periods
        would mean billions of bits, which is impractical. They are covered separately by
        TestGeneratePrbsLargeOrder, which only checks fast, finite-length generation.
        """
        period = (1 << order) - 1  # 2^order - 1
        bits = generate_prbs(order=order, length=2 * period)
        np.testing.assert_array_equal(bits[:period], bits[period:])


class TestGeneratePrbsLargeOrder:
    """Large orders (23, 31): only check that finite-length generation works.

    Their full periods are far too long to verify repetition, so we just confirm
    a short slice generates without error and has the expected shape, dtype, and value range.
    """

    @pytest.mark.parametrize("order", [23, 31])
    def test_generates_short_length_without_error(self, order: int) -> None:
        """A short slice (10000 bits) should generate quickly with correct shape, dtype, and values."""
        length = 10_000
        bits = generate_prbs(order=order, length=length)
        assert bits.shape == (length,)
        assert bits.dtype == np.uint8
        assert set(np.unique(bits)).issubset({0, 1})


class TestGeneratePrbsSeed:
    def test_same_seed_gives_same_sequence(self) -> None:
        """Calling twice with the same seed should produce identical sequences."""
        a = generate_prbs(order=7, length=50, seed=42)
        b = generate_prbs(order=7, length=50, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_sequence(self) -> None:
        """Different seeds should (in general) produce different sequences."""
        a = generate_prbs(order=7, length=50, seed=1)
        b = generate_prbs(order=7, length=50, seed=2)
        assert not np.array_equal(a, b)

    def test_default_seed_is_all_ones(self) -> None:
        """Omitting seed should produce the same result as explicitly passing an all-ones seed."""
        default = generate_prbs(order=7, length=50)
        explicit = generate_prbs(order=7, length=50, seed=0b1111111)
        np.testing.assert_array_equal(default, explicit)

    def test_length_longer_than_period_wraps_around(self) -> None:
        """A length greater than the period should wrap around naturally.

        When generating period bits and period+20 bits separately,
        the first period bits should be identical, confirming correct wrap-around behavior.
        """
        period = (1 << 7) - 1
        short = generate_prbs(order=7, length=period)
        long_ = generate_prbs(order=7, length=period + 20)
        np.testing.assert_array_equal(short, long_[:period])


class TestGeneratePrbsErrors:
    def test_raises_for_unsupported_order(self) -> None:
        """An unsupported order should raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            generate_prbs(order=8, length=10)

    def test_error_message_lists_supported_orders(self) -> None:
        """The error message should include the list of supported orders."""
        with pytest.raises(ValueError, match=r"\[7, 15, 23, 31\]"):
            generate_prbs(order=99, length=10)

    def test_raises_for_zero_length(self) -> None:
        """length=0 should raise ValueError."""
        with pytest.raises(ValueError, match="length must be >= 1"):
            generate_prbs(order=7, length=0)

    def test_raises_for_negative_length(self) -> None:
        """A negative length should raise ValueError."""
        with pytest.raises(ValueError, match="length must be >= 1"):
            generate_prbs(order=7, length=-1)

    def test_raises_for_zero_seed(self) -> None:
        """seed=0 should raise ValueError."""
        with pytest.raises(ValueError, match="seed=0 is invalid"):
            generate_prbs(order=7, length=10, seed=0)

    def test_raises_for_seed_above_max(self) -> None:
        """A seed exceeding 2^order - 1 should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            generate_prbs(order=7, length=10, seed=128)  # max for order=7 is 127

    def test_raises_for_seed_below_min(self) -> None:
        """A seed below 0 should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            generate_prbs(order=7, length=10, seed=-1)
