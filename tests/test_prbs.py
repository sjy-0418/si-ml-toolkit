import numpy as np
import pytest

from siml.channel.prbs import generate_prbs


class TestGeneratePrbsOutput:
    def test_returns_ndarray(self) -> None:
        """반환값이 numpy ndarray여야 한다."""
        result = generate_prbs(order=7, length=10)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_length(self) -> None:
        """반환 배열의 shape이 요청한 length와 일치해야 한다."""
        result = generate_prbs(order=7, length=50)
        assert result.shape == (50,)

    def test_dtype_is_uint8(self) -> None:
        """반환 배열의 dtype이 np.uint8이어야 한다."""
        result = generate_prbs(order=7, length=10)
        assert result.dtype == np.uint8

    def test_values_are_only_zero_or_one(self) -> None:
        """비트열에 0과 1만 존재해야 한다."""
        result = generate_prbs(order=7, length=200)
        assert set(np.unique(result)).issubset({0, 1})

    def test_contains_both_zero_and_one(self) -> None:
        """충분한 길이에서 0과 1이 모두 나타나야 한다 (all-zero/all-one 시퀀스 방지)."""
        result = generate_prbs(order=7, length=127)
        assert 0 in result
        assert 1 in result


class TestGeneratePrbsPeriod:
    """PRBS 핵심 특성: 주기 = 2^order - 1."""

    @pytest.mark.parametrize("order", [7, 15, 23, 31])
    def test_sequence_repeats_after_one_period(self, order: int) -> None:
        """2주기 분량을 생성했을 때 앞뒤 주기가 정확히 일치해야 한다.

        PRBS의 정의상 maximal-length LFSR은 2^order - 1 비트 후 반드시 반복된다.
        이 테스트가 통과하면 탭 다항식과 LFSR 로직이 올바르게 구현된 것이다.
        """
        period = (1 << order) - 1  # 2^order - 1
        bits = generate_prbs(order=order, length=2 * period)
        np.testing.assert_array_equal(bits[:period], bits[period:])


class TestGeneratePrbsSeed:
    def test_same_seed_gives_same_sequence(self) -> None:
        """같은 seed로 두 번 호출하면 동일한 시퀀스가 나와야 한다."""
        a = generate_prbs(order=7, length=50, seed=42)
        b = generate_prbs(order=7, length=50, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_sequence(self) -> None:
        """다른 seed는 (일반적으로) 다른 시퀀스를 만들어야 한다."""
        a = generate_prbs(order=7, length=50, seed=1)
        b = generate_prbs(order=7, length=50, seed=2)
        assert not np.array_equal(a, b)

    def test_default_seed_is_all_ones(self) -> None:
        """seed 미전달 시 all-ones seed와 동일한 결과여야 한다."""
        default = generate_prbs(order=7, length=50)
        explicit = generate_prbs(order=7, length=50, seed=0b1111111)
        np.testing.assert_array_equal(default, explicit)

    def test_length_longer_than_period_wraps_around(self) -> None:
        """length > 주기이면 자연스럽게 wrap around되어야 한다.

        period 비트와 2*period 비트를 각각 생성했을 때,
        앞 period 비트가 일치하면 wrap around가 정상 동작하는 것이다.
        """
        period = (1 << 7) - 1
        short = generate_prbs(order=7, length=period)
        long_ = generate_prbs(order=7, length=period + 20)
        np.testing.assert_array_equal(short, long_[:period])


class TestGeneratePrbsErrors:
    def test_raises_for_unsupported_order(self) -> None:
        """지원하지 않는 order를 넣으면 ValueError를 raise해야 한다."""
        with pytest.raises(ValueError, match="not supported"):
            generate_prbs(order=8, length=10)

    def test_error_message_lists_supported_orders(self) -> None:
        """에러 메시지에 지원 목록이 포함되어야 한다."""
        with pytest.raises(ValueError, match=r"\[7, 15, 23, 31\]"):
            generate_prbs(order=99, length=10)

    def test_raises_for_zero_length(self) -> None:
        """length=0이면 ValueError를 raise해야 한다."""
        with pytest.raises(ValueError, match="length must be >= 1"):
            generate_prbs(order=7, length=0)

    def test_raises_for_negative_length(self) -> None:
        """length가 음수면 ValueError를 raise해야 한다."""
        with pytest.raises(ValueError, match="length must be >= 1"):
            generate_prbs(order=7, length=-1)

    def test_raises_for_zero_seed(self) -> None:
        """seed=0이면 ValueError를 raise해야 한다."""
        with pytest.raises(ValueError, match="seed=0 is invalid"):
            generate_prbs(order=7, length=10, seed=0)

    def test_raises_for_seed_above_max(self) -> None:
        """seed가 2^order - 1을 초과하면 ValueError를 raise해야 한다."""
        with pytest.raises(ValueError, match="out of range"):
            generate_prbs(order=7, length=10, seed=128)  # max for order=7 is 127

    def test_raises_for_seed_below_min(self) -> None:
        """seed가 0 미만이면 ValueError를 raise해야 한다."""
        with pytest.raises(ValueError, match="out of range"):
            generate_prbs(order=7, length=10, seed=-1)
