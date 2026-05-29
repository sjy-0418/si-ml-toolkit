import numpy as np
import pytest

from siml.channel.waveform import bits_to_waveform


class TestBitsToWaveformOutput:
    def test_returns_ndarray(self) -> None:
        """반환값이 numpy ndarray여야 한다."""
        bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        result = bits_to_waveform(bits)
        assert isinstance(result, np.ndarray)

    def test_shape_is_bits_times_samples_per_bit(self) -> None:
        """반환 shape이 (len(bits) * samples_per_bit,)여야 한다."""
        bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=16)
        assert result.shape == (4 * 16,)

    def test_dtype_is_float32(self) -> None:
        """반환 dtype이 np.float32여야 한다."""
        bits = np.array([0, 1], dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=8, rise_time_ui=0.3)
        assert result.dtype == np.float32


class TestBitsToWaveformNrzMapping:
    def test_all_zeros_map_to_minus_one(self) -> None:
        """모든 비트가 0이면 출력 전체가 -1.0이어야 한다."""
        bits = np.zeros(8, dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=4, rise_time_ui=0.0)
        np.testing.assert_array_equal(result, np.full(32, -1.0, dtype=np.float32))

    def test_all_ones_map_to_plus_one(self) -> None:
        """모든 비트가 1이면 출력 전체가 +1.0이어야 한다."""
        bits = np.ones(8, dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=4, rise_time_ui=0.0)
        np.testing.assert_array_equal(result, np.full(32, 1.0, dtype=np.float32))

    def test_each_bit_fills_exactly_samples_per_bit_slots(self) -> None:
        """각 비트가 정확히 samples_per_bit개의 슬롯을 채워야 한다.

        [0, 1, 0]이고 samples_per_bit=4이면:
        - 샘플 0-3: -1.0 (비트 0)
        - 샘플 4-7: +1.0 (비트 1)
        - 샘플 8-11: -1.0 (비트 2)
        """
        bits = np.array([0, 1, 0], dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=4, rise_time_ui=0.0)
        np.testing.assert_array_equal(result[0:4], [-1.0, -1.0, -1.0, -1.0])
        np.testing.assert_array_equal(result[4:8], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result[8:12], [-1.0, -1.0, -1.0, -1.0])


class TestBitsToWaveformRiseTime:
    def test_zero_rise_time_returns_exact_square_wave(self) -> None:
        """rise_time_ui=0이면 Gaussian 필터 없이 정확한 ±1.0 사각파를 반환해야 한다."""
        bits = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=10, rise_time_ui=0.0)
        unique_values = np.unique(result)
        np.testing.assert_array_equal(unique_values, [-1.0, 1.0])

    def test_nonzero_rise_time_creates_intermediate_values_at_transition(self) -> None:
        """rise_time_ui > 0이면 천이 구간에 ±1.0이 아닌 중간 값이 생겨야 한다.

        Gaussian 필터가 실제로 동작했는지 확인하는 가장 직접적인 검증.
        """
        bits = np.array([0] * 5 + [1] * 5, dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=20, rise_time_ui=0.3)
        # 천이 직전/직후 샘플은 -1.0과 +1.0 사이의 중간 값이어야 한다
        transition_idx = 5 * 20  # 비트 5 시작점 (첫 번째 +1 비트)
        assert -1.0 < result[transition_idx - 1] < 0.0
        assert 0.0 < result[transition_idx] < 1.0

    def test_filtered_values_stay_within_minus_one_to_plus_one(self) -> None:
        """Gaussian 필터 후 값이 [-1, +1] 범위를 벗어나지 않아야 한다.

        Gaussian은 입력값의 가중 평균이므로 범위를 확대시키지 않는다.
        """
        bits = np.array([0, 1] * 20, dtype=np.uint8)
        result = bits_to_waveform(bits, samples_per_bit=32, rise_time_ui=0.3)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_transition_midpoint_is_near_zero(self) -> None:
        """천이 중점(n-1번과 n번 샘플의 평균)이 0에 가까워야 한다.

        Gaussian은 대칭 kernel이므로 step 함수 천이 중점에서
        w[n-1] + w[n] ≈ 0이 수학적으로 성립한다.
        양쪽에 flat 구간이 충분하면 경계 효과가 무시될 만큼 작아진다.
        """
        bits = np.array([0] * 10 + [1] * 10, dtype=np.uint8)
        spb = 100
        result = bits_to_waveform(bits, samples_per_bit=spb, rise_time_ui=0.3)
        transition_idx = 10 * spb  # 첫 번째 +1 비트의 시작 샘플
        midpoint = (result[transition_idx - 1] + result[transition_idx]) / 2
        assert abs(midpoint) < 0.05


class TestBitsToWaveformErrors:
    def test_raises_for_2d_bits(self) -> None:
        """bits가 2D 배열이면 ValueError를 raise해야 한다."""
        bits_2d = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        with pytest.raises(ValueError, match="1-D"):
            bits_to_waveform(bits_2d)

    def test_raises_for_invalid_bit_values(self) -> None:
        """bits에 0/1 외 값이 있으면 ValueError를 raise해야 한다."""
        bits = np.array([0, 1, 2], dtype=np.uint8)
        with pytest.raises(ValueError, match="only 0 and 1"):
            bits_to_waveform(bits)

    def test_raises_for_zero_samples_per_bit(self) -> None:
        """samples_per_bit=0이면 ValueError를 raise해야 한다."""
        bits = np.array([0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="samples_per_bit must be >= 1"):
            bits_to_waveform(bits, samples_per_bit=0)

    def test_raises_for_negative_samples_per_bit(self) -> None:
        """samples_per_bit가 음수면 ValueError를 raise해야 한다."""
        bits = np.array([0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="samples_per_bit must be >= 1"):
            bits_to_waveform(bits, samples_per_bit=-4)

    def test_raises_for_negative_rise_time(self) -> None:
        """rise_time_ui가 음수면 ValueError를 raise해야 한다."""
        bits = np.array([0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="rise_time_ui must be >= 0"):
            bits_to_waveform(bits, rise_time_ui=-0.1)
