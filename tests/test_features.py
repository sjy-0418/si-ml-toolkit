from pathlib import Path

import numpy as np
import pytest
import skrf

from siml.features import insertion_loss_db, return_loss_db


@pytest.fixture
def network(s2p_path: Path) -> skrf.Network:
    return skrf.Network(str(s2p_path))


class TestInsertionLossDb:
    def test_returns_float_at_exact_frequency(self, network: skrf.Network) -> None:
        """측정된 주파수 포인트를 그대로 넣으면 float을 반환해야 한다."""
        result = insertion_loss_db(network, network.f[0])
        assert isinstance(result, float)

    def test_interpolated_value_is_within_neighbor_bounds(
        self, network: skrf.Network
    ) -> None:
        """측정 포인트 사이 주파수를 넣으면 양쪽 이웃 dB 값 사이에 있어야 한다.

        보간을 linear 도메인에서 한 뒤 dB 변환하므로, 결과는 항상
        인접 두 포인트의 dB 값 사이에 위치해야 한다.
        """
        f_mid = (network.f[0] + network.f[1]) / 2.0
        result = insertion_loss_db(network, f_mid)

        il_at_f0 = insertion_loss_db(network, network.f[0])
        il_at_f1 = insertion_loss_db(network, network.f[1])

        assert min(il_at_f0, il_at_f1) <= result <= max(il_at_f0, il_at_f1)

    def test_raises_for_frequency_above_range(self, network: skrf.Network) -> None:
        """측정 범위 위의 주파수를 넣으면 ValueError를 raise해야 한다."""
        out_of_range = network.f[-1] + 1e9

        with pytest.raises(ValueError, match="outside the measured range"):
            insertion_loss_db(network, out_of_range)

    def test_raises_for_frequency_below_range(self, network: skrf.Network) -> None:
        """측정 범위 아래의 주파수를 넣으면 ValueError를 raise해야 한다."""
        out_of_range = network.f[0] - 1e9

        with pytest.raises(ValueError, match="outside the measured range"):
            insertion_loss_db(network, out_of_range)

    def test_raises_for_single_port_network(self) -> None:
        """1포트 network를 넣으면 ValueError를 raise해야 한다."""
        # skrf.Network를 직접 생성: 1포트, 주파수 3개, S11만 존재
        freq = skrf.Frequency(1, 10, 3, unit="ghz")
        one_port = skrf.Network(frequency=freq, s=np.ones((3, 1, 1)))

        with pytest.raises(ValueError, match="at least 2 ports"):
            insertion_loss_db(one_port, 5e9)

    def test_return_value_is_negative_for_passive_network(
        self, network: skrf.Network
    ) -> None:
        """수동 소자(passive network)의 insertion loss는 항상 0 dB 미만이어야 한다.

        수동 소자는 에너지를 생성하지 않으므로 |S21| <= 1, 즉 IL <= 0 dB.
        """
        result = insertion_loss_db(network, network.f[0])
        assert result <= 0.0


class TestReturnLossDb:
    def test_returns_float_at_exact_frequency(self, network: skrf.Network) -> None:
        """측정된 주파수 포인트를 그대로 넣으면 float을 반환해야 한다."""
        result = return_loss_db(network, network.f[0])
        assert isinstance(result, float)

    def test_interpolated_value_is_within_neighbor_bounds(
        self, network: skrf.Network
    ) -> None:
        """측정 포인트 사이 주파수를 넣으면 양쪽 이웃 dB 값 사이에 있어야 한다.

        insertion_loss_db와 동일한 이유: linear 도메인 보간 후 dB 변환이므로
        결과는 인접 두 포인트의 dB 값 사이에 위치해야 한다.
        """
        f_mid = (network.f[0] + network.f[1]) / 2.0
        result = return_loss_db(network, f_mid)

        rl_at_f0 = return_loss_db(network, network.f[0])
        rl_at_f1 = return_loss_db(network, network.f[1])

        assert min(rl_at_f0, rl_at_f1) <= result <= max(rl_at_f0, rl_at_f1)

    def test_raises_for_frequency_above_range(self, network: skrf.Network) -> None:
        """측정 범위 위의 주파수를 넣으면 ValueError를 raise해야 한다."""
        out_of_range = network.f[-1] + 1e9

        with pytest.raises(ValueError, match="outside the measured range"):
            return_loss_db(network, out_of_range)

    def test_raises_for_frequency_below_range(self, network: skrf.Network) -> None:
        """측정 범위 아래의 주파수를 넣으면 ValueError를 raise해야 한다."""
        out_of_range = network.f[0] - 1e9

        with pytest.raises(ValueError, match="outside the measured range"):
            return_loss_db(network, out_of_range)

    def test_works_on_one_port_network(self) -> None:
        """1포트 network도 S11이 존재하므로 정상 동작해야 한다.

        insertion_loss_db와 달리 nports 체크가 없으므로, 1포트에서도
        ValueError 없이 결과를 반환해야 한다.
        """
        freq = skrf.Frequency(1, 10, 3, unit="ghz")
        # |S11| = 0.5 → 20*log10(0.5) ≈ -6.02 dB
        s_data = np.full((3, 1, 1), 0.5 + 0j)
        one_port = skrf.Network(frequency=freq, s=s_data)

        result = return_loss_db(one_port, 5e9)

        assert isinstance(result, float)
        assert pytest.approx(result, abs=0.01) == 20.0 * np.log10(0.5)

    def test_return_value_is_negative_for_passive_network(
        self, network: skrf.Network
    ) -> None:
        """수동 소자(passive network)의 return loss는 항상 0 dB 이하여야 한다.

        수동 소자는 에너지를 생성하지 않으므로 |S11| <= 1, 즉 RL <= 0 dB.
        """
        result = return_loss_db(network, network.f[0])
        assert result <= 0.0
