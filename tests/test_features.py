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
        """Must return a float when given an exact measured frequency point."""
        result = insertion_loss_db(network, network.f[0])
        assert isinstance(result, float)

    def test_interpolated_value_is_within_neighbor_bounds(
        self, network: skrf.Network
    ) -> None:
        """When given a frequency between measured points, result must lie between the two neighboring dB values.

        Because interpolation is done in the linear domain before dB conversion, the result
        must always fall between the dB values of the two adjacent points.
        """
        f_mid = (network.f[0] + network.f[1]) / 2.0
        result = insertion_loss_db(network, f_mid)

        il_at_f0 = insertion_loss_db(network, network.f[0])
        il_at_f1 = insertion_loss_db(network, network.f[1])

        assert min(il_at_f0, il_at_f1) <= result <= max(il_at_f0, il_at_f1)

    def test_raises_for_frequency_above_range(self, network: skrf.Network) -> None:
        """Must raise ValueError for a frequency above the measured range."""
        out_of_range = network.f[-1] + 1e9

        with pytest.raises(ValueError, match="outside the measured range"):
            insertion_loss_db(network, out_of_range)

    def test_raises_for_frequency_below_range(self, network: skrf.Network) -> None:
        """Must raise ValueError for a frequency below the measured range."""
        out_of_range = network.f[0] - 1e9

        with pytest.raises(ValueError, match="outside the measured range"):
            insertion_loss_db(network, out_of_range)

    def test_raises_for_single_port_network(self) -> None:
        """Must raise ValueError for a single-port network."""
        # construct skrf.Network directly: 1 port, 3 frequency points, S11 only
        freq = skrf.Frequency(1, 10, 3, unit="ghz")
        one_port = skrf.Network(frequency=freq, s=np.ones((3, 1, 1)))

        with pytest.raises(ValueError, match="at least 2 ports"):
            insertion_loss_db(one_port, 5e9)

    def test_return_value_is_negative_for_passive_network(
        self, network: skrf.Network
    ) -> None:
        """Insertion loss of a passive network must always be below 0 dB.

        A passive device does not generate energy, so |S21| <= 1, meaning IL <= 0 dB.
        """
        result = insertion_loss_db(network, network.f[0])
        assert result <= 0.0


class TestReturnLossDb:
    def test_returns_float_at_exact_frequency(self, network: skrf.Network) -> None:
        """Must return a float when given an exact measured frequency point."""
        result = return_loss_db(network, network.f[0])
        assert isinstance(result, float)

    def test_interpolated_value_is_within_neighbor_bounds(
        self, network: skrf.Network
    ) -> None:
        """When given a frequency between measured points, result must lie between the two neighboring dB values.

        Same reason as insertion_loss_db: interpolation in the linear domain before dB conversion means
        the result must fall between the dB values of the two adjacent points.
        """
        f_mid = (network.f[0] + network.f[1]) / 2.0
        result = return_loss_db(network, f_mid)

        rl_at_f0 = return_loss_db(network, network.f[0])
        rl_at_f1 = return_loss_db(network, network.f[1])

        assert min(rl_at_f0, rl_at_f1) <= result <= max(rl_at_f0, rl_at_f1)

    def test_raises_for_frequency_above_range(self, network: skrf.Network) -> None:
        """Must raise ValueError for a frequency above the measured range."""
        out_of_range = network.f[-1] + 1e9

        with pytest.raises(ValueError, match="outside the measured range"):
            return_loss_db(network, out_of_range)

    def test_raises_for_frequency_below_range(self, network: skrf.Network) -> None:
        """Must raise ValueError for a frequency below the measured range."""
        out_of_range = network.f[0] - 1e9

        with pytest.raises(ValueError, match="outside the measured range"):
            return_loss_db(network, out_of_range)

    def test_works_on_one_port_network(self) -> None:
        """Must work on a single-port network because S11 exists for 1-port.

        Unlike insertion_loss_db, there is no nports check, so a 1-port network
        must return a result without raising ValueError.
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
        """Return loss of a passive network must always be 0 dB or below.

        A passive device does not generate energy, so |S11| <= 1, meaning RL <= 0 dB.
        """
        result = return_loss_db(network, network.f[0])
        assert result <= 0.0
