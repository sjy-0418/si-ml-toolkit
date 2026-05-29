import matplotlib
matplotlib.use("Agg")  # enable rendering in headless environments (including CI)

import matplotlib.figure
import pytest
import skrf

from siml.viz import plot_s_db


@pytest.fixture
def network(s2p_path):
    return skrf.Network(str(s2p_path))


class TestPlotSDb:
    def test_returns_figure(self, network):
        fig = plot_s_db(network)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_default_plots_all_four_params(self, network):
        """Default for a 2-port network should plot S11, S12, S21, S22 — 4 lines."""
        fig = plot_s_db(network)
        labels = [line.get_label() for line in fig.axes[0].get_lines()]
        assert sorted(labels) == ["S11", "S12", "S21", "S22"]

    def test_custom_parameters(self, network):
        fig = plot_s_db(network, parameters=["s11", "s21"])
        labels = [line.get_label() for line in fig.axes[0].get_lines()]
        assert sorted(labels) == ["S11", "S21"]

    def test_title_defaults_to_network_name(self, network):
        fig = plot_s_db(network)
        assert fig.axes[0].get_title() == network.name

    def test_custom_title(self, network):
        fig = plot_s_db(network, title="My Plot")
        assert fig.axes[0].get_title() == "My Plot"

    def test_save_path(self, network, tmp_path):
        out = tmp_path / "output.png"
        plot_s_db(network, save_path=out)
        assert out.exists() and out.stat().st_size > 0

    def test_raises_for_invalid_parameter_name(self, network):
        with pytest.raises(ValueError, match="Invalid S-parameter"):
            plot_s_db(network, parameters=["sx1"])

    def test_raises_for_out_of_range_port(self, network):
        """s33 does not exist on a 2-port network."""
        with pytest.raises(ValueError, match="port 3"):
            plot_s_db(network, parameters=["s33"])
