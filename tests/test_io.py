from pathlib import Path

import pytest
import skrf

from siml.io import load_s2p


class TestLoadS2p:
    def test_returns_network_object(self, s2p_path: Path) -> None:
        """Loading a valid file should return an skrf.Network object."""
        net = load_s2p(s2p_path)
        assert isinstance(net, skrf.Network)

    def test_network_has_two_ports(self, s2p_path: Path) -> None:
        """A 2-port file should have nports == 2."""
        net = load_s2p(s2p_path)
        assert net.nports == 2

    def test_network_has_frequency_data(self, s2p_path: Path) -> None:
        """The network should contain at least one frequency point."""
        net = load_s2p(s2p_path)
        assert len(net.f) > 0

    def test_accepts_string_path(self, s2p_path: Path) -> None:
        """A str path should behave identically to a Path object."""
        net = load_s2p(str(s2p_path))
        assert isinstance(net, skrf.Network)

    def test_raises_for_wrong_extension(self, tmp_path: Path) -> None:
        """A file without the .s2p extension should raise ValueError."""
        wrong_file = tmp_path / "data.s4p"
        wrong_file.touch()  # create an empty file just to make the path exist

        with pytest.raises(ValueError, match=r"\.s2p"):
            load_s2p(wrong_file)

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """A path that does not exist should raise FileNotFoundError."""
        missing = tmp_path / "nonexistent.s2p"

        with pytest.raises(FileNotFoundError, match="nonexistent.s2p"):
            load_s2p(missing)
