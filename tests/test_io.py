from pathlib import Path

import pytest
import skrf

from siml.io import load_s2p


class TestLoadS2p:
    def test_returns_network_object(self, s2p_path: Path) -> None:
        """정상 파일을 로드하면 skrf.Network가 반환되어야 한다."""
        net = load_s2p(s2p_path)
        assert isinstance(net, skrf.Network)

    def test_network_has_two_ports(self, s2p_path: Path) -> None:
        """2포트 파일이므로 nports == 2 이어야 한다."""
        net = load_s2p(s2p_path)
        assert net.nports == 2

    def test_network_has_frequency_data(self, s2p_path: Path) -> None:
        """주파수 포인트가 1개 이상 존재해야 한다."""
        net = load_s2p(s2p_path)
        assert len(net.f) > 0

    def test_accepts_string_path(self, s2p_path: Path) -> None:
        """str 타입 경로도 Path와 동일하게 동작해야 한다."""
        net = load_s2p(str(s2p_path))
        assert isinstance(net, skrf.Network)

    def test_raises_for_wrong_extension(self, tmp_path: Path) -> None:
        """확장자가 .s2p가 아니면 ValueError를 raise해야 한다."""
        wrong_file = tmp_path / "data.s4p"
        wrong_file.touch()  # 파일 내용 없이 경로만 존재하게 생성

        with pytest.raises(ValueError, match=r"\.s2p"):
            load_s2p(wrong_file)

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """존재하지 않는 파일 경로면 FileNotFoundError를 raise해야 한다."""
        missing = tmp_path / "nonexistent.s2p"

        with pytest.raises(FileNotFoundError, match="nonexistent.s2p"):
            load_s2p(missing)
