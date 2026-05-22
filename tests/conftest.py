from pathlib import Path

import pytest


# PROJECT_ROOT를 기준으로 경로를 잡아두면 어디서 pytest를 실행해도 깨지지 않는다
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def s2p_path() -> Path:
    """Return the Path to the sample ring_slot.s2p file."""
    return PROJECT_ROOT / "data" / "raw" / "ring_slot.s2p"
