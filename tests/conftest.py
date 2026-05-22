from pathlib import Path

import pytest


@pytest.fixture
def s2p_path() -> Path:
    """Return the Path to the sample ring_slot.s2p file."""
    return Path(__file__).parent / "fixtures" / "ring_slot.s2p"
