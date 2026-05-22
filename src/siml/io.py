from pathlib import Path

import skrf


def load_s2p(path: str | Path) -> skrf.Network:
    """Load a Touchstone .s2p file and return a scikit-rf Network object.

    Args:
        path: Path to the .s2p file. Accepts both str and Path.

    Returns:
        A skrf.Network object containing S-parameters, frequency, and port info.

    Raises:
        ValueError: If the file extension is not .s2p.
        FileNotFoundError: If the file does not exist at the given path.
    """
    # Work with Path internally so we get .suffix and .exists() for free
    path = Path(path)

    # Reject unsupported formats early — skrf can open other Touchstone types
    # (.s4p, .s1p, etc.) but this function is intentionally scoped to 2-port files
    if path.suffix.lower() != ".s2p":
        raise ValueError(
            f"Expected a .s2p file, got '{path.suffix}' (path: {path})"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"S2P file not found: {path.resolve()}"
        )

    # skrf.Network accepts a file path string directly and parses the
    # Touchstone format (frequency, S-parameters, port impedance) automatically
    return skrf.Network(str(path))
