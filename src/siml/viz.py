import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.figure
import skrf


def _parse_parameter(param: str, nports: int) -> tuple[int, int]:
    """Parse an S-parameter string into zero-based row/col indices.

    Args:
        param: S-parameter name, e.g. 's21' or 'S21'.
        nports: Number of ports in the network.

    Returns:
        (row, col) as zero-based indices into the s_db array.

    Raises:
        ValueError: If the format is invalid or indices exceed nports.
    """
    match = re.fullmatch(r"[sS](\d)(\d)", param)
    if match is None:
        raise ValueError(
            f"Invalid S-parameter name '{param}'. Expected format like 's11', 's21'."
        )

    # Touchstone uses 1-based port numbers; convert to 0-based array indices
    row = int(match.group(1)) - 1
    col = int(match.group(2)) - 1

    if row >= nports or col >= nports:
        raise ValueError(
            f"'{param}' references port {max(row, col) + 1}, "
            f"but the network only has {nports} port(s)."
        )

    return row, col


def plot_s_db(
    network: skrf.Network,
    parameters: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot S-parameters in dB scale.

    Args:
        network: A scikit-rf Network object.
        parameters: S-parameter names to plot, e.g. ['s11', 's21'].
            Defaults to all combinations for the network's port count.
        save_path: If given, save the figure to this path.
        title: Figure title. Defaults to the network's name.

    Returns:
        The matplotlib Figure object.

    Raises:
        ValueError: If any parameter name is invalid or out of range.
    """
    if parameters is None:
        n = network.nports
        parameters = [f"s{i+1}{j+1}" for i in range(n) for j in range(n)]

    # Validate all parameters before drawing anything
    indices = [_parse_parameter(p, network.nports) for p in parameters]

    fig, ax = plt.subplots()

    freq = network.frequency.f_scaled        # e.g. [75.0, 75.5, ...] in GHz
    freq_unit = network.frequency.unit       # e.g. 'GHz'

    for param, (row, col) in zip(parameters, indices):
        ax.plot(freq, network.s_db[:, row, col], label=param.upper())

    ax.set_xlabel(f"Frequency ({freq_unit})")
    ax.set_ylabel("|S| (dB)")
    ax.set_title(title if title is not None else network.name)
    ax.legend()
    ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path)

    return fig
