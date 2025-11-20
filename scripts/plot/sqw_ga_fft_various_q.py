"""Plot S(Q, ω) GA model results from FFT for various Q values.

This script reads S(Q, ω) data from a CSV file and generates a plot for
different Q values. The resulting plot is saved as a PNG file in the 'figs'
directory.
"""

from pathlib import Path

import numpy as np
from helper import plt_style_setup
from matplotlib import pyplot as plt

from sqw.consts import HBAR


def main() -> None:
    """Plot S(Q, ω) GA model results from FFT for various Q values."""
    data_file = Path(__file__).parents[2] / "data" / "results" / "sqw_ga_fft_various_q.csv"
    fig_dir = Path(__file__).parents[2] / "figs"

    data = np.loadtxt(data_file, delimiter=",")

    q_vals = np.arange(10, 22, 2)  # unit: Å⁻¹
    omega = data[:, 0]
    sqw_ga_fft_various_q = data[:, 1:]

    plt_style_setup()

    fig, ax = plt.subplots()
    for i, q in enumerate(q_vals):
        ax.plot(omega * HBAR, sqw_ga_fft_various_q[:, i], label=f"Q={q} Å⁻¹")

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Scattering Function (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(fig_dir / "sqw_ga_fft_various_q.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
