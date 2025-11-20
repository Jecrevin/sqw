"""Plot S(Q, omega) comparison between FFT and STC methods at Q=4.0 Å⁻¹.

This script reads precomputed S(Q, omega) data from a JSON file and generates a
comparison plot for both FFT and STC results, saving it as a PNG file and
displaying it interactively.
"""

from pathlib import Path

import numpy as np
from helper import plt_style_setup
from matplotlib import pyplot as plt

from sqw.consts import HBAR


def main() -> None:
    """Plot S(Q, omega) comparison between FFT and STC methods at Q=4.0 Å⁻¹."""
    data_dir = Path(__file__).parents[2] / "data" / "results" / "fft_stc_comparison_q40"
    fig_dir = Path(__file__).parents[2] / "figs"
    fig_dir.mkdir(exist_ok=True)

    omega_fft, sqw_fft = np.loadtxt(data_dir / "sqw_fft_q40.csv", delimiter=",", unpack=True)
    omega_stc, sqw_stc = np.loadtxt(data_dir / "sqw_stc_q40.csv", delimiter=",", unpack=True)

    plt_style_setup()

    fig, ax = plt.subplots()
    ax.plot(omega_stc * HBAR, sqw_stc, label="STC model")
    ax.plot(omega_fft * HBAR, sqw_fft, label="FFT result")
    ax.set_xlabel("Energy Transfer (eV)")
    ax.set_ylabel("Scattering Function (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "sqw_q40_fft_stc.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
