"""Plot S(Q, omega) results for Gaussian approximation model at Q=1.0 Å⁻¹.

This script reads precomputed S(Q, omega) data from a CSV file and generates
plots for both FFT and CDFT results, saving them as PNG files and displaying
them interactively.
"""

from pathlib import Path

import numpy as np
from helper import plt_style_setup
from matplotlib import pyplot as plt

from sqw.consts import HBAR


def main() -> None:
    """Plot S(Q, omega) results for Gaussian approximation model at Q=1.0 Å⁻¹."""
    data_file = Path(__file__).parents[2] / "data" / "results" / "sqw_ga_fft_cdft.csv"
    fig_dir = Path(__file__).parents[2] / "figs"

    omega, fft_res, cdft_res = np.loadtxt(data_file, delimiter=",", unpack=True)

    plt_style_setup()

    fig1, ax1 = plt.subplots()
    ax1.plot(omega * HBAR, fft_res)
    ax1.set_yscale("symlog", linthresh=1e-27)
    ax1.set_xlabel("Energy Transfer (eV)")
    ax1.set_ylabel("Scattering Function (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)")
    fig1.tight_layout()
    fig1.savefig(fig_dir / "sqw_ga_q1_fft.png", dpi=300)

    fig2, ax2 = plt.subplots()
    ax2.plot(omega * HBAR, cdft_res)
    ax2.set_yscale("log")
    ax2.set_xlabel("Energy Transfer (eV)")
    ax2.set_ylabel("Scattering Function (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "sqw_ga_q1_cdft.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
