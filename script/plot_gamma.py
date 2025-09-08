import argparse
import sys
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from helper import get_gamma_data, save_or_show_plot
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def main():
    parser = setup_parser()

    args = parser.parse_args()

    FILE_PATH: Final[str] = args.file_path
    OUTPUT: Final[str | None] = args.output

    print(f"Reading gamma data from {FILE_PATH}...")

    try:
        time_vec, gamma_qtm, _ = get_gamma_data(FILE_PATH)
    except Exception as e:
        sys.exit(f"Error occured while reading gamma data: {e}")

    print("Gamma data loaded successfully.")
    print("Plotting gamma function...")

    plt.figure(figsize=(8, 6), layout="constrained")

    plt.plot(time_vec, np.abs(gamma_qtm), label="Magnitude")
    plt.plot(time_vec, np.real(gamma_qtm), label="Real Part", linestyle=":")
    plt.plot(time_vec, np.imag(gamma_qtm), label="Imaginary Part", linestyle="--")
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel(r"Width Function $\Gamma(t)$ ($\text{Å}^2$)", fontsize=14)
    plt.grid()
    plt.legend(loc="center right", bbox_to_anchor=(0, 0, 1, 0.3))

    parent_axes = plt.gca()
    axins: Axes = inset_axes(parent_axes, width="40%", height="40%", loc="upper center")
    axins.plot(time_vec, np.abs(gamma_qtm), label="Magnitude")
    axins.plot(time_vec, np.real(gamma_qtm), label="Real Part", linestyle=":")
    axins.plot(time_vec, np.imag(gamma_qtm), label="Imaginary Part", linestyle="--")
    axins.set_xlim(-1e-13, 1e-13)
    axins.set_ylim(-0.05, 0.1)
    axins.set_xticks([-1e-13, 0, 1e-13])
    axins.set_yticks([-0.05, 0, 0.1])
    axins.grid()

    parent_axes.indicate_inset_zoom(axins, edgecolor="black")

    print("Plotting complete successfully.")

    save_or_show_plot(OUTPUT)

    print("Program complete successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot gamma function data from an HDF5 file.")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the gamma function data HDF5 file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="fig/gamma_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/gamma_plot.png).",
    )
    return parser


if __name__ == "__main__":
    main()
