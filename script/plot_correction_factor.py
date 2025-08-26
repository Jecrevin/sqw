import argparse
import sys
from typing import Final, Literal

import numpy as np
import scipy.constants as consts
from helper import get_gamma_data, save_or_show_plot
from matplotlib import pyplot as plt


def main() -> None:
    args = _parse_args()

    Q_VALUE: Final[float] = args.q
    ELEMENT: Final[Literal["H", "O"]] = args.element
    FILE_PATH: Final[str] = args.file_path.format(element=ELEMENT)
    OUTPUT: Final[str | None] = args.output

    print(f"Loading gamma data from {FILE_PATH}...")

    try:
        time_vec, gamma_qtm, gamma_cls = get_gamma_data(ELEMENT, FILE_PATH, include_classical=True)
        if gamma_cls is None:
            raise ValueError("The parameter `include_classical` must be set to True when calling `get_gamma_data`!")
    except Exception as e:
        sys.exit(f"Error occured while reading gamma data: {e}")

    print("Gamma data loaded successfully!")
    print(f"Calculating correction factor at Q={Q_VALUE} Angstrom^-1...")

    qtm_correction_factor = np.exp(-(Q_VALUE**2) / 2 * (gamma_qtm - gamma_cls))

    plt.figure(figsize=(8, 6), layout="constrained")
    plt.plot(time_vec / consts.pico, np.abs(qtm_correction_factor))
    plt.title(f"Correction Factor $R(Q, t)$ at $Q={Q_VALUE}$ Angstrom$^{{-1}}$")
    plt.xlabel("Time (ps)")
    plt.xscale("log")
    plt.grid()

    print("Plotting completed!")

    save_or_show_plot(OUTPUT)

    print("Program finished successfully!")


def _parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(
        description="Plot the quantum correction factor for the Classical SISF (Self-Intermediate Scattering Function)."
    )

    argparser.add_argument(
        "q",
        type=float,
        help="Q value in Angstrom^-1 at which to plot the correction factor.",
    )
    argparser.add_argument(
        "-e",
        "--element",
        type=str,
        choices=["H", "O"],
        default="H",
        help="Element symbol for which to plot the correction factor (default: H).",
    )
    argparser.add_argument(
        "-f",
        "--file-path",
        type=str,
        default="data/last_{element}.gamma",
        help="format string for the path to the hdf5 file containing the gamma data "
        "(default: data/last_{element}.gamma).",
    )
    argparser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="fig/correction_factor_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/correction_factor_plot.png).",
    )

    return argparser.parse_args()


if __name__ == "__main__":
    main()
