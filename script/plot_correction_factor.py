import argparse
import sys
from typing import Final

import numpy as np
import scipy.constants as consts
from helper import get_gamma_data, parse_q_values, save_or_show_plot
from matplotlib import pyplot as plt

from h2o_sqw_calc.typing import Array1D


def main() -> None:
    parser = setup_parser()

    args = parser.parse_args()

    GAMMA_FILE_PATH: Final[str] = args.gamma_file_path
    OUTPUT: Final[str | None] = args.output
    try:
        Q_VALUES: Final[Array1D[np.float64]] = parse_q_values(args.q_values)
    except ValueError as e:
        parser.error(f"Error parsing Q values: {e}")

    print(f"Loading gamma data from {GAMMA_FILE_PATH}...")

    try:
        time_vec, gamma_qtm, gamma_cls = get_gamma_data(GAMMA_FILE_PATH, include_classical=True)
    except Exception as e:
        sys.exit(f"Error occured while reading gamma data: {e}")

    assert gamma_cls is not None, "`include_classical` must be True to get classical gamma data."

    print("Gamma data loaded successfully.")
    print("Calculating correction factors...")

    qtm_correction_factors = map(lambda q: np.exp(-0.5 * q**2 * (gamma_qtm - gamma_cls)), Q_VALUES)

    plt.figure(figsize=(8, 6), layout="constrained")
    for q, qtm_cf in zip(Q_VALUES, qtm_correction_factors, strict=True):
        plt.semilogx(time_vec / consts.pico, np.abs(qtm_cf), label=f"{q = :.2f}", marker=",")
    plt.xlabel("Time (ps)", fontsize=14)
    plt.ylabel("Absolute Value of Quantum Correction Factor", fontsize=14)
    plt.legend()
    plt.grid()

    print("Plotting completed!")

    save_or_show_plot(OUTPUT)

    print("Program finished successfully.")


def setup_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(
        description="Plot the quantum correction factor for the Molecular "
        "Dynamics SISF (Self-Intermediate Scattering Function) results."
    )
    argparser.add_argument(
        "gamma_file_path",
        type=str,
        help="File path to the HDF5 file containing the gamma data.",
    )
    argparser.add_argument(
        "q_values",
        type=str,
        nargs="+",
        help="Momentum Transfer values (unit: angstrom^-1), format in START[:END[:STEP]].",
    )
    argparser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="fig/correction_factor_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/correction_factor_plot.png).",
    )
    return argparser


if __name__ == "__main__":
    main()
