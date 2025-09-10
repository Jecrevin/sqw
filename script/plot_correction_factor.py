import argparse
import sys
from functools import partial
from typing import Final

import numpy as np
import scipy.constants as consts
from helper import get_gamma_data, parse_q_values, save_or_show_plot
from matplotlib import pyplot as plt

from h2o_sqw_calc.core import HBAR, sqw_qtm_correction_factor
from h2o_sqw_calc.typing import Array1D


def main() -> None:
    parser = setup_parser()

    args = parser.parse_args()

    if not args.sqw:
        if args.scale != "linear":
            print("Warning: --scale option is only applicable when --sqw is set. Ignoring --scale option.")
        if args.energy_unit:
            print("Warning: --energy-unit option is only applicable when --sqw is set. Ignoring --energy-unit option.")

    GAMMA_FILE_PATH: Final[str] = args.gamma_file_path
    PLOT_SQW: Final[bool] = args.sqw
    SCALE: Final[str] = args.scale
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

    get_qc = (
        partial(sqw_qtm_correction_factor, time_vec=time_vec, gamma_qtm=gamma_qtm, gamma_cls=gamma_cls)
        if PLOT_SQW
        else lambda q: np.exp(-0.5 * q**2 * (gamma_qtm - gamma_cls))
    )
    qc_results = map(get_qc, Q_VALUES)

    plt.figure(figsize=(8, 6), layout="constrained")
    if PLOT_SQW:
        for q, qc_res in zip(Q_VALUES, qc_results, strict=True):
            plt.plot(qc_res[0] * HBAR, np.abs(qc_res[1]), label=f"{q = :.2f}")
        plt.yscale(SCALE)
        plt.xlabel(r"Energy (eV)", fontsize=14)
    else:
        for q, qc_data in zip(Q_VALUES, qc_results, strict=True):
            plt.semilogx(time_vec / consts.pico, np.abs(qc_data), label=f"{q = :.2f}")
        plt.xlabel("Time (ps)", fontsize=14)
    plt.ylabel("Absolute Value of Quantum Correction Factor", fontsize=14)
    plt.legend()
    plt.grid()

    print("Plotting completed!")

    save_or_show_plot(OUTPUT)

    print("Program finished successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the quantum correction factor for the Molecular "
        "Dynamics SISF (Self-Intermediate Scattering Function) results."
    )
    parser.add_argument(
        "gamma_file_path",
        type=str,
        help="File path to the HDF5 file containing the gamma data.",
    )
    parser.add_argument(
        "q_values",
        type=str,
        nargs="+",
        help="Momentum Transfer values (unit: angstrom^-1), format in START[:END[:STEP]].",
    )
    parser.add_argument(
        "--sqw",
        action="store_true",
        help="Plot the correction factor for the SQW (Scattering Function) instead of SISF.",
    )
    parser.add_argument(
        "--energy-unit",
        action="store_true",
        help="Use energy unit (eV) for x-axis instead of angular frequency (rad/s).",
    )
    parser.add_argument(
        "--scale",
        type=str,
        choices=["linear", "log"],
        default="linear",
        help="y-axis scale for SQW plot (default: linear).",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="fig/correction_factor_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/correction_factor_plot.png).",
    )
    return parser


if __name__ == "__main__":
    main()
