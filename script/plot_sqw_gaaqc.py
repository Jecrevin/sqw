import argparse
import sys
from functools import partial
from typing import Final, Literal

import numpy as np
from helper import (
    get_gamma_data,
    get_sqw_molecular_dynamics_data,
    parse_indices,
    reorder_legend_by_row,
    save_or_show_plot,
)
from matplotlib import pyplot as plt

from h2o_sqw_calc.core import HBAR, sqw_cdft, sqw_gaaqc
from h2o_sqw_calc.typing import Array1D


def main() -> None:
    parser = setup_parser()

    args = parser.parse_args()

    GAMMA_FILE_PATH: Final[str] = args.gamma_file_path
    MD_FILE_PATH: Final[str] = args.md_file_path
    ELEMENT: Final[Literal["H", "O"]] = args.elelement
    USE_ENERGY_UNIT: Final[bool] = args.energy_unit
    SCALE: Final[str] = args.scale
    PLOT_QTM_RESULTS: Final[bool] = args.qtm_results
    OUTPUT: Final[str | None] = args.output
    try:
        INDICES: Final[Array1D[np.int_]] = parse_indices(args.indices)
    except ValueError as e:
        parser.error(f"Error parsing indices: {e}")

    print(f"Loading gamma data for element '{ELEMENT}' from '{GAMMA_FILE_PATH}'...")

    try:
        time_vec, gamma_qtm, gamma_cls = get_gamma_data(GAMMA_FILE_PATH, include_classical=True)
    except Exception as e:
        sys.exit(f"Error loading gamma data: {e}")

    assert gamma_cls is not None, "`include_classical` must be True to get classical gamma data."

    print("Gamma data loaded successfully.")
    print(f"Loading MD simulation data from file '{MD_FILE_PATH}'...")

    try:
        q_vals, omega_md, sqw_md_vstack = get_sqw_molecular_dynamics_data(MD_FILE_PATH, ELEMENT)
    except Exception as e:
        sys.exit(f"Error loading MD data: {e}")

    print("MD data loaded successfully.")
    print("Calculating S(q,w) using GAAQC model...")

    q_vals = q_vals[INDICES]
    sqw_md_vstack = sqw_md_vstack[INDICES]

    sqw_gaaqc_results = map(
        lambda q, sqw_md: sqw_gaaqc(q, time_vec, gamma_qtm, gamma_cls, omega_md, sqw_md), q_vals, sqw_md_vstack
    )

    print("Calculation completed.")
    if PLOT_QTM_RESULTS:
        print("Calculating S(q,w) QTM results from CDFT for comparison...")

    sqw_qtm_results = map(partial(sqw_cdft, time_vec=time_vec, gamma=gamma_qtm), q_vals) if PLOT_QTM_RESULTS else None

    if PLOT_QTM_RESULTS:
        print("Calculation completed.")
    print("Plotting results...")

    plt.figure(figsize=(10, 6), layout="constrained")
    if sqw_qtm_results is not None:
        for q, (omega_gaaqc, sqw_gaaqc_data), (omega_qtm, sqw_qtm_data) in zip(
            q_vals, sqw_gaaqc_results, sqw_qtm_results, strict=True
        ):
            x_gaaqc = omega_gaaqc * HBAR if USE_ENERGY_UNIT else omega_gaaqc
            x_qtm = omega_qtm * HBAR if USE_ENERGY_UNIT else omega_qtm
            (line,) = plt.plot(x_gaaqc, np.abs(sqw_gaaqc_data), label="GAAQC")
            plt.plot(x_qtm, np.abs(sqw_qtm_data), label=f"GA-QTM    for {q = :.2f}", color=line.get_color(), ls="--")
        plt.legend(*reorder_legend_by_row(*plt.gca().get_legend_handles_labels(), ncol=2), ncol=2, loc="upper left")
    else:
        for q, (omega, sqw) in zip(q_vals, sqw_gaaqc_results, strict=True):
            plt.plot(omega * HBAR if USE_ENERGY_UNIT else omega, np.abs(sqw), label=f"{q = :.2f}")
        plt.legend()
    plt.yscale(SCALE)
    plt.xlabel("Energy (eV)" if USE_ENERGY_UNIT else r"Angular Frequency $\omega$ (rad/s)", fontsize=14)
    plt.ylabel(r"Scattering Function $S(q,\omega)$ (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)", fontsize=14)
    plt.grid()

    print("Plotting completed.")

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Scattering Function S(q,w) Calculated from Gaussian "
        "Approximation Assisted Quantum Correction (GAAQC) model."
    )
    parser.add_argument("gamma_file_path", type=str, help="File path to the HDF5 file containing the gamma data.")
    parser.add_argument(
        "md_file_path",
        type=str,
        help="File path to the HDF5 file containing the MD simulation data.",
    )
    parser.add_argument(
        "indices",
        type=str,
        nargs="+",
        help="Indices of MD simulation data to plot GAAQC from, formatted in START[:END[:STEP]]",
    )
    parser.add_argument(
        "-e",
        "--elelement",
        type=str,
        default="H",
        choices=["H", "O"],
        help="Element type to plot GAAQC for. Default is 'H'.",
    )
    parser.add_argument(
        "--energy-unit",
        action="store_true",
        help="Use energy unit (eV) for x-axis instead of angular frequency (rad/s).",
    )
    parser.add_argument(
        "--scale",
        choices=["linear", "log"],
        default="linear",
        help="Scale for the y-axis of the plot (default: linear).",
    )
    parser.add_argument(
        "--qtm-results",
        action="store_true",
        help="If set, plot S(q,w) QTM results from CDFT calculations for comparison.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        nargs="?",
        const="data/plot_sqw_gaaqc.png",
        help="Output file name to save the plot. If not provided, defaults to 'data/plot_sqw_gaaqc.png'.",
    )
    return parser


if __name__ == "__main__":
    main()
