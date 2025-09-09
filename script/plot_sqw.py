import argparse
import sys
from typing import Final, Literal

import numpy as np
from helper import (
    get_gamma_data,
    get_stc_model_data,
    parse_q_values,
    reorder_legend_by_row,
    save_or_show_plot,
)
from matplotlib import pyplot as plt

from h2o_sqw_calc.core import HBAR, sqw_cdft, sqw_stc_model
from h2o_sqw_calc.typing import Array1D


def main():
    parser = setup_parser()

    args = parser.parse_args()

    GAMMA_FILE_PATH: Final[str] = args.gamma_file_path
    STC_FILE_PATH: Final[str] = args.stc_file_path
    ELEMENT: Final[Literal["H", "O"]] = args.element
    TEMPERATURE: Final[float] = args.temperature  # unit: K
    SCALE: Final[Literal["linear", "log"]] = args.scale
    ENERGY_UNIT: Final[bool] = args.energy_unit
    OUTPUT: Final[str | None] = args.output
    try:
        Q_VALUES: Final[Array1D[np.float64]] = parse_q_values(args.q_values)
    except ValueError as e:
        parser.error(f"Error parsing q_values: {e}")

    print(f"Reading gamma data for element '{ELEMENT}' from {GAMMA_FILE_PATH}...")

    try:
        time_vec, gamma_qtm, _ = get_gamma_data(GAMMA_FILE_PATH)
    except Exception as e:
        sys.exit(f"Error occured while reading gamma data: {e}")

    print("Gamma data loaded successfully.")
    print(f"Reading STC model data from {STC_FILE_PATH}...")

    try:
        freq_dos, dos = get_stc_model_data(STC_FILE_PATH, element=ELEMENT)
    except Exception as e:
        sys.exit(f"Error occured while reading STC data: {e}")

    print("STC model data loaded successfully.")
    print("Calculating S(q,w) using CDFT...")

    omega_vals: tuple[Array1D[np.float64], ...]
    sqw_vals: tuple[Array1D[np.complex128], ...]
    omega_vals, sqw_vals = zip(*map(lambda q: sqw_cdft(q, time_vec, gamma_qtm), Q_VALUES), strict=True)

    print("CDFT calculation complete.")
    print("Calculating S(q,w) using STC model...")

    stc_sqw_vals: list[Array1D[np.float64]] = list(
        map(lambda q, omega: sqw_stc_model(q, omega, freq_dos, dos, TEMPERATURE), Q_VALUES, omega_vals)
    )

    print("STC model calculation complete.")
    print("Plotting S(q,w) comparison...")

    plt.figure(figsize=(8, 6), layout="constrained")
    for q, omega, sqw, stc_sqw in zip(Q_VALUES, omega_vals, sqw_vals, stc_sqw_vals, strict=True):
        x = omega * HBAR if ENERGY_UNIT else omega
        (stc_line,) = plt.plot(x, np.abs(sqw), label="CDFT Model")
        plt.plot(x, stc_sqw, label=f"STC Model    for {q = :.2f}", linestyle="--", color=stc_line.get_color())
    plt.xlim(-10 if ENERGY_UNIT else -10 / HBAR)
    if SCALE == "log":
        plt.yscale("log")
    plt.xlabel("Energy (eV)" if ENERGY_UNIT else "Angular Frequency (rad/s)", fontsize=14)
    plt.ylabel(r"Scattering Function $S(q,\omega)$ (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)", fontsize=14)
    plt.grid()
    plt.legend(*reorder_legend_by_row(*plt.gca().get_legend_handles_labels(), ncol=2), ncol=2, loc="upper left")

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Plot S(q,w) comparison between CDFT and STC model for given Momentum Transfer values.")
    )
    parser.add_argument("gamma_file_path", type=str, help="File path to the HDF5 file containing the gamma data.")
    parser.add_argument(
        "stc_file_path",
        type=str,
        help="File path to the HDF5 file containing the STC model data.",
    )
    parser.add_argument(
        "q_values",
        nargs="+",
        type=str,
        help="Momentum transfer values (in Angstrom^-1), formatted in START[:END[:STEP]] (e.g., 1:10).",
    )
    parser.add_argument(
        "-e",
        "--element",
        type=str,
        choices=["H", "O"],
        default="H",
        help="Element symbol for which to plot the S(q,w) function (default: H).",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=293.0,
        help="Temperature in Kelvin for the STC model calculation (default: 293 K).",
    )
    parser.add_argument(
        "--scale",
        type=str,
        choices=["linear", "log"],
        default="linear",
        help="Scale for y-axis: 'linear' or 'log' (default: linear).",
    )
    parser.add_argument(
        "--energy-unit",
        action="store_true",
        help="Use energy unit (eV) for x-axis instead of angular frequency (rad/s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="fig/sqw_comparison_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/sqw_comparison_plot.png).",
    )
    return parser


if __name__ == "__main__":
    main()
