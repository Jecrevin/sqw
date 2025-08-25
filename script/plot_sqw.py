import argparse
import sys
from typing import Final, Literal

import numpy as np
from helper import (
    get_gamma_data,
    get_q_values_from_cmdline,
    get_stc_model_data,
    reorder_legend_by_row,
    save_or_show_plot,
)
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from h2o_sqw_calc.core import HBAR, sqw_cdft, sqw_stc_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot S(q, w) comparison between CDFT and STC model for given Momentum Transfer values [Q1, Q2, ...]."
        )
    )
    parser.add_argument(
        "q",
        nargs="+",
        type=str,
        help="Momentum Transfer values in Angstrom^-1 for which to plot the S(q, w) function obtained by CDFT. "
        "Can be a list of numbers or a range in 'start-end' format.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="Step size for the range of momentum transfer values (default: 1.0 Angstrom^-1). "
        "This is only used if a range is specified in the q argument.",
    )
    parser.add_argument(
        "-e",
        "--element",
        type=str,
        choices=["H", "O"],
        default="H",
        help="Element symbol for which to plot the S(q, w) function (default: H).",
    )
    parser.add_argument(
        "-gamma",
        "--gamma-file-path",
        type=str,
        default="data/last_{element}.gamma",
        help="format string for the path to the hdf5 file containing the gamma data "
        "(default: data/last_{element}.gamma).",
    )
    parser.add_argument(
        "-stc",
        "--stc-file-path",
        type=str,
        default="data/last.sqw",
        help="file path for the hdf5 file containing the STC model data (default: data/last.sqw).",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=293.0,
        help="Temperature in Kelvin for the STC model calculation (default: 293 K).",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="fig/sqw_comparison_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/sqw_comparison_plot.png).",
    )
    parser.add_argument(
        "--energy-unit",
        action="store_true",
        help="Use energy units for the x axis of output plot (default: False, uses angular frequency).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    ELEMENT: Final[Literal["H", "O"]] = args.element
    ENERGY_UNIT: Final[bool] = args.energy_unit
    GAMMA_FILE_PATH: Final[str] = args.gamma_file_path.format(element=ELEMENT)
    OUTPUT: Final[str | None] = args.output
    STC_FILE_PATH: Final[str] = args.stc_file_path
    T: Final[float] = args.temperature  # unit: K
    Q_VALUES: Final[NDArray[np.float64]] = get_q_values_from_cmdline(args.q, args.step)

    print(f"Reading gamma data for element: {ELEMENT} from {GAMMA_FILE_PATH}...")

    try:
        time_vec, gamma_qtm, _ = get_gamma_data(ELEMENT, GAMMA_FILE_PATH)
    except Exception as e:
        sys.exit(f"Error occured while reading gamma data: {e}")

    print("Gamma data loaded successfully.")
    print(f"Reading STC model data from {STC_FILE_PATH}...")

    try:
        freq_dos, dos = get_stc_model_data(ELEMENT, STC_FILE_PATH)
    except Exception as e:
        sys.exit(f"Error occured while reading STC data: {e}")

    print("STC model data loaded successfully.")
    print("Calculating S(q, w) using CDFT...")

    omega_vals: tuple[NDArray[np.float64], ...]
    sqw_vals: tuple[NDArray[np.complex128], ...]
    omega_vals, sqw_vals = zip(*map(lambda q: sqw_cdft(q, time_vec, gamma_qtm), Q_VALUES), strict=True)

    print("CDFT calculation complete.")
    print("Calculating S(Q, w) using STC model...")

    stc_sqw_vals: list[NDArray[np.float64]] = list(
        map(lambda q, omega: sqw_stc_model(q, omega, freq_dos, dos, T), Q_VALUES, omega_vals)
    )

    print("STC model calculation complete.")
    print("Plotting S(q, w) comparison...")

    plt.figure(figsize=(8, 6), layout="constrained")
    for q, omega, sqw, stc_sqw in zip(Q_VALUES, omega_vals, sqw_vals, stc_sqw_vals, strict=True):
        x = omega * HBAR if ENERGY_UNIT else omega
        (stc_line,) = plt.plot(x, stc_sqw, label=f"q = {q:.2f}: STC Model")
        plt.plot(x, np.abs(sqw), label="CDFT Model", linestyle=":", color=stc_line.get_color())
    plt.xlabel("Energy (eV)" if ENERGY_UNIT else "Angular Frequency (rad/s)")
    plt.ylabel("S(q, w)")
    plt.title(f"S(q, w) Comparison for Element: {ELEMENT}")
    plt.grid()
    plt.legend(*reorder_legend_by_row(*plt.gca().get_legend_handles_labels(), ncol=2), ncol=2, loc="upper left")

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


if __name__ == "__main__":
    main()
