import argparse
import sys
from functools import partial
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

from sqw import sqw_ga_model, sqw_stc_model
from sqw._core import HBAR
from sqw._typing import Array1D


def main():
    parser = setup_parser()

    args = parser.parse_args()

    GAMMA_FILE_PATH: Final[str] = args.gamma_file_path
    ELEMENT: Final[Literal["H", "O"]] = args.element
    STC_FILE_PATH: Final[str | None] = args.stc_file_path
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
    print("Calculating S(q,w) using CDFT...")

    omega_vals, sqw_cdft_results = zip(
        *map(partial(sqw_ga_model, time_vec=time_vec, gamma=gamma_qtm), Q_VALUES), strict=True
    )

    print("CDFT calculation complete.")

    if STC_FILE_PATH:
        print(f"Reading STC model data from {STC_FILE_PATH}...")

        try:
            freq_dos, dos = get_stc_model_data(STC_FILE_PATH, element=ELEMENT)
        except Exception as e:
            sys.exit(f"Error occured while reading STC data: {e}")

        print("STC model data loaded successfully.")

        print("Calculating S(q,w) using STC model...")

        sqw_stc_results = map(
            partial(sqw_stc_model, freq_dos=freq_dos, density_of_states=dos, temperature=TEMPERATURE),
            Q_VALUES,
            omega_vals,
        )
        print("STC model calculation complete.")
    else:
        sqw_stc_results = None

    print("Plotting S(q,w) results...")

    plt.figure(figsize=(10, 6), layout="constrained")
    if sqw_stc_results is not None:
        for q, omega, sqw_cdft_data, sqw_stc_data in zip(
            Q_VALUES, omega_vals, sqw_cdft_results, sqw_stc_results, strict=True
        ):
            x = omega * HBAR if ENERGY_UNIT else omega
            (line,) = plt.plot(x, np.abs(sqw_cdft_data), label="CDFT model")
            plt.plot(x, sqw_stc_data, label=f"STC model    for {q = :.2f}", color=line.get_color(), linestyle="--")
            plt.legend(*reorder_legend_by_row(*plt.gca().get_legend_handles_labels(), ncol=2), ncol=2, loc="upper left")
    else:
        for q, omega, sqw_cdft_data in zip(Q_VALUES, omega_vals, sqw_cdft_results, strict=True):
            x = omega * HBAR if ENERGY_UNIT else omega
            plt.plot(x, np.abs(sqw_cdft_data), label=f"{q = :.2f}")
            plt.legend(loc="upper left")
    plt.yscale(SCALE)
    x_min = -10 if ENERGY_UNIT else -10 / HBAR
    x_max = 2 if ENERGY_UNIT else 2 / HBAR
    plt.xlim(max(plt.xlim()[0], x_min), min(plt.xlim()[1], x_max))
    plt.grid()
    plt.xlabel("Energy (eV)" if ENERGY_UNIT else r"Angular Frequency $\omega$ (rad/s)", fontsize=14)
    plt.ylabel(r"Scattering Function $S(q,\omega)$ (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)", fontsize=14)

    print("Plotting completed.")

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Plot S(q,w) comparison between CDFT and STC model for given Momentum Transfer values.")
    )
    parser.add_argument("gamma_file_path", type=str, help="File path to the HDF5 file containing the gamma data.")
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
        "-stc",
        "--stc-file-path",
        type=str,
        help="File path to the HDF5 file containing the STC model data.",
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
        const="fig/sqw_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/sqw_plot.png).",
    )
    return parser


if __name__ == "__main__":
    main()
