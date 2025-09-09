import argparse
import sys
from typing import Final, Literal

import matplotlib.pyplot as plt
import numpy as np
from helper import get_gamma_data, parse_q_values, reorder_legend_by_row, save_or_show_plot

from h2o_sqw_calc.core import HBAR
from h2o_sqw_calc.typing import Array1D


def main():
    parser = setup_parser()

    args = parser.parse_args()

    FILE_PATH: Final[str] = args.file_path
    OUTPUT: Final[str | None] = args.output
    SCALE: Final[Literal["linear", "log"]] = args.scale
    USE_ENERGY_UNIT: Final[bool] = args.energy_unit
    try:
        Q_VALUES: Final[Array1D[np.float64]] = parse_q_values(args.q_values)
    except ValueError as e:
        parser.error(f"Error parsing Q values: {e}")

    print(f"Reading gamma data from {FILE_PATH}...")

    try:
        time_vec, gamma_qtm, _ = get_gamma_data(FILE_PATH)
    except Exception as e:
        sys.exit(f"Error occured while reading gamma data: {e}")

    print("Gamma data loaded successfully.")
    print("Calculating S(q,w) via gamma function using FFT...")

    dt: float = np.diff(time_vec).mean()
    omega = np.fft.fftshift(np.fft.fftfreq(time_vec.size, d=dt)) * 2 * np.pi
    sisf_vals = np.exp(-((Q_VALUES[:, None]) ** 2) * gamma_qtm / 2)
    sqw_vals = np.fft.fftshift(np.fft.fft(sisf_vals, axis=1) / (2 * np.pi), axes=-1)

    print("S(q,w) calculation complete.")
    print("Plotting S(q,w) function...")

    plt.figure(figsize=(8, 6), layout="constrained")
    for q, sqw in zip(Q_VALUES, sqw_vals, strict=True):
        plt.plot(omega * HBAR if USE_ENERGY_UNIT else omega, np.abs(sqw), label=f"{q = :.2f}")
    if SCALE == "log":
        plt.yscale("log")
    plt.xlabel("Energy (eV)" if USE_ENERGY_UNIT else "Angular Frequency (rad/s)", fontsize=14)
    plt.ylabel(r"Scattering Function $S(q,\omega)$ (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)", fontsize=14)
    plt.grid()
    if len(Q_VALUES) > 10:
        ncol = 5
        plt.legend(
            *reorder_legend_by_row(*plt.gca().get_legend_handles_labels(), ncol),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=ncol,
        )
    else:
        plt.legend()

    print("Plotting complete successfully.")

    save_or_show_plot(OUTPUT)

    print("Program complete successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot H2O Scattering function S(q,w) results obtained by FFT "
        "for given Momentum Transfer values in normal or log scale."
    )
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file containing gamma function data.")
    parser.add_argument(
        "q_values",
        nargs="+",
        type=str,
        help="Momentum Transfer values (unit: angstrom^-1), format in START[:END[:STEP]].",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="fig/fft_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/fft_plot.png).",
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
    return parser


if __name__ == "__main__":
    main()
