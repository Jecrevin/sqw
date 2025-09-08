import argparse
import sys
from typing import Final, Literal

import numpy as np
from helper import get_gamma_data, get_stc_model_data, parse_q_values, save_or_show_plot
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import CubicSpline

from h2o_sqw_calc.core import HBAR, sqw_cdft, sqw_stc_model
from h2o_sqw_calc.typing import Array1D


def main():
    parser = setup_parser()

    args = parser.parse_args()

    GAMMA_FILE_PATH: Final[str] = args.gamma_file_path
    STC_FILE_PATH: Final[str] = args.stc_file_path
    ELEMENT: Final[Literal["H", "O"]] = args.element
    TEMPERATURE: Final[float] = args.temperature  # unit: K
    ENERGY_UNIT: Final[bool] = args.energy_unit
    OUTPUT: Final[str | None] = args.output
    try:
        Q_VALUES: Final[Array1D[np.float64]] = parse_q_values([args.q_values], step=5)
    except ValueError as e:
        parser.error(f"Error parsing q_values: {e}")

    print(f"Loading STC model data from {STC_FILE_PATH}...")

    try:
        freq_dos, dos = get_stc_model_data(STC_FILE_PATH, ELEMENT)
    except Exception as e:
        sys.exit(f"Error occured while reading STC data: {e}")

    print("STC model data loaded successfully.")
    print(f"Loading gamma data for '{ELEMENT}' from {GAMMA_FILE_PATH}...")

    try:
        time, gamma, _ = get_gamma_data(GAMMA_FILE_PATH)
    except Exception as e:
        sys.exit(f"Error occured while reading gamma data: {e}")

    print("Gamma data loaded successfully.")
    print("Calculating CDFT S(Q, w) values...")

    sqw_results = [sqw_cdft(q, time, gamma) for q in Q_VALUES]
    omega_vals, sqw_vals = zip(*sqw_results, strict=True)

    print("CDFT S(Q, w) values calculated successfully!")
    print("Interpolating S(Q, w) values on a common frequency grid...")

    omega = np.linspace(-10 / HBAR, 4 / HBAR, 3000)

    sqw_abs_vals = np.stack(
        [
            CubicSpline(omega_val, np.abs(sqw_val), extrapolate=False)(omega)
            for sqw_val, omega_val in zip(sqw_vals, omega_vals, strict=True)
        ]
    )

    print("S(Q, w) values interpolated successfully!")
    print("Getting max point of  STC Model...")

    stc_max = np.array([omega[sqw_stc_model(q, omega, freq_dos, dos, TEMPERATURE).argmax()] for q in Q_VALUES])

    print("Max points of STC Model obtained successfully!")
    print("Plotting results...")

    plt.figure(figsize=(10, 8), layout="constrained")
    y_coords = omega * HBAR if ENERGY_UNIT else omega
    plt.imshow(
        sqw_abs_vals.T,
        extent=(Q_VALUES.min(), Q_VALUES.max(), y_coords.min(), y_coords.max()),
        origin="lower",
        aspect="auto",
        cmap="jet",
        norm=LogNorm(vmin=1e-27),
        interpolation="none",
    )
    plt.plot(Q_VALUES, stc_max * HBAR if ENERGY_UNIT else stc_max, color="red", label="STC Model Max", linestyle="--")
    plt.xlabel("Momentum Transfer (Q) [Angstrom^-1]", fontsize=14)
    plt.ylabel("Energy (eV)" if ENERGY_UNIT else "Angular Frequency (rad/s)", fontsize=14)
    plt.colorbar(label=r"Scattering Function $S(q,\omega)$ (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)")
    plt.legend(loc="upper right", bbox_to_anchor=(0, 0, 1, 1.01))
    plt.grid()

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot S(q,w) function obtained by CDFT for a range of Momentum Transfer values stacking together.",
    )
    parser.add_argument("gamma_file_path", type=str, help="File path to the HDF5 file containing the gamma data.")
    parser.add_argument(
        "stc_file_path",
        type=str,
        help="File path to the HDF5 file containing the STC model data.",
    )
    parser.add_argument(
        "q_values",
        type=str,
        help="The range of Momentum Transfer values (in Angstrom^-1) to plot, formatted in [START:END[:STEP]]",
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
        "--energy-unit",
        action="store_true",
        help="Use energy unit (eV) for x-axis instead of angular frequency (rad/s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="fig/sqw_stack_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/sqw_stack_plot.png).",
    )
    return parser


if __name__ == "__main__":
    main()
