import argparse
import sys
from functools import partial
from typing import Final, Literal

import numpy as np
from helper import correct_sqw_by_detailed_balance, get_gamma_data, parse_q_values, save_or_show_plot
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import CubicSpline

from sqw import sqw_ga_model
from sqw._core import HBAR, NEUTRON_MASS
from sqw._typing import Array1D


def main():
    parser = setup_parser()

    args = parser.parse_args()

    GAMMA_FILE_PATH: Final[str] = args.gamma_file_path
    ELEMENT: Final[Literal["H", "O"]] = args.element
    TEMPERATURE: Final[float] = args.temperature  # unit: K
    MASS_NUM: Final[int] = args.mass_num
    USE_ENERGY_UNIT: Final[bool] = args.energy_unit
    NSAMPLES: Final[int] = args.nsamples
    OUTPUT: Final[str | None] = args.output
    try:
        Q_VALUES: Final[Array1D[np.float64]] = parse_q_values([args.q_values], step=5)
    except ValueError as e:
        parser.error(f"Error parsing q_values: {e}")

    print(f"Loading gamma data for '{ELEMENT}' from {GAMMA_FILE_PATH}...")

    try:
        time, gamma, _ = get_gamma_data(GAMMA_FILE_PATH)
    except Exception as e:
        sys.exit(f"Error occured while reading gamma data: {e}")

    print("Gamma data loaded successfully.")
    print("Calculating CDFT S(q,w) values...")

    correct_sqw = partial(correct_sqw_by_detailed_balance, temperature=TEMPERATURE)
    sqw_results = map(
        partial(
            sqw_ga_model,
            time_vec=time,
            gamma=gamma,
            assure_detailed_balance=correct_sqw,  # type: ignore
        ),
        Q_VALUES,
    )

    omega = np.linspace(-10 / HBAR, 4 / HBAR, NSAMPLES)

    sqw_abs_vals = np.vstack(
        [CubicSpline(omega_val, np.abs(sqw_val), extrapolate=False)(omega) for omega_val, sqw_val in sqw_results]
    )

    print("S(q,w) values calculated successfully!")
    print("Getting max point of  STC Model...")

    stc_max = np.array([-HBAR * q**2 / 2 / NEUTRON_MASS / MASS_NUM for q in Q_VALUES])

    print("Max points of STC Model obtained successfully!")
    print("Plotting results...")

    plt.figure(figsize=(10, 8), layout="constrained")
    y_coords = omega * HBAR if USE_ENERGY_UNIT else omega
    plt.imshow(
        sqw_abs_vals.T,
        extent=(Q_VALUES.min(), Q_VALUES.max(), y_coords.min(), y_coords.max()),
        origin="lower",
        aspect="auto",
        cmap="jet",
        norm=LogNorm(vmin=1e-27),
        interpolation="none",
    )
    plt.ylim(y_coords.min())
    plt.plot(
        Q_VALUES,
        stc_max * HBAR if USE_ENERGY_UNIT else stc_max,
        label="STC Model Max",
        color="tab:red",
        linestyle="--",
        clip_on=True,
    )
    plt.xlabel("Momentum Transfer $q$ (Å⁻¹)", fontsize=14)
    plt.ylabel("Energy (eV)" if USE_ENERGY_UNIT else "Angular Frequency (rad/s)", fontsize=14)
    plt.colorbar(label=r"Scattering Function $S(q,\omega)$ (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)")
    plt.legend(loc="upper right")
    plt.grid()

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot S(q,w) function obtained by CDFT for a range of Momentum Transfer values stacking together.",
    )
    parser.add_argument("gamma_file_path", type=str, help="File path to the HDF5 file containing the gamma data.")
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
        "--temperature",
        type=float,
        default=293.0,
        help="Temperature in Kelvin (default: 293 K).",
    )
    parser.add_argument(
        "--mass-num",
        type=int,
        default=1,
        help="Neutron mass number (default: 1)",
    )
    parser.add_argument(
        "--energy-unit",
        action="store_true",
        help="Use energy unit (eV) for x-axis instead of angular frequency (rad/s).",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=3000,
        help="Number of samples for plot interpolation of S(q,w) (default: 3000).",
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
