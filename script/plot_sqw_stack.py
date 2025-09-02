import argparse
import sys
from typing import Final, Literal

import numpy as np
from helper import get_gamma_data, get_stc_model_data, save_or_show_plot
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from h2o_sqw_calc.core import HBAR, sqw_cdft, sqw_stc_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot S(q, w) function obtained by CDFT for a range of Momentum "
        "Transfer values [Q_START-Q_END] stacking together.",
    )
    parser.add_argument(
        "q-range",
        type=str,
        help="The range of Momentum Transfer values in Angstrom^-1 for which to "
        "plot the S(q, w) function obtained by CDFT. ",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=5.0,
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
        const="fig/sqw_stack_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/sqw_stack_plot.png).",
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
    T: Final[float] = args.temperature  # unit: K
    GAMMA_FILE_PATH: Final[str] = args.gamma_file_path.format(element=ELEMENT)
    STC_FILE_PATH: Final[str] = args.stc_file_path
    OUTPUT: Final[str | None] = args.output
    ENERGY_UNIT: Final[bool] = args.energy_unit

    Q_START, Q_END = map(float, getattr(args, "q-range").split("-"))
    Q_STEP: Final[float] = args.step

    print(f"Loading STC model data from {STC_FILE_PATH}...")

    try:
        freq_dos, dos = get_stc_model_data(ELEMENT, STC_FILE_PATH)
    except Exception as e:
        sys.exit(f"Error occured while reading STC data: {e}")

    print("STC model data loaded successfully!")
    print(f"Loading gamma data from {GAMMA_FILE_PATH}...")

    try:
        time, gamma, _ = get_gamma_data(ELEMENT, GAMMA_FILE_PATH)
    except Exception as e:
        sys.exit(f"Error occured while reading gamma data: {e}")

    print("Gamma data loaded successfully!")
    print("Calculating CDFT S(Q, w) values...")

    dw = 2 * np.pi / np.diff(time).mean() / time.size

    q_vals = np.linspace(Q_START, Q_END, int((Q_END - Q_START) / Q_STEP), endpoint=False)
    sqw_results = [sqw_cdft(q, time, gamma) for q in q_vals]
    omega_vals, sqw_vals = zip(*sqw_results, strict=True)

    print("CDFT S(Q, w) values calculated successfully!")
    print("Interpolating S(Q, w) values on a common frequency grid...")

    omega_min: float = min([omega_val[0] for omega_val in omega_vals])
    omega_max: float = max([omega_val[-1] for omega_val in omega_vals])
    omega = np.linspace(omega_min, omega_max, int((omega_max - omega_min) / dw) + 1)

    sqw = np.stack(
        [
            np.interp(omega, omega_val, sqw_val, left=0, right=0)
            for sqw_val, omega_val in zip(sqw_vals, omega_vals, strict=True)
        ]
    )

    print("S(Q, w) values interpolated successfully!")
    print("Getting max point of  STC Model...")

    stc_max = np.array([omega[sqw_stc_model(q, omega, freq_dos, dos, T).argmax()] for q in q_vals])

    print("Max points of STC Model obtained successfully!")
    print("Plotting results...")

    plt.figure(figsize=(10, 8), layout="constrained")
    y_coords = omega * HBAR if ENERGY_UNIT else omega
    plt.imshow(
        np.abs(sqw.T),
        extent=(q_vals.min(), q_vals.max(), y_coords.min(), y_coords.max()),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(),
        interpolation="none",
    )
    plt.plot(q_vals, stc_max * HBAR if ENERGY_UNIT else stc_max, color="red", label="STC Model Max", linestyle="--")
    plt.title(f"S(Q, w) for {ELEMENT} at {T} K")
    plt.xlabel("Momentum Transfer (Q) [Angstrom^-1]")
    plt.ylabel("Energy (eV)" if ENERGY_UNIT else "Angular Frequency (rad/s)")
    plt.ylim(bottom=(-10 if ENERGY_UNIT else -10 / HBAR))
    plt.colorbar(label="S(Q, w)")
    plt.legend(loc="lower right", bbox_to_anchor=(1.01, 0.99))
    plt.grid()

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


if __name__ == "__main__":
    main()
