import argparse
import sys
from typing import Final, Literal

import numpy as np
from helper import get_gamma_data, get_stc_model_data, save_or_show_plot
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from numpy.typing import NDArray

from h2o_sqw_calc.core import HBAR, sqw_cdft, sqw_stc_model
from h2o_sqw_calc.utils import flow


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

    q_vals: tuple[float, ...]
    omega_vals: tuple[NDArray[np.float64], ...]
    sqw_vals: tuple[NDArray[np.complex128], ...]
    q_vals, omega_vals, sqw_vals = flow(
        np.linspace(Q_START, Q_END, int((Q_END - Q_START) / Q_STEP), endpoint=False),
        lambda q_range: [(q, *sqw_cdft(q, time, gamma)) for q in q_range],
        lambda results: zip(*results, strict=False),
    )

    print("CDFT S(Q, w) values calculated successfully!")
    print("Interpolating S(Q, w) values on a common frequency grid...")

    omega: NDArray[np.float64] = flow(
        omega_vals,
        lambda omega_vals: ([omega_val[0] for omega_val in omega_vals], [omega_val[-1] for omega_val in omega_vals]),
        lambda minmax_arr: (min(minmax_arr[0]), max(minmax_arr[1])),
        lambda minmax: np.linspace(minmax[0], minmax[1], int((minmax[1] - minmax[0]) / dw) + 1),
    )

    sqw = np.stack(
        flow(
            (sqw_vals, omega_vals),
            lambda sqw_omega: map(lambda sqw_val, omega_val: np.interp(omega, omega_val, sqw_val), *sqw_omega),
            lambda sqw: list(sqw),
        ),
    )

    print("S(Q, w) values interpolated successfully!")
    print("Getting max point of  STC Model...")

    stc_max = np.array(
        [
            flow(
                q,
                lambda q: sqw_stc_model(q, omega, freq_dos, dos, T).argmax(),
                lambda idx: omega[idx],
            )
            for q in q_vals
        ]
    )

    print("Max points of STC Model obtained successfully!")
    print("Plotting results...")

    plt.figure(figsize=(10, 8), layout="constrained")
    plt.pcolormesh(
        q_vals,
        omega * HBAR if ENERGY_UNIT else omega,
        np.abs(sqw.T),
        shading="auto",
        cmap="viridis",
        norm=LogNorm(),
    )
    plt.plot(q_vals, stc_max * HBAR if ENERGY_UNIT else stc_max, color="red", label="STC Model Max", linestyle="--")
    plt.title(f"S(Q, w) for {ELEMENT} at {T} K")
    plt.xlabel("Momentum Transfer (Q) [Angstrom^-1]")
    plt.ylabel("Energy (eV)" if ENERGY_UNIT else "Angular Frequency (rad/s)")
    plt.ylim(bottom=(-10 if ENERGY_UNIT else -10 / HBAR))
    plt.colorbar(label="S(Q, w)")
    plt.legend(loc="lower right", bbox_to_anchor=(1.05, 0.99))
    plt.grid()

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


if __name__ == "__main__":
    main()
