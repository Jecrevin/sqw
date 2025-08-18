import argparse
from typing import Final, Literal

import matplotlib.pyplot as plt
import numpy as np
from helper import get_gamma_data
from numpy.typing import NDArray

from h2o_sqw_calc.utils import flow


def parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(
        description="Plot H2O Scattering function S(q, w) results obtained by FFT for given Momentum Transfer values [Q1, Q2, ...] in normal or log scale."
    )
    argparser.add_argument(
        "q",
        nargs="+",
        type=float,
        help="Momentum Transfer values in Angstrom^-1 for which to plot the S(q, w) function.",
    )
    argparser.add_argument(
        "-e",
        "--element",
        type=str,
        default="H",
        help="Element symbol for which to plot the S(q, w) function (default: H).",
    )
    argparser.add_argument(
        "-f",
        "--file-path",
        type=str,
        default="data/last_{element}.gamma",
        help="format string for the path to the .gamma file (default: data/last_{element}.gamma).",
    )
    argparser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="fig/fft_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/fft_plot.png).",
    )
    argparser.add_argument(
        "--scale",
        choices=["linear", "log"],
        default="linear",
        help="Scale for the y-axis of the plot (default: linear).",
    )

    return argparser.parse_args()


def main():
    args = parse_args()

    ELEMENT: Final[str] = args.element
    FILE_PATH: Final[str] = args.file_path.format(element=ELEMENT)
    OUTPUT: Final[str | None] = args.output
    SCALE: Final[Literal["linear", "log"]] = args.scale
    Q_VALUES: Final[NDArray[np.float64]] = np.array(args.q)

    print(f"Reading gamma data for element: {ELEMENT} from {FILE_PATH}...")

    time_vec, gamma_qtm = get_gamma_data(element=ELEMENT, file_path=FILE_PATH)

    print("Gamma data loaded successfully.")
    print("Calculating S(q, w) via gamma function using FFT...")

    dt: float = np.diff(time_vec).mean()
    omega = np.fft.fftshift(np.fft.fftfreq(time_vec.size, d=dt)) * 2 * np.pi
    sqw_vals: NDArray[np.complex128] = flow(
        (gamma_qtm, np.expand_dims(Q_VALUES, axis=1)),
        lambda args: np.fft.fft(np.exp(-(args[1] ** 2) * args[0] / 2)) / (2 * np.pi),
        lambda sqw_vals: np.fft.fftshift(sqw_vals, axes=-1),
    )

    print("S(q, w) calculation complete.")
    print("Plotting S(q, w) function...")

    for q, sqw in zip(Q_VALUES, sqw_vals):
        plt.plot(omega, np.abs(sqw), label=f"Q = {q:.2f}")

    match SCALE:
        case "linear":
            plt.yscale("linear")
        case "log":
            plt.yscale("log")
    plt.xlabel("Angular Frequency (rad/s)")
    plt.ylabel("S(q, w)")
    plt.title(f"S(q, w) Function for Element: {ELEMENT}")
    plt.grid()
    plt.legend()

    if OUTPUT:
        plt.savefig(OUTPUT)
        print(f"Plot saved to {OUTPUT}")
    else:
        print("Displaying plot interactively.")
        plt.show()

    print("Program completed successfully.")


if __name__ == "__main__":
    main()
