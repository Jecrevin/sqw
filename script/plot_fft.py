import argparse
import sys
from typing import Final, Literal

import matplotlib.pyplot as plt
import numpy as np
from helper import get_gamma_data, get_q_values_from_cmdline, reorder_legend_by_row, save_or_show_plot

from h2o_sqw_calc.typing import Array1D


def _parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(
        description=(
            "Plot H2O Scattering function S(q, w) results obtained by FFT for given "
            "Momentum Transfer values [Q1, Q2, ...] in normal or log scale."
        )
    )
    argparser.add_argument(
        "q",
        nargs="+",
        type=str,
        help="Momentum Transfer values in Angstrom^-1 for which to plot the S(q, w) function. Can be a list of numbers "
        "or a range in 'start-end' format.",
    )
    argparser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="Step for the range of Momentum Transfer values (default: 1.0). "
        "Used only when 'q' is in 'start-end' format.",
    )
    argparser.add_argument(
        "-e",
        "--element",
        type=str,
        choices=["H", "O"],
        default="H",
        help="Element symbol for which to plot the S(q, w) function (default: H).",
    )
    argparser.add_argument(
        "-f",
        "--file-path",
        type=str,
        default="data/last_{element}.gamma",
        help="format string for the path to the hdf5 file containing the gamma data "
        "(default: data/last_{element}.gamma).",
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
    args = _parse_args()

    ELEMENT: Final[Literal["H", "O"]] = args.element
    FILE_PATH: Final[str] = args.file_path.format(element=ELEMENT)
    OUTPUT: Final[str | None] = args.output
    SCALE: Final[Literal["linear", "log"]] = args.scale
    Q_VALUES: Final[Array1D[np.float64]] = get_q_values_from_cmdline(args.q, args.step)

    print(f"Reading gamma data for element: {ELEMENT} from {FILE_PATH}...")

    try:
        time_vec, gamma_qtm, _ = get_gamma_data(element=ELEMENT, file_path=FILE_PATH)
    except Exception as e:
        sys.exit(f"Error occured while reading gamma data: {e}")

    print("Gamma data loaded successfully.")
    print("Calculating S(q, w) via gamma function using FFT...")

    dt: float = np.diff(time_vec).mean()
    omega = np.fft.fftshift(np.fft.fftfreq(time_vec.size, d=dt)) * 2 * np.pi
    sisf_vals = np.exp(-((Q_VALUES[:, None]) ** 2) * gamma_qtm / 2)
    sqw_vals = np.fft.fftshift(np.fft.fft(sisf_vals, axis=1) / (2 * np.pi), axes=-1)

    print("S(q, w) calculation complete.")
    print("Plotting S(q, w) function...")

    plt.figure(figsize=(8, 6), layout="constrained")
    for q, sqw in zip(Q_VALUES, sqw_vals, strict=True):
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

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


if __name__ == "__main__":
    main()
