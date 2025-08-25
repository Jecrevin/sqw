import argparse
import sys
from typing import Final, Literal

import matplotlib.pyplot as plt
import numpy as np
from helper import get_gamma_data


def parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-e",
        "--element",
        type=str,
        default="H",
        help="Element symbol for which to plot the gamma function (default: H).",
    )
    argparser.add_argument(
        "-f",
        "--file-path",
        type=str,
        default="data/last_H.gamma",
        help="Path to the .gamma file (default: data/last_H.gamma).",
    )
    argparser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="fig/gamma_plot.png",
        type=str,
        help="Output file name for the plot (default: fig/gamma_plot.png).",
    )
    return argparser.parse_args()


def main():
    args = parse_args()

    ELEMENT: Final[str] = args.element
    FILE_PATH: Final[str] = args.file_path
    OUTPUT: Final[str | None] = args.output

    print(f"Reading gamma data for element: {ELEMENT} from {FILE_PATH}...")

    time_vec, gamma_qtm = get_gamma_data(element=ELEMENT, file_path=FILE_PATH)

    print("Gamma data loaded successfully.")
    print("Plotting gamma function...")

    plt.plot(time_vec, np.abs(gamma_qtm), label="Magnitude")
    plt.plot(time_vec, np.real(gamma_qtm), label="Real Part", linestyle=":")
    plt.plot(time_vec, np.imag(gamma_qtm), label="Imaginary Part", linestyle="--")
    plt.xlabel("Time (ps)")
    plt.title(f"Gamma Function for Element: {ELEMENT}")
    plt.grid()
    plt.legend()

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


if __name__ == "__main__":
    main()
