import argparse
import sys
from typing import Final, Literal

import numpy as np
from helper import get_gamma_data, get_sqw_molecular_dynamics_data, save_or_show_plot
from matplotlib import pyplot as plt

from h2o_sqw_calc.core import sqw_gaaqc


def main() -> None:
    args = _parse_args()

    INDICES: Final[list[int]] = _get_indices(args.indices, args.step)
    ELEMENT: Final[Literal["H", "O"]] = args.elelement
    GAMMA_FILE_PATH: Final[str] = args.gamma_file_path
    MD_FILE_PATH: Final[str] = args.md_file_path
    OUTPUT: Final[str | None] = args.output

    print(f"Loading gamma data for element '{ELEMENT}' from '{GAMMA_FILE_PATH.format(element=ELEMENT)}'...")

    try:
        time_vec, gamma_qtm, gamma_cls = get_gamma_data(ELEMENT, GAMMA_FILE_PATH, include_classical=True)
    except Exception as e:
        sys.exit(f"Error loading gamma data: {e}")

    assert gamma_cls is not None, "`include_classical` must be True to get classical gamma data."

    print("Gamma data loaded successfully.")
    print(f"Loading MD simulation data from file '{MD_FILE_PATH}'...")

    try:
        q_vals, omega, sqw_md_vals = get_sqw_molecular_dynamics_data(ELEMENT)
    except Exception as e:
        sys.exit(f"Error loading MD data: {e}")

    print("MD data loaded successfully.")
    print("Calculating S(Q, w) using GAAQC model...")

    q_vals = q_vals[INDICES]
    sqw_md_vals = sqw_md_vals[INDICES]

    omega_vals, sqw_gaaqc_vals = zip(
        *[
            sqw_gaaqc(q, time_vec, gamma_qtm, gamma_cls, omega, sqw_md)
            for q, sqw_md in zip(q_vals, sqw_md_vals, strict=True)
        ],
        strict=True,
    )

    print("Calculation completed.")
    print("Plotting results...")

    plt.figure(figsize=(10, 6))
    for q, omega, sqw in zip(q_vals, omega_vals, sqw_gaaqc_vals, strict=True):
        plt.plot(omega, np.abs(sqw), label=f"Q = {q:.2f}")
    plt.xlabel("Angular Frequency ω (rad/s)")
    plt.ylabel("S(Q, ω)")
    plt.title(f"S(Q, ω) from GAAQC Model for Element '{ELEMENT}'")
    plt.legend()
    plt.grid()

    print("Plotting completed.")

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


def _parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(
        description="Plot Scattering Function S(Q, w) Calculated from Gaussian "
        "Approximation Assisted Quantum Correction (GAAQC) model."
    )

    argparser.add_argument(
        "indices",
        type=str,
        nargs="+",
        help="Indices of MD simulation data to plot GAAQC from, e.g., '0 1 2' or '0-3'.",
    )
    argparser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for reading MD simulation data. Default is 1 (read all data).",
    )
    argparser.add_argument(
        "-e",
        "--elelement",
        type=str,
        default="H",
        choices=["H", "O"],
        help="Element type to plot GAAQC for. Default is 'H'.",
    )
    argparser.add_argument(
        "-gamma",
        "--gamma-file-path",
        type=str,
        default="data/last_{element}.gamma",
        help="Format string for the path to the gamma data file. Default is 'data/last_{element}.gamma'.",
    )
    argparser.add_argument(
        "-md",
        "--md-file-path",
        type=str,
        default="data/merged_h2o_293k.sqw",
        help="Path to the MD simulation data file. Default is 'data/merged_h2o_293k.sqw'.",
    )
    argparser.add_argument(
        "-o",
        "--output",
        type=str,
        nargs="?",
        const="data/plot_sqw_gaaqc.png",
        help="Output file name to save the plot. If not provided, defaults to 'data/plot_sqw_gaaqc.png'.",
    )

    return argparser.parse_args()


def _get_indices(indices: list[str], step: int) -> list[int]:
    result: list[int] = []
    for index in indices:
        if "-" in index:
            start, end = map(int, index.split("-"))
            result.extend(range(start, end + 1, step))
        else:
            result.append(int(index))
    return sorted(set(result))


if __name__ == "__main__":
    main()
