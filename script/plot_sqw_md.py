import argparse
import sys
from typing import Final, Literal

import numpy as np
from helper import even_extend, odd_extend, save_or_show_plot
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from h2o_sqw_calc.io import get_data_from_h5py


def main():
    args = _parse_args()

    INDEXS: Final[list[int]] = _get_indexs(args.indexs, args.step)
    FILE_PATH: Final[str] = args.file_path
    ELEMENT: Final[Literal["H", "O"]] = args.element
    OUTPUT: Final[str | None] = args.output

    print(f"Reading data from {FILE_PATH}...")

    try:
        sqw_md_data: NDArray[np.float64] = np.apply_along_axis(
            even_extend,
            -1,
            get_data_from_h5py(FILE_PATH, f"inc_sqw_{ELEMENT}"),
        )
        q_vec: NDArray[np.float64] = get_data_from_h5py(FILE_PATH, f"qVec_{ELEMENT}")
        omega: NDArray[np.float64] = odd_extend(get_data_from_h5py(FILE_PATH, f"inc_omega_{ELEMENT}"))
    except Exception as e:
        sys.exit(f"Error occured while reading data: {e}")

    print("Data loaded successfully.")
    print(f"Plotting S(Q, w) for Element '{ELEMENT}'...")

    _plot_sqw_md(q_vec, omega, sqw_md_data, INDEXS)

    print("Plotting completed.")

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


def _parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="Plot the Classical Scattering Function S(Q, w).")

    argparser.add_argument(
        "indexs",
        nargs="+",
        type=str,
        help="List of indices or ranges (e.g., 1,2,5-10) to plot.",
    )
    argparser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for ranges in indices (default: 1).",
    )
    argparser.add_argument(
        "-f",
        "--file-path",
        type=str,
        default="data/merged_h2o_293k.sqw",
        help="Path to the data file (default: data/merged_h2o_293k.sqw).",
    )
    argparser.add_argument(
        "--element",
        type=str,
        choices=["H", "O"],
        default="H",
        help="Element symbol to plot (default: H).",
    )
    argparser.add_argument(
        "-o",
        "--output",
        nargs="?",
        type=str,
        const="fig/sqw_md_plot.png",
        help="Output file name for the plot (default: fig/sqw_md_plot.png).",
    )

    return argparser.parse_args()


def _get_indexs(indexs: list[str], step: int) -> list[int]:
    result: list[int] = []
    for part in indexs:
        if "-" in part:
            start, end = map(int, part.split("-"))
            result.extend(range(start, end + 1, step))
        else:
            result.append(int(part))
    return sorted(set(result))


def _plot_sqw_md(
    q_vals: NDArray[np.floating],
    omega: NDArray[np.floating],
    sqw_cls_data: NDArray[np.floating],
    indexs: list[int],
) -> None:
    plt.figure(figsize=(10, 8), layout="constrained")
    for idx in indexs:
        if idx < 0 or idx >= sqw_cls_data.shape[0]:
            print(f"Warning: Index {idx} is out of bounds. Skipping.")
            continue
        plt.plot(omega, sqw_cls_data[idx], label=f"Q = {q_vals[idx]:.2f} Å⁻¹")
    plt.xlabel("Angular Frequency (rad/s)")
    plt.ylabel("Arbitrary Units")
    plt.title("Scattering Function S(Q, w) Data from Molecular Dynamics Simulation")
    plt.legend()
    plt.grid()


if __name__ == "__main__":
    main()
