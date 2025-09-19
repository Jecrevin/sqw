import argparse
import sys
from typing import Final, Literal

import numpy as np
from helper import get_sqw_molecular_dynamics_data, parse_indices, reorder_legend_by_row, save_or_show_plot
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from sqw._core import HBAR
from sqw._typing import Array1D


def main():
    parser = setup_parser()

    args = parser.parse_args()

    MD_FILE_PATH: Final[str] = args.md_file_path
    INDICES: Final[Array1D[np.int_]] = parse_indices(args.indices)
    ELEMENT: Final[Literal["H", "O"]] = args.element
    USE_ENERGY_UNIT: Final[bool] = args.energy_unit
    SCALE: Final[Literal["linear", "log"]] = args.scale
    OUTPUT: Final[str | None] = args.output

    print(f"Reading data from {MD_FILE_PATH}...")

    try:
        q_values, omega, sqw_md_vstack = get_sqw_molecular_dynamics_data(MD_FILE_PATH, ELEMENT)
    except Exception as e:
        sys.exit(f"Error occured while reading data: {e}")

    print("Data loaded successfully.")
    print(f"Plotting S(q,w) for Element '{ELEMENT}'...")

    plt.figure(figsize=(10, 6))
    for idx in INDICES:
        if idx < 0 or idx >= sqw_md_vstack.shape[0]:
            print(f"Warning: Index {idx} is out of bounds. Skipping this index.")
            continue
        sqw_md_data: NDArray[np.float64] = sqw_md_vstack[idx, :]
        q = q_values[idx]
        x = omega * HBAR if USE_ENERGY_UNIT else omega
        plt.plot(x, sqw_md_data, label=f"{q = :.2f}")
    plt.yscale(SCALE)
    plt.xlabel("Energy (eV)" if USE_ENERGY_UNIT else "Angular Frequency (rad/s)", fontsize=14)
    plt.ylabel(r"Scattering Function $S(q,\omega)$ (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)", fontsize=14)
    plt.grid()
    if len(INDICES) > 10:
        ncol = 5
        plt.legend(
            *reorder_legend_by_row(*plt.gca().get_legend_handles_labels(), ncol),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=ncol,
        )
    else:
        plt.legend(loc="upper left")

    print("Plotting completed.")

    save_or_show_plot(OUTPUT)

    print("Program completed successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the Classical Scattering Function S(q,w).")
    parser.add_argument(
        "md_file_path",
        type=str,
        help="File path to the HDF5 file containing the MD simulation data.",
    )
    parser.add_argument(
        "indices",
        type=str,
        nargs="+",
        help="Indices of MD simulation data to plot GAAQC from, formatted in START[:END[:STEP]]",
    )
    parser.add_argument(
        "--element",
        type=str,
        choices=["H", "O"],
        default="H",
        help="Element symbol to plot (default: H).",
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
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        type=str,
        const="fig/sqw_md_plot.png",
        help="Output file name for the plot (default: fig/sqw_md_plot.png).",
    )
    return parser


if __name__ == "__main__":
    main()
