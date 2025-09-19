import argparse
import sys
from functools import partial
from typing import Final, Literal

import h5py
import numpy as np
from helper import get_gamma_data, get_sqw_molecular_dynamics_data
from scipy.interpolate import CubicSpline

from sqw import sqw_ga_model, sqw_gaaqc_model
from sqw._core import HBAR


def main() -> None:
    parser = setup_parser()

    args = parser.parse_args()

    GAMMA_FILE_PATH: Final[str] = args.gamma_file_path
    MD_FILE_PATH: Final[str] = args.md_file_path
    ELEMENT: Final[Literal["H", "O"]] = args.element
    OUTPUT_FILE_PATH: Final[str] = args.output

    print(f"Loading gamma function data from {GAMMA_FILE_PATH}...")

    try:
        time_vec, gamma_qtm, gamma_cls = get_gamma_data(GAMMA_FILE_PATH, include_classical=True)
    except Exception as e:
        sys.exit(f"Error loading gamma function data: {e}")

    assert gamma_cls is not None, "`include_classical` must be True to get classical gamma data."

    print("Gamma function data loaded successfully.")
    print("Loading MD simulation S(q,w) data...")

    try:
        q_vals, omega_md, sqw_md_stack = get_sqw_molecular_dynamics_data(MD_FILE_PATH, element=ELEMENT)
    except Exception as e:
        sys.exit(f"Error loading MD simulation data: {e}")

    print("MD simulation S(q,w) data loaded successfully.")
    print("Calculating S(q,w) QTM, S(q,w) GAAQC...")

    results_qtm = map(partial(sqw_ga_model, time_vec=time_vec, gamma=gamma_qtm), q_vals)
    results_gaaqc = map(
        lambda q, sqw_md: sqw_gaaqc_model(q, time_vec, gamma_qtm, gamma_cls, omega_md, sqw_md),
        q_vals,
        sqw_md_stack,
    )

    omega = np.linspace(-10 / HBAR, 4 / HBAR, 3000)

    sqw_qtm_vstack = np.zeros((q_vals.size, omega.size), dtype=np.complex128)
    for i, (o, sqw) in enumerate(results_qtm):
        norm_interp = CubicSpline(o, np.abs(sqw), extrapolate=False)(omega)
        phase_interp = CubicSpline(o, np.unwrap(np.angle(sqw)), extrapolate=False)(omega)
        sqw_qtm_vstack[i] = norm_interp * np.exp(1j * phase_interp)

    sqw_gaaqc_vstack = np.zeros((q_vals.size, omega.size), dtype=np.complex128)
    for i, (o, sqw) in enumerate(results_gaaqc):
        norm_interp = CubicSpline(o, np.abs(sqw), extrapolate=False)(omega)
        phase_interp = CubicSpline(o, np.unwrap(np.angle(sqw)), extrapolate=False)(omega)
        sqw_gaaqc_vstack[i] = norm_interp * np.exp(1j * phase_interp)

    print("Calculations completed successfully.")
    print(f"Saving results to {OUTPUT_FILE_PATH}...")

    try:
        with h5py.File(OUTPUT_FILE_PATH, "w") as f:
            f.create_dataset("q_vals", data=q_vals)
            f.create_dataset("omega", data=omega)
            f.create_dataset("sqw_qtm", data=sqw_qtm_vstack)
            f.create_dataset("sqw_gaaqc", data=sqw_gaaqc_vstack)
    except Exception as e:
        sys.exit(f"Error saving results to HDF5 file: {e}")

    print(f"Results saved successfully to {OUTPUT_FILE_PATH}.")
    print("Program completed successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save S(q,w) QTM and S(q,w) GAAQC data to HDF5 file.")
    parser.add_argument(
        "gamma_file_path",
        type=str,
        help="Path to the HDF5 file containing gamma function data (used for both QTM and QC calculations).",
    )
    parser.add_argument(
        "md_file_path",
        type=str,
        help="Path to the HDF5 file containing MD simulation S(q,w) data (used for GAAQC calculation).",
    )
    parser.add_argument(
        "-e",
        "--element",
        type=str,
        choices=["H", "O"],
        default="H",
        help="Element type to process ('H' for Hydrogen, 'O' for Oxygen). Default is 'H'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/h2o_sqw_calc_data.h5",
        help="Output HDF5 file path to save the results. Default is 'data/h2o_sqw_calc_data.h5'.",
    )
    return parser


if __name__ == "__main__":
    main()
