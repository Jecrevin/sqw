import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Final

import h5py
import numpy as np
from utils.helper import assure_detailed_balance
from utils.io import read_gamma_data, read_md_data

from sqw._core import sqw_ga_model, sqw_gaaqc_model
from sqw.consts import HBAR, PI


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    OUTPUT: Final[Path] = Path(args.output)
    GAMMA_FILE: Final[str] = args.gamma_file
    MD_FILE: Final[str] = args.md_file
    GAMMA_KEYS: Final[list[str]] = args.gamma_keys
    MD_KEYS: Final[list[str]] = args.md_keys

    try:
        time_ga, gamma_qtm, gamma_cls = read_gamma_data(GAMMA_FILE, GAMMA_KEYS, no_extend=False, include_cls=True)
        assert gamma_cls is not None, "Key param 'include_cls' for `read_gamma_data` must be True!"
    except Exception as e:
        sys.exit(f"Error reading width function data: {e}")

    try:
        q_vals, omega_md, raw_sqw_md_stack = read_md_data(MD_FILE, MD_KEYS, no_extend=False)
        sqw_md_stack = raw_sqw_md_stack * np.sqrt(2 * PI)
    except Exception as e:
        sys.exit(f"Error reading MD data: {e}")

    assure_db_at_293k = partial(assure_detailed_balance, temperature=293.0)

    print("Calculating S(q,w) using GA and GAAQC models...")

    ga_results = [sqw_ga_model(q, time_ga, gamma_qtm, correction=assure_db_at_293k) for q in q_vals]  # type: ignore
    gaaqc_results = [
        sqw_gaaqc_model(q, time_ga, gamma_qtm, gamma_cls, omega_md, sqw_md, correction=assure_db_at_293k)  # type: ignore
        for q, sqw_md in zip(q_vals, sqw_md_stack, strict=True)
    ]
    
    print("Calculations complete.")
    print("Interpolating data onto common omega grid...")
    
    omega_grid = np.linspace(-10 / HBAR, 2 / HBAR, 3000)
    sqw_ga_vstack = np.vstack([np.interp(omega_grid, omega, sqw, left=0, right=0) for omega, sqw in ga_results])
    sqw_gaaqc_vstack = np.vstack([np.interp(omega_grid, omega, sqw, left=0, right=0) for omega, sqw in gaaqc_results])

    print("Interpolation complete.")
    print(f"Saving results to '{OUTPUT}'...")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        print(f"Warning: Output file '{OUTPUT}' already exists! Do you want to overwrite it? (y/N)")
        response = input().strip().lower()
        if response != "y":
            sys.exit("Operation cancelled by user.")
    if OUTPUT.suffix not in {".h5", ".hdf5"}:
        sys.exit("Output file must have .h5 or .hdf5 extension!")
    with h5py.File(OUTPUT, "w") as f:
        f.create_dataset("q_values", data=q_vals)
        f.create_dataset("omega_grid", data=omega_grid)
        f.create_dataset("sqw_ga", data=sqw_ga_vstack)
        f.create_dataset("sqw_gaaqc", data=sqw_gaaqc_vstack)

    print("Data saved successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save data to a file.")

    parser.add_argument("output", help="Output file path.")
    parser.add_argument("gamma_file", help="Input gamma data file.")
    parser.add_argument("md_file", help="Input MD S(q,w) data file.")
    parser.add_argument(
        "--gamma-keys",
        nargs=4,
        default=["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"],
        help="Keys for gamma data in HDF5 file.",
    )
    parser.add_argument(
        "--md-keys",
        nargs=3,
        default=["qVec_H", "inc_omega_H", "inc_sqw_H"],
        help="Keys for MD S(q,w) data in HDF5 file.",
    )

    return parser


if __name__ == "__main__":
    main()
