"""Calculate the S(Q, ω) from 1 to 100 as 2d array.

This script calculates the S(Q, ω) for Q values from 1 to 100 and saves the
results in an HDF5 file. It uses the gamma data from a specified input file and
applies a correction for detailed balance at a temperature of 293 K.
"""

from functools import partial
from pathlib import Path

import h5py
import numpy as np
from helper import assure_detailed_balance, read_gamma_data

from sqw import sqw_ga_model
from sqw.consts import HBAR, NEUTRON_MASS


def main() -> None:
    """Calculate S(Q, ω) for Q values from 1 to 100 and save results."""
    data_dir = Path(__file__).parents[2] / "data"
    input_dir = data_dir / "molecular_dynamics"
    out_dir = data_dir / "results"

    out_dir.mkdir(exist_ok=True)

    time, gamma_qtm, _ = read_gamma_data(
        input_dir / "hydrogen_293k_gamma.h5", keys=["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"]
    )

    q_vals = np.arange(1, 101)

    omega = np.linspace(-10 / HBAR, 2 / HBAR, num=1000)
    sqw_ga_results = np.vstack(
        [
            np.interp(
                omega,
                *sqw_ga_model(q, time, gamma_qtm, correction=partial(assure_detailed_balance, temperature=293.0)),  # type: ignore
                left=np.nan,
                right=np.nan,
            )
            for q in q_vals
        ]
    )
    omega_where_stc_max = np.array([-HBAR * q**2 / (2 * NEUTRON_MASS) for q in q_vals])

    with h5py.File(out_dir / "sqw_ga_2d.h5", "w") as f:
        f.create_dataset("omega", data=omega)
        f.create_dataset("q_vals", data=q_vals)
        f.create_dataset("sqw_ga", data=sqw_ga_results)
        f.create_dataset("omega_where_stc_max", data=omega_where_stc_max)


if __name__ == "__main__":
    main()
