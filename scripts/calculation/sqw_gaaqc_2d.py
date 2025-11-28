"""Calculate the S(Q, ω) from 1 to 100 as 2d array.

This script calculates the S(Q, ω) for Q values from 1 to 100 and saves the
results in an HDF5 file. It uses the gamma data from a specified input file and
applies a correction for detailed balance at a temperature of 293 K.
"""

from functools import partial
from pathlib import Path

import h5py
import numpy as np
from helper import assure_detailed_balance, read_gamma_data, read_md_data

from sqw import sqw_gaaqc_model
from sqw.consts import HBAR, NEUTRON_MASS


def main() -> None:
    """Calculate S(Q, ω) for Q values from 1 to 100 and save results."""
    data_dir = Path(__file__).parents[2] / "data"
    input_dir = data_dir / "molecular_dynamics"
    out_dir = data_dir / "results"

    out_dir.mkdir(parents=True, exist_ok=True)

    time, gamma_qtm, gamma_cls = read_gamma_data(
        input_dir / "hydrogen_293k_gamma.h5",
        keys=["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"],
        include_cls=True,
    )

    assert gamma_cls is not None, "`include_cls` must be True to get `gamma_cls`."

    q_vals, omega_md, sqw_md_stack = read_md_data(
        input_dir / "h2o_293k_sqw_md.h5", keys=["qVec_H", "inc_omega_H", "inc_sqw_H"]
    )

    omega = np.linspace(-10 / HBAR, 2 / HBAR, num=1000)
    sqw_gaaqc_results = np.vstack(
        [
            np.interp(
                omega,
                *sqw_gaaqc_model(
                    q,  # type: ignore
                    time,
                    gamma_qtm,
                    gamma_cls,
                    omega_md,
                    sqw_md,
                    window=True,
                    correction=partial(assure_detailed_balance, temperature=293.0),
                ),
                left=np.nan,
                right=np.nan,
            )
            for q, sqw_md in zip(q_vals, sqw_md_stack, strict=True)
        ]
    )
    omega_where_stc_max = np.array([-HBAR * q**2 / (2 * NEUTRON_MASS) for q in q_vals])

    with h5py.File(out_dir / "sqw_gaaqc_2d.h5", "w") as f:
        f.create_dataset("omega", data=omega)
        f.create_dataset("q_vals", data=q_vals)
        f.create_dataset("sqw_gaaqc", data=sqw_gaaqc_results)
        f.create_dataset("omega_where_stc_max", data=omega_where_stc_max)


if __name__ == "__main__":
    main()
