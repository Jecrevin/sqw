"""Calculate S(Q,ω) GA CDFT and STC models for various Q values and save to csv files.

This script reads gamma(t) data and DOS data from HDF5 files, computes the
self-intermediate scattering function F(Q,t) for a range of Q values using both
GA CDFT and STC models, and saves the results to CSV files.
"""

from functools import partial
from pathlib import Path

import numpy as np
from helper import assure_detailed_balance, read_gamma_data, read_stc_data

from sqw import sqw_ga_model, sqw_stc_model


def main() -> None:
    """Calculate S(Q,ω) GA CDFT and STC models for various Q values and save to csv files."""
    data_dir = Path(__file__).parents[2] / "data"
    input_dir = data_dir / "molecular_dynamics"
    out_dir = data_dir / "results" / "sqw_ga_cdft_stc_comparison"

    out_dir.mkdir(parents=True, exist_ok=True)

    time, gamma_qtm, _ = read_gamma_data(
        input_dir / "hydrogen_293k_gamma.h5", keys=["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"]
    )
    omega_vdos, vdos = read_stc_data(input_dir / "h2o_293k_vdos.h5", keys=["inc_omega_H", "inc_vdos_H"])

    q_vals = np.arange(10, 90, 10)  # unit: Å⁻¹
    temperature = 293.0  # unit: K

    assure_db_at_293k = partial(assure_detailed_balance, temperature=temperature)

    sqw_ga_cdft_results = [sqw_ga_model(q, time, gamma_qtm, correction=assure_db_at_293k, window=True) for q in q_vals]  # type: ignore
    omega_list, sqw_ga_cdft_list = zip(*sqw_ga_cdft_results, strict=True)
    sqw_stc_list = [
        sqw_stc_model(q, omega, omega_vdos, vdos, temperature)  # type: ignore
        for q, omega in zip(q_vals, omega_list, strict=True)
    ]

    for i, q in enumerate(q_vals):
        np.savetxt(
            out_dir / f"sqw_ga_cdft_stc_q{q:.1f}.csv",
            np.column_stack((omega_list[i], sqw_ga_cdft_list[i], sqw_stc_list[i])),
            delimiter=",",
            header=f"Scattering function S(Q,ω) at Q={q:.1f} Å⁻¹ calculated using GA CDFT and STC models\n"
            + "omega, ga_cdft_result, stc_result",
        )


if __name__ == "__main__":
    main()
