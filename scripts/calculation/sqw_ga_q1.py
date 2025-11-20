"""Calculation of Scattering Function S(Q,ω) using FFT and CDFT results.

This script uses FFT and CDFT to calculate the scattering function S(Q,ω) from
self-intermediated correlation function F(Q,t) at Q=1.0 Å⁻¹, and saves the
result to a csv file.
"""

from functools import partial
from pathlib import Path

import numpy as np
from helper import assure_detailed_balance, read_gamma_data

from sqw import sqw_ga_model


def main() -> None:
    """Calculate S(Q,ω) GA model using FFT and CDFT and save to a csv file."""
    data_dir = Path(__file__).parents[2] / "data"
    input_dir = data_dir / "molecular_dynamics"
    out_dir = data_dir / "results"

    out_dir.mkdir(exist_ok=True)

    time, gamma_qtm, _ = read_gamma_data(
        input_dir / "hydrogen_293k_gamma.h5", keys=["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"]
    )

    q = 1.0  # unit: Å⁻¹
    temperature = 293.0  # unit: K
    omega, sqw_ga_fft = sqw_ga_model(
        q,
        time,
        gamma_qtm,
        correction=None,
        window=False,
    )
    _, sqw_ga_cdft = sqw_ga_model(
        q,
        time,
        gamma_qtm,
        correction=partial(assure_detailed_balance, temperature=temperature),
        window=True,
    )

    np.savetxt(
        out_dir / "sqw_ga_fft_cdft.csv",
        np.column_stack((omega, sqw_ga_fft, sqw_ga_cdft)),
        delimiter=",",
        header="Scattering function S(Q,ω) at Q=1.0 Å⁻¹ calculated using FFT and CDFT\nomega, fft_result, cdft_result",
    )


if __name__ == "__main__":
    main()
