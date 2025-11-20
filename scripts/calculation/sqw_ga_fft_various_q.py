"""Calculate S(Q,ω) using FFT for various Q values from gamma(t) data.

This script reads gamma(t) data from an HDF5 file, computes the
self-intermediate scattering function F(Q,t) for a range of Q values, performs
FFT to obtain S(Q,ω), and saves the results to a CSV file.
"""

from pathlib import Path

import numpy as np
from helper import read_gamma_data

from sqw._math import continuous_fourier_transform
from sqw.consts import PI


def main() -> None:
    """Calculate S(Q,ω) GA model using FFT for various Q values and save to a csv file."""
    data_dir = Path(__file__).parents[2] / "data"
    input_dir = data_dir / "molecular_dynamics"
    out_dir = data_dir / "results"

    out_dir.mkdir(exist_ok=True)

    time, gamma_qtm, _ = read_gamma_data(
        input_dir / "hydrogen_293k_gamma.h5", keys=["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"]
    )

    q_vals = np.linspace(10, 20, 6, endpoint=True)

    sisfs = [np.exp(-(q**2) / 2 * gamma_qtm) for q in q_vals]
    sqw_ga_fft_results = [continuous_fourier_transform(time, sisf) for sisf in sisfs]
    omega_duplicated, raw_list_fft = zip(*sqw_ga_fft_results, strict=True)
    omega = omega_duplicated[0]  # all omega are the same
    sqw_ga_fft_list = [np.real(f) / (2 * PI) for f in raw_list_fft]

    np.savetxt(
        out_dir / "sqw_ga_fft_various_q.csv",
        np.column_stack((omega, *sqw_ga_fft_list)),
        delimiter=",",
        header="Scattering function S(Q,ω) at Q=[10, 12, ..., 20] values calculated using FFT\nomega, "
        + ", ".join(f"fft_result_q{q:.1f}" for q in q_vals),
    )


if __name__ == "__main__":
    main()
