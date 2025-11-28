"""Calculate S(Q, ω) using FFT and STC model at Q=40 Å⁻¹ and save results to CSV files.

This script reads gamma data and vibrational density of states (VDOS) from HDF5
files, computes the scattering function S(Q, ω) using both FFT and STC model
methods, and saves the results to CSV files for comparison.
"""

from pathlib import Path

import numpy as np
from helper import read_gamma_data, read_stc_data

from sqw import sqw_stc_model
from sqw._math import continuous_fourier_transform
from sqw.consts import HBAR, PI


def main() -> None:
    """Calculate S(Q, ω) using FFT and STC model at Q=40 Å⁻¹ and save results to CSV files."""
    data_dir = Path(__file__).parents[2] / "data"
    input_dir = data_dir / "molecular_dynamics"
    out_dir = data_dir / "results" / "fft_stc_comparison_q40"

    out_dir.mkdir(parents=True, exist_ok=True)

    time, gamma_qtm, _ = read_gamma_data(
        input_dir / "hydrogen_293k_gamma.h5", keys=["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"]
    )
    omega_vdos, vdos = read_stc_data(input_dir / "h2o_293k_vdos.h5", keys=["inc_omega_H", "inc_vdos_H"])

    q = 40.0  # unit: Å⁻¹
    temperature = 293.0  # unit: K

    sisf = np.exp(-(q**2) / 2 * gamma_qtm)
    omega_fft, raw_sqw_fft = continuous_fourier_transform(time, sisf)
    sqw_fft = raw_sqw_fft.real / (2 * PI)

    omega_stc = np.linspace(-10 / HBAR, 2 / HBAR, 1000)
    sqw_stc = sqw_stc_model(q, omega_stc, omega_vdos, vdos, temperature)

    np.savetxt(
        out_dir / "sqw_fft_q40.csv",
        np.column_stack((omega_fft, sqw_fft)),
        delimiter=",",
        header="Scattering Function FFT Calculation at q=40 Å⁻¹\nomega, sqw_fft",
    )
    np.savetxt(
        out_dir / "sqw_stc_q40.csv",
        np.column_stack((omega_stc, sqw_stc)),
        delimiter=",",
        header="Scattering Function STC Model at q=40 Å⁻¹\nomega, sqw_stc",
    )


if __name__ == "__main__":
    main()
