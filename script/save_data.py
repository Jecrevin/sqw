import sys
from typing import Final, Literal

import h5py
import numpy as np
from helper import get_gamma_data
from scipy.interpolate import CubicSpline

from h2o_sqw_calc.core import HBAR, sqw_cdft, sqw_gaaqc, sqw_qtm_correction_factor
from h2o_sqw_calc.io import get_data_from_h5py
from h2o_sqw_calc.typing import Array1D


def main() -> None:
    ELEMENT: Final[Literal["H", "O"]] = "H"
    MD_DATA_FILE: Final[str] = "data/merged_h2o_293k.sqw"

    try:
        q_vals = get_data_from_h5py(MD_DATA_FILE, f"qVec_{ELEMENT}")
        omega_md = get_data_from_h5py(MD_DATA_FILE, f"inc_omega_{ELEMENT}")
        sqw_md_vals = get_data_from_h5py(MD_DATA_FILE, f"inc_sqw_{ELEMENT}")
        time_vec, gamma_qtm, gamma_cls = get_gamma_data(ELEMENT, include_classical=True)
    except Exception as e:
        sys.exit(f"Error loading data: {e}")

    assert gamma_cls is not None, "`include_classical` must set to True to get classical data."

    omega_cdft, sqw_cdft_data = zip(
        *map(lambda q: _interpolate_data(*sqw_cdft(q, time_vec, gamma_qtm)), q_vals), strict=True
    )
    omega_qc, sqw_qc_data = zip(
        *map(lambda q: _interpolate_data(*sqw_qtm_correction_factor(q, time_vec, gamma_qtm, gamma_cls)), q_vals),
        strict=True,
    )
    omega_gaaqc, sqw_gaaqc_data = zip(
        *map(
            lambda q, sqw_md: _interpolate_data(*sqw_gaaqc(q, time_vec, gamma_qtm, gamma_cls, omega_md, sqw_md)),
            q_vals,
            sqw_md_vals,
        ),
        strict=True,
    )

    with h5py.File("data/h2o_sqw_results.h5", "w") as f:
        f.create_dataset("qVals", data=q_vals)
        f.create_dataset("omega_cdft", data=omega_cdft)
        f.create_dataset("sqw_cdft", data=sqw_cdft_data)
        f.create_dataset("omega_qtm_correction", data=omega_qc)
        f.create_dataset("sqw_qtm_correction", data=sqw_qc_data)
        f.create_dataset("omega_gaaqc", data=omega_gaaqc)
        f.create_dataset("sqw_gaaqc", data=sqw_gaaqc_data)


def _interpolate_data(omega: Array1D, data: Array1D, data_num: int = 3000):
    omega_min, omega_max = -10 / HBAR, 10 / HBAR
    omega_interped = np.linspace(omega_min, omega_max, data_num)
    data_interped = CubicSpline(omega, data)(omega_interped)[:]

    return omega_interped, data_interped


if __name__ == "__main__":
    main()
