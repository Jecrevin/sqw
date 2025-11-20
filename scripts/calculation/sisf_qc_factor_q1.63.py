"""Calculate SISF QC factor at Q=1.63 Å⁻¹ and save results to a CSV file."""

from pathlib import Path

import numpy as np
from helper import read_gamma_data


def main() -> None:
    """Calculate SISF QC factor at Q=1.63 Å⁻¹ and save results to a CSV file."""
    data_dir = Path(__file__).parents[2] / "data"
    input_dir = data_dir / "molecular_dynamics"
    out_dir = data_dir / "results"

    out_dir.mkdir(exist_ok=True)

    time, gamma_qtm, gamma_cls = read_gamma_data(
        input_dir / "hydrogen_293k_gamma.h5",
        keys=["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"],
        include_cls=True,
    )

    assert gamma_cls is not None, "`gamma_cls` should not be None when `include_cls` is True."

    q = 1.63  # unit: Å⁻¹
    sisf_qtm = np.exp(-(q**2) / 2 * gamma_qtm)
    sisf_cls = np.exp(-(q**2) / 2 * gamma_cls)
    sisf_qc = np.exp(-(q**2) / 2 * (gamma_qtm - gamma_cls))

    np.savetxt(
        out_dir / "sisf_qc_factor_q1.63.csv",
        np.column_stack((time, np.abs(sisf_qtm), sisf_cls, np.abs(sisf_qc))),
        delimiter=",",
        header="SISF QC Factor Calculation at q=1.63 Å⁻¹\ntime, sisf_qtm_abs, sisf_cls, sisf_qc_abs",
    )


if __name__ == "__main__":
    main()
