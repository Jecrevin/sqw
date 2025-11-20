"""Plot the SISF QC factor for q=1.63 Å⁻¹."""

from pathlib import Path

import numpy as np
from helper import plt_style_setup
from matplotlib import pyplot as plt
from scipy import constants


def main() -> None:
    """Plot the SISF QC factor for q=1.63 Å⁻¹."""
    data_file = Path(__file__).parents[2] / "data" / "results" / "sisf_qc_factor_q1.63.csv"
    out_dir = Path(__file__).parents[2] / "figs"

    time, sisf_qtm_abs, sisf_cls, sisf_qc_abs = np.loadtxt(data_file, delimiter=",", unpack=True)

    plt_style_setup()

    fig, sisf_axis = plt.subplots()
    sisf_qc_axis = sisf_axis.twinx()

    line_qtm = sisf_axis.semilogx(time / constants.pico, sisf_qtm_abs)
    line_cls = sisf_axis.semilogx(time / constants.pico, sisf_cls)
    line_qc = sisf_qc_axis.semilogx(time / constants.pico, sisf_qc_abs, color="k", ls=":")

    sisf_axis.legend(
        handles=[line_qtm[0], line_cls[0], line_qc[0]],
        labels=["Quantum SISF", "Classical SISF", "SISF QC Factor"],
        loc="upper right",
    )
    sisf_axis.set_xlim(right=10)
    sisf_axis.set_ylim(0, 1.2)
    sisf_qc_axis.set_ylim(0.9, 1.1)
    sisf_qc_axis.set_yticks(np.linspace(0.9, 1.1, 7, endpoint=True))
    sisf_qc_axis.grid(False)
    sisf_axis.set_xlabel("Time (ps)")
    sisf_axis.set_ylabel("Absolute value of SISF for H")
    sisf_qc_axis.set_ylabel("Absolute value of QC Factor for H")

    fig.tight_layout()
    fig.savefig(out_dir / "sisf_qc_factor_1.63.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
