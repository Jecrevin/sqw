"""Plot cross section results from CSV files."""

from pathlib import Path

import numpy as np
from helper import plt_style_setup
from matplotlib import pyplot as plt

from sqw.consts import PI


def main() -> None:
    """Plot cross section results from CSV files."""
    data_dir = Path(__file__).parents[2] / "data" / "results" / "cross_section"
    file_list = sorted(p for p in data_dir.glob("*.csv") if not p.name.endswith(".exp.csv"))
    out_dir = Path(__file__).parents[2] / "figs" / "cross_section"

    out_dir.mkdir(exist_ok=True)

    plt_style_setup()

    for file_path in file_list:
        eout, cs_ga, cs_gaaqc = np.loadtxt(file_path, delimiter=",", unpack=True)
        eout_exp, cs_exp = np.loadtxt(file_path.with_suffix(".exp.csv"), delimiter=",", unpack=True)

        for cs_data, cs_name in ((cs_ga, "ga"), (cs_gaaqc, "gaaqc")):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.semilogy(eout, cs_data * np.pow(2 * PI, 1.5), label=f"{cs_name.upper()} model")
            ax.scatter(eout_exp, cs_exp, label="Esch et al. Exp.", marker="x", color="black")
            ax.set_xlabel("Scattered energy (eV)")
            ax.set_ylabel("Cross section (b/eV/sr)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{cs_name}_{file_path.stem}.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
