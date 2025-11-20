"""Plot 2D S(q, ω) calculated using the GAAQC method."""

from pathlib import Path
from typing import cast

import h5py
from helper import plt_style_setup
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from sqw.consts import HBAR


def main() -> None:
    """Plot 2D S(q, ω) calculated using the GA method."""
    data_file = Path(__file__).parents[2] / "data/results/sqw_gaaqc_2d.h5"
    out_dir = Path(__file__).parents[2] / "figs"

    with h5py.File(data_file) as f:
        q_vals = cast(h5py.Dataset, f["q_vals"])[()]
        omega = cast(h5py.Dataset, f["omega"])[()]
        sqw_gaaqc_results = cast(h5py.Dataset, f["sqw_gaaqc"])[()]
        omega_where_stc_max = cast(h5py.Dataset, f["omega_where_stc_max"])[()]
    energy = omega * HBAR

    plt_style_setup()

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        q_vals,
        energy,
        sqw_gaaqc_results.T,
        cmap="jet",
        norm=LogNorm(vmin=1e-27),
    )
    ax.set_ylim(bottom=energy.min())
    ax.plot(q_vals, omega_where_stc_max * HBAR, color="gold", label="STC max", linestyle="--")
    ax.set_xlabel("Momentum Transfer $q$ ($\\mathrm{\\AA}^{-1}$)")
    ax.set_ylabel("Energy Transfer (eV)")
    ax.legend(loc="upper left", labelcolor="white")

    fig.colorbar(mesh, ax=ax, label=r"$S(q, \omega)$ (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)")
    fig.tight_layout()
    fig.savefig(out_dir / "sqw_gaaqc_2d.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
