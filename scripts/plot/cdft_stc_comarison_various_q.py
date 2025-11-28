"""Plot S(Q, ω) GA CDFT and STC model comparison for various Q values.

This script reads S(Q, ω) data from CSV files and generates a comparison plot
for different Q values. Each Q value shows both CDFT and STC results with the
same color but different line styles. The resulting plot is saved as a PNG file
in the 'figs' directory.
"""

from pathlib import Path

import numpy as np
from helper import plt_style_setup
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from sqw.consts import HBAR


def main() -> None:
    """Plot S(Q, ω) GA CDFT and STC model comparison for various Q values."""
    data_dir = Path(__file__).parents[2] / "data" / "results" / "sqw_ga_cdft_stc_comparison"
    fig_dir = Path(__file__).parents[2] / "figs"

    fig_dir.mkdir(parents=True, exist_ok=True)

    q_vals = np.arange(10, 90, 10)  # unit: Å⁻¹

    plt_style_setup()

    fig, ax = plt.subplots()

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(q_vals)))

    legend_elements = []
    for i, q in enumerate(q_vals):
        data_file = data_dir / f"sqw_ga_cdft_stc_q{q:.1f}.csv"
        data = np.loadtxt(data_file, delimiter=",")

        omega = data[:, 0]
        sqw_cdft = data[:, 1]
        sqw_stc = data[:, 2]

        color = colors[i]
        line_cdft = ax.plot(omega * HBAR, sqw_cdft, color=color)
        line_stc = ax.plot(omega * HBAR, sqw_stc, color=color, linestyle="--")
        legend_elements.append((line_cdft[0], line_stc[0]))

    ax.set_xlim((-10, 2))

    ax.legend(
        [tuple(elem) for elem in legend_elements],
        [f"Q={q} Å⁻¹ (CDFT/STC)" for q in q_vals],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper left",
    )

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Scattering Function (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)")

    fig.tight_layout()
    fig.savefig(fig_dir / "cdft_stc_comparison_various_q.png", dpi=300)

    # plt.show()


if __name__ == "__main__":
    main()
