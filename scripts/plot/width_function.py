"""Plot the width function from molecular dynamics simulations.

This script reads the width function data from an HDF5 file and generates a
plot showing the amplitude, real part, and imaginary part of the width function
over time. An inset zooms in on the initial time region for better visibility.
"""

import importlib.util
from pathlib import Path

import numpy as np
from helper import plt_style_setup
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import constants


def _load_helper():
    helper_path = Path(__file__).parents[1] / "calculation" / "helper.py"

    spec = importlib.util.spec_from_file_location("helper", helper_path)
    helper = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(helper)  # type: ignore

    return helper


def main() -> None:
    """Plot the width function from molecular dynamics simulations."""
    data_file = Path(__file__).parents[2] / "data" / "molecular_dynamics" / "hydrogen_293k_gamma.h5"
    fig_dir = Path(__file__).parents[2] / "figs"

    time, gamma_qtm, _ = _load_helper().read_gamma_data(
        data_file, ["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"], include_cls=False
    )

    plt_style_setup()

    fig, ax = plt.subplots()

    ax.plot(time / constants.pico, np.abs(gamma_qtm), label="Amplitude")
    ax.plot(time / constants.pico, gamma_qtm.real, label="Real Part", ls="--")
    ax.plot(time / constants.pico, gamma_qtm.imag, label="Imaginary Part", ls="--")
    ax.legend(loc="lower right", bbox_to_anchor=(0, 0.05, 1, 1))
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Width Function ($\\mathrm{\\AA}^2$)")

    axins = inset_axes(
        ax,
        width="40%",
        height="40%",
        loc="upper center",
    )
    axins.plot(time / constants.pico, np.abs(gamma_qtm))
    axins.plot(time / constants.pico, np.real(gamma_qtm), linestyle="--")
    axins.plot(time / constants.pico, np.imag(gamma_qtm), linestyle="--")
    axins.set_xlim(-0.1, 0.1)
    axins.set_ylim(-0.05, 0.1)
    axins.set_xticks([-0.1, 0, 0.1])
    axins.set_yticks([-0.05, 0, 0.1])

    ax.indicate_inset_zoom(axins, edgecolor="gray")

    fig.savefig(fig_dir / "width_function.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
