import argparse
from typing import Final

import numpy as np
from helper import get_gamma_data, get_stc_model_data
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from numpy.typing import NDArray

from h2o_sqw_calc.core import HBAR, sqw_cdft, sqw_stc_model
from h2o_sqw_calc.utils import flow


def main():
    parser = argparse.ArgumentParser(description="Plot heatmap of S(Q, w) over a range of Q values.")
    parser.add_argument("--start", type=float, default=1, help="Start Q value (default: 1)")
    parser.add_argument("--end", type=float, default=80, help="End Q value (default: 80)")
    parser.add_argument("--step", type=float, default=1, help="Step size for Q values (default: 1)")
    args = parser.parse_args()

    ELEMENT: Final[str] = "H"
    T: Final[float] = 293  # unit: K
    Q_START: Final[float] = args.start  # unit: 1/angstrom
    Q_END: Final[float] = args.end  # unit: 1/angstrom
    Q_STEP: Final[float] = args.step  # unit: 1/angstrom

    freq_dos, dos = get_stc_model_data(ELEMENT)
    time, gamma = get_gamma_data(ELEMENT)
    dw = 2 * np.pi / np.diff(time).mean() / time.size

    q_vals: tuple[float, ...]
    omega_vals: tuple[NDArray[np.float64], ...]
    sqw_vals: tuple[NDArray[np.complex128], ...]
    q_vals, omega_vals, sqw_vals = flow(
        np.arange(Q_START, Q_END + Q_STEP, Q_STEP),
        lambda q_ruange: [(q, *sqw_cdft(q, time, gamma)) for q in q_ruange],
        lambda results: zip(*results),
    )

    print("All done!")

    omega: NDArray[np.float64] = flow(
        omega_vals,
        lambda omega_vals: ([omega_val[0] for omega_val in omega_vals], [omega_val[-1] for omega_val in omega_vals]),
        lambda minmax_arr: (min(minmax_arr[0]), max(minmax_arr[1])),
        lambda minmax: np.linspace(minmax[0], minmax[1], int((minmax[1] - minmax[0]) / dw) + 1),
    )

    print("Interpolating S(Q, w) values...")

    sqw = np.stack(
        flow(
            (sqw_vals, omega_vals),
            lambda sqw_omega: map(lambda sqw_val, omega_val: np.interp(omega, omega_val, sqw_val), *sqw_omega),
            # lambda sqw_interp: map(lambda sqw: (sqw - sqw.min()) / (sqw.max() - sqw.min()), sqw_interp),
            lambda sqw_norm: list(sqw_norm),
        ),
    )

    stc_max = np.array(
        [
            flow(
                q,
                lambda q: sqw_stc_model(q, omega, freq_dos, dos, T).argmax(),
                lambda idx: omega[idx],
            )
            for q in q_vals
        ]
    )

    print("Done!")
    print("Plotting heatmap...")

    plt.imshow(
        np.abs(sqw.T),
        aspect="auto",
        extent=(Q_START, Q_END, omega[0] * HBAR, omega[-1] * HBAR),
        origin="lower",
        cmap="magma",
        norm=LogNorm(),
    )
    plt.plot(q_vals, stc_max * HBAR, linestyle="--")
    plt.xlabel("Q (1/Å)")
    plt.ylabel("Energy (eV)")
    plt.title("Heatmap of S(Q, w) for H2O")
    plt.colorbar(label="S(Q, w) [arbitrary units]")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
