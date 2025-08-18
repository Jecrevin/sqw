import argparse
import sys
from typing import Final

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from numpy.fft import fft, fftfreq, fftshift
from numpy.typing import NDArray

from .core import HBAR, sqw_cdft, sqw_stc_model
from .io import get_gamma_data, get_stc_model_data
from .utils import flow


def sqw():
    parser = argparse.ArgumentParser(description="Compare CDFT and STC model results for a given Q value.")
    parser.add_argument("q_value", type=float, help="Q value")
    args = parser.parse_args()

    ELEMENT: Final[str] = "H"
    Q: Final[float] = args.q_value  # unit: 1/angstrom
    T: Final[float] = 293  # unit: K

    time, gamma = get_gamma_data(ELEMENT)
    freq_dos, dos = get_stc_model_data(ELEMENT)

    omega, res_cdft = sqw_cdft(Q, time, gamma)
    # omega, res_cdft = sqw_cdft(Q, time, gamma * np.hanning(gamma.size))
    res_stc = sqw_stc_model(Q, omega, freq_dos, dos, T)

    plt.plot(omega, np.abs(res_cdft), label="CDFT Result")
    plt.plot(omega, res_stc, label="STC Model Result", linestyle="--")
    plt.xlabel("Angular Frequency")
    plt.title(f"Comparison of CDFT and STC Model Results for Q={Q}")
    plt.grid()
    plt.legend()
    plt.show()


def sqw_stack():
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


def plot_fft():
    """
    Plots the FFT of the intermediate scattering function for given Q values.
    Supports single values and ranges like 10-20.
    """
    parser = argparse.ArgumentParser(description="Plot FFT of exp(-Q^2*gamma/2) for given Q values.")
    parser.add_argument(
        "q_values",
        nargs="*",
        metavar="Q",
        help="Q values (1-80) to plot. Supports single values or ranges like 10-20.",
    )
    args = parser.parse_args()
    raw_qs = args.q_values if args.q_values else list(range(5, 11))

    # Parse q values and ranges
    q_values = []
    for item in raw_qs:
        if isinstance(item, str) and "-" in item:
            try:
                start, end = map(int, item.split("-", 1))
                if start > end:
                    print(
                        f"Error: Invalid range '{item}'. Start value must be less than or equal to end value.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                q_values.extend(range(start, end + 1))
            except ValueError:
                print(f"Error: Invalid range format '{item}'. Please use 'start-end'.", file=sys.stderr)
                sys.exit(1)
        else:
            try:
                q_values.append(float(item))
            except ValueError:
                print(f"Error: Invalid Q value '{item}'. Must be a number.", file=sys.stderr)
                sys.exit(1)

    if not q_values:
        q_values = list(range(5, 11))

    ELEMENT = "H"
    time, gamma = get_gamma_data(ELEMENT)

    dt = time[1] - time[0]
    freq = fftshift(fftfreq(time.size, d=dt))
    omega = 2 * np.pi * freq

    for q in q_values:
        f_of_t = np.exp(-(q**2) * gamma / 2)
        sqw = fftshift(fft(f_of_t))
        plt.plot(omega, np.abs(sqw), label=f"Q = {q:.2f}")

    plt.xlabel("Angular Frequency (rad/ps)")
    plt.ylabel("S(Q, w) [arbitrary units]")
    plt.title("FFT of Intermediate Scattering Function")
    plt.grid()
    plt.legend()
    plt.show()


def plot_gamma():
    """
    Plots the gamma function for element 'H'.
    """
    parser = argparse.ArgumentParser(description="Plot the gamma function for element 'H'.")
    parser.parse_args()

    element = "H"
    time, gamma_qtm = get_gamma_data(element)

    plt.plot(time, np.abs(gamma_qtm), label="Magnitude")
    plt.plot(time, np.real(gamma_qtm), label="Real Part", linestyle=":")
    plt.plot(time, np.imag(gamma_qtm), label="Imaginary Part", linestyle="--")
    plt.xlabel("Time (ps)")
    plt.ylabel("Gamma Function")
    plt.title(f"Gamma Function for {element}")
    plt.grid()
    plt.legend()
    plt.show()
