import argparse

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft, fftfreq, fftshift

from .core import sqw_cdft, sqw_stc_model
from .io import get_gamma_data, get_stc_model_data


def main():
    parser = argparse.ArgumentParser(description="Compare CDFT and STC model results for a given Q value.")
    parser.add_argument("q_value", type=float, help="Q value")
    args = parser.parse_args()

    ELEMENT = "H"
    Q = args.q_value  # unit: 1/angstrom
    T = 293  # unit: K

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
        if isinstance(item, float) or isinstance(item, int):
            q_values.append(float(item))
        elif isinstance(item, str) and "-" in item:
            try:
                start, end = map(int, item.split("-", 1))
                q_values.extend(range(start, end + 1))
            except Exception:
                continue
        else:
            try:
                q_values.append(float(item))
            except Exception:
                continue

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
