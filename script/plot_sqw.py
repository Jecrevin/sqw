import argparse
from typing import Final

import numpy as np
from helper import get_gamma_data, get_stc_model_data
from matplotlib import pyplot as plt

from h2o_sqw_calc.core import sqw_cdft, sqw_stc_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CDFT and STC model results for a given Q value.")
    parser.add_argument("q-value", type=float, help="Q value")
    return parser.parse_args()


def main():
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


if __name__ == "__main__":
    main()
