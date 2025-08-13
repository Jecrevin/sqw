import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from sqw.io import get_data_from_h5py
from sqw.math import even_extend, odd_extend
from sqw.utils import flow


def get_gamma_data(file_path: str) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    t_vec: NDArray[np.float64] = flow(file_path, lambda fp: get_data_from_h5py(fp, "time_vec"), odd_extend)
    gamma_qtm_real: NDArray[np.float64] = flow(
        file_path, lambda fp: get_data_from_h5py(fp, "gamma_qtm_real"), even_extend
    )
    gamma_qtm_imag: NDArray[np.float64] = flow(
        file_path, lambda fp: get_data_from_h5py(fp, "gamma_qtm_imag"), odd_extend
    )
    gamma_qtm = gamma_qtm_real + 1j * gamma_qtm_imag

    return t_vec, gamma_qtm


def plot_gamma(gamma_data: tuple[NDArray[np.float64], NDArray[np.complex128]]) -> None:
    t_vec, gamma_qtm = gamma_data

    plt.plot(t_vec, np.abs(gamma_qtm), label="Amplitude")
    plt.plot(t_vec, gamma_qtm.real, label="Real part")
    plt.plot(t_vec, gamma_qtm.imag, label="Imaginary part")
    plt.xlabel("Time")
    plt.title("Gamma data")
    plt.grid()
    plt.legend()

    plt.show()


def main():
    flow(
        "data/last_H.gamma",
        get_gamma_data,
        plot_gamma,
    )


if __name__ == "__main__":
    main()
