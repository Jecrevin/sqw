from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

from sqw.core import sqw_cdft, sqw_stc_model
from sqw.io import get_data_from_h5py
from sqw.math import even_extend, odd_extend


def get_gamma_data(element: str) -> tuple[NDArray[np.float64], NDArray[np.complexfloating]]:
    time: NDArray[np.float64] = odd_extend(get_data_from_h5py(f"data/last_{element}.gamma", "time_vec"))
    gamma_qtm_re: NDArray[np.float64] = even_extend(get_data_from_h5py(f"data/last_{element}.gamma", "gamma_qtm_real"))
    gamma_qtm_im: NDArray[np.float64] = odd_extend(get_data_from_h5py(f"data/last_{element}.gamma", "gamma_qtm_imag"))
    gamma_qtm = gamma_qtm_re + 1j * gamma_qtm_im

    return time, gamma_qtm


def get_stc_model_data(element: str) -> tuple[NDArray, NDArray]:
    freq_dos = get_data_from_h5py("data/last.sqw", f"inc_omega_{element}")
    dos = get_data_from_h5py("data/last.sqw", f"inc_vdos_{element}")
    return freq_dos, dos


def main():
    ELEMENT = "H"
    Q = 20.0  # unit: 1/angstrom
    T = 293  # unit: K

    time, gamma = get_gamma_data(ELEMENT)
    freq_dos, dos = get_stc_model_data(ELEMENT)

    omega, res_cdft = sqw_cdft(Q, time, gamma)
    # omega, res_cdft = sqw_cdft(Q, time, gamma * np.hanning(gamma.size))
    res_stc = sqw_stc_model(Q, omega, freq_dos, dos, T)

    plt.plot(omega, np.abs(res_cdft), label="CDFT Result")
    plt.plot(omega, res_stc, label="STC Model Result", linestyle="--")
    plt.xlabel("Angular Frequency")
    plt.title("Comparison of CDFT and STC Model Results")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
