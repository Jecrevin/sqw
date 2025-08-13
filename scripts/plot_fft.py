from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

from sqw.io import get_data_from_h5py
from sqw.math import even_extend, odd_extend


def get_gamma_data(element: str) -> tuple[NDArray[np.float64], NDArray[np.complexfloating]]:
    time: NDArray[np.float64] = odd_extend(get_data_from_h5py(f"data/last_{element}.gamma", "time_vec"))
    gamma_qtm_re: NDArray[np.float64] = even_extend(get_data_from_h5py(f"data/last_{element}.gamma", "gamma_qtm_real"))
    gamma_qtm_im: NDArray[np.float64] = odd_extend(get_data_from_h5py(f"data/last_{element}.gamma", "gamma_qtm_imag"))
    gamma_qtm = gamma_qtm_re + 1j * gamma_qtm_im

    return time, gamma_qtm


def main():
    Q = np.expand_dims([5], axis=1)
    time, gamma = get_gamma_data("H")
    omega = np.fft.fftshift(np.fft.fftfreq(time.size, (dt := time[1] - time[0]))) * 2 * np.pi
    res_fft = np.fft.fftshift(np.fft.fft(np.exp(-(Q**2) * gamma / 2)), axes=-1) * dt / (2 * np.pi)

    for q, res in zip(np.take(Q, 0, 1), res_fft):
        plt.plot(omega, np.abs(res), label=f"Q = {q:.2f}")

    plt.xlabel("Angular Frequency")
    plt.title("FFT Spectrum")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
