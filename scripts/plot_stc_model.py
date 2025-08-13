import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from sqw.core import sqw_stc_model
from sqw.io import get_data_from_h5py
from sqw.utils import flow


def get_stc_model_data(file_path: str, element: str = "H") -> tuple[NDArray, NDArray]:
    freq_dos = get_data_from_h5py(file_path, f"inc_omega_{element}")
    dos = get_data_from_h5py(file_path, f"inc_vdos_{element}")
    return freq_dos, dos


def calc_sqw_data(stc_data: tuple[NDArray, NDArray], q: float, w: NDArray, temperature: float, mass_num: int = 1):
    freq_dos, density_of_states = stc_data
    return w, sqw_stc_model(q, w, freq_dos, density_of_states, temperature, mass_num)


def plot_sqw_stc_data(sqw_stc_data: tuple[NDArray, NDArray]):
    omega, sqw = sqw_stc_data

    plt.plot(omega, sqw, label="S(Q, ω)")
    plt.xlabel("ω")
    plt.ylabel("S(Q, ω)")
    plt.title("S(Q, ω) vs ω")
    plt.grid()
    plt.legend()

    plt.show()


def main():
    Q = 40  # unit: 1/angstrom
    T = 293  # unit: K

    omega = np.linspace(-1e16, 1e16, 10**6)

    flow(
        "data/last.sqw",
        get_stc_model_data,
        lambda stc_data: calc_sqw_data(stc_data, Q, omega, T),
        plot_sqw_stc_data,
    )


if __name__ == "__main__":
    main()
