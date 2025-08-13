from functools import cache

import numpy as np
import scipy.constants as consts
from numpy.typing import NDArray
from scipy.constants import pi as PI

from .math import continuous_fourier_transform, self_linear_convolve, self_linear_convolve_x_axis
from .utils import flow

HBAR: float = consts.value("reduced Planck constant in eV s")  # unit: eV·s
KB: float = consts.value("Boltzmann constant in eV/K")  # unit: eV/K
NEUTRON_MASS: float = (
    consts.value("neutron mass energy equivalent in MeV") * consts.mega / (consts.c / consts.angstrom) ** 2
)  # unit: eV/(Å/s)^2


def sqw_stc_model(
    q: float,
    w: NDArray,
    freq_dos: NDArray,
    density_of_states: NDArray,
    temperature: float,
    mass_num: int = 1,
) -> NDArray:
    beta_eff: float = flow(
        temperature,
        lambda t: HBAR / (KB * t),
        lambda hbar_beta: temperature
        * np.trapezoid(
            np.divide(
                density_of_states * hbar_beta * freq_dos,
                2 * np.tanh(hbar_beta * freq_dos / 2),
                out=density_of_states.copy(),
                where=(~np.isclose(freq_dos, 0, atol=0)),
            ),
            freq_dos,
        ),
        lambda t_eff: 1 / (KB * t_eff),
    )
    return flow(
        q,
        lambda q: (HBAR * q) ** 2 / (2 * mass_num * NEUTRON_MASS),
        lambda recoil_energy: HBAR
        * np.sqrt(beta_eff / (4 * PI * recoil_energy))
        * np.exp(-beta_eff / (4 * recoil_energy) * (HBAR * w - recoil_energy) ** 2)
        * np.exp(-HBAR * w * beta_eff),
    )


def sqw_cdft(q: float, time_vec: NDArray, gamma: NDArray[np.complexfloating]):
    omega = np.fft.fftshift(np.fft.fftfreq(time_vec.size, (dt := time_vec[1] - time_vec[0]))) * 2 * PI

    @cache
    def _cdft(q: float):
        print(f"Calculating CDFT recursively for q = {q:.2f} ...")

        if q <= 5:
            return omega, continuous_fourier_transform(np.exp(-(q**2) * gamma / 2), 1 / dt) / (2 * PI)

        dw = omega[1] - omega[0]
        recur_res = _cdft(q / np.sqrt(2))

        print(f"=> done with q = {q / np.sqrt(2):.2f}, now back to results for q = {q:.2f} ...")

        return self_linear_convolve_x_axis(recur_res[0]), self_linear_convolve(recur_res[1], dw)

    return _cdft(q)
