from collections.abc import Callable
from functools import lru_cache
from typing import Final

import numpy as np
import scipy.constants as consts
from numpy.typing import NDArray
from scipy.constants import pi as PI

from .math import (
    continuous_fourier_transform,
    is_all_array_1d,
    self_linear_convolve,
    self_linear_convolve_x_axis,
    trim_function,
)
from .typing import Array1D

HBAR: Final[float] = consts.value("reduced Planck constant in eV s")  # unit: eV·s
KB: Final[float] = consts.value("Boltzmann constant in eV/K")  # unit: eV/K
NEUTRON_MASS: Final[float] = (
    consts.value(key="neutron mass energy equivalent in MeV") * consts.mega / (consts.c / consts.angstrom) ** 2
)  # unit: eV/(Å/s)^2


def sqw_stc_model(
    q: float,
    w: Array1D,
    freq_dos: Array1D,
    density_of_states: Array1D,
    temperature: float,
    mass_num: int = 1,
) -> Array1D[np.float64]:
    """
    Calculate Scattering function S(q, w) using Short Time Collision Approximation (STC) model.
    """
    if not (q > 0):
        raise ValueError("Momentum transfer `q` must be greater than 0!")
    if not is_all_array_1d(w, freq_dos, density_of_states):
        raise ValueError("Input arrays must be all one-dimentional!")
    if not (temperature >= 0):
        raise ValueError("Temperature is in Kelvin and must be geater than or equal to 0!")
    if not (isinstance(mass_num, int) and mass_num > 0):
        raise ValueError("Mass number must be a positive integer!")

    hbar_beta = HBAR / (KB * temperature)
    t_eff = temperature * np.trapezoid(
        np.divide(
            density_of_states * hbar_beta * freq_dos,
            2 * np.tanh(hbar_beta * freq_dos / 2),
            out=density_of_states.copy(),
            where=(~np.isclose(freq_dos, 0, atol=0)),
        ),
        freq_dos,
    )
    beta_eff = 1 / (KB * t_eff)
    recoil_energy = (HBAR * q) ** 2 / (2 * mass_num * NEUTRON_MASS)
    result = (
        HBAR
        * np.sqrt(beta_eff / (4 * PI * recoil_energy))
        * np.exp(-beta_eff / (4 * recoil_energy) * (HBAR * w - recoil_energy) ** 2)
        * np.exp(-HBAR * w * beta_eff)
    )

    return result


@lru_cache
def _sqw_cdft_recursive[T: np.complexfloating, U: np.floating](
    q: float,
    gamma_tuple: tuple[T, ...],
    omega_tuple: tuple[U, ...],
    dt: float,
    dw: float,
    logger: Callable[[str], None] | None,
) -> tuple[Array1D[np.floating], NDArray[np.complex128]]:
    """Helper function to calculate S(q, w) using CDFT with logging."""
    gamma: Array1D[T] = np.array(gamma_tuple)
    omega: Array1D[U] = np.array(omega_tuple)

    if logger:
        logger(f"Calculating CDFT recursively for q = {q:.2f} ...")

    if q <= 5:
        result = omega, continuous_fourier_transform(np.exp(-(q**2) * gamma / 2), 1 / dt) / (2 * PI)

        if logger:
            logger(f"=> done with q = {q:.2f}.")

        return result

    recur_q = round(q / np.sqrt(2), 8)
    x_recur, y_recur = trim_function(
        *_sqw_cdft_recursive(recur_q, gamma_tuple, omega_tuple, dt, dw, logger), cut_ratio=1e-9
    )

    results = self_linear_convolve_x_axis(x_recur), self_linear_convolve(y_recur, dw).astype(np.complex128)

    if logger:
        logger(f"=> done with q = {q:.2f}.")

    return results


def sqw_cdft[T: np.floating, U: np.complexfloating](
    q: float,
    time_vec: Array1D[T],
    gamma: Array1D[U],
    *,
    logger: Callable[[str], None] | None = print,
) -> tuple[Array1D[np.floating], Array1D[np.complex128]]:
    if not (q > 0):
        raise ValueError("Momentum transfer `q` must be greater than 0!")
    if not is_all_array_1d(time_vec, gamma):
        raise ValueError("Input arrays must be one-dimensional!")

    dt = time_vec[1] - time_vec[0]
    omega = np.fft.fftshift(np.fft.fftfreq(time_vec.size, dt)) * 2 * PI
    dw = omega[1] - omega[0]

    gamma_tuple = tuple(gamma)
    omega_tuple = tuple(omega)

    result = _sqw_cdft_recursive(q, gamma_tuple, omega_tuple, dt, dw, logger=logger)

    if logger:
        logger(f"S(q, w) calculation for q = {q:.2f} completed.")

    return result
