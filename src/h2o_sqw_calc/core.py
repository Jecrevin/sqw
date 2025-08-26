from collections.abc import Callable
from functools import lru_cache
from typing import Final

import numpy as np
import scipy.constants as consts
from numpy.typing import NDArray
from scipy.constants import pi as PI

from .math import continuous_fourier_transform, self_linear_convolve, self_linear_convolve_x_axis
from .utils import flow

HBAR: Final[float] = consts.value("reduced Planck constant in eV s")  # unit: eV·s
KB: Final[float] = consts.value("Boltzmann constant in eV/K")  # unit: eV/K
NEUTRON_MASS: Final[float] = (
    consts.value(key="neutron mass energy equivalent in MeV") * consts.mega / (consts.c / consts.angstrom) ** 2
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


@lru_cache
def _sqw_cdft_recursive(
    q: float,
    gamma_tuple: tuple[NDArray[np.complexfloating], ...],
    omega_tuple: tuple[NDArray[np.floating], ...],
    dt: float,
    dw: float,
    logger: Callable[[str], None] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Helper function to calculate S(q, w) using CDFT with logging."""
    gamma = np.array(gamma_tuple)
    omega = np.array(omega_tuple)
    if logger:
        logger(f"Calculating CDFT recursively for q = {q:.2f} ...")

    if q <= 5:
        return omega, continuous_fourier_transform(np.exp(-(q**2) * gamma / 2), 1 / dt) / (2 * PI)

    next_q = round(q / np.sqrt(2), 8)
    recur_res = _sqw_cdft_recursive(next_q, gamma_tuple, omega_tuple, dt, dw, logger)

    x_recur, y_recur = recur_res
    y_abs = np.abs(y_recur)
    threshold = np.max(y_abs) * 1e-9
    significant_indices = np.where(y_abs > threshold)[0]

    if significant_indices.size > 0:
        start, end = significant_indices[0], significant_indices[-1] + 1
        if start > 0 or end < x_recur.size:
            x_recur = x_recur[start:end]
            y_recur = y_recur[start:end]

    # Ensure even length for convolution if needed, though self_linear_convolve might handle it.
    # This is a good practice for some FFT-based algorithms.
    if x_recur.size % 2 != 0:
        x_recur = np.append(x_recur, x_recur[-1] + (x_recur[-1] - x_recur[-2]))
        y_recur = np.append(y_recur, 0j)

    if logger:
        logger(f"=> done with q = {next_q:.2f}, now back to results for q = {q:.2f} ...")

    return self_linear_convolve_x_axis(x_recur), self_linear_convolve(y_recur, dw)


def sqw_cdft(
    q: float,
    time_vec: NDArray[np.floating],
    gamma: NDArray[np.complexfloating],
    logger: Callable[[str], None] | None = print,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    dt = time_vec[1] - time_vec[0]
    omega = np.fft.fftshift(np.fft.fftfreq(time_vec.size, dt)) * 2 * PI
    dw = omega[1] - omega[0]

    gamma_tuple = tuple(gamma)
    omega_tuple = tuple(omega)

    return _sqw_cdft_recursive(q, gamma_tuple, omega_tuple, dt, dw, logger=logger)
