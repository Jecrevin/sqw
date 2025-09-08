from collections.abc import Callable
from functools import lru_cache
from typing import Final

import numpy as np
import scipy.constants as consts
from scipy.constants import pi as PI
from scipy.interpolate import CubicSpline

from .math import (
    continuous_fourier_transform,
    is_all_array_1d,
    linear_convolve,
    linear_convolve_x_axis,
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
    Calculate Scattering function S(q,w) using Short Time Collision Approximation (STC) model.
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
) -> tuple[Array1D[np.floating], Array1D[np.complex128]]:
    """Helper function to calculate S(q,w) using CDFT with logging."""
    gamma: Array1D[T] = np.array(gamma_tuple)
    omega: Array1D[U] = np.array(omega_tuple)

    if logger:
        logger(f"Calculating CDFT recursively for {q = :.2f} ...")

    if q <= 5:
        result = omega, continuous_fourier_transform(np.exp(-(q**2) * gamma / 2), 1 / dt) / (2 * PI)

        if logger:
            logger(f"done with {q = :.2f}.")

        return result

    recur_q = round(q / np.sqrt(2), 8)
    x_recur, y_recur = trim_function(
        *_sqw_cdft_recursive(
            recur_q, gamma_tuple, omega_tuple, dt, dw, lambda s: logger("=> " + s) if logger else None
        ),
        cut_ratio=1e-9,
    )

    results = self_linear_convolve_x_axis(x_recur), self_linear_convolve(y_recur, dw).astype(np.complex128)

    if logger:
        logger(f"Done calculation for {q = :.2f}.")

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
        logger(f"S(q,w) calculation for {q = :.2f} all completed!")

    return result


def _get_sqw_qc(
    q: float, time_vec: Array1D, gamma_qtm: Array1D, gamma_cls: Array1D, logger: Callable[[str], None] | None
):
    dt = np.mean(np.diff(time_vec))
    omega = np.fft.fftshift(np.fft.fftfreq(time_vec.size, dt)) * 2 * PI
    dw = np.mean(np.diff(omega))

    gamma_tuple = tuple(gamma_qtm - gamma_cls)
    omega_tuple = tuple(omega)

    return _sqw_cdft_recursive(q, gamma_tuple, omega_tuple, dt, dw, logger)


def sqw_qtm_correction_factor(
    q: float, time_vec: Array1D, gamma_qtm: Array1D, gamma_cls: Array1D, *, logger: Callable[[str], None] | None = print
):
    if not is_all_array_1d(time_vec, gamma_qtm, gamma_cls):
        raise ValueError("Input arrays must be one-dimensional!")
    if not (q > 0):
        raise ValueError("Momentum transfer `q` must be greater than 0!")

    return _get_sqw_qc(q, time_vec, gamma_qtm, gamma_cls, logger)


def _interp_sqw_qc(omega, omega_qc, sqw_qc):
    sqw_norm, sqw_phase = np.abs(sqw_qc), np.unwrap(np.angle(sqw_qc))
    sqw_norm_interped = CubicSpline(omega_qc, sqw_norm, extrapolate=False)(omega)
    sqw_phase_interped = CubicSpline(omega_qc, sqw_phase, extrapolate=False)(omega)
    return sqw_norm_interped * np.exp(1j * sqw_phase_interped)


def sqw_gaaqc(
    q: float,
    time_vec: Array1D,
    gamma_qtm: Array1D,
    gamma_cls: Array1D,
    omega_md: Array1D,
    sqw_md: Array1D,
    *,
    logger: Callable[[str], None] | None = print,
):
    if not is_all_array_1d(time_vec, gamma_qtm, gamma_cls, omega_md, sqw_md):
        raise ValueError("Input arrays must be one-dimensional!")
    if not (q > 0):
        raise ValueError("Momentum transfer `q` must be greater than 0!")

    omega_qc, sqw_qc = _get_sqw_qc(q, time_vec, gamma_qtm, gamma_cls, logger)

    dw_qc = np.mean(np.diff(omega_qc))
    dw_md = np.mean(np.diff(omega_md))

    if not np.isclose(dw_qc, dw_md, atol=0):
        n_points = int(np.round((omega_qc[-1] - omega_qc[0]) / dw_md)) + 1
        omega_qc_interped = np.linspace(omega_qc[0], omega_qc[-1], n_points)
        sqw_qc = _interp_sqw_qc(omega_qc_interped, omega_qc, sqw_qc)
        omega_qc = omega_qc_interped

    return linear_convolve_x_axis(omega_qc, omega_md), linear_convolve(sqw_qc, sqw_md, dw_md)  # type: ignore
