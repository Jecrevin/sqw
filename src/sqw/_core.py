from collections.abc import Callable
from functools import lru_cache

import numpy as np
from scipy.constants import pi as PI

from ._math import (
    continuous_fourier_transform,
    interpolate_complex_function,
    is_all_array_1d,
    linear_convolve,
    linear_convolve_x_axis,
    self_linear_convolve,
    self_linear_convolve_x_axis,
    trim_function,
)
from .consts import HBAR, KB, NEUTRON_MASS
from .typing import Array1D


def sqw_stc_model[T: np.floating](
    q: float,
    w: Array1D[T],
    freq_dos: Array1D[np.floating],
    density_of_states: Array1D[np.floating],
    temperature: float,
    mass_num: int = 1,
) -> Array1D[T]:
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
def _sqw_ga_model[T: np.inexact, U: np.floating](
    q: float,
    gamma_tuple: tuple[T, ...],
    omega_tuple: tuple[U, ...],
    dt: float,
    dw: float,
    *,
    assure_detailed_balance: Callable[[Array1D[U], Array1D[np.complex128]], Array1D[np.complex128]] | None,
    logger: Callable[[str], None] | None,
) -> tuple[Array1D[U], Array1D[np.complex128]]:
    """Helper function to calculate S(q,w) using Gaussian Approximation (GA) with logging."""
    gamma: Array1D[T] = np.array(gamma_tuple)
    omega: Array1D[U] = np.array(omega_tuple)

    if logger:
        logger(f"Calculating CDFT recursively for {q = :.2f} ...")

    if q <= 5:  # direct calculation for small q
        sisf = np.exp(-0.5 * q**2 * gamma)  # Self-Intermediate Scattering Function
        sqw_raw = continuous_fourier_transform(sisf, 1 / dt) / (2 * PI)
        sqw = assure_detailed_balance(omega, sqw_raw) if assure_detailed_balance else sqw_raw

        if logger:
            logger(f"Done calculation for {q = :.2f}.")

        return omega, sqw

    recur_q = round(q / np.sqrt(2), 8)  # round q to avoid floating point issue for caching
    recur_logger = (lambda s: logger("=> " + s)) if logger else None

    # recursive calculation for larger q
    x_recur, y_recur = _sqw_ga_model(
        recur_q, gamma_tuple, omega_tuple, dt, dw, assure_detailed_balance=None, logger=recur_logger
    )

    # trim the value at front and end which nearly zero to speed up the convolution
    x_recur_trimed, y_recur_trimed = trim_function(x_recur, y_recur, cut_ratio=1e-9)

    res_omega = self_linear_convolve_x_axis(x_recur_trimed)
    res_sqw_raw = self_linear_convolve(y_recur_trimed, dw)
    res_sqw = assure_detailed_balance(res_omega, res_sqw_raw) if assure_detailed_balance else res_sqw_raw

    if logger:
        logger(f"Done calculation for {q = :.2f}.")

    return res_omega, res_sqw


def sqw_ga_model[T: np.floating, U: np.inexact](
    q: float,
    time_vec: Array1D[T],
    gamma: Array1D[U],
    *,
    assure_detailed_balance: Callable[[Array1D[U], Array1D[np.complex128]], Array1D[np.complex128]] | None = None,
    logger: Callable[[str], None] | None = print,
) -> tuple[Array1D[np.floating], Array1D[np.complex128]]:
    """Calculate S(q,w) using Gaussian Approximation (GA) model.

    Parameters
    ----------
    q : float
        Momentum transfer (in Å⁻¹), must be greater than 0.
    time_vec : Array1D[T]
        Time sequence, must be one-dimensional.
    gamma : Array1D[U]
        Width function :math:`\\Gamma(t)`, must be one-dimensional.
    logger : Callable[[str], None] | None, optional
        Logger function to log the progress, by default print to console.

    Returns
    -------
    tuple[Array1D[np.floating], Array1D[np.complex128]]
        A tuple containing:
        - omega: Frequency sequence.
        - S(q,w): Scattering function values (complex).

    Raises
    ------
    ValueError
        - If `q` is not greater than 0 or if input arrays are not one-dimensional.
        - If `time_vec` and `gamma` are not all one-dimensional.
        - If `time_vec` and `gamma` have different lengths.
    """
    if not (q > 0):
        raise ValueError("Momentum transfer `q` must be greater than 0!")
    if not is_all_array_1d(time_vec, gamma):
        raise ValueError("Input arrays must be one-dimensional!")
    if time_vec.size != gamma.size:
        raise ValueError("Input arrays must have the same length!")

    dt = np.mean(np.diff(time_vec))
    omega = np.fft.fftshift(np.fft.fftfreq(time_vec.size, dt)) * 2 * PI
    dw = np.mean(np.diff(omega))

    # Convert to tuple to make `gamma` & `omega` hashable for caching
    gamma_tuple = tuple(gamma)
    omega_tuple = tuple(omega)

    if not assure_detailed_balance and logger:
        logger("Warning: Detailed balance is not assured!")

    result = _sqw_ga_model(
        q, gamma_tuple, omega_tuple, dt, dw, assure_detailed_balance=assure_detailed_balance, logger=logger
    )

    if logger:
        logger(f"S(q,w) calculation for {q = :.2f} all completed!")

    return result


def _sqw_qc(
    q: float,
    time_vec: Array1D[np.floating],
    gamma_qtm: Array1D[np.complexfloating],
    gamma_cls: Array1D[np.floating],
    logger: Callable[[str], None] | None,
) -> tuple[Array1D[np.floating], Array1D[np.complex128]]:
    dt: float = np.diff(time_vec).mean()
    omega: Array1D[np.floating] = np.fft.fftshift(np.fft.fftfreq(time_vec.size, dt)) * 2 * PI
    dw: float = np.diff(omega).mean()

    gamma_tuple = tuple(gamma_qtm - gamma_cls)
    omega_tuple = tuple(omega)

    return _sqw_ga_model(q, gamma_tuple, omega_tuple, dt, dw, logger=logger)


def sqw_quantum_correction_factor(
    q: float,
    time_vec: Array1D[np.floating],
    gamma_qtm: Array1D[np.complexfloating],
    gamma_cls: Array1D[np.floating],
    *,
    logger: Callable[[str], None] | None = print,
) -> tuple[Array1D[np.floating], Array1D[np.complex128]]:
    if not is_all_array_1d(time_vec, gamma_qtm, gamma_cls):
        raise ValueError("Input arrays must be one-dimensional!")
    if not (q > 0):
        raise ValueError("Momentum transfer `q` must be greater than 0!")

    return _sqw_qc(q, time_vec, gamma_qtm, gamma_cls, logger=logger)


def sqw_gaaqc_model(
    q: float,
    time_vec: Array1D[np.floating],
    gamma_qtm: Array1D[np.complexfloating],
    gamma_cls: Array1D[np.floating],
    omega_md: Array1D[np.floating],
    sqw_md: Array1D[np.floating],
    *,
    logger: Callable[[str], None] | None = print,
) -> tuple[Array1D[np.floating], Array1D[np.complex128]]:
    if not is_all_array_1d(time_vec, gamma_qtm, gamma_cls, omega_md, sqw_md):
        raise ValueError("Input arrays must be one-dimensional!")
    if not (q > 0):
        raise ValueError("Momentum transfer `q` must be greater than 0!")

    omega_qc, sqw_qc = _sqw_qc(q, time_vec, gamma_qtm, gamma_cls, logger)

    dw_qc: float = np.diff(omega_qc).mean()
    dw_md: float = np.diff(omega_md).mean()

    if not np.isclose(dw_qc, dw_md, atol=0):
        n_points = int(np.round((omega_qc[-1] - omega_qc[0]) / dw_md)) + 1
        omega_qc_interped = np.linspace(omega_qc[0], omega_qc[-1], n_points)
        sqw_qc = interpolate_complex_function(omega_qc, omega_qc_interped, sqw_qc)
        omega_qc = omega_qc_interped

    return linear_convolve_x_axis(omega_qc, omega_md), linear_convolve(sqw_qc, sqw_md, dw_md)


def is_detailed_balance(omega: Array1D[np.floating], sqw: Array1D[np.complexfloating], temperature: float) -> bool:
    if not is_all_array_1d(omega, sqw):
        raise ValueError("Input arrays must be one-dimensional!")
    if omega.size != sqw.size:
        raise ValueError("Input arrays must have the same length!")

    neg_indices = omega < 0
    pos_indices = omega >= 0
    for indices in (neg_indices, pos_indices):
        o = omega[indices]
        s = sqw[indices]
        s_ = interpolate_complex_function(-o, omega, sqw) * np.exp(HBAR * -o / (KB * temperature))
        if not np.allclose(s_, s, atol=0):
            return False
    return True
