from collections.abc import Callable
from functools import lru_cache

import numpy as np

from ._math import (
    continuous_fourier_transform,
    is_all_array_1d,
    is_all_array_linspace,
    is_array_linspace,
    linear_convolve,
    linear_interpolate,
    self_linear_convolve,
    self_linear_convolve_x_axis,
    trim_function,
)
from .consts import HBAR, KB, NEUTRON_MASS, PI
from .typing import Array1D


def sqw_stc_model(
    q: float,
    w: Array1D[np.floating],
    freq_dos: Array1D[np.floating],
    density_of_states: Array1D[np.floating],
    temperature: float,
) -> Array1D[np.floating]:
    """Calculate Scattering function S(q,w) using Short Time Collision Approximation (STC) model."""
    if not (q > 0):
        raise ValueError("Momentum transfer `q` must be greater than 0!")
    if not is_all_array_1d(w, freq_dos, density_of_states):
        raise ValueError("Input arrays must be all one-dimensional!")
    if not (temperature >= 0):
        raise ValueError("Temperature is in Kelvin and must be greater than or equal to 0!")

    hbar_times_beta = HBAR / (KB * temperature)
    effective_temperature = temperature * np.trapezoid(
        np.divide(
            density_of_states * hbar_times_beta * freq_dos,
            2 * np.tanh(hbar_times_beta * freq_dos / 2),
            out=density_of_states.copy(),
            where=(~np.isclose(freq_dos, 0, atol=0)),
        ),
        freq_dos,
    )
    effective_beta = 1 / (KB * effective_temperature)
    recoil_energy = (HBAR * q) ** 2 / (2 * NEUTRON_MASS)
    result = (
        HBAR
        * np.sqrt(effective_beta / (4 * PI * recoil_energy))
        * np.exp(-effective_beta / (4 * recoil_energy) * (HBAR * w - recoil_energy) ** 2)
        * np.exp(-HBAR * w * effective_beta)  # detailed balance factor
    )

    return result


def sqw_ga_model(
    q: float,
    time: Array1D[np.floating],
    width_func: Array1D[np.inexact],
    *,
    correction: Callable[[Array1D[np.floating], Array1D[np.floating]], Array1D[np.floating]] | None = None,
    logger: Callable[[str], None] | None = print,
) -> tuple[Array1D[np.double], Array1D[np.floating]]:
    if not (q > 0):
        raise ValueError("Momentum transfer `q` must be greater than 0!")
    if not is_all_array_1d(time, width_func):
        raise ValueError("Input arrays must be all one-dimensional!")
    if not is_array_linspace(time):
        raise ValueError("Time array must be evenly spaced!")
    if time.size != width_func.size:
        raise ValueError("Input arrays must have the same length!")
    if not any(np.isclose(time, 0, atol=1e-24, rtol=0)):
        raise ValueError("Time array must contain zero time point for accurate Fourier Transform!")

    if logger:
        logger(f"Calculating S(q,w) with Gaussian Approximation model ({q = :.2f} 1/Ang)...")

    result = _gaussian_approximation_core(q, tuple(time), tuple(width_func), correction=correction, logger=logger)

    if logger:
        logger(f"S(q,w) GA result for {q = :.2f} calculation completed.")

    return result


@lru_cache
def _gaussian_approximation_core(
    q: float,
    time_tpl: tuple[np.floating, ...],
    gamma_tpl: tuple[np.inexact, ...],
    *,
    correction: Callable[[Array1D[np.floating], Array1D[np.floating]], Array1D[np.floating]] | None,
    logger: Callable[[str], None] | None,
) -> tuple[Array1D[np.floating], Array1D[np.floating]]:
    time = np.array(time_tpl)
    gamma = np.array(gamma_tpl)

    if logger:
        logger(f"...calculating {q = :.2f}...")

    if q <= 5:
        sisf = np.exp(-0.5 * q**2 * gamma)  # Self-Intermediate Scattering Function, F(q,t)
        omega, ft_sisf = continuous_fourier_transform(time, sisf)
        # NOTE: For quantum system, the width function (gamma) is Hermitian
        # thus the SISF is also Hermitian, and the FT result is real.
        # For classical system, gamma is real and even, thus SISF is also real and even,
        # and the FT result is also real.
        raw_sqw = np.real(ft_sisf) / (2 * PI)
        sqw = correction(omega, raw_sqw) if correction else raw_sqw

        return omega, sqw

    # round q to avoid precision issues affecting lru_cache
    recur_q = round(q / np.sqrt(2), 8)
    recur_x, recur_y = _gaussian_approximation_core(
        recur_q,
        time_tpl,
        gamma_tpl,
        correction=None,  # delay correction after convolution to avoid magnitude issues
        logger=logger,
    )
    # trim small values at both sides to avoid numerical noise
    trimed_x, raw_trimed_y = trim_function(recur_x, recur_y, 1e-9)
    trimed_y = correction(trimed_x, raw_trimed_y) if correction else raw_trimed_y

    return self_linear_convolve_x_axis(trimed_x), self_linear_convolve(trimed_y, trimed_x[1] - trimed_x[0])


def sqw_gaaqc_model(
    q: float,
    time_ga: Array1D[np.floating],
    width_func_qtm: Array1D[np.complexfloating],
    width_func_cls: Array1D[np.floating],
    freq_md: Array1D[np.floating],
    sqw_md: Array1D[np.floating],
    *,
    correction: Callable[[Array1D[np.floating], Array1D[np.floating]], Array1D[np.floating]] | None = None,
    logger: Callable[[str], None] | None = print,
) -> tuple[Array1D[np.floating], Array1D[np.floating]]:
    if not (q > 0):
        raise ValueError("Momentum transfer `q` must be greater than 0!")
    if not is_all_array_1d(time_ga, width_func_qtm, width_func_cls, freq_md, sqw_md):
        raise ValueError("Input arrays must be all one-dimensional!")
    if not is_all_array_linspace(time_ga, freq_md):
        raise ValueError("Time and frequency arrays must be evenly spaced!")
    if time_ga.size != width_func_qtm.size or time_ga.size != width_func_cls.size:
        raise ValueError("Time, quantum width function and classical width function arrays must have the same length!")
    if freq_md.size != sqw_md.size:
        raise ValueError("Frequency and MD S(q,w) arrays must have the same length!")
    if not any(np.isclose(time_ga, 0, atol=1e-24, rtol=0)):
        raise ValueError("Time array must contain zero time point for accurate Fourier Transform!")

    freq_qc, sqw_qc = _gaussian_approximation_core(
        q,
        tuple(time_ga),
        tuple(width_func_qtm - width_func_cls),
        correction=None,
        logger=logger,
    )

    dw_qc = freq_qc[1] - freq_qc[0]
    dw_md = freq_md[1] - freq_md[0]
    if dw_qc >= dw_md:
        # interpolate MD data to match the finer frequency grid of QC data
        qc_index_in_md = (freq_qc >= freq_md[0]) & (freq_qc <= freq_md[-1])
        sqw_qc_trimmed = sqw_qc[qc_index_in_md]

        freq_qc_trimmed = freq_qc[qc_index_in_md]
        sqw_md_interp = linear_interpolate(freq_qc_trimmed, freq_md, sqw_md)
        raw_sqw_gaaqc = linear_convolve(sqw_qc_trimmed, sqw_md_interp, dw_qc)

        omega = self_linear_convolve_x_axis(freq_qc_trimmed)
    else:
        # interpolate QC data to match the finer frequency grid of MD data
        md_index_in_qc = (freq_md >= freq_qc[0]) & (freq_md <= freq_qc[-1])
        sqw_md_trimmed = sqw_md[md_index_in_qc]

        freq_md_trimmed = freq_md[md_index_in_qc]
        sqw_qc_interp = linear_interpolate(freq_md_trimmed, freq_qc, sqw_qc)
        raw_sqw_gaaqc = linear_convolve(sqw_qc_interp, sqw_md_trimmed, dw_md)

        omega = self_linear_convolve_x_axis(freq_md_trimmed)
    sqw_gaaqc = correction(omega, raw_sqw_gaaqc) if correction else raw_sqw_gaaqc

    return omega, sqw_gaaqc
