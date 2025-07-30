from functools import reduce
from itertools import accumulate, chain, count, islice, repeat
import operator

import numpy as np
import scipy.constants as consts
from numpy.typing import NDArray
from returns.pipeline import flow

from app.math import ArrMulDFTMethod, NumberType, gen_func_self_n_mul_ft, is_linspaced_array

# Physical constants
# data used scipy data from CODATA Internationally recommended 2022 values of the
# Fundamental Physical Constants https://physics.nist.gov/cuu/Constants/index.html

HBAR = consts.value("reduced Planck constant in eV s")  # reduced Planck constant in eV·s
KB = consts.value("Boltzmann constant in eV/K")  # Boltzmann constant in eV/K
NEUTRON_MASS = (
    consts.value("neutron mass energy equivalent in MeV") * consts.mega / (consts.c / consts.angstrom) ** 2
)  # neutron mass in eV/(Å²·s²)

# mathematical constants
PI = consts.pi


def sqw_stc_model(
    q: float, w: NDArray[NumberType], t: float, freq: NDArray[NumberType], dos: NDArray[NumberType], mass_num: int = 1
) -> NDArray[NumberType]:
    beta = flow(
        t,
        lambda x: HBAR / (KB * x),
        lambda x: np.concatenate((dos[:1], dos[1:] * freq[1:] * x / (2 * np.tanh(freq[1:] * x / 2)))),
        lambda x: np.trapezoid(x, freq),
        lambda x: t * x,  # till now get the effective temperature
        lambda x: 1.0 / (x * KB),
    )
    return flow(
        q,
        lambda x: (HBAR * x) ** 2 / (2 * NEUTRON_MASS * mass_num),
        lambda x: np.sqrt(beta / (4 * PI * x)) * np.exp(-beta * (HBAR * w - x) ** 2 / (4 * x)),
    )


def sqw_from_gamma(
    q: float,
    time: NDArray[NumberType],
    gamma: NDArray[np.complex128],
    cutoff: int = 10,
    method: ArrMulDFTMethod = "direct",
) -> NDArray[np.complex128]:
    if time.ndim != 1 or gamma.ndim != 1:
        raise ValueError("Both `time` and `gamma` must be 1-dimensional arrays.")
    if time.size != gamma.size:
        raise ValueError("`time` and `gamma` must have the same size.")

    if not is_linspaced_array(time):
        raise ValueError("`time` must be a linearly spaced array.")

    dt = time[1] - time[0]
    ft = q**2 / 2 * gamma
    ft_max = np.max(ft)

    rt = ft_max - ft
    rt_at_zero = np.interp(0, time.astype(np.float64), rt)

    gn_coeff = accumulate((r / n for r, n in zip(repeat(rt_at_zero), count(1))), operator.mul)
    gn = chain(
        np.fft.fft([np.zeros_like(rt)]),
        (c * g for c, g in zip(gn_coeff, gen_func_self_n_mul_ft(rt / rt_at_zero, dt, method=method))),
    )

    return np.exp(-ft_max) * reduce(np.add, islice(gn, cutoff + 1))
