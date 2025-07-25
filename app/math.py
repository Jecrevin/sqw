from collections.abc import Generator, Iterator
from itertools import chain
from typing import TypeVar

import numpy as np
from numpy.fft import fft, ifft
from numpy.typing import NDArray

NumberType = TypeVar("NumberType", bound=np.number)


def cdft(x: NDArray[NumberType], fx: NDArray[NumberType], cutoff: int = 10):
    if x.ndim != 1 or fx.ndim != 1:
        raise ValueError("Both x and fx must be one-dimensional arrays.")
    if x.size != fx.size:
        raise ValueError("x and fx must have the same size.")
    if not is_linspaced_array(x):
        raise ValueError("x must be a linearly spaced array.")

    dx = x[1] - x[0]

    fx_max = np.max(fx)
    rx = fx_max - fx
    rx_at_zero = np.interp(0, x, rx)  # type: ignore

    first_two_gn = fft([np.ones_like(rx), rx / rx_at_zero])

    circ_conv_coeff = gen_circ_conv_to_dft_coeff(first_two_gn[1].size)
    gn: Iterator[NDArray[np.complexfloating]] = chain(
        first_two_gn,
        (coeff * gn for coeff, gn in zip(circ_conv_coeff, gen_self_n_circ_convolve(first_two_gn[1]))),
    )
    gn_coeff = _gen_cdft_gn_coeff(rx_at_zero)

    return np.exp(-fx_max) * np.sum([next(gn_coeff) * next(gn) for _ in range(cutoff + 1)], axis=0) * dx


def is_linspaced_array(arr: NDArray) -> bool:
    if arr.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")

    if arr.size <= 2:
        return True

    diff = np.diff(arr)
    return np.allclose(diff, diff[0], atol=0 if np.abs(diff[0]) < 1e-6 else 1e-8)


def gen_self_n_circ_convolve(fx: NDArray[NumberType]) -> Generator[NDArray[np.complexfloating]]:
    if fx.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")

    res = fx

    while True:
        # Use FFT for computing circular convolution
        res = ifft(fft(res) * fft(fx))

        # Check if the result contains inf or nan
        if not np.isfinite(res).all():
            raise OverflowError("Overflow occurred during FFT computation")

        yield res


def _gen_cdft_gn_coeff(val: float):
    res, n = 1.0, 0
    while True:
        yield res
        res *= val / (n := n + 1)


def gen_circ_conv_to_dft_coeff(n: int):
    res = 1.0
    while True:
        res *= 1 / n
        yield res
