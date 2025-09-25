from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq, fftshift, rfft
from scipy.signal import fftconvolve

from ._typing import Array1D


def is_all_array_1d(*arrays: NDArray[Any]) -> bool:
    """Check if all input arrays are 1-dimensional."""
    return all(arr.ndim == 1 for arr in arrays)


def is_array_linspace(arr: Array1D[np.number]) -> bool:
    """Check if the input array is evenly spaced."""
    if arr.size <= 2:
        return True
    diff = np.diff(arr)
    return all(diff == diff[0]) if np.issubdtype(arr.dtype, np.integer) else np.allclose(diff, diff[0], atol=0)


def is_all_array_linspace(*arrays: Array1D[np.number]) -> bool:
    """Check if all input arrays are evenly spaced."""
    return all(is_array_linspace(arr) for arr in arrays)


def linear_convolve(fx: Array1D[np.number], gx: Array1D[np.number], dx: float) -> Array1D[np.floating]:
    """Simulate the linear convolution of two functions sampled on evenly spaced grids."""
    return fftconvolve(fx, gx, mode="full") * dx


def self_linear_convolve(fx: Array1D[np.number], dx: float) -> Array1D[np.floating]:
    """Simulate the linear convolution of a function with itself sampled on an evenly spaced grid."""
    return linear_convolve(fx, fx, dx)


def linear_convolve_x_axis[T: np.floating, U: np.floating](x1: Array1D[T], x2: Array1D[U]) -> Array1D[T | U]:
    """Calculate the corresponding x-axis of the linear convolution of two functions sampled on evenly spaced grids."""
    return np.linspace(x1[0] + x2[0], x1[-1] + x2[-1], x1.size + x2.size - 1)


def self_linear_convolve_x_axis[T: np.floating](x: Array1D[T]) -> Array1D[T]:
    """Calculate the corresponding x-axis of the linear convolution of a
    function with itself sampled on an evenly spaced grid."""
    return linear_convolve_x_axis(x, x)


def continuous_fourier_transform[T: np.inexact](signal: Array1D[T], sampling_rate: float) -> Array1D[np.complex128]:
    """Compute the continuous-like Fourier Transform of a 1D signal.

    This function uses `numpy.fft.fft` and scales the result to approximate
    the continuous Fourier Transform. The zero-frequency component is shifted
    to the center of the spectrum.

    Parameters
    ----------
    signal : Array1D[T]
        The input 1D signal.
    sampling_rate : float | int | np.number
        The sampling rate of the signal (samples per unit of the domain).

    Returns
    -------
    Array1D[np.complex128]
        The Fourier-transformed signal.

    Raises
    ------
    ValueError
        If the input signal is not one-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> from h2o_sqw_calc.math import continuous_fourier_transform
    >>> sampling_rate = 100  # Hz
    >>> t = np.arange(0, 1.0, 1 / sampling_rate)
    >>> signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    >>> ft_signal = continuous_fourier_transform(signal, sampling_rate)
    >>> # The peak should be at +/- 5 Hz. The frequency array would be:
    >>> # freq = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1./sampling_rate))
    >>> # The magnitude at the peak is expected to be close to 0.5i (for sine).
    >>> np.round(ft_signal[55], 2)
    -0.5j
    >>> np.round(ft_signal[45], 2)
    0.5j
    """
    return np.fft.fftshift(np.fft.fft(signal)) / sampling_rate


def trim_function[T: np.inexact, U: np.inexact](
    x: Array1D[T], y: Array1D[U], cut_ratio: float
) -> tuple[Array1D[T], Array1D[U]]:
    """Trim the input function (x, y) by removing parts where |y| is below a certain ratio of its maximum value."""
    y_abs = np.abs(y)
    threshold = np.max(y_abs) * cut_ratio

    significant_indices = np.nonzero(y_abs > threshold)[0]
    start, end = significant_indices[0], significant_indices[-1] + 1

    return x[start:end], y[start:end]
