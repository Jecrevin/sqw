from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq, fftshift, rfft
from scipy.signal import fftconvolve

from .typing_ import Array1D


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
    """Calculate the corresponding x-axis of the linear convolution."""
    return linear_convolve_x_axis(x, x)


def continuous_fourier_transform(
    time: Array1D[np.floating], signal: Array1D[np.inexact]
) -> tuple[Array1D[np.double], Array1D[np.complexfloating]]:
    """Compute the continuous Fourier transform of a time-domain signal."""
    dt = time[1] - time[0]
    freq = fftshift(fftfreq(time.size, dt)) * 2 * np.pi  # angular frequency

    if np.iscomplexobj(signal):
        ft_signal_shifted = fftshift(fft(signal)) * dt
    else:
        ft_signal_pos = cast(Array1D[np.complexfloating], rfft(signal))
        ft_signal_neg = np.conj(ft_signal_pos[-2:0:-1] if signal.size % 2 == 0 else ft_signal_pos[:0:-1])
        ft_signal_shifted = fftshift(np.concatenate((ft_signal_pos, ft_signal_neg))) * dt

    ft_signal = ft_signal_shifted * np.exp(-1j * freq * time[0])

    return freq, ft_signal


def trim_function[T: np.floating, U: np.inexact](
    x: Array1D[T], y: Array1D[U], cut_ratio: float
) -> tuple[Array1D[T], Array1D[U]]:
    """Trim the input function (x, y) by removing parts where |y| is below a certain ratio of its maximum value."""
    y_abs = np.abs(y)
    threshold = np.max(y_abs) * cut_ratio

    significant_indices = np.nonzero(y_abs > threshold)[0]
    start, end = significant_indices[0], significant_indices[-1] + 1

    return x[start:end], y[start:end]
