import numpy as np
from numpy.typing import NDArray


def linear_convolve[T: np.number](
    fx: NDArray[T], gx: NDArray[T], dx: float | np.floating
) -> NDArray[np.complexfloating]:
    return np.convolve(fx, gx, mode="full") * dx


def self_linear_convolve[T: np.number](fx: NDArray[T], dx: float | np.floating) -> NDArray[np.complexfloating]:
    return linear_convolve(fx, fx, dx)


def linear_convolve_x_axis[T: np.number](x1: NDArray[T], x2: NDArray[T]) -> NDArray[T]:
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("Input arrays must be one-dimensional.")
    if not is_linspace(x1) or not is_linspace(x2):
        raise ValueError("Input arrays must be linear spaced.")
    if not np.isclose(x1[1] - x1[0], x2[1] - x2[0], atol=0):
        raise ValueError("Input arrays must have the same step size.")

    return np.linspace(x1[0] + x2[0], x1[-1] + x2[-1], x1.size + x2.size - 1, dtype=x1.dtype)


def self_linear_convolve_x_axis[T: np.number](x: NDArray[T]) -> NDArray[T]:
    return linear_convolve_x_axis(x, x)


def is_linspace[T: np.number](arr: NDArray[T]) -> bool:
    if arr.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")

    if arr.size <= 2:
        return True

    diff = np.diff(arr)

    if np.issubdtype(arr.dtype, np.integer):
        return bool(np.all(diff == diff[0]))

    return np.allclose(diff, diff[0], atol=0)


def continuous_fourier_transform[T: np.number](
    signal: NDArray[T], sampling_rate: float | np.floating
) -> NDArray[np.complex128]:
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional.")

    return np.fft.fftshift(np.fft.fft(signal)) / sampling_rate


def odd_extend[T: np.number](arr: NDArray[T]) -> NDArray[T]:
    return np.concatenate((-arr[:0:-1], arr))


def even_extend[T: np.number](arr: NDArray[T]) -> NDArray[T]:
    return np.concatenate((arr[:0:-1], arr))
