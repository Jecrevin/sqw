import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve


def linear_convolve[T: np.number](
    fx: NDArray[T], gx: NDArray[T], dx: float | np.floating
) -> NDArray[np.complexfloating]:
    """Compute the linear convolution of two one-dimensional sequences.

    Parameters
    ----------
    fx : NDArray[T]
        First one-dimensional input array.
    gx : NDArray[T]
        Second one-dimensional input array.
    dx : float | np.floating
        The sampling interval of the input arrays.

    Returns
    -------
    NDArray[np.complexfloating]
        The linear convolution of `fx` and `gx`, scaled by `dx`.
        The size of the output is `len(fx) + len(gx) - 1`.
    """
    if fx.ndim != 1 or gx.ndim != 1:
        raise ValueError("Input arrays must be one-dimensional.")

    return fftconvolve(fx, gx, mode="full") * dx


def self_linear_convolve[T: np.number](fx: NDArray[T], dx: float | np.floating) -> NDArray[np.complexfloating]:
    """Compute the self-convolution of a one-dimensional sequence.

    Parameters
    ----------
    fx : NDArray[T]
        One-dimensional input array.
    dx : float | np.floating
        The sampling interval of the input array.

    Returns
    -------
    NDArray[np.complexfloating]
        The self-convolution of `fx`, scaled by `dx`.
    """
    return linear_convolve(fx, fx, dx)


def linear_convolve_x_axis[T: np.number](x1: NDArray[T], x2: NDArray[T]) -> NDArray[T]:
    """Compute the x-axis for the convolution of two signals.

    The signals are assumed to be sampled on linearly spaced points defined by `x1` and `x2`.

    Parameters
    ----------
    x1 : NDArray[T]
        The x-axis for the first signal. Must be a 1D linearly spaced array.
    x2 : NDArray[T]
        The x-axis for the second signal. Must be a 1D linearly spaced array.

    Returns
    -------
    NDArray[T]
        The x-axis for the resulting convolved signal.

    Raises
    ------
    ValueError
        If input arrays are not one-dimensional, not linearly spaced, or do not have the same step size.
    """
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("Input arrays must be one-dimensional.")
    if not is_linspace(x1) or not is_linspace(x2):
        raise ValueError("Input arrays must be linear spaced.")
    if not np.isclose(x1[1] - x1[0], x2[1] - x2[0], atol=0):
        raise ValueError("Input arrays must have the same step size.")

    return np.linspace(x1[0] + x2[0], x1[-1] + x2[-1], x1.size + x2.size - 1, dtype=x1.dtype)


def self_linear_convolve_x_axis[T: np.number](x: NDArray[T]) -> NDArray[T]:
    """Compute the x-axis for the self-convolution of a signal.

    The signal is assumed to be sampled on linearly spaced points defined by `x`.

    Parameters
    ----------
    x : NDArray[T]
        The x-axis for the signal. Must be a 1D linearly spaced array.

    Returns
    -------
    NDArray[T]
        The x-axis for the resulting self-convolved signal.
    """
    return linear_convolve_x_axis(x, x)


def is_linspace[T: np.number](arr: NDArray[T]) -> bool:
    """Check if a 1D array is linearly spaced.

    Parameters
    ----------
    arr : NDArray[T]
        The input array to check.

    Returns
    -------
    bool
        True if the array is linearly spaced, False otherwise.

    Raises
    ------
    ValueError
        If the input array is not one-dimensional.
    """
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
    """Compute the continuous Fourier Transform of a 1D signal.

    This is approximated by the Discrete Fourier Transform (DFT).

    Parameters
    ----------
    signal : NDArray[T]
        The input 1D signal.
    sampling_rate : float | np.floating
        The sampling rate of the signal.

    Returns
    -------
    NDArray[np.complex128]
        The Fourier transform of the signal.

    Raises
    ------
    ValueError
        If the input signal is not one-dimensional.
    """
    if signal.ndim != 1:
        raise ValueError("Input signal must be one-dimensional.")

    return np.fft.fftshift(np.fft.fft(signal)) / sampling_rate


def odd_extend[T: np.number](arr: NDArray[T]) -> NDArray[T]:
    """Extend a 1D array to be odd-symmetric.

    The array is extended by reflecting it about the first element, with a sign change.
    The first element is the center of symmetry and is not repeated.

    Example
    -------
    >>> odd_extend(np.array([0, 1, 2]))
    array([-2, -1,  0,  1,  2])

    Parameters
    ----------
    arr : NDArray[T]
        The input 1D array. Assumed to start from or near zero.

    Returns
    -------
    NDArray[T]
        The odd-extended array.
    """
    return np.concatenate((-arr[:0:-1], arr))


def even_extend[T: np.number](arr: NDArray[T]) -> NDArray[T]:
    """Extend a 1D array to be even-symmetric.

    The array is extended by reflecting it about the first element.
    The first element is the center of symmetry and is not repeated.

    Example
    -------
    >>> even_extend(np.array([0, 1, 2]))
    array([2, 1, 0, 1, 2])

    Parameters
    ----------
    arr : NDArray[T]
        The input 1D array.

    Returns
    -------
    NDArray[T]
        The even-extended array.
    """
    return np.concatenate((arr[:0:-1], arr))
