from typing import Any, overload

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from ._typing import Array1D


def is_all_array_1d(*arrays: NDArray[Any]) -> bool:
    """Check if all input arrays are one-dimensional.

    Parameters
    ----------
    *arrays : NDArray[Any]
        Variable length argument list of numpy arrays.

    Returns
    -------
    bool
        True if all arrays are 1D, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from h2o_sqw_calc.math import is_all_array_1d
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> is_all_array_1d(a, b)
    True
    >>> c = np.array([[1, 2], [3, 4]])
    >>> is_all_array_1d(a, c)
    False
    """
    return all(arr.ndim == 1 for arr in arrays)


def is_array_linspace[T: np.number](arr: Array1D[T]) -> bool:
    """Check if a 1D array is linearly spaced.

    Parameters
    ----------
    arr : Array1D[T]
        The input 1D numpy array.

    Returns
    -------
    bool
        True if the array is linearly spaced, False otherwise.

    Raises
    ------
    ValueError
        If the input array is not one-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> from h2o_sqw_calc.math import is_linspace
    >>> a = np.array([1, 2, 3, 4])
    >>> is_linspace(a)
    True
    >>> b = np.array([1, 2, 4])
    >>> is_linspace(b)
    False
    >>> c = np.array([1.0, 1.5, 2.0])
    >>> is_linspace(c)
    True
    """
    if arr.size <= 2:
        return True

    diff = np.diff(arr)

    if np.issubdtype(arr.dtype, np.integer):
        return all(diff == diff[0])
    else:
        return np.allclose(diff, diff[0], atol=0)


def is_all_array_linspace(*arrays: Array1D[Any]) -> bool:
    """Check if all input arrays are linearly spaced.

    Parameters
    ----------
    *arrays : Array1D[Any]
        Variable length argument list of numpy arrays.

    Returns
    -------
    bool
        True if all arrays are linearly spaced, False otherwise.

    Raises
    ------
    ValueError
        If any input array is not one-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> from h2o_sqw_calc.math import is_all_array_linspace
    >>> a = np.linspace(0, 1, 5)
    >>> b = np.arange(5)
    >>> is_all_array_linspace(a, b)
    True
    >>> c = np.array([1, 2, 4])
    >>> is_all_array_linspace(a, c)
    False
    """
    return all(is_array_linspace(arr) for arr in arrays)


@overload
def linear_convolve[FloatType1: np.floating, FloatType2: np.floating](
    fx: Array1D[FloatType1], gx: Array1D[FloatType2], dx: float
) -> Array1D[FloatType1 | FloatType2 | np.float64]: ...
@overload
def linear_convolve[ComplexType: np.complexfloating, FloatType: np.floating](
    fx: Array1D[ComplexType], gx: Array1D[FloatType], dx: float
) -> Array1D[ComplexType]: ...
@overload
def linear_convolve[FloatType: np.floating, ComplexType: np.complexfloating](
    fx: Array1D[FloatType], gx: Array1D[ComplexType], dx: float
) -> Array1D[ComplexType]: ...
@overload
def linear_convolve[ComplexType1: np.complexfloating, ComplexType2: np.complexfloating](
    fx: Array1D[ComplexType1], gx: Array1D[ComplexType2], dx: float
) -> Array1D[ComplexType1 | ComplexType2]: ...
def linear_convolve(fx, gx, dx):
    """Calculate the linear convolution of two 1D signals.

    This uses `scipy.signal.fftconvolve` for fast computation.

    Parameters
    ----------
    fx : Array1D[T]
        First 1D input array.
    gx : Array1D[U]
        Second 1D input array.
    dx : float | int | np.number
        The sampling interval of the signals.

    Returns
    -------
    Array1D[np.number]
        The result of the linear convolution.

    Raises
    ------
    ValueError
        If input arrays are not one-dimensional.

    Notes
    -----
    Input arrays are assumed to be sampled on commonly spaced grids. The result
    is scaled by the sampling interval `dx` to approximate the continuous
    convolution.

    Examples
    --------
    >>> import numpy as np
    >>> from h2o_sqw_calc.math import linear_convolve
    >>> fx = np.array([1, 2, 3])
    >>> gx = np.array([4, 5])
    >>> dx = 0.1
    >>> linear_convolve(fx, gx, dx)
    array([0.4, 1.3, 2.2, 1.5])
    """
    return fftconvolve(fx, gx, mode="full") * dx


@overload
def self_linear_convolve[FloatType: np.floating](
    fx: Array1D[FloatType], dx: float
) -> Array1D[FloatType | np.float64]: ...
@overload
def self_linear_convolve[ComplexType: np.complexfloating](
    fx: Array1D[ComplexType], dx: float
) -> Array1D[ComplexType]: ...
def self_linear_convolve(fx, dx):
    """Calculate the self-convolution of a 1D signal.

    Parameters
    ----------
    fx : Array1D[T]
        The 1D input array.
    dx : float | int | np.number
        The sampling interval of the signal.

    Returns
    -------
    Array1D[np.number]
        The result of the self-convolution.

    Raises
    ------
    ValueError
        If the input array is not one-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> from h2o_sqw_calc.math import self_linear_convolve
    >>> fx = np.array([1, 2, 3])
    >>> dx = 1.0
    >>> self_linear_convolve(fx, dx)
    array([1., 4., 10., 12., 9.])
    """
    return linear_convolve(fx, fx, dx)


def linear_convolve_x_axis[T: np.inexact, U: np.inexact](x1: Array1D[T], x2: Array1D[U]) -> Array1D[T | U]:
    """Calculate the x-axis for the convolution of two signals.

    The signals are assumed to be sampled on linearly spaced grids.

    Parameters
    ----------
    x1 : Array1D[T]
        The x-axis for the first signal.
    x2 : Array1D[U]
        The x-axis for the second signal.

    Returns
    -------
    Array1D[np.floating]
        The x-axis for the convolved signal.

    Raises
    ------
    ValueError
        If input arrays are not 1D, not linearly spaced, or have
        different step sizes.

    Examples
    --------
    >>> import numpy as np
    >>> from h2o_sqw_calc.math import linear_convolve_x_axis
    >>> x1 = np.linspace(0, 1, 3)  # [0., 0.5, 1.]
    >>> x2 = np.linspace(0, 0.5, 2)  # [0., 0.5]
    >>> linear_convolve_x_axis(x1, x2)
    array([0., 0.5, 1., 1.5])
    """
    return np.linspace(x1[0] + x2[0], x1[-1] + x2[-1], x1.size + x2.size - 1)


def self_linear_convolve_x_axis[T: np.inexact](x: Array1D[T]) -> Array1D[T]:
    """Calculate the x-axis for the self-convolution of a signal.

    Parameters
    ----------
    x : Array1D[T]
        The x-axis for the signal.

    Returns
    -------
    Array1D[np.floating]
        The x-axis for the self-convolved signal.

    Raises
    ------
    ValueError
        If the input array is not 1D or not linearly spaced.

    Examples
    --------
    >>> import numpy as np
    >>> from h2o_sqw_calc.math import self_linear_convolve_x_axis
    >>> x = np.linspace(0, 1, 5)
    >>> self_linear_convolve_x_axis(x)
    array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ])
    """
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
    y_abs = np.abs(y)
    threshold = np.max(y_abs) * cut_ratio

    significant_indices = np.nonzero(y_abs > threshold)[0]
    start, end = significant_indices[0], significant_indices[-1] + 1

    return x[start:end], y[start:end]
