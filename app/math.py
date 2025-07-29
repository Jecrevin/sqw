from collections.abc import Generator
from itertools import accumulate, repeat
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

NumberType = TypeVar("NumberType", bound=np.number)
ArrMulDFTMethod = Literal["direct", "convolve", "convolve-fft"]


def circular_convolve(
    a: NDArray[NumberType], b: NDArray[NumberType], use_fft: bool = False
) -> NDArray[np.complexfloating]:
    """
    Compute the circular convolution of two 1D arrays.

    Circular convolution treats the input arrays as if they were periodic and computes
    their convolution accordingly.

    Parameters
    ----------
    a : NDArray[NumberType]
        First input array (must be 1-dimensional).
    b : NDArray[NumberType]
        Second input array (must be 1-dimensional).
    use_fft : bool, optional
        Whether to use Fast Fourier Transform for computation, by default False.
        When True, the function computes the convolution using FFT, which may be
        more efficient for large arrays.

    Returns
    -------
    NDArray[np.complexfloating]
        The circular convolution of the input arrays.

    Raises
    ------
    ValueError
        If either input array is not 1-dimensional.

    Notes
    -----
    - If input arrays have different sizes, the function falls back to regular
      convolution (numpy.convolve with 'full' mode).
    - When not using FFT, the convolution is computed manually by wrapping the
      values that would extend beyond the original array size.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Both input arrays must be 1-dimensional.")

    if a.size != b.size:
        return np.convolve(a, b, mode="full")

    if use_fft:
        return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))
    else:
        conv = np.convolve(a, b, mode="full")
        res = conv[: a.size]
        res[:-1] += conv[a.size :]
        return res


def gen_self_n_circ_convolve(arr: NDArray[NumberType], use_fft: bool = False) -> Generator[NDArray[np.complexfloating]]:
    """
    Generate successive self-circular convolutions of an array.

    This function is a generator that yields the result of circularly
    convolving an array with itself n times. The first yielded value is
    the array itself (n=1), the second is the array convolved with itself (n=2),
    and so on.

    Parameters
    ----------
    arr : NDArray[NumberType]
        The 1-dimensional input array.
    use_fft : bool, optional
        Whether to use Fast Fourier Transform for computation, by default False.
        See `circular_convolve` for more details.

    Yields
    ------
    Generator[NDArray[np.complexfloating]]
        A generator that produces the successive self-circular convolutions.

    Raises
    ------
    ValueError
        If the input array is not 1-dimensional.
    """
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")

    yield from accumulate(repeat(arr), func=lambda x, y: circular_convolve(x, y, use_fft=use_fft))


def arr_mul_dft(
    arr1: NDArray[NumberType], arr2: NDArray[NumberType], method: ArrMulDFTMethod = "direct", antialias: bool = True
):
    """
    Compute the Discrete Fourier Transform (DFT) of the element-wise product of two 1D arrays.

    This function calculates DFT(arr1 * arr2) using one of two methods:
    1. Direct method: Directly computes the DFT of the element-wise product of the arrays.
    2. Convolution method: Uses the convolution theorem, which states that the DFT of a
       product is proportional to the convolution of their DFTs.

    The function can handle arrays of different sizes by padding the smaller array.

    Parameters
    ----------
    arr1 : NDArray[NumberType]
        First input array (must be 1-dimensional).
    arr2 : NDArray[NumberType]
        Second input array (must be 1-dimensional).
    method : {'direct', 'convolve', 'convolve-fft'}, optional
        The method to use for computation, by default "direct".
        - 'direct': Computes `np.fft.fft(arr1 * arr2)`.
        - 'convolve': Computes the circular convolution of the DFTs of the input arrays.
        - 'convolve-fft': Same as 'convolve', but uses FFT for the convolution step.
    antialias : bool, optional
        Whether to use padding to prevent aliasing when input arrays have different
        sizes, by default True. If True, arrays are padded to a length of
        `arr1.size + arr2.size - 1`. If False, the smaller array is padded to match
        the length of the larger array. This parameter has no effect if the arrays
        have the same size.

    Returns
    -------
    NDArray[np.complexfloating]
        The Discrete Fourier Transform of the element-wise product of the input arrays.

    Raises
    ------
    ValueError
        If either input array is not 1-dimensional, or if an unknown method is specified.
    """
    if arr1.ndim != 1 or arr2.ndim != 1:
        raise ValueError("Both input arrays must be 1-dimensional.")

    if arr1.size != arr2.size:
        if arr1.size < arr2.size:
            arr1, arr2 = arr2, arr1

        if antialias:
            n = arr1.size + arr2.size - 1
            arr1 = np.pad(arr1, (0, n - arr1.size))
            arr2 = np.pad(arr2, (0, n - arr2.size))
        else:
            arr2 = np.pad(arr2, (0, arr1.size - arr2.size))

    match method:
        case "direct":
            return np.fft.fft(arr1 * arr2)
        case "convolve" | "convolve-fft":
            fw, gw = np.fft.fft(arr1), np.fft.fft(arr2)
            return circular_convolve(fw, gw, use_fft=(method == "convolve-fft")) / arr1.size
        case _:
            raise ValueError(f"Unknown method: {method}")


def func_mul_ft(
    fx: NDArray[NumberType],
    gx: NDArray[NumberType],
    dx: float,
    method: ArrMulDFTMethod = "direct",
    antialias: bool = True,
):
    """
    Compute the Fourier Transform of the product of two functions.

    This function approximates the continuous Fourier Transform of the product
    of two functions, `f(x)` and `g(x)`, which are provided as sampled arrays
    `fx` and `gx`.

    It computes `dx * DFT(fx * gx)`, where `DFT` is the Discrete Fourier
    Transform computed by `arr_mul_dft`.

    Parameters
    ----------
    fx : NDArray[NumberType]
        Sampled values of the first function, `f(x)`.
    gx : NDArray[NumberType]
        Sampled values of the second function, `g(x)`.
    dx : float
        The sampling interval in the x-domain.
    method : {'direct', 'convolve', 'convolve-fft'}, optional
        The method to use for computation, passed to `arr_mul_dft`.
        By default "direct".
    antialias : bool, optional
        Whether to use padding to prevent aliasing. Passed to `arr_mul_dft`.
        By default True.

    Returns
    -------
    NDArray[np.complexfloating]
        The Fourier Transform of `f(x) * g(x)`.

    Notes
    -----
    This function assumes that `fx` and `gx` are sampled at the same points and
    have the same length. If they differ in size, they will be padded to match
    the length of the larger array.

    See Also
    --------
    `arr_mul_dft` : The underlying function that computes the DFT of the product of arrays.
    """
    return arr_mul_dft(fx, gx, method=method, antialias=antialias) * dx
