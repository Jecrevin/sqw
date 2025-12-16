"""Helper functions for reading data from HDF5 files."""

from os import PathLike
from typing import cast

import h5py
import numpy as np
from numpy.typing import NDArray

from sqw.consts import HBAR, KB, PI
from sqw.typing_ import Array1D


def read_gamma_data(
    file_path: PathLike, keys: list[str], include_cls: bool = False, extend_to_neg_time: bool = True
) -> tuple[Array1D[np.floating], Array1D[np.complexfloating], Array1D[np.floating] | None]:
    """Read gamma data from an HDF5 file.

    Parameters
    ----------
    file_path : PathLike
        Path to the HDF5 file containing gamma data.
    keys : list[str]
        List of keys to access datasets in the HDF5 file., ordered as [time,
        gamma_qtm_real, gamma_qtm_imag, gamma_cls].
    include_cls : bool, optional
        Whether to include classical gamma data, by default False.
    extend_to_neg_time : bool, optional
        Whether to extend the time and gamma data to negative time values, by
        default True.

    Returns
    -------
    tuple[Array1D[np.floating], Array1D[np.complexfloating],
    Array1D[np.floating] | None]
        A tuple containing time array, complex gamma array, and classical gamma
        array (if included).

    """
    assert len(keys) == 4, "`keys` must contain exactly 4 keys."
    with h5py.File(file_path) as f:
        time = cast(h5py.Dataset, f[keys[0]])[()]
        gamma_qtm_real = cast(h5py.Dataset, f[keys[1]])[()]
        gamma_qtm_imag = cast(h5py.Dataset, f[keys[2]])[()]
        gamma_cls = cast(h5py.Dataset, f[keys[3]])[()]
    gamma_qtm = gamma_qtm_real + 1j * gamma_qtm_imag

    if extend_to_neg_time:
        time = np.concatenate((-time[::-1], time[1:]))
        gamma_qtm = np.concatenate((np.conj(gamma_qtm[::-1]), gamma_qtm[1:]))
        gamma_cls = np.concatenate((gamma_cls[::-1], gamma_cls[1:]))

    return time, gamma_qtm, gamma_cls if include_cls else None


def read_stc_data(file_path: PathLike, keys: list[str]) -> tuple[Array1D[np.floating], Array1D[np.floating]]:
    """Read DOS data for STC model from an HDF5 file.

    Parameters
    ----------
    file_path : PathLike
        Path to the HDF5 file containing DOS data.
    keys : list[str]
        List of keys to access datasets in the HDF5 file, ordered as [omega,
        vdos].

    Returns
    -------
    tuple[Array1D[np.number], Array1D[np.number]]
        A tuple containing omega array and VDOS array.

    """
    assert len(keys) == 2, "`keys` must contain exactly 2 keys."

    with h5py.File(file_path) as f:
        data = [cast(h5py.Dataset, f[key])[()] for key in keys]

    return tuple(data)


def read_md_data(
    file_path: PathLike, keys: list[str], normalize_fft_coeff: bool = True
) -> tuple[Array1D[np.floating], Array1D[np.floating], NDArray[np.floating]]:
    """Read MD simulation S(Q, ω) data from an HDF5 file.

    Parameters
    ----------
    file_path : PathLike
        Path to the HDF5 file containing MD simulation data.
    keys : list[str]
        List of keys to access datasets in the HDF5 file, ordered as [q_vals,
        omega_md, sqw_md_stack].
    normalize_fft_coeff : bool, optional
        Whether to normalize the FFT coefficients, by default True. If True,
        multiplies the S(Q, ω) data by sqrt(2π).

    """
    assert len(keys) == 3, "`keys` must contain exactly 3 keys."

    with h5py.File(file_path) as f:
        q_vals = cast(h5py.Dataset, f[keys[0]])[()]
        omega_md = cast(h5py.Dataset, f[keys[1]])[()]
        sqw_md_stack = cast(h5py.Dataset, f[keys[2]])[()]
    omega_md = np.concatenate((-omega_md[::-1], omega_md[1:]))
    sqw_md_stack = np.concatenate((sqw_md_stack[:, ::-1], sqw_md_stack[:, 1:]), axis=1)

    if normalize_fft_coeff:
        sqw_md_stack *= np.sqrt(2 * PI)

    return q_vals, omega_md, sqw_md_stack


def assure_detailed_balance(
    freq: Array1D[np.floating], sqw: Array1D[np.floating], temperature: float
) -> Array1D[np.floating]:
    """Apply detailed balance to the scattering function S(Q,ω).

    This function assumes that the input S(Q,ω) is correct on the negative
    frequency side and adjusts the positive frequency side according to the
    detailed balance condition.
    """
    freq_pos = freq[freq >= 0]
    sqw_pos = np.interp(-freq_pos, freq, sqw) * np.exp(-HBAR * freq_pos / (KB * temperature))

    return np.concatenate((sqw[freq < 0], sqw_pos))
