import os
import sys
from typing import cast

import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from sqw.typing import Array1D

from .helper import even_extend, odd_extend


def read_hdf5_dataset(file: str, dataset: str) -> NDArray[np.number]:
    try:
        with h5py.File(file) as f:
            data = f[dataset]
            if not isinstance(data, h5py.Dataset):
                raise DatasetNotFoundError(f"Dataset '{dataset}' not found in file '{file}'!")
            return data[()]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File '{file}' not found!") from e
    except OSError as e:
        raise NotHDF5FileError(f"{file} is not a valid HDF5 file!") from e
    except KeyError as e:
        raise DatasetNotFoundError(f"Dataset '{dataset}' not found in file '{file}'!") from e


class NotHDF5FileError(FileNotFoundError):
    """Not a valid HDF5 file."""


class DatasetNotFoundError(KeyError):
    """Dataset not found in HDF5 file."""


def read_gamma_data(
    file: str, keys: list[str], *, no_extend: bool, include_cls: bool
) -> tuple[Array1D[np.floating], Array1D[np.complexfloating], Array1D[np.floating] | None]:
    print(f"Reading width function data from file '{file}'...")

    assert len(keys) == 4, "GAMMA_KEYS must contain exactly 4 keys."

    needed = keys if include_cls else keys[:-1]
    data = [read_hdf5_dataset(file, key) for key in needed]

    if not no_extend:
        data[0] = odd_extend(data[0])  # time
        data[1] = even_extend(data[1])  # gamma_qtm_real
        data[2] = odd_extend(data[2])  # gamma_qtm_imag
        if include_cls:
            data[3] = even_extend(data[3])  # gamma_cls

    time = cast(Array1D[np.floating], data[0])
    gamma_qtm = cast(Array1D[np.complexfloating], data[1] + 1j * data[2])
    gamma_cls = cast(Array1D[np.floating], data[3]) if include_cls else None

    print("Width function data read completed.")

    return time, gamma_qtm, gamma_cls


def read_stc_data(file: str, keys: list[str]) -> tuple[Array1D[np.number], Array1D[np.number]]:
    print(f"Reading DOS data from file '{file}'...")

    assert len(keys) == 2, "STC_KEYS must contain exactly 2 keys."

    data = [read_hdf5_dataset(file, key) for key in keys]
    omega = data[0]
    vdos = data[1]

    print("DOS data for STC model read completed.")

    return omega, vdos


def read_md_data(file: str, keys: list[str], no_extend: bool):
    print(f"Reading MD S(q,w) data from file '{file}'...")

    assert len(keys) == 3, "MD_KEYS must contain exactly 3 keys."

    data = [read_hdf5_dataset(file, key) for key in keys]

    q_vals = cast(Array1D[np.floating], data[0])
    omega_md = cast(Array1D[np.floating], data[1] if no_extend else odd_extend(data[1]))
    sqw_md_stack = cast(NDArray[np.floating], data[2] if no_extend else np.apply_along_axis(even_extend, -1, data[2]))

    print("MD S(q,w) data read completed.")

    return q_vals, omega_md, sqw_md_stack


def save_or_show_plot(output: str | None) -> None:
    if output:
        print(f"Saving plot to '{output}'...")
        if os.path.dirname(output):
            os.makedirs(os.path.dirname(output), exist_ok=True)
        if os.path.exists(output):
            print(f"Warning: file '{output}' already exists, realy want to overwrite it? (y/N): ", end="")
            if input().lower() != "y":
                print("Aborted by user.")
                sys.exit(0)
        plt.savefig(output, dpi=300)
        print(f"Plot saved to '{output}'.")
    else:
        print("Showing plot interactively...")
        plt.show()
        print("Plot window closed.")
