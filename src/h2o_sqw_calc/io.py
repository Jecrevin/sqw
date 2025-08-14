import os

import h5py
import numpy as np
from numpy.typing import NDArray

from .math import even_extend, odd_extend


class NotH5FileError(Exception):
    pass


def get_data_from_h5py(file_path: str, dataset_name: str) -> NDArray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'The file "{file_path}" does not exist.')

    try:
        with h5py.File(file_path, "r") as f:
            if dataset_name not in f:
                raise KeyError(f'The dataset "{dataset_name}" does not exist in the file "{file_path}".')
            if not isinstance((dataset := f[dataset_name]), h5py.Dataset):
                raise TypeError(f'"{dataset_name}" is not a valid HDF5 dataset in the file "{file_path}".')

            return dataset[()]
    except OSError as e:
        raise NotH5FileError(f'The file "{file_path}" is not a valid HDF5 file.') from e


def get_gamma_data(element: str, data_dir: str = "data") -> tuple[NDArray[np.float64], NDArray[np.complexfloating]]:
    time = odd_extend(get_data_from_h5py(f"{data_dir}/last_{element}.gamma", "time_vec"))
    gamma_qtm_re = even_extend(get_data_from_h5py(f"{data_dir}/last_{element}.gamma", "gamma_qtm_real"))
    gamma_qtm_im = odd_extend(get_data_from_h5py(f"{data_dir}/last_{element}.gamma", "gamma_qtm_imag"))
    gamma_qtm = gamma_qtm_re + 1j * gamma_qtm_im

    return time, gamma_qtm


def get_stc_model_data(element: str, data_dir: str = "data") -> tuple[NDArray, NDArray]:
    freq_dos = get_data_from_h5py(f"{data_dir}/last.sqw", f"inc_omega_{element}")
    dos = get_data_from_h5py(f"{data_dir}/last.sqw", f"inc_vdos_{element}")
    return freq_dos, dos
