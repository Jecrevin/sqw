import os

import h5py
from numpy.typing import NDArray


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
