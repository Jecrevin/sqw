import h5py
import numpy as np
from numpy.typing import NDArray


class NotH5FileError(Exception):
    pass


def get_data_from_h5py[T: np.number](file_path: str, dataset_name: str) -> NDArray[T]:
    try:
        with h5py.File(file_path, "r") as f:
            if not isinstance((dataset := f[dataset_name]), h5py.Dataset):
                raise TypeError(f'"{dataset_name}" is not a valid HDF5 dataset in the file "{file_path}".')
            return dataset[()]
    except OSError as e:
        raise NotH5FileError(f'Could not open "{file_path}" as an HDF5 file.') from e
    except KeyError as e:
        raise KeyError(f'"{dataset_name}" not found in HDF5 file "{file_path}".') from e
