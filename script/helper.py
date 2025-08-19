from itertools import chain, zip_longest

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray

from h2o_sqw_calc.io import get_data_from_h5py
from h2o_sqw_calc.utils import flow


def odd_extend[T: np.number](arr: NDArray[T]) -> NDArray[T]:
    return np.concatenate((-arr[:0:-1], arr))


def even_extend[T: np.number](arr: NDArray[T]) -> NDArray[T]:
    return np.concatenate((arr[:0:-1], arr))


def get_gamma_data(
    element: str = "H", file_path: str = "data/last_{element}.gamma"
) -> tuple[NDArray[np.float64], NDArray[np.complexfloating]]:
    return flow(
        (element, file_path),
        lambda args: args[1].format(element=args[0]),
        lambda file_path: [
            get_data_from_h5py(file_path, key) for key in ("time_vec", "gamma_qtm_real", "gamma_qtm_imag")
        ],
        lambda data: [even_extend(arr) if (i + 1) % 2 == 0 else odd_extend(arr) for i, arr in enumerate(data)],
        lambda data: (data[0], data[1] + 1j * data[2]),
    )


def get_stc_model_data(
    element: str = "H", file_path: str = "data/last.sqw"
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        get_data_from_h5py(file_path, f"inc_omega_{element}"),
        get_data_from_h5py(file_path, f"inc_vdos_{element}"),
    )


def reorder_legend_by_row(handles: list[Artist], labels: list[str], ncol: int) -> tuple[list[Artist], list[str]]:
    """Reorders legend items to fill by row instead of by column."""
    return flow(
        (handles, labels, ncol),
        lambda args: (zip_longest(*[iter(args[0])] * args[2]), zip_longest(*[iter(args[1])] * args[2])),
        lambda items_by_col: (chain.from_iterable(zip(*col, strict=False)) for col in items_by_col),
        lambda items_by_row: tuple([item for item in group if item is not None] for group in items_by_row),
    )
