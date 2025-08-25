import os
import sys
from itertools import chain, zip_longest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from numpy.typing import NDArray

from h2o_sqw_calc.io import get_data_from_h5py
from h2o_sqw_calc.utils import flow


def odd_extend[T: np.number](arr: NDArray[T]) -> NDArray[T]:
    return np.concatenate((-arr[:0:-1], arr))


def even_extend[T: np.number](arr: NDArray[T]) -> NDArray[T]:
    return np.concatenate((arr[:0:-1], arr))


def reorder_legend_by_row(handles: list[Artist], labels: list[str], ncol: int) -> tuple[list[Artist], list[str]]:
    """Reorders legend items to fill by row instead of by column."""
    return flow(
        (handles, labels, ncol),
        lambda args: (zip_longest(*[iter(args[0])] * args[2]), zip_longest(*[iter(args[1])] * args[2])),
        lambda items_by_col: (chain.from_iterable(zip(*col, strict=False)) for col in items_by_col),
        lambda items_by_row: tuple([item for item in group if item is not None] for group in items_by_row),
    )


def get_gamma_data(
    element: str = "H", file_path: str = "data/last_{element}.gamma", include_classical: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.complexfloating], NDArray[np.float64] | None]:
    gamma_file = file_path.format(element)
    return flow(
        ("time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"),
        lambda keys: keys if include_classical else keys[:-1],
        lambda keys: [get_data_from_h5py(gamma_file, key) for key in keys],
        lambda data: [even_extend(e) if i % 2 != 0 else odd_extend(e) for i, e in enumerate(data)],
        lambda data: (data[0], data[1] + 1j * data[2], data[3] if include_classical else None),
    )


def get_stc_model_data(
    element: str = "H", file_path: str = "data/last.sqw"
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        get_data_from_h5py(file_path, f"inc_omega_{element}"),
        get_data_from_h5py(file_path, f"inc_vdos_{element}"),
    )


def get_sqw_classical_data(
    element: str = "H", file_path: str = "data/merged_h2o_293k.sqw"
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    return flow(
        (element, file_path),
        lambda args: [
            get_data_from_h5py(args[1], key)
            for key in (f"qVec_{args[0]}", f"inc_omega_{args[0]}", f"inc_sqw_{args[0]}")
        ],
        lambda data: (data[0], data[1], np.apply_along_axis(even_extend, -1, data[2])),
    )


def save_or_show_plot(output: str | None):
    if output:
        if os.path.exists(output):
            choice = input(f"File '{output}' already exists. Overwrite? (y/N): ").lower()
            if choice != "y":
                sys.exit("Aborted!")
        if output_dir := os.path.dirname(output):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output)
        print(f"Plot saved to {output}")
    else:
        print("Displaying plot interactively.")
        plt.show()


def get_q_values_from_cmdline(q_str_vals: list[str], q_step: float | None) -> NDArray[np.float64]:
    q_values: NDArray[np.float64]
    if len(q_str_vals) == 1 and "-" in q_str_vals[0]:
        start_q, end_q = map(float, q_str_vals[0].split("-"))
        q_values = np.arange(start_q, end_q, q_step, dtype=np.float64)
    else:
        q_values = np.array(q_str_vals, dtype=np.float64)
    return q_values
