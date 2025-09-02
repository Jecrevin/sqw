import os
import sys
from itertools import chain, zip_longest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.artist import Artist

from h2o_sqw_calc.io import get_data_from_h5py
from h2o_sqw_calc.typing import Array1D


def odd_extend[T: np.number](arr: Array1D[T]) -> Array1D[T]:
    return np.concatenate((-arr[:0:-1], arr))


def even_extend[T: np.number](arr: Array1D[T]) -> Array1D[T]:
    return np.concatenate((arr[:0:-1], arr))


def reorder_legend_by_row(handles: list[Artist], labels: list[str], ncol: int) -> tuple[list[Artist], list[str]]:
    """Reorders legend items to fill by row instead of by column."""
    handles_by_col = zip_longest(*[iter(handles)] * ncol)
    labels_by_col = zip_longest(*[iter(labels)] * ncol)

    items_by_col = (handles_by_col, labels_by_col)
    items_by_row = (chain.from_iterable(zip(*col, strict=False)) for col in items_by_col)

    handles_reordered, labels_reordered = ([item for item in group if item is not None] for group in items_by_row)

    return handles_reordered, labels_reordered


def get_gamma_data(
    element: str = "H", file_path: str = "data/last_{element}.gamma", include_classical: bool = False
) -> tuple[Array1D[np.float64], Array1D[np.complex128], Array1D[np.float64] | None]:
    gamma_file = file_path.format(element=element)
    keys = ("time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls")
    if not include_classical:
        keys = keys[:-1]

    data = [get_data_from_h5py(gamma_file, key) for key in keys]
    extended_data = [even_extend(e) if i % 2 != 0 else odd_extend(e) for i, e in enumerate(data)]
    time_vec = extended_data[0]
    gamma_qtm = extended_data[1] + 1j * extended_data[2]
    gamma_cls = extended_data[3] if include_classical else None

    return time_vec, gamma_qtm, gamma_cls


def get_stc_model_data(
    element: str = "H", file_path: str = "data/last.sqw"
) -> tuple[Array1D[np.float64], Array1D[np.float64]]:
    return (
        get_data_from_h5py(file_path, f"inc_omega_{element}"),
        get_data_from_h5py(file_path, f"inc_vdos_{element}"),
    )


def get_sqw_classical_data(
    element: str = "H", file_path: str = "data/merged_h2o_293k.sqw"
) -> tuple[Array1D[np.float64], Array1D[np.float64], Array1D[np.float64]]:
    keys = (f"qVec_{element}", f"inc_omega_{element}", f"inc_sqw_{element}")
    data = [get_data_from_h5py(file_path, key) for key in keys]
    q_vec, omega, sqw = data
    sqw_extended = np.apply_along_axis(even_extend, -1, sqw)
    return q_vec, omega, sqw_extended


def save_or_show_plot(output: str | None) -> None:
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


def get_q_values_from_cmdline(q_str_vals: list[str], q_step: float | None) -> Array1D[np.float64]:
    q_values: Array1D[np.float64]
    if len(q_str_vals) == 1 and "-" in q_str_vals[0]:
        start_q, end_q = map(float, q_str_vals[0].split("-"))
        q_values = np.arange(start_q, end_q, q_step, dtype=np.float64)
    else:
        q_values = np.array(q_str_vals, dtype=np.float64)
    return q_values
