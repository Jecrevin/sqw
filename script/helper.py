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
    file_path: str, include_classical: bool = False
) -> tuple[Array1D[np.float64], Array1D[np.complex128], Array1D[np.float64] | None]:
    all_keys = {"time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"}
    keys = all_keys if include_classical else all_keys - {"gamma_cls"}

    raw_data = {key: get_data_from_h5py(file_path, key) for key in keys}
    time = odd_extend(raw_data["time_vec"])
    gamma_qtm = even_extend(raw_data["gamma_qtm_real"]) + 1j * odd_extend(raw_data["gamma_qtm_imag"])
    gamma_cls = even_extend(raw_data["gamma_cls"]) if include_classical else None

    return time, gamma_qtm, gamma_cls


def get_stc_model_data(file_path: str, element: str) -> tuple[Array1D[np.float64], Array1D[np.float64]]:
    return (
        get_data_from_h5py(file_path, f"inc_omega_{element}"),
        get_data_from_h5py(file_path, f"inc_vdos_{element}"),
    )


def get_sqw_molecular_dynamics_data(
    file_path: str, element: str
) -> tuple[Array1D[np.float64], Array1D[np.float64], Array1D[np.float64]]:
    q_vals_key, omega_key, sqw_vstack_key = f"qVec_{element}", f"inc_omega_{element}", f"inc_sqw_{element}"

    q_vals = get_data_from_h5py(file_path, q_vals_key)
    omega = odd_extend(get_data_from_h5py(file_path, omega_key))
    sqw_vstack_data = np.apply_along_axis(even_extend, -1, get_data_from_h5py(file_path, sqw_vstack_key))

    return q_vals, omega, sqw_vstack_data


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
        print("Displaying plot interactively...")
        plt.show()


def _parse_q_interval(interval: str, default_step: float) -> set:
    parts = list(map(float, interval.split(":")))
    match parts:
        case [single]:
            return {single}
        case [start, end]:
            return set(np.arange(start, end + default_step / 2, default_step))
        case [start, end, step]:
            if step <= 0:
                raise ValueError(f"Step must be positive in interval '{interval}'.")
            return set(np.arange(start, end + step / 2, step))
        case _:
            raise ValueError(f"Invalid interval format: '{interval}'.")


def parse_q_values(q_intervals_str: list[str], step: float = 1.0) -> Array1D[np.float64]:
    return np.array(sorted(set.union(*(_parse_q_interval(q, step) for q in q_intervals_str))), dtype=np.float64)


def parse_indices(indices: list[str], default_step: int = 1) -> Array1D[np.int_]:
    all_indices = set()
    for interval in indices:
        parts = list(map(int, interval.split(":")))
        match parts:
            case [start, end, step]:
                if step <= 0:
                    raise ValueError("Step must be a positive integer.")
                all_indices.update(range(start, end + 1, step))
            case [start, end]:
                all_indices.update(range(start, end + 1, default_step))
            case [single]:
                all_indices.add(single)
            case _:
                raise ValueError(f"Invalid interval format: '{interval}'")
    return np.array(sorted(all_indices), dtype=np.int_)
