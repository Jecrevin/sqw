from itertools import chain, zip_longest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.artist import Artist

from sqw.consts import HBAR, KB
from sqw.typing import Array1D


def even_extend[T: np.number](arr: Array1D[T]) -> Array1D[T]:
    return np.concatenate((arr[-1:0:-1], arr))


def odd_extend[T: np.number](arr: Array1D[T]) -> Array1D[T]:
    return np.concatenate((-arr[-1:0:-1], arr))


def assure_detailed_balance(
    freq: Array1D[np.floating], sqw: Array1D[np.floating], temperature: float
) -> Array1D[np.floating]:
    freq_pos = freq[freq >= 0]
    sqw_pos = np.interp(-freq_pos, freq, sqw) * np.exp(-HBAR * freq_pos / (KB * temperature))

    return np.concatenate((sqw[freq < 0], sqw_pos))


def setup_pyplot_style() -> None:
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["axes.labelsize"] = 14


def parse_interval(interval: str, default_step: float) -> Array1D[np.double]:
    parts = map(float, interval.split(":"))
    match list(parts):
        case [start, end, step]:
            return np.linspace(start, end, int((end - start) / step) + 1)
        case [start, end]:
            return np.linspace(start, end, int((end - start) / default_step) + 1)
        case [single]:
            return np.array([single], dtype=np.double)
        case _:
            raise ValueError(f"Invalid interval format: '{interval}'!")


def parse_intervals(intervals: list[str], default_step: float) -> Array1D[np.double]:
    return np.unique(np.concatenate([parse_interval(interval, default_step) for interval in intervals]))


def parse_slice(slice_: str) -> Array1D[np.intp]:
    parts = map(int, slice_.split(":"))
    match list(parts):
        case [start, end, step]:
            return np.arange(start, end, step, dtype=np.intp)
        case [start, end]:
            return np.arange(start, end, dtype=np.intp)
        case [single]:
            return np.array([single], dtype=np.intp)
        case _:
            raise ValueError(f"Invalid slice format: '{slice_}'!")


def parse_slices(slices: list[str]) -> Array1D[np.intp]:
    return np.unique(np.concatenate([parse_slice(slice_) for slice_ in slices]))


def reorder_legend_by_row(handles: list[Artist], labels: list[str], ncol: int) -> tuple[list[Artist], list[str]]:
    """Reorders legend items to fill by row instead of by column."""
    handles_by_col = zip_longest(*[iter(handles)] * ncol)
    labels_by_col = zip_longest(*[iter(labels)] * ncol)

    items_by_col = (handles_by_col, labels_by_col)
    items_by_row = (chain.from_iterable(zip(*col, strict=False)) for col in items_by_col)

    handles_reordered, labels_reordered = ([item for item in group if item is not None] for group in items_by_row)

    return handles_reordered, labels_reordered
