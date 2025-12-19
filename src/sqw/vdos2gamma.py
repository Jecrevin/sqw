"""Convert VDOS data to Width Function (Gamma) using Filon's method."""

import argparse
import ctypes
import platform
import sys
import time
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from scipy.interpolate import CubicSpline

from sqw.consts import HBAR, PI
from sqw.typing_ import Array1D


def main():
    """Parse arguments and run the VDOS to Gamma conversion."""
    parser = _parse_args()
    args = parser.parse_args()

    input_file: str = args.input_file
    freq_dataset, vdos_dataset = args.input_datasets
    temperature: float = args.temperature
    num_mass: int = args.num_mass
    vdos_energy_max: float = args.vdos_energy_max
    size_shrink_factor: int = args.size_shrink_factor
    output_file: str = args.output_file
    output_datasets: list[str] = args.output_dataset

    print("Loading input VDOS data...")

    raw_freq_data, raw_vdos_data = _load_vdos_data(input_file, freq_dataset, vdos_dataset)

    freq_data = raw_freq_data[raw_freq_data <= vdos_energy_max / HBAR]
    vdos_data = raw_vdos_data[: freq_data.size] - raw_vdos_data[freq_data.size - 1]

    print("Loading Filon library...")

    filon_base_path = Path(__file__).parent / "_filon/libfilon"
    match platform.system():
        case "Linux":
            filon_path = filon_base_path.with_suffix(".so")
        case "Darwin":
            filon_path = filon_base_path.with_suffix(".dylib")
        case "Windows":
            filon_path = filon_base_path.with_suffix(".dll")
        case _:
            sys.exit("This script only supports Linux, macOS, and Windows currently, sorry for inconvenience.")
    global filon
    filon = ctypes.CDLL(filon_path)
    _add_filon_call_signatures()

    print("Starting VDOS to Gamma conversion...")

    begin_time = time.time()
    time_vec = np.linspace(0, 2 * PI / (freq_data[1] - freq_data[0]), freq_data.size * 4)
    gamma_cls, gamma_qtm_real, gamma_qtm_imag = calculate_gamma_from_vdos(
        time_vec,
        freq_data,
        vdos_data,
        temperature,
        num_mass,
        size_shrink_factor,
    )
    end_time = time.time()

    print(f"VDOS to Gamma conversion completed in {end_time - begin_time:.2f} seconds.")

    if output_file:
        print("Saving output Gamma data...")

        try:
            with h5py.File(output_file, "w") as f:
                f.create_dataset(output_datasets[0], data=time_vec)
                f.create_dataset(output_datasets[1], data=gamma_qtm_real)
                f.create_dataset(output_datasets[2], data=gamma_qtm_imag)
                f.create_dataset(output_datasets[3], data=gamma_cls)
        except OSError as e:
            sys.exit(f"Error writing to output file: {e}")
        except Exception as e:
            sys.exit(f"An unexpected error occurred while saving output data: {e}")

        print(f"Output Gamma data have been saved to '{Path(output_file).resolve()}'.")
    else:
        return gamma_cls, gamma_qtm_real, gamma_qtm_imag


def _parse_args():
    parser = argparse.ArgumentParser(description="Convert VDOS data to Width Function (Gamma).")

    parser.add_argument("input_file", type=str, help="Path to the input VDOS data file.")
    parser.add_argument(
        "input_datasets",
        type=str,
        nargs=2,
        metavar=("FREQ_DATASET", "VDOS_DATASET"),
        help="Names of the datasets in the input file for frequency and VDOS data.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        help="Temperature in Kelvin.",
    )
    parser.add_argument(
        "-n",
        "--num-mass",
        type=int,
        default=1,
        help="Number of neutron masses.",
    )
    parser.add_argument(
        "--vdos-energy-max",
        type=float,
        default=1.0,
        help="Maximum energy value for VDOS data (in eV). Default is 1.0 eV. (Higher data will be ignored)",
    )
    parser.add_argument(
        "--size-shrink-factor",
        type=int,
        default=1,
        help="Factor to shrink the size of input data for calculation acceleration. Default is 1 (no shrinking).",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Path to the output file for the Width Function (Gamma) data.",
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        nargs=4,
        metavar=("TIME", "GAMMA_QTM_REAL", "GAMMA_QTM_IMAG", "GAMMA_CLS"),
        default=["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"],
        help="Names of the datasets to create in the output file for time vector, "
        "real and imaginary parts of Gamma QTM, and Gamma CLS.",
    )

    return parser


def _load_vdos_data(input_file: str, freq_dataset: str, vdos_dataset: str) -> tuple[Array1D, Array1D]:
    try:
        with h5py.File(input_file) as f:
            freq_data: Array1D = cast(h5py.Dataset, f[freq_dataset])[()]
            vdos_data: Array1D = cast(h5py.Dataset, f[vdos_dataset])[()]
    except FileNotFoundError as e:
        sys.exit(f"Input file not found: {e}")
    except OSError as e:
        sys.exit(f"Error opening input file: {e}")
    except KeyError as e:
        sys.exit(f"Error accessing dataset in input file: {e}")
    except Exception as e:
        sys.exit(f"An unexpected error occurred while loading input data: {e}")

    return freq_data, vdos_data


def _add_filon_call_signatures() -> None:
    np1darray = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    filon.cal_limit.restype = ctypes.c_void_p
    filon.cal_limit.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np1darray,
        np1darray,
        ctypes.c_int,
        np1darray,
        np1darray,
        np1darray,
        np1darray,
    ]

    filon.cal_integral.restype = ctypes.c_void_p
    filon.cal_integral.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        np1darray,
        np1darray,
        ctypes.c_int,
        np1darray,
        np1darray,
        np1darray,
        np1darray,
    ]


def calculate_gamma_from_vdos(
    time_vec: Array1D, freq: Array1D, vdos: Array1D, temperature: float, num_mass: int, size_shrink_factor: int
) -> tuple[Array1D, Array1D, Array1D]:
    """Calculate the Width Function (Gamma) from VDOS data using Filon's method."""
    if size_shrink_factor > 1:
        time_thresh = 2 * PI / (freq[1] - freq[0]) / 10
        time_part1 = time_vec[time_vec <= time_thresh]
        time_part2 = time_vec[time_vec > time_thresh]
        shrinked_time_part2 = _log_shrink_vector(time_vec[time_vec > time_thresh], size_shrink_factor)

        cls_part1, qtm_real_part1, qtm_imag_part1 = _calculate_gamma_core(
            time_part1, freq, vdos, upsample_factor=10, num_mass=num_mass, temperature=temperature
        )
        shrinked_cls_part2, shrinked_qtm_real_part2, shrinked_qtm_imag_part2 = _calculate_gamma_core(
            shrinked_time_part2, freq, vdos, upsample_factor=10, num_mass=num_mass, temperature=temperature
        )

        cls_part2 = CubicSpline(shrinked_time_part2, shrinked_cls_part2)(time_part2)
        qtm_real_part2 = CubicSpline(shrinked_time_part2, shrinked_qtm_real_part2)(time_part2)
        qtm_imag_part2 = CubicSpline(shrinked_time_part2, shrinked_qtm_imag_part2)(time_part2)

        gamma_cls = np.concatenate([cls_part1, cls_part2])
        gamma_qtm_real = np.concatenate([qtm_real_part1, qtm_real_part2])
        gamma_qtm_imag = np.concatenate([qtm_imag_part1, qtm_imag_part2])
    else:
        gamma_cls, gamma_qtm_real, gamma_qtm_imag = _calculate_gamma_core(
            time_vec, freq, vdos, upsample_factor=10, num_mass=num_mass, temperature=temperature
        )

    return gamma_cls, gamma_qtm_real, gamma_qtm_imag


def _log_shrink_vector(vec: Array1D, shrink_factor: int) -> Array1D:
    idxs = np.unique(np.logspace(0, np.log10(vec.size), vec.size // shrink_factor, dtype=np.intp) - np.intp(1))
    return vec[idxs].copy()


def _calculate_gamma_core(
    time_vec: Array1D, freq: Array1D, dos: Array1D, /, upsample_factor: int, num_mass: int, temperature: float
) -> tuple[Array1D, Array1D, Array1D]:
    x, y = cubic_spline_upsample(freq, dos, upsample_factor)

    limit_cls = np.zeros_like(time_vec)
    limit_qtm_real = np.zeros_like(time_vec)
    limit_qtm_imag = np.zeros_like(time_vec)

    # NOTE: This changes the `limit_*` arrays in place.
    filon.cal_limit(num_mass, temperature, x, y, time_vec.size, time_vec, limit_cls, limit_qtm_real, limit_qtm_imag)

    x = x[1:] if x.size % 2 == 0 else x[1:-1]
    y = y[1:] if y.size % 2 == 0 else y[1:-1]
    panels = (x.size - 1) // 2

    integral_cls = np.zeros_like(time_vec)
    integral_qtm_real = np.zeros_like(time_vec)
    integral_qtm_imag = np.zeros_like(time_vec)

    # NOTE: This changes the `integral_*` arrays in place.
    filon.cal_integral(
        num_mass, temperature, panels, x, y, time_vec.size, time_vec, integral_cls, integral_qtm_real, integral_qtm_imag
    )

    gamma_cls = limit_cls + integral_cls
    gamma_qtm_real = limit_qtm_real + integral_qtm_real
    gamma_qtm_imag = limit_qtm_imag + integral_qtm_imag

    return gamma_cls, gamma_qtm_real, gamma_qtm_imag


def cubic_spline_upsample(x: Array1D, y: Array1D, upsample_factor: int) -> tuple[Array1D, Array1D]:
    """Upsample the given x and y data using cubic spline interpolation.

    Parameters
    ----------
    x : Array1D
        The x data points.
    y : Array1D
        The y data points.
    upsample_factor : int
        The factor by which to upsample the data.

    Returns
    -------
    Array1D, Array1D
        The upsampled x and y data points.

    """
    if upsample_factor <= 1:
        return x, y
    spline = CubicSpline(x, y)
    x_new = np.linspace(x[0], x[-1], x.size * upsample_factor)
    y_new = spline(x_new)
    return x_new, y_new  # type: ignore


if __name__ == "__main__":
    main()
