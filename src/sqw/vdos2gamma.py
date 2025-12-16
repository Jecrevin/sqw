"""Script to read VDOS from HDF5 and compute gamma using Filon integration."""

import argparse
import bisect
import ctypes
import platform
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sqw.consts import HBAR

try:
    import h5py
except ImportError:
    sys.exit("Install optional dependency group [gamma] to use this script.")


def conv_omega_to_time(tsize: int, dt: float, negative_axis: bool = True) -> tuple[float, np.ndarray]:
    """Convert omega axis to time axis."""
    data_range = 2 * np.pi / dt
    data_itv = data_range / (tsize - 1)
    data = np.linspace(0.0, data_range, tsize)
    if negative_axis:
        data = -np.flip(data)
    return data_itv, data


def gen_log_spacing(n: int, est_size: int, double_sized: bool = True) -> np.ndarray:
    """Generate logarithmically spaced indices for downsampling."""
    if double_sized:
        idx = np.logspace(0, np.log10(n // 2), est_size).astype(np.intp)
        idx = np.unique(idx)
        upidx = idx + n // 2
        upidx = np.insert(idx, 0, 0)
        lowidx = np.flip(upidx[0] - idx)
        slicing = np.concatenate((lowidx, upidx))
        slicing -= slicing[0]
        return slicing
    idx = np.logspace(0, np.log10(n), est_size).astype(np.intp)
    idx = np.unique(idx)
    slicing = np.insert(idx, 0, 0)
    return slicing


def load_h5_vdos(filename: str, omega_path: str, vdos_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and normalize VDOS from HDF5 file."""
    with h5py.File(filename, "r") as f:
        fre: NDArray = f[omega_path][()]  # type: ignore
        dos: NDArray = f[vdos_path][()]  # type: ignore

    if np.isnan(dos).any():
        raise RuntimeError("vdos has NaN values")

    # NumPy 2.0 compatibility: prefer trapezoid
    if hasattr(np, "trapezoid"):
        scale = np.trapezoid(dos, fre)
    else:
        scale = np.trapz(dos, fre)  # noqa: NPY201
    if scale == 0 or not np.isfinite(scale):
        raise RuntimeError("vdos normalization failed: zero or non-finite area")
    dos = dos / scale
    return fre, dos


def load_filon_lib() -> ctypes.CDLL:
    """Load the Filon integration C library."""
    lib_path = Path(__file__).parent / "_filon/libfilon"
    match platform.system():
        case "Linux":
            lib_path = lib_path.with_suffix(".so")
        case "Darwin":
            lib_path = lib_path.with_suffix(".dylib")
        case "Windows":
            lib_path = lib_path.with_suffix(".dll")

    if not lib_path.exists():
        raise FileNotFoundError(f"C library not found: {lib_path}")

    lib = ctypes.CDLL(str(lib_path))

    np1d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")

    lib.cal_limit.restype = ctypes.c_void_p
    lib.cal_limit.argtypes = [ctypes.c_int, ctypes.c_double, np1d, np1d, ctypes.c_int, np1d, np1d, np1d, np1d]
    lib.cal_integral.restype = ctypes.c_void_p
    lib.cal_integral.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        np1d,
        np1d,
        ctypes.c_int,
        np1d,
        np1d,
        np1d,
        np1d,
    ]

    return lib


def crop_vdos(freq: np.ndarray, dos: np.ndarray, e_max: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Crop VDOS at a maximum energy threshold."""
    thr = e_max / HBAR  # Convert to angular frequency threshold
    if freq[-1] > thr:
        idx = np.where(freq >= thr)[0][0]
        freq = freq[:idx]
        dos = dos[:idx]
        dos = dos - dos[-1]
    return freq, dos


def upsample_vdos(freq: np.ndarray, dos: np.ndarray, times: int) -> tuple[np.ndarray, np.ndarray]:
    """Upsample VDOS by linear interpolation."""
    if times <= 1:
        return freq, dos
    length = freq.size * int(times)
    freq_new = np.linspace(freq[0], freq[-1], length)
    dos_new = np.interp(freq_new, freq, dos)
    return freq_new, dos_new


def cal_taylor_limit(
    lib: ctypes.CDLL, mass_num: int, temperature: float, x: np.ndarray, y: np.ndarray, tarr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Taylor expansion limit part of gamma."""
    l_cls = np.zeros(tarr.size, dtype=np.float64)
    l_real = np.zeros(tarr.size, dtype=np.float64)
    l_imag = np.zeros(tarr.size, dtype=np.float64)
    lib.cal_limit(int(mass_num), float(temperature), x, y, int(tarr.size), tarr, l_cls, l_real, l_imag)
    return l_cls, l_real, l_imag


def cal_integral_filon(
    lib: ctypes.CDLL, mass_num: int, temperature: float, x: np.ndarray, y: np.ndarray, tarr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate integral part of gamma using Filon method."""
    if x.size % 2 == 0:
        x_use = x[1:]
        y_use = y[1:]
    else:
        x_use = x[1:-1]
        y_use = y[1:-1]
    panels = (x_use.size - 1) // 2

    i_cls = np.zeros(tarr.size, dtype=np.float64)
    i_real = np.zeros(tarr.size, dtype=np.float64)
    i_imag = np.zeros(tarr.size, dtype=np.float64)
    lib.cal_integral(
        int(mass_num), float(temperature), int(panels), x_use, y_use, int(tarr.size), tarr, i_cls, i_real, i_imag
    )
    return i_cls, i_real, i_imag


def cal_gamma_filon(
    lib: ctypes.CDLL, mass_num: int, temperature: float, x: np.ndarray, y: np.ndarray, tarr: np.ndarray, times: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate gamma using Filon integration (with optional upsampling)."""
    if times and times > 1:
        x, y = upsample_vdos(x, y, times)
    l_cls, l_real, l_imag = cal_taylor_limit(lib, mass_num, temperature, x, y, tarr)
    i_cls, i_real, i_imag = cal_integral_filon(lib, mass_num, temperature, x, y, tarr)
    return l_cls + i_cls, l_real + i_real, l_imag + i_imag


def compute_gamma(
    lib: ctypes.CDLL, freq: np.ndarray, dos: np.ndarray, temperature: float, mass_num: int, size_shrink_fact: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute gamma from VDOS using Filon integration with optional time vector shrinking."""
    df = freq[1] - freq[0]
    _, time_axis = conv_omega_to_time(freq.size * 4, df, negative_axis=False)

    if int(size_shrink_fact) > 1:
        tmp = 2 * np.pi / df / 10.0
        idx = bisect.bisect(time_axis.tolist(), tmp)
        time_p1 = time_axis[:idx]
        time_p2 = time_axis[idx:]

        shrink_idx = gen_log_spacing(time_p2.size, time_p2.size // int(size_shrink_fact), False)
        if shrink_idx[-1] == time_p2.size:
            shrink_idx[-1] -= 1
        time_p2_shrunk = time_p2[shrink_idx]

        cls1, real1, imag1 = cal_gamma_filon(lib, mass_num, temperature, freq, dos, time_p1, times=10)
        cls2_s, real2_s, imag2_s = cal_gamma_filon(lib, mass_num, temperature, freq, dos, time_p2_shrunk, times=10)

        # Linear interpolation back to original time_p2 (original used spline,
        # here np.interp is used to keep dependencies minimal)
        real2 = np.interp(time_p2, time_p2_shrunk, real2_s)
        imag2 = np.interp(time_p2, time_p2_shrunk, imag2_s)
        cls2 = np.interp(time_p2, time_p2_shrunk, cls2_s)

        g_cls = np.concatenate((cls1, cls2))
        g_real = np.concatenate((real1, real2))
        g_imag = np.concatenate((imag1, imag2))
    else:
        g_cls, g_real, g_imag = cal_gamma_filon(lib, mass_num, temperature, freq, dos, time_axis, times=10)

    return time_axis, g_cls, g_real, g_imag


def save_gamma_h5(
    filename: str,
    time_vec: np.ndarray,
    g_cls: np.ndarray,
    g_real: np.ndarray,
    g_imag: np.ndarray,
    ds_names: dict[str, str],
) -> None:
    """Save computed gamma data to HDF5 file."""
    with h5py.File(filename, "w") as f0:
        f0.create_dataset(ds_names["time"], data=time_vec, compression="gzip")
        f0.create_dataset(ds_names["real"], data=g_real, compression="gzip")
        f0.create_dataset(ds_names["imag"], data=g_imag, compression="gzip")
        f0.create_dataset(ds_names["cls"], data=g_cls, compression="gzip")


def run_once(
    lib: ctypes.CDLL,
    input_fname: str,
    omega_ds: str,
    vdos_ds: str,
    output_fname: str,
    temperature: float,
    mass_num: int,
    size_shrink_fact: int,
    ds_names: dict[str, str],
    emax_vdos: float = 1.0,
) -> None:
    """Run gamma computation for a single element and save results."""
    fre, dos = load_h5_vdos(input_fname, omega_ds, vdos_ds)
    fre, dos = crop_vdos(fre, dos, e_max=emax_vdos)

    time_vec, g_cls, g_real, g_imag = compute_gamma(lib, fre, dos, temperature, mass_num, size_shrink_fact)
    save_gamma_h5(output_fname, time_vec, g_cls, g_real, g_imag, ds_names)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        type=str,
        default="",
        dest="in_name",
        help="input HDF5 filename (path included)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        action="store",
        type=float,
        default=0.0,
        dest="temp",
        help="temperature in K",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        default="",
        dest="out_name",
        help="comma-separated output gamma filenames",
    )
    parser.add_argument(
        "--omega-ds",
        action="store",
        type=str,
        default="",
        dest="omega_ds",
        help="comma-separated omega dataset names",
    )
    parser.add_argument(
        "--vdos-ds",
        action="store",
        type=str,
        default="",
        dest="vdos_ds",
        help="comma-separated vdos dataset names",
    )
    parser.add_argument(
        "-n",
        "--num",
        action="store",
        type=str,
        default="1",
        dest="num",
        help="comma-separated atom masses (unit: neutron mass count)",
    )
    parser.add_argument(
        "-s",
        "--sizeShrinkFact",
        action="store",
        type=int,
        default=10,
        dest="size_shrink_fact",
        help="time vector shrinking factor for calculation acceleration",
    )
    parser.add_argument(
        "--ds-time",
        action="store",
        type=str,
        default="time_vec",
        dest="ds_time",
        help="output dataset name for time vector",
    )
    parser.add_argument(
        "--ds-cls",
        action="store",
        type=str,
        default="gamma_cls",
        dest="ds_cls",
        help="output dataset name for classical gamma",
    )
    parser.add_argument(
        "--ds-real",
        action="store",
        type=str,
        default="gamma_qtm_real",
        dest="ds_real",
        help="output dataset name for real part of quantum gamma",
    )
    parser.add_argument(
        "--ds-imag",
        action="store",
        type=str,
        default="gamma_qtm_imag",
        dest="ds_imag",
        help="output dataset name for imaginary part of quantum gamma",
    )
    return parser.parse_args()


def main() -> int:
    """Parse arguments and run gamma computations."""
    args = parse_args()
    temperature = args.temp
    input_fname = args.in_name
    output_list = str.split(args.out_name, ",") if args.out_name else []
    omega_ds_list = str.split(args.omega_ds, ",") if args.omega_ds else []
    vdos_ds_list = str.split(args.vdos_ds, ",") if args.vdos_ds else []
    nums = [int(x) for x in str.split(args.num, ",") if x]
    size_shrink_fact = args.size_shrink_fact

    ds_names = {
        "time": args.ds_time,
        "cls": args.ds_cls,
        "real": args.ds_real,
        "imag": args.ds_imag,
    }

    if not input_fname or not output_list or not omega_ds_list or not vdos_ds_list or not nums:
        print("Missing required arguments: input/output/omega-ds/vdos-ds/num")
        return 2
    if not (len(output_list) == len(omega_ds_list) == len(vdos_ds_list) == len(nums)):
        print("Arguments length mismatch among output/omega-ds/vdos-ds/num")
        return 2

    lib = load_filon_lib()
    emax_vdos = 1.0  # unit: eV

    for omega_ds, vdos_ds, out_fn, mass_n in zip(omega_ds_list, vdos_ds_list, output_list, nums, strict=True):
        begin = time.time()
        run_once(
            lib,
            input_fname,
            omega_ds,
            vdos_ds,
            out_fn,
            float(temperature),
            int(mass_n),
            int(size_shrink_fact),
            ds_names,
            emax_vdos,
        )
        print("finish time:", time.time() - begin, "dataset =", vdos_ds)

    return 0


if __name__ == "__main__":
    main()
