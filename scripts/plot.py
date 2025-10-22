#!./.venv/bin/python3

import argparse
import sys
from functools import partial
from typing import Final, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.typing import NDArray
from utils.helper import (
    assure_detailed_balance,
    parse_interval,
    parse_intervals,
    parse_slices,
    reorder_legend_by_row,
    setup_pyplot_style,
)
from utils.io import read_gamma_data, read_md_data, read_stc_data, save_or_show_plot

from sqw import sqw_ga_model
from sqw._core import sqw_gaaqc_model, sqw_stc_model
from sqw.consts import HBAR, NEUTRON_MASS, PI
from sqw.typing import Array1D


def main() -> None:
    setup_pyplot_style()
    parser = setup_parser()
    args = parser.parse_args()
    args.func(args)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()  # TODO: add description

    # helper parser stores arguments shared by sub-commands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--gamma-keys",
        type=str,
        nargs=4,
        metavar=("TIME", "GAMMA_QTM_R", "GAMMA_QTM_I", "GAMMA_CLS"),
        default=["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"],
        help="Keys to read width function datasets, ordered as time, quantum "
        "gamma real part, quantum gamma imaginary part and classical gamma "
        "(default: ['time_vec', 'gamma_qtm_real', 'gamma_qtm_imag', 'gamma_cls']).",
    )
    common_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file to save the plot (if not provided, show the plot interactively).",
    )

    subparsers = parser.add_subparsers(required=True)

    # sub-command to plot width function data
    parser_gamma = subparsers.add_parser(
        "gamma",
        parents=[common_parser],
        help="Plot quantum width function values from HDF5 file.",
    )
    parser_gamma.add_argument("file", type=str, help="HDF5 file containing width function data.")
    parser_gamma.add_argument(
        "--no-extend", action="store_true", help="Not extend the width function to negative time axis."
    )
    parser_gamma.set_defaults(func=plot_gamma_data)

    # sub-command to plot S(q,w) data
    parser_sqw = subparsers.add_parser(
        "sqw",
        parents=[common_parser],
        help="Plot scattering function values calculated from HDF5 files using different models.",
    )

    # helper parser stores arguments shared by sqw sub-commands
    common_sqw_parser = argparse.ArgumentParser(add_help=False)
    common_sqw_parser.add_argument(
        "--stc-keys",
        type=str,
        nargs=2,
        default=["inc_omega_H", "inc_vdos_H"],
        help="Keys to read datasets used in STC model, ordered as frequency "
        "and value of Density of States (default: ['inc_omega_H', 'inc_vdos_H']).",
    )
    common_sqw_parser.add_argument(
        "--temperature", type=float, default=293.0, help="Temperature in Kelvin (default: 293.0 K)."
    )
    common_sqw_parser.add_argument(
        "--scale", type=str, choices=["linear", "log"], default="log", help="Y axis scale (default: log)."
    )
    common_sqw_parser.add_argument(
        "--frequency-axis",
        action="store_true",
        help="Use frequency (in Hz) instead of energy (in eV) on axis.",
    )

    sqw_subparsers = parser_sqw.add_subparsers(required=True)

    # sub-command to plot S(q,w) using GA model
    parser_sqw_ga = sqw_subparsers.add_parser(
        "ga",
        parents=[common_parser, common_sqw_parser],
        help="Use Gaussian Approximation (GA) model to calculate S(q,w).",
    )
    parser_sqw_ga.add_argument(
        "q_interval",
        type=str,
        nargs="+",
        help="Interval(s) of momentum transfer Q in 1/Angstrom, format: START[:END[:STEP]].",
    )
    parser_sqw_ga.add_argument("gamma_file", type=str, help="HDF5 file containing width function data.")
    parser_sqw_ga.add_argument("--stc-file", type=str, help="HDF5 file containing DOS data for STC model.")
    parser_sqw_ga.add_argument(
        "--method",
        type=str,
        choices=["fft", "cdft"],
        default="cdft",
        help="Method to compute S(q,w), available options are 'fft' and 'cdft' (default: cdft).",
    )
    parser_sqw_ga.add_argument(
        "--restrict-view",
        "-r",
        action="store_true",
        help="Restrict x-axis to (-10, 2) eV in energy or the equivalent in frequency.",
    )
    parser_sqw_ga.set_defaults(func=plot_sqw_ga)

    # sub-command to plot S(q,w) as 2D color map using GA model
    parser_sqw_ga2d = sqw_subparsers.add_parser(
        "ga2d",
        parents=[common_parser, common_sqw_parser],
        help="Use Gaussian Approximation (GA) model to calculate and plot S(q,w) as 2D color map.",
    )
    parser_sqw_ga2d.add_argument(
        "q_slice",
        type=str,
        help="Slice of momentum transfer Q in 1/Ang, format: START:END[:STEP] (default STEP is 5).",
    )
    parser_sqw_ga2d.add_argument("file", type=str, help="HDF5 file containing width function data.")
    parser_sqw_ga2d.set_defaults(func=plot_sqw_ga2d)

    # sub-command to plot S(q,w) using GAAQC model
    parser_sqw_gaaqc = sqw_subparsers.add_parser(
        "gaaqc",
        parents=[common_parser, common_sqw_parser],
        help="Use GA with quantum correction model to calculate S(q,w).",
    )
    parser_sqw_gaaqc.add_argument(
        "indices",
        type=str,
        nargs="+",
        help="Indices of momentum transfer Q values in MD data, format: START[:END[:STEP]] or single INDEX.",
    )
    parser_sqw_gaaqc.add_argument("gamma_file", type=str, help="HDF5 file containing width function data.")
    parser_sqw_gaaqc.add_argument(
        "md_file", type=str, help="HDF5 file containing MD S(q,w) data for quantum correction."
    )
    parser_sqw_gaaqc.add_argument(
        "--with-ga-results",
        action="store_true",
        help="Also plot GA results without quantum correction for comparison.",
    )
    parser_sqw_gaaqc.add_argument(
        "--md-keys",
        type=str,
        nargs=3,
        default=["qVec_H", "inc_omega_H", "inc_sqw_H"],
        metavar=("Q_VALS", "OMEGA", "SQW_STACK"),
        help="Keys to read datasets used in GAAQC model, ordered as q values, "
        "frequency and S(q,w) 2D stack (default: ['qVec_H', 'inc_omega_H', 'inc_sqw_H']).",
    )
    parser_sqw_gaaqc.set_defaults(func=plot_sqw_gaaqc)

    return parser


############################## Main Plot Funcions #############################


def plot_gamma_data(args: argparse.Namespace) -> None:
    GAMMA_KEYS: Final[list[str]] = args.gamma_keys
    OUTPUT: Final[str | None] = args.output
    FILE: Final[str] = args.file
    NO_EXTEND: Final[bool] = args.no_extend

    try:
        time, gamma_qtm, _ = read_gamma_data(FILE, GAMMA_KEYS, no_extend=NO_EXTEND, include_cls=False)
    except Exception as e:
        sys.exit(f"Error reading width function data: {e}")

    plt.plot(time, np.abs(gamma_qtm), label="Magnitude")
    plt.plot(time, gamma_qtm.real, label="Real Part", ls=":")
    plt.plot(time, gamma_qtm.imag, label="Imaginary Part", ls=":")
    plt.legend(loc="lower right", bbox_to_anchor=(0, 0.05, 1, 1))
    plt.grid()
    plt.xlabel("Time (s)")
    plt.ylabel("Width Function ($\\mathrm{\\AA}^2$)")

    parent_axes = plt.gca()
    axins: Axes = inset_axes(
        parent_axes,
        width="40%",
        height="40%",
        loc="upper left" if NO_EXTEND else "upper center",
    )
    axins.plot(time, np.abs(gamma_qtm), label="Magnitude")
    axins.plot(time, np.real(gamma_qtm), label="Real Part", linestyle=":")
    axins.plot(time, np.imag(gamma_qtm), label="Imaginary Part", linestyle="--")
    if NO_EXTEND:
        axins.set_xlim(0, 1e-13)
        axins.set_ylim(0, 0.1)
        axins.set_xticks(np.linspace(0, 1e-13, 5))
        axins.set_yticks(np.linspace(0, 0.1, 5))
        # adjust tick label position to avoid overlap with main plot
        for label in axins.get_xticklabels():
            label.set_transform(label.get_transform() + ScaledTranslation(6 / 72.0, 0, plt.gcf().dpi_scale_trans))
        # yticks showing on right side to avoid overlap with main plot
        axins.yaxis.tick_right()
    else:
        axins.set_xlim(-1e-13, 1e-13)
        axins.set_ylim(-0.05, 0.1)
        axins.set_xticks([-1e-13, 0, 1e-13])
        axins.set_yticks([-0.05, 0, 0.1])
    axins.grid()

    parent_axes.indicate_inset_zoom(axins, edgecolor="black")

    save_or_show_plot(OUTPUT)


def plot_sqw_ga(args: argparse.Namespace) -> None:
    GAMMA_KEYS: Final[list[str]] = args.gamma_keys
    STC_KEYS: Final[list[str]] = args.stc_keys
    OUTPUT: Final[str | None] = args.output
    GAMMA_FILE: Final[str] = args.gamma_file
    STC_FILE: Final[str | None] = args.stc_file
    TEMPERATURE: Final[float] = args.temperature
    SCALE: Final[str] = args.scale
    METHOD: Final[Literal["fft", "cdft"]] = args.method
    USE_FREQUENCY_AXIS: Final[bool] = args.frequency_axis
    RESTRICT_VIEW: Final[bool] = args.restrict_view

    try:
        Q_VALUES: Final[NDArray[np.double]] = parse_intervals(args.q_interval, default_step=1)
    except ValueError as e:
        sys.exit(f"Error parsing momentum transfer intervals: {e}")

    try:
        time, gamma_qtm, _ = read_gamma_data(GAMMA_FILE, GAMMA_KEYS, no_extend=False, include_cls=False)
    except Exception as e:
        sys.exit(f"Error reading width function data: {e}")

    try:
        freq_vdos, vdos = read_stc_data(STC_FILE, STC_KEYS) if STC_FILE else (None, None)
    except Exception as e:
        sys.exit(f"Error reading DOS data for STC model: {e}")

    print(f"Plotting S(q,w) GA results calculated by method '{METHOD.upper()}'...")

    ga_results = map(
        partial(
            sqw_ga_model,
            time=time,
            width_func=gamma_qtm,
            # width_func=gamma_qtm * np.hanning(gamma_qtm.size),
            correction=partial(assure_detailed_balance, temperature=TEMPERATURE),
        ),
        Q_VALUES,
    )

    for q, (omega, sqw_ga) in zip(Q_VALUES, ga_results, strict=True):
        x = omega if USE_FREQUENCY_AXIS else omega * HBAR
        (line,) = plt.plot(x, sqw_ga, label="CDFT" if STC_FILE else f"CDFT    {q = :.2f} 1/Ang")
        if freq_vdos is not None and vdos is not None:
            sqw_stc: Array1D = sqw_stc_model(q, omega, freq_vdos, vdos, temperature=TEMPERATURE)  # type: ignore
            plt.plot(x, sqw_stc, label=f"STC    {q = :.2f} 1/Ang", ls="--", color=line.get_color())
    if RESTRICT_VIEW:
        plt.xlim((-10 / HBAR, 2 / HBAR) if USE_FREQUENCY_AXIS else (-10, 2))
    if STC_FILE:
        handles, labels = plt.gca().get_legend_handles_labels()
        handles, labels = reorder_legend_by_row(handles, labels, ncol=2)
        plt.legend(handles, labels, loc="upper left", ncol=2)
    else:
        plt.legend(loc="upper left")
    plt.grid()
    plt.xlabel("Angular Frequency (Hz)" if USE_FREQUENCY_AXIS else "Energy (eV)")
    plt.ylabel("Scattering Function $S(q,\\omega)$ (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)")
    plt.yscale(SCALE)

    save_or_show_plot(OUTPUT)


def plot_sqw_ga2d(args: argparse.Namespace) -> None:
    GAMMA_KEYS: Final[list[str]] = args.gamma_keys
    OUTPUT: Final[str | None] = args.output
    FILE: Final[str] = args.file
    TEMPERATURE: Final[float] = args.temperature
    USE_FREQUENCY_AXIS: Final[bool] = args.frequency_axis

    if args.stc_keys:
        print(
            "Warning: STC line is obtained by derivative function of S(q,w), "
            "'--stc-keys' argument is ignored in 'ga2d' sub-command."
        )
    if args.scale != "log":
        print("Warning: Y axis scale is always 'log' in 'ga2d' sub-command.")

    try:
        Q_VALUES: Final[NDArray[np.double]] = parse_interval(args.q_slice, default_step=5)
        if Q_VALUES.size < 2:
            raise ValueError("At least two momentum transfer values are required to plot 2D color map!")
    except ValueError as e:
        sys.exit(f"Error parsing momentum transfer slice: {e}")
    if Q_VALUES.size < 10:
        print("Warning: Less than 10 momentum transfer values may result in poor resolution.")
    if Q_VALUES.size > 100:
        print("Warning: More than 100 momentum transfer values may result in memory issues or long computation time.")

    try:
        time, gamma_qtm, _ = read_gamma_data(FILE, GAMMA_KEYS, no_extend=False, include_cls=False)
    except Exception as e:
        sys.exit(f"Error reading width function data: {e}")

    print("Plotting S(q,w) GA 2D color map results...")

    omega = np.linspace(-10 / HBAR, 2 / HBAR, num=1000)
    sqw_ga_results = np.vstack(
        [
            np.interp(
                omega,
                *sqw_ga_model(q, time, gamma_qtm, correction=partial(assure_detailed_balance, temperature=TEMPERATURE)),
                left=0,
                right=0,
            )
            for q in Q_VALUES
        ]
    )
    omega_where_stc_max = np.array([-HBAR * q**2 / (2 * NEUTRON_MASS) for q in Q_VALUES])

    y_coords = omega if USE_FREQUENCY_AXIS else omega * HBAR
    plt.imshow(
        sqw_ga_results.T,
        extent=(Q_VALUES.min(), Q_VALUES.max(), y_coords.min(), y_coords.max()),
        origin="lower",
        aspect="auto",
        cmap="jet",
        norm=LogNorm(vmin=1e-27),
        interpolation="none",
    )
    plt.ylim(y_coords.min())
    plt.plot(
        Q_VALUES,
        omega_where_stc_max if USE_FREQUENCY_AXIS else omega_where_stc_max * HBAR,
        label="STC Model Max",
        color="tab:red",
        linestyle="--",
        clip_on=True,
    )
    plt.xlabel("Momentum Transfer $q$ ($\\mathrm{\\AA}^{-1}$)", fontsize=14)
    plt.ylabel("Angular Frequency (rad/s)" if USE_FREQUENCY_AXIS else "Energy (eV)", fontsize=14)
    plt.colorbar(label=r"Scattering Function $S(q,\omega)$ (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)")
    plt.legend(loc="upper right")
    plt.grid()

    save_or_show_plot(OUTPUT)


def plot_sqw_gaaqc(args: argparse.Namespace) -> None:
    GAMMA_KEYS: Final[list[str]] = args.gamma_keys
    MD_KEYS: Final[list[str]] = args.md_keys
    OUTPUT: Final[str | None] = args.output
    GAMMA_FILE: Final[str] = args.gamma_file
    MD_FILE: Final[str] = args.md_file
    WITH_GA_RESULTS: Final[bool] = args.with_ga_results
    TEMPERATURE: Final[float] = args.temperature
    SCALE: Final[str] = args.scale
    USE_FREQUENCY_AXIS: Final[bool] = args.frequency_axis

    try:
        INDICES: Final[Array1D[np.intp]] = parse_slices(args.indices)
    except ValueError as e:
        sys.exit(f"Error parsing momentum transfer indices: {e}")

    try:
        time, gamma_qtm, gamma_cls = read_gamma_data(GAMMA_FILE, GAMMA_KEYS, no_extend=False, include_cls=True)
        assert gamma_cls is not None, "Key param 'include_cls' for `read_gamma_data` must be True!"
    except Exception as e:
        sys.exit(f"Error reading width function data: {e}")

    try:
        q_md, omega_md, raw_sqw_md_stack = read_md_data(MD_FILE, keys=MD_KEYS, no_extend=False)
        sqw_md_stack = raw_sqw_md_stack * np.sqrt(2 * PI)
    except Exception as e:
        sys.exit(f"Error reading MD S(q,w) data for quantum correction: {e}")

    for q, sqw_md in zip(q_md[INDICES], sqw_md_stack[INDICES], strict=True):
        assure_db = partial(assure_detailed_balance, temperature=TEMPERATURE)
        omega_gaaqc, sqw_gaaqc = sqw_gaaqc_model(q, time, gamma_qtm, gamma_cls, omega_md, sqw_md, correction=assure_db)
        (line,) = plt.plot(
            omega_gaaqc if USE_FREQUENCY_AXIS else omega_gaaqc * HBAR,
            sqw_gaaqc,
            label="GAAQC" if WITH_GA_RESULTS else f"GAAQC    {q = :.2f} 1/Ang",
        )
        if WITH_GA_RESULTS:
            omega_ga, sqw_ga = sqw_ga_model(
                q,
                time,
                gamma_qtm,
                correction=assure_db,
            )
            plt.plot(
                omega_ga if USE_FREQUENCY_AXIS else omega_ga * HBAR,
                sqw_ga,
                label=f"GA    {q = :.2f} 1/Ang",
                ls="--" if INDICES.size > 1 else None,
                color=line.get_color() if INDICES.size > 1 else None,
            )
    plt.yscale(SCALE)
    if WITH_GA_RESULTS:
        plt.legend(*reorder_legend_by_row(*plt.gca().get_legend_handles_labels(), ncol=2), ncol=2, loc="upper left")
    else:
        plt.legend()
    plt.xlabel("Angular Frequency (Hz)" if USE_FREQUENCY_AXIS else "Energy (eV)")
    plt.ylabel("Scattering Function $S(q,\\omega)$ (b·eV⁻¹·Sr⁻¹·ℏ⁻¹)")

    save_or_show_plot(OUTPUT)


if __name__ == "__main__":
    main()
