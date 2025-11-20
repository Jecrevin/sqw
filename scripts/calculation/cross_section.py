import json
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
from helper import assure_detailed_balance, read_gamma_data, read_md_data
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline

from sqw import sqw_ga_model, sqw_gaaqc_model
from sqw.consts import HBAR, NEUTRON_MASS
from sqw.typing import Array1D


def angle_to_momentum_transfer(angle: float, energy_in: float, energy_out: float) -> float:
    """Convert scattering angle and energies to momentum transfer Q.

    Parameters
    ----------
    angle : float
        Scattering angle in degrees.
    energy_in : float
        Incident neutron energy in eV.
    energy_out : float
        Scattered neutron energy in eV.

    Returns
    -------
    float
        Momentum transfer Q in inverse angstroms.

    """
    return np.sqrt(
        2
        * NEUTRON_MASS
        / HBAR**2
        * (energy_in + energy_out - 2 * np.sqrt(energy_in * energy_out) * np.cos(np.deg2rad(angle)))
    )


def cross_section_ga_model(
    angle: float, energy_in: float, energy_out: float, time: Array1D, width_func: Array1D, *, correction, window: bool
):
    """Calculate double differential cross section using Gaussian approximation model.

    Parameters
    ----------
    angle : float
        Scattering angle in degrees.
    energy_in : float
        Incident neutron energy in eV.
    energy_out : float
        Scattered neutron energy in eV.
    time : Array1D
        Time grid for the SQW model.
    width_func : Array1D
        Width function values corresponding to the time grid.
    correction : str
        Type of correction to apply in the SQW model.
    window : bool
        Whether to apply a window function in the SQW model.

    Returns
    -------
    float
        Double differential cross section value.

    """
    q = angle_to_momentum_transfer(angle, energy_in, energy_out)
    omega_grid, sqw_ga = sqw_ga_model(q, time, width_func, correction=correction, window=window)
    ddcs_ga = sqw_ga * np.sqrt(energy_out / energy_in) / HBAR

    target_omega = (energy_out - energy_in) / HBAR
    target_ddcs = np.interp(target_omega, omega_grid, ddcs_ga, left=0.0, right=0.0)

    if target_omega < omega_grid[0] or target_omega > omega_grid[-1]:
        print("Warning: target omega is out of the omega grid range. Returning zero Cross Section.")

    return target_ddcs


def cross_section_gaaqc_model(
    angle: float,
    energy_in: float,
    energy_out: float,
    q_vals: Array1D[np.floating],
    omega_grid: Array1D[np.floating],
    sqw: NDArray[np.floating],
):
    """Calculate double differential cross section using GAAQC model.

    Parameters
    ----------
    angle : float
        Scattering angle in degrees.
    energy_in : float
        Incident neutron energy in eV.
    energy_out : float
        Scattered neutron energy in eV.
    q_vals : Array1D[np.floating]
        Momentum transfer grid for the SQW model.
    omega_grid : Array1D[np.floating]
        Energy transfer grid for the SQW model.
    sqw : NDArray[np.floating]
        SQW values on the (q, omega) grid.

    Returns
    -------
    float
        Double differential cross section value.

    """
    q = angle_to_momentum_transfer(angle, energy_in, energy_out)
    omega = (energy_out - energy_in) / HBAR
    target_sqw = RectBivariateSpline(q_vals, omega_grid, sqw).ev(q, omega)

    return target_sqw * np.sqrt(energy_out / energy_in) / HBAR


def get_exp_data_at_fixed_condition(file: PathLike, cond: dict[Literal["angle", "energy_in", "energy_out"], float]):
    """Retrieve experimental data from a JSON file at fixed conditions.

    Parameters
    ----------
    file : PathLike
        Path to the JSON file containing experimental datasets.
    cond : dict[Literal["angle", "energy_in", "energy_out"], float]
        Conditions to match in the datasets.

    Returns
    -------
    NDArray[np.floating]
        Experimental data array matching the specified conditions.

    """

    def match_condition(dataset) -> bool:
        headers = dataset["headers"]
        real = {
            "energy_in": float(headers[2]["minval"]),
            "energy_out": float(headers[3]["minval"]),
            "angle": float(headers[4]["minval"]),
        }
        return all(np.isclose(real[key], value) for key, value in cond.items())

    with open(file) as f:
        json_data = json.load(f)
    data = list(map(lambda ds: np.asarray(ds["data"]), filter(match_condition, json_data["datasets"])))

    if len(data) == 0:
        raise ValueError(f"No dataset found matching conditions: {cond}")
    if len(data) > 1:
        print(f"Warning: multiple datasets found matching conditions: {cond}. Returning the first one.")

    return data[0]


def main() -> None:
    data_dir = Path(__file__).parents[2] / "data"
    input_dir = data_dir / "molecular_dynamics"
    out_dir = data_dir / "results" / "cross_section"

    out_dir.mkdir(exist_ok=True)

    time, gamma_qtm, gamma_cls = read_gamma_data(
        input_dir / "hydrogen_293k_gamma.h5",
        keys=["time_vec", "gamma_qtm_real", "gamma_qtm_imag", "gamma_cls"],
        include_cls=True,
    )

    assert gamma_cls is not None, "`gamma_cls` should not be None when `include_cls` is True."

    q_vals, omega_md, sqw_md_stack = read_md_data(
        input_dir / "h2o_293k_all.h5", keys=["qVec_H", "inc_omega_H", "inc_sqw_H"]
    )

    sqw_gaaqc_results = [
        sqw_gaaqc_model(q, time, gamma_qtm, gamma_cls, omega_md, sqw_md, window=False)  # type: ignore
        for q, sqw_md in zip(q_vals, sqw_md_stack, strict=True)
    ]
    omega_grid = np.linspace(-10 / HBAR, 2 / HBAR, 3000)
    sqw_gaaqc_stack = np.vstack(
        [np.interp(omega_grid, omega, sqw, left=0.0, right=0.0) for omega, sqw in sqw_gaaqc_results]
    )

    conditions = [{"angle": 25.0, "energy_in": 0.154}, {"angle": 40.0, "energy_in": 0.154}]

    for cond in conditions:
        exp_cross_section_data = get_exp_data_at_fixed_condition(
            data_dir / "cross_section_experiment/Esch/data.json",
            cond,  # type: ignore
        )
        energy_out = np.linspace(exp_cross_section_data[:, 3].min(), exp_cross_section_data[:, 3].max(), 200)
        cross_section_ga_result = np.array(
            [
                cross_section_ga_model(
                    cond["angle"],
                    cond["energy_in"],
                    e_out,
                    time,
                    gamma_qtm,
                    correction=partial(assure_detailed_balance, temperature=293.0),
                    window=True,
                )
                for e_out in energy_out
            ]
        )
        cross_section_gaaqc_result = np.array(
            [
                cross_section_gaaqc_model(cond["angle"], cond["energy_in"], e_out, q_vals, omega_grid, sqw_gaaqc_stack)
                for e_out in energy_out
            ]
        )
        out_file = out_dir / f"cross_section_angle_{int(cond['angle'])}_ein_{cond['energy_in']:.3f}.csv"
        np.savetxt(
            out_file,
            np.column_stack([energy_out, cross_section_ga_result, cross_section_gaaqc_result]),
            delimiter=",",
            header=f"Cross Section Results at angle {cond['angle']} deg and E_in {cond['energy_in']:.3f} eV\n"
            "energy_out, cross_section_ga, cross_section_gaaqc",
        )
        np.savetxt(
            out_file.with_suffix(".exp.csv"),
            np.column_stack((exp_cross_section_data[:, 3], exp_cross_section_data[:, 0])),
            delimiter=",",
            header=f"Cross Section Experimental Data at angle {cond['angle']} deg and E_in {cond['energy_in']:.3f} eV\n"
            "energy_out, cross_section_exp",
        )


if __name__ == "__main__":
    main()
