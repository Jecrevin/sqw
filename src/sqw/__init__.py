"""Neutron scattering function calculation package.

This package provides models for calculating neutron scattering functions based
on various theoretical frameworks. It includes implementations for the Gaussian
Approximation (GA) model, the GA-assisted Quantum Correction (GAAQC) model, and
the Short-Time Collision (STC) model.
"""

from ._core import sqw_ga_model, sqw_gaaqc_model, sqw_stc_model

__all__ = [
    "sqw_stc_model",
    "sqw_ga_model",
    "sqw_gaaqc_model",
]
