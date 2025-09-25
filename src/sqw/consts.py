from typing import Final

import scipy.constants as consts

HBAR: Final[float] = consts.value("reduced Planck constant in eV s")  # unit: eV·s
KB: Final[float] = consts.value("Boltzmann constant in eV/K")  # unit: eV/K
NEUTRON_MASS: Final[float] = (
    consts.value(key="neutron mass energy equivalent in MeV") * consts.mega / (consts.c / consts.angstrom) ** 2
)  # unit: eV/(Å/s)^2
