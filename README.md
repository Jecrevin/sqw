# sqw: Mathematical Calculation of Scattering Functions from Molecular Dynamics

## Project Overview

This repository provides a rigorous mathematical framework for calculating neutron scattering functions from molecular dynamics (MD) data. It implements the full computational pipeline, starting from vibrational density of states (VDOS) and culminating in the double differential cross section, using a variety of theoretical models. The package includes:

- **Gaussian Approximation (GA) Model**
- **GA-assisted Quantum Correction (GAAQC) Model**
- **Short-Time Collision (STC) Model**

All core model functions are implemented in [`src/sqw/_core.py`](src/sqw/_core.py) and exposed via the Python package interface ([`src/sqw/__init__.py`](src/sqw/__init__.py)). The workflow is designed to be reproducible and modular, facilitating both script-based data generation and direct Python usage.

## Environment Setup

### Dev Container

A Dev Container configuration is provided for seamless development and reproducibility. It is recommended to use the Dev Container for consistent environment management.

### Dependency Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management. To set up the environment:

1. **Install uv** (if not already available):

   ```bash
   pip install uv
   ```

2. **Install dependencies from `pyproject.toml`:**

   ```bash
   uv pip install -e .
   ```

   This will install all required packages as specified in [`pyproject.toml`](pyproject.toml).

3. **Set up a Python virtual environment (optional but recommended):**

   ```bash
   python3 -m venv .venv --prompt sqw
   source .venv/bin/activate
   pip install -e .
   ```

   If you intend to run the scripts in the `scripts/` directory, also install the packages listed under the `[dependency-groups.script]` section in `pyproject.toml`:

   ```bash
   pip install h5py matplotlib
   ```

## Usage Guide

### Workflow for Plotting Scripts

The recommended workflow for generating and visualizing scattering function data is as follows:

1. **Generate Gamma Data from VDOS**

   Use [`scripts/vdos2gamma.py`](scripts/vdos2gamma.py) to convert VDOS data (e.g., `h2o_293k_vdos.h5`) into gamma datasets required for subsequent calculations.

   ```bash
   python scripts/vdos2gamma.py -i data/molecular_dynamics/h2o_293k_vdos.h5 -o data/molecular_dynamics/hydrogen_293k_gamma.h5 -e H -n 1 -t 293
   ```

2. **Run Calculation Scripts**

   Execute all scripts in [`scripts/calculation/`](scripts/calculation/) (except `helper.py`) to generate the data required for plotting. These scripts perform various model calculations and save results in CSV or HDF5 formats.

   Example:

   ```bash
   python scripts/calculation/sqw_ga_2d.py
   python scripts/calculation/sqw_gaaqc_2d.py
   # ...and so on for each script
   ```

3. **Generate Plots**

   Run all scripts in [`scripts/plot/`](scripts/plot/) (except `helper.py`) to produce visualizations from the generated data.

   Example:

   ```bash
   python scripts/plot/plot_sqw_ga_2d.py
   # ...other plotting scripts
   ```

### Using as a Python Package

After installing the repository (`pip install .`), you can import and use the provided models directly in your Python scripts:

```python
from sqw import sqw_ga_model, sqw_gaaqc_model, sqw_stc_model

# Example usage:
omega, sqw = sqw_ga_model(q, time, gamma_qtm)
```

Refer to [`src/sqw/__init__.py`](src/sqw/__init__.py) for the list of available functions.

## Citation & Contact

For academic use, please cite appropriately. For questions or contributions, contact the author:

- Haoran Jiang (jianghr@ihep.ac.cn)

---
