# h2o-sqw-calc

Calculate and visualize the **neutron scattering function** $S(Q, \omega)$ of liquid water ($\mathrm{H_2O}$) based on molecular dynamics simulation data. This project implements multiple theoretical models to compute the neutron scattering function from the time-dependent partial function $\Gamma(t)$ of the self-intermediate scattering function (SISF).

## Theoretical Models

The project implements three neutron scattering function calculation models:

- **GA (Gaussian Approximation)**: Gaussian approximation model based on CDFT (Continuous Fourier Transform) method, supports recursive self-convolution for large Q values
- **GAAQC (GA-assisted Quantum Correction)**: GA-assisted quantum correction model that combines quantum and classical width functions
- **STC (Short-Time Collision)**: Short-time collision approximation model for comparison and peak tracking

## Project Structure

```
h2o-sqw-calc/
├── src/sqw/              # Core Python package
│   ├── _core.py          # Implementation of three models
│   ├── _math.py          # Mathematical utility functions
│   ├── consts.py         # Physical constants definitions
│   ├── typing.py         # Type definitions
│   └── __init__.py       # Package initialization
├── scripts/
│   ├── calculation/      # Calculation scripts
│   │   ├── sqw_ga_2d.py              # GA model 2D calculation
│   │   ├── sqw_gaaqc_2d.py           # GAAQC model 2D calculation
│   │   ├── cross_section.py          # Scattering cross-section calculation
│   │   ├── cdft_stc_comparison.py    # CDFT vs STC comparison
│   │   ├── sisf_qc_factor_q1.63.py   # Quantum correction factor calculation
│   │   ├── sqw_ga_fft_various_q.py   # Multi-Q FFT calculation
│   │   └── fft_stc_comarison_q40.py  # Q=40 FFT/STC comparison
│   └── plot/             # Visualization scripts
│       ├── sqw_ga_2d.py              # S(Q,ω) 2D heatmap
│       ├── sqw_gaaqc_2d.py           # GAAQC results visualization
│       ├── cross_section.py          # Scattering cross-section comparison plot
│       ├── cdft_stc_comarison_various_q.py  # Multi-Q comparison
│       ├── width_function.py         # Width function visualization
│       ├── sisf_qc_factor_1.63.py    # Quantum correction factor plot
│       ├── sqw_ga_fft_various_q.py   # FFT results visualization
│       └── sqw_ga_q1.py               # Single Q value result plot
├── data/                 # Data directory
│   ├── molecular_dynamics/           # MD simulation data
│   │   ├── h2o_293k_all.h5          # 293K water complete data
│   │   ├── h2o_293k_stc.h5          # STC model data
│   │   └── hydrogen_293k_gamma.h5    # Hydrogen width function data
│   ├── cross_section_experiment/     # Experimental scattering cross-section data
│   │   ├── Bischoff/
│   │   ├── Esch/
│   │   ├── Harling/
│   │   ├── Ramic/
│   │   ├── Wendorff/
│   │   └── total/
│   └── results/          # Calculation results output
├── figs/                 # Figure output directory
├── tests/                # Unit tests
│   └── test_math.py
└── pyproject.toml        # Project configuration
```

## Core Features

### Model Implementation (`src/sqw/_core.py`)

#### 1. GA Model (`sqw_ga_model`)
Gaussian approximation model that calculates the frequency-domain scattering function from the time-domain width function $\Gamma(t)$:
- For $Q \leq 5$: Direct calculation of SISF and Fourier transform
- For $Q > 5$: Recursive self-convolution method for improved computational efficiency
- Supports window functions and detailed balance correction

#### 2. GAAQC Model (`sqw_gaaqc_model`)
Quantum correction model combining quantum and classical width functions:
- Uses quantum width function to calculate high-frequency components
- Uses classical MD data to correct low-frequency components
- Combines both parts through convolution

#### 3. STC Model (`sqw_stc_model`)
Short-time collision approximation model:
- Based on density of states (DOS) data
- Calculates effective temperature
- Provides analytical form of scattering function

### Mathematical Tools (`src/sqw/_math.py`)

- **Fourier Transform**: `continuous_fourier_transform` - Continuous Fourier transform implementation
- **Convolution Operations**: 
  - `linear_convolve` - Linear convolution of two functions
  - `self_linear_convolve` - Function self-convolution
- **Array Validation**:
  - `is_array_linspace` - Check if array is uniformly spaced
  - `is_all_array_1d` - Check if all arrays are one-dimensional
- **Data Processing**: `trim_function` - Trim low-value regions of functions

### Physical Constants (`src/sqw/consts.py`)

- `HBAR`: Reduced Planck constant (eV·s)
- `KB`: Boltzmann constant (eV/K)
- `NEUTRON_MASS`: Neutron mass (eV/(Å/s)²)
- `PI`: Pi

## Environment Setup

This project requires Python 3.13+ and provides Dev Container support.

### Clone Repository

```bash
git clone https://code.ihep.ac.cn/jianghr/h2o-sqw-calc.git
cd h2o-sqw-calc
```

### Method A: Dev Container (Recommended)

**Prerequisites**
- Install Docker and ensure it's running (Windows/macOS use Docker Desktop, Linux uses Docker Engine)
- Install VS Code and the Dev Containers extension

**Open in Container**
1. Press Ctrl/Cmd+Shift+P in VS Code
2. Run "Dev Containers: Clone Repository in Container Volume"
3. Paste the repository URL

The container comes with uv package manager pre-installed and configured with mirror sources. Install dependencies:

```bash
uv sync
```

### Method B: Using uv (Local)

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/

2. Install project dependencies:

```bash
uv sync
```

### Method C: Using pip/venv

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[script]"
```

## Usage

### Running Calculation Scripts

Using uv:
```bash
uv run scripts/calculation/sqw_ga_2d.py
```

Or after activating the virtual environment:
```bash
source .venv/bin/activate
python scripts/calculation/sqw_ga_2d.py
```

### Running Plot Scripts

```bash
uv run scripts/plot/sqw_ga_2d.py
```

### Main Scripts Description

#### Calculation Scripts

1. **`sqw_ga_2d.py`**: Calculate 2D S(Q,ω) array for Q ∈ [1, 100]
   - Uses GA model
   - Applies 293K detailed balance correction
   - Outputs HDF5 file to `data/results/sqw_ga_2d.h5`

2. **`cross_section.py`**: Calculate double differential scattering cross-section
   - Supports GA and GAAQC models
   - Comparison with experimental data
   - Angle and energy conversion to momentum transfer

3. **`cdft_stc_comparison.py`**: CDFT and STC model comparison
   - Parallel computation for multiple Q values
   - Saves comparison data for plotting

#### Plot Scripts

1. **`sqw_ga_2d.py`**: Plot S(Q,ω) 2D heatmap
   - Color heatmap showing intensity distribution
   - Overlay STC model peak position curves

2. **`cross_section.py`**: Scattering cross-section vs experimental comparison
   - Theoretical calculation vs experimental data
   - Multiple scattering angles
   - Different incident energies

3. **`width_function.py`**: Width function visualization
   - Display |Γ(t)|, Re Γ(t), Im Γ(t)
   - Quantum and classical width function comparison

## Data Files Description

**Note**: The repository does not include large data files. You need to prepare the following data yourself:

### Input Data (`data/molecular_dynamics/`)

- **`hydrogen_293k_gamma.h5`**: Hydrogen width function data
  - Datasets: `time_vec`, `gamma_qtm_real`, `gamma_qtm_imag`, `gamma_cls`
  
- **`h2o_293k_all.h5`**: Complete 293K water MD data
  
- **`h2o_293k_stc.h5`**: Data required for STC model
  - Datasets: `inc_omega_H`, `inc_vdos_H` (hydrogen), 
            `inc_omega_O`, `inc_vdos_O` (oxygen)

### Experimental Data (`data/cross_section_experiment/`)

Contains experimental scattering cross-section data from multiple researchers:
- Bischoff, Esch, Harling, Ramic, Wendorff, etc.

Data format is JSON or text files.

### Output Results (`data/results/`)

Calculation scripts generate the following files:
- `sqw_ga_2d.h5`: GA model 2D results
- `sqw_gaaqc_2d.h5`: GAAQC model 2D results
- `cross_section_*.csv`: Scattering cross-section calculation results
- `sqw_ga_cdft_stc_comparison/`: CDFT and STC comparison data

## Testing

Run unit tests:

```bash
uv run pytest tests/
```

Or with coverage:

```bash
uv run coverage run -m pytest tests/
uv run coverage report
```

## Dependencies

### Core Dependencies
- `numpy >= 2.3.2`: Numerical computation
- `scipy >= 1.16.1`: Scientific computing and signal processing

### Script Dependencies
- `matplotlib >= 3.10.6`: Data visualization
- `h5py >= 3.14.0`: HDF5 file reading and writing

### Development Dependencies
- `ruff >= 0.12.9`: Code formatting and linting
- `coverage >= 7.10.4`: Test coverage
- `sphinx >= 8.2.3`: Documentation generation

## References

If you use this code, please cite the relevant paper (to be published).
