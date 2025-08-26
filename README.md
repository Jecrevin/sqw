# h2o-sqw-calc

Compute and visualize the **neutron scattering function** $S(Q, \omega)$
(Denoted as *sqw* in this project) of liquid water ($\mathrm{H_2O}$) from data of
time relevent part function $\Gamma(t)$ (Denoted as *gamma*) of SISF (Self-Intermediate
Scattering Function). The project implements a **CDFT (FFT-based)** approach and
compares it against an **STC (Short-Time-Collision)** model, with utilities to
explore classical data, the *gamma* function, and *quantum correction factors*.

## What’s inside

- **CDFT-based** *sqw* calculation with recursive self-convolution for large Q
- **STC model** for comparison and peak tracking
- **Plotting utilities:** plot scripts for
    - data from *gamma*
    - *sqw* results from direct FFT
    - comparison of *sqw* results from CDFT and STC model
    - stacking heat plot of different Q results of *sqw*
    - data of classical *sqw*
    - quantum correction factor data
- **HDF5 data** under `data/` and example figures under `fig/`

Repository layout (key parts):

- `src/h2o_sqw_calc/`: core math, I/O helpers, utilities
- `script/`: runnable plotting scripts (documented below)
- `data/`: example input files (gamma/STC/classical)
- `fig/`: output figures (created when using `-o`)

## Environment setup

Clone the repository:

```bash
git clone https://code.ihep.ac.cn/jianghr/h2o-sqw-calc.git
cd h2o-sqw-calc
```

> *NOTE:* If you use the Dev Container, you can skip cloning locally. In VS Code,
> use the command palette entry “Dev Containers: Clone Repository in …” to clone
> this repository directly into a container volume and open it there.

This project targets Python 3.13 and ships a Dev Container. Choose one of the
options below.

### Option A: Dev Container (recommended)

**Prerequisites**

- Install Docker and ensure it is running (Docker Desktop on macOS/Windows or
	Docker Engine on Linux).

**Open in a container**

- From VS Code, press Ctrl/Cmd+Shift+P and run:
	“Dev Containers: Clone Repository in …”, then paste the repo URL
	(no local clone needed). Or, if you already have the folder locally, open it
	and choose “Reopen in Container”.

The container comes with uv preinstalled and mirrors configured. Then install
deps:

```bash
uv sync
```

### Option B: uv (local)

1) Install uv: https://docs.astral.sh/uv/getting-started/installation/

2) Install deps for this project:

```bash
uv sync
```

### Option C: pip/venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

**Notes**

- Default data files live in `data/` and can be overridden via CLI flags (see
	each script).
- Without `-o/--output`, plots open interactively; with `-o`, they are saved to
	`fig/...` (overwrite prompt included).
- **Important:** The `data/` folder contents are **NOT** included in the remote
	repository. You must prepare/provide the required HDF5 files yourself (see
	Data files below) or adjust paths via CLI flags.

## Running

All entry points are regular Python scripts under `script/`.

If you have uv installed

```bash
uv run script/<script_name>.py [ARGS]
```

Without uv (activate your venv first)

```bash
source .venv/bin/activate
python3 script/<script_name>.py [ARGS]
```

### 1) Compare CDFT vs STC: `plot_sqw.py`

Plots S(Q, ω) from CDFT together with the STC model for one or more Q values.

- Positional Q can be a list: `10 20 30`, or a range: `5-50` with `--step`.
- Useful flags: `-e/--element {H,O}`, `-gamma/--gamma-file-path`,
  `-stc/--stc-file-path`, `-t/--temperature`, `--energy-unit`, `-o/--output`.

**Example**

```bash
uv run script/plot_sqw.py 5-40 --step 5 -e H --energy-unit \
	-o fig/sqw_comparison_plot.png
```

### 2) Stacked CDFT map + STC max line: `plot_sqw_stack.py`

Builds a Q–ω map from CDFT and overlays the STC peak trajectory.

- Positional `q-range` like `5-60`; spacing via `--step` (default 5.0).
- Same file/element/temperature flags as above; `--energy-unit`, `-o`
	supported.

**Example**

```bash
uv run script/plot_sqw_stack.py 5-60 --step 5 -e H --energy-unit \
	-o fig/sqw_stack_plot.png
```

### 3) Direct FFT S(Q, ω): `plot_fft.py`

Computes S(Q, ω) directly from *gamma(t)* with FFT for one or more Q; supports
linear/log y-scale.

- Q list or range; flags: `-e`, `-f/--file-path`, `--scale {linear,log}`, `-o`.

**Example**

```bash
uv run script/plot_fft.py 10 20 30 --scale log -e H -o fig/fft_plot.png
```

### 4) Gamma function: `plot_gamma.py`

Visualizes |*γ(t)*|, Re *γ(t)*, Im *γ(t)* for H or O.

- Flags: `-e`, `-f/--file-path`, `-o`.

**Example**

```bash
uv run script/plot_gamma.py -e H -f data/last_H.gamma -o fig/gamma_plot.png
```

### 5) Classical S(Q, ω): `plot_sqw_cls.py`

Plots classical S(Q, ω) curves from a merged `.sqw` file for selected index positions (corresponding Q shown in legend).

- Positional indices as list/ranges: e.g. `1-5 8 12-20 --step 2`.
- Flags: `-f/--file-path` (default `data/merged_h2o_293k.sqw`),
  `--element {H,O}`, `-o`.

Example

```bash
uv run script/plot_sqw_cls.py 1-5 8 --element H \
	-f data/merged_h2o_293k.sqw -o fig/sqw_cls_plot.png
```

### 6) Quantum correction factor R(Q, t): `plot_correction_factor.py`

Computes and plots `R(Q, t) = exp[-Q²/2 (γ_qtm - γ_cls)]` at a given Q using
quantum and classical gamma.

- Positional: `q` (float). Flags: `-e`, `-f/--file-path`, `-o`.

**Example**

```bash
uv run script/plot_correction_factor.py 40 -e H -o fig/correction_factor_plot.png
```

## Data files (defaults)

**The repository does not ship input data.** The paths below are the expected
defaults used by the scripts; you must create/provide these files yourself or
point the scripts to your own data via CLI flags.

- Gamma: `data/last_{element}.gamma` (datasets: time_vec, gamma_qtm_real/imag,
  optional gamma_cls)
- STC model: `data/last.sqw` (datasets: inc_omega_{element},
  inc_vdos_{element})
- Classical S(Q, ω): `data/merged_h2o_293k.sqw` (datasets: qVec_{element},
  inc_omega_{element}, inc_sqw_{element})

Adjust paths via the corresponding CLI flags when using your own files.

