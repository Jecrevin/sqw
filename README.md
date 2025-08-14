# ⚛️ h2o-sqw-calc

This repository containing codes to calculate the scattering function $S(q, \omega)$
(Denoting as SQW) of light water $\mathrm{H_2O}$ from gamma data.

## 🛠️ Environment Setup

You can get the whole repository by cloning using git:
```
git clone https://code.ihep.ac.cn/jianghr/h2o-sqw-calc.git
cd h2o-sqw-calc
```

This repository is a package based project, dependencies along with this project
itself must be installed. You can install through your favourate tools, following
the corresponding steps:

### Dev Container (Recommended)

This repository contains a pre-configured development container. If you are using
VS Code with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
or GitHub Codespaces, you can get a ready-to-use environment.

When you open this folder in VS Code, it will suggest you to "Reopen in Container".
By doing so, a Docker container will be built with all the necessary tools.
This dev container has `uv` pre-installed and network mirrors configured for users
in China. You can skip the `uv` installation step and directly use it to
install dependencies and run programs.

### [uv](https://docs.astral.sh/uv/#uv)

> [!NOTE]
> If you are using the Dev Container, `uv` is already installed and you can skip step 1.

1. Install uv by folloing instructions at https://docs.astral.sh/uv/getting-started/installation/
2. installing all dependencies by:
```
uv sync
```

### pip

1. Install pip by folling instructions at https://pip.pypa.io/en/stable/installation/.
2. Create virtual environment in the project to avoid polluting your global
environment:
```
python3 -m venv .venv
source .venv/bin/activate
```
3. install all needed package by one-shot command:
```
pip install .
```

## 💻 Running

This project now has 3 scripts can run dirrectly:

- `sqw [Q]`: show the comparison plot of Short-Time-Collision (STC) model and
Convolutional Discrete Fourier Transform (CDFT, which using FFT to calculate SQW
for small `Q`, and convolve from small `Q` result for large `Q`).
- `plot_fft Q1 Q2 ...`: plot the SQW using direct FFT for following `Q` value.
- `plot_gamma`: plot the gamma data for Hydrogen element (H).

After setting up the environment, you can run these scripts. For example, if you want to see
the comarison for CDFT and STC model at `Q = 40.0`, you can run:
```
sqw 40.0
```
If you are using `uv` (either installed manually or in the dev container), you can run:
```
uv run sqw 40.0
```
