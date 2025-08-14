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

### [uv](https://docs.astral.sh/uv/#uv)

1. Install uv by folloing instructions at https://docs.astral.sh/uv/getting-started/installation/
2. installing all dependencies by:
```
uv sync
```

## 💻 Running

This project now has 3 scripts can run dirrectly:

- `sqw [Q]`: show the comparison plot of Short-Time-Collision (STC) model and
Convolutional Discrete Fourier Transform (CDFT, which using FFT to calculate SQW
for small `Q`, and convolve from small `Q` result for large `Q`).
- `plot_fft Q1 Q2 ...`: plot the SQW using direct FFT for following `Q` value.
- `plot_gamma`: plot the gamma data for Hydrogen element (H).

If you're using pip and alread followed the instructions above, now you can simply
run these script dirrectly through your console. For example, if you want to see
the comarison for CDFT and STC model at `Q = 40.0`, you can run:
```
sqw 40.0
```
For uv users, for the same example above you can run:
```
uv run sqw 40.0
```
