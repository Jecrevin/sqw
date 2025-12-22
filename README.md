# SQW

sqw is a simple python package that can used to calculate the neutron
scattering fucntion $S(Q, \omega)$ of light water $\mathrm{H_2O}$ from
Vibrating Density of States (VDOS) and classical scattering function. The
following theoretical models are implemented:

- STC model: Short Time Collision approximation model, this makes the $S(Q,
  \omega)$ have the simmilar expression for free gas.
- GA model: Gaussian Approximation model, this assumes the $S(Q, \omega)$ is a
  gaussian function of width function $\Gamma(t)$.
- GAAQC model: GA-assisted Quantum Correction model, this adds the quantum
  correction on GA model.

All the detail of these models can be found in our [related paper]().

## installation

1. clone this repository to your local machine and change the directory:
```
git clone https://github.com/Jecrevin/sqw.git
cd sqw
```
2. (Optional) use python's venv module to add an virtual env:
```
python -m venv .venv --prompt sqw
```
3. compile the filon library used for calculation of width function from VDOS:
```
cmake -B src/filon/build -S src/filon
cmake --build src/filon/build
mkdir src/sqw/_filon/
cp src/filon/build/libfilon.so src/sqw/_filon/
```
3. use pip to install this package:
```
python -m pip install .
```
4. verifing the installation:
```
python -c "import sqw"
```

the installation is successed if there's no `ModuleNotFoundError` appears.

## usage

this package provides a CLI command `vdos2gamma` for converting VDOS to width
function, and three python functions for $S(Q, \omega)$ calculation of
different models.

the basic usage of `vdos2gamma` is:
```
vdos2gamma -t <temperature> -o path/to/output/file path/in/input/file <frequency_key> <vdos_key>
```
more info can be found using `vdos2gamma --help`.

for implementation of the models metioned above, we provide 3 python functions
`sqw_stc_model`, `sqw_ga_model` and `sqw_gaaqc_model`, for how to use these
python functions please refer to the docstring in [_core.py](src/sqw/_core.py).

## example data and plot

we provide an example VDOS and classical $S(Q, \omega)$ data in this repository
using LFS support, you can dowload these data and using all the scripts under
`scripts/calculation` to get the result and use `scripts/plot` to plot data.

to use the example data to plot the cross section calculated by GAAQC model, first
download the data using git-lfs:
```
git lfs fetch --all
```
and run the correspoding scripts:
```
python scripts/calculation/cross_section.py
python scripts/plot/cross_section.py
```
the plot will be saved under folder `figs`.