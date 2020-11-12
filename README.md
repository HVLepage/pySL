# pySL
 
## Table of Contents
1. [About](#about)
2. [Installation](#installation)
   1. [Prerequisites](#prerequisites)
      1. [CUDA SDK](#cuda-sdk)
      2. [Python 3](#python3)
      3. [Python modules](#python-modules)
3. [Usage](#usage)
   1. [Input parameters](#input-parameters)
   2. [Parameter sweeps](#parameter-sweeps)
4. [Authors](#authors)
5. [Publications](#publications)
6. [Acknowledgments](#acknowledgments)


## About
The python Staggered Leapfrog (pySL) package solves the Time Dependent Schrödinger Equation (TDSE) using CUDA for GPU-accerelation.

## Installation
TDSE3DCUDA works on modern NVIDIA GPU systems running Windows or Linux. This code was written using the CUDA 10.2 SDK. The current version has been tested on CUDA 10.0 and later. Earlier versions of CUDA are not supported.

### Prerequisites
1. **CUDA SDK** from: https://developer.nvidia.com/cuda-downloads
2. **Python 3** from: https://www.python.org/
3. **Python modules** install using: `pip install -r requirements.txt`

#### CUDA SDK
For installation on Linux, follow the instructions in https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

To add binaries and libraries permanently add the following lines to `~/.bash_profile`
```bash
# Export NVIDIA CUDA bin and libraries to PATH
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\
${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Make sure `nvcc` is working correctly on your system. You might have to specify a custum path to your local C++ compiler in the file `get_ccbin.py`.

#### Python3

#### Python modules


## Usage

To run the solver, use the command:
```
python pySL.py [TARGET_GPU_ID]
```
The argument following the command indicates which GPU to use in a multi-GPU system. If no argument is given or if the argument is incorrect, the simulation will default to GPU0.

An output directory will be created for each different simulation name. Within this directory several results are recorded in plain text files.
* `eigs.txt` lists the lowest 5 eigenstates found using the built-in eigensolver.
* `wave.txt` outputs the complex wave function of the initial state as determined by the user.
* `waves\wave_iteraton_XXX.txt` are the complex wave functions at every iteration as defined by `PrintStep` in the input parameters.

### Input parameters
Simulation parameters should be written to the input file `parameters.py`.

### Parameter sweeps
Future work includes the ability to sweep over certain parameters to optimise output results.


## Authors

* **H. V. Lepage**
* **A. A. Lasek**
* **C. H. W. Barnes**

See the list of [contributors](https://github.com/HVLepage/pySL/contributors) who participated in this project.

## Publications
* [Entanglement generation via power-of-swap operations between dynamic electron-spin qubits (2020)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.052305)
* [Sound-driven single-electron transfer in a circuit of coupled quantum rails (2019)](https://www.nature.com/articles/s41467-019-12514-w)
* [Protocol for fermionic positive-operator-valued measures (2017)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.052305)


## Acknowledgments

* D. R. M. Arvidsson Shukur
* A. Andreev
* E. T. Owen
* J. Mosakowski

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 642688.
