# pySL
 
## Table of Contents
- [pySL](#pysl)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
      - [CUDA SDK](#cuda-sdk)
      - [Python3](#python3)
      - [Python modules](#python-modules)
      - [(Optional) ffmpeg](#optional-ffmpeg)
  - [Usage](#usage)
    - [Input parameters](#input-parameters)
    - [Parameter sweeps](#parameter-sweeps)
  - [Authors](#authors)
  - [Publications](#publications)
  - [Acknowledgments](#acknowledgments)


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

:warning: **Linux driver issues**

Linux drivers installed from the system package manager will sometimes cause issues with CUDA installations. It is recommended to install both the driver and the CUDA SDK using the downloadable run file.

To uninstall all versions of CUDA as well as the drivers on Ubuntu, use the command:
```bash
sudo apt autoremove --purge cuda
sudo apt autoremove --purge *nvidia*
```

A system restart is required.

Next, install the Latest Production Branch Version driver from https://www.nvidia.com/en-gb/drivers/unix/ for example:
```bash
wget https://uk.download.nvidia.com/XFree86/Linux-x86_64/460.39/NVIDIA-Linux-x86_64-460.39.run
sudo sh NVIDIA-Linux-x86_64-460.39.run
```
Note that the exact version will depend on the host computer as well as NVIDIA updates.

A system restart is required.

To check that the drivers have been installed properly, run the command:
```bash
nvidia-smi
```
![nvidia-smi](https://www.dropbox.com/s/8x2pgdg3rt1q8gg/nvidia-smi.png?raw=1)


Next, download the latest version of the CUDA SDK from https://developer.nvidia.com/cuda-downloads make sure you choose the `runfile (local)`. For example:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda_11.2.1_460.32.03_linux.run
sudo sh cuda_11.2.1_460.32.03_linux.run
```
Follow the onscreen instructions and untick the NVIDIA driver as it should already be installed.

![cuda-install](https://www.dropbox.com/s/nu58d9wa4dl41z2/cuda-install.png?raw=1)

Finally, add binaries and libraries to PATH
```bash
# Export NVIDIA CUDA bin and libraries to PATH
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\
${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Make sure `nvcc` is working correctly on your system by running the command
```bash
nvcc --version
```
![nvcc --version](https://www.dropbox.com/s/727x405eo47ffi6/nvcc-version.png?raw=1)

You might have to specify a custum path to your local C++ compiler in the file `get_ccbin.py`.

#### Python3

#### Python modules

#### (Optional) ffmpeg

`ffmpeg` is used to generate plot animations. This is not a required feature since plotting is not necessary and other software can be used for analysis purposes. Installation will vary depending on the host operating system.


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

Other input files such as an initial wave function or static potentials should be written as text files and placed in the `/input/` directory.

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
