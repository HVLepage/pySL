import sys
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc
import humanize

# Set target CUDA device. Must be an integer. Currently no error checking.
if (len(sys.argv) > 1):
  print("\nTarget CUDA device: ",str(sys.argv[1]))
  os.environ['CUDA_DEVICE'] = str(sys.argv[1])

# Import pyCUDA modules
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

# Import project modules
import src.get_gpu_info as gpuinfo
from src.get_ccbin import get_ccbin
from src.input.parametersQubits import getparams
from src.input.wavefunction import wavefunction
from src.fft.convWavefunction import convWavefunction
from src.export.wavefunctionOut import wavefunctionOut
from src.export.exportTrace import exportTrace
from src.input.domain import domain
from src.index.kernelInds import kernelInds

from src.input.potential import potential
from src.export.makepath import makepath
from src.export.exportParameters import exportParameters

import reikna.cluda as cluda

# Built-in plotting
from src.plot.animate1D import animate1D
from src.plot.plot1Dtrace import plot1Dtrace


print("\n----------------------------------------------------------------")
print("pySL version 0.1.2")
print("----------------------------------------------------------------")



t0_full = time.time() # Keep track of total simulation time

################################################################################################################################
# Application profiling & GPU hardware configuration
################################################################################################################################

tracemalloc.start() # Memory allocation profile

gpuinfo.print_gpu_info()

################################################################################################################################
# CUDA Module
################################################################################################################################

# Specify C++ compiler
ccbin = get_ccbin()
api = cluda.cuda_api()
thr = api.Thread.create()

# Read CUDA kernel
with open('./src/kernel.cu', 'r') as file:
    source = file.read()

# Compile kernel using nvcc
mod = SourceModule(source, options=['-ccbin', ccbin])

# Load CUDA function into python
qubitsSL = mod.get_function("qubitsSL")
Axpy = mod.get_function("Axpy")

# Block size determines how the GPU parallelizes the problem. This shouldn't need to change.
blockSize = 256 

################################################################################################################################
# Staggered Leapfrog Setup
################################################################################################################################

# Import parameters
params = getparams()

# Create output directory
makepath(params)

# Save input parameters to a text file
exportParameters(params)


print("\n----------------------------------------------------------------")
print("Allocating potential")
print("----------------------------------------------------------------")
t0V = time.time()
V = potential(params, 0.0, mod) #  Potential
time_V = time.time() - t0V
t0V = time.time()
print("\nPotential allocated in %.4f seconds"%time_V)
dom = domain(params) # Simulation domain boundary conditions
time_V = time.time() - t0V
t0V = time.time()
print("Simulation domain defined in %.4f seconds"%time_V)


psi = wavefunction(params,mod, thr, V) #create wavefunction as per parameters

dt =  np.float32(params["dt"]) # Time step
TimeSteps = params["TimeSteps"] # Total solution iterations
PrintStep = params["PrintStep"]  # Used for saving the wave function


if (params["TDSESolver"] == "qubits"):
    psi2 = np.zeros((int(TimeSteps/PrintStep), psi.N))
k = 0 # Keeps track of print step, will be made obsolete in future release

################################################################################################################################
# CUDA malloc
################################################################################################################################

# Get all values from parameters that need to be passed to GPU kernels
N = np.int32(params["N"])

Np = np.int32(params["Np"])

fbb = np.float32(params["fbb"])
value = np.float32(params["dt"] * params["hbar"])

faa = cuda.mem_alloc_like(params["faa"])
cuda.memcpy_htod(faa, params["faa"])


################################################################################################################################
# IterateGPU
################################################################################################################################

print("\n----------------------------------------------------------------")
print("GPU-accelerated simulation using:", params["TDSESolver"])
print("Simulation Name:", params["SimName"])
print("----------------------------------------------------------------")

# Number of blocks needed to go over all N lattice sites
numBlocks = math.ceil(N / blockSize) 

k=0
t0 = time.time()
t1 = t0
for i in range(0, TimeSteps):
    t = np.float32(i*dt)

    if t in params["VUpdates"]:
        V = potential(params, 0.0, mod)

    # Iterate the staggered leapfrog kernel over the wavefunction

    elif (params["TDSESolver"] == "qubits"):
        qubitsSL(N, psi.re_gpu, psi.im_gpu, V.pot_gpu, dom.domain_gpu, t, np.int32(i%2),   fbb,  block=(blockSize,1,1), grid=(numBlocks,1))
    
    if (i % PrintStep) == 0:
      #Print to file / record to plot every printStep
        psi.dtoh()
      

        if (params["ExportTrace"]):
             exportTrace(psi.wave_gpu, params, i, False, kernelInds, mod)
        else:
            psi.exportFull(params,i)
        #psi2[k] = psi.wave.real**2 + psi.wave.imag**2
        
        k = k + 1
        t2 = time.time()
        t_print = t2-t1
        t1 = t2
        norm = psi.norm()
        n_zero = math.log10((float)(params["TimeSteps"]) + 1) + 1
        print("Iteration:", str(i).zfill(int(n_zero)), "    Time Elapsed: %.2f"%t_print, "    NORM: %.8f"%norm)

t2 = time.time()
t_print = t2-t0
print("\nTDSE time elapsed = ",t_print)


psi.dtoh()


################################################################################################################################
# Show Plots
################################################################################################################################

# Animate the 1D wave function as save as an mp4 file. Used in the Particle_in_a_box_1D example
# animate1D(params)

# Trace a 1D particle with the x axis as time and the y axis as the spatial dimension.
# plot1Dtrace(params)

################################################################################################################################
# Trace Memory Management
################################################################################################################################
t_print = time.time()-t0_full
print("\nTotal time elapsed = ",t_print)


snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
total = sum(stat.size for stat in top_stats)

print("\n----------------------------------------------------------------")
print("Total Memory Usage: %s"% humanize.naturalsize(total))
print("----------------------------------------------------------------")

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)