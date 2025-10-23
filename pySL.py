import sys
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc
import humanize
import cupy as cp

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
from src.input.parameters import getparams
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
print("pySLQubits version 0.1.0")
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



params={
        "nqubits_S":1,
        "nqubits_E":2,
        "psi_S_spec":'x+',
        "psi_E_spec":'mm',
        "dtype":cp.complex64
        
        
        
        
        }


psi = wavefunction(params) #create wavefunction as per parameters

dt =  np.float32(params["dt"]) # Time step
TimeSteps = params["TimeSteps"] # Total solution iterations
PrintStep = params["PrintStep"]  # Used for saving the wave function




################################################################################################################################
# IterateGPU
################################################################################################################################


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