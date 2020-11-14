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
from src.input.parameters import getparams
from src.input.wavefunction import wavefunction
from src.input.convWavefunction import convWavefunction
from src.export.wavefunctionOut import wavefunctionOut
from src.export.exportTrace import exportTrace
from src.fft.domain import domain
from src.fft.kernelInds import kernelInds

from src.input.potential import potential
from src.export.makepath import makepath
from src.export.exportParameters import exportParameters

import reikna.cluda as cluda

# Built-in plotting
from src.plot.animate1D import animate1D
from src.plot.plot1Dtrace import plot1Dtrace

print("\n----------------------------------------------------------------")
print("pySL version 0.0.5")
print("----------------------------------------------------------------")

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
if os.name == 'nt': # Compatibility with Windows
    mod = SourceModule(source, options=['-ccbin', ccbin])
else:
    mod = SourceModule(source, options=['-ccbin', ccbin])

# Load CUDA function into python
rSpaceSL = mod.get_function("rSpaceSL")
kSpaceSL = mod.get_function("kSpaceSL")
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


V = potential(params) # Static potential
dom = domain(params) # Simulation domain boundary conditions
kerInds = kernelInds(params) #1D index to ND indices lookup for the kernel - avoid calculating this every time
psi = wavefunction(params, thr, V) #create wavefunction as per parameters


dt =  np.float32(params["dt"]) # Time step
TimeSteps = params["TimeSteps"] # Total solution iterations
PrintStep = params["PrintStep"]  # Used for saving the wave function

if (params["TDSESolver"] == "kSpace"):
    phi = convWavefunction(psi, thr, params)
    psiOut = wavefunctionOut(params, thr)
    psi2 = np.zeros((int(TimeSteps/PrintStep), params["NOut"]))
else:
    psi2 = np.zeros((int(TimeSteps/PrintStep), psi.N)) #output plot data


k = 0 # Keeps track of print step, will be made obsolete in future release

################################################################################################################################
# CUDA malloc
################################################################################################################################

# Get all values from parameters that need to be passed to GPU kernels
N = np.int32(params["N"])

Np = np.int32(params["Np"])
N1 = np.int32(params["N1"])
N1K = np.int32(params["N1K"])
dim = np.int32(params["dim"])
ASAW = np.float32(params["ASAW"])
kSAW = np.float32(params["kSAW"])
SAWoffset = np.float32(params["SAWoffset"])
wSAW = np.float32(params["wSAW"])
cSAW = np.float32(params["cSAW"])
RS = np.float32(params["RS"])
DRt0 = np.float32(params["DRt0"])
URt0 = np.float32(params["URt0"])
ATB2 = np.float32(params["ATB2"])
s2TB2 = np.float32(params["s2TB2"])
fbb = np.float32(params["fbb"])
value = np.float32(params["dt"] * params["hbar"])

Xa = np.float32(params["Xa"])
Ya = np.float32(params["Ya"])

Nm1 = np.float32(1 / N)

faa = cuda.mem_alloc_like(params["faa"])
cuda.memcpy_htod(faa, params["faa"])

Sizes = cuda.mem_alloc_like(params["Sizes"])
cuda.memcpy_htod(Sizes, params["Sizes"])

Mass = cuda.mem_alloc_like(params["Mass"])
cuda.memcpy_htod(Mass, params["Mass"])

dXs = cuda.mem_alloc_like(params["dXs"])
cuda.memcpy_htod(dXs, params["dXs"])

dKs = cuda.mem_alloc_like(params["dKs"])
cuda.memcpy_htod(dKs, params["dKs"])

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

    # Iterate the staggered leapfrog kernel over the wavefunction
    if (params["TDSESolver"] == "rSpace"):
        rSpaceSL(N, psi.re_gpu, psi.im_gpu, V.pot_gpu, dom.domain_gpu, t, np.int32(i%2), kerInds.inds_gpu, Np, N1, dim, ASAW, kSAW, SAWoffset, wSAW, cSAW, RS, DRt0, URt0, ATB2, s2TB2, fbb, faa, Sizes, Mass, dXs, Xa, Ya, block=(blockSize,1,1), grid=(numBlocks,1))
    elif (params["TDSESolver"] == "kSpace"):
        psi.wave_gpu.get(phi.wave_gpu)
        phi.fft(thr, 1)
        Axpy(phi.wave_gpu, V.pot_gpu,  N,  t, kerInds.inds_gpu, Nm1,  Np,  N1,  dim,  ASAW,  kSAW,  SAWoffset,  wSAW,  cSAW,  RS,  DRt0,  URt0,  ATB2,  s2TB2, Mass,  dXs,  Xa,  Ya, block=(blockSize,1,1), grid=(numBlocks,1))
        phi.fft(thr, 0)
        kSpaceSL(N, psi.wave_gpu, phi.wave_gpu, t, dt, Np, N1K,  dim,  value,  Sizes,  Mass,  dKs,  np.int32(i%2),  kerInds.inds_gpu, block=(blockSize,1,1), grid=(numBlocks,1))
        
        
    if (i % PrintStep) == 0:
      #Print to file / record to plot every printStep
      psi.dtoh()
      
      if (params["TDSESolver"] == "kSpace"):
          psiOut.exportWaveKtoReal(psi, thr, params)
          if (params["ExportTrace"]):
              exportTrace(psiOut.wave_gpu, params, i, True, kernelInds, mod)
          else:
              psiOut.exportFull(params, i)
          #psi2[k] = psiOut.wave.real**2 + psiOut.wave.imag**2
      else:
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
print("\nTotal Time Elapsed = ",t_print)


psi.dtoh()


################################################################################################################################
# Show Plots
################################################################################################################################

# Animate the 1D wave function as save as an mp4 file. Used in the Particle_in_a_box_1D example
# animate1D(params)

# Trace a 1D particle with the x axis as time and the y axis as the spatial dimension.
plot1Dtrace(params)

################################################################################################################################
# Trace Memory Management
################################################################################################################################

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
total = sum(stat.size for stat in top_stats)

print("\n----------------------------------------------------------------")
print("Total Memory Usage: %s"% humanize.naturalsize(total))
print("----------------------------------------------------------------")

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)