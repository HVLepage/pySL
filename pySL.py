import sys
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc

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
from src.input.parameters import getparams
from src.input.wavefunction import wavefunction
from src.input.potential import potential
from src.export.makepath import makepath


print("\n----------------------------------------------------------------")
print("pySL version 0.0.4")
print("----------------------------------------------------------------")

################################################################################################################################
# Application profiling
################################################################################################################################

tracemalloc.start() # Memory allocation profile

################################################################################################################################
# GPU hardware configuration
################################################################################################################################

gpuinfo.print_gpu_info()

################################################################################################################################
# CUDA Module
################################################################################################################################

with open('./src/kernel.cu', 'r') as file:
    source = file.read()

if os.name == 'nt': # Compatibility with Windows
    mod = SourceModule(source, options=['-ccbin', 'cl.exe'])
else:
    mod = SourceModule(source, options=['-ccbin', 'cuda-g++'])

################################################################################################################################
# Staggered Leapfrog Setup
################################################################################################################################

# Import parameters
params = getparams()

# Create output directory
makepath(params)


V = potential(params) # Static potential
psi = wavefunction(params,V)

dt = params["dt"] # Time step
TimeSteps = params["TimeSteps"] # Total solution iterations
PrintStep = params["PrintStep"]  # Used for saving the wave function

psi2 = np.zeros((int(TimeSteps/PrintStep), psi.N)) #output plot data

dx = params["dx"]
aa = dt/(dx**2) # define constants for staggered leapfrog
bb = 2*dt

k = 0

################################################################################################################################
# CUDA malloc
################################################################################################################################


aa = np.float32(aa) #convert to singles for the GPU kernel
bb = np.float32(bb)
N = np.int32(params["N"])




################################################################################################################################
# IterateGPU
################################################################################################################################

print("\n----------------------------------------------------------------")
print("GPU-accelerated Simulation")
print("----------------------------------------------------------------")

func_re = mod.get_function("gqsl_x_re")
func_im = mod.get_function("gqsl_x_im")

blockSize = 256 
numBlocks = math.ceil(N / blockSize) #how many blocks you need to go over all the points -N

k=0
t0 = time.time()
t1 = t0
for i in range(0, TimeSteps):

  # func_re(psi_gpu,V_gpu,bb,aa,N, block=(blockSize,1,1), grid=(numBlocks,1))
  # func_im(psi_gpu,V_gpu,bb,aa,N, block=(blockSize,1,1), grid=(numBlocks,1))

    #iterate the leapfrog kernel over the wavefunction
  func_re(psi.re_gpu,psi.im_gpu,V.pot_gpu,bb,aa,N, block=(blockSize,1,1), grid=(numBlocks,1)) 
  func_im(psi.re_gpu,psi.im_gpu,V.pot_gpu,bb,aa,N, block=(blockSize,1,1), grid=(numBlocks,1))

  if (i % PrintStep) == 0:
      #Print to file / record to plot every printStep
    psi.dtoh()
    psi2[k] = psi.wave.real**2 + psi.wave.imag**2
    k = k + 1
    t2 = time.time()
    t_print = t2-t1
    t1 = t2
    norm = psi.norm()
    print("Iteration: ", str(i).zfill(5), "    Time Elapsed: %.4f"%t_print, "    NORM: %.8f"%norm)
    psi.exportFull(params,i)

t2 = time.time()
t_print = t2-t0
print("\nTotal Time Elapsed = ",t_print)


psi.dtoh()


################################################################################################################################
# Show Plots
################################################################################################################################

# plt. plot(x, re**2 + im**2, label = "line 1")
# plt. plot(x, re_new**2 + im_new**2, label = "line 2")
# # plt. plot(x, re_new + im_new, label = "line 2")
# plt.show()

plt.rcParams.update({'font.size': 10})
# plt.rc('font',**{'family':'serif','serif':['Times']})
plt.imshow(np.transpose(psi2), extent=[0,100,-200,200], aspect=0.1, cmap='inferno', vmin=0, vmax=0.01)
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
# plt.savefig('DoubleSlit.png', dpi=600)
# plt.savefig("graph.pdf")
plt.show()

################################################################################################################################
# Trace Memory Management
################################################################################################################################

print("\n----------------------------------------------------------------")
print("Total Memory Usage")
print("----------------------------------------------------------------")

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)