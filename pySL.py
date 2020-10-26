import sys
import os
# Set target CUDA device. Must be an integer. Currently no error checking.
if (len(sys.argv) > 1):
  print("\nTarget CUDA device: ",str(sys.argv[1]))
  os.environ['CUDA_DEVICE'] = str(sys.argv[1])
import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc

# Import project modules
import src.get_gpu_info as gpuinfo
from src.input.parameters import getparams
from src.input.wavefunction import Wavefunction
from src.fft.fftauxV import Potential

import pycuda.gpuarray as gpuarray

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

if os.name == 'nt':
  mod = SourceModule(source, options=['-ccbin', 'cl.exe'])
else:
  mod = SourceModule(source, options=['-ccbin', 'cuda-g++'])

################################################################################################################################
# Staggered Leapfrog Setup
################################################################################################################################

v_si2nu = 1.312342066e-2
t_si2nu = 1.157676458e2

# Import parameters
params = getparams()


# create output dir

dirPath = ".\\" + params["SimName"]
if not os.path.isdir(dirPath):
    os.mkdir(dirPath)
dirPath = dirPath + "\\waves"
if not os.path.isdir(dirPath):
    os.mkdir(dirPath)

#test_cpu = np.ones([4,4])

#test_cpu[0,0]=1
#test_cpu[0,1]=10
#test_cpu[1,0]=2
#test_cpu[2,0]=3
#test_cpu[3,1]=4

#test_cpu = test_cpu.astype(np.float32)
#test_gpu = gpuarray.to_gpu(test_cpu)
#test_gpu.set(test_cpu)
#test_gpu = test_gpu.astype(np.float32)
#func_test = mod.get_function("gputest")


#Ngpu = np.int32(16)

#func_test(test_gpu, Ngpu, block=(16,1,1), grid=(1,1))

x = np.linspace(params["Xa"], params["Xb"], params["Nx"])  # Define the simulation domain
dx = x[1] - x[0]      



V = Potential(params) # Static 1D potential (particle in a box)
psi = Wavefunction(params,V)

dt = params["dt"] # Time step
TimeSteps = params["TimeSteps"] # Total solution iterations
PrintStep = params["PrintStep"]  # Used for saving the wave function

psi2 = np.zeros((int(TimeSteps/PrintStep), psi.N))

aa = dt/dx**2
bb = 2*dt

k = 0

################################################################################################################################
# CUDA malloc
################################################################################################################################


aa = aa.astype(np.float32)
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
numBlocks = math.ceil(N / blockSize)

k=0
t0 = time.time()
t1 = t0
for i in range(0, TimeSteps):

  # func_re(psi_gpu,V_gpu,bb,aa,N, block=(blockSize,1,1), grid=(numBlocks,1))
  # func_im(psi_gpu,V_gpu,bb,aa,N, block=(blockSize,1,1), grid=(numBlocks,1))

  func_re(psi.re_gpu,psi.im_gpu,V.pot_gpu,bb,aa,N, block=(blockSize,1,1), grid=(numBlocks,1))
  func_im(psi.re_gpu,psi.im_gpu,V.pot_gpu,bb,aa,N, block=(blockSize,1,1), grid=(numBlocks,1))

  if (i % PrintStep) == 0:
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