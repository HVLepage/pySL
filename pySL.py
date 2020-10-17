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
import src.input.parameters as inputparameters
import src.input.wavefunction as wavefunction
from src.export.calculatenorm import getNorm

print("\n----------------------------------------------------------------")
print("pySL version 0.0.1")
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

mod = SourceModule(source, options=['-ccbin', 'cuda-g++'])

################################################################################################################################
# Staggered Leapfrog Setup
################################################################################################################################


v_si2nu = 1.312342066e-2
t_si2nu = 1.157676458e2

params = inputparameters.getparams()



x = np.linspace(params["x0"], params["x1"], params["Nx"])  # Define the simulation domain
dx = x[1] - x[0]                  # Extract spacing between lattice sites


# Define initial wave function
w = 0.1                           # Initial wave packet spread
xoffset = 40                      # Wave packet distance from center
re = ((w/np.pi)**(1/4)*np.exp(-w*(x-xoffset)**2/2) + (w/np.pi)**(1/4)*np.exp(-w*(x+xoffset)**2/2))/2 # Initial wave function
im = np.zeros(len(x)) # Initial wave function (imaginary part)

V = np.zeros(len(x)) # Static 1D potential (particle in a box)

dt = params["dt"] # Time step
N = params["TimeSteps"] # Total solution iterations
PrintStep = params["PrintStep"]  # Used for saving the wave function

psi2 = np.zeros((int(N/PrintStep),len(x)))

aa = dt/dx**2
bb = 2*dt

k = 0


################################################################################################################################
# CUDA malloc
################################################################################################################################

V = V.astype(np.float32)
V_gpu = cuda.mem_alloc(V.nbytes)
cuda.memcpy_htod(V_gpu, V)

re = re.astype(np.float32)
re_gpu = cuda.mem_alloc(re.nbytes)
cuda.memcpy_htod(re_gpu, re)

im = im.astype(np.float32)
im_gpu = cuda.mem_alloc(im.nbytes)
cuda.memcpy_htod(im_gpu, im)

aa = aa.astype(np.float32)
bb = np.float32(bb)
Nx = np.int32(params["Nx"])



re_new = np.empty_like(re)
im_new = np.empty_like(im)


################################################################################################################################
# IterateGPU
################################################################################################################################

print("\n----------------------------------------------------------------")
print("GPU-accelerated Simulation")
print("----------------------------------------------------------------")

func_re = mod.get_function("gqsl_re")
func_im = mod.get_function("gqsl_im")

blockSize = 256
numBlocks = math.ceil((Nx + blockSize - 1) / blockSize)


cuda.memcpy_htod(re_gpu, re)
cuda.memcpy_htod(im_gpu, im)

# psi2 = np.zeros((int(N/PrintStep),len(x)))

k=0
t0 = time.time()
t1 = t0
for i in range(0, N):
  func_re(
    re_gpu,
    im_gpu,
    V_gpu,
    bb,
    aa,
    Nx, 
    block=(blockSize,1,1), grid=(numBlocks,1))
  func_im(re_gpu,im_gpu,V_gpu,bb,aa,Nx, block=(blockSize,1,1), grid=(numBlocks,1))
  if (i % PrintStep) == 0:
    cuda.memcpy_dtoh(re_new, re_gpu)
    cuda.memcpy_dtoh(im_new, im_gpu)
    psi2[k] = re_new**2 + im_new**2
    k = k + 1
    t2 = time.time()
    t_print = t2-t1
    t1 = t2
    norm = getNorm(re_new,im_new,params)
    print("Iteration", i, "        k = ", k, "        Time Elapsed = ",t_print, "NORM: ", norm)

t2 = time.time()
t_print = t2-t0
print("Total Time Elapsed = ",t_print)


cuda.memcpy_dtoh(re_new, re_gpu)
cuda.memcpy_dtoh(im_new, im_gpu)



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