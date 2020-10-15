import sys
import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc

tracemalloc.start()

sys.version
pycuda.VERSION_TEXT


###############################################################################################################################
# Staggered Leapfrog Setup
###############################################################################################################################

Nx = 40000000
x = np.linspace(-200, 200, Nx)  # Define the simulation domain
dx = x[1] - x[0]                  # Extract spacing between lattice sites

w = 0.1                           # Initial wave packet spread
xoffset = 40                      # Wave packet distance from center
re = ((w/np.pi)**(1/4)*np.exp(-w*(x-xoffset)**2/2) + (w/np.pi)**(1/4)*np.exp(-w*(x+xoffset)**2/2))/2 # Initial wave function
im = np.zeros(len(x)) # Initial wave function (imaginary part)

V = np.zeros(len(x)) # Static 1D potential (particle in a box)

dt = 0.00000005 # Time step
N = 1000 # Total solution iterations
PrintStep = 100 # Used for saving the wave function

# psi2 = np.zeros((int(N/PrintStep),len(x)))

aa = dt/dx**2
bb = 2*dt

k = 0



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
# aa_gpu = cuda.mem_alloc(aa.nbytes)
# cuda.memcpy_htod(aa_gpu, aa)

bb = np.float32(bb)
# bb_gpu = cuda.mem_alloc(bb.nbytes)
# cuda.memcpy_htod(bb_gpu, bb)

Nx = np.int32(Nx)
# Nx_gpu = cuda.mem_alloc(Nx.nbytes)
# cuda.memcpy_htod(Nx_gpu, Nx_int)


re_new = np.empty_like(re)
im_new = np.empty_like(im)

###############################################################################################################################
# CUDA Module
###############################################################################################################################

mod = SourceModule("""

  __global__ void gqsl_re(float *re, float *im, float *V, float bb, float aa, int Nx)
  {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx > 0 && idx < Nx-1){
      //re[idx] = re[idx];
      re[idx] = re[idx] - (V[idx]*bb*im[idx] + aa*(2*im[idx] - (im[idx-1] + im[idx+1])));
    }
  }

  __global__ void gqsl_im(float *re, float *im, float *V, float bb, float aa, int Nx)
  {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx > 0 && idx < Nx-1){
      //im[idx] = im[idx];
      im[idx] = im[idx] +  V[idx]*bb*re[idx] + aa*(2*re[idx] - (re[idx-1] + re[idx+1]));
    }
  }

  """)

###############################################################################################################################
# IterateGPU
###############################################################################################################################

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
  func_re(re_gpu,im_gpu,V_gpu,bb,aa,Nx, block=(blockSize,1,1), grid=(numBlocks,1))
  func_im(re_gpu,im_gpu,V_gpu,bb,aa,Nx, block=(blockSize,1,1), grid=(numBlocks,1))
  if (i % PrintStep) == 0:
    cuda.memcpy_dtoh(re_new, re_gpu)
    cuda.memcpy_dtoh(im_new, im_gpu)
    # psi2[k] = re_new**2 + im_new**2
    k = k + 1
  if (i % (PrintStep)) == 0:
    t2 = time.time()
    t_print = t2-t1
    t1 = t2
    print("Iteration", i, "        k = ", k, "        Time Elapsed = ",t_print)

t2 = time.time()
t_print = t2-t1
t1 = t2
print("Iteration Time Elapsed = ",t_print)


# Flush context printf buffer
# cuda.Context.synchronize()



cuda.memcpy_dtoh(re_new, re_gpu)
cuda.memcpy_dtoh(im_new, im_gpu)



###############################################################################################################################
# Show Plots
###############################################################################################################################

# plt. plot(x, re**2 + im**2, label = "line 1")
# plt. plot(x, re_new**2 + im_new**2, label = "line 2")
# # plt. plot(x, re_new + im_new, label = "line 2")
# plt.show()



###############################################################################################################################
# Trace Memory Management
###############################################################################################################################


snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)