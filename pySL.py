import sys
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc
import humanize

# Set target CUDA device. Must be an integer. Currently no error checking.
if (len(sys.argv) > 1): #sys.argv is a list which contains the command line arguments
  print("\nTarget CUDA device: ",str(sys.argv[1]))
  os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1]) #set environment variable

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

from src.export.exportNpTrace import exportNpTrace
from src.export.export1pTrace import export1pTrace



from src.input.potential import potential
from src.export.makepath import makepath
from src.export.exportParameters import exportParameters

import reikna.cluda as cluda



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
rSpaceSL = mod.get_function("rSpaceSL")


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
V = potential(params, mod) # Static potential
#print(V.pot)

#have to use single quotes

stability=0
for dXi in params["dXs"]:
    stability+= dXi**-2
print(f"Max stable potential value in NU= {1/params['dt']-stability}")
stability+=np.max(V.pot)
stability=stability**-1 * params['t_si2nu']**-1
print(f"dt={params['dt']/params['t_si2nu']} Stable for current potential if dt < {stability}")
print(f"Max potential value= {np.max(V.pot)}")


time_V = time.time() - t0V
t0V = time.time()
print("\nPotential allocated in %.4f seconds"%time_V)

t0=time.time()
psi = wavefunction(params,mod, thr, V) #create wavefunction as per parameters, sets as superposition of eigenstates
print("Wavefunction allocated in %.4f seconds"%(time.time()-t0) )

dt =  np.float32(params["dt"]) # Time step
TimeSteps = params["TimeSteps"] # Total solution iterations
PrintStep = params["PrintStep"]  # Used for saving the wave function




k = 0 # Keeps track of print step, will be made obsolete in future release

################################################################################################################################
# CUDA malloc
################################################################################################################################

# Get all values from parameters that need to be passed to GPU kernels
N = np.int32(params["N"])

Np = np.int32(params["Np"])
N1 = np.int32(params["N1"])

dim = np.int32(params["dim"])


fbb = np.float32(params["fbb"])
value = np.float32(params["dt"] * params["hbar"])

oneptdepPotential=np.int32(params["oneptdepPotential"])

#qnc variables
potentialBarrier=np.int32(params["potentialBarrier"])
barrierHeight=np.float32(params["barrierHeight"])
barrierPos=np.float32(params["barrierPos"])
barrierWidth=np.float32(params["barrierWidth"])
tauBarrier=np.float32(params["tauBarrier"])
adiabaticSine=np.int32(params["adiabaticSine"])
adiabaticCoulomb=np.int32(params["adiabaticCoulomb"])
nonadiabaticBarrier=np.int32(params["nonadiabaticBarrier"])
num_pairs=np.int32(params["num_pairs"])



Acos = np.float32(params["Acos"])
kXcos = np.float32(params["kXcos"])
tau= np.float32(params["tau"])
tauCoulomb=np.float32(params["tauCoulomb"])
wX = np.float32(params["wX"])

Xa = np.float32(params["Xa"])
Ya = np.float32(params["Ya"])

Nm1 = np.float32(1 / N)

#why have some variables been copied to device?
faa = cuda.mem_alloc_like(params["faa"])
cuda.memcpy_htod(faa, params["faa"])

Sizes = cuda.mem_alloc_like(params["Sizes"])
cuda.memcpy_htod(Sizes, params["Sizes"])

Mass = cuda.mem_alloc_like(params["Mass"])
cuda.memcpy_htod(Mass, params["Mass"])

dXs = cuda.mem_alloc_like(params["dXs"])
cuda.memcpy_htod(dXs, params["dXs"])


xEqms= cuda.mem_alloc_like(params["xEqms"])
cuda.memcpy_htod(xEqms, params["xEqms"])

same_spin_pairs= cuda.mem_alloc_like(params["same_spin_pairs"])
cuda.memcpy_htod(same_spin_pairs, params["same_spin_pairs"])


#Npxdim=cuda.mem_alloc_like(params["Np"]*params["dim"])
#cuda.memcpy_htod(Npxdim, params["Np"]*params["dim"])

#inds_kernel=cuda.mem_alloc_like(np.zeros(params["Np"]*params["dim"],dtype=np.int32))
#cuda.memcpy_hto d(inds_kernel, np.zeros(params["Np"]*params["dim"],dtype=np.int32))


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
        
        rSpaceSL(N, psi.re_gpu, psi.im_gpu, V.pot_gpu, t, np.int32(i%2), Np, N1, dim, fbb, faa, Sizes, Mass, dXs, Xa, Ya, wX, xEqms, Acos, kXcos, tau, oneptdepPotential, V.VCoulomb_gpu, tauCoulomb,potentialBarrier, barrierPos,  barrierWidth, barrierHeight,tauBarrier,adiabaticSine,adiabaticCoulomb,nonadiabaticBarrier,same_spin_pairs, num_pairs, block=(blockSize,1,1), grid=(numBlocks,1))
        
    
        
    if (i % PrintStep) == 0:
      
      
      #Print to file / record to plot every printStep
        t3=time.time()
        psi.dtoh()
        t_copy=time.time()-t3
     
         
        
        if(params["ExportNpTrace"]):
            t4=time.time()
            exportNpTrace(psi,params,i) 
            t_export=time.time()-t4
        elif(params["Export1pTrace"]):
            t4=time.time()
            export1pTrace(psi,params,i) 
            t_export=time.time()-t4     
        else:
            t4=time.time()
            psi.exportFull(params,i)
            t_export=time.time()-t4
            #psi2[k] = psi.wave.real**2 + psi.wave.imag**2

        k = k + 1
        t2 = time.time()
        t_print = t2-t1
        t1 = t2
        norm = psi.norm() #calculates norm not renormalise
        n_zero = math.log10((float)(params["TimeSteps"]) + 1) + 1
        print("Iteration:", str(i).zfill(int(n_zero)), "    Time Elapsed: %.2f"%t_print,"    Export time: %.2f"%t_export,"    Copy time: %.2f"%t_copy, "    NORM: %.8f"%norm)
        t3=time.time()
  
t2 = time.time()
t_print = t2-t0
print("\nTDSE time elapsed = ",t_print)


psi.dtoh()
psi.exportFull( params, -1, "wave_final")
#print(np.flip()


########################c########################################################################################################
# Show Plots
################################################################################################################################

# Animate the 1D wave function as save as an mp4 file. Used in the Particle_in_a_box_1D example
if params["dim"]==1 and params["Np"]==1:
    print("animate separately")
elif params["dim"]==1 and params["Np"]>1:
    print("animate separately")
    #animate2p_sum(params)
elif params["dim"]==2:
    print("animate separately")
    # t0=time.time()
    # animate2D(params) #significantly slower here for some reason than when run separately
    # print(f"The time taken to make animation = {time.time()-t0}")

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