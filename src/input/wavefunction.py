################################################################################################################################
# Create the main wavefunction class
################################################################################################################################

import math
import numpy as np
import pycuda.driver as cuda
from  src.eigensolver.rSpaceSolverCPU import rSpaceSolverCPU
from  src.eigensolver.rSpaceSolverGPU import rSpaceSolverGPU
from  src.eigensolver.kSpaceSolverCPU import kSpaceSolverCPU
from  src.eigensolver.kSpaceSolverGPU import kSpaceSolverGPU
from  src.eigensolver.rSpaceSparseSolverGPU import rSpaceSparseSolverGPU
import reikna.fft as rfft
import pycuda.gpuarray as gpuarray
from src.get_ccbin import get_ccbin
import time
import random
from scipy.linalg import expm, logm, sqrtm

class wavefunction:
    
    def fft(self, thr, inv = 0):
        #Perform fft or ifft
        #api = cluda.cuda_api()
        #thr = api.Thread.create()
        #fft_h = rfft.FFT(self.wave)
        #fftc=fft_h.compile(thr)
        self.fftc(self.wave_gpu, self.wave_gpu, inv)
        thr.synchronize()

    #use appropriate eigensolver to get wavefunction
    def solve(self,params, mod, thr, V):
        print("\n----------------------------------------------------------------")
        print("Solving eigenvalue problem using:", str(params["Eigensolver"]))
        print("----------------------------------------------------------------\n")
        print("Eigenvalues:\n")
        t0 = time.time()
        if params["Eigensolver"] == "CPUrSpace":
            rSpaceSolverCPU(params,self,V, thr)
        elif params["Eigensolver"] == "GPUrSpace":
            rSpaceSolverGPU(params,self,V, thr)
        elif params["Eigensolver"] == "CPUkSpace":
            kSpaceSolverCPU(params,self, mod, thr)
        elif params["Eigensolver"] == "GPUkSpace":
            kSpaceSolverGPU(params,self, mod, thr)
        elif params["Eigensolver"] == "GPUrSpaceSparse":
            rSpaceSparseSolverGPU(params,self,V, thr)
        time_eig = time.time() - t0
        print("\nEigensolver completed in %.4f seconds"%time_eig)
        

    # Retrieve data from GPU device and copy to host array
    def dtoh(self):
        if (self.TDSESolver == "kSpace"):
            self.wave_gpu.get(self.wave)
        else:
            tempre = np.zeros(self.N, dtype=np.float32)
            cuda.memcpy_dtoh(tempre, self.re_gpu)
            
            tempim = np.zeros(self.N, dtype=np.float32)
            cuda.memcpy_dtoh(tempim, self.im_gpu)
            self.wave = tempre + 1j*tempim
            #cuda.memcpy_dtoh(temp, self.im_gpu)
            #self.wave.imag = temp.reshape(self.size)
           
            
            #cuda.memcpy_dtoh(self.wave.real, self.re_gpu.reshape(self.size))
            #cuda.memcpy_dtoh(self.wave.imag, self.im_gpu.reshape(self.size))
    
    # Copy wave function array from host to GPU device
    def htod(self):
        if self.TDSESolver=="kSpace":
            self.wave_gpu.set(self.wave)
        else:
            cuda.memcpy_htod(self.re_gpu, self.wave.real.ravel())
            cuda.memcpy_htod(self.im_gpu, self.wave.imag.ravel())
        
    # Calculate norm, used at every print step to check simulation convergence
    def norm(self):
        norm = sum(self.wave.ravel().real**2 + self.wave.ravel().imag**2) *self.dV
        return norm

    #write full wavefunction to file
    def exportFull(self, params, step, Filename="wave"):
        n_zero = math.log10((float)(params["TimeSteps"] / params["PrintStep"]) + 1) + 1
        count = int(step / params["PrintStep"])
        index = str(count).zfill((int(n_zero)))
        
        if (step >= 0):
            OutputFileName = "./out/" + params["SimName"] + "/waves/wave_iteration_" + index + ".txt"
        else:
            OutputFileName = "./out/" + params["SimName"] + "/" + Filename + ".txt"
	
        with open(OutputFileName, 'w+') as OutputFile:
            for x in np.nditer(self.wave):
                OutputFile.write("{:e} {:e} \n".format(x.real, x.imag))
	


    
    #if importing, read wavefunction from file
    def readFromFile(self,params):
        file = open("./input/" + params["WaveFileName"])
        lines = file.readlines()
        for i in range(0,params["N"]):
            self.wave.real[i] = float(lines[i].split()[0])
            self.wave.imag[i] = float(lines[i].split()[1])
        if (params["NormalizeWaveFunction"]):
            self.wave = self.wave / math.sqrt(self.norm())
    
    def demonInit(self, params):
        # Get the current time in seconds as a floating-point number
        zp=np.array([1,0], dtype=np.complex64)
        zm=np.array([0,1], dtype=np.complex64)
        z = np.array([[1,0],[0,-1]], dtype=np.complex64)
        
        current_time = time.time()

        # Convert the current time to an integer and use it as the seed for the random number generator
        random.seed(int(current_time))

        random_excited = random.random(0,1)
        random_phase=random.random(0,2*np.pi)
        
        rho_th = expm(- (1/params["T"])*z)
        
        rho_th=rho_th/np.trace(rho_th)
        
        psi_thermal=  np.sqrt(1-rho_th[0,0])*zm + np.sqrt(rho_th[0,0])*np.exp(1j*random_phase)*zp
        
        psi_init=psi_thermal
        
        for i in range(0, params["Np"]-1):
            psi_init=np.kron(psi_init,zm)
        #psi_thermal=psi_thermal/np.norm(psi_thermal)
    
    #thi sis called on creation of instance
    def __init__(self, params, mod, thr, V=0):
        

        self.TDSESolver = "qubits"
        
        self.N = params["N"]
        


        
        #self.wave=np.zeros(self.size, dtype = np.complex64)
        self.wave=np.zeros(self.N, dtype = np.complex64)
        
         #self.wave = self.wave.astype(np.complex64)
 
        self.re_gpu=cuda.mem_alloc_like(self.wave.real)
        self.im_gpu=cuda.mem_alloc_like(self.wave.imag)
        
        
        if (not params["ImportWavefunction"]):
            self.demonInit(params)
        else:
            self.readFromFile(params) #read from file
        
        
        self.htod()