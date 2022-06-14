################################################################################################################################
# Create the main wavefunction class
################################################################################################################################

import math
import numpy as np
import pycuda.driver as cuda
from  src.input.defWavefunction import defWavefunction
from  src.input.npfrom1p import npfrom1p
from  src.eigensolver.rSpaceSolverGPU import rSpaceSolverGPU
import time


class wavefunction:
    
   
    #use appropriate eigensolver to get wavefunction
    def solve(self,params, mod, thr, V):
        print("\n----------------------------------------------------------------")
        print("Solving eigenvalue problem using:", str(params["Eigensolver"]))
        print("----------------------------------------------------------------\n")
        print("Eigenvalues:\n")
        t0 = time.time()
     
      
        rSpaceSolverGPU(params,self,V, thr)
       
        time_eig = time.time() - t0
        print("\nEigensolver completed in %.4f seconds"%time_eig)
        

    # Retrieve data from GPU device and copy to host array
    def dtoh(self):
       
        tempre = np.zeros(self.N, dtype=np.float32)
        cuda.memcpy_dtoh(tempre, self.re_gpu) #allocated in init
        
        tempim = np.zeros(self.N, dtype=np.float32)
        cuda.memcpy_dtoh(tempim, self.im_gpu)
        self.wave = tempre + 1j*tempim
            #cuda.memcpy_dtoh(temp, self.im_gpu)
            #self.wave.imag = temp.reshape(self.size)
           
            
            #cuda.memcpy_dtoh(self.wave.real, self.re_gpu.reshape(self.size))
            #cuda.memcpy_dtoh(self.wave.imag, self.im_gpu.reshape(self.size))
    
    # Copy wave function array from host to GPU device
    def htod(self):
        
        
        cuda.memcpy_htod(self.re_gpu, self.wave.real.ravel()) #ravel flattens,if you modify the array returned by ravel, 
        #it may modify the entries in the original array.
        cuda.memcpy_htod(self.im_gpu, self.wave.imag.ravel())
        
    # Calculate norm, used at every print step to check simulation convergence
    def norm(self):
        norm = sum(self.wave.ravel().real**2 + self.wave.ravel().imag**2) #why dV?
        return norm

    #write full wavefunction to file
    def exportFull(self, params, step, Filename="wave"):
        n_zero = math.log10((float)(params["TimeSteps"] / params["PrintStep"]) + 1) + 1
        count = int(step / params["PrintStep"])
        index = str(count).zfill((int(n_zero))) #fill string with zeros until n_zero characters long
        
        
        if (step >= 0):
            OutputFileName = "./out/" + params["SimName"] + "/waves/wave_iteration_" + index + ".txt" 
        else:
            OutputFileName = "./out/" + params["SimName"] + "/" + Filename + ".txt"
	
        with open(OutputFileName, 'w+') as OutputFile:
            for x in np.nditer(self.wave): #efficient iterating
                OutputFile.write("{:e} {:e} \n".format(x.real, x.imag)) #exponent
	


    
    #if importing, read wavefunction from file
    def readFromFile(self,params):
        file = open("./src/input/" + params["WaveFileName"])
        lines = file.readlines()
        for i in range(0,params["N"]):
            self.wave.real[i] = float(lines[i].split()[0])
            self.wave.imag[i] = float(lines[i].split()[1])
        if (params["NormalizeWaveFunction"]):
            self.wave = self.wave / math.sqrt(self.norm())
        file.close()
    #set wavefunction manually using one of the functions in defWavefunction.py
    def setWavefunction(self,params):
        defWavefunction(params, self)
    
    #this is called on creation of instance
    def __init__(self, params, mod, thr, V=0): #0 is default argument, if V isn't passed to function then it will be set to 0
        
       
        
        self.TDSESolver = "rSpace"
        self.dV = (params["dx"] * params["dy"] *params["dz"]) ** params["Np"]
        
        self.N = params["N"]
        
        self.size1P = []
        self.size = []
        #seems redundant when can just repeat sizes array
        for i in range(0,params["dim"]):
            self.size1P.append(int(params["Sizes"][i]))
        for i in range(0,params["Np"]):
            self.size += self.size1P
        
        #self.wave=np.zeros(self.size, dtype = np.complex64)
        self.wave=np.zeros(self.N, dtype = np.complex64)
        
        
         #self.wave = self.wave.astype(np.complex64)
          
        
        self.re_gpu=cuda.mem_alloc_like(self.wave.real)
        self.im_gpu=cuda.mem_alloc_like(self.wave.imag)

        
        if params["defWavefunction"]:
            self.setWavefunction(params)
        elif params["npfrom1pWavefunction"]:
            npfrom1p(params,self,mod)
        elif (not params["ImportWavefunction"]):
            self.solve(params,mod, thr, V)
        
        
        else:
            self.readFromFile(params) #read from file
        
        # sumxs=trace_1p(params,self.wave.real,self.wave.imag)
        # Xa = params["Xa"]
        # Xb = params["Xb"]
        # Nx = params["Nx"]
        # x = np.linspace(Xa, Xb, Nx)
        
        # Xs=[f"x{p+1}" for p in range(params["Np"])]
        # for p in range(params["Np"]):
        #     Xs.pop(p)
        #     plt.figure()
        #     plt.xlabel(Xs[0])
        #     plt.ylabel(Xs[1])
        #     plt.contourf(x,x,np.transpose(sumxs[p].reshape([Nx,Nx])),levels=100,cmap='Greys')  #transpose since this is how contour works
        #     Xs=[f"x{p+1}" for p in range(params["Np"])]
        # plt.show()
        self.htod()