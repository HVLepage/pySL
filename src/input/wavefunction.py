import math
import numpy as np
import pycuda.driver as cuda
from  src.eigensolver.RSpaceSolverCPU import RSpaceSolverCPU
import pycuda.gpuarray as gpuarray

class Wavefunction:
        

    # Retrieve data from GPU device and copy to host array


    def solve(self,params,V):
        RSpaceSolverCPU(params,self,V)
        
    def dtoh(self):
        self.re_gpu.get(self.re)
        self.im_gpu.get(self.im)
    
    # Copy wave function array from host to GPU device
    def htod(self):
        self.re_gpu.set(self.re)
        self.im_gpu.set(self.im)
        
    # Calculate norm, used at every print step to check simulation convergence
    def norm(self):
        norm = sum(self.re**2 + self.im**2) *self.dV
        return norm

    def exportFull(self, params, step, Filename="wave"):
        n_zero = math.log10((float)(params["TimeSteps"] // params["PrintStep"]) + 1)
        count = int(step / params["PrintStep"])
        #index = (n_zero - count.length(), '0') + count
        index = str(count).zfill((int(n_zero)))
        
        if (step >= 0):
            OutputFileName = ".\\" + params["SimName"] + "\\waves\\wave_iteration_" + index + ".txt"
        else:
            OutputFileName = ".\\" + params["SimName"] + "\\" + Filename + ".txt"
	
        with open(OutputFileName, 'w+') as OutputFile:
            for i in range(0, self.N):
                OutputFile.write("{:e} {:e} \n".format(self.re[i], self.im[i]))
	


    

    def readFromFile(self,params):
        file = open(".\input\\" + params["WavefunctionName"])
        lines = file.readlines()
        for i in range(0,params["N"]):
            self.re[i] = float(lines[i].split()[0])
            self.im[i] = float(lines[i].split()[1])
    
    def __init__(self,params,V=0):
        
        if (params["KSpaceLeapfrog"]):
            self.dV = (params["dKx"] * params["dKy"] *params["dKz"]) ** params["Np"]
        else: 
            self.dV = (params["dx"] * params["dy"] *params["dz"]) ** params["Np"]
        
        self.N = params["N"]
        
        size = []
        for i in range(0,params["dim"]):
            size.append(int(params["Sizes"][i]))
        
        
        self.re=np.zeros(size)
        self.im=np.zeros(size)
        
        self.re = self.re.astype(np.float32)
        self.im = self.im.astype(np.float32)
        self.re_gpu=gpuarray.to_gpu(self.re)
        self.im_gpu=gpuarray.to_gpu(self.im)
        
        if (params["SolveEigenValueProblem"]):
            RSpaceSolverCPU(params,self,V)
        else:
            #self.readFromFile(params) #read from file
            
            # Define initial wave function
            w = 0.1                           # Initial wave packet spread
            xoffset = 40                      # Wave packet distance from center
            x = np.linspace(params["Xa"], params["Xb"], params["Nx"])  # Define the simulation domain
            
            self.re = (2**(0.5))*((w/np.pi)**(1/4)*np.exp(-w*(x-xoffset)**2/2) + (w/np.pi)**(1/4)*np.exp(-w*(x+xoffset)**2/2))/2 # Initial wave function
        
        
        self.re = self.re.astype(np.float32)
        self.im = self.im.astype(np.float32)
        self.htod()