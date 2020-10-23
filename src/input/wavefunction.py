import math
import numpy as np
import pycuda.driver as cuda
from  src.eigensolver.RSpaceSolverCPU import RSpaceSolverCPU

class Wavefunction:
        

    # Retrieve data from GPU device and copy to host array


    def solve(self,params,V):
        RSpaceSolverCPU(params,self,V)
        
    def dtoh(self):
        cuda.memcpy_dtoh(self.re, self.re_gpu)
        cuda.memcpy_dtoh(self.im, self.im_gpu)
    
    # Copy wave function array from host to GPU device
    def htod(self):
        cuda.memcpy_htod(self.re_gpu, self.re)
        cuda.memcpy_htod(self.im_gpu, self.im)
        
    # Calculate norm, used at every print step to check simulation convergence
    def norm(self):
        norm = sum(self.re**2 + self.im**2) *self.dV
        return norm

    def exportFull(self, params, step, Filename="wave"):
        #n_zero = math.log10((float)(params["TimeSteps"] / params["PrintStep"]) + 1)
        #count = step / params["PrintStep"]
        #index = (n_zero - count.length(), '0') + count
        index = step
        
        if (step >= 0):
            OutputFileName = ".\\" + params["SimName"] + "\\waves\\wave_iteration_" + index + ".txt"
        else:
            OutputFileName = ".\\" + params["SimName"] + "\\" + Filename + ".txt"
	
        with open(OutputFileName, 'w+') as OutputFile:
            for i in range(0, self.N):
                OutputFile.write("%.8e %.8e\n", self.re[i], self.im[i])
	


    

    def readFromFile(self,params):
        file = open(".\input\\" + params["WavefunctionName"])
        lines = file.readlines()
        for i in range(0,params["N"]):
            self.re[i] = float(lines[i].split()[0])
            self.im[i] = float(lines[i].split()[1])
    
    def __init__(self,params):
        
        if (params["KSpace"]):
            self.dV = (params["dKx"] * params["dKy"] *params["dKz"]) ** params["Np"]
        else: 
            self.dV = (params["dx"] * params["dy"] *params["dz"]) ** params["Np"]
        
        self.N = params["N"];
        
        self.re=np.zeros(params["N"])
        self.im=np.zeros(params["N"])
        
        self.re = self.re.astype(np.float32)
        self.im = self.im.astype(np.float32)
        self.re_gpu=cuda.mem_alloc(self.re.nbytes)
        self.im_gpu=cuda.mem_alloc(self.im.nbytes)
        
        if (params["SolveEigenValueProblem"]):
            a=1 # put eigensolvers here
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