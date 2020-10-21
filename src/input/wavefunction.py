import numpy as np
import pycuda.driver as cuda

class Wavefunction:
        

    def dtoh(self):
        cuda.memcpy_dtoh(self.re, self.re_gpu)
        cuda.memcpy_dtoh(self.im, self.im_gpu)
    
    def htod(self):
        cuda.memcpy_htod(self.re_gpu, self.re)
        cuda.memcpy_htod(self.im_gpu, self.im)
        
        
    def norm(self):

        norm = sum(self.re**2 + self.im**2) * self.dV

        return norm
    
    def readFromFile(self,params):
        file = open(".\src\\" + params["WavefunctionName"])
        lines = file.readlines()
        for i in range(0,params["N"]):
            self.re[i] = float(lines[i].split()[0])
            self.im[i] = float(lines[i].split()[1])
    
    def __init__(self,params):
        
        self.dV = params["dx"] * params["dy"] * params["dz"] * params["Np"]
        
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