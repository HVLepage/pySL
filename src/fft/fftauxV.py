import numpy as np
import pycuda.driver as cuda
import math
import src.fft.index6Dto1D as index6Dto1D
import src.fft.index3Dto1D as index3Dto1D
import src.fft.index1DtoND as index1DtoND
import src.fft.indexNDto1D as indexNDto1D
from src.fft.OnePpotential import OnePpotential
from src.fft.TwoPpotential import TwoPpotential

class Potential():
    def dtoh(self):
        cuda.memcpy_dtoh(self.pot, self.pot_gpu)
    
    def htod(self):
        cuda.memcpy_htod(self.pot_gpu, self.pot)
        
    def readFromFile(self,params):
        file = open(".\src\\" + params["PotentialFileName"])
        self.filePot = np.zeros(params["N1"])
        lines = file.readlines()
        for i in range(0,params["N1"]):
            self.filePot[i] = float(lines[i+3].split()[0])
        
        
    def construct(self,params):
        Xa = params["Xa"]
        Xb = params["Xb"]
        Ya = params["Ya"]
        Yb = params["Yb"]
        Za = params["Za"]
        Zb = params["Zb"]
        Nx = params["Nx"]
        Ny = params["Ny"]
        Nz = params["Nz"]


        Np = params["Np"]
        dim = params["dim"]


        x = np.linspace(Xa, Xb, Nx)
        y = np.linspace(Ya, Yb, Ny)
        z = np.linspace(Za, Zb, Nz)
	
        if (params["ImportPotential"]):
            self.readFromFile(params)


        inds = np.zeros([dim,Np])

        if (Np > 1):
            interactionsNr = (Np**2 - Np) / 2
            interactionsMap = np.zeros(interactionsNr,2)
		


            for i in range(0,Np):
                for j in range(i+1,Np):
                    interactionsMap[i,0] = i
                    interactionsMap[i,1] = j
        else:
            interactionsNr = 0

        if (interactionsNr > 0):
            V2P=np.zeros((Nx * Ny * Nz)**2)

            for i1 in range(0,Nx):
                for j1 in range(0,Ny):
                    for k1 in range(0,Nz):
                        for i2 in range(0,Nx):
                            for j2 in range(0,Ny):
                                for k2 in range(0,Nz):
                                    VVal = TwoPpotential(params,x[i1],y[j1],z[k1],x[i2],y[j2],z[k2])
                                    if (VVal > 100):
                                        VVal = 100.0
    
                                    V2P[index6Dto1D.index(i1, j1, k1, i2, j2, k2, Nx, Ny, Nz, Nx, Ny, Nz)] = VVal

		
            for p in range(0,interactionsNr):

                p1 = interactionsMap[p]
                p2 = interactionsMap[p + 1]
                inds2P = np.zeros([dim,2])

                for i in range(0,params["N"]):
                    index1DtoND.index(i, inds, params["Sizes"], params)

                for j in range(0,params["dim"]):
                    inds2P[j,0] = inds[p1 * params["dim"] + j]
                    inds2P[j,1] = inds[p2 * params["dim"] + j]
				
                ind2P = indexNDto1D.index(inds2P, params["Sizes"], params, 2)
                self.pot[i] += params["Charge"][p1] * params["Charge"][p2] * V2P[ind2P]



	#Add 1 particle static V
        V1P = np.zeros(Nx * Ny * Nz)
        
        for p in range(0, params["Np"]):

            for i in range(0,Nx):
                for j in range(0,Ny):
                    for k in range(0,Nz):
                        VVal = 0.0;
                        ind = index3Dto1D.index(i, j, k, Nx, Ny, Nz)
                        
                        if (params["ImportPotential"]):
                            VVal= self.filePot[ind]
                        else:
                            VVal = OnePpotential(params,p,x[i],y[j],z[k])

                        if (VVal > 100):
                            VVal = 100.0
					

                        V1P[ind] = VVal
		
        inds1P = np.zeros(dim)
		

		#Only go into if right indexes are 0, i.e. do this other way around ...
        for i in range(0,params["N"]):
            index1DtoND.index(i, inds, params["Sizes"], params)

            for j in range(0, params["dim"]):
                inds1P[j] = inds[p * params["dim"] + j]
			
            ind1P = indexNDto1D.index(inds1P, params["Sizes"], params, 1)
            self.pot[i] += V1P[int(ind1P)]


            
    
    def __init__(self,params):        
        
        self.pot=np.zeros(params["N"])
        self.pot = self.pot.astype(np.float32)
        
        self.pot_gpu=cuda.mem_alloc(self.pot.nbytes)
        
        self.construct(params)
        
        self.pot = self.pot.astype(np.float32)

        self.htod()