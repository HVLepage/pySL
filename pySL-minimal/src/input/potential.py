################################################################################################################################
# Create the potential class for storing the time-independent potential of the problem, including 2-particle interactions like Coulomb
################################################################################################################################

import math
import numpy as np
import pycuda.driver as cuda
import src.index.index6Dto1D as index6Dto1D
import src.index.index3Dto1D as index3Dto1D
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D
from src.input.electrostaticPotential import electrostaticPotential
from src.input.harmonicPotential import harmonicPotential
from src.input.potentialStep import potentialStep
from src.input.sinusoidalPotential import sinusoidalPotential

import matplotlib.pyplot as plt

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

#from src.input.coulombPotential import CoulombPotential

class potential():
    
    #copy data from device to host
    def dtoh(self):
        cuda.memcpy_dtoh(self.pot, self.pot_gpu)
    
    #copy data from host to device
    def htod(self):
        cuda.memcpy_htod(self.pot_gpu, self.pot)
        
    #read potential from file specified in params
    def readFromFile(self,params):
        file = open("./input/" + params["PotentialFileName"])
        self.filePot = np.zeros(params["N1"])
        lines = file.readlines()
        for i in range(0,params["N1"]):
            self.filePot[i] = float(lines[i+3].split()[0])
        
    # Build the potential form analytically
    def construct(self,params, mod):
        
        CoulombPotential = mod.get_function("CoulombPotential")
        AllocateCoulombPotential = mod.get_function("AllocateCoulombPotential")

        Xa = params["Xa"]
        Xb = params["Xb"]
        Ya = params["Ya"]
        Yb = params["Yb"]
        Za = params["Za"]
        Zb = params["Zb"]
        Nx = params["Nx"]
        Ny = params["Ny"]
        Nz = params["Nz"]


        Np = int(params["Np"])
        dim = int(params["dim"])


        x = np.linspace(Xa, Xb, Nx)
        y = np.linspace(Ya, Yb, Ny)
        z = np.linspace(Za, Zb, Nz)
	
        if (params["ImportPotential"]):
            self.readFromFile(params)


        inds = np.zeros(dim*Np)

        if (Np > 1):
            interactionsNr = int((Np**2 - Np) / 2) #Nr of interactions between particle pairs = Np choose 2
            #interactionsMap = np.zeros([interactionsNr,2]) #map of which particle interacts with which
            interactionsMap=[]
            for i in range(0,Np):
                for j in range(0,Np):
                    if j>i:
                        interactionsMap.append([i,j])
            interactionsMap=np.asarray(interactionsMap)  
            #print(interactionsMap)

            #alternative method, neither optimal, above faster for few particles
            # count=0
            # for i in range(0,Np):
            #     for j in range((i+1),Np):
            #         interactionsMap[count+(j-(i+1))][0]=i
            #         interactionsMap[count+(j-(i+1))][1]=j
            #     count+=len(range(i+1,Np))



        else:
            interactionsNr = 0

        if (interactionsNr > 0):
            V2P=np.zeros((Nx * Ny * Nz)**2, dtype = np.float32) #store 2-particle potential
            V2P_gpu = cuda.mem_alloc_like(V2P)

            Sizes = cuda.mem_alloc_like(params["Sizes"])
            cuda.memcpy_htod(Sizes, params["Sizes"])

            Xas = cuda.mem_alloc_like(params["Xas"])
            cuda.memcpy_htod(Xas, params["Xas"])

            dXs = cuda.mem_alloc_like(params["dXs"])
            cuda.memcpy_htod(dXs, params["dXs"])
            
            
            blockSize = 256 
            
            numBlocks = math.ceil((Nx * Ny * Nz)**2 / blockSize) #math.ceil returns type int, np.ceil returns np.float64
            
            #calculate 2-particle potential using Coulomb interaction
            if params["dim"]==1:
                
                softeningParam=(np.pi/2)**0.5*params["sigSoft"]
                
           

            CoulombPotential(np.int32((Nx * Ny * Nz)**2), V2P_gpu,  np.int32(Nx),  np.int32(Ny),  np.int32(Nz),  np.float32(Xa),   np.float32(params["dx"]),
                                  np.float32(Ya),   np.float32(params["dy"]),   np.float32(Za),   np.float32(params["dy"]),   np.float32(params["RelPerm"]),
                                  Sizes,np.int32(params["dim"]),Xas,dXs,np.float32(softeningParam),block=(blockSize,1,1), grid=(numBlocks,1))

            cuda.memcpy_dtoh(V2P, V2P_gpu)
            #print(V2P)

    		#for each pair, fill the potential with their corresponding 2-particle interaction potential
            
         
            VCoulomb=np.zeros((Nx * Ny * Nz)**params["Np"], dtype = np.float32) #store 2-particle potential
            VCoulomb_gpu = cuda.mem_alloc_like(VCoulomb)
            Charge = cuda.mem_alloc_like(params["Charge"])
            cuda.memcpy_htod(Charge, params["Charge"])
            numBlocks = math.ceil(params["N"] / blockSize)
            for p in range(0,interactionsNr):

                p1 = interactionsMap[p,0] #1st particle of pair
                p2 = interactionsMap[p,1]
                # #inds2P = np.zeros(dim*2) #1st particle then 2nd
                
                
                AllocateCoulombPotential(np.int32(params["N"]),VCoulomb_gpu,np.int32(p1),np.int32(p2),np.int32(params["dim"]),np.int32((Nx * Ny * Nz)**2),Sizes,Charge,V2P_gpu,np.int32(params["Np"]),block=(blockSize,1,1), grid=(numBlocks,1))

                cuda.memcpy_dtoh(VCoulomb,VCoulomb_gpu)
                #print(VCoulomb)
                if not params["oneptdepPotential"]: 
                    self.pot+=VCoulomb
                else:
                    self.VCoulomb+=VCoulomb
                

                VCoulomb=np.zeros((Nx * Ny * Nz)**params["Np"], dtype = np.float32)
                cuda.memcpy_htod(VCoulomb_gpu,VCoulomb)


                # inds2P = np.zeros(dim*2) #1st particle then 2nd

                # for i in range(0,params["N"]):
                #     index1DtoND.index(i, inds, params["Sizes"], params) #high dimensional (Np*dim) matrix with N elements, assigns inds values

                #     for j in range(0,dim):
                #         #get the indices of the 2 particles 
                #         inds2P[j] = inds[int(p1 * dim + j)] #for a given 1D index, the 1st particle's ND indices are assigned (starts with z)
                #         inds2P[j + dim] = inds[int(p2 * dim + j)]
                #     #print(inds2P)
                #     ind2P = int(indexNDto1D.index(inds2P, params["Sizes"], params, 2))
                #     #print(ind2P)
                    # if not params["oneptdepPotential"]: 
                    #     self.pot[i] += params["Charge"][int(p1)] * params["Charge"][int(p2)] * V2P[ind2P]
                    # else:
                    #     self.VCoulomb[i]+= params["Charge"][int(p1)] * params["Charge"][int(p2)] * V2P[ind2P]


            #add a single particle potential, may be different for each particle
        
        #print(self.VCoulomb)
        #this code is pretty bad, must make sure use x,y if 2D
        if not params["oneptdepPotential"]:
            if params["1pPotential"]: #allocating entire potential
                for p in range(0,params["Np"]):
                    inds1P=np.zeros(dim)
                    V1P = np.zeros(Nx * Ny * Nz)
                    for i in range(0,Nx):
                        for j in range(0,Ny):
                            for k in range(0,Nz):
                                VVal = 0.0
                                ind = index3Dto1D.index(i, j, k, Nx, Ny, Nz)
                                
                                #either import from file, or calculate from electrostatic equations
                                # if (params["ImportPotential"]):
                                #     VVal= self.filePot[ind]
                                    
                            
                                    #VVal = electrostaticPotential(params,p,x[i],y[j],z[k])
                                if params["1pPotentialType"]=="Harmonic":
                                    VVal = harmonicPotential(params,p,x[i],y[j],z[k]) #actually is dimension dependent even though passed 3 arguments
                                if params["1pPotentialType"]=="Sinusoidal":
                                    VVal=sinusoidalPotential(params,p,x[i],y[j],z[k])
                                    VVal+=harmonicPotential(params,p,x[i],y[j],z[k])
                                if params["1pPotentialStep"]:
                                    VVal+=potentialStep(params,x[i])
                                    
                                if (VVal > 100):
                                    VVal = 100.0
                            

                                V1P[ind] = VVal


                    for i in range(0,params["N"]):
                        index1DtoND.index(i, inds, params["Sizes"], params)

                        for j in range(0,dim):
                            inds1P[j]=inds[p*dim+j]
                        
                        ind1P=indexNDto1D.index(inds1P, params["Sizes"], params, 1)
                        self.pot[i]+=V1P[int(ind1P)]
                if params["dim"]==1:
                
                    plt.figure()
                    plt.plot(x,V1P)
                    plt.show()
            
    
    def __init__(self,params, mod):        
        
        self.pot=np.zeros(params["N"], dtype=np.float32)
        self.VCoulomb=np.zeros(params["N"], dtype=np.float32)
        
        self.pot_gpu=cuda.mem_alloc(self.pot.nbytes)
        self.VCoulomb_gpu=cuda.mem_alloc(self.VCoulomb.nbytes)
    
        self.construct(params, mod)
        
        if params["oneptdepPotential"]:
            cuda.memcpy_htod(self.VCoulomb_gpu, self.VCoulomb)


        self.size1P = []
        self.size = []
        for i in range(0,params["dim"]):
            self.size1P.append(int(params["Sizes"][i]))
        for i in range(0,params["Np"]):
            self.size += self.size1P
        
        #self.pot = np.reshape(self.pot, self.size)

        self.htod()