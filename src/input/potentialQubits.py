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
from scipy import sparse as sp
#from src.input.coulombPotential import CoulombPotential

def pauli_kron_sp(sigma,n,pair):
    sigma=sp.bsr_matrix(sigma)
    i1=pair[0]
    i2=pair[1]
    
    if abs(i2-i1)>1:
        eye1=sp.eye(2**(n-i2-1),dtype=np.int8)
        eye2=sp.eye(2**(i2-i1-1),dtype=np.int8)
        eye3=sp.eye(2**(i1),dtype=np.int8)
        return sp.kron(eye3,sp.kron(sigma,sp.kron(eye2,sp.kron(sigma,eye1))))
    else:
        eye1=sp.eye(2**(n-i2-1),dtype=np.int8)
        eye3=sp.eye(2**(i2-1),dtype=np.int8)
        return sp.kron(eye3,sp.kron(sigma,sp.kron(sigma,eye1)))

def pauli_kron(sigma,n,pair):
    arr=[np.array([[1,0],[0,1]],dtype=np.complex64)]*n
    arr[pair[0]]=sigma
    arr[pair[1]]=sigma
    outvar=arr[0]
    for ll in range(1,n):
            outvar=np.kron(outvar,arr[ll])
    return outvar

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
        
    # Build the potential from analytically
    def construct(self,params,t, mod):

        eye=np.array([[1,0],[0,1]], dtype=np.complex64)
        z = np.array([[1,0],[0,-1]], dtype=np.complex64)
        x = np.array([[0,1],[1,0]], dtype=np.complex64)
        y = np.array([[0,-1j],[1j,0]], dtype=np.complex64)

        Np = int(params["Np"])



	
        if (params["ImportPotential"]):
            self.readFromFile(params)



        if (Np > 1):
           
            interactionsMap = [[0,(t//params["FinalTime"])+1]] #map of which particle interacts with which
		


        else:
            interactionsNr = 0

        if (interactionsNr > 0):
            
            
            

    		#for each pair, fill the potential with their corresponding 2-particle interaction potential
            for p in range(0,interactionsNr):

                H=pauli_kron_sp(z,Np,interactionsMap[p])
                self.pot+=H.toarray()
               
            
        
    


            
    
    def __init__(self,params,t, mod):        
        
        self.pot=np.zeros(params["N"], dtype=np.float32)
        
        self.pot_gpu=cuda.mem_alloc(self.pot.nbytes)
    
        self.construct(params,t, mod)
        
     
        self.size = params["N"]

        
        #self.pot = np.reshape(self.pot, self.size)

        self.htod()