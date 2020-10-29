################################################################################################################################
# Real space eigensolver using the GPU
################################################################################################################################


import numpy as np
import ctypes
import pycuda.gpuarray as gpuarray
import os
from skcuda import linalg
from src.export.exportEigenvalues import exportEigenvalues

def rSpaceSolverGPU(params, PsiOut, V): #Real space solver GPU
	#read input parameters from params & declare stuff

	
    Nx = params["Nx"]
    Ny = params["Ny"]
    Nz = params["Nz"]
     
    Np = params["Np"]
    dim = params["dim"]
    PsiSize = (Nx * Ny * Nz) ** Np
	
	
	#generate Hamiltonian matrix for eigenfunction solving
	
	#Init Hamiltonian
    H = np.zeros([PsiSize, PsiSize], dtype=np.float32)
    
    H2 = params["hbar"] ** 2 #hbar squared

    for k in range(0, PsiSize):# Go over each row



		#fill Hamiltonian, making use of hermiticity to only fill half and do conjugate

		#on-site potential
        H[k,k] += V.pot[k] #!!!!!!!!!!!!!!!!!!!!!!!
        for d in range(0,dim):
            for p in range(0,Np):
                H[k,k] += H2 / (params["Mass"][p] * params["dXs"][d] ** 2) # on-site kinetic part

                if ((k-1)>=0):
                    H[k,k-1] -= H2 / (2 * params["Mass"][p] * params["dXs"][d] ** 2)
				
                if ((k+1)<PsiSize):
                    H[k,k+1] -=  H2 / (2 * params["Mass"][p] * params["dXs"][d] ** 2)
				

   #Do eig things here
    H=np.reshape(H.ravel(), [PsiSize,PsiSize], order='F')
    H_gpu = gpuarray.to_gpu(H)
    D=np.zeros(PsiSize, dtype=np.float32)
    
    linalg.init()
    V_gpu, D_gpu = linalg.eig(H_gpu, 'N', 'V')


    D_gpu.get(D)
    V_gpu.get(H)
    #H=H.reshape([PsiSize,PsiSize])
    H=np.reshape(H.ravel(), [PsiSize,PsiSize], order='C')

    for i in range(0,5):
        print(D[i])
	


	# Go over eigenstates to construct the init wave function as specified by Input Params
    for J in range(0,5):
        if ((params["InputStates"][2 * J] != 0.0) or (params["InputStates"][2 * J + 1] != 0.0)):
            PsiOut.wave += params["InputStates"][2 * J] * H[:,J] 

	
							# do a final normalisation
    norm = PsiOut.norm()

    PsiOut.wave /= np.sqrt(norm)
    
		


	# Write wave and eigenvalues to file as appropriate
    #if (KSpaceLeapfrog):
	#	ExportFullWave(PsiOut, params, -1, PsiSize, params["SimName, "wave")
	#	FFTAuxPsi(params, PsiOut, planConvND, true, false)
	#	ExportFullWave(PsiOut, params, -1, PsiSize, params["SimName, "Kwave")
	#else:
    PsiOut.exportFull( params, -1, "wave")
	
    exportEigenvalues(params, D, 5)
