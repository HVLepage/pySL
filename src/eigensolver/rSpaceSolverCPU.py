################################################################################################################################
# Real space eigensolver using the CPU
################################################################################################################################

import numpy as np
from numpy import linalg as LA
from src.export.exportEigenvalues import exportEigenvalues


def rSpaceSolverCPU(params, PsiOut, V, thr): #Real space solver GPU
	#read input parameters from params & declare stuff

	
    Nx = params["Nx"]
    Ny = params["Ny"]
    Nz = params["Nz"]
     
    Np = params["Np"]
    dim = params["dim"]
    PsiSize = (Nx * Ny * Nz) ** Np
	
	
	#generate Hamiltonian matrix for eigenfunction solving
	
	#Init Hamiltonian
    H = np.zeros([PsiSize, PsiSize], dtype = np.float32)
    
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
				

    D, eV = LA.eigh(H)

    for i in range(0,5):
        print(D[i])
	


	# Go over eigenstates to construct the init wave function as specified by Input Params
    for J in range(0,5):
        if ((params["InputStates"][2 * J] != 0.0) or (params["InputStates"][2 * J + 1] != 0.0)):
            PsiOut.wave += (params["InputStates"][2 * J] + 1j * params["InputStates"][2 * J + 1]) * eV[:,J].reshape(PsiOut.size) 

	
					
    if (params["TDSESolver"] == "rSpace"):		# do a final normalisation
        norm = PsiOut.norm()
        PsiOut.wave /= np.sqrt(norm)
    	# Write wave and eigenvalues to file as appropriate
        PsiOut.exportFull( params, -1, "wave")
    elif (params["TDSESolver"] == "kSpace"):
        PsiOut.exportFull( params, -1, "wave")
        PsiOut.wave = np.reshape(PsiOut.wave, PsiOut.size)
        PsiOut.htod()
        PsiOut.fft(thr, 0)
        PsiOut.dtoh()
        norm = PsiOut.norm()
        PsiOut.wave /= np.sqrt(norm)
        PsiOut.exportFull( params, -1, "Kwave")


	
    exportEigenvalues(params, D, 5)