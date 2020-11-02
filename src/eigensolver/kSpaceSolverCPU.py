################################################################################################################################
# Real space eigensolver using the CPU
################################################################################################################################

import numpy as np
from numpy import linalg as LA
from src.export.exportEigenvalues import exportEigenvalues
from src.input.potentialKSpace import potentialKSpace
import src.fft.index1DtoND as index1DtoND
import src.fft.indexNDto1D as indexNDto1D

def kSpaceSolverCPU(params, PsiOut): #Real space solver GPU
	#read input parameters from params & declare stuff

	

    KxCO = params["KxECO"]
    KyCO = params["KyECO"]
    KzCO = params["KzECO"]

    Kx = 2 * KxCO + 1
    Ky = 2 * KyCO + 1
    Kz = 2 * KzCO + 1
    
    dKx = params["dKx"]
    dKy = params["dKy"]
    dKz = params["dKz"]
    
    Np = params["Np"]
    dim = params["dim"]
    PsiSize = (Kx * Ky * Kz) ** Np
	
    
    SizesK = []
    SizesKE = []
    KCOs = []
    dKs = []
    for i in range(0,dim):
        if (i == 0):
            SizesK.append(Kx)
            SizesKE.append(4 * KxCO + 1)
            KCOs.append(KxCO)
            dKs.append(dKx)
        elif (i ==1):
            SizesK.append(Ky)
            SizesKE.append(4 * KyCO + 1)
            KCOs.append(KyCO)
            dKs.append(dKy)
        elif (i ==2):
            SizesK.append(Kz)
            SizesKE.append(4 * KzCO + 1)
            KCOs.append(KzCO)
            dKs.append(dKz)
	
	#generate Hamiltonian matrix for eigenfunction solving
	
	#Init Hamiltonian
    H = np.zeros(PsiSize*PsiSize, dtype = np.complex64)
    
    H2 = params["hbar"] ** 2 #hbar squared
    
    Vk = potentialKSpace(params, True)

    indsCol = np.zeros(dim*Np, dtype=np.int32)
    indsRow = np.zeros(dim*Np, dtype=np.int32)
    indsDiff = np.zeros(dim*Np, dtype=np.int32)
    
    for k in range(0, PsiSize**2):# Go over each row

        j = k % PsiSize # column index
        i = np.floor(np.float64(k) / np.float64(PsiSize)) #i is row index

        index1DtoND.index(j, indsCol, SizesK, params)
        index1DtoND.index(i, indsRow, SizesK, params)
		# Calculate offset from diagonal
        for l in range(0, dim * Np):
            indsDiff[l] = indsCol[l] - indsRow[l] if  (indsCol[l] - indsRow[l]) >= 0 else (4 * KCOs[l % dim] + 1) + (indsCol[l] - indsRow[l]) #((indsCol[l] - indsRow[l]) >= 0 ? (indsCol[l] - indsRow[l]) : (4 * KCOs[l % dim] + 1) + (indsCol[l] - indsRow[l]))
		
        
		#fill Hamiltonian, making use of hermiticity to only fill half and do conjugate
        H[j + PsiSize * i] = Vk[indexNDto1D.index(indsDiff, SizesKE, params)]
        
        if (i == j): #on - diagonal element
            for p in range(0,Np):
                for d in range(0,dim):
                    H[j + PsiSize * i] +=  ((indsRow[p * dim + d] - KCOs[d]) * dKs[d]) ** 2 / (2 * params["Mass"][p])
        else:
            H[i - 1 + PsiSize*(j - 1)] = H[j - 1 + PsiSize*(i - 1)].conj
			
			
		#on-site potential
        
    H = H.reshape([PsiSize,PsiSize])			

    D, eV = LA.eigh(H)

    for i in range(0,5):
        print(D[i])
	


	# Go over eigenstates to construct the init wave function as specified by Input Params
    for J in range(0,5):
        if ((params["InputStates"][2 * J] != 0.0) or (params["InputStates"][2 * J + 1] != 0.0)):
            PsiOut.wave += params["InputStates"][2 * J] * eV[:,J] 

	
					
    if (params["TDSESolver"] == "rSpace"):		# do a final normalisation
        norm = PsiOut.norm()
        PsiOut.wave /= np.sqrt(norm)
    	# Write wave and eigenvalues to file as appropriate
        PsiOut.exportFull( params, -1, "wave")
    elif (params["TDSESolver"] == "kSpace"):
        PsiOut.exportFull( params, -1, "wave")
        PsiOut.wave = np.reshape(PsiOut.wave, PsiOut.size)
        PsiOut.htod()
        PsiOut.fft(0)
        PsiOut.dtoh()
        norm = PsiOut.norm()
        PsiOut.wave /= np.sqrt(norm)
        PsiOut.exportFull( params, -1, "Kwave")


	
    exportEigenvalues(params, D, 5)