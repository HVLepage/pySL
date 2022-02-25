################################################################################################################################
# Real space eigensolver using the GPU
################################################################################################################################


import numpy as np
import ctypes
import pycuda.gpuarray as gpuarray
import os
from skcuda import linalg
from src.export.exportEigenvalues import exportEigenvalues
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D
import matplotlib.pyplot as plt

def rSpaceSolverGPU(params, PsiOut, V, thr): #Real space solver GPU
	#read input parameters from params & declare stuff

	
    Nx = params["Nx"]
    Ny = params["Ny"]
    Nz = params["Nz"]
     
    Np = params["Np"]
    dim = params["dim"]
    PsiSize = (Nx * Ny * Nz) ** Np
	
	
	#generate Hamiltonian matrix for eigenfunction solving
	
	#Init Hamiltonian
    H = np.zeros([PsiSize**2], dtype=np.float32)
    
    H2 = params["hbar"] ** 2 #hbar squared
    inds = np.zeros(dim*Np)
    
    for k in range(0, PsiSize):# Go over each row
        ind = k + k*PsiSize
        indPMone=0
        index1DtoND.index(k,inds,params["Sizes"],params)

		#fill Hamiltonian, making use of hermiticity to only fill half and do conjugate (hasn't been done yet)

		#on-site potential
        H[ind] += V.pot[k] #!!!!!!!!!!!!!!!!!!!!!!! #add potential to diagonal terms
        for d in range(0,dim):
            for p in range(0,Np):
                H[ind] += H2 / (params["Mass"][p] * params["dXs"][d] ** 2) # on-site kinetic part, need to add for each dim and particle
                
                inds[p*dim+d]-=1 #for a given k and for pth particle and dth dimension
                indPMone = indexNDto1D.index(inds,params["Sizes"],params)
                if (inds[p*dim+d]>=0 and inds[p*dim+d]< params["Sizes"][d] ): #since would otherwise be -1 for 0 elements, 2nd term not needed here
                    H[int(k*PsiSize + indPMone)] -= H2 / (2 * params["Mass"][p] * params["dXs"][d] ** 2)
                    
                inds[p*dim+d]+=2 #recover the -1 already taken off
                indPMone = indexNDto1D.index(inds,params["Sizes"],params)
                
                if (inds[p*dim+d]>=0 and inds[p*dim+d]< params["Sizes"][d] and (k+1)<PsiSize): #1st term not needed here, for very last term don't have 
                    #a value on the "right" (not needed though as 2nd term prevents)
                    H[int(k*PsiSize + indPMone)] -=  H2 / (2 * params["Mass"][p] * params["dXs"][d] ** 2)
                    
                inds[p*dim+d]-=1 #back to original value

   #Do eig things here
    H=np.reshape(H.ravel(), [PsiSize,PsiSize], order='F') #convert to square matrix, Fortran order, down columns
    H_gpu = gpuarray.to_gpu(H) #allocates memory and copies presumably
    D=np.zeros(PsiSize, dtype=np.float32)
    
    linalg.init()
    V_gpu, D_gpu = linalg.eig(H_gpu, 'N', 'V') #1st argument is input matrix, N in pos 2 means left eigenvectors aren't computed
    #V means right ones are, returns eigenvectors and eigenvalues, normalised


    D_gpu.get(D) #copies to host presumably, not necessarily ordered according to docs but seems to be here
    V_gpu.get(H) #ith eigenvector corresponding to ith eigenvalue down ith column, copied to H (since it's the same size?)
    #H=H.reshape([PsiSize,PsiSize])
    #H=np.reshape(H.ravel(), [PsiSize,PsiSize], order='C') #eigenvectors down columns still, seems redundant?

    for i in range(0,5): #first 5 eigenvalues
        print(D[i])
	


	# Go over eigenstates to construct the init wave function as specified by Input Params
    for J in range(0,5):
        if ((params["InputStates"][2 * J] != 0.0) or (params["InputStates"][2 * J + 1] != 0.0)): #real then imag
            if (params["TDSESolver"] == "kSpace"):
                PsiOut.wave += (params["InputStates"][2 * J] + 1j * params["InputStates"][2 * J + 1]) * np.reshape(H[:,J], PsiOut.size)
            else:
                PsiOut.wave += (params["InputStates"][2 * J] + 1j * params["InputStates"][2 * J + 1]) *  H[:,J] #complex weighted superposition of eigenstates
            
    #PsiOut.wave[0]=0
    #PsiOut.wave[params["N"]-1]=0
	
						
    if (params["TDSESolver"] == "rSpace"):		# do a final normalisation
        norm = PsiOut.norm()
        #print(norm)
        PsiOut.wave /= np.sqrt(norm)

        if params["Np"]==2 and params["dim"]==1:
            x1 = np.linspace(float(params["Xa"]), float(params["Xb"]), int(params["Nx"]))
            x2 = np.linspace(float(params["Xa"]), float(params["Xb"]), int(params["Nx"]))

            wavefunc2D=np.zeros([Nx,Nx])
            for i in range(Nx*Nx):
                inds=np.zeros(2)
                index1DtoND.index(i,inds,params["Sizes"],params)
                wavefunc2D[int(inds[1])][int(inds[0])]=np.abs(PsiOut.wave[i])**2
            fig = plt.figure()
            ax = plt.axes(xlim=(params["Xa"],params["Xb"]), ylim=(params["Ya"],params["Yb"]))
            ax.contourf(x1,x2,wavefunc2D,levels=100,cmap='Greys')  
            plt.show()
            




        #norm = PsiOut.norm()
        #print(norm)
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
