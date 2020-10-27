import numpy as np
import ctypes
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import os

def RSpaceSolverGPU(params, PsiOut, V): #Real space solver GPU
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
    H=H.ravel()
    dH = gpuarray.to_gpu(H)
    dH.set(H)
    D=np.zeros(PsiSize, dtype=np.float32)
    dD = gpuarray.to_gpu(D)
    

    if os.name == 'nt':
        _libcusolver = ctypes.cdll.LoadLibrary('cusolver64_11.dll')
    else:
        _libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')

    #_libcusolver = ctypes.cdll.LoadLibrary('libcusolver.dll')

    _libcusolver.cusolverDnCreate.restype = int
    _libcusolver.cusolverDnCreate.argtypes = [ctypes.c_void_p]

    _libcusolver.cusolverDnDestroy.restype = int
    _libcusolver.cusolverDnDestroy.argtypes = [ctypes.c_void_p]


    _libcusolver.cusolverDnSsyevd.restype = int
    _libcusolver.cusolverDnSsyevd.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_float,
                                            ctypes.c_int,
                                            ctypes.c_float,
                                            ctypes.c_double,
                                            ctypes.c_int,
                                            ctypes.c_void_p]


    n = ctypes.c_int(PsiSize)
    lda = ctypes.c_int(PsiSize)
    jobz = ctypes.c_int(1)
    uplo = ctypes.c_int(1)
    
    #dwork = ctypes.double()
    lwork = ctypes.c_int(0)
    devinfo = ctypes.c_void_p()
    
    
    #create cusolver handle
    _cuso_handle = ctypes.c_void_p()
    status = _libcusolver.cusolverDnCreate(ctypes.byref(_cuso_handle))
    print('status: ' + str(status))
    cuso_handle = _cuso_handle.value
    
    #create MatDescriptor
    #status = _libcusolver.cusolverDnCreateMatDescr(ctypes.byref(descrA))
    #print('status: ' + str(status))
    
    print('cuso handle: ' + str(cuso_handle))
    
 
    
    _libcusolver.cusolverDnSsyevd_bufferSize.restype = int
    _libcusolver.cusolverDnSsyevd_bufferSize.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_float,
                                            ctypes.c_int,
                                            ctypes.c_float,
                                            ctypes.c_int]
    
    _libcusolver.cusolverDnSsyevd_bufferSize(cuso_handle,
                                 jobz,
                                 uplo,
                                 n,
                                 (dH.gpudata),
                                 lda,
                                 (dD.gpudata),
                                lwork)
    
    dwork=gpuarray.to_gpu(np.zeros(lwork, dtype=np.float64))
    
                                            
    
    _libcusolver.cusolverDnSsyevd(cuso_handle,
                                 jobz,
                                 uplo,
                                 n,
                                 (dH.gpudata),
                                 lda,
                                  (dD.gpudata),
                                 dwork.gpudata,
                                 lwork,
                                 devinfo)
                                 

     #destroy handles
    status = _libcusolver.cusolverDnDestroy(cuso_handle)
    print('status: ' + str(status))

    #

    dD.get(D)
    dH.get(H)
    H=H.reshape([PsiSize,PsiSize])

    for i in range(0,5):
        print(D[i])
	


	# Go over eigenstates to construct the init wave function as specified by Input Params
    for J in range(0,5):
        if ((params["InputStates"][2 * J] != 0.0) or (params["InputStates"][2 * J + 1] != 0.0)):
            PsiOut.wave += params["InputStates"][2 * J] * H[:,J] 

	
							# do a final normalisation
    norm = PsiOut.norm()

    PsiOut.wave /= norm
    
		


	# Write wave and eigenvalues to file as appropriate
    #if (KSpaceLeapfrog):
	#	ExportFullWave(PsiOut, params, -1, PsiSize, params["SimName, "wave")
	#	FFTAuxPsi(params, PsiOut, planConvND, true, false)
	#	ExportFullWave(PsiOut, params, -1, PsiSize, params["SimName, "Kwave")
	#else:
    PsiOut.exportFull( params, -1, "wave")
	
    ExportEigenvalues(params, D, 5)

    print("EigenSolver finished!  \n")

def ExportEigenvalues(params, D, count):
    OutputFileName = ".\\" + params["SimName"] + "eigs.txt"
    with open(OutputFileName, 'w+') as OutputFile:
        for i in range(0, count):
            OutputFile.write("{:e} \n".format(D[i]))