from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from src.input.parameters import getparams
import src.index.index1DtoND as index1DtoND
from src.input.trace1D import trace1D
import pycuda.driver as cuda
import time
import math

def npfrom1p(params,PsiOut,mod):
    SimName = params['SimName']

    # Read simulation files
    FilePath = './out/' + SimName + '/npfrom1p/'
    onepwaves = [f for f in listdir(FilePath) if isfile(join(FilePath, f))] #all the files in folder (wave at each print step)
    onepwaves.sort()
    
    
    npfrom1p_gpu = mod.get_function("npfrom1p_gpu")
    blockSize = 256 
            
    numBlocks = math.ceil(params["N"] / blockSize) #math.ceil returns type int, np.ceil returns np.float64

    onepwaves_arr=np.zeros([params["Np"],params["Nx"],2],dtype=np.float32)
    
    for p in range(params["Np"]):
        onepwaves_arr[p]=np.loadtxt(fname = FilePath + onepwaves[p],dtype=np.float32)
    
    wave_real=cuda.mem_alloc_like(onepwaves_arr[0,:,0]) #any particle
    wave_im=cuda.mem_alloc_like(onepwaves_arr[0,:,1])

    Sizes = cuda.mem_alloc_like(params["Sizes"])
    cuda.memcpy_htod(Sizes, params["Sizes"])

    for p in range(params["Np"]):  
        
        cuda.memcpy_htod(wave_real,onepwaves_arr[p,:,0].ravel())
        cuda.memcpy_htod(wave_im,onepwaves_arr[p,:,1].ravel())

        npfrom1p_gpu(np.int32(params["N"]),PsiOut.re_gpu,PsiOut.im_gpu,wave_real,wave_im,np.int32(params["Np"]),np.int32(p),Sizes,np.int32(params["dim"]),block=(blockSize,1,1), grid=(numBlocks,1))
        
    tempre = np.zeros(params["N"], dtype=np.float32)
    tempim = np.zeros(params["N"], dtype=np.float32)
    cuda.memcpy_dtoh(tempre,PsiOut.re_gpu)
    cuda.memcpy_dtoh(tempim,PsiOut.im_gpu)   
    
    

    PsiOut.wave=tempre+1j*tempim

    # for p in range(params["Np"]): 
    #     for i in range(params["N"]):
    #         inds=np.zeros(params["Np"],dtype=int) #if doing in higher dimensions have to convert 
    #         index1DtoND.index(i,inds,params["Sizes"],params)
         
            
    #         if p==0:
    #             PsiOut.wave[i]=onepwaves_arr[p,inds[p],0]+1j*onepwaves_arr[p,inds[p],1]
    #         else:
    #              PsiOut.wave[i]*=onepwaves_arr[p,inds[p],0]+1j*onepwaves_arr[p,inds[p],1]


    

    norm=PsiOut.norm()
    PsiOut.wave/=norm**0.5
    sumxs=trace1D(params,PsiOut.wave.real,PsiOut.wave.imag)
    # PsiOut.wave.real[abs(PsiOut.wave.real) < 1e-11] = 0.0
    # PsiOut.wave.imag[abs(PsiOut.wave.imag) < 1e-11] = 0.0

    x=np.linspace(params["Xa"],params["Xb"],params["Nx"])
    plt.figure()
    for i in range(params["Np"]):
        plt.plot(x,sumxs[i])
       
    plt.xlabel("x/nm")
    plt.ylabel("Traced over mod^2 for each particle")
    plt.title("Traced over mod^2 for each particle as a function of position")
    PsiOut.exportFull( params, -1, "wave_full")
    plt.show()

    
