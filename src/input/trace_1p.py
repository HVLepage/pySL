import numpy as np
import src.index.index1DtoND as index1DtoND

#trace over just 1 particle
def trace_1p (params,wave_real,wave_im):

    
    #psi=np.zeros(psiShape,dtype=np.complex64)
    psi=np.zeros(params["N"],dtype=np.complex64)
    psi=wave_real+1j*wave_im

    # for i in range(params["N"]):
    #     inds=np.zeros(params["Np"],dtype=int)
    #     index1DtoND.index(i,inds,params["Sizes"],params)
    #     #psiIndex=()
    #     # for i in range(params["Np"]):
    #     #     psiIndex+=(inds[i],)
    
    #     psi[tuple(inds)]=wave_real[i] + 1j*wave_im[i]    #PsiOut.wave[i]
    psi=np.reshape(psi,np.tile(params["Sizes"],params["Np"])) #reshape psi for easy summing
    sumxs=np.zeros([params["Np"],params["Nx"]**2]) #will hold traces for each particle
   
    for i in range(params["Np"]):
     
        sumxs[i]=np.ndarray.flatten(np.sum(np.abs(psi)**2,axis=i))#e.g. 0th element is sum over x1 etc., flatten here so can be saved easily, reshape later
        
    return sumxs    
        
        
        
