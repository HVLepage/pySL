import numpy as np
import src.index.index1DtoND as index1DtoND

#reshape wave from 1D to NpD, then sum over all particles but 1
def trace1D (params,wave_real,wave_im):

    
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
    sumxs=np.zeros([params["Np"],params["Nx"]]) #willh hold traces for each particle
    psi_temp=psi
    for i in range(params["Np"]):
        temp_indices=np.arange(params["Np"])
        temp_indices[i]=0
        temp_indices[0]=i
        psi_temp=np.transpose(psi,axes=temp_indices) #permute axes of psi so can easily sum over the other particles
        
        
        sumxs[i]=np.sum(np.reshape(np.abs(psi_temp)**2,[params["Nx"],params["Nx"]**(params["Np"]-1)]),axis=1)
        
    return sumxs    
        
        
        
