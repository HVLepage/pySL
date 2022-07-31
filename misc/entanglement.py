import numpy as np
import src.index.index1DtoND as index1DtoND
from src.input.parameters import getparams
from src.input.antisymmetrise import antisymmetrise
#trace over just 1 particle
def entanglement(params,wave_real,wave_im):

    
    #psi=np.zeros(psiShape,dtype=np.complex64)
    psi=np.zeros(params["N"],dtype=np.complex64)
    psi=wave_real+1j*wave_im
   

    
    #     psi[tuple(inds)]=wave_real[i] + 1j*wave_im[i]    #PsiOut.wave[i]
    psi=np.reshape(psi,np.tile(params["Sizes"],params["Np"])) #reshape psi for easy summing
    sumxs=np.zeros([params["Np"],params["Nx"]**2]) #will hold traces for each particle

    rho_1=np.zeros([params["Nx"],params["Nx"]])
    rho_2=np.zeros([params["Nx"],params["Nx"]])
    rho_3=np.zeros([params["Nx"],params["Nx"]])

    # rho_1=np.einsum('ilm,jlm->ij',psi,np.conjugate(psi))
    # rho_2=np.einsum('lim,ljm->ij',psi,np.conjugate(psi))
    # rho_3=np.einsum('lmi,lmj->ij',psi,np.conjugate(psi))

    rho_1=np.einsum('ilm,jlm->ij',psi,np.conjugate(psi))
    rho_2=np.einsum('lim,ljm->ij',psi,np.conjugate(psi))
    rho_3=np.einsum('lmi,lmj->ij',psi,np.conjugate(psi))
    
    eigs_1=np.linalg.eig(rho_1)[0]
    eigs_2=np.linalg.eig(rho_2)[0]
    eigs_3=np.linalg.eig(rho_3)[0]

    s1=-np.sum(np.multiply(eigs_1,np.log2(eigs_1)))
    s2=-np.sum(np.multiply(eigs_2,np.log2(eigs_2)))
    s3=-np.sum(np.multiply(eigs_3,np.log2(eigs_3)))

    print(f"s1={s1},s2={s2},s3={s3}")

    
        
        
#wave=np.loadtxt('wave_initial.txt')       
params=getparams()  
#entanglement(params,wave[:,0],wave[:,1])


x=np.linspace(-100,100,201)
wave=np.ones([201,201,201])
wave/=np.sum(np.abs(wave)**2)**0.5
entanglement(params,wave.real,wave.imag)