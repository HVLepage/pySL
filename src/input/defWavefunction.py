import numpy as np
import matplotlib.pyplot as plt
import src.index.index1DtoND as index1DtoND
import src.index.indexNDto1D as indexNDto1D
from src.input.trace1D import trace1D
import time

def defWavefunction(params, PsiOut):

    Xa = params["Xa"]
    Xb = params["Xb"]
    Ya = params["Ya"]
    Yb = params["Yb"]
    Za = params["Za"]
    Zb = params["Zb"]
    Nx = params["Nx"]
    Ny = params["Ny"]
    Nz = params["Nz"]


    Np = int(params["Np"])
    dim = int(params["dim"])


    x = np.linspace(Xa, Xb, Nx)
    x2=np.linspace(Xa, Xb, Nx)
    x3=np.linspace(Xa, Xb, Nx)
    y = np.linspace(Ya, Yb, Ny)
    z = np.linspace(Za, Zb, Nz)
        
    if params["wavefunctionType"]=="Gaussian":

        if params["dim"]==1 and params["Np"]==1:
            PsiOut.wave=np.complex64(np.exp(-(x-params["muxG"])**2/(2*params["sigmaxG"]**2))*np.exp(1j*params["kGx"]*x)) #128 bit otherwise
            
        if params["dim"]==2 and params["Np"]==1:
            for i in range(params["N"]):
                inds=np.zeros(2)
                index1DtoND.index(i,inds,params["Sizes"],params)
                PsiOut.wave[i]=np.complex64(np.exp(-(x[int(inds[0])]-params["muxG"])**2/(2*params["sigmaxG"]**2))* np.exp(-(y[int(inds[1])]-params["muyG"])**2/(2*params["sigmayG"]**2))*np.exp(1j*(params["kGx"]*x[int(inds[0])]+params["kGy"]*y[int(inds[1])])))
        
            # wavefunc2D=np.zeros([Ny,Nx])
            # for i in range(Nx*Ny):
            #     inds=np.zeros(2)
            #     index1DtoND.index(i,inds,params["Sizes"],params)
            #     wavefunc2D[int(inds[1])][int(inds[0])]=np.abs(PsiOut.wave[i])**2
            # fig = plt.figure()
            # ax = plt.axes(xlim=(Xa,Xb), ylim=(Ya,Yb))
            # ax.contourf(x,y,wavefunc2D,levels=100,cmap='Greys')  
            # plt.show()


        if params["dim"]==1 and params["Np"]==2:
            for i in range(params["N"]):
                inds=np.zeros(2,dtype=int)
                index1DtoND.index(i,inds,params["Sizes"],params)
                PsiOut.wave[i]=np.complex64( (np.exp(-(x[inds[0]]-params["muxG"])**2/(2*params["sigmaxG"]**2)) *np.exp(1j*params["kGx"]*x[inds[0]])) *(np.exp(-(x2[inds[1]]-params["mux2G"])**2/(2*params["sigmax2G"]**2)) *np.exp(1j*params["kGx2"]*x2[inds[1]]))) 

            
            norm=PsiOut.norm()
            
            PsiOut.wave/=norm**0.5


            sumxs=trace1D(params,PsiOut.wave.real,PsiOut.wave.imag)
            
            # psi=np.zeros([params["Nx"],params["Nx"]],dtype=np.complex64) #psi at all x1 sites for a given x2, x1 vals along a particular row
            
            # #print(PsiOut.norm())


            # for i in range(params["N"]):
            #     inds=np.zeros(2,dtype=int)
            #     index1DtoND.index(i,inds,params["Sizes"],params)
            #     psi[inds[1]][inds[0]]=PsiOut.wave[i]
            # sumx1=np.sum(np.abs(psi)**2,axis=1)
            # sumx2=np.sum(np.abs(psi)**2,axis=0)
            plt.figure()
            plt.plot(x2,sumxs[1]) #summed over x1 for each x2
            plt.plot(x,sumxs[0])
            plt.xlabel("x/nm")
            plt.ylabel("Traced over mod^2 for each particle")
            plt.title("Traced over mod^2 for each particle as a function of position")
            #plt.show()

            wavefunc2D=np.zeros([Nx,Nx],dtype=np.complex64)
            for i in range(Nx*Nx):
                inds=np.zeros(2)
                index1DtoND.index(i,inds,params["Sizes"],params)
                wavefunc2D[int(inds[1])][int(inds[0])]= PsiOut.wave[i]    #np.abs(PsiOut.wave[i])**2
            plt.figure()
            #ax = plt.axes(xlim=(Xa,Xb), ylim=(Xa,Xb))
            plt.contourf(x,x2,np.abs(wavefunc2D)**2,levels=100,cmap='Greys')  
            plt.xlabel("x1/nm")
            plt.ylabel("x2/nm")
            plt.title("Mod^2 wavefunction")
            plt.show()

        if params["dim"]==1 and params["Np"]==3:    
            #can set this generally
            for i in range(params["N"]):
                inds=np.zeros(3,dtype=int)
                index1DtoND.index(i,inds,params["Sizes"],params)
                PsiOut.wave[i]=np.complex64( (np.exp(-(x[inds[0]]-params["muxG"])**2/(2*params["sigmaxG"]**2)) *np.exp(1j*params["kGx"]*x[inds[0]])) *(np.exp(-(x2[inds[1]]-params["mux2G"])**2/(2*params["sigmax2G"]**2)) *np.exp(1j*params["kGx2"]*x2[inds[1]])) * (np.exp(-(x[inds[2]]-params["mux3G"])**2/(2*params["sigmax3G"]**2)) *np.exp(1j*params["kGx3"]*x[inds[2]]))    )
            
            
            norm=PsiOut.norm()
            
            PsiOut.wave/=norm**0.5
            t0=time.time()
            sumxs=trace1D(params,PsiOut.wave.real,PsiOut.wave.imag)
            print(time.time()-t0)
            plt.figure()
            plt.plot(x3,sumxs[2]) #summed over x1 for each x2
            plt.plot(x2,sumxs[1]) #summed over x1 for each x2
            plt.plot(x,sumxs[0])
            plt.xlabel("x/nm")
            plt.ylabel("Traced over mod^2 for each particle")
            plt.title("Traced over mod^2 for each particle as a function of position")
            plt.show()

    

    norm = PsiOut.norm()
    PsiOut.wave /= np.sqrt(norm)
    #plt.plot(x,PsiOut.wave.real**2+PsiOut.wave.imag**2)
    #plt.show()
    #print((PsiOut.wave))
    