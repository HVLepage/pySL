# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.linalg import expm


def commutator (A,B):
    return A@B -B@A


Nx = 101
PsiSize = (Nx)
m = 1

Xa = -100
Xb = 100
dx = (Xb-Xa)/(Nx-1)

hbar = 1

x=np.linspace(Xa,Xb,Nx)
	#generate Hamiltonian matrix for eigenfunction solving
	
	#Init Hamiltonian
H = np.zeros([PsiSize**2], dtype=np.complex64)

# Define total hamiltonian
H=np.reshape(H, [PsiSize,PsiSize])
# Kinetic part of the Hamiltonian
HK=np.reshape(H, [PsiSize,PsiSize])

H2 = hbar ** 2 #hbar squared
V=0.00005*x**2
   
    
for k in range(0, PsiSize):# Go over each row
    #fill Hamiltonian, making use of hermiticity to only fill half and do conjugate

    #on-site potential
    # H[k][k] += V[k] 
 
    HK[k][k] += H2 / (m * dx ** 2) # on-site kinetic part
                
    pmone= k -1
    if (pmone>=0 and pmone<Nx):
        HK[k][k-1] -= H2 / (2 * m * dx ** 2)
                    
    pmone= k+1
               
                
    if (pmone>=0 and pmone< Nx ):
        HK[k][k+1] -= H2 / (2 * m * dx ** 2)
                    
# Add potential term to the Hamiltonian (can be time dependent)
H = HK + np.diag(V)


D, eV = LA.eigh(H)

psi0=eV[:,0]
psi1=eV[:,1]

psi = (1/np.sqrt(2)) *(psi1+psi0)
# psi = psi0

rho=np.outer(psi,psi)

rhodot= commutator(H,rho)/(hbar*1j)

T=200
dt=0.01

Nt=T/dt

#rhofinal=expm(-1j *H *T/hbar)@rho@expm(1j *H *T/hbar)


psi00 = np.zeros(int(Nt), dtype=np.complex64)
psi01 = np.zeros(int(Nt), dtype=np.complex64)
psi10 = np.zeros(int(Nt), dtype=np.complex64)
psi11 = np.zeros(int(Nt), dtype=np.complex64)


# print(HK.shape)
# print(H.shape)
# print(V.shape)


for i in range(0,int(Nt)):
    # Adjust time-dependent potential
    if (i>2000):
        V = (0.0007) * x**2
    
    H = HK + np.diag(V)

    rho+=rhodot*dt
    rhodot= -1j*commutator(H,rho)/(hbar)
    
    psi00[i] = np.conjugate(psi0)@rho@psi0
    psi01[i] = np.conjugate(psi0)@rho@psi1
    psi10[i] = np.conjugate(psi1)@rho@psi0
    psi11[i] = np.conjugate(psi1)@rho@psi1


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Potential step at T=2000')

ax1.plot(range(int(Nt)), psi00.real, label = "psi00")
ax1.plot(range(int(Nt)), psi01.real, label = "psi01")
ax1.plot(range(int(Nt)), psi10.real, label = "psi10")
ax1.plot(range(int(Nt)), psi11.real, label = "psi11")

ax1.set_xlabel('Time')
ax1.set_ylabel('Psi')
ax1.set_ylim(-1, 1)
ax1.set_title("Pure State Evolution")
ax1.legend()
ax1.grid()


DMGraph = np.zeros((PsiSize,PsiSize), dtype=np.complex64)
for i in range(PsiSize):
    for j in range(PsiSize):
        DMGraph[i][j] = np.conjugate(eV[:,i])@rho@eV[:,j]

ax2.bar(np.arange(0, PsiSize)-.5, np.diag(DMGraph.real))
ax2.set_title("Pure State ocupation")

ax3.pcolormesh(x, x, DMGraph.real)
ax3.set_title("Full State ocupation")

# fig.savefig("test.png")
plt.show()

# psifinal00 = np.conjugate(psi0)@rho@psi0
# psifinal10 = np.conjugate(psi1)@rho@psi0
# psifinal01 = np.conjugate(psi0)@rho@psi1
# psifinal11 = np.conjugate(psi1)@rho@psi1