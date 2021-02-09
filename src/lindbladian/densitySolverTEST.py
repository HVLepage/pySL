# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.linalg import expm


def commutator (A,B):
    return A@B -B@A


Nx = 51
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
    
H=np.reshape(H, [PsiSize,PsiSize])

H2 = hbar ** 2 #hbar squared
V=0.00005*x**2
   
    
for k in range(0, PsiSize):# Go over each row
    

		#fill Hamiltonian, making use of hermiticity to only fill half and do conjugate

		#on-site potential
    H[k][k] += V[k] 
 
    H[k][k] += H2 / (m * dx ** 2) # on-site kinetic part
                
    pmone= k -1
    if (pmone>=0 and pmone<Nx):
        H[k][k-1] -= H2 / (2 * m * dx ** 2)
                    
    pmone= k+1
               
                
    if (pmone>=0 and pmone< Nx ):
        H[k][k+1] -= H2 / (2 * m * dx ** 2)
                    

				

D, eV = LA.eigh(H)

psi0=eV[:,0]
psi1=eV[:,1]

psi = (1/np.sqrt(2)) *(psi1+psi0)

rho=np.outer(psi,psi)

rhodot= commutator(H,rho)/(hbar*1j)

T=100
dt=0.05

Nt=T/dt

#rhofinal=expm(-1j *H *T/hbar)@rho@expm(1j *H *T/hbar)

for i in range(0,int(Nt)):
    rho+=rhodot*dt
    rhodot= -1j*commutator(H,rho)/(hbar)
    
rhofinal=rho

psifinal00 = np.conjugate(psi0)@rhofinal@psi0
psifinal10 = np.conjugate(psi1)@rhofinal@psi0
psifinal01 = np.conjugate(psi0)@rhofinal@psi1
psifinal11 = np.conjugate(psi1)@rhofinal@psi1


