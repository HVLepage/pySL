# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import norm
import time
import tracemalloc
import humanize

from src.lindbladian.commutation import *

t0_full = time.time() # Keep track of total simulation time

################################################################################################################################
# Application profiling & GPU hardware configuration
################################################################################################################################

tracemalloc.start() # Memory allocation profile

# gpuinfo.print_gpu_info()

################################################################################################################################
# 
################################################################################################################################


Nx = 21
PsiSize = (Nx)
m = 1

Xa = -40
Xb = 40
dx = (Xb-Xa)/(Nx-1)

hbar = 1

x = np.linspace(Xa,Xb,Nx)

#Init Hamiltonian
H = np.zeros([PsiSize**2], dtype=np.complex64)

# Define total hamiltonian
H=np.reshape(H, [PsiSize,PsiSize])
# Kinetic part of the Hamiltonian
HK=np.reshape(H, [PsiSize,PsiSize])

hbar2 = hbar ** 2 #hbar squared
w = 0.015
V = 0.5 * w**2 * x**2   

hk = hbar2 / (2 * m * dx ** 2)

for k in range(0, PsiSize):# Go over each row
    #fill Hamiltonian, making use of hermiticity to only fill half and do conjugate
 
    HK[k][k] += 2 * hk # on-site kinetic part

    pmone= k -1
    if (pmone>=0 and pmone<Nx):
        HK[k][k-1] -= hk

    pmone= k+1
    if (pmone>=0 and pmone< Nx ):
        HK[k][k+1] -= hk

# Add potential term to the Hamiltonian (can be time dependent)
H = HK + np.diag(V)

D, psi_eig = LA.eigh(H)

# Define creation and annihilation operators
a = np.zeros((PsiSize,PsiSize), dtype=np.complex64)
rng = np.arange(PsiSize-1)
a[rng, rng+1] = np.sqrt(np.diff(D))

ax = psi_eig@a@np.transpose(psi_eig)
adx = np.transpose(ax)





psi0=psi_eig[:,0]
psi1=psi_eig[:,1]

# psi = (1/np.sqrt(2)) *(psi1+psi0)
# psi = psi0

xoffset = 10 # Wave packet distance from center
re = (w/np.pi)**(1/4)*np.exp(-w*(x-xoffset)**2/2) # + (w/np.pi)**(1/4)*np.exp(-w*(x+xoffset)**2/2) # Initial wave function
psi = np.zeros(PsiSize, dtype=np.complex64)
psi += re

rho=np.outer(psi,psi)

rhodot = commutator(H,rho)/(hbar*1j)

T = 2000
dt = 0.01

Nt=T/dt


psi2 = np.zeros((int(Nt), int(PsiSize)))

# Dissipation Lindblad term
T1 = 6
L1 = np.sqrt(1/T1) * ax
L1d = np.transpose(L1)

# Dephasing Lindblad term
T2 = 600
L2 = np.sqrt(1/T2)*(1/(hbar*w))*H
L2d = np.transpose(L2)

# Thermalization Lindblad term
Temperature = 300
kB = 0.1
kBT = kB * Temperature
L3 = np.sqrt(1/T1)*np.exp(- 0.5 * hbar*w/kBT ) * adx
L3d = np.transpose(L3)


for i in range(0,int(Nt)):
    # Adjust time-dependent potential
    # if (i*dt > 500):
    #     V = (0.00005) * x**2
    #     H = HK + np.diag(V)

    rhodot = -1j*commutator(H,rho)/(hbar)
    rhodot += L1@rho@L1d - 0.5 * anticommutator( L1d@L1, rho )
    # rhodot += L2@rho@L2d - 0.5 * anticommutator( L2d@L2, rho )
    rhodot += L3@rho@L3d - 0.5 * anticommutator( L3d@L3, rho )

    rho += rhodot*dt

    psi2[i] = np.diag(rho).real

# print(rhodot)

fig, (ax2, ax3) = plt.subplots(1, 2)
fig.suptitle('Potential step at T=2000')


DMGraph = np.zeros((PsiSize,PsiSize), dtype=np.complex64)
for i in range(PsiSize):
    for j in range(PsiSize):
        DMGraph[i][j] = np.conjugate(psi_eig[:,i])@rho@psi_eig[:,j]

ax2.bar(np.arange(0, PsiSize), np.diag(DMGraph.real))
ax2.set_title("Pure State ocupation")

t = np.linspace(0,T,int(Nt))
ax3.pcolormesh(x, t, psi2)
ax3.set_title("Full State Evolution")
ax3.set_xlabel('x (nm)')
ax3.set_ylabel('Time (ps)')

# fig.savefig("test.png")
plt.show()


################################################################################################################################
# Trace Memory Management
################################################################################################################################
t_print = time.time()-t0_full
print("\nTotal time elapsed = ",t_print)


snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
total = sum(stat.size for stat in top_stats)

print("\n----------------------------------------------------------------")
print("Total Memory Usage: %s"% humanize.naturalsize(total))
print("----------------------------------------------------------------")

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)