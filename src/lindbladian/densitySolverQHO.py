# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time
import tracemalloc
import humanize


t0_full = time.time() # Keep track of total simulation time

################################################################################################################################
# Application profiling & GPU hardware configuration
################################################################################################################################

tracemalloc.start() # Memory allocation profile

# gpuinfo.print_gpu_info()

################################################################################################################################
# 
################################################################################################################################

def commutator (A,B):
    return A@B - B@A

def anticommutator (A,B):
    return A@B + B@A

Nx = 51
PsiSize = (Nx)
m = 1

Xa = -25
Xb = 25
dx = (Xb-Xa)/(Nx-1)

hbar = 1

x=np.linspace(Xa,Xb,Nx)

#generate Hamiltonian matrix for eigenfunction solving

# Define ladder operators (This should change when not dealing with a QHO)
a = np.zeros((PsiSize,PsiSize))
rng = np.arange(PsiSize-1)
a[rng, rng+1] = np.sqrt(rng+1)
ad = np.transpose(a)

#Init Hamiltonian
H = np.zeros([PsiSize**2], dtype=np.complex64)

# Define total hamiltonian
H=np.reshape(H, [PsiSize,PsiSize])
# Kinetic part of the Hamiltonian
HK=np.reshape(H, [PsiSize,PsiSize])

hbar2 = hbar ** 2 #hbar squared
w = 0.015
V = w**2 * x**2   

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

psi0=psi_eig[:,0]
psi1=psi_eig[:,1]

psi = (1/np.sqrt(2)) *(psi1+psi0)
# psi = psi0

rho=np.outer(psi,psi)

rhodot = commutator(H,rho)/(hbar*1j)

T = 2000
dt = 0.01

Nt=T/dt

#rhofinal=expm(-1j *H *T/hbar)@rho@expm(1j *H *T/hbar)


psi00 = np.zeros(int(Nt), dtype=np.complex64)
psi01 = np.zeros(int(Nt), dtype=np.complex64)
psi10 = np.zeros(int(Nt), dtype=np.complex64)
psi11 = np.zeros(int(Nt), dtype=np.complex64)

psi2 = np.zeros((int(Nt), int(PsiSize)))

# print(HK.shape)
# print(H.shape)
# print(V.shape)

# Dissipation Lindblad term
T1 = 600
L1 = np.sqrt(1/T1) * a
L1d = np.transpose(L1)

# Dephasing Lindblad term
T2 = 10
L2 = 0
L2d = np.transpose(L2)

# Thermalization Lindblad term
Temperature = 300
kBT = 0
L3 = 0
L3d = np.transpose(L3)


for i in range(0,int(Nt)):
    # Adjust time-dependent potential
    # if (i*dt > 500):
    #     V = (0.00005) * x**2
    #     H = HK + np.diag(V)

    rhodot = -1j*commutator(H,rho)/(hbar)
    rhodot += L1@rho@L1d - 0.5 * anticommutator( L1d@L1, rho )

    rho += rhodot*dt

    # psi00[i] = np.conjugate(psi0)@rho@psi0
    # psi01[i] = np.conjugate(psi0)@rho@psi1
    # psi10[i] = np.conjugate(psi1)@rho@psi0
    # psi11[i] = np.conjugate(psi1)@rho@psi1

    psi2[i] = np.diag(rho).real

# print(rhodot)

fig, (ax2, ax3) = plt.subplots(1, 2)
fig.suptitle('Potential step at T=2000')

# ax1.plot(range(int(Nt)), psi00.real, label = "psi00")
# ax1.plot(range(int(Nt)), psi01.real, label = "psi01")
# ax1.plot(range(int(Nt)), psi10.real, label = "psi10")
# ax1.plot(range(int(Nt)), psi11.real, label = "psi11")

# ax1.set_xlabel('Time')
# ax1.set_ylabel('Psi')
# ax1.set_ylim(-1, 1)
# ax1.set_title("Pure State Evolution")
# ax1.legend()
# ax1.grid()


DMGraph = np.zeros((PsiSize,PsiSize), dtype=np.complex64)
for i in range(PsiSize):
    for j in range(PsiSize):
        DMGraph[i][j] = np.conjugate(psi_eig[:,i])@rho@psi_eig[:,j]

ax2.bar(np.arange(0, PsiSize)-.5, np.diag(DMGraph.real))
ax2.set_title("Pure State ocupation")

t = np.linspace(0,T,int(Nt))
ax3.pcolormesh(x, t, psi2)
ax3.set_title("Full State Evolution")
ax3.set_xlabel('x (nm)')
ax3.set_ylabel('Time (ps)')

# fig.savefig("test.png")
plt.show()

# psifinal00 = np.conjugate(psi0)@rho@psi0
# psifinal10 = np.conjugate(psi1)@rho@psi0
# psifinal01 = np.conjugate(psi0)@rho@psi1
# psifinal11 = np.conjugate(psi1)@rho@psi1



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