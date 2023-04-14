#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import expm, logm
from scipy.stats import norm
import time
from numba import *


# In[25]:


#All the functions, i have many fairly degenerate function but it makes it easier and quicker to have  many adv_timestep 
#for each wanted combination of lindblad opps, due to numbanot liking certain types, and only suing needed linblads

@njit(nogil = True)
def commutator (A,B):
    return A@B - B@A

@njit(nogil = True)
def anticommutator (A,B):
    return A@B + B@A

@jit(forceobj=True)
def get_ladderops(H):
  """Gets anhilations and creation opperators from a hamiltonian"""
  D, psi_eig = LA.eigh(H)
  PsiSize = int(H[0].size)
  # Define creation and annihilation operators
  a = np.zeros((PsiSize,PsiSize), dtype=np.complex128)
  rng = np.arange(PsiSize-1)
  a[rng, rng+1] = np.sqrt(np.diff(D))

  ax = psi_eig@a@np.transpose(psi_eig)
  adx = np.transpose(ax)
  return ax, adx

@njit()
def get_Lindblad(H,ax,adx,w):
    """Returns phenomonoligcal linblad opperators from Hamiltonian, anhilation, creation opperators
    as well as a charrecterisitc energy scale w, w is from harmonic oscialtor, but used as energy scale in L" and L3 forms"""
    # Dissipation Lindblad term
    T1 = 200
    L1 = np.sqrt(1/T1) * ax
    #L1d = np.transpose(L1)

    # Dephasing Lindblad term
    T2 = 600
    L2 = np.sqrt(1/T2)*(1/(w))*H
    #L2d = np.transpose(L2)

    # Thermalization Lindblad term
    Temperature = 300
    kB = 0.001 #[Kb] = Length^2 * mass * time^-2 * K^-1
    kBT = kB * Temperature
    L3 = np.sqrt(1/T1)*np.exp(- (0.5 * w**2)/kBT ) * adx  #can addapt boltzman factor here to better suit the specific potetnial
    #L3d = np.transpose(L3)
    return L1, L2, L3

@njit()
def get_Lindblad_T(H,ax,adx,w, T1, T2):
    """Returns phenomonoligcal linblad opperators from Hamiltonian, anhilation, creation opperators
    as well as a charrecterisitc energy scale w, w is from harmonic oscialtor, but used as energy scale in L" and L3 forms"""
    # Dissipation Lindblad term
    T1 = T1
    L1 = np.sqrt(1/T1) * ax
    #L1d = np.transpose(L1)

    # Dephasing Lindblad term
    T2 = T2
    L2 = np.sqrt(1/T2)*(1/(w))*H
    #L2d = np.transpose(L2)

    # Thermalization Lindblad term
    Temperature = 300
    kB = 0.001 #[Kb] = Length^2 * mass * time^-2 * K^-1
    kBT = kB * Temperature
    L3 = np.sqrt(1/T1)*np.exp(- (0.5 * w**2)/kBT ) * adx  #can addapt boltzman factor here to better suit the specific potetnial
    #L3d = np.transpose(L3)
    return L1, L2, L3

#Forward euler time step methods for varying combinations of lindblad opps.
#Rough params Nx=81 dt = 0.001, V = 0.5 * 0.015**2 *x**2
@njit(nogil = True)
def adv_timestep(H,rho):
  rhodot = -1j*commutator(H,rho)
  return rhodot

@njit(nogil = True)
def adv_timestep_L1(H,rho,L1):
  rhodot = -1j*commutator(H,rho) 
  rhodot += L1@rho@np.transpose(L1) - 0.5 * anticommutator( np.transpose(L1)@L1, rho )
  return rhodot

@njit(nogil = True)
def adv_timestep_L2(H,rho,L2):
  rhodot = -1j*commutator(H,rho)
  rhodot += L2@rho@np.transpose(L2) - 0.5 * anticommutator( np.transpose(L2)@L2, rho )
  return rhodot

@njit(nogil = True)
def adv_timestep_L3(H,rho,L3):
  rhodot = -1j*commutator(H,rho)
  rhodot += L3@rho@np.transpose(L3) - 0.5 * anticommutator( np.transpose(L3)@L3, rho )
  return rhodot

@njit(nogil = True)
def adv_timestep_L12(H,rho,L1,L2):
  rhodot = -1j*commutator(H,rho) 
  rhodot += L1@rho@np.transpose(L1) - 0.5 * anticommutator( np.transpose(L1)@L1, rho )
  rhodot += L2@rho@np.transpose(L2) - 0.5 * anticommutator( np.transpose(L2)@L2, rho )
  return rhodot

@njit(nogil = True)
def adv_timestep_L123(H,rho,L1,L2,L3):
  rhodot = -1j*commutator(H,rho) 
  rhodot += L1@rho@np.transpose(L1) - 0.5 * anticommutator( np.transpose(L1)@L1, rho )
  rhodot += L2@rho@np.transpose(L2) - 0.5 * anticommutator( np.transpose(L2)@L2, rho )
  rhodot += L3@rho@np.transpose(L3) - 0.5 * anticommutator( np.transpose(L3)@L3, rho )
  return rhodot

#Time dependant potentials
@jit(forceobj=True)
def get_V2(Nx, Nt, Xa, scale):
    """Gives a time dependant potetnial which is  convergance of a double qaudratic well"""
    x = np.linspace(-Xa, Xa, Nx)
    V2 = np.zeros([Nt, Nx])

    for i in range(0,Nt):
        x_0 = Xa/2
        a = np.linspace(x_0,0,Nt)
        V2[i] = 0.5 * ((abs(x) + a[i] ) / Xa) ** 2
    return V2 * scale


def gaussian(x,height ,sigma):
    """returns a standard gaussian"""
    return height * np.exp(-x**2/ (2*sigma**2))

def get_Vg2(Nt,w,Nx, Xa,wi,hi, hf,wf):
  """ Returns a time dependannt potetnial which is a quadratic well 0.5* w**2 *x**2, with a gausian barrier of height hi standard deviation wi
  which linearly moves in time Nt to a gaussian with height hf and std wf
  HWHM ~ 1.177*sigma, HWTM ~ 2.146 sigma"""
  x = np.linspace(Xa,-Xa,Nx)
  Vg = np.zeros((Nt,Nx))
  start_height = hi
  start_width = wi
  end_height = hf
  end_width = wf  
  height_loss = start_height - end_height
  width_loss = start_width - end_width
  Vh = 0.5 * w**2 * x**2
  for i in range(Nt):
    height = height_loss * (Nt-i)/Nt + end_height
    sigma = width_loss * (Nt - i)/Nt + end_width
    Vg[i] = Vh + gaussian(x,height,sigma)
  return Vg

def get_Vg2m(Nt,w,Nx, Xa,wi,hi, hf,wf):
  """Same as above, however the minimum value fo the potential is subtracted at each step, so the minimum potetnial value is always 0,
  to make the potetnial more resamble the 2 channel system"""
  x = np.linspace(-Xa,Xa,Nx)
  Vg = np.zeros((Nt,Nx))
  start_height = hi
  start_width = wi
  end_height = hf
  end_width = wf  
  height_loss = start_height - end_height
  width_loss = start_width - end_width
  Vh = 0.5 * w**2 * x**2
  for i in range(Nt):
    height = height_loss * (Nt-i)/Nt + end_height
    sigma = width_loss * (Nt - i)/Nt + end_width
    Vg[i] = Vh + gaussian(x,height,sigma)
    Vg[i] = Vg[i] - np.min(Vg[i])
  return Vg

def get_Vg2lbm(Nt,w,Nx, Xa,wi,hi, hf,wf,m):
  """Same as above however with a linear bias m*x, such that the left half of the channel is at a lower energy"""
  x = np.linspace(Xa,-Xa,Nx)
  Vg = np.zeros((Nt,Nx))
  start_height = hi
  start_width = wi
  end_height = hf
  end_width = wf  
  height_loss = start_height - end_height
  width_loss = start_width - end_width
  Vh = 0.5 * w**2 * x**2
  Vl = m * x
  for i in range(Nt):
    height = height_loss * (Nt-i)/Nt + end_height
    sigma = width_loss * (Nt - i)/Nt + end_width
    Vg[i] = Vh + Vl + gaussian(x,height,sigma)
    Vg[i] = Vg[i] - np.min(Vg[i])
  return Vg

def get_QWGBi(Nt,i,w,x,Xa,wi,hi,hf,wf):
    """Returns Quadratic well with gaussian abrrier at a specfic time step, to get potetnial step wise"""
#   start_height = hi
#   start_width = wi
#   end_height = hf
#   end_width = wf    
    height_loss_step = (hi - hf) / Nt
    width_loss_step = (wi - wf) / Nt
    height = hi - i * height_loss_step
    width = wi - i * width_loss_step
    Vqw = 0.5 * w**2 * x**2
    Vg = Vqw + gaussian(x, height, width)
    Vg = Vg - np.min([Vg])
    return Vg

def get_QWGBLBi(Nt,i,w,x,Xa,wi,hi,hf,wf,m):
    height_loss_step = (hi - hf) / Nt
    width_loss_step = (wi - wf) / Nt
    height = hi - i * height_loss_step
    width = wi - i * width_loss_step
    Vqw = 0.5 * w**2 * x**2 
    Vl =  m * x
    Vg = Vqw + gaussian(x, height, width) + Vl
    Vg = Vg - np.min([Vg])
    return Vg

#RK4 individual step time evolution funcs
@njit(nogil = True)
def F(H,rho):
    K_1 = -1j*commutator(H,rho) 
    return K_1

@njit()
def F_L1(H,rho,L1):
    K_1 = -1j*commutator(H,rho) 
    K_1 += L1@rho@np.transpose(L1) - 0.5 * anticommutator( np.transpose(L1)@L1, rho )
    return K_1

@njit()
def F_L2(H,rho,L2):
    K_1 = -1j*commutator(H,rho) 
    K_1 += L2@rho@np.transpose(L2) - 0.5 * anticommutator( np.transpose(L2)@L2, rho )
    return K_1

@njit()
def F_L3(H,rho,L3):
    K_1 = -1j*commutator(H,rho) 
    K_1 += L3@rho@np.transpose(L3) - 0.5 * anticommutator( np.transpose(L3)@L3, rho )
    return K_1

@njit()
def F_L1L2(H,rho,L1,L2):
    K_1 = -1j*commutator(H,rho) 
    K_1 += L1@rho@np.transpose(L1) - 0.5 * anticommutator( np.transpose(L1)@L1, rho )
    K_1 += L2@rho@np.transpose(L2) - 0.5 * anticommutator( np.transpose(L2)@L2, rho )    
    return K_1

@njit()
def F_L1L2L3(H,rho,L1,L2,L3):
    K_1 = -1j*commutator(H,rho) 
    K_1 += L1@rho@np.transpose(L1) - 0.5 * anticommutator( np.transpose(L1)@L1, rho )
    K_1 += L2@rho@np.transpose(L2) - 0.5 * anticommutator( np.transpose(L2)@L2, rho )
    K_1 += L3@rho@np.transpose(L3) - 0.5 * anticommutator( np.transpose(L3)@L3, rho ) 
    return K_1


#Constatn H Runge Kutta method, the hamiltonian only updates between steps and not every step.. 
#small dt this can a valid aprox, and is simpler to use
@njit(nogil = True)
def RK4_step(H,rho,dt):
    """Runge Kutta 4th order, Constant H"""
    K_1 = F(H, rho)
    K_2 = F(H, rho + dt * K_1 / 2) 
    K_3 = F(H, rho + dt * K_2 / 2) 
    K_4 = F(H, rho + dt * K_3) 
    drho = (K_1 + 2*K_2 + 2*K_3 + K_4) / 6
    return drho


@njit()
def RK4_step_L1(H,rho,L1,dt):
    K_1 = F_L1(H, rho, L1)
    K_2 = F_L1(H, rho + dt * K_1 / 2, L1) 
    K_3 = F_L1(H, rho + dt * K_2 / 2, L1) 
    K_4 = F_L1(H, rho + dt * K_3, L1) 
    drho = (K_1 + 2*K_2 + 2*K_3 + K_4) / 6
    return drho

@njit()
def RK4_step_L2(H,rho,L2,dt):
    K_1 = F_L2(H, rho, L2)
    K_2 = F_L2(H, rho + dt * K_1 / 2, L2) 
    K_3 = F_L2(H, rho + dt * K_2 / 2, L2) 
    K_4 = F_L2(H, rho + dt * K_3, L2) 
    drho = (K_1 + 2*K_2 + 2*K_3 + K_4) / 6
    return drho

@njit()
def RK4_step_L3(H,rho,L3,dt):
    K_1 = F_L3(H, rho, L3)
    K_2 = F_L3(H, rho + dt * K_1 / 2, L3) 
    K_3 = F_L3(H, rho + dt * K_2 / 2, L3) 
    K_4 = F_L3(H, rho + dt * K_3,L3) 
    drho = (K_1 + 2*K_2 + 2*K_3 + K_4) / 6
    return drho

@njit()
def RK4_step_L1L2(H,rho,L1,L2,dt):
    K_1 = F_L1L2(H, rho, L1, L2)
    K_2 = F_L1L2(H, rho + dt * K_1 / 2, L1, L2) 
    K_3 = F_L1L2(H, rho + dt * K_2 / 2, L1, L2) 
    K_4 = F_L1L2(H, rho + dt * K_3, L1, L2) 
    drho = (K_1 + 2*K_2 + 2*K_3 + K_4) / 6
    return drho

@njit()
def RK4_step_L1L2L3(H,rho,dt,L1,L2,L3):
    K_1 = F_L1L2L3(H,rho,L1,L2,L3)
    K_2 = F_L1L2L3(H,rho+dt*K_1/2,L1,L2,L3) 
    K_3 = F_L1L2L3(H,rho+dt *K_2/2,L1,L2,L3)
    K_4 = F_L1L2L3(H,rho+dt*K_3,L1,L2,L3)
    drho = (K_1 + 2*K_2 + 2*K_3 + K_4) / 6
    return drho


#Runge Kutta 4th order steps, where it has Hamiltonian and lindblad opperator update for each half step
@njit()
def RK4t_step_L1L2L3(Ht,Htstep,Htdt,rho,dt,L1_t,L2_t,L3_t,
                     L1_step,L2_step,L3_step,L1_dt,L2_dt,L3_dt):
    """Time depnedant hamilton RK4, 3 lindblad modalities"""
    
    K_1 = F_L1L2L3(Ht,rho,L1_t,L2_t,L3_t)
    K_2 = F_L1L2L3(Htstep,rho+dt*K_1/2,L1_step,L2_step,L3_step) 
    K_3 = F_L1L2L3(Htstep,rho+dt *K_2/2,L1_step,L2_step,L3_step)
    K_4 = F_L1L2L3(Htdt,rho+dt*K_3,L1_dt,L2_dt,L3_dt)
    drho = (K_1 + 2*K_2 + 2*K_3 + K_4) / 6
    return drho

@njit()
def RK4t_step_L1L2(Ht,Htstep,Htdt,rho,dt,L1_t,L2_t,
                     L1_step,L2_step,L1_dt,L2_dt):
    
    K_1 = F_L1L2(Ht,rho,L1_t,L2_t)
    K_2 = F_L1L2(Htstep,rho+dt*K_1/2,L1_step,L2_step) 
    K_3 = F_L1L2(Htstep,rho+dt *K_2/2,L1_step,L2_step)
    K_4 = F_L1L2(Htdt,rho+dt*K_3,L1_dt,L2_dt)
    drho = (K_1 + 2*K_2 + 2*K_3 + K_4) / 6
    return drho

@njit()
def RK4t_step_L1(Ht,Htstep,Htdt,rho,dt,L1_t,
                     L1_step,L1_dt):
    
    K_1 = F_L1(Ht,rho,L1_t)
    K_2 = F_L1(Htstep,rho+dt*K_1/2,L1_step) 
    K_3 = F_L1(Htstep,rho+dt *K_2/2,L1_step)
    K_4 = F_L1(Htdt,rho+dt*K_3,L1_dt)
    drho = (K_1 + 2*K_2 + 2*K_3 + K_4) / 6
    return drho

@njit
def RK4t_step(Ht,Htstep,Htdt,rho,dt):
    
    K_1 = F(Ht,rho)
    K_2 = F(Htstep, rho+dt * K_1/2) 
    K_3 = F(Htstep,rho+dt * K_2/2)
    K_4 = F(Htdt,rho+dt * K_3)
    drho = (K_1 + 2*K_2 + 2*K_3 + K_4) / 6
    return drho   

#Need to have normalised wavefunction or the entopry doesnt make sense
def get_entropy(rho):
    """Von-Neuman entropy of a density matrix"""
    log_rho = logm(rho)/np.log(2.0)
    return -np.trace(rho@log_rho)
