################################################################################################################################
# Create the main wavefunction class
################################################################################################################################

import math
import numpy as np
import pycuda.driver as cuda
from  src.eigensolver.rSpaceSolverCPU import rSpaceSolverCPU
from  src.eigensolver.rSpaceSolverGPU import rSpaceSolverGPU
from  src.eigensolver.kSpaceSolverCPU import kSpaceSolverCPU
from  src.eigensolver.kSpaceSolverGPU import kSpaceSolverGPU
from  src.eigensolver.rSpaceSparseSolverGPU import rSpaceSparseSolverGPU
import reikna.fft as rfft
import pycuda.gpuarray as gpuarray
from src.get_ccbin import get_ccbin
import time

import cupy as cp


from kernel_qubits import standard_evolve

def build_state_ud(self):
    """
    spins: list of 'up' or 'down'
    returns: CuPy array representing the tensor product state on GPU
    """
    up = cp.array([1, 0], dtype=self.dtype)
    down = cp.array([0, 1], dtype=self.dtype)
    state = cp.array([1], dtype=self.dtype)  # start with scalar 1
    for n in self.nqubitsE:
        state = cp.kron(state, up if n%2==0 else down)
    return state
   

def build_state_mm(self):
    """
    spins: list of 'up' or 'down'
    returns: CuPy array representing the tensor product state on GPU
    """
    up = cp.array([1, 0], dtype=self.dtype)
    down = cp.array([0, 1], dtype=self.dtype)
    state = cp.array([1], dtype=self.dtype)  # start with scalar 1
    for n in self.nqubitsE:
        state = cp.kron(state, (up + down) / cp.sqrt(2))
    return state

def entropy_vn_svd(self, keep):
    """
    Compute von Neumann entropy of subsystem `keep` (list of qubit indices)
    from full pure statevector `psi` (CuPy array).
    All operations stay on GPU.
    """
    
    N=self.nqubits
    keep = sorted(keep)
    nA = len(keep)
    nB = N - nA

    # Permute qubits so A is first, B is last
    perm = keep + [i for i in range(N) if i not in keep]

    # Reshape to bipartite matrix
    psi_matrix = self.psi.reshape([2]*N).transpose(perm).reshape(2**nA, 2**nB)

    # Singular values (no U/V needed)
    s = cp.linalg.svd(psi_matrix, compute_uv=False)

    # Probabilities = squared singular values
    p = s**2
    p = p[p > 1e-15]  # numerical stability

    # von Neumann entropy in bits
    S = -cp.sum(p * cp.log2(p))
    return float(S.get())

class wavefunction:
    
    
    
    #this is called on creation of instance
    def __init__(self,  params):
        
        self.dtype=params["dtype"]
        self.nqubitsS=params["nqubitsS"]
        self.nqubitsE=params["nqubitsE"]
        self.nqubits=self.nqubitsS+self.nqubitsE
        self.psi_S_spec=params["psi_S_spec"]
        self.psi_E_spec=params["psi_E_spec"]
        
        
        up = cp.array([1, 0], dtype=self.dtype)
        down = cp.array([0, 1], dtype=self.dtype)
        
        match self.psi_S_spec:
           case "x+":
               psi_S= (up + down) / cp.sqrt(2)
           case "x-":
               psi_S = (up - down) / cp.sqrt(2)
           case "z+":
               psi_S= up
           case "z+":
               psi_S= down
               
        match self.psi_E_spec:
            case "mm":
                psi_E=build_state_mm(self)
            case "ud":
                psi_E=build_state_ud(self)
                
        self.psi=cp.kron(psi_S,psi_E)
        
        
    # def evolve(self):
        
        
        
