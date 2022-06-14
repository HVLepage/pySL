import numpy as np

from itertools import permutations


def parity(perm):
    #gives parity of a permutation of 0 to n-1
    parity_matrix=np.zeros([len(perm),len(perm)]) #create permutation matrix
    for ind,val in enumerate(perm):
        x=np.zeros(len(perm))
        x[val]=1
        parity_matrix[ind]=x
    return np.linalg.det(parity_matrix) #parity given by determinant of parity matrix

def antisymmetrise(params,wave_real,wave_im):
    psi=np.zeros(params["N"],dtype=np.complex64)
    psi=wave_real+1j*wave_im
    psi=np.reshape(psi,np.tile(params["Sizes"],params["Np"]))
    psi_a=np.zeros(np.tile(params["Sizes"],params["Np"]),dtype=np.complex64) #will hold antisymmetrised psi
    particle_perms=permutations(np.arange(params["Np"]))

    for perm in particle_perms:
        psi_a+=parity(perm)*np.transpose(psi,axes=perm) #permutes psi and then adds to psi_a with sign determined by parity of perm
    return np.ndarray.flatten(psi_a)
