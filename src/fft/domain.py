################################################################################################################################
# Define simulation domain
################################################################################################################################

# This function returns an array with the same dimensions as the wave function.
# Values on the boundary of the simulation domain are set to false, everything inside is set to true.
# Although this method uses more GPU memory, it reduces the number of operations required per iteration.
import numpy as np
import src.fft.index1DtoND as index1DtoND
import pycuda.driver as cuda

class domain:
     def __init__(self,params):
         self.domain = np.ones(params["N"], dtype=np.bool_)
         inds = np.zeros(params["Np"] * params["dim"], dtype=np.int32)
         for i in range(0,params["N"]):
             index1DtoND.index(i,inds, params["Sizes"], params)
             for j in range(0,(params["Np"] * params["dim"])):
                 if (inds[j]==0 or inds[j]==(params["Sizes"][j%params["dim"]] - 1)):
                     self.domain[i] = 0
                     break
                 
         self.domain_gpu = cuda.mem_alloc_like(self.domain)
         cuda.memcpy_htod(self.domain_gpu, self.domain)