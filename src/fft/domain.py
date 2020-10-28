import numpy as np
import src.fft.index1DtoND as index1DtoND

class Domain:
     def __init__(self,params):
         self.domain = np.ones(params["N"], dtype=np.bool_)
         inds = np.zeros(params["Np"] * params["dim"], dtype=np.int32)
         for i in range(0,params["N"]):
             index1DtoND.index(i,inds, params["Sizes"], params)
             for j in range(0,(params["Np"] * params["dim"])):
                 if (inds[j]==0 or inds[j]==(params["Sizes"][j%params["dim"]] - 1)):
                     self.domain[i] = 0
                     break