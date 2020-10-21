import numpy as np
import pycuda.driver as cuda

class Potential():
    def dtoh(self):
        cuda.memcpy_dtoh(self.pot, self.pot_gpu)
    
    def htod(self):
        cuda.memcpy_htod(self.pot_gpu, self.pot)
        
        
    
    def __init__(self,params):        
        
        self.pot=np.zeros(params["N"])
        self.pot = self.pot.astype(np.float32)
        
        self.pot_gpu=cuda.mem_alloc(self.pot.nbytes)
        
        
        self.pot = self.pot.astype(np.float32)

        self.htod()