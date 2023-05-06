################################################################################################################################
# Take a 1D index and return indices for each dimension of the wave function (x/y/z for each particle)
################################################################################################################################

def index(ind, inds, params, p = 0): 
    if (p == 0):
        p = params["Np"]

   
    N = 2**p
    
    stride = N
    
    temp = stride
    stride = temp // sizes[i % params["dim"]]
    inds[i] = ind // stride
    ind = ind - inds[i] * stride
