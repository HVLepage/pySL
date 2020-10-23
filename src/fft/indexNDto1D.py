def index(inds, sizes, params, p = 0):
    if (p == 0):
        p = params["Np"]

    indOut = 0
    l = p * params["dim"]
    for i in range(0, l):
        stride = 1
        for j in range( i + 1, l):
            stride *= sizes[j % params["dim"]]

        indOut += inds[i] * stride
        
    return indOut
