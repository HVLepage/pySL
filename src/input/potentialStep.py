import numpy as np

def potentialStep(params,x):
    step=np.exp(-(x-params["stepPos"])**2/(2*params["stepWidth"]**2))*params["stepHeight"]
    return step