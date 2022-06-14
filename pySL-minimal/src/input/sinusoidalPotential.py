import numpy as np

def sinusoidalPotential(params,p,x,y,z): #if y and z aren't passed
    
    if params["dim"]==1:
        #VSinusoidal=params["Acos"]*(1-np.cos(params["kXcos"]*x))
        VSinusoidal=params["Acos"]*(1+np.cos(params["kXcos"]*x)+0.01*x/(params["Xb"]-params["Xa"])) #+x/(params["Xb"]-params["Xa"])
        if params["potentialBarrier"]:
            VSinusoidal+=np.exp(-(x-params["barrierPos"])**2/(2*params["barrierWidth"]**2))*params["barrierHeight"]
        #VSinusoidal=params["Acos"]*(1+np.cos(params["kXcos"]*x))
    if params["dim"]==2:
        1==1
    if params["dim"]==3:
        1==1

    return VSinusoidal