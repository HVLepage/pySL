import numpy as np

def harmonicPotential(params,p,x,y,z): #if y and z aren't passed
    
    if params["dim"]==1:
        VHarmonic=0.5*params["Mass"][p]*(params["wX"]**2 * (x-params["xEqms"][p])**2 )
    if params["dim"]==2:
         VHarmonic=0.5*params["Mass"][p]*(params["wX"]**2 * (x-params["xEqm"][p])**2 + params["wY"]**2 * (y-params["yEqm"])**2)
    if params["dim"]==3:
        VHarmonic=0.5*params["Mass"][p]*(params["wX"]**2 * (x-params["xEqm"][p])**2 + params["wY"]**2 * (y-params["yEqm"])**2 + params["wZ"]**2 * (z-params["zEqm"])**2)

    return VHarmonic