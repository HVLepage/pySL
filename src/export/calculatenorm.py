def getNorm(re,im,params):
    dV = params["dx"] * params["dy"] * params["dz"] * params["nP"]

    norm = sum(re**2 + im**2) * dV

    return norm