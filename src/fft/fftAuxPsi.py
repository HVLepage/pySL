import math

def ThetaOfMax(psi, L):
    maxi = 0.0
    maxindex = 0
    current = 0.0
    for i in range(0,L):
        current = (psi[i].real ** 2) + (psi[i].imag ** 2)
        if (current > maxi):
            maxi = current
            maxindex = i
    return math.atan2(psi[maxindex].imag, psi[maxindex].real)


def ToReal(params, psi, OutK):
	if (OutK):
		Nx = params["NxOutK"]
		Ny = params["NyOutK"]
		Nz = params["NzOutK"]
	else:
		Nx = params["Nx"]
		Ny = params["Ny"]
		Nz = params["Nz"]



	 #dKx = params["dKx"] #delta momentum
	 #dKy = params["dKy"]
	 #dKz = params["dKz"]

	 #dx = params["dx"]
	 #dy = params["dy"]
	 #dz = params["dz"]



	#normalise and rotate to real


		theta = ThetaOfMax(psi, Nx*Ny*Nz ** params["Np"])
		for i in range(0, (Nx*Ny*Nz ** params["Np"] )):
			psiR = psi[i].real
			psiI = psi[i].imag
			psi[i] = psiR * math.cos(theta) + psiI * math.sin(theta) + 1j*(psiI * math.cos(theta) - psiR * math.sin(theta))

