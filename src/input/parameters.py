def getparams():
    
    v_si2nu = 1.312342066e-2
    t_si2nu = 1.157676458e2

    params = {
        "SimName": "CUDA_1D_Double_Slit_Experiment",

        # Flags
        "Np": 1,
        "SolveEigenValueProblem": False,
        "KSpaceLeapfrog": False,
        "GPUSolver": True,
        "SparseSolver": False,
        "KSpaceSolver": False,
        "NormalizeWaveFunction": True,
        "SlidingWindow": False,
        "ImportPotential": False,
        "IncludeSpin": False,
        "PotentialFileName": "potential.txt",
        "WaveFileName": "wave.txt",
        "ExportTrace": False,

        # Mass is quoted in [electron masses] (so 1 for a free electron in a vacuum)
        "Mass": 1.0,

        # Relative permittivity of quantum well material
        "RelPerm": 13.1,

        # SAW amplitude [meV] , wavenumber 1/[nm], and frequency 1/[s] see MATLAB Code
        "ASAW": 0.0,
        "cSAW": 3.0,
        "lambdaSAW": 1000.0,
        "SAWoffset": 0.0,

        # Soon to be obsolete (calculated in InputParameters.cu)
        "kSAW": 0.006283185307180,
        "wSAW": 0.018849555921539,


        # dt is time in [ps] per each timeStep
        "dt": 0.00005,
        "StartTime": 0.0,

        # Total time steps and how often to print data
        "TimeSteps": 50000,
        "PrintStep": 5000,


        # Lattice boundaries
        "x0": -200.0,
        "x1": 200.0,
        "y0": 0.0,
        "y1": 0.0,

        "z0": 0.0,
        "z1": 0.0,

        # Lattice Sites
        "Nx": 801,
        "Ny": 1,
        "Nz": 1,

        "NxOutK": 43,
        "NyOutK": 41,
        "NzOutK": 1,

        # Momentum Solver Inputs
        "KxECO": 6,
        "KyECO": 6,
        "KzECO": 0,
        "KxKCO": 6,
        "KyKCO": 6,
        "KzKCO": 0,


        # Lattice padding (for expanding TDSE solution window)
        "padLeft": 0,
        "padRight": 0,
        "padTop": 0,
        "padBottom": 0,

        # Absorbing Boudary Conditions (Leave as zero for Dirichlet boundary)
        "ABC_L": 0,
        "ABC_R": 0,
        "ABC_T": 0,
        "ABC_B": 0,
        "ABC_U": 0,
        "ABC_D": 0,

        # Potential variables
        "w2X": 0.0,
        "w2Y": 0.0,
        "ATB1": 0.0,
        "s2TB1": 0.0,
        "ATB2": 0.0,
        "s2TB2": 0.0,
        "detuneX": 0.0,

        "RS": 0.0,
        "DRt0": 1.0,
        "URt0": 12.4,

        "InputStates": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "sigmaFit": [1.74707037e-17, -2.73761743e-14, 1.78462882e-11, -6.26868417e-09, 1.28479963e-06, -1.55625754e-04, 1.07952353e-02, 6.02006265e-01, 6.93382649e+00],
        "Traces": 0,
        
    }

    # Convert variables to Natural Units
    params["dt"] *= t_si2nu
    params["StartTime"] *= t_si2nu
    params["ASAW"] *= v_si2nu
    params["wSAW"] /= t_si2nu
    params["cSAW"] /= t_si2nu
    params["DRt0"] *= t_si2nu
    params["URt0"] *= t_si2nu

    # Construct Hamiltonian
    params["dx"] = 1.0
    params["dy"] = 1.0
    params["dz"] = 1.0
    if (params["Nx"] > 1):
        params["dx"] = (params["x1"] - params["x0"]) / (params["Nx"] - 1)
    if (params["Ny"] > 1):
        params["dy"] = (params["y1"] - params["y0"]) / (params["Ny"] - 1)
    if (params["Nz"] > 1):
        params["dz"] = (params["z1"] - params["z0"]) / (params["Nz"] - 1)
        
        
    if (params["SolveEigenValueProblem"] != True):
        params["KxECO"] = 0
        params["KyECO"] = 0
        params["KzECO"] = 0
	

    if (params["KSpaceLeapfrog"]!=True):
        params["NxOutK"] = 1
        params["NyOutK"] = 1
        params["NzOutK"] = 1
        params["KxKCO"] = 0
        params["KyKCO"] = 0
        params["KzKCO"] = 0
	
        
    params["N"] =(params["Nx"] * params["Ny"] * params["Nz"]) ** params["Np"]
    params["N1"] = params["Nx"] * params["Ny"] * params["Nz"]
    params["NOut"] = (params["NxOutK"] * params["NyOutK"] * params["NzOutK"])**params["Np"]
    params["NK"] = ((2 * params["KxKCO"] + 1) * (2 * params["KyKCO"] + 1) * (2 * params["KzKCO"] + 1))**params["Np"]
    
    params["dim"] = 0
    if (params["Nx"] > 1):
        params["dim"]=params["dim"]+1
    if (params["Ny"] > 1):
        params["dim"]=params["dim"]+1
    if (params["Nz"] > 1):
        params["dim"]=params["dim"]+1
    if (params["dim"] == 0):
        print( "No dimension size is greater than 1")
        exit(1)
	

	# Check if padded output is greater than Psi
    if ((params["KSpaceLeapfrog"]) and (params["NxOutK"] < params["Nx"] or params["NyOutK"] < params["Ny"] or params["NzOutK"] < params["Nz"])) :
        print("NOut parameters cannot be smaller than N for KSpaceLeapfrog")
        exit(1)
	
    if ((params["KSpaceLeapfrog"]) and ((2 * params["KxKCO"] + 1) >params["Nx"] or (2 * params["KyKCO"] + 1) > params["Ny"] or (2 * params["KzKCO"] + 1) > params["Nz"])):
        print("(2 * Kx/y/zKCO + 1)  cannot be greater than Nx/y/z for KSpaceLeapfrog")
        exit(1)
	



    return params