def getparams():
    params = {
        "SimName": "CUDA_1D_Double_Slit_Experiment",

        # Flags
        "ParticlesNr": 1,
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
        "dt": 0.0005,
        "StartTime": 0.0,

        # Total time steps and how often to print data
        "TimeSteps": 10000,
        "PrintStep": 10,


        # Lattice boundaries
        "Xa": -200.0,
        "Xb": 200.0,
        "Ya": 0.0,
        "Yb": 0.0,

        "Za": 0.0,
        "Zb": 0.0,

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
        "Traces": 0 
    }

    return params