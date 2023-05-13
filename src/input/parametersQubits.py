import math
import numpy as np
import sys

def getparams():
    
    params = {
        
        "SimName": "Demon",

        ################################################################
        # FLAGS 
        ################################################################
        
        "Np": 2, # Number of particles
        # "IncludeSpin": False, # Solve Pauli equation instead of Schrödinger equation (future release)
        
        "T": 1.0, #Temperature of bath
        
        "TDSESolver": "qubits", # Type of staggered leapfrog solver [rSpace, kSpace]
        
        # Define the initial superposition state of the wave function. 
        # The first two elements are the real and imaginary components of the ground state. 
        # The second two elements are the real and imaginary components of the first excited state. 
        "InputStates": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        # Import external files (if set to sal)
        "ImportPotential": False, # If set to false, the potential will be created functionally in fftauxV.py
        "PotentialFileName": "potential.txt",

        "ImportWavefunction": False, # If set to false, the wave function will be solved using the specified eigensolver
        "WaveFileName": "wave_DSE1D.txt",
        "NormalizeWaveFunction": True, # Set to false if importing a partial wave function

        
        ################################################################
        # Particle properties
        ################################################################

        # Mass is quoted in [electron masses] (so 1 for a free electron in a vacuum)
        

        ################################################################
        # Surface acoustic wave
        ################################################################

      

        ################################################################
        # Simulation time
        ################################################################

        
        "dt": 0.00001, # duration of a single time step in [ps]
        "StartTime": 0.0,
        "TimeSteps": 1000,
        "PrintStep": 1000, # How often the wave function is saved to a file

        ################################################################
        # Simulation domain
        ################################################################

      


        ################################################################
        # Potential 
        ################################################################

        # Potential variables
     

        "RS": 0.0,
        "DRt0": 1.0,
        "URt0": 12.4,
        
    }

    ################################################################
    # Post processing and conversion to Natural Units
    ################################################################

    # Conversion between SI units and natural units (hbar = 1, m_e = 1)
    t_si2nu = 1.157676458e2
    v_si2nu = 1.312342066e-2

    params["FinalTime"]=  params["StartTime"] +  params["dt"]*params["timeSteps"]

    # Convert variables to Natural Units
    params["dt"] *= t_si2nu
    params["StartTime"] *= t_si2nu
  

    # Construct Hamiltonian
   

    params["hbar"] = 1
    
    params["N"]=params["Np"]**2
    
    params["VUpdates"]=np.arange( params["dt"]*params["timeSteps"]//(params["Np"]-1))
    
    params["fbb"] = 2.0 * params["dt"] / params["hbar"] #time parameter fbb for Hamiltonians/kernels

    return params