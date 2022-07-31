import math
import numpy as np
import sys

def getparams():
    
    params = {
        
        "SimName": "Particle_in_a_box_1D",

        ################################################################
        # FLAGS 
        ################################################################
        
        "Np": 3, # Number of particles
      
        
        "TDSESolver": "rSpace", # Type of staggered leapfrog solver [rSpace, kSpace] (only rSpace supported in this version)
        "Eigensolver": "GPUrSpace", # Type of eigensolver [CPUrSpace, GPUrSpace] future [GPUkSpace, GPUrSparse]

        # Define the initial superposition state of the wave function. 
        # The first two elements are the real and imaginary components of the ground state. 
        # The second two elements are the real and imaginary components of the first excited state. 
        "InputStates": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        # Import external files (if set to sal)
        "ImportPotential": False, # If set to false, the potential will be created functionally in fftauxV.py
        "1pPotential":True, #if want to add 1particle potential defined by function e.g harmonicPotential.py then turn this on
        "1pPotentialType":"Sinusoidal", #Harmonic or Sinusoidal (sinusoidal is actually sinusoidal +harmonic, can change this in potential.py)
        "1pPotentialStep":False, #gaussian step, irrelevant for newton's cradle
        "PotentialFileName": "potential.txt",

        "oneptdepPotential":1, #time dependent potential parameters, switches off all of the below except potential barrier, 0 for off, 1 for on
        "potentialBarrier":1, #gaussian barrier, 0 for off and 1 for on
        "barrierHeight":0.7, #height in NU
        "barrierPos":-90, #position in nm
        "barrierWidth":4, #width in nm
        "tauBarrier":100, #decay constant for barrier when nonadibatically evolving system
        "adiabaticSine":1, #either 0 or 1, used when adibatically putting system in eigenstate
        "adiabaticCoulomb":1, #either 0 or 1, used when adibatically putting system in eigenstate
        "nonadiabaticBarrier":0, #when nonadibatically evolving system

        "ImportWavefunction":False, #from file, used when evolving newton's cradle(put this to false and npfrom1p to false if want 1 particle eigenstates)
        "WaveFileName": "wave.txt",
        "NormalizeWaveFunction": True, # Set to false if importing a partial wave function

        "npfrom1pWavefunction":True, #used when finding 3 particle eigenstate (n particle wave function from 1 particle wave functions)
     

        #irrelevant parameters used when want to manually define wave function
        "defWavefunction":False, #if set to false eigenvalue solver is used
        "wavefunctionType":"Gaussian", #only this at the moment
        "coherentState":False,
        "muxG":400, #centre of Gaussian in x, particle 1
        "mux2G": -400, #centre of Gaussian in x, particle 2
        "mux3G":0,#centre of Gaussian in x, particle 3
        "muyG":0, #centre of Gaussian in y, particle 1
        "sigmaxG":200, #width of Gaussian in x, particle 1
        "sigmax2G":200,#width of Gaussian in x, particle 2
        "sigmax3G":200, #width of Gaussian in x, particle 3
        "sigmayG":200, #width of Gaussian in y, particle 1
        "kGx": -2*np.pi/100,    #plane wave momentum for particle 1 in x (wavepacket is Gaussian * plane wave)
        "kGx2":2*np.pi/100, #plane wave momentum for particle 2 in x (wavepacket is Gaussian * plane wave)
        "kGx3":0, #plane wave momentum for particle 3 in x (wavepacket is Gaussian * plane wave)
        "kGy":2*np.pi/100, #plane wave momentum for particle 1 in y (wavepacket is Gaussian * plane wave)
      

       
        "ExportNpTrace":True, #trace over all particles but 1
        "Export1pTrace":False, #trace over just 1 particle out of e.g. 3
        


        "antisymmetrise":False, #antisymmetrise wave function
        "spins":[1,1,0], #1 for down, 0 for up, convention I use is particle to left of barrier is 3rd particle, central particle is particle 1 and right particle is 2

        ################################################################
        # Particle properties
        ################################################################

        # Mass is quoted in [electron masses] (so 1 for a free electron in a vacuum)
        "Mass": np.array([1.0,1.0,1.0], dtype=np.float32)*0.067, #can keep as many particles even if only 1 used
        
        # Charge in [electron charges]
        "Charge": np.array([1.0,1.0,1.0], dtype=np.float32), #can keep as many particles even if only 1 used

        # Relative permittivity of quantum well material
        "RelPerm": 13.1,

        # Conversion between SI units and natural units (hbar = 1, m_e = 1)
        "v_si2nu": 1.312342066e-2,
        "t_si2nu": 1.157676458e2,
   
        ################################################################
        # Simulation time
        ################################################################

        
        "dt": 0.00025, # duration of a single time step in [ps]
        "StartTime": 0.0,
        "TimeSteps": 1600000,
        "PrintStep": 20000, # How often the wave function is saved to a file

        ################################################################
        # Simulation domain
        ################################################################

        # Lattice boundaries [nm]
        "Xa": -200.0, 
        "Xb": 200.0,
        "Ya": -1000,
        "Yb": 1000.0,

        "Za": 0.0,
        "Zb": 0.0,

        # Number of lattice sites
        "Nx": 201, 
        "Ny": 1, 
        "Nz": 1,

        ################################################################
        # Potential 
        ################################################################

        
        "w2X": 0.0, #these aren't used
        "w2Y": 0.0,
        "w2Z": 0.0,

        #Harmonic Potential variables
        "wX":0.004*1.157676458e2*2,# 0.000025*1.157676458e2,#0.0001*1.157676458e2, 0.003 for xwidth=200nm, 0.0001 for xwidth=2000nm
        "wY": 0.0*1.157676458e2,
        "wZ": 0.0,
        "xEqms": np.array([0,0,0]), #nm, x centre of harmonic potential
        "yEqm": 0.0,
        "zEqm": 0.0,

        #Sinusoidal potential variables
        "kXcos": 2*np.pi/90, #wave vector i.e. wavelength is 90nm 
        "Acos":  0.2/ 1.312342066e-2,  #amplitude of cosine
        "tau":6250, #decay constant for sinusoidal potential (NU) 
        "tauCoulomb":6250, #decay constant for Coulomb potential (NU)

        #Step parameters
        "stepPos":0,
        "stepWidth":5,
        "stepHeight":0.00297/ 1.312342066e-2,


        "sigSoft":30 #same softenting parameter in y and z
    
    }

    t_si2nu = params["t_si2nu"]
    v_si2nu = params["v_si2nu"]

    # Convert variables to Natural Units
    params["dt"] *= t_si2nu
    params["StartTime"] *= t_si2nu
   
    
    if params["coherentState"]:
        params["wX"]=1/(params["sigmaxG"]**2)*t_si2nu

    params["wX"]/=t_si2nu
    params["wY"]/=t_si2nu
    params["wZ"]/=t_si2nu

    params["Acos"]*=v_si2nu
    params["stepHeight"]*=v_si2nu

    params["same_spin_pairs"]=[]
    for i,vali in enumerate(params["spins"]):
        for j,valj in enumerate(params["spins"]):
            if j<i:
                if vali==valj:
                    params["same_spin_pairs"].append((i,j))
    params["same_spin_pairs"]=np.ndarray.flatten(np.asarray(params["same_spin_pairs"],dtype=np.int32))
    print(params["same_spin_pairs"])
    params["num_pairs"]=len(params["same_spin_pairs"])/2
    print(params["num_pairs"])
    

    # Construct Hamiltonian
    params["dx"] = 1.0
    params["dy"] = 1.0
    params["dz"] = 1.0
    if (params["Nx"] > 1):
        params["dx"] = (params["Xb"] - params["Xa"]) / (params["Nx"] - 1)
    if (params["Ny"] > 1):
        params["dy"] = (params["Yb"] - params["Ya"]) / (params["Ny"] - 1)
    if (params["Nz"] > 1):
        params["dz"] = (params["Zb"] - params["Za"]) / (params["Nz"] - 1)
        
	
        
    params["N"] =(params["Nx"] * params["Ny"] * params["Nz"]) ** params["Np"]
    params["N1"] = params["Nx"] * params["Ny"] * params["Nz"]

    
    params["dim"] = 0

    params["dim"]=np.count_nonzero(np.array([params["Nx"]-1,params["Ny"]-1,params["Nz"]-1])) #slightly faster
    if (params["dim"] == 0):
        print( "No dimension size is greater than 1")
        exit(1)
	

    params["dXs"] = np.zeros( params["dim"], dtype = np.float32)
    params["Xas"] = np.zeros( params["dim"], dtype = np.float32)
    params["faa"] = np.zeros( params["dim"], dtype = np.float32)
    params["Sizes"] = np.zeros( params["dim"], dtype = np.int32)
    params["hbar"] = 1
    #change to for loop
    i = 0
    if (params["Nx"] > 1):
        params["Sizes"][i] = params["Nx"] #Psi size
        params["dXs"][i] = params["dx"]
        params["Xas"][i] = params["Xa"]
        params["faa"][i] = params["hbar"] * params["dt"] / (params["dx"] **2) #kinetic parameter faa for Hamiltonians/kernels
        i=i+1
    if (params["Ny"] > 1):
        params["Sizes"][i] = params["Ny"] #Psi size
        params["dXs"][i] = params["dy"]
        params["Xas"][i] = params["Ya"]
        params["faa"][i] = params["hbar"] * params["dt"] / (params["dy"] **2) #kinetic parameter faa for Hamiltonians/kernels
        i=i+1
    if (params["Nz"] > 1):
        params["Sizes"][i] = params["Nz"] #Psi size
        params["dXs"][i] = params["dz"]
        params["Xas"][i] = params["Za"]
        params["faa"][i] = params["hbar"] * params["dt"] / (params["dz"] **2) #kinetic parameter faa for Hamiltonians/kernels
        i=i+1
	

    params["fbb"] = 2.0 * params["dt"] / params["hbar"] #time parameter fbb for Hamiltonians/kernels

    return params
