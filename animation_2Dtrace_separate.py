import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from matplotlib import animation
import src.index.index1DtoND as index1DtoND
from src.input.parameters import getparams
import time
from src.input.trace_1p import trace_1p
def animate2Dtrace(params):

    
    global lobj #variables inside a function are local so must make global
    SimName = 'Particle_in_a_box_1D'

        # Read simulation files
    FilePath = './out/' + SimName+ '/waves/'
    waves = [f for f in listdir(FilePath) if isfile(join(FilePath, f))] #all the files in folder (wave at each print step)
    waves.sort()


    
    Xa = params["Xa"]
    Xb = params["Xb"]
    Nx = params["Nx"]
    x = np.linspace(Xa, Xb, Nx)

    for p in range(params["Np"]):
        
        Xs=[f"x{i+1}" for i in range(params["Np"])]
        Xs.pop(p)
        fig = plt.figure()
        ax = plt.axes(xlim=(Xa,Xb), ylim=(Xa,Xb))
        
        plt.xlabel(f"{Xs[0]}/nm")
        plt.ylabel(f"{Xs[1]}/nm")

        x = np.linspace(Xa, Xb, Nx)

        wavefunc2D=np.zeros([params["Nx"],params["Nx"]])
        lobj = ax.contourf(x, x, wavefunc2D, levels=100,cmap='Greys')
        
        def animate(i):
            print(i)
            global lobj #so can modify lobj
            wave = np.loadtxt(fname = FilePath + waves[i])
            for c in lobj.collections:
                    c.remove() 
            
            wavefunc2D=np.zeros([params["Nx"],params["Nx"]])
            wavefunc2D=wave[:,p].reshape([Nx,Nx])
            

            lobj = ax.contourf(x,x,np.transpose(wavefunc2D),levels=100,cmap='Greys')
            
            return lobj.collections

        
        anim = animation.FuncAnimation(fig, animate, 
                                        frames=len(waves), interval=1,blit=True)
        SavePath = './out/' + SimName + f'/animation_p{p+1}.mp4'
            # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Cambridge Quantum Information Group'))

        anim.save(SavePath, writer=writer,dpi=400)

        
params = getparams()
t0=time.time()
animate2Dtrace(params)
print(time.time()-t0)