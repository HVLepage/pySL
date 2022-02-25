import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from matplotlib import animation
import src.index.index1DtoND as index1DtoND
from src.input.parameters import getparams
import time
def animate2p(params):

    
    global lobj #variables inside a function are local so must make global
    SimName = 'Particle_in_a_box_1D'

        # Read simulation files
    FilePath = './out/' + SimName+ '/waves/'
    waves = [f for f in listdir(FilePath) if isfile(join(FilePath, f))] #all the files in folder (wave at each print step)
    waves.sort()

    fig = plt.figure()
    ax = plt.axes(xlim=(params["Xa"],params["Xb"]), ylim=(params["Xa"],params["Xb"]))
    ax.set_aspect('equal')
    fig.suptitle('2 Particles in a Box')
    plt.xlabel('x1 (nm)')
    plt.ylabel('x2 (nm)')

    x1 = np.linspace(float(params["Xa"]), float(params["Xb"]), int(params["Nx"]))
    x2 = np.linspace(float(params["Xa"]), float(params["Xb"]), int(params["Nx"]))

    wavefunc2D=np.zeros([params["Nx"],params["Nx"]])
    lobj = ax.contourf(x1, x2, wavefunc2D, levels=100,cmap='Greys')
    
    def animate(i):
        print(i)
        global lobj #so can modify lobj
        wave = np.loadtxt(fname = FilePath + waves[i])
        for c in lobj.collections:
                c.remove() 
        
        wavefunc2D=np.zeros([params["Nx"],params["Nx"]])
        for i in range(params["Nx"]*params["Nx"]):
            
            inds=np.zeros(2)
            index1DtoND.index(i,inds,params["Sizes"], params)
            wavefunc2D[int(inds[1])][int(inds[0])]=np.square(wave[i][0])+np.square(wave[i][1]) #mod^2

        lobj = ax.contourf(x1,x2,wavefunc2D,levels=100,cmap='Greys') 
        
        return lobj.collections

    
    anim = animation.FuncAnimation(fig, animate, 
                                    frames=len(waves), interval=1,blit=True)
    SavePath = './out/' + SimName + '/animation.mp4'
        # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Cambridge Quantum Information Group'))

    t0=time.time()
    anim.save(SavePath, writer=writer,dpi=400)
    print(time.time()-t0)
    plt.show()

params=getparams()
animate2p(params)