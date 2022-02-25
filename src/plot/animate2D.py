import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from matplotlib import animation
import src.index.index1DtoND as index1DtoND

def animate2D(params):

    
    global lobj #variables inside a function are local so must make global
    SimName = 'Particle_in_a_box_1D'

        # Read simulation files
    FilePath = './out/' + SimName+ '/waves/'
    waves = [f for f in listdir(FilePath) if isfile(join(FilePath, f))] #all the files in folder (wave at each print step)
    waves.sort()

    fig = plt.figure()
    ax = plt.axes(xlim=(params["Xa"],params["Xb"]), ylim=(params["Ya"],params["Yb"]))
    ax.set_aspect('equal')
    fig.suptitle('2D Particle in a Box')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')

    x = np.linspace(float(params["Xa"]), float(params["Xb"]), int(params["Nx"]))
    y = np.linspace(float(params["Ya"]), float(params["Yb"]), int(params["Ny"]))

    wavefunc2D=np.zeros([params["Ny"],params["Nx"]])
    lobj = ax.contourf(x, y, wavefunc2D, levels=100,cmap='Greys')
    
    def animate(i):
        global lobj #so can modify lobj
        wave = np.loadtxt(fname = FilePath + waves[i])
        for c in lobj.collections:
                c.remove() 
        
        wavefunc2D=np.zeros([params["Ny"],params["Nx"]])
        for i in range(params["Nx"]*params["Ny"]):
            
            inds=np.zeros(2)
            index1DtoND.index(i,inds,params["Sizes"], params)
            wavefunc2D[int(inds[1])][int(inds[0])]=np.square(wave[i][0])+np.square(wave[i][1])

        lobj = ax.contourf(x,y,wavefunc2D,levels=100,cmap='Greys')
        
        return lobj.collections

    
    anim = animation.FuncAnimation(fig, animate, 
                                    frames=len(waves), interval=1,blit=True)
    SavePath = './out/' + SimName + '/animation.mp4'
        # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Cambridge Quantum Information Group'))

    anim.save(SavePath, writer=writer,dpi=400)

    #plt.show()