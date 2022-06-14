from cProfile import label
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import src.index.index1DtoND as index1DtoND
from src.input.trace1D import trace1D


from src.input.parameters import getparams
import time




SimName = 'CUDA_1D_Particle_in_a_Box'

def animatenp_sum(params):
    #plot traces from traces files
    
    SimName = params['SimName']

    # Read simulation files
    FilePath = './out/' + SimName + '/waves/'
    waves = [f for f in listdir(FilePath) if isfile(join(FilePath, f))] #all the files in folder (wave at each print step)
    waves.sort()
    

    wave = np.loadtxt(fname = FilePath + waves[0])
    # print(sum(np.square(wave[:,0]) + np.square(wave[:,1])))
    x = np.linspace(float(params["Xa"]), float(params["Xb"]), int(params["Nx"]))

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(float(params["Xa"]), float(params["Xb"])), ylim=(0, 0.30))
    fig.suptitle(f'{params["Np"]} Particles in a 1D Box')
    plt.xlabel('x (nm)')
    plt.ylabel('ψ')
    
    #plotcols = ["blue","red"]
    lines = []
    for index in range(params["Np"]):
        #lobj = ax.plot([],[],lw=2,color=plotcols[index])[0]
        lobj = ax.plot([],[],lw=2,label=f"Particle {index+1}")[0]
        lines.append(lobj)

    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    # animation function.  This is called sequentially
    def animate(i):
        print(i)
        wave = np.loadtxt(fname = FilePath + waves[i])
     

        ylist = []
        for i in range (params["Np"]):
            ylist.append(wave[:,i])

        for lnum,line in enumerate(lines):
            line.set_data(x, ylist[lnum]) # set data for each line separately.

        return lines

    # call the animator.  blit=True means only re-draw the parts that have changed.
    plt.legend()
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=len(waves), interval=1, blit=True)

    SavePath = './out/' + SimName + '/animation.mp4'
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Cambridge Quantum Information Group'))

    t0=time.time()
    anim.save(SavePath, writer=writer)
    print(time.time()-t0)

    #plt.show()


params=getparams()
animatenp_sum(params)
