from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import src.index.index1DtoND as index1DtoND
from src.input.trace1D import trace1D

SimName = 'CUDA_1D_Particle_in_a_Box'

def animate2p_sum(params):

    
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
    ax = plt.axes(xlim=(float(params["Xa"]), float(params["Xb"])), ylim=(0, 0.3))
    fig.suptitle('2 Particles in a 1D Box')
    plt.xlabel('x (nm)')
    plt.ylabel('ψ')
    #plotcols = ["blue","red"]
    lines = []
    for index in range(params["Np"]):
        #lobj = ax.plot([],[],lw=2,color=plotcols[index])[0]
        lobj = ax.plot([],[],lw=2)[0]
        lines.append(lobj)

    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    # animation function.  This is called sequentially
    def animate(i):
        wave = np.loadtxt(fname = FilePath + waves[i])
        # psi=np.zeros([params["Nx"],params["Nx"]],dtype=np.complex64)
        # for i in range(params["N"]):
        #     inds=np.zeros(2,dtype=int)
        #     index1DtoND.index(i,inds,params["Sizes"],params)
        #     psi[inds[1]][inds[0]]=wave[i][0]+1j*wave[i][1] #real +imag
        # sumx1=np.sum(np.abs(psi)**2,axis=1)
        # sumx2=np.sum(np.abs(psi)**2,axis=0)

        sumxs=trace1D(params,wave[:,0],wave[:,1])



        ylist = []
        for i in range (params["Np"]):
            ylist.append(sumxs[i])

        for lnum,line in enumerate(lines):
            line.set_data(x, ylist[lnum]) # set data for each line separately.

        return lines

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=len(waves), interval=1, blit=True)

    SavePath = './out/' + SimName + '/animation.mp4'
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Cambridge Quantum Information Group'))

    anim.save(SavePath, writer=writer)

    plt.show()