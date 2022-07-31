
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from mpl_toolkits import mplot3d
from src.input.parameters import getparams
import time


def animatenp_sum(params):
    #plot traces from traces files
    
    SimName = params['SimName']

    # Read simulation files
    FilePath = './out/' + SimName + '/waves/'
    waves = [f for f in listdir(FilePath) if isfile(join(FilePath, f))] #all the files in folder (wave at each print step)
    waves.sort()
    def plot4d(data):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection="3d")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        mask = data > 0.00001
        idx = np.arange(int(np.prod(data.shape)))
        x, y, z = np.unravel_index(idx, data.shape)
        ax.scatter(x, y, z, c=data.flatten(), s=10.0 * mask, edgecolor="face", alpha=0.2, marker="o", cmap="magma", linewidth=0)
        plt.tight_layout()
        plt.savefig("test_scatter_4d.png", dpi=250)
        plt.show()
        
        



   
    
    

    wave = np.loadtxt(fname = FilePath + waves[0])
    psi=wave[:,0]+1j*wave[:,1]
    psi=np.reshape(psi,np.tile(params["Sizes"],params["Np"]))
    print(np.max(np.abs(psi)**2))
    # print(sum(np.square(wave[:,0]) + np.square(wave[:,1])))
    x = np.linspace(float(params["Xa"]), float(params["Xb"]), int(params["Nx"]))
    x1,x2,x3=np.meshgrid(x,x,x)
    plot4d(np.abs(psi)**2)
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    #ax.scatter3D(x1,x2,x3,np.abs(psi)**2,alpha=0.7,marker='.')
    #plt.savefig('./out/' + SimName +'/3D.png')
    #plt.show()
    #plotcols = ["blue","red"]
    

params=getparams()
animatenp_sum(params)
