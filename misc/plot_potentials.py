from sklearn.metrics import plot_confusion_matrix
from src.input.parameters import getparams
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
plt.rcParams.update({'font.size': 14})
def plot_coulomb(params):
    x=np.linspace(0,100,200)
    sigma=10
    V=(18.8950673 / params["RelPerm"])/np.sqrt(x**2+np.pi*0.5*sigma**2)
    V/= 1.312342066e-2
    plt.plot(x,V,label='$\sigma=10$ nm',linewidth=2.5)
    plt.xlabel('$|x_i-x_j|$/nm')
    plt.ylabel('$V_C$/meV')
    #plt.title('$V_C$ plotted against $|x_i-x_j|$ for different $\sigma$ values')
    sigma=30
    V=(18.8950673 / params["RelPerm"])/np.sqrt(x**2+np.pi*0.5*sigma**2)
    V/= 1.312342066e-2
    plt.plot(x,V,label='$\sigma=30$ nm',linewidth=2.5)

    sigma=0
    V=(18.8950673 / params["RelPerm"])/np.sqrt(x**2+np.pi*0.5*sigma**2)
    V/= 1.312342066e-2
    plt.plot(x,V,label='No softening ($\sigma=0$ nm)',linewidth=2.5)
    plt.legend()
    plt.ylim([0,0.2/1.312342066e-2])
    
    plt.savefig('./Plots/coulomb_si.png',bbox_inches='tight')
    plt.show()
    
def plot_sp(params):
    x=np.linspace(params["Xa"],params["Xb"],params["Nx"])
    VSinusoidal=params["Acos"]*(1+np.cos(params["kXcos"]*x))
    VBarrier=np.exp(-(x-params["barrierPos"])**2/(2*params["barrierWidth"]**2))*params["barrierHeight"]
    VHarmonic=0.5*params["Mass"][0]*(params["wX"]**2 * (x-params["xEqms"][0])**2 )
    V=VSinusoidal+VBarrier+VHarmonic
    V/= 1.312342066e-2
    plt.plot(x,V,linewidth=2.5)
    plt.xlabel("x/nm")
    plt.ylabel("$V_{sp}$/meV")
    plt.savefig('./Plots/potential_si.png',bbox_inches='tight')
    plt.show()



params=getparams()
plot_coulomb(params)
#plot_sp(params)