import numpy as np
import src.index.index1DtoND as index1DtoND
import pycuda.driver as cuda
from src.input.parameters import getparams
from os.path import isfile, join
from os import listdir
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.fftpack
plt.rcParams.update({'font.size': 14})

#plot expectation for n particles in 1D as a function of time, iteration files must be traces

x=np.linspace(-100,100,201)
# Read simulation files
FilePath = './Plotting/expectations/'

sigs = [f for f in listdir(FilePath)] #all the files in folder (wave at each print step)
sigs.sort()

print(sigs)
fig,axs=plt.subplots(3,2,figsize=(10,12))
for i in range(len(sigs)):
    sigma=sigs[i].split('_')[1][3:5]
    waves=[f for f in listdir(FilePath+sigs[i])]
    waves.sort()
    expectations=np.zeros([3,len(waves)]) #will hold the expectation for each particle at each iteration
    for j in range(len(waves)):

        wave = np.loadtxt(fname = FilePath +sigs[i]+ '/'+waves[j])
        for p1 in range(3):
            
            expectations[p1][j]=np.sum(np.multiply(x,wave[:,p1])) #just sum once since have already summed in traces

    t=np.linspace(0,1000,2000,endpoint=False)
    for p2 in range(3):
        #plt.plot(t,np.sum(expectations[:,:],axis=0)/3,label=f'Particle {p+1}')
        #break
        axs.flatten()[i].plot(t,expectations[p2,:],linewidth=1.5)
        axs.flatten()[i].set(xlabel='Time/ps',ylabel='Expectation of \n particle position/nm')
    axs.flatten()[i].text(0.66,0.92,f"$\sigma$={sigma} nm", transform=axs.flatten()[i].transAxes)
fig.tight_layout()
plt.savefig(f"./Plots/expecs.png",bbox_inches='tight',dpi=200)
plt.show()


# expectations=np.zeros([params["Np"],len(waves)]) #will hold the expectation for each particle at each iteration
# for i in range(len(waves)):

#     wave = np.loadtxt(fname = FilePath + waves[i])
#     for p in range(params["Np"]):
#         x=np.linspace(params["Xa"],params["Xb"],params["Nx"])
#         expectations[p][i]=np.sum(np.multiply(x,wave[:,p])) #just sum once since have already summed in traces
# plt.figure()
# t=np.linspace(params["StartTime"],params["PrintStep"]*len(waves)*params["dt"]/params["t_si2nu"],len(waves))
# for p in range(params["Np"]):
#     #plt.plot(t,np.sum(expectations[:,:],axis=0)/3,label=f'Particle {p+1}')
#     #break
#     plt.plot(t,expectations[p,:],label=f'Particle {p+1}',linewidth=2.5)

# # def sine(x,a,b,c):
# #     return a*np.sin(b*x+c)
# # popt, pcov = curve_fit(sine, t[10:], expectations[0,10:],p0=[7.15,2*np.pi/13.8,-2*np.pi/13.8*5.15])
# # print(popt)
# # plt.plot(t[10:],sine(t[10:],*popt))
# # plt.plot( t[10:],expectations[0,10:])

# #plt.legend()
# plt.xlabel('Time/ps')
# #plt.xlim([params["Xa"],params["Xb"]])
# plt.ylabel("Expectation of particle position/nm")
# #plt.title("Expectation of the position of each particle as a function of time")
# #plt.plot(t,np.sin((2*np.pi/13.8)*(t-5.15))*7.15)
# plt.savefig(f"./Plots/sig30dduexpec.png",bbox_inches='tight',dpi=200)
# #plt.figure()

# # sample_spacing=params["PrintStep"]*params["dt"]/params["t_si2nu"]
# # num_samples=len(waves)
# # print(sample_spacing)

# # f=np.linspace(0,1//(2*sample_spacing),num_samples//2)



# # fft=scipy.fftpack.fft(expectations[2,:])

# # plt.plot(f,np.abs(fft[:num_samples//2])*2/num_samples)
# # plt.xlabel("Frequency/Hz")
# plt.show()    


    





