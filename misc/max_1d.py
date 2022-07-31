from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
#from src.input.parameters import getparams
import time
from src.plot.plot_multiple_1D import plot_multiple_1D
from src.plot.plot_multiple_2D_same import plot_multiple_2D_same
from src.plot.plot_multiple_2D_separate import plot_multiple_2D_separate
from scipy.signal import argrelextrema
import scipy.fftpack
plt.rcParams.update({'font.size': 14})


file_name='1D_anti_30_0005_1000'
t=np.linspace(0,1000,2000,endpoint=False)
x=np.linspace(-100,100,201)

waves = [f for f in listdir('./Plotting/'+file_name) if isfile(join('./Plotting/'+file_name, f))]
# max_values=np.zeros(len(waves))
# for i in range(len(waves)):
#     wave = np.loadtxt(fname = './Plotting/'+file_name+'/'+ waves[i])
#     max_val=np.max(wave[:,0])
#     max_index=list(wave[:,0]).index(max_val)
#     x_max=x[max_index]
#     max_values[i]=x_max
# plt.plot(t,max_values)
# plt.show()

# zero_ind=list(x).index(55)
# zero_vals=np.zeros(len(waves))
# for i in range(len(waves)):
#     wave = np.loadtxt(fname = './Plotting/'+file_name+'/'+ waves[i])
#     zero_vals[i]=wave[zero_ind,0]
# plt.plot(t,zero_vals)
# plt.show()
# plt.figure()
# sample_spacing=0.5
# num_samples=len(waves)


# f=np.linspace(0,1//(2*sample_spacing),num_samples//2)



# fft=scipy.fftpack.fft(zero_vals)

# plt.plot(f,np.abs(fft[:num_samples//2])*2/num_samples)
# plt.xlabel("Frequency/Hz")
# plt.show()    

# peak1_vals=np.zeros(len(waves))
# peak2_vals=np.zeros(len(waves))
# peak3_vals=np.zeros(len(waves))
# for i in range(len(waves)):
#     wave = np.loadtxt(fname = './Plotting/'+file_name+'/'+ waves[i])
#     if len(argrelextrema(wave[:,0],np.greater)[0])==3:
#         peak1_vals[i]=x[argrelextrema(wave[:,0],np.greater)[0][1]]
# plt.plot(t,peak1_vals)
# plt.show()
