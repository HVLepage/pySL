import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

plt.rcParams.update({'font.size': 14})
def tau(x):
    return 0.25*np.log(3*x**2+6*x+1)-6**0.5/24*np.log((x+1-(2/3)**0.5)/(x+1+(2/3)**0.5))
freq=np.linspace(0,0.01,100)
freq_estimate_b=np.zeros(len(freq))
freq_estimate_g=np.zeros(len(freq))
freq_estimate=np.zeros(len(freq))
for i_f in range(len(freq)):
    t=np.linspace(0,1000,2000)

    random_wave=np.zeros(len(t))

    for i in range(10):
        random_wave+=np.sin(2*np.pi*t*np.random.rand(1)*0.073695)*0.5

    wave=np.sin(2*np.pi*0.073695*t)+2*np.sin(2*np.pi*freq[i_f]*t)+random_wave
    #plt.figure()
    #plt.plot(t,wave)
    sample_spacing=0.5
    num_samples=2000
    #print(sample_spacing)
    
    f=np.linspace(0,1//(2*sample_spacing),num_samples//2)
    #print(f)
    f_res=f[1]-f[0]


    fft=scipy.fftpack.fft(wave)
    #plt.figure()
    #plt.plot(f,np.abs(fft[:num_samples//2])*2/num_samples)

    fft=(fft[:num_samples//2])
    max_index=np.abs(fft).argsort()[-1]

    ap=(fft[max_index+1].real*fft[max_index].real+fft[max_index+1].imag*fft[max_index].imag)/(fft[max_index].real*fft[max_index].real+fft[max_index].imag*fft[max_index].imag)
    dp=-ap/(1-ap)
    am=(fft[max_index-1].real*fft[max_index].real+fft[max_index-1].imag*fft[max_index].imag)/(fft[max_index].real*fft[max_index].real+fft[max_index].imag*fft[max_index].imag)
    dm=am/(1-am)
    d=(dp+dm)/2+tau(dp*dp)-tau(dm*dm)
    kp=(max_index+d)*f_res
    freq_estimate_b[i_f]=kp
    print("kp =" +str(kp))

    fft=np.abs(fft[:num_samples//2])

    #print(sorted(fft)[-5:][::-1])
    #print(f[fft.argsort()[-5:][::-1]])
    max_index=fft.argsort()[-1]
    #print(max_index)
    #print(fft[:10])
    delta=np.log(fft[max_index+1]/fft[max_index-1])/(2*np.log(fft[max_index]**2/(fft[max_index-1]*fft[max_index+1])))
    #print(delta)
    #print(f[max_index])
    freq_estimate[i_f]=f[max_index]
    f_interp=f_res*(max_index+delta)
    freq_estimate_g[i_f]=f_interp
    #print(f_interp)


plt.scatter(freq,freq_estimate_b,c='r',s=10,label="Quinn's second estimator")
plt.scatter(freq,freq_estimate_g,c='b',s=10,label='Gaussian interpolation')
plt.scatter(freq,freq_estimate,c='g',s=10,label="Uncorrected peak frequency")
plt.plot(freq,freq,c='k',alpha=0.5)
plt.legend(loc='lower right')
plt.xlabel("True frequency of slow component/THz")
plt.ylabel("Estimated frequency/THz")
plt.title("Estimated frequency vs true frequency")
plt.xlim([0,0.01])
plt.ylim([0,0.01])
plt.savefig('./Plots/freqinterp.png',dpi=200,bbox_inches='tight')
plt.show()