import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
entropies=np.loadtxt('./Plotting/entropy/entropy_10.txt')
t=np.linspace(0,100,200)
plt.plot(t,entropies[:,0],label='$S_{[1]}$',linewidth=2.5)
plt.plot(t,entropies[:,1],label='$S_{[2]}$',linewidth=2.5)
plt.plot(t,entropies[:,2],label='$S_{[3]}$',linewidth=2.5)
plt.legend(loc='lower right')
plt.xlabel('Time/ps')
plt.ylabel('Entanglement entropy')
times_arr=[0,10,20,30]
print_step=1000
dt=0.0005

times=np.linspace(0,print_step*dt*len(entropies[:,0]),len(entropies[:,0]),endpoint=False)
times=list(times)
time_indices=[]
actual_times=[]
for time in times_arr:
    temp_t=np.abs(np.asarray(times)-time)
    t_closest=np.min(temp_t)
    closest_ind=list(temp_t).index(t_closest)
    time_indices.append(closest_ind)
    actual_times.append(times[closest_ind])
#print(time_indices)

print(f"Actual times {actual_times}")

print(entropies[time_indices,:])
#plt.savefig('./Plots/sig10entropy100ps',dpi=200,bbox_inches='tight')
plt.show()