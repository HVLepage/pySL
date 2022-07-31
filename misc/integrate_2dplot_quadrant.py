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
from scipy import ndimage

plt.rcParams.update({'font.size': 14})

# wave = np.loadtxt('./Plotting/2D_anti_10_0005_1000/wave_iteration_010.txt')
# #wave = np.loadtxt('./Plotting/2D_nospin_10_0005_1000/wave_iteration_000.txt')
# wavefunc2D=np.zeros([Nx,Nx])
# wavefunc2D=wave[:,0].reshape([Nx,Nx]) #antisymmetrised so all are the same, [particle1, particle 2]


# wavefunc2D/=np.sum(wavefunc2D)
#wavefunc2D=np.tril(wavefunc2D)


# print(wave)

def integrate_2dplot(lims):
    Nx=201
    Xa=-100
    Xb=100
    x=np.linspace(Xa,Xb,Nx)
    lims_actual=[]#
    inds_actual=[]
    for i in range(4):
        x_temp=np.abs(x-lims[i])
        min_val=np.min(x_temp)
        min_ind=list(x_temp).index(min_val)
        inds_actual.append(int(min_ind))
        lims_actual.append(x[min_ind])
    print(inds_actual)
    print(lims_actual) #might round down
    waves = [f for f in listdir('./Plotting/'+'2D_ddu_30_0005_1000/') if isfile(join('./Plotting/'+'2D_ddu_30_0005_1000/', f))]
    arr=np.zeros(len(waves))
    for i,val in enumerate( waves):
        wave = np.loadtxt('./Plotting/'+'2D_ddu_30_0005_1000/'+waves[i])
        
        wavefunc2D=np.zeros([Nx,Nx])
        wavefunc2D=wave[:,0].reshape([Nx,Nx]) #antisymmetrised so all are the same, [particle1, particle 2]


        wavefunc2D/=np.sum(wavefunc2D)
        arr[i]=np.sum(wavefunc2D[inds_actual[0]:inds_actual[1],inds_actual[2]:inds_actual[3]])
        print(np.sum(wavefunc2D[inds_actual[0]:inds_actual[1],inds_actual[2]:inds_actual[3]]))
        #print(np.max(wavefunc2D[inds_actual[0]:inds_actual[1],inds_actual[2]:inds_actual[3]]))
        wavefunc2D_0=np.copy(wavefunc2D)
    
        #wavefunc2D[:,:][wavefunc2D[:,:]>0]=1
        total_probability =0
        #print(np.count_nonzero(wavefunc2D[:,:]))
        initial_value=4*10**-4
        step=10**-7
        counter=0
    plt.plot(np.linspace(0,100,200,endpoint=False),arr,linewidth=2.5)
    plt.xlabel('Time/ps')
    plt.ylabel('Probability of both particles having x>0')
    plt.savefig('./Plots/sig30quadrant.png',dpi=200,bbox_inches='tight')
    plt.show()
    # while total_probability<0.8:
    #     wavefunc2D=np.copy(wavefunc2D_0)
    #     #print(initial_value-step*counter)
    #     wavefunc2D[:,:][wavefunc2D[:,:]-(initial_value-step*counter)<0]=0
    #     #print(np.count_nonzero(wavefunc2D[:,:]))
    #     counter+=1
    #     #print(np.count_nonzero(wavefunc2D[:,:]))
    #     #print( ndimage.find_objects((np.array(wavefunc2D,dtype=int))))
    #     labeled_array,num_features=ndimage.measurements.label(wavefunc2D,[[0,1,0],[1,1,1],[0,1,0]])
    #     #print(labeled_array)
    #     slices=ndimage.find_objects(labeled_array)
    #     #print(slices)
    #     feature_integrated=np.zeros(num_features)
    #     boxes=np.zeros([Nx,Nx])
        
    #     for i in range(num_features):
    #         feature_integrated[i]=np.sum((wavefunc2D)[slices[i]])

    #         #print(f"Feature {i+1}, sum = {feature_integrated[i]}, x limits: {x[slices[i][0].start]}, {x[slices[i][0].stop]}, y limits: {x[slices[i][1].start]}, {x[slices[i][1].stop]}")
    #         boxes[slices[i]]=1
    #     total_probability=np.sum(feature_integrated)
    #     #print(total_probability)
        
    #     #wave = np.loadtxt('./Plotting/2D_anti_10_0005_1000/wave_iteration_000.txt')
        
    #     #wavefunc2D=wave[:,0].reshape([Nx,Nx])
    #     #wavefunc2D/=np.sum(wavefunc2D)
        
    # for i in range(num_features):
        

    #     print(f"Feature {i+1}, sum = {feature_integrated[i]}, x limits: {x[slices[i][0].start]}, {x[slices[i][0].stop]}, y limits: {x[slices[i][1].start]}, {x[slices[i][1].stop]}")
        
    # print(f"Total probability in features= {np.sum(feature_integrated)}")
    # print(f"Remove all values <{initial_value-step*(counter-1)}, counter={counter}")
    # plt.figure()
    # plt.imshow(np.transpose(labeled_array),extent=[-100,100,100,-100])
    # plt.xlabel("$x_i$/nm")
    # plt.ylabel("$x_j$/nm (j ≠ i)")
    # plt.gca().invert_yaxis()
    # plt.figure()
    # plt.imshow(np.transpose(wavefunc2D),extent=[-100,100,100,-100],cmap='Greys')
    # plt.xlabel("$x_i$/nm")
    # plt.ylabel("$x_j$/nm (j ≠ i)")
    # plt.gca().invert_yaxis()
    # plt.figure()
    # plt.imshow(np.transpose(boxes),extent=[-100,100,100,-100])
    # plt.xlabel("$x_i$/nm")
    # plt.ylabel("$x_j$/nm (j ≠ i)")
    # plt.gca().invert_yaxis()
    # plt.show()
    
    

integrate_2dplot([0,100,0,100])
