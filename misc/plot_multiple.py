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
plt.rcParams.update({'font.size': 14})





#plot_multiple_1D(1,1,'','','',[400],-100,100,201,ratio=0.2,fig_size=(6,8),times_label=False,times_text=True,legend=False,file_name='test')
#plot_multiple_1D(5,1,'','','',[0,5,10,15,20],-100,100,201,ratio=0.2,fig_size=(6,8),times_label=False,times_text=True,legend=False,file_name='test')
plot_multiple_1D(4,1,'','','',[0,3,6,9],-100,100,201,ratio=0.2,fig_size=(6,8),times_label=False,times_text=True,legend=False,anti=False,
              file_path='1D_uuu_noint_0005_1000',file_name='1Dnointuuu')

#plot_multiple_2D_separate('',[0,17,18,19],-100,100,201,(4,12),times_label=False,times_text=True,anti=True,file_path='2D_anti_10_0005_1000',file_name='10anti_test')
#plot_multiple_2D_separate('',[0,5,10,15],-100,100,201,(4,12),times_label=False,times_text=True,anti=True,file_path='2D_anti_30_0005_1000',file_name='30anti_test') #noint
#plot_multiple_2D_separate('',[0,3,9,9+13.5],-100,100,201,(4,12),times_label=False,times_text=True,anti=True,file_path='2D_anti_noint_0005_1000',file_name='nointanti_Test')
#plot_multiple_2D_separate('',[0,5,10,15],-100,100,201,(4,12),times_label=False,times_text=True,anti=False,file_path='2D_anti_30_0005_1000',file_name='test14')
#plot_multiple_2D_same('Probability density for each particle ',[0,2,4,6,8],-100,100,201)

#plot_multiple_2D_separate('',[0,10,20,30],-100,100,201,(8,12),times_label=False,times_text=True,anti=False,file_path='2D_ddu_30_0005_1000',file_name='2Dsig30ddu')
