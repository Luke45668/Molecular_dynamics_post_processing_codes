#%%
import os
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import regex as re
import pandas as pd
import sigfig
# plt.rcParams.update(plt.rcParamsDefault)
# plt.rcParams['text.usetex'] = True
#from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import time 
import scipy.stats
from datetime import datetime
import h5py as h5 
import multiprocessing as mp
from log2numpy import *
from mom2numpy import *
from velP2numpy import *
from dump2numpy import * 



import glob 
from post_MPCD_MP_processing_module import *
colour = [
 'black',
 'blueviolet',
 'cadetblue',
 'chartreuse',
 'coral',
 'cornflowerblue',
 'crimson',
 'darkblue',
 'darkcyan',
 'darkgoldenrod',
 'darkgray']


from scipy.optimize import curve_fit
def linearfunc(x,a):
    return (a*x)
steps=1000000000 
filepath="/Users/luke_dev/Documents/simulation_test_folder/langevin_test"
os.chdir(filepath)
log_name_list=glob.glob("log.lang*")
log_files =()

thermo_vars='         KinEng          Temp          TotEng        c_msd[3]      c_vacf[3]   '
for log_name in log_name_list:
        log_files=log_files+(log2numpy_reader(log_name,
                            filepath,
                            thermo_vars),)
log_array=np.zeros((11,1000001,6))


for i in range(11):

        time=steps*0.00101432490424207
        log_array[i,:,:]=log_files[i]

        # timepoints=np.linspace(0,time,log_files[i].shape[0])
        # plt.plot(timepoints,log_files[i][:,4])
        # popt,pcov=curve_fit(linearfunc,timepoints,log_files[i][:,4])
        # plt.plot(timepoints,(popt[0]*(timepoints)))
#plt.show()
#%%
log_mean=np.mean(log_array,axis=0)
timepoints=np.linspace(0,time,log_mean.shape[0])
cutoff=330000
plt.plot(timepoints[:],log_mean[:,4])
popt,pcov=curve_fit(linearfunc,timepoints[:cutoff],log_mean[:cutoff,4])
plt.plot(timepoints[:cutoff],(popt[0]*(timepoints[:cutoff])))
plt.ylabel("$\\langle \mathbf{r}^{2} \\rangle $", rotation=0, labelpad=10)
plt.xlabel("$t$")
plt.show()

# %% VACF curve 

plt.plot(timepoints,log_mean[:,5])

plt.ylabel("VACF", rotation=0)
print("mean VACF",np.mean(log_mean[:,5]))

plt.show()
# %% temp plot

plt.plot(timepoints,log_mean[:,2])
print("temp",np.mean(log_mean[:,2]))

plt.ylabel("T")

plt.show()
# %%
