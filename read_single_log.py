#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
steps=5000000
#steps=1315000 
filepath="/Users/luke_dev/Documents/simulation_test_folder/langevin_test"
#filepath="/Users/luke_dev/Documents/simulation_test_folder/low_damp_test"
os.chdir(filepath)
log_name_list=glob.glob("log.lang*")
log_files =()

thermo_vars='         KinEng          Temp          TotEng        c_msd[3]      c_vacf[3]   '
#thermo_vars='         KinEng         PotEng        c_myTemp        c_bias         TotEng    '
for log_name in log_name_list:
        log_files=log_files+(log2numpy_reader(log_name,
                            filepath,
                            thermo_vars),)
#log_array=np.zeros((10,133,6))
log_array=np.zeros((11,500001,6))
col=4

for i in range(3):

        time=steps*0.00101432490424207
        log_array[i,:,:]=log_files[i]

        timepoints=np.linspace(0,time,log_files[i].shape[0])
        plt.plot(timepoints[1:],log_files[i][1:,col])
        plt.axhline(np.mean(log_files[i][1:,col]))
        print(log_name_list[i])
        print(np.mean(log_files[i][1:,col]))
        # popt,pcov=curve_fit(linearfunc,timepoints,log_files[i][:,4])
        # plt.plot(timepoints,(popt[0]*(timepoints)))
plt.show()

#%%
col=2
cutoff=25
for i in range(3):

        time=steps*0.00101432490424207
        log_array[i,:,:]=log_files[i]

        timepoints=np.linspace(0,time,log_files[i].shape[0])
        plt.plot(timepoints[1:],log_files[i][1:,col])
        plt.axhline(np.mean(log_files[i][1,col]))

       
        print(log_name_list[i])
        print(np.mean(log_files[i][1,col]))
        print(i)
        # popt,pcov=curve_fit(linearfunc,timepoints,log_files[i][:,4])
        # plt.plot(timepoints,(popt[0]*(timepoints)))
plt.show()
#%% total energy
col=3
cutoff=25
for i in range(3):

        time=steps*0.00101432490424207
        log_array[i,:,:]=log_files[i]

        timepoints=np.linspace(0,time,log_files[i].shape[0])
        plt.plot(timepoints[1:],log_files[i][1:,col])
        plt.axhline(np.mean(log_files[i][1,col]))

       
        print(log_name_list[i])
        print(np.mean(log_files[i][1,col]))
        print(i)
        # popt,pcov=curve_fit(linearfunc,timepoints,log_files[i][:,4])
        # plt.plot(timepoints,(popt[0]*(timepoints)))
plt.show()
#%%
log_mean=np.mean(log_array,axis=0)
temp_mean=np.mean(log_array[:,2],axis=0)
timepoints=np.linspace(0,time,log_mean.shape[0])
cutoff=500000
plt.plot(timepoints[:cutoff],log_mean[:cutoff,4])
popt,pcov=curve_fit(linearfunc,timepoints[:cutoff],log_mean[:cutoff,4])
plt.plot(timepoints[:cutoff],(popt[0]*(timepoints[:cutoff])))
plt.ylabel("$\\langle \mathbf{r}^{2} \\rangle $", rotation=0, labelpad=10)
plt.xlabel("$t$")
plt.show()




#%% 
eta_H20_non_dim=30
r_particle=0.125
diff_particle=popt[0]/6
einstein_diffusion=1/(6*np.pi*eta_H20_non_dim*r_particle)


diff_particle/einstein_diffusion

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
