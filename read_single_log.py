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
import seaborn as sns
import h5py as h5


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
steps=1000000
filepath="/Users/luke_dev/Documents/simulation_test_folder/langevin_test"
#filepath="/Users/luke_dev/Documents/simulation_test_folder/low_damp_test"
filepath="/Users/luke_dev/Documents/simulation_test_folder/uef_tests"
filepath="/Users/luke_dev/Documents/simulation_test_folder/flat_elastic_high_shear_rate_test"
filepath="/Users/luke_dev/Documents/simulation_test_folder/many_plates_test"
os.chdir(filepath)
log_name_list=glob.glob("log.lang*gdot_0_*")
hdf5_name_list=glob.glob("*h5")
log_files =()

thermo_vars='         KinEng          Temp          TotEng        c_msd[3]      c_vacf[3]   '
#thermo_vars='         KinEng         PotEng        c_myTemp        c_bias         TotEng    '
thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
#thermo_vars='         KinEng         PotEng         Press           Temp      c_uniaxnvttemp'
for log_name in log_name_list:
        log_files=log_files+(log2numpy_reader(log_name,
                            filepath,
                            thermo_vars),)
#log_array=np.zeros((10,133,6))
log_array=np.zeros((len(log_name_list),log_files[0].shape[0],7))
col=5


for i in range(len(log_name_list)):

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
log_mean_temp=np.mean(log_array[:,5:,col],axis=0)


#%% kinetic  energy
col=1
cutoff=25
for i in range(1):

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
#%% potential energy
col=2
cutoff=25
for i in range(1):

        time=steps*0.00101432490424207
        log_array[i,:,:]=log_files[i]

        timepoints=np.linspace(0,time,log_files[i].shape[0])
        plt.plot(timepoints[1:],log_files[i][1:,col])
       
        print(log_name_list[i])
        print(np.mean(log_files[i][1,col]))
        print(i)
        # popt,pcov=curve_fit(linearfunc,timepoints,log_files[i][:,4])
        # plt.plot(timepoints,(popt[0]*(timepoints)))
plt.show()



#%% pressure
col=3
cutoff=25
for i in range(1):

        time=steps*0.00101432490424207
        log_array[i,:,:]=log_files[i]

        timepoints=np.linspace(0,time,log_files[i].shape[0])
        plt.plot(timepoints[1:],log_files[i][1:,col])
       
        print(log_name_list[i])
        print(np.mean(log_files[i][1,col]))
        print(i)
        # popt,pcov=curve_fit(linearfunc,timepoints,log_files[i][:,4])
        # plt.plot(timepoints,(popt[0]*(timepoints)))
plt.show()
#%% bias temp
col=5
cutoff=25
for i in range(1):

        time=steps*0.00101432490424207
        log_array[i,:,:]=log_files[i]

        timepoints=np.linspace(0,time,log_files[i].shape[0])
        plt.plot(timepoints[1:],log_files[i][1:,col])
       
        print(log_name_list[i])
        print(np.mean(log_files[i][1,col]))
        print(i)
        # popt,pcov=curve_fit(linearfunc,timepoints,log_files[i][:,4])
        # plt.plot(timepoints,(popt[0]*(timepoints)))
plt.show()
# %%
#%% total energy
col=6
cutoff=25
for i in range(1):

        time=steps*0.00101432490424207
        log_array[i,:,:]=log_files[i]

        timepoints=np.linspace(0,time,log_files[i].shape[0])
        plt.plot(timepoints[1:],log_files[i][1:,col])
       
        print(log_name_list[i])
        print(np.mean(log_files[i][1,col]))
        print(i)
        # popt,pcov=curve_fit(linearfunc,timepoints,log_files[i][:,4])
        # plt.plot(timepoints,(popt[0]*(timepoints)))
plt.show()
# %%
