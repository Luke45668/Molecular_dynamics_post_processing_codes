##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will calculate the MPCD stress tensor for a pure fluid under forward NEMD
"""
#%% Importing packages
import os
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import regex as re
import pandas as pd
import sigfig
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime

path_2_post_proc_module= '/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/LAMMPS python run and analysis scripts/Analysis codes'
os.chdir(path_2_post_proc_module)
from log2numpy import *
from mom2numpy import *
from velP2numpy import *
from dump2numpy import * 
import glob 
from post_MPCD_MP_processing_module import *


#%% importing one log file 

realisation_name = "log.vtargetnubar_no806324_wall_pure_output_no_rescale_541709_2_60835_23.0_0.005071624521210362_10_10000_1000_1000000_T_1_lbda_0.05071624521210362_SR_15_SN_1_VT_10"
Path_2_log= "/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/"
thermo_vars="         KinEng          Temp          TotEng        c_ke1[1]       c_ke1[2]       c_ke1[3]       c_ke1[5]       c_mom[1]       c_mom[2]       c_mom[3]       c_mom[5]   "
log_file_for_test= log2numpy_reader(realisation_name,Path_2_log,thermo_vars)


#%% equilibrium
delta_t_srd=0.05071624521210362
box_vol=23**3
kinetic_energy_tensor_xx= np.array([(2*log_file_for_test[1:2000,4])/box_vol ]).T
kinetic_energy_tensor_yy= np.array([(2*log_file_for_test[1:2000,5])/box_vol ]).T
kinetic_energy_tensor_zz= np.array([(2*log_file_for_test[1:2000,6])/box_vol ]).T
delta_mom_pos_tensor=np.zeros((log_file_for_test.shape[0]-2,3))
for i in range(0,log_file_for_test.shape[0]-2):
    delta_mom_pos_tensor[i,0]= log_file_for_test[i+1,8]- log_file_for_test[i,8]
    delta_mom_pos_tensor[i,1]= log_file_for_test[i+1,9]- log_file_for_test[i,9]
    delta_mom_pos_tensor[i,2]= log_file_for_test[i+1,10]- log_file_for_test[i,10]
delta_mom_pos_tensor=delta_mom_pos_tensor/((delta_t_srd*box_vol))
stress_tensor_xx= kinetic_energy_tensor_xx[:,0] +delta_mom_pos_tensor[:,0]
stress_tensor_yy= kinetic_energy_tensor_yy[:,0] +delta_mom_pos_tensor[:,1]
stress_tensor_zz= kinetic_energy_tensor_zz[:,0] +delta_mom_pos_tensor[:,2]

plt.plot(log_file_for_test[2:,0],stress_tensor_xx[:])
plt.title("stress tensor xx vs time steps ")
plt.show()
plt.plot(log_file_for_test[2:,0],stress_tensor_yy[:])
plt.title("stress tensor yy vs time steps ")
plt.show()
plt.plot(log_file_for_test[2:,0],stress_tensor_zz[:])
plt.title("stress tensor zz vs time steps ")
plt.show()
pressure = np.mean(stress_tensor_xx+stress_tensor_yy +stress_tensor_zz)
print("Pressure",pressure)

#factor of 10000 out 



#%% constants for shear 
box_vol=23**3
delta_t_srd=0.05071624521210362
erate=0.00001

#%% 
delta_mom_pos_tensor=np.zeros((log_file_for_test.shape[0]-2,1))
for i in range(0,log_file_for_test.shape[0]-2):
    delta_mom_pos_tensor[i,0]= log_file_for_test[i+1,6]- log_file_for_test[i,6]

delta_mom_pos_tensor=delta_mom_pos_tensor*(1/(delta_t_srd*box_vol))
kinetic_energy_tensor_zz=np.array([(erate*delta_t_srd*log_file_for_test[2:,4])/box_vol]).T # skip off the first value 
kinetic_energy_tensor_xz=np.array([2*log_file_for_test[1:6000,5]/box_vol ]).T# skip off the first value 
stress_tensor_xz = kinetic_energy_tensor_xz + kinetic_energy_tensor_zz +delta_mom_pos_tensor
stress_tensor_xz_rms_mean = np.sqrt(np.mean(stress_tensor_xz**2))





# %%
#plt.plot(log_file_for_test[1:,0],stress_tensor_xz[:,0])


plt.plot(log_file_for_test[2:,0],delta_mom_pos_tensor[:,0])
plt.title("$\Delta p_{x}r_{z}$ vs timesteps ")

plt.show()

plt.plot(log_file_for_test[2:,0],kinetic_energy_tensor_xz[:,0])
plt.title("Kinetic energy tensor xz vs time steps ")

plt.show()

plt.plot(log_file_for_test[2:,0],kinetic_energy_tensor_zz[:,0])
plt.title("Kinetic energy tensor zz vs time steps ")
plt.show()


plt.plot(log_file_for_test[2:,0],stress_tensor_xz[:,0])
plt.title("stress tensor xz vs time steps ")
plt.show()


# %%
