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

realisation_name = "log.testout_no806324_wall_pure_output_no_rescale_541709_2_60835_23.0_0.005071624521210362_10_10000_10000_1000010_T_1_lbda_0.05071624521210362_SR_15_SN_1_VT_10" # with shear 
Path_2_log= "/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/"
thermo_vars="         KinEng          Temp          TotEng        c_pe[1]        c_pe[2]        c_pe[3]        c_pe[5]        c_mom[1]       c_mom[2]       c_mom[3]       c_mom[5]   "
log_file_for_test= log2numpy_reader(realisation_name,Path_2_log,thermo_vars)
#%%
Path_2_VP=Path_2_log
chunk=20
VP_ave_freq=10000
no_timesteps=1000000
VP_output_col_count=6 
equilibration_timesteps=0
no_SRD=60835
#realisation_name="vel.testout_pure_output_no806324_no_rescale_541709_2_60835_23.0_0.005071624521210362_10_10000_10000_500010_T_1_lbda_0.05071624521210362_SR_15_SN_1_VT_10"
#vel_profile_for_test_x= np.mean(np.mean(velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[2],axis=1))
#vel_profile_for_test_y= np.mean(np.mean(velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[3],axis=1))
#vel_profile_for_test_z= np.mean(np.mean(velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[4],axis=1))
#pressure= np.sqrt(vel_profile_for_test_x**2 + vel_profile_for_test_y**2 + vel_profile_for_test_z**2)/3
  

#%% equilibrium
delta_t_srd=0.05071624521210362
box_vol=23**3
erate=0.001


kinetic_energy_tensor_xx= np.array([(log_file_for_test[1:202,4]) ]).T
kinetic_energy_tensor_yy= np.array([(log_file_for_test[1:202,5])]).T
kinetic_energy_tensor_zz= np.array([(log_file_for_test[1:202,6])]).T
kinetic_energy_tensor_xz= np.array([(log_file_for_test[1:202,7])]).T


delta_mom_pos_tensor=np.zeros((log_file_for_test.shape[0]-1,4))
for i in range(0,log_file_for_test.shape[0]-1):
    delta_mom_pos_tensor[i,0]= log_file_for_test[i+1,8]- log_file_for_test[i,8]
    delta_mom_pos_tensor[i,1]= log_file_for_test[i+1,9]- log_file_for_test[i,9]
    delta_mom_pos_tensor[i,2]= log_file_for_test[i+1,10]- log_file_for_test[i,10]
    delta_mom_pos_tensor[i,3]= log_file_for_test[i+1,10]- log_file_for_test[i,11]
delta_mom_pos_tensor=delta_mom_pos_tensor/(delta_t_srd*box_vol)
stress_tensor_xx= kinetic_energy_tensor_xx[:,0] +delta_mom_pos_tensor[:,0]
stress_tensor_yy= kinetic_energy_tensor_yy[:,0] +delta_mom_pos_tensor[:,1]
stress_tensor_zz= kinetic_energy_tensor_zz[:,0] +delta_mom_pos_tensor[:,2]
shear_rate_term=(erate*delta_t_srd/2)*kinetic_energy_tensor_zz 

stress_tensor_xz=kinetic_energy_tensor_xz[:,0] + shear_rate_term[:,0] + delta_mom_pos_tensor[:,3]
viscosity=stress_tensor_xz/erate

plt.plot(log_file_for_test[1:,0],stress_tensor_xx[:])
plt.title("stress tensor xx vs collision steps ")
plt.show()
plt.plot(log_file_for_test[1:,0],stress_tensor_yy[:])
plt.title("stress tensor yy vs collision steps ")
plt.show()
plt.plot(log_file_for_test[1:,0],stress_tensor_zz[:])
plt.title("stress tensor zz vs collision steps ")
plt.show()
plt.plot(log_file_for_test[1:,0],delta_mom_pos_tensor[:,0])
plt.title("$\Delta p_{x}r_{x}$ vs timesteps ")
plt.show()
plt.plot(log_file_for_test[1:,0],delta_mom_pos_tensor[:,1])
plt.title("$\Delta p_{y}r_{y}$ vs timesteps ")
plt.show()
plt.plot(log_file_for_test[1:,0],delta_mom_pos_tensor[:,2])
plt.title("$\Delta p_{z}r_{z}$ vs timesteps ")
plt.show()
pressure_plot= (stress_tensor_xx+stress_tensor_yy +stress_tensor_zz)/3
plt.plot(log_file_for_test[1:,0],pressure_plot[:])
plt.title("$P$ vs timesteps ")
plt.show()
pressure = np.mean((stress_tensor_xx+stress_tensor_yy +stress_tensor_zz)/3)
print("Pressure",pressure)

#factor of 10000 out 

#%% reading a dump file 
Path_2_dump= Path_2_log
dump_start_line="ITEM: ATOMS id x y z vx vy vz"
dump_realisation_name= "testout_541709_2_60835_23.0_0.005071624521210362_10_10000_10000_500010_equilibrium_test.dump"
number_of_particles_per_dump =no_SRD

dump_file= dump2numpy_f(dump_start_line,Path_2_dump,dump_realisation_name,number_of_particles_per_dump)
dump_file_1=dump_file[0]
dump_file_2=dump_file[1]
dump_file_unsorted=dump_file[2]
# need to write a test to check this was done properly 
dump_file_shaped=np.reshape(dump_file_unsorted,(102,no_SRD,7))

#%% sorting rows 

dump_file_sorted=np.zeros((102,60835,7))

for i in range(0,102):
    for j in range(0,60835):
        id= int(float(dump_file_shaped[i,j,0]))-1
        dump_file_sorted[i,id,:]= dump_file_shaped[i,j,:]

# boolean to check order is correct 
comparison_list =np.arange(1,60836,1)
error_count=0
for i in range(0,102):

     boolean_result = dump_file_sorted[i,:,0]==comparison_list
     if np.all(boolean_result)==True:
         print("success")
     else:
        error_count=error_count+1
print(error_count)
     
#%% now can do calculation
box_vol=23**3
kinetic_energy_tensor=np.zeros((102,3))

for i in range(0,102):
    kinetic_energy_tensor[i,0]= np.sum(dump_file_sorted[i,:,4]* dump_file_sorted[i,:,4])/box_vol#xx
    kinetic_energy_tensor[i,1]= np.sum(dump_file_sorted[i,:,5]* dump_file_sorted[i,:,5])/box_vol#yy
    kinetic_energy_tensor[i,2]= np.sum(dump_file_sorted[i,:,6]* dump_file_sorted[i,:,6])/box_vol#zz

pressure= np.mean(np.mean(kinetic_energy_tensor,axis=1))






        


      
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
kinetic_energy_tensor_xz=np.array([2*log_file_for_test[1:10000,5]/box_vol ]).T# skip off the first value 
stress_tensor_xz = kinetic_energy_tensor_xz + kinetic_energy_tensor_zz +delta_mom_pos_tensor
stress_tensor_xz_rms_mean = np.sqrt(np.mean(stress_tensor_xz**2))





# %%
#plt.plot(log_file_for_test[1:,0],stress_tensor_xz[:,0])


plt.plot(log_file_for_test[2:,0],delta_mom_pos_tensor[:,0])
plt.title("$\Delta p_{x}r_{z}$ vs timesteps ")

plt.show()

plt.plot(log_file_for_test[2:,0],kinetic_energy_tensor_xz[:,0])
plt.title("Kinetic energy tensor xz vs collision steps ")

plt.show()

plt.plot(log_file_for_test[2:,0],kinetic_energy_tensor_zz[:,0])
plt.title("Kinetic energy tensor zz vs collision steps ")
plt.show()


plt.plot(log_file_for_test[2:,0],stress_tensor_xz[:,0])
plt.title("stress tensor xz vs collision steps ")
plt.show()


# %%
