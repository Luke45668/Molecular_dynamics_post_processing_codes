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


#%% key inputs 

no_SRD=60835
box_size=23
#nu_bar=3
#delta_t_srd=0.014872025172594354
#nu_bar=0.9 
delta_t_srd=0.05071624521210362
box_vol=box_size**3
erate=0.002
no_timesteps=2000010

# estimating number of steps  required
strain=3
delta_t_md=delta_t_srd/10
strain_rate= np.array([0.001,0.002,0.003,0.01,0.0005])

number_steps_needed= np.ceil(strain/(strain_rate*delta_t_md))

#%% importing one log file 

realisation_name = "log.testout_no806324_wall_pure_output_no_rescale_541709_2_60835_23.0_0.005071624521210362_10_10000_10000_2000010_T_1_lbda_0.05071624521210362_gdot_0.002" # with shear 
Path_2_log= "/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/"
thermo_vars="         KinEng          Temp          TotEng        c_pe[1]        c_pe[2]        c_pe[3]        c_pe[5]        c_mom[1]       c_mom[2]       c_mom[3]       c_mom[5]   "
log_file_for_test= log2numpy_reader(realisation_name,Path_2_log,thermo_vars)
#%% Check velocity profiles 
Path_2_VP=Path_2_log
chunk=20
VP_ave_freq=10000

VP_output_col_count=6 
equilibration_timesteps=0
realisation_name="vel.testout_pure_output_no806324_no_rescale_541709_2_60835_23.0_0.005071624521210362_10_10000_10000_295764_T_1_lbda_0.05071624521210362_gdot_0.002"
vel_profile_in= velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)
vel_profile_x=vel_profile_in[0]
vel_profile_z_data= vel_profile_in[1]
vel_profile_z_data=vel_profile_z_data.astype('float')*box_size

# plotting time series of vps
shear_rate_list =[]
for i in range(0,int(no_timesteps/VP_ave_freq)):
    shear_rate_list.append(scipy.stats.linregress(vel_profile_z_data[:],vel_profile_x[:,i]).slope)
    plt.plot(vel_profile_x[:,i],vel_profile_z_data[:])
plt.show()

# could do a standard error aswell and an R^2


#%% calculation from log file 
size_array=402
kinetic_energy_tensor_xx= np.array([(log_file_for_test[1:size_array,4]) ]).T
kinetic_energy_tensor_yy= np.array([(log_file_for_test[1:size_array,5])]).T
kinetic_energy_tensor_zz= np.array([(log_file_for_test[1:size_array,6])]).T
kinetic_energy_tensor_xz= np.array([(log_file_for_test[1:size_array,7])]).T


delta_mom_pos_tensor=np.zeros((log_file_for_test.shape[0]-1,4))
for i in range(1,log_file_for_test.shape[0]-1):
    delta_mom_pos_tensor[i,0]= log_file_for_test[i+1,8]- log_file_for_test[i,8]
    delta_mom_pos_tensor[i,1]= log_file_for_test[i+1,9]- log_file_for_test[i,9]
    delta_mom_pos_tensor[i,2]= log_file_for_test[i+1,10]- log_file_for_test[i,10]
    delta_mom_pos_tensor[i,3]= log_file_for_test[i+1,11]- log_file_for_test[i,11]
delta_mom_pos_tensor=delta_mom_pos_tensor/(delta_t_srd*box_vol)


stress_tensor_xx= kinetic_energy_tensor_xx[:,0] +delta_mom_pos_tensor[:,0]
stress_tensor_yy= kinetic_energy_tensor_yy[:,0] +delta_mom_pos_tensor[:,1]
stress_tensor_zz= kinetic_energy_tensor_zz[:,0] +delta_mom_pos_tensor[:,2]
shear_rate_term=(erate*delta_t_srd/2)*kinetic_energy_tensor_zz 

stress_tensor_xz=kinetic_energy_tensor_xz[:,0] +shear_rate_term[:,0] +delta_mom_pos_tensor[:,3]
stress_tensor_xz_rms_mean = np.sqrt(np.mean(stress_tensor_xz**2))
stress_tensor_xz_mean= np.mean(stress_tensor_xz[2:])
viscosity_from_log=stress_tensor_xz_mean/erate


plt.plot(log_file_for_test[1:,0],stress_tensor_xx[:])
plt.title("stress tensor xx vs collision steps ")
plt.show()
plt.plot(log_file_for_test[1:,0],stress_tensor_yy[:])
plt.title("stress tensor yy vs collision steps ")
plt.show()
plt.plot(log_file_for_test[1:,0],stress_tensor_zz[:])
plt.title("stress tensor zz vs collision steps ")
plt.show()
plt.plot(log_file_for_test[1:,0],stress_tensor_xz[:])
plt.title("stress tensor xz vs collision steps ")
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
dump_realisation_name= "testout_541709_2_60835_23.0_0.0014872025172594354_10_gdot_0.003_10000_1008606_equilibrium_test.dump"
number_of_particles_per_dump = no_SRD

dump_file= dump2numpy_f(dump_start_line,Path_2_dump,dump_realisation_name,number_of_particles_per_dump)
#%% reshaping 
dump_file_1=dump_file[0]
dump_file_2=dump_file[1]
dump_file_unsorted=dump_file[2]

loop_size=36
# need to write a test to check this was done properly 
dump_file_shaped=np.reshape(dump_file_unsorted,(loop_size,number_of_particles_per_dump,7))

#%% sorting rows 

dump_file_sorted=np.zeros((loop_size,number_of_particles_per_dump,7))

for i in range(0,loop_size):
    for j in range(0,number_of_particles_per_dump):
        id= int(float(dump_file_shaped[i,j,0]))-1
        dump_file_sorted[i,id,:]= dump_file_shaped[i,j,:]

# boolean to check order is correct 
comparison_list =np.arange(1,number_of_particles_per_dump+1,1)
error_count=0
for i in range(0,loop_size):

     boolean_result = dump_file_sorted[i,:,0]==comparison_list
     if np.all(boolean_result)==True:
         print("success")
     else:
        error_count=error_count+1
print(error_count)
     
#%% now can do calculation

kinetic_energy_tensor=np.zeros((loop_size,4))

for i in range(0,loop_size):
    kinetic_energy_tensor[i,0]= np.sum(dump_file_sorted[i,:,4]* dump_file_sorted[i,:,4])/box_vol#xx
    kinetic_energy_tensor[i,1]= np.sum(dump_file_sorted[i,:,5]* dump_file_sorted[i,:,5])/box_vol#yy
    kinetic_energy_tensor[i,2]= np.sum(dump_file_sorted[i,:,6]* dump_file_sorted[i,:,6])/box_vol#zz
    kinetic_energy_tensor[i,3]= np.sum(dump_file_sorted[i,:,4]* dump_file_sorted[i,:,6])/box_vol#xz

pressure= np.mean(np.mean(kinetic_energy_tensor[:,0:2],axis=1))

#%%
delta_mom_pos_tensor_from_dump=np.zeros((loop_size,4))
for i in range(0,loop_size-1):
    delta_mom_pos_tensor_from_dump[i,0]= np.sum((dump_file_sorted[i+1,:,4]- dump_file_sorted[i,:,4]) *  dump_file_sorted[i,:,1])#xx
    delta_mom_pos_tensor_from_dump[i,1]= np.sum(dump_file_sorted[i+1,:,5]- dump_file_sorted[i,:,5] *  dump_file_sorted[i,:,2])#yy
    delta_mom_pos_tensor_from_dump[i,2]= np.sum(dump_file_sorted[i+1,:,6]- dump_file_sorted[i,:,6] * dump_file_sorted[i,:,3]) #zz 
    delta_mom_pos_tensor_from_dump[i,3]= np.sum((dump_file_sorted[i+1,:,4]- dump_file_sorted[i,:,4]) *  dump_file_sorted[i,:,3]) #xz
delta_mom_pos_tensor_from_dump=delta_mom_pos_tensor_from_dump/(delta_t_srd*box_vol)
#%%
stress_tensor_xz= kinetic_energy_tensor[:,3]+ (erate*delta_t_srd/2)*kinetic_energy_tensor[:,2]  + delta_mom_pos_tensor_from_dump[:,3]
stress_tensor_xz_rms_mean_from_dump= np.sqrt(np.mean(stress_tensor_xz**2))
stress_tensor_xz_mean_from_dump= np.mean(stress_tensor_xz)
# # stress_tensor_xx= kinetic_energy_tensor_xx[:,0] +delta_mom_pos_tensor_from_dump[:,0]
# # stress_tensor_yy= kinetic_energy_tensor_yy[:,0] +delta_mom_pos_tensor_from_dump[:,1]
# # stress_tensor_zz= kinetic_energy_tensor_zz[:,0] +delta_mom_pos_tensor_from_dump[:,2]
# # shear_rate_term=(erate*delta_t_srd/2)*kinetic_energy_tensor_zz 
# plt.plot(log_file_for_test[:,0],stress_tensor_xz[:])
# plt.show()

viscosity_from_dump=stress_tensor_xz_mean_from_dump/erate
#plt.plot(log_file_for_test[:,0],viscosity[:])
#plt.show()



        


      
