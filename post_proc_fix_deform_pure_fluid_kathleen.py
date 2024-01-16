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
erate=0
no_timesteps=10010
# estimating number of steps  required
strain=3
delta_t_md=delta_t_srd/10
strain_rate= np.array([0.001,0.002,0.003,0.01,0.0005])

number_steps_needed= np.ceil(strain/(strain_rate*delta_t_md))
Path_2_log= "/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/"

#%% reading a dump file 
Path_2_dump= Path_2_log
dump_start_line="ITEM: ATOMS id x y z vx vy vz fx fy fz"
number_of_particles_per_dump = no_SRD


dump_realisation_name_before= "equilibrium_1234_2_60835_23.0_0.005071624521210362_10_gdot_0_10_10010_before_rotation.dump"
dump_file_before= dump2numpy_f(dump_start_line,Path_2_dump,dump_realisation_name_before,number_of_particles_per_dump)
dump_realisation_name_after= "equilibrium_1234_2_60835_23.0_0.005071624521210362_10_gdot_0_10_10010_after_rotation.dump"
dump_file_after= dump2numpy_f(dump_start_line,Path_2_dump,dump_realisation_name_after,number_of_particles_per_dump)

# saving for speed 
np.save("equilibrium_1234_2_60835_23.0_0.005071624521210362_10_gdot_0_10_10010_after_rotation.py",dump_file_after)
np.save("equilibrium_1234_2_60835_23.0_0.005071624521210362_10_gdot_0_10_10010_before_rotation.py",dump_file_before)

#%% finding all the dump files in a folder

VP_general_name_string='vel.*'

Mom_general_name_string='mom.*'

log_general_name_string='log.*'
                         #log.H20_no466188_wall_VACF_output_no_rescale_
TP_general_name_string='temp.*'

dump_general_name_string_after='*after*.dump'
dump_general_name_string_before='*before*.dump'

filepath="/Volumes/Backup\ Plus\ 1/PhD_/Rouse\ Model\ simulations/Using\ LAMMPS\ imac/KATHLEEN_LAMMPS_RUNS/equilibrium_fix_deform_pure_mpcd_results"

dump_realisation_name_info_before= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_before)
dump_realisation_name_info_after= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_after)


#%% reshaping 

dump_file_unsorted_before=dump_file_before[0]
dump_file_unsorted_after=dump_file_after[0]

columns= 10
# need to write a test to check this was done properly 
loop_size=1002
dump_file_shaped_before=np.reshape(dump_file_unsorted_before,(loop_size,number_of_particles_per_dump,columns))
loop_size=1001
dump_file_shaped_after=np.reshape(dump_file_unsorted_after,(loop_size,number_of_particles_per_dump,columns))
# freeing memory 
del dump_file_unsorted_before
del dump_file_unsorted_after
#%% sorting rows 

dump_file_sorted_before=np.zeros((loop_size,number_of_particles_per_dump,columns))
dump_file_sorted_after=np.zeros((loop_size,number_of_particles_per_dump,columns))

# for i in range(0,loop_size):
#     for j in range(0,number_of_particles_per_dump):
#         id= int(float(dump_file_shaped_before[i,j,0]))-1
#         dump_file_sorted_before[i,id,:]= dump_file_shaped_before[i,j,:]
# # for i in range(0,loop_size):
# #     for j in range(0,number_of_particles_per_dump):
        
#         id_1= int(float(dump_file_shaped_after[i,j,0]))-1
#         dump_file_sorted_after[i,id_1,:]= dump_file_shaped_after[i,j,:]


for i in range(0,loop_size):
    dump_file_sorted_before[i,:,:]=dump_file_shaped_before[dump_file_shaped_before[:, 0].argsort()]
    dump_file_sorted_after[i,:,:]=dump_file_shaped_after[dump_file_shaped_after[:, 0].argsort()]




# boolean to check order is correct 
comparison_list =np.arange(1,number_of_particles_per_dump+1,1)
error_count=0
for i in range(0,loop_size):

     boolean_result_1 = dump_file_sorted_before[i,:,0]==comparison_list
     if np.all(boolean_result_1)==True:
         print("success")
     else:
        error_count=error_count+1

     boolean_result_2 = dump_file_sorted_after[i,:,0]==comparison_list
     if np.all(boolean_result_2)==True:
         print("success")
     else:
        error_count=error_count+1

print(error_count)

# freeing memory 
del dump_file_shaped_after
del dump_file_shaped_before
     
#%% now can do calculation

kinetic_energy_tensor=np.zeros((loop_size,4))

for i in range(0,loop_size):
    kinetic_energy_tensor[i,0]= np.sum(dump_file_sorted_before[i,:,4]* dump_file_sorted_before[i,:,4])/box_vol#xx
    kinetic_energy_tensor[i,1]= np.sum(dump_file_sorted_before[i,:,5]* dump_file_sorted_before[i,:,5])/box_vol#yy
    kinetic_energy_tensor[i,2]= np.sum(dump_file_sorted_before[i,:,6]* dump_file_sorted_before[i,:,6])/box_vol#zz
    kinetic_energy_tensor[i,3]= np.sum(dump_file_sorted_before[i,:,4]* dump_file_sorted_before[i,:,6])/box_vol#xz


#%%
vel_magnitude_before_col=np.zeros((dump_file_sorted_after.shape[0]-1,no_SRD,1))    
delta_mom_pos_tensor_from_dump=np.zeros((dump_file_sorted_after.shape[0]-1,no_SRD,4))
mom_pos_tensor_from_dump=np.zeros((dump_file_sorted_after.shape[0]-1,no_SRD,4))

# for i in range(0,loop_size-1):
#     mom_pos_tensor_from_dump[i,:,0]= dump_file_sorted_after[i,:,4]*  dump_file_sorted_after[i,:,1]#xx
#     mom_pos_tensor_from_dump[i,:,1]= dump_file_sorted_after[i,:,5]*  dump_file_sorted_after[i,:,2]#yy
#     mom_pos_tensor_from_dump[i,:,2]= dump_file_sorted_after[i,:,6]* dump_file_sorted_after[i,:,3] #zz 
#     mom_pos_tensor_from_dump[i,:,3]= dump_file_sorted_after[i,:,4]*  dump_file_sorted_after[i,:,3] #xz

# mom_pos_tensor_from_dump=np.mean(mom_pos_tensor_from_dump,axis=1)



for i in range(1,loop_size-1):
    delta_mom_pos_tensor_from_dump[i,:,0]=(dump_file_sorted_after[i,:,4]- dump_file_sorted_before[i,:,4]) *  dump_file_sorted_after[i,:,1]#xx
    delta_mom_pos_tensor_from_dump[i,:,1]= (dump_file_sorted_after[i,:,5]- dump_file_sorted_before[i,:,5]) *  dump_file_sorted_after[i,:,2]#yy
    delta_mom_pos_tensor_from_dump[i,:,2]= (dump_file_sorted_after[i,:,6]- dump_file_sorted_before[i,:,6]) * dump_file_sorted_after[i,:,3] #zz 
    delta_mom_pos_tensor_from_dump[i,:,3]= (dump_file_sorted_after[i,:,4]- dump_file_sorted_before[i,:,4]) *  dump_file_sorted_after[i,:,3] #xz
    # delta_mom_pos_tensor_from_dump[i,1]= np.sum(dump_file_sorted[i+1,:,5]- dump_file_sorted[i,:,5] *  dump_file_sorted[i,:,2])#yy
    # delta_mom_pos_tensor_from_dump[i,2]= np.sum(dump_file_sorted[i+1,:,6]- dump_file_sorted[i,:,6] * dump_file_sorted[i,:,3]) #zz 
    # delta_mom_pos_tensor_from_dump[i,3]= np.sum((dump_file_sorted[i+1,:,4]- dump_file_sorted[i,:,4]) *  dump_file_sorted[i,:,3]) #xz
    # print(delta_mom_pos_tensor_from_dump[i,3]/(delta_t_srd*box_vol))
#delta_mom_pos_tensor_from_dump=delta_mom_pos_tensor_from_dump/(delta_t_srd*box_vol)
delta_mom_pos_tensor_from_dump_mean=np.mean(np.sum( delta_mom_pos_tensor_from_dump[1:,:],axis=1),axis=0)/(delta_t_srd*box_vol)

delta_mom_pos_tensor_from_dump_summed=np.sum( delta_mom_pos_tensor_from_dump[1:,:],axis=1)/(delta_t_srd*box_vol)

for i in range(1,loop_size-1):
    vel_magnitude_before_col[i,:,0]=(dump_file_sorted_after[i,:,4]**2 + dump_file_sorted_before[i,:,5]**2 + dump_file_sorted_before[i,:,6]**2)**0.5

summed_vel_magnitude_before_col=np.sum(vel_magnitude_before_col[:,1:],axis=1)
mean_summed_vel_magnitude= np.mean(summed_vel_magnitude_before_col[1:,:]) 

Pressure= mean_summed_vel_magnitude/(3*box_vol)


#%% checking momentum exchange sums to 0 for equilibrium only 

#NOTE: This isnt correct

delta_mom_from_dump=np.zeros((dump_file_sorted_after.shape[0]-1,no_SRD,3))

for i in range(1,loop_size-1):
    delta_mom_from_dump[i,:,0]=(dump_file_sorted_after[i,:,4]- dump_file_sorted_before[i,:,4])
    delta_mom_from_dump[i,:,1]= (dump_file_sorted_after[i,:,5]- dump_file_sorted_before[i,:,5])
    delta_mom_from_dump[i,:,2]= (dump_file_sorted_after[i,:,6]- dump_file_sorted_before[i,:,6])

delta_mom_from_dump_summed=np.sum(delta_mom_from_dump,axis=1)

# delta_mom_from_dump_abs_mean= np.mean(np.abs(delta_mom_from_dump), axis=1)


# # doesnt work but I think 
# standard_error_in_momentum=np.std(delta_mom_from_dump_summed, axis=0)
# realtive_error= standard_error_in_momentum/delta_mom_from_dump_abs_mean
# this is an error margin, I think the errror is propagated through the whole calculation    
    
#%% checking convergence of the sum 

delta_mom_pos_tensor_from_dump_partial_sum = np.zeros((loop_size,4))

for i in range(0,loop_size):
      delta_mom_pos_tensor_from_dump_partial_sum[i,:]=np.mean(delta_mom_pos_tensor_from_dump_summed[:i,:],axis=0)

labels=["dP_c,xx","dP_c,yy","dP_c,zz"]
for i in range(0,3):
    plt.plot(delta_mom_pos_tensor_from_dump_partial_sum[:,i],label=labels[i])
    plt.ylabel('$\\langle \Delta P_{coll} \\rangle_{T}$', rotation=0)
    plt.xlabel("$N_{coll}$")
    plt.ylim((None,1))
    plt.legend()

plt.show()


#%%

shear_rate_term_from_dump=(erate*delta_t_srd/2)*kinetic_energy_tensor[2:,2]
stress_tensor_xz_from_dump= kinetic_energy_tensor[2:,3]+ shear_rate_term_from_dump+ delta_mom_pos_tensor_from_dump_summed[:,3]
# # stress_tensor_xz_rms_mean_from_dump= np.sqrt(np.mean(stress_tensor_xz**2))
stress_tensor_xz_mean_from_dump= -np.mean(stress_tensor_xz_from_dump)
# viscosity_from_dump=stress_tensor_xz_mean_from_dump/erate

stress_tensor_xx_from_dump=( kinetic_energy_tensor[2:,0] +delta_mom_pos_tensor_from_dump_summed[:,0])
stress_tensor_yy_from_dump=( kinetic_energy_tensor[2:,1] +delta_mom_pos_tensor_from_dump_summed[:,1])
stress_tensor_zz_from_dump=( kinetic_energy_tensor[2:,2] +delta_mom_pos_tensor_from_dump_summed[:,2])

pressure_dump=-np.mean((stress_tensor_xx_from_dump+stress_tensor_yy_from_dump+stress_tensor_zz_from_dump)/3)

# # shear_rate_term=(erate*delta_t_srd/2)*kinetic_energy_tensor_zz 
# plt.plot(log_file_for_test[:,0],stress_tensor_xz[:])
# plt.show()

#viscosity_from_dump=stress_tensor_xz/erate
#plt.plot(log_file_for_test[:,0],viscosity[:])
#plt.show()


        


      

# %%
