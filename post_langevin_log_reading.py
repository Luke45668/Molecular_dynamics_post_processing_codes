##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file processes the log files from brownian dynamics simulations 

after an MPCD simulation. 
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
import mmap
import h5py as h5

path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
os.chdir(path_2_post_proc_module)

from log2numpy import *
from dump2numpy import *
import glob 
from post_MPCD_MP_processing_module import *

#%% 
# damp 0.1 seems to work well, need to add window averaging, figure out how to get imposed shear to match velocity of particle
# neeed to then do a huge run 


path_2_log_files='/Users/luke_dev/Documents/simulation_test_folder/damp_0.1'
erate= np.array([0.02,0.0175,0.015,0.0125,0.01]) 
K=20

pol_general_name_string='*pol*h5'

phantom_general_name_string='*phantom*h5'

Mom_general_name_string='mom.*'

log_general_name_string='log.*'

dump_general_name_string='*dump'




(realisation_name_Mom,
 realisation_name_phantom,
 count_mom,count_phantom,
 realisation_name_log,
 count_log,
 realisation_name_dump,
 count_dump,
 realisation_name_pol,
 count_pol)= VP_and_momentum_data_realisation_name_grabber(pol_general_name_string,
                                                                     log_general_name_string,
                                                                     phantom_general_name_string,
                                                                     Mom_general_name_string,
                                                                     path_2_log_files,
                                                                     dump_general_name_string)




class realisation():
     def __init__(self,realisation_full_str,data_set,realisation_index_):
          self.realisation_full_str= realisation_full_str
          self.data_set= data_set
          self.realisation_index_=realisation_index_
     def __repr__(self):
        return '({},{},{})'.format(self.realisation_full_str,self.data_set,self.realisation_index_)
realisations_for_sorting_after_srd=[]
realisation_split_index=6
erate_index=15

def org_names(split_list_for_sorting,unsorted_list,first_sort_index,second_sort_index):
    for i in unsorted_list:
          realisation_index_=i.split('_')[first_sort_index]
          data_set =i.split('_')[second_sort_index]
          split_list_for_sorting.append(realisation(i,data_set,realisation_index_))


    realisation_name_sorted=sorted(split_list_for_sorting,
                                                key=lambda x: ( x.realisation_index_,x.data_set))
    realisation_name_sorted_final=[]
    for i in realisation_name_sorted:
          realisation_name_sorted_final.append(i.realisation_full_str)
    
    return realisation_name_sorted_final


#%%
realisations_for_sorting_after_pol=[]
realisation_name_h5_after_sorted_final_pol=org_names(realisations_for_sorting_after_pol,
                                                     realisation_name_pol,
                                                     realisation_split_index,
                                                     erate_index)

realisations_for_sorting_after_phantom=[]
realisation_name_h5_after_sorted_final_phantom=org_names(realisations_for_sorting_after_phantom,
                                                     realisation_name_phantom,
                                                     realisation_split_index,
                                                     erate_index)
realisations_for_sorting_after_log=[]
realisation_name_log_sorted_final=org_names(realisations_for_sorting_after_log,
                                                     realisation_name_log,
                                                     realisation_split_index,
                                                     erate_index)

     




#%%read the  log files 

thermo_vars='         KinEng         PotEng          Temp          c_bias         TotEng    '

# should prbably add window averaging 
count=0
log_file_tuple=()
for i in range(0,count_log):
    print(realisation_name_log_sorted_final[i])
    count=count+1
    print(count)
    log_file_tuple=log_file_tuple+(log2numpy_reader(realisation_name_log_sorted_final[i],
                                                    path_2_log_files,
                                                    thermo_vars),)

eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
count=0
pol_velocities_tuple=()
COM_velocity_tuple=()
COM_position_tuple=()
erate_velocity_tuple=()
averaged_z_tuple=()
spring_force_positon_tensor_tuple=()

for i in range(0,count_pol):
    print(realisation_name_h5_after_sorted_final_pol[i])
    with h5.File(realisation_name_h5_after_sorted_final_pol[i],'r') as f_c:
        with h5.File(realisation_name_h5_after_sorted_final_phantom[i],'r') as f_ph:

            outputdim=f_c['particles']['small']['velocity']['value'].shape[0]
            pol_velocities_array=np.zeros((outputdim,3,3))
            pol_positions_array=np.zeros((outputdim,3,3))
            COM_velocity=np.zeros((outputdim,3))
            COM_position=np.zeros((outputdim,3))
            averaged_z_array=np.zeros((outputdim,1))
            spring_force_positon_array=np.zeros((outputdim,6))
            erate_velocity_prediciton=np.zeros((outputdim,1))

            for j in range(0,outputdim):
                pol_positions_array[j,:,:]=f_c['particles']['small']['position']['value'][j]
                pol_velocities_array[j,:,:]=f_c['particles']['small']['velocity']['value'][j]
                COM_velocity[j,:]=np.mean(pol_velocities_array[j,:,:],axis=0)
                COM_position[j,:]=np.mean(pol_positions_array[j,:,:],axis=0)
                erate_velocity_prediciton[j,0]=COM_position[j,2]*erate[i] # this works for now 



                pol_positions_after=f_c['particles']['small']['position']['value'][j]
                phantom_positions_after=f_ph['particles']['phantom']['position']['value'][j]
                averaged_z_array[j,0]=np.mean(pol_velocities_array[j,:,2])

                f_spring_1_dirn=pol_positions_after[0,:]-phantom_positions_after[1,:]
                f_spring_1_mag=np.sqrt(np.sum((f_spring_1_dirn)**2))
                f_spring_1=K*(f_spring_1_dirn/f_spring_1_mag)*(f_spring_1_mag-eq_spring_length)
                # spring 2
                f_spring_2_dirn=pol_positions_after[1,:]-phantom_positions_after[2,:]
                f_spring_2_mag=np.sqrt(np.sum((f_spring_2_dirn)**2))
                f_spring_2=K*(f_spring_2_dirn/f_spring_2_mag)*(f_spring_2_mag-eq_spring_length)
                # spring 3
                f_spring_3_dirn=pol_positions_after[2,:]-phantom_positions_after[0,:]
                f_spring_3_mag=np.sqrt(np.sum((f_spring_3_dirn)**2))
                f_spring_3=K*(f_spring_3_dirn/f_spring_3_mag)*(f_spring_3_mag-eq_spring_length)

                spring_force_positon_tensor_xx=f_spring_1[0]*f_spring_1_dirn[0] + f_spring_2[0]*f_spring_2_dirn[0] +f_spring_3[0]*f_spring_3_dirn[0] 
                spring_force_positon_tensor_yy=f_spring_1[1]*f_spring_1_dirn[1] + f_spring_2[1]*f_spring_2_dirn[1] +f_spring_3[1]*f_spring_3_dirn[1] 
                spring_force_positon_tensor_zz=f_spring_1[2]*f_spring_1_dirn[2] + f_spring_2[2]*f_spring_2_dirn[2] +f_spring_3[2]*f_spring_3_dirn[2] 
                spring_force_positon_tensor_xz=f_spring_1[0]*f_spring_1_dirn[2] + f_spring_2[0]*f_spring_2_dirn[2] +f_spring_3[0]*f_spring_3_dirn[2] 
                spring_force_positon_tensor_xy=f_spring_1[0]*f_spring_1_dirn[1] + f_spring_2[0]*f_spring_2_dirn[1] +f_spring_3[0]*f_spring_3_dirn[1] 
                spring_force_positon_tensor_yz=f_spring_1[1]*f_spring_1_dirn[2] + f_spring_2[1]*f_spring_2_dirn[2] +f_spring_3[1]*f_spring_3_dirn[2] 
              
                
                np_array_spring_pos_tensor=np.array([spring_force_positon_tensor_xx,
                                                    spring_force_positon_tensor_yy,
                                                    spring_force_positon_tensor_zz,
                                                    spring_force_positon_tensor_xz,
                                                    spring_force_positon_tensor_xy,
                                                    spring_force_positon_tensor_yz, 
                                                     ])
                spring_force_positon_array[j,:]= np_array_spring_pos_tensor


    
    pol_velocities_tuple=pol_velocities_tuple+(pol_velocities_array,)
    averaged_z_tuple=averaged_z_tuple+(averaged_z_array,)
    spring_force_positon_tensor_tuple=spring_force_positon_tensor_tuple+(spring_force_positon_array,)
    COM_velocity_tuple=COM_velocity_tuple+(COM_velocity,)
    COM_position_tuple=COM_position_tuple+(COM_position,)
    erate_velocity_tuple=erate_velocity_tuple+(erate_velocity_prediciton,)

# should now look at the COM z coordinate and compare it to the velocity profile at that z value

#%% compute erate prediction 


for i in range(erate.size):
      
      #plt.plot(COM_velocity_tuple[i][:,0])
      print(np.mean(COM_velocity_tuple[i][:,0]))
      plt.axhline(np.mean(COM_velocity_tuple[i][:,0]), label="COM mean",linestyle=':')
      #plt.plot(erate_velocity_tuple[i][:,0])
      plt.axhline(np.mean(erate_velocity_tuple[i][:,0]), label="VP mean",linestyle='--')
      print("error:",np.mean(erate_velocity_tuple[i][:,0])-np.mean(COM_velocity_tuple[i][:,0]))
      plt.legend()
      plt.show()


#%% look at internal stresses
labels_stress=["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{yz}$"]

for i in range(erate.size):
     for j in range(3,6):
        plt.plot(spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
        plt.legend()
     plt.show()

for i in range(erate.size):
     for j in range(0,3):
        plt.plot(spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
        plt.legend()
     plt.show()

#%%
N_1_mean=np.zeros((5))
for i in range(erate.size):
    N_1=spring_force_positon_tensor_tuple[i][:,0]-spring_force_positon_tensor_tuple[i][:,2]
    N_1_mean[i]=np.mean(N_1)
    plt.plot(N_1, label=labels_stress[j])
    plt.axhline(np.mean(N_1))
    plt.ylabel("$N_{1}$")
    plt.legend()
    plt.show()

#%%
N_2_mean=np.zeros((5))
for i in range(erate.size):
    N_2= spring_force_positon_tensor_tuple[i][:,2]-spring_force_positon_tensor_tuple[i][:,1]
    N_2_mean[i]=np.mean(N_2)
    plt.plot(N_2, label=labels_stress[j])
    plt.axhline(np.mean(N_2))
    plt.ylabel("$N_{2}$")
    plt.legend()
    plt.show()

#%%
# could add fittings to this run 
plt.plot(erate,N_1_mean)

plt.show()

plt.plot(erate,N_2_mean)
plt.show()
# %% plot
column=3
for i in range(0,count_log):
    plt.plot(log_file_tuple[i][:,column])
    mean_temp=np.mean(log_file_tuple[i][:,column])
    plt.axhline(np.mean(log_file_tuple[i][:,column]))
    print(mean_temp)
    
    plt.show()


# %%
