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
import pickle as pck

#%% 
# damp 0.1 seems to work well, need to add window averaging, figure out how to get imposed shear to match velocity of particle
# neeed to then do a huge run 
damp=0.05
strain_total=300

path_2_log_files='/Users/luke_dev/Documents/simulation_test_folder/damp_'+str(damp)


erate=np.array([0.02,0.0175,0.015,0.0125,0.01])

# #300 strain 
no_timesteps=np.array([ 59150000,  67600000 , 78870000,  94640000,118310000])
# no_timesteps=np.array([52580000,  57360000,  63096000,  70107000,  78870000,  90137000,
#        105160000, 126192000, 157740000, 210321000])
thermo_freq=10000

lgf_row_count=np.ceil((no_timesteps/thermo_freq )).astype("int")

thermo_vars='         KinEng         PotEng          Temp          c_bias         TotEng    '
j_=1
K=20
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 

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
          realisation_index_=int(i.split('_')[first_sort_index])
          data_set =i.split('_')[second_sort_index]
          split_list_for_sorting.append(realisation(i,data_set,realisation_index_))


    realisation_name_sorted=sorted(split_list_for_sorting,
                                                key=lambda x: ( x.data_set,x.realisation_index_),reverse=True)
    realisation_name_sorted_final=[]
    for i in realisation_name_sorted:
          realisation_name_sorted_final.append(i.realisation_full_str)
    
    return realisation_name_sorted_final

def folder_check_or_create(filepath,folder):
     os.chdir(filepath)
     # combine file name with wd path
     check_path=filepath+"/"+folder
     print((check_path))
     if os.path.exists(check_path) == 1:
          print("file exists, proceed")
          os.chdir(check_path)
     else:
          print("file does not exist, making new directory")
          os.chdir(filepath)
          os.mkdir(folder)
          os.chdir(filepath+"/"+folder)
  


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

     


      
     






#%% when looking at very small number of files 

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
#%%
strainplot_tuple=()
for i in range(erate.size):
     strain_unit=strain_total/lgf_row_count[i]
     strain_plotting_points= np.linspace(0,strain_total, spring_force_positon_tensor_tuple[i].shape[0])
   
     strainplot_tuple=strainplot_tuple+(strain_plotting_points,)  
     print(strainplot_tuple[i].size)

#%% compute erate prediction 


# for i in range(erate.size):
      
#       #plt.plot(COM_velocity_tuple[i][:-1,0],label="COM",linestyle=':')
#       #print(np.mean(COM_velocity_tuple[i][:,0]))
#      # plt.axhline(np.mean(COM_velocity_tuple[i][:-1,0]), label="COM mean",linestyle=':')
#       plt.plot(strainplot_tuple[i][:-1],erate_velocity_tuple[i][:-1,0],label="erate prediction",linestyle='--')
#       plt.axhline(np.mean(erate_velocity_tuple[i][:-1,0]), label="erate mean",linestyle=':')
#       #print("error:",np.mean(erate_velocity_tuple[i][:,0])-np.mean(COM_velocity_tuple[i][:,0]))
#       plt.legend()
#       plt.show()

for i in range(erate.size):
      
      #plt.plot(strainplot_tuple[i][:-1],COM_velocity_tuple[i][:-1,0],label="COM",linestyle='--')
      #print(np.mean(COM_velocity_tuple[i][:,0]))
      plt.axhline(np.mean(COM_velocity_tuple[i][:-1,0]), label="COM mean",linestyle=':',color='blueviolet')
      #plt.plot(erate_velocity_tuple[i][:-1,0],label="erate prediction",linestyle='--')
      plt.axhline(np.mean(erate_velocity_tuple[i][:-1,0]), label="erate mean",linestyle='--',color='black')
      #print("error:",np.mean(erate_velocity_tuple[i][:,0])-np.mean(COM_velocity_tuple[i][:,0]))
      plt.legend()
      plt.show()


#%% look at internal stresses
folder="stress_tensor_plots"
folder_check_or_create(path_2_log_files,folder)
labels_stress=["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{yz}$"]

for i in range(erate.size):
     for j in range(3,6):
        plt.plot(strainplot_tuple[i],spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
        plt.legend()
     plt.show()

for i in range(erate.size):
     for j in range(0,3):
        plt.plot(strainplot_tuple[i],spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
        plt.legend()
     plt.show()

#%%
cutoff_ratio=0.75
folder="N_1_plots"

N_1_mean=np.zeros((erate.size))
for i in range(erate.size):
    
    N_1=spring_force_positon_tensor_tuple[i][:-1,0]-spring_force_positon_tensor_tuple[i][:-1,2]
    cutoff=int(np.ceil(cutoff_ratio*N_1.size))
    N_1_mean[i]=np.mean(N_1[cutoff:])
    plt.plot(strainplot_tuple[i][:-1],N_1, label=labels_stress[j])
    plt.axhline(np.mean(N_1))
    plt.ylabel("$N_{1}$")
    plt.legend()
    plt.show()

#%%
folder="N_2_plots"

N_2_mean=np.zeros((erate.size))
for i in range(erate.size):
    N_2= spring_force_positon_tensor_tuple[i][:-1,2]-spring_force_positon_tensor_tuple[i][:-1,1]
    cutoff=int(np.ceil(cutoff_ratio*N_2.size))
    N_2_mean[i]=np.mean(N_2[cutoff:])
    plt.plot(strainplot_tuple[i][:-1],N_2)
    plt.axhline(np.mean(N_2), label="$\\bar{N}_{2}$")
    plt.ylabel("$N_{2}$")
    plt.legend()
    plt.show()

#%%
folder="shear_stress_plots"

xz_shear_stress_mean=np.zeros((erate.size))
for i in range(erate.size):
    
    xz_shear_stress= spring_force_positon_tensor_tuple[i][:,3]
    cutoff=int(np.ceil(cutoff_ratio*xz_shear_stress.size))
    xz_shear_stress_mean[i]=np.mean(xz_shear_stress[cutoff:])
    plt.plot(strainplot_tuple[i][:],xz_shear_stress, label=labels_stress[3])
    plt.axhline(xz_shear_stress_mean[i])
    plt.ylabel("$\sigma_{xz}$")
    plt.legend()
    plt.show()

#%%
# could add fittings to this run 
from scipy.optimize import curve_fit
 
def quadfunc(x, a):

    return a*(x**2)
plt.scatter(erate,N_1_mean)
popt,pcov=curve_fit(quadfunc,erate,N_1_mean)
plt.plot(erate,(popt[0]*(erate**2)))
plt.show()

#%%
plt.scatter(erate,N_2_mean)
popt,pcov=curve_fit(quadfunc,erate,N_2_mean)
plt.plot(erate,(popt[0]*(erate**2)))
plt.show()
#%%
plt.scatter(erate,xz_shear_stress_mean)
plt.show()
# %% plot
column=3
for i in range(erate.size):
    plt.plot(log_file_tuple[i][:,column])
    mean_temp=np.mean(log_file_tuple[i][:,column])
    plt.axhline(np.mean(log_file_tuple[i][:,column]))
    print(mean_temp)
    
    plt.show()


# %%
