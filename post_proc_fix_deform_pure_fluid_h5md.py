##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will calculate the MPCD stress tensor for a pure fluid under forward NEMD using hdf5 files 
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
import h5py as h5 


path_2_post_proc_module= '/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/LAMMPS python run and analysis scripts/Analysis codes'
#os.chdir(path_2_post_proc_module)
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

#%% key inputs 

no_SRD=506530
box_size=37
# no_SRD=2160
# box_size=6
#nu_bar=3
#delta_t_srd=0.014872025172594354
#nu_bar=0.9 
delta_t_srd=0.05674857690605889

box_vol=box_size**3
erate=0
no_timesteps=10000
# estimating number of steps  required
strain=3
delta_t_md=delta_t_srd/10
strain_rate= np.array([0.001,0.002,0.003,0.01,0.0005])
number_steps_needed= np.ceil(strain/(strain_rate*delta_t_md))
dump_freq=10

#%% finding all the dump files in a folder

VP_general_name_string='vel.*'

Mom_general_name_string='mom.*'

log_general_name_string='log.*'
                         #log.H20_no466188_wall_VACF_output_no_rescale_
TP_general_name_string='temp.*'

dump_general_name_string_after='*'+str(no_timesteps)+'*after*.h5'
dump_general_name_string_before='*'+str(no_timesteps)+'*before*.h5'

filepath="/KATHLEEN_LAMMPS_RUNS/equilibrium_fix_deform_pure_mpcd_test_file"
filepath="/Users/lukedebono/Documents/test_analysis_small_equilibrium_test_box_"+str(int(box_size))+"_M_10"
Path_2_dump=filepath
# can chnage this to another array on kathleen


dump_realisation_name_info_before= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_before)
realisation_name_VP=dump_realisation_name_info_before[1]
count_mom=dump_realisation_name_info_before[2]
count_VP=dump_realisation_name_info_before[3]
realisation_name_log=dump_realisation_name_info_before[4]
count_log=dump_realisation_name_info_before[5]
realisation_name_h5_before=dump_realisation_name_info_before[6]
count_h5_before=dump_realisation_name_info_before[7]

dump_realisation_name_info_after= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_after)

realisation_name_h5_after=dump_realisation_name_info_after[6]
count_h5_after=dump_realisation_name_info_after[7]

if count_h5_before==count_h5_after:
    print("consistent dump file count")
else:
    print("inconsistent dump file count, Please check")
# %%

# find dump file size
with h5.File(realisation_name_h5_after[0], 'r') as f:
    shape_after= f['particles']['SRDs']['position']['value'].shape
    print(f['particles']['SRDs']['species'].keys())
with h5.File(realisation_name_h5_before[0], 'r') as f:
    shape_before= f['particles']['SRDs']['position']['value'].shape

if shape_after==shape_before:
    print("shapes match, no need to check")
else:
    print("shapes do not match check if code is still correct")
#%%
j_=4
delta_mom_summed= np.zeros((j_,shape_before[0]-1,3))
delta_mom_pos_tensor_summed= np.zeros((j_,shape_before[0]-1,9))

for k in range(0,j_):
    for j in range(1,shape_before[0]-1):
    #for j in range(0,10):
        with h5.File(realisation_name_h5_after[k], 'r') as f_a:
           #with h5.File(realisation_name_h5_before[k], 'r') as f_b:

                SRD_positions_initial= f_a['particles']['SRDs']['position']['value'][j-1]
                
                SRD_positions_after= f_a['particles']['SRDs']['position']['value'][j]
                #print(SRD_positions_after)

                SRD_velocities_initial=f_a['particles']['SRDs']['velocity']['value'][j-1]
                SRD_velocities_after=f_a['particles']['SRDs']['velocity']['value'][j]
                #print(SRD_velocities_after)
                delta_mom=SRD_velocities_after-SRD_velocities_initial
                delta_mom_summed[k,j,:]=np.sum(SRD_velocities_after-SRD_velocities_initial,axis=0)



                
                delta_mom_pos_tensor_summed[k,j,0]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#xx
                delta_mom_pos_tensor_summed[k,j,1]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#yy
                delta_mom_pos_tensor_summed[k,j,2]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#zz
                delta_mom_pos_tensor_summed[k,j,3]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#xz
                delta_mom_pos_tensor_summed[k,j,4]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#xy
                delta_mom_pos_tensor_summed[k,j,5]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#yz
                delta_mom_pos_tensor_summed[k,j,6]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#zx
                delta_mom_pos_tensor_summed[k,j,7]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#zy
                delta_mom_pos_tensor_summed[k,j,8]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#yx

                
                
#%%

                      
        

delta_mom_pos_tensor_summed_mean=np.mean(delta_mom_pos_tensor_summed,axis=1)

delta_mom_pos_tensor_summed_realisation_mean=np.mean(delta_mom_pos_tensor_summed_mean,axis=0)

np.save("delta_mom_pos_tensor_summed_realisation_mean_M_10_L_"+str(box_size)+".py",delta_mom_pos_tensor_summed_realisation_mean)





                

         
        

   


  

# %%
