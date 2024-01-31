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
#from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import h5py as h5 
import multiprocessing as mp


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
# no_SRD=1038230
# box_size=47
# no_SRD=506530
# box_size=37
no_SRD=121670
box_size=23
no_SRD=270
box_size=3
# no_SRD=58320
# box_size=18
# no_SRD=2160
# box_size=6
# no_SRD=2560
# box_size=8
#nu_bar=3
#delta_t_srd=0.014872025172594354
#nu_bar=0.9 
delta_t_srd=0.05674857690605889

box_vol=box_size**3
erate= np.array([0.001,0.002,0.003])
no_timesteps=50000
# estimating number of steps  required
strain=3
delta_t_md=delta_t_srd/10
strain_rate= np.array([0.001,0.002,0.003])
number_steps_needed= np.ceil(strain/(strain_rate*delta_t_md))
dump_freq=10
rho=10 
#rho=5
realisation_index=np.array([1,2,3])
#%% finding all the dump files in a folder

VP_general_name_string='vel.*'

Mom_general_name_string='mom.*'

log_general_name_string='log.*'
                         #log.H20_no466188_wall_VACF_output_no_rescale_
TP_general_name_string='temp.*'

dump_general_name_string_after='*'+str(no_timesteps)+'*after*.h5'
dump_general_name_string_before='*'+str(no_timesteps)+'*before*.h5'

filepath="/KATHLEEN_LAMMPS_RUNS/equilibrium_fix_deform_pure_mpcd_test_file"
filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/hfd5_runs/non_equilibrium_tests/2dumps/test_non_eq_box_"+str(int(box_size))+"_M_"+str(rho)
#filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/hfd5_runs/tests_equilibrium_with_more_regular_neighbour_listing_box_"+str(int(box_size))+"_M_10"
Path_2_dump=filepath
# can chnage this to another array on kathleen
dump_realisation_name_info_before=VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_before)
realisation_name_h5_before=dump_realisation_name_info_before[6]
count_h5_before=dump_realisation_name_info_before[7]

dump_realisation_name_info_after= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_after)

realisation_name_h5_after=dump_realisation_name_info_after[6]
count_h5_after=dump_realisation_name_info_after[7]
#%% should check the time series of velocity profiles aswell. 

# find dump file size
with h5.File(realisation_name_h5_after[0], 'r') as f:
    shape_after= f['particles']['SRDs']['position']['value'].shape
    print(f['particles']['SRDs']['position']['step'].shape)
#%%
with h5.File(realisation_name_h5_before[1], 'r') as f_i:
    # first_step= f_i['particles']
    # print(first_step)
    print(f_i['particles']['SRDs']['position']['step'].shape)
#%% reorganising the list of realisations 

realisation_name_h5_after_sorted=['0']*9
realisation_name_h5_before_sorted=['0']*9
for i in range(9):
     realisation_index_=int(np.where(realisation_index==float(realisation_name_h5_after[i].split('_')[9]))[0][0])

     print(realisation_index_)
     data_set = int(np.where(erate==float(realisation_name_h5_after[i].split('_')[15]))[0][0])
     print(data_set)
     if data_set==0:
        realisation_name_h5_after_sorted[realisation_index_]=realisation_name_h5_after[i]
     elif data_set==1:
        realisation_name_h5_after_sorted[3+realisation_index_]=realisation_name_h5_after[i]
     else:
        realisation_name_h5_after_sorted[6+realisation_index_]=realisation_name_h5_after[i]
for i in range(9):
     realisation_index_=int(np.where(realisation_index==float(realisation_name_h5_before[i].split('_')[9]))[0][0])
     
     print(realisation_index_)
     data_set = int(np.where(erate==float(realisation_name_h5_before[i].split('_')[15]))[0][0])
     print(data_set)
     if data_set==0:
        realisation_name_h5_before_sorted[realisation_index_]=realisation_name_h5_before[i]
     elif data_set==1:
        realisation_name_h5_before_sorted[3+realisation_index_]=realisation_name_h5_before[i]
     else:
        realisation_name_h5_before_sorted[6+realisation_index_]=realisation_name_h5_before[i]
          
realisation_name_h5_before=realisation_name_h5_before_sorted
realisation_name_h5_after=realisation_name_h5_after_sorted   



#%%
# this needs to be changed back to the old version where we looked at file N and N-1, since the shear could  change things in the collision step

# need to look into adding multi-processing to this section of the code
j_=3
no_data_sets=erate.shape[0]
delta_mom_summed= np.zeros((no_data_sets,j_,shape_after[0]-1,3))
delta_mom_pos_tensor_summed= np.zeros((no_data_sets,j_,shape_after[0]-1,9))
stress_tensor_summed=np.zeros((no_data_sets,j_,shape_after[0]-1,9))
kinetic_energy_tensor_summed=np.zeros((no_data_sets,j_,shape_after[0]-1,6))

for i in range(0,len(realisation_name_h5_after)):
    with h5.File(realisation_name_h5_after[i], 'r') as f_a:
        with h5.File(realisation_name_h5_before[i], 'r') as f_b:
            data_set = np.where(erate==float(realisation_name_h5_after[i].split('_')[15]))[0][0]
            k=np.where(realisation_index==float(realisation_name_h5_after[i].split('_')[9]))[0][0]
            for j in range(1,shape_after[0]-1):
    #for j in range(0,10):
       
           
                data_set = np.where(erate==float(realisation_name_h5_after[i].split('_')[15]))[0][0]
                k=np.where(realisation_index==float(realisation_name_h5_after[i].split('_')[9]))[0][0]
                SRD_positions_initial= f_b['particles']['SRDs']['position']['value'][j-1]
                
                SRD_positions_after= f_a['particles']['SRDs']['position']['value'][j]
                #print(SRD_positions_after)

                SRD_velocities_initial=f_b['particles']['SRDs']['velocity']['value'][j-1]
                SRD_velocities_after=f_a['particles']['SRDs']['velocity']['value'][j]
                #print(SRD_velocities_after)
                delta_mom=SRD_velocities_after-SRD_velocities_initial
                delta_mom_summed[data_set,k,j,:]=np.sum(SRD_velocities_after-SRD_velocities_initial,axis=0)

                # need to add kinetic contribution

                
                delta_mom_pos_tensor_summed[data_set,k,j,0]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#xx
                delta_mom_pos_tensor_summed[data_set,k,j,1]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#yy
                delta_mom_pos_tensor_summed[data_set,k,j,2]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#zz
                delta_mom_pos_tensor_summed[data_set,k,j,3]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#xz
                delta_mom_pos_tensor_summed[data_set,k,j,4]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#xy
                delta_mom_pos_tensor_summed[data_set,k,j,5]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#yz
                delta_mom_pos_tensor_summed[data_set,k,j,6]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#zx
                delta_mom_pos_tensor_summed[data_set,k,j,7]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#zy
                delta_mom_pos_tensor_summed[data_set,k,j,8]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#yx

                kinetic_energy_tensor_summed[data_set,k,j,0]=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,0],axis=0)/(box_vol)#xx
                kinetic_energy_tensor_summed[data_set,k,j,1]=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#yy
                kinetic_energy_tensor_summed[data_set,k,j,2]=np.sum(SRD_velocities_initial[:,2]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#zz
                kinetic_energy_tensor_summed[data_set,k,j,3]=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#xy
                kinetic_energy_tensor_summed[data_set,k,j,4]=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#xz
                kinetic_energy_tensor_summed[data_set,k,j,5]=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#yz
                
                stress_tensor_summed[data_set,k,j,0]=delta_mom_pos_tensor_summed[data_set,k,j,0] + kinetic_energy_tensor_summed[data_set,k,j,0]#xx
                stress_tensor_summed[data_set,k,j,1]=delta_mom_pos_tensor_summed[data_set,k,j,1] + kinetic_energy_tensor_summed[data_set,k,j,1]#yy
                stress_tensor_summed[data_set,k,j,2]=delta_mom_pos_tensor_summed[data_set,k,j,2] + kinetic_energy_tensor_summed[data_set,k,j,2]#zz
                
                
                stress_tensor_summed[data_set,k,j,3]=delta_mom_pos_tensor_summed[data_set,k,j,3] + kinetic_energy_tensor_summed[data_set,k,j,4] + (erate[data_set]*delta_t_srd*0.5)*kinetic_energy_tensor_summed[data_set,k,j,2]#xz
                stress_tensor_summed[data_set,k,j,4]=delta_mom_pos_tensor_summed[data_set,k,j,4] + kinetic_energy_tensor_summed[data_set,k,j,3] #xy 
                stress_tensor_summed[data_set,k,j,5]=delta_mom_pos_tensor_summed[data_set,k,j,5] + kinetic_energy_tensor_summed[data_set,k,j,5]#yz
                stress_tensor_summed[data_set,k,j,6]=delta_mom_pos_tensor_summed[data_set,k,j,6] + kinetic_energy_tensor_summed[data_set,k,j,4] + (erate[data_set]*delta_t_srd*0.5)*kinetic_energy_tensor_summed[data_set,k,j,2] #zx
                stress_tensor_summed[data_set,k,j,7]=delta_mom_pos_tensor_summed[data_set,k,j,7] + kinetic_energy_tensor_summed[data_set,k,j,5]#zy
                stress_tensor_summed[data_set,k,j,8]=delta_mom_pos_tensor_summed[data_set,k,j,8] + kinetic_energy_tensor_summed[data_set,k,j,3]#yx


             
#%% using multiprocessing to speed up the code 

# first need to turn the previous calc into a function 
# def stress_tensor_total_compute(realisation_name_h5_after,shape_after,j_,no_data_sets,erate,delta_t_srd):
#     delta_mom_summed= np.zeros((no_data_sets,j_,shape_after[0]-1,3))
#     delta_mom_pos_tensor_summed= np.zeros((no_data_sets,j_,shape_after[0]-1,9))
#     stress_tensor_summed=np.zeros((no_data_sets,j_,shape_after[0]-1,9))
#     kinetic_energy_tensor_summed=np.zeros((no_data_sets,j_,shape_after[0]-1,6))
#     #for i in range(0,len(realisation_name_h5_after)):
#     with h5.File(realisation_name_h5_after, 'r') as f_a:
#             data_set = np.where(erate==float(realisation_name_h5_after.split('_')[15]))[0][0]
#             k=np.where(realisation_index==float(realisation_name_h5_after.split('_')[9]))[0][0]
#             for j in range(1,shape_after[0]-1):
#         #for j in range(0,10):
        
#             #with h5.File(realisation_name_h5_before[k], 'r') as f_b:
#                     data_set = np.where(erate==float(realisation_name_h5_after.split('_')[15]))[0][0]
#                     k=np.where(realisation_index==float(realisation_name_h5_after.split('_')[9]))[0][0]
                    
#                     SRD_positions_initial= f_a['particles']['SRDs']['position']['value'][j-1]
                    
#                     SRD_positions_after= f_a['particles']['SRDs']['position']['value'][j]
#                     #print(SRD_positions_after)

#                     SRD_velocities_initial=f_a['particles']['SRDs']['velocity']['value'][j-1]
#                     SRD_velocities_after=f_a['particles']['SRDs']['velocity']['value'][j]
#                     #print(SRD_velocities_after)
#                     delta_mom=SRD_velocities_after-SRD_velocities_initial
#                     delta_mom_summed[data_set,k,j,:]=np.sum(SRD_velocities_after-SRD_velocities_initial,axis=0)

#                     # need to add kinetic contribution

                    
#                     delta_mom_pos_tensor_summed[data_set,k,j,0]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#xx
#                     delta_mom_pos_tensor_summed[data_set,k,j,1]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#yy
#                     delta_mom_pos_tensor_summed[data_set,k,j,2]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#zz
#                     delta_mom_pos_tensor_summed[data_set,k,j,3]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#xz
#                     delta_mom_pos_tensor_summed[data_set,k,j,4]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#xy
#                     delta_mom_pos_tensor_summed[data_set,k,j,5]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#yz
#                     delta_mom_pos_tensor_summed[data_set,k,j,6]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#zx
#                     delta_mom_pos_tensor_summed[data_set,k,j,7]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#zy
#                     delta_mom_pos_tensor_summed[data_set,k,j,8]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#yx

#                     kinetic_energy_tensor_summed[data_set,k,j,0]=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,0],axis=0)/(box_vol)#xx
#                     kinetic_energy_tensor_summed[data_set,k,j,1]=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#yy
#                     kinetic_energy_tensor_summed[data_set,k,j,2]=np.sum(SRD_velocities_initial[:,2]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#zz
#                     kinetic_energy_tensor_summed[data_set,k,j,3]=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#xy
#                     kinetic_energy_tensor_summed[data_set,k,j,4]=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#xz
#                     kinetic_energy_tensor_summed[data_set,k,j,5]=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#yz
                    
#                     stress_tensor_summed[data_set,k,j,0]=delta_mom_pos_tensor_summed[data_set,k,j,0] + kinetic_energy_tensor_summed[data_set,k,j,0]#xx
#                     stress_tensor_summed[data_set,k,j,1]=delta_mom_pos_tensor_summed[data_set,k,j,1] + kinetic_energy_tensor_summed[data_set,k,j,1]#yy
#                     stress_tensor_summed[data_set,k,j,2]=delta_mom_pos_tensor_summed[data_set,k,j,2] + kinetic_energy_tensor_summed[data_set,k,j,2]#zz
                    
                    
#                     stress_tensor_summed[data_set,k,j,3]=delta_mom_pos_tensor_summed[data_set,k,j,3] + kinetic_energy_tensor_summed[data_set,k,j,4] + (erate[data_set]*delta_t_srd*0.5)*kinetic_energy_tensor_summed[data_set,k,j,2]#xz
#                     stress_tensor_summed[data_set,k,j,4]=delta_mom_pos_tensor_summed[data_set,k,j,4] + kinetic_energy_tensor_summed[data_set,k,j,3] #xy 
#                     stress_tensor_summed[data_set,k,j,5]=delta_mom_pos_tensor_summed[data_set,k,j,5] + kinetic_energy_tensor_summed[data_set,k,j,5]#yz
#                     stress_tensor_summed[data_set,k,j,6]=delta_mom_pos_tensor_summed[data_set,k,j,6] + kinetic_energy_tensor_summed[data_set,k,j,4] + (erate[data_set]*delta_t_srd*0.5)*kinetic_energy_tensor_summed[data_set,k,j,2] #zx
#                     stress_tensor_summed[data_set,k,j,7]=delta_mom_pos_tensor_summed[data_set,k,j,7] + kinetic_energy_tensor_summed[data_set,k,j,5]#zy
#                     stress_tensor_summed[data_set,k,j,8]=delta_mom_pos_tensor_summed[data_set,k,j,8] + kinetic_energy_tensor_summed[data_set,k,j,3]#yx

#             return stress_tensor_summed,kinetic_energy_tensor_summed,delta_mom_pos_tensor_summed
    





#%% taking realisation mean

delta_mom_pos_tensor_summed_realisation_mean=np.mean(delta_mom_pos_tensor_summed,axis=1)
stress_tensor_summed_realisation_mean=np.mean(stress_tensor_summed,axis=1)        
kinetic_energy_tensor_summed_realisation_mean=np.mean(kinetic_energy_tensor_summed,axis=1)   

# calculating rolling average 
delta_mom_pos_tensor_summed_realisation_mean_rolling=np.zeros((no_data_sets,shape_after[0]-1,9))
stress_tensor_summed_realisation_mean_rolling= np.zeros((no_data_sets,shape_after[0]-1,9))
kinetic_energy_tensor_summed_realisation_mean_rolling=np.zeros((no_data_sets,shape_after[0]-1,3))
for j in range(0,erate.shape[0]):
    for k in range(0,9):
        for i in range(0,shape_after[0]-1): 
                    delta_mom_pos_tensor_summed_realisation_mean_rolling[j,i,k]=np.mean(delta_mom_pos_tensor_summed_realisation_mean[j,:i,k],axis=0)
                    stress_tensor_summed_realisation_mean_rolling[j,i,k]=np.mean(stress_tensor_summed_realisation_mean[j,:i,k],axis=0)
for j in range(0,erate.shape[0]):
    for k in range(0,3):
        for i in range(0,shape_after[0]-1):  
                    kinetic_energy_tensor_summed_realisation_mean_rolling[j,i,k]= np.mean(kinetic_energy_tensor_summed_realisation_mean[j,:i,k],axis=0)
                        
                
#%% loading bloc reproduce plots
stress_tensor_summed_realisation_mean_rolling=np.load("shear_stress_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size)+".npy")
delta_mom_pos_tensor_summed_realisation_mean_rolling=np.load("shear_delta_mom_pos_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size)+".npy")
kinetic_energy_tensor_summed_realisation_mean=np.load("shear_kinetic_energy_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size)+".npy")
                
#%% plotting rolling average diagonal 

labels_coll=["$\Delta p_{x}r_{x}$","$\Delta p_{y}r_{y}$","$\Delta p_{z}r_{z}$","$\Delta p_{x}r_{z}$","$\Delta p_{x}r_{y}$","$\Delta p_{y}r_{z}$","$\Delta p_{z}r_{x}$","$\Delta p_{y}r_{x}$","$\Delta p_{z}r_{y}$"]
labels_stress=["$\sigma_{xx}$","$\sigma_{yy}$","$\sigma_{zz}$","$\sigma_{xz}$","$\sigma_{xy}$","$\sigma_{yz}$","$\sigma_{zx}$","$\sigma_{zy}$","$\sigma_{yx}$"]


stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[:,3000:,0:3])
labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})
for i in range(0,erate.shape[0]):
    for j in range(0,3):
        plt.plot(stress_tensor_summed_realisation_mean_rolling[i,:,j],label=labels_stress[j],color=colour[j])
        plt.ylabel('$\sigma_{\\alpha \\beta}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$N_{coll}$")
        #plt.ylim((9,11))

    plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\alpha}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
    plt.legend(loc='best')
    #plt.tight_layout()
    plt.savefig("rolling_ave_shear_stress_tensor_elements_1_3_gdot_"+str(erate[i])+"_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
    plt.show()

#%% first normal stress difference 
stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[:,3000:,0:3])
N_1=stress_tensor_summed_realisation_mean_rolling[:,:,0]-stress_tensor_summed_realisation_mean_rolling[:,:,1]
labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})
for i in range(0,erate.shape[0]):
    #for j in range(0,3):
        plt.plot(N_1[i,:])
        plt.ylabel('$N_{1}$', rotation=0, labelpad=labelpady)# m, label="$\dot{\gamma}="+str(erate[i])+"$")
        plt.xlabel("$N_{coll}$")
        plt.ylim((-1,1))

    #plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\alpha}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
        plt.legend(loc='best')
    #plt.tight_layout()
    #plt.savefig("rolling_ave_shear_stress_tensor_elements_1_3_gdot_"+str(erate[i])+"_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
plt.show()
#%% plotting off diagonal 
labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})

for i in range(0,erate.shape[0]):
    for j in range(3,4):
        stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[i,3000:,3])
        plt.plot(stress_tensor_summed_realisation_mean_rolling[i,:,j],label=labels_stress[j],color=colour[j])
        plt.ylabel('$\sigma_{\\alpha \\beta}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$N_{coll}$")
        plt.ylim((0,0.5))

    plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\beta}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
    plt.legend(loc='best')
    #plt.tight_layout()
    plt.savefig("rolling_ave_shear_stress_tensor_elements_xy_gdot_"+str(erate[i])+"_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
    plt.show()


# saving bloc so graphs can be reproduced 

np.save("shear_stress_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size),stress_tensor_summed_realisation_mean_rolling)
np.save("shear_delta_mom_pos_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size),delta_mom_pos_tensor_summed_realisation_mean_rolling)
np.save("shear_kinetic_energy_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size),kinetic_energy_tensor_summed_realisation_mean)

#%% plotting whole off diagonal 
labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})

for i in range(0,erate.shape[0]):
    for j in range(3,9):
        stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[i,3000:,3])
        plt.plot(stress_tensor_summed_realisation_mean_rolling[i,:,j],label=labels_stress[j],color=colour[j])
        plt.ylabel('$\sigma_{\\alpha \\beta}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$N_{coll}$")
        plt.ylim((-0.5,0.5))

    #plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\beta}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
    plt.legend(bbox_to_anchor=(1.2,1),loc='upper right')
    plt.tight_layout()
    plt.savefig("rolling_ave_shear_stress_tensor_elements_4_9_gdot_"+str(erate[i])+"_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
    plt.show()



#%% viscosity estimate
stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[:,3000:,3],axis=1)
viscosity=stress_tensor_summed_realisation_mean_rolling_hline/erate
alpha=np.pi
dim=3
def collisional_visc(alpha,rho,dim):
    coll_visc= (1/(6*dim*rho))*(rho-1+np.exp(-rho))*(1-np.cos(alpha))
    return coll_visc
def kinetic_visc(alpha,rho):
    kin_visc= (5*rho)/((rho-1+np.exp(-rho))*(2-np.cos(alpha)-np.cos(2*alpha)))  -1 
    return kin_visc

total_kinematic_visc= kinetic_visc(alpha,rho) + collisional_visc(alpha,rho,dim)
shear_dynamic_visc_prediction= total_kinematic_visc*rho
np.save("stress_tensor_summed_realisation_mean_rolling_hline"+str(rho)+"_L_"+str(box_size),stress_tensor_summed_realisation_mean_rolling_hline)
np.save("shear_visc_"+str(rho)+"_L_"+str(box_size),viscosity)
#%% stress vs strain rate plot 
fit=np.polyfit(erate,stress_tensor_summed_realisation_mean_rolling_hline,1)
plt.scatter(np.asarray(erate[:],float), stress_tensor_summed_realisation_mean_rolling_hline[:])
plt.plot(erate,fit[0]*erate + fit[1])
plt.xticks(erate) 
plt.xlabel("$\dot{\gamma}$",rotation=0)

plt.ylabel("$\sigma_{xz}$",rotation=0,labelpad=labelpady)
plt.show()

# %% testing parallel results 
stress_tensor_summed=np.load("stress_tensor_summed_test.npy")
delta_mom_pos_tensor_summed=np.load("delta_mom_pos_tensor_summed_test.npy")
kinetic_energy_tensor_summed=np.load("kinetic_energy_tensor_summed_test.npy")

# %%
