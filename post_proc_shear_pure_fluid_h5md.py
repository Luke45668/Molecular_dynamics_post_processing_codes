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
# no_SRD=1038230
# box_size=47
# no_SRD=506530
# box_size=37
no_SRD=58320
box_size=18
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
rho=10 

#%% finding all the dump files in a folder

VP_general_name_string='vel.*'

Mom_general_name_string='mom.*'

log_general_name_string='log.*'
                         #log.H20_no466188_wall_VACF_output_no_rescale_
TP_general_name_string='temp.*'

dump_general_name_string_after='*'+str(no_timesteps)+'*after*.h5'
dump_general_name_string_before='*'+str(no_timesteps)+'*before*.h5'

filepath="/KATHLEEN_LAMMPS_RUNS/equilibrium_fix_deform_pure_mpcd_test_file"
filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/hfd5_runs/non_equilibrium_tests/test_non_eq_box_"+str(int(box_size))+"_M_"+str(rho)
#filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/hfd5_runs/tests_equilibrium_with_more_regular_neighbour_listing_box_"+str(int(box_size))+"_M_10"
Path_2_dump=filepath
# can chnage this to another array on kathleen


dump_realisation_name_info_after= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_after)

realisation_name_h5_after=dump_realisation_name_info_after[6]
count_h5_after=dump_realisation_name_info_after[7]
#%% should check the time series of velocity profiles aswell. 

# find dump file size
with h5.File(realisation_name_h5_after[0], 'r') as f:
    shape_after= f['particles']['SRDs']['position']['value'].shape
    print(f['particles']['SRDs']['species'].keys())
#%%
j_=4
delta_mom_summed= np.zeros((j_,shape_after[0]-1,3))
delta_mom_pos_tensor_summed= np.zeros((j_,shape_after[0]-1,9))
stress_tensor_summed=np.zeros((j_,shape_after[0]-1,9))
kinetic_energy_tensor_summed=np.zeros((j_,shape_after[0]-1,6))

for k in range(0,j_):
    for j in range(1,shape_after[0]-1):
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

                # need to add kinetic contribution

                
                delta_mom_pos_tensor_summed[k,j,0]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#xx
                delta_mom_pos_tensor_summed[k,j,1]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#yy
                delta_mom_pos_tensor_summed[k,j,2]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#zz
                delta_mom_pos_tensor_summed[k,j,3]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#xz
                delta_mom_pos_tensor_summed[k,j,4]=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#xy
                delta_mom_pos_tensor_summed[k,j,5]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#yz
                delta_mom_pos_tensor_summed[k,j,6]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#zx
                delta_mom_pos_tensor_summed[k,j,7]=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#zy
                delta_mom_pos_tensor_summed[k,j,8]=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#yx

                kinetic_energy_tensor_summed[k,j,0]=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,0],axis=0)/(box_vol)#xx
                kinetic_energy_tensor_summed[k,j,1]=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#yy
                kinetic_energy_tensor_summed[k,j,2]=np.sum(SRD_velocities_initial[:,2]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#zz
                kinetic_energy_tensor_summed[k,j,3]=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#xy
                kinetic_energy_tensor_summed[k,j,4]=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#xz
                kinetic_energy_tensor_summed[k,j,5]=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#yz
                
                stress_tensor_summed[k,j,0]=delta_mom_pos_tensor_summed[k,j,0] + kinetic_energy_tensor_summed[k,j,0]#xx
                stress_tensor_summed[k,j,1]=delta_mom_pos_tensor_summed[k,j,1] + kinetic_energy_tensor_summed[k,j,1]#yy
                stress_tensor_summed[k,j,2]=delta_mom_pos_tensor_summed[k,j,2] + kinetic_energy_tensor_summed[k,j,2]#zz
                
                
                stress_tensor_summed[k,j,3]=delta_mom_pos_tensor_summed[k,j,3] + kinetic_energy_tensor_summed[k,j,4] + (erate*delta_t_srd*0.5)*kinetic_energy_tensor_summed[k,j,2]#xz
                #stress_tensor_summed[k,j,4]=delta_mom_pos_tensor_summed[k,j,4] + kinetic_energy_tensor_summed[k,j,3]#xy
                #stress_tensor_summed[k,j,5]=delta_mom_pos_tensor_summed[k,j,5] + kinetic_energy_tensor_summed[k,j,5]#yz
                stress_tensor_summed[k,j,6]=delta_mom_pos_tensor_summed[k,j,6] + kinetic_energy_tensor_summed[k,j,4] + (erate*delta_t_srd*0.5)*kinetic_energy_tensor_summed[k,j,2] #zx
               # stress_tensor_summed[k,j,7]=delta_mom_pos_tensor_summed[k,j,7] + kinetic_energy_tensor_summed[k,j,5]#zy
               # stress_tensor_summed[k,j,8]=delta_mom_pos_tensor_summed[k,j,8] + kinetic_energy_tensor_summed[k,j,3]#yx
                
                
#%% taking realisation mean

delta_mom_pos_tensor_summed_realisation_mean=np.mean(delta_mom_pos_tensor_summed,axis=0)
stress_tensor_summed_realisation_mean=np.mean(stress_tensor_summed,axis=0)        
kinetic_energy_tensor_summed_realisation_mean=np.mean(kinetic_energy_tensor_summed,axis=0)   

# calculating rolling average 
delta_mom_pos_tensor_summed_realisation_mean_rolling=np.zeros((shape_after[0]-1,9))
stress_tensor_summed_realisation_mean_rolling= np.zeros((shape_after[0]-1,9))
kinetic_energy_tensor_summed_realisation_mean_rolling=np.zeros((shape_after[0]-1,3))

for k in range(0,9):
    for i in range(0,shape_after[0]-1): 
                 delta_mom_pos_tensor_summed_realisation_mean_rolling[i,k]=np.mean(delta_mom_pos_tensor_summed_realisation_mean[:i,k],axis=0)
                 stress_tensor_summed_realisation_mean_rolling[i,k]=np.mean(stress_tensor_summed_realisation_mean[:i,k],axis=0)
for k in range(0,3):
    for i in range(0,shape_after[0]-1):  
                 kinetic_energy_tensor_summed_realisation_mean_rolling[i,k]= np.mean(kinetic_energy_tensor_summed_realisation_mean[:i,k],axis=0)
                       
                
#%% loading bloc reproduce plots
stress_tensor_summed_realisation_mean_rolling=np.load("shear_stress_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size)+".py")
delta_mom_pos_tensor_summed_realisation_mean_rolling=np.load("shear_delta_mom_pos_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size)+".py")
kinetic_energy_tensor_summed_realisation_mean=np.load("shear_kinetic_energy_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size)+".py")
                
#%% plotting rolling average diagonal 

labels_coll=["$\Delta p_{x}r_{x}$","$\Delta p_{y}r_{y}$","$\Delta p_{z}r_{z}$","$\Delta p_{x}r_{z}$","$\Delta p_{x}r_{y}$","$\Delta p_{y}r_{z}$","$\Delta p_{z}r_{x}$","$\Delta p_{y}r_{x}$","$\Delta p_{z}r_{y}$"]
labels_stress=["$\sigma_{xx}$","$\sigma_{yy}$","$\sigma_{zz}$","$\sigma_{xz}$","$\sigma_{xy}$","$\sigma_{yz}$","$\sigma_{zx}$","$\sigma_{zy}$","$\sigma_{yx}$"]


stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[400:,0:3])
labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})

for j in range(0,3):
    plt.plot(stress_tensor_summed_realisation_mean_rolling[:,j],label=labels_stress[j],color=colour[j])
    plt.ylabel('$\sigma_{\\alpha \\beta}$', rotation=0, labelpad=labelpady)
    plt.xlabel("$N_{coll}$")
    plt.ylim((9,11))

plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\alpha}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
plt.legend(loc='best')
#plt.tight_layout()
plt.savefig("rolling_ave_shear_stress_tensor_elements_1_3_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
plt.show()

#%% plotting off diagonal 
labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})
stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[400:,3:])
for j in range(3,9):
    plt.plot(stress_tensor_summed_realisation_mean_rolling[:,j],label=labels_stress[j],color=colour[j])
    plt.ylabel('$\sigma_{\\alpha \\beta}$', rotation=0, labelpad=labelpady)
    plt.xlabel("$N_{coll}$")
    plt.ylim((-1,1))

plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\beta}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
plt.legend(loc='best')
#plt.tight_layout()
plt.savefig("rolling_ave_shear_stress_tensor_elements_4_9_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
plt.show()


#%% saving bloc so graphs can be reproduced 

np.save("shear_stress_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size)+".py",stress_tensor_summed_realisation_mean_rolling)
np.save("shear_delta_mom_pos_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size)+".py",delta_mom_pos_tensor_summed_realisation_mean_rolling)
np.save("shear_kinetic_energy_tensor_summed_realisation_mean_rolling_M_"+str(rho)+"_L_"+str(box_size)+".py",kinetic_energy_tensor_summed_realisation_mean)




  

# %%
