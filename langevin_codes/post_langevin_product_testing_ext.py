##!/usr/bin/env python3.11
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
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import regex as re
import pandas as pd
import sigfig
plt.rcParams.update(plt.rcParamsDefault)
#plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import mmap
import h5py as h5
from scipy.optimize import curve_fit


#path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
# os.chdir(path_2_post_proc_module)
import seaborn as sns

import glob 
# from post_MPCD_MP_processing_module import *
import pickle as pck
from post_langevin_module import *
from reading_lammps_module import *

linestyle_tuple = ['-', 
  'dotted', 
 'dashed', 'dashdot', 
 'None', ' ', '', 'solid', 
 'dashed', 'dashdot', '--']

#%% 


damp=np.array([ 0.035, 0.035 ,0.035,0.035])
K=np.array([  60,480,960,1500   ,
            ])
# thermal_damp_multiplier=np.array([75,150,300,600,1200])
K=np.array([  50 ,500  
             ])
erate=np.flip(np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.01,0.005,0]))

no_timesteps=np.flip(np.array([ 3944000,  4382000,  4929000,  5634000,  6573000,  7887000,
         9859000, 13145000, 19718000,  2253000,  2629000,  3155000,
         3944000,  4929000,  6573000,  9859000, 39435000,
        78870000, 10000000]))

erate=np.flip(np.array([0.5,0.45,0.4,0.35,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.02,0.01,0.005]))

no_timesteps=np.flip(np.array([ 473221000,  525801000,  591526000,  676030000,  788702000,
        1183053000, 1352060000, 1577404000, 1892885000,  236611000,
         295763000,  394351000,  591526000, 1183053000,  236611000,
         473221000]))


e_in=0
e_end=erate.size
n_plates=100

strain_total=60

# need to check if we have the same amount of strain in all cases. could just do a bulk re-run with 10000 data points 
# instead 

path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/run_279865/saved_tuples"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/dumbell_run/log_tensor_files/saved_tuples"
path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/product_run"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/run_226020/saved_tuples"
path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/non_zero_natural_length_run/run_727608/sucessful_runs_5_reals/saved_tuples"
thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
thermo_vars='         KinEng         PotEng         Press           Temp         Ecouple       Econserve   '
thermo_vars='         KinEng         PotEng         Press           Temp         Ecouple       Econserve    c_uniaxnvttemp'

j_=5
sim_fluid=30.315227255599112

eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
damp_ratio=mass_pol/damp


#%% save tuples
label='damp_'+str(damp)+'_K_'+str(K)+'_'


os.chdir(path_2_log_files)
#os.mkdir("tuple_results")
#os.chdir("tuple_results")

def batch_load_tuples(label,tuple_name):

    with open(label+tuple_name, 'rb') as f:
         load_in= pck.load(f)

    return load_in

erate_velocity_batch_tuple=()
spring_force_positon_tensor_batch_tuple=()
COM_velocity_batch_tuple=()
conform_tensor_batch_tuple=()
log_file_batch_tuple=()
area_vector_spherical_batch_tuple=()
interest_vectors_batch_tuple=()


# loading all data into one 
#for i in range(thermal_damp_multiplier.size):
for i in range(K.size):
    label='damp_'+str(damp[i])+'_K_'+str(K[i])+'_'
    #label='damp_'+str(thermal_damp_multiplier[i])+'_K_'+str(K[0])+'_'
    print(label)

    spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(batch_load_tuples(label,
                                                            "spring_force_positon_tensor_tuple.pickle"),)
    # erate_velocity_batch_tuple=erate_velocity_batch_tuple+(batch_load_tuples(label,
    #                                                         "erate_velocity_tuple.pickle"),)
    # COM_velocity_batch_tuple=COM_velocity_batch_tuple+(batch_load_tuples(label,
    #                                                         "COM_velocity_tuple.pickle"),)
    # conform_tensor_batch_tuple=conform_tensor_batch_tuple+(batch_load_tuples(label,
    #                                                         "conform_tensor_tuple.pickle"),)
    log_file_batch_tuple=log_file_batch_tuple+(batch_load_tuples(label,
                                                            "log_file_tuple.pickle"),)
    # area_vector_spherical_batch_tuple=area_vector_spherical_batch_tuple+(batch_load_tuples(label,"area_vector_tuple.pickle"),)
    
    # interest_vectors_batch_tuple=interest_vectors_batch_tuple+(batch_load_tuples(label,
    #                                                                              "interest_vectors_tuple.pickle"),)

    


     

#%% strain points for temperatuee data 
strainplot_tuple=()

for i in range(erate.size):
    
    strain_plotting_points= np.linspace(0,strain_total,log_file_batch_tuple[0][i].shape[0])

    strainplot_tuple=strainplot_tuple+(strain_plotting_points,)  
    print(strainplot_tuple[i].size)

def strain_plotting_points(total_strain,points_per_iv):
     #points_per_iv= number of points for the variable measured against strain 
     strain_unit=total_strain/points_per_iv
     strain_plotting_points=np.arange(0,total_strain,strain_unit)
     return  strain_plotting_points


#%% fix k vary tdamp
folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=4
final_temp=np.zeros((erate.size))
mean_temp_tuple=()

e_end=[11,14,15,16]
e_end=[20,20,20,20,19]
#for j in range(K.size):
for i in range(erate[:e_end[j]].size):
    for j in range(thermal_damp_multiplier.size):
        mean_temp_array=np.zeros((erate[:e_end[j]].size))
        #for i in range(erate[:e_end[j]].size):
        i=15
            
        plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column],
        label="tdamp="+str(thermal_damp_multiplier[j]))
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
        mean_temp_array[i]=np.mean(log_file_batch_tuple[j][i][1000:,column])

        mean_temp_tuple=mean_temp_tuple+(mean_temp_array,)
        
            #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
        #     plt.ylabel("$T$", rotation=0)
        #     plt.xlabel("$\gamma$")
        

        # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
    plt.legend()
    plt.show()

#%% fix tdamp vary k 
folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=4
final_temp=np.zeros((erate.size))
mean_temp_tuple=()

e_end=[11,14,15,16]
e_end=[16,16]
j=0
for i in range(erate[:e_end[j]].size):
        for j in range(K.size):

    
        
            
            plt.plot(log_file_batch_tuple[j][i][:,column],
            label="K="+str(K[j])+",$\dot{\gamma}="+str(erate[i])+"$")
            # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
            
            mean_temp_array[i]=np.mean(log_file_batch_tuple[j][i][1000:,column])

            mean_temp_tuple=mean_temp_tuple+(mean_temp_array,)
            
            #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
        #     plt.ylabel("$T$", rotation=0)
        #     plt.xlabel("$\gamma$")
        

        # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
        plt.legend()
        plt.show()

#%% potential energy fix k vary tdamp 
j=0
for i in range(erate[:e_end[j]].size):
    for j in range(thermal_damp_multiplier.size):
        column=2
        #for i in range(erate[:e_end[j]].size):
        
            
        plt.plot(log_file_batch_tuple[j][i][:,column],
        label="tdamp="+str(thermal_damp_multiplier[j]))
        
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
    
        
            #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
        #     plt.ylabel("$T$", rotation=0)
        #     plt.xlabel("$\gamma$")
        

        # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
    plt.yscale('log')
    plt.legend()
    plt.show()
#%% potential energy fix tdamp vary K 
j=0
for i in range(erate[:e_end[j]].size):
        for j in range(1,2):
            column=2
            #for i in range(erate[:e_end[j]].size):
            
                
            plt.plot(log_file_batch_tuple[j][i][:,column],
             label="K="+str(K[j])+",$\dot{\gamma}="+str(erate[i])+"$")
            plt.ylabel("$E_{p}$")
            # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
    
        
            #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
        #     plt.ylabel("$T$", rotation=0)
        #     plt.xlabel("$\gamma$")
        

        # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
        #plt.yscale('log')
        plt.legend()
        plt.show()

#%%
marker=['x','+','^',"1","X","d","*","P","v"]

# for j in range(K.size):
for j in range(thermal_damp_multiplier.size-1):
    plt.scatter(erate[:e_end[j]],mean_temp_tuple[j],label="$K="+str(K[j])+"$" ,marker=marker[j])
    plt.ylabel("$T$", rotation=0)
    plt.xlabel('$\dot{\gamma}$')
    #plt.xscale('log')
   # plt.yscale('log')
plt.axhline(1,label="$T_{0}=1$")
plt.legend()
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/temp_vs_erate.pdf",dpi=1200,bbox_inches='tight')


plt.show()


#%% look at internal stresses
aftcut=1
cut=0.9
folder_check_or_create(path_2_log_files,folder)
labels_stress=np.array(["\sigma_{xx}$",
               "\sigma_{yy}$",
               "\sigma_{zz}$",
               "\sigma_{xz}$",
               "\sigma_{xy}$",
               "\sigma_{yz}$"])


#compute stress tensor 
##y_ticks_stress=[-10,0,20,40,60,80] # for plates 
#y_ticks_stress=[0.95,1,1.05,1.1,1.15,1.2,1.25,1.3]
stress_tensor_tuple=()
stress_tensor_std_tuple=()

for j in range(K.size):
    stress_tensor=np.zeros((e_end[j],6))
    stress_tensor_std=np.zeros((e_end[j],6))   
    stress_tensor,stress_tensor_std= stress_tensor_averaging(e_end[j],
                            labels_stress,
                            cut,
                            aftcut,
                           spring_force_positon_tensor_batch_tuple[j],j_)
    
    stress_tensor_tuple=stress_tensor_tuple+(stress_tensor,)
    stress_tensor_std_tuple=stress_tensor_std_tuple+(stress_tensor_std,)



    
    

#%%
# for j in range(K.size):    
#     for i in range(6):

#         plotting_stress_vs_strain( spring_force_positon_tensor_batch_tuple[j],
#                                 e_in,e_end[j],j_,
#                                 strain_total,cut,aftcut,i,labels_stress[i],erate)
#     plt.legend(fontsize=legfont) 
#     plt.tight_layout()
#    # plt.savefig(path_2_log_files+"/plots/"+str(K[j])+"_SS_grad_plots.pdf",dpi=1200,bbox_inches='tight')       
   
#     plt.show()

#%%

for j in range(K.size): 
    for l in range(3):
        plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[j])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        

    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
plt.legend(fontsize=12) 
plt.xlabel("$\dot{\gamma}$")
#plt.yticks(y_ticks_stress)
#plt.ylim(0.9,1.3)
plt.tight_layout()
#plt.xscale('log')
#plt.savefig(path_2_log_files+"/plots/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.show()

#%%
for j in range(K.size): 
    for l in range(3,6):
        plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[j])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        

    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
plt.legend(fontsize=10) 
plt.xlabel("$\dot{\gamma}$")
#plt.yticks(y_ticks_stress)
#plt.ylim(0.9,1.3)
#plt.xscale('log')
plt.tight_layout()

#plt.savefig(path_2_log_files+"/plots/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.show()

#%%
def ext_visc_compute(stress_tensor,stress_tensor_std,i1,i2,n_plates,e_end):
    extvisc=(stress_tensor[:,i1]- stress_tensor[:,i2])/erate[:e_end]/30.3
    extvisc_error=np.sqrt(stress_tensor_std[:,i1]**2 +stress_tensor_std[:,i2]**2)/np.sqrt(j_*n_plates)

    return extvisc,extvisc_error

for j in range(K.size):


    ext_visc_1,ext_visc_1_error=ext_visc_compute(stress_tensor_tuple[j],stress_tensor_std_tuple[j],0,2,n_plates,e_end[j])
    cutoff=1
    #plt.errorbar(erate[:e_end[j]],ext_visc_1,yerr=ext_visc_1_error, label="$\eta_{1},K="+str(K[j])+"$", linestyle='none', marker=marker[j])
    plt.plot(erate[:e_end[j]],ext_visc_1, label="$\eta_{1},K="+str(K[j])+"$", linestyle='none', marker=marker[j])
    plt.ylabel("$\eta/\eta_{s}$", rotation=0, labelpad=20)
    plt.xlabel("$\dot{\gamma}$")

plt.legend()
plt.show()



# %%
