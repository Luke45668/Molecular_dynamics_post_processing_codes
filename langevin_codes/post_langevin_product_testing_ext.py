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
# plt.rcParams["figure.figsize"] = (8,6 )
# plt.rcParams.update({'font.size': 16})
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
marker=['x','+','^',"1","X","d","*","P","v","."]

damp=np.array([ 0.035, 0.035 ,0.035,0.035])
K=np.array([  60,480,960,1500   ,
            ])

thermal_damp_multiplier=np.array([750,1000,1500,2000])
thermal_damp_multiplier=np.array([750,1000,1500])
#thermal_damp_multiplier=np.array([50,250,500,750,1000])

K=np.array([  25,50,100,200])
erate=np.flip(np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.01,0.005,0]))

no_timesteps=np.flip(np.array([ 3944000,  4382000,  4929000,  5634000,  6573000,  7887000,
         9859000, 13145000, 19718000,  2253000,  2629000,  3155000,
         3944000,  4929000,  6573000,  9859000, 39435000,
        78870000, 10000000]))

erate=np.flip(np.array([0.5,0.45,0.4,0.375,0.35,0.325,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.02,0.01,0.005]))

no_timesteps=np.flip(np.array([ 492939000,  547710000,  616173000,  657252000,  704198000,
         758367000,  821565000, 1232347000, 1408396000, 1643129000,
        1971755000,  246469000,  308087000,  410782000,  616173000,
        1232347000,  246469000,  492939000]))

erate=np.flip(np.array([0.55,0.5,0.45,0.4,0.375,0.3675,0.35,0.3375 ,0.325,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.02,0.01,0.005]))

no_timesteps=np.flip(np.array([ 352099000,  410782000,  448126000,  492939000,  547710000,
         616173000,  657252000,  670665000,  704198000,  730280000,
         758367000,  821565000, 1232347000, 1408396000, 1643129000,
        1971755000,  246469000,  308087000,  410782000,  616173000,
        1232347000,  246469000,  492939000]))

erate=np.flip(np.array([0.5,0.45,0.4,0.375,0.3725,0.37,0.365,0.36,0.355,0.35,0.3375 ,0.325,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.02,0.01,0.005]))

no_timesteps=np.flip(np.array([  197175000, 219084000, 246469000, 262901000, 264665000, 266453000,
        270103000, 273855000, 277712000, 281679000, 292112000, 303347000,
        328626000, 492939000, 563359000, 657252000, 788702000,  98588000,
        123235000, 164313000, 246469000, 492939000,  98588000, 197175000]))

# timestep_multiplier=np.array([
# [0.00005,0.00005,0.00005,0.00005,
# 0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
# 0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
# 0.0005,0.0005,0.0005,0.0005,0.0005,0.005,
# 0.005],

# [0.00005,0.00005,0.00005,0.00005,
# 0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
# 0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
# 0.0005,0.0005,0.0005,0.0005,0.0005,0.005,
# 0.005]])*4

e_in=0
e_end=erate.size
n_plates=100

strain_total=100

# need to check if we have the same amount of strain in all cases. could just do a bulk re-run with 10000 data points 
# instead 

path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/run_279865/saved_tuples"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/dumbell_run/log_tensor_files/saved_tuples"
path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/product_run"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/run_226020/saved_tuples"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/non_zero_natural_length_run/run_118874/saved_tuples"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/non_zero_natural_length_run/run_813285/sucessful_runs_5_reals/saved_tuples"
path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/non_zero_natural_length_run/run_171767/sucessful_runs_10_reals/saved_tuples"
path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/non_zero_natural_length_run/run_709773/sucessful_runs_10_reals/saved_tuples"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/non_zero_natural_length_run/run_74420/sucessful_runs_5_reals/saved_tuples"
thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
thermo_vars='         KinEng         PotEng         Press           Temp         Ecouple       Econserve   '
thermo_vars='         KinEng         PotEng         Press           Temp         Ecouple       Econserve    c_uniaxnvttemp'

j_=10
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
posvel_batch_tuple=()


damp=2000
# loading all data into one 
#for i in range(thermal_damp_multiplier.size):
for i in range(K.size):
    label='damp_'+str(damp)+'_K_'+str(K[i])+'_'
    #label='damp_'+str(thermal_damp_multiplier[i])+'_K_'+str(K[0])+'_'
    print(label)

    spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(batch_load_tuples(label,
                                                            "spring_force_positon_tensor_tuple.pickle"),)
    posvel_batch_tuple=posvel_batch_tuple+(batch_load_tuples(label,"posvelocities_tuple.pickle"),)
    

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
plt.rcParams["figure.figsize"] = (24,6 )
plt.rcParams.update({'font.size': 16})
e_end=[11,14,15,16]
e_end=[20,20,20,20,19]
e_end=[18,18,18,18,18,18,18,18,18,18,18]
e_end=[21,21,21,21]
e_end=[20,20,20,20]
#for j in range(K.size):
j=0
for i in range(erate[:e_end[j]].size):
    mean_temp_array=np.zeros((erate[:e_end[j]].size))
    for j in range(thermal_damp_multiplier.size):
           

        
        #for i in range(erate[:e_end[j]].size):
        #i=15
            plt.subplot(1, 3, 1)
            column=4
            signal_std=sigfig.round(np.std(log_file_batch_tuple[j][i][100:,column]), sigfigs=3)
            signal_mean=sigfig.round(np.mean(log_file_batch_tuple[j][i][100:,column]), sigfigs=5)
            plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column],
            label="tdamp="+str(thermal_damp_multiplier[j])+",$\\bar{T}="+str(signal_mean)+",\sigma_{T}="+str(signal_std)+"$")
            plt.ylabel("$T$", rotation=0)
            plt.title("$"+str(erate[i])+"$")
            plt.legend(loc='upper right', bbox_to_anchor=(4.5,1))


            plt.subplot(1, 3, 2)
            column=2
            grad_pe=np.gradient(log_file_batch_tuple[j][i][:,column])
            grad_mean=np.mean(grad_pe[500:])
            plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column],
            label="tdamp="+str(thermal_damp_multiplier[j])+",$\\bar{grad}="+str(grad_mean)+"$")
            plt.plot(strainplot_tuple[i][:],grad_pe)
            plt.ylabel("$E_{p}$")
            
            plt.title("$\dot{\gamma}="+str(erate[i])+"$")
            plt.legend(loc='upper right', bbox_to_anchor=(4.5,0.75))
            # final_temp[i]=log_file_batch_tuple[j][i][-1,column]

            
        #for i in range(erate[:e_end[j]].size):
            plt.subplot(1, 3, 3)
            column=1
            plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column],
            label="tdamp="+str(thermal_damp_multiplier[j])+",$\dot{\gamma}="+str(erate[i])+"$")
            plt.ylabel("$E_{k}$")
            plt.title("$\dot{\gamma}="+str(erate[i])+"$")
            
            mean_temp_array[i]=np.mean(log_file_batch_tuple[j][i][500:,column])

            mean_temp_tuple=mean_temp_tuple+(mean_temp_array,)
        
            #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
           
                #     plt.xlabel("$\gamma$")
                

            
                # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
            #plt.ylim(0,2)
            #plt.yscale('log')
            
            plt.show()
#%% fix tdamp vary k
from fitter import Fitter

folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=4
final_temp=np.zeros((erate.size))
mean_temp_tuple=()
plt.rcParams["figure.figsize"] = (24,12 )
plt.rcParams.update({'font.size': 16})

e_end=[24,24,24,24]
#e_end=[1,1,1,1]
#for j in range(K.size):
j=0
for i in range(erate[:e_end[j]].size):
    mean_temp_array=np.zeros((erate[:e_end[j]].size))
    for j in range(K.size):
           

        
        #for i in range(erate[:e_end[j]].size):
        #i=15
            plt.subplot(2, 3, 1)
            column=4
            signal_std=sigfig.round(np.std(log_file_batch_tuple[j][i][100:,column]), sigfigs=3)
            signal_mean=sigfig.round(np.mean(log_file_batch_tuple[j][i][100:,column]), sigfigs=5)
            plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column],
            label="K="+str(K[j])+",$\\bar{T}="+str(signal_mean)+",\sigma_{T}="+str(signal_std)+"$")
            plt.ylabel("$T$", rotation=0)
            plt.title("$"+str(erate[i])+"$")
            #plt.legend(loc='upper right', bbox_to_anchor=(4.25,1))
            plt.legend()


            plt.subplot(2, 3, 2)
            column=2
            grad_pe=np.gradient(log_file_batch_tuple[j][i][:,column])
            grad_mean=np.mean(grad_pe[500:])
            plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column],
            label="K="+str(K[j])+",$\\bar{grad}="+str(grad_mean)+"$")
            #plt.plot(strainplot_tuple[i][:],grad_pe)
            plt.ylabel("$E_{p}$")
            plt.yscale('log')
            plt.title("$\dot{\gamma}="+str(erate[i])+"$")
            #plt.legend(loc='upper right', bbox_to_anchor=(4.25,0.75))
            plt.legend()
            # final_temp[i]=log_file_batch_tuple[j][i][-1,column]

            
        #for i in range(erate[:e_end[j]].size):
            plt.subplot(2,3 , 3)
            column=1
            plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column],
            label="K="+str(K[j])+",$\dot{\gamma}="+str(erate[i])+"$")
            plt.ylabel("$E_{k}$")
            plt.title("$\dot{\gamma}="+str(erate[i])+"$")

            vel_data=posvel_batch_tuple[j][i].astype('float')[:,5:8]
            plt.subplot(2,3,4)
           
            x_vel=np.ravel(vel_data[:,0])
            # f = Fitter(x_vel)
            
            # f.distributions =  ['gennorm']
            # f.fit()
            # # # may take some time since by default, all distributions are tried
            # # # but you call manually provide a smaller set of distributions
            # f.summary()
            sns.kdeplot(x_vel, bw_adjust=1)

            plt.xlabel("$v_{x}$")
            #plt.legend()
            plt.subplot(2,3,5)
           
            y_vel=np.ravel(vel_data[:,1])
            # f = Fitter(y_vel)
            
            # f.distributions =  ['gennorm']
            # f.fit()
            # # # may take some time since by default, all distributions are tried
            # # # but you call manually provide a smaller set of distributions
            # f.summary()
            sns.kdeplot(y_vel, bw_adjust=1)
            plt.xlabel("$v_{y}$")
            #plt.legend()
            plt.subplot(2,3,6)
            z_vel=np.ravel(vel_data[:,2])
            # f = Fitter(z_vel)
            
            # f.distributions =  ['gennorm']
            # f.fit()
            # # # may take some time since by default, all distributions are tried
            # # # but you call manually provide a smaller set of distributions
            # f.summary()
            test=scipy.stats.kstest(z_vel, 'norm')
            sns.kdeplot(z_vel, bw_adjust=1,label="$D="+str(test[0])+",p="+str(test[1])+"$")
            #plt.legend()
           

            plt.xlabel("$v_{z}$")




            
            mean_temp_array[i]=np.mean(log_file_batch_tuple[j][i][500:,column])

            mean_temp_tuple=mean_temp_tuple+(mean_temp_array,)
        
            #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
           
                #     plt.xlabel("$\gamma$")
                

            
                # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
            #plt.ylim(0,2)
            #plt.yscale('log')
            
            plt.show()

#%% fix tdamp vary k 
folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=4
final_temp=np.zeros((erate.size))
mean_temp_tuple=()

e_end=[11,14,15,16]
#e_end=[16,16]
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
        plt.legend(loc='upper right', bbox_to_anchor=(1.5,1))
        plt.show()

#%% potential energy fix k vary tdamp 
j=0
for i in range(0,erate[:e_end[j]].size):
#for i in range(0,2):
    for j in range(0,thermal_damp_multiplier.size):
        column=2
        #for i in range(erate[:e_end[j]].size):
        
            
        plt.plot(log_file_batch_tuple[j][i][:,column],
        label="tdamp="+str(thermal_damp_multiplier[j])+",$\dot{\gamma}="+str(erate[i])+"$")
        plt.ylabel("$E_{p}$")
        plt.title("$\dot{\gamma}="+str(erate[i])+"$")
        
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
    
        
            #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
        #     plt.ylabel("$T$", rotation=0)
        #     plt.xlabel("$\gamma$")
        

    # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
        plt.yscale('log')
        plt.ylim(1e-10,10)
        plt.legend(loc='upper right', bbox_to_anchor=(1.5,1))
        plt.show()

#%% kinetic energy fix k vary tdamp 
j=0
for i in range(0,erate[:e_end[j]].size):
    for j in range(0,thermal_damp_multiplier.size):
        column=1
        #for i in range(erate[:e_end[j]].size):
        
            
        plt.plot(log_file_batch_tuple[j][i][:,column],
        label="tdamp="+str(thermal_damp_multiplier[j])+",$\dot{\gamma}="+str(erate[i])+"$")
        plt.ylabel("$E_{k}$")
        plt.title("$\dot{\gamma}="+str(erate[i])+"$")
        
        
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
    
        
            #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
        #     plt.ylabel("$T$", rotation=0)
        #     plt.xlabel("$\gamma$")
        

        # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
    plt.yscale('log')
    plt.ylim(1e-8,10)
    plt.legend(loc='upper right', bbox_to_anchor=(1.5,1))
    plt.show()
#%% potential energy fix tdamp vary K 
j=0
for i in range(erate[:e_end[j]].size):
        for j in range(K.size):
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
marker=['x','+','^',"1","X","d","*","P","v","."]

# for j in range(K.size):
for j in range(thermal_damp_multiplier.size):
    #plt.scatter(erate[:e_end[j]],mean_temp_tuple[j],label="$K="+str(K[j])+"$" ,marker=marker[j])
    plt.scatter(erate[:e_end[j]],mean_temp_tuple[j],label="$tdamp="+str(thermal_damp_multiplier[j])+"$" ,marker=marker[j])
    plt.ylabel("$T$", rotation=0)
    plt.xlabel('$\dot{\gamma}$')
    #plt.xscale('log')
   # plt.yscale('log')
plt.axhline(1,label="$T_{0}=1$")
plt.legend()
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/temp_vs_erate.pdf",dpi=1200,bbox_inches='tight')


plt.show()


#%% look at stress vs strain fix k vary tsdamp
labels_stress=np.array(["\sigma_{xx}$",
               "\sigma_{yy}$",
               "\sigma_{zz}$",
               "\sigma_{xz}$",
               "\sigma_{xy}$",
               "\sigma_{yz}$"])
damp_cutoff=0
for j in range(K.size):
    for i in range(0,erate[:e_end[j]].size):
    # for j in range(damp_cutoff,thermal_damp_multiplier.size):
    #for j in range(K.size):

        stress_tensor_mean=np.mean(spring_force_positon_tensor_batch_tuple[j][i],axis=0)
        stress_tensor_mean=np.mean(stress_tensor_mean,axis=1)

        for l in range(1):

            # plt.plot( stress_tensor_mean[:,l],label="$"+labels_stress[l]+\
            #     ",$tdamp="+str(thermal_damp_multiplier[j])+"$")
            plt.plot( stress_tensor_mean[:,l],label="$\dot{\gamma}="+str(erate[i])+"$")

    plt.legend(loc='upper right', bbox_to_anchor=(1.2,1))
    plt.title("$K="+str(K[j])+","+labels_stress[l])

    plt.yscale('log')
    plt.ylim(-0.2,0.2)
    plt.show()


#%% yy stress

for i in range(0,erate[:e_end[j]].size):
    for j in range(damp_cutoff,thermal_damp_multiplier.size):

        stress_tensor_mean=np.mean(spring_force_positon_tensor_batch_tuple[j][i],axis=0)
        stress_tensor_mean=np.mean(stress_tensor_mean,axis=1)

        for l in range(1,2):

            plt.plot( stress_tensor_mean[:,l],label="$"+labels_stress[l]+\
                ",$tdamp="+str(thermal_damp_multiplier[j])+"$")

    plt.legend(loc='upper right', bbox_to_anchor=(1.5,1))
    plt.title("$\dot{\gamma}="+str(erate[i])+"$")
    #plt.ylim(-250,250)
    plt.show()

for i in range(0,erate[:e_end[j]].size):
    for j in range(damp_cutoff,thermal_damp_multiplier.size):

        stress_tensor_mean=np.mean(spring_force_positon_tensor_batch_tuple[j][i],axis=0)
        stress_tensor_mean=np.mean(stress_tensor_mean,axis=1)

        for l in range(2,3):

            plt.plot( stress_tensor_mean[:,l],label="$"+labels_stress[l]+\
                ",$tdamp="+str(thermal_damp_multiplier[j])+"$")

    plt.legend(loc='upper right', bbox_to_anchor=(1.5,1))
    plt.title("$\dot{\gamma}="+str(erate[i])+"$")
    #plt.ylim(-250,250)
    plt.show()
#%% look at internal stresses
aftcut=1
cut=0.5
# aftcut=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.25,0.25,0.2,0.2,0.175,0.15,0.15,0.1,0.1,0.1]
# cut=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.1,0.1,0.075,0.075,0.075,0.075,0.075,0.05,0.05,0.05]

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
    stress_tensor,stress_tensor_std=stress_tensor_averaging(e_end[j],labels_stress,
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
plt.rcParams["figure.figsize"] = (10,6 )
plt.rcParams.update({'font.size': 16})
# for j in range(thermal_damp_multiplier.size): 
for j in range(K.size): 
    for l in range(3):
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[0])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]), marker=marker[j])
        plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[j])+","+str(labels_stress[l]), marker=marker[j])
        

    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
       
        plt.xlabel("$\dot{\gamma}$")
        plt.ylabel("$\sigma_{\\alpha \\alpha}$")
        #plt.yticks(y_ticks_stress)
        #plt.ylim(0.9,1.3)
        #plt.tight_layout()
        #plt.yscale('log')
        #plt.savefig(path_2_log_files+"/plots/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.legend(loc='upper right', bbox_to_anchor=(1.5,1))
plt.show()

#%%
plt.rcParams["figure.figsize"] = (10,6 )
plt.rcParams.update({'font.size': 16})
#for j in range(thermal_damp_multiplier.size): 
for j in range(K.size): 
    for l in range(3,6):
         #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[0])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[j])+","+str(labels_stress[l]), marker=marker[j])
        
    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
        plt.legend(fontsize=10) 
        plt.xlabel("$\dot{\gamma}$")
        plt.ylabel("$\sigma_{\\alpha \\beta}$")
        #plt.yticks(y_ticks_stress)
        #plt.ylim(0.9,1.3)
        #plt.xscale('log')
        #plt.tight_layout()

        #plt.savefig(path_2_log_files+"/plots/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight') 
    
#plt.ylim(-3,3)
plt.show()

#%%
def ext_visc_compute(stress_tensor,stress_tensor_std,i1,i2,n_plates,e_end):
    extvisc=(stress_tensor[:,i1]- stress_tensor[:,i2])/erate[:e_end]/30.3
    extvisc_error=np.sqrt(stress_tensor_std[:,i1]**2 +stress_tensor_std[:,i2]**2)/np.sqrt(j_*n_plates*10)

    return extvisc,extvisc_error

#for j in range(thermal_damp_multiplier.size):
for j in range(K.size):



    ext_visc_1,ext_visc_1_error=ext_visc_compute(stress_tensor_tuple[j],stress_tensor_std_tuple[j],0,2,n_plates,e_end[j])
    cutoff=0
    #plt.errorbar(erate[cutoff:e_end[j]],ext_visc_1[cutoff:],yerr=ext_visc_1_error, label="$\eta_{1},K="+str(K[j])+"$", linestyle='none', marker=marker[j])
    #plt.errorbar(erate[cutoff:e_end[j]],ext_visc_1[cutoff:],yerr=ext_visc_1_error, label="$\eta_{1},tdamp="+str(thermal_damp_multiplier[j])+"$", marker=marker[j])
    plt.errorbar(erate[cutoff:e_end[j]],ext_visc_1[cutoff:],yerr=ext_visc_1_error[cutoff:], label="$\eta_{1},K="+str(K[j])+"$", marker=marker[j])
    #plt.plot(erate[cutoff:e_end[j]],ext_visc_1[cutoff:], label="$\eta_{1},K="+str(K[j])+"$", marker=marker[j])
    #plt.plot(erate[cutoff:e_end[j]],ext_visc_1_error,label="$\eta_{1},tdamp="+str(thermal_damp_multiplier[j])+"$")
    #plt.plot(erate[:e_end[j]],ext_visc_1, label="$\eta_{1},K="+str(K[j])+"$", linestyle='none', marker=marker[j])
   # plt.plot(erate[cutoff:e_end[j]],ext_visc_1[cutoff:], label="$tdamp="+str(thermal_damp_multiplier[j])+"$", marker=marker[j])
    #plt.plot(erate[cutoff:e_end[j]], k_50_ext_visc[cutoff:], label="$tdampk50="+str(thermal_damp_multiplier[j])+"$", marker=marker[j])
   
    plt.ylabel("$\eta/\eta_{s}$", rotation=0, labelpad=20)
    plt.xlabel("$\dot{\gamma}$")
#plt.yscale('log')
#plt.ylim(-2,7)
plt.legend(loc='upper right', bbox_to_anchor=(1.5,1))
plt.show()



# %%
