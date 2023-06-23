##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will given the correct set of input data produce all the plots and velocity profile data 

after an MPCD simulation. 
"""
#%%
#from imp import reload
import os

#from sys import exception
#from tkinter import HORIZONTAL
import numpy as np

import matplotlib.pyplot as plt
import regex as re
import pandas as pd
#import pyswarms as ps

plt.rcParams.update(plt.rcParamsDefault)
#plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
#import seaborn as sns

import scipy.stats
from datetime import datetime


path_2_post_proc_module= '/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/LAMMPS python run and analysis scripts/Analysis codes'
#path_2_post_proc_module= '/Users/lukedebono/Documents/LAMMPS_projects_mac_book/OneDrive_1_24-02-2023/LAMMPS python run and analysis scripts/Analysis codes'
os.chdir(path_2_post_proc_module)
#import them in the order they should be used 
from mom2numpy import *
from velP2numpy import *
import glob 
from post_MPCD_MP_processing_module import *
#importlib.reload(post_MPCD_MP_processing_module)

#%% define key inputs 
j_=3

swap_rate = np.array([3,7,15,30,60,150,300,600,900,1200]) # values chosen from original mp paper
swap_number = np.array([1,10,100,1000])
swap_number = np.array([1])
#swap_rate= np.array([7])
equilibration_timesteps= 2000 # number of steps to do equilibration with 
#equilibration_timesteps=1000
VP_ave_freq =1000
chunk = 20

dump_freq=1000 # if you change the timestep rememebr to chaneg this 
thermo_freq = 10000
scaled_temp=1
scaled_timestep=0.001
realisation=np.array([0.0,1.0,2.0])
VP_output_col_count = 4 
r_particle =10e-6
phi=0.0005
N=2
Vol_box_at_specified_phi=(N* (4/3)*np.pi*r_particle**3 )/phi
box_side_length=np.cbrt(Vol_box_at_specified_phi)
fluid_name='H20'
run_number='2_2e6'
no_timesteps=2500000 # rememebr to change this depending on run 

#%% grabbing file names 
#vel.Ar_579862_mom_output__no_rescale_851961_2.0_61516_11877.258268303078_0.01_197_1000_10000_500000_T_1.0_lbda_1.3517157706256893_SR_450_SN_10
VP_general_name_string='vel.'+fluid_name+'*_mom_output_*_no_rescale_*'
#vel.Nitrogen_mom_output_no221643_no_rescale_6913_1.0_139775_7725.899083231229_0.01_6_1000_10000_500000_T_1.0_lbda_0.8097165710872696_SR_5_SN_10
#VP_general_name_string='vel.profile_mom_output__no_rescale_*'
#mom.Nitrogen_no_tstat_no810908_no_rescale_400902_0.0_48384_3755.9188485904992
Mom_general_name_string='mom.'+fluid_name+'*_no_tstat_*_no_rescale_*'
#mom.Nitrogen_no_tstat__no_rescale_
#Mom_general_name_string='mom.'+fluid_name+'_*_tstat__no_rescale_*'

#filepath='N_phi_0.005_0.00005_data_T_1'
filepath='T_1_phi_'+str(phi)+'_data/'+fluid_name+'_data_T_1_phi_'+str(phi)+'/run_'+str(run_number)+'/'
filepath='T_1_compiled_data_all_phi'
filepath='T_1_phi_0.005_data/H20_data_T_1_phi_0.0005/run_952534'
filepath='Test_data_solid/phi_0.005'
filepath='T_1_phi_0.005_solid_inc_data/H20_data/run_208958'
filepath= 'T_1_phi_0.0005_solid_inc_data/H20_data/run_484692'  


realisation_name_info=VP_and_momentum_data_realisation_name_grabber(j_,swap_number,swap_rate,VP_general_name_string,Mom_general_name_string,filepath)
realisation_name_Mom=realisation_name_info[0]
realisation_name_VP=realisation_name_info[1]
count_mom=realisation_name_info[2]
count_VP=realisation_name_info[3]
number_of_solutions=realisation_name_info[4]

##LEARN HOW TO USE DEBUGGER


# checking the number of different solutions used in the run 
# locations of key info when string name is split by undescores
# AR 
# loc_no_SRD=9
# loc_EF=21
# loc_SN=23
# loc_Realisation_index= 8
# Nitrogen (old versions)
loc_no_SRD=8
loc_EF=20
loc_SN=22
loc_Realisation_index= 7
loc_box_size=9
# 
no_SRD=[]
box_size=[]
for i in range(0,count_VP):
    no_srd=realisation_name_VP[i].split('_')
    no_SRD.append(no_srd[loc_no_SRD])
    box_size.append(no_srd[loc_box_size])
    
no_SRD.sort(key=int)
no_SRD.sort()
no_SRD_key=[]
box_size_key=[]
#using list comprehension to remove duplicates
[no_SRD_key.append(x) for x in no_SRD if x not in no_SRD_key]
[box_size_key.append(x) for x in box_size if x not in box_size_key]
box_side_length_scaled=[]
for item in box_size_key:
    box_side_length_scaled.append(float(item))
box_side_length_scaled=np.array([box_side_length_scaled])

#vel.Nitrogen_mom_output_no221643_no_rescale_6913_1.0_139775_7725.899083231229_0.01_6_1000_10000_500000_T_1.0_lbda_0.8097165710872696_SR_5_SN_10
#no_SRD_key= SRD_counter_solution_grabber_duplicate_remover(loc_no_SRD,count_VP,realisation_name_VP)[0]

# %%
simulation_file="MYRIAD_LAMMPS_runs/"+filepath

Path_2_VP="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file
#from velP2numpy import velP2numpy_f
#Path_2_VP='/Users/lukedebono/Documents/LAMMPS_projects_mac_book/OneDrive_1_24-02-2023/'

VP_raw_data= VP_organiser_and_reader(loc_no_SRD,loc_EF,loc_SN,loc_Realisation_index,box_side_length_scaled,j_,number_of_solutions,swap_number,swap_rate,no_SRD_key,realisation_name_VP,Path_2_VP,chunk,equilibration_timesteps,VP_ave_freq,no_timesteps,VP_output_col_count,count_VP)

# #np.save(fluid_name+'_T_'+str(scaled_temp)+'_box_size_'+str(box_side_length_scaled)+'_'+VP_raw_data)
VP_data_upper=VP_raw_data[0]
VP_data_lower=VP_raw_data[1]
error_count=VP_raw_data[2]
filename=VP_raw_data[3]
VP_z_data_upper=VP_raw_data[4]
VP_z_data_lower=VP_raw_data[5]

if error_count != 0: 
    print('Error reading velocity profiles, check data !')
else:
    print('Velocity profile data success')

box_size_loc=9
lengthscale=box_side_length/float(filename[box_size_loc])

#%% to analyse one file 
from velP2numpy import velP2numpy_f
marker=-1
error_count=0 
VP_data_upper=np.zeros((number_of_solutions,swap_rate.size,swap_number.size,9,int(no_timesteps/VP_ave_freq),j_))
VP_data_lower=np.zeros((number_of_solutions,swap_rate.size,swap_number.size,9,int(no_timesteps/VP_ave_freq),j_))

for i in range(0,count_VP):
    filename=realisation_name_VP[i].split('_')
    marker=marker+1
    no_SRD=filename[loc_no_SRD]
    z=no_SRD_key.index(no_SRD)
    realisation_index=filename[loc_Realisation_index]
    j=int(float(realisation_index))
    EF=int(filename[loc_EF])
    m=np.where(swap_rate==EF)
    SN=int(filename[loc_SN])
    k=np.where(swap_number==SN)
    realisation_name=realisation_name_VP[i]
    try: 
        VP_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[0]
        VP_data_upper= VP_data[1:10,:]
        VP_data_lower= VP_data[11:,:]
        
    except Exception as e:
        print('Velocity Profile Data faulty')
        error_count=error_count+1 
        continue
VP_z_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[1]     

VP_z_data_upper = np.array([VP_z_data[1:10].astype('float64')])* box_side_length_scaled.T    
VP_z_data_lower =np.array([ VP_z_data[11:].astype('float64') ])* box_side_length_scaled.T


box_size_loc=9
lengthscale=box_side_length/float(filename[box_size_loc])



 #%% saving VP data as function is very slow 
# np.save(fluid_name+'_VP_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_data_upper)    
# np.save(fluid_name+'_VP_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_data_lower)   
# np.save(fluid_name+'_VP_z_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_z_data_upper)    
# np.save(fluid_name+'_VP_z_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_z_data_lower)          
# #%% loading vP data to avoid running fucntion 

# VP_data_upper=np.load(fluid_name+'_VP_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_data_upper)    
# VP_data_lower=np.load(fluid_name+'_VP_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_data_lower)   
# VP_z_data_upper=np.load(fluid_name+'_VP_z_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_z_data_upper)    
# VP_z_data_lower=np.load(fluid_name+'_VP_z_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_z_data_lower)          

# %% obtaining the mom data size
#Path_2_mom_file="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file
Path_2_mom_file=Path_2_VP

mom_data_pre_process= mom_file_data_size_reader(j_,number_of_solutions,count_mom,realisation_name_Mom,no_SRD_key,swap_rate,swap_number,Path_2_mom_file)
size_array=mom_data_pre_process[0]
mom_data= mom_data_pre_process[1]
pass_count= mom_data_pre_process[2]

if pass_count!=count_VP:
    print("Data import error, number of momentum files should match number of velocity files" )
else:
    print("Data size assessment success!")

#%% Reading in mom data files 
mom_data_files=Mom_organiser_and_reader(mom_data,count_mom,realisation_name_Mom,no_SRD_key,swap_rate,swap_number,Path_2_mom_file)

mom_data=mom_data_files[0]
error_count_mom=mom_data_files[1]
failed_list_realisations=[2]

if error_count_mom !=0:
    print("Mom file error, check data files aren't damaged")
else:
    print("Mom data import success")
    
#%% Now assess the steady state of the VP data 

VP_shear_rate_and_stat_data=VP_data_averaging_and_stat_test_data(VP_z_data_upper,VP_z_data_lower,no_timesteps,VP_data_lower,VP_data_upper,number_of_solutions,swap_rate,swap_number,VP_ave_freq)
pearson_coeff_upper=VP_shear_rate_and_stat_data[0]
shear_rate_upper=VP_shear_rate_and_stat_data[1]
pearson_coeff_lower=VP_shear_rate_and_stat_data[2]
shear_rate_lower=VP_shear_rate_and_stat_data[3]
timestep_points=VP_shear_rate_and_stat_data[4]
VP_data_lower_realisation_averaged=VP_shear_rate_and_stat_data[5]
VP_data_upper_realisation_averaged=VP_shear_rate_and_stat_data[6]
shear_rate_upper_error=VP_shear_rate_and_stat_data[7]
shear_rate_lower_error=VP_shear_rate_and_stat_data[8]
#%% Averaging for one file 

VP_z_data_upper_repeated= np.repeat(VP_z_data_upper.T,VP_data_upper.shape[1],axis=1)
VP_z_data_lower_repeated= np.repeat(VP_z_data_lower.T,VP_data_lower.shape[1],axis=1)
pearson_coeff_upper= np.zeros(VP_data_upper.shape[1])
pearson_coeff_lower= np.zeros(VP_data_lower.shape[1])
shear_rate_upper= np.zeros(VP_data_upper.shape[1])    
shear_rate_lower= np.zeros(VP_data_lower.shape[1])
shear_rate_upper_error= np.zeros(VP_data_upper.shape[1]) 
shear_rate_lower_error= np.zeros(VP_data_lower.shape[1])

for i in range(0,VP_data_upper.shape[1]):
    pearson_coeff_upper[i]=scipy.stats.pearsonr(VP_data_upper[:,i],VP_z_data_upper_repeated[:,i])[0]
    shear_rate_upper[i]= scipy.stats.linregress(VP_z_data_upper_repeated[:,i],VP_data_upper[:,i]).slope
    shear_rate_upper_error[i]= scipy.stats.linregress(VP_data_upper[:,i],VP_z_data_upper_repeated[:,i] ).stderr
    pearson_coeff_lower[i] =scipy.stats.pearsonr(VP_data_lower[:,i],VP_z_data_lower_repeated[:,i] )[0]
    shear_rate_lower[i]= scipy.stats.linregress(VP_z_data_lower_repeated[:,i] ,VP_data_lower[:,i]).slope  
    shear_rate_lower_error[i]= scipy.stats.linregress(VP_data_lower[:,i],VP_z_data_lower_repeated [:,i] ).stderr 
timestep_points=np.array([[[np.linspace(1,VP_data_upper.shape[1],int(float(no_timesteps)/float(VP_ave_freq)))]]])*VP_ave_freq

plt.plot(timestep_points[0,0,0,:],shear_rate_upper[:])
plt.plot(timestep_points[0,0,0,:],shear_rate_lower[:])
plt.xlabel('$N_{t}[-]$')
plt.ylabel('$\dot{\gamma}[\\tau]$',rotation='horizontal')
plt.title(fluid_name+" simulation run with all $f_{v,x}$ and $N_{v,x}$, $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")

plt.show()
#%%
import log2numpy
Path_2_log='/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/Test_data_solid/logs'
realisation_name='log.H20_solid396988_inc_mom_output_no_rescale_243550_0.0_9112_118.77258268303078_0.001_1781_1000_10000_1000000_T_1.0_lbda_1.3166259218664098_SR_7_SN_1_rparticle_10.0'
thermo_vars='         KinEng          Temp          TotEng'
log_data= log2numpy.log2numpy(Path_2_log,thermo_vars,realisation_name)[0]
fontsize=15
labelpad=20
#plotting temp vs time 
plt.plot(log_data[:,0],log_data[:,2])
temp=1
x=np.repeat(temp,log_data[:,0].shape[0])
plt.plot(log_data[:,0],x[:])
plt.xlabel('$N_{t}[-]$',fontsize=fontsize)
plt.ylabel('$T[\\frac{T k_{B}}{\\varepsilon}]$', rotation=0,fontsize=fontsize,labelpad=labelpad)
plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+", $f_{v,x}=$"+str(swap_rate[0])+", $N_{v,x}=$"+str(swap_number[0])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
                
plt.show()

#plotting energy vs time 
plt.plot(log_data[:,0],log_data[:,3])
plt.xlabel('$N_{t}[-]$',fontsize=fontsize)
plt.ylabel('$E_{t}[\\frac{\\tau^{2}}{\mu \ell^{2}}]$', rotation=0,fontsize=fontsize,labelpad=labelpad)
plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+", $f_{v,x}=$"+str(swap_rate[0])+", $N_{v,x}=$"+str(swap_number[0])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
     
plt.show()




# need to fix log file removing warning messages

# %%
import sigfig
lengthscale= sigfig.round(lengthscale,sigfigs=3)
box_size_nd= box_side_length_scaled
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

swap_rate_index_start=0
swap_rate_index_end=10
swap_number_index_start=0
swap_number_index_end=1
def plot_shear_rate_to_asses_SS(swap_number_index_end,swap_number_index_start,swap_rate_index_start,swap_rate_index_end,no_timesteps,phi,lengthscale,timestep_points,scaled_temp,number_of_solutions,swap_rate,swap_number,shear_rate_upper,shear_rate_lower,fluid_name,box_size_nd):
    for z in range(0,number_of_solutions): 
        for k in range(swap_number_index_start,swap_number_index_end):
            for m in range(swap_rate_index_start,swap_rate_index_end):
            
                plt.plot(timestep_points[0,1,0,:],shear_rate_upper[z,m,k,:])
                plt.plot(timestep_points[0,1,0,:],shear_rate_lower[z,m,k,:])
                plt.xlabel('$N_{t}[-]$')
                plt.ylabel('$\dot{\gamma}[\\tau]$',rotation='horizontal')
                plt.title(fluid_name+" simulation run with all $f_{v,x}$ and $N_{v,x}$, $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
                
            plt.show()
        #plot_save=input("save figure?, YES/NO")
        # if plot_save=='YES':
        #     plt.savefig(fluid_name+'_T_'+str(scaled_temp)+'_length_scale_'+str(lengthscale)+'_phi_'+str(phi)+'_no_timesteps_'+str(no_timesteps)+'.png')
        # else:
        #     print('Thanks for checking steady state')

plot_shear_rate_to_asses_SS(swap_number_index_end,swap_number_index_start,swap_rate_index_start,swap_rate_index_end,no_timesteps,phi,lengthscale,timestep_points,scaled_temp,number_of_solutions,swap_rate,swap_number,shear_rate_upper,shear_rate_lower,fluid_name,box_size_nd)
# need to save this plot 
#%% assess gradient of truncated shear rate data to determine steady state




# %%
truncation_timestep=1000000
truncation_and_SS_averaging_data=  truncation_step_and_SS_average_of_VP_and_stat_tests(shear_rate_upper_error,shear_rate_lower_error,timestep_points,pearson_coeff_lower,pearson_coeff_upper,shear_rate_upper,shear_rate_lower,VP_ave_freq,truncation_timestep,VP_data_lower_realisation_averaged,VP_data_upper_realisation_averaged)
standard_deviation_upper_error=truncation_and_SS_averaging_data[0]
standard_deviation_lower_error=truncation_and_SS_averaging_data[1]
pearson_coeff_upper_mean_SS=truncation_and_SS_averaging_data[2]
pearson_coeff_lower_mean_SS=truncation_and_SS_averaging_data[3]
shear_rate_lower_steady_state_mean=truncation_and_SS_averaging_data[4]
shear_rate_upper_steady_state_mean=truncation_and_SS_averaging_data[5]
VP_steady_state_data_lower_truncated_time_averaged=truncation_and_SS_averaging_data[6]
VP_steady_state_data_upper_truncated_time_averaged=truncation_and_SS_averaging_data[7]
shear_rate_upper_steady_state_mean_error=truncation_and_SS_averaging_data[8]
shear_rate_lower_steady_state_mean_error=truncation_and_SS_averaging_data[9]

# could probably vectorise this or use a method 
error_count = 0
for z in range(0,number_of_solutions):
    for k in range(swap_number_index_start,swap_number_index_end):
            for m in range(swap_rate_index_start,swap_rate_index_end):
                 if pearson_coeff_upper_mean_SS[z,m,k]<0.7:
                     print('Non-linear simulation run please inspect')
                     error_count=error_count +1 
                 else:
                     print('Great success')
                     
print('Non-linear simulation count: ',error_count)
                
#%% assess gradient of truncated shear rate data to determine steady state
# can then truncate again 
slope_shear_rate_upper=  np.zeros((number_of_solutions,swap_rate.size,swap_number.size))
slope_shear_rate_lower=  np.zeros((number_of_solutions,swap_rate.size,swap_number.size))

gradient_tolerance= 1e-9
for z in range(0,number_of_solutions): 
        for m in range(swap_rate_index_start,swap_rate_index_end):
            for k in range(swap_number_index_start,swap_number_index_end):
                slope_shear_rate_upper[z,m,k]=np.polyfit(timestep_points[0,1,0,:],shear_rate_upper[z,m,k,:],1)[0]
                slope_shear_rate_lower[z,m,k]=np.polyfit(timestep_points[0,1,0,:],shear_rate_upper[z,m,k,:],1)[0]
                if np.abs(slope_shear_rate_upper[z,m,k]) < gradient_tolerance:
                    slope_shear_rate_upper[z,m,k] =slope_shear_rate_upper[z,m,k] 
                else: 
                    #slope_shear_rate_upper[z,m,k]='NaN'
                    print('FAILED run, exclude from data ')
                if np.abs(slope_shear_rate_lower[z,m,k]) < gradient_tolerance:
                    slope_shear_rate_lower[z,m,k] =slope_shear_rate_lower[z,m,k] 
                else: 
                    #slope_shear_rate_lower[z,m,k]='NaN'
                    print('FAILED run, exclude from data ')
print("if no fail statements, data can be considered steady")

#%% plotting gradient of the shear vs time plot 
for z in range(0,number_of_solutions): 
    for k in range(swap_number_index_start,swap_number_index_end):
         plt.yscale('log')
         plt.ylabel('grad $\dot{\gamma}$')
         plt.xlabel('$f_{v,x}$')
         plt.scatter(swap_rate[:],slope_shear_rate_upper[z,:,k])
         plt.scatter(swap_rate[:],slope_shear_rate_lower[z,:,k])
    
plt.show

# is this sufficient ?? 


# need to save this plot 
##%% save to debug 
#%%
np.save(fluid_name+'_standard_deviation_upper_error_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),standard_deviation_upper_error)    
np.save(fluid_name+'_standard_deviation_lower_error_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),standard_deviation_lower_error)    
np.save(fluid_name+'_pearson_coeff_upper_mean_SS_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),pearson_coeff_upper_mean_SS)    
np.save(fluid_name+'_pearson_coeff_lower_mean_SS_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),pearson_coeff_lower_mean_SS)    
np.save(fluid_name+'_shear_rate_lower_steady_state_mean_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),shear_rate_lower_steady_state_mean)          
np.save(fluid_name+'_shear_rate_upper_steady_state_mean_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),shear_rate_upper_steady_state_mean)          
np.save(fluid_name+'VP_steady_state_data_lower_truncated_time_averaged'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_steady_state_data_lower_truncated_time_averaged)          
np.save(fluid_name+'VP_steady_state_data_upper_truncated_time_averaged'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_steady_state_data_upper_truncated_time_averaged)          
#np.save(fluid_name+'mom_data_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),mom_data)
np.save(fluid_name+'VP_z_data_upper_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_z_data_upper)
np.save(fluid_name+'VP_z_data_lower_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_z_data_lower)

#%% load to skip above cells 

standard_deviation_upper_error=np.load(fluid_name+'_standard_deviation_upper_error_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps)+'.npy')    
standard_deviation_lower_error=np.load(fluid_name+'_standard_deviation_lower_error_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps)+'.npy')    
pearson_coeff_upper_mean_SS=np.load(fluid_name+'_pearson_coeff_upper_mean_SS_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps)+'.npy')    
pearson_coeff_lower_mean_SS=np.load(fluid_name+'_pearson_coeff_lower_mean_SS_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps)+'.npy')    
shear_rate_lower_steady_state_mean=np.load(fluid_name+'_shear_rate_lower_steady_state_mean_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps)+'.npy')          
shear_rate_upper_steady_state_mean=np.load(fluid_name+'_shear_rate_upper_steady_state_mean_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps)+'.npy')          
VP_steady_state_data_lower_truncated_time_averaged=np.load(fluid_name+'VP_steady_state_data_lower_truncated_time_averaged'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps)+'.npy')          
VP_steady_state_data_upper_truncated_time_averaged=np.load(fluid_name+'VP_steady_state_data_upper_truncated_time_averaged'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps)+'.npy')          
VP_z_data_upper=np.save(fluid_name+'VP_z_data_upper_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps)+'.npy')
VP_z_data_lower=np.save(fluid_name+'VP_z_data_lower_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps)+'.npy')

# still need to load in mom data
# reset truncation step if needed 
truncation_timestep=150000


#%% mom_data_averaging_and_flux_calc

from post_MPCD_MP_processing_module import *
 
flux_ready_for_plotting=mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,swap_number,truncation_timestep,swap_rate,scaled_timestep,no_timesteps,box_side_length_scaled[0,0],mom_data)[0]
mom_data_realisation_averaged_truncated=mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,swap_number,truncation_timestep,swap_rate,scaled_timestep,no_timesteps,box_side_length_scaled[0,0],mom_data)[1]
#(1,4,10)
#%% importin momentum after steady state
def mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,swap_number,truncation_timestep,swap_rate,scaled_timestep,no_timesteps,box_side_length_scaled,mom_data):
    mom_data_realisation_averaged=()
    number_swaps_before_truncation=(np.ceil(truncation_timestep/swap_rate)).astype(int)
    mom_data_realisation_averaged_truncated=()
    flux_x_momentum_z_direction=np.zeros((number_of_solutions,swap_number.size,swap_rate.size))
    total_run_time=scaled_timestep* (no_timesteps-truncation_timestep)
    
    flux_ready_for_plotting=np.zeros((number_of_solutions,swap_number.size,swap_rate.size))
    for z in range(0,number_of_solutions):    
        for i in range(0,swap_rate.size):
            box_area_nd=float(box_size_key[z])**2
            mom_data_realisation_averaged=mom_data_realisation_averaged+(np.mean(mom_data[i],axis=2),)


        #for i in range(0,swap_rate.size):

            mom_data_realisation_averaged_truncated=mom_data_realisation_averaged_truncated+(mom_data_realisation_averaged[i][:,:,number_swaps_before_truncation[i]:],)


        # now apply the MP formula 
            mom_difference= mom_data_realisation_averaged_truncated[i][z,:,-1]-mom_data_realisation_averaged_truncated[i][z,:,0]
            flux_x_momentum_z_direction[z,:,i]=(mom_difference)/(2*total_run_time*float(box_area_nd))
            
    flux_ready_for_plotting=np.log((np.abs(flux_x_momentum_z_direction)))
    
    return flux_ready_for_plotting,mom_data_realisation_averaged_truncated


#%%
flux_ready_for_plotting=mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,swap_number,truncation_timestep,swap_rate,scaled_timestep,no_timesteps,box_side_length_scaled[0,0],mom_data)[0]
mom_data_realisation_averaged_truncated=mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,swap_number,truncation_timestep,swap_rate,scaled_timestep,no_timesteps,box_side_length_scaled[0,0],mom_data)[1]
# flux vs shear regression line 
shear_rate_mean_of_both_cells=(((np.abs(shear_rate_lower_steady_state_mean)+np.abs(shear_rate_upper_steady_state_mean))*0.5))
shear_rate_mean_error_of_both_cells=(np.abs(shear_rate_lower_steady_state_mean_error)+np.abs(shear_rate_upper_steady_state_mean_error))*0.5
print(shear_rate_mean_of_both_cells.shape)
print(shear_rate_mean_error_of_both_cells.shape)
shear_rate_mean_error_of_both_cells_relative=shear_rate_mean_error_of_both_cells/shear_rate_mean_of_both_cells
shear_rate_mean_of_both_cells=np.log(((np.abs(shear_rate_lower_steady_state_mean)+np.abs(shear_rate_upper_steady_state_mean))*0.5))
shear_rate_mean_error_of_both_cells=shear_rate_mean_of_both_cells*shear_rate_mean_error_of_both_cells_relative

# reshape

#shear_rate_mean_of_both_cells= np.reshape(shear_rate_mean_of_both_cells,(flux_ready_for_plotting.shape))
# shear_rate_mean_error_of_both_cells_relative= np.reshape(shear_rate_mean_error_of_both_cells_relative,(flux_ready_for_plotting.shape))
# shear_rate_mean_error_of_both_cells=shear_rate_mean_of_both_cells*shear_rate_mean_error_of_both_cells_relative
print(shear_rate_mean_of_both_cells.shape)
print(shear_rate_mean_error_of_both_cells.shape)

flux_vs_shear_regression_line_params=()
x=shear_rate_mean_of_both_cells
#shear_rate_mean_of_both_cells=np.reshape(shear_rate_mean_of_both_cells,(flux_ready_for_plotting.shape))

def func4(x, a, b):
   #return np.log(a) + np.log(b*x)
   #return (a*(x**b))
   return (a*x) +b 
   #return a*np.log(b*x)+c



for z in range(0,number_of_solutions):    
    for i in range(0,swap_number.size):
      
        flux_vs_shear_regression_line_params= flux_vs_shear_regression_line_params+(scipy.optimize.curve_fit(func4,shear_rate_mean_of_both_cells[z,:,i],flux_ready_for_plotting[z,i,:],method='lm',maxfev=5000)[0],)
        #print(scipy.optimize.curve_fit(func4,shear_rate_mean_of_both_cells[z,:,i],flux_ready_for_plotting[z,i,:],method='lm',maxfev=5000)[0])

params=flux_vs_shear_regression_line_params 

#%%
 #calculating error of flux
# plot cumulative momentum exchange vs time 
# fit to linear grad and take error 
# need to calculate number of swaps done, plot that as time axes 

#total_number_of_swaps_after_SS=(np.floor( (no_timesteps-truncation_timestep)/swap_rate))
swap_timestep_vector=()
total_run_time=scaled_timestep* (no_timesteps-truncation_timestep)
for z in range(swap_rate_index_start,swap_rate_index_end):
    total_number_of_swaps_after_SS=mom_data_realisation_averaged_truncated[z].shape[2]
    final_swap_step= truncation_timestep +(total_number_of_swaps_after_SS*swap_rate[z])
    #print(final_swap_step)
    swap_timestep_vector= swap_timestep_vector+ (np.arange(truncation_timestep,final_swap_step,int(swap_rate[z])),)

slope_momentum_vector_error=()
slope_momentum_vector_error_1=()
pearson_coeff_momentum=()
slope_momentum_vector_mean_abs_error= np.zeros((number_of_solutions,swap_number_index_end,swap_rate_index_end))
slope_flux_abs_error=np.zeros((number_of_solutions,swap_number_index_end,swap_rate_index_end))
for z in range(0,number_of_solutions):
    box_area_nd=float(box_size_key[z])**2
    for k in range(swap_number_index_start,swap_number_index_end):
            for m in range(swap_rate_index_start,swap_rate_index_end):
                plt.scatter(swap_timestep_vector[m],-mom_data_realisation_averaged_truncated[m][z,k,:],label='$f_v=${}'.format(swap_rate[m]),marker='x', s=0.00005)
                plt.legend()
                plt.ylabel('$P_x$',rotation=0)
                plt.xlabel('$N_t$')
                
                #slope_momentum_vector_error=slope_momentum_vector_error + (np.polyfit(swap_timestep_vector[m],-mom_data_realisation_averaged_truncated[m][z,k,:],1,full=True)[1],)
                #slope_momentum_vector_error_1=slope_momentum_vector_error_1 + (scipy.stats.linregress(swap_timestep_vector[m],-mom_data_realisation_averaged_truncated[m][z,k,:] ).stderr,)
                pearson_coeff_momentum=pearson_coeff_momentum+ (scipy.stats.pearsonr(swap_timestep_vector[m],-mom_data_realisation_averaged_truncated[m][z,k,:])[0],)
                if pearson_coeff_momentum[m]  > 0.9999:
                    print('All pearson coeffs are perfectly linear, therefore there is no error in the total momentum')
                else:
                    print('Cumulative total momentum is not linear in time, please check data is has reached SS')
                #scipy.stats.linregress(swap_timestep_vector[m],-mom_data_realisation_averaged_truncated[m][z,k,:] ).stderr
                # print(mom_data_realisation_averaged_truncated[m][z,k,:].shape[0])
                # slope_momentum_vector_mean_abs_error[z,k,m]= np.sqrt(slope_momentum_vector_error[m][0]/mom_data_realisation_averaged_truncated[m][z,k,:].shape[0])
                # slope_flux_abs_error[z,k,m]=slope_momentum_vector_mean_abs_error[z,k,m]/(2*total_run_time*float(box_area_nd))
                #print(swap_timestep_vector[m].shape,mom_data_realisation_averaged_truncated[m][z,k,:].shape)
plt.show()




#NOTE this section needs to be finished.


#%%

 
# flux_from_fit=np.zeros((number_of_solutions,swap_number.size,swap_rate.size))
# for z in range(0,number_of_solutions):
#     for i in range(0,swap_number.size):
#         for j in range(0,swap_rate.size):
#           flux_from_fit[z,i,j]=func4(shear_rate_mean_of_both_cells[z,j,i],params[i][0],params[i][1])


# abs_error_in_flux= flux_from_fit-flux_ready_for_plotting

#msq_error_in_flux= 


#%% 
#save_string_for_plot= 'Flux_vs_shear_rate_'+fluid_name+'_phi_range_'+str(phi[0])+'_'+str(phi[1])+'_l_scale_'+str(lengthscale)+'_T_'+str(scaled_temp)+'.png'
labelpadx=15
labelpady=40
fontsize=15
count=1
swap_number_index=1 

       
#plotting_flux_vs_shear_rate(func4,labelpadx,labelpady,params,fontsize,box_side_length_scaled,number_of_solutions,flux_ready_for_plotting,swap_number_index,shear_rate_mean_of_both_cells)
def plotting_flux_vs_shear_rate(shear_rate_mean_error_of_both_cells,func4,labelpadx,labelpady,params,fontsize,box_side_length_scaled,number_of_solutions,flux_ready_for_plotting,swap_number_index,shear_rate_mean_of_both_cells):
    
    for z in range(0,number_of_solutions):
        
        
        x=shear_rate_mean_of_both_cells[z,:,:]
        x_pos_error=np.abs(shear_rate_mean_error_of_both_cells[z,:,:])
        #y_pos_error=np.abs(abs_error_in_flux[z,:,:])
        y=flux_ready_for_plotting[z,:,:]
        
        for i in range(0,swap_number_index):
        
        #for i in range(0,1):
            if z==0:
                j=i
                
                # need to add legend to this 
                plt.scatter(x[:,i],y[i,:],label='$L=${}'.format(np.around(box_side_length_scaled[0,z]),decimals=0),marker='x')
                plt.errorbar(x[:,i],y[i,:],xerr=x_pos_error[:,i],ls ='',capsize=3,color='r')
                plt.plot(x[:,i],func4(x[:,i],params[j][0],params[j][1]))
                #plt.fill_between(y[:,i], x_neg_error[i,:], x_pos_error[i,:])
                #plt.xscale('log')
                plt.xlabel('log($\dot{\gamma}\ [\\tau]$)', labelpad=labelpadx,fontsize=fontsize)
                #plt.yscale('log')
                plt.ylabel('log($J_{z}(p_{x})$$\ [\\frac{\\tau^{3}}{\mu}]$)',rotation=0,labelpad=labelpady,fontsize=fontsize)
                plt.legend()
            else: 
                j=z*(i+4)
                plt.scatter(x[:,i],y[i,:],label='$L=${}'.format(np.around(box_side_length_scaled[0,z]),decimals=0),marker='x')
                plt.errorbar(x[:,i],y[i,:],xerr=x_pos_error[:,i],ls ='',capsize=3,color='r')
                plt.plot(x[:,i],func4(x[:,i],params[j][0],params[j][1]))
                #plt.fill_between(y[:,i], x_neg_error[i,:], x_pos_error[i,:])
                #plt.xscale('log')
                plt.xlabel('log($\dot{\gamma}\ [\\tau]$)', labelpad=labelpadx,fontsize=fontsize)
                #plt.yscale('log')
                plt.ylabel('log($J_{z}(p_{x})$$\ [\\frac{\\tau^{3}}{\mu}]$)',rotation=0,labelpad=labelpady,fontsize=fontsize)
                plt.legend()
                 
    plt.show() 
    

plotting_flux_vs_shear_rate(shear_rate_mean_error_of_both_cells,func4,labelpadx,labelpady,params,fontsize,box_side_length_scaled,number_of_solutions,flux_ready_for_plotting,swap_number_index,shear_rate_mean_of_both_cells)    
# need to adjust this so we get the visocsities of both plots 
shear_viscosity=10** (params[0][1])
print('Dimensionless_shear_viscosity:',shear_viscosity)
# to get the error in viscosity need to look whether we take the mean of the relative or absolute errors. 

#dimensionful_shear_viscosity= shear_viscosity * mass_scale / lengthscale*timescale
#%% plotting qll 4 SS V_Ps

# need to fix legend location 
swap_number_choice_index=0
fontsize=22
labelpadx=15
labelpady=35
width_plot=15
height_plot=10
legend_x_pos=1.2
legend_y_pos=1
swap_rate_index_start=0
swap_rate_index_end=10
  
plotting_SS_velocity_profiles(swap_rate_index_start,swap_rate_index_end,legend_x_pos, legend_y_pos,labelpadx,labelpady,fontsize,number_of_solutions,swap_number_choice_index,width_plot,height_plot,swap_number,swap_rate,VP_ave_freq,no_timesteps,VP_steady_state_data_lower_truncated_time_averaged,VP_steady_state_data_upper_truncated_time_averaged,VP_z_data_lower,VP_z_data_upper)
#%%















