##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will produces all the plots needed to validate the MPCD model with the MP algorithm, apart from the run time data which is in another script.
It is currrently set up to use swaprate as variable 

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

path_2_post_proc_module= '/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/LAMMPS python run and analysis scripts/Analysis codes'
os.chdir(path_2_post_proc_module)

from mom2numpy import *
from velP2numpy import *
from dump2numpy import * 
import glob 
from post_MPCD_MP_processing_module import *
#importlib.reload(post_MPCD_MP_processing_module)

# define key inputs 
j_=3
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
swap_rate=np.array([15,30,60,90,120,150,180,360])
vel_target=np.array([0.8]) 
swap_number=np.array([1])
fluid_name='swaprate'
fluid_name='SRDedittest'
equilibration_timesteps=1000
VP_ave_freq =10000
chunk = 20
dump_freq=10000 # if you change the timestep rememebr to chaneg this 
thermo_freq = 10000
scaled_temp=1
scaled_timestep=0.009270002009500069 #
nubar=0.52
# scaled_timestep=0.0066350508792359635 # 
# nubar=0.7 
# scaled_timestep= 0.005071624521210362 # 
# nubar=0.9
realisation=np.array([1,2,3])
VP_output_col_count = 4 
r_particle =25e-6 # for some solutions, rememebrr to check if its 25 or 10

run_number=''
batchcode='966397'
no_timesteps=2000000 # rememebr to change this depending on run 

# grabbing file names 

VP_general_name_string='vel.'+fluid_name+'**'

Mom_general_name_string='mom.'+fluid_name+'**'

log_general_name_string='log.'+fluid_name+'_**'
                         #log.H20_no466188_wall_VACF_output_no_rescale_
TP_general_name_string='temp.'+fluid_name+'**'

dump_general_name_string='test_run_dump_'+fluid_name+'_*'

#swap_rate file path 
filepath='pure_fluid_new_method_validations/Final_MPCD_val_run/fluid_visc_0.52_data/swaprate_test'
# srdtest filepath 
filepath='pure_fluid_new_method_validations/Final_MPCD_val_run/fluid_visc_'+str(nubar)+'_data/SRDedittest'
fluid_name='SRDedittest'

realisation_name_info= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string)
realisation_name_Mom=realisation_name_info[0]
realisation_name_VP=realisation_name_info[1]
count_mom=realisation_name_info[2]
count_VP=realisation_name_info[3]
realisation_name_log=realisation_name_info[4]
count_log=realisation_name_info[5]
realisation_name_dump=realisation_name_info[6]
count_dump=realisation_name_info[7]
realisation_name_TP=realisation_name_info[8]
count_TP=realisation_name_info[9]
box_size_loc=9
filename_for_lengthscale=realisation_name_VP[0].split('_')
#lengthscale=box_side_length/float(filename_for_lengthscale[box_size_loc])


#checking the number of different solutions used in the run
# # locations of key info when string name is split by undescores

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
box_size.sort()
no_SRD_key=[]
box_size_key=[]
#using list comprehension to remove duplicates
[no_SRD_key.append(x) for x in no_SRD if x not in no_SRD_key]
no_SRD_key.sort(key=int)
[box_size_key.append(x) for x in box_size if x not in box_size_key]
box_side_length_scaled=[]
for item in box_size_key:
    box_side_length_scaled.append(float(item))
box_side_length_scaled=np.array([box_side_length_scaled])
number_of_solutions=len(no_SRD_key)

simulation_file="MYRIAD_LAMMPS_runs/"+filepath
Path_2_VP="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file
# these are the organisation variables which are held in the file names and allow the code to sort them
#reading in velocity profiles, avoid if possible
org_var_1=swap_rate
loc_org_var_1=20
# org_var_1=vel_target
# loc_org_var_1=24
org_var_2=swap_number #spring_constant
loc_org_var_2=22#25

#%% Velocity profiles 
# this cell reads lammps output files, for velocity and temperature profiles. 
# if possible just load in the data as this method is slower. 

VP_raw_data=VP_organiser_and_reader_swap_rate_no_strings_input(loc_no_SRD,loc_org_var_1,loc_org_var_2,loc_Realisation_index,box_side_length_scaled,j_,number_of_solutions,org_var_1,org_var_2,no_SRD_key,realisation_name_VP,Path_2_VP,chunk,equilibration_timesteps,VP_ave_freq,no_timesteps,VP_output_col_count,count_VP)

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

# Temperature profiles
# as long as the VP data have the same structure as the TP data this will work 
TP_raw_data=VP_organiser_and_reader(loc_no_SRD,loc_org_var_1,loc_org_var_2,loc_Realisation_index,box_side_length_scaled,j_,number_of_solutions,org_var_1,org_var_2,no_SRD_key,realisation_name_TP,Path_2_VP,chunk,equilibration_timesteps,VP_ave_freq,no_timesteps,VP_output_col_count,count_VP)

TP_data_upper=TP_raw_data[0]
TP_data_lower=TP_raw_data[1]
error_count=TP_raw_data[2]
filename=TP_raw_data[3]
TP_z_data_upper=TP_raw_data[4]
TP_z_data_lower=TP_raw_data[5]

if error_count != 0: 
    print('Error reading temp profiles, check data !')
else:
    print('Temp profile data success')


#%% reading in mom files (much faster)
#  obtaining the mom data size
#Path_2_mom_file="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file
from post_MPCD_MP_processing_module import *
Path_2_mom_file=Path_2_VP
org_var_mom_1=swap_rate
loc_org_var_mom_1=20
# loc_org_var_mom_1=24
# org_var_mom_1=vel_target
org_var_mom_2=swap_number #spring_constant 
loc_org_var_mom_2=22#25

mom_data_pre_process= mom_file_data_size_reader_swap_rate(j_,number_of_solutions,count_mom,realisation_name_Mom,no_SRD_key,org_var_mom_1,org_var_mom_2,Path_2_mom_file)

size_array=mom_data_pre_process[0]
mom_data= mom_data_pre_process[1]
pass_count= mom_data_pre_process[2]

if pass_count!=count_VP:
    print("Data import error, number of momentum files should match number of velocity files" )
else:
    print("Data size assessment success!")

# Reading in mom data files 

mom_data_files=Mom_organiser_and_reader_swap_rate(mom_data,count_mom,realisation_name_Mom,no_SRD_key,org_var_mom_1,loc_org_var_mom_1,org_var_mom_2,loc_org_var_mom_2,Path_2_mom_file)

mom_data=mom_data_files[0]
error_count_mom=mom_data_files[1]
failed_list_realisations=[2]

if error_count_mom !=0:
    print("Mom file error, check data files aren't damaged")
else:
    print("Mom data import success")


#%% Realisation averaging and calculating statistics 

#VP data 
VP_shear_rate_and_stat_data=VP_data_averaging_and_stat_test_data(VP_z_data_upper,VP_z_data_lower,no_timesteps,VP_data_lower,VP_data_upper,number_of_solutions,org_var_1,org_var_2,VP_ave_freq)
pearson_coeff_upper=VP_shear_rate_and_stat_data[0]
shear_rate_upper=VP_shear_rate_and_stat_data[1]
pearson_coeff_lower=VP_shear_rate_and_stat_data[2]
shear_rate_lower=VP_shear_rate_and_stat_data[3]
timestep_points=VP_shear_rate_and_stat_data[4]
VP_data_lower_realisation_averaged=VP_shear_rate_and_stat_data[5]
VP_data_upper_realisation_averaged=VP_shear_rate_and_stat_data[6]
shear_rate_upper_error=VP_shear_rate_and_stat_data[7]
shear_rate_lower_error=VP_shear_rate_and_stat_data[8]
# TP data

TP_shear_rate_and_stat_data=VP_data_averaging_and_stat_test_data(TP_z_data_upper,TP_z_data_lower,no_timesteps,TP_data_lower,TP_data_upper,number_of_solutions,org_var_1,org_var_2,VP_ave_freq)

TP_data_lower_realisation_averaged=TP_shear_rate_and_stat_data[5]
TP_data_upper_realisation_averaged=TP_shear_rate_and_stat_data[6]

#%% plotting shear rate against timesteps to identify steady state

plt.rcParams.update({'font.size': 20})   
box_size_nd= box_side_length_scaled 

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
labelpadx=15
labelpady=20
fontsize=28
width_plot=8
height_plot=6
org_var_1_index_start=0
org_var_1_index_end=8
org_var_2_index_start=0
org_var_2_index_end=1
shear_rate_plot=(np.abs(shear_rate_upper)+np.abs(shear_rate_lower))*0.5

#yticks= np.arange(0,0.005,0.0004)
def plot_shear_rate_to_asses_SS(org_var_2_index_end,org_var_2_index_start,org_var_1_index_start,org_var_1_index_end,no_timesteps,timestep_points,scaled_temp,number_of_solutions,org_var_1,org_var_2,shear_rate_upper,shear_rate_lower,fluid_name,box_size_nd):
    for z in range(0,number_of_solutions): 
        plt.figure(figsize=(width_plot,height_plot))
        for m in range(org_var_1_index_start,org_var_1_index_end):
             for k in range(org_var_2_index_start,org_var_2_index_end):
                plt.plot(timestep_points[0,0,0,:],shear_rate_plot[z,m,k,:], label="$f_{p}="+str(swap_rate[m])+"$",color=colour[m])
                #plt.plot(timestep_points[0,0,0,:],shear_rate_lower[z,m,k,:])
                plt.xlabel('$N_{t}$', labelpad=labelpadx, fontsize=fontsize)
                plt.ylabel('$\dot{\gamma}$',rotation='horizontal',labelpad=labelpady,fontsize=fontsize)
                
                #plt.yticks(yticks,usetex=True)
             
        plt.legend(loc='best',bbox_to_anchor=(1,1)) 
        
        plt.savefig("plots/"+fluid_name+"_gammadot_vs_timesteps_box_size_var_swap_rate_"+str(int(box_side_length_scaled[0,z]))+"_.pdf", dpi=500, bbox_inches='tight')     
        plt.show()      

plot_shear_rate_to_asses_SS(org_var_2_index_end,org_var_2_index_start,org_var_1_index_start,org_var_1_index_end,no_timesteps,timestep_points,scaled_temp,number_of_solutions,org_var_1,org_var_2,shear_rate_upper,shear_rate_lower,fluid_name,box_size_nd)
#%% R squared plots to assess steady state 

def plot_Rsqaured_to_asses_SS(pearson_coeff_upper,pearson_coeff_lower,org_var_1_index_start,org_var_1_index_end,timestep_points,number_of_solutions,box_side_length_scaled,fluid_name):
    
    Rsquared_for_plot=((pearson_coeff_upper**2) +(pearson_coeff_lower**2) )* 0.5
    for z in range(0,number_of_solutions): 
        for m in range(org_var_1_index_start,org_var_1_index_end):
    
                k=0     
                plt.plot(timestep_points[0,0,0,:],Rsquared_for_plot[z,m,k,:], label="$f_{p}="+str(swap_rate[m])+"$",color=colour[m])  
                plt.xlabel('$N_{t}$', labelpad=labelpadx, fontsize=fontsize)
                plt.ylabel('$R^{2}$',rotation='horizontal',labelpad=labelpady,fontsize=fontsize)
                #plt.yticks(yticks,usetex=True)
                #plt.yscale('log')
                
        plt.legend(loc='best',bbox_to_anchor=(1,1))       
        plt.savefig("plots/"+fluid_name+"_Rsqaured_vs_timesteps_box_size_swap_rate"+str(int(box_side_length_scaled[0,z]))+"_.pdf", dpi=500, bbox_inches='tight')     
        plt.show()
        
plot_Rsqaured_to_asses_SS(pearson_coeff_upper,pearson_coeff_lower,org_var_1_index_start,org_var_1_index_end,timestep_points,number_of_solutions,box_side_length_scaled,fluid_name)


#%% saving data after reading 
name_of_run_for_save=fluid_name+"_scaled_box_size_"+str(box_side_length_scaled[0,0])+"_"+str(box_side_length_scaled[0,-1])+"_"
print(name_of_run_for_save)
np.save("timestep_points_"+name_of_run_for_save,timestep_points)
np.save("shear_rate_lower_"+name_of_run_for_save,shear_rate_lower)
np.save("shear_rate_upper_"+name_of_run_for_save,shear_rate_upper)
np.save("pearson_coeff_upper_"+name_of_run_for_save,pearson_coeff_lower)
np.save("pearson_coeff_lower_"+name_of_run_for_save,pearson_coeff_upper)
np.save("VP_data_lower_realisation_averaged_"+name_of_run_for_save,VP_data_lower_realisation_averaged)
np.save("VP_data_upper_realisation_averaged_"+name_of_run_for_save,VP_data_upper_realisation_averaged)
np.save("TP_data_lower_realisation_averaged_"+name_of_run_for_save,TP_data_lower_realisation_averaged)
np.save("TP_data_upper_realisation_averaged_"+name_of_run_for_save,TP_data_upper_realisation_averaged)
np.save("VP_z_data_lower_"+name_of_run_for_save,VP_z_data_lower)
np.save("VP_z_data_upper_"+name_of_run_for_save,VP_z_data_upper)
np.save("TP_z_data_lower_"+name_of_run_for_save,TP_z_data_lower)
np.save("TP_z_data_upper_"+name_of_run_for_save,TP_z_data_upper)

np.save("shear_rate_upper_error_"+name_of_run_for_save,shear_rate_upper_error)
np.save("shear_rate_lower_error_"+name_of_run_for_save,shear_rate_lower_error)


#%% loading in data option 

name_of_run_for_save=fluid_name+"_scaled_box_size_"+str(box_side_length_scaled[0,0])+"_"+str(box_side_length_scaled[0,-1])+"_"
print(name_of_run_for_save)
timestep_points=np.load("timestep_points_"+name_of_run_for_save+".npy")
shear_rate_lower=np.load("shear_rate_lower_"+name_of_run_for_save+".npy")
shear_rate_upper=np.load("shear_rate_upper_"+name_of_run_for_save+".npy")
pearson_coeff_lower=np.load("pearson_coeff_upper_"+name_of_run_for_save+".npy")
pearson_coeff_upper=np.load("pearson_coeff_lower_"+name_of_run_for_save+".npy")
VP_data_lower_realisation_averaged=np.load("VP_data_lower_realisation_averaged_"+name_of_run_for_save+".npy")
VP_data_upper_realisation_averaged=np.load("VP_data_upper_realisation_averaged_"+name_of_run_for_save+".npy")
TP_data_lower_realisation_averaged=np.load("TP_data_lower_realisation_averaged_"+name_of_run_for_save+".npy")
TP_data_upper_realisation_averaged=np.load("TP_data_upper_realisation_averaged_"+name_of_run_for_save+".npy")
shear_rate_upper_error=np.load("shear_rate_upper_error_"+name_of_run_for_save+".npy")
shear_rate_upper_error=np.load("shear_rate_upper_error_"+name_of_run_for_save+".npy")
shear_rate_lower_error=np.load("shear_rate_lower_error_"+name_of_run_for_save+".npy")
VP_z_data_lower=np.load("VP_z_data_lower_"+name_of_run_for_save+".npy")
VP_z_data_upper=np.load("VP_z_data_upper_"+name_of_run_for_save+".npy")
TP_z_data_lower=np.load("TP_z_data_lower_"+name_of_run_for_save+".npy")
TP_z_data_upper=np.load("TP_z_data_upper_"+name_of_run_for_save+".npy")
#np.save(fluid_name+'_VP_z_data_upper_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_z_data_upper)

#%%
truncation_timestep=500000
truncation_and_SS_averaging_data=  truncation_step_and_SS_average_of_VP_and_stat_tests(shear_rate_upper_error,shear_rate_lower_error,timestep_points,pearson_coeff_lower,pearson_coeff_upper,shear_rate_upper,shear_rate_lower,VP_ave_freq,truncation_timestep,VP_data_lower_realisation_averaged,VP_data_upper_realisation_averaged)
shear_rate_standard_deviation_upper_error_relative=truncation_and_SS_averaging_data[0]/np.sqrt((no_timesteps-truncation_timestep)/VP_ave_freq)# error from fluctuations
shear_rate_standard_deviation_lower_error_relative=truncation_and_SS_averaging_data[1]/np.sqrt((no_timesteps-truncation_timestep)/VP_ave_freq)# error from fluctuations 
shear_rate_standard_deviation_both_cell_error_relative= (np.abs(shear_rate_standard_deviation_upper_error_relative)+np.abs(shear_rate_standard_deviation_lower_error_relative)) * 0.5 
pearson_coeff_upper_mean_SS=truncation_and_SS_averaging_data[2]
r_squared_upper_mean= pearson_coeff_upper_mean_SS**2
pearson_coeff_lower_mean_SS=truncation_and_SS_averaging_data[3]
r_squared_lower_mean= pearson_coeff_lower_mean_SS**2
pearson_coeff_mean_SS= (np.abs(pearson_coeff_lower_mean_SS)+np.abs(pearson_coeff_upper_mean_SS))*0.5
shear_rate_lower_steady_state_mean=truncation_and_SS_averaging_data[4]
shear_rate_upper_steady_state_mean=truncation_and_SS_averaging_data[5]
VP_steady_state_data_lower_truncated_time_averaged=truncation_and_SS_averaging_data[6]
VP_steady_state_data_upper_truncated_time_averaged=truncation_and_SS_averaging_data[7]
shear_rate_upper_steady_state_mean_error=truncation_and_SS_averaging_data[8] # error in fitting velocity profile
shear_rate_lower_steady_state_mean_error=truncation_and_SS_averaging_data[9]# error in fitting velocity profile


# checking the linear fit 
rsquared_mean= (r_squared_lower_mean+r_squared_upper_mean)*0.5
mean_fitting_error= (np.abs(shear_rate_upper_steady_state_mean_error)+np.abs(shear_rate_lower_steady_state_mean_error))*0.5
cell_mean_shear_rate= (np.abs(shear_rate_upper_steady_state_mean)+np.abs(shear_rate_lower_steady_state_mean))*0.5
relative_error_in_velocity_profile_fit=shear_rate_standard_deviation_both_cell_error_relative/cell_mean_shear_rate

shear_rate_mean= (shear_rate_upper_steady_state_mean+ np.abs(shear_rate_lower_steady_state_mean)) *0.5

def plot_meanfitting_error_vs_SS_shear(number_of_solutions,mean_fitting_error,shear_rate_mean):
    for z in range(0,number_of_solutions):
        plt.scatter(shear_rate_mean[z,:,0],mean_fitting_error[z,:,0],label="$L="+str(box_side_length_scaled[0,z])+"$", marker='x', color=colour[z])
        plt.yscale('log')
        plt.xlabel("$\dot{\\gamma}_{SS}$")
        plt.ylabel("$\sigma_{m}$", rotation=0)
        plt.xscale('log')
        plt.legend(loc='best', bbox_to_anchor=(1,1.05))   
    plt.show()


def plot_relative_error_vs_SS_shear(number_of_solutions,relative_error_in_velocity_profile_fit,shear_rate_mean):
    for z in range(0,number_of_solutions):
        plt.scatter(shear_rate_mean[z,:,0],relative_error_in_velocity_profile_fit[z,:,0],label="$L="+str(box_side_length_scaled[0,z])+"$", marker='x', color=colour[z])
        plt.yscale('log')
        plt.xlabel("$\dot{\\gamma}_{SS}$")
        plt.ylabel("$\\frac{\sigma_{std}}{\dot{\\gamma}_{SS}}$", rotation=0)
        plt.xscale('log')
        plt.legend(loc='best', bbox_to_anchor=(1,1.05))   
    plt.show()

def plot_Rsquared_vs_SS_shear(number_of_solutions,rsquared_mean,shear_rate_mean):
    for z in range(0,number_of_solutions):
        plt.scatter(shear_rate_mean[z,:,0],rsquared_mean[z,:,0],label="$L="+str(box_side_length_scaled[0,z])+"$", marker='x', color=colour[z])
       
        #plt.yscale('log')
        plt.xlabel("$\dot{\\gamma}_{SS}$")
        plt.ylabel("$R^{2}$", rotation=0, labelpad=20)
        plt.xscale('log')
        plt.legend(loc='best', bbox_to_anchor=(1,1.05))   
    plt.hlines(0.55,0,np.max(shear_rate_mean), linestyles='dashed')
    plt.show()
    
    
plot_meanfitting_error_vs_SS_shear(number_of_solutions,mean_fitting_error,shear_rate_mean)

plot_relative_error_vs_SS_shear(number_of_solutions,relative_error_in_velocity_profile_fit,shear_rate_mean)

plot_Rsquared_vs_SS_shear(number_of_solutions,rsquared_mean,shear_rate_mean)

# temperature data 
truncation_and_SS_averaging_data_TP=  truncation_step_and_SS_average_of_VP_and_stat_tests(shear_rate_upper_error,shear_rate_lower_error,timestep_points,pearson_coeff_lower,pearson_coeff_upper,shear_rate_upper,shear_rate_lower,VP_ave_freq,truncation_timestep,TP_data_lower_realisation_averaged,TP_data_upper_realisation_averaged)

TP_steady_state_data_lower_truncated_time_averaged=truncation_and_SS_averaging_data_TP[6]
TP_steady_state_data_upper_truncated_time_averaged=truncation_and_SS_averaging_data_TP[7]

#%% plotting mean shear rate for each box size 
plt.rcParams['text.usetex'] = True
marker=['x','o','+','^',"1","X","d","*","P","v"]
#yticks= np.arange(0,0.00275,0.00025)
box_side_length_scaled_for_plot=np.repeat(box_side_length_scaled,org_var_1.size,axis=0)
shear_rate_mean= (shear_rate_upper_steady_state_mean+ np.abs(shear_rate_lower_steady_state_mean)) *0.5
# get fitting scaling for box size with shear rate 
# def func_exp(a,b,c,x):
def func_exp(a,b,x):   
    #return a*np.exp(b*x)+c
    #return 10**((a*x) + b)
    #return np.log(a) + np.log(b*x)
    #return a*np.log(x)+b
    return  (a*np.exp((b)*x))
   

fitting_for_shear_box_size_curve = np.zeros((org_var_1.size,3))

for z in range(0,org_var_1.size):

        fit = scipy.optimize.curve_fit(func_exp,box_side_length_scaled_for_plot[z,:],shear_rate_mean[:,z,0], p0=[0, 0], bounds=(-np.inf, np.inf))[0]
        #fit = np.polyfit(box_side_length_scaled_for_plot[z,:],np.log(shear_rate_mean[:,z,0]),1)
        fitting_for_shear_box_size_curve[z,0]=fit[0]
        fitting_for_shear_box_size_curve[z,1]=fit[1]
        #fitting_for_shear_box_size_curve[z,2]=fit[2]

labelpadx=15
labelpady=20
fontsize=28
width_plot=8
height_plot=6
plt.figure(figsize=(width_plot,height_plot))
for z in range(0,org_var_1.size):
#for z in range(0, org_var_1.size):
    #for i in range(0,org_var_1.size):
        
        #plt.plot(box_side_length_scaled_for_plot[z,:],func_exp(fitting_for_shear_box_size_curve[z,0],fitting_for_shear_box_size_curve[z,1],box_side_length_scaled_for_plot[z,:]))
        plt.scatter(box_side_length_scaled_for_plot[z,:],shear_rate_mean[:,z,0], label="$f_{p}="+str(swap_rate[z])+"$",marker=marker[z], color=colour[z])  
        plt.xlabel("$L$")
        plt.ylabel("$\dot{\gamma}_{SS}$", rotation=0, labelpad=labelpady)
        #plt.yscale('log')
        #plt.yticks(yticks,usetex=True)
plt.legend(loc='best', bbox_to_anchor=(1,1))   

plt.savefig("plots/test_with_"+str(number_of_solutions)+"_solutions_steady_shear_rate.pdf",dpi=500, bbox_inches='tight')
plt.show()


#%% R^2 test on steady state velocity profiles 
plt.rcParams['text.usetex'] = True
marker=['x','o','+','^',"1","X","d","*","P","v",'.']
linestyle_tuple = [
     ('dotted',                (0, (1, 1))),
     ('dashed',                (0, (5, 5))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
# need to fix legend location 
plt.rcParams.update({'font.size': 25})
org_var_1_choice_index=org_var_1.size
fontsize=35
labelpadx=10
labelpady=15
width_plot=12
height_plot=6
plt.figure(figsize=(width_plot,height_plot))
legend_x_pos=1
legend_y_pos=1
org_var_1_index_start=0
org_var_1_index_end=8
org_var_2_index_start=0
org_var_2_index_end=1
def func_linear(x,a,b):
    return (a*x) + b 

# now testing R^2 of steady state profiles 
def R_squared_test_on_steady_state_vps_swaprate(number_of_solutions,org_var_2,org_var_1,VP_steady_state_data_lower_truncated_time_averaged):
    r_squared_of_steady_state_VP_upper= np.zeros((number_of_solutions,org_var_2.size,org_var_1.size))
    r_squared_of_steady_state_VP_lower= np.zeros((number_of_solutions,org_var_2.size,org_var_1.size))
    for z in range(0,number_of_solutions):
    
        for m in range(0,org_var_2.size):
       
            for k in range(0,org_var_1.size):  
                y_1=VP_steady_state_data_lower_truncated_time_averaged[z,k,m,:]
                #print(x_1.shape)
                y_2=VP_steady_state_data_upper_truncated_time_averaged[z,k,m,:]
                x_1=VP_z_data_lower[z,0,:]
                #print(y_1.shape)
                x_2=VP_z_data_upper[z,0,:]
                #print(k)
                #x_1[:],y_1[:]
                #for i in range(org_var_2_index_start,org_var_1_index_end):
                # a=scipy.stats.linregress(x_1[:],y_1[:]).slope
                # b=scipy.stats.linregress(x_1[:],y_1[:]).intercept
                # c=scipy.stats.linregress(x_1[:],y_1[:]).stderr
                r_squared_of_steady_state_VP_lower[z,m,k]=(scipy.stats.linregress(x_1[:],y_1[:]).rvalue)**2
                r_squared_of_steady_state_VP_upper[z,m,k]=(scipy.stats.linregress(x_2[:],y_2[:]).rvalue)**2
    
    r_squared_of_steady_state_VP=  (r_squared_of_steady_state_VP_lower + r_squared_of_steady_state_VP_upper)*0.5
    plt.figure(figsize=(width_plot,height_plot))
    for z in range(0,number_of_solutions):
        plt.scatter(org_var_1[:], r_squared_of_steady_state_VP[z,0,:], label="$L="+str(int(box_side_length_scaled[:,z]))+"$", color=colour[z])
        #plt.yscale('log')
        plt.xlabel("$f_{p}$", rotation=0, labelpad=labelpadx)
        plt.ylabel("$R^{2}$",rotation=0, labelpad=20)
        
    plt.hlines(0.7,0,360,linestyle='dashed',label="$R^{2}_{tol}=0.7$")
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig("plots/"+fluid_name+"_R_squared_swaprate_test_with_cutoff.pdf",dpi=500, bbox_inches='tight')
    plt.show()
    
    
    return  r_squared_of_steady_state_VP
    
R_squared_test_on_steady_state_vps_swaprate(number_of_solutions,org_var_2,org_var_1,VP_steady_state_data_lower_truncated_time_averaged)

#%% plottingvelocity profiles that passed R^2 and those that didnt 
# need to add all these settings for every plot
#yticks=np.arange(-0.09,0.11,0.02)
width_plot=9
height_plot=8
org_var_1_index_start=0
org_var_1_index_end=6
org_var_2_index_start=0
org_var_2_index_end=1
def plotting_SS_velocity_profiles(org_var_2_index_start,org_var_1_index_end,legend_x_pos, legend_y_pos,labelpadx,labelpady,fontsize,number_of_solutions,org_var_1_choice_index,width_plot,height_plot,org_var_1,org_var_2,VP_ave_freq,no_timesteps,VP_steady_state_data_lower_truncated_time_averaged,VP_steady_state_data_upper_truncated_time_averaged,VP_z_data_lower,VP_z_data_upper):
    for z in range(0,number_of_solutions):
    
        for m in range(0,org_var_2.size):
        #for k in range(0,org_var_1.size):  
            
            fig=plt.figure(figsize=(width_plot,height_plot))
            gs=GridSpec(nrows=1,ncols=1)

            ax1= fig.add_subplot(gs[0,0])
    
        


            for k in range(org_var_1_index_start,org_var_1_index_end):  
                y_1=VP_steady_state_data_lower_truncated_time_averaged[z,k,m,:]
                #print(x_1.shape)
                x_2=VP_steady_state_data_upper_truncated_time_averaged[z,k,m,:]
                x_1=VP_z_data_lower[z,0,:]
                #print(y_1.shape)
                y_2=VP_z_data_upper[z,0,:]
                #print(k)
                #x_1[:],y_1[:]
                #for i in range(org_var_2_index_start,org_var_1_index_end):
                a=scipy.stats.linregress(x_1[:],y_1[:]).slope
                b=scipy.stats.linregress(x_1[:],y_1[:]).intercept
                c=scipy.stats.linregress(x_1[:],y_1[:]).stderr
                d=scipy.stats.linregress(x_1[:],y_1[:]).rvalue
                
                
                #print(k)
                #for i in range(org_var_2_index_start,org_var_1_index_end):
                
                ax1.plot(x_1[:],func_linear(x_1[:],a,b),linestyle=linestyle_tuple[k][1],linewidth=3,color=colour[k])
                ax1.scatter(x_1[:],y_1[:],label='$f_{p}= '+str(org_var_1[k])+', R^{2}='+str(sigfig.round(d**2,sigfigs=4))+'$',marker=marker[k],color=colour[k])
                
               # ax1.plot(y_1[:],x_1[:],label='$v_{target}=\\pm '+str(org_var_1[k])+'$',marker=marker[k], markersize=6 ,linestyle=linestyle_tuple[k][1],linewidth=3,color=colour[k])
                ax1.set_ylabel('$v_{x}$',rotation=0,labelpad=labelpady,fontsize=fontsize)
                ax1.set_xlabel('$x_{z}$',rotation=0,labelpad=labelpadx,fontsize=fontsize)
               # ax1.set_title('$\\bar{L}='+str(box_side_length_scaled[0,z])+'$')
                ax1.legend(loc=0,bbox_to_anchor=(legend_x_pos, legend_y_pos))     
        
            plt.savefig("plots/"+fluid_name+"_velocity_profile_swaprate_var_box_size_"+str(int(box_side_length_scaled[0,z]))+"_range_vt_"+str(org_var_1[org_var_1_index_start-1])+"_"+str(org_var_1[org_var_1_index_end-1])+".pdf", dpi=500, bbox_inches='tight')             #plt.yticks(yticks,usetex=True)  
        plt.show()
    

plotting_SS_velocity_profiles(org_var_2_index_start,org_var_1_index_end,legend_x_pos, legend_y_pos,labelpadx,labelpady,fontsize,number_of_solutions,org_var_1_choice_index,width_plot,height_plot,org_var_1,org_var_2,VP_ave_freq,no_timesteps,VP_steady_state_data_lower_truncated_time_averaged,VP_steady_state_data_upper_truncated_time_averaged,VP_z_data_lower,VP_z_data_upper)



#%% checking grad of datta
#assess gradient of truncated shear rate data to determine steady state
# can then truncate again   
os.getcwd()
truncation_index=int(truncation_timestep/VP_ave_freq)
slope_shear_rate_upper=  np.zeros((number_of_solutions,org_var_1.size,org_var_2.size))
slope_shear_rate_lower=  np.zeros((number_of_solutions,org_var_1.size,org_var_2.size))
slope_Rsquared=  np.zeros((number_of_solutions,org_var_1.size,org_var_2.size))
Rsquared_for_plot=((pearson_coeff_upper**2) +(pearson_coeff_lower**2) )* 0.5

gradient_tolerance= 1e-7
for z in range(0,number_of_solutions): 
        for m in range(org_var_1_index_start,org_var_1_index_end):
            for k in range(0,1):
                slope_Rsquared[z,m,k]=np.polyfit(timestep_points[0,0,0,truncation_index:],Rsquared_for_plot[z,m,k,truncation_index:],1)[0]
              
                if np.abs(slope_Rsquared[z,m,k]) < gradient_tolerance:
                    slope_Rsquared[z,m,k] =slope_Rsquared[z,m,k]
                    #slope_shear_rate_upper[z,m,k]='NaN'
                else: 
                   # slope_Rsquared[z,m,k] ='NaN'
                    print('FAILED run, exclude from data ')
#print("if no fail statements, data can be considered steady")

for z in range(0,number_of_solutions): 
        for m in range(org_var_1_index_start,org_var_1_index_end):
            for k in range(0,1):
                slope_shear_rate_upper[z,m,k]=np.polyfit(timestep_points[0,0,0,truncation_index:],shear_rate_upper[z,m,k,truncation_index:],1)[0]
                slope_shear_rate_lower[z,m,k]=np.polyfit(timestep_points[0,0,0,truncation_index:],shear_rate_lower[z,m,k,truncation_index:],1)[0]
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
# # plotting gradient of the shear vs time plot 
# fontsize=35
# labelpadx=15
# labelpady=30

# legend_x_pos=1
# legend_y_pos=1
# for z in range(0,number_of_solutions): 
#      #for m in range(org_var_1_index_start,org_var_1_index_end):
#         #for k in range(org_var_2_index_start,org_var_2_index_end):
#            # plt.yscale('log')
#             plt.ylabel(' $\\frac{d \dot{\gamma}\\tau^{2}}{d t} $',rotation=0)
#             plt.xlabel('$f_{p}[-]$')
#             #plt.xlabel('$\\pm v_{target}$',labelpad=labelpadx)
#             #plt.xlabel('$K[\\frac{\\tau}{\mu}]$')
#             #plt.xscale('log')
#             plt.scatter(org_var_1[:],slope_shear_rate_upper[z,:,0],color=colour[z])
#             plt.scatter(org_var_1[:],slope_shear_rate_lower[z,:,0],color=colour[z])
#             #plt.title("Needs a title")
# plt.savefig("plots/"+fluid_name+"_shear_rate_steady_state_plot_all_boxes_var_swap.pdf", dpi=500, bbox_inches='tight')    
# plt.show()


#%%
# importing momentum after steady state
def mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,org_var_1,truncation_timestep,org_var_2,scaled_timestep,no_timesteps,box_side_length_scaled,mom_data):
    mom_data_realisation_averaged=()
    number_swaps_before_truncation=(np.ceil(truncation_timestep/swap_rate)).astype(int)
    mom_data_realisation_averaged_truncated=()
    flux_x_momentum_z_direction=np.zeros((number_of_solutions,org_var_1.size,org_var_2.size))
    total_run_time=scaled_timestep* (no_timesteps-truncation_timestep)
    
    flux_ready_for_plotting=np.zeros((number_of_solutions,org_var_1.size,org_var_2.size))
    for z in range(0,number_of_solutions):  
        box_area_nd=float(box_size_key[z])**2
        for j in range(0,org_var_1.size):
               
                mom_data_realisation_averaged=mom_data_realisation_averaged+(np.mean(mom_data[j],axis=2),)
                mom_data_realisation_averaged_truncated=mom_data_realisation_averaged_truncated+(mom_data_realisation_averaged[j][:,:,number_swaps_before_truncation[j]:],)
                for i in range(0,org_var_2.size):
                    


            #for i in range(0,org_var_2.size):

                    #mom_data_realisation_averaged_truncated=mom_data_realisation_averaged_truncated+(mom_data_realisation_averaged[j][:,:,number_swaps_before_truncation[i]:],)
                    #mom_data_realisation_averaged_truncated=mom_data_realisation_averaged_truncated+(mom_data_realisation_averaged[j][:,:,number_swaps_before_truncation[j]:],)
                    #print(mom_data_realisation_averaged_truncated.shape)


            # now apply the MP formula 
                    mom_difference= mom_data_realisation_averaged_truncated[j][z,i,-1]-mom_data_realisation_averaged_truncated[j][z,i,0]
    #                 print(mom_difference)
                    flux_x_momentum_z_direction[z,j,i]=(mom_difference)/(2*total_run_time*float(box_area_nd))
                
    flux_ready_for_plotting=np.log((np.abs(flux_x_momentum_z_direction)))
    
    return flux_ready_for_plotting,mom_data_realisation_averaged_truncated
#%%
# only use this version for varying vel swap 
#def mom_data_averaging_and_flux_calc(org_var_1,org_var_2,box_size_key,number_of_solutions,swap_number,truncation_timestep,swap_rate,scaled_timestep,no_timesteps,box_side_length_scaled,mom_data):
mom_data_realisation_averaged=()
number_swaps_before_truncation=(np.ceil(truncation_timestep/swap_rate)).astype(int)
mom_data_realisation_averaged_truncated=()
flux_x_momentum_z_direction=np.zeros((number_of_solutions,org_var_2.size,org_var_1.size))
total_run_time=scaled_timestep* (no_timesteps-truncation_timestep)

flux_ready_for_plotting=np.zeros((number_of_solutions,org_var_2.size,org_var_1.size))
box_area_nd=(box_side_length_scaled)**2
box_area_nd=np.asarray(box_area_nd, dtype = np.float64 ) 

#for z in range(0,number_of_solutions):    
for i in range(0,org_var_1.size):
   # box_area_nd=float(box_size_key)**2
    mom_data_realisation_averaged=mom_data_realisation_averaged+(np.mean(mom_data[i],axis=2),)
    # print( mom_data_realisation_averaged[i])
    # print(i)
        #print(mom_data_realisation_averaged[i])

#for z in range(0,number_of_solutions):    
for i in range(0,org_var_1.size):

    #for i in range(0,swap_rate.size):

    mom_data_realisation_averaged_truncated=mom_data_realisation_averaged_truncated+(mom_data_realisation_averaged[i][:,:,number_swaps_before_truncation[i]:],)
        
        #print(mom_data_realisation_averaged_truncated[i])

for z in range(0,number_of_solutions):    
    for i in range(0,org_var_1.size):
        # now apply the MP formula 
            mom_difference= mom_data_realisation_averaged_truncated[i][z,:,-1]-mom_data_realisation_averaged_truncated[i][z,:,0]
            flux_x_momentum_z_direction[z,:,i]=(mom_difference)/(2*total_run_time*box_area_nd[0,z])
        
flux_ready_for_plotting=np.log((np.abs(flux_x_momentum_z_direction)))
    
   # return flux_ready_for_plotting,mom_data_realisation_averaged_truncated

#flux_ready_for_plotting=mom_data_averaging_and_flux_calc(org_var_1,org_var_2,box_size_key,number_of_solutions,org_var_1,truncation_timestep,org_var_2,scaled_timestep,no_timesteps,box_side_length_scaled[0,0],mom_data)[0]
#mom_data_realisation_averaged_truncated=mom_data_averaging_and_flux_calc(org_var_1,org_var_2,box_size_key,number_of_solutions,org_var_1,truncation_timestep,org_var_2,scaled_timestep,no_timesteps,box_side_length_scaled[0,0],mom_data)[1]


shear_rate_mean_of_both_cells=(((np.abs(shear_rate_lower_steady_state_mean)+np.abs(shear_rate_upper_steady_state_mean))*0.5))
shear_rate_mean_error_of_both_cells=np.sqrt(((shear_rate_lower_steady_state_mean_error**2)+(shear_rate_upper_steady_state_mean_error**2))*0.5)
# print(shear_rate_mean_of_both_cells.shape)
# print(shear_rate_mean_error_of_both_cells.shape)
shear_rate_mean_error_of_both_cells_relative=(shear_rate_mean_error_of_both_cells/shear_rate_mean_of_both_cells) 
shear_rate_mean_of_both_cells=np.log(((np.abs(shear_rate_lower_steady_state_mean)+np.abs(shear_rate_upper_steady_state_mean))*0.5))
shear_rate_mean_error_of_both_cells=shear_rate_mean_of_both_cells*shear_rate_mean_error_of_both_cells_relative
shear_rate_mean_error_of_both_cells=shear_rate_mean_of_both_cells*shear_rate_standard_deviation_both_cell_error_relative
# print(shear_rate_mean_of_both_cells.shape)
# print(shear_rate_mean_error_of_both_cells.shape)

flux_vs_shear_regression_line_params=()
x=shear_rate_mean_of_both_cells
#shear_rate_mean_of_both_cells=np.reshape(shear_rate_mean_of_both_cells,(flux_ready_for_plotting.shape))
# fiting params
# def func4(x, a, b):
#    #return np.log(a) + np.log(b*x)
#    #return (a*(x**b))
#    #return 10**(a*np.log(x) + b)
#    return (a*x) +b 
#    #return a*np.log(b*x)+c

def func4(x,c):
    
    return x + c



org_var_1_fitting_start_index=0
org_var_1_fitting_end_index=4
size_of_new_data=org_var_1_fitting_end_index-org_var_1_fitting_start_index
#shear_rate_error_of_both_cell_mean_over_all_points_relative = shear_rate_mean_error_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:]/shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:]
shear_rate_mean_error_of_both_cell_mean_over_selected_points_relative= np.zeros((box_side_length_scaled.size))
relative_y_residual_mean=np.zeros((box_side_length_scaled.size))
# mean_flux_ready_for_plotting=np.mean(flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:],axis=1)
# mean_flux_ready_for_plotting_relative_error=np.zeros((org_var_2.size))
# mean_error_in_fit =np.zeros((org_var_2.size))
y_residual_in_fit=np.zeros((number_of_solutions,size_of_new_data,org_var_2.size))
for z in range(0,number_of_solutions):    
    shear_rate_mean_error_of_both_cell_mean_over_selected_points_relative[z]= np.mean(np.abs(shear_rate_mean_error_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:]),axis=0)

    for i in range(0,org_var_2.size):
        
        # with varaible swap rate 
        # flux_vs_shear_regression_line_params= flux_vs_shear_regression_line_params+(scipy.optimize.curve_fit(func4,shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],method='lm',maxfev=5000)[0],)
        # #y_residual_in_fit[z,:,i]=func4(shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],flux_vs_shear_regression_line_params[i][0] ,flux_vs_shear_regression_line_params[i][1])-flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i]
        # y_residual_in_fit[z,:,i]=func4(shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],flux_vs_shear_regression_line_params[i][0])-flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i]
        
         # with only one swap rate 
        flux_vs_shear_regression_line_params= flux_vs_shear_regression_line_params+(scipy.optimize.curve_fit(func4,shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],flux_ready_for_plotting[z,i,org_var_1_fitting_start_index:org_var_1_fitting_end_index],method='lm',maxfev=5000)[0],)
        #y_residual_in_fit[z,:,i]=func4(shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],flux_vs_shear_regression_line_params[i][0] ,flux_vs_shear_regression_line_params[i][1])-flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i]
        # y_residual_in_fit[z,:,i]=func4(shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],flux_vs_shear_regression_line_params[i][0])-flux_ready_for_plotting[z,i,org_var_1_fitting_start_index:org_var_1_fitting_end_index]
        
        # print(y_residual_in_fit.shape)
        # print(flux_ready_for_plotting[z,i,org_var_1_fitting_start_index:org_var_1_fitting_end_index].shape)
        # relative_y_residual_mean[z]= np.mean(np.abs(y_residual_in_fit[z,:,i]/flux_ready_for_plotting[z,i,org_var_1_fitting_start_index:org_var_1_fitting_end_index]),axis=0)


        #mean_error_in_fit[i] = np.sqrt(np.mean((func4(shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],flux_vs_shear_regression_line_params[i][0] ,flux_vs_shear_regression_line_params[i][1])-flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i])**2))
        #mean_flux_ready_for_plotting_relative_error[i]=mean_error_in_fit[i]/mean_flux_ready_for_plotting[i]
        #print(scipy.optimize.curve_fit(func4,shear_rate_mean_of_both_cells[z,:,i],flux_ready_for_plotting[z,i,:],method='lm',maxfev=5000)[0])

params=flux_vs_shear_regression_line_params
#relative_y_residual_mean= np.abs(np.mean(y_residual_in_fit/flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:],axis=1))
# only for vel swap tests 
#relative_y_residual_mean= np.abs(np.mean(y_residual_in_fit/flux_ready_for_plotting[z,i,org_var_1_fitting_start_index:org_var_1_fitting_end_index],axis=1))
# need to do this as a fit to each viscosity take a mean, then take the mean of the residuals 

#total_error_relative_in_flux_fit= relative_y_residual_mean+shear_rate_mean_error_of_both_cell_mean_over_selected_points_relative

viscosity_fit_individual=  np.zeros((number_of_solutions,org_var_1_fitting_end_index-org_var_1_fitting_start_index))
viscosity_fit_mean_individual=[]
for z in range(0,number_of_solutions):    
    viscosity_fit_mean_individual.append(np.exp(params[z][0]))
    for i in range(0,org_var_1_fitting_end_index-org_var_1_fitting_start_index):
      viscosity_fit_individual[z,i]=np.exp(flux_ready_for_plotting[z,0,i]-shear_rate_mean_of_both_cells[z,i,0])

#np.mean(viscosity_fit_individual,axis=1)
viscosity_fit_mean_individual=np.array([viscosity_fit_mean_individual])
viscosity_fit_mean_for_comparison = np.repeat(viscosity_fit_mean_individual,org_var_1_fitting_end_index-org_var_1_fitting_start_index,axis=0).T
viscosity_fit_residual=  viscosity_fit_individual- viscosity_fit_mean_for_comparison
tolerance=0.08
tolerance_test_=np.abs(viscosity_fit_residual)/viscosity_fit_mean_for_comparison
tolerance_test=np.abs(viscosity_fit_residual)/viscosity_fit_mean_for_comparison
print(tolerance_test_.shape)
print(tolerance_test)
if np.any(tolerance_test>tolerance):
    print('fitting not acceptable')
else:
    print('viscosity fit accepted')

viscosity_fit_absolute_error= np.mean(np.abs(viscosity_fit_residual), axis=1)
viscosity_fit_relative_error= viscosity_fit_absolute_error/viscosity_fit_mean_individual[0,:]
total_error_relative_in_flux_fit= viscosity_fit_relative_error+shear_rate_mean_error_of_both_cell_mean_over_selected_points_relative




#%% plotting with only one swap rate 
# need to fit the curve to just the linear part of plot 
os.getcwd()
plt.rcParams.update({'font.size': 15})
labelpadx=15
labelpady=55
fontsize=25
width_plot=8
height_plot=6
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
marker=['x','o','+','^',"1","X","d","*","P","v","."]
shear_viscosity=[]
shear_viscosity_abs_error=[]
plt.figure(figsize=(width_plot,height_plot))
for z in range(0,number_of_solutions):
    # only fitting section 
    
    x_=shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:]

    # x_pos_error=np.abs(shear_rate_mean_error_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:])
    # #y_pos_error=np.abs(abs_error_in_flux[z,:,:])
    #y=flux_ready_for_plotting[z,:,org_var_1_fitting_start_index:org_var_1_fitting_end_index]
    
    # only fitting section 
    x=shear_rate_mean_of_both_cells[z,:,:]

    x_pos_error=np.abs(shear_rate_mean_error_of_both_cells[z,:,:])
    #y_pos_error=np.abs(abs_error_in_flux[z,:,:])
    y=flux_ready_for_plotting[z,:,:]



# for i in range(0,org_var_2_index):

    for i in range(0,1):
    #     #if z==0:

        j=i
        shear_viscosity_=np.exp( (params[z][0]))
        shear_viscosity_abs_error_= (shear_viscosity_ * total_error_relative_in_flux_fit[z])
        
        shear_viscosity.append((np.exp( (params[z][0]))))
        shear_viscosity_abs_error.append(shear_viscosity[z] * total_error_relative_in_flux_fit[z])
        


        # need to add legend to this 
        plt.scatter(x[:,i],y[i,:],label='$L='+str(np.around(box_side_length_scaled[0,z]))+",\ \eta="+str(sigfig.round(shear_viscosity_,sigfigs=2))+"\pm"+str(sigfig.round(shear_viscosity_abs_error_,sigfigs=2))+"$",marker=marker[z], color= colour[z])
        #plt.errorbar(x[:,i],y[i,:],xerr=x_pos_error[:,i],ls ='',capsize=3,color='r')
        # # plt.plot(x[:,i],func4(x[:,i],params[z][0],params[z][1]))
        plt.plot(x_[:,i],func4(x_[:,i],params[z][0]), color= colour[z])
        # #plt.fill_between(y[:,i], x_neg_error[i,:], x_pos_error[i,:])
        #plt.xscale('log')
        plt.xlabel('$log(\dot{\gamma})$', labelpad=labelpadx,fontsize=fontsize)
        #plt.yscale('log')
        plt.ylabel('$log(J_{z}(p_{x}))$',rotation=0,labelpad=labelpady,fontsize=fontsize)
        plt.legend(loc='best',bbox_to_anchor=(1,1))
        #plt.show() 
        # shear_viscosity_=10** (params[z][0])
        # shear_viscosity.append(shear_viscosity_)
        # shear_viscosity_abs_error.append(shear_viscosity_*total_error_relative_in_flux_fit[z,i])

        # grad_fit=(params[z][0])
        # grad_fit_abs_error= grad_fit*shear_rate_mean_error_of_both_cell_mean_over_selected_points_relative[z]
        # gradient_of_fit.append(grad_fit)

    # print('Dimensionless_shear_viscosity:',shear_viscosity_,',abs error',shear_viscosity_abs_error)
        #print('Grad of fit =',grad_fit,',abs error', grad_fit_abs_error)


plt.savefig("plots/"+fluid_name+"_flux_vs_shear_all_sizes_swap_rate.pdf",dpi=500, bbox_inches='tight' )
plt.show() 
print("shear visc mean:",np.mean(shear_viscosity),"$\pm$",np.mean(shear_viscosity_abs_error))




#%% plotting temp profiles 
T_P_for_plotting = np.mean(np.mean(1-(TP_steady_state_data_lower_truncated_time_averaged+TP_steady_state_data_upper_truncated_time_averaged)*0.5, axis=3),axis=1)
print(np.mean(T_P_for_plotting)*100,"%")
plt.rcParams.update({'font.size': 25})
org_var_1_choice_index=org_var_1.size
fontsize=45
labelpadx=15
labelpady=45
width_plot=16.5
height_plot=7
legend_x_pos=1
legend_y_pos=1
org_var_1_index_start=0
org_var_1_index_end=8
org_var_2_index_start=0
org_var_2_index_end=1
def plotting_SS_Temp_profiles(org_var_2_index_start,org_var_1_index_end,legend_x_pos, legend_y_pos,labelpadx,labelpady,fontsize,number_of_solutions,org_var_1_choice_index,width_plot,height_plot,org_var_1,org_var_2,VP_ave_freq,no_timesteps,TP_steady_state_data_lower_truncated_time_averaged,TP_steady_state_data_upper_truncated_time_averaged,TP_z_data_lower,TP_z_data_upper):
    for z in range(0,number_of_solutions):
    
        for m in range(0,org_var_2.size):
        #for k in range(0,org_var_1.size):  
            
            fig=plt.figure(figsize=(width_plot,height_plot))
            gs=GridSpec(nrows=1,ncols=2)

            ax1= fig.add_subplot(gs[0,1])
            ax2= fig.add_subplot(gs[0,0])
    
        


            for k in range(org_var_1_index_start,org_var_1_index_end):  
                x_1=TP_steady_state_data_lower_truncated_time_averaged[z,k,m,:]
                #print(x_1.shape)
                x_2=TP_steady_state_data_upper_truncated_time_averaged[z,k,m,:]
                y_1=TP_z_data_lower[z,0,:]
                #print(y_1.shape)
                y_2=TP_z_data_upper[z,0,:]
                #print(k)
                #for i in range(org_var_2_index_start,org_var_1_index_end):
                
                ax1.plot(y_1[:],x_1[:],label='$f_p=${}'.format(org_var_1[k]),marker='x')
                
                #ax1.set_ylabel('$v_{x}\ [\\frac{\\tau}{\ell}]$',rotation=0,labelpad=labelpady, fontsize=fontsize)
                ax1.set_xlabel('$L_{z}/\ell$',rotation=0,labelpad=labelpadx,fontsize=fontsize)
                ax2.plot(y_2[:],x_2[:],label='$f_p=${}'.format(org_var_1[k]),marker='x')
                ax2.set_ylabel('$ T\\frac{ k_{B}}{\epsilon}$',rotation=0,labelpad=labelpady, fontsize=fontsize)
                ax2.set_xlabel('$L_{z}/\ell$',rotation=0,labelpad=labelpadx,fontsize=fontsize)
                ax1.legend(loc=0,bbox_to_anchor=(legend_x_pos, legend_y_pos))      
            plt.savefig("plots/"+fluid_name+"_temp_profile_box_size_swap_var_"+str(int(box_side_length_scaled[0,z]))+"_.pdf",dpi=500, bbox_inches='tight')
            plt.show()
    

plotting_SS_Temp_profiles(org_var_2_index_start,org_var_1_index_end,legend_x_pos, legend_y_pos,labelpadx,labelpady,fontsize,number_of_solutions,org_var_1_choice_index,width_plot,height_plot,org_var_1,org_var_2,VP_ave_freq,no_timesteps,TP_steady_state_data_lower_truncated_time_averaged,TP_steady_state_data_upper_truncated_time_averaged,TP_z_data_lower,TP_z_data_upper)
     
#%%
#calculating theoretical viscosity 

def kinetic_visc_dimensionless(M):
    kinetic_visc_dimensionless=((5*M)/((M-1+np.exp(-M))*(2-np.cos(np.pi/2)-np.cos(np.pi))))-1
    
    return kinetic_visc_dimensionless

def collisional_visc_dimensionless(d,M):
    collisional_visc_dimensionless= (1/(6*d*M)) * ((M-1+np.exp(-M))* (1-np.cos(np.pi/2)))
    
    return collisional_visc_dimensionless
    

d=3
M=5
predicted_dimensionless_shear_viscosity= np.repeat((collisional_visc_dimensionless(d,M) +kinetic_visc_dimensionless(M)),len(shear_viscosity))
    
    
##%% shear viscosity vs box size
#NOTE: need to add error bars to this 
plt.rcParams['text.usetex'] = True
labelpadx=5
labelpady=25
fontsize=25
plt.rcParams.update({'font.size': 15})
width_plot=8
height_plot=6
#shear_viscosity_abs_error_max=np.amax(shear_viscosity_abs_error,axis=0)
os.getcwd()
x=box_side_length_scaled[0,:]
y=shear_viscosity[:]
y_error_bar=np.abs(shear_viscosity_abs_error[:])/np.sqrt(org_var_1.size)
plt.figure(figsize=(width_plot,height_plot))  
#plt.scatter(x[:,i],y[:,i],label="$L=$"+str(box_side_length_scaled[z])+", grad$=$"+str(sigfig.round(grad_fit,sigfigs=2))+"$\pm$"+str(sigfig.round(grad_fit_abs_error,sigfigs=1)),marker='x')
#plt.plot(x,y,"--",marker= 'x')#label="$N_{v,x}=$"+str(org_var_2[z])+", $\Delta\eta_{max}=$"+str(sigfig.round(shear_viscosity_abs_error_max[z],sigfigs=2)),marker='x')
plt.errorbar(x,y,yerr=y_error_bar,capsize=3,color='r', label="Simulation data",linestyle='dashed')
plt.xlabel('$L$', labelpad=labelpadx,fontsize=fontsize)
plt.plot(x, predicted_dimensionless_shear_viscosity[:], linestyle='solid', label="Tuzel,Ihle and Kroll, 2006")
#plt.yscale('log')
#plt.xscale('log')
plt.ylabel('$\eta $',rotation=0,labelpad=labelpady,fontsize=fontsize)
plt.legend(loc='center right',bbox_to_anchor=(0.8,0.3))
plt.tight_layout()     

#plt.savefig(fluid_name+"_shear_eta_vs_phi_"+str(org_var_1[org_var_1_fitting_start_index])+"_"+str(org_var_1[org_var_1_fitting_end_index-1])+"_run_number_"+str(run_number[0])+"_"+str(run_number[1])+"_"+str(run_number[2])+".pdf",dpi=500, bbox_inches='tight')

plt.savefig("plots/"+fluid_name+"shear_eta_vs_box_size_swap_rate_var.pdf",dpi=500, bbox_inches='tight' )
plt.show() 



#%% Reynolds number vs box size 
def Reynolds_number(shear_rate,shear_viscosity,box_size,rho_density):
    Re = (shear_rate).T*(box_size**2)*(rho_density/np.array([shear_viscosity]))
    return Re
rho_density=5
Re= Reynolds_number(np.exp(shear_rate_mean_of_both_cells[:,:,0]),shear_viscosity,box_side_length_scaled[0,:],rho_density)
x=box_side_length_scaled[0,:]
y=Re 

plt.rcParams.update({'font.size': 15})
linestyle=['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
labelpady=25
labelpady=15
width_plot=8
height_plot=6
#%y_error_bar=np.abs(np.exp(shear_rate_mean_error_of_both_cells[:,:,0]))
#for z in range(org_var_1_fitting_start_index,org_var_1_fitting_end_index):

os.getcwd()
plt.figure(figsize=(width_plot,height_plot))  

for z in range(0,8):
    #plt.scatter(x[:,i],y[:,i],label="$L=$"+str(box_side_length_scaled[z])+", grad$=$"+str(sigfig.round(grad_fit,sigfigs=2))+"$\pm$"+str(sigfig.round(grad_fit_abs_error,sigfigs=1)),marker='x')
    #plt.plot(x,y,"--",marker= 'x')#label="$N_{v,x}=$"+str(org_var_2[z])+", $\Delta\eta_{max}=$"+str(sigfig.round(shear_viscosity_abs_error_max[z],sigfigs=2)),marker='x')
    #plt.errorbar(x,y[:,z],yerr=y_error_bar[:,z],capsize=3,color='r', label="Simulation data",linestyle='dashed')
    plt.plot(x,y[z,:],color=colour[z], label="$f_{p}="+str(swap_rate[z])+"$",marker=marker[z],markersize=8,linestyle=linestyle[z])

    plt.xlabel('$L$', labelpad=labelpadx,fontsize=fontsize)
    #plt.plot(x, predicted_dimensionless_shear_viscosity[:], linestyle='solid', label="Tuzel,Ihle and Kroll, 2006")
    #plt.yscale('log')
    plt.ylabel('$\mathrm{Re}$',labelpad=labelpady, rotation=0,fontsize=fontsize)
    #plt.xscale('log')
    #plt.ylabel('$\eta \\frac{\ell^{3}}{\epsilon\\tau}$',rotation=0,labelpad=labelpady,fontsize=fontsize)
    plt.legend(loc='best',bbox_to_anchor=(1,1))
    #plt.tight_layout()     
    #plt.savefig(fluid_name+"_shear_eta_vs_phi_"+str(org_var_1[org_var_1_fitting_start_index])+"_"+str(org_var_1[org_var_1_fitting_end_index-1])+"_run_number_"+str(run_number[0])+"_"+str(run_number[1])+"_"+str(run_number[2])+".pdf",dpi=500, bbox_inches='tight')

plt.savefig("plots/"+fluid_name+"Re_vs_box_size_swap_rate_var_all.pdf",dpi=500, bbox_inches='tight' )  
#plt.savefig("plots/"+fluid_name+"Re_vs_box_size_swap_rate_var_selected.pdf",dpi=500, bbox_inches='tight' )
plt.show() 

#%%Mach number 
c_f=np.sqrt(5/3)
labelpady=15

def characteric_vel(Re,shear_viscosity,rho_density,box_side_length_scaled):
    u_characteristic= Re * shear_viscosity / (rho_density * box_side_length_scaled)
    
    return  u_characteristic

u_characteristic= characteric_vel(Re,shear_viscosity,rho_density,box_side_length_scaled)

Mach_number= u_characteristic/c_f
y=Mach_number

if np.all(Mach_number<0.1):
    print("Mach number low enough")
else:
    print("mach not low enough")
fontsize=25
labelpadx=5
labelpady=18
width_plot=8
height_plot=6
#for z in range(org_var_1_fitting_start_index,org_var_1_fitting_end_index):
os.getcwd()
plt.figure(figsize=(width_plot,height_plot)) 
for z in range(0,8):
    #plt.scatter(x[:,i],y[:,i],label="$L=$"+str(box_side_length_scaled[z])+", grad$=$"+str(sigfig.round(grad_fit,sigfigs=2))+"$\pm$"+str(sigfig.round(grad_fit_abs_error,sigfigs=1)),marker='x')
    #plt.plot(x,y,"--",marker= 'x')#label="$N_{v,x}=$"+str(org_var_2[z])+", $\Delta\eta_{max}=$"+str(sigfig.round(shear_viscosity_abs_error_max[z],sigfigs=2)),marker='x')
    #plt.errorbar(x,y[:,z],yerr=y_error_bar[:,z],capsize=3,color='r', label="Simulation data",linestyle='dashed')
    plt.plot(x,y[z,:],color=colour[z], label="$f_{p}="+str(swap_rate[z])+"$",marker=marker[z],markersize=8,linestyle=linestyle[z])

    plt.xlabel('$L$', labelpad=labelpadx,fontsize=fontsize)
  
    #plt.yscale('log')
    plt.ylabel('$\mathrm{Ma}$',labelpad=labelpady, rotation=0,fontsize=fontsize)
    #plt.xscale('log')
    #plt.ylabel('$\eta \\frac{\ell^{3}}{\epsilon\\tau}$',rotation=0,labelpad=labelpady,fontsize=fontsize)
    plt.legend(loc='best',bbox_to_anchor=(1,1))
    #plt.tight_layout()     
    #plt.savefig(fluid_name+"_shear_eta_vs_phi_"+str(org_var_1[org_var_1_fitting_start_index])+"_"+str(org_var_1[org_var_1_fitting_end_index-1])+"_run_number_"+str(run_number[0])+"_"+str(run_number[1])+"_"+str(run_number[2])+".pdf",dpi=500, bbox_inches='tight')

plt.savefig("plots/"+fluid_name+"Ma_vs_box_size_swap_rate_var_all.pdf",dpi=500, bbox_inches='tight' )  
#plt.savefig("plots/"+fluid_name+"Ma_vs_box_size_swap_rate_var_selected.pdf",dpi=500, bbox_inches='tight' )
plt.show()     

    

#%%Schmidt number 
D_f=0.15

Sc_after =np.array([shear_viscosity]) /( D_f * rho_density)
Sc_mean=np.repeat(np.mean(Sc_after),number_of_solutions)
fontsize=25
labelpady=20
labelpadx=10
width_plot=8
height_plot=6
plt.figure(figsize=(width_plot,height_plot)) 
plt.xlabel('$L$', labelpad=labelpadx,fontsize=fontsize)  
plt.ylabel('$\mathrm{Sc}$',labelpad=labelpady, rotation=0,fontsize=fontsize)
plt.scatter(box_side_length_scaled[0,:],Sc_after[0,:])
plt.plot(box_side_length_scaled[0,:],Sc_mean[:], linestyle='--', label="$\mathrm{Sc}_{mean}$")
plt.legend(loc='best')

plt.savefig("plots/"+fluid_name+"Sc_vs_box_size_swap_rate_var_selected.pdf",dpi=500, bbox_inches='tight' )
plt.show()


# %% saving all the arrays which are needed for plots
# need to save orginal untruncated VP / shear rate data 

# logged shear rate 
#np.save(fluid_name+'_logged_shear_rate_mean_both_cells_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),shear_rate_mean_of_both_cells)
np.save("shear_rate_mean_of_both_cells_"+name_of_run_for_save,shear_rate_mean_of_both_cells)

#  shear rate errors 
#np.save(fluid_name+'_logged_shear_rate_mean_error_both_cells_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),shear_rate_mean_error_of_both_cells)
np.save("shear_rate_mean_error_of_both_cells_"+name_of_run_for_save,shear_rate_mean_error_of_both_cells)

# fitting parameters and viscosity 

#np.save(fluid_name+'_fitting_params_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),params)
np.save("flux_fitting_params_"+name_of_run_for_save,params)

# flux ready for plotting 

#np.save(fluid_name+'_flux_ready_for_plotting_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),flux_ready_for_plotting)
np.save("flux_ready_for_plotting_"+name_of_run_for_save,flux_ready_for_plotting)

# steady state velocity profiles 
# v_x data upper/ lower 
#np.save(fluid_name+'_VP_steady_state_data_lower_truncated_time_averaged_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_steady_state_data_lower_truncated_time_averaged)
np.save("VP_steady_state_data_lower_truncated_time_averaged_"+name_of_run_for_save,VP_steady_state_data_lower_truncated_time_averaged)

#np.save(fluid_name+'_VP_steady_state_data_upper_truncated_time_averaged_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_steady_state_data_upper_truncated_time_averaged)
np.save("VP_steady_state_data_upper_truncated_time_averaged_"+name_of_run_for_save,VP_steady_state_data_upper_truncated_time_averaged)

# z coordinate upper/lower
#np.save(fluid_name+'_VP_z_data_lower_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_z_data_lower)
np.save("VP_z_data_lower_"+name_of_run_for_save,VP_z_data_lower)

#np.save(fluid_name+'_VP_z_data_upper_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_z_data_upper)
np.save("VP_z_data_upper_"+name_of_run_for_save,VP_z_data_upper)



# %%
