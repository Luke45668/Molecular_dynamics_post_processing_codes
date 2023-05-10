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
no_timesteps=500000
swap_rate = np.array([3,5,7,9,12,15,22,30,45,60,180,300,450,600,750,900,1050,1200]) # values chosen from original mp paper
swap_number = np.array([1,10,100,1000])
equilibration_timesteps= 1000 # number of steps to do equilibration with 
VP_ave_freq =1000
chunk = 20

dump_freq=1000 # if you change the timestep rememebr to chaneg this 
thermo_freq = 10000
scaled_temp=1
scaled_timestep=0.01
realisation=np.array([0.0,1.0,2.0])
VP_output_col_count = 4 
r_particle =50e-6
phi=0.005
N=2
Vol_box_at_specified_phi=(N* (4/3)*np.pi*r_particle**3 )/phi
box_side_length=np.cbrt(Vol_box_at_specified_phi)
fluid_name='Ar'

#%% grabbing file names 
#vel.Ar_579862_mom_output__no_rescale_851961_2.0_61516_11877.258268303078_0.01_197_1000_10000_500000_T_1.0_lbda_1.3517157706256893_SR_450_SN_10
VP_general_name_string='vel.'+fluid_name+'_mom_output_*_no_rescale_*'
#vel.Nitrogen_mom_output_no221643_no_rescale_6913_1.0_139775_7725.899083231229_0.01_6_1000_10000_500000_T_1.0_lbda_0.8097165710872696_SR_5_SN_10
#VP_general_name_string='vel.profile_mom_output__no_rescale_*'
#mom.Nitrogen_no_tstat_no810908_no_rescale_400902_0.0_48384_3755.9188485904992
Mom_general_name_string='mom.'+fluid_name+'_no_tstat_*_no_rescale_*'
#mom.Nitrogen_no_tstat__no_rescale_
#Mom_general_name_string='mom.'+fluid_name+'_*_tstat__no_rescale_*'

#filepath='N_phi_0.005_0.00005_data_T_1'
filepath='Ar_phi_0.005_0.00005_data_T_1'


realisation_name_info=VP_and_momentum_data_realisation_name_grabber(j_,swap_number,swap_rate,VP_general_name_string,Mom_general_name_string,filepath)
realisation_name_Mom=realisation_name_info[0]
realisation_name_VP=realisation_name_info[1]
count_mom=realisation_name_info[2]
count_VP=realisation_name_info[3]
number_of_solutions=realisation_name_info[4]

##LEARN HOW TO USE DEBUGGER

#%%

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

box_size_loc=10
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
#%%
VP_data_lower_realisation_averaged = np.mean(VP_data_lower,axis=5) 
VP_data_upper_realisation_averaged = np.mean(VP_data_upper,axis=5) 
x_u= np.array(VP_data_upper_realisation_averaged)
x_l= np.array(VP_data_lower_realisation_averaged)
y_u=np.zeros((number_of_solutions,VP_z_data_upper.shape[1],VP_data_upper.shape[4]))
y_l=np.zeros((number_of_solutions,VP_z_data_upper.shape[1],VP_data_upper.shape[4]))

for z in range(number_of_solutions):
    y_u[z,:,:]= np.reshape(np.repeat(VP_z_data_upper[z,:],VP_data_upper.shape[4],axis=0),(VP_z_data_upper.shape[1],VP_data_upper.shape[4]))

    y_l[z,:,:]= np.reshape(np.repeat(VP_z_data_lower[z,:],VP_data_lower.shape[4],axis=0),(VP_z_data_lower.shape[1],VP_data_lower.shape[4]))

pearson_coeff_upper= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_upper_realisation_averaged.shape[4]))
pearson_coeff_lower= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_lower_realisation_averaged.shape[4]))
shear_rate_upper= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_upper_realisation_averaged.shape[4]))    
shear_rate_lower= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_lower_realisation_averaged.shape[4]))
timestep_points=np.array([[[np.linspace(1,VP_data_upper.shape[4],int(float(no_timesteps)/float(VP_ave_freq)))]]])*VP_ave_freq
timestep_points=np.repeat(timestep_points, number_of_solutions,axis=0)
timestep_points=np.repeat(timestep_points, swap_rate.size,axis=1)     
timestep_points=np.repeat(timestep_points, swap_number.size,axis=2)   



for z in range(0,number_of_solutions): 
    for m in range(0,swap_rate.size):
        for k in range(0,swap_number.size):
            for i in range(0,VP_data_upper.shape[4]):
                    
                    pearson_coeff_upper[z,m,k,i] =scipy.stats.pearsonr(y_u[z,:,i],x_u[z,m,k,:,i] )[0]
                    shear_rate_upper[z,m,k,i]= scipy.stats.linregress(y_u[z,:,i],x_u[z,m,k,:,i] ).slope
                    pearson_coeff_lower[z,m,k,i] =scipy.stats.pearsonr(y_l[z,:,i],x_l[z,m,k,:,i] )[0]
                    shear_rate_lower[z,m,k,i]= scipy.stats.linregress(y_l[z,:,i],x_l[z,m,k,:,i] ).slope  
    
# %%
box_size_nd= box_side_length_scaled
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

plot_shear_rate_to_asses_SS(no_timesteps,phi,lengthscale,timestep_points,scaled_temp,number_of_solutions,swap_rate,swap_number,shear_rate_upper,shear_rate_lower,fluid_name,box_size_nd)
# this needs to go into a new plot function with name save in could have a plot one to check 
# then have a one that just plots and saves
#plt.savefig(fluid_name+'_simulation_full_shear_rate_data_untruncated_T_'+str(scaled_temp)+'_box_size_'+str(box_side_length_scaled)+'.png')
truncation_timestep=100000
# need to save this plot 
# %%
truncation_and_SS_averaging_data= truncation_step_and_SS_average_of_VP_and_stat_tests(timestep_points,pearson_coeff_lower,pearson_coeff_upper,shear_rate_upper,shear_rate_lower,VP_ave_freq,truncation_timestep,VP_data_lower_realisation_averaged,VP_data_upper_realisation_averaged)
standard_deviation_upper_error=truncation_and_SS_averaging_data[0]
standard_deviation_lower_error=truncation_and_SS_averaging_data[1]
pearson_coeff_upper_mean_SS=truncation_and_SS_averaging_data[2]
pearson_coeff_lower_mean_SS=truncation_and_SS_averaging_data[3]
shear_rate_lower_steady_state_mean=truncation_and_SS_averaging_data[4]
shear_rate_upper_steady_state_mean=truncation_and_SS_averaging_data[5]
VP_steady_state_data_lower_truncated_time_averaged=truncation_and_SS_averaging_data[6]
VP_steady_state_data_upper_truncated_time_averaged=truncation_and_SS_averaging_data[7]

# np.save(fluid_name+'_standard_deviation_upper_error_'+str(box_side_length_scaled)+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),standard_deviation_upper_error)    
# np.save(fluid_name+'_standard_deviation_lower_error_'+str(box_side_length_scaled)+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),standard_deviation_lower_error)    
# np.save(fluid_name+'_pearson_coeff_upper_mean_SS_'+str(box_side_length_scaled)+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),pearson_coeff_upper_mean_SS)    
# np.save(fluid_name+'_pearson_coeff_lower_mean_SS_'+str(box_side_length_scaled)+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),pearson_coeff_lower_mean_SS)    
# np.save(fluid_name+'_shear_rate_lower_steady_state_mean_'+str(box_side_length_scaled)+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),shear_rate_lower_steady_state_mean)          
# np.save(fluid_name+'VP_steady_state_data_lower_truncated_time_averaged'+str(box_side_length_scaled)+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_steady_state_data_lower_truncated_time_averaged)          
# np.save(fluid_name+'VP_steady_state_data_upper_truncated_time_averaged'+str(box_side_length_scaled)+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_steady_state_data_upper_truncated_time_averaged)          
#%% mom_data_averaging_and_flux_calc
flux_data= mom_data_averaging_and_flux_calc(number_of_solutions,swap_number,truncation_timestep,swap_rate,scaled_timestep,no_timesteps,box_side_length_scaled,mom_data)
flux_x_momentum_z_direction=flux_data[0]
flux_ready_for_plotting=flux_data[1]
#%%
mom_data_realisation_averaged=()
number_swaps_before_truncation=(np.ceil(truncation_timestep/swap_rate)).astype(int)
mom_data_realisation_averaged_truncated=()
flux_x_momentum_z_direction=np.zeros((number_of_solutions,swap_number.size,swap_rate.size))
total_run_time=scaled_timestep* no_timesteps
box_area_nd=np.array(box_size_key)
flux_ready_for_plotting=np.zeros((number_of_solutions,swap_number.size,swap_rate.size))
for z in range(0,number_of_solutions):    
    for i in range(0,swap_rate.size):
        mom_data_realisation_averaged=mom_data_realisation_averaged+(np.mean(mom_data[i],axis=2),)


    #for i in range(0,swap_rate.size):

        mom_data_realisation_averaged_truncated=mom_data_realisation_averaged_truncated+(mom_data_realisation_averaged[i][:,:,number_swaps_before_truncation[i]:],)


    # now apply the MP formula 
        mom_difference= mom_data_realisation_averaged_truncated[i][z,:,-1]-mom_data_realisation_averaged_truncated[i][z,:,0]
        flux_x_momentum_z_direction[z,:,i]=(mom_difference)/(2*total_run_time*float(box_area_nd[z]))
        
flux_ready_for_plotting=np.log((np.abs(flux_x_momentum_z_direction)))

#%% flux vs shear regression line 
shear_rate_mean_of_both_cells=np.log((((np.abs(shear_rate_lower_steady_state_mean)+np.abs(shear_rate_upper_steady_state_mean))*0.5)))
flux_vs_shear_regression_line_params=()
x=shear_rate_mean_of_both_cells
yfit1=()
def func4(x, a, b, c):
   #return np.log(a) + np.log(b*x)
   #return (a*(x**b))
   return (a*x) +b 
   #return a*np.log(b*x)+c

for z in range(0,number_of_solutions):    
    for i in range(0,swap_number.size):
      
        flux_vs_shear_regression_line_params= flux_vs_shear_regression_line_params+(scipy.optimize.curve_fit(func4,shear_rate_mean_of_both_cells[z,:,i],flux_ready_for_plotting[z,i,:],method='lm',maxfev=5000),)

params=flux_vs_shear_regression_line_params 

#%%
plotting_flux_vs_shear_rate(box_side_length_scaled,number_of_solutions,shear_rate_lower_steady_state_mean,shear_rate_upper_steady_state_mean,flux_ready_for_plotting,swap_number)
#%%
phi=[0.005,0.0005]
#%% 
save_string_for_plot= 'Flux_vs_shear_rate_'+fluid_name+'_phi_range_'+str(phi[0])+'_'+str(phi[1])+'_l_scale_'+str(lengthscale)+'_T_'+str(scaled_temp)+'.png'
labelpadx=15
labelpady=40
fontsize=15
count=1
shear_rate_mean_of_both_cells=np.log((np.abs(shear_rate_lower_steady_state_mean)+np.abs(shear_rate_upper_steady_state_mean))*0.5)
for z in range(0,number_of_solutions):
        
        #shear_rate_mean_of_both_cells=((np.abs(shear_rate_lower_steady_state_mean)+np.abs(shear_rate_upper_steady_state_mean))*0.5)

        x=shear_rate_mean_of_both_cells[z,:,:]
        y=flux_ready_for_plotting[z,:,:]
        #for i in range(0,swap_number.size):
        
        for i in range(0,1):
            if z==0:
                j=i
                
                # need to add legend to this 
                plt.scatter(x[:,i],y[i,:],label='$L=${}'.format(np.around(box_side_length_scaled[0,z]),decimals=0))
                plt.plot(x[:,i],func4(x[:,i],params[j][0][0],params[j][0][1],params[j][0][2]))
                #plt.xscale('log')
                plt.xlabel('log($\dot{\gamma}\ [\\tau]$)', labelpad=labelpadx,fontsize=fontsize)
                #plt.yscale('log')
                plt.ylabel('log($J_{z}(p_{x})$$\ [\\frac{\\tau^{3}}{\mu}]$)',rotation=0,labelpad=labelpady,fontsize=fontsize)
                plt.legend()
            else: 
                j=z*(i+4)
                plt.scatter(x[:,i],y[i,:],label='$L=${}'.format(np.around(box_side_length_scaled[0,z]),decimals=0))
                plt.plot(x[:,i],func4(x[:,i],params[j][0][0],params[j][0][1],params[j][0][2]))
                #plt.xscale('log')
                plt.xlabel('log($\dot{\gamma}\ [\\tau]$)', labelpad=labelpadx,fontsize=fontsize)
                #plt.yscale('log')
                plt.ylabel('log($J_{z}(p_{x})$$\ [\\frac{\\tau^{3}}{\mu}]$)',rotation=0,labelpad=labelpady,fontsize=fontsize)
                plt.legend()
                 
plt.show() 
#plt.savefig(save_string_for_plot)        
       



#%% plotting qll 4 SS V_Ps


width_plot=10
height_plot=5
# need to fix legend location 
for i in range(0,swap_number.size):
    swap_number_choice_index=i
    plotting_SS_velocity_profiles_for_4_swap_numbers(number_of_solutions,swap_number_choice_index,width_plot,height_plot,swap_number,swap_rate,VP_ave_freq,no_timesteps,VP_steady_state_data_lower_truncated_time_averaged,VP_steady_state_data_upper_truncated_time_averaged,VP_z_data_lower,VP_z_data_upper)
#%%
fontsize=22
labelpadx=15
labelpady=35
width_plot=8
height_plot=5
for z in range(0,number_of_solutions):
        fig=plt.figure(figsize=(width_plot,height_plot))
        gs=GridSpec(nrows=1,ncols=1)

        ax1= fig.add_subplot(gs[0,0])
        #ax2= fig.add_subplot(gs[1,0])
        k=swap_number_choice_index



        x_1=VP_steady_state_data_lower_truncated_time_averaged[z,:,:]
        x_2=VP_steady_state_data_upper_truncated_time_averaged[z,:,:]
        y_1=VP_z_data_lower[z,:,]
        y_2=VP_z_data_upper[z,:]

        no_vps=int(no_timesteps/VP_ave_freq)
        #for k in range(0,swap_number.size):
        #for i in range(0,swap_rate.size):
        for i in range(0,3):
        

            ax1.plot(y_1[:],x_1[i,k,:],label='$f_p=${}'.format(swap_rate[i]),marker='x')
            ax1.set_ylabel('$v_{x}\ [\\frac{\\tau}{\ell}]$',rotation=0,labelpad=labelpady, fontsize=fontsize)
            ax1.set_xlabel('$L_{z}\ [\ell^{-1}]$',rotation=0,labelpad=labelpadx,fontsize=fontsize)
            #ax2.plot(x_2[i,k,:],y_2[:],label='$f_p=${}'.format(swap_rate[i]))
            ax1.legend(frameon=False,loc='right',fontsize=fontsize-4)
            #ax2.set_xlabel('$v_{x}\ [\\frac{\\tau}{\ell}]$',fontsize=fontsize)
            #ax2.set_ylabel('$L_{z}\ [\ell^{-1}]$',rotation=0,labelpad=labelpad,fontsize=fontsize)
            #ax2.legend(frameon=False,loc='right')
            
        plt.show()

save_string_for_plot= 'Velocity_Profiles_'+fluid_name+'_phi_range_'+str(phi[0])+'_'+str(phi[1])+'_l_scale_'+str(lengthscale)+'_T_'+str(scaled_temp)+'.png'




  

# %%
