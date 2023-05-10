#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:27:44 2023
This script will read in lammps log or velocity profiles then produce all the required plots


@author: lukedebono
"""
#%%
import os
from pyexpat.model import XML_CQUANT_PLUS
from sys import exception

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

#%% definitions 
#os.chdir('/Volumes/Backup Plus/PhD_/Rouse Model simulations/Using LAMMPS imac/LAMMPS python run and analysis scripts/Analysis codes/key arrays for post processing ')
j_=3
no_timesteps=500000
#swap_rate = np.array([5,9,12,22,45,180,450,750,1050])
swap_rate = np.array([3,5,7,9,12,15,22,30,45,60,180,300,450,600,750,900,1050,1200]) # values chosen from original mp paper
swap_number = np.array([1,10,100,1000])
#locations_of_non_nan_neg=np.load('locations_of_non_nan_neg_16_Ar.npy')
#number_SRD_particles_wrt_pf_cp_mthd_1=np.load('number_SRD_particles_Box_16_Ar.npy')
equilibration_timesteps= 1000 # number of steps to do equilibration with 
VP_ave_freq =1000
chunk = 20
box_side_length_scaled=3755.9188485904992
dump_freq=1000 # if you change the timestep rememebr to chaneg this 
thermo_freq = 10000
scaled_temp=1

# VP_z_data_upper=np.zeros((9,1))
# VP_z_data_lower=np.zeros((9,1))
realisation=np.array([0.0,1.0,2.0])
fluid_name='Nitrogen'
os.chdir('/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/LAMMPS python run and analysis scripts/Analysis codes')
from velP2numpy import *
from mom2numpy import *



#%% getting names of files
filepath='pure_fluid_validations_batch_scripts_500k_tstps_phi_0.005'
os.chdir('/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/'+filepath)
import glob 
VP_output_col_count = 4 

count_VP=0
realisation_name_VP = []     

#for name in glob.glob('/Volumes/Backup Plus/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/VP_data using pearson coeff/vel.profile_no_tstat__no_rescale_*'):
# make sure you are in the correct location before you read in the data
for name in glob.glob('vel.profile_mom_output__no_rescale_*'):
   # vel.profile_mom_output_
    count_VP=count_VP+1    
    realisation_name_VP.append(name)
    

#%%
count_mom=0
realisation_name_Mom=[]
for name in glob.glob('mom.'+fluid_name+'_no_tstat__no_rescale_*'):
   
    count_mom=count_mom+1    
    realisation_name_Mom.append(name)
    
    
    

no_mom_data_points=(np.ceil((no_timesteps/swap_rate))).astype(int)
number_of_solutions= int(count_VP/(j_*swap_number.size*swap_rate.size))
# BECAREFUL NOT TO RUN THIS BOX AGAIN AS THE LOOP IN NEXT BOX TAKES FUCKING AGES
VP_data_upper=np.zeros((number_of_solutions,swap_rate.size,swap_number.size,9,int(no_timesteps/VP_ave_freq),j_))
VP_data_lower=np.zeros((number_of_solutions,swap_rate.size,swap_number.size,9,int(no_timesteps/VP_ave_freq),j_))
mom_data=()
for i in range(0,no_mom_data_points.size):
    
      mom_data= mom_data+(np.zeros((number_of_solutions,swap_number.size,j_,(no_mom_data_points[i]))),)
# for i in range(0,swap_rate.size):
#     for j in range(0,swap_number.size):
#          mom_data= mom_data+((swap_rate[i],swap_number[j]))
#          mom_data= mom_data+'_'+str(swap_rate[i])+'_'+str(swap_number[j])

#VP_data_upper=() 
#VP_data_lower=()


#%% 
# finding SRD counts and removing duplicates
no_SRD=[]
for i in range(0,count_VP):
    no_srd=realisation_name_VP[i].split('_')
    no_SRD.append(no_srd[8])
    
no_SRD.sort(key=int)
no_SRD_key=[]
#using list comprehension to remove duplicates
[no_SRD_key.append(x) for x in no_SRD if x not in no_SRD_key]


simulation_file="MYRIAD_LAMMPS_runs/"+filepath

Path_2_mom_file="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file
#%%finding and organising velocity profiles 
#mom.Nitrogen_no_tstat__no_rescale_104054_0.0_48384_3755.9188485904992_0.01_8_1000_10000_500000_T_1.0_lbda_0.5556691060097062_SR_180_SN_1000
#vel.profile_mom_output__no_rescale_104054_0.0_48384_3755.9188485904992_0.01_8_1000_10000_500000_T_1.0_lbda_0.5556691060097062_SR_180_SN_1000
simulation_file="MYRIAD_LAMMPS_runs/"+filepath
Path_2_VP="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file

marker=-1
error_count=0 
for i in range(0,count_VP):
    filename=realisation_name_VP[i].split('_')
    marker=marker+1
    no_SRD=filename[8]
    z=no_SRD_key.index(no_SRD)
    realisation_index=filename[7]
    j=int(float(realisation_index))
    EF=int(filename[20])
    m=np.where(swap_rate==EF)
    SN=int(filename[22])
    k=np.where(swap_number==SN)
    realisation_name=realisation_name_VP[i]
    try: 
        VP_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[0]
        VP_data_upper[z,m,k,:,:,j] = VP_data[1:10,:]
        VP_data_lower[z,m,k,:,:,j] = VP_data[11:,:]
        
    except Exception as e:
      print('Velocity Profile Data faulty')
      error_count=error_count+1 
      continue
#%% finding and organising the mom files 
simulation_file="MYRIAD_LAMMPS_runs/"+filepath

Path_2_mom_file="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file
error_count_mom=0
pass_count=0
size_list=[]
failed_list_realisations=[]
# this for loop makes the size list for the mom_data tuple 
for i in range(0,count_mom):
    filename=realisation_name_Mom[i].split('_')
   
    no_SRD=filename[8]
    z=no_SRD_key.index(no_SRD)
   
    realisation_index=filename[7]
    j=int(float(realisation_index))
    EF=int(filename[20])
    m=np.where(swap_rate==EF)[0][0,]
    
    SN=int(filename[22])
    k=np.where(swap_number==SN)[0][0,]
    
    realisation_name=realisation_name_Mom[i]
    
    #try: 
    #mom_data[m][z,k,j,:]=mom2numpy_f(realisation_name,Path_2_mom_file)  
    
    mom_data_test=mom2numpy_f(realisation_name,Path_2_mom_file)  
    size_list.append(mom_data_test.shape)
    
    pass_count=pass_count+1 

       
        
    # except Exception as e:
    #    print('Mom Data faulty')
    #    error_count_mom=error_count_mom+1 
    #    failed_list_realisations.append(realisation_name)
    #    break
#%%
size_list.sort()
size_list.reverse()
size_list=list(dict.fromkeys(size_list))
size_array=np.array(size_list)


mom_data=()
for i in range(0,no_mom_data_points.size):
    
      mom_data= mom_data+(np.zeros((number_of_solutions,swap_number.size,j_,(size_array[i,0]))),)
# for i in range(0,swap_rate.size):
#%%
error_count_mom=0
failed_list_realisations=[]
for i in range(0,count_mom):
    filename=realisation_name_Mom[i].split('_')
   
    no_SRD=filename[8]
    z=no_SRD_key.index(no_SRD)
   
    realisation_index=filename[7]
    j=int(float(realisation_index))
    EF=int(filename[20])
    m=np.where(swap_rate==EF)[0][0,]
    
    SN=int(filename[22])
    k=np.where(swap_number==SN)[0][0,]
    
    realisation_name=realisation_name_Mom[i]
    
    try:
       mom_data[m][z,k,j,:]=mom2numpy_f(realisation_name,Path_2_mom_file)  
    
    

       
        
    except Exception as e:
       print('Mom Data faulty')
       error_count_mom=error_count_mom+1 
       failed_list_realisations.append(realisation_name)
       break
#%% averaging the momentum data
mom_data_realisation_averaged=()
truncation_timestep=100000
number_swaps_before_truncation=(np.ceil(truncation_timestep/swap_rate)).astype(int)
mom_data_realisation_averaged_truncated=()
flux_x_momentum_z_direction=()
scaled_timestep=0.01
total_run_time=scaled_timestep* no_timesteps
flux_ready_for_plotting=np.zeros((swap_number.size,swap_rate.size))


box_area_nd= box_side_length_scaled**2

for i in range(0,swap_rate.size):
       mom_data_realisation_averaged=mom_data_realisation_averaged+(np.mean(mom_data[i],axis=2),)
    

#for i in range(0,swap_rate.size):
    
       mom_data_realisation_averaged_truncated=mom_data_realisation_averaged_truncated+(mom_data_realisation_averaged[i][:,:,number_swaps_before_truncation[i]:],)
    
    
# now apply the MP formula 
       flux_x_momentum_z_direction=flux_x_momentum_z_direction+((np.sum(mom_data_realisation_averaged_truncated[i],axis=2,keepdims=True)/(2*total_run_time*box_area_nd),),)

#for i in range(0,swap_rate.size):  #
   # for j in range(0,swap_number.size): 
    
       flux_ready_for_plotting[:,i]=np.abs(flux_x_momentum_z_direction[i][0][0,:,:].flatten())
       


        

#%%

#%% testing reader alone 
# so the reader works on failed files in isolation, its something wrong with the rest of the loop
realisation_name='mom.Nitrogen_no_tstat__no_rescale_856735_1.0_48384_3755.9188485904992_0.01_8_1000_10000_500000_T_1.0_lbda_0.5556691060097062_SR_9_SN_1'
swap_rate=9
mom_data= mom2numpy_f(realisation_name,Path_2_mom_file)

#%% saving the array so we dont have to run the very slow loop again    
np.save(fluid_name+'_VP_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_data_upper)    
np.save(fluid_name+'_VP_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_data_lower)   

#%%
VP_z_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[1]      
VP_z_data_upper = VP_z_data[1:10].astype('float64')* box_side_length_scaled         
VP_z_data_lower = VP_z_data[11:].astype('float64') * box_side_length_scaled
np.save(fluid_name+'_VP_z_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_z_data_upper)    
np.save(fluid_name+'_VP_z_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_z_data_lower)          
 


 
 
#%%
VP_data_upper=np.load(fluid_name+'_VP_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq)+'.npy')
VP_data_lower=np.load(fluid_name+'_VP_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq)+'.npy') 
VP_data_lower_realisation_averaged = np.mean(VP_data_lower,axis=5) 
# 4 indicates to take the mean of the 
VP_data_upper_realisation_averaged = np.mean(VP_data_upper,axis=5) 

 #%% plot some VP time series data 
 #could do with turning this into a function
fig=plt.figure(figsize=(13,6))
gs=GridSpec(nrows=2,ncols=1)
#fig.suptitle('",$\Delta t_{MD}^{*}$: "+str(scaled_timestep)+", M:"+str(Solvent_bead_SRD_box_density_cp_1[0,0])+", $\Delta x^{*}$"+str(SRD_box_size_wrt_solid_beads[0,3]) ,size='large',ha='center')
ax1= fig.add_subplot(gs[1,0])
ax2= fig.add_subplot(gs[0,0])
z=0
m=0
k=0

x_1=VP_data_upper_realisation_averaged
x_2=VP_data_lower_realisation_averaged
y_1=VP_z_data_upper
y_2=VP_z_data_lower

no_vps=int(no_timesteps/VP_ave_freq)

for i in range(0,no_vps):
    ax1.plot(x_1[z,m,k,:,i],y_1[:])
    ax2.plot(x_2[z,m,k,:,i],y_2[:])
    # ax1.plot(y_1[:],x_1[z,m,k,:,i])
    # ax2.plot(y_2[:],x_2[z,m,k,:,i])
    
plt.show()


#%%
x_u= np.array(VP_data_upper_realisation_averaged)
y_u= np.repeat(np.array([VP_z_data_upper]).T,VP_data_upper.shape[4],1)
x_l= np.array(VP_data_lower_realisation_averaged)
y_l= np.repeat(np.array([VP_z_data_lower]).T,VP_data_lower.shape[4],1)
pearson_coeff_upper= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_upper_realisation_averaged.shape[4]))
pearson_coeff_lower= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_lower_realisation_averaged.shape[4]))
shear_rate_upper= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_upper_realisation_averaged.shape[4]))
shear_rate_grad_upper=np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_upper_realisation_averaged.shape[4]))
shear_rate_lower= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_lower_realisation_averaged.shape[4]))
shear_rate_grad_lower=np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_lower_realisation_averaged.shape[4]))
#pearson_mean_upper=np.zeros((number_of_solutions,swap_rate.size))    
#standard_deviation_upper=np.zeros((number_of_solutions,swap_rate.size))                  
perfect_linearity_comparison_upper =np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_upper_realisation_averaged.shape[4]))
perfect_linearity_comparison_lower =np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_lower_realisation_averaged.shape[4]))
t_s=np.array([[[np.linspace(1,VP_data_upper.shape[4],int(float(no_timesteps)/float(VP_ave_freq)))]]])*VP_ave_freq
t_s=np.repeat(t_s, number_of_solutions,axis=0)
t_s=np.repeat(t_s, swap_rate.size,axis=1)     
t_s=np.repeat(t_s, swap_number.size,axis=2)     



# swap_rate_for_plotting_ = np.array([[swap_rate]])    
# swap_rate_for_plotting_=np.repeat(swap_rate_for_plotting_,number_of_solutions,axis=0)
# swap_rate_for_plotting_=np.repeat(swap_rate_for_plotting_,int(float(no_timesteps)/float(VP_ave_freq)),axis=1)

#%%  
                 
for z in range(0,number_of_solutions): 
    for m in range(0,swap_rate.size):
        for k in range(0,swap_number.size):
            for i in range(0,VP_data_upper.shape[4]):
                    
                    pearson_coeff_upper[z,m,k,i] =scipy.stats.pearsonr(y_u[:,i],x_u[z,m,k,:,i] )[0]
                    #pearson_coeff_upper[z,m,k,:]= np.corrcoef(x[z,m,k,:,:].T, VP_z_data_upper,rowvar=False)
                    shear_rate_upper[z,m,k,i]= scipy.stats.linregress(y_u[:,i],x_u[z,m,k,:,i] ).slope
                    pearson_coeff_lower[z,m,k,i] =scipy.stats.pearsonr(y_l[:,i],x_l[z,m,k,:,i] )[0]
                    shear_rate_lower[z,m,k,i]= scipy.stats.linregress(y_l[:,i],x_l[z,m,k,:,i] ).slope
#%%                    


np.save(fluid_name+'_shear_rate_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),shear_rate_upper)    
np.save(fluid_name+'_shear_rate_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),shear_rate_lower)    
np.save(fluid_name+'_pearson_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq), pearson_coeff_upper)
np.save(fluid_name+'_pearson_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq), pearson_coeff_lower)
#%%
shear_rate_upper=np.load(fluid_name+'_shear_rate_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq)+'.npy')   
shear_rate_lower=np.load(fluid_name+'_shear_rate_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq)+'.npy')    
pearson_coeff_upper=np.load(fluid_name+'_pearson_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq)+'.npy')
pearson_coeff_lower=np.load(fluid_name+'_pearson_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq)+'.npy')
#%%

# need to add box size 
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
# get rid of start up effects via inspection 
for z in range(0,number_of_solutions): 
    for m in range(0,swap_rate.size):
        for k in range(0,swap_number.size):
            plt.plot(t_s[0,1,1,:],shear_rate_upper[z,m,k,:])
            plt.plot(t_s[0,1,1,:],shear_rate_lower[z,m,k,:])
            plt.xlabel('$N_{t}[-]$')
            plt.ylabel('$\dot{\gamma}[-]$',rotation='horizontal')
            plt.title(fluid_name+"bsimulation run with all $f_{v,x}$ and $N_{v,x}$, $\\bar{T}="+str(scaled_temp)+"$")
            
            
    plt.show()
#%%            
##%% truncate data 
truncation_timestep=100000#np.array([200000,300000,400000])
truncation_index=int(truncation_timestep/VP_ave_freq)
shear_rate_upper=shear_rate_upper[:,:,:,truncation_index:]
#shear_rate_upper_mean = np.mean(shear_rate_upper, axis=3)
shear_rate_lower=shear_rate_lower[:,:,:,truncation_index:]
#shear_rate_lower_mean = np.mean(shear_rate_lower, axis=3)
pearson_coeff_upper=pearson_coeff_upper[:,:,:,truncation_index:]
pearson_coeff_lower=pearson_coeff_lower[:,:,:,truncation_index:]
t_s=t_s[:,:,:,truncation_index:]

VP_steady_state_data_lower_truncated=VP_data_lower_realisation_averaged[:,:,:,:,truncation_index:]
VP_steady_state_data_upper_truncated=VP_data_upper_realisation_averaged[:,:,:,:,truncation_index:]
#%%
VP_steady_state_data_lower_truncated_time_averaged=np.mean(VP_steady_state_data_lower_truncated,axis=4)
VP_steady_state_data_upper_truncated_time_averaged=np.mean(VP_steady_state_data_upper_truncated,axis=4)
#%%
shear_rate_upper_mean = np.mean(shear_rate_upper, axis=3)
shear_rate_lower_mean = np.mean(shear_rate_lower, axis=3)

pearson_coeff_lower_mean_SS=np.mean(pearson_coeff_lower,axis=3)
pearson_coeff_upper_mean_SS=np.mean(pearson_coeff_upper,axis=3)



#%%

fig=plt.figure(figsize=(20,6))
gs=GridSpec(nrows=2,ncols=1)
#fig.suptitle('",$\Delta t_{MD}^{*}$: "+str(scaled_timestep)+", M:"+str(Solvent_bead_SRD_box_density_cp_1[0,0])+", $\Delta x^{*}$"+str(SRD_box_size_wrt_solid_beads[0,3]) ,size='large',ha='center')
ax1= fig.add_subplot(gs[0,0],projection='3d')
ax2= fig.add_subplot(gs[1,0],projection='3d')
z=0
m=0
k=0

x_1=VP_steady_state_data_lower_time_averaged_truncated
x_2=VP_steady_state_data_upper_time_averaged_truncated
y_1=VP_z_data_lower
y_2=VP_z_data_upper

no_vps=int(no_timesteps/VP_ave_freq)
for k in range(0,swap_number.size):
    for i in range(0,swap_rate.size):
   
        ax1.plot(x_1[0,i,k,:],y_1[:],swap_number[k])
        ax2.plot(x_2[0,i,k,:],y_2[:],swap_number[k])
    # ax1.plot(y_1[:],x_1[z,m,k,:,i])
    # ax2.plot(y_2[:],x_2[z,m,k,:,i])
    
plt.show()
#%% Then take mean pearson coefficient and standard deviation of mean pearson 
pearson_mean_upper= np.mean(pearson_coeff_upper,axis=3)
pearson_mean_lower= np.mean(pearson_coeff_lower,axis=3)
standard_deviation_upper=np.std(pearson_coeff_upper,axis=3)
standard_deviation_lower=np.std(pearson_coeff_lower,axis=3)

standard_deviation_upper_error=standard_deviation_upper/pearson_mean_upper
standard_deviation_lower_error=standard_deviation_lower/pearson_mean_lower
#%% 



# as long as the values pass the stat test 




#%%                  
                   
perfect_linearity_comparison_lower=  1-pearson_mean_upper
perfect_linearity_comparison_upper=  1-pearson_mean_lower







# we will accept 0.7 as the lower bound  ref https://link.springer.com/article/10.1057/jt.2009.5                  
linearity_tol=0.25        
          
pearson_mean_upper=np.where(abs(perfect_linearity_comparison_upper)>linearity_tol, np.nan, pearson_mean_upper)
#shear_rate_upper= np.where(abs(perfect_linearity_comparison_upper)>linearity_tol, np.nan, shear_rate_upper)
pearson_mean_lower=np.where(abs(perfect_linearity_comparison_lower)>linearity_tol, np.nan, pearson_mean_lower)
#shear_rate_lower= np.where(abs(perfect_linearity_comparison_lower)>linearity_tol, np.nan,shear_rate_lower )
#%%
# now applying Std deviation tolerance 

std_dev_tol=0.05
pearson_mean_upper=np.where(standard_deviation_upper_error>std_dev_tol, np.nan, pearson_mean_upper)
pearson_mean_lower=np.where(standard_deviation_lower_error>std_dev_tol, np.nan, pearson_mean_lower)

#%% 







#now we need to go through and make a matrix of all the data that passed the stat tests or do we could just keep them in there and use as ref
passed_stat_test_upper=np.count_nonzero(~np.isnan(pearson_mean_upper)) 
print(passed_stat_test_upper)
passed_stat_test_lower= np.count_nonzero(~np.isnan(pearson_mean_lower)) 
print(passed_stat_test_lower)
locations_of_passed_stat_test_upper= np.argwhere(~np.isnan(pearson_mean_upper))
locations_of_passed_stat_test_lower= np.argwhere(~np.isnan(pearson_mean_lower))
shear_rate_upper_mean=np.zeros((locations_of_passed_stat_test_upper.shape[0]))
shear_rate_lower_mean=np.zeros((locations_of_passed_stat_test_lower.shape[0]))

#%%
#now go through and take mean shear rate of those which pass the test 
for z in range(0,locations_of_passed_stat_test_upper.shape[0]):
    shear_rate_upper_mean[z]=np.mean(shear_rate_upper[locations_of_passed_stat_test_upper[z,0],locations_of_passed_stat_test_upper[z,1],locations_of_passed_stat_test_upper[z,2],:])
 #%%
for z in range(0,locations_of_passed_stat_test_lower.shape[0]):
    shear_rate_lower_mean[z]=np.mean(shear_rate_lower[locations_of_passed_stat_test_lower[z,0],locations_of_passed_stat_test_lower[z,1],locations_of_passed_stat_test_lower[z,2],:])    
 #%%   


# then place the acceptable solutions into an array and produce bash scripts to run momentum output 
# need to go back to list of file names and collect the input data from that to feed into simulation_file producer
passed_realisations=[]
for i in range(0,locations_of_passed_stat_test_upper.shape[0]):
    #filename=realisation_name_[i].split('_')
    
    no_SRD=no_SRD_key[locations_of_passed_stat_test_upper[i,0]]
    swap_rate_passed=swap_rate[locations_of_passed_stat_test_upper[i,1]]
    swap_number_passsed=swap_number[locations_of_passed_stat_test_upper[i,2]]
    realisation_index=str(0.0)
    
    for name in  glob.glob('vel.profile_'+fluid_name+'_no_tstat__no_rescale_*_0.0_'+str(no_SRD)+'*_SR_'+str(swap_rate_passed)+'_SN_'+str(swap_number_passsed)):
         passed_realisations.append(name)
    
    
    # non mom output filenames
    #vel.profile_${fluid_name}_no_tstat__no_rescale_${rand_int}_${realisation_index}_${no_SRD}_${box_size}_${timestep_input}_${SRD_MD_ratio}_${dump_freq}_${thermo_freq}_${no_timesteps}_T_${temp_}_lbda_${lambda}_SR_${swap_rate}_SN_${swap_number}
#%%    
SRD_MD_ratio_passed=[]
no_SRD_passed=[]
lamda_passed=[]
SR_passed=[]
SN_passed=[]
for i in range(0,locations_of_passed_stat_test_upper.shape[0]):
    filename=passed_realisations[i].split('_')
    
    no_SRD_passed.append(filename[9])
    box_side_length_scaled=filename[10]
    timestep=filename[11]
    SRD_MD_ratio_passed.append(filename[12])
    dump_freq=filename[13]
    thermo_freq=filename[14]
    no_timesteps=filename[15]
    temp=filename[16]
    lamda_passed.append(filename[19])
    SR_passed.append(filename[21])
    SN_passed.append(filename[23])


#%% 
# given an integer number prpoduce 4 integers either side with a specified step 
def values_either_side_20_percent(V):
    
    int_2=np.linspace(V,V*1.2,5)
    int_1=np.linspace(V*0.8,V,5)
    intdone=list(np.concatenate((np.round(int_1),np.round(int_2)),axis=0))
    intdone=list(dict.fromkeys(intdone))
    return intdone

for i in range(0,locations_of_passed_stat_test_upper.shape[0]):
     
     V=int(SR_passed[i])
     New_SR=values_either_side_20_percent(V)
    


#SR_passed=np.array([])
# turn to dict then back to list to remove duplicates
# no_SRD_passed=list(dict.fromkeys(no_SRD_passed))
# SRD_MD_ratio_passed=list(dict.fromkeys(SRD_MD_ratio_passed))
# lamda_passed=list(dict.fromkeys(lamda_passed))
# SR_passed=list(dict.fromkeys(SR_passed))
# SN_passed=list(dict.fromkeys(SN_passed))

      
    # z=no_SRD_key.index(no_SRD)
    # realisation_index=filename[8]
    # j=int(float(realisation_index))
    # EF=int(filename[21])
    # m=np.where(swap_rate==EF)
    # SN=int(filename[23])
    # k=np.where(swap_number==SN)
    
    
    #sim_file_producer_SRD inputs
    # data_transfer_instructions,
    # extra_code,
    # wd_path,np_req,num_task_req,
    # tempdir_req,
    # wall_time,
    # ram_requirement,
    # prod_run_file_name,
    # realisation_index_,
    # equilibration_timesteps,
    # VP_ave_freq,
    # abs_path_2_lammps_exec,
    # abs_path_2_lammps_script,
    # num_proc,no_timesteps,
    # thermo_freq,dump_freq,
    # SRD_box_size_wrt_solid_beads,
    # mean_free_path_pf_SRD_particles_cp_mthd_1_neg,
    # scaled_timestep,
    # mass_fluid_particle_wrt_pf_cp_mthd_1,
    # Number_MD_steps_per_SRD_with_pf_cp_mthd_1_neg,
    # number_SRD_particles_wrt_pf_cp_mthd_1,
    # locations_of_non_nan_neg,swap_number,
    # i_,
    # j_,
    # num_solns,
    # tolerance,
    # number_of_test_points,
    # swap_rate,
    # box_side_length_scaled,
    # scaled_temp,
    # eta_s,
    # Path_2_shell_scirpts,
    # Path_2_generic,
    # fluid_name
    
    
   
    





                    #       pearson_coeff_upper[z,m,k,i]=pearson_coeff_upper[z,m,k,i]
                    # else:
                    #       pearson_coeff_upper[z,m,k,i]=float('NAN')
                    #       shear_rate_upper[z,m,k,i]=float('NAN')
                    #       t_s[z,m,k,i]=float('NAN')
                    # if abs(perfect_linearity_comparison_lower[z,m,k,i])<0.01:
                    #     pearson_coeff_lower[z,m,k,i]=pearson_coeff_lower[z,m,k,i]
                    # else:
                    #     pearson_coeff_lower[z,m,k,i]=float('NAN')
                    #     shear_rate_lower[z,m,k,i]=float('NAN')
                    #     t_s[z,m,k,i]=float('NAN')
    
#%% plotting flux against shear rate

shear_rate_mean_of_both_cells=((np.abs(shear_rate_upper_mean)+np.abs(shear_rate_lower_mean))*0.5)

x=shear_rate_mean_of_both_cells
y=flux_ready_for_plotting
for i in range(0,swap_number.size):
    
   plt.plot(x[0,:,i],y[i,:])
   plt.xscale('log')
   plt.xlabel('$\dot{\gamma}\ [-]$')
   plt.yscale('log')
   plt.ylabel('$J_{z}(p_{x})\ [-]$',rotation=90)
plt.show()





# %%
