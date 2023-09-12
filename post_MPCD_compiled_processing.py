##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will import data arrays for flux vs shear and velocity profiles 

after an MPCD simulation. 
"""
#%%
import sigfig
import os
import numpy as np
import matplotlib.pyplot as plt
import regex as re
import pandas as pd
#import pyswarms as ps

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
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
import fnmatch
#%% loading in data 
phi=[0.005,0.0005,5e-05]
box_side_length_scaled=[119,256,551]
#file_search_strings=['flux_fitting_params_*','flux_ready_for_plotting_*','shear_rate_mean_error_of_both_cells_*','shear_rate_mean_of_both_cells_*']
fluid_name='H20'
path_2_compiled_data= '/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/T_1_compiled_data_all_phi/H20_data'
os.chdir(path_2_compiled_data)

flux_ready_for_plotting=()
shear_rate_mean_of_both_cells=()
shear_rate_mean_error_of_both_cells=()
flux_fitting_params=()
files_gen_string= '*'+fluid_name+'*'
all_files_in_folder=[]
for name in glob.glob(files_gen_string):
     
     all_files_in_folder.append(name)

all_files_in_tuple=()
for name in all_files_in_folder:
     all_files_in_tuple=all_files_in_tuple+(np.load(name),)

# problem is if order changes in file 
flux_fitting_params=np.array(all_files_in_tuple[0:3])
flux_ready_for_plotting=np.vstack(np.array(all_files_in_tuple[3:6]))
shear_rate_mean_error_of_both_cells=np.vstack(np.array(all_files_in_tuple[6:9]))
shear_rate_mean_of_both_cells=np.vstack(np.array(all_files_in_tuple[9:12]))


org_var_1=np.array([3,7,15,30,60,150,300,600,900,1200])
org_var_2=np.array([1,10,100,1000])
# need to make sure you note which fitting indices you use below
org_var_1_fitting_start_index=0
org_var_1_fitting_end_index=6
size_of_new_data=org_var_1_fitting_end_index-org_var_1_fitting_start_index
shear_rate_mean_error_of_both_cells_relative=shear_rate_mean_error_of_both_cells/shear_rate_mean_of_both_cells
shear_rate_mean_error_of_both_cell_mean_over_selected_points_relative= np.mean(shear_rate_mean_error_of_both_cells_relative[:,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:],axis=1)

def func4(x, a, b):
     return (a*x) +b 

y_residual_in_fit=np.zeros((len(phi),size_of_new_data,org_var_2.size))
for z in range(0,len(phi)):    
    for i in range(0,org_var_2.size):     
     y_residual_in_fit[z,:,i]=func4(shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],flux_fitting_params[z,i,0] ,flux_fitting_params[z,i,1])-flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i]
        
relative_y_residual_mean= np.mean(y_residual_in_fit/flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:],axis=1)

total_error_relative_in_flux_fit= relative_y_residual_mean+shear_rate_mean_error_of_both_cell_mean_over_selected_points_relative


labelpadx=15
labelpady=60
fontsize=20
count=1
org_var_1_index=org_var_1_fitting_start_index
org_var_2_index=4
plt.rcParams.update({'font.size': 15})

shear_viscosity=[]
gradient_of_fit=[]


for i in range(0,org_var_2_index):

        
     
    
     
     for z in range(0,len(phi)):
          x=shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:]
          y=flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:]
     
     
        
          shear_viscosity_=10** (flux_fitting_params[z,i,1])
          shear_viscosity_abs_error = shear_viscosity_*total_error_relative_in_flux_fit[z,i]
          
          grad_fit=(flux_fitting_params[z,i,0])
          grad_fit_abs_error= grad_fit*shear_rate_mean_error_of_both_cell_mean_over_selected_points_relative[z,i]
          print('Dimensionless_shear_viscosity:',shear_viscosity_,',abs error',shear_viscosity_abs_error)
          
          print('Grad of fit =',grad_fit,',abs error', grad_fit_abs_error)
          plt.scatter(x[:,i],y[:,i],label="$L=$"+str(box_side_length_scaled[z])+", grad$=$"+str(sigfig.round(grad_fit,sigfigs=2))+"$\pm$"+str(sigfig.round(grad_fit_abs_error,sigfigs=1)),marker='x')
          plt.plot(x[:,i],func4(x[:,i],flux_fitting_params[z,i,0],flux_fitting_params[z,i,1]),'--')
          plt.xlabel('$log(\dot{\gamma}\\tau)$', labelpad=labelpadx,fontsize=fontsize)
          plt.ylabel('$log(J_{z}(p_{x})$$\ \\frac{\\tau^{3}}{\\varepsilon})$',rotation=0,labelpad=labelpady,fontsize=fontsize)
          plt.legend()
     
     plt.show() 
     

       

# def plotting_flux_vs_shear_rate(shear_rate_mean_error_of_both_cells,func4,labelpadx,labelpady,params,fontsize,box_side_length_scaled,number_of_solutions,flux_ready_for_plotting,org_var_1_index,shear_rate_mean_of_both_cells):
    
#     for z in range(0,number_of_solutions):
        
        
#         x=shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:]
#         y=flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:]
        
#         for i in range(0,org_var_2_index):
        
#         #for i in range(0,1):
#             #if z==0:
#                 j=i
                
#                 # need to add legend to this 
#                 plt.scatter(x[:,i],y[:,i],label='$L=${}'.format(np.around(box_side_length_scaled[0,z]),decimals=0),marker='x')
#                 #plt.errorbar(x[:,i],y[:,i],xerr=x_pos_error[:,i],ls ='',capsize=3,color='r')
#                 plt.plot(x[:,i],func4(x[:,i],flux_fitting_params[z,i,0],flux_fitting_params[z,i,1]))
#                 #plt.xscale('log')
#                 plt.xlabel('$log(\dot{\gamma}\\tau)$', labelpad=labelpadx,fontsize=fontsize)
#                 #plt.yscale('log')
#                 plt.ylabel('$log(J_{z}(p_{x})$$\ \\frac{\\tau^{3}}{\\varepsilon})$',rotation=0,labelpad=labelpady,fontsize=fontsize)
#                 plt.legend()
#                 #plt.show() 
#                 shear_viscosity_=10** (flux_fitting_params[z,i,1])
#                 shear_viscosity.append(shear_viscosity_)
#                 shear_viscosity_abs_error = shear_viscosity_*total_error_relative_in_flux_fit[z,i]
                
#                 grad_fit=(flux_fitting_params[z,i,0])
#                 grad_fit_abs_error= grad_fit*shear_rate_mean_error_of_both_cell_mean_over_selected_points_relative[i]
#                 gradient_of_fit.append(grad_fit)
#                 print('Dimensionless_shear_viscosity:',shear_viscosity_,',abs error',shear_viscosity_abs_error)
#                 print('Grad of fit =',grad_fit,',abs error', grad_fit_abs_error)
#                 plt.show() 
           
        
                    
# plotting_flux_vs_shear_rate(shear_rate_mean_error_of_both_cells,func4,labelpadx,labelpady,params,fontsize,box_side_length_scaled,number_of_solutions,flux_ready_for_plotting,org_var_1_index,shear_rate_mean_of_both_cells)    









#%%


for name in glob.glob(numpy_arrays_gen_string):
   
      
        numpy_arrays.append(name)
        
       
# loading flux files need to label the pure fluid ones differently 
flux_data_005_tuple=()       
pattern = '*flux_ready_for_plotting_phi0.005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     flux_data_005_tuple=flux_data_005_tuple+(flux_in,)
     print(count) 
# loading flux files need to label the pure fluid ones differently 
flux_data_0005_tuple=()       
pattern = '*flux_ready_for_plotting_phi0.0005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     flux_data_0005_tuple=flux_data_0005_tuple+(flux_in,)
     print(count)
     
# loading flux files need to label the pure fluid ones differently 
flux_data_00005_tuple=()       
pattern = '*flux_ready_for_plotting_phi0.00005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     flux_data_00005_tuple=flux_data_00005_tuple+(flux_in,)
     print(count)
     
     
##shear rate tuples     
#0.005
shear_rate_005_data_tuple=()       
pattern = '*shear_rate_mean_of_both_cells_phi0.005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     shear_rate_005_data_tuple=shear_rate_005_data_tuple+(flux_in,)
     print(count)
     
# shear rate data 
shear_rate_005_error_data_tuple=()       
pattern = '*shear_rate_mean_error_of_both_cells_phi0.005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     shear_rate_005_error_data_tuple=shear_rate_005_error_data_tuple+(flux_in,)
     print(count)
#0.0005    
# shear rate data 
shear_rate_0005_data_tuple=()       
pattern = '*shear_rate_mean_of_both_cells_phi0.0005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     shear_rate_0005_data_tuple=shear_rate_0005_data_tuple+(flux_in,)
     print(count)
     
# shear rate data 
shear_rate_0005_error_data_tuple=()       
pattern = '*shear_rate_mean_error_of_both_cells_phi0.0005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     shear_rate_0005_error_data_tuple=shear_rate_0005_error_data_tuple+(flux_in,)
     print(count)
     
#0.00005
# shear rate data 
shear_rate_00005_data_tuple=()       
pattern = '*shear_rate_mean_of_both_cells_phi0.00005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     shear_rate_00005_data_tuple=shear_rate_00005_data_tuple+(flux_in,)
     print(count)
     
# shear rate data 
shear_rate_00005_error_data_tuple=()       
pattern = '*shear_rate_mean_error_of_both_cells_phi0.00005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     shear_rate_00005_error_data_tuple=shear_rate_00005_error_data_tuple+(flux_in,)
     print(count)
 
 
# fitted curve parameters
#0.005
params_005_tuple=()       
pattern = '*_params_phi0.005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     params_005_tuple=params_005_tuple+(flux_in,)
     print(count)
#0.0005
params_0005_tuple=()       
pattern = '*_params_phi0.0005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     params_0005_tuple=params_0005_tuple+(flux_in,)
     print(count)
     
#0.00005
params_00005_tuple=()       
pattern = '*_params_phi0.00005*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     params_0005_tuple=params_0005_tuple+(flux_in,)
     print(count)     
     
     
# velocity profile data
VP_tuple=()       
pattern = '*VP_steady_state_data_upper_truncated_time_averaged_*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     VP_tuple=VP_tuple+(flux_in,)
     print(count)

VP_z_tuple=()       
pattern = '*VP_z_data_upper_phi*'
count=0
filtered_list = fnmatch.filter(numpy_arrays, pattern) 
for name in filtered_list:
     count=count+1
     flux_in=np.load(name)
     
     VP_z_tuple=VP_z_tuple+(flux_in,)

# %%

#plotting flux vs shear 
def func4(x, a, b):
   #return np.log(a) + np.log(b*x)
   #return (a*(x**b))
   return (a*x) +b 
   #return a*np.log(b*x)+c
   
labelpadx=15
labelpady=40
fontsize=15
count=1
swap_number_index=1 
number_of_solutions=1



for z in range(0,2):
        
        
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
    
