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

#file_search_strings=['flux_fitting_params_*','flux_ready_for_plotting_*','shear_rate_mean_error_of_both_cells_*','shear_rate_mean_of_both_cells_*']
fluid_name='Nitrogen'
path_2_compiled_data= '/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/T_1_compiled_data_all_phi/'+str(fluid_name)+'_data'
os.chdir(path_2_compiled_data)

flux_ready_for_plotting=()
shear_rate_mean_of_both_cells=()
shear_rate_mean_error_of_both_cells=()
flux_fitting_params=()
files_gen_string= '*'+fluid_name+'*'
all_files_in_folder=[]
for name in glob.glob(files_gen_string):
     
     all_files_in_folder.append(name)
     
box_side_length_scaled=[]
box_size_loc=14
run_number=[]
run_number_loc=16
phi=[]
phi_loc=5
# only need the first 3 files as the rest are duplicate info    
for i in range(0,3):
     
     filename_split=all_files_in_folder[i].split('_')
     box_side_length_scaled.append(int(float(filename_split[box_size_loc])))
     run_number.append(int(filename_split[run_number_loc][:-4]))
     phi.append(filename_split[phi_loc])

box_side_length_scaled.sort(key=int)
phi.sort(key=float,reverse=True)



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
#%%
# need to make sure you note which fitting indices you use below
org_var_1_fitting_start_index=6
org_var_1_fitting_end_index=10


size_of_new_data=org_var_1_fitting_end_index-org_var_1_fitting_start_index
shear_rate_mean_error_of_both_cells_relative=shear_rate_mean_error_of_both_cells/shear_rate_mean_of_both_cells
shear_rate_mean_error_of_both_cell_mean_over_selected_points_relative= np.mean(shear_rate_mean_error_of_both_cells_relative[:,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:],axis=1)

def func4(x, a, b):
     return (a*x) +b 

y_residual_in_fit=np.zeros((len(phi),size_of_new_data,org_var_2.size))
flux_vs_shear_regression_line_params=np.zeros((len(phi),org_var_2.size,2))

for z in range(0,len(phi)):    
    for i in range(0,org_var_2.size):     
     flux_vs_shear_regression_line_params[z,i,:]= scipy.optimize.curve_fit(func4,shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],method='lm',maxfev=5000)[0]
     y_residual_in_fit[z,:,i]=func4(shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i],flux_fitting_params[z,i,0] ,flux_fitting_params[z,i,1])-flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,i]
        
print(flux_vs_shear_regression_line_params)

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

flux_fitting_params=flux_vs_shear_regression_line_params

#for i in range(0,org_var_2_index):
for i in range(0,1):
        
     
    
     
     for z in range(0,len(phi)):
          x=shear_rate_mean_of_both_cells[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:]
          y=flux_ready_for_plotting[z,org_var_1_fitting_start_index:org_var_1_fitting_end_index,:]
     
     
        
          shear_viscosity_=10** (flux_fitting_params[z,i,1])
          shear_viscosity_abs_error = shear_viscosity_*total_error_relative_in_flux_fit[z,i]
          
          grad_fit=(flux_fitting_params[z,i,0])
          grad_fit_abs_error= grad_fit*shear_rate_mean_error_of_both_cell_mean_over_selected_points_relative[z,i]
          print('Dimensionless_shear_viscosity:',shear_viscosity_,',abs error',shear_viscosity_abs_error)
          
          print('Grad of fit =',grad_fit,',abs error', grad_fit_abs_error)
          plt.scatter(x[:,i],y[:,i],label="$L/\ell=$"+str(box_side_length_scaled[z])+", grad$=$"+str(sigfig.round(grad_fit,sigfigs=2))+"$\pm$"+str(sigfig.round(grad_fit_abs_error,sigfigs=1)),marker='x')
          plt.plot(x[:,i],func4(x[:,i],flux_fitting_params[z,i,0],flux_fitting_params[z,i,1]),'--')
          plt.xlabel('$log(\dot{\gamma}\\tau)$', labelpad=labelpadx,fontsize=fontsize)
          plt.ylabel('$log(J_{z}(p_{x})$$\ \\frac{\\tau^{3}}{\\varepsilon})$',rotation=0,labelpad=labelpady,fontsize=fontsize)
          plt.legend(loc='upper right',bbox_to_anchor=(0.25,-0.1))
          #plt.legend(loc='best')
     plt.tight_layout()
     plt.savefig(fluid_name+"_flux_vs_shear_swap_rates_"+str(org_var_1[org_var_1_fitting_start_index])+"_"+str(org_var_1[org_var_1_fitting_end_index-1])+"_run_number_"+str(run_number[0])+"_"+str(run_number[1])+"_"+str(run_number[2])+".pdf",dpi=500, bbox_inches='tight')
     
     plt.show() 
     
 
shear_viscosity= 10**flux_fitting_params[:,:,1]
shear_viscosity_abs_error = shear_viscosity_*total_error_relative_in_flux_fit[:,:]

#%%

np.save(fluid_name+"_shear_viscosity_data_run_swap_rates_"+str(org_var_1[org_var_1_fitting_start_index])+"_"+str(org_var_1[org_var_1_fitting_end_index-1])+"_run_number_"+str(run_number[0])+"_"+str(run_number[1])+"_"+str(run_number[2])+".npy",shear_viscosity)
np.save(fluid_name+"_shear_viscosity_abs_error_data_run_swap_rates_"+str(org_var_1[org_var_1_fitting_start_index])+"_"+str(org_var_1[org_var_1_fitting_end_index-1])+"_run_number_"+str(run_number[0])+"_"+str(run_number[1])+"_"+str(run_number[2])+".npy",shear_viscosity_abs_error)
       
  


        
     
#%% viscosity vs phi plot 
labelpadx=5
labelpady=15
shear_viscosity_abs_error_max=np.amax(shear_viscosity_abs_error,axis=0)
for z in range(0,4):
          x=phi[:]
          y=shear_viscosity[:,z]
     
          #plt.scatter(x[:,i],y[:,i],label="$L=$"+str(box_side_length_scaled[z])+", grad$=$"+str(sigfig.round(grad_fit,sigfigs=2))+"$\pm$"+str(sigfig.round(grad_fit_abs_error,sigfigs=1)),marker='x')
          plt.plot(x,y,"--",label="$N_{v,x}=$"+str(org_var_2[z])+", $\Delta\eta_{max}=$"+str(sigfig.round(shear_viscosity_abs_error_max[z],sigfigs=2)),marker='x')
          plt.xlabel('$\phi$', labelpad=labelpadx,fontsize=fontsize)
          plt.yscale('log')
          #plt.xscale('log')
          plt.ylabel('$\eta \\frac{\ell^{3}}{\epsilon\\tau}$',rotation=0,labelpad=labelpady,fontsize=fontsize)
          plt.legend()
plt.tight_layout()     
plt.savefig(fluid_name+"_shear_eta_vs_phi_"+str(org_var_1[org_var_1_fitting_start_index])+"_"+str(org_var_1[org_var_1_fitting_end_index-1])+"_run_number_"+str(run_number[0])+"_"+str(run_number[1])+"_"+str(run_number[2])+".pdf",dpi=500, bbox_inches='tight')
plt.show() 


# %%
