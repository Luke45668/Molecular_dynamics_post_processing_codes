#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:20:52 2023

This script will do the SRD calculations in dimensional form, then plot a number of graphs
from the petersen paper, the key advantage will be the ability to see the effect of dimensionless parameter 
choices


@author: lukedebono
"""
#%%

import os
import numpy as np

import matplotlib.pyplot as plt
import regex as re
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
#import seaborn as sns
import math as m
import scipy.stats
from datetime import datetime

#%% fixed values 

tolerance=0.01 # for solution error 
atol=0.01
rtol=0.01
number_of_test_points =500
Solvent_bead_SRD_box_density_cp_1 = np.array([np.linspace(3.5,1000,number_of_test_points)])
number_of_M_cp_1=Solvent_bead_SRD_box_density_cp_1.shape[1]
scaled_timestep=0.01



#%% fixed values for nitrogen 
# fluid_name='Nitrogen'
# rho_s = 847 #kg/m^3
# T_K=72.2 #+273.15 #Kelvin
# k_b= 1.380649e-23 #boltzmann in J K^-1

# # linear interpolation
# # to determine true visc from NIST data
# eta_s_1 = 0.00022081
# rho_1=	843.96
# eta_s_2 = 0.00023138
# rho_2 = 850.73	
# delta_rho=rho_2-rho_1 
# delta_eta_s =eta_s_2 - eta_s_1 
# grad_eta_s_rho_s = delta_eta_s/delta_rho

# eta_s_NIST=0.00022081 + ((rho_s -rho_1)*grad_eta_s_rho_s) 
eta_s=eta_s_NIST#*1000 #*1000 to convert kg to g
nu_s = eta_s/rho_s

#determine side length of simulaton box 
r_particle =50e-6
phi=0.005
N=2
Vol_box_at_specified_phi= N* (4/3)*np.pi*r_particle**3 /phi
box_side_length=np.cbrt(Vol_box_at_specified_phi)

rho_s = 1005##kg/m^3
r_particle =50e-6 #m 
T_K=300#+273.15 #Kelvin
k_b= 1.380649e-23 #boltzmann in J K^-1
eta_s_NIST=0.00085253	

eta_s=eta_s_NIST#*1000 #*1000 to convert kg to g
nu_s = eta_s/rho_s
rho_particle = 1200 #kg m^-3 PMMA spheres
mass_solid_particle= rho_particle * (4/3)*np.pi*(r_particle**3)
# calculating stokes number in fluid conditions for solid particle tests
Stokes_number=0.0001
Gamma_dot= 4.5*Stokes_number*eta_s_NIST/ (rho_particle * r_particle**2)
number_boxes_vec=np.linspace(2,64,63)
box_size_vec = np.array([box_side_length/number_boxes_vec])
mass_fluid_particle_wrt_pf_cp_mthd_1=(rho_s * (box_size_vec**3))/Solvent_bead_SRD_box_density_cp_1.T
fluid_name='H2O'

#%%
length_multiplier=0.05
lengthscale_parameter = length_multiplier*r_particle
box_side_length_scaled=box_side_length/lengthscale_parameter
box_size_to_lengthscale=box_size_vec/lengthscale_parameter
mass_multiplier=1
SRD_mass_scale_parameter = mass_multiplier* rho_s * (lengthscale_parameter**3)
r_particle_scaled = r_particle/lengthscale_parameter
#energy_parameter= 0.99607 * moles *1000*energy_multiplier #J x 100000
import units_lj_scalings
timescale_parameter=units_lj_scalings.units_lj_scalings(SRD_mass_scale_parameter,lengthscale_parameter,k_b,rho_s,eta_s,T_K)[1]
temperature_parameter=units_lj_scalings.units_lj_scalings(SRD_mass_scale_parameter,lengthscale_parameter,k_b,rho_s,eta_s,T_K)[2]
scaled_temp=T_K/temperature_parameter
#%%
import numpy as np
from SRD_master import *

box_size_vec = np.array([box_side_length/number_boxes_vec])
box_size_vec_nd=np.array([box_side_length_scaled/number_boxes_vec])
number_of_boxes_in_each_dim=number_boxes_vec
SRD_box_size_wrt_solid_beads =box_size_vec_nd
SRD_box_size_wrt_solid_beads_check = box_size_vec



SRD_non_dimensional_master_data=SRD_MASTER_calc_(mass_fluid_particle_wrt_pf_cp_mthd_1,box_side_length,number_boxes_vec,tolerance,scaled_timestep,atol,rtol,nu_s,Solvent_bead_SRD_box_density_cp_1 ,r_particle, box_size_vec ,box_side_length_scaled,T_K,SRD_mass_scale_parameter,lengthscale_parameter,energy_parameter,k_b,rho_s,eta_s)
sc_pos_soln=SRD_non_dimensional_master_data[0]
sc_neg_soln=SRD_non_dimensional_master_data[1]

mean_free_path_pf_SRD_particles_cp_mthd_1_neg=SRD_non_dimensional_master_data[2]
mean_free_path_to_box_ratio_neg=mean_free_path_pf_SRD_particles_cp_mthd_1_neg/SRD_box_size_wrt_solid_beads
mean_free_path_pf_SRD_particles_cp_mthd_1_pos=SRD_non_dimensional_master_data[3]
mean_free_path_to_box_ratio_pos=mean_free_path_pf_SRD_particles_cp_mthd_1_pos/SRD_box_size_wrt_solid_beads

Number_MD_steps_per_SRD_with_pf_cp_mthd_1_neg=SRD_non_dimensional_master_data[4]
Number_MD_steps_per_SRD_with_pf_cp_mthd_1_pos=SRD_non_dimensional_master_data[5]

number_SRD_particles_wrt_pf_cp_mthd_1_neg=SRD_non_dimensional_master_data[6]
number_SRD_particles_wrt_pf_cp_mthd_1_pos=SRD_non_dimensional_master_data[7]

mass_fluid_particle_wrt_pf_cp_mthd_1=SRD_non_dimensional_master_data[8]

comparison_neg=SRD_non_dimensional_master_data[9]
comparison_pos=SRD_non_dimensional_master_data[10]

SRD_timestep_cp_1_based_on_sphere_pf_neg_nd=SRD_non_dimensional_master_data[11]
SRD_MD_ratio_neg=SRD_timestep_cp_1_based_on_sphere_pf_neg_nd/scaled_timestep
SRD_step_neg_nd=SRD_timestep_cp_1_based_on_sphere_pf_neg_nd
SRD_timestep_cp_1_based_on_sphere_pf_pos_nd=SRD_non_dimensional_master_data[12]
SRD_step_pos_nd=SRD_timestep_cp_1_based_on_sphere_pf_pos_nd
SRD_MD_ratio_pos=SRD_timestep_cp_1_based_on_sphere_pf_pos_nd/scaled_timestep
 
#%% plotting the graphs 
#SC vs lamda/deltax
#Make one to plot this specific type of plot, then make function 


fig=plt.figure(figsize=(10,6))
gs=GridSpec(nrows=1,ncols=1)

fig.suptitle(fluid_name+': $Sc\ vs\ \\frac{\lambda}{\Delta x}$',size='large', wrap=True)

ax1= fig.add_subplot(gs[0]) 
for z in range(0,number_of_test_points):
    
    ax1.plot(mean_free_path_to_box_ratio_neg[z,:],sc_neg_soln[z,:],label='M+={}'.format(Solvent_bead_SRD_box_density_cp_1[0,z]))
    
    #ax1.legend(Solvent_bead_SRD_box_density_cp_1[0,z])
    ax1.plot(mean_free_path_to_box_ratio_pos[z,:],sc_pos_soln[z,:])
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$\\frac{\lambda}{\Delta x}\  [-]$', rotation='horizontal',ha='right',size='large')
    ax1.set_ylabel( '$Sc\ [-]$', rotation='horizontal',ha='right',size='large')
    ax1.grid('on')
    
plt.show()
    
# sort out legend for thius one 

#%% plot 2 
# Sc vs box size/lengthscale
fig=plt.figure(figsize=(10,6))
gs=GridSpec(nrows=1,ncols=1)

fig.suptitle(fluid_name+': $Sc\ vs$ $\\frac{\Delta x}{\\bar{\ell}}\\ $',size='large', wrap=True)

ax1= fig.add_subplot(gs[0]) 
for z in range(0,number_of_test_points):
    
    ax1.plot(box_size_to_lengthscale[0,:],sc_neg_soln[z,:],label='M+={}'.format(Solvent_bead_SRD_box_density_cp_1[0,z]))
    
    #ax1.legend(Solvent_bead_SRD_box_density_cp_1[0,z])
    ax1.plot(box_size_to_lengthscale[0,:],sc_pos_soln[z,:])
    
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.set_xlabel('$\\frac{\Delta x}{\\bar{\ell}}\   [-]$', rotation='horizontal',ha='right',size='large')
    ax1.set_ylabel( '$Sc\ [-]$', rotation='horizontal',ha='right',size='large')
    ax1.grid('on')
    
plt.show()
  

#%% plot 3 
#box size to length_scale  vs lammda to box size
fig=plt.figure(figsize=(10,6))
gs=GridSpec(nrows=1,ncols=1)

fig.suptitle(fluid_name+': $\\frac{\lambda}{\Delta x}\ $ vs $\\frac{\Delta x}{\\bar{\\ell}}\ $',size='large', wrap=True)

ax1= fig.add_subplot(gs[0]) 
for z in range(0,number_of_test_points):
    
    ax1.plot(box_size_to_lengthscale[0,:],mean_free_path_to_box_ratio_neg[z,:])#label='M+={}'.format(Solvent_bead_SRD_box_density_cp_1[0,z]))
    
    #ax1.legend(Solvent_bead_SRD_box_density_cp_1[0,z])
    #ax1.plot(box_size_to_lengthscale[0,:],mean_free_path_to_box_ratio_pos[z,:])
    
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    ax1.set_ylabel('$\\frac{\lambda{\Delta x}\$$[-]$', rotation='horizontal',ha='right',size='large')
    ax1.set_xlabel('$\\frac{\Delta x}{\\bar{\ell}}\ [-]$', rotation='horizontal',ha='right',size='large')
    ax1.grid('on')
    
plt.show()

#%% Plot 4

#box size to length_scale  vs  scaled SRD step 
fig=plt.figure(figsize=(10,6),)
gs=GridSpec(nrows=1,ncols=1)

fig.suptitle(fluid_name+': $\\frac{\Delta t_{SRD}}{\\bar{\\tau}}\ $ vs $\\frac{\Delta x}{\\bar{\ell}}\ $',size='large', wrap=True)

ax1= fig.add_subplot(gs[0]) 
for z in range(0,number_of_test_points):
    
    ax1.plot(box_size_to_lengthscale[0,:],SRD_step_pos_nd[z,:],label='M+={}'.format(Solvent_bead_SRD_box_density_cp_1[0,z]))
    
    #ax1.legend(Solvent_bead_SRD_box_density_cp_1[0,z])
    ax1.plot(box_size_to_lengthscale[0,:],SRD_step_neg_nd[z,:])
    
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.set_ylabel('$\\frac{\Delta t_{SRD}}{\\bar{\\tau}}\ [-]$', rotation='horizontal',ha='right',size='large')
    ax1.set_xlabel( '$\\frac{\Delta x}{\\bar{\ell}}\ [-]$', rotation='horizontal',ha='right',size='large')
    ax1.grid('on')
    
plt.show()

#%% Plot 5 

# plotting mean free path to collision cell size ratio vs timestep ratio  vs  Sc
fig=plt.figure(figsize=(18,7)) #width x height
gs=GridSpec(nrows=1,ncols=2)

fig.suptitle(fluid_name+':  $\\frac{\lambda}{\Delta x}\ $ vs $\\frac{\Delta t_{SRD}}{\Delta t_{MD}}$ vs  $Sc$',size='x-large', wrap=True)
#$\\frac{\lambda}{\Delta x}\ $
ax1= fig.add_subplot(gs[0,0],projection='3d') 
ax2= fig.add_subplot(gs[0,1],projection='3d') 
for z in range(0,number_of_test_points):
    
    ax1.plot(mean_free_path_to_box_ratio_neg[z,:],SRD_MD_ratio_neg[z,:],sc_neg_soln[z,:],marker ='x')
    ax2.plot(mean_free_path_to_box_ratio_pos[z,:],SRD_MD_ratio_pos[z,:],sc_pos_soln[z,:],marker ='o')
    
    #ax1.legend(Solvent_bead_SRD_box_density_cp_1[0,z])
   # ax1.plot(mean_free_path_to_box_ratio_pos[z,:],SRD_MD_ratio_pos[z,:],sc_pos_soln[z,:])
    
    #ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # ax1.set_zscale('log')
    ax1.set_xlabel('$\\frac{\lambda}{\Delta x}\ $', rotation='horizontal',ha='right',size='large')
    ax1.set_ylabel( '$\\frac{\Delta t_{SRD}}{\Delta t_{MD}}$', rotation='vertical',ha='right',size='large')
    ax1.set_zlabel( '$Sc$', rotation='horizontal',ha='right',size='large')
    #ax2.grid('on')
    ax2.set_xlabel('$\\frac{\lambda}{\Delta x}\ $', rotation='horizontal',ha='right',size='large')
    ax2.set_ylabel( '$\\frac{\Delta t_{SRD}}{\Delta t_{MD}}$', rotation='vertical',ha='right',size='large')
    ax2.set_zlabel( '$Sc$', rotation='horizontal',ha='right',size='large')
    #ax2.grid('on')
    
    
plt.show()

  



# %%
