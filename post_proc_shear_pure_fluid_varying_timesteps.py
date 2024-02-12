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
#from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import h5py as h5 
import multiprocessing as mp


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
# no_SRD=121670
# box_size=23
# no_SRD=270
# box_size=3
# no_SRD=58320
# # box_size=18
# no_SRD=2160
# box_size=6
# no_SRD=2560
# box_size=8
no_SRD=60835
box_size=23
#nu_bar=3
#delta_t_srd=0.014872025172594354
#nu_bar=0.9 
#delta_t_srd=0.05674857690605889
delta_t_srd=0.05071624521210362

box_vol=box_size**3
#erate= np.array([0.01,0.001,0.0001])
# #erate=np.array([0.01])
#erate=np.array([0.001,0.002,0.003])
erate= np.array([0.0005,0.001,0.002,0.005,0.01])

no_timestep=np.array([800000,400000,200000,80000,40000])

# estimating number of steps  required
strain=2
delta_t_md=delta_t_srd/10
strain_rate= erate
number_steps_needed= np.ceil(strain/(strain_rate*delta_t_md))
dump_freq=10
total_strain_actual=no_timestep*strain_rate*delta_t_md
#rho=10 
j_=3
rho=5
realisation_index=np.array([1,2,3])
filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/pure_srd_fix_deform_results/2_strain_units_test"
os.chdir(filepath)

#%% loading in arrays to tuple 
stress_tensor_summed_1d_tuple=()
stress_tensor_summed_tuple_shaped=()
delta_mom_pos_tensor_summed_1d_tuple=()
delta_mom_pos_tensor_summed_tuple_shaped=()
kinetic_energy_tensor_summed_1d_tuple=()
kinetic_energy_tensor_summed_tuple_shaped=()



array_size=((no_timestep/dump_freq)-1).astype('int')

     


for i in range(erate.size):
       cumsum_divsor_array=np.repeat(np.array([np.arange(1,array_size[i]+1)]), repeats=9, axis=0).T
       #loading in 
       stress_tensor_summed_1d_tuple=stress_tensor_summed_1d_tuple+(np.load("stress_tensor_summed_1d_test_erate_"+str(erate[i])+"_M_"+str(rho)+"_L_"+str(box_size)+".npy"),)
       # reshaping and taking realisation mean , then taking cumulative sum along timestep axis, then dividing by the array of total values 
       stress_tensor_summed_tuple_shaped=stress_tensor_summed_tuple_shaped+(np.cumsum(np.mean(np.reshape(stress_tensor_summed_1d_tuple[i],(1,j_,array_size[i],9)),axis=1),axis=1)/cumsum_divsor_array,)
       
       delta_mom_pos_tensor_summed_1d_tuple=delta_mom_pos_tensor_summed_1d_tuple+(np.load("delta_mom_pos_tensor_summed_1d_test_erate_"+str(erate[i])+"_M_"+str(rho)+"_L_"+str(box_size)+".npy"),)
       delta_mom_pos_tensor_summed_tuple_shaped=delta_mom_pos_tensor_summed_tuple_shaped+(np.cumsum(np.mean(np.reshape(delta_mom_pos_tensor_summed_1d_tuple[i],(1,j_,array_size[i],9)),axis=1),axis=1)/cumsum_divsor_array,)
       
       # different divisor needed for 6 values 
       cumsum_divsor_array_6=np.repeat(np.array([np.arange(1,array_size[i]+1)]), repeats=6, axis=0).T
      
       kinetic_energy_tensor_summed_1d_tuple=kinetic_energy_tensor_summed_1d_tuple+(np.load("kinetic_energy_tensor_summed_1d_test_erate_"+str(erate[i])+"_M_"+str(rho)+"_L_"+str(box_size)+".npy"),)
       kinetic_energy_tensor_summed_tuple_shaped=kinetic_energy_tensor_summed_tuple_shaped+(np.cumsum(np.mean(np.reshape(kinetic_energy_tensor_summed_1d_tuple[i],(1,j_,array_size[i],6)),axis=1),axis=1)/cumsum_divsor_array_6,)
       
       

labels_coll=["$\Delta p_{x}r_{x}$","$\Delta p_{y}r_{y}$","$\Delta p_{z}r_{z}$","$\Delta p_{x}r_{z}$","$\Delta p_{x}r_{y}$","$\Delta p_{y}r_{z}$","$\Delta p_{z}r_{x}$","$\Delta p_{y}r_{x}$","$\Delta p_{z}r_{y}$"]
labels_stress=["$\sigma_{xx}$","$\sigma_{yy}$","$\sigma_{zz}$","$\sigma_{xz}$","$\sigma_{xy}$","$\sigma_{yz}$","$\sigma_{zx}$","$\sigma_{zy}$","$\sigma_{yx}$"]
labels_gdot=["$\dot{\gamma}= "]

no_data_sets=erate.size

#%% calculating strain x points

strainplot_tuple=()
for i in range(0,erate.shape[0]):
    units_strain=(total_strain_actual[i]/array_size[i])
    strainplot=np.zeros((array_size[i]))
    for j in range(0,array_size[i]):
         strainplot[j]=j*units_strain
    
    strainplot_tuple=strainplot_tuple+(strainplot,)

        
mean_step=np.array([79998,39998,19998,7998,3998])

#%% plotting rolling average diagonal against collisions  

#stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[:,:,0:3])
labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})
for i in range(0,erate.shape[0]):
   
    for j in range(0,3):
        plt.plot(stress_tensor_summed_tuple_shaped[i][0,:,j],label=labels_stress[j],color=colour[j])
        plt.ylabel('$\sigma_{\\alpha \\beta}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$N_{coll}$")
      
        #plt.ylim((10.5,11))
        plt.ylim(5.3,5.4)
        plt.ylim(5,6)

    #plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\alpha}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
    plt.legend(loc='best')
    #plt.tight_layout()
    plt.savefig("rolling_ave_shear_stress_vs_collisions_tensor_elements_1_3_gdot_"+str(erate[i])+"_M_"+str(rho)+"_L_"+str(box_size)+".pdf",dpi=1200,bbox_inches='tight')
    plt.show()
 
#%% plotting rolling average diagonal against strain 

#stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[:,:,0:3])
labelpady=15
fontsize=15
std_dv_stress=[]
plt.rcParams.update({'font.size': 12})
for i in range(2,erate.shape[0]):
   

    for j in range(0,3):
        std_dv_stress.append(np.std(stress_tensor_summed_tuple_shaped[i][0,mean_step[i]:,j]))
        plt.plot(strainplot_tuple[i][:],stress_tensor_summed_tuple_shaped[i][0,:,j],label=labels_stress[j],color=colour[j])
        plt.ylabel('$\sigma_{\\alpha \\beta}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$\gamma$")
        #plt.ylim((10.5,11))
        plt.ylim(5.3,5.4)
        plt.ylim(5,6)

    #plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\alpha}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
    plt.legend(loc='best')
    #plt.tight_layout()
    plt.savefig("rolling_ave_shear_stress_vs_strain_tensor_elements_1_3_gdot_"+str(erate[i])+"_M_"+str(rho)+"_L_"+str(box_size)+".pdf",dpi=1200,bbox_inches='tight')
    plt.show()

#%% first normal stress difference rolling vs collisions 



labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})
for i in range(0,erate.shape[0]):
    #for j in range(0,3):
        N_1=stress_tensor_summed_tuple_shaped[i][0,:,0]-stress_tensor_summed_tuple_shaped[i][0,:,1]
        N_1_mean=np.mean(N_1[mean_step[i]:])
        plt.plot(N_1[:],label=labels_gdot[0]+str(erate[i])+", \\bar{N_{1}}="+str(sigfig.round(N_1_mean,sigfigs=3))+"$",color=colour[i])
        plt.ylabel('$N_{1}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$N_{coll}$")
        plt.ylim((-0.1,0.1))

plt.legend(loc='best')
    #plt.tight_layout()
plt.savefig("N_1_vs_coll_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
plt.show()

#%%first normal stress difference rolling vs strain 



labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})
for i in range(0,erate.shape[0]):
    #for j in range(0,3):
       
        N_1=stress_tensor_summed_tuple_shaped[i][0,:,0]-stress_tensor_summed_tuple_shaped[i][0,:,1]
        N_1_mean=np.mean(N_1[mean_step[i]:])
        plt.plot(strainplot_tuple[i][:],N_1[:],label=labels_gdot[0]+str(erate[i])+", \\bar{N_{1}}="+str(sigfig.round(N_1_mean,sigfigs=3))+"$",color=colour[i])
        plt.ylabel('$N_{1}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$\gamma$")
        plt.ylim((-0.1,0.5))

plt.legend(loc='best')
    #plt.tight_layout()
plt.savefig("N_1_vs_strain_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
plt.show()


#%% second normal vs collisions 


labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})
for i in range(0,erate.shape[0]):
    #for j in range(0,3):
        N_2=stress_tensor_summed_tuple_shaped[i][0,:,1]-stress_tensor_summed_tuple_shaped[i][0,:,2]
        N_2_mean=np.mean(N_2[mean_step[i]:])
        plt.plot(N_2[:],label=labels_gdot[0]+str(erate[i])+", \\bar{N_{2}}="+str(sigfig.round(N_2_mean,sigfigs=3))+"$",color=colour[i])
        plt.ylabel('$N_{2}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$N_{coll}$")
        #plt.ylim((-0.1,0.1))

plt.legend(loc='best')
    #plt.tight_layout()
plt.savefig("N_2_coll_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
plt.show()

#%% second normal vs strain 


labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})
for i in range(0,erate.shape[0]):
    #for j in range(0,3):
        N_2=stress_tensor_summed_tuple_shaped[i][0,:,1]-stress_tensor_summed_tuple_shaped[i][0,:,2]
        N_2_mean=np.mean(N_2[mean_step[i]:])
        plt.plot(strainplot_tuple[i][:],N_2[:],label=labels_gdot[0]+str(erate[i])+", \\bar{N_{2}}="+str(sigfig.round(N_2_mean,sigfigs=3))+"$",color=colour[i])
        plt.ylabel('$N_{2}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$\gamma$")
        plt.ylim((-0.1,0.1))

plt.legend(loc='best')
    #plt.tight_layout()
plt.savefig("N_2_strain_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
plt.show()



#%% plotting off diagonal vs collisions 
labelpady=10
fontsize=15
plt.rcParams.update({'font.size': 12})

for i in range(0,erate.shape[0]):
    for j in range(3,4):
        #stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[i,mean_step:,3])
        plt.plot(stress_tensor_summed_tuple_shaped[i][0,:,j],label=labels_stress[j]+", "+labels_gdot[0]+str(erate[i])+"$",color=colour[i])
        plt.ylabel('$\sigma_{\\alpha \\beta}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$N_{coll}$")
        plt.ylim((0,0.075))
        
    #plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\beta}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
    plt.legend()
    #plt.tight_layout()
    #plt.savefig("rolling_ave_shear_stress_tensor_elements_xy_gdot_allgdot_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
    plt.savefig("rolling_ave_shear_stress_tensor_vs_coll_elements_xy_gdot_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
plt.show()

#%% plotting whole off diagonal vs strain 
labelpady=10
fontsize=15
plt.rcParams.update({'font.size': 12})

for i in range(0,erate.shape[0]):
    for j in range(3,4):
        #stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[i,mean_step:,3])
        plt.plot(strainplot_tuple[i][:],stress_tensor_summed_tuple_shaped[i][0,:,j],label=labels_stress[j]+", "+labels_gdot[0]+str(erate[i])+"$",color=colour[i])
        plt.ylabel('$\sigma_{\\alpha \\beta}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$\gamma$")
        plt.ylim((0,1))
        
    #plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\beta}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
    plt.legend()
    #plt.tight_layout()
    #plt.savefig("rolling_ave_shear_stress_tensor_elements_xy_gdot_allgdot_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
    plt.savefig("rolling_ave_shear_stress_tensor_vs_strain_elements_xy_gdot_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
plt.show()


#%% plotting whole off diagonal 
labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})

for i in range(0,erate.shape[0]):
    for j in range(3,9):
        #stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[i,mean_step:,3])
        plt.plot(stress_tensor_summed_tuple_shaped[i][0,:,j],label=labels_stress[j]+", "+labels_gdot[0]+str(erate[i])+"$",color=colour[j])
        plt.ylabel('$\sigma_{\\alpha \\beta}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$N_{coll}$")
        plt.ylim((-0.02,0.075))

    #plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\beta}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
    plt.legend()
    #plt.tight_layout()
    #plt.savefig("rolling_ave_shear_stress_tensor_elements_xy_gdot_allgdot_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
    plt.savefig("rolling_ave_shear_stress_tensor_vs_coll_all_elements_gdot_"+str(erate[i])+"_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
    plt.show()

#%%
labelpady=15
fontsize=15
plt.rcParams.update({'font.size': 12})

for i in range(0,erate.shape[0]):
    for j in range(3,9):
        #stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[i,mean_step:,3])
        plt.plot(strainplot_tuple[i][:],stress_tensor_summed_tuple_shaped[i][0,:,j],label=labels_stress[j]+", "+labels_gdot[0]+str(erate[i])+"$",color=colour[j])
        plt.ylabel('$\sigma_{\\alpha \\beta}$', rotation=0, labelpad=labelpady)
        plt.xlabel("$\gamma$")
        plt.ylim((-0.02,0.075))

    #plt.axhline(stress_tensor_summed_realisation_mean_rolling_hline,0,1000, label="$\\bar{\sigma_{\\alpha \\beta}}="+str(sigfig.round(stress_tensor_summed_realisation_mean_rolling_hline,sigfigs=3))+"$",linestyle='dashed',color=colour[6])
    plt.legend()
    #plt.tight_layout()
    #plt.savefig("rolling_ave_shear_stress_tensor_elements_xy_gdot_allgdot_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
    plt.savefig("rolling_ave_shear_stress_tensor_vs_strain_all_elements_gdot_"+str(erate[i])+"_M_"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
    plt.show()



#%% viscosity estimate
#stress_tensor_summed_realisation_mean_rolling_hline=np.mean(stress_tensor_summed_realisation_mean_rolling[:,mean_step:,3],axis=1)

alpha=np.pi
dim=3
def collisional_visc(alpha,rho,dim):
    coll_visc= (1/(6*dim*rho))*(rho-1+np.exp(-rho))*(1-np.cos(alpha))
    return coll_visc
def kinetic_visc(alpha,rho):
    kin_visc= (5*rho)/((rho-1+np.exp(-rho))*(2-np.cos(alpha)-np.cos(2*alpha)))  -1 
    return kin_visc

total_kinematic_visc= kinetic_visc(alpha,rho) + collisional_visc(alpha,rho,dim)
shear_dynamic_visc_prediction= total_kinematic_visc*rho

#%% stress vs strain rate plot 
viscosity=np.zeros((erate.size))
stress_tensor_summed_mean=np.zeros((erate.size))
for i in range(5):
   stress_tensor_summed_mean[i]=np.mean(stress_tensor_summed_tuple_shaped[i][0,mean_step[i]:,3])
   viscosity[i]=stress_tensor_summed_mean[i]/erate[i]


#%%
fit=np.polyfit(erate,stress_tensor_summed_mean,1)
plt.scatter(np.asarray(erate[:],float), stress_tensor_summed_mean[:])
plt.plot(erate,fit[0]*erate + fit[1], label="$\eta="+str(sigfig.round(fit[0],sigfigs=4))+", \eta_{T}="+str(sigfig.round(shear_dynamic_visc_prediction,sigfigs=4))+"$")
plt.xticks(erate) 
plt.xlabel("$\dot{\gamma}$",rotation=0)

plt.ylabel("$\sigma_{xz}$",rotation=0,labelpad=labelpady)
plt.legend()
plt.savefig("shear_stress_vs_shear_rate_gdot_"+str(erate[0])+"_"+str(erate[-1])+"_M"+str(rho)+"_L_"+str(box_size)+".png",dpi=1200)
plt.show()



# %%
