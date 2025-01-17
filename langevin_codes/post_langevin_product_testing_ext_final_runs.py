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
  'solid', 
 'dashed', 'dashdot', '--']

linestyle_tuple = [
    
     ('dotted',                (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dotted',        (0, (1, 1))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

#%% 
marker=['x','+','^',"1","X","d","*","P","v","."]

damp=np.array([ 0.035, 0.035 ,0.035,0.035,0.035,0.035])
K=np.array([ 30, 60,100,150,300,600 ])
K=np.array([ 30,60,100,300 ])
K=np.array([ 15,60,120 ])

#K=np.array([  100,300,600,1200 ])
thermal_damp_multiplier=np.flip(np.array([25,25,25,25,25,25,25,100,100,100,100,100,
100,100,100,100,250,250]))/10

erate=np.flip(np.linspace(1,0.005,24))


e_in=0
e_end=erate.size
n_plates=100

strain_total=100

path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/final_plate_run_x_stretch/no_visc_10_reals_5tstats"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/final_plate_runs_tuples/"

thermo_vars='         KinEng         PotEng         Press           Temp         Ecouple       Econserve    c_uniaxnvttemp'

j_=10

sim_fluid=30.315227255599112

eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
damp_ratio=mass_pol/damp


#%% load in tuples
label='damp_'+str(damp)+'_K_'+str(K)+'_'


os.chdir(path_2_log_files)
#os.mkdir("tuple_results")
#os.chdir("tuple_results")

def batch_load_tuples(label,tuple_name):

    with open(label+tuple_name, 'rb') as f:
         load_in= pck.load(f)

    return load_in


spring_force_positon_tensor_batch_tuple=()
log_file_batch_tuple=()
dirn_vector_batch_tuple=()
transformed_pos_batch_tuple=()
transformed_vel_batch_tuple=()
e_end=[]



for i in range(K.size):
    label='damp_'+str(damp[i])+'_K_'+str(K[i])+'_'
    #label='damp_'+str(thermal_damp_multiplier[i])+'_K_'+str(K[0])+'_'
    print(label)

    spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(batch_load_tuples(label,
                                                            "spring_force_positon_tensor_tuple.pickle"),)
    print(len( spring_force_positon_tensor_batch_tuple[i]))
   
    log_file_batch_tuple=log_file_batch_tuple+(batch_load_tuples(label,
                                                            "log_file_tuple.pickle"),)
    print(len(log_file_batch_tuple[i]))
    
    dirn_vector_batch_tuple=dirn_vector_batch_tuple+(batch_load_tuples(label,
                                                            "dirn_vector_tuple.pickle"),)
    print(len(dirn_vector_batch_tuple[i]))
    
    transformed_pos_batch_tuple=transformed_pos_batch_tuple+(batch_load_tuples(label,
                                                            "transformed_pos_tuple.pickle"),)
    
    print(len(transformed_pos_batch_tuple[i]))

    transformed_vel_batch_tuple=transformed_vel_batch_tuple+(batch_load_tuples(label,
                                                            "transformed_vel_tuple.pickle"),)
    print(len(transformed_vel_batch_tuple[i]))
    
    e_end.append(len(spring_force_positon_tensor_batch_tuple[i]))
    

# need to add velocity and speed distributions, but need an equilirbium simulation to verify 


     

#%% strain points for temperatuee data 

strainplot_tuple=()

for i in range(erate.size):
    
    strain_plotting_points= np.linspace(0,strain_total,1002)

    strainplot_tuple=strainplot_tuple+(strain_plotting_points,)  
    print(strainplot_tuple[i].size)

def strain_plotting_points(total_strain,points_per_iv):
     #points_per_iv= number of points for the variable measured against strain 
     strain_unit=total_strain/points_per_iv
     strain_plotting_points=np.arange(0,total_strain,strain_unit)
     return  strain_plotting_points


#%% fix tdamp vary k
from fitter import Fitter
#NOTE: have I included the phantom particles in the velocity distributions ?
folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=5
final_temp=np.zeros((erate.size))
mean_temp_tuple=()
plt.rcParams["figure.figsize"] = (24,12 )
plt.rcParams.update({'font.size': 16})
#NOTE need to add an econserve plot as this should be constant 

for j in range(K.size-2):
    j=1
    mean_temp_array=np.zeros((erate[:e_end[j]].size))

    skip_array=np.array([7,8,9,10])
   
    for i in range(erate[:e_end[j]].size):
    # for i in range(skip_array.size):
    #         i=skip_array[i]
        
           

        
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
            mean_temp_array[i]=np.mean(log_file_batch_tuple[j][i][500:,column])


            plt.subplot(2, 3, 2)
            column=2
            grad_pe=np.gradient(log_file_batch_tuple[j][i][:,column])
            grad_mean=np.mean(grad_pe[500:])
            plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column],
            label="K="+str(K[j])+",$\\bar{grad}="+str(grad_mean)+"$")
            #plt.plot(strainplot_tuple[i][:],grad_pe)
            plt.ylabel("$E_{p}$")
            #plt.yscale('log')
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

            # assume first 3 particles are stokes beads

            vel_data_unsorted=transformed_vel_batch_tuple[j][i].astype('float')
            vel_data=np.zeros((j_,1000,300,3))
            for l in range(100):
                 start=l*6
                 end=start+3
                
                 start_in=l*3
                 end_in=start_in+3
                 vel_data[:,:,start_in:end_in,:]=vel_data_unsorted[:,:,start:end,:]
                 

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
            
            


        
            #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
           
                #     plt.xlabel("$\gamma$")
                

            
                # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
            #plt.ylim(0,2)
            #plt.yscale('log')
            
            plt.show()

            speed=np.sqrt(x_vel**2 +y_vel**2 +z_vel**2)
            maxwell = scipy.stats.maxwell
            params = maxwell.fit(speed)
            print(params)
            # (0, 4.9808603062591041)

            plt.hist(speed, bins=20,density=True)
            x = np.linspace(np.min(speed),np.max(speed),speed.size)
            plt.plot(x, maxwell.pdf(x, *params), lw=3)
            plt.xlabel("$|v|$")
            plt.ylabel("Density")
           
            plt.show()

            energy=0.5*5*(speed**2)
            maxwell = scipy.stats.maxwell
            params = maxwell.fit(energy)
            print(params)
            # (0, 4.9808603062591041)

            plt.hist(speed, bins=20,density=True)
            x = np.linspace(np.min(energy),np.max(energy),energy.size)
            plt.plot(x, maxwell.pdf(x, *params), lw=3)
            plt.xlabel("$E_{k}$")
            plt.ylabel("Density")
           
            plt.show()


    mean_temp_tuple=mean_temp_tuple+(mean_temp_array,)




#%%
marker=['x','+','^',"1","X","d","*","P","v","."]

# for j in range(K.size):
for j in range(K.size):
    #plt.scatter(erate[:e_end[j]],mean_temp_tuple[j],label="$K="+str(K[j])+"$" ,marker=marker[j])
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

#%% time series plots of stress

for j in range(K.size):
    for i in range(e_end[j]):
         mean_stress_tensor=np.mean(spring_force_positon_tensor_batch_tuple[j][i],axis=0)
         mean_stress_tensor=np.mean(mean_stress_tensor,axis=1)
         
         for l in range(0,3):
             strain_plot=np.linspace(0,strain_total,mean_stress_tensor[:,l].size)
             plt.plot(strain_plot,mean_stress_tensor[:,l],label="$\dot{\gamma}="+str(erate[i])+"$")
    plt.legend(bbox_to_anchor=(1,1))
    #plt.yscale('log')
    
    plt.show()


for j in range(K.size):
    for i in range(e_end[j]):
         mean_stress_tensor=np.mean(spring_force_positon_tensor_batch_tuple[j][i],axis=0)
         mean_stress_tensor=np.mean(mean_stress_tensor,axis=1)
         
         for l in range(3,6):
             strain_plot=np.linspace(0,strain_total,mean_stress_tensor[:,l].size)
             plt.plot(strain_plot,mean_stress_tensor[:,l],label="$\dot{\gamma}="+str(erate[i])+"$")
    plt.legend(bbox_to_anchor=(1,1))
    #plt.yscale('log')
    
    plt.show()




#%% look at internal stresses

def stress_tensor_averaging(e_end,
                            labels_stress,
                            trunc1,
                            trunc2,
                            spring_force_positon_tensor_tuple,j_):
    stress_tensor=np.zeros((e_end,6))
    stress_tensor_std=np.zeros((e_end,6))
    stress_tensor_reals=np.zeros((e_end,j_,6))
    stress_tensor_std_reals=np.zeros((e_end,j_,6))
    for l in range(6):
        for i in range(e_end):
            for j in range(j_):
                cutoff=int(np.round(trunc1*spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0]))
                aftercutoff=int(np.round(trunc2*spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0]))
                # print(spring_force_positon_tensor_tuple[i][j,:,:,l].shape)
                # print(cutoff)
                # print(aftercutoff)
                data=np.ravel(spring_force_positon_tensor_tuple[i][j,cutoff:aftercutoff,:,l])
              
                stress_tensor_reals[i,j,l]=np.mean(data)
                stress_tensor_std_reals[i,j,l]=np.std(data)
    stress_tensor=np.mean(stress_tensor_reals, axis=1)
    stress_tensor_std=np.mean(stress_tensor_std_reals, axis=1)
    return stress_tensor,stress_tensor_std




aftcut=1
cut=0.5 # or 0.4 
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
sns.set_palette('colorblind')
# sns.color_palette("mako", as_cmap=True)
# sns.color_palette("viridis")
#sns.set_palette('virdris')
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 14})
SIZE_DEFAULT = 14
SIZE_LARGE = 16
#plt.rcParams['text.usetex'] = True
# plt.rc("font", family="Roboto")  # controls default font
# plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False 

plt.rcParams.update({'font.size': 16})
# for j in range(thermal_damp_multiplier.size): 

for j in range(0,3): 
    for l in range(3):
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[0])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]), marker=marker[j])
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[j])+","+str(labels_stress[l]),linestyle=linestyle_tuple[j][1], marker=marker[j])
        plt.errorbar(erate[:e_end[j]],stress_tensor_tuple[j][:,l],
                     yerr=stress_tensor_std_tuple[j][:,l]/np.sqrt(n_plates*j_)
                     ,label="$K="+str(K[j])+","+str(labels_stress[l]),linestyle=linestyle_tuple[j][1],
                       marker=marker[j])
        

    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
       
        plt.xlabel("$\dot{\\varepsilon}$")
        plt.ylabel("$\sigma_{\\alpha \\alpha}$",rotation=0,labelpad=15)
        #plt.yticks(y_ticks_stress)
        #plt.ylim(0.9,1.3)
       
plt.tight_layout()
        #plt.xscale('log')

plt.legend(frameon=False)
plt.savefig(path_2_log_files+"/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.show()

#%%
for j in range(K.size): 
    for l in range(3,6):
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[0])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]), marker=marker[j])
       # plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[j])+","+str(labels_stress[l]), marker=marker[j])
        plt.errorbar(erate[:e_end[j]-1],stress_tensor_tuple[j][:-1,l],yerr=stress_tensor_std_tuple[j][:-1,l]/np.sqrt(n_plates*j_),label="$K="+str(K[j])+","+str(labels_stress[l]), marker=marker[j])
        
        

    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
       
       
        plt.xlabel("$\dot{\\varepsilon}$")
        plt.ylabel("$\sigma_{\\alpha \\alpha}$",rotation=0,labelpad=10)
        #plt.yticks(y_ticks_stress)
        #plt.ylim(0.9,1.3)
        #plt.tight_layout()
        #plt.xscale('log')
        #plt.savefig(path_2_log_files+"/plots/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.tight_layout()
        #plt.xscale('log')

plt.legend(frameon=False)
plt.savefig(path_2_log_files+"/stress_tensor_3_6_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.show()

#%%
def ext_visc_compute(stress_tensor,stress_tensor_std,i1,i2,i3,n_plates,e_end):
    extvisc=((2/3)*(stress_tensor[:,i1]- (1/2) *(stress_tensor[:,i2]+stress_tensor[:,i3]))/ erate[:e_end])
    # check this error calc 
    extvisc_error=np.sqrt(stress_tensor_std[:,i1]**2 +stress_tensor_std[:,i2]**2)/np.sqrt(j_*n_plates)

    return extvisc,extvisc_error

# def ext_visc_compute(stress_tensor,stress_tensor_std,i1,i2,n_plates,e_end):
#     extvisc=(stress_tensor[:,i1]- stress_tensor[:,i2])/erate[:e_end]
#     extvisc_error=np.sqrt((stress_tensor_std[:,i1]**2) +(stress_tensor_std[:,i2]**2))/np.sqrt(j_*n_plates)

#     return extvisc,extvisc_error


#for j in range(thermal_damp_multiplier.size):
for j in range(K.size):



    ext_visc_1,ext_visc_1_error=ext_visc_compute(stress_tensor_tuple[j],stress_tensor_std_tuple[j],0,1,2,n_plates,e_end[j])
    cutoff=1
    endcut=1
    #plt.errorbar(erate[cutoff:e_end[j]],ext_visc_1[cutoff:],yerr=ext_visc_1_error, label="$\eta_{1},K="+str(K[j])+"$", linestyle='none', marker=marker[j])
    #plt.errorbar(erate[cutoff:e_end[j]],ext_visc_1[cutoff:],yerr=ext_visc_1_error, label="$\eta_{1},tdamp="+str(thermal_damp_multiplier[j])+"$", marker=marker[j])
    plt.errorbar(erate[cutoff:e_end[j]-endcut],ext_visc_1[cutoff:-endcut],yerr=ext_visc_1_error[cutoff:-endcut], label="$\eta_{1},K="+str(K[j])+"$", marker=marker[j])
   # plt.plot(erate[cutoff:e_end[j]-endcut],ext_visc_1[cutoff:-endcut], label="$\eta_{1},K="+str(K[j])+"$", marker=marker[j])
    #plt.plot(erate[cutoff:e_end[j]],ext_visc_1_error,label="$\eta_{1},tdamp="+str(thermal_damp_multiplier[j])+"$")
    #plt.plot(erate[:e_end[j]],ext_visc_1, label="$\eta_{1},K="+str(K[j])+"$", linestyle='none', marker=marker[j])
   # plt.plot(erate[cutoff:e_end[j]],ext_visc_1[cutoff:], label="$tdamp="+str(thermal_damp_multiplier[j])+"$", marker=marker[j])
    #plt.plot(erate[cutoff:e_end[j]], k_50_ext_visc[cutoff:], label="$tdampk50="+str(thermal_damp_multiplier[j])+"$", marker=marker[j])
   
    plt.ylabel("$\eta_{ext}$", rotation=0, labelpad=20)
    plt.xlabel("$\dot{\\varepsilon}$")
#plt.yscale('log')
#plt.ylim(-2,7)
plt.legend(loc='upper right', bbox_to_anchor=(1.5,1))
plt.savefig(path_2_log_files+"/eta_vs_strain_rate.pdf",dpi=1200,bbox_inches='tight') 
plt.show()



# %%
plt.rcParams["figure.figsize"] = (24,12 )
for j in range(K.size):
    spherical_coords_tuple=()
    for i in range(e_end[j]):
        
        area_vector_ray=dirn_vector_batch_tuple[j][i]
        # detect all z coords less than 0 and multiply all 3 coords by -1
        area_vector_ray[area_vector_ray[:,:,:,2]<0]*=-1
        spherical_coords_array=np.zeros((j_,area_vector_ray.shape[1],n_plates*3,3))
        x=area_vector_ray[:,:,:,0]
        y=area_vector_ray[:,:,:,1]
        z=area_vector_ray[:,:,:,2]


        # radial coord
        spherical_coords_array[:,:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))
        #  theta coord 
        spherical_coords_array[:,:,:,1]=np.sign(y)*np.arccos(x/(np.sqrt((x**2)+(y**2))))
        # phi coord
        spherical_coords_array[:,:,:,2]=np.arccos(z/spherical_coords_array[:,:,:,0])

        spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_array,)

    # general skip array
    skip_array=np.array([[0,3,6,7],
                         [0,9,10,18],
                         [0,12,18,23]])
    
    skip_array=np.array([[0,2,4,6,8,10],
                         [0,4,8,11,13]
                         [0,4,8,12,15]
                         [0,4,8,14,17]
                         [0,6,10,15,20]])
    # skip array for turning points
    

    plt.subplot(1,2,1)
    for i in range(skip_array.shape[1]):
    #for i in range(e_end[j]):
        #for j in range(j_):


            i=skip_array[j,i]
            
            # sns.displot( data=np.ravel(spherical_coords_tuple[i][:,200000,:,1]),
            #             label ="$\dot{\gamma}="+str(erate[i])+"$", kde=True)
            # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,skip_array_2[j],:,1]),
            #             label="output_range:"+str(skip_array_2[j]))
            data=np.ravel(spherical_coords_tuple[i][:,500:,:,1])
            periodic_data=np.array([data-2*np.pi,data,data+2*np.pi])  

            sns.kdeplot( data=np.ravel(periodic_data),
                        label ="$\dot{\gamma}="+str(erate[i],)+"$")#bw_adjust=0.1
            
            # mean_data=np.mean(spherical_coords_tuple[0][:,-1,:,1],axis=0)      
            #plt.hist(np.ravel(spherical_coords_tuple[i][:,-100,:,1]))
            # bw adjust effects the degree of smoothing , <1 smoothes less
            plt.xlabel("$\Theta$")
            #plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
            plt.xlim(-np.pi,np.pi)
            plt.ylabel('Density')
           #plt.legend(bbox_to_anchor=[1.1, 0.45])
    #plt.show()
    plt.subplot(1,2,2)
    for i in range(skip_array.shape[1]):
    #for i in range(e_end[j]):
        #for j in range(j_):


            i=skip_array[j,i]

            # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,skip_array_2[j],:,2]),
            #              label="output_range:"+str(skip_array_2[j]))
            # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,-1,:,2]),
            #              label ="$\dot{\gamma}="+str(erate[i])+"$")
            data=np.ravel(spherical_coords_tuple[i][:,500:,:,2])
            periodic_data=np.array([data,np.pi-data])  
            sns.kdeplot( data=np.ravel(periodic_data),
                        label ="$\dot{\gamma}="+str(erate[i])+"$")
                    
            #plt.hist(np.ravel(spherical_coords_tuple[i][:,-1,:,2]))

    plt.xlabel("$\Phi$")
    #plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=[1.1, 0.45])
    plt.xlim(0,np.pi/2)
    plt.show()
#%% one at a time so turning points can be analysed 
pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
pi_phi_ticks=[ 0,np.pi/8,np.pi/4,3*np.pi/8, np.pi/2]
pi_phi_tick_labels=[ '0','π/8','π/4','3π/8', 'π/2']

def conv_cart_2_spherical_coords(cutoff,e_end,j,dirn_vector_batch_tuple,n_plates):
    spherical_coords_tuple=()
    for i in range(e_end[j]):
        
        area_vector_ray=dirn_vector_batch_tuple[j][i]
        # detect all z coords less than 0 and multiply all 3 coords by -1
        area_vector_ray[area_vector_ray[:,:,:,0]<0]*=-1
        spherical_coords_array=np.zeros((j_,1000-cutoff,n_plates*3,3))
        #spherical_coords_array=np.zeros((j_,n_plates*3,3))
        # x=np.mean(area_vector_ray[:,cutoff:,:,0],axis=1)
        # y=np.mean(area_vector_ray[:,cutoff:,:,1],axis=1)
        # z=np.mean(area_vector_ray[:,cutoff:,:,2],axis=1)
        x=area_vector_ray[:,cutoff:,:,0]
        y=area_vector_ray[:,cutoff:,:,1]
        z=area_vector_ray[:,cutoff:,:,2]


          
        # # radial coord  
        # spherical_coords_array[:,:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))
        #   #  theta coord 
        # spherical_coords_array[:,:,:,1]=np.sign(z)*np.arccos(y/(np.sqrt((z**2)+(y**2))))
        # #phi coord
        # spherical_coords_array[:,:,:,2]=np.arccos(x/spherical_coords_array[:,:,:,0])

        # radial coord  
        spherical_coords_array[:,:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))
          #  theta coord 
        spherical_coords_array[:,:,:,1]=np.sign(z)*np.arccos(y/(np.sqrt((z**2)+(y**2))))
        #phi coord
        spherical_coords_array[:,:,:,2]=np.arccos(x/spherical_coords_array[:,:,:,0])

        spherical_coords_mean=np.mean(spherical_coords_array,axis=0)

        #  # radial coord  
        # spherical_coords_array[:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))
        #   #  theta coord 
        # spherical_coords_array[:,:,1]=np.sign(z)*np.arccos(y/(np.sqrt((z**2)+(y**2))))
        # #phi coord
        # spherical_coords_array[:,:,2]=np.arccos(x/spherical_coords_array[:,:,0])
       
       
       
        # swapping nan for 0 
        #spherical_coords_array[:,:,:,0]=np.nan_to_num(spherical_coords_array[:,:,:,0], copy=False, nan=0.0,posinf=np.pi, neginf=0)
       
        # swapping nan for 0 
        #spherical_coords_array[:,:,:,1]=np.nan_to_num(spherical_coords_array[:,:,:,1], copy=False, nan=0.0,posinf=np.pi, neginf=0)

        
        # print("Number of elements",np.ravel(spherical_coords_array).size)
        # print("Number of NaN",np.count_nonzero(np.isnan(np.ravel(spherical_coords_array))))
        # print("Number of inf",np.count_nonzero(np.isinf(np.ravel(spherical_coords_array))))

        spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_mean,)

    return spherical_coords_tuple




def phi_dist_plot_find_minima(skip_array,spherical_coords_tuple):
        for i in range(skip_array.size):

                i=skip_array[i]

                data=np.ravel(spherical_coords_tuple[i][:,500:,:,2])
                periodic_data=np.array([data,np.pi-data])  
                ax=sns.kdeplot( data=np.ravel(periodic_data),
                            label ="$\dot{\gamma}="+str(erate[i])+"$")
                x = ax.lines[0].get_xdata() # Get the x data of the distribution
                y = ax.lines[0].get_ydata() # Get the y data of the distribution
                xy = [[x[j], y[j]] for j in range(len(x))]
                peak_coord = [xy[j] for j in find_peaks(-y)[0]]
                sorted_peak = sorted(peak_coord, key=lambda x: x[1])

                # sort peak based on its `y` coord
                sorted_peak.reverse() 
                print(sorted_peak)
                                
                                        
                #plt.hist(np.ravel(spherical_coords_tuple[i][:,-1,:,2]))

        plt.xlabel("$\Phi$")
        plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=[1.1, 0.45])
        plt.xlim(0,np.pi/2)
        plt.show()

# time phi plot

def phi_dist_plot_time(skip_array,spherical_coords_tuple):
        for i in range(skip_array.size):

                i=skip_array[i]

                for k in range(1000):

                    data=spherical_coords_tuple[i][:,k,:,2]
                    periodic_data=np.array([data,np.pi-data])  
                    sns.kdeplot( data=np.ravel(periodic_data),
                                label ="$\dot{\gamma}="+str(erate[i])+"$")
                            
                    #plt.hist(np.ravel(spherical_coords_tuple[i][:,-1,:,2]))

                plt.xlabel("$\Phi$")
                plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
                plt.ylabel('Density')
                #plt.legend(bbox_to_anchor=[1.1, 0.45])
                plt.xlim(0,np.pi/2)
                plt.show()

skip_array=np.array([0,3,6,7])
# phi_dist_plot_time(skip_array,spherical_coords_tuple)





#%% inspecting discontinuities  
j=1
skip_array=np.array([1,2,3,4,5,6,7])
spherical_coords_tuple=conv_cart_2_spherical_coords(cutoff,e_end,j,dirn_vector_batch_tuple,n_plates)
theta_dist_plot(skip_array,spherical_coords_tuple,j)
phi_dist_plot(skip_array,spherical_coords_tuple,j) 

j=2
skip_array=np.array([5,6,7])
spherical_coords_tuple=conv_cart_2_spherical_coords(cutoff,e_end,j,dirn_vector_batch_tuple,n_plates)
theta_dist_plot(skip_array,spherical_coords_tuple,j)
phi_dist_plot(skip_array,spherical_coords_tuple,j)

j=3
skip_array=np.array([14,15,16,17,18,19])
spherical_coords_tuple=conv_cart_2_spherical_coords(cutoff,e_end,3,dirn_vector_batch_tuple,n_plates)
theta_dist_plot(skip_array,spherical_coords_tuple,j)
phi_dist_plot(skip_array,spherical_coords_tuple,j)

 

#%% produce skipped extension distributions 

f, axs = plt.subplots(1, 4, figsize=(16, 6),sharey=True,sharex=True)
for j in range(K.size):
   # for i in range(e_end[j]):
    skip_array=np.array([[0,6,10,14,18,23],
                         [0,4,6,8,11,13],
                         [0,4,8,12,14,15],
                         [0,6,8,12,14,17],
                         [0,6,10,14,18,21],
                         [0,6,10,14,18,23]])
    #for i in range(1):
    for i in range(skip_array.shape[1]):
        i=skip_array[j,i]

        R_x=dirn_vector_batch_tuple[j][i][:,:,:,0]
        R_y=dirn_vector_batch_tuple[j][i][:,:,:,1]
        R_z=dirn_vector_batch_tuple[j][i][:,:,:,2]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$\dot{\gamma}="+str(sigfig.round(erate[i],sigfigs=3))+"$",ax=axs[j])
plt.legend(bbox_to_anchor=(1,1))
# plt.xlabel("$\Delta x$")
f.supxlabel("$\Delta x$")
f.tight_layout()
plt.savefig(path_2_log_files+"/extension_dist_skipped_K_"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight') 
plt.show()

#%%
mean_extension_tuple=()
mean_extension_error_tuple=()
for j in range(K.size):
    mean_extension_list=[]
    mean_extension_error_list=[]
    skip_array=np.array([[0,4,6,8,11,13],
                         [0,4,6,8,11,13],
                         [0,4,8,12,14,15],
                         [0,6,8,12,14,17],
                         [0,6,10,14,18,21],
                         [0,6,10,14,18,23]])
    # for i in range(skip_array.shape[1]):
    #     i=skip_array[j,i]
    for i in range(e_end[j]):
       

        R_x=dirn_vector_batch_tuple[j][i][:,:,:,0]
        R_y=dirn_vector_batch_tuple[j][i][:,:,:,1]
        R_z=dirn_vector_batch_tuple[j][i][:,:,:,2]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$\dot{\gamma}="+str(erate[i])+",K="+str(K[j])+"$")
        mean_extension=np.mean(np.ravel(magnitude_spring)-eq_spring_length)
        mean_extension_error=np.std(np.ravel(magnitude_spring)-eq_spring_length)
        mean_extension_list.append(mean_extension)
        mean_extension_error_list.append(mean_extension_error)

        plt.axvline(mean_extension,label ="$\\bar{\Delta x}="+str(sigfig.round(mean_extension,sigfigs=3))+\
                    ",\dot{\gamma}="+str(erate[i])+",K="+str(K[j])+"$")
    plt.xlabel("$\Delta x$")
  
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()
    mean_extension_tuple=mean_extension_tuple+(mean_extension_list,)
    mean_extension_error_tuple=mean_extension_error_tuple+(mean_extension_error_list,)

#%%
# this is more clear than the distributions in this case 
for j in range(K.size):
    
    #plt.plot(erate[:e_end[j]],mean_extension_tuple[j],marker=marker[j], label="$K="+str(K[j])+"$")
    plt.errorbar(erate[:e_end[j]],mean_extension_tuple[j],yerr=mean_extension_error_tuple[j]/np.sqrt(j_*n_plates),marker=marker[j], label="$K="+str(K[j])+"$")
    plt.xlabel("$\dot{\\varepsilon}$")
    plt.ylabel("$\\bar{\Delta x}$",rotation=0, labelpad=15)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(path_2_log_files+"/mean_extension_vs_strain_rate"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight') 
plt.show()
#%%
for j in range(K.size):
    skip_array=np.array([[0,6,10,14,18,23],
                         [0,4,6,8,11,13],
                         [0,4,8,12,14,15],
                         [0,6,8,12,14,17],
                         [0,6,10,14,18,21],
                         [0,6,10,14,18,23]])
    for i in range(skip_array.shape[1]):
        i=skip_array[j,i]

        R_x=dirn_vector_batch_tuple[j][i][:,:,:,0]
        R_y=dirn_vector_batch_tuple[j][i][:,:,:,1]
        R_z=dirn_vector_batch_tuple[j][i][:,:,:,2]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
        sns.kdeplot(np.ravel(R_x),
                    label ="$\dot{\gamma}="+str(erate[i])+",K="+str(K[j])+"$")
    plt.xlabel("$R_{x}$")
    plt.legend(bbox_to_anchor=(1,1))

    plt.show()

for j in range(K.size):
    skip_array=np.array([[0,6,10,14,18,23],
                         [0,4,6,8,11,13],
                         [0,4,8,12,14,15],
                         [0,6,8,12,14,17],
                         [0,6,10,14,18,21],
                         [0,6,10,14,18,23]])
    for i in range(skip_array.shape[1]):
        i=skip_array[j,i]

        R_x=dirn_vector_batch_tuple[j][i][:,:,:,0]
        R_y=dirn_vector_batch_tuple[j][i][:,:,:,1]
        R_z=dirn_vector_batch_tuple[j][i][:,:,:,2]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
        sns.kdeplot(np.ravel(R_x),
                    label ="$\dot{\gamma}="+str(erate[i])+",K="+str(K[j])+"$")
    plt.xlabel("$R_{y}$")
    plt.legend(bbox_to_anchor=(1,1))

    plt.show()


for j in range(K.size):
    skip_array=np.array([[0,6,10,14,18,23],
                         [0,4,6,8,11,13],
                         [0,4,8,12,14,15],
                         [0,6,8,12,14,17],
                         [0,6,10,14,18,21],
                         [0,6,10,14,18,23]])
    for i in range(skip_array.shape[1]):
        i=skip_array[j,i]

        R_x=dirn_vector_batch_tuple[j][i][:,:,:,0]
        R_y=dirn_vector_batch_tuple[j][i][:,:,:,1]
        R_z=dirn_vector_batch_tuple[j][i][:,:,:,2]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
        sns.kdeplot(np.ravel(R_x),
                    label ="$\dot{\gamma}="+str(erate[i])+",K="+str(K[j])+"$")
    plt.xlabel("$R_{z}$")
    plt.legend(bbox_to_anchor=(1,1))

    plt.show()
# %% extension distribution 
skip_array=[0,9,13,18]
plt.rcParams["figure.figsize"] = (12,6 )

#dist_xticks=([[-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-7.5,-5,-2.5,0,2.5,5]])


f, axs = plt.subplots(1, 4, figsize=(12, 6),sharey=True,sharex=True)
legfont=12
#for i in range(len(skip_array)):
adjust=1
for j in range(K.size):
        R_x=dirn_vector_batch_tuple[j][0][:,:,:,0]
        R_y=dirn_vector_batch_tuple[j][0][:,:,:,1]
        R_z=dirn_vector_batch_tuple[j][0][:,:,:,2]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)

        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$\dot{\gamma}="+str(erate[0])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[0])
        axs[0].legend(fontsize=legfont)
        # axs[0].xticks(dist_xticks[0][:])

        R_x=dirn_vector_batch_tuple[j][5][:,:,:,0]
        R_y=dirn_vector_batch_tuple[j][5][:,:,:,1]
        R_z=dirn_vector_batch_tuple[j][5][:,:,:,2]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
       
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$\dot{\gamma}="+str(erate[9])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[1])
        axs[1].legend(fontsize=legfont)
        # axs[1].xticks(dist_xticks[1][:])
        R_x=dirn_vector_batch_tuple[j][10][:,:,:,0]
        R_y=dirn_vector_batch_tuple[j][10][:,:,:,1]
        R_z=dirn_vector_batch_tuple[j][10][:,:,:,2]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$\dot{\gamma}="+str(erate[13])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[2])
        axs[2].legend(fontsize=legfont)
        # axs[2].xticks(dist_xticks[2][:])
        #plt.legend(fontsize=legfont) 
        R_x=dirn_vector_batch_tuple[j][13][:,:,:,0]
        R_y=dirn_vector_batch_tuple[j][13][:,:,:,1]
        R_z=dirn_vector_batch_tuple[j][13][:,:,:,2]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$\dot{\gamma}="+str(erate[18])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[3])
        axs[3].legend(fontsize=legfont)
        
        

#plt.yticks(extension_ticks)
#plt.xticks()


f.supxlabel("$\Delta x$")
f.tight_layout()

#plt.savefig(path_2_log_files+"/plots/deltax_dist_.pdf",dpi=1200,bbox_inches='tight')
   
plt.show()

# %%
def convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff):
    spherical_coords_tuple=()
     
    #for j in range(K.size):
    #for i in range(e_end[j]):
    for i in range(skip_array.shape[1]):
            k=skip_array[j,i]
            reshaped_coords=np.reshape(transformed_pos_batch_tuple[j][k],(j_,1000,100,6,3))
            ell_1=reshaped_coords[:,:,:,0] - reshaped_coords[:,:,:,1]
            ell_2=reshaped_coords[:,:,:,0] - reshaped_coords[:,:,:,2]

            mag_ell_1= np.sqrt(np.sum(ell_1**2,axis=3))
            mag_ell_2= np.sqrt(np.sum(ell_2**2,axis=3))
            
            ell_1[mag_ell_1>50]=float('NaN')
            ell_2[mag_ell_2>50]=float('NaN')



            area_vector=np.cross(ell_1,ell_2,axis=3)
            #print("percent of data with faulty rho vector",100*np.count_nonzero(area_vector[:,:,:,0]>50)/area_vector[:,:,:,0].size)
            
            # detect all z coords less than 0 and multiply all 3 coords by -1
            area_vector[area_vector[:,:,:,0]<0]*=-1
            spherical_coords_array=np.zeros((j_,area_vector.shape[1]-cutoff,n_plates,3))
            
            x=area_vector[:,cutoff:,:,0]
            y=area_vector[:,cutoff:,:,1]
            z=area_vector[:,cutoff:,:,2]

        
            
            # using x as inclination
             # radial coord  
            spherical_coords_array[:,:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))
              #  theta coord 
            spherical_coords_array[:,:,:,1]=np.sign(z)*np.arccos(y/(np.sqrt((z**2)+(y**2))))
            #phi coord
            spherical_coords_array[:,:,:,2]=np.arccos(x/spherical_coords_array[:,:,:,0])

            #spherical_coords_array[spherical_coords_array[:,:,:,0]>50]=float('NaN')

                
            #spherical_coords_mean=np.mean(spherical_coords_array,axis=0)
            
            print("percent of data with faulty rho vector",100*np.count_nonzero( spherical_coords_array[:,:,:,0]>50)/ spherical_coords_array[:,:,:,0].size)
            

            spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_array,)

    return spherical_coords_tuple

def convert_cart_2_spherical_y_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff):
    spherical_coords_tuple=()
     
    #for j in range(K.size):
    #for i in range(e_end[j]):
    for i in range(skip_array.shape[1]):
            k=skip_array[j,i]
            reshaped_coords=np.reshape(transformed_pos_batch_tuple[j][k],(j_,1000,100,6,3))
            ell_1=reshaped_coords[:,:,:,0] - reshaped_coords[:,:,:,1]
            ell_2=reshaped_coords[:,:,:,0] - reshaped_coords[:,:,:,2]

            mag_ell_1= np.sqrt(np.sum(ell_1**2,axis=3))
            mag_ell_2= np.sqrt(np.sum(ell_2**2,axis=3))
            
            ell_1[mag_ell_1>50]=float('NaN')
            ell_2[mag_ell_2>50]=float('NaN')



            area_vector=np.cross(ell_1,ell_2,axis=3)
            #print("percent of data with faulty rho vector",100*np.count_nonzero(area_vector[:,:,:,0]>50)/area_vector[:,:,:,0].size)
            
            # detect all z coords less than 0 and multiply all 3 coords by -1
            area_vector[area_vector[:,:,:,1]<0]*=-1
            spherical_coords_array=np.zeros((j_,area_vector.shape[1]-cutoff,n_plates,3))
            
            x=area_vector[:,cutoff:,:,0]
            y=area_vector[:,cutoff:,:,1]
            z=area_vector[:,cutoff:,:,2]

        
            
            # using x as inclination
             # radial coord  
            spherical_coords_array[:,:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))
              #  theta coord 
            spherical_coords_array[:,:,:,1]=np.sign(z)*np.arccos(x/(np.sqrt((z**2)+(x**2))))
            #phi coord
            spherical_coords_array[:,:,:,2]=np.arccos(y/spherical_coords_array[:,:,:,0])

            #spherical_coords_array[spherical_coords_array[:,:,:,0]>50]=float('NaN')

                
            #spherical_coords_mean=np.mean(spherical_coords_array,axis=0)
            print("percent of data with faulty rho vector",100*np.count_nonzero( spherical_coords_array[:,:,:,0]>50)/ spherical_coords_array[:,:,:,0].size)
            

            spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_array,)

    return spherical_coords_tuple

def convert_cart_2_spherical_z_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff):
    spherical_coords_tuple=()
     
    #for j in range(K.size):
    #for i in range(e_end[j]):
    for i in range(skip_array.shape[1]):
            k=skip_array[j,i]
            reshaped_coords=np.reshape(transformed_pos_batch_tuple[j][k],(j_,1000,100,6,3))
            ell_1=reshaped_coords[:,:,:,0] - reshaped_coords[:,:,:,1]
            ell_2=reshaped_coords[:,:,:,0] - reshaped_coords[:,:,:,2]

            mag_ell_1= np.sqrt(np.sum(ell_1**2,axis=3))
            mag_ell_2= np.sqrt(np.sum(ell_2**2,axis=3))
            
            ell_1[mag_ell_1>50]=float('NaN')
            ell_2[mag_ell_2>50]=float('NaN')



            area_vector=np.cross(ell_1,ell_2,axis=3)
            #print("percent of data with faulty rho vector",100*np.count_nonzero(area_vector[:,:,:,0]>50)/area_vector[:,:,:,0].size)
            
            # detect all z coords less than 0 and multiply all 3 coords by -1
            area_vector[area_vector[:,:,:,2]<0]*=-1
            spherical_coords_array=np.zeros((j_,area_vector.shape[1]-cutoff,n_plates,3))
            
            x=area_vector[:,cutoff:,:,0]
            y=area_vector[:,cutoff:,:,1]
            z=area_vector[:,cutoff:,:,2]

        
            #using z as inclination 
            # radial coord
            spherical_coords_array[:,:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))
            #  theta coord 
            spherical_coords_array[:,:,:,1]=np.sign(y)*np.arccos(x/(np.sqrt((x**2)+(y**2))))
            # phi coord
            spherical_coords_array[:,:,:,2]=np.arccos(z/spherical_coords_array[:,:,:,0])

            # # using x as inclination
            #  # radial coord  
            # spherical_coords_array[:,:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))
            #   #  theta coord 
            # spherical_coords_array[:,:,:,1]=np.sign(z)*np.arccos(y/(np.sqrt((z**2)+(y**2))))
            # #phi coord
            # spherical_coords_array[:,:,:,2]=np.arccos(x/spherical_coords_array[:,:,:,0])

            #spherical_coords_array[spherical_coords_array[:,:,:,0]>50]=float('NaN')

            
                 

                
            #spherical_coords_mean=np.mean(spherical_coords_array,axis=0)

            #spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_mean,)
            print("percent of data with faulty rho vector",100*np.count_nonzero( spherical_coords_array[:,:,:,0]>50)/ spherical_coords_array[:,:,:,0].size)
            
            spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_array,)

    return spherical_coords_tuple

def theta_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor):
    for i in range(len(spherical_coords_tuple)):
            
        # skip_tstep=np.array([0,30,60,90,120,150])
        # for k in range(skip_tstep.size):
            #k=skip_tstep[k]
    


           
            
            
            data=spherical_coords_tuple[i][:,:,:,1]
            #data=np.ravel(spherical_coords_tuple[i][:,:,1])
            periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
            adjust=adjfactor#*periodic_data.size**(-1/5)

            sns.kdeplot( data=periodic_data,
                        label ="$\dot{\gamma}="+str(erate[skip_array[j,i]])+"$",
                        bw_adjust=adjust,linestyle=linestyle_tuple[i][1])#bw_adjust=0.1
            
           
           
      
            # bw adjust effects the degree of smoothing , <1 smoothes less
    plt.axhline(1/(6*np.pi), label="Ref_uniform")
    plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")
    plt.xlabel("$\Theta$")
    plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
    plt.xlim(-np.pi,np.pi)
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=[1.1, 0.45])
    #plt.show()

def theta_dist_plot_timestep(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps):
    for i in range(len(spherical_coords_tuple)):
        
            for k in range(len(skip_steps)):
                l=skip_steps[k]
            
            
        # skip_tstep=np.array([0,30,60,90,120,150])
        # for k in range(skip_tstep.size):
            #k=skip_tstep[k]
    


           
            
            
                data=spherical_coords_tuple[i][:,l,:,1]
                #data=np.ravel(spherical_coords_tuple[i][:,:,1])
                periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
                adjust=adjfactor#*periodic_data.size**(-1/5)

                sns.kdeplot( data=periodic_data,
                            label ="$\dot{\gamma}="+str(erate[skip_array[j,i]])+",step= "+str(skip_steps[k])+"$",bw_adjust=adjust)#bw_adjust=0.1
                
            
               
        
                # bw adjust effects the degree of smoothing , <1 smoothes less
            plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")
            plt.axhline(1/(6*np.pi),label="Ref_uniform", linestyle='dashed')
            plt.xlabel("$\Theta$")
            plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
            plt.xlim(-np.pi,np.pi)
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=[1.1, 0.45])
            plt.show()



# Plot the probability density

     
def phi_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor):
        for i in range(len(spherical_coords_tuple)):
                 
            # skip_tstep=np.array([0,30,60,90,120,150])
            # for k in range(skip_tstep.size):
            #     k=skip_tstep[k]
    


                

                data=spherical_coords_tuple[i][:,:,:,2]
                #data=np.ravel(spherical_coords_tuple[i][:,:,2])
                periodic_data=np.ravel(np.array([data,np.pi-data]))
                adjust=adjfactor#*periodic_data.size**(-1/5)
                
                
                sns.kdeplot( data=np.ravel(periodic_data),
                            label ="$\dot{\gamma}="+str(erate[skip_array[j,i]])+"$",
                            bw_adjust=adjust,linestyle=linestyle_tuple[i][1])
                        
                #plt.hist(np.ravel(spherical_coords_tuple[i][:,-1,:,2]))
        t = np.linspace(0, np.pi/2, data.size)
        pdf = 0.5 * np.sin(t)
        plt.plot(t, pdf, label="Ref_sin")
        plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")
        plt.xlabel("$\Phi$")
        plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=[1.1, 0.45])
        plt.xlim(0,np.pi/2)
        #plt.show()

def phi_dist_plot_timstep(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps):
        for i in range(len(spherical_coords_tuple)):
            
            

            for k in range(len(skip_steps)):
                l=skip_steps[k]
            
         

                data=spherical_coords_tuple[i][:,l,:,2]
                #data=np.ravel(spherical_coords_tuple[i][:,:,2])
                periodic_data=np.ravel(np.array([data,np.pi-data]))
                adjust=adjfactor#*periodic_data.size**(-1/5)
               
                
                sns.kdeplot( data=np.ravel(periodic_data),
                            label ="$\dot{\gamma}="+str(erate[skip_array[j,i]])+",step= "+str(skip_steps[k])+"$",bw_adjust=adjust)
                        
                #plt.hist(np.ravel(spherical_coords_tuple[i][:,-1,:,2]))
            t = np.linspace(0, np.pi/2, data.size)
            pdf = 0.5 * np.sin(t)
            plt.plot(t, pdf, label="Ref_sin", linestyle='dashed')
            plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")
            plt.xlabel("$\Phi$")
            plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=[1.1, 0.45])
            plt.xlim(0,np.pi/2)
            plt.show()

def phi_theta_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps):
        plt.subplots(1, 2, figsize=(15, 6),sharey=False,sharex=False)
        for i in range(len(spherical_coords_tuple)):
            
            
            
            plt.subplot(1, 2, 1)
            
            
         

            data=spherical_coords_tuple[i][:,:,:,2]
            #data=np.ravel(spherical_coords_tuple[i][:,:,2])
            periodic_data=np.ravel(np.array([data,np.pi-data]))
            adjust=adjfactor#*periodic_data.size**(-1/5)
            
            
            sns.kdeplot( data=np.ravel(periodic_data),
                        label ="$\dot{\gamma}="+str(erate[skip_array[j,i]])+"$",bw_adjust=adjust)
                    
                #plt.hist(np.ravel(spherical_coords_tuple[i][:,-1,:,2]))
            t = np.linspace(0, np.pi/2, data.size)
            pdf = 0.5 * np.sin(t)
            plt.plot(t, pdf, label="Ref_sin", linestyle='dashed')
            plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")
            plt.xlabel("$\Phi$")
            plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
            plt.ylabel('Density')
            #plt.legend(bbox_to_anchor=[1.1, 0.45])
            plt.xlim(0,np.pi/2)
            #plt.show()

            plt.subplot(1, 2, 2)
            
        # skip_tstep=np.array([0,30,60,90,120,150])
        # for k in range(skip_tstep.size):
            #k=skip_tstep[k]
    


           
            
            
            data=spherical_coords_tuple[i][:,:,:,1]
            #data=np.ravel(spherical_coords_tuple[i][:,:,1])
            periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
            adjust=adjfactor#*periodic_data.size**(-1/5)

            sns.kdeplot( data=periodic_data,
                        label ="$\dot{\gamma}="+str(erate[skip_array[j,i]])+"$",bw_adjust=adjust)#bw_adjust=0.1
            
        
            
    
                # bw adjust effects the degree of smoothing , <1 smoothes less
            plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")
            plt.axhline(1/(6*np.pi),label="Ref_uniform", linestyle='dashed')
            plt.xlabel("$\Theta$")
            plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
            plt.xlim(-np.pi,np.pi)
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=[1.4, 0.35])
        plt.show()

def phi_theta_dist_plot_timstep(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps):
        for i in range(len(spherical_coords_tuple)):
            plt.subplots(1, 2, figsize=(15, 6),sharey=False,sharex=False)
            
            
            plt.subplot(1, 2, 1)
            for k in range(len(skip_steps)):
                l=skip_steps[k]
            
         

                data=spherical_coords_tuple[i][:,l,:,2]
                #data=np.ravel(spherical_coords_tuple[i][:,:,2])
                periodic_data=np.ravel(np.array([data,np.pi-data]))
                adjust=adjfactor#*periodic_data.size**(-1/5)
               
                
                sns.kdeplot( data=np.ravel(periodic_data),
                            label ="$\dot{\gamma}="+str(erate[skip_array[j,i]])+",step= "+str(skip_steps[k])+"$",bw_adjust=adjust)
                        
                #plt.hist(np.ravel(spherical_coords_tuple[i][:,-1,:,2]))
            t = np.linspace(0, np.pi/2, data.size)
            pdf = 0.5 * np.sin(t)
            plt.plot(t, pdf, label="Ref_sin", linestyle='dashed')
            plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")
            plt.xlabel("$\Phi$")
            plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
            plt.ylabel('Density')
            #plt.legend(bbox_to_anchor=[1.1, 0.45])
            plt.xlim(0,np.pi/2)
            #plt.show()

            plt.subplot(1, 2, 2)
            for k in range(len(skip_steps)):
                l=skip_steps[k]
            
            
        # skip_tstep=np.array([0,30,60,90,120,150])
        # for k in range(skip_tstep.size):
            #k=skip_tstep[k]
    


           
            
            
                data=spherical_coords_tuple[i][:,l,:,1]
                #data=np.ravel(spherical_coords_tuple[i][:,:,1])
                periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
                adjust=adjfactor#*periodic_data.size**(-1/5)

                sns.kdeplot( data=periodic_data,
                            label ="$\dot{\gamma}="+str(erate[skip_array[j,i]])+",step= "+str(skip_steps[k])+"$",bw_adjust=adjust)#bw_adjust=0.1
                
            
               
        
                # bw adjust effects the degree of smoothing , <1 smoothes less
            plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")
            plt.axhline(1/(6*np.pi),label="Ref_uniform", linestyle='dashed')
            plt.xlabel("$\Theta$")
            plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
            plt.xlim(-np.pi,np.pi)
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=[1.2, 0.45])
            plt.show()

#%% phi, theta dists with phi against theta 
pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
pi_phi_ticks=[ 0,np.pi/8,np.pi/4,3*np.pi/8, np.pi/2]
pi_phi_tick_labels=[ '0','π/8','π/4','3π/8', 'π/2']
skip_array=np.array([[0,2,4,6,8,9],
                         [0,4,6,8,11,13],
                         [0,4,8,12,14,15],
                         [0,6,8,12,14,17],
                         [0,6,10,14,18,21],
                         [0,6,10,14,18,23]])
cutoff=0
skip_steps=[0,100,300,600,900,950,995]
for j in range(K.size):
    f, axs = plt.subplots(skip_array.shape[1], 3, figsize=(30, 20),sharey=False,sharex=False)
    spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)
        

    for i in range(skip_array.shape[1]):
        #for i in range(e_end[j]):
        for k in range(len(skip_steps)):
            l=skip_steps[k]
            
        
            #phi 
            data=np.ravel(spherical_coords_tuple[i][:,l,:,2])
            periodic_data=np.array([data,np.pi-data])  
            sns.kdeplot( data=np.ravel(periodic_data),
                        label ="$\dot{\gamma}="+str(erate[skip_array[j,i]])+",step= "+str(skip_steps[k])+"$",ax=axs[i,0])
            axs[i,0].set_xlim(0,np.pi/2)
            axs[i,0].set_xlabel("$\phi$")
            axs[i,0].set_xticks(pi_phi_ticks,pi_phi_tick_labels)
            axs[i,0].legend()
            # theta 

            data=np.ravel(spherical_coords_tuple[i][:,l,:,1])
            periodic_data=np.array([data-2*np.pi,data,data+2*np.pi])  

            sns.kdeplot( data=np.ravel(periodic_data),
                        label ="$\dot{\gamma}="+str(erate[skip_array[j,i]])+",step= "+str(skip_steps[k])+"$",ax=axs[i,1])#bw_adjust=0.1
            
            axs[i,1].set_xlim(-np.pi,np.pi)
            axs[i,1].set_xlabel("$\\theta$")
            axs[i,1].set_xticks(pi_theta_ticks,pi_theta_tick_labels)
            axs[i,1].legend()
            
        phi=np.ravel(spherical_coords_tuple[i][:,:,:,2])
        theta=np.ravel(spherical_coords_tuple[i][:,:,:,1])
        axs[i,2].scatter(phi,theta,label ="$\dot{\\varepsilon}="+str(sigfig.round(erate[i],sigfigs=3))+"$")
        axs[i,2].set_xlabel("$\phi$")
        axs[i,2].set_yticks(pi_theta_ticks,pi_theta_tick_labels)
        axs[i,2].set_ylabel("$\\theta$")
        #axs[i,2].set_xticks(pi_phi_ticks,pi_phi_tick_labels)
        #axs[i,2].set_xlim(0,0.35)
        axs[i,2].legend()



    

        #f.supxlabel("$\Delta x$")
    
    f.tight_layout()
    f.suptitle("$K="+str(K[j])+"$",x=0.5,y=1.05,fontsize=25)

    #plt.savefig(path_2_log_files+"/plots/deltax_dist_.pdf",dpi=1200,bbox_inches='tight')

    plt.show()  



# %%
# %%
pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
pi_phi_ticks=[ 0,np.pi/8,np.pi/4,3*np.pi/8, np.pi/2]
pi_phi_tick_labels=[ '0','π/8','π/4','3π/8', 'π/2']
skip_array=np.array([[0,4,6,8,11,13],
                         [0,4,6,8,11,13],
                         [0,4,8,12,14,15],
                         [0,6,8,12,14,17],
                         [0,6,10,14,18,21],
                         [0,6,10,14,18,23]])
# skip_array=np.array([[0,2],
#                          [0,4],
#                          [0,4],
#                          [0,6],
#                          [0,6],
#                          [0,6]])


cutoff=500
skip_steps=[0,100,300,600,900,950,980,995]
skip_steps=[0,800,900,950,980,995]
skip_steps=[0,100,200,300,400,499]
# skip_steps=[900, 905, 910, 915, 920, 925, 930, 935, 940, 945, 950, 955, 960,
#        965, 970, 975, 980, 985, 990]
# skip_steps=[700,800,900,950,999]
adjfactor=1#0.1

j=0
spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)
sns.color_palette("magma")
plt.ylim(0,2.2)

phi_dist_plot_timstep(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)
plt.title("Phi inclined to x axis selection of steps")



plt.title("Theta on yz plane selection of steps")
theta_dist_plot_timestep(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)


# j=1
# spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)

# phi_dist_plot_timstep(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)
# plt.title("Phi inclined to x axis selection of steps")


# plt.title("Theta on yz plane selection of steps")
# theta_dist_plot_timestep(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)

# j=2

# spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)

# phi_dist_plot_timstep(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)
# plt.title("Phi inclined to x axis selection of steps")


# plt.title("Theta on zy plane selection of steps")
# theta_dist_plot_timestep(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)

# j=3

# spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)

# phi_dist_plot_timstep(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)
# plt.title("Phi inclined to x axis selection of steps")


# plt.title("Theta on yz plane selection of steps")
# theta_dist_plot_timestep(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)

# %%
# %% looking at batches
linestyle_tuple = [
    
     ('dotted',                (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dotted',        (0, (1, 1))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]



cutoff=500
adjfactor=0.5 #0.25#0.05
j=0
spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)
# phi_theta_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)
phi_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor)
plt.show()
theta_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor)
plt.show()
j=1
spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)
# phi_theta_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)
phi_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor)
plt.show()
theta_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor)
plt.show()
# j=2
# spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)
# # phi_theta_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)
# phi_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor)
# plt.show()
# theta_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor)
# plt.show()
# j=2
# spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)
# phi_theta_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)
# j=3
# spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)
# phi_theta_dist_plot(skip_array,spherical_coords_tuple,j,adjfactor,skip_steps)

# %%
#%% different style plot of theta
cutoff=500
#theta
linestyle_tuple = ['dashdot', 
  'dotted', 
 'dashed', '-', 
 'None', ' ', '', 'solid', 
 'dashed', 'dashdot', '--']
pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
pi_phi_ticks=[ 0,np.pi/8,np.pi/4,3*np.pi/8, np.pi/2]
pi_phi_tick_labels=[ '0','π/8','π/4','3π/8', 'π/2']
skip_array=np.array([[0,5,10,12,11,13],
                         [0,5,10,12,11,1],
                         [0,5,10,12,14,15],
                         [0,6,8,12,14,17],
                         [0,6,10,14,18,21],
                         [0,6,10,14,18,23]])
f, axs = plt.subplots(1, 4, figsize=(15, 6),sharey=True,sharex=True)
adjust_factor=1
for j in range(K.size):
    spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)
   
    i=0
    data=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[0],bw_adjust=adjust_factor)
    
    axs[0].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
    #axs[0].axhline(1/(6*np.pi),label="Ref_uniform", linestyle='dashed')

    i=1
    data=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[1],bw_adjust=adjust_factor)
  
    axs[1].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
    # axs[1].axhline(1/(6*np.pi),label="Ref_uniform", linestyle='dashed')

    i=2
    data=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[2],bw_adjust=adjust_factor)
    
    axs[2].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
    #axs[2].axhline(1/(6*np.pi),label="Ref_uniform", linestyle='dashed')

    i=3
    data=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[3],bw_adjust=adjust_factor)
   
    axs[3].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
   # axs[3].axhline(1/(6*np.pi),label="Ref_uniform", linestyle='dashed')


f.supxlabel("$\Theta$")
axs[0].axhline(1/(6*np.pi),label="Ref_uniform", linestyle='dashed')
axs[1].axhline(1/(6*np.pi),label="Ref_uniform", linestyle='dashed')
axs[2].axhline(1/(6*np.pi),label="Ref_uniform", linestyle='dashed')
axs[3].axhline(1/(6*np.pi),label="Ref_uniform", linestyle='dashed')

plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
plt.legend(bbox_to_anchor=(1,0.55),frameon=False)


#plt.yticks(phi_y_ticks)

plt.ylabel('Density')
plt.xlim(-np.pi,np.pi)
#plt.xlim(0,np.pi)
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/theta_dist_.pdf",dpi=1200,bbox_inches='tight')
plt.show()
# %%
#%% different style plot of phi using kdeplot 
adjust_factor=5
#phi 
f, axs = plt.subplots(1, 4, figsize=(15, 6),sharex=True)


for j in range(K.size):
    spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)
    i=0
    data=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    print(np.count_nonzero(np.isnan(data)))
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    
    sns.kdeplot( data=periodic_data,
                      label ="K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[0],bw_adjust=adjust_factor)
    
    axs[0].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
    t = np.linspace(0, np.pi/2, data.size)
    pdf = 0.5 * np.sin(t)
    #axs[0].plot(t, pdf, label="Ref_sin", linestyle='dashed')

    i=1
    data=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    print(np.count_nonzero(np.isnan(data)))
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[i]])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[1],bw_adjust=adjust_factor)
    axs[1].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
    t = np.linspace(0, np.pi/2, data.size)
    pdf = 0.5 * np.sin(t)
    #axs[1].plot(t, pdf, label="Ref_sin", linestyle='dashed')

    i=2
    data=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    print(np.count_nonzero(np.isnan(data)))
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[2],bw_adjust=adjust_factor)
    axs[2].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
    t = np.linspace(0, np.pi/2, data.size)
    pdf = 0.5 * np.sin(t)
    #axs[2].plot(t, pdf, label="Ref_sin", linestyle='dashed')

    i=3
    data=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    print(np.count_nonzero(np.isnan(data)))
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[3],bw_adjust=adjust_factor)
    axs[3].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
    t = np.linspace(0, np.pi/2, data.size)
    pdf = 0.5 * np.sin(t)
    #axs[3].plot(t, pdf, label="Ref_sin", linestyle='dashed')

f.supxlabel("$\Phi$")
axs[0].plot(t, pdf, label="Ref_sin", linestyle='dashed')
axs[1].plot(t, pdf, label="Ref_sin", linestyle='dashed')
axs[2].plot(t, pdf, label="Ref_sin", linestyle='dashed')
axs[3].plot(t, pdf, label="Ref_sin", linestyle='dashed')

plt.xticks(pi_phi_ticks,pi_phi_tick_labels)

#plt.yticks(phi_y_ticks)
plt.ylabel('Density')
plt.legend(bbox_to_anchor=(1,0.5),frameon=False)
plt.xlim(0,np.pi/2)
#plt.xlim(0,np.pi)
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/phi_dist_.pdf",dpi=1200,bbox_inches='tight')
plt.show()

#%% different style plot of rho using kdeplot 
adjust_factor=100
#phi 
f, axs = plt.subplots(1, 4, figsize=(15, 6),sharex=True)


for j in range(K.size):
    spherical_coords_tuple=convert_cart_2_spherical_x_incline(j_,j,skip_array,transformed_pos_batch_tuple,n_plates,cutoff)
    i=0
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
    periodic_data=np.ravel(np.array([data]))
    sns.kdeplot( data=periodic_data,
                      label ="K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[0],bw_adjust=adjust_factor)
    
    axs[0].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
   
    i=1
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
    periodic_data=np.ravel(np.array([data]))
    sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[i]])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[1],bw_adjust=adjust_factor)
    axs[1].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
   

    i=2
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
    periodic_data=np.ravel(np.array([data]))
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[2],bw_adjust=adjust_factor)
    axs[2].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
   

    i=3
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
    periodic_data=np.ravel(np.array([data]))
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[3],bw_adjust=adjust_factor)
    axs[3].set_title("$\dot{\\varepsilon}="+str(sigfig.round(erate[skip_array[j,i]],sigfigs=3))+"$")
    

f.supxlabel("$\\rho$")

#plt.xticks(pi_phi_ticks,pi_phi_tick_labels)

#plt.yticks(phi_y_ticks)
plt.ylabel('Density')
plt.legend(bbox_to_anchor=(1,0.5),frameon=False)
#plt.xlim(0,np.pi/2)
#plt.xlim(0,np.pi)
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/phi_dist_.pdf",dpi=1200,bbox_inches='tight')
plt.show()
# %%
