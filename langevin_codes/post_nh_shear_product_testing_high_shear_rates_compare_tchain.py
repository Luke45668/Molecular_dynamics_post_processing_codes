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
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats

from datetime import datetime
import mmap
import h5py as h5
from scipy.optimize import curve_fit

marker=['x','+','^',"1","X","d","*","P","v"]
# path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
# os.chdir(path_2_post_proc_module)
import seaborn as sns
# from log2numpy import *
# from dump2numpy import *
import glob 
#from MPCD_codes.post_MPCD_MP_processing_module import *
import pickle as pck
from post_langevin_module import *

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

#note: currently we have one missing shear rate for k=30,60 so need to run them again with erate 0.02 to get the full picture 
# I have k=20 downloaded, K=120 is running , need to produce run files for k=30,60

damp=np.array([ 0.035, 0.035,0.035,0.035 ])
tchain=["60_30"]
K=np.array([  30,  60  ,90
            ])
K=np.array([ 120
            ])
K=np.array([ 60])


erate=np.linspace(0.005,1,24)
no_timesteps=np.array([1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000 ])

timestep_multiplier=np.flip(np.array([9.85877894e-06, 1.03045633e-05, 1.07925709e-05, 1.13290990e-05,
        1.19217621e-05, 1.25798566e-05, 1.33148512e-05, 1.41410611e-05,
        1.50765901e-05, 1.61446718e-05, 1.73756257e-05, 1.88097815e-05,
        2.05019815e-05, 2.25287546e-05, 2.50002112e-05, 2.80807326e-05,
        3.20271067e-05, 3.72640781e-05, 4.45485099e-05, 5.53728731e-05,
        7.31457792e-05, 1.07720625e-04, 2.04281005e-04, 1.97175579e-03]))


thermo_vars="         KinEng      c_spring_pe       PotEng         Press         c_myTemp        c_bias         TotEng    "

erate=np.array([1.34      , 1.34555556, 1.35111111, 1.35666667, 1.36222222,
       1.36777778, 1.37333333, 1.37888889, 1.38444444, 1.39,1.395     , 1.41222222, 1.42944444, 1.44666667, 1.46388889,
       1.48111111, 1.49833333, 1.51555556, 1.53277778, 1.55,1.6       , 1.62222222, 1.64444444, 1.66666667, 1.68888889,
       1.71111111, 1.73333333, 1.75555556, 1.77777778, 1.8])

e_in=0
#e_end=erate.size
n_plates=100

strain_total=250


path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/shear_runs/strain_250_6_reals_erate_over_1.34_comparison/"




j_=6
sim_fluid=30.315227255599112

eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
damp_ratio=mass_pol/damp

def one_term_poly(x,a,b):
     return b*(x**a)

def quadratic_no_constant(x,a,b):
     return a*x + b*(x**2)
     
def quadratic(x,a):
     return a*(x**2)
#%% load in tuples
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
log_file_real_batch_tuple=()
area_vector_spherical_batch_tuple=()
interest_vectors_batch_tuple=()
pos_batch_tuple=()
vel_batch_tuple=()
e_end=[]

# loading all data into one 
for i in range(len(tchain)):

    label='tchain_'+str(tchain[i])+'_K_'+str(K[i])+'_'

   
    spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(batch_load_tuples(label,
                                                            "spring_force_positon_tensor_tuple.pickle"),)
    
    print(len( spring_force_positon_tensor_batch_tuple[i]))
    e_end.append(len(spring_force_positon_tensor_batch_tuple[i]))

    pos_batch_tuple=pos_batch_tuple+(batch_load_tuples(label,"p_positions_tuple.pickle"),)

    vel_batch_tuple=vel_batch_tuple+(batch_load_tuples(label,"p_velocities_tuple.pickle"),)


    log_file_batch_tuple=log_file_batch_tuple+(batch_load_tuples(label,
                                                            "log_file_tuple.pickle"),)
    
    log_file_real_batch_tuple=log_file_real_batch_tuple+(batch_load_tuples(label,
                                                            "log_file_real_tuple.pickle"),)
    # print(len(log_file_batch_tuple[i]))
    area_vector_spherical_batch_tuple=area_vector_spherical_batch_tuple+(batch_load_tuples(label,"area_vector_tuple.pickle"),)
    
   
   # e_end.append(len(spring_force_positon_tensor_batch_tuple[i]))


                                                                                      

    


     

#%% strain points for temperatuee data 
strainplot_tuple=()

for i in range(erate.size):
    
    strain_plotting_points= np.linspace(0,strain_total,1002)
    #strain_plotting_points= np.linspace(150,strain_total,501)

    strainplot_tuple=strainplot_tuple+(strain_plotting_points,)  
    print(strainplot_tuple[i].size)

# def strain_plotting_points(total_strain,points_per_iv):
#      #points_per_iv= number of points for the variable measured against strain 
#      strain_unit=total_strain/points_per_iv
#      strain_plotting_points=np.arange(0,total_strain,strain_unit)
#      return  strain_plotting_points

#%% energy vs time plot
plt.rcParams["figure.figsize"] = (26,10 )
# pe vs time 

for j in range(K.size):
    for i in range(e_end[j]):
        column=2
        plt.subplot(1,3,1)
        #plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
       #strainplot= np.linspace(0,strain_total,log_file_batch_tuple[j][i].shape[0])
        strainplot= np.linspace(0,strain_total,1002)
        plt.plot(strainplot,log_file_batch_tuple[j][i][:,column])
        #plt.yscale('log')
        plt.xlabel("$\gamma$")
        plt.ylabel("$E_{p}$")
        plt.ylim(0,3)
        plt.title("$N_{c}=$"+str(tchain[j]))
        #plt.show()
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        column=6

        plt.subplot(1,3,2)
        plt.plot(strainplot[50:],log_file_batch_tuple[j][i][50:,column])
       # plt.yscale('log')
        plt.xlabel("$\gamma$")
        plt.ylabel("$T$")
        plt.title("$N_{c}=$"+str(tchain[j]))
 
        column=7


        plt.subplot(1,3,3)
        plt.plot(strainplot[50:],log_file_batch_tuple[j][i][50:,column])
       # plt.yscale('log')
        plt.xlabel("$\gamma$")
        plt.ylabel("$E_{t}$")
        plt.title("$N_{c}=$"+str(tchain[j]))
    plt.show()
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]


#%% inspecting energies realisation by realisation 
plt.rcParams["figure.figsize"] = (26,10 )
j=0
for i in range(e_end[j]):
    for j in range(K.size):
        for k in range(j_):
    
            column=2
            plt.subplot(1,3,1)
            #plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
        #strainplot= np.linspace(0,strain_total,log_file_batch_tuple[j][i].shape[0])
            strainplot= np.linspace(0,strain_total,1002)
            plt.plot(strainplot,log_file_real_batch_tuple[j][i][k,:,column])
            #plt.yscale('log')
            plt.xlabel("$\gamma$")
            plt.ylabel("$E_{p}$")
            plt.ylim(0,3)
            plt.title("$N_{c}=$"+str(tchain[j]))
            #plt.show()
            # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
            column=6

            plt.subplot(1,3,2)
            plt.plot(strainplot[50:],log_file_real_batch_tuple[j][i][k,50:,column])
        # plt.yscale('log')
            plt.xlabel("$\gamma$")
            plt.ylabel("$T$")
            plt.title("$N_{c}=$"+str(tchain[j]))
    
            column=7


            plt.subplot(1,3,3)
            plt.plot(strainplot[50:],log_file_real_batch_tuple[j][i][k,50:,column])
        # plt.yscale('log')
            plt.xlabel("$\gamma$")
            plt.ylabel("$E_{t}$")
            plt.title("$N_{c}=$"+str(tchain[j]))
        plt.suptitle("$\dot{\gamma}="+str(erate[i])+"$")
        plt.show()
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]       



#%%
folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=6
final_temp=np.zeros((erate.size))
mean_temp_array=np.zeros((K.size,erate.size))
for j in range(K.size):
    for i in range(e_end[j]):
        
        # plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
        mean_temp_array[j,i]=np.mean(log_file_batch_tuple[j][i][1000:,column])
      
        #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
    #     plt.ylabel("$T$", rotation=0)
    #     plt.xlabel("$\gamma$")
    

    # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
    #     plt.show()

#

marker=['x','+','^',"1","X","d","*","P","v"]

for j in range(K.size):
    plt.scatter(erate,mean_temp_array[j,:],label="$K="+str(K[j])+"$" ,marker=marker[j])
    plt.ylabel("$T$", rotation=0)
    plt.xlabel('$\dot{\gamma}$')
    plt.xscale('log')
   # plt.yscale('log')
plt.axhline(1,label="$T_{0}=1$")
plt.legend()
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/temp_vs_erate.pdf",dpi=1200,bbox_inches='tight')


plt.show()

#%



#%%
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



 # only for low shear rate regime 
aftcut=1
cut=0.6# or 0.4 
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



    
    
#%% time series plots of stress
j=0
for i in range(e_end[j]):
    for j in range(len(tchain)):   
    
    
    
        mean_stress_tensor=np.mean(spring_force_positon_tensor_batch_tuple[j][i],axis=0)
        mean_stress_tensor=np.mean(mean_stress_tensor,axis=1)
        std_dev_stress_tensor=np.std(spring_force_positon_tensor_batch_tuple[j][i],axis=0)
        
 
        for l in range(3,4):
                strain_plot=np.linspace(0,strain_total,mean_stress_tensor[:,l].size)
                grad_cutoff=int(np.round(cut*mean_stress_tensor[:,l].size))
                SS_grad=np.mean(np.gradient(mean_stress_tensor[grad_cutoff:,l],axis=0))
                SS_grad=np.around(SS_grad,5)

                plt.plot(strain_plot,mean_stress_tensor[:,l],label="tchain="+tchain[j]+",$SSgrad="+str(SS_grad)+","+labels_stress[l])

    plt.legend(bbox_to_anchor=(1,1))
    plt.title("$\dot{\\gamma}="+str(erate[i])+"="+str(K[j])+"$")
            #plt.yscale('log')
    plt.ylim(-5,40)
        #plt.savefig(path_2_log_files+"/"+str(K[j])+"_stress_time_series_"+str(erate[i])+".pdf",dpi=1200,format="pdf",bbox_inches='tight') 
            
    plt.show()

#%% time series plots of stress each realisation 
for i in range(e_end[j]):
#for i in range(13,14):
    for j in range(len(tchain)):   
    
    
    
        mean_stress_tensor=np.mean(spring_force_positon_tensor_batch_tuple[j][i],axis=2)
        
        std_dev_stress_tensor=np.std(spring_force_positon_tensor_batch_tuple[j][i],axis=2)

        for l in range(3,4):

            for k in range(j_):
        
 
            
                    strain_plot=np.linspace(0,strain_total,mean_stress_tensor[k,:,l].size)
                    grad_cutoff=int(np.round(cut*mean_stress_tensor[k,:,l].size))
                    SS_grad=np.mean(np.gradient(mean_stress_tensor[k,grad_cutoff:,l],axis=0))
                    SS_grad=np.around(SS_grad,5)

                    plt.plot(strain_plot,mean_stress_tensor[k,:,l],label="tchain="+tchain[j]+",real="+str(k)+",$SSgrad="+str(SS_grad)+","+labels_stress[l])

        plt.legend(bbox_to_anchor=(1,1))
        plt.title("$\dot{\\gamma}="+str(erate[i])+",K="+str(K[j])+"$")
        plt.xlabel("$\gamma$")
        plt.ylabel("$\sigma_{\\alpha\\beta}$")
                #plt.yscale('log')
        plt.ylim(-5,40)
            #plt.savefig(path_2_log_files+"/"+str(K[j])+"_stress_time_series_"+str(erate[i])+".pdf",dpi=1200,format="pdf",bbox_inches='tight') 
                
        plt.show()
# for j in range(K.size):
#     for i in range(e_end[j]):
#          mean_stress_tensor=np.mean(spring_force_positon_tensor_batch_tuple[j][i],axis=0)
#          mean_stress_tensor=np.mean(mean_stress_tensor,axis=1)
         
#          for l in range(3,6):
#              strain_plot=np.linspace(0,strain_total,mean_stress_tensor[:,l].size)
#              plt.plot(strain_plot,mean_stress_tensor[:,l],label="$\dot{\gamma}="+str(erate[i])+"$")
#     plt.legend(bbox_to_anchor=(1,1))
#     #plt.yscale('log')
    
#     plt.show()
#%% checking gradients for success
pe_column=2
cut=0.8
SS_grad_array=np.zeros((2,22,7,6))

for j in range(len(tchain)): 
    
    for i in range(e_end[j]):
      
        # obtaining gradient of last cut% of signal 
        mean_stress_tensor=np.mean(spring_force_positon_tensor_batch_tuple[j][i],axis=2)
        grad_cutoff=int(np.round(cut*mean_stress_tensor[0,:,0].size))
        SS_grad_stress=np.mean(np.gradient(mean_stress_tensor[:,grad_cutoff:,:],axis=1),axis=1)
        SS_grad_stress=np.around(SS_grad_stress,6)
        SS_grad_pe=np.mean(np.gradient(log_file_real_batch_tuple[j][i][:,grad_cutoff:,pe_column],axis=1),axis=1)
        SS_grad_array[j,i,0,:]=SS_grad_pe
        SS_grad_array[j,i,1:,:]=SS_grad_stress


Bool_tolerance_array=np.abs( SS_grad_array)<0.005
Bool_count_array=np.count_nonzero(Bool_tolerance_array,axis=3)
Bool_count_min=np.min(Bool_count_array, axis=2)
Bool_count_min=np.min(Bool_count_min, axis=1)
print(Bool_count_min)
# bool count min is the minimum number of successful runs in each set of data. 

# now we will sort through them and extract bool count min successful runs for each data set 
# since it is 0 for the 70_45 runs

adjusted_spring_force_position_tensor_batch_tuple=()
adjusted_log_file_tuple=()
lgf_rows=1002
lgf_columns=8
spring_force_pos_rows=1000
spring_force_columns=6


for j in range(len(tchain)): 
    print(j)
    log_file_array=np.zeros((K.size,Bool_count_min[j],lgf_rows,lgf_columns))
    spring_force_positon_tensor_array=np.zeros((K.size,Bool_count_min[j],spring_force_pos_rows,spring_force_columns))
 
    for i in range(e_end[j]):
        count=0
        
        # find the first true index in each row 
        for l in range(7):
            True_indices=np.argwhere(Bool_tolerance_array[j,i,l]==True)
            count=0
            while count<Bool_count_min[j]:
                passed_file_indices=True_indices[count]
                print(passed_file_indices)

                log_file_array[j,i,]


                count+=1
             




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
legfont=10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False 


plt.rcParams.update({'font.size': 16})
# for j in range(thermal_damp_multiplier.size): 

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



for j in range(K.size): 
    for l in range(3):
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[0])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]), marker=marker[j])
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[j])+","+str(labels_stress[l]),linestyle=linestyle_tuple[j][1], marker=marker[j])
        plt.errorbar(erate[:e_end[j]],stress_tensor_tuple[j][:,l],
                     yerr=stress_tensor_std_tuple[j][:,l]/np.sqrt(j_)
                     ,label="$K="+str(K[j])+","+str(labels_stress[l]),linestyle=linestyle_tuple[j][1],
                       marker=marker[j])
        

    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
       
        plt.xlabel("$\dot{\\gamma}$")
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
        plt.errorbar(erate[:e_end[j]-1],stress_tensor_tuple[j][:-1,l],yerr=stress_tensor_std_tuple[j][:-1,l]/np.sqrt(j_),label="$K="+str(K[j])+","+str(labels_stress[l]), marker=marker[j])
        
        

    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
       
       
        plt.xlabel("$\dot{\gamma}$")
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






#%% area vector plots 
cutoff=0
plt.rcParams['text.usetex'] = "false"
def convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff):
        spherical_coords_tuple=()
        for i in range(len(skip_array)):
            i=skip_array[i]

            
            area_vector_ray=area_vector_spherical_batch_tuple[j][i]
            area_vector_ray[area_vector_ray[:,:,:,2]<0]*=-1
            
            x=area_vector_ray[:,cutoff:,:,0]
            y=area_vector_ray[:,cutoff:,:,1]
            z=area_vector_ray[:,cutoff:,:,2]
        

            spherical_coords_array=np.zeros((j_,area_vector_ray.shape[1]-cutoff,n_plates,3))
        



            # radial coord
            spherical_coords_array[:,:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))

            #  theta coord 
            spherical_coords_array[:,:,:,1]=np.sign(y)*np.arccos(x/(np.sqrt((x**2)+(y**2))))

            #spherical_coords_array[:,:,:,1]=np.sign(x)*np.arccos(y/(np.sqrt((x**2)+(y**2))))
            #spherical_coords_array[:,:,:,1]=np.arctan(y/x)
            
            # phi coord
            # print(spherical_coords_array[spherical_coords_array[:,:,:,0]==0])
            spherical_coords_array[:,:,:,2]=np.arccos(z/np.sqrt((x**2)+(y**2)+(z**2)))

        

            spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_array,)

        return spherical_coords_tuple

def stat_test_on_theta(periodic_data,sample_size):
    KS_test_result=[]
    MW_test_result=[]
    for m in range(100):                
                    # scotts factor 
                    np.random.seed(m)
                    uniform=np.random.uniform(low=np.min(periodic_data), high=np.max(periodic_data),size=periodic_data.size)
                    sample1 = np.random.choice(uniform,size=sample_size, replace = True, p = None)
                    periodic_sample=np.random.choice( np.ravel(periodic_data) , size = sample_size, replace = True, p = None)
                    print(f'Uniform vs. My data: {scipy.stats.ks_2samp( periodic_sample,sample1)}')
                    KS_test_result.append(scipy.stats.ks_2samp(  periodic_sample,sample1)[1])
                   # MW_test_result.append(scipy.stats.mannwhitneyu(  periodic_sample,sample1)[1])

    return KS_test_result,MW_test_result


                                    
                   
#sns.set_theme(font_scale=1.5, rc={'text.usetex' : True})

def producing_random_points_with_theta(number_of_points,rand_int):

    rng = np.random.default_rng(rand_int)
    Phi=np.arccos(1-2*(rng.random((number_of_points))))
    
    Theta=2*np.pi*rng.random((number_of_points))
    rho=1#7.7942286341
    A=Phi
    B=Theta
    R=np.array([rho*np.sin(A)*np.cos(B),rho*np.sin(B)*np.sin(A),rho*np.cos(A)])


    return Phi,Theta,R
                
                # scotts factor 
def stat_test_on_phi(periodic_data,sample_size):
    KS_test_result=[]
    MW_test_result=[]
    for m in range(100):                
                
                    Phi,Theta,R=producing_random_points_with_theta(periodic_data.size,m)

                    sample_sin=np.random.choice( Phi , size = sample_size, replace = True, p = None)
                    periodic_sample=np.random.choice( np.ravel(periodic_data) , size = sample_size, replace = True, p = None)
                    KS_test_result.append(scipy.stats.ks_2samp( periodic_sample,sample_sin)[1])
                    #MW_test_result.append(scipy.stats.mannwhitneyu( periodic_sample,sample_sin)[1])
                    
                    print(f'sampled sine vs. My data sample KS test: {scipy.stats.ks_2samp( periodic_sample,sample_sin)}')
                    #MW only makes sense in ordinal data - no natural ranking 
                   # print(f'sampled sine vs. My data sample Mannwhitney test: {scipy.stats.mannwhitneyu( periodic_sample,sample_sin)}')
                   # print(f'sampled sine vs. My data sample ranksums test: {scipy.stats.ranksums( periodic_sample,sample_sin)}')

    return KS_test_result,MW_test_result

def generic_stat_kolmogorov_2samp(dist1,dist2):
    
    KS_test_result=scipy.stats.ks_2samp( dist1,dist2)
     
    return KS_test_result
     
    
    

def plot_MW_test(MW_test_result):

    plt.plot(MW_test_result,label="MW_test_phi")
    plt.axhline(np.mean(MW_test_result),label="MWmean$="+str(np.mean(MW_test_result))+",\pm"+str(np.std(MW_test_result))+"$",color="red",linestyle="dashed")
    plt.ylabel("pvalue")
    plt.legend()
    plt.show()

def plot_KS_test(KS_test_result,):

    plt.plot(KS_test_result,label="KS_test")
    plt.axhline(np.mean(KS_test_result),label="KSmean$="+str(np.mean(KS_test_result))+",\pm"+str(np.std(KS_test_result))+"$",color="green",linestyle="dotted")
    plt.ylabel("pvalue")
    plt.legend()
    plt.show()

#%%


pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
phi_y_ticks=[0,0.2,0.4,0.6,0.8,1,1.2]
pi_phi_ticks=[ 0,np.pi/8,np.pi/4,3*np.pi/8, np.pi/2]
pi_phi_tick_labels=[ '0','π/8','π/4','3π/8', 'π/2']
theta_y_ticks=[0,0.02,0.04,0.06,0.08,0.1]
skip_array=[0,6,15,23]
spherical_coords_tuple=()
sample_cut=0
cutoff=0
sample_size=500


#%% time series plot of  phi 
linestyle_tuple = ['-', 
  'dotted', 
 'dashed', 'dashdot', 
  'solid', 
 'dashed', 'dashdot', '--']
skip_array=[0,10,21,23]
#skip_array=[0,18,19,23]
skip_array=[0,3,5,7]
skip_array=[8,9,10,11]
skip_array=[12,13,14,15]
skip_array=[0,13,16,18,19]
skip_array=[13]
# skip_array=[20,21,22,23]
# skip_array=[22,24,26,28]
# skip_array=[29,30,31,32]
# skip_array=[33,34,35,36]
#phi 

cutoff=0
timestep_skip_array=[0,5,10,100,200,500,900]
steady_state_index=800
adjust_factor=1
#for j in range(1,K.size):
for j in range(0,1):
    spherical_coords_tuple=convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff)
    p_value_array=np.zeros((len(skip_array),j_,len(timestep_skip_array)))
    KS_stat_array=np.zeros((len(skip_array),j_,len(timestep_skip_array)))
    for i in range(len(skip_array)):

        k=skip_array[i]
        data= spherical_coords_tuple[i][:,:,:,2]
        periodic_data=np.array([data,np.pi-data])
       
        for n in range(j_):

            steady_state_dist=np.ravel(periodic_data[:,n,steady_state_index:,:])
         
            for l in range(len(timestep_skip_array)):
                m=timestep_skip_array[l]
                #timstep_dist=np.ravel(periodic_data[:,n,m,:])
                timstep_dist=np.ravel(periodic_data[:,:,m,:])
                KS_stat,p_value=generic_stat_kolmogorov_2samp(steady_state_dist,timstep_dist)
                p_value_array[i,n,l]=p_value
                KS_stat_array[i,n,l]=KS_stat
                sns.kdeplot( data=np.ravel(periodic_data[:,:,m,:]),
                                label ="$N_{t}="+str(timestep_skip_array[l])+"$",linestyle=linestyle_tuple[j],bw_adjust=adjust_factor)
                #plt.title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$, real="+str(n))
                plt.title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")

           
            plt.xlabel("$\Phi$")
            plt.xticks(pi_phi_ticks,pi_phi_tick_labels)

            #plt.yticks(phi_y_ticks)
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=(1,0.5),frameon=False)
            plt.xlim(0,np.pi/2)
            #plt.xlim(0,np.pi)
            plt.tight_layout()
            #plt.savefig(path_2_log_files+"/plots/phi_dist_.pdf",dpi=1200,bbox_inches='tight')
            plt.show()
#%%plotting KS statistics
for j in range(len(skip_array)):
    k=skip_array[j]
     
    for i in range(j_):
        plt.plot(timestep_skip_array , KS_stat_array[j,i],label="real="+str(i), marker=marker[i])
    plt.ylabel("KS difference ")
    plt.xlabel("output count")
    plt.title("$\dot{\gamma}="+str(erate[k])+"$")
    plt.legend()
    plt.show()

    for i in range(j_):
        plt.plot(timestep_skip_array , p_value_array[j,i],label="real="+str(i), marker=marker[i])

    plt.ylabel("P value")
    plt.xlabel("output count")
    plt.title("$\dot{\gamma}="+str(erate[k])+"$")

    plt.legend()
    plt.show()

#%% different style plot of theta
periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
#theta
adjust_factor=0.25
for j in range(0,1):
    spherical_coords_tuple=convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff)
    
    for i in range(len(skip_array)):

        k=skip_array[i]
   

    
        data= spherical_coords_tuple[i][:,:,:,1]

        periodic_data=np.array([data-2*np.pi,data,data+2*np.pi])
       
        for l in range(len(timestep_skip_array)):
            m=timestep_skip_array[l]
            sns.kdeplot( data=np.ravel(periodic_data[:,:,m,:]),
                            label ="$N_{t}="+str(timestep_skip_array[l])+"$",linestyle=linestyle_tuple[j],bw_adjust=adjust_factor)
            plt.title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


       
        plt.xlabel("$\Theta$")
        plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
        plt.legend(bbox_to_anchor=(1,0.55),frameon=False)


       
        plt.xticks(theta_y_ticks)

        plt.ylabel('Density')
        plt.xlim(-np.pi,np.pi)
       
        #plt.xlim(0,np.pi)
        plt.tight_layout()
        #plt.savefig(path_2_log_files+"/plots/theta_dist_.pdf",dpi=1200,bbox_inches='tight')
        plt.show()

#%%

#theta
adjust_factor=0.25
for j in range(0,1):
    spherical_coords_tuple=convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff)
    
    for i in range(len(skip_array)):

        k=skip_array[i]
   


       
       
        for l in range(len(timestep_skip_array)):
            m=timestep_skip_array[l]
            theta= spherical_coords_tuple[i][:,m,:,1]
            phi=spherical_coords_tuple[i][:,m,:,2]

            plt.scatter(theta,phi, label ="$N_{t}="+str(timestep_skip_array[l])+"$", s=5, marker=marker[l])
            plt.title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


        
            plt.xlabel("$\Theta$")
            plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
            plt.yticks(pi_phi_ticks,pi_phi_tick_labels)
            plt.legend(bbox_to_anchor=(1,0.55),frameon=False)


            #plt.yticks(phi_y_ticks)

            # plt.yticks(phi_y_ticks)
            # plt.xticks(theta_y_ticks)

            # plt.ylabel('Density')
            # plt.xlim(-np.pi,np.pi)
            # plt.ylim(0,np.pi/2)
            #plt.xlim(0,np.pi)
            plt.tight_layout()
            #plt.savefig(path_2_log_files+"/plots/theta_dist_.pdf",dpi=1200,bbox_inches='tight')
            plt.show()

  
#%% different style plot of rho

#rho

f, axs = plt.subplots(1, 4, figsize=(15, 6),sharey=True,sharex=True)
adjust_factor=2
for j in range(0,K.size):
    spherical_coords_tuple=convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff)

   
    i=0
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
   
    sns.kdeplot( data=data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[0],bw_adjust=adjust_factor)
    
    axs[0].axvline(np.mean(data))
    
    axs[0].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")
    print("mean rho 0",np.mean(data))


    i=1
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
    
    sns.kdeplot( data=data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[1],bw_adjust=adjust_factor)
    axs[1].axvline(np.mean(data), label="$\\bar{\\rho}="+str(np.mean(data))+"$")
    print("mean rho 1",np.mean(data))
    axs[1].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=2
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
   
    sns.kdeplot( data=data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[2],bw_adjust=adjust_factor)
    
    axs[2].axvline(np.mean(data))
    print("mean rho 2",np.mean(data))
    
    axs[2].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=3
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
   
    sns.kdeplot( data=data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[3],bw_adjust=adjust_factor)
   
    axs[3].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")
    print("mean rho 3",np.mean(data))
    axs[3].axvline(np.mean(data))


f.supxlabel("$\\rho$")

plt.legend(bbox_to_anchor=(1,0.55),frameon=False)


#plt.yticks(phi_y_ticks)

plt.ylabel('Density')

#plt.xlim(0,np.pi)
plt.tight_layout()

plt.show()
#%% plto theta against phi 

#rho
markersize=0.0005
f, axs = plt.subplots(1, 4, figsize=(15, 6),sharey=True,sharex=True)
adjust_factor=1
for j in range(0,K.size):
    spherical_coords_tuple=convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff)

   
    i=0
    theta=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    phi=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    axs[0].scatter(theta,phi,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],s=markersize)
    
    axs[0].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=1
    theta=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    phi=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    axs[1].scatter(theta,phi,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],s=markersize)
    
   
  
    axs[1].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=2
    theta=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    phi=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    axs[2].scatter(theta,phi,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],s=markersize)
    
    axs[2].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=3
    theta=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    phi=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    axs[3].scatter(theta,phi,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],s=markersize)
   
    axs[3].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    f.supxlabel("$\\Theta$")
    f.supylabel("$\\Phi$", rotation=0)
    plt.legend(bbox_to_anchor=(1,0.55),frameon=False)

    # plt.yticks(pi_phi_ticks,pi_phi_tick_labels)

    # plt.xticks(theta_y_ticks,pi_theta_tick_labels)

    # #plt.ylabel('Density')

    plt.ylim(0,np.pi/2)
    plt.xlim(-np.pi,np.pi)
    plt.tight_layout()

    plt.show()

#%% comparing across K
skip_array=[0,9,13,18]



f, axs = plt.subplots(1, 4, figsize=(15, 6),sharey=True,sharex=True)

#for i in range(len(skip_array)):
adjust=1
for j in range(K.size):
        i=0
        magnitude_spring=interest_vectors_batch_tuple[j][i][:,:,2:5]

        mean_spring_mag=np.mean(magnitude_spring)
          
        
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[0])
        axs[0].set_title("$\dot{\gamma}="+str(erate[i])+"$")
        i=9
        magnitude_spring=interest_vectors_batch_tuple[j][9][:,:,2:5]
       
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[1])
        axs[1].set_title("$\dot{\gamma}="+str(erate[i])+"$")
        i=13
        magnitude_spring=interest_vectors_batch_tuple[j][13][:,:,2:5]
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[2])
        axs[2].set_title("$\dot{\gamma}="+str(erate[i])+"$")
        i=18
        magnitude_spring=interest_vectors_batch_tuple[j][18][:,:,2:5]
        
      
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[3])
        axs[3].set_title("$\dot{\gamma}="+str(erate[i])+"$")
        
        

#plt.yticks(extension_ticks)
#plt.xticks()

plt.legend(bbox_to_anchor=(1,0.55),frameon=False)
f.supxlabel("$\Delta x$")
f.tight_layout()

plt.savefig(path_2_log_files+"/plots/deltax_dist_.pdf",dpi=1200,bbox_inches='tight')
   
plt.show()

#%% comparing erates
skip_array=[0,9,13,18]
#skip_array=[0]

#dist_xticks=([[-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-7.5,-5,-2.5,0,2.5,5]])

#for i in range(len(skip_array)):

adjust=1
for j in range(K.size):
  
    for i in range(len(skip_array)):
            k=skip_array[i]
           
            magnitude_spring=interest_vectors_batch_tuple[j][k][:,:,2:5]

            mean_spring_mag=np.mean(magnitude_spring)
          

            sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                        label ="$\dot{\gamma}="+str(erate[k])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[i])
            plt.axvline(mean_spring_mag,label="$\\bar{\Delta x}="+str(sigfig.round(mean_spring_mag,sigfigs=3))+", \dot{\gamma}="+str(erate[k])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[i])
         
        
            

    #plt.yticks(extension_ticks)

    #plt.xticks()


    plt.xlabel("$\Delta x$")
    plt.legend(fontsize=legfont,bbox_to_anchor=(0.75,0.5), frameon=False)
    plt.xlim(-2,5.5)
    plt.tight_layout()
    

    plt.savefig(path_2_log_files+"/plots/deltax_dist_K"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')
    
    plt.show()





#%% plotting particle velocity against postion 
skip_array=[1,5,10,15,18]
for j in range(K.size):
    for i in range(len(skip_array)):
        i=skip_array[i]


        
        z_position =np.mean(np.mean(pos_batch_tuple[j][i][:,:,:,:,2],axis=0),axis=0)
        x_vel=np.mean(np.mean(vel_batch_tuple[j][i][:,:,:,:,0],axis=0),axis=0)
        pred_x_vel=erate[i]* z_position
        plt.scatter(z_position,x_vel, label="$\dot{\gamma}="+str(erate[i])+"$")
        plt.plot(z_position,pred_x_vel)
        plt.xlabel("$z$")
        plt.ylabel("$v_{x}$",rotation=0,labelpad=20)
        
    plt.legend(bbox_to_anchor=(1,1))
   # plt.savefig(path_2_log_files+"/plots/vx_vs_z_"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')
    plt.show()

#%% plotting initial velocity distribution 

for j in range(K.size):
    # only need the velocity distribution at gammadot=0
    for i in range(erate.size):
        f, axs = plt.subplots(1, 3, figsize=(16, 6),sharey=True,sharex=True)
        
        x_vel=np.ravel(pos_vel_batch_tuple[j][i][0,:,3,0])
        sns.kdeplot( data=x_vel, ax=axs[0])
        
        y_vel=np.ravel(pos_vel_batch_tuple[j][i][0,:,3,1])
        sns.kdeplot( data=y_vel, ax=axs[1])
       
        z_vel=np.ravel(pos_vel_batch_tuple[j][i][0,:,3,2])
        sns.kdeplot( data=z_vel, ax=axs[2])
       
        plt.show()

#%%fitting gaussian for velocity component 
for j in range(K.size):
    # only need the velocity distribution at gammadot=0
    for i in range(1):
        maxwell = scipy.stats.norm
        data = np.ravel(vel_batch_tuple[j][i][:,:,:,:,0])

        params = maxwell.fit(data)
        print(params)
        # (0, 4.9808603062591041)

        plt.hist(data, bins=20,density=True)
        x = np.linspace(np.min(data),np.max(data),data.size)
        plt.plot(x, maxwell.pdf(x, *params), lw=3)
        plt.xlabel("$v_{x}$")
        plt.ylabel("Density")
        #plt.savefig(path_2_log_files+"/plots/vx_dist_"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')
        plt.show()
        data = np.ravel(vel_batch_tuple[j][i][:,:,:,:,1])

        params = maxwell.fit(data)
        print(params)
        # (0, 4.9808603062591041)

        plt.hist(data, bins=20,density=True)
        x = np.linspace(np.min(data),np.max(data),data.size)
        plt.plot(x, maxwell.pdf(x, *params), lw=3)
        plt.xlabel("$v_{y}$")
        plt.ylabel("Density")
       # plt.savefig(path_2_log_files+"/plots/vy_dist_"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')
        plt.show()
        data = np.ravel(vel_batch_tuple[j][i][:,:,:,:,2])

        params = maxwell.fit(data)
        print(params)
        # (0, 4.9808603062591041)

        plt.hist(data, bins=20,density=True)
        x = np.linspace(np.min(data),np.max(data),data.size)
        plt.plot(x, maxwell.pdf(x, *params), lw=3)
        plt.xlabel("$v_{z}$")
        plt.ylabel("Density")
       # plt.savefig(path_2_log_files+"/plots/vz_dist_"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')

        plt.show()

#%% looking at maxwell_ botlzman  for speed dist 

for j in range(K.size):
    # only need the velocity distribution at gammadot=0
    for i in range(1):
        
        x_vel=np.ravel(pos_vel_batch_tuple[j][i][:,:,3:6,0])
       
        
        y_vel=np.ravel(pos_vel_batch_tuple[j][i][:,:,3:6,1])
        
       
        z_vel=np.ravel(pos_vel_batch_tuple[j][i][:,:,3:6,2])
        speed= np.sqrt(x_vel**2+ y_vel**2 + z_vel**2)
        maxwell = scipy.stats.maxwell
        params = maxwell.fit(speed)
        print(params)
        # (0, 4.9808603062591041)

        plt.hist(speed, bins=20,density=True)
        x = np.linspace(np.min(speed),np.max(speed),speed.size)
        plt.plot(x, maxwell.pdf(x, *params), lw=3)
        plt.xlabel("$|v|$")
        plt.ylabel("Density")
      #  plt.savefig(path_2_log_files+"/plots/v_dist_"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')
        plt.show()
# %% interest vectors

ell_1=interest_vectors_batch_tuple[0][15][:,:,4]
ell_2=interest_vectors_batch_tuple[0][0][:,:,1]

sns.kdeplot(np.ravel(ell_1))
plt.show()

#%% ridgeline plot of phi distributions 


#%% ridgeline plot of theta distributions 


#%% violin plot of phi 
sns.set_palette('colorblind')
# sns.color_palette("mako", as_cmap=True)
# sns.color_palette("viridis")
#sns.set_palette('virdris')

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
legfont=10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False 

plt.rcParams.update({'font.size': 16})
adjust_factor = 2
erate_1=0
erate_2=21
plt.rcParams["figure.figsize"] = (25,6 )
for j in range(K.size):
    skip_array = np.arange(erate_1, erate_2, 1)
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff)
    periodic_data_list = []
    erate_list = []
    for i in range(skip_array.size):
        data = np.ravel(spherical_coords_tuple[i][:, :, :, 2])  # Assuming this extracts the spherical data
        periodic_data = np.ravel(np.array([data, np.pi - data]))  # Handling the periodic nature
        periodic_data_list.append(periodic_data)


    # Convert lists to DataFrames at the end
    periodic_data_df = pd.DataFrame(periodic_data_list)
    periodic_data_df=periodic_data_df.transpose()
    erate_str=np.around(erate[erate_1:erate_2],3).astype("str")
    periodic_data_df.columns= erate_str
    print(periodic_data_df.isna().sum())
    # erate_df = pd.DataFrame(erate[:e_end[j]])
    # full_df = pd.concat([erate_df, periodic_data_df], axis=0)
    # full_df = full_df.rename(columns={full_df.columns[0]: "erate"})

    # # rename columns 1 to end 
    # full_df.columns = full_df.columns[:1].tolist() + [f"part_angle" for i in range(1, len(full_df.columns))]

    # # Combine both DataFrames into a final DataFrame
  
        


    
   
    sns.violinplot( data=periodic_data_df, inner=None, linewidth=0 ,scale="width")
    plt.yticks(pi_phi_ticks,pi_phi_tick_labels)
    plt.ylim(0,np.pi/2)
    plt.ylabel("$\Phi$")
    plt.xlabel("$\dot{\gamma}$")
    plt.show()

   
#%% violin plot of theta 
adjust_factor = 0.005
erate_1=0
erate_2=21

for j in range(0, K.size):
    skip_array = np.arange(erate_1, erate_2, 1)
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff)
    periodic_data_list = []
    erate_list = []
    for i in range(skip_array.size):
        data = np.ravel(spherical_coords_tuple[i][:, :, :, 1])  # Assuming this extracts the spherical data
        periodic_data = np.ravel(np.array([data, np.pi - data]))  # Handling the periodic nature
        periodic_data_list.append(periodic_data)

        

    # Convert lists to DataFrames at the end
    periodic_data_df = pd.DataFrame(periodic_data_list)
    periodic_data_df=periodic_data_df.transpose()
    
    erate_str=erate[erate_1:erate_2].astype("str")
    periodic_data_df.columns= erate_str
    print(periodic_data_df.isna().sum())
    # erate_df = pd.DataFrame(erate[:e_end[j]])
    # full_df = pd.concat([erate_df, periodic_data_df], axis=0)
    # full_df = full_df.rename(columns={full_df.columns[0]: "erate"})

    # # rename columns 1 to end 
    # full_df.columns = full_df.columns[:1].tolist() + [f"part_angle" for i in range(1, len(full_df.columns))]

    # # Combine both DataFrames into a final DataFrame
  
        


    
   
    sns.violinplot( data=periodic_data_df, inner=None, linewidth=0 ,scale="width")
    plt.ylim(-np.pi,np.pi)
    plt.yticks(pi_theta_ticks,pi_theta_tick_labels)
    plt.ylabel("$\Theta$")
    plt.xlabel("$\dot{\gamma}$")
    plt.show()
# %%
