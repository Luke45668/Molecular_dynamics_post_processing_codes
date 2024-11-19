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
 'None', ' ', '', 'solid', 
 'dashed', 'dashdot', '--']

#%% 

#note: currently we have one missing shear rate for k=30,60 so need to run them again with erate 0.02 to get the full picture 
# I have k=20 downloaded, K=120 is running , need to produce run files for k=30,60

damp=np.array([ 0.035, 0.035,0.035 ])
K=np.array([  30,  60  ,90
            ])
K=np.array([ 20, 30,  60  
            ])
# K=np.array([  50   ,
#             ])
erate=np.flip(np.array([1.   , 0.9  , 0.8  , 0.7  , 0.6  , 0.5  , 0.4  , 0.3  , 0.2  ,
       0.175, 0.15 , 0.125, 0.1  , 0.08 , 0.06 , 0.04 , 0.02 , 0.01 ,
       0.005, 0.  ]))

erate=np.flip(np.array([1.   , 0.9  , 0.8  , 0.7  , 0.6  , 0.5  , 0.4  , 0.3  , 0.2  ,
       0.175, 0.15 , 0.125, 0.1  , 0.08 , 0.06 , 0.04  , 0.01 ,
       0.005, 0.  ]))

no_timesteps=np.flip(np.array([ 157740000,  175267000,  197175000,  225343000,  262901000,
         315481000,  394351000,  525801000,  788702000,   90137000,
         105160000,  126192000,  157740000,  197175000,  262901000,
         394351000,  394351000,  788702000, 1577404000,   10000000]))

e_in=0
e_end=erate.size
n_plates=100

strain_total=100

path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/run_279865/saved_tuples"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/dumbell_run/log_tensor_files/saved_tuples"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/final_tuples"


thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
j_=5
sim_fluid=30.315227255599112

eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
damp_ratio=mass_pol/damp


#%% save tuples
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
area_vector_spherical_batch_tuple=()
interest_vectors_batch_tuple=()
pos_vel_batch_tuple=()


# loading all data into one 
for i in range(K.size):
    label='damp_'+str(damp[i])+'_K_'+str(K[i])+'_'

    spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(batch_load_tuples(label,
                                                            "spring_force_positon_tensor_tuple.pickle"),)
    # erate_velocity_batch_tuple=erate_velocity_batch_tuple+(batch_load_tuples(label,
    #                                                         "erate_velocity_tuple.pickle"),)
    # COM_velocity_batch_tuple=COM_velocity_batch_tuple+(batch_load_tuples(label,
    #                                                         "COM_velocity_tuple.pickle"),)
    # conform_tensor_batch_tuple=conform_tensor_batch_tuple+(batch_load_tuples(label,
    #                                                         "conform_tensor_tuple.pickle"),)
    pos_vel_batch_tuple=pos_vel_batch_tuple+(batch_load_tuples(label,
                                                            "new_pos_vel_tuple.pickle"),)

    log_file_batch_tuple=log_file_batch_tuple+(batch_load_tuples(label,
                                                            "log_file_tuple.pickle"),)
    area_vector_spherical_batch_tuple=area_vector_spherical_batch_tuple+(batch_load_tuples(label,"area_vector_tuple.pickle"),)
    
    interest_vectors_batch_tuple=interest_vectors_batch_tuple+(batch_load_tuples(label,
                                                                                 "interest_vectors_tuple.pickle"),)

    


     

#%% strain points for temperatuee data 
strainplot_tuple=()

for i in range(erate.size):
    
    strain_plotting_points= np.linspace(0,strain_total,log_file_batch_tuple[0][i].shape[0])

    strainplot_tuple=strainplot_tuple+(strain_plotting_points,)  
    print(strainplot_tuple[i].size)

def strain_plotting_points(total_strain,points_per_iv):
     #points_per_iv= number of points for the variable measured against strain 
     strain_unit=total_strain/points_per_iv
     strain_plotting_points=np.arange(0,total_strain,strain_unit)
     return  strain_plotting_points

#%% energy vs time plot
plt.rcParams["figure.figsize"] = (12,6 )
# pe vs time 

for j in range(K.size):
    for i in range(erate.size):
        column=2
        plt.subplot(1,2,1)
        plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
    #plt.yscale('log')
        plt.xlabel("$\gamma$")
        plt.ylabel("$E_{p}$")
        plt.title("$K="+str(K[j])+"$")
        #plt.show()
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
 
        column=5

        plt.subplot(1,2,2)
        plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
       # plt.yscale('log')
        plt.xlabel("$\gamma$")
        plt.ylabel("$T$")
        plt.title("$K="+str(K[j])+"$")
    plt.show()
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
#%5
folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=5
final_temp=np.zeros((erate.size))
mean_temp_array=np.zeros((K.size,erate.size))
for j in range(K.size):
    for i in range(erate.size):
        
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




#%% look at internal stresses
sns.set_palette('colorblind')
# sns.color_palette("mako", as_cmap=True)
# sns.color_palette("viridis", as_cmap=True)
# sns.set_palette('virdris')
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
legfont=12
folder="stress_tensor_plots"
marker=['x','+','^',"1","X","d","*","P","v"]
aftcut=1
cut=0.8
folder_check_or_create(path_2_log_files,folder)
labels_stress=np.array(["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{yz}$"])



#compute stress tensor 
##y_ticks_stress=[-10,0,20,40,60,80] # for plates 
#y_ticks_stress=[0.95,1,1.05,1.1,1.15,1.2,1.25,1.3]
stress_tensor=np.zeros((K.size,e_end,6))
stress_tensor_std=np.zeros((K.size,e_end,6))
n_1=np.zeros((K.size,e_end))
n_1_error=np.zeros((K.size,e_end))
n_2=np.zeros((K.size,e_end))
n_2_error=np.zeros((K.size,e_end))
for j in range(K.size):
    stress_tensor[j],stress_tensor_std[j]= stress_tensor_averaging(e_end,
                            labels_stress,
                            cut,
                            aftcut,
                           spring_force_positon_tensor_batch_tuple[j],j_)
    n_1[j],n_1_error[j]=compute_n_stress_diff(stress_tensor[j], 
                          stress_tensor_std[j],
                          0,2,
                          j_,n_plates,
                         )

    n_2[j],n_2_error[j]=compute_n_stress_diff(stress_tensor[j], 
                          stress_tensor_std[j],
                          2,1,
                          j_,n_plates,
                          )
  
for j in range(K.size):    
    for i in range(6):

        plotting_stress_vs_strain( spring_force_positon_tensor_batch_tuple[j],
                                e_in,e_end,j_,
                                strain_total,cut,aftcut,i,labels_stress[i],erate)
    plt.legend(fontsize=legfont) 
    plt.title("$K="+str(K[j])+"$")
    plt.tight_layout()
    #plt.savefig(path_2_log_files+"/plots/"+str(K[j])+"_SS_grad_plots.pdf",dpi=1200,bbox_inches='tight')       
   
    plt.show()



for j in range(K.size): 
    plot_stress_tensor(0,3,
                       stress_tensor[j],
                       stress_tensor_std[j],
                       j_,n_plates, labels_stress,marker,0,erate,e_end,linestyle_tuple[j])
    plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
plt.legend(fontsize=legfont) 
#plt.yticks(y_ticks_stress)
#plt.ylim(0.9,1.3)
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.show()

for j in range(K.size): 
    plot_stress_tensor(3,6,
                       stress_tensor[j],
                       stress_tensor_std[j],
                       j_,n_plates, labels_stress,marker,0,erate,e_end,linestyle_tuple[j])
    plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
plt.legend(fontsize=legfont) 
#plt.yticks(y_ticks_stress)
plt.tight_layout()

#plt.savefig(path_2_log_files+"/plots/_stress_tensor_3_6_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.show()

#%% collapse plot
for j in range(K.size): 
    plot_stress_tensor(0,3,
                       stress_tensor[j]/K[j],
                       stress_tensor_std[j]/K[j],
                       j_,n_plates, labels_stress,marker,0,erate,e_end,linestyle_tuple[j])
    plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
plt.ylabel("$\sigma_{\\alpha\\beta}/K$",rotation=0,labelpad=25)
plt.legend(fontsize=legfont) 
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/stress_tensor_K_scaled_0_3_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()

for j in range(K.size): 
    plot_stress_tensor(3,6,
                       stress_tensor[j]/K[j],
                       stress_tensor_std[j]/K[j],
                       j_,n_plates, labels_stress,marker,0,erate,e_end,linestyle_tuple[j])
    plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
plt.ylabel("$\sigma_{\\alpha\\beta}/K$",rotation=0,labelpad=25)
plt.legend(fontsize=legfont) 
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/stress_tensor_K_scaled_3_6_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()


#%%
# now plot n1 and n2 
#probably need to turn this into a a function 
n_y_ticks=[-10,0,20,40,60,80]
cutoff=0
for j in range(K.size):
    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
  

    plt.errorbar(erate[cutoff:e_end], n_1[j,cutoff:e_end], yerr =n_1_error[j,cutoff:e_end],
                  ls='none',label="$N_{1},K="+str(K[j])+"$",marker=marker[j] )
    popt,cov_matrix_n1=curve_fit(linearthru0,erate[cutoff:e_end], n_1[j,cutoff:e_end])
    # difference=np.sqrt(np.sum((n_1[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

    # plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),ls=linestyle_tuple[j],#)#,
    #         label="$N_{1,fit,K="+str(K[j])+"},m="+str(sigfig.round(popt[0],sigfigs=2))+\
    #             ",\\varepsilon="+str(sigfig.round(difference,sigfigs=2))+"$")

    #plt.xscale('log')
    #plt.show()
    #print(difference)

    plt.errorbar(erate[cutoff:e_end], n_2[j,cutoff:e_end], yerr =n_2_error[j,cutoff:e_end],
                  ls='none',label="$N_{2},K="+str(K[j])+"$",marker=marker[j] )
    # popt,cov_matrix_n2=curve_fit(linearthru0,erate[cutoff:e_end], n_2[j,cutoff:e_end])
    # difference=np.sqrt(np.sum((n_2[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

    # plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),ls=linestyle_tuple[j],#)#,
    #         label="$N_{2,fit,K="+str(K[j])+"},m="+str(sigfig.round(popt[0],sigfigs=2))+\
    #         ",\\varepsilon="+str(sigfig.round(difference,sigfigs=2))+"$")
    # # #
plt.legend(fontsize=legfont, frameon=False)
#plt.xscale('log')
plt.xlabel("$\dot{\gamma}$")
plt.ylabel("$N_{\\alpha}$",rotation=0)
#plt.yticks(n_y_ticks)
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/N1_N2_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()
#print(difference)

#collapse N1 and N2 /sigma_xz
cutoff=1
for j in range(K.size):
    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
  

    plt.errorbar(erate[cutoff:e_end], n_1[j,cutoff:e_end]/stress_tensor[j,cutoff:e_end,3], yerr =np.abs(n_1_error[j,cutoff:e_end]/stress_tensor[j,cutoff:e_end,3]),
                  ls='none',label="$N_{1},K="+str(K[j])+"$",marker=marker[j] )
    # popt,cov_matrix_n1=curve_fit(linearthru0,erate[cutoff:e_end], n_1[j,cutoff:e_end])
    # difference=np.sqrt(np.sum((n_1[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

    # plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),
    #         label="$N_{1,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
    #             ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$", ls=linestyle_tuple[j])

    #plt.xscale('log')
    #plt.show()
    #print(difference)

    # plt.errorbar(erate[cutoff:e_end], n_2[j,cutoff:e_end]/K[j], yerr =n_2_error[j,cutoff:e_end]/K[j],
    #               ls='none',label="$N_{2},K="+str(K[j])+"$",marker=marker[j] )
    # popt,cov_matrix_n2=curve_fit(linearthru0,erate[cutoff:e_end], n_2[j,cutoff:e_end])
    # difference=np.sqrt(np.sum((n_2[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

    # plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),
    #         label="$N_{2,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
    #         ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$",ls=linestyle_tuple[j])
    #
    plt.legend(fontsize=legfont)
    plt.ylabel("$N_{1}/\sigma_{xz}$",rotation=0)
    #plt.xscale('log')
    plt.xlabel("$\dot{\gamma}$")
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/N1_N2_scaled_K_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()
#print(difference)





# now do N1 N2 comparison plots
# this plot isnt so useful 
n2_factor=[-10,-18]
for j in range(2):

    #sns.set_palette('icefire')
    plt.scatter(erate,n_1[j],label="$N_{1},K="+str(K[j])+"$", marker=marker[j])
    plt.scatter(erate,n2_factor[j]*n_2[j],label="$"+str(n2_factor[j])+"N_{2},K="+str(K[j])+"$", marker=marker[j+2])
    plt.xlabel("$\dot{\gamma}$")
    plt.ylabel("$N_{\\alpha}$",rotation=0)
    #plt.ylabel("$\\frac{N_{1}}{N_{2}}$", rotation=0)
    plt.legend()
    #plt.xscale('log')
    plt.legend(fontsize=legfont) 
    #plt.yticks(n_y_ticks)
    plt.tight_layout()

   # plt.savefig(path_2_log_files+"/plots/N1_N2_multi_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
    plt.show()


#%%viscosity for plate 

cutoff=1
for j in range(K.size):
    xz_stress= stress_tensor[j,cutoff:,3]
    xz_stress_std=stress_tensor_std[j,:,3]/np.sqrt(j_*n_plates)
    #powerlaw
    plt.errorbar(erate[cutoff:e_end], xz_stress/erate[cutoff:e_end]/sim_fluid, yerr =xz_stress_std[cutoff:]/erate[cutoff:e_end]/sim_fluid,
                  ls='none',label="$\eta,K="+str(K[j])+"$",marker=marker[j] )
    popt,cov_matrix_xz=curve_fit(powerlaw,erate[cutoff:e_end], xz_stress/erate[cutoff:e_end]/sim_fluid)
    y=xz_stress/erate[cutoff:e_end]/sim_fluid
    y_pred=popt[0]*(erate[cutoff:e_end]**(popt[1]))/sim_fluid
    difference=np.sqrt(np.sum((y-y_pred)**2)/e_end-cutoff)
    plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end]**(popt[1]))),
            label="$\eta_{fit},a="+str(sigfig.round(popt[0],sigfigs=3))+",n="+str(sigfig.round(popt[1],sigfigs=3))+

            ",\\varepsilon=\pm"+str(sigfig.round(difference,sigfigs=3))+"$")

    plt.legend(fontsize=legfont) 
    plt.ylabel("$\eta/\eta_{s}$", rotation=0,labelpad=10)
    plt.xlabel("$\dot{\gamma}$")
    plt.tight_layout()
# plt.xscale('log')
# plt.yscale('log')
#plt.savefig(path_2_log_files+"/plots/eta_vs_K_"+str(K[j])+"gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show() 

#%%viscosity for dumbell

cutoff=1 
for j in range(K.size):
    xz_stress= stress_tensor[j,cutoff:,3]
    xz_stress_std=stress_tensor_std[j,:,3]/np.sqrt(j_*n_plates)
    #powerlaw
    plt.errorbar(erate[cutoff:e_end], xz_stress/erate[cutoff:e_end]/sim_fluid, yerr =xz_stress_std[cutoff:],
                  ls='none',label="$\eta,K="+str(K[j])+"$",marker=marker[j] )
    # popt,cov_matrix_xz=curve_fit(powerlaw,erate[cutoff:e_end], xz_stress/erate[cutoff:e_end])
    # y=xz_stress/erate[cutoff:e_end]
    # y_pred=popt[0]*(erate[cutoff:e_end]**(popt[1]))
    # difference=np.sqrt(np.sum((y-y_pred)**2)/e_end-cutoff)
    # plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end]**(popt[1]))),
    #         label="$\eta_{fit},a="+str(sigfig.round(popt[0],sigfigs=3))+",n="+str(sigfig.round(popt[1],sigfigs=3))+

    #         ",\\varepsilon=\pm"+str(sigfig.round(difference,sigfigs=3))+"$")

   
    plt.ylabel("$\eta/\eta_{s}$", rotation=0,labelpad=10)
    plt.xlabel("$\dot{\gamma}$")
plt.tight_layout()
plt.legend(fontsize=legfont) 
    # plt.xscale('log')
    # plt.yscale('log')
plt.savefig(path_2_log_files+"/plots/eta_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show() 

#%% n1  fit for dumbells

cutoff=0
j=0
plt.errorbar(erate[cutoff:e_end], n_1[j,cutoff:e_end], yerr =n_1_error[j,cutoff:e_end], ls='none',label="$N_{1}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(quadfunc,erate[cutoff:e_end], n_1[j,cutoff:e_end])
difference=np.sqrt(np.sum((n_1[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])**2))**2)/(e_end))

plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])**2),
         label="$N_{1,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$",marker='none')
plt.legend()
plt.ylabel("$N_{1}$", rotation=0, labelpad=20)
plt.xlabel("$\dot{\gamma}$")
plt.tight_layout()
    # plt.xscale('log')
    # plt.yscale('log')
plt.savefig(path_2_log_files+"/plots/N1_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')

plt.show()


#%% area vector plots 
#sns.set_theme(font_scale=1.5, rc={'text.usetex' : True})

plt.rcParams["figure.figsize"] = (6,4 )
plt.rcParams.update({'font.size': 14})
SIZE_DEFAULT = 14
SIZE_LARGE = 16
legfont=12
# plt.rcParams['text.usetex'] = True


# plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels


pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
phi_y_ticks=[0,0.1,0.2,0.3,0.4,0.5,0.6]
pi_phi_ticks=[ 0,np.pi/8,np.pi/4,3*np.pi/8, np.pi/2]
pi_phi_tick_labels=[ '0','π/8','π/4','3π/8', 'π/2']
theta_y_ticks=[0,0.02,0.04,0.06,0.08,0.1]
skip_array=[0,6,12,18]
spherical_coords_tuple=()
sample_cut=0
cutoff=3000
bin_count=27
spherical_coords_batch_tuple=()
# fig = plt.figure(constrained_layout=True)
# spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
for j in range(K.size):
    spherical_coords_tuple=()
    for i in range(len(skip_array)):
        i=skip_array[i]

        
        area_vector_ray=area_vector_spherical_batch_tuple[j][i]
        # x_mean=np.mean(np.mean(area_vector_ray[:,:,:,0],axis=0),axis=0)
        # y_mean=np.mean(np.mean(area_vector_ray[:,:,:,1],axis=0),axis=0)
        # z_mean=np.mean(np.mean(area_vector_ray[:,:,:,2],axis=0),axis=0)
        x_mean=np.mean(area_vector_ray[:,cutoff:,:,0],axis=1)
        y_mean=np.mean(area_vector_ray[:,cutoff:,:,1],axis=1)
        z_mean=np.mean(area_vector_ray[:,cutoff:,:,2],axis=1)
        # x_mean=area_vector_ray[:,cutoff:,:,0]
        # y_mean=area_vector_ray[:,cutoff:,:,1]
        # z_mean=area_vector_ray[:,cutoff:,:,2]
        x=np.ravel(x_mean)
        y=np.ravel(y_mean)
        z=np.ravel(z_mean)

        spherical_coords_array=np.zeros((z.shape[0],3))
       
        
        for k in range(z.shape[0]):
            if z[k]<0:
                z[k]=-1*z[k]
                y[k]=-1*y[k]
                x[k]=-1*x[k]

            else:
                continue
        
        # detect all z coords less than 0 and multiply all 3 coords by -1

        # area_vector_ray[area_vector_ray[:,:,:,2]<0]=area_vector_ray[:,:,:,0]*-1
        # area_vector_ray[area_vector_ray[:,:,:,2]<0]=area_vector_ray[:,:,:,1]*-1
        #area_vector_ray[area_vector_ray[:,:,:,2]<0]*=-1
        #area_vector_ray[area_vector_ray[:,:,:,2]<0]*=-1
        
       


        # radial coord
        spherical_coords_array[:,0]=np.sqrt((x**2)+(y**2)+(z**2))

        #  theta coord 
        spherical_coords_array[:,1]=np.sign(y)*np.arccos(x/(np.sqrt((x**2)+(y**2))))
        # phi coord
        # print(spherical_coords_array[spherical_coords_array[:,:,:,0]==0])
        spherical_coords_array[:,2]=np.arccos(z/np.sqrt((x**2)+(y**2)+(z**2)))


        spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_array,)




    plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")
    for l in range(4):
    #for j in range(j_):


        #l=skip_array[l]
        
        
        
        data=np.ravel(spherical_coords_tuple[l][-sample_cut:,1])
        periodic_data=np.array([data-2*np.pi,data,data+2*np.pi])  

        sns.kdeplot( data=np.ravel(periodic_data),
                    label ="$\dot{\gamma}="+str(erate[skip_array[l]],)+"$",linestyle=linestyle_tuple[l])#bw_adjust=0.1
        # plt.hist(np.ravel(periodic_data),bins=bin_count,label="$\dot{\gamma}="+str(erate[skip_array[l]])+"$",
        #          histtype='step',
        #             stacked=True, fill=False, density=True, linestyle=linestyle_tuple[l])
       
    plt.xlabel("$\Theta$")
    plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
    #plt.yticks(theta_y_ticks)
    plt.xlim(-np.pi,np.pi)

    plt.ylabel('Density')
    plt.legend(fontsize=legfont) 
    plt.tight_layout()
    plt.savefig(path_2_log_files+"/plots/theta_dist_K_"+str(K[j])+"_.pdf",dpi=1200,bbox_inches='tight')
    plt.show()
 
    plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")  
    for l in range(4):
    #for j in range(skip_array_2.size):
        #l=skip_array[l]
        

        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,skip_array_2[j],:,2]),
        #              label="output_range:"+str(skip_array_2[j]))
        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,-1,:,2]),
        #              label ="$\dot{\gamma}="+str(erate[i])+"$")
        data=np.ravel(spherical_coords_tuple[l][-sample_cut:,2])
        #periodic_data=np.array([data,0.5*np.pi+data])  
        #NOTE ask helen about this 
        periodic_data=np.ravel(np.array([data,np.pi-data]))

        
        #periodic_data=data
        #periodic_data[periodic_data==np.pi*0.5]=0
        sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[l]])+"$",linestyle=linestyle_tuple[l])
                   
        # plt.hist(periodic_data,bins=bin_count,
        #           label="$\dot{\gamma}="+str(erate[skip_array[l]])+"$",histtype='step',
        #             stacked=True, fill=False, density=True, linestyle=linestyle_tuple[l])
        # bincheck=np.histogram(data,bins=40)
   
    plt.xlabel("$\Phi$")
    plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
    #
    plt.yticks(phi_y_ticks)
    plt.ylabel('Density')
    plt.legend(fontsize=legfont,loc='upper right') 
    plt.xlim(0,np.pi/2)
    #plt.xlim(0,np.pi)
    plt.tight_layout()
   
    plt.savefig(path_2_log_files+"/plots/phi_dist_K_"+str(K[j])+"_.pdf",dpi=1200,bbox_inches='tight')
    plt.show()

    spherical_coords_batch_tuple=spherical_coords_batch_tuple+(spherical_coords_tuple,)

#%% different style plot of theta and phi 

#phi 
f, axs = plt.subplots(1, 4, figsize=(10, 6),sharey=True,sharex=True)

for j in range(K.size):

    i=0
    data=np.ravel(spherical_coords_batch_tuple[j][i][-sample_cut:,2])
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[i]])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[0])
    axs[0].legend(fontsize=legfont)


    i=1
    data=np.ravel(spherical_coords_batch_tuple[j][i][-sample_cut:,2])
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[i]])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[1])
    axs[1].legend(fontsize=legfont)


    i=2
    data=np.ravel(spherical_coords_batch_tuple[j][i][-sample_cut:,2])
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[i]])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[2])
    axs[2].legend(fontsize=legfont)


    i=3
    data=np.ravel(spherical_coords_batch_tuple[j][i][-sample_cut:,2])
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[i]])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[3])
    axs[3].legend(fontsize=legfont)

f.supxlabel("$\Phi$")
plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
    #
plt.yticks(phi_y_ticks)
plt.ylabel('Density')

plt.xlim(0,np.pi/2)
#plt.xlim(0,np.pi)
plt.tight_layout()
plt.show()


#%%theta 
f, axs = plt.subplots(1, 4, figsize=(10, 6),sharey=True,sharex=True)
theta_y_ticks=[0,0.02,0.04,0.06,0.08,0.1]
for j in range(K.size):

    i=0
    data=np.ravel(spherical_coords_batch_tuple[j][i][-sample_cut:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )

    sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[i]])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[0])
    axs[0].legend(fontsize=legfont)


    i=1
    data=np.ravel(spherical_coords_batch_tuple[j][i][-sample_cut:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
    sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[i]])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[1])
    axs[1].legend(fontsize=legfont)


    i=2
    data=np.ravel(spherical_coords_batch_tuple[j][i][-sample_cut:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )

    sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[i]])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[2])
    axs[2].legend(fontsize=legfont)


    i=3
    data=np.ravel(spherical_coords_batch_tuple[j][i][-sample_cut:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) ) 

    sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[i]])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[3])
    axs[3].legend(fontsize=legfont)


f.supxlabel("$\Theta$")
plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
    
#plt.yticks(theta_y_ticks)
plt.ylabel('Density')
plt.ylim(0.03,0.07)
plt.xlim(-np.pi,np.pi)
#plt.xlim(0,np.pi)
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/theta_dist_.pdf",dpi=1200,bbox_inches='tight')
plt.show()


  

#%% 
plt.rcParams["figure.figsize"] = (12,6 )
extension_ticks=[0,1,2,3,4]
skip_array=[0,9,18]
for i in range(len(skip_array)):
    
    for j in range(K.size):
    
        l=skip_array[i]
    # for i in range(e_in,e_end):

        # sns.kdeplot(eq_spring_length-np.ravel(interest_vectors_tuple[i][:,:,2:5]),
        #              label ="$K="+str(K)+"$")
                    #label ="$\dot{\gamma}="+str(erate[i])+"$")

        plt.subplot(1, 3, i+1)
        sns.kdeplot(np.ravel(interest_vectors_batch_tuple[j][l][:,:,2:5])-eq_spring_length,
                    label ="$\dot{\gamma}="+str(erate[l])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j])
        #NOTE: is this level of smoothing appropriate
        # plt.hist(eq_spring_length+0.125-np.ravel(interest_vectors_batch_tuple[j][i][:,:,2:5]),
        #         histtype='step', stacked=True,
        #         fill=False, density=True, linestyle=linestyle_tuple[l], label ="$\dot{\gamma}="+str(erate[i])+"$")
   
    
plt.xlabel("$\Delta x$")


#plt.legend(fontsize=legfont,loc='upper right') 
plt.yticks(extension_ticks)

plt.legend(fontsize=legfont)
#plt.savefig(path_2_log_files+"/plots/deltax_dist_K_"+str(K[j])+"_.pdf",dpi=1200,bbox_inches='tight')
#plt.xlim(-3,2)
plt.ylabel('Density')
plt.show()


#%% 
skip_array=[0,9,13,18]
plt.rcParams["figure.figsize"] = (12,6 )

dist_xticks=([[-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-7.5,-5,-2.5,0,2.5,5]])


f, axs = plt.subplots(1, 4, figsize=(12, 6),sharey=True,sharex=True)

#for i in range(len(skip_array)):
adjust=1
for j in range(K.size):
        R_x=interest_vectors_batch_tuple[j][0][:,:,2]
        R_y=interest_vectors_batch_tuple[j][0][:,:,3]
        R_z=interest_vectors_batch_tuple[j][0][:,:,4]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)

        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$\dot{\gamma}="+str(erate[0])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[0])
        axs[0].legend(fontsize=legfont)
        # axs[0].xticks(dist_xticks[0][:])

        R_x=interest_vectors_batch_tuple[j][9][:,:,2]
        R_y=interest_vectors_batch_tuple[j][9][:,:,3]
        R_z=interest_vectors_batch_tuple[j][9][:,:,4]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
       
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$\dot{\gamma}="+str(erate[9])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[1])
        axs[1].legend(fontsize=legfont)
        # axs[1].xticks(dist_xticks[1][:])
        R_x=interest_vectors_batch_tuple[j][13][:,:,2]
        R_y=interest_vectors_batch_tuple[j][13][:,:,3]
        R_z=interest_vectors_batch_tuple[j][13][:,:,4]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$\dot{\gamma}="+str(erate[13])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[2])
        axs[2].legend(fontsize=legfont)
        # axs[2].xticks(dist_xticks[2][:])
        #plt.legend(fontsize=legfont) 
        R_x=interest_vectors_batch_tuple[j][18][:,:,2]
        R_y=interest_vectors_batch_tuple[j][18][:,:,3]
        R_z=interest_vectors_batch_tuple[j][18][:,:,4]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$\dot{\gamma}="+str(erate[18])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[3])
        axs[3].legend(fontsize=legfont)
        
        

plt.yticks(extension_ticks)
plt.xticks()


f.supxlabel("$\Delta x$")
f.tight_layout()

plt.savefig(path_2_log_files+"/plots/deltax_dist_.pdf",dpi=1200,bbox_inches='tight')
   
plt.show()


#%% x dist 
plt.rcParams["figure.figsize"] = (12,6 )

dist_xticks=([[-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-7.5,-5,-2.5,0,2.5,5]])
f, axs = plt.subplots(1, 3, figsize=(10, 6),sharey=True,sharex=True)

#for i in range(len(skip_array)):
    
for j in range(0,2):
        R_x=interest_vectors_batch_tuple[j][0][:,:,2]
        R_y=interest_vectors_batch_tuple[j][0][:,:,3]
        R_z=interest_vectors_batch_tuple[j][0][:,:,4]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)

        sns.kdeplot(np.ravel(R_x),
                    label ="$R_{x},\dot{\gamma}="+str(erate[0])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[1], ax=axs[0])
        sns.kdeplot(np.ravel(R_y),
                    label ="$R_{y},\dot{\gamma}="+str(erate[0])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[2], ax=axs[0])
        sns.kdeplot(np.ravel(R_z),
                    label ="$R_{z},\dot{\gamma}="+str(erate[0])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[3], ax=axs[0])

        axs[0].legend(fontsize=legfont)
        # axs[0].xticks(dist_xticks[0][:])

        R_x=interest_vectors_batch_tuple[j][9][:,:,2]
        R_y=interest_vectors_batch_tuple[j][9][:,:,3]
        R_z=interest_vectors_batch_tuple[j][9][:,:,4]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
       
        sns.kdeplot(np.ravel(R_x),
                    label ="$R_{x},\dot{\gamma}="+str(erate[9])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[1], ax=axs[1])
        sns.kdeplot(np.ravel(R_y),
                    label ="$R_{y},\dot{\gamma}="+str(erate[9])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[2], ax=axs[1])
        sns.kdeplot(np.ravel(R_z),
                    label ="$R_{z},\dot{\gamma}="+str(erate[9])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[3], ax=axs[1])
        axs[1].legend(fontsize=legfont)
        # axs[1].xticks(dist_xticks[1][:])
        R_x=interest_vectors_batch_tuple[j][18][:,:,2]
        R_y=interest_vectors_batch_tuple[j][18][:,:,3]
        R_z=interest_vectors_batch_tuple[j][18][:,:,4]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
        sns.kdeplot(np.ravel(R_x),
                    label ="$R_{x},\dot{\gamma}="+str(erate[18])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[1], ax=axs[2])
        sns.kdeplot(np.ravel(R_y),
                    label ="$R_{y},\dot{\gamma}="+str(erate[18])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[2], ax=axs[2])
        sns.kdeplot(np.ravel(R_z),
                    label ="$R_{z},\dot{\gamma}="+str(erate[18])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[3], ax=axs[2])
        axs[2].legend(fontsize=legfont)
        # axs[2].xticks(dist_xticks[2][:])
        #plt.legend(fontsize=legfont) 
        

plt.yticks(extension_ticks)
f.supxlabel("$\Delta R_{x}$")
f.tight_layout()

plt.savefig(path_2_log_files+"/plots/deltar_x_dist_.pdf",dpi=1200,bbox_inches='tight')
   
plt.show()


#%% y dist 
plt.rcParams["figure.figsize"] = (12,6 )

dist_xticks=([[-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-7.5,-5,-2.5,0,2.5,5]])
f, axs = plt.subplots(1, 3, figsize=(10, 6),sharey=True,sharex=True)

#for i in range(len(skip_array)):
    
for j in range(K.size):
        R_x=interest_vectors_batch_tuple[j][0][:,:,2]
        R_y=interest_vectors_batch_tuple[j][0][:,:,3]
        R_z=interest_vectors_batch_tuple[j][0][:,:,4]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)

        sns.kdeplot(np.ravel(R_y),
                    label ="$\dot{\gamma}="+str(erate[0])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[0])
        axs[0].legend(fontsize=legfont)
        # axs[0].xticks(dist_xticks[0][:])

        R_x=interest_vectors_batch_tuple[j][9][:,:,2]
        R_y=interest_vectors_batch_tuple[j][9][:,:,3]
        R_z=interest_vectors_batch_tuple[j][9][:,:,4]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
       
        sns.kdeplot(np.ravel(R_y),
                    label ="$\dot{\gamma}="+str(erate[9])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[1])
        axs[1].legend(fontsize=legfont)
        # axs[1].xticks(dist_xticks[1][:])
        R_x=interest_vectors_batch_tuple[j][18][:,:,2]
        R_y=interest_vectors_batch_tuple[j][18][:,:,3]
        R_z=interest_vectors_batch_tuple[j][18][:,:,4]
        magnitude_spring=np.sqrt(R_x**2 +R_y**2 + R_z**2)
        sns.kdeplot(np.ravel(R_y),
                    label ="$\dot{\gamma}="+str(erate[18])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[2])
        axs[2].legend(fontsize=legfont)
        # axs[2].xticks(dist_xticks[2][:])
        #plt.legend(fontsize=legfont) 
        

plt.yticks(extension_ticks)

f.supxlabel("$\Delta R_{y}$")
f.tight_layout()

plt.savefig(path_2_log_files+"/plots/deltar_y_dist_.pdf",dpi=1200,bbox_inches='tight')
   
plt.show()




# %%
#%% z dist 
plt.rcParams["figure.figsize"] = (12,6 )

dist_xticks=([[-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-7.5,-5,-2.5,0,2.5,5]])
f, axs = plt.subplots(1, 3, figsize=(10, 6),sharey=True,sharex=True)

#for i in range(len(skip_array)):
    
for j in range(K.size):

        sns.kdeplot(np.ravel(interest_vectors_batch_tuple[j][0][:,:,4]),
                    label ="$\dot{\gamma}="+str(erate[0])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[0])
        axs[0].legend(fontsize=legfont)
        # axs[0].xticks(dist_xticks[0][:])

        
       
        sns.kdeplot(np.ravel(interest_vectors_batch_tuple[j][9][:,:,4]),
                    label ="$\dot{\gamma}="+str(erate[9])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[1])
        axs[1].legend(fontsize=legfont)
        # axs[1].xticks(dist_xticks[1][:])
       
        sns.kdeplot(np.ravel(interest_vectors_batch_tuple[j][18][:,:,4]),
                    label ="$\dot{\gamma}="+str(erate[18])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[2])
        axs[2].legend(fontsize=legfont)
        # axs[2].xticks(dist_xticks[2][:])
        #plt.legend(fontsize=legfont) 
        
        

plt.yticks(extension_ticks)

f.supxlabel("$\Delta R_{z}$")
f.tight_layout()

plt.savefig(path_2_log_files+"/plots/deltar_z_dist_.pdf",dpi=1200,bbox_inches='tight')
   
plt.show()

#%% plotting particle velocity against postion 
for j in range(K.size):
    for i in range(erate.size):


        
        z_position =np.mean(pos_vel_batch_tuple[j][i][:,:,0:3,2],axis=0)
        x_vel=np.mean(pos_vel_batch_tuple[j][i][:,:,3:6,0],axis=0)
        pred_x_vel=erate[i]* z_position
        plt.scatter(z_position,x_vel, label="$\dot{\gamma}="+str(erate[i])+"$")
        plt.plot(z_position,pred_x_vel)
        plt.xlabel("$z$")
        plt.ylabel("$v_{x}$")
        
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()

#%% plotting initial velocity distribution 
f, axs = plt.subplots(1, 3, figsize=(16, 6),sharey=True,sharex=True)
for j in range(K.size):
    # only need the velocity distribution at gammadot=0
    for i in range(1):
        
        x_vel=np.ravel(pos_vel_batch_tuple[j][i][:,:,3:6,0])
        sns.kdeplot( data=x_vel, ax=axs[0])
        
        y_vel=np.ravel(pos_vel_batch_tuple[j][i][:,:,3:6,1])
        sns.kdeplot( data=y_vel, ax=axs[1])
       
        z_vel=np.ravel(pos_vel_batch_tuple[j][i][:,:,3:6,2])
        sns.kdeplot( data=z_vel, ax=axs[2])
       
        plt.show()

#%%fitting gaussian for velocity component 
for j in range(K.size):
    # only need the velocity distribution at gammadot=0
    for i in range(1):
        maxwell = scipy.stats.norm
        data = np.ravel(pos_vel_batch_tuple[j][i][:,:,3:6,0])

        params = maxwell.fit(data)
        print(params)
        # (0, 4.9808603062591041)

        plt.hist(data, bins=20,density=True)
        x = np.linspace(np.min(data),np.max(data),data.size)
        plt.plot(x, maxwell.pdf(x, *params), lw=3)

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
        plt.show()
# %%
