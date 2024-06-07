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
import regex as re
import pandas as pd
import sigfig
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import mmap
import h5py as h5

path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
os.chdir(path_2_post_proc_module)
import seaborn as sns
from log2numpy import *
from dump2numpy import *
import glob 
from post_MPCD_MP_processing_module import *
import pickle as pck

linestyle_tuple = [
     'loosely dotted',
     'dotted',   
     'densely dotted',  
     'long dash with offset', 
     'loosely dashed',     
    'dashed',             
    'densely dashed',        

     'loosely dashdotted',  
     'dashdotted',         
     'densely dashdotted',  

     'dashdotdotted', 
     'loosely dashdotdotted', 
     'densely dashdotdotted']

#%% 
# damp 0.1 seems to work well, need to add window averaging, figure out how to get imposed shear to match velocity of particle
# neeed to then do a huge run 
# damp=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ])
# K=np.array([300.        , 149.99999        , 99.9999        ,  74.9999   ,
#         59.999       ,  49.9999      ,  42.85714286,  37.5       ,
#         33.33333333,  30.       ])

# damp=np.array([0.01, 0.02, 0.03,0.04, 0.05 ])
# K=np.array([300.0        , 149.999     ,99.999,  74.999  ,
#         59.999     ])


damp=np.array([ 0.03633, 0.03633 ])
K=np.array([ 500     , 2000   ,
            ])

# damp=np.array([ 0.03633])
# K=np.array([ 500   ])


strain_total=400

path_2_log_files='/Users/luke_dev/Documents/simulation_test_folder/kdamp_prod_3_init_test_10_reals_200_strain/tuple_results'
path_2_log_files='/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/run_868387_high_T_iteration_moreoutputs/tuple_results'
#path_2_log_files='/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/run_run_694356/tuple_results'

# erate= np.array([0.0075, 0.01  , 0.0125, 0.015 , 0.0175, 0.02  , 0.0225, 0.025 ,
#       0.0275, 0.03])


# # #300 strain 
# no_timesteps=np.flip(np.array([7887000,  15774000,  31548000,  63096000,  70107000,  78870000,
#         90137000, 105160000, 126192000, 157740000]))


erate=np.flip(np.array([1,0.9,0.7,0.5,0.35,0.2,0.1,0.09,0.08,
                0.07,0.06,0.05,0.04,
                0.03,0.0275,0.025,0.0225,
                0.02,0.0175,0.015,0.0125,
                0.01,0.0075,0.005,0.0025,
                0.001,0.00075,0.0005]))

# #600 strain 


no_timesteps=np.flip(np.array([   394000,
          438000,    563000,    789000, 1127000,  1972000,   3944000,   4382000,
         4929000,   5634000,   6573000,   7887000,   9859000,  13145000,
        14340000,  15774000,  17527000,  19718000,  22534000,  26290000,
        31548000,  39435000,  52580000,  78870000, 157740000, 394351000,
       525801000, 788702000]))




thermo_freq=10000
dump_freq=10000
lgf_row_count=np.ceil((no_timesteps/thermo_freq )).astype("int")
dp_row_count=np.ceil((no_timesteps/dump_freq)).astype("int")

thermo_vars='         KinEng         PotEng          Temp          c_bias         TotEng    '
j_=10

eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
damp_ratio=mass_pol/damp



def folder_check_or_create(filepath,folder):
     os.chdir(filepath)
     # combine file name with wd path
     check_path=filepath+"/"+folder
     print((check_path))
     if os.path.exists(check_path) == 1:
          print("file exists, proceed")
          os.chdir(check_path)
     else:
          print("file does not exist, making new directory")
          os.chdir(filepath)
          os.mkdir(folder)
          os.chdir(filepath+"/"+folder)
  
def window_averaging(i,window_size,input_tuple,array_size,outdim1,outdim3):
    
    output_array_final=np.zeros((outdim1,array_size[i],outdim3))
    for k in range(outdim1):
        input_array=input_tuple[i][k,:,:]
        df = pd.DataFrame(input_array)
        output_dataframe=df.rolling(window_size,axis=0).mean()
        output_array_temp=output_dataframe.to_numpy()

        #print(output_array_temp)
        non_nan_size=int(np.count_nonzero(~np.isnan(output_array_temp))/outdim3)
        print("non_nan_size", non_nan_size)
        output_array_final[k,:,:]=output_array_temp

    return output_array_final, non_nan_size

from scipy.optimize import curve_fit
 
def quadfunc(x, a):

    return a*(x**2)

def linearfunc(x,a,b):
    return (a*x)+b 


     
#%%load in tuples 

# os.chdir(path_2_log_files)

# with open('spring_force_positon_tensor_tuple.pickle', 'rb') as f:
#     spring_force_positon_tensor_tuple=pck.load(f)

# with open('COM_velocity_tuple.pickle', 'rb') as f:
#     COM_velocity_tuple=pck.load(f)

# with open('COM_position_tuple.pickle', 'rb') as f:
#     COM_position_tuple=pck.load(f)

# with open('erate_velocity_tuple.pickle', 'rb') as f:
#     erate_velocity_tuple=pck.load(f)

# with open('log_file_tuple.pickle', 'rb') as f:
#     log_file_tuple=pck.load(f)

# with open('pol_velocities_tuple.pickle', 'rb') as f:
#     pol_velocities_tuple=pck.load(f)

# with open('pol_positions_tuple.pickle', 'rb') as f:
#     pol_positions_tuple=pck.load(f)

# with open('ph_velocities_tuple.pickle', 'rb') as f:
#     ph_velocities_tuple=pck.load(f)

# with open('ph_positions_tuple.pickle', 'rb') as f:
#     ph_positions_tuple= pck.load(f)





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


# loading all data into one 
for i in range(K.size):
    label='damp_'+str(damp[i])+'_K_'+str(K[i])+'_'

    spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(batch_load_tuples(label,
                                                            "spring_force_positon_tensor_tuple.pickle"),)
    erate_velocity_batch_tuple=erate_velocity_batch_tuple+(batch_load_tuples(label,
                                                            "erate_velocity_tuple.pickle"),)
    COM_velocity_batch_tuple=COM_velocity_batch_tuple+(batch_load_tuples(label,
                                                            "COM_velocity_tuple.pickle"),)
    conform_tensor_batch_tuple=conform_tensor_batch_tuple+(batch_load_tuples(label,
                                                            "conform_tensor_tuple.pickle"),)
    log_file_batch_tuple=log_file_batch_tuple+(batch_load_tuples(label,
                                                            "log_file_tuple.pickle"),)
    area_vector_spherical_batch_tuple=area_vector_spherical_batch_tuple+(batch_load_tuples(label,"area_vector_spherical_tuple.pickle"),)
    
    interest_vectors_batch_tuple=interest_vectors_batch_tuple+(batch_load_tuples(label,
                                                                                 "interest_vectors_tuple.pickle"))

    


     

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



folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=4  
final_temp=np.zeros((erate.size))
mean_temp_array=np.zeros((K.size,erate.size))
for j in range(K.size):
    for i in range(erate.size):
        
        # plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
        mean_temp_array[j,i]=np.mean(log_file_batch_tuple[j][i][:,column])
      
        #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
    #     plt.ylabel("$T$", rotation=0)
    #     plt.xlabel("$\gamma$")
    

    # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
    #     plt.show()

#

marker=['x','o','+','^',"1","X","d","*","P","v"]
for j in range(K.size):
    plt.scatter(erate,mean_temp_array[j,:],label="$K="+str(K[j])+"$" ,marker=marker[j])
    plt.ylabel("$T$", rotation=0)
    plt.xlabel('$\dot{\gamma}$')
    plt.xscale('log')
   # plt.yscale('log')
plt.axhline(1,label="$T_{0}=1$")
plt.legend()
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/temp_vs_erate.pdf",dpi=1200,bbox_inches='tight')


plt.show()


      


#%% strain points for viscoelastic data 
strainplot_tuple=()
for i in range(erate.size):
     strain_plotting_points= np.linspace(0,strain_total,spring_force_positon_tensor_batch_tuple[0][i].shape[0])
   
     strainplot_tuple=strainplot_tuple+(strain_plotting_points,)  
     print(strainplot_tuple[i].size)


COM_mean_vel=np.zeros((K.size,erate.size))
erate_mean_vel=np.zeros((K.size,erate.size))
for j in range(K.size):
    for i in range(erate.size):

        COM_mean_vel[j,i]=np.mean(COM_velocity_batch_tuple[j][i][:,0])
        erate_mean_vel[j,i]=np.mean(erate_velocity_batch_tuple[j][i][:,0])

for j in range(K.size):
    popt,pcov=curve_fit(linearfunc,COM_mean_vel[j,:],erate_mean_vel[j,:])
    plt.plot(COM_mean_vel,(popt[0]*(COM_mean_vel)+popt[1]))
    plt.scatter(COM_mean_vel,erate_mean_vel,label="$\\frac{d v_{x,\dot{\gamma}}}{d v_{x,COM}}="+str(sigfig.round(popt[0],sigfigs=3))+",c="+str(sigfig.round(popt[1],sigfigs=3))+"$", marker=marker[j])
    plt.xlabel("$v_{x,COM}$")
    plt.ylabel("$v_{x,\dot{\gamma}}$", rotation=0, labelpad=10)
    plt.legend()
    plt.tight_layout()

    #  plt.axhline(np.mean(COM_velocity_tuple[i][:,0]/erate_velocity_tuple[i][:,0]))
plt.savefig(path_2_log_files+"/plots/vcom_vs_verate.pdf",dpi=1200,bbox_inches='tight')       
plt.show()




#%% look at internal stresses
folder="stress_tensor_plots"
folder_check_or_create(path_2_log_files,folder)
labels_stress=["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{yz}$"]

# for i in range(erate.size):
#     for j in range(3,6):
#         #plt.plot(strainplot_tuple[i],spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
#         #plt.plot(spring_force_positon_tensor_tuple_wa[i][:,j], label=labels_stress[j]+",$\dot{\gamma}="+str(erate[i])+"$")
#         plt.plot(viscoelastic_stress_tuple_wa[i][:,j], label=labels_stress[j]+",$\dot{\gamma}="+str(erate[i])+"$")
#     plt.legend()
#     plt.show()

# #%% normal stress
for j in range(K.size):
    for i in range(erate.size):
        
        for k in range(0,3):
            #plt.plot(strainplot_tuple[i],spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
            plt.plot(strainplot_tuple[i],spring_force_positon_tensor_batch_tuple[j][i][:,k], label=labels_stress[j]+",$\dot{\gamma}="+str(erate[i])+"$")
            #plt.plot(viscoelastic_stress_tuple_w[i][:,j], label=labels_stress[j]+",$\dot{\gamma}="+str(erate[i])+"$")
            #plt.ylim(-8000,10) 
            plt.legend()
        plt.show()

#%%
cutoff_ratio=0.1
end_cutoff_ratio=1
j_point_1=0
j_point_2=K.size
folder="N_1_plots"

N_1_mean=np.zeros((K.size,erate.size))
N_1_mean_std_dev=np.zeros((K.size,erate.size))
for j in range(j_point_1,j_point_2):
    for i in range(erate.size):

        cutoff= int(np.ceil(cutoff_ratio*(spring_force_positon_tensor_batch_tuple[j][i][:-1,0].size)))
        end_cutoff=int(np.ceil(end_cutoff_ratio*(spring_force_positon_tensor_batch_tuple[j][i][:-1,0].size)))
        #cutoff=int(nan_size[i]) +int(np.ceil(cutoff_ratio*(spring_force_positon_tensor_tuple_wa[i][:-1,0].size-nan_size[i])))

        #N_1=spring_force_positon_tensor_tuple_wa[i][cutoff:-1,0]-spring_force_positon_tensor_tuple_wa[i][cutoff:end_cutoff,2]
        N_1=spring_force_positon_tensor_batch_tuple[j][i][cutoff:end_cutoff-1,0]-spring_force_positon_tensor_batch_tuple[j][i][cutoff:end_cutoff-1,2]
    
        N_1_mean[j,i]=np.mean(N_1[:])
        N_1_mean_std_dev[j,i]=np.mean(N_1[:])
        
        print(N_1_mean)
        
        plt.plot(strainplot_tuple[i][cutoff:end_cutoff-1],N_1,label="$\dot{\gamma}="+str(erate[i])+"$")
        #plt.plot(N_1, label="$\dot{\gamma}="+str(erate[i])+"$")
        plt.axhline(np.mean(N_1))
        plt.ylabel("$N_{1}$")
        plt.xlabel("$\gamma$")
        plt.legend()
        plt.show()

#%% 
for j in range(j_point_1,j_point_2):
   # plt.scatter((erate[:]),N_1_mean[j,:],label="$K="+str(K[j])+"$" ,marker=marker[j])
    plt.errorbar((erate[:]),N_1_mean[j,:],yerr= np.abs(N_1_mean_std_dev[j,:]/np.sqrt(j_)),label="$K="+str(K[j])+"$" ,marker=marker[j])
    #popt,pcov=curve_fit(linearfunc,erate[:],N_1_mean[j,:])
    #plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))
    plt.ylabel("$N_{1}$", rotation=0, labelpad=20)
    plt.xlabel("$\dot{\gamma}$")
    # plt.xscale('log')
    # plt.yscale('log')
plt.tight_layout()
plt.legend()
#plt.savefig(path_2_log_files+"/plots/N1_vs_logscale_erate.pdf",dpi=1200)
plt.savefig(path_2_log_files+"/plots/N1_vs_erate.pdf",dpi=1200)
plt.show()

#%% 
for j in range(j_point_1,j_point_2):
    plt.scatter((erate[:]),N_1_mean[j,:]/K[j],label="$K="+str(K[j])+"$" ,marker=marker[j])
    #plt.errorbar((erate[:]),N_1_mean[j,:]/K[j],yerr= N_1_mean_std_dev[j,:]/np.sqrt(j_),label="$K="+str(K[j])+"$" ,marker=marker[j])
    #popt,pcov=curve_fit(linearfunc,erate[:],N_1_mean[j,:])
    #plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))
    plt.ylabel("$\\frac{N_{1}}{K}$", rotation=0, labelpad=20)
    plt.xlabel("$\dot{\gamma}$")
    # plt.xscale('log')
    # plt.yscale('log')
plt.tight_layout()
plt.legend()
#plt.savefig(path_2_log_files+"/plots/N1_vs_logscale_erate.pdf",dpi=1200)
plt.savefig(path_2_log_files+"/plots/N1_K_vs_erate.pdf",dpi=1200)
plt.show()
#%% against temp

for j in range(j_point_1,j_point_2):
    plt.scatter((mean_temp_array[j,:]),N_1_mean[j,:]/K[j],label="$K="+str(K[j])+"$" ,marker=marker[j])
    #popt,pcov=curve_fit(linearfunc,erate[:],N_1_mean[j,:])
    #plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))
    plt.ylabel("$\\frac{N_{1}}{K}$", rotation=0)
    plt.xlabel("$T$")
    # plt.xscale('log')
    # plt.yscale('log')
plt.tight_layout()
plt.legend()
#plt.savefig(path_2_log_files+"/plots/N1_vs_logscale_erate.pdf",dpi=1200)
plt.savefig(path_2_log_files+"/plots/N1_K_vs_T.pdf",dpi=1200)
plt.show()

 #%%
cutoff_ratio=0.1
end_cutoff_ratio=1
folder="N_2_plots"

N_2_mean=np.zeros((K.size,erate.size))
N_2_mean_std_dev=np.zeros((K.size,erate.size))
for j in range(j_point_1,j_point_2):
    for i in range(erate.size):

        cutoff= int(np.ceil(cutoff_ratio*(spring_force_positon_tensor_batch_tuple[j][i][:-1,0].size)))
        end_cutoff=int(np.ceil(end_cutoff_ratio*(spring_force_positon_tensor_batch_tuple[j][i][:-1,0].size)))
        #cutoff=int(nan_size[i]) +int(np.ceil(cutoff_ratio*(spring_force_positon_tensor_tuple_wa[i][:-1,0].size-nan_size[i])))

        #N_1=spring_force_positon_tensor_tuple_wa[i][cutoff:-1,0]-spring_force_positon_tensor_tuple_wa[i][cutoff:end_cutoff,2]
        N_2=spring_force_positon_tensor_batch_tuple[j][i][cutoff:end_cutoff-1,2]-spring_force_positon_tensor_batch_tuple[j][i][cutoff:end_cutoff-1,1]
      
        N_2_mean[j,i]=np.mean(N_2[:])
        N_2_mean_std_dev[j,i]=np.mean(N_2[:])
        print(N_2_mean)
        
        plt.plot(strainplot_tuple[i][cutoff:end_cutoff-1],N_2,label="$\dot{\gamma}="+str(erate[i])+"$")
        #plt.plot(N_1, label="$\dot{\gamma}="+str(erate[i])+"$")
        plt.axhline(np.mean(N_2))
        plt.ylabel("$N_{2}$")
        plt.xlabel("$\gamma$")
        plt.legend()
        plt.show()

#%%
for j in range(j_point_1,j_point_2):
    plt.scatter((erate[:]),(N_2_mean[j,:]),label="$K="+str(K[j])+"$" ,marker=marker[j])
    plt.errorbar((erate[:]),N_2_mean[j,:],yerr= np.abs(N_2_mean_std_dev[j,:]/np.sqrt(j_)),label="$K="+str(K[j])+"$" ,marker=marker[j])
   # popt,pcov=curve_fit(linearfunc,erate[:],N_2_mean[j,:])
    #plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))
    # popt,pcov=curve_fit(quadfunc,erate[:],N_2_mean[j,:])
    # plt.plot(erate[:],(popt[0]*(erate[:]**2)))  
    plt.ylabel("$N_{2}$",rotation=0)
    plt.xlabel("$\dot{\gamma}$")
   # plt.xscale('log')
    #plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.savefig(path_2_log_files+"/plots/N2_vs_erate.pdf",dpi=1200)
#plt.savefig(path_2_log_files+"/plots/N2_vs_logscale_erate.pdf",dpi=1200)
plt.show()
#%%
for j in range(j_point_1,j_point_2):
    plt.scatter((erate[:]),(N_2_mean[j,:]/K[j]),label="$K="+str(K[j])+"$" ,marker=marker[j])
   # popt,pcov=curve_fit(linearfunc,erate[:],N_2_mean[j,:])
    #plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))
    # popt,pcov=curve_fit(quadfunc,erate[:],N_2_mean[j,:])
    # plt.plot(erate[:],(popt[0]*(erate[:]**2)))  
    plt.ylabel("$\\frac{N_{2}}{K}$",rotation=0, labelpad=20)
    plt.xlabel("$\dot{\gamma}$")
   # plt.xscale('log')
    #plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.savefig(path_2_log_files+"/plots/N2_K_vs_erate.pdf",dpi=1200)
#plt.savefig(path_2_log_files+"/plots/N2_vs_logscale_erate.pdf",dpi=1200)
plt.show()

#%%
folder="shear_stress_plots"
cutoff_ratio=0.1
end_cutoff_ratio=1
xz_shear_stress_mean=np.zeros((K.size,erate.size))
xz_shear_stress_std_dev=np.zeros((K.size,erate.size))
xz_shear_stress_rel_err=np.zeros((K.size,erate.size))

for j in range(j_point_1,j_point_2):
    for i in range(erate.size):
        # cutoff=int(nan_size[i]) +int(np.ceil(cutoff_ratio*(spring_force_positon_tensor_tuple_wa[i][:-1,0].size-nan_size[i])))
        # xz_shear_stress= spring_force_positon_tensor_tuple_wa[i][cutoff:,3]
        cutoff= int(np.ceil(cutoff_ratio*(spring_force_positon_tensor_batch_tuple[j][i][:-1,0].size)))
        end_cutoff=int(np.ceil(end_cutoff_ratio*(spring_force_positon_tensor_batch_tuple[j][i][:-1,0].size)))
        xz_shear_stress=spring_force_positon_tensor_batch_tuple[j][i][cutoff:end_cutoff,3]
     

        xz_shear_stress_mean[j,i]=np.mean(xz_shear_stress[:])
        xz_shear_stress_std_dev[j,i]=np.std(xz_shear_stress[:])
        xz_shear_stress_rel_err[j,i]= xz_shear_stress_std_dev[j,i]/xz_shear_stress_mean[j,i]

        #plt.plot(strainplot_tuple[i][:],xz_shear_stress, label=labels_stress[3])
        plt.plot(strainplot_tuple[i][cutoff:end_cutoff],xz_shear_stress, label=labels_stress[3]+",$\dot{\gamma}="+str(erate[i])+"$")
        #plt.axhline(xz_shear_stress_mean[i])
        plt.ylabel("$\sigma_{xz}$")
        plt.xlabel("$\gamma$")
        plt.legend()
    plt.show()
#%%
for j in range(j_point_1,j_point_2):
    plt.plot(erate[:],xz_shear_stress_mean[j,:],label="$K="+str(K[j])+"$" ,marker=marker[j])
    plt.errorbar(erate[:],xz_shear_stress_mean[j,:],yerr= ( xz_shear_stress_std_dev[j,:]/np.sqrt(j_)),label="$K="+str(K[j])+"$" ,marker=marker[j])
    
    plt.ylabel("$\sigma_{xz}$",rotation=0, labelpad=10)
    plt.xlabel("$\dot{\gamma}$")
    # plt.ylim(0,10)
    
    

plt.tight_layout()
plt.legend()
plt.savefig(path_2_log_files+"/plots/sigma_xz_vs_erate.pdf",dpi=1200)
plt.show()

for j in range(j_point_1,j_point_2):
    #plt.plot(erate[:],xz_shear_stress_mean[j,:]/(erate),label="$K="+str(K[j])+"$" ,marker=marker[j])
    plt.errorbar(erate[:],xz_shear_stress_mean[j,:]/(erate),
                 yerr= np.abs(xz_shear_stress_mean[j,:]/(erate))*xz_shear_stress_rel_err[j,:]/np.sqrt(j_),label="$K="+str(K[j])+"$" ,marker=marker[j])
    
    plt.ylabel("$\eta$",rotation=0)
    plt.xlabel("$\dot{\gamma}$")
    plt.xscale('log')
    #plt.yscale('log')
    # plt.ylim(0,10)
    
    

plt.tight_layout()
plt.legend()
plt.savefig(path_2_log_files+"/plots/eta_vs_erate.pdf",dpi=1200)
plt.show()


for j in range(j_point_1,j_point_2):
    plt.plot(erate[:],xz_shear_stress_mean[j,:]/(erate*K[j]),label="$K="+str(K[j])+"$" ,marker=marker[j])
    plt.ylabel("$\\frac{\eta}{K}$",rotation=0)
    plt.xlabel("$\dot{\gamma}$")
    plt.xscale('log')
    plt.yscale('log')
    # plt.ylim(0,10)
    
    

plt.tight_layout()
plt.legend()
plt.savefig(path_2_log_files+"/plots/etaK_vs_erate.pdf",dpi=1200)
plt.show()

   

#%% plot N_1 and N_2 together

for j in range(j_point_1,j_point_2):
    plt.scatter((erate[:]),N_1_mean[j,:],label="$N_{1},K="+str(K[j])+"$" ,marker=marker[j])
    plt.scatter((erate[:]),(-1.25*N_2_mean[j,:]),label="$-1.25N_{2},K="+str(K[j])+"$" ,marker=marker[j])
   # popt,pcov=curve_fit(linearfunc,erate[:],N_2_mean[j,:])
    #plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))
    # popt,pcov=curve_fit(quadfunc,erate[:],N_2_mean[j,:])
    # plt.plot(erate[:],(popt[0]*(erate[:]**2)))  
    #plt.ylabel("$N_{2}$")
    plt.xlabel("$\dot{\gamma}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

plt.savefig(path_2_log_files+"/plots/n1_scaled_n2_vs_erate.pdf",dpi=1200)
plt.show()



#%%
for j in range(j_point_1,j_point_2):
    plt.scatter((erate[:]),np.abs(N_1_mean[j,:]/N_2_mean[j,:]),label="$K="+str(K[j])+"$" ,marker=marker[j])
   
   # popt,pcov=curve_fit(linearfunc,erate[:],N_2_mean[j,:])
    #plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))
    # popt,pcov=curve_fit(quadfunc,erate[:],N_2_mean[j,:])
    # plt.plot(erate[:],(popt[0]*(erate[:]**2)))  
    plt.ylabel("$|N_{1}/N_{2}|$", rotation=0, labelpad=20)
  
    plt.xlabel("$\dot{\gamma}$")
    plt.xscale('log')
    #plt.yscale('log')
    
plt.axhline(1,label="$|N_{1}/N_{2}|=1$",linestyle='--')
plt.axhline(np.mean(np.abs(N_1_mean[j,3:]/N_2_mean[j,3:])),
            label="$\\bar{|N_{1}/N_{2}}|="+str(sigfig.round(np.mean(np.abs(N_1_mean[j,2:]/N_2_mean[j,2:])),sigfigs=3))+"$",
            linestyle=':')

plt.legend()
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/n1_div_n2_vs_erate.pdf",dpi=1200)
plt.show()






#%% density plot using seaborn distplot theta
# could add fittings to this run 
import seaborn as sns
os.chdir(path_2_log_files+"/plots/")
K_string=K.astype("str")
#os.mkdir("distributions")
bin_count = int(np.ceil(np.log2((j_))) + 1)# sturges rule


pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
for j in range(K.size):
    for i in range(0,7):
               
          
            
                 
            x=area_vector_spherical_batch_tuple[j][i][:,0]
            y=area_vector_spherical_batch_tuple[j][i][:,1]
           
            z=area_vector_spherical_batch_tuple[j][i][:,2]
    
    # Draw the density plot
            sns.kdeplot(data=y, label ="$\dot{\gamma}="+str(erate[i])+"$" )
          
    plt.xlabel("$\Theta$")
    plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
    plt.ylabel('Density')

    plt.xlim(-np.pi,np.pi)
    plt.legend()
    
# Plot formatting
    plt.savefig(path_2_log_files+"/plots/distributions/Azimuthal_K_"+str(K[j])+"_erate_"+str(erate[0])+"_"+str(erate[7])+".pdf",dpi=1200)
    plt.show(sns)

for j in range(K.size):
    for i in range(7,14):
               
          
            
                 
            x=area_vector_spherical_batch_tuple[j][i][:,0]
            y=area_vector_spherical_batch_tuple[j][i][:,1]
           
            z=area_vector_spherical_batch_tuple[j][i][:,2]
    
    # Draw the density plot
            sns.kdeplot(data=y, label ="$\dot{\gamma}="+str(erate[i])+"$" )
          
    plt.xlabel("$\Theta$")
    plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
    plt.ylabel('Density')
    plt.xlim(-np.pi,np.pi)
    plt.legend()
    
    plt.savefig(path_2_log_files+"/plots/distributions/Azimuthal_K_"+\
               str(K[j])+"_erate_"+str(erate[7])+"_"+str(erate[14])+".pdf",dpi=1200)
    plt.show(sns)

for j in range(K.size):
    for i in range(14,21):
               
          
            
                 
            x=area_vector_spherical_batch_tuple[j][i][:,0]
            y=area_vector_spherical_batch_tuple[j][i][:,1]
           
            z=area_vector_spherical_batch_tuple[j][i][:,2]
    
    # Draw the density plot
            sns.kdeplot(data=y, label ="$\dot{\gamma}="+str(erate[i])+"$" )
          
    plt.xlabel("$\Theta$")
    plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
    plt.ylabel('Density')
    plt.xlim(-np.pi,np.pi)
    plt.legend()
    plt.savefig(path_2_log_files+"/plots/distributions/Azimuthal_K_"+\
               str(K[j])+"_erate_"+str(erate[14])+"_"+str(erate[21])+".pdf",dpi=1200)
    
# Plot formatting
    plt.show(sns)

for j in range(K.size):
    for i in range(21,27):
               
          
            
                 
            x=area_vector_spherical_batch_tuple[j][i][:,0]
            y=area_vector_spherical_batch_tuple[j][i][:,1]
           
            z=area_vector_spherical_batch_tuple[j][i][:,2]
    
    # Draw the density plot
            sns.kdeplot(data=y, label ="$\dot{\gamma}="+str(erate[i])+"$" )
          
    plt.xlabel("$\Theta$")
    plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
    plt.xlim(-np.pi,np.pi)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(path_2_log_files+"/plots/distributions/Azimuthal_K_"+\
               str(K[j])+"_erate_"+str(erate[21])+"_"+str(erate[27])+".pdf",dpi=1200)
    
# Plot formatting
    plt.show()







#%% density plot using seaborn distplot phi
pi_phi_ticks=[ 0,np.pi/4, np.pi/2]
pi_phi_tick_labels=[ '0','π/4', 'π/2']
for j in range(K.size):
    for i in range(0,7):
               
          
            
                 
            x=area_vector_spherical_batch_tuple[j][i][:,0]
            y=area_vector_spherical_batch_tuple[j][i][:,1]
           
            z=area_vector_spherical_batch_tuple[j][i][:,2]
    
    # Draw the density plot
            sns.kdeplot(data=z, label ="$\dot{\gamma}="+str(erate[i])+"$" )
          
    plt.xlabel("$\Phi$")
    plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(0,np.pi/2)
# Plot formatting
    plt.savefig(path_2_log_files+"/plots/distributions/inclination_K_"+str(K[j])+"_erate_"+str(erate[0])+"_"+str(erate[7])+".pdf",dpi=1200)
    plt.show(sns)

for j in range(K.size):
    for i in range(7,14):
               
          
            
                 
            x=area_vector_spherical_batch_tuple[j][i][:,0]
            y=area_vector_spherical_batch_tuple[j][i][:,1]
           
            z=area_vector_spherical_batch_tuple[j][i][:,2]
    
    # Draw the density plot
            sns.kdeplot(data=z, label ="$\dot{\gamma}="+str(erate[i])+"$" )
          
    plt.xlabel("$\Phi$")
    plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(0,np.pi/2)
    
    plt.savefig(path_2_log_files+"/plots/distributions/inclination_K_"+\
               str(K[j])+"_erate_"+str(erate[7])+"_"+str(erate[14])+".pdf",dpi=1200)
    plt.show(sns)

for j in range(K.size):
    for i in range(14,21):
               
          
            
                 
            x=area_vector_spherical_batch_tuple[j][i][:,0]
            y=area_vector_spherical_batch_tuple[j][i][:,1]
           
            z=area_vector_spherical_batch_tuple[j][i][:,2]
    
    # Draw the density plot
            sns.kdeplot(data=z, label ="$\dot{\gamma}="+str(erate[i])+"$" )
          
    plt.xlabel("$\Phi$")
    plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
    plt.ylabel('Density')
    plt.xlim(0,np.pi/2)
    plt.legend()
    plt.savefig(path_2_log_files+"/plots/distributions/inclination_K_"+\
               str(K[j])+"_erate_"+str(erate[14])+"_"+str(erate[21])+".pdf",dpi=1200)
    
# Plot formatting
    plt.show(sns)

for j in range(K.size):
    for i in range(21,27):
               
          
            
                 
            x=area_vector_spherical_batch_tuple[j][i][:,0]
            y=area_vector_spherical_batch_tuple[j][i][:,1]
           
            z=area_vector_spherical_batch_tuple[j][i][:,2]
    
    # Draw the density plot
            sns.kdeplot(data=z, label ="$\dot{\gamma}="+str(erate[i])+"$" )
          
    plt.xlabel("$\phi$")
    plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
    plt.ylabel('Density')
    plt.xlim(0,np.pi/2)
    plt.legend()
    plt.savefig(path_2_log_files+"/plots/distributions/inclination_K_"+\
               str(K[j])+"_erate_"+str(erate[21])+"_"+str(erate[27])+".pdf",dpi=1200)
    
# Plot formatting
    plt.show()




#%% spring extension distributions


for j in range(0,27):
        
        spring_extensions=np.zeros((len(interest_vectors_batch_tuple[j]),3))
        
        for i in range(0,len(interest_vectors_batch_tuple[j])):

     
            spring_extensions[i,0]=np.sqrt(np.sum(interest_vectors_batch_tuple[j][i][2,:]**2))-eq_spring_length
            spring_extensions[i,1]=np.sqrt(np.sum(interest_vectors_batch_tuple[j][i][3,:]**2))-eq_spring_length
            spring_extensions[i,2]=np.sqrt(np.sum(interest_vectors_batch_tuple[j][i][4,:]**2))-eq_spring_length
 
        sns.kdeplot(data=spring_extensions)

        plt.xlabel("$\Delta x$")
    
        plt.ylabel('Density')

        plt.legend()
   # plt.savefig(path_2_log_files+"/plots/distributions/spring_1_extension_K_"+\
      #         str(K[j])+"_erate_"+str(erate[21])+"_"+str(erate[27])+".pdf",dpi=1200)
    
# Plot formatting
plt.show()

#%%
for j in range(27,56):
        spring_extensions=np.zeros((len(interest_vectors_batch_tuple[j]),3))
        
        for i in range(0,len(interest_vectors_batch_tuple[j])):

     
            spring_extensions[i,0]=np.sqrt(np.sum(interest_vectors_batch_tuple[j][i][2,:]**2))-eq_spring_length
            spring_extensions[i,1]=np.sqrt(np.sum(interest_vectors_batch_tuple[j][i][3,:]**2))-eq_spring_length
            spring_extensions[i,2]=np.sqrt(np.sum(interest_vectors_batch_tuple[j][i][4,:]**2))-eq_spring_length
 
        sns.kdeplot(data=spring_extensions)

        plt.xlabel("$\Delta x$")
    
        plt.ylabel('Density')

        plt.legend()
   # plt.savefig(path_2_log_files+"/plots/distributions/spring_1_extension_K_"+\
      #         str(K[j])+"_erate_"+str(erate[21])+"_"+str(erate[27])+".pdf",dpi=1200)
    
# Plot formatting
plt.show()
            
           


# %%
