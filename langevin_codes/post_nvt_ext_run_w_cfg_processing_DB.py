##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code processes data from the continuous extension simulations of dumbells utlising cfg files to 
apply the correct transformation to the dump outputs


"""
#%% 
import os
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import regex as re
import pandas as pd
import sigfig
plt.rcParams.update(plt.rcParamsDefault)
# plt.rcParams["figure.figsize"] = (8,6 )
# plt.rcParams.update({'font.size': 16})
#plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import mmap
import h5py as h5
import math as m 
import glob 
from reading_lammps_module import *
from fitter import Fitter, get_common_distributions, get_distributions
path_2_post_proc_module= '/Users/luke_dev/Documents/molecular_dynamics_post_processing_codes/MPCD_codes/'
os.chdir(path_2_post_proc_module)
import seaborn as sns
import glob 
from post_MPCD_MP_processing_module import *
from post_langevin_module import * 

import pickle as pck
from numpy.linalg import norm
from numpy.linalg import inv as matinv

#%% simulation parameters
damp=0.035
strain_total=100

erate=np.array([0.005 , 0.01  , 0.02  , 0.04  , 0.06  , 0.08  , 0.1   , 0.125 ,
       0.15  , 0.175 , 0.2   , 0.3   , 0.325 , 0.3375, 0.35  , 0.355 ,
       0.36  , 0.365 , 0.37  , 0.3725, 0.375 , 0.4   , 0.45  , 0.5 ])

no_timesteps=np.array([197175000,  98588000, 492939000, 246469000, 164313000, 123235000,
        98588000, 788702000, 657252000, 563359000, 492939000, 328626000,
       303347000, 292112000, 281679000, 277712000, 273855000, 270103000,
       266453000, 264665000, 262901000, 246469000, 219084000, 197175000])

timestep_multiplier=np.array(
[0.00005,0.00005,0.00005,0.00005,
0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
0.0005,0.0005,0.0005,0.0005,0.0005,0.005,
0.005])*4
md_step=0.005071624521210362*timestep_multiplier
thermo_vars='         KinEng         PotEng         Press           Temp         Ecouple       Econserve    c_uniaxnvttemp'

dump_start_line ='ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]'
dump_start_line_posvel = "ITEM: ATOMS id type x y z vx vy vz"

K=200
j_=10
box_size=100
eq_spring_length=0.25
mass_pol=5 
n_plates=100
n_particles=2*n_plates
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/cfg_run/cfg_run/"
path_2_log_files=filepath
Path_2_dump=filepath

# file name search strings

log_general_name_string="log.*K_"+str(K)
cfg_general_name_before_string="before*.cfg"
cfg_general_name_after_string="after*K_"+str(K)+"_*.cfg"
log_general_name_string=("log.**K_"+str(K))
force_dump_general_name_string="*_dump_*K_"+str(K)+".dump"
posvel_dump_general_name_string="*_tensor_*K_"+str(K)+".dump"


#%% grabbing file names and organising 


# grab file names 
(realisation_name_posvel_dump,
 realisation_name_cfg_before,
 count_mom,count_phantom,
 realisation_name_log,
 count_log,
 realisation_name_force_dump,
 count_dump,
 realisation_name_cfg_after,
 count_pol)= VP_and_momentum_data_realisation_name_grabber(cfg_general_name_after_string,
                                                                     log_general_name_string,
                                                                     cfg_general_name_before_string,
                                                                     force_dump_general_name_string,
                                                                     path_2_log_files,
                                                                     posvel_dump_general_name_string)


realisations_for_sorting_after_cfg=[]
# first sort cfg via realisation and erate 
realisation_split_index=7
erate_index=16
realisation_name_after_sorted_final_cfg=org_names(realisations_for_sorting_after_cfg,
                                                      realisation_name_cfg_after,
                                                     realisation_split_index,
                                                     erate_index)
# then sort cfg via timestep and erate 
timestep_split_index=21
erate_index=16
realisations_for_sorting_after_cfg=[]
realisation_name_after_sorted_final_cfg=org_names(realisations_for_sorting_after_cfg,
                                                      realisation_name_after_sorted_final_cfg,
                                                     timestep_split_index,
                                                     erate_index)

# grabbing log files and sorting 
realisation_split_index=6
erate_index=15
realisations_for_sorting_after_log=[]
realisation_name_log_sorted_final=org_names(realisations_for_sorting_after_log,
                                                     realisation_name_log,
                                                     realisation_split_index,
                                                     erate_index)

# grabbing force dump file and sorting 
realisations_for_sorting_force_dump=[]
realisation_name_force_dump_sorted_final=org_names(realisations_for_sorting_force_dump,
                                                     realisation_name_force_dump,
                                                     realisation_split_index,
                                                     erate_index)
# grabbing position velocity dump and sorting 

realisations_for_sorting_posvel_dump=[]
realisation_name_posvel_dump_sorted_final=org_names(realisations_for_sorting_posvel_dump,
                                                     realisation_name_posvel_dump,
                                                     realisation_split_index,
                                                     erate_index)


print(len(realisation_name_after_sorted_final_cfg))
# the following lists should be equal length 
print(len(realisation_name_log_sorted_final))
print(len(realisation_name_force_dump_sorted_final))
print(len(realisation_name_posvel_dump_sorted_final))


#%% file processing loop
Path_2_dump=filepath

# tuples for storing info 
spring_force_positon_tensor_tuple=()
log_file_tuple=()
dir_vector_tuple=()
new_pos_vel_tuple=()

e_in=0
e_end=erate.size
count=e_in

for i in range(e_in,e_end):
    i_=(count*j_)
    print("i_",i_)
    i_=0


    
    outputdim_force_dump=dump2numpy_tensor_1tstep(dump_start_line,
                                    Path_2_dump,
                                     realisation_name_force_dump_sorted_final[i_],
                                     n_plates,100,6).shape[0]

    outputdim_posvel_dump=int(dump2numpy_f(dump_start_line_posvel,
                                    Path_2_dump,
                                    realisation_name_posvel_dump_sorted_final[i_],
                                    n_plates*2).shape[0]/(n_plates*2))

    # outputdim_cfg_dump=cfg2numpy_coords(Path_2_dump,realisation_name_after_sorted_final_cfg[i_],
    #                   n_plates*2, 4).shape[0]
    
    outputdim_log=log2numpy_reader(realisation_name_log_sorted_final[i_],
                                                    path_2_log_files,
                                                    thermo_vars).shape[0]
    
    spring_force_positon_array=np.zeros((j_,outputdim_force_dump,100,6))
    pos_vel_array=((j_,outputdim_posvel_dump,8))
    dirn_vector_array=np.zeros((j_,outputdim_force_dump,100,3))
    log_file_array=np.zeros((j_,outputdim_log,8))


    for j in range(j_):
            j_index=j+(j_*count)

            log_file_array[j,:,:]=log2numpy_reader(realisation_name_log_sorted_final[j_index],
                                                        path_2_log_files,
                                                        thermo_vars)

            box_coords_from_dump=dump2numpy_box_coords_1tstep( Path_2_dump,realisation_name_posvel_dump_sorted_final[j_index],n_plates)
            posvel_from_dump_all=dump2numpy_f(dump_start_line_posvel,
                                    Path_2_dump,
                                    realisation_name_posvel_dump_sorted_final[j_index],
                                    n_plates*2).astype("float")
            posvel_from_dump_all=np.reshape(posvel_from_dump_all,(1000,200,8))

            force_dirn_dump=dump2numpy_tensor_1tstep(dump_start_line,
                                    Path_2_dump,
                                    realisation_name_force_dump_sorted_final[j_index],
                                    n_plates,100,6)
            db_forces=force_dirn_dump[:,:,0:3]
            db_dirns=force_dirn_dump[:,:,3:6]
           
            list_box_vec=[]
            box_vec_array_dump_unsort=np.zeros((1000,3,3))
            box_vec_array_dump=np.zeros((1000,3,3))


            count_cfg=0
            cfg_indices=np.arange(j,10000+j,10)
            for k in range(1000):
                cfg_k_index=cfg_indices[k]
            
                #accquiring dump box vector for timestep k
                

                box_vec_list_dump=dump2numpy_box_coords_1tstep( Path_2_dump,
                                                                realisation_name_posvel_dump_sorted_final[j_index],
                                                                200)[k][5:8]
                
                #accquire cfg box vector and coordinates
                full_cfg_coord_after,box_vector_array_cfg=cfg2numpy_coords(Path_2_dump,
                                                                            realisation_name_after_sorted_final_cfg[cfg_k_index],
                                                                            n_plates*2, 4)
                full_cfg_coord_after_sorted=full_cfg_coord_after[full_cfg_coord_after[:,3].argsort()]
                posvel_from_dump_sing=posvel_from_dump_all[k]
                posvel_from_dump_sing_sorted=posvel_from_dump_sing[posvel_from_dump_sing[:,0].argsort()]
                
                
                box_vec_array_dump=np.zeros((3,3))
                box_vec_array_upper_tri=np.zeros((3,3))
                
                # putting dump box vectors into upper triangular matrix 
                # this can be a function 
                # def bounds_2_upper_tri()
                for l in range(3):
                    box_vec_array_dump[l,:]= box_vec_list_dump[l].split(" ")
                    xy= box_vec_array_dump[0,2]
                    xz= box_vec_array_dump[1,2]
                    yz= box_vec_array_dump[2,2]
                    xlo= box_vec_array_dump[0,0]-np.min(np.array([0,xy,xz,xy+xz]))
                    xhi= box_vec_array_dump[0,1]-np.max(np.array([0,xy,xz,xy+xz]))
                    ylo= box_vec_array_dump[1,0]-np.min(np.array([0,yz]))
                    yhi= box_vec_array_dump[1,1]-np.max(np.array([0,yz]))
                    zlo= box_vec_array_dump[2,0]
                    zhi= box_vec_array_dump[2,1]
                    box_vec_array_upper_tri=np.array([[xhi-xlo,xy,xz],
                                           [0,yhi-ylo,yz],
                                           [0,0,zhi-zlo]])
                    
                # sort positions from dump
                
                # calculate q transform and test against cfg



                transform_dump_coords,transform_force_dump=q_matrix_transform(box_vector_array_cfg,box_vec_array_upper_tri,
                        full_cfg_coord_after_sorted,posvel_from_dump_sing_sorted
                        ,n_particles,n_plates,db_forces,db_dirns,k)
                
                dump_data=transform_force_dump
    
                spring_force_positon_array[j,k,:,0]=-dump_data[:,0]*dump_data[:,3]#xx
                spring_force_positon_array[j,k,:,1]=-dump_data[:,1]*dump_data[:,4]#yy
                spring_force_positon_array[j,k,:,2]=-dump_data[:,2]*dump_data[:,5]#zz
                spring_force_positon_array[j,k,:,3]=-dump_data[:,0]*dump_data[:,5]#xz
                spring_force_positon_array[j,k,:,4]=-dump_data[:,0]*dump_data[:,4]#xy
                spring_force_positon_array[j,k,:,5]=-dump_data[:,1]*dump_data[:,5]#yz
                
    spring_force_positon_mean=np.mean(np.mean(spring_force_positon_array,axis=0),
                            axis=1)
    lgf_mean=np.mean(log_file_array,axis=0)    
    log_file_tuple=log_file_tuple+(lgf_mean,)
    spring_force_positon_tensor_tuple=spring_force_positon_tensor_tuple+\
        (spring_force_positon_array,)
    dir_vector_tuple=dir_vector_tuple+(dirn_vector_array,)
    count+=1
            
                        
                
                    
                
                    
                


                   

                     


        


#%% save tuples to avoid needing the next stage 
#make sure to comment this out after use
label='damp_'+str(damp)+'_K_'+str(K)+'_'
import pickle as pck

folder_check_or_create(filepath,"saved_tuples")

with open(label+'spring_force_positon_tensor_tuple.pickle', 'wb') as f:
    pck.dump(spring_force_positon_tensor_tuple, f)

with open(label+'log_file_tuple.pickle', 'wb') as f:
    pck.dump(log_file_tuple, f)


#%%plot temp vs strain 
thermal_damp_multiplier=np.flip(np.array([750,  750,  750,  650,  550,  450,  450,  450,  500,  500,  750,
        750, 1000, 1500,  800,  550,  550,  600,  600,  600,  750,  750,
       2500, 3000])*0.5 )
column=4# temp 
final_temp=np.zeros((erate.size))
mean_temp_array=np.zeros((erate.size))
pe_final_list=[]



for i in range(0,len(log_file_tuple)):
        
    
        strain_plot=np.linspace(0,strain_total,log_file_tuple[i][:,column].shape[0])
        std_dev=np.std(log_file_tuple[i][:,column])
        #column=4
        plt.plot(strain_plot,log_file_tuple[i][:,column],label="$\dot{\gamma}="+\
                 str(erate[i])+", tdamp=,"+str(thermal_damp_multiplier[i])+\
                    ",t_{\sigma}="+str(std_dev)+"$")
        #plt.plot(strain_plot,1-log_file_tuple[i][:,column],label='Convergence $\dot{\gamma}='+str(erate[i])+'$')
        #print(i)
       
        plt.ylabel("$T$")
        plt.xlabel("$\gamma$")
        #plt.legend(bbox_to_anchor=(1.5,1))
        #plt.yscale('log')
        plt.legend()
        plt.show() 


        # column=5
        # plt.plot(log_file_tuple[i][:,column],label="uef_temp")
      

        # column=5 # ecouple 
        # plt.plot(log_file_tuple[i][:,column],label="Ecouple")
        
        # column=6 # econserve 
        # plt.plot(log_file_tuple[i][:,column],label="Econserve")
#%%

for i in range(0,len(log_file_tuple)):   
        column=2 # pe 
        
        plt.plot(strain_plot,log_file_tuple[i][:,column],label="$\dot{\gamma}="+str(erate[i])+"$")
        #print(log_file_tuple[i][-1,column])
        pe_final_list.append(log_file_tuple[i][-1,column])
        plt.ylabel("$E_{p}$")
        plt.xlabel("$\gamma$")
        #plt.ylim(1e-10,10)       
        plt.yscale('log')
        plt.legend()
        plt.show() 

# for i in range(0,len(log_file_tuple)):  
#         column=1 # ke 

#         plt.plot(log_file_tuple[i][:,column], label="Kinetic E")
#         # total_energy=log_file_tuple[i][:,1]+log_file_tuple[i][:,2]
#         # plt.plot(total_energy,label="total energy")
        
#         plt.legend(loc='upper right')
#         #plt.yscale('log')
#         plt.show() 


#%% mean temp
column=7 # uef temp 
final_temp=np.zeros((erate.size))
mean_temp_array=np.zeros((erate.size))

for i in range(e_in,e_end):
        
        # plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
        mean_temp_array[i]=np.mean(log_file_tuple[i][500:,column])
#         plt.plot(log_file_tuple[i][:,column])
# plt.show() 
        #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
    #     plt.ylabel("$T$", rotation=0)
    #     plt.xlabel("$\gamma$")
    

    # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
    #     plt.show()

#

marker=['x','o','+','^',"1","X","d","*","P","v"]
plt.scatter(erate[e_in:e_end],mean_temp_array[e_in:e_end])
plt.ylabel("$T$", rotation=0)
plt.xlabel('$\dot{\gamma}$')
#plt.xscale('log')
# plt.yscale('log')
plt.axhline(np.mean(mean_temp_array[e_in:e_end]),label="$\\bar{T}="+str(sigfig.round(np.mean(mean_temp_array[e_in:e_end]),sigfigs=5))+"$")
#plt.ylim(0.95,1.05)
plt.legend()
plt.tight_layout()
plt.show()


#%% computing stress distributions 
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import curve_fit
marker=['x','o','+','^',"1","X","d","*","P","v"]
aftcut=1
cut=0.7

labels_stress=["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{yz}$"]

stress_tensor,stress_tensor_std=stress_tensor_averaging(len(log_file_tuple),
                            labels_stress,
                            cut,
                            aftcut,
                            spring_force_positon_tensor_tuple,j_)


plot_stress_tensor(0,3,
                       stress_tensor,
                       stress_tensor_std,
                       j_,n_plates, labels_stress,marker,0,erate,e_end,'--')
plt.show()

# plot_stress_tensor(3,6,
#                        stress_tensor,
#                        stress_tensor_std,
#                        j_,n_plates, labels_stress,marker,0,erate,e_end,'--')

#%%
def ext_visc_compute(stress_tensor,stress_tensor_std,i1,i2,n_plates,e_end,e_in):
    extvisc=(stress_tensor[:,i1]- stress_tensor[:,i2])/erate[e_in:e_end]/30.3
    extvisc_error=np.sqrt(stress_tensor_std[:,i1]**2 +stress_tensor_std[:,i2]**2)/np.sqrt(j_*n_plates)

    return extvisc,extvisc_error

ext_visc_1,ext_visc_1_error=ext_visc_compute(stress_tensor,stress_tensor_std,0,2,n_plates,e_end,e_in)
     

cutoff=0
plt.errorbar(erate[cutoff:e_end],ext_visc_1[cutoff:],yerr=ext_visc_1_error[cutoff:], label="$\eta_{1}$", linestyle='none', marker='x')
#plt.plot(erate[:e_end],ext_visc_1,label="$\eta_{1}$", linestyle='none', marker='x')
plt.ylabel("$\eta/\eta_{s}$", rotation=0, labelpad=20)
plt.xlabel("$\dot{\gamma}$")
plt.show()
# %%
