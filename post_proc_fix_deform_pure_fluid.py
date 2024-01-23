##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will calculate the MPCD stress tensor for a pure fluid under forward NEMD
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

path_2_post_proc_module= '/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/LAMMPS python run and analysis scripts/Analysis codes'
os.chdir(path_2_post_proc_module)
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

no_SRD=10000
box_size=10
#nu_bar=3
#delta_t_srd=0.014872025172594354
#nu_bar=0.9 
delta_t_srd=0.05674857690605889
box_vol=box_size**3
erate=0
no_timesteps=10000
# estimating number of steps  required
strain=3
delta_t_md=delta_t_srd/10
strain_rate= np.array([0.001,0.002,0.003,0.01,0.0005])
number_steps_needed= np.ceil(strain/(strain_rate*delta_t_md))
dump_freq=10
#%% importing one log file 

realisation_name = "log.M_5_no5591_pure_srd_fix_deform_9840_8_60835_23.0_0.005071624521210362_10_10_10_200000_T_1_lbda_0.05071624521210362_gdot_0.002" # with shear 
Path_2_log= "/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/"
thermo_vars="         KinEng          Temp          TotEng        c_pe[1]        c_pe[2]        c_pe[3]        c_pe[5]        f_3[13]        f_3[14]        f_3[15]        f_3[16]        f_3[17]        f_3[18]    "
log_file_for_test= log2numpy_reader(realisation_name,Path_2_log,thermo_vars)
#%%
steady_state_index=10000

fix_srd_collisional_contributions_trace=np.mean(log_file_for_test[steady_state_index:,8:11],axis=0)
fix_srd_collisional_contributions_xz=np.mean(log_file_for_test[steady_state_index:,13],axis=0)
fix_srd_kinetic_contributions_trace=np.mean(log_file_for_test[steady_state_index:,4:7],axis=0)
fix_srd_kinetic_contributions_xz=np.mean(log_file_for_test[steady_state_index:,7],axis=0)

fix_srd_stress_tensor_trace= np.sum(fix_srd_collisional_contributions_trace +fix_srd_kinetic_contributions_trace)/3
fix_srd_stress_tensor_xz=fix_srd_collisional_contributions_xz + fix_srd_kinetic_contributions_trace[2] +fix_srd_kinetic_contributions_xz
viscosity=fix_srd_stress_tensor_xz/erate
#%% finding all the dump files in a folder

VP_general_name_string='vel.*'

Mom_general_name_string='mom.*'

log_general_name_string='log.*'
                         #log.H20_no466188_wall_VACF_output_no_rescale_
TP_general_name_string='temp.*'

dump_general_name_string_after='*after*.dump'
dump_general_name_string_before='*before*.dump'

filepath="/KATHLEEN_LAMMPS_RUNS/equilibrium_fix_deform_pure_mpcd_test_file"
filepath="Simulation_run_folder/test_analysis_small_equilibrium_test_box_10_M_10"
Path_2_dump="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/"+filepath
# can chnage this to another array on kathleen


dump_realisation_name_info_before= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_before)
realisation_name_VP=dump_realisation_name_info_before[1]
count_mom=dump_realisation_name_info_before[2]
count_VP=dump_realisation_name_info_before[3]
realisation_name_log=dump_realisation_name_info_before[4]
count_log=dump_realisation_name_info_before[5]
realisation_name_dump_before=dump_realisation_name_info_before[6]
count_dump_before=dump_realisation_name_info_before[7]

dump_realisation_name_info_after= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_after)

realisation_name_dump_after=dump_realisation_name_info_after[6]
count_dump_after=dump_realisation_name_info_after[7]

if count_dump_before==count_dump_after:
    print("consistent dump file count")
else:
    print("inconsistent dump file count, Please check")

#%% Check velocity profiles 
# this section can be done with other data 
Path_2_VP=Path_2_log
chunk=20
VP_ave_freq=10000

VP_output_col_count=6 
equilibration_timesteps=0
realisation_name="vel.testout_pure_output_no806324_no_rescale_541709_2_60835_23.0_0.005071624521210362_10_10000_10000_1000010_T_1_lbda_0.05071624521210362_gdot_0"
vel_profile_in= velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)
vel_profile_x=vel_profile_in[0]
vel_profile_z_data= vel_profile_in[1]
vel_profile_z_data=vel_profile_z_data.astype('float')*box_size

# plotting time series of vps
shear_rate_list =[]
for i in range(0,int(no_timesteps/VP_ave_freq)):
    shear_rate_list.append(scipy.stats.linregress(vel_profile_z_data[:],vel_profile_x[:,i]).slope)
    plt.plot(vel_profile_x[:,i],vel_profile_z_data[:])
plt.show()

# could do a standard error aswell and an R^2




#%% reading a dump file 

dump_start_line="ITEM: ATOMS id x y z vx vy vz xu yu zu"
number_of_particles_per_dump = no_SRD
number_of_dumps_per_realisation=int(no_timesteps/dump_freq)
number_of_repeats=4
total_number_of_data_sets = int(count_dump_before/number_of_repeats)
columns= 10



# this needs a fail to read statement in it 

def reading_in_unsorted_dumps(number_of_particles_per_dump,dump_start_line,Path_2_dump,total_number_of_data_sets,number_of_repeats,number_of_dumps_per_realisation,no_SRD,columns,realisation_name_dump_before,realisation_name_dump_after):
    from dump2numpy import dump2numpy_f
    dump_file_before=np.zeros((total_number_of_data_sets,number_of_repeats,number_of_dumps_per_realisation*no_SRD,columns))
    dump_file_after=np.zeros((total_number_of_data_sets,number_of_repeats,(number_of_dumps_per_realisation+1)*no_SRD,columns))


    for j in range(0,total_number_of_data_sets):
        for i in range(0,number_of_repeats): 
           
            dump_file_before[j,i,:,:]= dump2numpy_f(dump_start_line,Path_2_dump,realisation_name_dump_before[i],number_of_particles_per_dump)
            dump_file_after[j,i,:,:]= dump2numpy_f(dump_start_line,Path_2_dump,realisation_name_dump_after[i],number_of_particles_per_dump)



   # np.save("raw_data_array_after.py",dump_file_after)
    #np.save("raw_data_array_before.py",dump_file_before)

    return dump_file_after, dump_file_before
dump_files=reading_in_unsorted_dumps(number_of_particles_per_dump,dump_start_line,Path_2_dump,total_number_of_data_sets,number_of_repeats,number_of_dumps_per_realisation,no_SRD,columns,realisation_name_dump_before,realisation_name_dump_after)

dump_file_after=dump_files[0]
dump_file_before=dump_files[1]
# reshaping 
# need to write a test to check this was done properly 
loop_size=1000
dump_file_shaped_before=np.reshape(dump_file_before,(total_number_of_data_sets,number_of_repeats,loop_size,number_of_particles_per_dump,columns))
loop_size=1001
dump_file_shaped_after=np.reshape(dump_file_after,(total_number_of_data_sets,number_of_repeats,loop_size,number_of_particles_per_dump,columns))
# freeing memory 
del dump_file_before
del dump_file_after


#%% sorting rows 


# sorting by the atom id column
def sorting_and_checking_sort_dump_files(dump_file_shaped_after,dump_file_shaped_before,total_number_of_data_sets,number_of_repeats,loop_size,number_of_particles_per_dump,columns):
        dump_file_sorted_before=np.zeros((total_number_of_data_sets,number_of_repeats,loop_size,number_of_particles_per_dump,columns))
        dump_file_sorted_after=np.zeros((total_number_of_data_sets,number_of_repeats,loop_size,number_of_particles_per_dump,columns))


        for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
                    for j in range(0,loop_size):
                    
                     dummy_sort_array_after=dump_file_shaped_after[i,k,j,:,:]
                     dump_file_sorted_after[i,k,j,:,:]=dummy_sort_array_after[dummy_sort_array_after[:,0].argsort()]
        
        for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
                    for j in range(0,loop_size-1):
                     dummy_sort_array_before=dump_file_shaped_before[i,k,j,:,:]
                     dump_file_sorted_before[i,k,j,:,:]=dummy_sort_array_before[dummy_sort_array_before[:,0].argsort()]
                        


        # boolean to check order is correct 
        comparison_list =np.arange(1,number_of_particles_per_dump+1,1)
        error_count=0
        success_count=0
        for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
                for j in range(0,loop_size):

                    boolean_result_1 = dump_file_sorted_before[i,k,j,:,0]==comparison_list
                    if np.all(boolean_result_1)==True:
                        success_count=success_count+1
                        

                    else:
                        error_count=error_count+1
        for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
                for j in range(0,loop_size-1):


                    boolean_result_2 = dump_file_sorted_after[i,k,j,:,0]==comparison_list
                    if np.all(boolean_result_2)==True:
                        
                        success_count=success_count+1
                    else:
                        error_count=error_count+1

        print(error_count)
        if success_count==total_number_of_data_sets*number_of_repeats*loop_size*2:
            print("sucessful sort")
        else:
            print("There were ",error_count," sorting errors")

        # freeing memory 
        #np.save("dump_file_sorted_before.py",dump_file_sorted_before)
        #np.save("dump_file_sorted_after.py",dump_file_sorted_after)

        return dump_file_sorted_before,dump_file_sorted_after,error_count,success_count


sorting_results=sorting_and_checking_sort_dump_files(dump_file_shaped_after,dump_file_shaped_before,total_number_of_data_sets,number_of_repeats,loop_size,number_of_particles_per_dump,columns)

dump_file_sorted_before=sorting_results[0]
dump_file_sorted_after=sorting_results[1]
error_count=sorting_results[2]
success_count=sorting_results[3]

#del dump_file_shaped_after
#del dump_file_shaped_before

#%% calculating the kinetic energy tensor 

kinetic_energy_tensor=np.zeros((total_number_of_data_sets,number_of_repeats,loop_size,6))
delta_mom_pos_tensor_from_dump=np.zeros((total_number_of_data_sets,number_of_repeats,loop_size-1,no_SRD,9))
# is this right 
for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
                for j in range(0,loop_size-1): # needs to finish one behind due to dump output 
                    kinetic_energy_tensor[i,k,j,0]= np.sum(dump_file_sorted_before[i,k,j,:,4]* dump_file_sorted_before[i,k,j,:,4])/box_vol#xx
                    kinetic_energy_tensor[i,k,j,1]= np.sum(dump_file_sorted_before[i,k,j,:,5]* dump_file_sorted_before[i,k,j,:,5])/box_vol#yy
                    kinetic_energy_tensor[i,k,j,2]= np.sum(dump_file_sorted_before[i,k,j,:,6]* dump_file_sorted_before[i,k,j,:,6])/box_vol#zz
                    kinetic_energy_tensor[i,k,j,3]= np.sum(dump_file_sorted_before[i,k,j,:,4]* dump_file_sorted_before[i,k,j,:,6])/box_vol#xz
                    kinetic_energy_tensor[i,k,j,4]= np.sum(dump_file_sorted_before[i,k,j,:,4]* dump_file_sorted_before[i,k,j,:,5])/box_vol#xy
                    kinetic_energy_tensor[i,k,j,5]= np.sum(dump_file_sorted_before[i,k,j,:,5]* dump_file_sorted_before[i,k,j,:,6])/box_vol#zy
                



# collisional tensor 
for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
                for j in range(0,loop_size-1): # needs to start 1 ahead due to dump output

                    delta_mom_pos_tensor_from_dump[i,k,j,:,0]=(dump_file_sorted_after[i,k,j+1,:,4]- dump_file_sorted_before[i,k,j,:,4]) *  dump_file_sorted_after[i,k,j+1,:,1]#xx
                    delta_mom_pos_tensor_from_dump[i,k,j,:,1]=(dump_file_sorted_after[i,k,j+1,:,5]- dump_file_sorted_before[i,k,j,:,5]) *  dump_file_sorted_after[i,k,j+1,:,2]#yy
                    delta_mom_pos_tensor_from_dump[i,k,j,:,2]=(dump_file_sorted_after[i,k,j+1,:,6]- dump_file_sorted_before[i,k,j,:,6]) *  dump_file_sorted_after[i,k,j+1,:,3]#zz
                    delta_mom_pos_tensor_from_dump[i,k,j,:,3]=(dump_file_sorted_after[i,k,j+1,:,4]- dump_file_sorted_before[i,k,j,:,4]) * dump_file_sorted_after[i,k,j+1,:,3]#xz
                    delta_mom_pos_tensor_from_dump[i,k,j,:,4]=(dump_file_sorted_after[i,k,j+1,:,4]- dump_file_sorted_before[i,k,j,:,4]) * dump_file_sorted_after[i,k,j+1,:,2]#xy
                    delta_mom_pos_tensor_from_dump[i,k,j,:,5]=(dump_file_sorted_after[i,k,j+1,:,5]- dump_file_sorted_before[i,k,j,:,5]) * dump_file_sorted_after[i,k,j+1,:,3]#yz
                    delta_mom_pos_tensor_from_dump[i,k,j,:,6]=(dump_file_sorted_after[i,k,j+1,:,6]- dump_file_sorted_before[i,k,j,:,6]) * dump_file_sorted_after[i,k,j+1,:,1]#zx
                    delta_mom_pos_tensor_from_dump[i,k,j,:,7]=(dump_file_sorted_after[i,k,j+1,:,5]- dump_file_sorted_before[i,k,j,:,5]) * dump_file_sorted_after[i,k,j+1,:,1]#yx
                    delta_mom_pos_tensor_from_dump[i,k,j,:,8]=(dump_file_sorted_after[i,k,j+1,:,6]- dump_file_sorted_before[i,k,j,:,6]) * dump_file_sorted_after[i,k,j+1,:,2]#zy
                
                                

# delta_mom_pos_tensor_from_dump_mean=np.mean(np.sum( delta_mom_pos_tensor_from_dump[1:,:],axis=3),axis=0)/(delta_t_srd*box_vol)
delta_mom_pos_tensor_from_dump_summed=np.sum( delta_mom_pos_tensor_from_dump,axis=3)/(delta_t_srd*box_vol)
delta_mom_pos_tensor_from_dump_magnitude_mean= np.mean(np.abs(delta_mom_pos_tensor_from_dump_summed),axis=2)

delta_mom_pos_tensor_from_dump_summed_mean=np.mean(delta_mom_pos_tensor_from_dump_summed,axis=2)




#%% calculating with lattice vector 
# lattice_vector=np.zeros((total_number_of_data_sets,number_of_repeats,loop_size,no_SRD,3))
# for i in range(0,total_number_of_data_sets): 
#             for k in range(0,number_of_repeats):
#                 for j in range(0,loop_size-1):
#                       lattice_vector[i,k,j,:,0]=dump_file_sorted_after[i,k,j,:,7]-dump_file_sorted_after[i,k,j,:,1]
#                       lattice_vector[i,k,j,:,1]=dump_file_sorted_after[i,k,j,:,8]-dump_file_sorted_after[i,k,j,:,2]
#                       lattice_vector[i,k,j,:,2]=dump_file_sorted_after[i,k,j,:,9]-dump_file_sorted_after[i,k,j,:,3]

# external_stress_tensor = np.zeros((total_number_of_data_sets,number_of_repeats,loop_size-1,no_SRD,3))


# for i in range(0,total_number_of_data_sets): 
#             for k in range(0,number_of_repeats):
#                 for j in range(0,loop_size-1):
#                       external_stress_tensor[i,k,j,:,0]=(dump_file_sorted_after[i,k,j+1,:,4]- dump_file_sorted_before[i,k,j,:,4])* lattice_vector[i,k,j,:,0]
#                       external_stress_tensor[i,k,j,:,1]=(dump_file_sorted_after[i,k,j+1,:,5]- dump_file_sorted_before[i,k,j,:,5])* lattice_vector[i,k,j,:,1]
#                       external_stress_tensor[i,k,j,:,2]=(dump_file_sorted_after[i,k,j+1,:,5]- dump_file_sorted_before[i,k,j,:,5])* lattice_vector[i,k,j,:,2]
                      
# external_stress_tensor_summed=np.sum(external_stress_tensor,axis=3)/(delta_t_srd*box_vol)
# external_stress_tensor_summed_mean=np.mean(external_stress_tensor_summed,axis=2)
# external_stress_tensor_realisation_mean=np.mean(external_stress_tensor_summed_mean)

# plt.plot(external_stress_tensor_summed[0,:,:,0])

# plt.show()

# #%%
# # computing equation 14 in winkler 2009 

# Force_term=np.zeros((total_number_of_data_sets,number_of_repeats,loop_size,no_SRD,3))


# for i in range(0,total_number_of_data_sets): 
#             for k in range(0,number_of_repeats):
#                 for j in range(0,loop_size-1): 
#                       Force_term[i,k,j,:,0]=external_stress_tensor[i,k,j,:,0]+ delta_mom_pos_tensor_from_dump[i,k,j,:,0]
#                       Force_term[i,k,j,:,1]=external_stress_tensor[i,k,j,:,1]+ delta_mom_pos_tensor_from_dump[i,k,j,:,1]
#                       Force_term[i,k,j,:,2]=external_stress_tensor[i,k,j,:,2]+ delta_mom_pos_tensor_from_dump[i,k,j,:,2]

# Force_term=Force_term/(delta_t_srd*box_vol)
# Force_term_summed=np.sum(Force_term,axis=3)
# Force_term_mean=np.mean(Force_term_summed,axis=2)

                      




#%% checking momentum exchange sums to 0 for equilibrium only 

#NOTE: This isnt correct

delta_mom_from_dump=np.zeros((total_number_of_data_sets,number_of_repeats,loop_size-1,no_SRD,3))

for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
                for j in range(0,loop_size-1): # needs to start 1 ahead due to dump output
                    delta_mom_from_dump[i,k,j,:,0]=(dump_file_sorted_after[i,k,j+1,:,4]- dump_file_sorted_before[i,k,j,:,4]) 
                    delta_mom_from_dump[i,k,j,:,1]=(dump_file_sorted_after[i,k,j+1,:,5]- dump_file_sorted_before[i,k,j,:,5]) 
                    delta_mom_from_dump[i,k,j,:,2]=(dump_file_sorted_after[i,k,j+1,:,6]- dump_file_sorted_before[i,k,j,:,6]) 
                   

delta_mom_from_dump_summed=np.sum(delta_mom_from_dump,axis=3)
delta_mom_from_dump_mean=np.mean(delta_mom_from_dump_summed,axis=2)
delta_mom_from_dump_realisation_mean=np.mean(delta_mom_from_dump_mean)

for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
               
               plt.plot(delta_mom_from_dump_summed[i,k,:,0], label="x")
             
               plt.plot(delta_mom_from_dump_summed[i,k,:,1],label="y")
      
               plt.plot(delta_mom_from_dump_summed[i,k,:,2],label='z')
               plt.legend()
            
              
plt.show()
               


#%%
for i in range(0,total_number_of_data_sets): 
            #for k in range(0,number_of_repeats):
               
               plt.hist(delta_mom_from_dump_summed[i,:,:,0],label="x")
               plt.show()
               
               plt.hist(delta_mom_from_dump_summed[i,:,:,1],label="y")
               plt.show()
               
               plt.hist(delta_mom_from_dump_summed[i,:,:,2],label="z")
               plt.show()
               
               #plt.hist(delta_mom_from_dump_summed[i,k,:,3])
               
plt.show()
                            
for i in range(0,total_number_of_data_sets): 
            #for k in range(0,number_of_repeats):
               
               plt.plot(delta_mom_pos_tensor_from_dump_summed[i,:,:,0], label="x")
             
               plt.plot(delta_mom_pos_tensor_from_dump_summed[i,:,:,1],label="y")
      
               plt.plot(delta_mom_pos_tensor_from_dump_summed[i,:,:,2],label='z')
               plt.legend()
            
              
plt.show()

# delta_mom_from_dump_abs_mean= np.mean(np.abs(delta_mom_from_dump), axis=1)


# # doesnt work but I think 
# standard_error_in_momentum=np.std(delta_mom_from_dump_summed, axis=0)
# realtive_error= standard_error_in_momentum/delta_mom_from_dump_abs_mean
# this is an error margin, I think the errror is propagated through the whole calculation    
    
#%% checking convergence of the sum 

delta_mom_pos_tensor_from_dump_partial_sum = np.zeros((total_number_of_data_sets,number_of_repeats,loop_size-1,9))

for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
                for j in range(0,loop_size-1):
                 delta_mom_pos_tensor_from_dump_partial_sum[i,k,j,:]=np.mean(delta_mom_pos_tensor_from_dump_summed[i,k,:j,:],axis=0)
delta_mom_pos_tensor_from_dump_partial_sum_realisation_mean=np.mean(delta_mom_pos_tensor_from_dump_partial_sum,axis=1)

np.save("delta_mom_pos_tensor_from_dump_partial_sum_realisation_mean_"+str(box_size)+".py",delta_mom_pos_tensor_from_dump_partial_sum_realisation_mean)



#%%
labels=["$\Delta p_{x}r_{x}$","$\Delta p_{y}r_{y}$","$\Delta p_{z}r_{z}$","$\Delta p_{x}r_{z}$","$\Delta p_{x}r_{y}$","$\Delta p_{y}r_{z}$","$\Delta p_{z}r_{x}$","$\Delta p_{y}r_{x}$","$\Delta p_{z}r_{y}$"]
labelpady=30
fontsize=15
plt.rcParams.update({'font.size': 9})
for i in range(0,total_number_of_data_sets): 
                for j in range(0,3):
                    plt.plot(delta_mom_pos_tensor_from_dump_partial_sum_realisation_mean[i,:,j],label=labels[j],color=colour[j])
                    plt.ylabel('$\\langle \Delta p_{coll,\\alpha}r_{\\beta} \\rangle_{T}$', rotation=0, labelpad=labelpady)
                    plt.xlabel("$N_{coll}$")
                    plt.title("Rolling average of collisional contribution, $\\alpha \parallel \\beta$, $L="+str(box_size)+"$")

                    #plt.ylim((None,1))
                    plt.axhline(delta_mom_pos_tensor_from_dump_partial_sum_realisation_mean[i,-1,j],0,4, label="Final mean "+labels[j]+"="+str(sigfig.round(delta_mom_pos_tensor_from_dump_partial_sum_realisation_mean[i,-1,j],sigfigs=2)),linestyle='dashed',color=colour[j])
                    plt.legend(bbox_to_anchor=(1,1))
                plt.tight_layout()
                plt.savefig("rolling_ave_collisional_contribution_L"+str(box_size)+".png")
                plt.show()


for i in range(0,total_number_of_data_sets): 
                for j in range(3,9):
                        plt.plot(delta_mom_pos_tensor_from_dump_partial_sum_realisation_mean[i,:,j],label=labels[j],color=colour[j])
                        plt.ylabel('$\\langle \Delta p_{coll,\\alpha}r_{\\beta} \\rangle_{T}$', rotation=0, labelpad=labelpady)
                        plt.xlabel("$N_{coll}$")
                        plt.title("Rolling average of collisional contribution, $\\alpha \perp \\beta$, $L="+str(box_size)+"$")

                        #plt.ylim((None,1))
                        plt.axhline(delta_mom_pos_tensor_from_dump_partial_sum_realisation_mean[i,-1,j],0,4, label="Final mean "+labels[j]+"="+str(sigfig.round(delta_mom_pos_tensor_from_dump_partial_sum_realisation_mean[i,-1,j],sigfigs=2)),linestyle='dashed',color=colour[j])
                        plt.legend(loc='right',bbox_to_anchor=(1,0.3))
                plt.tight_layout()
                plt.savefig("rolling_ave_collisional_contribution_perp_L"+str(box_size)+".png")
                plt.show()


# for i in range(0,total_number_of_data_sets): 
#                 for j in range(0,3):
#                     plt.plot(delta_mom_pos_tensor_from_dump_summed_mean[i,:,j],label=labels[j])
                  
#                     plt.ylabel('$\\langle \Delta p_{coll,\\alpha}r_{\\beta} \\rangle_{T}$', rotation=0, labelpad=labelpady)
#                     plt.xlabel("Realisation index")
#                     plt.title("Final mean of  $\Delta p_{coll,\\alpha}r_{\\beta}, \\alpha \parallel \\beta $, $L="+str(box_size)+"$")

#                     #plt.ylim((None,1))
                    
#                 plt.axhline(np.mean(delta_mom_pos_tensor_from_dump_summed_mean[0,:,0:3]),0,4, label="$\\bar{\Delta p_{coll,\\alpha}r_{\\beta}}$",linestyle='dashed')
#                 plt.legend()
#                 plt.tight_layout()
#                 plt.savefig("final_mean_of_dp_coll_L"+str(box_size)+".png")
#                 plt.show()

# for i in range(0,total_number_of_data_sets): 
#                 for j in range(3,9):
#                     plt.plot(delta_mom_pos_tensor_from_dump_summed_mean[i,:,j],label=labels[j])
                  
#                     plt.ylabel('$\\langle \Delta p_{coll,\\alpha}r_{\\beta} \\rangle_{T}$', rotation=0, labelpad=labelpady)
#                     plt.xlabel("Realisation index")
#                     plt.title("Final mean of  $\Delta p_{coll,\\alpha}r_{\\beta}, \\alpha \perp \\beta$, $L="+str(box_size)+"$")

#                     #plt.ylim((None,1))
                    
#                 plt.axhline(np.mean(delta_mom_pos_tensor_from_dump_summed_mean[0,:,3:]),0,4, label="$\\bar{\Delta p_{coll,\\alpha}r_{\\beta}}$",linestyle='dashed')
#                 plt.legend()
#                 plt.tight_layout()
#                 plt.savefig("final_mean_of_dp_coll_perp_L"+str(box_size)+".png")
#                 plt.show()


# saving block
                
np.save("delta_mom_pos_tensor_from_dump_summed_mean_L_"+str(box_size)+".py",delta_mom_pos_tensor_from_dump_summed_mean)





#%%

#shear_rate_term_from_dump=(erate*delta_t_srd/2)*kinetic_energy_tensor[0,0,2:,2]
#stress_tensor_xz_from_dump= kinetic_energy_tensor[2:,3]+ shear_rate_term_from_dump+ delta_mom_pos_tensor_from_dump_summed[:,3]
# # stress_tensor_xz_rms_mean_from_dump= np.sqrt(np.mean(stress_tensor_xz**2))
#stress_tensor_xz_mean_from_dump= np.mean(stress_tensor_xz_from_dump)
# viscosity_from_dump=stress_tensor_xz_mean_from_dump/erate

for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
                stress_tensor_xx_from_dump=( kinetic_energy_tensor[i,k,1:,0] +delta_mom_pos_tensor_from_dump_summed[i,k,:,0])
                # collisional_xx_mean= np.mean(stress_tensor_xx_from_dump)
                # plt.axhline(collisional_xx_mean,0,200, label="\sigma_xx_mean")
                plt.plot(stress_tensor_xx_from_dump,label="\sigma_xx")
plt.show()
for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
                stress_tensor_yy_from_dump=( kinetic_energy_tensor[i,k,1:,1] +delta_mom_pos_tensor_from_dump_summed[i,k,:,1])
                # collisional_yy_mean= np.mean(stress_tensor_yy_from_dump)
                # plt.axhline(collisional_yy_mean,0,200, label="\sigma_yy_mean")
                plt.plot(stress_tensor_yy_from_dump,label="\sigma_yy")
plt.show()
for i in range(0,total_number_of_data_sets): 
            for k in range(0,number_of_repeats):
                stress_tensor_zz_from_dump=( kinetic_energy_tensor[i,k,1:,2] +delta_mom_pos_tensor_from_dump_summed[i,k,:,2])
                # collisional_zz_mean= np.mean(stress_tensor_zz_from_dump)
                # plt.axhline(collisional_zz_mean,0,200, label="\sigma_zz_mean")
                plt.plot(stress_tensor_zz_from_dump,label="\sigma_zz")

# plt.legend()
#plt.ylim(9.9,10.1)
plt.show()

pressure_dump=np.mean((stress_tensor_xx_from_dump+stress_tensor_yy_from_dump+stress_tensor_zz_from_dump)/3)

# plt.plot(pressure_dump)
# plt.show()
# # shear_rate_term=(erate*delta_t_srd/2)*kinetic_energy_tensor_zz 
#plt.plot(stress_tensor_xz_from_dump[:])
#plt.show()

#viscosity_from_dump=stress_tensor_xz/erate
#plt.plot(log_file_for_test[:,0],viscosity[:])
#plt.show()

#%% plotting some histograms 

number_of_plots=100
for i in range(0,number_of_plots):
    plt.hist(dump_file_sorted_before[i,:,4])
    plt.xlabel("$v_{x}$")
    plt.ylabel("$N$",rotation=0)
plt.show()
for i in range(0,number_of_plots):
    plt.hist(dump_file_sorted_before[i,:,5])
    plt.xlabel("$v_{y}$")
    plt.ylabel("$N$",rotation=0)
plt.show()
for i in range(0,number_of_plots):
    plt.hist(dump_file_sorted_before[i,:,6])
    plt.xlabel("$v_{z}$")
    plt.ylabel("$N$",rotation=0)
plt.show()
        


      

# %%
