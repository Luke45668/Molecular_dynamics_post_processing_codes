##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will load in an array of run time data  by reading lammps log files. T
"""
#%%
#NOTE need to turn script bck into functions 
import os
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import regex as re
import pandas as pd
import sigfig
plt.rcParams.update(plt.rcParamsDefault)
#plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import *
path_2_post_proc_module= '/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/LAMMPS python run and analysis scripts/Analysis codes'
os.chdir(path_2_post_proc_module)
from post_MPCD_MP_processing_module import *
swap_rate = np.array([3,7,15,30,60,150,300,600,900,1200]) # values chosen from original mp paper
swap_number = np.array([1,10,100,1000])
#%% # need to use the same set up as reading and organising log files before, then only read the run time 


VP_general_name_string='vel.*_VACF_output_*_no_rescale_*'

Mom_general_name_string='mom.*_VACF_out_*_no_rescale_*'

log_general_name_string='log.*'
                         #log.H20_no466188_wall_VACF_output_no_rescale_
TP_general_name_string='temp.*_VACF_output_*_no_rescale_*'

dump_general_name_string='test_run_dump__*'

filepath='T_1_pure_fluid_run_time/'
realisation_name_info= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string)
realisation_name_Mom=realisation_name_info[0]
realisation_name_VP=realisation_name_info[1]
count_mom=realisation_name_info[2]
count_VP=realisation_name_info[3]
realisation_name_log=realisation_name_info[4]
count_log=realisation_name_info[5]
realisation_name_dump=realisation_name_info[6]
count_dump=realisation_name_info[7]
realisation_name_TP=realisation_name_info[8]
count_TP=realisation_name_info[9]

def run_time_reader(realisation_name):
   format_string = "%H:%M:%S"
   run_data_start= ("Total wall time: ")
   run_data_start_bytes=bytes(run_data_start,'utf-8')

   with open(realisation_name) as f:
      read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ ) 

      run_data_seek=read_data.find(run_data_start_bytes)  
      read_data.seek(run_data_seek) 

      
      run_time_bytes=read_data.read()
      
      read_data.close()
   run_time=str(run_time_bytes)[18:26].replace(" ", "")
   epoch_time = datetime(1900, 1, 1)
   run_time=datetime.strptime( run_time, format_string)
   delta = (run_time - epoch_time)
   run_time=int(delta.total_seconds())
   
   return run_time


#%%log file reader and organiser
fluid_names=['Ar','Nitrogen','H20','C6H14']
log_EF=21
log_SN=23
log_realisation_index=8

loc_fluid_name=0
j_=3
loc_no_SRD=9
no_SRD=[]
box_size=[]
no_SRD=np.zeros((count_log,1),dtype=int)
fluid_name_array=np.zeros((count_log,1),dtype=str)

# not happy with this bit of code, could ceertainly be more elegant 
for i in range(0,count_log):
    filename=realisation_name_log[i].split('_')
    print(filename[loc_no_SRD])
    no_SRD[i,0] = filename[loc_no_SRD]
    print(filename[loc_fluid_name][4:])
    fluid_name_array[i,0] = filename[loc_fluid_name][4:]

no_SRD_and_fluid_names = np.concatenate((fluid_name_array,no_SRD),axis=1)
fluid_name_array=np.unique(no_SRD_and_fluid_names,axis=0)



#%%
    
# no_SRD.sort(key=int)
# no_SRD.sort()
# no_SRD_key=[]

#using list comprehension to remove duplicates
# [no_SRD_key.append(x) for x in no_SRD if x not in no_SRD_key]

Path_2_log='/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/'+filepath
thermo_vars='         KinEng          Temp          TotEng       c_vacf[4]   '
from log2numpy import * 
total_cols_log=5
org_var_log_1=swap_rate
loc_org_var_log_1=log_EF
org_var_log_2=swap_number#spring_constant
loc_org_var_log_2=log_SN

#%%
def log_run_time_reader(loc_no_SRD,org_var_log_1,loc_org_var_log_1,org_var_log_2,loc_org_var_log_2,j_,count_log,realisation_name_log,log_realisation_index,fluid_names,loc_fluid_name):
    
   
   
   
    run_time_array_with_all_realisations=np.zeros((len(fluid_names),j_,org_var_log_1.size,org_var_log_2.size,))
    run_time_array_unsorted_srd_bins=()
    for i in range(0,fluid_name_array.shape[0]):
        run_time_array_unsorted_srd_bins=run_time_array_unsorted_srd_bins+(run_time_array_with_all_realisations,)
        
        
   

    for i in range(0,count_log):
        filename=realisation_name_log[i].split('_')
        realisation_name_for_run_time=realisation_name_log[i]
        realisation_index=int(float(realisation_name_log[i].split('_')[log_realisation_index]))
        fluid_identifier=filename[loc_fluid_name][4:]
        
        

        if isinstance(fluid_identifier,(str)):
            fluid_name_find_in_selection=fluid_identifier
            fluid_name_index=fluid_names.index(fluid_name_find_in_selection)
        else:
            break
        if isinstance(filename[loc_no_SRD],(int)):
            no_srd_in_selection=filename[loc_no_SRD]
            #no_srd_index=no_SRD_key.index(no_srd_in_selection)
            fluid_name_array_fluid_col_index = np.where(fluid_name_array[:,0]==fluid_names[fluid_name_index][0])
            no_srd_row_index=np.where(int(filename[loc_no_SRD])==fluid_name_array[:,1])
            if no_srd_row_index == fluid_name_array_fluid_col_index :
                
                fluid_name_no_srd_tuple_index = no_srd_row_index
                print(fluid_name_no_srd_tuple_index)
                
            else:
                break 
        else:
            break
        if isinstance(filename[loc_org_var_log_1],int):
            org_var_log_1_find_in_name=int(filename[loc_org_var_log_1])
            org_var_1_index=np.where(org_var_log_1==org_var_log_1_find_in_name)[0][0]
        else:
            org_var_log_1_find_in_name=float(filename[loc_org_var_log_1])
            org_var_1_index=np.where(org_var_log_1==org_var_log_1_find_in_name)[0][0]
        
        if isinstance(filename[loc_org_var_log_2],int):
            org_var_log_2_find_in_name=int(filename[loc_org_var_log_2])
            org_var_2_index= np.where(org_var_log_2==org_var_log_2_find_in_name)[0][0] 
        else:
            org_var_log_2_find_in_name=float(filename[loc_org_var_log_2])
            org_var_2_index= np.where(org_var_log_2==org_var_log_2_find_in_name)[0][0] 
            
        # multiple swap numbers and swap rates
        #log_file_tuple[np.where(swap_rate==swap_rate_org)[0][0]][np.where(swap_number==swap_number_org)[0][0],realisation_index,:,:]=log2numpy_reader(realisation_name_log[i],Path_2_log,thermo_vars)
    
        run_time_array_unsorted_srd_bins[fluid_name_no_srd_tuple_index][fluid_name_index,realisation_index,org_var_1_index, org_var_2_index]=run_time_reader(realisation_name_for_run_time) # need a new function that simply grabs the run time from end of the file 
    
    return run_time_array_unsorted_srd_bins
    

#%%

#run_time_array= log_run_time_reader(loc_no_SRD,org_var_log_1,loc_org_var_log_1,org_var_log_2,loc_org_var_log_2,j_,count_log,realisation_name_log,log_realisation_index,fluid_names,loc_fluid_name)


#%%
run_time_array_with_all_realisations=np.zeros((fluid_name_array.shape[0],j_,org_var_log_1.size,org_var_log_2.size,))
run_time_array_unsorted_srd_bins=()
#%%
for i in range(0,fluid_name_array.shape[0]):
    run_time_array_unsorted_srd_bins=run_time_array_unsorted_srd_bins+(run_time_array_with_all_realisations,)
    
#%%   

#for i in range(0,count_log):
#for i in range(0,120):
#for i in range(120,125):    
for i in range(0,count_log): 
    filename=realisation_name_log[i].split('_')
    realisation_name_for_run_time=realisation_name_log[i]
    #print(realisation_name_for_run_time)
    realisation_index=int(float(realisation_name_log[i].split('_')[log_realisation_index]))
    
    fluid_identifier=filename[loc_fluid_name][4:]
    #print("fluid_identifier",fluid_identifier)
    fluid_name_index=fluid_names.index(fluid_identifier)
    #print("fluid_name_index", fluid_name_index)
    #print(filename[loc_no_SRD])
    fluid_name_array_fluid_col_index = np.where(fluid_name_array[:,0]==fluid_names[fluid_name_index][0])
    #print("fluid_name_array_fluid_col_index",fluid_name_array_fluid_col_index)
    no_srd_row_index=np.where(filename[loc_no_SRD]==fluid_name_array[:,1])[0]
    #print("no_srd_row_index",no_srd_row_index)
    a = no_srd_row_index[:]
    b =fluid_name_array_fluid_col_index[0].flatten()
    name_number_comparison = [np.where(a==x) for x in b]
    correct_fluid_index=[]
    for j in range(0,len(name_number_comparison)):
        if name_number_comparison[j][0].size == 0:
           continue 
        elif name_number_comparison[j][0] == 0:
            correct_fluid_index.append(j)
        elif name_number_comparison[j][0] == 1:
            correct_fluid_index.append(j)
        else: 
            print("SRD and fluid name not matched")
            break 
    #print("correct_fluid_index",correct_fluid_index)
    
    #print(fluid_name_no_srd_tuple_index)
    if isinstance(filename[loc_org_var_log_1],int):
        org_var_log_1_find_in_name=int(filename[loc_org_var_log_1])
        org_var_1_index=np.where(org_var_log_1==org_var_log_1_find_in_name)[0][0]
    else:
        org_var_log_1_find_in_name=float(filename[loc_org_var_log_1])
        org_var_1_index=np.where(org_var_log_1==org_var_log_1_find_in_name)[0][0]
    
    if isinstance(filename[loc_org_var_log_2],int):
        org_var_log_2_find_in_name=int(filename[loc_org_var_log_2])
        org_var_2_index= np.where(org_var_log_2==org_var_log_2_find_in_name)[0][0] 
    else:
        org_var_log_2_find_in_name=float(filename[loc_org_var_log_2])
        org_var_2_index= np.where(org_var_log_2==org_var_log_2_find_in_name)[0][0] 
        
    # multiple swap numbers and swap rates
    #log_file_tuple[np.where(swap_rate==swap_rate_org)[0][0]][np.where(swap_number==swap_number_org)[0][0],realisation_index,:,:]=log2numpy_reader(realisation_name_log[i],Path_2_log,thermo_vars)
    #print(fluid_name_no_srd_tuple_index[0])
    # print(realisation_index)
    # print(org_var_1_index) 
    # print(org_var_2_index)
    # print(run_time_reader(realisation_name_for_run_time))
    fluid_name_no_srd_index =fluid_name_array_fluid_col_index[0][correct_fluid_index]
    print( fluid_name_no_srd_index)
    
    run_time_array_with_all_realisations[fluid_name_no_srd_index,realisation_index,org_var_1_index, org_var_2_index]=run_time_reader(realisation_name_for_run_time)
    #print( run_time_array_unsorted_srd_bins[fluid_name_no_srd_tuple_index[0]])
# %%

# run time vs swap rate
#NOTE: need to consider this averaging 
#NOTE: need to consider how exactly to plot, could be a quad plot with one set of data for each swap rate

run_time_array_with_all_realisations_real_av=np.mean(run_time_array_with_all_realisations,axis=1)
run_time_array_with_all_realisations_swap_av=np.mean(run_time_array_with_all_realisations_real_av,axis=1)
run_time_array_with_all_realisations_swap_num_av=np.mean(run_time_array_with_all_realisations_swap_av,axis=1)
for j in range(0,swap_number.size):
    for i in range(0,fluid_name_array.shape[0]):
   
        plt.plot(swap_rate[:],run_time_array_with_all_realisations_real_av[i,:,j])
        #plt.yscale('log')
        plt.ylabel("$f_{p}[-]$")
        plt.ylabel("$t_{\infty}[s]$", rotation=0)

    plt.show()

#%%
# run time vs SRD count 
plt.rcParams.update({'font.size': 15})
fluid_names_=["Argon","Cyclohexane","Water","Nitrogen"]
linestyle=["--","-",":","-."]
marker=["x","o","+","^"]
no_srd_values= np.asarray(fluid_name_array[:,1] , dtype=float)
#plt.plot(fluid_name_array[0:3,1],run_time_array_with_all_realisations_swap_num_av[0:3])
#plt.plot(fluid_name_array[3:6,1],run_time_array_with_all_realisations_swap_num_av[3:6])
#plt.plot(fluid_name_array[6:9,1],run_time_array_with_all_realisations_swap_num_av[6:9])
i_=[0,3,6,9]
for i in range(0,4):
    #plt.figure(figsize=(20,5))
   # linestyle=linestyle[i]
    plt.scatter(np.flip(no_srd_values[i_[i]:i_[i]+3]),np.flip(run_time_array_with_all_realisations_swap_num_av[i_[i]:i_[i]+3]),label=str(fluid_names_[i]), marker=marker[i])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("$N_{SRD}[-]$")
    plt.ylabel("$t_{\infty}[s]$", rotation=0)

    plt.legend()

plt.show()






    

# %%
