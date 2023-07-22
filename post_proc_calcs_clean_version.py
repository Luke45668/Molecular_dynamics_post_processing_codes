##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will given the correct set of input data produce all the plots and velocity profile data 

after an MPCD simulation. 
"""
#%%
#from imp import reload
import os

#from sys import exception
#from tkinter import HORIZONTAL
import numpy as np

import matplotlib.pyplot as plt
import regex as re
import pandas as pd
#import pyswarms as ps

plt.rcParams.update(plt.rcParamsDefault)
#plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
#import seaborn as sns

import scipy.stats
from datetime import datetime


path_2_post_proc_module= '/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/LAMMPS python run and analysis scripts/Analysis codes'
os.chdir(path_2_post_proc_module)
#import them in the order they should be used 
from mom2numpy import *
from velP2numpy import *
from dump2numpy import * 
import glob 
from post_MPCD_MP_processing_module import *
#importlib.reload(post_MPCD_MP_processing_module)

#%% define key inputs 
j_=3
swap_rate = np.array([3,7,15,30,60,150,300,600,900,1200]) # values chosen from original mp paper
swap_number = np.array([1,10,100,1000])
swap_number = np.array([1])
spring_constant= np.array([0.01,0.1,1,10,20,40,50,60,100,1000])
#swap_rate = np.array([15,15,15,15,15,15,15,15,15,15])
swap_rate= np.array([15])
equilibration_timesteps= 2000 # number of steps to do equilibration with 
#equilibration_timesteps=1000
VP_ave_freq =1000
chunk = 20
dump_freq=10000 # if you change the timestep rememebr to chaneg this 
thermo_freq = 10000
scaled_temp=1
scaled_timestep=0.01
realisation=np.array([0.0,1.0,2.0])
VP_output_col_count = 4 
r_particle =25e-6
phi=0.005
N=2
Vol_box_at_specified_phi=(N* (4/3)*np.pi*r_particle**3 )/phi
box_side_length=np.cbrt(Vol_box_at_specified_phi)
fluid_name='Ar'
run_number=''
no_timesteps=8000000 # rememebr to change this depending on run 

# grabbing file names 

VP_general_name_string='vel.'+fluid_name+'*_mom_output_*_no_rescale_*'

Mom_general_name_string='mom.'+fluid_name+'*_no_tstat_*_no_rescale_*'

log_general_name_string='log.'+fluid_name+'_*_inc_mom_output_no_rescale_*'

# only need this section if you are analysing dumps 
dump_general_name_string='test_run_dump_'+fluid_name+'_*'
count_dump=0
realisation_name_dump =[]


for name in glob.glob(dump_general_name_string):
        
        count_dump=count_dump+1    
        realisation_name_dump.append(name)
    

# filepath='T_1_phi_'+str(phi)+'_data/'+fluid_name+'_data_T_1_phi_'+str(phi)+'/run_'+str(run_number)+'/'
# filepath='T_1_compiled_data_all_phi'
# filepath='T_1_phi_0.005_data/H20_data_T_1_phi_0.0005/run_952534'
# filepath='Test_data_solid/phi_0.005'
filepath='T_1_phi_0.005_solid_inc_data/H20_data/run_208958'
filepath='T_1_phi_0.005_data/Nitrogen_data_T_1_phi_0.005/run_84688'
#filepath='T_1_spring_tests_solid_in_simple_shear/phi_0.005_tests'
filepath= 'T_1_spring_tests_solid_in_simple_shear/phi_0.005_tests'
#filepath= 'T_1_phi_0.0005_solid_inc_data/H20_data/run_484692'  
#filepath='T_1_phi_0.005_data/H20_data_T_1_phi_0.005/run_830466'
#filepath='T_1_phi_0.00005_solid_inc_data/H20_data/run_680943'
#filepath='T_1_phi_0.005_data/H20_data_T_1_phi_0.005/run_830466'
#filepath='T_1_compiled_data_all_phi/H20_data'

realisation_name_info=VP_and_momentum_data_realisation_name_grabber(log_general_name_string,j_,swap_number,swap_rate,VP_general_name_string,Mom_general_name_string,filepath)
realisation_name_Mom=realisation_name_info[0]
realisation_name_VP=realisation_name_info[1]
count_mom=realisation_name_info[2]
count_VP=realisation_name_info[3]

realisation_name_log=realisation_name_info[5]
count_log=realisation_name_info[6]

#checking the number of different solutions used in the run
# # locations of key info when string name is split by undescores

loc_no_SRD=8
loc_EF=20
loc_SN=22
loc_Realisation_index= 7
loc_box_size=9

# 
no_SRD=[]
box_size=[]
for i in range(0,count_VP):
    no_srd=realisation_name_VP[i].split('_')
    no_SRD.append(no_srd[loc_no_SRD])
    box_size.append(no_srd[loc_box_size])
    
no_SRD.sort(key=int)
no_SRD.sort()
no_SRD_key=[]
box_size_key=[]
#using list comprehension to remove duplicates
[no_SRD_key.append(x) for x in no_SRD if x not in no_SRD_key]
[box_size_key.append(x) for x in box_size if x not in box_size_key]
box_side_length_scaled=[]
for item in box_size_key:
    box_side_length_scaled.append(float(item))
box_side_length_scaled=np.array([box_side_length_scaled])

number_of_solutions=len(no_SRD_key)

simulation_file="MYRIAD_LAMMPS_runs/"+filepath

Path_2_VP="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file
#%% new more general version of VP organiser and reader
def VP_organiser_and_reader(loc_no_SRD,loc_org_var_1,loc_org_var_2,loc_Realisation_index,box_side_length_scaled,j_,number_of_solutions,org_var_1,org_var_2,no_SRD_key,realisation_name_VP,Path_2_VP,chunk,equilibration_timesteps,VP_ave_freq,no_timesteps,VP_output_col_count,count_VP):
    from velP2numpy import velP2numpy_f
    marker=-1
    error_count=0 
    VP_data_upper=np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,9,int(no_timesteps/VP_ave_freq),j_))
    VP_data_lower=np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,9,int(no_timesteps/VP_ave_freq),j_))
    VP_z_data_upper=np.zeros((number_of_solutions,1,9))
    VP_z_data_lower=np.zeros((number_of_solutions,1,9))
    
    for i in range(0,count_VP):
        filename=realisation_name_VP[i].split('_')
        marker=marker+1
        no_SRD=filename[loc_no_SRD]
        z=no_SRD_key.index(no_SRD)
        realisation_index=filename[loc_Realisation_index]
        j=int(float(realisation_index))
        if isinstance(filename[loc_org_var_1], int):
            org_var_1_find_from_file_name=int(filename[loc_org_var_1])
            m=np.where(org_var_1==org_var_1_find_from_file_name)
        else: 
            org_var_1_find_from_file_name=float(filename[loc_org_var_1])
            m=np.where(org_var_1==org_var_1_find_from_file_name)
        if  isinstance(filename[loc_org_var_2], int):
            org_var_2_find_from_file_name=int(filename[loc_org_var_2])
            k=np.where(org_var_2==org_var_2_find_from_file_name)
        else:    
            org_var_2_find_from_file_name=float(filename[loc_org_var_2])
            k=np.where(org_var_2==org_var_2_find_from_file_name)
        realisation_name=realisation_name_VP[i]
        try: 
            VP_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[0]
            VP_data_upper[z,m,k,:,:,j] = VP_data[1:10,:]
            VP_data_lower[z,m,k,:,:,j] = VP_data[11:,:]
            
        except Exception as e:
            print('Velocity Profile Data faulty')
            error_count=error_count+1 
            continue
        VP_z_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[1]     
       
        VP_z_data_upper[z,0,:] = VP_z_data[1:10].astype('float64')* box_side_length_scaled.T    
        VP_z_data_lower[z,0,:] = VP_z_data[11:].astype('float64')* box_side_length_scaled.T
        
    return VP_data_upper,VP_data_lower,error_count,filename,VP_z_data_upper,VP_z_data_lower

#%%
org_var_1=swap_rate
loc_org_var_1=20
org_var_2=spring_constant
loc_org_var_2=25
#VP_raw_data= VP_organiser_and_reader(loc_no_SRD,loc_EF,loc_SN,loc_Realisation_index,box_side_length_scaled,j_,number_of_solutions,swap_number,swap_rate,no_SRD_key,realisation_name_VP,Path_2_VP,chunk,equilibration_timesteps,VP_ave_freq,no_timesteps,VP_output_col_count,count_VP)
VP_raw_data= VP_organiser_and_reader(loc_no_SRD,loc_org_var_1,loc_org_var_2,loc_Realisation_index,box_side_length_scaled,j_,number_of_solutions,org_var_1,org_var_2,no_SRD_key,realisation_name_VP,Path_2_VP,chunk,equilibration_timesteps,VP_ave_freq,no_timesteps,VP_output_col_count,count_VP)

VP_data_upper=VP_raw_data[0]
VP_data_lower=VP_raw_data[1]
error_count=VP_raw_data[2]
filename=VP_raw_data[3]
VP_z_data_upper=VP_raw_data[4]
VP_z_data_lower=VP_raw_data[5]

if error_count != 0: 
    print('Error reading velocity profiles, check data !')
else:
    print('Velocity profile data success')


box_size_loc=9
lengthscale=box_side_length/float(filename[box_size_loc])

#%%log file reader and organiser
log_EF=21
log_SN=23
log_K=27
log_realisation_index=8
#def log_file_reader_and_organiser(count_log,):
log_file_col_count=4
log_file_row_count=((no_timesteps)/thermo_freq) +2 # to fit full log file  not sure on it 
log_file_tuple=()
Path_2_log='/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/'+filepath
thermo_vars='         KinEng          Temp          TotEng    '
from log2numpy import * 
org_var_log_1=swap_rate
loc_org_var_log_1=log_EF
org_var_log_2=spring_constant
loc_org_var_log_2=log_K


for i in range(0,org_var_log_1.size):
    log_4d_array_with_all_realisations=np.zeros((org_var_log_2.size,j_,int(log_file_row_count),log_file_col_count))
    log_file_tuple=log_file_tuple+(log_4d_array_with_all_realisations,)
    
averaged_log_file=np.zeros((org_var_log_1.size,org_var_log_2.size,int(log_file_row_count),4))

for i in range(0,count_log):
    filename=realisation_name_log[i].split('_')
    realisation_index=int(float(realisation_name_log[i].split('_')[log_realisation_index]))
    print(realisation_index)
    if isinstance(filename[loc_org_var_log_1],int):
       org_var_log_1_find_in_name=int(filename[loc_org_var_log_1])
       tuple_index=np.where(org_var_log_1==org_var_log_1_find_in_name)[0][0]
    else:
       org_var_log_1_find_in_name=float(filename[loc_org_var_log_1])
       tuple_index=np.where(org_var_log_1==org_var_log_1_find_in_name)[0][0]
    
    if isinstance(filename[loc_org_var_2],int):
        org_var_log_2_find_in_name=int(filename[loc_org_var_log_2])
        array_index_1= np.where(org_var_log_2==org_var_log_2_find_in_name)[0][0] 
    else:
        org_var_log_2_find_in_name=float(filename[loc_org_var_log_2])
        array_index_1= np.where(org_var_log_2==org_var_log_2_find_in_name)[0][0] 
        
    # multiple swap numbers and swap rates
    #log_file_tuple[np.where(swap_rate==swap_rate_org)[0][0]][np.where(swap_number==swap_number_org)[0][0],realisation_index,:,:]=log2numpy_reader(realisation_name_log[i],Path_2_log,thermo_vars)
    log_file_tuple[tuple_index][array_index_1,realisation_index,:,:]=log2numpy_reader(realisation_name_log[i],Path_2_log,thermo_vars)


for k in range(0,org_var_log_1.size):
    for i in range(0,org_var_log_2.size):
        averaged_log_file[k,i,:,:]=np.mean(log_file_tuple[k][i],axis=0)
        


    
    
    
#%% to analyse one file 
from velP2numpy import velP2numpy_f
marker=-1
error_count=0 
VP_data_upper=np.zeros((number_of_solutions,swap_rate.size,swap_number.size,9,int(no_timesteps/VP_ave_freq),j_))
VP_data_lower=np.zeros((number_of_solutions,swap_rate.size,swap_number.size,9,int(no_timesteps/VP_ave_freq),j_))

for i in range(0,count_VP):
    filename=realisation_name_VP[i].split('_')
    marker=marker+1
    no_SRD=filename[loc_no_SRD]
    z=no_SRD_key.index(no_SRD)
    realisation_index=filename[loc_Realisation_index]
    j=int(float(realisation_index))
    EF=int(filename[loc_EF])
    m=np.where(swap_rate==EF)
    SN=int(filename[loc_SN])
    k=np.where(swap_number==SN)
    realisation_name=realisation_name_VP[i]
    try: 
        VP_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[0]
        VP_data_upper= VP_data[1:10,:]
        VP_data_lower= VP_data[11:,:]
        
    except Exception as e:
        print('Velocity Profile Data faulty')
        error_count=error_count+1 
        continue
VP_z_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[1]     

VP_z_data_upper = np.array([VP_z_data[1:10].astype('float64')])* box_side_length_scaled.T    
VP_z_data_lower =np.array([ VP_z_data[11:].astype('float64') ])* box_side_length_scaled.T


box_size_loc=9
lengthscale=box_side_length/float(filename[box_size_loc])



 #%% saving VP data as function is very slow 
# np.save(fluid_name+'_VP_data_upper_phi'+str(phi)+'_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_data_upper)    
# np.save(fluid_name+'_VP_data_lower_phi'+str(phi)+'_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_data_lower)   
# np.save(fluid_name+'_VP_z_data_upper_phi'+str(phi)+'_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_z_data_upper)    
# np.save(fluid_name+'_VP_z_data_lower_phi'+str(phi)+'_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_z_data_lower)          
# # #%% loading vP data to avoid running fucntion 

# VP_data_upper=np.load(fluid_name+'_VP_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_data_upper)    
# VP_data_lower=np.load(fluid_name+'_VP_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_data_lower)   
# VP_z_data_upper=np.load(fluid_name+'_VP_z_data_upper_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_z_data_upper)    
# VP_z_data_lower=np.load(fluid_name+'_VP_z_data_lower_'+str(box_side_length_scaled)+'_no_timesteps_'+str(no_timesteps)+'VP_ave_freq'+str(VP_ave_freq),VP_z_data_lower)          

# %% obtaining the mom data size
#Path_2_mom_file="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file
Path_2_mom_file=Path_2_VP
org_var_mom_1=swap_rate
loc_org_var_mom_1=20
org_var_mom_2=spring_constant 
loc_org_var_mom_2=25
def mom_file_data_size_reader(j_,number_of_solutions,count_mom,realisation_name_Mom,no_SRD_key,org_var_mom_1,org_var_mom_2,Path_2_mom_file):
    from mom2numpy import mom2numpy_f
    pass_count=0
    size_list=[]
    
    for i in range(0,count_mom):
       
        
        realisation_name=realisation_name_Mom[i]
        
        
        
        mom_data_test=mom2numpy_f(realisation_name,Path_2_mom_file)  
        size_list.append(mom_data_test.shape)
        
        pass_count=pass_count+1 
    
    size_list.sort()
    size_list.reverse()
    size_list=list(dict.fromkeys(size_list))
    size_array=np.array(size_list)
    
    mom_data=()
    for i in range(0,org_var_mom_1.size):
    
         mom_data= mom_data+(np.zeros((number_of_solutions,org_var_mom_2.size,j_,(size_array[i,0]))),)
         
         
         
    return size_array,mom_data,pass_count
#%%
mom_data_pre_process= mom_file_data_size_reader(j_,number_of_solutions,count_mom,realisation_name_Mom,no_SRD_key,org_var_mom_1,org_var_mom_2,Path_2_mom_file)
size_array=mom_data_pre_process[0]
mom_data= mom_data_pre_process[1]
pass_count= mom_data_pre_process[2]

if pass_count!=count_VP:
    print("Data import error, number of momentum files should match number of velocity files" )
else:
    print("Data size assessment success!")

#%% re-writing mom organiser and reader 
def Mom_organiser_and_reader(mom_data,count_mom,realisation_name_Mom,no_SRD_key,org_var_mom_1,loc_org_var_mom_1,org_var_mom_2,loc_org_var_mom_2,Path_2_mom_file):
    from mom2numpy import mom2numpy_f
    error_count_mom=0
    failed_list_realisations=[]
    for i in range(0,count_mom):
        filename=realisation_name_Mom[i].split('_')
    
        no_SRD=filename[8]
        z=no_SRD_key.index(no_SRD)
    
        realisation_index=filename[7]
        j=int(float(realisation_index))
        if isinstance(filename[loc_org_var_mom_1],int):
            org_var_mom_1_find_in_name=int(filename[loc_org_var_mom_1])
            tuple_index=np.where(org_var_mom_1==org_var_mom_1_find_in_name)[0][0]
        else:
            org_var_mom_1_find_in_name=float(filename[loc_org_var_mom_1])
            tuple_index=np.where(org_var_mom_1==org_var_mom_1_find_in_name)[0][0]
    
        if isinstance(filename[loc_org_var_mom_2],int):
            org_var_mom_2_find_in_name=int(filename[loc_org_var_mom_2])
            array_index_1= np.where(org_var_mom_2==org_var_mom_2_find_in_name)[0][0] 
        else:
            org_var_mom_2_find_in_name=float(filename[loc_org_var_mom_2])
            array_index_1= np.where(org_var_mom_2==org_var_mom_2_find_in_name)[0][0] 
        
        realisation_name=realisation_name_Mom[i]
        
        try:
           mom_data[tuple_index][z,array_index_1,j,:]=mom2numpy_f(realisation_name,Path_2_mom_file)  
        
        

        
            
        except Exception as e:
            print('Mom Data faulty')
            error_count_mom=error_count_mom+1 
            failed_list_realisations.append(realisation_name)
            break
    return mom_data,error_count_mom,failed_list_realisations


#%% Reading in mom data files 
mom_data_files=Mom_organiser_and_reader(mom_data,count_mom,realisation_name_Mom,no_SRD_key,org_var_mom_1,loc_org_var_mom_1,org_var_mom_2,loc_org_var_mom_2,Path_2_mom_file)

mom_data=mom_data_files[0]
error_count_mom=mom_data_files[1]
failed_list_realisations=[2]

if error_count_mom !=0:
    print("Mom file error, check data files aren't damaged")
else:
    print("Mom data import success")
    
#%% Dump file reader and organiser 
def dump2numpy_f(dump_start_line,Path_2_dump,dump_realisation_name,number_of_particles_per_dump):
       
       
        dump_start_line_bytes = bytes(dump_start_line,'utf-8')

        os.chdir(Path_2_dump) #+simulation_file+"/" +filename


        with open(dump_realisation_name) as f:
                    read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) 
                    if read_data.find(dump_start_line_bytes) != -1:
                        print('Start of run found')
                        dump_byte_start=read_data.find(dump_start_line_bytes)
                        end_pattern = re.compile(b"\d\n$")
                        end_position_search = end_pattern.search(read_data)  
                    
                
                    else:
                        print('could not find dump variables')
            

        # find end of run        
        
                    if end_pattern.search(read_data) != -1:
                        print('End of run found')
                        dump_byte_end= end_position_search.span(0)[1]
                    else:
                        print('could not find end of run')
            
   
        read_data.seek(dump_byte_start) #setting relative psotion of the file 
        size_of_data =dump_byte_end-dump_byte_start
        dump_array_bytes=read_data.read(dump_byte_end)

        #finding all the matches and putting their indicies into lists 

        timestep_start_pattern = re.compile(dump_start_line_bytes) #b"ITEM: ATOMS id type x y z vx vy vz mass"
        timestep_end_pattern = re.compile(b"ITEM: TIMESTEP")

        dumpstart = timestep_start_pattern.finditer(dump_array_bytes)
        dumpend = timestep_end_pattern.finditer(dump_array_bytes)
        count=0
        dump_start_timestep=[]
        dump_end_timestep=[]


        for matches in dumpstart:
                count=count+1 # this is counted n times as we include the first occurence 
                #print(matches)
                dump_start_timestep.append(matches.span(0)[1])
                

        count1=0
        for match in dumpend:
                count1=count1+1 # this is counted n-1 times as we dont include the first occurence
                #print(match)
                dump_end_timestep.append(match.span(0)[0])


        #Splicing and storing the dumps into a list containing the dump at each timestep  
        dump_one_timestep=[]    
       
            
        for i in range(0,count-1):
                dump_one_timestep.append(dump_array_bytes[dump_start_timestep[i]:dump_end_timestep[i]])
            
        # getting last dump not marked by ITEM: TIMESTEP      
        dump_one_timestep.append(dump_array_bytes[dump_start_timestep[count-1]:])
        #dump_one_timestep_tuple=dump_one_timestep_tuple+(str(dump_array_bytes[dump_start_timestep[count-1]:]),)
        dump_one_timestep=str(dump_one_timestep)

        newline_regex_pattern= re.compile(r'\\n')
        remove_b_regex_pattern= re.compile(r'b\'')
        remove_left_bracket_regex_pattern=re.compile(r'\[')
        remove_right_bracket_regex_pattern=re.compile(r'\]')

        remove_stray_comma_regex_pattern=re.compile(r'\,')
        remove_stray_appost_regex_pattern = re.compile(r'\'')
        empty=' '


        dump_one_timestep=re.sub(newline_regex_pattern,empty,dump_one_timestep)

        dump_one_timestep=re.sub(remove_b_regex_pattern,empty,dump_one_timestep)
        dump_one_timestep=re.sub(remove_left_bracket_regex_pattern,empty,dump_one_timestep)
        dump_one_timestep=re.sub(remove_right_bracket_regex_pattern,empty,dump_one_timestep)
        dump_one_timestep=re.sub(remove_stray_comma_regex_pattern,empty,dump_one_timestep)
        dump_one_timestep=re.sub(remove_stray_appost_regex_pattern ,empty,dump_one_timestep)
            # needs to remove all the line markers and the b'
            
        dump_one_timestep=dump_one_timestep.split()
        for i in range(0,count): 
           # print(i)
            dump_one_timestep[i]=float(dump_one_timestep[i])
            
        
        #print(len(dump_one_timestep))
        particle_sorted_array_1=[]  #np.zeros((1,len(dump_start_line.split()) -2)) #*number_of_particles_per_dump))    
        particle_sorted_array_2=[] 
        total_number_of_readouts_for_run=count
        #print(total_number_of_readouts_for_run)
        
        number_cols_dump = (len(dump_start_line.split()) -2) #*number_of_particles_per_dump  # minus one to remove item: 
        number_rows_dump= count *number_of_particles_per_dump 

        dump_file_unsorted=np.reshape(np.array(dump_one_timestep),(number_rows_dump,number_cols_dump))
        #particle_sorted_array=np.array([[]])
        # need to extend this loop for more particles and generalise it if you want N particles
        j=0
        # particle 1
        for i in range(0,number_rows_dump):
                
                
                if dump_file_unsorted[i,0]=='1.0':
                        #insert_point==[j+1,0:11]
                        particle_sorted_array_1.append(dump_file_unsorted[i,:])
                        #print(np.append(dump_file_unsorted[i,:],particle_sorted_array_1[0,0:11],axis=0))
                        #print(dump_file_unsorted[i,:])
                elif dump_file_unsorted[i,0]=='1':   
                        #insert_point=[j+1,0:11]
                        particle_sorted_array_1.append(dump_file_unsorted[i,:])
                        #print(dump_file_unsorted[i,:])
                else:
                     continue
        #particle 2
        for i in range(0,number_rows_dump):
                
                
                if dump_file_unsorted[i,0]=='2.0':
                        #insert_point==[j+1,0:11]
                        particle_sorted_array_2.append(dump_file_unsorted[i,:])
                        #print(np.append(dump_file_unsorted[i,:],particle_sorted_array_1[0,0:11],axis=0))
                        #print(dump_file_unsorted[i,:])
                elif dump_file_unsorted[i,0]=='2':   
                        #insert_point=[j+1,0:11]
                        particle_sorted_array_2.append(dump_file_unsorted[i,:])
                        #print(dump_file_unsorted[i,:])
                else:
                     continue
        
        dump_file_1=np.stack(particle_sorted_array_1)
        dump_file_2=np.stack(particle_sorted_array_2)
       
       
        
        return dump_file_1,dump_file_2,dump_file_unsorted
#%%
org_var_dump_1=swap_rate
loc_org_var_dump_1=19
org_var_dump_2=spring_constant
loc_org_var_dump_2=23

dump_start_line='ITEM: ATOMS id type x y z vx vy vz omegax omegay omegaz'
dump_realisation_name_size_check=realisation_name_dump[0]
Path_2_dump="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/"+filepath
number_of_particles_per_dump=2 # only loooking at the solids
dump_file_test_1=dump2numpy_f(dump_start_line,Path_2_dump,dump_realisation_name_size_check,number_of_particles_per_dump)[0]
dump_file_test_2=dump2numpy_f(dump_start_line,Path_2_dump,dump_realisation_name_size_check,number_of_particles_per_dump)[1]
# dump_one_timestep=dump2numpy_f(dump_start_line,Path_2_dump,dump_realisation_name_size_check,number_of_particles_per_dump)[1]

dump_file_tuple_1=()
dump_file_tuple_2=()
# particle 1
for i in range(0,org_var_dump_1.size):
    dump_file_array=np.zeros((org_var_dump_2.size,j_,dump_file_test_1.shape[0],dump_file_test_1.shape[1]))
    dump_file_tuple_1=dump_file_tuple_1+(dump_file_array,)
    #dump_file_tuple_2=dump_file_tuple_2+(dump_file_array,)

for i in range(0,count_dump):
        dump_file=dump2numpy_f(dump_start_line,Path_2_dump,realisation_name_dump[i],number_of_particles_per_dump)
        filename=realisation_name_dump[i].split('_')
    
        
        realisation_index=filename[6]
        j=int(float(realisation_index))
        if isinstance(filename[loc_org_var_dump_1],int):
            org_var_dump_1_find_in_name=int(filename[loc_org_var_dump_1])
            tuple_index=np.where(org_var_dump_1==org_var_dump_1_find_in_name)[0][0]
        else:
            org_var_dump_1_find_in_name=float(filename[loc_org_var_dump_1])
            tuple_index=np.where(org_var_dump_1==org_var_dump_1_find_in_name)[0][0]
        #[:-5] is due to the final number being attached to .dump eg. 0.01.dump
        if isinstance(filename[loc_org_var_dump_2],int):
            org_var_dump_2_find_in_name=int(filename[loc_org_var_dump_2][:-5])
            array_index_1= np.where(org_var_dump_2==org_var_dump_2_find_in_name)[0][0] 
        else:
            org_var_dump_2_find_in_name=float(filename[loc_org_var_dump_2][:-5])
            array_index_1= np.where(org_var_dump_2==org_var_dump_2_find_in_name)[0][0] 
        
        dump_file_1=dump_file[0]
        #print(dump_file_1)
        dump_file_tuple_1[tuple_index][array_index_1,j,:,:]=dump_file_1
       # print(dump_file_tuple_1[tuple_index][array_index_1,j,:,:])
        

        # dump_file_2=dump_file[1]
        # print(dump_file_2)
        # dump_file_tuple_2[tuple_index][array_index_1,j,:,:]=dump_file_2
        # print(dump_file_tuple_2[tuple_index][array_index_1,j,:,:])
  
# particle 2
for i in range(0,org_var_dump_1.size):
    dump_file_array=np.zeros((org_var_dump_2.size,j_,dump_file_test_1.shape[0],dump_file_test_1.shape[1]))
    dump_file_tuple_2=dump_file_tuple_2+(dump_file_array,)

for i in range(0,count_dump):
        dump_file=dump2numpy_f(dump_start_line,Path_2_dump,realisation_name_dump[i],number_of_particles_per_dump)
        filename=realisation_name_dump[i].split('_')
    
        
        realisation_index=filename[6]
        j=int(float(realisation_index))
        if isinstance(filename[loc_org_var_dump_1],int):
            org_var_dump_1_find_in_name=int(filename[loc_org_var_dump_1])
            tuple_index=np.where(org_var_dump_1==org_var_dump_1_find_in_name)[0][0]
        else:
            org_var_dump_1_find_in_name=float(filename[loc_org_var_dump_1])
            tuple_index=np.where(org_var_dump_1==org_var_dump_1_find_in_name)[0][0]
        #[:-5] is due to the final number being attached to .dump eg. 0.01.dump
        if isinstance(filename[loc_org_var_dump_2],int):
            org_var_dump_2_find_in_name=int(filename[loc_org_var_dump_2][:-5])
            array_index_1= np.where(org_var_dump_2==org_var_dump_2_find_in_name)[0][0] 
        else:
            org_var_dump_2_find_in_name=float(filename[loc_org_var_dump_2][:-5])
            array_index_1= np.where(org_var_dump_2==org_var_dump_2_find_in_name)[0][0] 
        
        dump_file_2=dump_file[1]
       # print(dump_file_2)
        dump_file_tuple_2[tuple_index][array_index_1,j,:,:]=dump_file_2

       # print(dump_file_tuple_2[tuple_index][array_index_1,j,:,:])
    
#%% Now assess the steady state of the VP data 
org_var_1=swap_rate
org_var_2=spring_constant 
def VP_data_averaging_and_stat_test_data(VP_z_data_upper,VP_z_data_lower,no_timesteps,VP_data_lower,VP_data_upper,number_of_solutions,org_var_1,org_var_2,VP_ave_freq):
    VP_data_lower_realisation_averaged = np.mean(VP_data_lower,axis=5) 
    VP_data_upper_realisation_averaged = np.mean(VP_data_upper,axis=5) 
    x_u= np.array(VP_data_upper_realisation_averaged)
    x_l= np.array(VP_data_lower_realisation_averaged)
    y_u=np.zeros((number_of_solutions,VP_z_data_upper.shape[2],VP_data_upper.shape[4]))
    y_l=np.zeros((number_of_solutions,VP_z_data_upper.shape[2],VP_data_upper.shape[4]))

    for z in range(number_of_solutions):
        y_u[z,:,:]= np.reshape(np.repeat(VP_z_data_upper[z,0,:],VP_data_upper.shape[4],axis=0),(VP_z_data_upper.shape[2],VP_data_upper.shape[4]))

        y_l[z,:,:]= np.reshape(np.repeat(VP_z_data_lower[z,0,:],VP_data_lower.shape[4],axis=0),(VP_z_data_lower.shape[2],VP_data_lower.shape[4]))

    pearson_coeff_upper= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_upper_realisation_averaged.shape[4]))
    pearson_coeff_lower= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_lower_realisation_averaged.shape[4]))
    shear_rate_upper= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_upper_realisation_averaged.shape[4]))    
    shear_rate_lower= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_lower_realisation_averaged.shape[4]))
    shear_rate_upper_error= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_upper_realisation_averaged.shape[4]))    
    shear_rate_lower_error= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_lower_realisation_averaged.shape[4]))
    timestep_points=np.array([[[np.linspace(1,VP_data_upper.shape[4],int(float(no_timesteps)/float(VP_ave_freq)))]]])*VP_ave_freq
    timestep_points=np.repeat(timestep_points, number_of_solutions,axis=0)
    timestep_points=np.repeat(timestep_points, org_var_1.size,axis=1)     
    timestep_points=np.repeat(timestep_points, org_var_2.size,axis=2)   



    for z in range(0,number_of_solutions): 
        for m in range(0,org_var_1.size):
            for k in range(0,org_var_2.size):
                for i in range(0,VP_data_upper.shape[4]):
                        
                        pearson_coeff_upper[z,m,k,i] =scipy.stats.pearsonr(y_u[z,:,i],x_u[z,m,k,:,i] )[0]
                        shear_rate_upper[z,m,k,i]= scipy.stats.linregress(y_u[z,:,i],x_u[z,m,k,:,i] ).slope
                        shear_rate_upper_error[z,m,k,i]= scipy.stats.linregress(y_u[z,:,i],x_u[z,m,k,:,i] ).stderr
                        pearson_coeff_lower[z,m,k,i] =scipy.stats.pearsonr(y_l[z,:,i],x_l[z,m,k,:,i] )[0]
                        shear_rate_lower[z,m,k,i]= scipy.stats.linregress(y_l[z,:,i],x_l[z,m,k,:,i] ).slope  
                        shear_rate_lower_error[z,m,k,i]= scipy.stats.linregress(y_l[z,:,i],x_l[z,m,k,:,i] ).stderr 
     
    return pearson_coeff_upper,shear_rate_upper,pearson_coeff_lower,shear_rate_lower,timestep_points,VP_data_lower_realisation_averaged,VP_data_upper_realisation_averaged,shear_rate_upper_error,shear_rate_lower_error
        
VP_shear_rate_and_stat_data=VP_data_averaging_and_stat_test_data(VP_z_data_upper,VP_z_data_lower,no_timesteps,VP_data_lower,VP_data_upper,number_of_solutions,org_var_1,org_var_2,VP_ave_freq)
pearson_coeff_upper=VP_shear_rate_and_stat_data[0]
shear_rate_upper=VP_shear_rate_and_stat_data[1]
pearson_coeff_lower=VP_shear_rate_and_stat_data[2]
shear_rate_lower=VP_shear_rate_and_stat_data[3]
timestep_points=VP_shear_rate_and_stat_data[4]
VP_data_lower_realisation_averaged=VP_shear_rate_and_stat_data[5]
VP_data_upper_realisation_averaged=VP_shear_rate_and_stat_data[6]
shear_rate_upper_error=VP_shear_rate_and_stat_data[7]
shear_rate_lower_error=VP_shear_rate_and_stat_data[8]

#%% Averaging for one file 

VP_z_data_upper_repeated= np.repeat(VP_z_data_upper.T,VP_data_upper.shape[1],axis=1)
VP_z_data_lower_repeated= np.repeat(VP_z_data_lower.T,VP_data_lower.shape[1],axis=1)
pearson_coeff_upper= np.zeros(VP_data_upper.shape[1])
pearson_coeff_lower= np.zeros(VP_data_lower.shape[1])
shear_rate_upper= np.zeros(VP_data_upper.shape[1])    
shear_rate_lower= np.zeros(VP_data_lower.shape[1])
shear_rate_upper_error= np.zeros(VP_data_upper.shape[1]) 
shear_rate_lower_error= np.zeros(VP_data_lower.shape[1])

for i in range(0,VP_data_upper.shape[1]):
    pearson_coeff_upper[i]=scipy.stats.pearsonr(VP_data_upper[:,i],VP_z_data_upper_repeated[:,i])[0]
    shear_rate_upper[i]= scipy.stats.linregress(VP_z_data_upper_repeated[:,i],VP_data_upper[:,i]).slope
    shear_rate_upper_error[i]= scipy.stats.linregress(VP_data_upper[:,i],VP_z_data_upper_repeated[:,i] ).stderr
    pearson_coeff_lower[i] =scipy.stats.pearsonr(VP_data_lower[:,i],VP_z_data_lower_repeated[:,i] )[0]
    shear_rate_lower[i]= scipy.stats.linregress(VP_z_data_lower_repeated[:,i] ,VP_data_lower[:,i]).slope  
    shear_rate_lower_error[i]= scipy.stats.linregress(VP_data_lower[:,i],VP_z_data_lower_repeated [:,i] ).stderr 
timestep_points=np.array([[[np.linspace(1,VP_data_upper.shape[1],int(float(no_timesteps)/float(VP_ave_freq)))]]])*VP_ave_freq

plt.plot(timestep_points[0,0,0,:],shear_rate_upper[:])
plt.plot(timestep_points[0,0,0,:],shear_rate_lower[:])
plt.xlabel('$N_{t}[-]$')
plt.ylabel('$\dot{\gamma}[\\tau]$',rotation='horizontal')
plt.title(fluid_name+" simulation run with all $f_{v,x}$ and $N_{v,x}$, $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")

plt.show()

import log2numpy
Path_2_log='/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/Test_data_solid/logs'
realisation_name='log.H20_solid396988_inc_mom_output_no_rescale_243550_0.0_9112_118.77258268303078_0.001_1781_1000_10000_1000000_T_1.0_lbda_1.3166259218664098_SR_7_SN_1_rparticle_10.0'
thermo_vars='         KinEng          Temp          TotEng'
log_data= log2numpy.log2numpy(Path_2_log,thermo_vars,realisation_name)[0]
fontsize=15
labelpad=20
#plotting temp vs time 
plt.plot(log_data[:,0],log_data[:,2])
temp=1
x=np.repeat(temp,log_data[:,0].shape[0])
plt.plot(log_data[:,0],x[:])
plt.xlabel('$N_{t}[-]$',fontsize=fontsize)
plt.ylabel('$T[\\frac{T k_{B}}{\\varepsilon}]$', rotation=0,fontsize=fontsize,labelpad=labelpad)
plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+", $f_{v,x}=$"+str(swap_rate[0])+", $N_{v,x}=$"+str(swap_number[0])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
                
plt.show()

#plotting energy vs time 
plt.plot(log_data[:,0],log_data[:,3])
plt.xlabel('$N_{t}[-]$',fontsize=fontsize)
plt.ylabel('$E_{t}[\\frac{\\tau^{2}}{\mu \ell^{2}}]$', rotation=0,fontsize=fontsize,labelpad=labelpad)
plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+", $f_{v,x}=$"+str(swap_rate[0])+", $N_{v,x}=$"+str(swap_number[0])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
     
plt.show()




# need to fix log file removing warning messages

#%%
plt.rcParams.update({'font.size': 15})   
import sigfig
lengthscale= sigfig.round(lengthscale,sigfigs=3)
box_size_nd= box_side_length_scaled 
# get rid of this on laptop or code will fail 
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

org_var_1_index_start=0
org_var_1_index_end=1
org_var_2_index_start=0
org_var_2_index_end=10

def plot_shear_rate_to_asses_SS(org_var_2_index_end,org_var_2_index_start,org_var_1_index_start,org_var_1_index_end,no_timesteps,phi,lengthscale,timestep_points,scaled_temp,number_of_solutions,org_var_1,org_var_2,shear_rate_upper,shear_rate_lower,fluid_name,box_size_nd):
    for z in range(0,number_of_solutions): 
        #for k in range(org_var_2_index_start,org_var_2_index_end):
        for m in range(org_var_1_index_start,org_var_1_index_end):
             for k in range(org_var_2_index_start,org_var_2_index_end):
                plt.plot(timestep_points[0,0,0,:],shear_rate_upper[z,m,k,:])
                plt.plot(timestep_points[0,0,0,:],shear_rate_lower[z,m,k,:])
                plt.xlabel('$N_{t}[-]$')
                plt.ylabel('$\dot{\gamma}[\\tau]$',rotation='horizontal')
                #plt.title(fluid_name+" simulation run with all $f_{v,x}$ and $N_{v,x}=$"+str(org_var_2[m])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
                plt.title(fluid_name+" simulation run with all $K$ and $f_{v,x}=$"+str(org_var_1[m])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
                
             plt.show()
        #plot_save=input("save figure?, YES/NO")
        # if plot_save=='YES':
        #     plt.savefig(fluid_name+'_T_'+str(scaled_temp)+'_length_scale_'+str(lengthscale)+'_phi_'+str(phi)+'_no_timesteps_'+str(no_timesteps)+'.png')
        # else:
        #     print('Thanks for checking steady state')

plot_shear_rate_to_asses_SS(org_var_2_index_end,org_var_2_index_start,org_var_1_index_start,org_var_1_index_end,no_timesteps,phi,lengthscale,timestep_points,scaled_temp,number_of_solutions,org_var_1,org_var_2,shear_rate_upper,shear_rate_lower,fluid_name,box_size_nd)
# need to save this plot 

# %%
truncation_timestep=1000000 # for H20 and Nitrogen 
truncation_and_SS_averaging_data=  truncation_step_and_SS_average_of_VP_and_stat_tests(shear_rate_upper_error,shear_rate_lower_error,timestep_points,pearson_coeff_lower,pearson_coeff_upper,shear_rate_upper,shear_rate_lower,VP_ave_freq,truncation_timestep,VP_data_lower_realisation_averaged,VP_data_upper_realisation_averaged)
standard_deviation_upper_error=truncation_and_SS_averaging_data[0]
standard_deviation_lower_error=truncation_and_SS_averaging_data[1]
pearson_coeff_upper_mean_SS=truncation_and_SS_averaging_data[2]
pearson_coeff_lower_mean_SS=truncation_and_SS_averaging_data[3]
pearson_coeff_mean_SS= (np.abs(pearson_coeff_lower_mean_SS)+np.abs(pearson_coeff_upper_mean_SS))*0.5
shear_rate_lower_steady_state_mean=truncation_and_SS_averaging_data[4]
shear_rate_upper_steady_state_mean=truncation_and_SS_averaging_data[5]
VP_steady_state_data_lower_truncated_time_averaged=truncation_and_SS_averaging_data[6]
VP_steady_state_data_upper_truncated_time_averaged=truncation_and_SS_averaging_data[7]
shear_rate_upper_steady_state_mean_error=truncation_and_SS_averaging_data[8]
shear_rate_lower_steady_state_mean_error=truncation_and_SS_averaging_data[9]

# could probably vectorise this or use a method 
error_count = 0
for z in range(0,number_of_solutions):
    for k in range(org_var_1_index_start,org_var_1_index_end):
            for m in range(org_var_2_index_start,org_var_2_index_end):
                 
                 
                 if pearson_coeff_mean_SS[z,k,m]<0.7:
                     print('Non-linear simulation run please inspect')
                     error_count=error_count +1 
                 else:
                     print('Great success')
                     
  
                     
                
                
                 
print('Non-linear simulation count: ',error_count)
marker=['x','o','+','^',"1","X","d","*","P","v"]
for z in range(0,number_of_solutions):
    for k in range(org_var_1_index_start,org_var_1_index_end):
            for m in range(org_var_2_index_start,org_var_2_index_end):
              plt.scatter(org_var_2[m],pearson_coeff_mean_SS[z,k,m],marker=marker[m])
              #plt.xlabel('$f_{v,x}[-]$')
              plt.xlabel('$K[\\frac{\\tau}{\mu}]$')
              plt.xscale('log')
              plt.ylabel('$P_{C}[-]$',rotation=0,labelpad=30)
    #plt.legend(loc=7,bbox_to_anchor=(1.3, 0.5))
              
    plt.show()

print('Non-linear simulation count: ',error_count)
                
#%% assess gradient of truncated shear rate data to determine steady state
# can then truncate again 
slope_shear_rate_upper=  np.zeros((number_of_solutions,org_var_1.size,org_var_2.size))
slope_shear_rate_lower=  np.zeros((number_of_solutions,org_var_1.size,org_var_2.size))

gradient_tolerance= 5e-9
for z in range(0,number_of_solutions): 
        for m in range(org_var_1_index_start,org_var_1_index_end):
            for k in range(org_var_2_index_start,org_var_2_index_end):
                slope_shear_rate_upper[z,m,k]=np.polyfit(timestep_points[0,0,0,:],shear_rate_upper[z,m,k,:],1)[0]
                slope_shear_rate_lower[z,m,k]=np.polyfit(timestep_points[0,0,0,:],shear_rate_upper[z,m,k,:],1)[0]
                if np.abs(slope_shear_rate_upper[z,m,k]) < gradient_tolerance:
                    slope_shear_rate_upper[z,m,k] =slope_shear_rate_upper[z,m,k] 
                else: 
                    #slope_shear_rate_upper[z,m,k]='NaN'
                    print('FAILED run, exclude from data ')
                if np.abs(slope_shear_rate_lower[z,m,k]) < gradient_tolerance:
                    slope_shear_rate_lower[z,m,k] =slope_shear_rate_lower[z,m,k] 
                else: 
                    #slope_shear_rate_lower[z,m,k]='NaN'
                    print('FAILED run, exclude from data ')
print("if no fail statements, data can be considered steady")

# plotting gradient of the shear vs time plot 
for z in range(0,number_of_solutions): 
     for m in range(org_var_1_index_start,org_var_1_index_end):
        #for k in range(org_var_2_index_start,org_var_2_index_end):
            plt.yscale('log')
            plt.ylabel(' $d \dot{\gamma}/d t\ [1/\\tau^{2}]$',rotation=0)
            #plt.xlabel('$f_{v,x}[-]$')
            plt.xlabel('$K[\\frac{\\tau}{\mu}]$')
            plt.xscale('log')
            plt.scatter(org_var_2[:],slope_shear_rate_upper[z,m,:])
            plt.scatter(org_var_2[:],slope_shear_rate_lower[z,m,:])
            #plt.title("Needs a title")
plt.show

# is this sufficient ?? 

#%%
# np.save(fluid_name+'_standard_deviation_upper_error_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),standard_deviation_upper_error)    
# np.save(fluid_name+'_standard_deviation_lower_error_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),standard_deviation_lower_error)    
# np.save(fluid_name+'_pearson_coeff_upper_mean_SS_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),pearson_coeff_upper_mean_SS)    
# np.save(fluid_name+'_pearson_coeff_lower_mean_SS_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),pearson_coeff_lower_mean_SS)    
# np.save(fluid_name+'_shear_rate_lower_steady_state_mean_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),shear_rate_lower_steady_state_mean)          
# np.save(fluid_name+'_shear_rate_upper_steady_state_mean_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),shear_rate_upper_steady_state_mean)          
# np.save(fluid_name+'VP_steady_state_data_lower_truncated_time_averaged_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_steady_state_data_lower_truncated_time_averaged)          
# np.save(fluid_name+'VP_steady_state_data_upper_truncated_time_averaged_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_steady_state_data_upper_truncated_time_averaged)          
# #np.save(fluid_name+'mom_data_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),mom_data)
# np.save(fluid_name+'VP_z_data_upper_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_z_data_upper)
# np.save(fluid_name+'VP_z_data_lower_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_z_data_lower)

#%% plotting E vs Nt and T vs Nt
# legend colours still dont match 
fontsize=15
labelpad=20
#plotting temp vs time 
temp=1
legendx=1.4
for k in range(0,org_var_1.size):
    for i in range(org_var_2.size):
        
        plt.plot(averaged_log_file[k,i,:,0],averaged_log_file[k,i,:,2],label='$K=${}'.format(org_var_2[i]))
    
        x=np.repeat(temp,averaged_log_file[k,i,:,0].shape[0])
        #plt.plot(averaged_log_file[k,i,:,0],x[:],label='$f_v=${}'.format(org_var_2[i]))
       # plt.plot(averaged_log_file[k,i,:,0],x[:],label='$K=${}'.format(org_var_2[i]))
        plt.xlabel('$N_{t}[-]$',fontsize=fontsize)
        plt.ylabel('$T[\\frac{T k_{B}}{\\varepsilon}]$', rotation=0,fontsize=fontsize,labelpad=labelpad)
        #plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+",All $f_{v,x}=$, $N_{v,x}=$"+str(org_var_1[k])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
        plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+", All K, $f_{v,x}=$"+str(org_var_1[k])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
        plt.legend(loc=7,bbox_to_anchor=(legendx, 0.5))
    plt.show()

    #plotting energy vs time 
    for i in range(org_var_2.size):
        #plotting energy vs time 
        #plt.plot(averaged_log_file[k,i,:,0],averaged_log_file[k,i,:,3],label='$f_v=${}'.format(org_var_2[i]))
        plt.plot(averaged_log_file[k,i,:,0],averaged_log_file[k,i,:,3],label='$K=${}'.format(org_var_2[i]))
        plt.xlabel('$N_{t}[-]$',fontsize=fontsize)
        plt.ylabel('$E_{t}[\\frac{\\tau^{2}}{\mu \ell^{2}}]$', rotation=0,fontsize=fontsize,labelpad=labelpad)
        #plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+",All $f_{v,x}=$, $N_{v,x}=$"+str(org_var_1[k])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
        plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+", All $K$, $f_{v,x}=$"+str(org_var_1[k])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
        
        plt.legend(loc=7,bbox_to_anchor=(legendx, 0.5))
    plt.show()

#%% for variable spring constant 
for k in range(0,variable_choice.size):
    
        plt.plot(averaged_log_file[k,0,:,0],averaged_log_file[k,0,:,2])
    
        x=np.repeat(temp,averaged_log_file[k,i,:,0].shape[0])
        plt.plot(averaged_log_file[k,i,:,0],x[:],label='$K=${}'.format(spring_constant[k]))
        plt.xlabel('$N_{t}[-]$',fontsize=fontsize)
        plt.ylabel('$T[\\frac{T k_{B}}{\\varepsilon}]$', rotation=0,fontsize=fontsize,labelpad=labelpad)
        #plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+",All $f_{v,x}=$, $N_{v,x}=$"+str(org_var_1[k])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
        plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+", $f_{v,x}=$"+str(org_var_2[i])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
        plt.legend(loc=7,bbox_to_anchor=(1.3, 0.5))
plt.show()

    #plotting energy vs time 
for k in range(0,variable_choice.size):
    
        #plotting energy vs time 
        plt.plot(averaged_log_file[k,0,:,0],averaged_log_file[k,0,:,3],label='$K=${}'.format(spring_constant[k]))
        plt.xlabel('$N_{t}[-]$',fontsize=fontsize)
        plt.ylabel('$E_{t}[\\frac{\\tau^{2}}{\mu \ell^{2}}]$', rotation=0,fontsize=fontsize,labelpad=labelpad)
        #plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+",All $f_{v,x}=$, $N_{v,x}=$"+str(org_var_1[k])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
        plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+", $f_{v,x}=$"+str(org_var_2[i])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
        plt.legend(loc=7,bbox_to_anchor=(1.3, 0.5))
plt.show()

#%% mom_data_averaging_and_flux_calc

from post_MPCD_MP_processing_module import *
 
flux_ready_for_plotting=mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,org_var_1,truncation_timestep,org_var_2,scaled_timestep,no_timesteps,box_side_length_scaled[0,0],mom_data)[0]
mom_data_realisation_averaged_truncated=mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,org_var_1,truncation_timestep,org_var_2,scaled_timestep,no_timesteps,box_side_length_scaled[0,0],mom_data)[1]
#np.save(fluid_name+'_flux_ready_for_plotting_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),flux_ready_for_plotting)
#(1,4,10)
#%% importin momentum after steady state
def mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,org_var_1,truncation_timestep,org_var_2,scaled_timestep,no_timesteps,box_side_length_scaled,mom_data):
    mom_data_realisation_averaged=()
    number_swaps_before_truncation=(np.ceil(truncation_timestep/org_var_2)).astype(int)
    mom_data_realisation_averaged_truncated=()
    flux_x_momentum_z_direction=np.zeros((number_of_solutions,org_var_1.size,org_var_2.size))
    total_run_time=scaled_timestep* (no_timesteps-truncation_timestep)
    
    flux_ready_for_plotting=np.zeros((number_of_solutions,org_var_1.size,org_var_2.size))
    for z in range(0,number_of_solutions):    
        for i in range(0,org_var_2.size):
            box_area_nd=float(box_size_key[z])**2
            mom_data_realisation_averaged=mom_data_realisation_averaged+(np.mean(mom_data[i],axis=2),)


        #for i in range(0,org_var_2.size):

            mom_data_realisation_averaged_truncated=mom_data_realisation_averaged_truncated+(mom_data_realisation_averaged[i][:,:,number_swaps_before_truncation[i]:],)


        # now apply the MP formula 
            mom_difference= mom_data_realisation_averaged_truncated[i][z,:,-1]-mom_data_realisation_averaged_truncated[i][z,:,0]
            flux_x_momentum_z_direction[z,:,i]=(mom_difference)/(2*total_run_time*float(box_area_nd))
            
    flux_ready_for_plotting=np.log((np.abs(flux_x_momentum_z_direction)))
    
    return flux_ready_for_plotting,mom_data_realisation_averaged_truncated


#%%
flux_ready_for_plotting=mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,org_var_1,truncation_timestep,org_var_2,scaled_timestep,no_timesteps,box_side_length_scaled[0,0],mom_data)[0]
mom_data_realisation_averaged_truncated=mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,org_var_1,truncation_timestep,org_var_2,scaled_timestep,no_timesteps,box_side_length_scaled[0,0],mom_data)[1]
# flux vs shear regression line 
#%%
shear_rate_mean_of_both_cells=(((np.abs(shear_rate_lower_steady_state_mean)+np.abs(shear_rate_upper_steady_state_mean))*0.5))
shear_rate_mean_error_of_both_cells=(np.abs(shear_rate_lower_steady_state_mean_error)+np.abs(shear_rate_upper_steady_state_mean_error))*0.5
print(shear_rate_mean_of_both_cells.shape)
print(shear_rate_mean_error_of_both_cells.shape)
shear_rate_mean_error_of_both_cells_relative=shear_rate_mean_error_of_both_cells/shear_rate_mean_of_both_cells
shear_rate_mean_of_both_cells=np.log(((np.abs(shear_rate_lower_steady_state_mean)+np.abs(shear_rate_upper_steady_state_mean))*0.5))
shear_rate_mean_error_of_both_cells=shear_rate_mean_of_both_cells*shear_rate_mean_error_of_both_cells_relative
# np.save(fluid_name+'_shear_rate_mean_error_of_both_cells_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),shear_rate_mean_error_of_both_cells)
# np.save(fluid_name+'_shear_rate_mean_of_both_cells_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),shear_rate_mean_of_both_cells)
# # reshape

#shear_rate_mean_of_both_cells= np.reshape(shear_rate_mean_of_both_cells,(flux_ready_for_plotting.shape))
# shear_rate_mean_error_of_both_cells_relative= np.reshape(shear_rate_mean_error_of_both_cells_relative,(flux_ready_for_plotting.shape))
# shear_rate_mean_error_of_both_cells=shear_rate_mean_of_both_cells*shear_rate_mean_error_of_both_cells_relative
print(shear_rate_mean_of_both_cells.shape)
print(shear_rate_mean_error_of_both_cells.shape)

flux_vs_shear_regression_line_params=()
x=shear_rate_mean_of_both_cells
#shear_rate_mean_of_both_cells=np.reshape(shear_rate_mean_of_both_cells,(flux_ready_for_plotting.shape))

def func4(x, a, b):
   #return np.log(a) + np.log(b*x)
   #return (a*(x**b))
   return (a*x) +b 
   #return a*np.log(b*x)+c



for z in range(0,number_of_solutions):    
    for i in range(0,org_var_1.size):
      
        flux_vs_shear_regression_line_params= flux_vs_shear_regression_line_params+(scipy.optimize.curve_fit(func4,shear_rate_mean_of_both_cells[z,:,i],flux_ready_for_plotting[z,i,:],method='lm',maxfev=5000)[0],)
        #print(scipy.optimize.curve_fit(func4,shear_rate_mean_of_both_cells[z,:,i],flux_ready_for_plotting[z,i,:],method='lm',maxfev=5000)[0])

params=flux_vs_shear_regression_line_params 

#np.save(fluid_name+'_params_phi'+str(phi)+'_'+str(box_side_length_scaled[0,0])+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),params)

#%%
 #calculating error of flux
# plot cumulative momentum exchange vs time 
# fit to linear grad and take error 
# need to calculate number of swaps done, plot that as time axes 

#total_number_of_swaps_after_SS=(np.floor( (no_timesteps-truncation_timestep)/org_var_2))
swap_timestep_vector=()
total_run_time=scaled_timestep* (no_timesteps-truncation_timestep)
for z in range(org_var_2_index_start,org_var_1_index_end):
    total_number_of_swaps_after_SS=mom_data_realisation_averaged_truncated[z].shape[2]
    final_swap_step= truncation_timestep +(total_number_of_swaps_after_SS*org_var_2[z])
    #print(final_swap_step)
    swap_timestep_vector= swap_timestep_vector+ (np.arange(truncation_timestep,final_swap_step,int(org_var_2[z])),)

slope_momentum_vector_error=()
slope_momentum_vector_error_1=()
pearson_coeff_momentum=()
slope_momentum_vector_mean_abs_error= np.zeros((number_of_solutions,org_var_1_index_end,org_var_1_index_end))
slope_flux_abs_error=np.zeros((number_of_solutions,org_var_1_index_end,org_var_1_index_end))
for z in range(0,number_of_solutions):
    box_area_nd=float(box_size_key[z])**2
    for k in range(org_var_1_index_start,org_var_1_index_end:
            for m in range(org_var_2_index_start,org_var_1_index_end):
                plt.scatter(swap_timestep_vector[m],-mom_data_realisation_averaged_truncated[m][z,k,:],label='$f_v=${}'.format(org_var_2[m]),marker='x', s=0.00005)
                plt.legend()
                plt.ylabel('$P_x$',rotation=0)
                plt.xlabel('$N_t$')
                
                #slope_momentum_vector_error=slope_momentum_vector_error + (np.polyfit(swap_timestep_vector[m],-mom_data_realisation_averaged_truncated[m][z,k,:],1,full=True)[1],)
                #slope_momentum_vector_error_1=slope_momentum_vector_error_1 + (scipy.stats.linregress(swap_timestep_vector[m],-mom_data_realisation_averaged_truncated[m][z,k,:] ).stderr,)
                pearson_coeff_momentum=pearson_coeff_momentum+ (scipy.stats.pearsonr(swap_timestep_vector[m],-mom_data_realisation_averaged_truncated[m][z,k,:])[0],)
                if pearson_coeff_momentum[m]  > 0.9999:
                    print('All pearson coeffs are perfectly linear, therefore there is no error in the total momentum')
                else:
                    print('Cumulative total momentum is not linear in time, please check data is has reached SS')
                #scipy.stats.linregress(swap_timestep_vector[m],-mom_data_realisation_averaged_truncated[m][z,k,:] ).stderr
                # print(mom_data_realisation_averaged_truncated[m][z,k,:].shape[0])
                # slope_momentum_vector_mean_abs_error[z,k,m]= np.sqrt(slope_momentum_vector_error[m][0]/mom_data_realisation_averaged_truncated[m][z,k,:].shape[0])
                # slope_flux_abs_error[z,k,m]=slope_momentum_vector_mean_abs_error[z,k,m]/(2*total_run_time*float(box_area_nd))
                #print(swap_timestep_vector[m].shape,mom_data_realisation_averaged_truncated[m][z,k,:].shape)
plt.show()




#NOTE this section needs to be finished.


#%%

 
# flux_from_fit=np.zeros((number_of_solutions,org_var_1.size,org_var_2.size))
# for z in range(0,number_of_solutions):
#     for i in range(0,org_var_1.size):
#         for j in range(0,org_var_2.size):
#           flux_from_fit[z,i,j]=func4(shear_rate_mean_of_both_cells[z,j,i],params[i][0],params[i][1])


# abs_error_in_flux= flux_from_fit-flux_ready_for_plotting

#msq_error_in_flux= 


#%% 
#save_string_for_plot= 'Flux_vs_shear_rate_'+fluid_name+'_phi_range_'+str(phi[0])+'_'+str(phi[1])+'_l_scale_'+str(lengthscale)+'_T_'+str(scaled_temp)+'.png'
labelpadx=15
labelpady=55
fontsize=20
count=1
org_var_1_index=org_var_1.size
plt.rcParams.update({'font.size': 15})

       
#plotting_flux_vs_shear_rate(func4,labelpadx,labelpady,params,fontsize,box_side_length_scaled,number_of_solutions,flux_ready_for_plotting,org_var_1_index,shear_rate_mean_of_both_cells)
def plotting_flux_vs_shear_rate(shear_rate_mean_error_of_both_cells,func4,labelpadx,labelpady,params,fontsize,box_side_length_scaled,number_of_solutions,flux_ready_for_plotting,org_var_1_index,shear_rate_mean_of_both_cells):
    
    for z in range(0,number_of_solutions):
        
        
        x=shear_rate_mean_of_both_cells[z,:,:]
        x_pos_error=np.abs(shear_rate_mean_error_of_both_cells[z,:,:])
        #y_pos_error=np.abs(abs_error_in_flux[z,:,:])
        y=flux_ready_for_plotting[z,:,:]
        
        for i in range(0,org_var_1_index):
        
        #for i in range(0,1):
            if z==0:
                j=i
                
                # need to add legend to this 
                plt.scatter(x[:,i],y[i,:],label='$L=${}'.format(np.around(box_side_length_scaled[0,z]),decimals=0),marker='x')
                plt.errorbar(x[:,i],y[i,:],xerr=x_pos_error[:,i],ls ='',capsize=3,color='r')
                plt.plot(x[:,i],func4(x[:,i],params[j][0],params[j][1]))
                #plt.fill_between(y[:,i], x_neg_error[i,:], x_pos_error[i,:])
                #plt.xscale('log')
                plt.xlabel('log($\dot{\gamma}\ [\\tau]$)', labelpad=labelpadx,fontsize=fontsize)
                #plt.yscale('log')
                plt.ylabel('log($J_{z}(p_{x})$$\ [\\frac{\\tau^{3}}{\mu}]$)',rotation=0,labelpad=labelpady,fontsize=fontsize)
                plt.legend()
                plt.show() 
                shear_viscosity=10** (params[i][1])
                grad_fit=(params[i][0])
                print('Dimensionless_shear_viscosity:',shear_viscosity)
                print('Grad of fit =',grad_fit)
            else: 
                j=z*(i+4)
                plt.scatter(x[:,i],y[i,:],label='$L=${}'.format(np.around(box_side_length_scaled[0,z]),decimals=0),marker='x')
                plt.errorbar(x[:,i],y[i,:],xerr=x_pos_error[:,i],ls ='',capsize=3,color='r')
                plt.plot(x[:,i],func4(x[:,i],params[j][0],params[j][1]))
                #plt.fill_between(y[:,i], x_neg_error[i,:], x_pos_error[i,:])
                #plt.xscale('log')
                plt.xlabel('log($\dot{\gamma}\ [\\tau]$)', labelpad=labelpadx,fontsize=fontsize)
                #plt.yscale('log')
                plt.ylabel('log($J_{z}(p_{x})$$\ [\\frac{\\tau^{3}}{\mu}]$)',rotation=0,labelpad=labelpady,fontsize=fontsize)
                plt.legend()
                plt.show() 
                plt.show() 
                shear_viscosity=10** (params[i][1])
                grad_fit=(params[i][0])
                print('Dimensionless_shear_viscosity:',shear_viscosity)
                print('Grad of fit =',grad_fit)
                 
               
    

plotting_flux_vs_shear_rate(shear_rate_mean_error_of_both_cells,func4,labelpadx,labelpady,params,fontsize,box_side_length_scaled,number_of_solutions,flux_ready_for_plotting,org_var_1_index,shear_rate_mean_of_both_cells)    
# need to adjust this so we get the visocsities of both plots 
# shear_viscosity=10** (params[0][1])
# grad_fit=(params[0][0])
# print('Dimensionless_shear_viscosity:',shear_viscosity)
# print('Grad of fit =',grad_fit)
# to get the error in viscosity need to look whether we take the mean of the relative or absolute errors. 

#dimensionful_shear_viscosity= shear_viscosity * mass_scale / lengthscale*timescale
#%% plotting qll 4 SS V_Ps

# need to fix legend location 
#plt.rcParams.update({'font.size': 25})
org_var_1_choice_index=org_var_1.size
fontsize=35
labelpadx=15
labelpady=35
width_plot=15
height_plot=10
legend_x_pos=1
legend_y_pos=1
org_var_2_index_start=0
org_var_1_index_end=10
# need to add all these settings for every plot
def plotting_SS_velocity_profiles(org_var_2_index_start,org_var_1_index_end,legend_x_pos, legend_y_pos,labelpadx,labelpady,fontsize,number_of_solutions,org_var_1_choice_index,width_plot,height_plot,org_var_1,org_var_2,VP_ave_freq,no_timesteps,VP_steady_state_data_lower_truncated_time_averaged,VP_steady_state_data_upper_truncated_time_averaged,VP_z_data_lower,VP_z_data_upper):
    for z in range(0,number_of_solutions):
    
        
        for k in range(0,org_var_1.size):  
            
            fig=plt.figure(figsize=(width_plot,height_plot))
            gs=GridSpec(nrows=1,ncols=1)

            ax1= fig.add_subplot(gs[0,0])
    
        


            for m in range(0,org_var_2.size):
                x_1=VP_steady_state_data_lower_truncated_time_averaged[z,k,m,:]
                #print(x_1.shape)
                x_2=VP_steady_state_data_upper_truncated_time_averaged[z,k,m,:]
                y_1=VP_z_data_lower[z,k,:]
                #print(y_1.shape)
                y_2=VP_z_data_upper[z,k,:]
                print(k)
                #for i in range(org_var_2_index_start,org_var_1_index_end):
                
                ax1.plot(y_1[:],x_1[:],label='$f_p=${}'.format(org_var_2[m]),marker='x')
                ax1.set_ylabel('$v_{x}\ [\\frac{\\tau}{\ell}]$',rotation=0,labelpad=labelpady, fontsize=fontsize)
                ax1.set_xlabel('$L_{z}\ [\ell^{-1}]$',rotation=0,labelpad=labelpadx,fontsize=fontsize)
                ax1.legend(frameon=False,loc=0,bbox_to_anchor=(legend_x_pos, legend_y_pos),fontsize=fontsize-4)       
            plt.show()
    
            
            

plotting_SS_velocity_profiles(org_var_2_index_start,org_var_1_index_end,legend_x_pos, legend_y_pos,labelpadx,labelpady,fontsize,number_of_solutions,org_var_1_choice_index,width_plot,height_plot,org_var_1,org_var_2,VP_ave_freq,no_timesteps,VP_steady_state_data_lower_truncated_time_averaged,VP_steady_state_data_upper_truncated_time_averaged,VP_z_data_lower,VP_z_data_upper)


# %% saving all the arrays which are needed for plots


# logged shear rate 
np.save(fluid_name+'_logged_shear_rate_mean_both_cells_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),shear_rate_mean_of_both_cells)

#  shear rate errors 
np.save(fluid_name+'_logged_shear_rate_mean_error_both_cells_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),shear_rate_mean_error_of_both_cells)

# fitting parameters and viscosity 

np.save(fluid_name+'_fitting_params_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),params)

# flux ready for plotting 

np.save(fluid_name+'_flux_ready_for_plotting_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),flux_ready_for_plotting)

# steady state velocity profiles 
# v_x data upper/ lower 
np.save(fluid_name+'_VP_steady_state_data_lower_truncated_time_averaged_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_steady_state_data_lower_truncated_time_averaged)
np.save(fluid_name+'_VP_steady_state_data_upper_truncated_time_averaged_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_steady_state_data_upper_truncated_time_averaged)

# z coordinate upper/lower
np.save(fluid_name+'_VP_z_data_lower_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_z_data_lower)
np.save(fluid_name+'_VP_z_data_upper_'+str(run_number)+'_phi_'+str(phi)+'_'+str(np.round(box_side_length_scaled[0,0]))+'_T_'+str(scaled_temp)+'_no_timesteps_'+str(no_timesteps),VP_z_data_upper)


# %% plotting dump file tuples
# 25mum
# nitrogen 
original_position_particle_1=np.array([1877.9594242952496, 1877.9594242952496, 2816.9391364428748])
original_position_particle_2=np.array([1877.9594242952496, 1877.9594242952496, 938.9797121476248])
# H20
original_position_particle_1=np.array([59.38629134151539, 59.38629134151539, 89.07943701227309])
original_position_particle_2=np.array([59.38629134151539, 59.38629134151539, 29.693145670757694])
# Ar
original_position_particle_1=np.array([5938.629134151539, 5938.629134151539, 8907.943701227308])
original_position_particle_2=np.array([5938.629134151539, 5938.629134151539, 2969.3145670757694])
# C6H14
#original_position_particle_1=np.array([187.795942429525, 187.795942429525, 281.6939136442875])
#original_position_particle_2=np.array([187.795942429525, 187.795942429525 ,93.8979712147625])

dump_file_p1_realisation_averaged = np.mean(dump_file_tuple_1[0],axis=1)
dump_file_p2_realisation_averaged = np.mean(dump_file_tuple_2[0],axis=1)
truncation_index=int(truncation_timestep/dump_freq)

z_deviation_from_centre_p1 = dump_file_p1_realisation_averaged[:,truncation_index:,4] -original_position_particle_1[2]
z_deviation_from_centre_p2 = dump_file_p2_realisation_averaged[:,truncation_index:,4] -original_position_particle_2[2]
z_mean_deviation_from_centre_scaled_box_side = (np.sqrt((z_deviation_from_centre_p1**2)+(z_deviation_from_centre_p2**2))*0.5)/ box_side_length_scaled
z_time_mean_deviation_from_centre_scaled_box_side=np.mean(z_mean_deviation_from_centre_scaled_box_side,axis=1)
z_std_dev_mean_deviation_from_centre_scaled_box_side = np.std(z_mean_deviation_from_centre_scaled_box_side,axis=1)
y_deviation_from_centre_p1 = dump_file_p1_realisation_averaged[:,truncation_index:,3] -original_position_particle_1[1]
y_deviation_from_centre_p2 = dump_file_p2_realisation_averaged[:,truncation_index:,3] -original_position_particle_2[1]
output_vector = np.arange(0,z_deviation_from_centre_p1.shape[1],1)
#/box_side_length_scaled

for i in range(0,spring_constant.size):
     
              plt.plot(output_vector[:],z_mean_deviation_from_centre_scaled_box_side[i,:],label='$K=${}'.format(spring_constant[i]))
              plt.xlabel('$N_{outputs}$')
              #plt.yscale('log')
              plt.ylabel('$z-z_{0}/L_{z}$',rotation=0)
              plt.legend(loc='best',bbox_to_anchor=(legendx, 1))
plt.show()

# need to add a fitting to this plot 
plt.scatter(spring_constant[:],z_time_mean_deviation_from_centre_scaled_box_side[:], marker='x')
plt.xscale('log')
plt.xlabel('$K[\\frac{\\tau}{\mu}]$')
plt.ylabel('$\Delta\\bar{Z}/L_{z}[-]$',rotation=0)
plt.yscale('log')
plt.show()

for i in range(0,spring_constant.size):
     
              plt.plot(output_vector[:],y_deviation_from_centre_p1[i,:])
              plt.xscale('log')
plt.show()
# %%
