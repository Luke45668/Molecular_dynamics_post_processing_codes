#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 17th July 2023

@author: lukedebono
This script will test the dump2numpy functionality
"""
#%%

import os
import mmap
import re 
#import time
import numpy as np

def dump2numpy_f(dump_start_line,Path_2_dump,simulation_file,filename,dump_realisation_name):
       
    

#%%


simulation_file="MYRIAD_LAMMPS_runs/T_1_spring_tests_solid_in_simple_shear/phi_0.005_tests"
dump_start_line='ITEM: ATOMS id type x y z vx vy vz omegax omegay omegaz'
filename=''
dump_realisation_name= 'test_run_dump_Ar_812518_1238_0.0_9146_11877.258268303078_0.2_679_10000_10000_8000000_T_1.0_lbda_31.80892704591236_SR_15_SN_1_K_40.0.dump'
Path_2_dump="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file
number_of_particles_per_dump =2

#dump_test= dump2numpy_f(dump_start_line,Path_2_dump,simulation_file,filename,dump_realisation_name)
dump2numpy_f(dump_start_line,Path_2_dump,dump_realisation_name)
# %%
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
            
        number_cols_dump = (len(dump_start_line.split()) -2) # minus one to remove item 
        number_rows_dump= count * number_of_particles_per_dump 

        dump_file=np.reshape(np.array(dump_one_timestep),(number_rows_dump,number_cols_dump))
        
        return dump_file




