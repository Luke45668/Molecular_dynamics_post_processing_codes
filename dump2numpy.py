#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:33:00 2022

@author: lukedebono
This script will read LAMMPS .dump files and store the data in a list of numpy arrays with dimensions
Number of dumps per run x number of particles x number of parameters
"""

import os
import mmap
import re 
#import time
import numpy as np

"""
Example inputs:
    

dump_start_line = "ITEM: ATOMS id type x y z vx vy vz mass"
This is found in the dump file the line above the first data output.

Path_2_dump="/Volumes/Backup Plus/PhD_/Rouse Model simulations/Using LAMMPS imac/"
The above Path_2_dump should be the path to the main directory where you store dump files.

simulation_file="fix_wall_simulations" 
filename="timestep_0.0001"

The above inputs allow further sorting of dump files. They can be left empty if necessary.

dump_realisation_name = "fix_induced_wall_test_no_min_eratexz_1_ts_00001_dump_every_10.dump"

The above is the name of the specific dumpfile you wish to read.


"""
simulation_file="Simulation_run_folder"
dump_start_line='ITEM: ATOMS id type x y z vx vy vz omegax omegay omegaz'
filename=''
dump_realisation_name= 'fixed_wall_t_27706_16.0_1e-05_34_1.0203153383802774_1000_100_300000_T_1.0_1.0_td_0.1.dump'
Path_2_dump="/Volumes/Backup Plus/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file


#t_0=time.time()  
def dump2numpy(dump_start_line,Path_2_dump,simulation_file,filename,dump_realisation_name):
       
    dump_vars = dump_start_line.split()
    dump_start_line_bytes = bytes(dump_start_line,'utf-8')
    
    
    os.system('cd ~')
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
    
    
    dump_file=[]
    for i in range(0,count): 
          dump_file.append(dump_one_timestep[i].split())
          dump_file[i]=np.array(dump_file[i],dtype=object)
          dim_dump_file = np.shape(dump_file[i])[0]
          col_dump_file= len(dump_vars)-2 #-2 to remove atoms and item 
          new_dim_dump_file = int(np.divide(dim_dump_file,col_dump_file))
          dump_file[i] = np.reshape(dump_file[i],(new_dim_dump_file,col_dump_file))

#for i in range(0,count): 
   # dump_file[i] = np.reshape(dump_file[i],(new_dim_dump_file,col_dump_file))
    
  
    return(dump_file)
#t=time.time()    
#Run_time = t-t_0 
#print(Run_time)


    
    
    


