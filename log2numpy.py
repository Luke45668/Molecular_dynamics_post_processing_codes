# -*- coding: utf-8 -*-
"""
This file will read log.lammps files and output a numpy array containing each thermo column. 
"""

import os
import mmap
import re 
import time
import numpy as np
import matplotlib.pyplot as plt

#keep code DRY(dont repeat yourself)

# =============================================================================

#realisation line below could be made to contain vars 
# simulation_file="Simulation_run_folder"
# no_timesteps = 20000
# no_eq_timesteps =1000
# total_no_timesteps = str(no_eq_timesteps+no_timesteps)
# Path_2_dump="/Volumes/Backup Plus/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file
# Path_2_log=Path_2_dump
# realisation_name = "log.no_wall_no_tstat_no_rescale_8194_1.0_7507_16.0_0.001_381_0.01_1000_1000_20000_T_5e-05_lbda_0.00934921866630143_SR_3_SN_1"
# # #
# thermo_vars ="         KinEng          Temp          TotEng    "
 
# =============================================================================
#%%
def log2numpy(Path_2_log,thermo_vars,realisation_name):
    


    # Finding the logfile 
    
        os.system('cd ~')
        os.chdir(Path_2_log)
           
        
        # Searching for thermo output start and end line
        
        Thermo_data_start_line= ("   Step" + thermo_vars)
        Thermo_data_end_line =("Loop time of") 
        SRD_temp_lambda_start_line = "SRD temperature & lamda ="
        SRD_temp_lambda_end_line = "SRD max distance & max velocity"# re.compile(b"Loop time of [+-]?[0-9]+\.[0-9]+ on . procs for "+bytes(total_no_timesteps,'utf-8')) 
        
        #Loop time of 316.462 on 6 procs for 300000 steps with 27653 atoms
        
        
        
        
        
        
        
        
           
        
        
        warning_start = "\\nWARNING:"
        warning_end = "(../fix_srd.cpp:2492)\\n"
        # convert string to bytes for faster searching 
        Thermo_data_start_line_bytes=bytes(Thermo_data_start_line,'utf-8')
        Thermo_data_end_line_bytes =bytes(Thermo_data_end_line,'utf-8')
        SRD_temp_lambda_end_bytes = bytes(SRD_temp_lambda_end_line,'utf-8')
        SRD_temp_lambda_start_bytes = bytes(SRD_temp_lambda_start_line,'utf-8')
        
        # find begining of data run 
        with open(realisation_name) as f:
         read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) 
         if read_data.find(Thermo_data_start_line_bytes) != -1:
            print('true')
            thermo_byte_start=read_data.find(Thermo_data_start_line_bytes)
            
         else:
            print('could not find thermo variables')
        
        
        # find end of run         
        
         if read_data.rfind(Thermo_data_end_line_bytes) != -1:
            print('true')
            thermo_byte_end =read_data.rfind(Thermo_data_end_line_bytes)      #Thermo_data_end_line_bytes.search(read_data)#read_data.find(Thermo_data_end_line_bytes)
         else:
            print('could not find end of run')
            # correct 
         if read_data.find(SRD_temp_lambda_start_bytes)!= -1:
            print('true')
            SRD_temp_lambda_start = read_data.find(SRD_temp_lambda_start_bytes)
         else: 
            print('Could not find SRD temperature')
        
         if read_data.find(SRD_temp_lambda_end_bytes)!= -1:
            print('true')
            SRD_temp_lambda_end = read_data.find(SRD_temp_lambda_end_bytes)
         else:     
             print("Could not find SRD temperature ending")
        #%%
        
        
        
        
        
        
        # Splicing out the thermo data and closing the file
        with open(realisation_name) as f:
         read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ ) 
        
           
         read_data.seek(thermo_byte_start) 
        
         size_of_data =thermo_byte_end-thermo_byte_start
         log_array_bytes=read_data.read(thermo_byte_end)
         log_array_bytes_trim=log_array_bytes[0:size_of_data]
         read_data.seek(SRD_temp_lambda_start)
         SRD_temp_lambda_byte_array= read_data.read(SRD_temp_lambda_end)
         size_of_temp_lambda_data = SRD_temp_lambda_end-SRD_temp_lambda_start
         SRD_temp_lambda_byte_array_trim = SRD_temp_lambda_byte_array[0:size_of_temp_lambda_data]
         
         read_data.close()
           
        # comparison    
        #    log_string_before_warning_removal =  str(log_array_bytes_trim)
        # Convert byte array to string
        log_string= str(log_array_bytes_trim)
        SRD_temp_lambda_string = str(SRD_temp_lambda_byte_array_trim)
        
        
        
        #Count SRD warnings   
        warn_count  = log_string.count(warning_start) 
        #if warn_count == 0:
        # could skip to string split line from here
        # correct 
        #%%  
        # Removing non-data parts 
         # need to re structure the loop to find the first warning idicies then delete it, then find the neext etc etc 
         
        i_0 = list(range(0,warn_count+1))
           # size_i_0=len(i_0)
        warning_start = "WARNING:"
        warning_end = "(../fix_srd.cpp:2492)"# as long as the new number in place of 2492 is the same number of digits the exact didigits dont matter in this case
           # warn_offset=len(warning_start)
        warn2_offset= len(warning_end)
           # length_final_warning = len("(../thermo.cpp:422)")
         
        pattern_warn = re.compile(r"WARNING: Fix") # separates pattern into a variable, allows us to reuse for multiple searches
        pattern_warn_end =re.compile(r"(../fix_srd.cpp:....)")
        pattern_final_warn_start = re.compile(r"WARNING: Too")
        pattern_final_warn =re.compile(r"(../thermo.cpp:...)") # 418 may need to change depending on the final warning error 
        
        # correct 
        
        #%% Removing standard warning 
        warning_start_search= pattern_warn.search(log_string)
        warning_end_search= pattern_warn_end.search(log_string)   
        
        warning_start_index=0
        warning_end_index =0
        #%%
        
        for i in i_0:
        
          if warning_start_search == None:
           if warning_end_search== None:
            warning_start_index=None
            warning_end_index=None
            break
          
          else:
           warning_start_index=warning_start_search.span(0)[0]
           warning_end_index = warning_end_search.span(0)[0]+ warn2_offset
           log_string=log_string[:warning_start_index]+log_string[warning_end_index:]     
          
         
           
         
         
          
          
        
         
         #correct
         #%%
        
        
        #remove final warning 
        final_warning_start= pattern_final_warn_start.search(log_string)
        
        final_warning_end= pattern_final_warn.search(log_string)
        print(final_warning_start)
        print(final_warning_end)
        final_warning_start_index= 0  # FIX THIS LINE TO REMOVE WHOLE FINALWARN 
        final_warning_end_index = 0 
        #%%
        
        # =============================================================================
        # this whole section needs work
        #for i in i_0:
        for i in i_0:
        
         if final_warning_start==None:
            if final_warning_end==None:
             
             break
         else:
           final_warning_start_index= final_warning_start.span(0)[0]-2 # the plus -2 and plus 2 indicies remove the linebreaks eithers side of the statement
           final_warning_end_index = final_warning_end.span(0)[1]+4 
           
           break 
         #%%     
        log_string=log_string[:final_warning_start_index]+log_string[final_warning_end_index:]      
        
        #correct 
         #%%
        # =============================================================================
        # use regex again for final check 
        check_warn = pattern_warn.search(log_string)
        check_warn_end = pattern_warn_end.search(log_string)       
        check_warn_end_final = pattern_final_warn.search(log_string)
        
        #%%
        for i in i_0:
        
         if check_warn ==None:
          if check_warn_end ==None:
           if check_warn_end_final==None:
             
             print("All Warning messages removed, proceed.")
           else:
             print("Data still contains warning messages.")
        
        #%%Using a regex to remove all the \n's
        #log_string = re.sub(r'[\n]','',log_string,re.M)
        
            
        # Converting to array and removing '\n'
        # getting rid of the spacing '\n'
        #%%
        
        log_string_array_raw = log_string
        # Splitting the string into a string array 
        #log_string_array_raw = log_string.split()
        #log_string_array = [ i for i in log_string_array_raw if i!=r'\n' ] 
        
         # Potentially not the fastest way as we loop through the whole list
        backslash_pattern = re.compile(r'\\n ')# removing all the \ns' 
        log_string_array = re.sub(backslash_pattern," ",log_string_array_raw) 
        
        #log_string_array = [ i for i in log_string_array_raw if i!='\\n' ]
        #%%
        #thermo_vars = "KinEng          Temp          TotEng             "
        step = "b'   Step"
        thermo_vars = step+thermo_vars#.split()
        thermo_vars_length = len(thermo_vars)
        log_string_array =log_string_array[thermo_vars_length:]
        log_string_array = log_string_array[:-3]
        
        #%%
        log_string_array= log_string_array.split()
        #log_string_array = log_string_array[thermo_vars_length:]  # this needs to be a vraible calculated from thermovars
        log_float_array = [0.00]*len(log_string_array)
        #log_string_array = [ i for i in log_string_array_raw if i!='\\\n' ]
        #t_0 =time.time()
        
         #%% 
        for i in range(len(log_string_array)):
         log_float_array[i] = float(log_string_array[i])
        # print(log_string_array.pop())
        #t=time.time()
        #print(t-t_0)
        #%%
        # Creating numpy array of the data
        
        log_file = np.array(log_float_array, dtype=np.float64)
        dim_log_file =np.shape(log_file)[0]
        col_log_file = len(Thermo_data_start_line.split())
        new_dim_log_file = int(np.divide(dim_log_file,col_log_file))
        
        #%%
        log_file = np.reshape(log_file,(new_dim_log_file,col_log_file))

        return log_file,SRD_temp_lambda_string

















