# -*- coding: utf-8 -*-
"""
This file will read log.lammps files and output a numpy array containing each VP column. 
"""

import os
import mmap
import re 
#mport numba
#import time
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd


#keep code DRY(dont repeat yourself)

# =============================================================================
# simulation_file="Simulation_run_folder"
# chunk = 20 # number of chunks to use for VP averaging 
# equilibration_timesteps= 10000 # number of steps to do equilibration with 
# VP_ave_freq =10000
# no_SRD=str(33315)
# no_timesteps = str(200000)
# Path_2_VP="/Volumes/Backup Plus/PhD_/Rouse Model simulations/Using LAMMPS imac/"+simulation_file
# VP_output_col_count = 4 
# realisation_name=realisation_name="vel.profile_solid_inc_no_tstat__no_rescale_203267_1.0_33315_16.0_0.01_5_1000_10000_200000_T_5e-05_lbda_0.0025850369836944932_SR_1200_SN_1"
# #=============================================================================
#%%
def velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count):
    


    # Finding the logfile 
    
        #os.system('cd ~')
        #os.chdir(Path_2_VP)
           
        
        # Searching for VP output start and end line
        # use this version wehn averaging freqency is close eq steps 
        #VP_data_start_line= (str(VP_ave_freq+equilibration_timesteps)+" "+str(chunk))+" "#+no_SRD)# can we put a wild card in for no_SRD
        # use the versio  below when averaging frequency is much larger than eq steps 
        VP_data_start_line= (str(VP_ave_freq+equilibration_timesteps)+" "+str(chunk))+" "#+no_SRD)# can we put a wild card in for no_SRD
        VP_data_end_line =(str(int(equilibration_timesteps+float(no_timesteps)))+" "+str(chunk))#+" "+no_SRD) 
        #SRD_temp_lambda_start_line = "SRD temperature & lamda ="
        #SRD_temp_lambda_end_line = "SRD max distance & max velocity"# re.compile(b"Loop time of [+-]?[0-9]+\.[0-9]+ on . procs for "+bytes(total_no_timesteps,'utf-8')) 
        
        #Loop time of 316.462 on 6 procs for 300000 steps with 27653 atoms
        
        
        
        
        
        
        
        
           
        
        
        #warning_start = "\\nWARNING:"
        #warning_end = "(../fix_srd.cpp:2492)\\n"
        # convert string to bytes for faster searching 
        VP_data_start_line_bytes=bytes(VP_data_start_line,'utf-8')
        VP_data_end_line_bytes =bytes(VP_data_end_line,'utf-8')
        #SRD_temp_lambda_end_bytes = bytes(SRD_temp_lambda_end_line,'utf-8')
        #SRD_temp_lambda_start_bytes = bytes(SRD_temp_lambda_start_line,'utf-8')
        
        # find begining of data run 
        with open(realisation_name) as f:
             read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) 
             if read_data.find(VP_data_start_line_bytes) != -1:
               print('true')
               VP_byte_start=read_data.find(VP_data_start_line_bytes)
            
             else:
               print('could not find VP variables')
            
            
            # find end of run         
            
             if read_data.rfind(VP_data_end_line_bytes) != -1:
               print('true')
               VP_byte_end =read_data.rfind(VP_data_end_line_bytes)      #VP_data_end_line_bytes.search(read_data)#read_data.find(VP_data_end_line_bytes)
             else:
               print('could not find end of run')
            # correct 
             # if read_data.find(SRD_temp_lambda_start_bytes)!= -1:
             #    print('true')
             #    SRD_temp_lambda_start = read_data.find(SRD_temp_lambda_start_bytes)
             # else: 
             #    print('Could not find SRD temperature')
            
             # if read_data.find(SRD_temp_lambda_end_bytes)!= -1:
             #    print('true')
             #    SRD_temp_lambda_end = read_data.find(SRD_temp_lambda_end_bytes)
             # else:     
             #     print("Could not find SRD temperature ending")
        #%%
        
        
        
        
        
        
        # Splicing out the VP data and closing the file
        with open(realisation_name) as f:
         read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ ) 
        
           
         read_data.seek(VP_byte_start) #takes reader position to first line of interest 
        
         #size_of_data =VP_byte_end-VP_byte_start
         log_array_bytes=read_data.read() # reads whole file from line of interest
        # log_array_bytes_trim=log_array_bytes[0:size_of_data]
         #read_data.seek(SRD_temp_lambda_start)
         #SRD_temp_lambda_byte_array= read_data.read(SRD_temp_lambda_end)
         #size_of_temp_lambda_data = SRD_temp_lambda_end-SRD_temp_lambda_start
         #SRD_temp_lambda_byte_array_trim = SRD_temp_lambda_byte_array[0:size_of_temp_lambda_data]
         
         read_data.close()
           
        # comparison    
        #    log_string_before_warning_removal =  str(log_array_bytes_trim)
        # Convert byte array to string
        log_string= str(log_array_bytes)
        #SRD_temp_lambda_string = str(SRD_temp_lambda_byte_array_trim)
        
        #%%
        
        # this potentially works better than the original log file reader code 
        # Splitting the string into a string array 
        log_string_array_raw = log_string#.split()# removing the b' symbol 
        
        
        backslash_pattern = re.compile(r'\\n ')# removing all the \ns' 
        log_string_array_raw = re.sub(backslash_pattern," ",log_string_array_raw) 
        log_string_array_raw =log_string_array_raw[2:]  # gets rid of b'
        log_string_array_raw =log_string_array_raw[:-3] # gets rid of \n' file end marker 
        backslash_pattern = re.compile(r'\\n')# removing any \n's embedded in words 
        log_string_array_raw = re.sub(backslash_pattern," ",log_string_array_raw) 
        #%% 
        VP_unsorted_list= log_string_array_raw.split() 
        start_line_size =len(VP_data_start_line.split())
        VP_data_cols = VP_output_col_count
        VP_data_rows = chunk 
        #%%
        #from collections import deque
        number_of_VP_outputs = int((float(no_timesteps))/int(float(VP_ave_freq)))
        #VP_sorted_list = np.zeros(number_of_VP_outputs)
        #for i in range(0,number_of_VP_outputs):
        #VP_sorted_list= VP_unsorted_list[0:(VP_data_cols*VP_data_rows)]
        #VP_unsorted_list=VP_unsorted_list[(VP_data_cols*VP_data_rows):]        
        #VP_unsorted_list=deque(VP_unsorted_list)
        #%%
        #df_VP= pd.DataFrame()
        VP_list_size=VP_data_cols*VP_data_rows
        VP_final_array=np.zeros((VP_data_rows,number_of_VP_outputs)) #(np.array([[]])
        #%%
        count=0
        for i in range(0,number_of_VP_outputs):# need to chnage 50 to number of VP read outs variable 
        
            #timestep=VP_unsorted_list[0+((count*VP_list_size)+3)]
            if count==0:
             #  timestep=VP_unsorted_list[0+((count*VP_list_size))]
                VP_unsorted_ = VP_unsorted_list[3:]
                VP_numpy_array = np.array(VP_unsorted_[:VP_list_size])
                VP_numpy_array = VP_numpy_array.reshape(VP_data_rows,VP_data_cols)              
            
            else:
              #  timestep=VP_unsorted_list[0+(count*VP_list_size)+(3*count)]
                VP_unsorted_ = VP_unsorted_list[(3*(count+1))+(count*VP_list_size):]  
                VP_numpy_array = np.array(VP_unsorted_[:VP_list_size])
                VP_numpy_array = VP_numpy_array.reshape(VP_data_rows,VP_data_cols)                           
            #VP_unsorted_ = VP_unsorted_list[3+(count*VP_list_size):]
            #VP_numpy_array = np.array(VP_unso_rted_[:VP_list_size])
            
            #VP_numpy_array = VP_numpy_array.reshape(VP_data_rows,VP_data_cols)
            #df_VP[i] = pd.DataFrame({str(timestep): VP_numpy_array[:,3]})
            VP_final_array[:,i]= VP_numpy_array[:,3]
            VP_z_data = VP_numpy_array[:,1]
            count=count+1
        VP_z_data = VP_numpy_array[:,1]   
                
        print(VP_final_array)
        return VP_final_array,VP_z_data 
                
    
    #%%
    
    # for i in range(0,50):
    #     x=VP_numpy_array[:,1]
    #     y = df[i][:]
    #     plt.plot(y,x)
    #     plt.show()
    #VP_vars = "c_p0[4] Xz KinEng c_OMGA[1][1] c_OMGA[1][2] c_OMGA[1][3]"
    ##VP_vars = VP_vars.split()
    #VP_vars_length = len(VP_vars)+1
    #log_string_array = log_string_array[:-1]
    #log_string_array = log_string_array[VP_vars_length:]  # this needs to be a vraible calculated from VPvars
    ##3log_float_array = [0.00]*len(log_string_array)
    #log_string_array = [ i for i in log_string_array_raw if i!='\\\n' ]
    #t_0 =time.time()
    #%%
    # y=VP_numpy_array[:9,1]
    # x = df_VP[49][:9]
    # plt.xscale('linear')
    # plt.plot(y,x,linestyle='--', marker="x")
    # plt.show()
    
    
     
   

    

















