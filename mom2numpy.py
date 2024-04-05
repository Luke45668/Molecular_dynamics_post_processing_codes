#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file will read momentum output files 
"""
#%%
import regex as re
import os
import numpy as np
import glob 
import mmap
#def mom2numpy_f(Path_2_mom_file)):

#%%

# realisation_name_ = []     
# for name in glob.glob('mom.'+fluid_name+'_no_tstat__no_rescale_*'):
   
#     count=count+1    
#     realisation_name_.append(name)
def mom2numpy_f(realisation_name,Path_2_mom_file):    
    os.chdir(Path_2_mom_file)
    Mom_data_start_line_bytes=bytes('# Fix print output for fix Px','utf-8')


# read in file data 
#for i in (0,count):
    with open(realisation_name) as f:
        read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) 
        if read_data.find(Mom_data_start_line_bytes) != -1:
               print('true')
               mom_byte_start=read_data.rfind(Mom_data_start_line_bytes)
            
        else:
               print('could not find data start')
               breakpoint()
        read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) 
        mom_data_bytes_not_trim=read_data.read()
        mom_str_trim= str(mom_data_bytes_not_trim[len(Mom_data_start_line_bytes):])
        backslash_pattern = re.compile(r'\\n')# removing all the \ns' 
        mom_data_ = re.sub(backslash_pattern," ", mom_str_trim) 
        remove_b_pattern= re.compile(r'b')
        mom_data_ = re.sub(remove_b_pattern,"", mom_data_ ) 
        #getting rid of cells with just a ' 
        mom_data__= mom_data_[1:] 
        mom_data___=mom_data__[:-1].split()
        
        # if len(mom_data___) != np.round(swap_rate*no_timesteps):
        #       breakpoint()
        # else:
        mom_data____=np.array(mom_data___)
        
       
    return    mom_data____
        

         

# %%
