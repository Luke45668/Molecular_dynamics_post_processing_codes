##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code will check the number of corrupted/aborted files in a data set
"""
#%%
import os 
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from log2numpy import *

path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
os.chdir(path_2_post_proc_module)
thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '

j_=20
damp=0.01
strain_total=400
K=2000
erate=np.flip(np.array([1,0.9,0.7,0.5,0.2,0.1,0.09,0.08,
                0.07,0.06,0.05,0.04,
                0.03,0.0275,0.025,0.0225,
                0.02,0.0175,0.015,0.0125,
                0.01,0.0075,0.005,0.0025,
                0.001,0.00075,0.0005]))



filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/damp_0.01"
os.chdir(filepath)
log_file_size_array=np.zeros((2,erate.size,j_))

log_name_list=glob.glob("log.*K_"+str(K)+"*")
count=np.zeros((erate.size)).astype("int")

failed_files=[]
passed_files=[]

# can scan all the files and produce a list of files that pass test
# check number of files in log file, this will be more clear than size
for file in log_name_list:

    split_name=file.split('_')
    erate_ind=int(np.where(erate==float(split_name[15]))[0][0])
    
    realisation_ind=int(split_name[6])
    spring_stiff=int(split_name[19])


    try:
        file_size_rows=log2numpy_reader(file,
                                filepath,
                                thermo_vars).shape[0]
        log_file_size_array[0,erate_ind,count[erate_ind]]=file_size_rows
        if count[erate_ind]==10:
            continue
        else:
            passed_files.append(file)
            count[erate_ind]+=1
        

    except:
        # if count[erate_ind]==10:
            failed_files.append(file)
            continue
        
              
        # log_file_size_array[0,erate_ind,count[erate_ind]]=0
        # count[erate_ind]+=1
        # continue 


# now i need n passed files for each shear rate




    
#    elif spring_stiff==2000:

#       log_file_size_array[1,erate_ind,realisation_ind]=file_size_mbytes
   

for i in range(0,erate.size):
   plt.plot(log_file_size_array[0,i,:], marker='x', linestyle='None')
   plt.yscale('log')
plt.show()

#%% making copy of file only with sucessful runs 
#os.mkdir("sucessful_runs")
# need to put in check if file exists test
for file in passed_files:
    unique_barcode=file.split('_')[5]
    # os.system("cp -r *_"+str(unique_barcode)+"_*h5 sucessful_runs/")
    os.system("cp -r log*_"+str(unique_barcode)+"_* sucessful_runs/")
    os.system("cp -r *_"+str(unique_barcode)+"_*dump sucessful_runs/")

#%%
os.chdir("sucessful_runs")

count=np.zeros((erate.size)).astype("int")
log_name_list=glob.glob("log.*K_"+str(K)+"*")
log_file_size_array=np.zeros((erate.size,20))
for file in log_name_list:

    split_name=file.split('_')
    erate_ind=int(np.where(erate==float(split_name[15]))[0][0])
    
    realisation_ind=int(split_name[6])
    spring_stiff=int(split_name[19])


    
    file_size_rows=log2numpy_reader(file,
                            filepath,
                            thermo_vars).shape[0]
    log_file_size_array[erate_ind,count[erate_ind]]=file_size_rows
    
    count[erate_ind]+=1

    # except:
        
    #     failed_files.append(file)
    #     log_file_size_array[0,erate_ind,count[erate_ind]]=0
    #     count[erate_ind]+=1
    #     continue 
#%%
for i in range(0,erate.size):
   plt.plot(log_file_size_array[i,:], marker='x', linestyle='None')
   plt.yscale('log')

plt.show()
print("max")

# %%
