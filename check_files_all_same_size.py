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

j_=10
damp=0.035
strain_total=400
K=100

def folder_check_or_create(filepath,folder):
     os.chdir(filepath)
     # combine file name with wd path
     check_path=filepath+"/"+folder
     print((check_path))
     if os.path.exists(check_path) == 1:
          print("file exists, proceed")
          os.chdir(check_path)
     else:
          print("file does not exist, making new directory")
          os.chdir(filepath)
          os.mkdir(folder)
          os.chdir(filepath+"/"+folder)




# erate=np.flip(np.array([1,0.8,0.6,0.4,0.2,0.175,0.15,0.125,0.1,0.08,
#                 0.06,0.04,
#                 0.03,0.025,
#                 0.02,0.015,
#                 0.01,0.005,
#                 0.001,0.00075,0]))

# no_timesteps=np.flip(np.array([   394000,    493000,    657000,    986000,   1972000,   2253000,
#          2629000,   3155000,   3944000,   4929000,   6573000,   9859000,
#         13145000,  15774000,  19718000,  26290000,  39435000,  78870000,
#        394351000, 525801000,1000000]))

# code to remove failed erates








#K=100
erate=np.array([0, 0.00075, 0.001, 0.04, 0.06, 0.08,
        0.1, 0.125,0.15, 0.175, 0.2, 0.4,
        0.8, 1])

no_timesteps=np.array([  1000000, 525801000, 394351000,   9859000,   6573000,   4929000,
          3944000,   3155000,   2629000,   2253000,   1972000,    986000,
           493000,    394000])


# considering failed runs for K=50
# erate=np.flip(np.array([1,0.8,0.6,0.4,0.2,0.175,0.15,0.125,0.1,0.08,
#                 0.06,0.001,0.00075,0]))

# considering failed runs for K=100
# erate=np.flip(np.array([1,0.8,0.4,0.2,0.175,0.15,0.125,0.1,0.08,
#                 0.06,0.04,
#                 0.03,0.025,
#                 0.02,0.015,
#                 0.01,0.005,
#                 0.001,0.00075,0]))

# no_timesteps=np.flip(np.array([   394000,    493000,    657000,    986000,   1972000,   2253000,
#          2629000,   3155000,   3944000,   4929000,   6573000,   9859000,
#         13145000,  15774000,  19718000,  26290000,  39435000,  78870000,
#        394351000, 525801000,1000000]))



# erate=np.array([0])
# no_timesteps=np.array([1000000])

filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/damp_0.01"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/no_rattle/run_156147_no_rattle"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/run_63179_844598_495895/damp_0.035"
# filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle_pentagon/run_748702"
# filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/run_22190"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/run_335862"
#filepath="/Users/luke_dev/Documents/simulation_run_folder/eq_run_tri_plate_damp_0.035_K_500_4000"
filepath="/Users/luke_dev/Documents/simulation_run_folder/eq_run_tri_plate_damp_0.03633_K_500_4000"
filepath="/Users/luke_dev/Documents/simulation_run_folder/eq_run_tri_plate_damp_0.05_K_500_4000"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/run_692855"
os.chdir(filepath)
log_file_size_array=np.zeros((2,erate.size,j_))

log_name_list=glob.glob("log.*K_"+str(K))
count=np.zeros((erate.size)).astype("int")
count_failed=np.zeros((erate.size)).astype("int")
failed_files=[]
passed_files=[]
real_target=3
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
        #print(file_size_rows)
        log_file_size_array[0,erate_ind,count[erate_ind]]=file_size_rows
        if count[erate_ind]==real_target:
           
            continue

        elif file_size_rows<10000:
            continue
    
        else:
            passed_files.append(file)
            count[erate_ind]+=1
        
       
        

    except:
        # if count[erate_ind]==10:
            failed_files.append(file)
            count_failed[erate_ind]+=1

            continue
        
              
        # log_file_size_array[0,erate_ind,count[erate_ind]]=0
        # count[erate_ind]+=1
        # continue 


# now i need n passed files for each shear rate




    
#    elif spring_stiff==2000:

#       log_file_size_array[1,erate_ind,realisation_ind]=file_size_mbytes
   

for i in range(0,erate.size):
   plt.plot(log_file_size_array[0,i,:], marker='x', linestyle='None')
   #plt.yscale('log')
plt.show()

#%% making copy of file only with sucessful runs 

folder_check_or_create(filepath,"sucessful_runs_"+str(real_target)+"_reals")
os.chdir(filepath)
# need to put in check if file exists test
for file in passed_files:
    unique_barcode=file.split('_')[5]
    realisation_ind=file.split('_')[6]
    timestep=file.split('_')[12]
    # os.system("cp -r *_"+str(unique_barcode)+"_*h5 sucessful_runs/")
    os.system("cp -r log*_"+str(int(unique_barcode))+"_"+str(realisation_ind)+"_*"+str(timestep)+"*K_"+str(K)+" sucessful_runs_"+str(real_target)+"_reals/")
    #'log.langevinrun_no63179_hookean_flat_elastic_590951_9_100_0.035_0.005071624521210362_10000_10000_78870000_0.2_gdot_0.005_BK_500_K_2000'   
    os.system("cp -r *_"+str(int(unique_barcode))+"_"+str(realisation_ind)+"_*"+str(timestep)+"*K_"+str(K)+".dump sucessful_runs_"+str(real_target)+"_reals/")
    #langevinrun_no63179_hookean_flat_elastic_74035_2_100_0.035_0.005071624521210362_10000_10000_39435000_0.2_gdot_0.01_BK_500_K_2000.dump

#%%
os.chdir("sucessful_runs_"+str(real_target)+"_reals")


count=np.zeros((erate.size)).astype("int")
log_name_list=glob.glob("log.*K_"+str(K)+"*")
log_file_size_array=np.zeros((erate.size,real_target))

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
   plt.plot(log_file_size_array[i,:], marker='x', linestyle='None', label=erate[i])
   plt.yscale('log')
plt.legend()
plt.show()
print("file count:",np.sum(count))
print("expected file count",real_target*erate.size)

# %%
erate,no_timesteps=remove_failed_erates(erate,no_timesteps,count)
