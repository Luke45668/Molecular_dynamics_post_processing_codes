##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file processes the all files from brownian dynamics simulations of many flat elastic particles.


after an MPCD simulation. 
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
# plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import mmap
import h5py as h5
import math as m 
import glob 

path_2_post_proc_module= '/home/ucahlrl/python_scripts/MPCD_post_processing_codes'
os.chdir(path_2_post_proc_module)

from log2numpy import *
from dump2numpy import *
import glob 
from MPCD_codes.post_MPCD_MP_processing_module import *
import pickle as pck
from numpy.linalg import norm
import pickle as pck

#%%
damp=0.035
strain_total=100


# erate=np.array([0])
# no_timesteps=np.array([10000000])

erate=np.flip(np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.01,0.005,0]))

no_timesteps=np.flip(np.array([ 3944000,  4382000,  4929000,  5634000,  6573000,  7887000,
         9859000, 13145000, 19718000,  2253000,  2629000,  3155000,
         3944000,  4929000,  6573000,  9859000, 39435000,
        78870000, 10000000]))

timestep_multiplier=np.flip(np.array(
[0.0005,0.0005,0.0005,0.0005,
0.0005,0.0005,0.0005,0.0005,0.0005,
0.0005,0.0005,0.0005,0.0005,0.0005,0.0005,
0.0005,0.0005,0.005,0.005,0.2]))


thermo_vars='         KinEng         PotEng         Press           Temp      c_uniaxnvttemp'
#thermo_vars='         KinEng         PotEng         Press         c_myTemp        TotEng    '
thermo_vars='         KinEng         PotEng         Press           Temp         Ecouple       Econserve   '



K=30
j_=5
box_size=100
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
n_plates=100
barcode=765583

filepath=filepath="/home/ucahlrl/Scratch/output/nvt_runs/run_"+str(barcode)+"/sucessful_runs_5_reals"
#filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/run_765583_damp_75dt/sucessful_runs_5_reals"
path_2_log_files=filepath
pol_general_name_string='*K_'+str(K)+'*pol*h5'

phantom_general_name_string='*K_'+str(K)+'*phantom*h5'

Mom_general_name_string='mom.*'

log_general_name_string='log.langevin*K_'+str(K)

dump_general_name_string='*K_'+str(K)+'.dump'




(realisation_name_Mom,
 realisation_name_phantom,
 count_mom,count_phantom,
 realisation_name_log,
 count_log,
 realisation_name_dump,
 count_dump,
 realisation_name_pol,
 count_pol)= VP_and_momentum_data_realisation_name_grabber(pol_general_name_string,
                                                                     log_general_name_string,
                                                                     phantom_general_name_string,
                                                                     Mom_general_name_string,
                                                                     path_2_log_files,
                                                                     dump_general_name_string)

class realisation():
     def __init__(self,realisation_full_str,data_set,realisation_index_):
          self.realisation_full_str= realisation_full_str
          self.data_set= data_set
          self.realisation_index_=realisation_index_
     def __repr__(self):
        return '({},{},{})'.format(self.realisation_full_str,self.data_set,self.realisation_index_)
realisations_for_sorting_after_srd=[]
realisation_split_index=6
erate_index=15

def org_names(split_list_for_sorting,unsorted_list,first_sort_index,second_sort_index):
    for i in unsorted_list:
          realisation_index_=int(i.split('_')[first_sort_index])
          data_set =i.split('_')[second_sort_index]
          split_list_for_sorting.append(realisation(i,data_set,realisation_index_))


    realisation_name_sorted=sorted(split_list_for_sorting,
                                                key=lambda x: ( x.data_set,x.realisation_index_))
    realisation_name_sorted_final=[]
    for i in realisation_name_sorted:
          realisation_name_sorted_final.append(i.realisation_full_str)
    
    return realisation_name_sorted_final


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

def linearfunc(x,a,b):
    return (a*x)+b 
def linearthru0(x,a):
     return a*x 

def powerlaw(x,a,n):
    return a*(x**(n))

def quadfunc(x,a):
     return a*x**(2)

def logfunc(x,a,b,c):
     return a*np.log(x-b) +c

def block_averaging(oned_arr_in,number_blocks):
    block_means=np.zeros((number_blocks))
    size_block=int(oned_arr_in.shape[0]/number_blocks)

    block_count=0
    for i in range(number_blocks):
          block_means[i]=np.mean(oned_arr_in[block_count*size_block:
            (block_count+1)*size_block] )
          block_count+=1

    return block_means


realisations_for_sorting_after_pol=[]
realisation_name_h5_after_sorted_final_pol=org_names(realisations_for_sorting_after_pol,
                                                     realisation_name_pol,
                                                     realisation_split_index,
                                                     erate_index)

realisations_for_sorting_after_phantom=[]
realisation_name_h5_after_sorted_final_phantom=org_names(realisations_for_sorting_after_phantom,
                                                     realisation_name_phantom,
                                                     realisation_split_index,
                                                     erate_index)
realisations_for_sorting_after_log=[]
realisation_name_log_sorted_final=org_names(realisations_for_sorting_after_log,
                                                     realisation_name_log,
                                                     realisation_split_index,
                                                     erate_index)

realisations_for_sorting_after_dump=[]
realisation_name_dump_sorted_final=org_names(realisations_for_sorting_after_dump,
                                                     realisation_name_dump,
                                                     realisation_split_index,
                                                     erate_index)

print(len(realisation_name_log_sorted_final))
print(len(realisation_name_dump_sorted_final))




#NOTE: make sure the lists are all correct with precisely the right number of repeats or this code below
# will not work properly. 

dump_start_line ='ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]'
Path_2_dump=filepath


md_step=0.005071624521210362*timestep_multiplier


box_size_div=2
strain_total=np.repeat(strain_total,erate.size)
log_file_tuple=()
p_velocities_tuple=()
p_positions_tuple=()
area_vector_tuple=()
spring_force_positon_tensor_tuple=()
new_pos_vel_tuple=()
interest_vectors_tuple=()
tilt_test=[]
e_in=0
e_end=13# for extension runs 
count=e_in
from collections import Counter
# need to ake this geenral 
def dump2numpy_tensor_1tstep(dump_start_line,
                      Path_2_dump,dump_realisation_name,
                      number_of_particles_per_dump,lines_per_dump, cols_per_dump):
       
       
        

        os.chdir(Path_2_dump) #+simulation_file+"/" +filename

        

        with open(dump_realisation_name, 'r') as file:
            

            lines = file.readlines()
            
            counter = Counter(lines)
            
            #print(counter.most_common(3))
            n_outs=int(counter["ITEM: TIMESTEP\n"])
            dump_outarray=np.zeros((n_outs,lines_per_dump,cols_per_dump))
            #print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])
            skip_spacing=lines_per_dump+9
            skip_array=np.arange(1,len(lines),skip_spacing)
            for i in range(n_outs):
                k=skip_array[i]
                # timestep_list=[]
                start=k-1
                end=start+skip_spacing
                timestep_list=lines[start:end]
                data_list=timestep_list[9:]
                #print(data_list[0])
                #print(len(data_list))
                data=np.zeros((lines_per_dump,cols_per_dump))
                for j in range(len(data_list)):
                    data[j,:]=data_list[j].split(" ")[0:cols_per_dump]
            
                dump_outarray[i,:,:]=data


        return dump_outarray




            

# need to write dump to numpy to only look at chunks to save on ram 
for i in range(e_in,e_end):
    i_=(count*j_)
    print("i_",i_)


    
    outputdim_dump=dump2numpy_tensor_1tstep(dump_start_line,
                                    Path_2_dump,
                                     realisation_name_dump_sorted_final[i_],
                                     n_plates,300,6).shape[0]
    
    outputdim_log=log2numpy_reader(realisation_name_log_sorted_final[i_],
                                                    path_2_log_files,
                                                    thermo_vars).shape[0]

    
    log_file_array=np.zeros((j_,outputdim_log,6)) #nemd
   
    
    spring_force_positon_array=np.zeros((j_,outputdim_dump,300,6))
    area_vector_array=np.zeros((j_,outputdim_dump,100,3))



    for j in range(j_):
            j_index=j+(j_*count)



            log_file_array[j,:,:]=log2numpy_reader(realisation_name_log_sorted_final[j_index],
                                                        path_2_log_files,
                                                        thermo_vars)
            dump_data=dump2numpy_tensor_1tstep(dump_start_line,
                                    Path_2_dump,
                                    realisation_name_dump_sorted_final[j_index],
                                    n_plates,300,6)
            print(dump_data.shape)

            # bond local reverses the sign of direction, lower id - higher id 

            spring_force_positon_array[j,:,:,0]=-dump_data[:,:,0]*dump_data[:,:,3]#xx
            spring_force_positon_array[j,:,:,1]=-dump_data[:,:,1]*dump_data[:,:,4]#yy
            spring_force_positon_array[j,:,:,2]=-dump_data[:,:,2]*dump_data[:,:,5]#zz
            spring_force_positon_array[j,:,:,3]=-dump_data[:,:,0]*dump_data[:,:,5]#xz
            spring_force_positon_array[j,:,:,4]=-dump_data[:,:,0]*dump_data[:,:,4]#xy
            spring_force_positon_array[j,:,:,5]=-dump_data[:,:,1]*dump_data[:,:,5]#yz

            #area_vector_array[j]=np.cross(-dump_data[:,:,0],-dump_data[:,:,1])

            # can we get it to dump a hdf5 file for the tensor, this would be in order






            # need to compute the ell vectors from this. 

          

    lgf_mean=np.mean(log_file_array,axis=0)    
    spring_force_positon_mean=np.mean(np.mean(spring_force_positon_array,axis=0),
                                      axis=1)
    area_vector_tuple=area_vector_tuple+(area_vector_array,)
    
  
   
    log_file_tuple=log_file_tuple+(lgf_mean,)
    spring_force_positon_tensor_tuple=spring_force_positon_tensor_tuple+\
        (spring_force_positon_array,)
    count+=1



#%% save tuples to avoid needing the next stage 
#make sure to comment this out after use
label='damp_'+str(damp)+'_K_'+str(K)+'_'
import pickle as pck

folder_check_or_create(filepath,"saved_tuples")

with open(label+'spring_force_positon_tensor_tuple.pickle', 'wb') as f:
    pck.dump(spring_force_positon_tensor_tuple, f)

with open(label+'log_file_tuple.pickle', 'wb') as f:
    pck.dump(log_file_tuple, f)

# with open(label+"new_pos_vel_tuple.pickle",'wb') as f:
#      pck.dump(new_pos_vel_tuple,f)

# with open(label+'p_velocities_tuple.pickle', 'wb') as f:
#     pck.dump( p_velocities_tuple, f)

# with open(label+'p_positions_tuple.pickle', 'wb') as f:
#     pck.dump( p_positions_tuple, f)

# with open(label+"area_vector_tuple.pickle", 'wb') as f:
#     pck.dump(area_vector_tuple,f)

# with open(label+"interest_vectors_tuple.pickle",'wb') as f:
#     pck.dump(interest_vectors_tuple,f)


