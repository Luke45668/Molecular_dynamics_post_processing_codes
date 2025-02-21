##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file processes the all files from brownian dynamics simulations of many flat elastic particles.
"""
#%% Importing packages
import os
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import regex as re
import pandas as pd
import sigfig
plt.rcParams.update(plt.rcParamsDefault)
#plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats

from datetime import datetime
import mmap
import h5py as h5
from scipy.optimize import curve_fit


# path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
# os.chdir(path_2_post_proc_module)
import seaborn as sns
# from log2numpy import *
# from dump2numpy import *
import glob 
from post_MPCD_MP_processing_module import *
import pickle as pck
from post_langevin_module import *
from reading_lammps_module import *


damp=0.035
strain_total=250

erate=np.linspace(0,1.7,48)

no_timesteps=np.array([1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000 ])

erate=np.array([0.        , 0.00388889, 0.00777778, 0.01166667, 0.01555556,
       0.01944444, 0.02333333, 0.02722222, 0.03111111, 0.035  ,0.07      , 0.13894737, 0.20789474, 0.27684211, 0.34578947,
        0.41473684, 0.48368421, 0.55263158, 0.62157895, 0.69052632,
        0.75947368, 0.82842105, 0.89736842, 0.96631579, 1.03526316,
        1.10421053, 1.17315789, 1.24210526, 1.31105263, 1.38,1.4  , 1.42222222, 1.44444444, 1.46666667, 1.48888889,
        1.51111111, 1.53333333, 1.55555556, 1.57777778, 1.6 ])

thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
K=120
j_=3
box_size=100
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5
n_plates=100


filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/shear_runs/strain_250_3_reals_15_45_60_tchain"
path_2_log_files=filepath
pol_general_name_string='*K_'+str(K)+'*pol*h5'

phantom_general_name_string='*K_'+str(K)+'*phantom*h5'

Mom_general_name_string='mom.*'

log_general_name_string='log.langevin*K_'+str(K)

dump_general_name_string="*K_"+str(K)+".dump"




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

#%%
#NOTE: make sure the lists are all correct with precisely the right number of repeats or this code below
# will not work properly. 
dump_start_line = "ITEM: ATOMS id type xu yu zu vx vy vz"
Path_2_dump=filepath
md_step=0.00101432490424207
box_size_div=2
strain_total=np.repeat(400,erate.size)
log_file_tuple=()
p_velocities_tuple=()
p_positions_tuple=()
area_vector_tuple=()
spring_force_positon_tensor_tuple=()
new_pos_vel_tuple=()
interest_vectors_tuple=()
tilt_test=[]
e_in=0
e_end=37
count=e_in

# need to write dump to numpy to only look at chunks to save on ram 
for i in range(e_in,e_end):
    i_=(count*j_)
    print("i_",i_)
    
    outputdim_dump=int(dump2numpy_f(dump_start_line,
                                    Path_2_dump,
                                    realisation_name_dump_sorted_final[i_],
                                    n_plates*3).shape[0]/(n_plates*3))
   
    dump_freq=int(realisation_name_dump_sorted_final[i_].split('_')[10])
    print(dump_freq)
    
   
    p_velocities_array=np.zeros((j_,outputdim_dump,n_plates,3,3))
    p_positions_array=np.zeros((j_,outputdim_dump,n_plates,3,3))
    
    area_vector_array=np.zeros((j_,outputdim_dump,n_plates,3))
    
 
    interest_vectors_array=np.zeros((j_,outputdim_dump,n_plates,2,3))
    for j in range(j_):
            j_index=j+(j_*count)

            # need to get rid of print statements in log2numpy 
            # print(realisation_name_log_sorted_final[j_index])
            print(j_index)
           
            dump_data=dump2numpy_f(dump_start_line,
                                    Path_2_dump,
                                    realisation_name_dump_sorted_final[j_index],
                                    n_plates*3)
            dump_data=np.reshape(dump_data,(outputdim_dump,n_plates,3,8)).astype('float')
            new_pos_vel_array=np.zeros((outputdim_dump,n_plates,6,3))
            
            p_positions_array[j,:,:,:]= dump_data[:,:,:,2:5]
            
            p_velocities_array[j,:,:,:]= dump_data[:,:,:,5:8]

            ell_1=p_positions_array[j,:,:,1,:]-p_positions_array[j,:,:,0,:]
            ell_2=p_positions_array[j,:,:,2,:]-p_positions_array[j,:,:,0,:]
            
            area_vector_array[j,:,:,:]=np.cross(ell_1,
                                      ell_2
                                        ,axisa=2,axisb=2)

          
                     
    
    area_vector_tuple=area_vector_tuple+(area_vector_array,)

    p_velocities_tuple=p_velocities_tuple+(p_velocities_array,)
    p_positions_tuple=p_positions_tuple+(p_positions_array,)
 
    count+=1



# save tuples to avoid needing the next stage 
#make sure to comment this out after use
label='damp_'+str(damp)+'_K_'+str(K)+'_'


folder_check_or_create(filepath,"saved_tuples")


with open(label+'p_velocities_tuple.pickle', 'wb') as f:
    pck.dump( p_velocities_tuple, f)

with open(label+'p_positions_tuple.pickle', 'wb') as f:
    pck.dump( p_positions_tuple, f)

with open(label+"area_vector_tuple.pickle", 'wb') as f:
    pck.dump(area_vector_tuple,f)





# %%
