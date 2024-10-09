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
# plt.rcParams["figure.figsize"] = (8,6 )
# plt.rcParams.update({'font.size': 16})
#plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import mmap
import h5py as h5
import math as m 
import glob 
from reading_lammps_module import *
from fitter import Fitter, get_common_distributions, get_distributions
path_2_post_proc_module= '/Users/luke_dev/Documents/molecular_dynamics_post_processing_codes/MPCD_codes/'
os.chdir(path_2_post_proc_module)
import seaborn as sns
#sns.set_palette('colorblind')
# from log2numpy import *
# from dump2numpy import *
import glob 
from post_MPCD_MP_processing_module import *

import pickle as pck
from numpy.linalg import norm

#%%
damp=0.035
strain_total=200


# erate=np.array([0])
# no_timesteps=np.array([10000000])

erate=np.flip(np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.01,0.005,0]))

no_timesteps=np.flip(np.array([ 3944000,  4382000,  4929000,  5634000,  6573000,  7887000,
         9859000, 13145000, 19718000,  2253000,  2629000,  3155000,
         3944000,  4929000,  6573000,  9859000, 39435000,
        78870000, 10000000]))

no_timesteps=np.flip(np.array(
[ 157740000,  175267000,  197175000,  225343000,  262901000,
         315481000,  394351000,  525801000,  788702000,  901374000,
        1051603000, 1261923000,  157740000,  197175000,  262901000,
         394351000,  788702000,  157740000,  315481000,   10000000]))



timestep_multiplier=np.flip(np.array(
[0.0005,0.0005,0.0005,0.0005,
0.0005,0.0005,0.0005,0.0005,0.0005,
0.0005,0.0005,0.0005,0.0005,0.0005,0.0005,
0.0005,0.0005,0.005,0.005,0.2]))

timestep_multiplier=np.flip(np.array(
[0.00005,0.00005,0.00005,0.00005,
0.00005,0.00005,0.00005,0.00005,0.00005,
0.00005,0.00005,0.00005,0.0005,0.0005,0.0005,
0.0005,0.0005,0.005,0.005,0.2]))

erate=np.flip(np.array([1.   , 0.9  , 0.8  , 0.7  , 0.6  , 0.5  , 0.4  , 0.3  , 0.2  ,
       0.175, 0.15 , 0.125, 0.1  , 0.08 , 0.06 , 0.04  ,0.02 ,0.01 ,
       0.005, 0   ]))

erate=np.flip(np.array([0.5,0.45,0.4,0.35,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.02,0.01,0.005]))

no_timesteps=np.flip(np.array([ 473221000,  525801000,  591526000,  676030000,  788702000,
        1183053000, 1352060000, 1577404000, 1892885000,  236611000,
         295763000,  394351000,  591526000, 1183053000,  236611000,
         473221000]))


erate=np.flip(np.array([0.5,0.45,0.4,0.375,0.35,0.325,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.02,0.01,0.005]))

no_timesteps=np.flip(np.array([ 492939000,  547710000,  616173000,  657252000,  704198000,
         758367000,  821565000, 1232347000, 1408396000, 1643129000,
        1971755000,  246469000,  308087000,  410782000,  616173000,
        1232347000,  246469000,  492939000]))

erate=np.flip(np.array([0.55,0.5,0.45,0.4,0.375,0.3675,0.35,0.3375 ,0.325,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.02,0.01,0.005]))

erate=np.flip(np.array([0.5,0.45,0.4,0.375,0.3725,0.37,0.365,0.36,0.355,0.35,0.3375 ,0.325,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.02,0.01,0.005]))

no_timesteps=np.flip(np.array([ 197175000, 219084000, 246469000, 262901000, 264665000, 266453000,
        270103000, 273855000, 277712000, 281679000, 292112000, 303347000,
        328626000, 492939000, 563359000, 657252000, 788702000,  98588000,
        123235000, 164313000, 246469000, 492939000,  98588000, 197175000]))

timestep_multiplier=np.array([
[0.00005,0.00005,0.00005,0.00005,
0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
0.0005,0.0005,0.0005,0.0005,0.0005,0.005,
0.005],

[0.00005,0.00005,0.00005,0.00005,
0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
0.0005,0.0005,0.0005,0.0005,0.0005,0.005,
0.005]])*4

thermo_vars='         KinEng         PotEng         Press           Temp      c_uniaxnvttemp'
#thermo_vars='         KinEng         PotEng         Press         c_myTemp        TotEng    '
thermo_vars='         KinEng         PotEng         Press           Temp         Ecouple       Econserve   '



K=50
j_=10
box_size=100
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
n_plates=100

filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/extension_runs/run_94057/sucessful_runs_5_reals"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/run_759848/sucessful_runs_5_reals"
#filepath='/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/run_224048/sucessful_runs_5_reals'
#filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/run_64228/sucessful_runs_5_reals"
thermo_vars='         KinEng         PotEng         Press           Temp         Ecouple       Econserve    c_uniaxnvttemp'
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/sucessful_runs_5_reals"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/run_956647/sucessful_runs_5_reals"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/run_226020"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/non_zero_natural_length_run/run_55599/sucessful_runs_5_reals"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/non_zero_natural_length_run/run_727608/sucessful_runs_5_reals"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/non_zero_natural_length_run/run_452444/sucessful_runs_10_reals"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/non_zero_natural_length_run/run_709773/sucessful_runs_10_reals"
#filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/non_zero_natural_length_run/run_781839"


path_2_log_files=filepath

pol_general_name_string='*K_'+str(K)+'*pol*h5'

phantom_general_name_string='*K_'+str(K)+'*phantom*h5'

dump_vel_name_string='*_dump_**K_'+str(K)+'.dump'

log_general_name_string='log.*K_'+str(K)
thermal_damp_array=np.array([5,12.5,25,50,75,100,125,150,200,400])
thermal_damp_array=np.array([750,1000,1500,2000])
thermal_damp=thermal_damp_array[3]
#log_general_name_string=("log.*_100_"+str(thermal_damp)+"_*K_"+str(K))

#dump_general_name_string='*_tensor_*_100_'+str(thermal_damp)+'_*K_'+str(K)+'.dump'

dump_general_name_string='*_tensor_**K_'+str(K)+'.dump'




(realisation_name_dump_vel,
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
                                                                     dump_vel_name_string,
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

#%%
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
realisations_for_sorting_after_dump_vel=[]
realisation_name_dump_vel_sorted_final=org_names(realisations_for_sorting_after_dump_vel,
                                                     realisation_name_dump_vel,
                                                     realisation_split_index,
                                                     erate_index)

print(len(realisation_name_log_sorted_final))
print(len(realisation_name_dump_sorted_final))


# %%

#NOTE: make sure the lists are all correct with precisely the right number of repeats or this code below
# will not work properly. 

dump_start_line ='ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]'
dump_start_line_posvel = "ITEM: ATOMS id type x y z vx vy vz"
Path_2_dump=filepath


md_step=0.005071624521210362*timestep_multiplier


box_size_div=2
strain_total=np.repeat(strain_total,erate.size)
log_file_tuple=()
velocities_tuple=()
p_positions_tuple=()
area_vector_tuple=()
spring_force_positon_tensor_tuple=()
new_pos_vel_tuple=()
interest_vectors_tuple=()
spring_extension_tuple=()
dir_vector_tuple=()
tilt_test=[]
e_in=0
e_end=24# for extension runs 
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




            
    
    


#%%
# need to write dump to numpy to only look at chunks to save on ram 
for i in range(e_in,e_end):
    i_=(count*j_)
    print("i_",i_)


    
    outputdim_dump=dump2numpy_tensor_1tstep(dump_start_line,
                                    Path_2_dump,
                                     realisation_name_dump_sorted_final[i_],
                                     n_plates,100,6).shape[0]
    
    outputdim_log=log2numpy_reader(realisation_name_log_sorted_final[i_],
                                                    path_2_log_files,
                                                    thermo_vars).shape[0]
    outputdim_posvel_dump=int(dump2numpy_f(dump_start_line_posvel,
                                    Path_2_dump,
                                    realisation_name_dump_vel_sorted_final[i_],
                                    n_plates*2).shape[0]/(n_plates*2))

    
    log_file_array=np.zeros((j_,outputdim_log,8)) #nemd
   
    
    spring_force_positon_array=np.zeros((j_,outputdim_dump,100,6))
    dirn_vector_array=np.zeros((j_,outputdim_dump,100,3))
    area_vector_array=np.zeros((j_,outputdim_dump,100,3))
    spring_extension_array=np.zeros((j_,outputdim_dump,100))



    for j in range(j_):
            j_index=j+(j_*count)



            log_file_array[j,:,:]=log2numpy_reader(realisation_name_log_sorted_final[j_index],
                                                        path_2_log_files,
                                                        thermo_vars)
            dump_data=dump2numpy_tensor_1tstep(dump_start_line,
                                    Path_2_dump,
                                    realisation_name_dump_sorted_final[j_index],
                                    n_plates,100,6)
            dump_vel_data=dump2numpy_f(dump_start_line_posvel,
                                    Path_2_dump,
                                    realisation_name_dump_vel_sorted_final[i_],
                                    n_plates*2)
            print(dump_data.shape)

            # bond local reverses the sign of direction, lower id - higher id 

            spring_force_positon_array[j,:,:,0]=-dump_data[:,:,0]*dump_data[:,:,3]#xx
            spring_force_positon_array[j,:,:,1]=-dump_data[:,:,1]*dump_data[:,:,4]#yy
            spring_force_positon_array[j,:,:,2]=-dump_data[:,:,2]*dump_data[:,:,5]#zz
            spring_force_positon_array[j,:,:,3]=-dump_data[:,:,0]*dump_data[:,:,5]#xz
            spring_force_positon_array[j,:,:,4]=-dump_data[:,:,0]*dump_data[:,:,4]#xy
            spring_force_positon_array[j,:,:,5]=-dump_data[:,:,1]*dump_data[:,:,5]#yz
            dirn_vector_array[j]=dump_data[:,:,3:6]

            #area_vector_array[j]=np.cross(-dump_data[:,:,0],-dump_data[:,:,1])
            x=dump_data[:,:,3]
            y=dump_data[:,:,4]
            z=dump_data[:,:,5]
            spring_extension_array[j]=np.sqrt(x**2 +y**2 +z**2)

            # can we get it to dump a hdf5 file for the tensor, this would be in order






            # need to compute the ell vectors from this. 

          

    lgf_mean=np.mean(log_file_array,axis=0)    
    spring_force_positon_mean=np.mean(np.mean(spring_force_positon_array,axis=0),
                                      axis=1)
    area_vector_tuple=area_vector_tuple+(area_vector_array,)
    spring_extension_tuple=spring_extension_tuple+(spring_extension_array,)

    velocities_tuple=velocities_tuple+(dump_vel_data,)


    
  
   
    log_file_tuple=log_file_tuple+(lgf_mean,)
    spring_force_positon_tensor_tuple=spring_force_positon_tensor_tuple+\
        (spring_force_positon_array,)
    dir_vector_tuple=dir_vector_tuple+(dirn_vector_array,)
    count+=1



#%% save tuples to avoid needing the next stage 
#make sure to comment this out after use
label='damp_'+str(thermal_damp)+'_K_'+str(K)+'_'
import pickle as pck

folder_check_or_create(filepath,"saved_tuples")

with open(label+'spring_force_positon_tensor_tuple.pickle', 'wb') as f:
    pck.dump(spring_force_positon_tensor_tuple, f)

with open(label+'log_file_tuple.pickle', 'wb') as f:
    pck.dump(log_file_tuple, f)

with open(label+'posvelocities_tuple.pickle', 'wb') as f:
    pck.dump(velocities_tuple, f)

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

#%% load in tuples

folder_check_or_create(filepath,"saved_tuples")


label='damp_'+str(damp)+'_K_'+str(K)+'_'


with open(label+'spring_force_positon_tensor_tuple.pickle', 'rb') as f:
    spring_force_positon_tensor_tuple=pck.load(f)

with open(label+'log_file_tuple.pickle', 'rb') as f:
    log_file_tuple=pck.load(f)

# with open(label+'p_velocities_tuple.pickle', 'rb') as f:
#     p_velocities_tuple=pck.load(f)

# with open(label+'p_positions_tuple.pickle', 'rb') as f:
#     pol_positions_tuple=pck.load(f)

# with open(label+"new_pos_vel_tuple.pickle",'rb') as f:
#      new_pos_vel_tuple=pck.load(f)

# with open(label+"area_vector_tuple.pickle", 'rb') as f:
#     area_vector_tuple= pck.load(f)

# with open(label+"interest_vectors_tuple.pickle",'rb') as f:
#      interest_vectors_tuple=pck.load(f)


#%% look at velocity dist

for i in range(erate.size):
    vel_data=np.ravel(velocities_tuple[i].astype('float')[:,5:8])
    sns.kdeplot(vel_data, bw_adjust=1)
plt.show()

for i in range(erate.size):
    vel_data=np.ravel(velocities_tuple[i].astype('float')[:,5:8])
    pos_data=np.ravel(velocities_tuple[i].astype('float')[:,2:5])
    plt.scatter(pos_data,vel_data)
plt.show()



#%% looking at energy, and temp 

#plot temp vs strain 
column=5# temp 
final_temp=np.zeros((erate.size))
mean_temp_array=np.zeros((erate.size))
pe_final_list=[]



for i in range(0,len(log_file_tuple)):
        
    
        strain_plot=np.linspace(0,strain_total,log_file_tuple[i][:,column].shape[0])
        column=4
        plt.plot(strain_plot,log_file_tuple[i][:,column],label="$\dot{\gamma}="+str(erate[i])+"$")
        #plt.plot(strain_plot,1-log_file_tuple[i][:,column],label='Convergence $\dot{\gamma}='+str(erate[i])+'$')
        #print(i)
       
        plt.ylabel("$T$")
        plt.xlabel("$\gamma$")
        #plt.legend(bbox_to_anchor=(1.5,1))
        #plt.yscale('log')
        #plt.legend()
        plt.show() 


        # column=5
        # plt.plot(log_file_tuple[i][:,column],label="uef_temp")
      

        # column=5 # ecouple 
        # plt.plot(log_file_tuple[i][:,column],label="Ecouple")
        
        # column=6 # econserve 
        # plt.plot(log_file_tuple[i][:,column],label="Econserve")
#%%

for i in range(0,len(log_file_tuple)):   
        column=2 # pe 
        
        plt.plot(strain_plot,log_file_tuple[i][:,column],label="$\dot{\gamma}="+str(erate[i])+"$")
        #print(log_file_tuple[i][-1,column])
        pe_final_list.append(log_file_tuple[i][-1,column])
        plt.ylabel("$E_{p}$")
        plt.xlabel("$\gamma$")
        #plt.ylim(1e-10,10)       
        plt.yscale('log')
        plt.legend()
        plt.show() 

# for i in range(0,len(log_file_tuple)):  
#         column=1 # ke 

#         plt.plot(log_file_tuple[i][:,column], label="Kinetic E")
#         # total_energy=log_file_tuple[i][:,1]+log_file_tuple[i][:,2]
#         # plt.plot(total_energy,label="total energy")
        
#         plt.legend(loc='upper right')
#         #plt.yscale('log')
#         plt.show() 


#%% mean temp
column=7 # uef temp 
final_temp=np.zeros((erate.size))
mean_temp_array=np.zeros((erate.size))

for i in range(e_in,e_end):
        
        # plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
        mean_temp_array[i]=np.mean(log_file_tuple[i][500:,column])
#         plt.plot(log_file_tuple[i][:,column])
# plt.show() 
        #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
    #     plt.ylabel("$T$", rotation=0)
    #     plt.xlabel("$\gamma$")
    

    # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
    #     plt.show()

#

marker=['x','o','+','^',"1","X","d","*","P","v"]
plt.scatter(erate[e_in:e_end],mean_temp_array[e_in:e_end])
plt.ylabel("$T$", rotation=0)
plt.xlabel('$\dot{\gamma}$')
#plt.xscale('log')
# plt.yscale('log')
plt.axhline(np.mean(mean_temp_array[e_in:e_end]),label="$\\bar{T}="+str(sigfig.round(np.mean(mean_temp_array[e_in:e_end]),sigfigs=5))+"$")
#plt.ylim(0.95,1.05)
plt.legend()
plt.tight_layout()
plt.show()


#%% computing stress distributions 
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import curve_fit
marker=['x','o','+','^',"1","X","d","*","P","v"]
aftcut=1
cut=0.7

labels_stress=["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{yz}$"]

def stress_tensor_averaging(e_end,
                            labels_stress,
                            trunc1,
                            trunc2,
                            spring_force_positon_tensor_tuple):
    stress_tensor=np.zeros((e_end,6))
    stress_tensor_std=np.zeros((e_end,6))
    stress_tensor_reals=np.zeros((e_end,j_,6))
    stress_tensor_std_reals=np.zeros((e_end,j_,6))
    for l in range(6):
        for i in range(e_end):
            for j in range(j_):
                cutoff=int(np.round(trunc1*spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0]))
                aftercutoff=int(np.round(trunc2*spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0]))
                # print(spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0])
                # print(cutoff)
                # print(aftercutoff)
                data=np.ravel(np.mean(spring_force_positon_tensor_tuple[i][j,cutoff:aftercutoff,:,l],axis=0))
                print(data.size)
                # plt.axhline(np.mean(data))
                plt.plot(data,label=labels_stress[l]+",$\dot{\gamma}="+str(erate[i])+"$")
                plt.legend()
                plt.show()
                stress_tensor_reals[i,j,l]=np.mean(data)
                stress_tensor_std_reals[i,j,l]=np.std(data)
                stress_tensor=np.mean(stress_tensor_reals, axis=1)
                stress_tensor_std=np.mean(stress_tensor_std_reals, axis=1)
    return stress_tensor,stress_tensor_std

stress_tensor,stress_tensor_std=stress_tensor_averaging(len(log_file_tuple),
                            labels_stress,
                            cut,
                            aftcut,
                            spring_force_positon_tensor_tuple)

#%% plot stress vs strain 


def plotting_stress_vs_strain(spring_force_positon_tensor_tuple,
                              i_1,i_2,j_,
                              strain_total,cut,aftcut,stress_component,label_stress):
    

    mean_grad_l=[] 
    for i in range(i_1,i_2):
        #for j in range(j_):
            cutoff=int(np.round(cut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            aftcutoff=int(np.round(aftcut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))


            strain_plot=np.linspace(cut*strain_total,aftcut*strain_total,spring_force_positon_tensor_tuple[i][0,cutoff:aftcutoff,:,stress_component].shape[0])
            cutoff=int(np.round(cut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            aftcutoff=int(np.round(aftcut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            stress=np.mean(spring_force_positon_tensor_tuple[i][:,:,:,stress_component],axis=0)
            stress=stress[cutoff:aftcutoff]
            gradient_vec=np.gradient(np.mean(stress,axis=1))
            mean_grad=np.mean(gradient_vec)
            mean_grad_l.append(mean_grad)
            print(stress.shape)
            # plt.plot(strain_plot,np.mean(stress,axis=1))
            # plt.axhline(np.mean(np.mean(stress,axis=1)))
            # #plt.ylabel(labels_stress[stress_component],rotation=0)
            # plt.xlabel("$\gamma$")
            # #plt.plot(strain_plot,gradient_vec, label="$\\frac{dy}{dx}="+str(mean_grad)+"$")

            # plt.legend()
            #plt.show()

    plt.scatter(erate[:e_end],mean_grad_l, label=label_stress)
    plt.xlabel("$\dot{\gamma}$")
    plt.ylabel("$\\frac{d\\bar{\sigma}_{\\alpha\\beta}}{dt}$", rotation=0,labelpad=20)
    plt.show()

for i in range(6):

    plotting_stress_vs_strain(spring_force_positon_tensor_tuple,
                              e_in,e_end,j_,
                              strain_total,cut,aftcut,i,labels_stress[i])
plt.legend(fontsize=14) 
plt.show()




#%% plotting N1 N2 vs strain 
def plotting_n_difff_vs_strain(spring_force_positon_tensor_tuple,
                              i_1,i_2,j_,
                              strain_total,i1,i2,cut,aftcut):
     
    for i in range(i_1,i_2):
        #for j in range(j_):
            cutoff=int(np.round(cut*spring_force_positon_tensor_tuple[i][0,:,:,0].shape[0]))
            aftcutoff=int(np.round(aftcut*spring_force_positon_tensor_tuple[i][0,:,:,0].shape[0]))
            strain_plot=np.linspace(cut*strain_total,aftcut*strain_total,spring_force_positon_tensor_tuple[i][0,cutoff:aftcutoff,:,0].shape[0])
            
            n_diff=spring_force_positon_tensor_tuple[i][:,cutoff:aftcutoff,:,i1]-spring_force_positon_tensor_tuple[i][:,cutoff:aftcutoff,:,i2]
            n_diff=np.mean(n_diff,axis=0)
            print(n_diff.shape)
            plt.xlabel("")
            plt.plot(strain_plot,np.mean(n_diff,axis=1), label ="$\dot{\gamma}="+str(erate[i])+"$")
    plt.show()


plotting_n_difff_vs_strain(spring_force_positon_tensor_tuple,e_in,e_end,j_,strain_total,0,2,cut,aftcut)

plotting_n_difff_vs_strain(spring_force_positon_tensor_tuple,e_in,e_end,j_,strain_total,2,1,cut,aftcut)


#%% plotting stress tensor with shear rate

def plot_stress_tensor(t_0,t_1,
                       stress_tensor,
                       stress_tensor_std,
                       j_,n_plates, labels_stress,marker,cutoff):
    for l in range(t_0,t_1):
          plt.errorbar(erate[cutoff:e_end], stress_tensor[:,l],
                        yerr =stress_tensor_std[:,l]/np.sqrt(j_*n_plates), 
                        ls='none',label=labels_stress[l],marker=marker[l] )
          plt.xlabel("$\dot{\gamma}$")
          plt.ylabel("$\sigma_{\\alpha\\beta}$",rotation=0,labelpad=20)
    #plt.yscale('log')
    plt.legend()      
    plt.show()

plot_stress_tensor(0,3,
                       stress_tensor,
                       stress_tensor_std,
                       j_,n_plates, labels_stress,marker,0)

plot_stress_tensor(3,6,
                       stress_tensor,
                       stress_tensor_std,
                       j_,n_plates, labels_stress,marker,0)

# for i in range(3,6):
#      plt.axhline(np.mean(stress_tensor[:,i],axis=0))
# plt.show()


    

#%%normal stress differences 

def compute_plot_n_stress_diff(stress_tensor, 
                          stress_tensor_std,
                          i1,i2,
                          j_,n_plates,
                          erate,e_end,
                          ylab):
    n_diff=stress_tensor[:,i1]- stress_tensor[:,i2]
    n_diff_error=np.sqrt(stress_tensor_std[:,i1]**2 +stress_tensor_std[:,i2]**2)/np.sqrt(j_*n_plates)
    #plt.errorbar(erate[:e_end], n_diff, yerr =n_diff_error, ls='none' )
    plt.scatter(erate[:e_end],n_diff )
    plt.ylabel(ylab, rotation=0, labelpad=20)
    plt.xlabel("$\dot{\gamma}$")
   
    plt.legend()  
    plt.show() 
        
    return n_diff,n_diff_error

#n1 and n2 not real in extension 
# n_1,n_1_error=compute_plot_n_stress_diff(stress_tensor, 
#                           stress_tensor_std,
#                           0,2,
#                           j_,n_plates,
#                           erate,e_end,
#                           "$N_{1}$")

# n_2,n_2_error=compute_plot_n_stress_diff(stress_tensor, 
#                           stress_tensor_std,
#                           2,1,
#                           j_,n_plates,
#                           erate,e_end,
#                           "$N_{2}$")
  

#compute extensional viscosity

def ext_visc_compute(stress_tensor,stress_tensor_std,i1,i2,n_plates,e_end,e_in):
    extvisc=(stress_tensor[:,i1]- stress_tensor[:,i2])/erate[e_in:e_end]/30.3
    extvisc_error=np.sqrt(stress_tensor_std[:,i1]**2 +stress_tensor_std[:,i2]**2)/np.sqrt(j_*n_plates)

    return extvisc,extvisc_error

ext_visc_1,ext_visc_1_error=ext_visc_compute(stress_tensor,stress_tensor_std,0,2,n_plates,e_end,e_in)
     
#ext_visc_2,ext_visc_2_error=ext_visc_compute(stress_tensor,stress_tensor_std,1,2,n_plates,e_end)
cutoff=2
plt.errorbar(erate[cutoff:e_end],ext_visc_1[cutoff:],yerr=ext_visc_1_error[cutoff:], label="$\eta_{1}$", linestyle='none', marker='x')
#plt.plot(erate[:e_end],ext_visc_1,label="$\eta_{1}$", linestyle='none', marker='x')
plt.ylabel("$\eta/\eta_{s}$", rotation=0, labelpad=20)
plt.xlabel("$\dot{\gamma}$")
# popt,cov_matrix_n1=curve_fit(linearthru0,erate[cutoff:e_end], ext_visc_1[:e_end])
# difference=np.sqrt(np.sum((ext_visc_1[:e_end]-(popt[0]*(erate[cutoff:e_end])**2))**2)/(e_end))
# plt.plot(erate[cutoff:e_end],popt[0]*erate[cutoff:e_end], label="fit")
# plt.plot(erate[1:e_end],ext_visc_2, label="$\eta_{2}$")
# plt.xscale('log')
#plt.yscale('log')

plt.show()

#%% extension distribution 
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
eq_spring_length=3*np.sqrt(3)/2
skip_array=np.arange(0,e_end,3)
#for i in range(1,e_end):
for i in range(skip_array.size):
    i=skip_array[i]
# for i in range(e_in,e_end):

    # sns.kdeplot(eq_spring_length-np.ravel(interest_vectors_tuple[i][:,:,2:5]),
    #              label ="$K="+str(K)+"$")
                 #label ="$\dot{\gamma}="+str(erate[i])+"$")
    #spring_extension=np.ravel(np.mean(spring_extension_tuple[i],axis=0))
    spring_extension=np.ravel(spring_extension_tuple[i])
    sns.kdeplot(eq_spring_length-spring_extension,
                  label ="$\dot{\gamma}="+str(erate[i])+"$", bw_adjust=5)
    
    plt.xlabel("$\Delta x$")

    plt.ylabel('Density')
    #plt.legend(bbox_to_anchor=[1.1, 0.45])

    plt.legend(fontsize=11) 
    #plt.xlim(-3,2)
    
plt.show()

# this calculation works
# i think we just need to get a higher value for damp
#%% quiver plot of directions 
# need positions for this to work 

import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')
ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

plt.show()
        

      
# %% dumbell vector analysis 

pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
pi_phi_ticks=[ 0,np.pi/4, np.pi/2]
pi_phi_tick_labels=[ '0','π/4', 'π/2']
spherical_coords_tuple=()
cutoff=500
for i in range(e_in,e_end):
     
    dirn_vector_ray=dir_vector_tuple[i]
    # detect all z coords less than 0 and multiply all 3 coords by -1
    dirn_vector_ray[dirn_vector_ray[:,:,:,2]<0]*=-1
    spherical_coords_array=np.zeros((j_* n_plates,3))
    x=np.ravel(np.mean(dirn_vector_ray[:,cutoff:,:,0],axis=1))
    y=np.ravel(np.mean(dirn_vector_ray[:,cutoff:,:,1],axis=1))
    z=np.ravel(np.mean(dirn_vector_ray[:,cutoff:,:,2],axis=1))


     # radial coord
    spherical_coords_array[:,0]=np.sqrt((x**2)+(y**2)+(z**2))
     #  theta coord 
    spherical_coords_array[:,1]=np.sign(y)*np.arccos(x/(np.sqrt((x**2)+(y**2))))
     # phi coord
    spherical_coords_array[:,2]=np.arccos(z/spherical_coords_array[:,0])

    spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_array,)

#%% look at chnage of theta with time 
for i in range(e_in,e_end):
     strain_plot=np.linspace(0,strain_total,spherical_coords_array[i,:,0,2].shape[0])
     plt.plot( strain_plot,spherical_coords_array[i,:,0,1])
     plt.show()

#%%
for i in range(e_in,e_end):
     strain_plot=np.linspace(0,strain_total,spherical_coords_array[i,:,0,2].shape[0])
     plt.plot( strain_plot,spherical_coords_array[i,:,0,2])
     plt.xlabel("$\gamma$")
     plt.ylabel("$\phi$")
     pi_phi_ticks=[ 0,np.pi/4, np.pi/2]
     pi_phi_tick_labels=[ '0','π/4', 'π/2']
     plt.yticks(pi_phi_ticks,pi_phi_tick_labels)
     #plt.ylim(0,np.pi/2)
     plt.show()
     


#%% rho 
for i in range(e_in,e_end):
    sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,:,:,0]),
                label ="$\dot{\gamma}="+str(erate[i])+"$")

plt.legend()
plt.show()
#%% theta 
# could just plot a few of them 
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
skip_array=np.arange(0,e_end,4)
skip_array_2=np.arange(0,int(no_timesteps[0]/100),100)



for i in range(erate.size):
    #for j in range(j_):


        #i=skip_array[i]
        
        # sns.displot( data=np.ravel(spherical_coords_tuple[i][:,200000,:,1]),
        #             label ="$\dot{\gamma}="+str(erate[i])+"$", kde=True)
        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,skip_array_2[j],:,1]),
        #             label="output_range:"+str(skip_array_2[j]))
        data=np.ravel(spherical_coords_tuple[i][:,1])
        periodic_data=np.array([data-2*np.pi,data,data+2*np.pi])  

        sns.kdeplot( data=np.ravel(periodic_data),
                    label ="$\dot{\gamma}="+str(erate[i],)+"$")#bw_adjust=0.1
        
        # mean_data=np.mean(spherical_coords_tuple[0][:,-1,:,1],axis=0)      
        #plt.hist(np.ravel(spherical_coords_tuple[i][:,-100,:,1]))
        # bw adjust effects the degree of smoothing , <1 smoothes less
        plt.xlabel("$\Theta$")
        #plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
        #plt.xlim(-np.pi,np.pi)
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=[1.1, 0.45])
plt.show()


#%% chi squared test
# generate uniform data 
rng = np.random.default_rng(1234)
s = rng.uniform(-np.pi,np.pi,n_plates*j_)
bins=10
# make into histograms 
data=spherical_coords_tuple[0][0,-1,:,1]
freq_exp=np.histogram(s,bins=bins)[0]
freq_obv=np.histogram(data,bins=bins)[0]
degrees_freedom=bins-1
chi_stat,pvalue=scipy.stats.chisquare(f_obs=freq_obv,ddof=0)
print(chi_stat)
print(pvalue)

chi_stat_table=scipy.stats.chi2.ppf(0.05,degrees_freedom)
print(chi_stat_table)





#%% phi 
pi_phi_ticks=[ 0,np.pi/4, np.pi/2]
pi_phi_tick_labels=[ '0','π/4', 'π/2']
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
for i in range(erate.size):
    #for j in range(skip_array_2.size):
        #i=skip_array[i]

        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,skip_array_2[j],:,2]),
        #              label="output_range:"+str(skip_array_2[j]))
        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,-1,:,2]),
        #              label ="$\dot{\gamma}="+str(erate[i])+"$")
        data=np.ravel(spherical_coords_tuple[i][:,2])
        periodic_data=np.array([data,np.pi-data])  
        #periodic_data=np.array([data])  
        sns.kdeplot( data=np.ravel(periodic_data),
                      label ="$\dot{\gamma}="+str(erate[i])+"$")
                   
        #plt.hist(np.ravel(spherical_coords_tuple[i][:,-1,:,2]))

plt.xlabel("$\Phi$")
plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
plt.ylabel('Density')
plt.legend(bbox_to_anchor=[1.1, 0.45])
plt.xlim(0,np.pi/2)
plt.show()

#%% extension_vectors
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
eq_spring_length=3*np.sqrt(3)/2
skip_array=np.arange(0,e_end,3)
for i in range(skip_array.size):
    i=skip_array[i]
# for i in range(e_in,e_end):

    # sns.kdeplot(eq_spring_length-np.ravel(interest_vectors_tuple[i][:,:,2:5]),
    #              label ="$K="+str(K)+"$")
                 #label ="$\dot{\gamma}="+str(erate[i])+"$")
    sns.kdeplot(eq_spring_length+0.125-np.ravel(interest_vectors_tuple[i][:,:,2:5]),
                  label ="$\dot{\gamma}="+str(erate[i])+"$")
    
plt.xlabel("$\Delta x$")

plt.ylabel('Density')
#plt.legend(bbox_to_anchor=[1.1, 0.45])

plt.legend(fontsize=11) 
plt.xlim(-3,2)
plt.show()




#%% cleaner plots 
#stress tensor 
plt.rcParams['text.usetex'] = True
def plot_stress_tensor(t_0,t_1,
                       stress_tensor,
                       stress_tensor_std,
                       j_,n_plates, labels_stress,marker,cutoff):
    
    fig, ax = plt.subplots(figsize=(6, 5))
    color = ["#2B2F42", "#8D99AE", "#EF233C","#2B2F42", "#8D99AE", "#EF233C"]
    # Define font sizes
    SIZE_DEFAULT = 14
    SIZE_LARGE = 16
    plt.rcParams['text.usetex'] = True
    plt.rc("font", family="Roboto")  # controls default font
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
        
    for l in range(t_0,t_1):
        ax.errorbar(erate[cutoff:e_end], 
                    stress_tensor[cutoff:,l],
                      yerr =stress_tensor_std[cutoff:,l]/np.sqrt(j_*n_plates), 
                      ls='--',label=labels_stress[l],marker=marker[l], color=color[l] )
        #   ax.text(
        #         erate[-1] * 1.05,
        #         stress_tensor[-1,l],
        #         labels_stress[l],
        #         fontweight="bold",
        #         horizontalalignment="left",
        #         verticalalignment="center",
        #     )
        ax.spines["right"].set_visible(False)
       # ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.xlabel("$\dot{\gamma}$")
    plt.ylabel("$\sigma_{\\alpha\\beta}$",rotation=0,labelpad=15)
    plt.legend(loc="best")      
    plt.show()

plot_stress_tensor(0,3,
                       stress_tensor,
                       stress_tensor_std,
                       j_,n_plates, labels_stress,marker,0)

plot_stress_tensor(3,6,
                       stress_tensor,
                       stress_tensor_std,
                       j_,n_plates, labels_stress,marker,0)

# %%
