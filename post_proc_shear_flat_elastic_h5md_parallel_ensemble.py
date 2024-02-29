# ##!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# This script will calculate the MPCD stress tensor for a pure fluid under forward NEMD using hdf5 files and python multiprocessing
# to ensure fast analysis  
# """
# Importing packages
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import regex as re
import pandas as pd
import multiprocessing as mp
from multiprocessing import Process
import time
import h5py as h5
import seaborn as sns
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True
#from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import h5py as h5 
import multiprocessing as mp
from log2numpy import *
from mom2numpy import *
from velP2numpy import *
from dump2numpy import * 
import glob 
from post_MPCD_MP_processing_module import *
colour = [
 'black',
 'blueviolet',
 'cadetblue',
 'chartreuse',
 'coral',
 'cornflowerblue',
 'crimson',
 'darkblue',
 'darkcyan',
 'darkgoldenrod',
 'darkgray']

# key inputs 
# no_SRD=1038230
# box_size=47
# no_SRD=506530
# box_size=37
# no_SRD=121670
# box_size=23
# no_SRD=58320
# box_size=18
no_SRD=2160
box_size=6
# no_SRD=270
# box_size=3
# no_SRD=2560
# box_size=8
no_SRD=60835
box_size=23
#nu_bar=3
#delta_t_srd=0.014872025172594354
#nu_bar=0.9 
#rho=10
#delta_t_srd=0.05674857690605889
rho=5
delta_t_srd=0.05071624521210362

box_vol=box_size**3

erate= np.array([0.001])

no_timesteps=300
# estimating number of steps  required
strain=3
delta_t_md=delta_t_srd/10
strain_rate= np.array([0.01,0.001,0.0001])
number_steps_needed= np.ceil(strain/(strain_rate*delta_t_md))
dump_freq=10
spring_stiffness =np.array([20,40,80,100])
bending_stiffness=10000

#rho=5
realisation_index=np.array([1,2,3])
# finding all the dump files in a folder

VP_general_name_string='vel.*'

Mom_general_name_string='mom.*'

log_general_name_string='log.*'
                         
TP_general_name_string='temp.*'

dump_general_name_string_after_srd='*SRDs*after*.h5'
dump_general_name_string_before_srd='*SRDs*before*.h5'
dump_general_name_string_after_pol='*pol*after*.h5'
dump_general_name_string_before_pol='*pol*before*.h5'
dump_general_name_string_after_phantom='*phantom*after*.h5'
dump_general_name_string_before_phantom='*phantom*before*.h5'


#filepath="/home/ucahlrl/Scratch/output"
filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/dist_test_1000/"

Path_2_dump=filepath
# can chnage this to another array on kathleen

#srd before 
dump_realisation_name_info_before_srd=VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_before_srd)
realisation_name_h5_before_srd=dump_realisation_name_info_before_srd[6]
count_h5_before_srd=dump_realisation_name_info_before_srd[7]
# pol before 
dump_realisation_name_info_before_pol=VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_before_pol)
realisation_name_h5_before_pol=dump_realisation_name_info_before_pol[6]
count_h5_before_pol=dump_realisation_name_info_before_pol[7]
# phantom before 
dump_realisation_name_info_before_phantom=VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_before_phantom)
realisation_name_h5_before_phantom=dump_realisation_name_info_before_phantom[6]
count_h5_before_phantom=dump_realisation_name_info_before_phantom[7]

# srd after 
dump_realisation_name_info_after_srd= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_after_srd)
realisation_name_h5_after_srd=dump_realisation_name_info_after_srd[6]
count_h5_after_srd=dump_realisation_name_info_after_srd[7]
# pol after 
dump_realisation_name_info_after_pol= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_after_pol)
realisation_name_h5_after_pol=dump_realisation_name_info_after_pol[6]
count_h5_after_pol=dump_realisation_name_info_after_pol[7]
#phantom after 
dump_realisation_name_info_after_phantom= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_after_phantom)
realisation_name_h5_after_phantom=dump_realisation_name_info_after_phantom[6]
count_h5_after_phantom=dump_realisation_name_info_after_phantom[7]



# # find dump file size
with h5.File(realisation_name_h5_after_pol[0], 'r') as f:
    shape_after= f['particles']['pol']['position']['value'].shape
    f.close()
print(shape_after[0])

with h5.File(realisation_name_h5_before_pol[0], 'r') as f_i:
      first_step= f_i['particles']['pol']['position']['step'][0]
      first_posn= f_i['particles']['pol']['position']['value'][0]
      f.close()

j_=1000
no_data_sets=erate.shape[0]

#
# creating indices for mapping 3-D data to 1-D sequence
def producing_plus_N_sequence_for_ke(increment,n_terms):
    n=np.arange(1,n_terms) # if steps change need to chnage this number 
    terms_n=[0]
    for i in n:
            terms_n.append(terms_n[i-1]+int(increment))
       
    return terms_n 

n_terms=j_*no_data_sets*(shape_after[0]-1) 
#n_terms=j_*no_data_sets*(shape_after[0]) # if only one step 
terms_9=producing_plus_N_sequence_for_ke(9,n_terms)
terms_6=producing_plus_N_sequence_for_ke(6,n_terms)
terms_3=producing_plus_N_sequence_for_ke(3,n_terms)

##sorting realisation lists


# found this method https://www.youtube.com/watch?v=D3JvDWO-BY4&ab_channel=CoreySchafer
class realisation():
     def __init__(self,realisation_full_str,data_set,realisation_index_):
          self.realisation_full_str= realisation_full_str
          self.data_set= data_set
          self.realisation_index_=realisation_index_
     def __repr__(self):
        return '({},{},{})'.format(self.realisation_full_str,self.data_set,self.realisation_index_)
realisations_for_sorting_after_srd=[]
realisation_split_index=6
spring_stiff_index=20
for i in realisation_name_h5_after_srd:
          realisation_index_=i.split('_')[realisation_split_index]
          data_set =i.split('_')[spring_stiff_index]
          realisations_for_sorting_after_srd.append(realisation(i,data_set,realisation_index_))

realisations_for_sorting_before_srd=[]
for i in realisation_name_h5_before_srd:
          realisation_index_=i.split('_')[realisation_split_index]
          data_set =i.split('_')[spring_stiff_index]
          realisations_for_sorting_before_srd.append(realisation(i,data_set,realisation_index_))

realisations_for_sorting_after_pol=[]
for i in realisation_name_h5_after_pol:
          realisation_index_=i.split('_')[realisation_split_index]
          data_set =i.split('_')[spring_stiff_index]
          realisations_for_sorting_after_pol.append(realisation(i,data_set,realisation_index_))

realisations_for_sorting_before_pol=[]
for i in realisation_name_h5_before_pol:
          realisation_index_=i.split('_')[realisation_split_index]
          data_set =i.split('_')[spring_stiff_index]
          realisations_for_sorting_before_pol.append(realisation(i,data_set,realisation_index_))

realisations_for_sorting_after_phantom=[]
for i in realisation_name_h5_after_phantom:
          realisation_index_=i.split('_')[realisation_split_index]
          data_set =i.split('_')[spring_stiff_index]
          realisations_for_sorting_after_phantom.append(realisation(i,data_set,realisation_index_))

realisations_for_sorting_before_phantom=[]
for i in realisation_name_h5_before_phantom:
          realisation_index_=i.split('_')[realisation_split_index]
          data_set =i.split('_')[spring_stiff_index]
          realisations_for_sorting_before_phantom.append(realisation(i,data_set,realisation_index_))





#### srd 
realisation_name_h5_after_sorted_srd=sorted(realisations_for_sorting_after_srd,key=lambda x: (x.data_set, x.realisation_index_))
realisation_name_h5_after_sorted_final_srd=[]
for i in realisation_name_h5_after_sorted_srd:
     realisation_name_h5_after_sorted_final_srd.append(i.realisation_full_str)

realisation_name_h5_before_sorted_srd=sorted(realisations_for_sorting_before_srd,key=lambda x: (x.data_set, x.realisation_index_))
realisation_name_h5_before_sorted_final_srd=[]
for i in realisation_name_h5_before_sorted_srd:
     realisation_name_h5_before_sorted_final_srd.append(i.realisation_full_str)
### pol
realisation_name_h5_after_sorted_pol=sorted(realisations_for_sorting_after_pol,key=lambda x: (x.data_set, x.realisation_index_))
realisation_name_h5_after_sorted_final_pol=[]
for i in realisation_name_h5_after_sorted_pol:
     realisation_name_h5_after_sorted_final_pol.append(i.realisation_full_str)

realisation_name_h5_before_sorted_pol=sorted(realisations_for_sorting_before_pol,key=lambda x: (x.data_set, x.realisation_index_))
realisation_name_h5_before_sorted_final_pol=[]
for i in realisation_name_h5_before_sorted_pol:
     realisation_name_h5_before_sorted_final_pol.append(i.realisation_full_str)

## phantom
realisation_name_h5_after_sorted_phantom=sorted(realisations_for_sorting_after_phantom,key=lambda x: (x.data_set, x.realisation_index_))
realisation_name_h5_after_sorted_final_phantom=[]
for i in realisation_name_h5_after_sorted_phantom:
     realisation_name_h5_after_sorted_final_phantom.append(i.realisation_full_str)

realisation_name_h5_before_sorted_phantom=sorted(realisations_for_sorting_before_phantom,key=lambda x: (x.data_set, x.realisation_index_))
realisation_name_h5_before_sorted_final_phantom=[]
for i in realisation_name_h5_before_sorted_phantom:
     realisation_name_h5_before_sorted_final_phantom.append(i.realisation_full_str)


print(realisation_name_h5_after_sorted_final_srd)
print(realisation_name_h5_before_sorted_final_srd)
print(realisation_name_h5_after_sorted_final_pol)
print(realisation_name_h5_before_sorted_final_pol)
print(realisation_name_h5_after_sorted_final_phantom)
print(realisation_name_h5_before_sorted_final_phantom)
          

realisation_name_h5_before_srd=realisation_name_h5_before_sorted_final_srd
realisation_name_h5_after_srd=realisation_name_h5_after_sorted_final_srd
realisation_name_h5_before_pol=realisation_name_h5_before_sorted_final_pol
realisation_name_h5_after_pol=realisation_name_h5_after_sorted_final_pol
realisation_name_h5_before_phantom=realisation_name_h5_before_sorted_final_phantom
realisation_name_h5_after_phantom=realisation_name_h5_after_sorted_final_phantom

# Computing full stress tensor of flat elastic.
mass_pol=5 

#NOTE sorting not working properly 


def area_vector_calculation(spring_stiffness,terms_3,area_vector_summed_shared,realisation_name_h5_after_pol,shape_after,erate,p):    
    
    # make sure to put the realisation name arguments with list index [p]
   
     with h5.File(realisation_name_h5_after_pol,'r') as f_c:
                

                            # need to change this to looking for spring constant 
          # erate
          data_set = int(np.where(erate==float(realisation_name_h5_after_pol.split('_')[16]))[0][0])
          # spring constant 
          K=int(np.where(spring_stiffness==float(realisation_name_h5_after_pol.split('_')[20]))[0][0])


           # has -1 when there is more than one set 
          count=int((shape_after[0]-1)*p) # to make sure the data is in the correct location in 1-D array 
          
          # looping through N-1 dumps, since we are using a delta 
          #for j in range(1,shape_after[0]):
          for j in range(0,1):
               

                   
                    # have to use small for hirotori edit , not pol
                    pol_positions_after= f_c['particles']['pol']['position']['value'][j]
                    # below for hirtori
                    #pol_positions_after= f_c['particles']['small']['position']['value'][j]

                    ell_1=pol_positions_after[1,:]-pol_positions_after[0,:]
                    ell_2=pol_positions_after[2,:]-pol_positions_after[0,:]
                    area_vector=np.cross(ell_1,ell_2)

                  
                    # for 3-d vector 
                    start_index_3_element=terms_3[j+(count)-1] # has -1 when there is more than one set 
                    end_index_3_element=   start_index_3_element+3

     
                    
                    # inserting values into final 1-D array 
                    #kinetic_energy_tensor_summed_shared[start_index_6_element:end_index_6_element]=np_array_ke_entries
                    #delta_mom_pos_tensor_summed_shared[start_index_9_element:end_index_9_element]=np_array_delta_mom_pos
                    area_vector_summed_shared[start_index_3_element:end_index_3_element]=area_vector


     return area_vector_summed_shared


# parallel analysis of small run 

# determine 1-D array size 
# size_delta_mom_pos_tensor_summed=int(no_data_sets*j_*((shape_after[0])-1)*9)
# print(size_delta_mom_pos_tensor_summed)
# size_stress_tensor_summed=int(no_data_sets*j_*((shape_after[0])-1)*9) 
# print(size_stress_tensor_summed)
# size_kinetic_energy_tensor_summed=int(no_data_sets*j_*((shape_after[0])-1))*6
# print(size_kinetic_energy_tensor_summed)
size_area_vector_summed=int(no_data_sets*j_*((shape_after[0]))*3) 
print(size_area_vector_summed)
processes=[]
print(count_h5_after_srd)
  


tic=time.perf_counter()

if __name__ =='__main__':
        
        # need to change this code so it only does 5 arrays at a time 
        
        # creating shared memory arrays that can be updated by multiple processes
        #delta_mom_pos_tensor_summed_shared=  mp.Array('d',range(size_delta_mom_pos_tensor_summed))
        area_vector_summed_shared=   mp.Array('d',range(size_area_vector_summed))
        # kinetic_energy_tensor_summed_shared=   mp.Array('d',range(size_kinetic_energy_tensor_summed))
        list_done=[]
        
        simulations_analysed=0   
        increment=10 
        #creating processes, iterating over each realisation name 
        while simulations_analysed < j_:
               for p in  range(simulations_analysed,simulations_analysed+increment):
                    
                    proc= Process(target=area_vector_calculation,args=(spring_stiffness,terms_3,area_vector_summed_shared,realisation_name_h5_after_pol[p],shape_after,erate,p,))
                                                  
                    processes.append(proc)
                    proc.start()
               
               for proc in  processes:
                    proc.join()
                    print(proc)

               simulations_analysed+=increment
             


        toc=time.perf_counter()
        print("Parallel analysis done in ", (toc-tic)/60," mins")
        print("Simulations analysed:",simulations_analysed)
       
        # could possibly make this code save the unshaped arrays to save the problem 
        np.save("area_vector_summed_1d_test_erate_"+str(erate[0])+"_bk_"+str(bending_stiffness)+"_M_"+str(rho)+"_L_"+str(box_size)+"_no_timesteps_"+str(no_timesteps),area_vector_summed_shared)
        # np.save("delta_mom_pos_tensor_summed_1d_test_M_"+str(rho)+"_L_"+str(box_size),delta_mom_pos_tensor_summed_shared)
        # np.save("kinetic_energy_tensor_summed_1d_test_M_"+str(rho)+"_L_"+str(box_size),kinetic_energy_tensor_summed_shared)


# area_vector_1d=np.load("area_vector_summed_1d_test_erate_"+str(erate[0])+"_bk_"+str(bending_stiffness)+"_M_"+str(rho)+"_L_"+str(box_size)+"_no_timesteps_"+str(no_timesteps)+".npy")
# area_vector_3d=np.reshape(area_vector_1d,(j_,3))
# # not so simple to calculate spherical coords, using mathematics convention from wiki

# spherical_coordinates_area_vector=np.zeros((j_,3))

# x=area_vector_3d[:,0]
# y=area_vector_3d[:,1]
# z=area_vector_3d[:,2]
# # radial coord
# spherical_coordinates_area_vector[:,0]=np.sqrt((x**2)+(y**2)+(z**2))
# # theta coord 
# spherical_coordinates_area_vector[:,1]=np.sign(y)*np.arccos(x/(np.sqrt((x**2)+(y**2))))
# # phi coord
# spherical_coordinates_area_vector[:,2]=np.arccos(z/spherical_coordinates_area_vector[:,0])

# # plot theta histogram
# plt.hist(spherical_coordinates_area_vector[:,1],density=True, bins=int(j_/10))
# plt.show()

# #plot phi histogram 
# plt.hist(spherical_coordinates_area_vector[:,2],density=True, bins=int(j_/10))
# plt.show()

# sns.displot(data=spherical_coordinates_area_vector[:,1], kde=True, bins=int(j_/10))
# sns.displot(data=spherical_coordinates_area_vector[:,2], kde=True, bins=int(j_/10))
                

        
