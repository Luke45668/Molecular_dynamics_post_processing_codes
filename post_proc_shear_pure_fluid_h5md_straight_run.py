# ##!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# This script will calculate the MPCD stress tensor for a pure fluid under forward NEMD using hdf5 files and python multiprocessing
# to ensure fast analysis  
# """
# Importing packages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import regex as re
import pandas as pd

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
# no_SRD=2160
# box_size=6
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
delta_t_srd=0.05674857690605889
rho=5
delta_t_srd=0.05071624521210362

box_vol=box_size**3
erate= np.array([0.01,0.001,0.0001])
erate=np.array([0.01])
no_timesteps=100000
# estimating number of steps  required
strain=3
delta_t_md=delta_t_srd/10
strain_rate= np.array([0.01,0.001,0.0001])
number_steps_needed= np.ceil(strain/(strain_rate*delta_t_md))
dump_freq=10

realisation_index=np.array([1,2,3])
# finding all the dump files in a folder

VP_general_name_string='vel.*'

Mom_general_name_string='mom.*'

log_general_name_string='log.*'
                         #log.H20_no466188_wall_VACF_output_no_rescale_
TP_general_name_string='temp.*'

dump_general_name_string_after='*'+str(no_timesteps)+'*after*.h5'
dump_general_name_string_before='*'+str(no_timesteps)+'*before*.h5'

filepath="/KATHLEEN_LAMMPS_RUNS/equilibrium_fix_deform_pure_mpcd_test_file"
filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/hfd5_runs/non_equilibrium_tests/2dumps_10k_collisions/test_non_eq_box_"+str(int(box_size))+"_M_"+str(rho)
#filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/hfd5_runs/tests_equilibrium_with_more_regular_neighbour_listing_box_"+str(int(box_size))+"_M_10"
Path_2_dump=filepath
# can chnage this to another array on kathleen
dump_realisation_name_info_before=VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_before)
realisation_name_h5_before=dump_realisation_name_info_before[6]
count_h5_before=dump_realisation_name_info_before[7]

dump_realisation_name_info_after= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_after)

realisation_name_h5_after=dump_realisation_name_info_after[6]
count_h5_after=dump_realisation_name_info_after[7]
print(count_h5_after)

# # find dump file size
with h5.File(realisation_name_h5_after[0], 'r') as f:
    shape_after= f['particles']['SRDs']['position']['value'].shape
    f.close()
print(shape_after[0])
fraction_of_steps=0#500/50000
shape_truncated_in=int(shape_after[0]*fraction_of_steps)
start_point_after_truncation=shape_after[0]-shape_truncated_in
# print(shape_truncated_in)
# print(shape_after[0]-shape_truncated_in)
# print("start point ",start_point_after_truncation)


with h5.File(realisation_name_h5_before[0], 'r') as f_i:
      first_step= f_i['particles']['SRDs']['position']['step'][0]
      first_posn= f_i['particles']['SRDs']['position']['value'][0]
    #   print(first_step)
    #   print(first_posn)
      f.close()

j_=1
no_data_sets=erate.shape[0]


# creating indices for mapping 3-D data to 1-D sequence
def producing_plus_N_sequence_for_ke(increment,n_terms):
    n=np.arange(1,n_terms) # if steps change need to chnage this number 
    terms_n=[0]
    for i in n:
            terms_n.append(terms_n[i-1]+int(increment))
       
    return terms_n 

n_terms=j_*no_data_sets*(shape_after[0]-shape_truncated_in-1)
terms_9=producing_plus_N_sequence_for_ke(9,n_terms)
terms_6=producing_plus_N_sequence_for_ke(6,n_terms)

##sorting realisation lists

# found this method https://www.youtube.com/watch?v=D3JvDWO-BY4&ab_channel=CoreySchafer
class realisation():
     def __init__(self,realisation_full_str,data_set,realisation_index_):
          self.realisation_full_str= realisation_full_str
          self.data_set= data_set
          self.realisation_index_=realisation_index_
     def __repr__(self):
        return '({},{},{})'.format(self.realisation_full_str,self.data_set,self.realisation_index_)
realisations_for_sorting_after=[]
for i in realisation_name_h5_after:
          realisation_index_=i.split('_')[9]
          data_set =i.split('_')[15]
          realisations_for_sorting_after.append(realisation(i,data_set,realisation_index_))

realisations_for_sorting_before=[]
for i in realisation_name_h5_before:
          realisation_index_=i.split('_')[9]
          data_set =i.split('_')[15]
          realisations_for_sorting_before.append(realisation(i,data_set,realisation_index_))

        
realisation_name_h5_after_sorted=sorted(realisations_for_sorting_after,key=lambda x: (x.data_set, x.realisation_index_))
realisation_name_h5_after_sorted_final=[]
for i in realisation_name_h5_after_sorted:
     realisation_name_h5_after_sorted_final.append(i.realisation_full_str)
realisation_name_h5_before_sorted=sorted(realisations_for_sorting_before,key=lambda x: (x.data_set, x.realisation_index_))
realisation_name_h5_before_sorted_final=[]
for i in realisation_name_h5_before_sorted:
     realisation_name_h5_before_sorted_final.append(i.realisation_full_str)


print(realisation_name_h5_after_sorted_final)
print(realisation_name_h5_before_sorted_final)
          
realisation_name_h5_before=realisation_name_h5_before_sorted_final
realisation_name_h5_after=realisation_name_h5_before_sorted_final




 
def stress_tensor_total_compute_shear(shape_truncated_in,terms_9,terms_6,box_vol,realisation_index,delta_mom_pos_tensor_summed_shared,stress_tensor_summed_shared,kinetic_energy_tensor_summed_shared,realisation_name_h5_after,realisation_name_h5_before,shape_after,j_,no_data_sets,erate,delta_t_srd,p):    
    import h5py as h5 

    with h5.File(realisation_name_h5_after, 'r') as f_a:
        with h5.File(realisation_name_h5_before, 'r') as f_b:
            data_set = int(np.where(erate==float(realisation_name_h5_after.split('_')[15]))[0][0])
            print(data_set)
            k=int(np.where(realisation_index==float(realisation_name_h5_after.split('_')[9]))[0][0])
        
            # if we are using the whole array    
            if shape_truncated_in==0:
                count=int((shape_after[0]-1)*p)# to make sure the data is in the correct location in 1-D arrat 

                #print(count)
                for j in range(1,shape_after[0]):
                        # the plus 
            
                        # taking the position and vel data out of hdf5 format
                        SRD_positions_initial= f_b['particles']['SRDs']['position']['value'][j-1]
                        SRD_positions_after= f_a['particles']['SRDs']['position']['value'][j]
                        SRD_velocities_initial=f_b['particles']['SRDs']['velocity']['value'][j-1]
                        SRD_velocities_after=f_a['particles']['SRDs']['velocity']['value'][j]
                        
                        # calculating the mom_pos tensor 
                        delta_mom_pos_tensor_summed_xx=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#xx
                        delta_mom_pos_tensor_summed_yy=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#yy
                        delta_mom_pos_tensor_summed_zz=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#zz
                        delta_mom_pos_tensor_summed_xz=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#xz
                        delta_mom_pos_tensor_summed_xy=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#xy
                        delta_mom_pos_tensor_summed_yz=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#yz
                        delta_mom_pos_tensor_summed_zx=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#zx
                        delta_mom_pos_tensor_summed_zy=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#zy
                        delta_mom_pos_tensor_summed_yx=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#yx
                        
                        # calculating ke tensor 
                        kinetic_energy_tensor_summed_xx=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,0],axis=0)/(box_vol)#xx
                        kinetic_energy_tensor_summed_yy=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#yy
                        kinetic_energy_tensor_summed_zz=np.sum(SRD_velocities_initial[:,2]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#zz
                        kinetic_energy_tensor_summed_xy=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#xy
                        kinetic_energy_tensor_summed_xz=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#xz
                        kinetic_energy_tensor_summed_yz=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#yz
                        
                        # calculating Stress tensor 
                        stress_tensor_summed_xx=delta_mom_pos_tensor_summed_xx + kinetic_energy_tensor_summed_xx
                        stress_tensor_summed_yy=delta_mom_pos_tensor_summed_yy + kinetic_energy_tensor_summed_yy#yy
                        stress_tensor_summed_zz=delta_mom_pos_tensor_summed_zz + kinetic_energy_tensor_summed_zz#zz
                        
                        
                        stress_tensor_summed_xz=delta_mom_pos_tensor_summed_xz + kinetic_energy_tensor_summed_xz + (erate[data_set-1]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_zz#xz
                        stress_tensor_summed_xy=delta_mom_pos_tensor_summed_xy + kinetic_energy_tensor_summed_xy #xy 
                        stress_tensor_summed_yz=delta_mom_pos_tensor_summed_yz + kinetic_energy_tensor_summed_yz#yz
                        stress_tensor_summed_zx=delta_mom_pos_tensor_summed_zx + kinetic_energy_tensor_summed_xz + (erate[data_set-1]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_xz #zx
                        stress_tensor_summed_zy=delta_mom_pos_tensor_summed_zy+ kinetic_energy_tensor_summed_yz#zy
                        stress_tensor_summed_yx=delta_mom_pos_tensor_summed_yx + kinetic_energy_tensor_summed_xy#yx
                        
                        # making sure the sets of data are laid into the 1-D array in correct positions 
                    
                            
                        # j=j-(shape_after[0]-shape_truncated_in) # need to edit this 
                            #print("j",j)
                        start_index_6_element=terms_6[j+(count)-1]
                        print(start_index_6_element)
                
                        end_index_6_element=start_index_6_element+6
                        print(end_index_6_element)
                    
                        start_index_9_element=terms_9[j+(count)-1]
                        end_index_9_element=   start_index_9_element+9

                        np_array_ke_entries=np.array([ kinetic_energy_tensor_summed_xx, kinetic_energy_tensor_summed_yy, kinetic_energy_tensor_summed_zz, kinetic_energy_tensor_summed_xy, kinetic_energy_tensor_summed_xz, kinetic_energy_tensor_summed_yz])
                        np_array_delta_mom_pos=np.array([delta_mom_pos_tensor_summed_xx,delta_mom_pos_tensor_summed_yy,delta_mom_pos_tensor_summed_zz,delta_mom_pos_tensor_summed_xz,delta_mom_pos_tensor_summed_xy,delta_mom_pos_tensor_summed_yz,delta_mom_pos_tensor_summed_zx,delta_mom_pos_tensor_summed_zy,delta_mom_pos_tensor_summed_yx])
                        np_array_stress_tensor=np.array([stress_tensor_summed_xx,stress_tensor_summed_yy,stress_tensor_summed_zz,stress_tensor_summed_xz,stress_tensor_summed_xy,stress_tensor_summed_yz,stress_tensor_summed_zx,stress_tensor_summed_zy,stress_tensor_summed_yx])
                        
                        # inserting values into final 1-D array 
                        kinetic_energy_tensor_summed_shared[start_index_6_element:end_index_6_element]=np_array_ke_entries
                        delta_mom_pos_tensor_summed_shared[start_index_9_element:end_index_9_element]=np_array_delta_mom_pos
                        stress_tensor_summed_shared[start_index_9_element:end_index_9_element]=np_array_stress_tensor

        


            
            else:
                count=int((shape_after[0]-1-shape_truncated_in)*p)# to make sure the data is in the correct location in 1-D arrat 

                #print(count)
                for j in range(shape_after[0]-shape_truncated_in+1,shape_after[0]):
                        # the plus 
            
                        # taking the position and vel data out of hdf5 format
                        SRD_positions_initial= f_b['particles']['SRDs']['position']['value'][j-1]
                        SRD_positions_after= f_a['particles']['SRDs']['position']['value'][j]
                        SRD_velocities_initial=f_b['particles']['SRDs']['velocity']['value'][j-1]
                        SRD_velocities_after=f_a['particles']['SRDs']['velocity']['value'][j]
                        
                        # calculating the mom_pos tensor 
                        delta_mom_pos_tensor_summed_xx=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#xx
                        delta_mom_pos_tensor_summed_yy=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#yy
                        delta_mom_pos_tensor_summed_zz=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#zz
                        delta_mom_pos_tensor_summed_xz=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#xz
                        delta_mom_pos_tensor_summed_xy=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#xy
                        delta_mom_pos_tensor_summed_yz=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#yz
                        delta_mom_pos_tensor_summed_zx=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#zx
                        delta_mom_pos_tensor_summed_zy=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#zy
                        delta_mom_pos_tensor_summed_yx=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#yx
                        
                        # calculating ke tensor 
                        kinetic_energy_tensor_summed_xx=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,0],axis=0)/(box_vol)#xx
                        kinetic_energy_tensor_summed_yy=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#yy
                        kinetic_energy_tensor_summed_zz=np.sum(SRD_velocities_initial[:,2]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#zz
                        kinetic_energy_tensor_summed_xy=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#xy
                        kinetic_energy_tensor_summed_xz=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#xz
                        kinetic_energy_tensor_summed_yz=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#yz
                        
                        # calculating Stress tensor 
                        stress_tensor_summed_xx=delta_mom_pos_tensor_summed_xx + kinetic_energy_tensor_summed_xx
                        stress_tensor_summed_yy=delta_mom_pos_tensor_summed_yy + kinetic_energy_tensor_summed_yy#yy
                        stress_tensor_summed_zz=delta_mom_pos_tensor_summed_zz + kinetic_energy_tensor_summed_zz#zz
                        
                        
                        stress_tensor_summed_xz=delta_mom_pos_tensor_summed_xz + kinetic_energy_tensor_summed_xz + (erate[data_set-1]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_zz#xz
                        stress_tensor_summed_xy=delta_mom_pos_tensor_summed_xy + kinetic_energy_tensor_summed_xy #xy 
                        stress_tensor_summed_yz=delta_mom_pos_tensor_summed_yz + kinetic_energy_tensor_summed_yz#yz
                        stress_tensor_summed_zx=delta_mom_pos_tensor_summed_zx + kinetic_energy_tensor_summed_xz + (erate[data_set-1]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_xz #zx
                        stress_tensor_summed_zy=delta_mom_pos_tensor_summed_zy+ kinetic_energy_tensor_summed_yz#zy
                        stress_tensor_summed_yx=delta_mom_pos_tensor_summed_yx + kinetic_energy_tensor_summed_xy#yx
                            
                            
                        j=j-(shape_after[0]-shape_truncated_in)
                        start_index_6_element=terms_6[j+(count)-1]
                        #print(start_index_6_element)
            
                        end_index_6_element=start_index_6_element+6
                        #print(end_index_6_element)
                
                        start_index_9_element=terms_9[j+(count)-1]
                        end_index_9_element=   start_index_9_element+9
                            
                        
                        
                        # temp variables to hold result of loop j 
                        np_array_ke_entries=np.array([ kinetic_energy_tensor_summed_xx, kinetic_energy_tensor_summed_yy, kinetic_energy_tensor_summed_zz, kinetic_energy_tensor_summed_xy, kinetic_energy_tensor_summed_xz, kinetic_energy_tensor_summed_yz])
                        np_array_delta_mom_pos=np.array([delta_mom_pos_tensor_summed_xx,delta_mom_pos_tensor_summed_yy,delta_mom_pos_tensor_summed_zz,delta_mom_pos_tensor_summed_xz,delta_mom_pos_tensor_summed_xy,delta_mom_pos_tensor_summed_yz,delta_mom_pos_tensor_summed_zx,delta_mom_pos_tensor_summed_zy,delta_mom_pos_tensor_summed_yx])
                        np_array_stress_tensor=np.array([stress_tensor_summed_xx,stress_tensor_summed_yy,stress_tensor_summed_zz,stress_tensor_summed_xz,stress_tensor_summed_xy,stress_tensor_summed_yz,stress_tensor_summed_zx,stress_tensor_summed_zy,stress_tensor_summed_yx])
                        
                        # inserting values into final 1-D array 
                        kinetic_energy_tensor_summed_shared[start_index_6_element:end_index_6_element]=np_array_ke_entries
                        delta_mom_pos_tensor_summed_shared[start_index_9_element:end_index_9_element]=np_array_delta_mom_pos
                        stress_tensor_summed_shared[start_index_9_element:end_index_9_element]=np_array_stress_tensor


        f_b.close()
    f_a.close()
           

                

    return stress_tensor_summed_shared,kinetic_energy_tensor_summed_shared,delta_mom_pos_tensor_summed_shared


import multiprocessing as mp
from multiprocessing import Process, Queue, Array ,Lock
import multiprocessing.managers as mpm
import time
import ctypes




if shape_truncated_in==0:

# determine 1-D array size 
    size_delta_mom_pos_tensor_summed=int(no_data_sets*j_*((shape_after[0])-1)*9)
    print(size_delta_mom_pos_tensor_summed)
    size_stress_tensor_summed=int(no_data_sets*j_*((shape_after[0])-1)*9) 
    print(size_stress_tensor_summed)
    size_kinetic_energy_tensor_summed=int(no_data_sets*j_*((shape_after[0])-1))*6
    print(size_kinetic_energy_tensor_summed)
    processes=[]
else:
    size_delta_mom_pos_tensor_summed=int(no_data_sets*j_*((shape_after[0]*fraction_of_steps)-1)*9)
    print(size_delta_mom_pos_tensor_summed)
    size_stress_tensor_summed=int(no_data_sets*j_*((shape_after[0]*fraction_of_steps)-1)*9) 
    print(size_stress_tensor_summed)
    size_kinetic_energy_tensor_summed=int(no_data_sets*j_*((shape_after[0]*fraction_of_steps)-1))*6
    print(size_kinetic_energy_tensor_summed)
    processes=[]


tic=time.perf_counter()

if __name__ =='__main__':
        # need to change this code so it only does 5 arrays at a time 
        
        # creating shared memory arrays that can be updated by multiple processes
        delta_mom_pos_tensor_summed_shared=  mp.Array('d',range(size_delta_mom_pos_tensor_summed))
        stress_tensor_summed_shared=   mp.Array('d',range(size_stress_tensor_summed))
        kinetic_energy_tensor_summed_shared=   mp.Array('d',range(size_kinetic_energy_tensor_summed))
        list_done=[]
        
            
        #creating processes, iterating over each realisation name 
        for p in  range(count_h5_after):
              
            proc= Process(target=stress_tensor_total_compute_shear,args=(shape_truncated_in,terms_9,terms_6,box_vol,realisation_index,delta_mom_pos_tensor_summed_shared,stress_tensor_summed_shared,kinetic_energy_tensor_summed_shared,realisation_name_h5_after[p],realisation_name_h5_before[p],shape_after,j_,no_data_sets,erate,delta_t_srd,p,))
                                        
            processes.append(proc)
            proc.start()
        
        for proc in  processes:
             proc.join()
             print(proc)
             
        toc=time.perf_counter()
        print("Parallel analysis done in ", (toc-tic)/60," mins")
       
        # could possibly make this code save the unshaped arrays to save the problem 
        np.save("stress_tensor_summed_1d_test_M_"+str(rho)+"_L_"+str(box_size),stress_tensor_summed_shared)
        np.save("delta_mom_pos_tensor_summed_1d_test_M_"+str(rho)+"_L_"+str(box_size),delta_mom_pos_tensor_summed_shared)
        np.save("kinetic_energy_tensor_summed_1d_test_M_"+str(rho)+"_L_"+str(box_size),kinetic_energy_tensor_summed_shared)



        

# could also use MPI4PY     