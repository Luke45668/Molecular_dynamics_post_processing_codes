# ##!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# This script will calculate the MPCD stress tensor for a pure fluid under forward NEMD using hdf5 files 
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
no_SRD=58320
box_size=18
# no_SRD=2160
# box_size=6
# no_SRD=270
# box_size=3
# no_SRD=2560
# box_size=8
#nu_bar=3
#delta_t_srd=0.014872025172594354
#nu_bar=0.9 
delta_t_srd=0.05674857690605889

box_vol=box_size**3
erate= np.array([0.001,0.002,0.003])
no_timesteps=50000
# estimating number of steps  required
strain=3
delta_t_md=delta_t_srd/10
strain_rate= np.array([0.001,0.002,0.003])
number_steps_needed= np.ceil(strain/(strain_rate*delta_t_md))
dump_freq=10
rho=10 
#rho=5
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
filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/hfd5_runs/non_equilibrium_tests/2dumps/test_non_eq_box_"+str(int(box_size))+"_M_"+str(rho)
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


#should check the time series of velocity profiles aswell. 


# # find dump file size
with h5.File(realisation_name_h5_after[0], 'r') as f:
    shape_after= f['particles']['SRDs']['position']['value'].shape
    f.close()
print(shape_after)

# #
with h5.File(realisation_name_h5_before[0], 'r') as f_i:
      first_step= f_i['particles']['SRDs']['position']['step'][0]
      f.close()

#     # print(first_step)
#     print(f_i['particles'].keys())

    
# # this needs to be changed back to the old version where we looked at file N and N-1, since the shear could  change things in the collision step
# #%%
# # need to look into adding multi-processing to this section of the code
j_=3
no_data_sets=erate.shape[0]
def producing_plus_6_sequence_for_ke():
    n=np.arange(1,44992) # if steps change need to chnage this number 
    terms_6=[0]
    for i in n:
            terms_6.append(terms_6[i-1]+6)
       
    return terms_6 
    
def producing_plus_9_sequence_for_ke(): 
    n=np.arange(1,44992)
    terms_9=[0]
    for i in n:
            terms_9.append(terms_9[i-1]+9)

    return terms_9
        
terms_9=producing_plus_9_sequence_for_ke()
terms_6=producing_plus_6_sequence_for_ke()

##sorting realisation lists
realisation_name_h5_after_sorted=['0']*9
realisation_name_h5_before_sorted=['0']*9
for i in range(9):
     realisation_index_=int(np.where(realisation_index==float(realisation_name_h5_after[i].split('_')[9]))[0][0])

    
     data_set = int(np.where(erate==float(realisation_name_h5_after[i].split('_')[15]))[0][0])
   
     if data_set==0:
        realisation_name_h5_after_sorted[realisation_index_]=realisation_name_h5_after[i]
     elif data_set==1:
        realisation_name_h5_after_sorted[3+realisation_index_]=realisation_name_h5_after[i]
     else:
        realisation_name_h5_after_sorted[6+realisation_index_]=realisation_name_h5_after[i]
for i in range(9):
     realisation_index_=int(np.where(realisation_index==float(realisation_name_h5_before[i].split('_')[9]))[0][0])
     
     
     data_set = int(np.where(erate==float(realisation_name_h5_before[i].split('_')[15]))[0][0])

     if data_set==0:
        realisation_name_h5_before_sorted[realisation_index_]=realisation_name_h5_before[i]
     elif data_set==1:
        realisation_name_h5_before_sorted[3+realisation_index_]=realisation_name_h5_before[i]
     else:
        realisation_name_h5_before_sorted[6+realisation_index_]=realisation_name_h5_before[i]
          
realisation_name_h5_before=realisation_name_h5_before_sorted
realisation_name_h5_after=realisation_name_h5_after_sorted   






 
def stress_tensor_total_compute_shear(terms_9,terms_6,box_vol,realisation_index,delta_mom_pos_tensor_summed_shared,stress_tensor_summed_shared,kinetic_energy_tensor_summed_shared,realisation_name_h5_after,realisation_name_h5_before,shape_after,j_,no_data_sets,erate,delta_t_srd,p):    
    import h5py as h5 

    with h5.File(realisation_name_h5_after, 'r') as f_a:
        with h5.File(realisation_name_h5_before, 'r') as f_b:
            data_set = int(np.where(erate==float(realisation_name_h5_after.split('_')[15]))[0][0])
            k=int(np.where(realisation_index==float(realisation_name_h5_after.split('_')[9]))[0][0])
            count=4999*p
            for j in range(1,shape_after[0]):
                    
        
                        
                    SRD_positions_initial= f_b['particles']['SRDs']['position']['value'][j-1]
                    
                    SRD_positions_after= f_a['particles']['SRDs']['position']['value'][j]
                    #print(SRD_positions_after)

                    SRD_velocities_initial=f_b['particles']['SRDs']['velocity']['value'][j-1]
                    SRD_velocities_after=f_a['particles']['SRDs']['velocity']['value'][j]
                    #print(SRD_velocities_after)
                    # delta_mom=SRD_velocities_after-SRD_velocities_initial
                    # delta_mom_summed[data_set,k,j,:]=np.sum(SRD_velocities_after-SRD_velocities_initial,axis=0)

                    # need to add kinetic contribution

                        
                    delta_mom_pos_tensor_summed_xx=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#xx
                    delta_mom_pos_tensor_summed_yy=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#yy
                    delta_mom_pos_tensor_summed_zz=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#zz
                    delta_mom_pos_tensor_summed_xz=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#xz
                    delta_mom_pos_tensor_summed_xy=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#xy
                    delta_mom_pos_tensor_summed_yz=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#yz
                    delta_mom_pos_tensor_summed_zx=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#zx
                    delta_mom_pos_tensor_summed_zy=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#zy
                    delta_mom_pos_tensor_summed_yx=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#yx

                    kinetic_energy_tensor_summed_xx=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,0],axis=0)/(box_vol)#xx
                    kinetic_energy_tensor_summed_yy=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#yy
                    kinetic_energy_tensor_summed_zz=np.sum(SRD_velocities_initial[:,2]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#zz
                    kinetic_energy_tensor_summed_xy=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#xy
                    kinetic_energy_tensor_summed_xz=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#xz
                    kinetic_energy_tensor_summed_yz=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#yz
                    
                    stress_tensor_summed_xx=delta_mom_pos_tensor_summed_xx + kinetic_energy_tensor_summed_xx
                    stress_tensor_summed_yy=delta_mom_pos_tensor_summed_yy + kinetic_energy_tensor_summed_yy#yy
                    stress_tensor_summed_zz=delta_mom_pos_tensor_summed_zz + kinetic_energy_tensor_summed_zz#zz
                    
                    
                    stress_tensor_summed_xz=delta_mom_pos_tensor_summed_xz + kinetic_energy_tensor_summed_xz + (erate[data_set-1]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_zz#xz
                    stress_tensor_summed_xy=delta_mom_pos_tensor_summed_xy + kinetic_energy_tensor_summed_xy #xy 
                    stress_tensor_summed_yz=delta_mom_pos_tensor_summed_yz + kinetic_energy_tensor_summed_yz#yz
                    stress_tensor_summed_zx=delta_mom_pos_tensor_summed_zx + kinetic_energy_tensor_summed_xz + (erate[data_set-1]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_xz #zx
                    stress_tensor_summed_zy=delta_mom_pos_tensor_summed_zy+ kinetic_energy_tensor_summed_yz#zy
                    stress_tensor_summed_yx=delta_mom_pos_tensor_summed_yx + kinetic_energy_tensor_summed_xy#yx

                    start_index_6_element=terms_6[j+(count)-1]
                    #print("start_index_6_element",start_index_6_element)
                    end_index_6_element=start_index_6_element+6
                    #print("end_index_6_element",end_index_6_element)

                    start_index_9_element=terms_9[j+(count)-1]
                    end_index_9_element=   start_index_9_element+9
                    #print("start_index_9_element",start_index_9_element)
                    #print("end_index_9_element",end_index_9_element)
                    
                    

                    np_array_ke_entries=np.array([ kinetic_energy_tensor_summed_xx, kinetic_energy_tensor_summed_yy, kinetic_energy_tensor_summed_zz, kinetic_energy_tensor_summed_xy, kinetic_energy_tensor_summed_xz, kinetic_energy_tensor_summed_yz])
                    np_array_delta_mom_pos=np.array([delta_mom_pos_tensor_summed_xx,delta_mom_pos_tensor_summed_yy,delta_mom_pos_tensor_summed_zz,delta_mom_pos_tensor_summed_xz,delta_mom_pos_tensor_summed_xy,delta_mom_pos_tensor_summed_yz,delta_mom_pos_tensor_summed_zx,delta_mom_pos_tensor_summed_zy,delta_mom_pos_tensor_summed_yx])
                    np_array_stress_tensor=np.array([stress_tensor_summed_xx,stress_tensor_summed_yy,stress_tensor_summed_zz,stress_tensor_summed_xz,stress_tensor_summed_xy,stress_tensor_summed_yz,stress_tensor_summed_zx,stress_tensor_summed_zy,stress_tensor_summed_yx])
                    
                    kinetic_energy_tensor_summed_shared[start_index_6_element:end_index_6_element]=np_array_ke_entries
                    delta_mom_pos_tensor_summed_shared[start_index_9_element:end_index_9_element]=np_array_delta_mom_pos
                    stress_tensor_summed_shared[start_index_9_element:end_index_9_element]=np_array_stress_tensor
           

                

    return stress_tensor_summed_shared,kinetic_energy_tensor_summed_shared,delta_mom_pos_tensor_summed_shared
#from post_MPCD_MP_processing_module import *
import multiprocessing as mp
from multiprocessing import Process, Queue, Array ,Lock
import multiprocessing.managers as mpm
import time
import ctypes
# count=[1,2,3,4]
# count2=[1,2,3,4]

#p1=Process(target=stress_tensor_total_compute_shear,args=[realisation_name_h5_after[0],realisation_name_h5_before[0],shape_after,j_,no_data_sets,erate,delta_t_srd])
# p2=Process(target=stress_tensor_total_compute_shear,args=[realisation_name_h5_after[1],realisation_name_h5_before[1],shape_after,j_,no_data_sets,erate,delta_t_srd])
# p3=Process(target=stress_tensor_total_compute_shear,args=[realisation_name_h5_after[2],realisation_name_h5_before[2],shape_after,j_,no_data_sets,erate,delta_t_srd])
# p4=Process(target=stress_tensor_total_compute_shear,args=[realisation_name_h5_after[3],realisation_name_h5_before[3],shape_after,j_,no_data_sets,erate,delta_t_srd])
# p5=Process(target=stress_tensor_total_compute_shear,args=[realisation_name_h5_after[4],realisation_name_h5_before[4],shape_after,j_,no_data_sets,erate,delta_t_srd])
# p6=Process(target=stress_tensor_total_compute_shear,args=[realisation_name_h5_after[5],realisation_name_h5_before[5],shape_after,j_,no_data_sets,erate,delta_t_srd])
# p7=Process(target=stress_tensor_total_compute_shear,args=[realisation_name_h5_after[6],realisation_name_h5_before[6],shape_after,j_,no_data_sets,erate,delta_t_srd])
# p8=Process(target=stress_tensor_total_compute_shear,args=[realisation_name_h5_after[7],realisation_name_h5_before[7],shape_after,j_,no_data_sets,erate,delta_t_srd])
# p9=Process(target=stress_tensor_total_compute_shear,args=[realisation_name_h5_after[8],realisation_name_h5_before[8],shape_after,j_,no_data_sets,erate,delta_t_srd])


stress_tensor_total_compute_shear_tuple=()
realisation_count_to_do= count_h5_after
processes=[]


size_delta_mom_summed=int(no_data_sets*j_*(shape_after[0]-1)*3)
size_delta_mom_pos_tensor_summed=int(no_data_sets*j_*(shape_after[0]-1)*9)
size_stress_tensor_summed=int(no_data_sets*j_*(shape_after[0]-1)*9)
size_kinetic_energy_tensor_summed=int(no_data_sets*j_*(shape_after[0]-1)*6)

tic=time.perf_counter()
if __name__ =='__main__':
        
        delta_mom_summed_shared= mp.Array('d',range(size_delta_mom_summed))
        delta_mom_pos_tensor_summed_shared=  mp.Array('d',range(size_delta_mom_pos_tensor_summed))
        stress_tensor_summed_shared=   mp.Array('d',range(size_stress_tensor_summed))
        kinetic_energy_tensor_summed_shared=   mp.Array('d',range(size_kinetic_energy_tensor_summed))
       
        # queue= Queue()
        for p in  range(count_h5_after):
            

        
            # not sure about this yet 
            proc= Process(target=stress_tensor_total_compute_shear,args=(terms_9,terms_6,box_vol,realisation_index,delta_mom_pos_tensor_summed_shared,stress_tensor_summed_shared,kinetic_energy_tensor_summed_shared,realisation_name_h5_after[p],realisation_name_h5_before[p],shape_after,j_,no_data_sets,erate,delta_t_srd,p,))
                                        
            processes.append(proc)
            proc.start()
        
        for proc in  processes:
             proc.join()
             print(proc)
             
        toc=time.perf_counter()
        print("done in ", (toc-tic)/60)
       # print(stress_tensor_summed_shared[:])
       
        #np.save("stress_tensor_unshaped",stress_tensor_summed_shared)

        stress_tensor_summed=np.reshape(stress_tensor_summed_shared,((3,3,4999,9)))
        kinetic_energy_tensor_summed=np.reshape(kinetic_energy_tensor_summed_shared,((3,3,4999,6)))
        delta_mom_pos_tensor_summed=np.reshape(delta_mom_pos_tensor_summed_shared,((3,3,4999,9)))

        print(stress_tensor_summed.shape)
        print(kinetic_energy_tensor_summed.shape)
        print(delta_mom_pos_tensor_summed.shape)

        np.save("stress_tensor_summed_test",stress_tensor_summed)
        np.save("delta_mom_pos_tensor_summed_test",delta_mom_pos_tensor_summed)
        np.save("kinetic_energy_tensor_summed_test",kinetic_energy_tensor_summed)

        

        


# standard loop took 26 mins approx 

# need to write this loop so it only uses the right amount of cores, eg. six at a time, until its analysed all the files 
   
# need to organise tuple output 
