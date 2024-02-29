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
import multiprocessing as mp
from multiprocessing import Process
import time
import h5py as h5

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

no_timesteps=1000000
# estimating number of steps  required
strain=3
delta_t_md=delta_t_srd/10
strain_rate= np.array([0.01,0.001,0.0001])
number_steps_needed= np.ceil(strain/(strain_rate*delta_t_md))
dump_freq=10

#rho=5
realisation_index=np.array([1,2,3])
spring_stiffness =np.array([20,40,80,100])
bending_stiffness=5000
# finding all the dump files in a folder

VP_general_name_string='vel.*'

Mom_general_name_string='mom.*'

log_general_name_string='log.*'
                         
TP_general_name_string='temp.*'

dump_general_name_string_after_srd='*BK_'+str(bending_stiffness)+'*SRDs*after*.h5'
dump_general_name_string_before_srd='*BK_'+str(bending_stiffness)+'*SRDs*before*.h5'
dump_general_name_string_after_pol='*BK_'+str(bending_stiffness)+'*pol*after*.h5'
dump_general_name_string_before_pol='*BK_'+str(bending_stiffness)+'*pol*before*.h5'
dump_general_name_string_after_phantom='*BK_'+str(bending_stiffness)+'*phantom*after*.h5'
dump_general_name_string_before_phantom='*BK_'+str(bending_stiffness)+'*phantom*before*.h5'


filepath="/home/ucahlrl/Scratch/output"
#filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/hfd5_runs/flat_elastic_tests/test_box_23_M_5_k_10_30"

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
with h5.File(realisation_name_h5_after_srd[0], 'r') as f:
    shape_after= f['particles']['SRDs']['position']['value'].shape
    f.close()
print(shape_after[0])

with h5.File(realisation_name_h5_before_srd[0], 'r') as f_i:
      first_step= f_i['particles']['SRDs']['position']['step'][0]
      first_posn= f_i['particles']['SRDs']['position']['value'][0]
      f.close()

j_=3
no_data_sets=spring_stiffness.shape[0]


# creating indices for mapping 3-D data to 1-D sequence
def producing_plus_N_sequence_for_ke(increment,n_terms):
    n=np.arange(1,n_terms) # if steps change need to chnage this number 
    terms_n=[0]
    for i in n:
            terms_n.append(terms_n[i-1]+int(increment))
       
    return terms_n 

n_terms=j_*no_data_sets*(shape_after[0]-1)
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

# need to make this into a formula 
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 


def stress_tensor_flat_elastic_total_compute_shear_full_run(mass_pol,eq_spring_length,spring_stiffness,terms_9,box_vol,realisation_index,stress_tensor_summed_shared,realisation_name_h5_after_srd,realisation_name_h5_before_srd,realisation_name_h5_after_pol,realisation_name_h5_before_pol,realisation_name_h5_after_phantom,shape_after,erate,delta_t_srd,p):    
    
    # make sure to put the realisation name arguments with list index [p]
    with h5.File(realisation_name_h5_after_srd, 'r') as f_a:
        with h5.File(realisation_name_h5_before_srd, 'r') as f_b:
            with h5.File(realisation_name_h5_after_pol,'r') as f_c:
                with h5.File(realisation_name_h5_before_pol,'r') as f_d:
                      with h5.File(realisation_name_h5_after_phantom,'r') as f_ph:


                            # need to change this to looking for spring constant 
                            # erate
                            data_set = int(np.where(erate==float(realisation_name_h5_after_srd.split('_')[16]))[0][0])
                            # spring constant 
                            K=int(np.where(spring_stiffness==float(realisation_name_h5_after_srd.split('_')[20]))[0][0])
                        
                        
                            
                            count=int((shape_after[0]-1)*p) # to make sure the data is in the correct location in 1-D array 
                            
                            # looping through N-1 dumps, since we are using a delta 
                            for j in range(1,shape_after[0]):
                                
                        
                                    
                                    SRD_positions_after= f_a['particles']['SRDs']['position']['value'][j]
                                    SRD_velocities_initial=f_b['particles']['SRDs']['velocity']['value'][j-1]
                                    SRD_velocities_after=f_a['particles']['SRDs']['velocity']['value'][j]
                                    # have to use small for hirotori edit , not pol
                                    pol_positions_after= f_c['particles']['small']['position']['value'][j]
                                    pol_velocities_initial=f_d['particles']['small']['velocity']['value'][j-1]
                                    pol_velocities_after=f_c['particles']['small']['velocity']['value'][j]

                                    phantom_positions_after=f_ph['particles']['phantom']['position']['value'][j]
                                  
                                    # calculating the mom_pos tensor srds
                                    delta_mom_pos_tensor_summed_xx_srd=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#xx
                                    delta_mom_pos_tensor_summed_yy_srd=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#yy
                                    delta_mom_pos_tensor_summed_zz_srd=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#zz
                                    delta_mom_pos_tensor_summed_xz_srd=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#xz
                                    delta_mom_pos_tensor_summed_xy_srd=np.sum((SRD_velocities_after[:,0]- SRD_velocities_initial[:,0])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#xy
                                    delta_mom_pos_tensor_summed_yz_srd=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,2],axis=0)/(box_vol*delta_t_srd)#yz
                                    delta_mom_pos_tensor_summed_zx_srd=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#zx
                                    delta_mom_pos_tensor_summed_zy_srd=np.sum((SRD_velocities_after[:,2]- SRD_velocities_initial[:,2])*SRD_positions_after[:,1],axis=0)/(box_vol*delta_t_srd)#zy
                                    delta_mom_pos_tensor_summed_yx_srd=np.sum((SRD_velocities_after[:,1]- SRD_velocities_initial[:,1])*SRD_positions_after[:,0],axis=0)/(box_vol*delta_t_srd)#yx
                                    
                                    # calculating the mom_pos tensor pol
                                    delta_mom_pos_tensor_summed_xx_pol=np.sum((pol_velocities_after[:,0]- pol_velocities_initial[:,0])*pol_positions_after[:,0],axis=0)*mass_pol/(box_vol*delta_t_srd)#xx
                                    delta_mom_pos_tensor_summed_yy_pol=np.sum((pol_velocities_after[:,1]- pol_velocities_initial[:,1])*pol_positions_after[:,1],axis=0)*mass_pol/(box_vol*delta_t_srd)#yy
                                    delta_mom_pos_tensor_summed_zz_pol=np.sum((pol_velocities_after[:,2]- pol_velocities_initial[:,2])*pol_positions_after[:,2],axis=0)*mass_pol/(box_vol*delta_t_srd)#zz
                                    delta_mom_pos_tensor_summed_xz_pol=np.sum((pol_velocities_after[:,0]- pol_velocities_initial[:,0])*pol_positions_after[:,2],axis=0)*mass_pol/(box_vol*delta_t_srd)#xz
                                    delta_mom_pos_tensor_summed_xy_pol=np.sum((pol_velocities_after[:,0]- pol_velocities_initial[:,0])*pol_positions_after[:,1],axis=0)*mass_pol/(box_vol*delta_t_srd)#xy
                                    delta_mom_pos_tensor_summed_yz_pol=np.sum((pol_velocities_after[:,1]- pol_velocities_initial[:,1])*pol_positions_after[:,2],axis=0)*mass_pol/(box_vol*delta_t_srd)#yz
                                    delta_mom_pos_tensor_summed_zx_pol=np.sum((pol_velocities_after[:,2]- pol_velocities_initial[:,2])*pol_positions_after[:,0],axis=0)*mass_pol/(box_vol*delta_t_srd)#zx
                                    delta_mom_pos_tensor_summed_zy_pol=np.sum((pol_velocities_after[:,2]- pol_velocities_initial[:,2])*pol_positions_after[:,1],axis=0)*mass_pol/(box_vol*delta_t_srd)#zy
                                    delta_mom_pos_tensor_summed_yx_pol=np.sum((pol_velocities_after[:,1]- pol_velocities_initial[:,1])*pol_positions_after[:,0],axis=0)*mass_pol/(box_vol*delta_t_srd)#yx
                                    
                                


                                    
                                    # calculating ke tensor srds
                                    kinetic_energy_tensor_summed_xx_srd=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,0],axis=0)/(box_vol)#xx
                                    kinetic_energy_tensor_summed_yy_srd=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#yy
                                    kinetic_energy_tensor_summed_zz_srd=np.sum(SRD_velocities_initial[:,2]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#zz
                                    kinetic_energy_tensor_summed_xy_srd=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,1],axis=0)/(box_vol)#xy
                                    kinetic_energy_tensor_summed_xz_srd=np.sum(SRD_velocities_initial[:,0]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#xz
                                    kinetic_energy_tensor_summed_yz_srd=np.sum(SRD_velocities_initial[:,1]*SRD_velocities_initial[:,2],axis=0)/(box_vol)#yz

                                    # calculating ke tensor pol
                                    kinetic_energy_tensor_summed_xx_pol=np.sum(pol_velocities_initial[:,0]*pol_velocities_initial[:,0],axis=0)*mass_pol/(box_vol)#xx
                                    kinetic_energy_tensor_summed_yy_pol=np.sum(pol_velocities_initial[:,1]*pol_velocities_initial[:,1],axis=0)*mass_pol/(box_vol)#yy
                                    kinetic_energy_tensor_summed_zz_pol=np.sum(pol_velocities_initial[:,2]*pol_velocities_initial[:,2],axis=0)*mass_pol/(box_vol)#zz
                                    kinetic_energy_tensor_summed_xy_pol=np.sum(pol_velocities_initial[:,0]*pol_velocities_initial[:,1],axis=0)*mass_pol/(box_vol)#xy
                                    kinetic_energy_tensor_summed_xz_pol=np.sum(pol_velocities_initial[:,0]*pol_velocities_initial[:,2],axis=0)*mass_pol/(box_vol)#xz
                                    kinetic_energy_tensor_summed_yz_pol=np.sum(pol_velocities_initial[:,1]*pol_velocities_initial[:,2],axis=0)*mass_pol/(box_vol)#yz
                                    
                                
                                    #calculating spring position tensor 
                                    # start here 
                                     # spring 1
                                    f_spring_1_dirn=pol_positions_after[0,:]-phantom_positions_after[1,:]
                                    f_spring_1_mag=np.sqrt(np.sum((f_spring_1_dirn)**2))
                                    f_spring_1=K*(f_spring_1_dirn/f_spring_1_mag)*(f_spring_1_mag-eq_spring_length)
                                    # spring 2
                                    f_spring_2_dirn=pol_positions_after[1,:]-phantom_positions_after[2,:]
                                    f_spring_2_mag=np.sqrt(np.sum((f_spring_2_dirn)**2))
                                    f_spring_2=K*(f_spring_2_dirn/f_spring_2_mag)*(f_spring_2_mag-eq_spring_length)
                                    # spring 3
                                    f_spring_3_dirn=pol_positions_after[2,:]-phantom_positions_after[0,:]
                                    f_spring_3_mag=np.sqrt(np.sum((f_spring_3_dirn)**2))
                                    f_spring_3=K*(f_spring_3_dirn/f_spring_3_mag)*(f_spring_3_mag-eq_spring_length)


                                    # force position tensor 

                                    spring_force_positon_tensor_xx=f_spring_1[0]*f_spring_1_dirn[0] + f_spring_2[0]*f_spring_2_dirn[0] +f_spring_3[0]*f_spring_3_dirn[0] 
                                    spring_force_positon_tensor_yy=f_spring_1[1]*f_spring_1_dirn[1] + f_spring_2[1]*f_spring_2_dirn[1] +f_spring_3[1]*f_spring_3_dirn[1] 
                                    spring_force_positon_tensor_zz=f_spring_1[2]*f_spring_1_dirn[2] + f_spring_2[2]*f_spring_2_dirn[2] +f_spring_3[2]*f_spring_3_dirn[2] 
                                    spring_force_positon_tensor_xz=f_spring_1[0]*f_spring_1_dirn[2] + f_spring_2[0]*f_spring_2_dirn[2] +f_spring_3[0]*f_spring_3_dirn[2] 
                                    spring_force_positon_tensor_xy=f_spring_1[0]*f_spring_1_dirn[1] + f_spring_2[0]*f_spring_2_dirn[1] +f_spring_3[0]*f_spring_3_dirn[1] 
                                    spring_force_positon_tensor_yz=f_spring_1[1]*f_spring_1_dirn[2] + f_spring_2[1]*f_spring_2_dirn[2] +f_spring_3[1]*f_spring_3_dirn[2] 
                                    spring_force_positon_tensor_zx=f_spring_1[2]*f_spring_1_dirn[0] + f_spring_2[2]*f_spring_2_dirn[0] +f_spring_3[2]*f_spring_3_dirn[0] 
                                    spring_force_positon_tensor_zy=f_spring_1[2]*f_spring_1_dirn[1] + f_spring_2[2]*f_spring_2_dirn[1] +f_spring_3[2]*f_spring_3_dirn[1] 
                                    spring_force_positon_tensor_yx=f_spring_1[1]*f_spring_1_dirn[0] + f_spring_2[1]*f_spring_2_dirn[0] +f_spring_3[1]*f_spring_3_dirn[0] 
                                    

                                    # calculating Stress tensor 
                                    stress_tensor_summed_xx=spring_force_positon_tensor_xx+delta_mom_pos_tensor_summed_xx_srd + kinetic_energy_tensor_summed_xx_srd+ delta_mom_pos_tensor_summed_xx_pol + kinetic_energy_tensor_summed_xx_pol #xx
                                    stress_tensor_summed_yy=spring_force_positon_tensor_yy+delta_mom_pos_tensor_summed_yy_srd + kinetic_energy_tensor_summed_yy_srd + delta_mom_pos_tensor_summed_yy_pol + kinetic_energy_tensor_summed_yy_pol #yy
                                    stress_tensor_summed_zz=spring_force_positon_tensor_zz+delta_mom_pos_tensor_summed_zz_srd + kinetic_energy_tensor_summed_zz_srd + delta_mom_pos_tensor_summed_zz_pol + kinetic_energy_tensor_summed_zz_pol#zz
                                    
                                    
                                    stress_tensor_summed_xz=spring_force_positon_tensor_xz+delta_mom_pos_tensor_summed_xz_srd + kinetic_energy_tensor_summed_xz_srd + (erate[data_set]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_zz_srd +    delta_mom_pos_tensor_summed_xz_pol + kinetic_energy_tensor_summed_xz_pol + (erate[data_set]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_zz_pol#xz
                                    stress_tensor_summed_xy=spring_force_positon_tensor_xy+delta_mom_pos_tensor_summed_xy_srd + kinetic_energy_tensor_summed_xy_srd+delta_mom_pos_tensor_summed_xy_pol + kinetic_energy_tensor_summed_xy_pol #xy 
                                    stress_tensor_summed_yz=spring_force_positon_tensor_yz+delta_mom_pos_tensor_summed_yz_srd + kinetic_energy_tensor_summed_yz_srd+ delta_mom_pos_tensor_summed_yz_pol + kinetic_energy_tensor_summed_yz_pol #yz
                                    stress_tensor_summed_zx=spring_force_positon_tensor_zx+delta_mom_pos_tensor_summed_zx_srd + kinetic_energy_tensor_summed_xz_srd + (erate[data_set]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_xz_srd + delta_mom_pos_tensor_summed_zx_pol + kinetic_energy_tensor_summed_xz_pol + (erate[data_set]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_zz_pol #zx
                                    stress_tensor_summed_zy=spring_force_positon_tensor_zy+delta_mom_pos_tensor_summed_zy_srd+ kinetic_energy_tensor_summed_yz_srd + delta_mom_pos_tensor_summed_zy_pol+ kinetic_energy_tensor_summed_yz_pol#zy
                                    stress_tensor_summed_yx=spring_force_positon_tensor_yx+delta_mom_pos_tensor_summed_yx_srd + kinetic_energy_tensor_summed_xy_srd+ delta_mom_pos_tensor_summed_yx_pol + kinetic_energy_tensor_summed_xy_pol#yx
                                    
                                    # making sure the sets of data are laid into the 1-D array in correct positions
                                    # for symmetric tensor 
                                    # start_index_6_element=terms_6[j+(count)-1]
                                    # end_index_6_element=start_index_6_element+6

                                    # for non-symmetric tensor 
                                    start_index_9_element=terms_9[j+(count)-1]
                                    end_index_9_element=   start_index_9_element+9

                                    #np_array_ke_entries=np.array([ kinetic_energy_tensor_summed_xx, kinetic_energy_tensor_summed_yy, kinetic_energy_tensor_summed_zz, kinetic_energy_tensor_summed_xy, kinetic_energy_tensor_summed_xz, kinetic_energy_tensor_summed_yz])
                                    #np_array_delta_mom_pos=np.array([delta_mom_pos_tensor_summed_xx,delta_mom_pos_tensor_summed_yy,delta_mom_pos_tensor_summed_zz,delta_mom_pos_tensor_summed_xz,delta_mom_pos_tensor_summed_xy,delta_mom_pos_tensor_summed_yz,delta_mom_pos_tensor_summed_zx,delta_mom_pos_tensor_summed_zy,delta_mom_pos_tensor_summed_yx])
                                    np_array_stress_tensor=np.array([stress_tensor_summed_xx,stress_tensor_summed_yy,stress_tensor_summed_zz,stress_tensor_summed_xz,stress_tensor_summed_xy,stress_tensor_summed_yz,stress_tensor_summed_zx,stress_tensor_summed_zy,stress_tensor_summed_yx])
                                    
                                    # inserting values into final 1-D array 
                                    #kinetic_energy_tensor_summed_shared[start_index_6_element:end_index_6_element]=np_array_ke_entries
                                    #delta_mom_pos_tensor_summed_shared[start_index_9_element:end_index_9_element]=np_array_delta_mom_pos
                                    stress_tensor_summed_shared[start_index_9_element:end_index_9_element]=np_array_stress_tensor


    return stress_tensor_summed_shared


# determine 1-D array size 
# size_delta_mom_pos_tensor_summed=int(no_data_sets*j_*((shape_after[0])-1)*9)
# print(size_delta_mom_pos_tensor_summed)
size_stress_tensor_summed=int(no_data_sets*j_*((shape_after[0])-1)*9) 
print(size_stress_tensor_summed)
# size_kinetic_energy_tensor_summed=int(no_data_sets*j_*((shape_after[0])-1))*6
# print(size_kinetic_energy_tensor_summed)
processes=[]
print(count_h5_after_srd)
  


tic=time.perf_counter()

if __name__ =='__main__':
        
        # need to change this code so it only does 5 arrays at a time 
        
        # creating shared memory arrays that can be updated by multiple processes
        #delta_mom_pos_tensor_summed_shared=  mp.Array('d',range(size_delta_mom_pos_tensor_summed))
        stress_tensor_summed_shared=   mp.Array('d',range(size_stress_tensor_summed))
        # kinetic_energy_tensor_summed_shared=   mp.Array('d',range(size_kinetic_energy_tensor_summed))
        list_done=[]
        
            
        #creating processes, iterating over each realisation name 
        for p in  range(count_h5_after_srd):
              
            proc= Process(target=stress_tensor_flat_elastic_total_compute_shear_full_run,args=(mass_pol,eq_spring_length,spring_stiffness,terms_9,box_vol,realisation_index,stress_tensor_summed_shared,realisation_name_h5_after_srd[p],realisation_name_h5_before_srd[p],realisation_name_h5_after_pol[p],realisation_name_h5_before_pol[p],realisation_name_h5_after_phantom[p],shape_after,erate,delta_t_srd,p,))
                                        
            processes.append(proc)
            proc.start()
        
        for proc in  processes:
             proc.join()
             print(proc)
             
        toc=time.perf_counter()
        print("Parallel analysis done in ", (toc-tic)/60," mins")
       
        # could possibly make this code save the unshaped arrays to save the problem 
        np.save("stress_tensor_summed_1d_test_erate_"+str(erate[0])+"_bk_"+str(bending_stiffness)+"_M_"+str(rho)+"_L_"+str(box_size),stress_tensor_summed_shared)
        # np.save("delta_mom_pos_tensor_summed_1d_test_M_"+str(rho)+"_L_"+str(box_size),delta_mom_pos_tensor_summed_shared)
        # np.save("kinetic_energy_tensor_summed_1d_test_M_"+str(rho)+"_L_"+str(box_size),kinetic_energy_tensor_summed_shared)



        

        

# could also use MPI4PY     