# ##!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# This script will calculate the MPCD stress tensor for a single elastic plate under forward NEMD using hdf5 files and python multiprocessing
# to ensure fast analysis  
# """
# Importing packages
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import regex as re
import pandas as pd
import multiprocessing as mp
from multiprocessing import Process
import time
import h5py as h5
import sys
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
rho=5
# key inputs 

box_size=34
no_SRD=int((box_size**3)*rho)
#nu_bar=3
#delta_t_srd=0.014872025172594354
#nu_bar=0.9 
#rho=10
#delta_t_srd=0.05674857690605889

delta_t_srd=0.05071624521210362

box_vol=box_size**3

erate_selection= np.array([0.02,0.0175,0.015,0.0125,0.01]) 
#no_timesteps=np.array([7887000, 3944000, 1972000,  789000,  394000,   39000])
no_timesteps_selection=np.array([5915000,  6760000,  7887000,  9464000, 11831000])
# estimating number of steps  required
index_step_erate=int(sys.argv[1]) # make sure to specify python argument 
erate=erate_selection[index_step_erate]
no_timesteps=int(no_timesteps_selection[index_step_erate])
restart_count=0
dump_freq=10
internal_stiffness =np.array([60,80,100])
bending_stiffness=10000



j_=30
realisation_index=np.arange(0,j_,1)
# finding all the dump files in a folder

VP_general_name_string='vel.*'

Mom_general_name_string='mom.*'

log_general_name_string='log.*'
                         
TP_general_name_string='temp.*'


dump_general_name_string_after_pol='*_restartcount_'+str(restart_count)+'_*gdot_'+str(erate)+'_BK_10000_K_*_pol_after_rotation.h5'

dump_general_name_string_before_pol='*_restartcount_'+str(restart_count)+'*gdot_'+str(erate)+'_BK_10000_K_*_pol_before_rotation.h5'

dump_general_name_string_after_phantom='*_restartcount_'+str(restart_count)+'*gdot_'+str(erate)+'_BK_10000_K_*_phantom_after_rotation.h5'


filepath="/home/ucahlrl/Scratch/output"
#filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/test_run_flat_elastic_10_realisations"

Path_2_dump=filepath
# can chnage this to another array on kathleen

# pol before 
dump_realisation_name_info_before_pol=VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,
                                                                                    log_general_name_string,
                                                                                    VP_general_name_string,
                                                                                    Mom_general_name_string,
                                                                                    filepath,
                                                                                    dump_general_name_string_before_pol)
realisation_name_h5_before_pol=dump_realisation_name_info_before_pol[6]
count_h5_before_pol=dump_realisation_name_info_before_pol[7]

# pol after 
dump_realisation_name_info_after_pol= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,
                                                                                    log_general_name_string,
                                                                                    VP_general_name_string,
                                                                                    Mom_general_name_string,
                                                                                    filepath,
                                                                                    dump_general_name_string_after_pol)
realisation_name_h5_after_pol=dump_realisation_name_info_after_pol[6]
count_h5_after_pol=dump_realisation_name_info_after_pol[7]
#phantom after 
dump_realisation_name_info_after_phantom= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,
                                                                                        log_general_name_string,
                                                                                        VP_general_name_string,
                                                                                        Mom_general_name_string,
                                                                                        filepath,
                                                                                        dump_general_name_string_after_phantom)
realisation_name_h5_after_phantom=dump_realisation_name_info_after_phantom[6]
count_h5_after_phantom=dump_realisation_name_info_after_phantom[7]



# # find dump file size
with h5.File(realisation_name_h5_after_pol[0], 'r') as f:
    shape_after= f['particles']['small']['position']['value'].shape
    f.close()
print(shape_after[0])

with h5.File(realisation_name_h5_after_pol[0], 'r') as f_i:
      first_step= f_i['particles']['small']['position']['step'][0]
      first_posn= f_i['particles']['small']['position']['value'][0]
      f.close()


no_data_sets=internal_stiffness.shape[0]

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
          self.data_set= int(data_set) # int required to sort the data properly
          self.realisation_index_=int(realisation_index_)
     def __repr__(self):
        return '({},{},{})'.format(self.realisation_full_str,self.data_set,self.realisation_index_)
     


realisation_split_index=6
internal_stiff_index=20
# this can be turned into one function 
# before pol 
realisations_for_sorting_before_pol=[]
for i in realisation_name_h5_before_pol:
          realisation_index_=i.split('_')[realisation_split_index]
          data_set =i.split('_')[internal_stiff_index]
          realisations_for_sorting_before_pol.append(realisation(i,data_set,realisation_index_))


#after pol
realisations_for_sorting_after_pol=[]
for i in realisation_name_h5_after_pol:
          realisation_index_=i.split('_')[realisation_split_index]
          print(realisation_index_)
          data_set =i.split('_')[internal_stiff_index]
          print(data_set)
          realisations_for_sorting_after_pol.append(realisation(i,data_set,realisation_index_))


# after phantom 
realisations_for_sorting_after_phantom=[]
for i in realisation_name_h5_after_phantom:
          realisation_index_=i.split('_')[realisation_split_index]
          data_set =i.split('_')[internal_stiff_index]
          realisations_for_sorting_after_phantom.append(realisation(i,data_set,realisation_index_))



#NOTE this lambda function doesnt work properly 
# before pol      
realisation_name_h5_before_sorted_pol=sorted(realisations_for_sorting_before_pol,
                                             key=lambda x: (x.data_set, x.realisation_index_))
realisation_name_h5_before_sorted_final_pol=[]
for i in realisation_name_h5_before_sorted_pol:
     realisation_name_h5_before_sorted_final_pol.append(i.realisation_full_str)

# after pol
realisation_name_h5_after_sorted_pol=sorted(realisations_for_sorting_after_pol,
                                            key=lambda x: ( x.data_set, x.realisation_index_))
realisation_name_h5_after_sorted_final_pol=[]
for i in realisation_name_h5_after_sorted_pol:
     realisation_name_h5_after_sorted_final_pol.append(i.realisation_full_str)


## phantom
realisation_name_h5_after_sorted_phantom=sorted(realisations_for_sorting_after_phantom,
                                                key=lambda x: (x.data_set, x.realisation_index_))
realisation_name_h5_after_sorted_final_phantom=[]
for i in realisation_name_h5_after_sorted_phantom:
     realisation_name_h5_after_sorted_final_phantom.append(i.realisation_full_str)




realisation_name_h5_before_pol=realisation_name_h5_before_sorted_final_pol
realisation_name_h5_after_pol=realisation_name_h5_after_sorted_final_pol
realisation_name_h5_after_phantom=realisation_name_h5_after_sorted_final_phantom
print(realisation_name_h5_after_sorted_final_phantom)

# Computing full stress tensor of flat elastic.

eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 

#NOTE sorting not working properly 


def area_vector_calculation(eq_spring_length,
                            internal_stiffness,
                            terms_3,
                            terms_9,
                            area_vector_summed_shared,
                            stress_tensor_summed_shared,
                            realisation_name_h5_after_pol,
                            realisation_name_h5_before_pol,
                            realisation_name_h5_after_phantom,
                            shape_after,
                            erate_selection,
                            p):    
    
    # make sure to put the realisation name arguments with list index [p]
   
     with h5.File(realisation_name_h5_after_pol,'r') as f_c:
          with h5.File(realisation_name_h5_before_pol,'r') as f_d:
                with h5.File(realisation_name_h5_after_phantom,'r') as f_ph:
                
                

                            # need to change this to looking for spring constant 
                    # erate
                    data_set = int(np.where(erate_selection==float(realisation_name_h5_after_pol.split('_')[16]))[0][0])
                    # spring constant 
                    K=int(np.where(internal_stiffness==float(realisation_name_h5_after_pol.split('_')[20]))[0][0])


                    # has -1 when there is more than one set 
                    count=int((shape_after[0]-1)*p) # to make sure the data is in the correct location in 1-D array 
                    
                    # looping through N-1 dumps, since we are using a delta 
                    for j in range(1,shape_after[0]):
                         

                         
                         
                              # below for hirtori
                              pol_positions_after= f_c['particles']['small']['position']['value'][j]
                              pol_velocities_initial=f_d['particles']['small']['velocity']['value'][j-1]
                              pol_velocities_after=f_c['particles']['small']['velocity']['value'][j]
                              phantom_positions_after=f_ph['particles']['phantom']['position']['value'][j]
                              #pol_velocities_after=f_c['particles']['small']['velocity']['value'][j]

                              delta_mom_pos_tensor_summed_xx_pol=np.sum((pol_velocities_after[:,0]- 
                                                                         pol_velocities_initial[:,0])*pol_positions_after[:,0],axis=0)*mass_pol/(box_vol*delta_t_srd)#xx
                              delta_mom_pos_tensor_summed_yy_pol=np.sum((pol_velocities_after[:,1]-
                                                                          pol_velocities_initial[:,1])*pol_positions_after[:,1],axis=0)*mass_pol/(box_vol*delta_t_srd)#yy
                              delta_mom_pos_tensor_summed_zz_pol=np.sum((pol_velocities_after[:,2]- 
                                                                         pol_velocities_initial[:,2])*pol_positions_after[:,2],axis=0)*mass_pol/(box_vol*delta_t_srd)#zz
                              delta_mom_pos_tensor_summed_xz_pol=np.sum((pol_velocities_after[:,0]- 
                                                                         pol_velocities_initial[:,0])*pol_positions_after[:,2],axis=0)*mass_pol/(box_vol*delta_t_srd)#xz
                              delta_mom_pos_tensor_summed_xy_pol=np.sum((pol_velocities_after[:,0]- 
                                                                         pol_velocities_initial[:,0])*pol_positions_after[:,1],axis=0)*mass_pol/(box_vol*delta_t_srd)#xy
                              delta_mom_pos_tensor_summed_yz_pol=np.sum((pol_velocities_after[:,1]- 
                                                                         pol_velocities_initial[:,1])*pol_positions_after[:,2],axis=0)*mass_pol/(box_vol*delta_t_srd)#yz
                              delta_mom_pos_tensor_summed_zx_pol=np.sum((pol_velocities_after[:,2]- 
                                                                         pol_velocities_initial[:,2])*pol_positions_after[:,0],axis=0)*mass_pol/(box_vol*delta_t_srd)#zx
                              delta_mom_pos_tensor_summed_zy_pol=np.sum((pol_velocities_after[:,2]- 
                                                                         pol_velocities_initial[:,2])*pol_positions_after[:,1],axis=0)*mass_pol/(box_vol*delta_t_srd)#zy
                              delta_mom_pos_tensor_summed_yx_pol=np.sum((pol_velocities_after[:,1]- 
                                                                         pol_velocities_initial[:,1])*pol_positions_after[:,0],axis=0)*mass_pol/(box_vol*delta_t_srd)#yx

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
                              # Are these indices correct? yes checked on lammps script 
                    
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
                              stress_tensor_summed_xx=spring_force_positon_tensor_xx+ delta_mom_pos_tensor_summed_xx_pol + kinetic_energy_tensor_summed_xx_pol #xx
                              stress_tensor_summed_yy=spring_force_positon_tensor_yy + delta_mom_pos_tensor_summed_yy_pol + kinetic_energy_tensor_summed_yy_pol #yy
                              stress_tensor_summed_zz=spring_force_positon_tensor_zz + delta_mom_pos_tensor_summed_zz_pol + kinetic_energy_tensor_summed_zz_pol#zz
                              
                              
                              stress_tensor_summed_xz=(spring_force_positon_tensor_xz +    delta_mom_pos_tensor_summed_xz_pol + kinetic_energy_tensor_summed_xz_pol 
                                                       + (erate_selection[data_set]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_zz_pol)#xz # shear rate term
                              
                              stress_tensor_summed_xy=spring_force_positon_tensor_xy+delta_mom_pos_tensor_summed_xy_pol + kinetic_energy_tensor_summed_xy_pol #xy 
                              stress_tensor_summed_yz=spring_force_positon_tensor_yz+ delta_mom_pos_tensor_summed_yz_pol + kinetic_energy_tensor_summed_yz_pol #yz

                              stress_tensor_summed_zx=(spring_force_positon_tensor_zx + delta_mom_pos_tensor_summed_zx_pol + kinetic_energy_tensor_summed_xz_pol 
                                                       + (erate_selection[data_set]*delta_t_srd*0.5)*kinetic_energy_tensor_summed_zz_pol) #zx # shear rate term
                              
                              stress_tensor_summed_zy=spring_force_positon_tensor_zy + delta_mom_pos_tensor_summed_zy_pol+ kinetic_energy_tensor_summed_yz_pol#zy
                              stress_tensor_summed_yx=spring_force_positon_tensor_yx+ delta_mom_pos_tensor_summed_yx_pol + kinetic_energy_tensor_summed_xy_pol#yx

                              np_array_stress_tensor=np.array([stress_tensor_summed_xx,
                                                               stress_tensor_summed_yy,
                                                               stress_tensor_summed_zz,
                                                               stress_tensor_summed_xz,
                                                               stress_tensor_summed_xy,
                                                               stress_tensor_summed_yz,
                                                               stress_tensor_summed_zx,
                                                               stress_tensor_summed_zy,
                                                               stress_tensor_summed_yx])
                                   

                              ell_1=pol_positions_after[1,:]-pol_positions_after[0,:]
                              ell_2=pol_positions_after[2,:]-pol_positions_after[0,:]
                              area_vector=np.cross(ell_1,ell_2)

                              # for 9 element vector 

                         
                              # for 3-d vector 
                              start_index_3_element=terms_3[j+(count)-1] # has -1 when there is more than one set 
                              end_index_3_element=   start_index_3_element+3
                              start_index_9_element=terms_9[j+(count)-1] # has -1 when there is more than one set 
                              end_index_9_element=  start_index_9_element+9

               
                              
                              # inserting values into final 1-D array 
                              #kinetic_energy_tensor_summed_shared[start_index_6_element:end_index_6_element]=np_array_ke_entries
                              #delta_mom_pos_tensor_summed_shared[start_index_9_element:end_index_9_element]=np_array_delta_mom_pos
                              area_vector_summed_shared[start_index_3_element:end_index_3_element]=area_vector
                              stress_tensor_summed_shared[start_index_9_element:end_index_9_element]=np_array_stress_tensor


     return area_vector_summed_shared,stress_tensor_summed_shared


# parallel analysis of small run 
# determine size of 1-d array
size_area_vector_summed=int(no_data_sets*j_*((shape_after[0]-1))*3) 
size_stress_tensor_summed=int(no_data_sets*j_*((shape_after[0]-1))*9)
print(size_area_vector_summed)
processes=[]
print(count_h5_after_pol)
  


tic=time.perf_counter()
#
if __name__ =='__main__':
        
  
        
        # creating shared memory arrays that can be updated by multiple processes
       
        area_vector_summed_shared=   mp.Array('d',range(size_area_vector_summed))
        stress_tensor_summed_shared=mp.Array('d',range(size_stress_tensor_summed))

       
        list_done=[]
        
        simulations_analysed=0   
        increment=20  
        #creating processes, iterating over each realisation name 
        while simulations_analysed < count_h5_after_pol:
               for p in  range(simulations_analysed,simulations_analysed+increment):
                    
                    proc= Process(target=area_vector_calculation,args=(eq_spring_length,
                                                                      internal_stiffness,
                                                                      terms_3,
                                                                      terms_9,
                                                                      area_vector_summed_shared,
                                                                      stress_tensor_summed_shared,
                                                                      realisation_name_h5_after_pol[p],
                                                                      realisation_name_h5_before_pol[p],
                                                                      realisation_name_h5_after_phantom[p],
                                                                      shape_after,
                                                                      erate_selection,
                                                                      p,))
                                                  
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
        np.save("area_vector_summed_1d_test_restartcount_"+str(restart_count)+"_erate_"+str(erate)+"_bk_"\
                +str(bending_stiffness)+"_K_"+str(internal_stiffness[0])+\
                "_"+str(internal_stiffness[-1])+"_L_"+str(box_size)+\
                "_no_timesteps_"+str(no_timesteps),area_vector_summed_shared)
        np.save("stress_tensor_summed_1d_test_restartcount_"+str(restart_count)+"_erate_"\
                +str(erate)+"_bk_"+str(bending_stiffness)+\
                "_K_"+str(internal_stiffness[0])+\
                "_"+str(internal_stiffness[-1])+\
                "_L_"+str(box_size)+"_no_timesteps_"+\
                str(no_timesteps),stress_tensor_summed_shared)
      

                


# %%
