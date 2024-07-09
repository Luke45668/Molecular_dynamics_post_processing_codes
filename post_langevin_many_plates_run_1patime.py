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

path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
os.chdir(path_2_post_proc_module)

from log2numpy import *
from dump2numpy import *
import glob 
from post_MPCD_MP_processing_module import *
import pickle as pck

#%%
damp=0.01
strain_total=400
erate=np.flip(np.array([1,0.9,0.7,0.5,0.35,0.2,0.1,0.09,0.08,
                0.07,0.06,0.05,0.04,
                0.03,0.0275,0.025,0.0225,
                0.02,0.0175,0.015,0.0125,
                0.01,0.0075,0.005,0.0025,
                0.001,0.00075,0.0005]))
no_timesteps=np.flip(np.array([   394000,
          438000,    563000,    789000, 1127000,  1972000,   3944000,   4382000,
         4929000,   5634000,   6573000,   7887000,   9859000,  13145000,
        14340000,  15774000,  17527000,  19718000,  22534000,  26290000,
        31548000,  39435000,  52580000,  78870000, 157740000, 394351000,
       525801000, 788702000]))

erate=np.flip(np.array([1,0.9,0.7,0.5,0.2,0.1,0.09,0.08,
                0.07,0.06,0.05,0.04,
                0.03,0.0275,0.025,0.0225,
                0.02,0.0175,0.015,0.0125,
                0.01,0.0075,0.005,0.0025,
                0.001,0.00075,0.0005]))
no_timesteps=np.flip(np.array([   394000,
          438000,    563000,    789000,  1972000,   3944000,   4382000,
         4929000,   5634000,   6573000,   7887000,   9859000,  13145000,
        14340000,  15774000,  17527000,  19718000,  22534000,  26290000,
        31548000,  39435000,  52580000,  78870000, 157740000, 394351000,
       525801000, 788702000]))
# erate=np.flip(np.array([1,0.9,0.7,0.5,0.2,0.1,0.09,0.08,
#                 0.07,0.06,0.05,
#                 0.03]))
# no_timesteps=np.flip(np.array([   394000,
#           438000,    563000,    789000,  1972000,   3944000,   4382000,
#          4929000,   5634000,   6573000,   7887000,  13145000]))

# erate=np.flip(np.array([1,0.9,0.7,0.5,0.35,0.2,0.1,
#                 0.03,0.025,0.0225,
#                 0.02,0.0175,0.0125,
#                 0.01,0.0075,0.005,0.0025,
#                 0.001,0.00075,0.0005]))
# no_timesteps=np.flip(np.array([   394000,
#           438000,    563000,    789000, 1127000,  1972000,   3944000,   13145000, 
#            15774000,  17527000,  19718000,  22534000,
#         31548000,  39435000,  52580000,  78870000, 157740000, 394351000,
#        525801000, 788702000]))




thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
K=2000
j_=10
box_size=100
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 

filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/damp_0.01/sucessful_runs"
path_2_log_files=filepath
pol_general_name_string='*K_'+str(K)+'*pol*h5'

phantom_general_name_string='*K_'+str(K)+'*phantom*h5'

Mom_general_name_string='mom.*'

log_general_name_string='log.langevin*K_'+str(K)+'*'

dump_general_name_string='*K_'+str(K)+'**dump'




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
# %%

#NOTE: make sure the lists are all correct with precisely the right number of repeats or this code below
# will not work properly. 
import dump2numpy
dump_start_line = "ITEM: ATOMS id type x y z vx vy vz"
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
e_end=23
count=e_in
for i in range(e_in,e_end):
    i_=(count*j_)
    print("i_",i_)
    
    outputdim_dump=int(dump2numpy_f(dump_start_line,
                                    Path_2_dump,
                                    realisation_name_dump_sorted_final[i_],
                                    60).shape[0]/60)
    outputdim_log=log2numpy_reader(realisation_name_log_sorted_final[i_],
                                                    path_2_log_files,
                                                    thermo_vars).shape[0]
    
    dump_freq=int(realisation_name_log_sorted_final[i_].split('_')[10])

    log_file_array=np.zeros((j_,outputdim_log,7))
    p_velocities_array=np.zeros((j_,outputdim_dump,60,3))
    p_positions_array=np.zeros((j_,outputdim_dump,60,3))
    
    area_vector_array=np.zeros((j_,outputdim_dump,10,3))
    
    erate_velocity_array=np.zeros(((j_,outputdim_dump,30,1)))
    spring_force_positon_array=np.zeros((j_,outputdim_dump,10,6))
    interest_vectors_array=np.zeros((j_,outputdim_dump,10,5,3))
    for j in range(j_):
            j_index=j+(j_*count)

            # need to get rid of print statements in log2numpy 
            # print(realisation_name_log_sorted_final[j_index])
            print(j_index)
            log_file_array[j,:,:]=log2numpy_reader(realisation_name_log_sorted_final[j_index],
                                                        path_2_log_files,
                                                        thermo_vars)
            dump_data=dump2numpy_f(dump_start_line,
                                    Path_2_dump,
                                    realisation_name_dump_sorted_final[j_index],
                                    60)
            dump_data=np.reshape(dump_data,(outputdim_dump,60,8)).astype('float')
            new_pos_vel_array=np.zeros((outputdim_dump,10,6,3))
            
            p_positions_array[j,:,:,:]= dump_data[:,:,2:5]
            
            p_velocities_array[j,:,:,:]= dump_data[:,:,5:8]

            
            interest_vectors_store=np.zeros((outputdim_dump,10,5,3))

            for m in range(10):
                splicing_indices=np.arange(0,60,6)
                r1=0
                r2=outputdim_dump
                # r1=8025
                # r2=8100




                for l in range(r1,r2):
                        unsrt_dump= dump_data[l,:,:]
                        srt_dump=unsrt_dump[unsrt_dump[:,0].argsort()]
                        p_positions_array[j,l,:,:]=srt_dump[:,2:5]
                        p_velocities_array[j,l,:,:]=srt_dump[:,5:8]
                       

                    
                        particle_array=srt_dump[splicing_indices[m]:splicing_indices[m]+6,:]
                        ell_1=particle_array[1,2:5]-particle_array[0,2:5]
                        ell_2=particle_array[2,2:5]-particle_array[0,2:5]
                        phant_1=particle_array[0,2:5]-particle_array[4,2:5]
                        phant_2=particle_array[1,2:5]-particle_array[5,2:5]
                        phant_3=particle_array[2,2:5]-particle_array[3,2:5]
                        interest_vectors=np.array([ell_1,
                                                    ell_2,
                                                    phant_1,
                                                    phant_2,
                                                    phant_3])
                        
                        strain=l*dump_freq*md_step*erate[i] -\
                                np.floor(l*dump_freq*md_step*erate[i])
                            
                            
                        if strain <= 0.5:
                            tilt= (box_size)*(strain)
                            #tilt_test.append(tilt)
                        else:
                                
                            tilt=-(1-strain)*box_size
                        # tilt_test.append(tilt)
                        #print("tilt",tilt)

                        #z loop
                        if np.any(np.abs(interest_vectors[:,2])>box_size/box_size_div):
                                #print("periodic anomalies detected")
                                

                                for r in range(6):
                                            if particle_array[r,4]>box_size/2:
                                                particle_array[r,2:5]+=np.array([-tilt,0,-box_size])
                                                # adjusting the x velocity for the down shift
                                                particle_array[r,5]+=-box_size*erate[i]

                                                #print("z shift performed")

                               # print("post z mod particle array",particle_array)


                        ell_1=particle_array[1,2:5]-particle_array[0,2:5]
                        ell_2=particle_array[2,2:5]-particle_array[0,2:5]
                        phant_1=particle_array[0,2:5]-particle_array[4,2:5]
                        #f_spring_1_dirn=pol_positions_tuple[i][j,0,:]-ph_positions_tuple[i][j,1,:]
                        phant_2=particle_array[1,2:5]-particle_array[5,2:5]
                        #f_spring_2_dirn=pol_positions_tuple[i][j,1,:]-ph_positions_tuple[i][j,2,:]
                        phant_3=particle_array[2,2:5]-particle_array[3,2:5]
                        
                        interest_vectors=np.array([ell_1,
                                                    ell_2,
                                                    phant_1,
                                                    phant_2,
                                                    phant_3])
                        # y shift loop
                            

                            
                            

                        if np.any(np.abs(interest_vectors[:,1])>box_size/box_size_div):
                            # print("periodic anomalies detected")
                                
                                    
                                y_coords=particle_array[:,1]
                                y_coords[y_coords<box_size/2]+=box_size
                                particle_array[:,3]=y_coords
                                #print("y shift performed")

                        #print("post y shift mod particle array",particle_array)


                        ell_1=particle_array[1,2:5]-particle_array[0,2:5]
                        ell_2=particle_array[2,2:5]-particle_array[0,2:5]
                        phant_1=particle_array[0,2:5]-particle_array[4,2:5]
                        #f_spring_1_dirn=pol_positions_tuple[i][j,0,:]-ph_positions_tuple[i][j,1,:]
                        phant_2=particle_array[1,2:5]-particle_array[5,2:5]
                        #f_spring_2_dirn=pol_positions_tuple[i][j,1,:]-ph_positions_tuple[i][j,2,:]
                        phant_3=particle_array[2,2:5]-particle_array[3,2:5]

                        interest_vectors=np.array([ell_1,
                                                    ell_2,
                                                    phant_1,
                                                    phant_2,
                                                    phant_3])



                        if np.any(np.abs(interest_vectors[:,0])>box_size/box_size_div):
                            #print("periodic anomalies detected")
                            # x shift convention shift to smaller side
                            
                            x_coords=particle_array[:,2]
                            z_coords=particle_array[:,4]
                            box_position_x= x_coords -(z_coords/box_size)*tilt
                            x_coords[box_position_x<box_size/2]+=box_size
                            particle_array[:,2]=x_coords
                            #print("x shift performed")

                        #print("post x shift mod particle array",particle_array)


                        ell_1=particle_array[1,2:5]-particle_array[0,2:5]
                        ell_2=particle_array[2,2:5]-particle_array[0,2:5]
                        phant_1=particle_array[0,2:5]-particle_array[4,2:5]
                        #f_spring_1_dirn=pol_positions_tuple[i][j,0,:]-ph_positions_tuple[i][j,1,:]
                        phant_2=particle_array[1,2:5]-particle_array[5,2:5]
                        #f_spring_2_dirn=pol_positions_tuple[i][j,1,:]-ph_positions_tuple[i][j,2,:]
                        phant_3=particle_array[2,2:5]-particle_array[3,2:5]

                        interest_vectors_store[l,m,0,:]=ell_1
                        interest_vectors_store[l,m,1,:]=ell_2
                        interest_vectors_store[l,m,2,:]=phant_1
                        interest_vectors_store[l,m,3,:]=phant_2
                        interest_vectors_store[l,m,4,:]=phant_3

                        new_pos_vel_array[l,m,0:3,:]=particle_array[0:3,2:5]
                        new_pos_vel_array[l,m,3:6,:]=particle_array[0:3,5:8]





                        #NOTE need to add some sort of way to store new positions and velocities





            # final correction for undected anomalies
                      
                  

            interest_vectors_store[interest_vectors_store>box_size/2]-=box_size
            interest_vectors_store[interest_vectors_store<-box_size/2]+=box_size  

                     
        
            # plt.plot( interest_vectors_store[r1:r2,:,0,0])
            # plt.title("$\ell_{1},x$")
            # plt.xlabel("output count")
            # plt.legend()
            # plt.show()

            # plt.plot(interest_vectors_store[r1:r2,:,1,0])
            # plt.title("$\ell_{2},x$")
            # plt.xlabel("output count")
            # plt.show()

            # plt.plot(interest_vectors_store[r1:r2,:,2,0])
            # plt.title("$\Delta_{1},x$")
            # plt.xlabel("output count")
            # plt.show()

            # plt.plot(interest_vectors_store[r1:r2,:,3,0])
            # plt.title("$\Delta_{2},x$")
            # plt.xlabel("output count")
            # plt.show()


            # plt.plot(interest_vectors_store[r1:r2,:,4,0])
            # plt.title("$\Delta_{3},x$")
            # plt.xlabel("output count")
            # plt.show()

            f_spring_1_dirn= interest_vectors_store[:,:,2,:]
           
            f_spring_1_mag=np.transpose(np.tile(np.sqrt(np.sum((f_spring_1_dirn)**2,axis=2)),\
                                                (3,1,1)),(1,2,0))

            f_spring_1=(K*(f_spring_1_dirn/f_spring_1_mag)*(f_spring_1_mag-eq_spring_length))
        
            # spring 2
            f_spring_2_dirn= interest_vectors_store[:,:,3,:]
        
            f_spring_2_mag=np.transpose(np.tile(np.sqrt(np.sum((f_spring_2_dirn)**2,axis=2)),\
                                                (3,1,1)),(1,2,0))

            f_spring_2=(K*(f_spring_2_dirn/f_spring_2_mag)*(f_spring_2_mag-eq_spring_length))
        
            # spring 3

            f_spring_3_dirn= interest_vectors_store[:,:,4,:]
        
            f_spring_3_mag=np.transpose(np.tile(np.sqrt(np.sum((f_spring_3_dirn)**2,axis=2)),\
                                                (3,1,1)),(1,2,0))

            f_spring_3=(K*(f_spring_3_dirn/f_spring_3_mag)*(f_spring_3_mag-eq_spring_length))


            spring_force_positon_tensor_xx=f_spring_1[:,:,0]*f_spring_1_dirn[:,:,0] +\
              f_spring_2[:,:,0]*f_spring_2_dirn[:,:,0] +f_spring_3[:,:,0]*f_spring_3_dirn[:,:,0] 
            spring_force_positon_tensor_yy=f_spring_1[:,:,1]*f_spring_1_dirn[:,:,1] +\
              f_spring_2[:,:,1]*f_spring_2_dirn[:,:,1] +f_spring_3[:,:,1]*f_spring_3_dirn[:,:,1] 
            spring_force_positon_tensor_zz=f_spring_1[:,:,2]*f_spring_1_dirn[:,:,2] +\
              f_spring_2[:,:,2]*f_spring_2_dirn[:,:,2] +f_spring_3[:,:,2]*f_spring_3_dirn[:,:,2] 
            spring_force_positon_tensor_xz=f_spring_1[:,:,0]*f_spring_1_dirn[:,:,2] +\
              f_spring_2[:,:,0]*f_spring_2_dirn[:,:,2] +f_spring_3[:,:,0]*f_spring_3_dirn[:,:,2] 
            spring_force_positon_tensor_xy=f_spring_1[:,:,0]*f_spring_1_dirn[:,:,1] +\
              f_spring_2[:,:,0]*f_spring_2_dirn[:,:,1] +f_spring_3[:,:,0]*f_spring_3_dirn[:,:,1] 
            spring_force_positon_tensor_yz=f_spring_1[:,:,1]*f_spring_1_dirn[:,:,2] +\
              f_spring_2[:,:,1]*f_spring_2_dirn[:,:,2] +f_spring_3[:,:,1]*f_spring_3_dirn[:,:,2] 
            
                
            np_array_spring_pos_tensor=np.transpose(np.array([spring_force_positon_tensor_xx,
                                                spring_force_positon_tensor_yy,
                                                spring_force_positon_tensor_zz,
                                                spring_force_positon_tensor_xz,
                                                spring_force_positon_tensor_xy,
                                                spring_force_positon_tensor_yz, 
                                                    ]),(1,2,0))
            
            
  

            spring_force_positon_array[j,:,:,:]= np_array_spring_pos_tensor


            # this can be switched off for general runs
            # just to check there isnt any PBC anomalies
            
    


    area_vector_array[j,:,:,:]=np.cross(interest_vectors_store[:,:,0,:],
                                        interest_vectors_store[:,:,1,:]
                                        ,axisa=2,axisb=2)
    interest_vectors_mean_mag=np.mean(np.sqrt(np.sum(interest_vectors_store**2,
                                                     axis=3)),axis=0)    
    lgf_mean=np.mean(log_file_array,axis=0)    
    spring_force_positon_mean=np.mean(np.mean(spring_force_positon_array,axis=0),
                                      axis=1)
    
    new_pos_vel_tuple=new_pos_vel_tuple+(new_pos_vel_array,)
    p_positions_tuple=p_positions_tuple+(p_positions_array,)
    p_velocities_tuple=p_velocities_tuple+(p_velocities_array,)
    log_file_tuple=log_file_tuple+(lgf_mean,)
    area_vector_tuple=area_vector_tuple+(area_vector_array,)
    spring_force_positon_tensor_tuple=spring_force_positon_tensor_tuple+\
        (spring_force_positon_array,)
    interest_vectors_tuple=interest_vectors_tuple+( interest_vectors_mean_mag,)
    count+=1



#%% save tuples to avoid needing the next stage 
label='damp_'+str(damp)+'_K_'+str(K)+'_'
import pickle as pck

folder_check_or_create(filepath,"saved_tuples")

with open(label+'spring_force_positon_tensor_tuple.pickle', 'wb') as f:
    pck.dump(spring_force_positon_tensor_tuple, f)

with open(label+'log_file_tuple.pickle', 'wb') as f:
    pck.dump(log_file_tuple, f)

with open(label+"new_pos_vel_tuple.pickle",'wb') as f:
     pck.dump(new_pos_vel_tuple,f)

with open(label+'p_velocities_tuple.pickle', 'wb') as f:
    pck.dump( p_velocities_tuple, f)

with open(label+'p_positions_tuple.pickle', 'wb') as f:
    pck.dump( p_positions_tuple, f)

with open(label+"area_vector_tuple.pickle", 'wb') as f:
    pck.dump(area_vector_tuple,f)

#%% load in tuples
folder_check_or_create(filepath,"saved_tuples")


label='damp_'+str(damp)+'_K_'+str(K)+'_'


with open(label+'spring_force_positon_tensor_tuple.pickle', 'rb') as f:
    spring_force_positon_tensor_tuple=pck.load(f)

with open(label+'log_file_tuple.pickle', 'rb') as f:
    log_file_tuple=pck.load(f)

with open(label+'p_velocities_tuple.pickle', 'rb') as f:
    p_velocities_tuple=pck.load(f)

with open(label+'p_positions_tuple.pickle', 'rb') as f:
    pol_positions_tuple=pck.load(f)

with open(label+"new_pos_vel_tuple.pickle",'rb') as f:
     new_pos_vel_tuple=pck.load(f)


#%% computing stress distributions 
import seaborn as sns
stress_tensor=np.zeros((e_end,10))
line_values=np.zeros((10))
for i in range(e_in,e_end):
        for m in range(10):
           

                data=np.mean(spring_force_positon_tensor_tuple[i][:,:,m,2]-spring_force_positon_tensor_tuple[i][:,:,m,1], axis=0)
             
                kde=sns.kdeplot(data)
                lines = kde.get_lines()

                for line in lines:
                    x, y = line.get_data()
                    line_values[m]=x[np.argmax(y)]

                
                    plt.axvline(x[np.argmax(y)], ls='--', color='black')
                
                stress_tensor[i,:]=line_values[:]
        

        plt.show()

plt.scatter(erate[:e_end],np.mean(stress_tensor,axis=1))
    
plt.show()    


#%%

#NOTE this works but i think the damping parameter is too low 

from scipy.optimize import curve_fit
COM_velocity_tuple=()
COM_position_tuple=()
erate_velocity_tuple=()

count=0
box_error_count=0
for i in range(e_in,e_end):
    i_=(count*j_)
    COM_position= np.mean(new_pos_vel_tuple[i][:,:,0:3,:],axis=2)
    COM_velocity=np.mean(new_pos_vel_tuple[i][:,:,3:6,:],axis=2)
    erate_predicted_vel=COM_position[:,:,2]*erate[i]
    x=np.ravel(erate_predicted_vel)
    y= np.ravel( COM_velocity[:,:,0])
    plt.scatter(x,y)
    plt.show()


# this calculation works
# i think we just need to get a higher value for damp
#%% temp check 
column=5
final_temp=np.zeros((erate.size))
mean_temp_array=np.zeros((erate.size))

for i in range(e_in,e_end):
        
        # plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
        mean_temp_array[i]=np.mean( log_file_tuple[i][5:,column])
      
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
plt.xscale('log')
# plt.yscale('log')
plt.axhline(1,label="$T_{0}=1$")
plt.legend()
plt.tight_layout()
plt.show()

 #%%             
strainplot_tuple=()
for i in range(e_in,e_end):
     # is this the correct way to plot 
     strain_plotting_points= np.linspace(0,strain_total, spring_force_positon_tensor_tuple[i].shape[0])
   
     strainplot_tuple=strainplot_tuple+(strain_plotting_points,)  
     print(strainplot_tuple[i].size)

folder="stress_tensor_plots"
folder_check_or_create(path_2_log_files,folder)
labels_stress=["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{yz}$"]

# for i in range(10):
#      for j in range(3,6):
#         plt.plot(strainplot_tuple[i],spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
#         plt.legend()
#      plt.show()

#%% normal stresses
for i in range(e_in,e_end):
    for j in range(0,3):
        plt.plot(strainplot_tuple[i],spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
        plt.xlabel("$\gamma$")
        plt.ylabel("$\sigma$")
    
    plt.legend()
    plt.show()
#%% shear stresses
for i in range(e_in,e_end):
    for j in range(3,6):
        plt.plot(strainplot_tuple[i],spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
        plt.xlabel("$\gamma$")
        plt.ylabel("$\sigma$")
    
    
    plt.legend()
    plt.show()                       


# %%
cutoff_ratio=0.1
end_cutoff_ratio=1
folder="N_1_plots"

N_1_mean=np.zeros((erate.size))
for i in range(e_in,e_end):

    # cutoff=int(nan_size[i]) +int(np.ceil(cutoff_ratio*(viscoelastic_stress_tuple_wa[i][:-1,0].size-nan_size[i])))
    # end_cutoff=int(nan_size[i]) +int(np.ceil(end_cutoff_ratio*(viscoelastic_stress_tuple_wa[i][:-1,0].size-nan_size[i])))
    # #cutoff=int(nan_size[i]) +int(np.ceil(cutoff_ratio*(spring_force_positon_tensor_tuple_wa[i][:-1,0].size-nan_size[i])))

    N_1=spring_force_positon_tensor_tuple[i][:,0]-spring_force_positon_tensor_tuple[i][:,2]
    #N_1=viscoelastic_stress_tuple_wa[i][cutoff:end_cutoff-1,0]-viscoelastic_stress_tuple_wa[i][cutoff:end_cutoff-1,2]
    #N_1[np.abs(N_1)>400]=0
    N_1_mean[i]=np.mean(N_1[:])
    print(N_1_mean)
    
    plt.plot(strainplot_tuple[i][:],N_1,label="$\dot{\gamma}="+str(erate[i])+"$")
    #plt.plot(N_1, label="$\dot{\gamma}="+str(erate[i])+"$")
    plt.axhline(np.mean(N_1))
    plt.ylabel("$N_{1}$")
    plt.xlabel("$\gamma$")
    plt.legend()
    plt.show()
#%%

plt.scatter((erate[:e_end]),N_1_mean[:e_end])
# popt,pcov=curve_fit(linearfunc,erate[:],N_1_mean[:])

# plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))

# popt,pcov=curve_fit(quadfunc,erate[:],N_1_mean[:])
# plt.plot(erate[:],(popt[0]*(erate[:]**2)))  
#plt.xscale('log')
# plt.yscale('log')
plt.ylabel("$N_{1}$")
plt.xlabel("$\dot{\gamma}$")
plt.show()
# %%

folder="N_2_plots"
# cutoff_ratio=0.5
# end_cutoff_ratio=0.7
cutoff_ratio=0.2
end_cutoff_ratio=1
N_2_mean=np.zeros((erate.size))
for i in range(e_in,e_end):
   # cutoff=int(nan_size[i]) +int(np.ceil(cutoff_ratio*(spring_force_positon_tensor_tuple_wa[i][:-1,0].size-nan_size[i])))
   
    # cutoff=int(nan_size[i]) +int(np.ceil(cutoff_ratio*(viscoelastic_stress_tuple_wa[i][:-1,0].size-nan_size[i])))
    # end_cutoff=int(nan_size[i]) +int(np.ceil(end_cutoff_ratio*(viscoelastic_stress_tuple_wa[i][:-1,0].size-nan_size[i])))
    #N_2= viscoelastic_stress_tuple_wa[i][cutoff:end_cutoff,2]-viscoelastic_stress_tuple_wa[i][cutoff:end_cutoff,1]
    N_2= spring_force_positon_tensor_tuple[i][:,2]-spring_force_positon_tensor_tuple[i][:,1]
    # N_2[np.abs(N_2)>2000]=0
    N_2_mean[i]=np.mean(N_2[:])
   
    print(N_2_mean)
    #plt.plot(strainplot_tuple[i][:-1],N_2)
    plt.plot(strainplot_tuple[i][:],N_2)
    plt.axhline(np.mean(N_2),label="$\dot{\gamma}="+str(erate[i])+"$")
    plt.ylabel("$N_{2}$")
    plt.xlabel("$\gamma$")
    plt.legend()
    plt.show()

#%%
plt.scatter((erate[:e_end]),N_2_mean[:e_end])

# popt,pcov=curve_fit(linearfunc,erate[:],N_2_mean[:])

# plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))

plt.ylabel("$N_{2}$")
plt.xlabel("$\dot{\gamma}$")
plt.xscale('log')
plt.show()

#%% N_1/N_2

plt.scatter((erate[:e_end]),N_1_mean[:e_end],label="$N_{1}$" ,marker='x')
plt.scatter((erate[:e_end]),(-1*N_2_mean[:e_end]),label="$-4N_{2}$" ,marker='o')
# popt,pcov=curve_fit(linearfunc,erate[:],N_2_mean[j,:])
#plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))
# popt,pcov=curve_fit(quadfunc,erate[:],N_2_mean[j,:])
# plt.plot(erate[:],(popt[0]*(erate[:]**2)))  
#plt.ylabel("$N_{2}$")
plt.xlabel("$\dot{\gamma}$")
# plt.xscale('log')
# plt.yscale('log')
plt.legend()


plt.show()




#%%
cutoff_ratio=0.1
end_cutoff_ratio=1
xz_shear_stress_mean=np.zeros((erate.size))
xy_shear_stress_mean=np.zeros((erate.size))
yz_shear_stress_mean=np.zeros((erate.size))
for i in range(e_in,e_end):
   
   
    #cutoff=int(nan_size[i]) +int(np.ceil(cutoff_ratio*(viscoelastic_stress_tuple_wa[i][:-1,0].size-nan_size[i])))
    # end_cutoff=int(nan_size[i]) +int(np.ceil(end_cutoff_ratio*(viscoelastic_stress_tuple_wa[i][:-1,0].size-nan_size[i])))
    xz_shear_stress= spring_force_positon_tensor_tuple[i][:,3]
    xy_shear_stress= spring_force_positon_tensor_tuple[i][:,4]
    yz_shear_stress= spring_force_positon_tensor_tuple[i][:,5]
    
    # end_cutoff=int(nan_size[i]) +int(np.ceil(end_cutoff_ratio*(viscoelastic_stress_tuple_wa[i][:-1,0].size-nan_size[i])))
    # xz_shear_stress= viscoelastic_stress_tuple_wa[i][cutoff:end_cutoff,3]
   
    xz_shear_stress_mean[i]=np.mean(xz_shear_stress[:])
    xy_shear_stress_mean[i]=np.mean(xy_shear_stress[:])
    yz_shear_stress_mean[i]=np.mean(yz_shear_stress[:])
    #plt.plot(strainplot_tuple[i][:],xz_shear_stress, label=labels_stress[3])
    plt.plot(strainplot_tuple[i][:],xz_shear_stress, label=labels_stress[3]+",$\dot{\gamma}="+str(erate[i])+"$")
    plt.axhline(xz_shear_stress_mean[i])
    plt.ylabel("$\sigma_{xz}$")
    plt.xlabel("$\gamma$")
    plt.legend()
    plt.show()

#%%
plt.scatter(erate[:e_end],xz_shear_stress_mean[:e_end])
plt.axhline(np.mean(xz_shear_stress_mean[:]))
# plt.xscale('log')
# plt.yscale('log')
#plt.ylim(-5,2)
plt.show()

#%%
plt.scatter(erate[:e_end],xz_shear_stress_mean[:e_end]/erate[:e_end])
#plt.axhline(np.mean(xz_shear_stress_mean[:]))
# plt.xscale('log')
# plt.yscale('log')
#plt.ylim(-5,2)
plt.show()
#%%
plt.scatter(erate[:e_end],yz_shear_stress_mean[:e_end])
plt.axhline(np.mean(yz_shear_stress_mean[:e_end]))
#plt.ylim(-5,2)
plt.show()


#%%
plt.scatter(erate[:e_end],xy_shear_stress_mean[:e_end])
plt.axhline(np.mean(xy_shear_stress_mean[:e_end]))
#plt.ylim(-5,2)
plt.show()

      
# %%
with h5.File("langevinrun_no999_hookean_flat_elastic_889634_1_100_0.03633_0.005071624521210362_1000_1000_5634000_0.2_gdot_0.07_BK_10000_K_2000_phantom_after_rotation.h5",'r') as f_ph:
        print("phantom shape",f_ph['particles']['phantom']['position']['value'].shape)


# %%
