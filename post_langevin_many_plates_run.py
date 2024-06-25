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
damp=0.03633
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
                0.07,0.06,0.05,
                0.03]))
no_timesteps=np.flip(np.array([   394000,
          438000,    563000,    789000,  1972000,   3944000,   4382000,
         4929000,   5634000,   6573000,   7887000,  13145000]))

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
K=500
j_=5
box_size=100
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 

filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/run_835070_104809_233523"
path_2_log_files=filepath
pol_general_name_string='*K_'+str(K)+'*pol*h5'

phantom_general_name_string='*K_'+str(K)+'*phantom*h5'

Mom_general_name_string='mom.*'

log_general_name_string='log.langevin*K_'+str(K)+'*'

dump_general_name_string='*dump'




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
# %%
md_step=0.00101432490424207
strain_total=np.repeat(400,erate.size)
log_file_tuple=()
pol_velocities_tuple=()
pol_positions_tuple=()
ph_positions_tuple=()
ph_velocities_tuple=()
area_vector_tuple=()
conform_tensor_tuple=()
spring_force_positon_tensor_tuple=()
count=0
interest_vectors_tuple=()
compare_vel_to_profile_tuple=()
tilt_test=[]

for i in range(erate.size):
     i_=(count*j_)
     print("i_",i_)
     with h5.File(realisation_name_h5_after_sorted_final_pol[i_],'r') as f_check:
        outputdim_hdf5=f_check['particles']['small']['velocity']['value'].shape[0]
        outputdim_log=log2numpy_reader(realisation_name_log_sorted_final[i_],
                                                        path_2_log_files,
                                                        thermo_vars).shape[0]
        dump_freq=int(realisation_name_h5_after_sorted_final_pol[i_].split('_')[10])

        log_file_array=np.zeros((j_,outputdim_log,7))
        pol_velocities_array=np.zeros((j_,outputdim_hdf5,30,3))
        pol_positions_array=np.zeros((j_,outputdim_hdf5,30,3))
        ph_velocities_array=np.zeros((j_,outputdim_hdf5,30,3))
        ph_positions_array=np.zeros((j_,outputdim_hdf5,30,3))
        area_vector_array=np.zeros((j_,outputdim_hdf5,10,3))
        conform_tensor_array=np.zeros((j_,outputdim_hdf5,30,9))
        COM_velocity_array=np.zeros((j_,outputdim_hdf5,30,3))
        COM_position_array=np.zeros((j_,outputdim_hdf5,30,3))
        erate_velocity_array=np.zeros(((j_,outputdim_hdf5,30,1)))
        spring_force_positon_array=np.zeros((j_,outputdim_hdf5,10,6))
        interest_vectors_array=np.zeros((j_,outputdim_hdf5,10,5,3))
        for j in range(j_):
            j_index=j+(j_*count)

            # need to get rid of print statements in log2numpy 
            # print(realisation_name_log_sorted_final[j_index])
            print(j_index)
            log_file_array[j,:,:]=log2numpy_reader(realisation_name_log_sorted_final[j_index],
                                                        path_2_log_files,
                                                        thermo_vars)
            with h5.File(realisation_name_h5_after_sorted_final_pol[j_index],'r') as f_c:
                
                  with h5.File(realisation_name_h5_after_sorted_final_phantom[j_index],'r') as f_ph:
                    # print(realisation_name_h5_after_sorted_final_pol[j_index])
                    # print("pol shape ",f_c['particles']['small']['velocity']['value'].shape)
                    print(realisation_name_h5_after_sorted_final_phantom[j_index])
                    print("phantom shape",f_ph['particles']['phantom']['velocity']['value'].shape)
                    pol_velocities_array[j,:,:,:]=f_c['particles']['small']['velocity']['value']
                    pol_positions_array[j,:,:,:]=f_c['particles']['small']['position']['value']
                    ph_velocities_array[j,:,:,:]=f_ph['particles']['phantom']['velocity']['value']
                    ph_positions_array[j,:,:,:]=f_ph['particles']['phantom']['position']['value']

                    pol_velocities_array_=np.reshape(pol_velocities_array,(j_,outputdim_hdf5,10,3,3))
                    pol_positions_array_=np.reshape(pol_positions_array,(j_,outputdim_hdf5,10,3,3))
                    ph_velocities_array_=np.reshape(ph_velocities_array,(j_,outputdim_hdf5,10,3,3))
                    ph_positions_array_=np.reshape(ph_positions_array,(j_,outputdim_hdf5,10,3,3))

                    ell_1=pol_positions_array_[j,:,:,1,:]-pol_positions_array_[j,:,:,0,:]
                    ell_2=pol_positions_array_[j,:,:,2,:]-pol_positions_array_[j,:,:,0,:]
                    phant_1=pol_positions_array_[j,:,:,0,:]-ph_positions_array_[j,:,:,1,:]
                    #f_spring_1_dirn=pol_positions_tuple[i][j,0,:]-ph_positions_tuple[i][j,1,:]
                    phant_2=pol_positions_array_[j,:,:,1,:]-ph_positions_array_[j,:,:,2,:]
                    #f_spring_2_dirn=pol_positions_tuple[i][j,1,:]-ph_positions_tuple[i][j,2,:]
                    phant_3=pol_positions_array_[j,:,:,2,:]-ph_positions_array_[j,:,:,0,:]

                    r1=0
                    r2=outputdim_hdf5
                    
                   
                    for l in range(r1,r2):
                        # z correction for ell_1
                        particle_array=np.array([pol_positions_array_[j,l,:,0,:],
                                                 pol_positions_array_[j,l,:,1,:],
                                                 pol_positions_array_[j,l,:,2,:],
                                                ph_positions_array_[j,l,:,0,:],
                                                ph_positions_array_[j,l,:,1,:],
                                                ph_positions_array_[j,l,:,2,:]])
                        particle_COM=np.mean(particle_array,axis=0)
                       
                        particle_vel_array=np.array([pol_velocities_array_[j,l,:,0,:],
                                                 pol_velocities_array_[j,l,:,1,:],
                                                 pol_velocities_array_[j,l,:,2,:],
                                                ph_velocities_array_[j,l,:,0,:],
                                                ph_velocities_array_[j,l,:,1,:],
                                                ph_velocities_array_[j,l,:,2,:]])
                        
                        particle_COM_vel=np.mean(particle_vel_array,axis=0)
                        pred_velocity_profile=particle_array[:3,2]*erate[i]


                    

                    
                       
                        
                        interest_vectors=np.array([ell_1[l,:,:],ell_2[l,:,:],phant_1[l,:,:],phant_2[l,:,:],phant_3[l,:,:]])
                        
                        # print("unmodified particle array ",particle_array)
                        # print("unmodified interest vectors", interest_vectors)

                        
                        strain=l*dump_freq*md_step*erate[i] -\
                              np.floor(l*dump_freq*md_step*erate[i])
                        
                        
                        if strain <= 0.5:
                            tilt= (int(box_size))*(strain)
                            #tilt_test.append(tilt)
                        else:
                                
                            tilt=-(1-strain)*int(box_size)
                           # tilt_test.append(tilt)
                        #print("tilt",tilt)

                      

                        for m in range(10):

                              # z shift 
                              # convention shift down to lower boundary 

                            if np.any(np.abs(interest_vectors[:,m,2])>int(box_size)/2):
                                # print("periodic anomalies detected")
                                # print("z shift performed")
                                
                            

                                for r in range(6):
                                    if particle_array[r,m,2]>int(box_size)/2:
                                        particle_array[r,m,:]+=np.array([-tilt,0,-int(box_size)])


                              #print("post z mod particle array",particle_array)


                        ell_1_c=particle_array[1,:,:]-particle_array[0,:,:]
                        ell_2_c=particle_array[2,:,:]-particle_array[0,:,:]
                        phant_1_c=particle_array[0,:,:]-particle_array[4,:,:]
                        phant_2_c=particle_array[1,:,:]-particle_array[5,:,:]
                        phant_3_c=particle_array[2,:,:]-particle_array[3,:,:]

                        interest_vectors=np.array([ell_1_c,ell_2_c,phant_1_c,phant_2_c,phant_3_c])
                        
                        # y shift 
                        for m in range(10):

                          
                        

                            if np.any(np.abs(interest_vectors[:,m,1])>int(box_size)/2):
                                #print("periodic anomalies detected")
                                y_coords=particle_array[:,m,1]
                                y_coords[y_coords<int(box_size)/2]+=int(box_size)
                                particle_array[:,m,1]=y_coords
                            # print("y shift performed")

                            # print("post y shift mod particle array",particle_array)
                                
                               

                        ell_1_c=particle_array[1,:,:]-particle_array[0,:,:]
                        ell_2_c=particle_array[2,:,:]-particle_array[0,:,:]
                        phant_1_c=particle_array[0,:,:]-particle_array[4,:,:]
                        phant_2_c=particle_array[1,:,:]-particle_array[5,:,:]
                        phant_3_c=particle_array[2,:,:]-particle_array[3,:,:]

                        interest_vectors=np.array([ell_1_c,ell_2_c,phant_1_c,phant_2_c,phant_3_c])

                        # x shift 

                        for m in range(10):

                            if np.any(np.abs(interest_vectors[:,m,0])>int(box_size)/2):
                                #print("periodic anomalies detected")
                                # x shift convention shift to smaller side
                                
                                x_coords=particle_array[:,m,0]
                                z_coords=particle_array[:,m,2]
                                box_position_x= x_coords -(z_coords/int(box_size))*tilt
                                x_coords[box_position_x<int(box_size)/2]+=int(box_size)
                                particle_array[:,m,0]=x_coords
                            # print("x shift performed")

                        # print("post x shift mod particle array",particle_array)

                        ell_1_c=particle_array[1,:,:]-particle_array[0,:,:]
                        ell_2_c=particle_array[2,:,:]-particle_array[0,:,:]
                        phant_1_c=particle_array[0,:,:]-particle_array[4,:,:]
                        phant_2_c=particle_array[1,:,:]-particle_array[5,:,:]
                        phant_3_c=particle_array[2,:,:]-particle_array[3,:,:]

                        interest_vectors=np.array([ell_1_c,ell_2_c,phant_1_c,phant_2_c,phant_3_c])
                            

                            #print("interest vectors",interest_vectors)



                            # if np.any(np.abs(interest_vectors)>23/2):
                            #     interest_vectors[interest_vectors>23/2]-=23
                            #     interest_vectors[interest_vectors<-23/2]+=23

                        ell_1[l,:,:]=interest_vectors[0,:,:]
                        ell_2[l,:,:]=interest_vectors[1,:,:]
                        phant_1[l,:,:]=interest_vectors[2,:,:]
                        phant_2[l,:,:]=interest_vectors[3,:,:]
                        phant_3[l,:,:]=interest_vectors[4,:,:]
                        interest_vectors=np.reshape(interest_vectors,(10,5,3))
                        interest_vectors_array[j,l,:,:,:]=interest_vectors


                            # if np.any(np.abs(interest_vectors)>23/2):
                            #     # print(interest_vectors)
                            #     # print("anomalies still present")

                        
                            #     breakpoint

                      


                    ell_1[ell_1[:,:,:]>int(box_size)/2]-=int(box_size)
                    ell_1[ell_1[:,:,:]<-int(box_size)/2]+=int(box_size)
                    ell_2[ell_2[:,:,:]>int(box_size)/2]-=int(box_size)
                    ell_2[ell_2[:,:,:]<-int(box_size)/2]+=int(box_size)
                    
                    phant_1[phant_1[:,:,:]>int(box_size)/2]-=int(box_size)
                    phant_1[phant_1[:,:,:]<-int(box_size)/2]+=int(box_size)
                    
                  
                    phant_2[phant_2[:,:,:]>int(box_size)/2]-=int(box_size)
                    phant_2[phant_2[:,:,:]<-int(box_size)/2]+=int(box_size)
                   
                    phant_3[phant_3[:,:,:]>int(box_size)/2]-=int(box_size)
                    phant_3[phant_3[:,:,:]<-int(box_size)/2]+=int(box_size)


                  
                    # plt.plot(ell_1[r1:r2,0])
                    # plt.title("$|\ell_{1}|,x$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # plt.plot(ell_2[r1:r2,0])
                    # plt.title("$|\ell_{2}|,x$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # plt.plot(phant_1[r1:r2,0])
                    # plt.title("$|ph_{1}|,x$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # plt.plot(phant_2[r1:r2,0])
                    # plt.title("$|ph_{2}|,x$")
                    # plt.xlabel("output count")
                    # plt.show()


                    # plt.plot(phant_3[r1:r2,0])
                    # plt.title("$|ph_{3}|,x$")
                    # plt.xlabel("output count")
                    # plt.show()

                        
                    
                    interest_vectors=np.array([ell_1,ell_2,phant_1,phant_2,phant_3])
                    interest_vectors=np.reshape(interest_vectors,(outputdim_hdf5,10,5,3))
                    interest_vectors_array[j,:,:,:,:]=interest_vectors


                    # spring extension calcs

                    for m in range(10):

                        f_spring_1_dirn=phant_1[:,m,:]
                    
                        f_spring_1_mag=np.sqrt(np.sum((f_spring_1_dirn)**2,axis=1))

                        f_spring_1=(K*(f_spring_1_dirn.T/f_spring_1_mag)*(f_spring_1_mag-eq_spring_length)).T
                    
                        # spring 2
                        f_spring_2_dirn=phant_2[:,m,:]
                    
                        f_spring_2_mag=np.sqrt(np.sum((f_spring_2_dirn)**2,axis=1))
                    
                        f_spring_2=(K*(f_spring_2_dirn.T/f_spring_2_mag)*(f_spring_2_mag-eq_spring_length)).T
                    
                        # spring 3

                        f_spring_3_dirn=phant_3[:,m,:]
                    
                        f_spring_3_mag=np.sqrt(np.sum((f_spring_3_dirn)**2,axis=1))
                        
                        f_spring_3=(K*(f_spring_3_dirn.T/f_spring_3_mag)*(f_spring_3_mag-eq_spring_length)).T


                        spring_force_positon_tensor_xx=f_spring_1[:,0]*f_spring_1_dirn[:,0] + f_spring_2[:,0]*f_spring_2_dirn[:,0] +f_spring_3[:,0]*f_spring_3_dirn[:,0] 
                        spring_force_positon_tensor_yy=f_spring_1[:,1]*f_spring_1_dirn[:,1] + f_spring_2[:,1]*f_spring_2_dirn[:,1] +f_spring_3[:,1]*f_spring_3_dirn[:,1] 
                        spring_force_positon_tensor_zz=f_spring_1[:,2]*f_spring_1_dirn[:,2] + f_spring_2[:,2]*f_spring_2_dirn[:,2] +f_spring_3[:,2]*f_spring_3_dirn[:,2] 
                        spring_force_positon_tensor_xz=f_spring_1[:,0]*f_spring_1_dirn[:,2] + f_spring_2[:,0]*f_spring_2_dirn[:,2] +f_spring_3[:,0]*f_spring_3_dirn[:,2] 
                        spring_force_positon_tensor_xy=f_spring_1[:,0]*f_spring_1_dirn[:,1] + f_spring_2[:,0]*f_spring_2_dirn[:,1] +f_spring_3[:,0]*f_spring_3_dirn[:,1] 
                        spring_force_positon_tensor_yz=f_spring_1[:,1]*f_spring_1_dirn[:,2] + f_spring_2[:,1]*f_spring_2_dirn[:,2] +f_spring_3[:,1]*f_spring_3_dirn[:,2] 
                        
                            
                        np_array_spring_pos_tensor=np.array([spring_force_positon_tensor_xx,
                                                            spring_force_positon_tensor_yy,
                                                            spring_force_positon_tensor_zz,
                                                            spring_force_positon_tensor_xz,
                                                            spring_force_positon_tensor_xy,
                                                            spring_force_positon_tensor_yz, 
                                                                ])
                    

                        spring_force_positon_array[j,:,m,:]= np_array_spring_pos_tensor.T
                        area_vector_array[j,:,:]=np.cross(ell_1,ell_2,axisa=2,axisb=2)


        interest_vectors_mean=np.mean(np.sqrt(np.sum(interest_vectors_array**2,axis=4)),axis=0)    
        lgf_mean=np.mean(log_file_array,axis=0)    
        pol_velocities_mean=np.mean(pol_velocities_array_,axis=0)
        pol_positions_mean=np.mean(pol_positions_array_,axis=0)
        ph_velocities_mean=np.mean(ph_velocities_array_,axis=0)
        ph_positions_mean=np.mean(ph_positions_array_,axis=0)
        area_vector_mean=np.mean(area_vector_array,axis=0)
       # conform_tensor_mean=np.mean( conform_tensor_array,axis=0)
        spring_force_positon_mean=np.mean(np.mean(spring_force_positon_array,axis=0),axis=1)


        log_file_tuple=log_file_tuple+(lgf_mean,)
        pol_velocities_tuple=pol_velocities_tuple+(pol_velocities_array_,)
        pol_positions_tuple=pol_positions_tuple+(pol_positions_array_,)
        ph_velocities_tuple=ph_velocities_tuple+(ph_velocities_array_,)
        ph_positions_tuple=ph_positions_tuple+(ph_positions_array_,)
        area_vector_tuple=area_vector_tuple+(area_vector_array,)
        #conform_tensor_tuple=conform_tensor_tuple+(conform_tensor_mean,)
        spring_force_positon_tensor_tuple=spring_force_positon_tensor_tuple+(spring_force_positon_mean,)
        interest_vectors_tuple=interest_vectors_tuple+(interest_vectors_mean,)
        count+=1

#%%

#NOTE need to complete this code 
from scipy.optimize import curve_fit
COM_velocity_tuple=()
COM_position_tuple=()
erate_velocity_tuple=()

count=0
box_error_count=0
for i in range(erate.size):
    i_=(count*j_)
    row_count=pol_velocities_tuple[i].shape[1]
    indv_velocity_array=np.zeros((j_,row_count,10,3,3))
    indv_position_array=np.zeros((j_,row_count,10,3,3))
    erate_velocity_array=np.zeros(((j_,row_count,10,3))) # only need x component
    #averaged_z_array=np.zeros(((row_count,1)))
   
    # change this to each particle instead of com 
    indv_velocity_array[:,:,:,:,:]=pol_velocities_tuple[i]
    indv_position_array[:,:,:,:,:]=pol_positions_tuple[i]
    erate_velocity_array[:,:,:,:]=indv_position_array[:,:,:,:,2]*erate[i] # only need x component
    
    # for a in range(j_):
    #     for b in range(10):
    #         for c in range(3):
    #             x= indv_velocity_array[a,:,b,c,0]
    #             y=erate_velocity_array[a,:,b,c]

    #             plt.scatter(x,y)
    # plt.show()



    # note i think the points that deviate are potentially to do with periodic boundary conditions 
    # if it is has had its position corrected the velocity may not be consistent 
    # could start outputing images in hdf5 files 

   
    
    x=np.ravel(indv_velocity_array[:,:,:,:,0])
    y= np.ravel(erate_velocity_array)
    cutoff=int(np.round(0.3*x.shape[0]))
    plt.scatter(x[cutoff:],y[cutoff:])
    plt.show()
    

    popt,pcov=curve_fit(linearfunc,x,y)
    plt.plot(x,(popt[0]*(x)+popt[1]))
    print(popt[0])
    
   
    count+=1

#%% temp check 
column=5
final_temp=np.zeros((erate.size))
mean_temp_array=np.zeros((erate.size))

for i in range(erate.size):
        
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
plt.scatter(erate,mean_temp_array)
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
for i in range(erate.size):
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

#%%
for i in range(erate.size):
    for j in range(0,3):
        plt.plot(strainplot_tuple[i],spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
    
    
    plt.show()

                       




# %%
cutoff_ratio=0.2
end_cutoff_ratio=1
folder="N_1_plots"

N_1_mean=np.zeros((erate.size))
for i in range(erate.size):

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

plt.scatter((erate[:]),N_1_mean[:])
# popt,pcov=curve_fit(linearfunc,erate[:],N_1_mean[:])

# plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))

# popt,pcov=curve_fit(quadfunc,erate[:],N_1_mean[:])
# plt.plot(erate[:],(popt[0]*(erate[:]**2)))  
plt.xscale('log')
plt.yscale('log')
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
for i in range(erate.size):
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


plt.scatter((erate[:]),N_2_mean[:])

# popt,pcov=curve_fit(linearfunc,erate[:],N_2_mean[:])

# plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))

plt.ylabel("$N_{2}$")
plt.xlabel("$\dot{\gamma}$")
plt.show()

#%% N_1/N_2

plt.scatter((erate[:]),N_1_mean,label="$N_{1}$" ,marker='x')
plt.scatter((erate[:]),(-4*N_2_mean),label="$-4N_{2}$" ,marker='o')
# popt,pcov=curve_fit(linearfunc,erate[:],N_2_mean[j,:])
#plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))
# popt,pcov=curve_fit(quadfunc,erate[:],N_2_mean[j,:])
# plt.plot(erate[:],(popt[0]*(erate[:]**2)))  
#plt.ylabel("$N_{2}$")
plt.xlabel("$\dot{\gamma}$")
plt.xscale('log')
plt.yscale('log')
plt.legend()


plt.show()




#%%
cutoff_ratio=0.1
end_cutoff_ratio=1
xz_shear_stress_mean=np.zeros((erate.size))
xy_shear_stress_mean=np.zeros((erate.size))
yz_shear_stress_mean=np.zeros((erate.size))
for i in range(erate.size):
   
   
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
plt.scatter(erate[:],xz_shear_stress_mean[:])
plt.axhline(np.mean(xz_shear_stress_mean[:]))
plt.xscale('log')
plt.yscale('log')
#plt.ylim(-5,2)
plt.show()

#%%
plt.scatter(erate[:],xz_shear_stress_mean[:]/erate[:])
#plt.axhline(np.mean(xz_shear_stress_mean[:]))
plt.xscale('log')
plt.yscale('log')
#plt.ylim(-5,2)
plt.show()
#%%
plt.scatter(erate[:],yz_shear_stress_mean[:])
plt.axhline(np.mean(yz_shear_stress_mean[:]))
#plt.ylim(-5,2)
plt.show()


#%%
plt.scatter(erate[:],xy_shear_stress_mean[:])
plt.axhline(np.mean(xy_shear_stress_mean[:]))
#plt.ylim(-5,2)
plt.show()

      
# %%
with h5.File("langevinrun_no999_hookean_flat_elastic_889634_1_100_0.03633_0.005071624521210362_1000_1000_5634000_0.2_gdot_0.07_BK_10000_K_2000_phantom_after_rotation.h5",'r') as f_ph:
        print("phantom shape",f_ph['particles']['phantom']['position']['value'].shape)


# %%
