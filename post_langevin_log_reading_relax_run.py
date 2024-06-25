##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file processes the log files from brownian dynamics simulations 

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
plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import mmap
import h5py as h5
import math as m 



path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
os.chdir(path_2_post_proc_module)

from log2numpy import *
from dump2numpy import *
import glob 
from post_MPCD_MP_processing_module import *
import pickle as pck

#%% 
# damp 0.1 seems to work well, need to add window averaging, figure out how to get imposed shear to match velocity of particle
# neeed to then do a huge run 
damp=0.03633

strain_total=400

path_2_log_files='/Users/luke_dev/Documents/simulation_test_folder/relaxation_tests'




erate=np.flip(np.array([1,0.9,0.7,0.5,0.35,0.2,0.1,0.09,0.08,
                0.07,0.06,0.05,0.04,
                0.03,0.0275,0.025,0.0225,
                0.02,0.0175,0.015,0.0125,
                0.01,0.0075,0.005,0.0025,
                0.001,0.00075,0.0005]))

# #600 strain 


no_timesteps=np.flip(np.array([   394000,
          438000,    563000,    789000, 1127000,  1972000,   3944000,   4382000,
         4929000,   5634000,   6573000,   7887000,   9859000,  13145000,
        14340000,  15774000,  17527000,  19718000,  22534000,  26290000,
        31548000,  39435000,  52580000,  78870000, 157740000, 394351000,
       525801000, 788702000]))


no_timesteps=no_timesteps*2

# erate=np.flip(np.array([1,0.9,0.7,0.5]))

# # #600 strain 


# no_timesteps=np.flip(np.array([   394000,
#           438000,    563000,    789000]))


#no_timesteps=np.flip(np.array([       4000,      8000,     16000,     39000,     79000,    394000]))
#200 strain 
# no_timesteps=np.flip(np.array([26290000,  28680000,  31548000,  35053000,  39435000,  45069000,
#         52580000,  63096000,  78870000, 105160000,157740000,  315481000,  788702000, 1051603000, 1577404000]))

# no_timesteps=np.array([52580000,  57360000,  63096000,  70107000,  78870000,  90137000,
#        105160000, 126192000, 157740000, 210321000])
thermo_freq=100
dump_freq=100
lgf_row_count=np.ceil((no_timesteps/thermo_freq )).astype("int")
dp_row_count=np.ceil((no_timesteps/dump_freq)).astype("int")

thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
j_=1
K=500
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
damp_ratio=mass_pol/damp

#NOTE: the damping force may be very large compared to the spring force at high shear rates, which could be cancelling 
# out the spring forces.

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
  
def window_averaging(i,window_size,input_tuple,array_size,outdim1,outdim3):
    
    output_array_final=np.zeros((outdim1,array_size[i],outdim3))
    for k in range(outdim1):
        input_array=input_tuple[i][k,:,:]
        df = pd.DataFrame(input_array)
        output_dataframe=df.rolling(window_size,axis=0).mean()
        output_array_temp=output_dataframe.to_numpy()

        #print(output_array_temp)
        non_nan_size=int(np.count_nonzero(~np.isnan(output_array_temp))/outdim3)
        print("non_nan_size", non_nan_size)
        output_array_final[k,:,:]=output_array_temp

    return output_array_final, non_nan_size

from scipy.optimize import curve_fit
 
def quadfunc(x, a):

    return a*(x**2)

def linearfunc(x,a,b):
    return (a*x)+b 

def round_down_to_even(f):
    return m.floor(f / 2.) * 2

def sinusoidfit(a,b,c,x):
    return a*np.sin(b*x)+c 

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

     



#%%read the  log files 
#no_timesteps=np.array([ 2958000,  5915000, 11831000, 23661000, 26290000, 29576000,
 #      33802000, 39435000, 47322000, 59153000])

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
#need to fix this issue where the arrays are all slightly different sizes by one or two 
for i in range(16,17):
     i_=(count*j_)
     print("i_",i_)
     with h5.File(realisation_name_h5_after_sorted_final_pol[i_],'r') as f_check:
         
        
        outputdim_hdf5=f_check['particles']['small']['velocity']['value'].shape[0]
        outputdim_log=log2numpy_reader(realisation_name_log_sorted_final[i_],
                                                            path_2_log_files,
                                                            thermo_vars).shape[0]
        dump_freq=int(realisation_name_h5_after_sorted_final_pol[i_].split('_')[10])
        
        log_file_array=np.zeros((j_,outputdim_log,7))
        pol_velocities_array=np.zeros((j_,outputdim_hdf5,3,3))
        pol_positions_array=np.zeros((j_,outputdim_hdf5,3,3))
        ph_velocities_array=np.zeros((j_,outputdim_hdf5,3,3))
        ph_positions_array=np.zeros((j_,outputdim_hdf5,3,3))
        area_vector_array=np.zeros((j_,outputdim_hdf5,3))
        conform_tensor_array=np.zeros((j_,outputdim_hdf5,9))
        COM_velocity_array=np.zeros((j_,outputdim_hdf5,3))
        COM_position_array=np.zeros((j_,outputdim_hdf5,3))
        erate_velocity_array=np.zeros(((j_,outputdim_hdf5,1)))

        # need to compute the total strain based on the number of outputs, not 
        # just assume it reaches 400
     
     
       
        #NOTE check you have the correct total strain or these corrections dont work
        spring_force_positon_array=np.zeros((j_,outputdim_hdf5,6))
       
        interest_vectors_array=np.zeros((j_,outputdim_hdf5,5,3))
        compare_vel_to_profile=np.zeros((j_,outputdim_hdf5,3))


        for j in range(j_):
                j_index=j+(j_*count)

                # need to get rid of print statements in log2numpy 
                # print(realisation_name_log_sorted_final[j_index])
                print(j_index)
                log_file_array[j,:,:]=log2numpy_reader(realisation_name_log_sorted_final[j_index],
                                                            path_2_log_files,
                                                            thermo_vars)
                #  print(realisation_name_h5_after_sorted_final_pol[j_index])
                #  print(j_index)
                with h5.File(realisation_name_h5_after_sorted_final_pol[j_index],'r') as f_c:
                
                  with h5.File(realisation_name_h5_after_sorted_final_phantom[j_index],'r') as f_ph:

                    

                    #for l in range(0,outputdim):
                    # this isnt working properly !
                    # I think we need to compute row by t
                    pol_velocities_array[j,:,:,:]=f_c['particles']['small']['velocity']['value'][:]
                    pol_positions_array[j,:,:,:]=f_c['particles']['small']['position']['value'][:]
                    ph_velocities_array[j,:,:,:]=f_ph['particles']['phantom']['velocity']['value'][:]
                    ph_positions_array[j,:,:,:]=f_ph['particles']['phantom']['position']['value'][:]
                    



                    


                    
                    # erate_velocity_array[j,:,:,:]=COM_position_array[j,2]*erate[i] 
                    
                    ell_1=pol_positions_array[j,:,1,:]-pol_positions_array[j,:,0,:]
                    ell_2=pol_positions_array[j,:,2,:]-pol_positions_array[j,:,0,:]
                    phant_1=pol_positions_array[j,:,0,:]-ph_positions_array[j,:,1,:]
                    #f_spring_1_dirn=pol_positions_tuple[i][j,0,:]-ph_positions_tuple[i][j,1,:]
                    phant_2=pol_positions_array[j,:,1,:]-ph_positions_array[j,:,2,:]
                    #f_spring_2_dirn=pol_positions_tuple[i][j,1,:]-ph_positions_tuple[i][j,2,:]
                    phant_3=pol_positions_array[j,:,2,:]-ph_positions_array[j,:,0,:]
                    #f_spring_3_dirn=pol_positions_tuple[i][j,2,:]-ph_positions_tuple[i][j,0,:]
                    # ell_1[ell_1[:,0]>10]-=23
                    # ell_1[ell_1[:,0]<-10]+=23
                    # ell_2[ell_2[:,0]>10]-=23
                    # ell_2[ell_2[:,0]<-10]+=23
                   
                    #outputdim_hdf5
                    # r1=17251
                    # r2=17252
                    # r1=52544
                    # r2=52545

                    # r1=1851
                    # r2=1852
                   
                    r1=0
                    r2=789
                    
                   
                    for l in range(r1,r2):
                        # z correction for ell_1
                        particle_array=np.array([pol_positions_array[j,l,0,:],
                                                 pol_positions_array[j,l,1,:],
                                                 pol_positions_array[j,l,2,:],
                                                ph_positions_array[j,l,0,:],
                                                ph_positions_array[j,l,1,:],
                                                ph_positions_array[j,l,2,:]])
                        particle_COM=np.mean(particle_array,axis=0)
                       
                        particle_vel_array=np.array([pol_velocities_array[j,l,0,:],
                                                 pol_velocities_array[j,l,1,:],
                                                 pol_velocities_array[j,l,2,:],
                                                ph_velocities_array[j,l,0,:],
                                                ph_velocities_array[j,l,1,:],
                                                ph_velocities_array[j,l,2,:]])
                        
                        particle_COM_vel=np.mean(particle_vel_array,axis=0)
                        pred_velocity_profile=particle_array[:3,2]*erate[i]


                        compare_vel_to_profile[j,l,:]= particle_vel_array[:3,0]/pred_velocity_profile

                    
                       
                        
                        interest_vectors=np.array([ell_1[l,:],ell_2[l,:],phant_1[l,:],phant_2[l,:],phant_3[l,:]])
                        
                        # print("unmodified particle array ",particle_array)
                        # print("unmodified interest vectors", interest_vectors)

                        
                        
                        strain=l*dump_freq*md_step*erate[i] -\
                              np.floor(l*dump_freq*md_step*erate[i])
                        
                        
                        if strain <= 0.5:
                            tilt= (23)*(strain)
                            #tilt_test.append(tilt)
                        else:
                                
                            tilt=-(1-strain)*23
                           # tilt_test.append(tilt)
                        #print("tilt",tilt)

                        # z shift 
                        # convention shift down to lower boundary 

                        if np.any(np.abs(interest_vectors[:,2])>23/2):
                            #print("periodic anomalies detected")
                           # print("z shift performed")
                            
                           

                            for r in range(6):
                                if particle_array[r,2]>23/2:
                                    particle_array[r,:]+=np.array([-tilt,0,-23])



                        #print("post z mod particle array",particle_array)


                        ell_1_c=particle_array[1,:]-particle_array[0,:]
                        ell_2_c=particle_array[2,:]-particle_array[0,:]
                        phant_1_c=particle_array[0,:]-particle_array[4,:]
                        phant_2_c=particle_array[1,:]-particle_array[5,:]
                        phant_3_c=particle_array[2,:]-particle_array[3,:]

                        interest_vectors=np.array([ell_1_c,ell_2_c,phant_1_c,phant_2_c,phant_3_c])
                        
                      

                        # y shift 
                       

                        if np.any(np.abs(interest_vectors[:,1])>23/2):
                            #print("periodic anomalies detected")
                            y_coords=particle_array[:,1]
                            y_coords[y_coords<23/2]+=23
                            particle_array[:,1]=y_coords
                           # print("y shift performed")

                       # print("post y shift mod particle array",particle_array)
                        
                        # x shift 

                        ell_1_c=particle_array[1,:]-particle_array[0,:]
                        ell_2_c=particle_array[2,:]-particle_array[0,:]
                        phant_1_c=particle_array[0,:]-particle_array[4,:]
                        phant_2_c=particle_array[1,:]-particle_array[5,:]
                        phant_3_c=particle_array[2,:]-particle_array[3,:]

                        interest_vectors=np.array([ell_1_c,ell_2_c,phant_1_c,phant_2_c,phant_3_c])
                       

                        if np.any(np.abs(interest_vectors[:,0])>23/2):
                            #print("periodic anomalies detected")
                            # x shift convention shift to smaller side
                            
                            x_coords=particle_array[:,0]
                            z_coords=particle_array[:,2]
                            box_position_x= x_coords -(z_coords/23)*tilt
                            x_coords[box_position_x<23/2]+=23
                            particle_array[:,0]=x_coords
                           # print("x shift performed")

                       # print("post x shift mod particle array",particle_array)


                       
   

                        # #particle_array[particle_array[:,0]>23/2]-=23

                        #print("mod particle array",particle_array)
                        #recompute interest vectors
                        ell_1_c=particle_array[1,:]-particle_array[0,:]
                        ell_2_c=particle_array[2,:]-particle_array[0,:]
                        phant_1_c=particle_array[0,:]-particle_array[4,:]
                        phant_2_c=particle_array[1,:]-particle_array[5,:]
                        phant_3_c=particle_array[2,:]-particle_array[3,:]

                        interest_vectors=np.array([ell_1_c,ell_2_c,phant_1_c,phant_2_c,phant_3_c])
                        #interest_vectors[np.abs(interest_vectors)>23/2]=float('nan')
                        

                        #print("interest vectors",interest_vectors)



                        # if np.any(np.abs(interest_vectors)>23/2):
                        #     interest_vectors[interest_vectors>23/2]-=23
                        #     interest_vectors[interest_vectors<-23/2]+=23

                        ell_1[l,:]=interest_vectors[0,:]
                        ell_2[l,:]=interest_vectors[1,:]
                        phant_1[l,:]=interest_vectors[2,:]
                        phant_2[l,:]=interest_vectors[3,:]
                        phant_3[l,:]=interest_vectors[4,:]
                        interest_vectors_array[j,l,:,:]=interest_vectors



                       


                    
                        # now we can nan the 



                        #NOTE write in test here 

                        if np.any(np.abs(interest_vectors)>23/2):
                            # print(interest_vectors)
                            # print("anomalies still present")

                    
                            breakpoint
                    # correction after shear 
                    r1=789
                    r2=outputdim_hdf5
                    
                   
                    for l in range(r1,r2):
                        # z correction for ell_1
                        particle_array=np.array([pol_positions_array[j,l,0,:],
                                                 pol_positions_array[j,l,1,:],
                                                 pol_positions_array[j,l,2,:],
                                                ph_positions_array[j,l,0,:],
                                                ph_positions_array[j,l,1,:],
                                                ph_positions_array[j,l,2,:]])
                        particle_COM=np.mean(particle_array,axis=0)
                       
                        particle_vel_array=np.array([pol_velocities_array[j,l,0,:],
                                                 pol_velocities_array[j,l,1,:],
                                                 pol_velocities_array[j,l,2,:],
                                                ph_velocities_array[j,l,0,:],
                                                ph_velocities_array[j,l,1,:],
                                                ph_velocities_array[j,l,2,:]])
                        
                        particle_COM_vel=np.mean(particle_vel_array,axis=0)
                        pred_velocity_profile=particle_array[:3,2]*erate[i]


                        compare_vel_to_profile[j,l,:]= particle_vel_array[:3,0]/pred_velocity_profile

                    
                       
                        
                        interest_vectors=np.array([ell_1[l,:],ell_2[l,:],phant_1[l,:],phant_2[l,:],phant_3[l,:]])
                        
                        # print("unmodified particle array ",particle_array)
                        # print("unmodified interest vectors", interest_vectors)

                        
                       
                        #print("tilt",tilt)

                        # z shift 
                        # convention shift down to lower boundary 

                        if np.any(np.abs(interest_vectors[:,2])>23/2):
                              #print("periodic anomalies detected")
                            z_coords=particle_array[:,2]
                            z_coords[z_coords<23/2]+=23
                            particle_array[:,2]=z_coords
                           # print("y shift performed")



                        #print("post z mod particle array",particle_array)


                        ell_1_c=particle_array[1,:]-particle_array[0,:]
                        ell_2_c=particle_array[2,:]-particle_array[0,:]
                        phant_1_c=particle_array[0,:]-particle_array[4,:]
                        phant_2_c=particle_array[1,:]-particle_array[5,:]
                        phant_3_c=particle_array[2,:]-particle_array[3,:]

                        interest_vectors=np.array([ell_1_c,ell_2_c,phant_1_c,phant_2_c,phant_3_c])
                        
                      

                        # y shift 
                       

                        if np.any(np.abs(interest_vectors[:,1])>23/2):
                            #print("periodic anomalies detected")
                            y_coords=particle_array[:,1]
                            y_coords[y_coords<23/2]+=23
                            particle_array[:,1]=y_coords
                           # print("y shift performed")

                       # print("post y shift mod particle array",particle_array)
                        
                        # x shift 

                        ell_1_c=particle_array[1,:]-particle_array[0,:]
                        ell_2_c=particle_array[2,:]-particle_array[0,:]
                        phant_1_c=particle_array[0,:]-particle_array[4,:]
                        phant_2_c=particle_array[1,:]-particle_array[5,:]
                        phant_3_c=particle_array[2,:]-particle_array[3,:]

                        interest_vectors=np.array([ell_1_c,ell_2_c,phant_1_c,phant_2_c,phant_3_c])
                       

                        if np.any(np.abs(interest_vectors[:,0])>23/2):
                            #print("periodic anomalies detected")
                            # x shift convention shift to smaller side
                            
                            x_coords=particle_array[:,0]
                            z_coords=particle_array[:,2]
                            box_position_x= x_coords -(z_coords/23)*tilt
                            x_coords[box_position_x<23/2]+=23
                            particle_array[:,0]=x_coords
                           # print("x shift performed")

                       # print("post x shift mod particle array",particle_array)


                       
   

                        # #particle_array[particle_array[:,0]>23/2]-=23

                        #print("mod particle array",particle_array)
                        #recompute interest vectors
                        ell_1_c=particle_array[1,:]-particle_array[0,:]
                        ell_2_c=particle_array[2,:]-particle_array[0,:]
                        phant_1_c=particle_array[0,:]-particle_array[4,:]
                        phant_2_c=particle_array[1,:]-particle_array[5,:]
                        phant_3_c=particle_array[2,:]-particle_array[3,:]

                        interest_vectors=np.array([ell_1_c,ell_2_c,phant_1_c,phant_2_c,phant_3_c])
                        #interest_vectors[np.abs(interest_vectors)>23/2]=float('nan')
                        

                        #print("interest vectors",interest_vectors)



                        # if np.any(np.abs(interest_vectors)>23/2):
                        #     interest_vectors[interest_vectors>23/2]-=23
                        #     interest_vectors[interest_vectors<-23/2]+=23

                        ell_1[l,:]=interest_vectors[0,:]
                        ell_2[l,:]=interest_vectors[1,:]
                        phant_1[l,:]=interest_vectors[2,:]
                        phant_2[l,:]=interest_vectors[3,:]
                        phant_3[l,:]=interest_vectors[4,:]
                        interest_vectors_array[j,l,:,:]=interest_vectors



                       


                    
                        # now we can nan the 



                        #NOTE write in test here 

                        if np.any(np.abs(interest_vectors)>23/2):
                            # print(interest_vectors)
                            # print("anomalies still present")

                    
                            breakpoint
                                
                    # print(realisation_name_h5_after_sorted_final_pol[j_index])
                    # data=compare_vel_to_profile[j,r1:r2,:]
                    # data[np.abs(data)>1000]=0
                    # plt.plot(data ) 
                    # plt.show()

                    ell_1[ell_1[:,:]>23/2]-=23
                    ell_1[ell_1[:,:]<-23/2]+=23
                    ell_2[ell_2[:,:]>23/2]-=23
                    ell_2[ell_2[:,:]<-23/2]+=23
                    # ell_1[ell_1[:,1]>10]-=23
                    # ell_1[ell_1[:,1]<-10]+=23
                    # ell_2[ell_2[:,1]>10]-=23
                    # ell_2[ell_2[:,1]<-10]+=23
                    phant_1[phant_1[:,:]>23/2]-=23
                    phant_1[phant_1[:,:]<-23/2]+=23
                    # phant_1[phant_1[:,1]>10]-=23
                    # phant_1[phant_1[:,1]<-10]+=23
                  
                    phant_2[phant_2[:,:]>23/2]-=23
                    phant_2[phant_2[:,:]<-23/2]+=23
                    # phant_2[phant_2[:,1]>10]-=23
                    # phant_2[phant_2[:,1]<-10]+=23

                    phant_3[phant_3[:,:]>23/2]-=23
                    phant_3[phant_3[:,:]<-23/2]+=23


                    phant_3[phant_3[:,1]>10]-=23
                    phant_3[phant_3[:,1]<-10]+=23

                    
                    interest_vectors=np.array([ell_1,ell_2,phant_1,phant_2,phant_3])
                    # interest_vectors_array[j,:,:,:]=interest_vectors
                                
                    # print(realisation_name_h5_after_sorted_final_pol[j_index])
                    # data=compare_vel_to_profile[j,r1:r2,:]
                    # data[np.abs(data)>1000]=0
                    # plt.plot(data ) 
                    # plt.show()

                    ell_1[ell_1[:,:]>23/2]-=23
                    ell_1[ell_1[:,:]<-23/2]+=23
                    ell_2[ell_2[:,:]>23/2]-=23
                    ell_2[ell_2[:,:]<-23/2]+=23
                    # ell_1[ell_1[:,1]>10]-=23
                    # ell_1[ell_1[:,1]<-10]+=23
                    # ell_2[ell_2[:,1]>10]-=23
                    # ell_2[ell_2[:,1]<-10]+=23
                    phant_1[phant_1[:,:]>23/2]-=23
                    phant_1[phant_1[:,:]<-23/2]+=23
                    # phant_1[phant_1[:,1]>10]-=23
                    # phant_1[phant_1[:,1]<-10]+=23
                  
                    phant_2[phant_2[:,:]>23/2]-=23
                    phant_2[phant_2[:,:]<-23/2]+=23
                    # phant_2[phant_2[:,1]>10]-=23
                    # phant_2[phant_2[:,1]<-10]+=23

                    phant_3[phant_3[:,:]>23/2]-=23
                    phant_3[phant_3[:,:]<-23/2]+=23


                    phant_3[phant_3[:,1]>10]-=23
                    phant_3[phant_3[:,1]<-10]+=23

                    
                    interest_vectors=np.array([ell_1,ell_2,phant_1,phant_2,phant_3])
                    # interest_vectors_array[j,:,:,:]=interest_vectors


                









                    
                    # need to check magnitude of ell_1 and ell_2 if its huge particle is split over the boundary 
                    # need to add 23 if component is negative or subtract 23 if component is pos
                    f_spring_1_dirn=phant_1
                  
                    f_spring_1_mag=np.sqrt(np.sum((f_spring_1_dirn)**2))

                    f_spring_1=K*(f_spring_1_dirn/f_spring_1_mag)*(f_spring_1_mag-eq_spring_length)
                
                    # spring 2
                    f_spring_2_dirn=phant_2
                   
                    f_spring_2_mag=np.sqrt(np.sum((f_spring_2_dirn)**2))
                  
                    f_spring_2=K*(f_spring_2_dirn/f_spring_2_mag)*(f_spring_2_mag-eq_spring_length)
                  
                    # spring 3

                    f_spring_3_dirn=phant_3
                   
                    f_spring_3_mag=np.sqrt(np.sum((f_spring_3_dirn)**2))
                    
                    f_spring_3=K*(f_spring_3_dirn/f_spring_3_mag)*(f_spring_3_mag-eq_spring_length)

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
                   

                    spring_force_positon_array[j,:,:]= np_array_spring_pos_tensor.T
                    


                    area_vector_array[j,:,:]=np.cross(ell_1,ell_2,axisa=1,axisb=1)
                    conform_tensor_array[j,:,0]=area_vector_array[j,:,0]*area_vector_array[j,:,0]#xx
                    conform_tensor_array[j,:,1]=area_vector_array[j,:,1]*area_vector_array[j,:,1]#yy
                    conform_tensor_array[j,:,2]=area_vector_array[j,:,2]*area_vector_array[j,:,2]#zz
                    conform_tensor_array[j,:,3]=area_vector_array[j,:,0]*area_vector_array[j,:,2]#xz
                    conform_tensor_array[j,:,4]=area_vector_array[j,:,0]*area_vector_array[j,:,1]#xy
                    conform_tensor_array[j,:,5]=area_vector_array[j,:,2]*area_vector_array[j,:,0]#zx
                    conform_tensor_array[j,:,6]=area_vector_array[j,:,1]*area_vector_array[j,:,0]#yx
                    conform_tensor_array[j,:,7]=area_vector_array[j,:,1]*area_vector_array[j,:,2]#yz
                    conform_tensor_array[j,:,8]=area_vector_array[j,:,2]*area_vector_array[j,:,1]#zy
                    
                    
        interest_vectors_mean=np.mean(interest_vectors_array,axis=0)                        
        lgf_mean=np.mean(log_file_array,axis=0)    
        pol_velocities_mean=np.mean(pol_velocities_array,axis=0)
        pol_positions_mean=np.mean(pol_positions_array,axis=0)
        ph_velocities_mean=np.mean(ph_velocities_array,axis=0)
        ph_positions_mean=np.mean(ph_positions_array,axis=0)
        area_vector_mean=np.mean(area_vector_array,axis=0)
        conform_tensor_mean=np.mean( conform_tensor_array,axis=0)
        spring_force_positon_mean=np.mean(spring_force_positon_array,axis=0)
        
        compare_vel_to_profile_mean=np.mean(compare_vel_to_profile,axis=0)
        





        log_file_tuple=log_file_tuple+(lgf_mean,)
        pol_velocities_tuple=pol_velocities_tuple+(pol_velocities_mean,)
        pol_positions_tuple=pol_positions_tuple+(pol_positions_mean,)
        ph_velocities_tuple=ph_velocities_tuple+(ph_velocities_mean,)
        ph_positions_tuple=ph_positions_tuple+(ph_positions_mean,)
        area_vector_tuple=area_vector_tuple+(area_vector_mean,)
        conform_tensor_tuple=conform_tensor_tuple+(conform_tensor_mean,)
        spring_force_positon_tensor_tuple=spring_force_positon_tensor_tuple+(spring_force_positon_mean,)
        compare_vel_to_profile_tuple=compare_vel_to_profile_tuple+(compare_vel_to_profile_mean,)
        interest_vectors_tuple=interest_vectors_tuple+(interest_vectors_mean,)
        
        count+=1

#%% particle relaxation 
ell_1_mag=np.zeros((outputdim_hdf5))
ell_2_mag=np.zeros((outputdim_hdf5))
spring_1_mag=np.zeros((outputdim_hdf5))
spring_2_mag=np.zeros((outputdim_hdf5))
spring_3_mag=np.zeros((outputdim_hdf5))
for i in range(outputdim_hdf5):
   spring_1_mag[i]=np.sqrt(np.sum( interest_vectors_tuple[0][i,2,:]**2))-eq_spring_length
   spring_2_mag[i]=np.sqrt(np.sum( interest_vectors_tuple[0][i,3,:]**2))-eq_spring_length
   spring_3_mag[i]=np.sqrt(np.sum( interest_vectors_tuple[0][i,4,:]**2))-eq_spring_length
   ell_1_mag[i]=np.sqrt(np.sum( interest_vectors_tuple[0][i,0,:]**2))-3
   ell_2_mag[i]=np.sqrt(np.sum( interest_vectors_tuple[0][i,1,:]**2))-3




plt.plot(spring_1_mag)
plt.plot(spring_2_mag)
plt.plot(spring_3_mag)
plt.xlabel("$N_{\Delta t}$")
plt.ylabel("$|\Delta L_{sp}|$")
plt.show()

plt.plot(ell_1_mag)
plt.plot(ell_2_mag)
plt.xlabel("$N_{\Delta t}$")
plt.ylabel("$|\Delta \ell_{i}|$")
plt.show()
# %%
