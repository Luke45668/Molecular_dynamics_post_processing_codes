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
# plt.rcParams.update(plt.rcParamsDefault)
# plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import mmap
import h5py as h5
import math as m 

import seaborn as sns

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

path_2_log_files='/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/run_868387_high_T_iteration_moreoutputs'


# erate=np.flip(np.array([100,50,25,10,5,1,0.9,0.7,0.5,0.2,0.1,0.09,0.08,
#                 0.07,0.06,0.05,0.04,
#                 0.03,0.0275,0.025,0.0225,
#                 0.02,0.0175,0.015,0.0125,
#                 0.01,0.0075,0.005,0.0025,
#                 0.001,0.00075,0.0005]))

# no_timesteps=np.flip(np.array([       4000,      8000,     16000,     39000,     79000,    394000,
#           438000,    563000,    789000,   1972000,   3944000,   4382000,
#          4929000,   5634000,   6573000,   7887000,   9859000,  13145000,
#         14340000,  15774000,  17527000,  19718000,  22534000,  26290000,
#         31548000,  39435000,  52580000,  78870000, 157740000, 394351000,
#        525801000, 788702000]))

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
thermo_freq=10000
dump_freq=10000
lgf_row_count=np.ceil((no_timesteps/thermo_freq )).astype("int")
dp_row_count=np.ceil((no_timesteps/dump_freq)).astype("int")

thermo_vars='         KinEng         PotEng        c_myTemp        c_bias         TotEng    '
j_=10
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
strain_total=np.flip(np.array([400,400,400,400,400,400,400,400,400,800,800,800,800,800,800,
              800,800,800,800,800,800,400,400,400,400,400,400,400]))
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
for i in range(erate.size):
     i_=(count*j_)
     print("i_",i_)
     with h5.File(realisation_name_h5_after_sorted_final_pol[i_],'r') as f_check:
         
        
        outputdim_hdf5=f_check['particles']['small']['velocity']['value'].shape[0]
        outputdim_log=log2numpy_reader(realisation_name_log_sorted_final[i_],
                                                            path_2_log_files,
                                                            thermo_vars).shape[0]
        dump_freq=int(realisation_name_h5_after_sorted_final_pol[i_].split('_')[10])
        
        log_file_array=np.zeros((j_,outputdim_log,6))
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

                    # plt.plot(ell_1[r1:r2,1])
                    # plt.title("$|\ell_{1}|,y$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # plt.plot(ell_2[r1:r2,1])
                    # plt.title("$|\ell_{2}|,y$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # plt.plot(ell_1[r1:r2,2])
                    # plt.title("$|\ell_{1}|,z$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # plt.plot(ell_2[r1:r2,2])
                    # plt.title("$|\ell_{2}|,z$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # # plt.plot(ell_1[:,1])
                    # # plt.title("$|\ell_{1}|,y$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # plt.plot(ell_2[:,1])
                    # plt.title("$|\ell_{2}|,y$")
                    # plt.xlabel("output count")
                    # plt.show

                    # plt.plot(ell_1[:,2])
                    # plt.title("$|\ell_{1}|,z$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # plt.plot(ell_2[:,2])
                    # plt.title("$|\ell_{2}|,z$")
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


                    # plt.plot(phant_1[r1:r2,1])
                    # plt.title("$|ph_{1}|,y$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # plt.plot(phant_2[r1:r2,1])
                    # plt.title("$|ph_{2}|,y$")
                    # plt.xlabel("output count")
                    # plt.show()


                    # plt.plot(phant_3[r1:r2,1])
                    # plt.title("$|ph_{3}|,y$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # plt.plot(phant_1[r1:r2,2])
                    # plt.title("$|ph_{1}|,z$")
                    # plt.xlabel("output count")
                    # plt.show()

                    # plt.plot(phant_2[r1:r2,2])
                    # plt.title("$|ph_{2}|,z$")
                    # plt.xlabel("output count")
                    # plt.show()


                    # plt.plot(phant_3[r1:r2,2])
                    # plt.title("$|ph_{3}|,z$")
                    # plt.xlabel("output count")
                    # plt.show()

                #     fig = plt.figure()
                #     ax = fig.add_subplot(projection='3d')
                #     ax.plot(particle_array[0,:], particle_array[1,:], particle_array[2,:])
                #    # ax.scatter(particle_array[3,:], particle_array[4,:], particle_array[5,:])
                #     plt.show()

                









                    
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
                    
                    
        interest_vectors_mean=np.mean(np.sqrt(np.sum(interest_vectors_array**2,axis=3)),axis=0)    
        # need to take magnitude before averaging 
                            
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

#%% checking velocity profile matches

for i in range(erate.size):
    plt.plot( compare_vel_to_profile_tuple[i][:,0]) 
    plt.show()

#%% check conform tesnor 
# check this calculation is correct 
viscoelastic_stress_tuple=()
total_energy_tuple=()
labels_stress=["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{zx}$",
               "$\sigma_{yx}$",
                "$\sigma_{yz}$",
                 "$\sigma_{zy}$",]
for i in range(erate.size):
    trace=np.sum(conform_tensor_tuple[i][:,0:3], axis=1)
    total_energy=np.sum(area_vector_tuple[i]**2 ,axis=1)
    rows=conform_tensor_tuple[i].shape[0]
    trace_matrix=np.zeros((rows,9))
    trace_matrix[:,0:3]=np.tile(trace,(3,1)).T
    viscoelastic_stress= trace_matrix-conform_tensor_tuple[i]
    
    viscoelastic_stress_tuple=viscoelastic_stress_tuple+(viscoelastic_stress,)
    total_energy_tuple=total_energy_tuple+(total_energy,)
    for j in range(2,3):
     plt.plot(viscoelastic_stress[:,j], label=labels_stress[j]+", $\dot{\gamma}="+str(erate[i])+"$")
     plt.legend()
    plt.show()

#%% plotting total energy 

for i in range(erate.size):
    plt.plot(K*total_energy_tuple[i]*0.5, label="$\dot{\gamma}="+str(erate[i])+"$")
    plt.legend()
plt.show()

#%% plotting 

#%%now do computes on mean files   

# COM compute 
particle_velocity_tuple=()
COM_velocity_tuple=()
COM_position_tuple=()
erate_velocity_tuple=()
particle_position_tuple=()
erate_velocity_tuple=()
averaged_z_tuple=()

count=0
box_error_count=0
for i in range(erate.size):
    i_=(count*j_)
    row_count=pol_velocities_tuple[i].shape[0]
    COM_velocity_array=np.zeros((row_count,3))
    COM_position_array=np.zeros((row_count,3))
    erate_velocity_array=np.zeros(((row_count,1))) # only need x component
    #averaged_z_array=np.zeros(((row_count,1)))
   
    # change this to each particle instead of com 

    for j in range(row_count):
       
        COM_velocity_array[j,:]=np.mean(pol_velocities_tuple[i][j,:,:], axis=0)
        COM_position_array[j,:]=np.mean(pol_positions_tuple[i][j,:,:], axis=0)
        erate_velocity_array[j,0]=COM_position_array[j,2]*erate[i] # only need x component


   
    COM_velocity_tuple=COM_velocity_tuple+(COM_velocity_array,)
    COM_position_tuple=COM_position_tuple+(COM_position_array,)
    erate_velocity_tuple=erate_velocity_tuple+(erate_velocity_array,)
    count+=1
    
#%% Checking centre of mass matches 



COM_mean_vel=np.zeros((erate.size))
erate_mean_vel=np.zeros((erate.size))
for i in range(erate.size):
    COM_mean_vel[i]=np.mean(COM_velocity_tuple[i][:,0])
    erate_mean_vel[i]=np.mean(erate_velocity_tuple[i][:,0])


    popt,pcov=curve_fit(linearfunc,COM_mean_vel,erate_mean_vel)
    plt.plot(COM_mean_vel,(popt[0]*(COM_mean_vel)+popt[1]))
    plt.scatter(COM_mean_vel,erate_mean_vel,label="gradient="+str(sigfig.round(popt[0],sigfigs=3))+",intercept="+str(sigfig.round(popt[1],sigfigs=3)))
    plt.xlabel("$v_{x,COM}$")
    plt.ylabel("$v_{x,\dot{\gamma}}$")
    plt.legend()
    print("gradient=",sigfig.round(popt[0],sigfigs=3))
    print("intercept=",sigfig.round(popt[1]))
    #  plt.axhline(np.mean(COM_velocity_tuple[i][:,0]/erate_velocity_tuple[i][:,0]))
        
plt.show()








#%% apply window averaging 
  


def window_averaging_pre_av(i,window_size,input_tuple,array_size,outdim3):
    
    output_array_final=np.zeros((array_size,outdim3))
    
    input_array=input_tuple[i]
    df = pd.DataFrame(input_array)
    output_dataframe=df.rolling(window_size,axis=0).mean()
    output_array_temp=output_dataframe.to_numpy()

    #print(output_array_temp)
    non_nan_size=int(np.count_nonzero(~np.isnan(output_array_temp))/outdim3)
    print("non_nan_size", non_nan_size)
    output_array_final=output_array_temp

    return output_array_final, non_nan_size

spring_force_positon_tensor_tuple_wa=()
COM_velocity_tuple_wa=()
COM_position_tuple_wa=()
erate_velocity_tuple_wa=()
viscoelastic_stress_tuple_wa=()
nan_size=np.zeros((erate.size))
window_size=1  #50 made n2 look good

for i in range(erate.size):
    array_size=spring_force_positon_tensor_tuple[i].shape[0]
    spring_force_positon_tensor_tuple_wa=  spring_force_positon_tensor_tuple_wa+(window_averaging_pre_av(i,
                                                                                window_size,
                                                                                spring_force_positon_tensor_tuple,
                                                                                array_size,
                                                                                6)[0],)
      
    array_size=COM_velocity_tuple[i].shape[0]
    COM_velocity_tuple_wa= COM_velocity_tuple_wa+ (window_averaging_pre_av(i,
                                                                window_size,
                                                                COM_velocity_tuple,
                                                                array_size,
                                                                3)[0],)
    array_size=COM_position_tuple[i].shape[0]
    COM_position_tuple_wa=COM_position_tuple_wa+(window_averaging_pre_av(i,
                                                                window_size,
                                                                COM_position_tuple,
                                                                array_size,
                                                                3)[0],)
    array_size=erate_velocity_tuple[i].shape[0]
    erate_velocity_tuple_wa=erate_velocity_tuple_wa+( window_averaging_pre_av(i,
                                                                window_size,
                                                                erate_velocity_tuple,
                                                                array_size,
                                                                1)[0],)
    # array_size=viscoelastic_stress_tuple[i].shape[0]
    # viscoelastic_stress_tuple_wa=viscoelastic_stress_tuple_wa+( window_averaging_pre_av(i,
    #                                                             window_size,
    #                                                             viscoelastic_stress_tuple,
    #                                                             array_size,
    #                                                             9)[0],)
    
    nan_size[i]=array_size- window_averaging_pre_av(i,
                                                window_size,
                                                erate_velocity_tuple,
                                                array_size,
                                                1)[1]










     

 #%% plots   
erate=np.flip(np.array([1,0.9,0.7,0.5,0.2,0.1,0.09,0.08,
                0.07,0.06,0.05,0.04,
                0.03,0.0275,0.025,0.0225,
                0.02,0.0175,0.015,0.0125,
                0.01,0.0075,0.005,0.0025,
                0.001,0.00075,0.0005]))
# strain_total=np.flip(np.array([400,400,400,400,400,400,400,400,400,800,800,800,800,800,800,
#               800,800,800,800,800,800,400,400,400,400,400,400])) #NOTE finish this so there can be different amounts of strain 
strainplot_tuple=()
for i in range(erate.size):
     strain_unit=strain_total/lgf_row_count[i]
     strain_plotting_points= np.linspace(0,strain_total[i],log_file_tuple[i].shape[0])
   
     strainplot_tuple=strainplot_tuple+(strain_plotting_points,)  
     print(strainplot_tuple[i].size)

def strain_plotting_points(total_strain,points_per_iv):
     #points_per_iv= number of points for the variable measured against strain 
     strain_unit=total_strain/points_per_iv
     strain_plotting_points=np.arange(0,total_strain,strain_unit)
     return  strain_plotting_points



#%%
folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=4
final_temp=np.zeros((erate.size))
for i in range(erate.size):
     
    plt.plot(strainplot_tuple[i][:],log_file_tuple[i][:,column])
    final_temp[i]=np.mean(log_file_tuple[i][:,column])
    
    mean_temp=np.mean(log_file_tuple[i][:,column])
    plt.axhline(np.mean(log_file_tuple[i][:,column]))
    plt.ylabel("$T$", rotation=0)
    plt.xlabel("$\gamma$")
    print(mean_temp)

    plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
    plt.show()


#%%
plt.scatter(erate[:],final_temp)
plt.ylabel("$T$", rotation=0)
plt.xlabel('$\dot{\gamma}$')

plt.savefig("temp_vs_gdot_damp_"+str(damp)+"_tstrain_"+str(strain_total)+"_.pdf",dpi=1200,bbox_inches='tight')


plt.show()


      


#%%
strainplot_tuple=()
for i in range(erate.size):
     #strain_unit=strain_total/lgf_row_count[i]
     strain_plotting_points= np.linspace(0,strain_total[i], spring_force_positon_tensor_tuple[i].shape[0])
   
     strainplot_tuple=strainplot_tuple+(strain_plotting_points,)  
     print(strainplot_tuple[0].size)

# compute erate prediction 


# for i in range(erate.size):
      
#       #plt.plot(COM_velocity_tuple[i][:-1,0],label="COM",linestyle=':')
#       #print(np.mean(COM_velocity_tuple[i][:,0]))
#      # plt.axhline(np.mean(COM_velocity_tuple[i][:-1,0]), label="COM mean",linestyle=':')
#       plt.plot(strainplot_tuple[i][:-1],erate_velocity_tuple[i][:-1,0],label="erate prediction",linestyle='--')
#       plt.axhline(np.mean(erate_velocity_tuple[i][:-1,0]), label="erate mean",linestyle=':')
#       #print("error:",np.mean(erate_velocity_tuple[i][:,0])-np.mean(COM_velocity_tuple[i][:,0]))
#       plt.legend()
#       plt.show()
COM_mean_vel=np.zeros(())
for i in range(erate.size):
      COM_mean_vel
      
      #plt.plot(strainplot_tuple[i][:-1],COM_velocity_tuple[i][:-1,0],label="COM",linestyle='--')
      #print(np.mean(COM_velocity_tuple[i][:,0]))

      plt.axhline(np.mean(COM_velocity_tuple[i][:-1,0])/np.mean(erate_velocity_tuple[i][:-1,0]), label="COM mean",linestyle=':',color='blueviolet')
      plt.plot(erate[:])
      #plt.plot(erate_velocity_tuple[i][:-1,0],label="erate prediction",linestyle='--')
      #plt.axhline(np.mean(erate_velocity_tuple[i][:-1,0]), label="erate mean",linestyle='--',color='black')
      #print("error:",np.mean(erate_velocity_tuple[i][:,0])-np.mean(COM_velocity_tuple[i][:,0]))
      #plt.legend()
plt.show()


#%% spring_extensions
# still some spuriously large values of extension
labels_interest=["$\ell_{1}$",
               "$\ell_{2}$",
               "$sp_{1}$",
               "$sp_{2}$",
               "$sp_{3}$"]
for i in range(erate.size):
    for j in range(2,3):
        spring_extension=interest_vectors_tuple[i][:,j]-eq_spring_length
        spring_extension[np.abs(spring_extension)>0.5]=0
        #plt.plot(strainplot_tuple[i],spring_extension, label=labels_interest[j])
        #plt.hist(spring_extension,density=True)
        sns.kdeplot(spring_extension,label="$\dot{\gamma}="+str(erate[i])+"$")
        plt.ylabel("$\Delta x$", rotation=0)
        plt.title("$\dot{\gamma}="+str(erate[i])+"$")
        #plt.xlabel("$\gamma$")
        plt.legend()
plt.show()

for i in range(erate.size):
    for j in range(0,1):
        spring_extension=interest_vectors_tuple[i][:,j]-3
        spring_extension[np.abs(spring_extension)>0.5]=0
        #plt.plot(strainplot_tuple[i],spring_extension, label=labels_interest[j])
        #plt.hist(spring_extension,density=True)
        sns.kdeplot(spring_extension,label="$\dot{\gamma}="+str(erate[i])+"$")
        plt.ylabel("$\Delta x$", rotation=0)
        plt.title("$\dot{\gamma}="+str(erate[i])+"$")
        #plt.xlabel("$\gamma$")
        plt.legend()
plt.show()




#%% look at internal stresses
folder="stress_tensor_plots"
folder_check_or_create(path_2_log_files,folder)
labels_stress=["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{yz}$"]

for i in range(erate.size):
    for j in range(3,6):
        plt.plot(strainplot_tuple[i],spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
        #plt.plot(spring_force_positon_tensor_tuple_wa[i][:,j], label=labels_stress[j]+",$\dot{\gamma}="+str(erate[i])+"$")
        #plt.plot(viscoelastic_stress_tuple_wa[i][:,j], label=labels_stress[j]+",$\dot{\gamma}="+str(erate[i])+"$")
        plt.xlabel("$\gamma$")

        plt.legend()
    plt.show()

#%% normal stress
#NOTE analyse the frequencies of these plots there is something periodic 

for i in range(erate.size):
     for j in range(0,3):
       # popt,pcov=curve_fit(sinusoidfit,strainplot_tuple[i],spring_force_positon_tensor_tuple[i][:,j])
        plt.plot(strainplot_tuple[i],spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
        #plt.plot(spring_force_positon_tensor_tuple[i][:,j], label=labels_stress[j])
      #  plt.plot(strainplot_tuple[i], popt[0]*np.sin(popt[1]*strainplot_tuple[i])+popt[2])
        #plt.plot(spring_force_positon_tensor_tuple_wa[i][:,j], label=labels_stress[j]+",$\dot{\gamma}="+str(erate[i])+"$")
        #plt.plot(viscoelastic_stress_tuple_wa[i][:,j], label=labels_stress[j]+",$\dot{\gamma}="+str(erate[i])+"$")
        #plt.ylim(-8000,10) 
        plt.title("$\dot{\gamma}="+str(erate[i])+"$")
        plt.legend()
        plt.xlabel("$\gamma$")
     plt.show()

#%%
cutoff_ratio=0.1
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

plt.scatter((erate[:]),N_1_mean[:])
# popt,pcov=curve_fit(linearfunc,erate[:],N_1_mean[:])

# plt.plot(erate[:],(popt[0]*(erate[:])+popt[1]))

# popt,pcov=curve_fit(quadfunc,erate[:],N_1_mean[:])
# plt.plot(erate[:],(popt[0]*(erate[:]**2)))  

plt.ylabel("$N_{1}$")
plt.xlabel("$\dot{\gamma}$")
plt.show()
 #%%
folder="N_2_plots"
# cutoff_ratio=0.5
# end_cutoff_ratio=0.7
cutoff_ratio=0.1
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
#%%
folder="shear_stress_plots"
cutoff_ratio=0.2
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

#%% mean shear stress plots
plt.scatter(erate[:],xz_shear_stress_mean[:])
plt.axhline(np.mean(xz_shear_stress_mean[:]))
#plt.ylim(-5,2)
plt.show()

#%%
plt.scatter(erate[:],xz_shear_stress_mean[:]/erate[:])
plt.axhline(np.mean(xz_shear_stress_mean[:]))
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




#%%
bin_count = int(np.ceil(np.log2((j_))) + 1)# sturges rule


# I think this is only looking at one shear rate at a time 
# folder="theta_and_phi_histograms"
# folder_check_or_create(filepath,folder)  
area_vector_spherical_tuple=()
for i in range(erate.size):
     
        spherical_coordinates_area_vector=np.zeros((area_vector_tuple[i].shape[0],3))
        x=area_vector_tuple[i][:,0]
        y=area_vector_tuple[i][:,1]
        z=area_vector_tuple[i][:,2]
        # for l in range(internal_stiffness.size):
        for j in range(z.shape[0]):
            if z[j]<0:
                z[j]=-1*z[j]
                y[j]=-1*y[j]
                x[j]=-1*x[j]

            else:
                continue

        # x[z<0]=-1*x
        # y[z<0]=-1*y
        # z[z<0]=-1*z

        # radial coord
        spherical_coordinates_area_vector[:,0]=np.sqrt((x**2)+(y**2)+(z**2))
        # theta coord 
        spherical_coordinates_area_vector[:,1]=np.sign(y)*np.arccos(x/(np.sqrt((x**2)+(y**2))))
        # phi coord
        spherical_coordinates_area_vector[:,2]=np.arccos(z/spherical_coordinates_area_vector[:,0])


   
        # for k in range(internal_stiffness.size):
                # plot theta histogram
        pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
        pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
        plt.hist((spherical_coordinates_area_vector[:,1]))
        plt.xticks(pi_theta_ticks, pi_theta_tick_labels)
        plt.title("Azimuthal angle $\Theta$ histogram,$\dot{\gamma}="\
            +str(erate[i])+"$ and $K="+str(K)+"$")
        plt.xlabel('$\\theta$')
       # plt.tight_layout()
        #plt.savefig("theta_histogram_"+str(j_)+"_points_erate_"+str(erate[i])+"_K_"+str(internal_stiffness[k])+".pdf",dpi=1200)
        plt.show()


        pi_phi_ticks=[ 0,np.pi/4, np.pi/2,3*np.pi/4,np.pi]
        pi_phi_tick_labels=[ '0','π/4', 'π/2','3π/4' ,'π']
        frequencies_phi= np.histogram(spherical_coordinates_area_vector[:,2],bins=bin_count)[0]


                # plot phi hist

        plt.hist(spherical_coordinates_area_vector[:,2])
        plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
        plt.xlabel('$\phi$')
        plt.title("Inclination angle $\phi$ histogram,$\dot{\gamma}="\
            +str(erate[i])+"$ and $K="+str(K)+"$")
       # plt.tight_layout()
        #plt.savefig("phi_histogram_"+str(j_)+"_points_erate_"+str(erate[i])+"_K_"+str(internal_stiffness[k])+".pdf",dpi=1200)
        plt.show()


        area_vector_spherical_tuple=area_vector_spherical_tuple+(spherical_coordinates_area_vector,)


#%% save tuples
label='damp_'+str(damp)+'_K_'+str(K)+'_'
import pickle as pck
os.chdir(path_2_log_files)
#os.mkdir("tuple_results")
os.chdir("tuple_results")

with open(label+'spring_force_positon_tensor_tuple.pickle', 'wb') as f:
    pck.dump(spring_force_positon_tensor_tuple, f)

with open(label+'conform_tensor_tuple.pickle', 'wb') as f:
    pck.dump(conform_tensor_tuple, f)

with open(label+'interest_vectors_tuple.pickle','wb') as f:
    pck.dump(interest_vectors_tuple,f)

# with open(label+'viscoelastic_stress_tuple.pickle', 'wb') as f:
#     pck.dump(viscoelastic_stress_tuple, f)

with open(label+'COM_velocity_tuple.pickle', 'wb') as f:
    pck.dump(COM_velocity_tuple, f)

#with open(label+'COM_position_tuple.pickle', 'wb') as f:
    #pck.dump(COM_position_tuple, f)

with open(label+'erate_velocity_tuple.pickle', 'wb') as f:
    pck.dump(erate_velocity_tuple, f)

with open(label+'log_file_tuple.pickle', 'wb') as f:
    pck.dump(log_file_tuple, f)

with open(label+'pol_velocities_tuple.pickle', 'wb') as f:
    pck.dump( pol_velocities_tuple, f)

with open(label+'pol_positions_tuple.pickle', 'wb') as f:
    pck.dump(  pol_positions_tuple, f)

with open(label+'ph_velocities_tuple.pickle', 'wb') as f:
    pck.dump( ph_velocities_tuple, f)

with open(label+'ph_positions_tuple.pickle', 'wb') as f:
    pck.dump(  ph_positions_tuple, f)


with open(label+"area_vector_spherical_tuple.pickle", 'wb') as f:
    pck.dump(area_vector_spherical_tuple,f)
# %%
