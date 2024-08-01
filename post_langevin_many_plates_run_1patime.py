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
from fitter import Fitter, get_common_distributions, get_distributions
path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
os.chdir(path_2_post_proc_module)

from log2numpy import *
from dump2numpy import *
import glob 
from post_MPCD_MP_processing_module import *
import pickle as pck
from numpy.linalg import norm

#%%
damp=0.035
strain_total=400
# erate=np.flip(np.array([1,0.9,0.7,0.5,0.35,0.2,0.1,0.09,0.08,
#                 0.07,0.06,0.05,0.04,
#                 0.03,0.0275,0.025,0.0225,
#                 0.02,0.0175,0.015,0.0125,
#                 0.01,0.0075,0.005,0.0025,
#                 0.001,0.00075,0.0005]))
# no_timesteps=np.flip(np.array([   394000,
#           438000,    563000,    789000, 1127000,  1972000,   3944000,   4382000,
#          4929000,   5634000,   6573000,   7887000,   9859000,  13145000,
#         14340000,  15774000,  17527000,  19718000,  22534000,  26290000,
#         31548000,  39435000,  52580000,  78870000, 157740000, 394351000,
#        525801000, 788702000]))

# erate=np.flip(np.array([1,0.9,0.7,0.5,0.2,0.1,0.09,0.08,
#                 0.07,0.06,0.05,0.04,
#                 0.03,0.0275,0.025,0.0225,
#                 0.02,0.0175,0.015,0.0125,
#                 0.01,0.0075,0.005,0.0025,
#                 0.001,0.00075,0.0005]))
# no_timesteps=np.flip(np.array([   394000,
#           438000,    563000,    789000,  1972000,   3944000,   4382000,
#          4929000,   5634000,   6573000,   7887000,   9859000,  13145000,
#         14340000,  15774000,  17527000,  19718000,  22534000,  26290000,
#         31548000,  39435000,  52580000,  78870000, 157740000, 394351000,
#        525801000, 788702000]))

# erate=np.flip(np.array([1,0.9,0.7,0.5,0.2,0.1,0.09,0.08,
#                 0.07,0.06,0.05,0.04,
#                 0.03,0.0275,0.025,0.0225,
#                 0.02,0.0175,0.015,0.0125,
#                 0.01,0.0075,0.005,0.0025,
#                 0.001,0.00075]))
# no_timesteps=np.flip(np.array([   394000,
#           438000,    563000,    789000,  1972000,   3944000,   4382000,
#          4929000,   5634000,   6573000,   7887000,   9859000,  13145000,
#         14340000,  15774000,  17527000,  19718000,  22534000,  26290000,
#         31548000,  39435000,  52580000,  78870000, 157740000, 394351000,
#        525801000]))

erate=np.flip(np.array([1,0.8,0.6,0.4,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,
                0.03,0.025,
                0.02,0.015,
                0.01,0.005,
                0.001,0.00075,0]))

no_timesteps=np.flip(np.array([   394000,    493000,    657000,    986000,   1972000,   2253000,
         2629000,   3155000,   3944000,   4929000,   6573000,   9859000,
        13145000,  15774000,  19718000,  26290000,  39435000,  78870000,
       394351000, 525801000, 1000000]))

erate=np.array([0])
no_timesteps=np.array([20000000])



thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
K=4000
j_=1
box_size=100
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
n_plates=100


filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/no_rattle/run_156147_no_rattle/sucessful_runs_10_reals"

filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/run_63179_844598_495895/damp_0.035/sucessful_runs_15_reals/"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/run_63179_844598_495895/damp_0.035/sucessful_runs_10_reals"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/run_335862/sucessful_runs_37_reals"
#filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/10_particle/run_22190/sucessful_runs_15_reals"
#filepath="/Users/luke_dev/Documents/simulation_run_folder/eq_run_tri_plate_damp_0.035_K_500_4000/sucessful_runs_37_reals"
filepath="/Users/luke_dev/Documents/simulation_run_folder/eq_run_tri_plate_damp_0.03633_K_500_4000/sucessful_runs_37_reals"
#filepath="/Users/luke_dev/Documents/simulation_run_folder/eq_run_tri_plate_damp_0.035_K_500_4000/sucessful_runs_27_reals"
filepath="/Users/luke_dev/Documents/simulation_run_folder/eq_run_tri_plate_damp_0.05_K_500_4000/sucessful_runs_"+str(j_)+"_reals"
filepath="/Users/luke_dev/Documents/simulation_run_folder/tri_plate_with_6_angles/eq_run_tri_plate_damp_0.035_K_500_4000_100_particles"
path_2_log_files=filepath
pol_general_name_string='*K_'+str(K)+'*pol*h5'

phantom_general_name_string='*K_'+str(K)+'*phantom*h5'

Mom_general_name_string='mom.*'

log_general_name_string='log.langevin*K_'+str(K)

dump_general_name_string='*K_'+str(K)+'.dump'




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
def linearthru0(x,a):
     return a*x 

def powerlaw(x,a,n):
    return a*(x**(n))

def quadfunc(x,a):
     return a*x**(2)

def logfunc(x,a,b,c):
     return a*np.log(x-b) +c


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

print(len(realisation_name_log_sorted_final))
print(len(realisation_name_dump_sorted_final))
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
e_end=erate.size
count=e_in
#%%
# need to write dump to numpy to only look at chunks to save on ram 
for i in range(e_in,e_end):
    i_=(count*j_)
    print("i_",i_)
    
    outputdim_dump=int(dump2numpy_f(dump_start_line,
                                    Path_2_dump,
                                    realisation_name_dump_sorted_final[i_],
                                    n_plates*6).shape[0]/(n_plates*6))
    outputdim_log=log2numpy_reader(realisation_name_log_sorted_final[i_],
                                                    path_2_log_files,
                                                    thermo_vars).shape[0]
    
    dump_freq=int(realisation_name_log_sorted_final[i_].split('_')[10])

    log_file_array=np.zeros((j_,outputdim_log,7))
    p_velocities_array=np.zeros((j_,outputdim_dump,n_plates*6,3))
    p_positions_array=np.zeros((j_,outputdim_dump,n_plates*6,3))
    
    area_vector_array=np.zeros((j_,outputdim_dump,n_plates,3))
    
    erate_velocity_array=np.zeros(((j_,outputdim_dump,int(n_plates*6/2),1)))
    spring_force_positon_array=np.zeros((j_,outputdim_dump,n_plates,6))
    interest_vectors_array=np.zeros((j_,outputdim_dump,n_plates,5,3))
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
                                    n_plates*6)
            dump_data=np.reshape(dump_data,(outputdim_dump,n_plates*6,8)).astype('float')
            new_pos_vel_array=np.zeros((outputdim_dump,n_plates,6,3))
            
            p_positions_array[j,:,:,:]= dump_data[:,:,2:5]
            
            p_velocities_array[j,:,:,:]= dump_data[:,:,5:8]

            
            interest_vectors_store=np.zeros((outputdim_dump,n_plates,5,3))

            for m in range(n_plates):
                splicing_indices=np.arange(0,n_plates*6,6)
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

                        # old correction 
                        # ell_1=particle_array[1,2:5]-particle_array[0,2:5]
                        # ell_2=particle_array[2,2:5]-particle_array[0,2:5]
                        # ell_3=particle_array[2,2:5]-particle_array[1,2:5]
                        # ph_1_st_1=particle_array[3,2:5]-particle_array[0,2:5]
                        # ph_2_st_2=particle_array[4,2:5]-particle_array[1,2:5]
                        # ph_3_st_3=particle_array[5,2:5]-particle_array[0,2:5]
                        # dot_1=np.dot(ell_1,ph_1_st_1)
                        # dot_2=np.dot(ell_3,ph_2_st_2)
                        # dot_3=np.dot(ell_2,ph_3_st_3)

                        # theta_1=np.arccos(dot_1/(norm(ell_1,axis=0)*norm(ph_1_st_1,axis=0)))
                        # #print(dot_1/(norm(ell_1,axis=0)*norm(ph_1_st_1,axis=0)))
                        # theta_2=np.arccos(dot_2/(norm(ell_3,axis=0)*norm(ph_2_st_2,axis=0)))
                        # #print(dot_2/(norm(ell_3)*norm(ph_2_st_2)))
                        # theta_3=np.arccos(dot_3/(norm(ell_2,axis=0)*norm(ph_3_st_3,axis=0)))
                        # #print(dot_3/(norm(ell_2)*norm(ph_3_st_3)))
                        # # shift particles back to position 
                        # particle_array[3,2:5]-= ph_1_st_1*np.sin(theta_1)
                        # particle_array[4,2:5]-= ph_2_st_2*np.sin(theta_2)
                        # particle_array[5,2:5]-= ph_3_st_3*np.sin(theta_3)

                        # final recalc of interest vectors 

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
            
    
            
            # this isnt being filled up properly 
            area_vector_array[j,:,:,:]=np.cross(interest_vectors_store[:,:,0,:],
                                        interest_vectors_store[:,:,1,:]
                                        ,axisa=2,axisb=2)
            
            
    interest_vectors_mag=np.sqrt(np.sum(interest_vectors_store**2,
                                                     axis=3))
    
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
    interest_vectors_tuple=interest_vectors_tuple+( interest_vectors_mag,)
    count+=1



#%% save tuples to avoid needing the next stage 
#make sure to comment this out after use
# label='damp_'+str(damp)+'_K_'+str(K)+'_'
# import pickle as pck

# folder_check_or_create(filepath,"saved_tuples")

# with open(label+'spring_force_positon_tensor_tuple.pickle', 'wb') as f:
#     pck.dump(spring_force_positon_tensor_tuple, f)

# with open(label+'log_file_tuple.pickle', 'wb') as f:
#     pck.dump(log_file_tuple, f)

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

with open(label+'p_velocities_tuple.pickle', 'rb') as f:
    p_velocities_tuple=pck.load(f)

with open(label+'p_positions_tuple.pickle', 'rb') as f:
    pol_positions_tuple=pck.load(f)

with open(label+"new_pos_vel_tuple.pickle",'rb') as f:
     new_pos_vel_tuple=pck.load(f)

with open(label+"area_vector_tuple.pickle", 'rb') as f:
    area_vector_tuple= pck.load(f)

with open(label+"interest_vectors_tuple.pickle",'rb') as f:
     interest_vectors_tuple=pck.load(f)



#%% computing stress distributions 
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import curve_fit
labels_stress=["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{yz}$"]
marker=['x','o','+','^',"1","X","d","*","P","v"]
stress_tensor=np.zeros((e_end,6))
stress_tensor_std=np.zeros((e_end,6))
line_values=np.zeros((10))
for l in range(6):
    for i in range(e_end):
    #for l in range(6):
        ##for m in range(10):
           

                #data=np.mean(spring_force_positon_tensor_tuple[i][:,:,m,2]-spring_force_positon_tensor_tuple[i][:,:,m,1], axis=0)
                # need to include a cut off in this to get rid of any start up effects
                cutoff=int(np.round(0.7*spring_force_positon_tensor_tuple[i][:,:,:,l].shape[1]))
                data=np.ravel(spring_force_positon_tensor_tuple[i][:,cutoff:,:,l])
                #strain_plot=np.linspace(0,400,data.shape[0])
                #plt.plot(strain_plot,data, label=labels_stress[l])
                mean_np=np.mean(data)
                std_np=np.std(data)

                
            
                
                # f = Fitter(data,
                # distributions=[
                #           "norm"])
                # f.fit()
                # #f.get_best(method = 'sumsquare_error')
                # fitting_params=f.fitted_param.get('norm')
                # mean,std=fitting_params
                # f.summary()
                
                stress_tensor[i,l]=mean_np
                stress_tensor_std[i,l]=std_np
                # plt.show()

                # plt.plot(data, norm.pdf(data, mean_np, std_np)) 
           
                # plt.legend()
                # plt.show()
                



    

#%%normal stresses

for l in range(3):
        
    #plt.scatter(erate[:e_end],stress_tensor[:,l],label=labels_stress[l],marker=marker[l] )
    plt.errorbar(erate[:e_end], stress_tensor[:,l], yerr =stress_tensor_std[:,l]/np.sqrt(j_*100), ls='--',label=labels_stress[l],marker=marker[l] )
    plt.xlabel("$\dot{\gamma}$")
    # check if its number of realisations -1 or just number of realisations 


    plt.legend()  
#plt.xscale('log')
plt.show()    

for l in range(3,6):
        
    #plt.scatter(erate[:e_end],stress_tensor[:,l],label=labels_stress[l],marker=marker[l] )
    plt.errorbar(erate[:e_end], stress_tensor[:,l], yerr =stress_tensor_std[:,l]/np.sqrt(j_*100), ls='--',label=labels_stress[l],marker=marker[l] )
    plt.xlabel("$\dot{\gamma}$")
    # check if its number of realisations -1 or just number of realisations 

    plt.legend()  
plt.show()    
#%% n1 
n_1= stress_tensor[:,0]- stress_tensor[:,2]
n_1_error=np.sqrt(stress_tensor_std[:,0]**2 +stress_tensor_std[:,2]**2)/np.sqrt(j_*n_plates)
        
plt.scatter(erate[:e_end],(n_1),label="$N_{1}$",marker=marker[0] )
#plt.errorbar(erate[:e_end], n_1, yerr =n_1_error, ls='none',label="$N_{1}$",marker=marker[0] )
plt.ylabel("$N_{1}$", rotation=0)
plt.xlabel("$\dot{\gamma}$")


plt.legend()  
plt.show() 

#%% n2
n_2= stress_tensor[:,2]- stress_tensor[:,1]
n_2_error=np.sqrt(stress_tensor_std[:,1]**2 +stress_tensor_std[:,2]**2)/np.sqrt(j_*n_plates)
        
plt.scatter(erate[:e_end],n_2,label="$N_{2}$",marker=marker[0] )
plt.ylabel("$N_{2}$", rotation=0)
plt.xlabel("$\dot{\gamma}$")
#plt.ylim(-20,75)
plt.legend()  

#plt.xscale('log')
plt.show() 


#%% polyfit
fit=np.polyfit(erate,n_2,2)
plt.plot(erate,fit[0]*(erate**2)+fit[1]*erate+ fit[2])
#plt.plot(erate,fit[0]*(erate**3)+(fit[1]*erate**2)+ fit[2]*erate +fit[3])

plt.scatter(erate[:e_end],n_2,label="$N_{2}$",marker=marker[0] )
plt.xscale('log')
plt.show()
# plt.errorbar(erate[:e_end], n_2, yerr =n_2_error, ls='none',label="$N_{2}$",marker=marker[0] )

# plt.legend()  
# plt.show() 


     
     
#%% linear fit n1
plt.errorbar(erate[:e_end], n_1, yerr =n_1_error, ls='none',label="$N_{1}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(linearthru0,erate[:e_end], n_1)
difference=np.sqrt(np.sum((n_1-(popt[0]*(erate[:e_end])))**2)/(e_end))

plt.plot(erate[:e_end],(popt[0]*(erate[:e_end])),
         label="$N_{1,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")
plt.legend()

plt.show()
print(difference)


#%%quadratic fit n1
plt.errorbar(erate[:e_end], n_1, yerr =n_1_error, ls='none',label="$N_{1}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(quadfunc,erate[:e_end], n_1)
difference=np.sqrt(np.sum((n_1-(popt[0]*(erate[:e_end])**2))**2)/(e_end))

plt.plot(erate[:e_end],(popt[0]*(erate[:e_end])**2),
         label="$N_{1,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")
plt.legend()
plt.xscale('log')
plt.show()

print(difference)



#%% linear fit n2
plt.errorbar(erate[:e_end], n_2, yerr =n_2_error, ls='none',label="$N_{2}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(linearthru0,erate[:e_end], n_2)
difference=np.sqrt(np.sum((n_2-(popt[0]*(erate[:e_end])))**2)/(e_end))

plt.plot(erate[:e_end],(popt[0]*(erate[:e_end])),
         label="$N_{2,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
           ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")
plt.legend()
plt.xscale('log')
plt.show()
print(difference)



#%%quadratic fit n2
plt.errorbar(erate[:e_end], n_2, yerr =n_2_error, ls='none',label="$N_{2}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(quadfunc,erate[:e_end], n_2)
difference=np.sqrt(np.sum((n_2-(popt[0]*(erate[:e_end])**2))**2)/(e_end))

plt.plot(erate[:e_end],(popt[0]*(erate[:e_end])**2),
         label="$N_{2,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")
plt.legend()
plt.xscale('log')
plt.show()
print(difference)


#%% shear stress xz
xz_stress= stress_tensor[:,3]
xz_stress_std=stress_tensor_std[:,3]/np.sqrt(j_*10)
plt.errorbar(erate[:e_end], xz_stress, yerr =xz_stress_std, ls='none',label="$\sigma_{xz}$",marker=marker[0] )
popt,cov_matrix_xz=curve_fit(linearthru0,erate[:e_end], xz_stress)
difference=np.sqrt(np.sum((xz_stress-popt[0]*(erate[:e_end]))**2))/(e_end)
plt.plot(erate[:e_end],(popt[0]*(erate[:e_end])),
         label="$\sigma_{xz,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+

         ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")

plt.legend()  
#plt.xscale('log')
plt.ylabel("$\sigma_{xz}$", rotation=0)
plt.xlabel("$\dot{\gamma}$")

plt.show() 
#%%
xz_stress= stress_tensor[:,3]
xz_stress_std=stress_tensor_std[:,3]/np.sqrt(j_*10)
plt.errorbar(erate[:e_end], xz_stress/erate[:e_end], yerr =xz_stress_std, ls='none',label="$\sigma_{xz}$",marker=marker[0] )
popt,cov_matrix_xz=curve_fit(linearthru0,erate[:e_end], xz_stress)
difference=np.sqrt(np.sum((xz_stress-popt[0]*(erate[:e_end]))**2))/(e_end)
# plt.plot(erate[:e_end],(popt[0]*(erate[:e_end])),
#          label="$\sigma_{xz,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+

#          ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")

plt.legend()  

plt.ylabel("$\sigma_{xz}$", rotation=0)
plt.xlabel("$\dot{\gamma}$")

plt.show() 


#%% plot n1 and n2 quadratic 
plt.errorbar(erate[:e_end], n_1, yerr =n_1_error, ls='none',label="$N_{1}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(quadfunc,erate[:e_end], n_1)
difference=np.sqrt(np.sum((n_1-(popt[0]*(erate[:e_end])**2))**2)/(e_end))

plt.plot(erate[:e_end],(popt[0]*(erate[:e_end])**2),
         label="$N_{1,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")


print(difference)
plt.legend()
plt.xscale('log')
plt.show()


plt.errorbar(erate[:e_end], n_2, yerr =n_2_error, ls='none',label="$N_{2}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(quadfunc,erate[:e_end], n_2)
difference=np.sqrt(np.sum((n_2-(popt[0]*(erate[:e_end])**2))**2)/(e_end))

plt.plot(erate[:e_end],(popt[0]*(erate[:e_end])**2),
         label="$N_{2,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")
plt.legend()
plt.xscale('log')
plt.show()
print(difference)

#%% plot n1 linear

plt.errorbar(erate[:e_end], n_1, yerr =n_1_error, ls='none',label="$N_{1}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(linearthru0,erate[:e_end], n_1)
difference=np.sqrt(np.sum((n_1-(popt[0]*(erate[:e_end])))**2)/(e_end))

plt.plot(erate[:e_end],(popt[0]*(erate[:e_end])),
         label="$N_{1,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")
plt.legend()
plt.xscale('log')
plt.show()
print(difference)
#%% n2 linear 

plt.errorbar(erate[:e_end], n_2, yerr =n_2_error, ls='none',label="$N_{2}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(linearthru0,erate[:e_end], n_2)
difference=np.sqrt(np.sum((n_2-(popt[0]*(erate[:e_end])))**2)/(e_end))

plt.plot(erate[:e_end],(popt[0]*(erate[:e_end])),
         label="$N_{2,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")
plt.legend()
plt.xscale('log')
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
    xline=np.arange(0,np.max(x),0.1)
    yline=np.arange(0,np.max(x),0.1)

    plt.scatter(x,y, marker='x',alpha=0.5, color='g')
    plt.plot(xline,yline,ls='dashed')
    plt.xlabel("$v_{x}(z)$",rotation=0)
    plt.ylabel("$v_{x,CoM}$",rotation=0)
    
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
#plt.xscale('log')
# plt.yscale('log')
#plt.axhline(1,label="$T_{0}=1$")
plt.legend()
plt.tight_layout()
plt.show()

          

      
# %% area vector analysis 
import seaborn as sns
pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
pi_phi_ticks=[ 0,np.pi/4, np.pi/2]
pi_phi_tick_labels=[ '0','π/4', 'π/2']
spherical_coords_tuple=()
for i in range(e_in,e_end):
     
    area_vector_ray=area_vector_tuple[i]
    # detect all z coords less than 0 and multiply all 3 coords by -1
    area_vector_ray[area_vector_ray[:,:,:,2]<0]*=-1
    spherical_coords_array=np.zeros((j_,area_vector_ray.shape[1],n_plates,3))
    x=area_vector_ray[:,:,:,0]
    y=area_vector_ray[:,:,:,1]
    z=area_vector_ray[:,:,:,2]


     # radial coord
    spherical_coords_array[:,:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))
     #  theta coord 
    spherical_coords_array[:,:,:,1]=np.sign(y)*np.arccos(x/(np.sqrt((x**2)+(y**2))))
     # phi coord
    spherical_coords_array[:,:,:,2]=np.arccos(z/spherical_coords_array[:,:,:,0])

    spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_array,)

#%% look at chnage of theta with time 
for i in range(e_in,e_end):
     strain_plot=np.linspace(0,400,spherical_coords_array[i,:,0,2].shape[0])
     plt.plot( strain_plot,spherical_coords_array[i,:,0,1])
     plt.show()

#%%
for i in range(e_in,e_end):
     strain_plot=np.linspace(0,400,spherical_coords_array[i,:,0,2].shape[0])
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
    sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,:,:,0]))
plt.show()
#%% theta 
# could just plot a few of them 
skip_array=np.arange(0,e_end,4)
skip_array_2=np.arange(0,int(no_timesteps[0]/100),10000)
for i in range(skip_array.size):
    for j in range(skip_array_2.size):


        i=skip_array[i]

        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,:,:,1]),
        #             label ="$\dot{\gamma}="+str(erate[i])+"$",bw_adjust=1)
        sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,skip_array_2[j]:,:,1]),
                    label ="$\dot{\gamma}="+str(erate[i])+"$")
        #plt.hist(np.ravel(spherical_coords_tuple[i][:,skip_array_2[j]:,:,1]))
        # bw adjust effects the degree of smoothing , <1 smoothes less
        plt.xlabel("$\Theta$")
        plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
        plt.xlim(-np.pi,np.pi)
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=[1.1, 0.45])
        plt.show()

#%% phi 
for i in range(skip_array.size):
    for j in range(skip_array_2.size):
        i=skip_array[i]

        sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,skip_array_2[j]:,:,2]),
                    label ="$\dot{\gamma}="+str(erate[i])+"$")
        print(np.mean(np.ravel(spherical_coords_tuple[i][:,:,:,2])))
        plt.xlabel("$\Phi$")
        plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=[1.1, 0.45])
        plt.xlim(0,np.pi/2)
        plt.show()

#%% extension_vectors
eq_spring_length=3*np.sqrt(3)/2
for i in range(skip_array.size):
    i=skip_array[i]
# for i in range(e_in,e_end):

    sns.kdeplot(eq_spring_length-np.ravel(interest_vectors_tuple[i][:,:,2:5]),
                 label ="$\dot{\gamma}="+str(erate[i])+"$")
plt.xlabel("$\Delta x$")

plt.ylabel('Density')
plt.legend(bbox_to_anchor=[1.1, 0.45])
#plt.xlim(-0.25,0.25)
plt.show()
    




# %%m stress distributions 
stretch_events_ratio=np.zeros((6,erate.size))
for l in range(6):
    
               
   
           

                #data=np.mean(spring_force_positon_tensor_tuple[i][:,:,m,2]-spring_force_positon_tensor_tuple[i][:,:,m,1], axis=0)
                # need to include a cut off in this to get rid of any start up effects
                i=0
                cutoff=int(np.round(0.1*spring_force_positon_tensor_tuple[i][:,:,:,l].shape[1]))
                data=np.ravel(spring_force_positon_tensor_tuple[i][:,cutoff:,:,l])
                #plt.plot(data)
                sns.kdeplot(data=data, label=labels_stress[l]+", $\\bar{\sigma}="+\
                            str(sigfig.round(np.mean(data),sigfigs=4))+\
                                ", \dot{\gamma}="+str(erate[i])+\
                                    ", N_{p}<0/N_{p}>0="+str(sigfig.round(data[data<0].size/data[data>0].size,sigfigs=3))+"$")
                
                # i=5
                # cutoff=int(np.round(0.1*spring_force_positon_tensor_tuple[i][:,:,:,l].shape[1]))
                # data=np.ravel(spring_force_positon_tensor_tuple[i][:,cutoff:,:,l])

                
                # #plt.plot(data)
                # sns.kdeplot(data=data, label=labels_stress[l]+", $\\bar{\sigma}="+\
                #             str(sigfig.round(np.mean(data),sigfigs=4))+\
                #                 ", \dot{\gamma}="+str(erate[i])+\
                #                     ", N_{p}<0/N_{p}>0="+str(sigfig.round(data[data<0].size/data[data>0].size,sigfigs=3))+"$")
                
           

                
                # i=10
                # cutoff=int(np.round(0.1*spring_force_positon_tensor_tuple[i][:,:,:,l].shape[1]))
                # data=np.ravel(spring_force_positon_tensor_tuple[i][:,cutoff:,:,l])

                
                # #plt.plot(data)
                # sns.kdeplot(data=data, label=labels_stress[l]+", $\\bar{\sigma}="+\
                #             str(sigfig.round(np.mean(data),sigfigs=4))+\
                #                 ", \dot{\gamma}="+str(erate[i])+\
                #                     ", N_{p}<0/N_{p}>0="+str(sigfig.round(data[data<0].size/data[data>0].size,sigfigs=3))+"$")
                
           

                # #data=np.mean(spring_force_positon_tensor_tuple[i][:,:,m,2]-spring_force_positon_tensor_tuple[i][:,:,m,1], axis=0)
                # # need to include a cut off in this to get rid of any start up effects
                # i=15
                # cutoff=int(np.round(0.1*spring_force_positon_tensor_tuple[i][:,:,:,l].shape[1]))
                # data=np.ravel(spring_force_positon_tensor_tuple[i][:,cutoff:,:,l])

                
                # #plt.plot(data)
                # sns.kdeplot(data=data, label=labels_stress[l]+", $\\bar{\sigma}="+\
                #             str(sigfig.round(np.mean(data),sigfigs=4))+\
                #                 ", \dot{\gamma}="+str(erate[i])+\
                #                     ", N_{p}<0/N_{p}>0="+str(sigfig.round(data[data<0].size/data[data>0].size,sigfigs=3))+"$")
                
           
                # i=e_end-1
                # cutoff=int(np.round(0.1*spring_force_positon_tensor_tuple[i][:,:,:,l].shape[1]))
                # data=np.ravel(spring_force_positon_tensor_tuple[i][:,cutoff:,:,l])

                
                # #plt.plot(data)
                # sns.kdeplot(data=data, label=labels_stress[l]+", $\\bar{\sigma}="+\
                #             str(sigfig.round(np.mean(data),sigfigs=4))+\
                #                 ", \dot{\gamma}="+str(erate[i])+\
                #                     ", N_{p}<0/N_{p}>0="+str(sigfig.round(data[data<0].size/data[data>0].size,sigfigs=3))+"$")
                

                
            
                
                plt.legend(bbox_to_anchor=[1.1, 0.45])              
                plt.show()



# %% looking a relative number of stretch and compression events in each entry of stress tensor 

ex_stretch_events_ratio=np.zeros((6,erate.size))
stress_bound=0
for l in range(6):
    for i in range(erate.size):
        cutoff=int(np.round(0.1*spring_force_positon_tensor_tuple[i][:,:,:,l].shape[1]))
        data=np.ravel(spring_force_positon_tensor_tuple[i][:,cutoff:,:,0])-np.ravel(spring_force_positon_tensor_tuple[i][:,cutoff:,:,2])
        try: 
            ex_stretch_events_ratio[l,i]=data[data<-stress_bound].size/data[data>stress_bound].size
        except:
            ex_stretch_events_ratio[l,i]=0

for l in range(3):
    plt.scatter(erate,ex_stretch_events_ratio[l,:], label=labels_stress[l], marker=marker[l])
    plt.xlabel("$\dot{\gamma}$", rotation=0)
    plt.ylabel("$\\frac{N_{e}<"+str(stress_bound)+"}{N_{e}>"+str(stress_bound)+"}$", rotation=0, labelpad=20)
plt.legend()
#plt.xscale('log')
plt.show()
ex_stretch_events_ratio=np.zeros((6,erate.size))
stress_bound=0
for l in range(6):
    for i in range(erate.size):
        cutoff=int(np.round(0.1*spring_force_positon_tensor_tuple[i][:,:,:,l].shape[1]))
        data=np.ravel(spring_force_positon_tensor_tuple[i][:,cutoff:,:,2])-np.ravel(spring_force_positon_tensor_tuple[i][:,cutoff:,:,1])
        try: 
            ex_stretch_events_ratio[l,i]=data[data<-stress_bound].size/data[data>stress_bound].size
        except:
            ex_stretch_events_ratio[l,i]=0

for l in range(3):
    plt.scatter(erate,ex_stretch_events_ratio[l,:], label=labels_stress[l], marker=marker[l])
    plt.xlabel("$\dot{\gamma}$", rotation=0)
    plt.ylabel("$\\frac{N_{e}<"+str(stress_bound)+"}{N_{e}>"+str(stress_bound)+"}$", rotation=0, labelpad=20)
plt.legend()
#plt.xscale('log')
plt.show()




#note need to consider if we can include the weighting 


# %%
