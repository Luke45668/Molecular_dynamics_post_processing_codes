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
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
#plt.rcParams['text.usetex'] = True
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
import seaborn as sns
sns.set_palette('colorblind')
from log2numpy import *
from dump2numpy import *
import glob 
from MPCD_codes.post_MPCD_MP_processing_module import *
import pickle as pck
from numpy.linalg import norm
from post_langevin_module import *

#%%
damp=0.035
strain_total=100

# old run with bad low shear rate data 
# K=50 
# no_timesteps=np.flip(np.array([   394000,    493000,    657000,    986000,   1972000,   2253000,
#          2629000,   3155000,   3944000,   4929000,   6573000, 
#        394351000, 525801000, 1000000]))


# erate=np.flip(np.array([1,0.8,0.6,0.4,0.2,0.175,0.15,0.125,0.1,0.08,
#                 0.06,0.001,0.00075,0]))

# timestep_multiplier=np.array([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,
#                               0.05,0.05,0.005,0.005,0.005,0.005])



# erate=np.array([0])
# no_timesteps=np.array([10000000])
K=60
erate=np.flip(np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.01,0.005,0]))

no_timesteps=np.flip(np.array([ 3944000,  4382000,  4929000,  5634000,  6573000,  7887000,
         9859000, 13145000, 19718000,  2253000,  2629000,  3155000,
         3944000,  4929000,  6573000,  9859000, 39435000,
        78870000, 10000000]))

timestep_multiplier=np.flip(np.array(
[0.005,0.005,0.005,0.005,
0.005,0.005,0.005,0.005,0.005,
0.05,0.05,0.05,0.05,0.05,0.05,
0.05,0.05,0.05,0.2]))


thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
#thermo_vars='         KinEng         PotEng         Press         c_myTemp        TotEng    '




j_=5
box_size=100
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
n_plates=100

filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/run_692855/sucessful_runs_3_reals"
#filepath='/Users/luke_dev/Documents/simulation_run_folder/tri_plate_with_6_angles/eq_run_tri_plate_damp_0.035_K_50_100_particles'
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/run_312202/"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/run_667325/"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/run_279865/"
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

def block_averaging(oned_arr_in,number_blocks):
    block_means=np.zeros((number_blocks))
    size_block=int(oned_arr_in.shape[0]/number_blocks)

    block_count=0
    for i in range(number_blocks):
          block_means[i]=np.mean(oned_arr_in[block_count*size_block:
            (block_count+1)*size_block] )
          block_count+=1

    return block_means


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


md_step=0.005071624521210362*timestep_multiplier


box_size_div=2
strain_total=np.repeat(strain_total,erate.size)
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
    
    #log_file_array=np.zeros((j_,outputdim_log,6)) #eq
    log_file_array=np.zeros((j_,outputdim_log,7)) #nemd
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
                        
                        strain=l*dump_freq*md_step[i]*erate[i] -\
                                np.floor(l*dump_freq*md_step[i]*erate[i])
                            
                            
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
marker=['x','o','+','^',"1","X","d","*","P","v"]
aftcut=1
cut=0.4

labels_stress=["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{yz}$"]

def stress_tensor_averaging(e_end,
                            labels_stress,
                            trunc1,
                            trunc2,
                            spring_force_positon_tensor_tuple,j_):
    stress_tensor=np.zeros((e_end,6))
    stress_tensor_std=np.zeros((e_end,6))
    stress_tensor_reals=np.zeros((e_end,j_,6))
    stress_tensor_std_reals=np.zeros((e_end,j_,6))
    for l in range(6):
        for i in range(e_end):
            for j in range(j_):
                cutoff=int(np.round(trunc1*spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0]))
                aftercutoff=int(np.round(trunc2*spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0]))
                # print(spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0])
                # print(cutoff)
                # print(aftercutoff)
                data=np.ravel(spring_force_positon_tensor_tuple[i][j,cutoff:aftercutoff,:,l])
                stress_tensor_reals[i,j,l]=np.mean(data)
                stress_tensor_std_reals[i,j,l]=np.std(data)
                stress_tensor=np.mean(stress_tensor_reals, axis=1)
                stress_tensor_std=np.mean(stress_tensor_std_reals, axis=1)
    return stress_tensor,stress_tensor_std

stress_tensor,stress_tensor_std=stress_tensor_averaging(e_end,
                            labels_stress,
                            cut,
                            aftcut,
                            spring_force_positon_tensor_tuple,j_)

#%% plot stress vs strain 


def plotting_stress_vs_strain(spring_force_positon_tensor_tuple,
                              i_1,i_2,j_,
                              strain_total,cut,aftcut,stress_component,label_stress):
    

    mean_grad_l=[] 
    for i in range(i_1,i_2):
        #for j in range(j_):
            cutoff=int(np.round(cut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            aftcutoff=int(np.round(aftcut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))


            strain_plot=np.linspace(cut*strain_total,aftcut*strain_total,spring_force_positon_tensor_tuple[i][0,cutoff:aftcutoff,:,stress_component].shape[0])
            cutoff=int(np.round(cut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            aftcutoff=int(np.round(aftcut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            stress=np.mean(spring_force_positon_tensor_tuple[i][:,:,:,stress_component],axis=0)
            stress=stress[cutoff:aftcutoff]
            gradient_vec=np.gradient(np.mean(stress,axis=1))
            mean_grad=np.mean(gradient_vec)
            mean_grad_l.append(mean_grad)
            #print(stress.shape)
            # plt.plot(strain_plot,np.mean(stress,axis=1))
            # plt.ylabel(labels_stress[stress_component],rotation=0)
            # plt.xlabel("$\gamma$")
            # plt.plot(strain_plot,gradient_vec, label="$\\frac{dy}{dx}="+str(mean_grad)+"$")

            #plt.legend()
            #plt.show()

    plt.scatter(erate,mean_grad_l, label=label_stress)
    plt.xlabel("$\dot{\gamma}$")
    plt.ylabel("$\\frac{d\\bar{\sigma}_{\\alpha\\beta}}{dt}$", rotation=0,labelpad=20)
    #plt.show()

for i in range(6):

    plotting_stress_vs_strain(spring_force_positon_tensor_tuple,
                              e_in,e_end,j_,
                              strain_total,cut,aftcut,i,labels_stress[i])
plt.legend(fontsize=14) 
plt.show()




#%% plotting N1 N2 vs strain 
def plotting_n_difff_vs_strain(spring_force_positon_tensor_tuple,
                              i_1,i_2,j_,
                              strain_total,i1,i2,cut,aftcut):
     
    for i in range(i_1,i_2):
        #for j in range(j_):
            cutoff=int(np.round(cut*spring_force_positon_tensor_tuple[i][0,:,:,0].shape[0]))
            aftcutoff=int(np.round(aftcut*spring_force_positon_tensor_tuple[i][0,:,:,0].shape[0]))
            strain_plot=np.linspace(cut*strain_total,aftcut*strain_total,spring_force_positon_tensor_tuple[i][0,cutoff:aftcutoff,:,0].shape[0])
            
            n_diff=spring_force_positon_tensor_tuple[i][:,cutoff:aftcutoff,:,i1]-spring_force_positon_tensor_tuple[i][:,cutoff:aftcutoff,:,i2]
            n_diff=np.mean(n_diff,axis=0)
            print(n_diff.shape)
            plt.xlabel("")
            plt.plot(strain_plot,np.mean(n_diff,axis=1), label ="$\dot{\gamma}="+str(erate[i])+"$")
    plt.show()


plotting_n_difff_vs_strain(spring_force_positon_tensor_tuple,e_in,e_end,j_,strain_total,0,2,cut,aftcut)

plotting_n_difff_vs_strain(spring_force_positon_tensor_tuple,e_in,e_end,j_,strain_total,2,1,cut,aftcut)


#%% plotting stress tensor with shear rate

def plot_stress_tensor(t_0,t_1,
                       stress_tensor,
                       stress_tensor_std,
                       j_,n_plates, labels_stress,marker,cutoff,erate,e_end):
    for l in range(t_0,t_1):
          plt.errorbar(erate[cutoff:e_end], stress_tensor[cutoff:,l], yerr =stress_tensor_std[cutoff:,l]/np.sqrt(j_*n_plates), ls='--',label=labels_stress[l],marker=marker[l] )
          plt.xlabel("$\dot{\gamma}$")
          plt.ylabel("$\sigma_{\\alpha\\beta}$",rotation=0,labelpad=20)
    plt.legend()      
    plt.show()

plot_stress_tensor(0,3,
                       stress_tensor,
                       stress_tensor_std,
                       j_,n_plates, labels_stress,marker,0,erate,e_end)

plot_stress_tensor(3,6,
                       stress_tensor,
                       stress_tensor_std,
                       j_,n_plates, labels_stress,marker,0,erate,e_end)


    
#%%normal stress differences 

def compute_plot_n_stress_diff(stress_tensor, 
                          stress_tensor_std,
                          i1,i2,
                          j_,n_plates,
                          erate,e_end,
                          ylab):
    n_diff=stress_tensor[:,i1]- stress_tensor[:,i2]
    n_diff_error=np.sqrt(stress_tensor_std[:,i1]**2 +stress_tensor_std[:,i2]**2)/np.sqrt(j_*n_plates)
    #plt.errorbar(erate[:e_end], n_diff, yerr =n_diff_error, ls='none' )
    plt.scatter(erate[:e_end],n_diff )
    plt.ylabel(ylab, rotation=0, labelpad=20)
    plt.xlabel("$\dot{\gamma}$")
    plt.legend()  
    plt.show() 
        
    return n_diff,n_diff_error

n_1,n_1_error=compute_plot_n_stress_diff(stress_tensor, 
                          stress_tensor_std,
                          0,2,
                          j_,n_plates,
                          erate,e_end,
                          "$N_{1}$")

n_2,n_2_error=compute_plot_n_stress_diff(stress_tensor, 
                          stress_tensor_std,
                          2,1,
                          j_,n_plates,
                          erate,e_end,
                          "$N_{2}$")
  


#%% collecting n1 and n2 
# n_1_list=[]
# n_1_list_er=[]
# n_2_list=[]
# n_2_list_er=[]
# n_1_list.append(n_1)
# n_2_list.append(n_2)
# n_1_list_er.append(n_1_error)
# n_2_list_er.append(n_2_error)
# polyfit
# fit=np.polyfit(erate[3:e_end],n_2[3:e_end],2)
# plt.plot(erate[3:e_end],fit[0]*(erate[3:e_end]**2)+fit[1]*erate[3:e_end])
# #plt.plot(erate,fit[0]*(erate**3)+(fit[1]*erate**2)+ fit[2]*erate +fit[3])

# #plt.scatter(erate[:e_end],n_2,label="$N_{2}$",marker=marker[0] )
# #plt.xscale('log')
# #plt.show()
# plt.errorbar(erate[3:e_end], n_2[3:e_end], yerr =n_2_error[3:e_end], ls='none',label="$N_{2}$",marker=marker[0] )

# plt.legend()  
# plt.show() 


     
     

#%% linear fit n1 and n2
cutoff=0
plt.errorbar(erate[cutoff:e_end], n_1[cutoff:e_end], yerr =n_1_error[cutoff:e_end], ls='none',label="$N_{1}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(linearthru0,erate[cutoff:e_end], n_1[cutoff:e_end])
difference=np.sqrt(np.sum((n_1[cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),
         label="$N_{1,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")

#plt.xscale('log')
#plt.show()
print(difference)

plt.errorbar(erate[cutoff:e_end], n_2[cutoff:e_end], yerr =n_2_error[cutoff:e_end], ls='none',label="$N_{2}$",marker=marker[0] )
popt,cov_matrix_n2=curve_fit(linearthru0,erate[cutoff:e_end], n_2[cutoff:e_end])
difference=np.sqrt(np.sum((n_2[cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),
         label="$N_{2,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
           ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")
plt.legend(fontsize=12)
plt.xscale('log')
plt.xlabel("$\dot{\gamma}$")
plt.show()
print(difference)


#%%quadratic fit n1
plt.errorbar(erate[cutoff:e_end], n_1[cutoff:e_end], yerr =n_1_error[cutoff:e_end], ls='none',label="$N_{1}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(quadfunc,erate[cutoff:e_end], n_1[cutoff:e_end])
difference=np.sqrt(np.sum((n_1[cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])**2))**2)/(e_end))

plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])**2),
         label="$N_{1,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")
plt.legend()
#plt.xscale('log')
plt.show()

print(difference)



#%% linear fit n2
plt.errorbar(erate[cutoff:e_end], n_2[cutoff:e_end], yerr =n_2_error[cutoff:e_end], ls='none',label="$N_{2}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(linearthru0,erate[cutoff:e_end], n_2[cutoff:e_end])
difference=np.sqrt(np.sum((n_2[cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),
         label="$N_{2,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
           ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")
plt.legend()
plt.xscale('log')
plt.show()
print(difference)



#%%quadratic fit n2
plt.errorbar(erate[cutoff:e_end], n_2[cutoff:e_end], yerr =n_2_error[cutoff:e_end], ls='none',label="$N_{2}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(quadfunc,erate[cutoff:e_end], n_2[cutoff:e_end])
difference=np.sqrt(np.sum((n_2[cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])**2))**2)/(e_end))

plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])**2),
         label="$N_{2,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")
plt.legend()
#plt.xscale('log')
plt.show()
print(difference)


#%% shear stress xz
xz_stress= stress_tensor[:,3]
xz_stress_std=stress_tensor_std[:,3]/np.sqrt(j_*n_plates)
plt.errorbar(erate[:e_end], xz_stress, yerr =xz_stress_std, ls='none',label="$\sigma_{xz}$",marker=marker[0] )
popt,cov_matrix_xz=curve_fit(linearthru0,erate[:e_end], xz_stress)
difference=np.sqrt(np.sum((xz_stress-popt[0]*(erate[:e_end]))**2))/(e_end)
plt.plot(erate[:e_end],(popt[0]*(erate[:e_end])),
         label="$\sigma_{xz,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+

         ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$")

plt.legend()  
plt.xscale('log')
plt.yscale('log')
plt.ylabel("$\sigma_{xz}$", rotation=0)
plt.xlabel("$\dot{\gamma}$")

plt.show() 
#%% viscosity plot 
def powerlaw(x,a,n):
    return a*(x**(n))

def herschbuck(x,a,b,n):
     
     return (a/x)+ b*(x**(n))
     
cutoff=1
xz_stress= stress_tensor[cutoff:,3]
xz_stress_std=stress_tensor_std[:,3]/np.sqrt(j_*n_plates)
#powerlaw
plt.errorbar(erate[cutoff:e_end], xz_stress/erate[cutoff:e_end], yerr =xz_stress_std[cutoff:], ls='none',label="$\eta$",marker=marker[0] )
popt,cov_matrix_xz=curve_fit(powerlaw,erate[cutoff:e_end], xz_stress/erate[cutoff:e_end])
y=xz_stress/erate[cutoff:e_end]
y_pred=popt[0]*(erate[cutoff:e_end]**(popt[1]))
difference=np.sqrt(np.sum((y-y_pred)**2)/e_end-cutoff)
plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end]**(popt[1]))),
         label="$\eta_{fit},a="+str(sigfig.round(popt[0],sigfigs=3))+",n="+str(sigfig.round(popt[1],sigfigs=3))+

          ",\\varepsilon=\pm"+str(sigfig.round(difference,sigfigs=3))+"$")

plt.legend(fontsize=14) 
plt.ylabel("$\eta$", rotation=0,labelpad=20)
plt.xlabel("$\dot{\gamma}$")
# plt.xscale('log')
# plt.yscale('log')

plt.show() 

# herschel buckley fit yield stress is negative so dont use it 
# plt.errorbar(erate[cutoff:e_end], xz_stress/erate[cutoff:e_end], yerr =xz_stress_std[cutoff:], ls='none',label="$\sigma_{xz}$",marker=marker[0] )
# popt,cov_matrix_xz=curve_fit(herschbuck,erate[cutoff:e_end], xz_stress/erate[cutoff:e_end])
# y=xz_stress/erate[cutoff:e_end]
# y_pred=(popt[0]/erate[cutoff:e_end])+ popt[1]*(erate[cutoff:e_end]**(popt[2]))
# difference=np.sqrt(np.sum((y-y_pred)**2)/e_end-cutoff)

# plt.plot(erate[cutoff:e_end],((popt[0]/erate[cutoff:e_end])+ popt[1]*(erate[cutoff:e_end]**(popt[2]))),
#          label="$\eta_{fit},a="+str(sigfig.round(popt[0],sigfigs=3))+",b="+str(sigfig.round(popt[1],sigfigs=3))+
#          ",n="+str(sigfig.round(popt[2],sigfigs=3))+

#           ",\\varepsilon=\pm"+str(sigfig.round(difference,sigfigs=3))+"$")#

# #plt.legend()  
# plt.legend(fontsize=11) 

# plt.ylabel("$\eta$", rotation=0,labelpad=20)
# plt.xlabel("$\dot{\gamma}$")
# # plt.xscale('log')
# # plt.yscale('log')

# plt.show() 

# need to find power law fit 
# power law model for viscosity 



#%% plot N1 and N2 together 
multi=-10
sns.set_palette('icefire')
plt.scatter(erate,n_1,label="$N_{1}$", marker="x")
plt.scatter(erate,multi*n_2,label="$"+str(multi)+"N_{2}$", marker="*")
plt.xlabel("$\dot{\gamma}$")
#plt.ylabel("$\\frac{N_{1}}{N_{2}}$", rotation=0)
plt.legend()
plt.xscale('log')
plt.show()

#%%
sns.set_palette('icefire')
plt.scatter(erate,n_1/n_2,label="$N_{1}$")
plt.axhline(np.mean(n_1/n_2))
plt.xlabel("$\dot{\gamma}$")
plt.ylabel("$\\frac{N_{1}}{N_{2}}$", rotation=0)
plt.legend()
plt.xscale('log')
plt.show()



#%% taking mean of all timesteps to check velocity profile matches up

for i in range(e_in,e_end):
    
        i_=(count*j_)
        COM_position= np.mean(new_pos_vel_tuple[i][:,:,0:3,:],axis=2)
        COM_velocity=np.mean(new_pos_vel_tuple[i][:,:,3:6,:],axis=2)

        erate_predicted_vel=COM_position[:,:,2]*erate[i]
        x=np.mean(erate_predicted_vel,axis=0)
        y=np.mean(COM_velocity[:,:,0],axis=0)
        xline=np.arange(0,np.max(x),0.001)
        yline=np.arange(0,np.max(x),0.001)

        plt.scatter(x,y,alpha=0.5)
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
        
        mean_temp_array[i]=np.mean( log_file_tuple[i][1000:,column])
        plt.plot(log_file_tuple[i][1000:,column])
plt.show() 
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
plt.axhline(np.mean(mean_temp_array),label="$\\bar{T}="+str(sigfig.round(np.mean(mean_temp_array),sigfigs=5))+"$")
plt.ylim(0.95,1.05)
plt.legend()
plt.tight_layout()
plt.show()

          

      
# %% area vector analysis 

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
     strain_plot=np.linspace(0,strain_total,spherical_coords_array[i,:,0,2].shape[0])
     plt.plot( strain_plot,spherical_coords_array[i,:,0,1])
     plt.show()

#%%
for i in range(e_in,e_end):
     strain_plot=np.linspace(0,strain_total,spherical_coords_array[i,:,0,2].shape[0])
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
    sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,:,:,0]),
                label ="$\dot{\gamma}="+str(erate[i])+"$")

plt.legend()
plt.show()
#%% theta 
# could just plot a few of them 
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
skip_array=np.arange(0,e_end,4)
skip_array_2=np.arange(0,int(no_timesteps[0]/100),100)



for i in range(skip_array.size):
    #for j in range(j_):


        i=skip_array[i]
        
        # sns.displot( data=np.ravel(spherical_coords_tuple[i][:,200000,:,1]),
        #             label ="$\dot{\gamma}="+str(erate[i])+"$", kde=True)
        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,skip_array_2[j],:,1]),
        #             label="output_range:"+str(skip_array_2[j]))
        data=np.ravel(spherical_coords_tuple[i][:,-500:,:,1])
        periodic_data=np.array([data-2*np.pi,data,data+2*np.pi])  

        sns.kdeplot( data=np.ravel(periodic_data),
                    label ="$\dot{\gamma}="+str(erate[i],)+"$")#bw_adjust=0.1
        
        # mean_data=np.mean(spherical_coords_tuple[0][:,-1,:,1],axis=0)      
        #plt.hist(np.ravel(spherical_coords_tuple[i][:,-100,:,1]))
        # bw adjust effects the degree of smoothing , <1 smoothes less
        plt.xlabel("$\Theta$")
        plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
        plt.xlim(-np.pi,np.pi)
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=[1.1, 0.45])
plt.show()


#%% chi squared test
# generate uniform data 
rng = np.random.default_rng(1234)
s = rng.uniform(-np.pi,np.pi,n_plates*j_)
bins=10
# make into histograms 
data=spherical_coords_tuple[0][0,-1,:,1]
freq_exp=np.histogram(s,bins=bins)[0]
freq_obv=np.histogram(data,bins=bins)[0]
degrees_freedom=bins-1
chi_stat,pvalue=scipy.stats.chisquare(f_obs=freq_obv,ddof=0)
print(chi_stat)
print(pvalue)

chi_stat_table=scipy.stats.chi2.ppf(0.05,degrees_freedom)
print(chi_stat_table)





#%% phi 
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
for i in range(skip_array.size):
    #for j in range(skip_array_2.size):
        i=skip_array[i]

        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,skip_array_2[j],:,2]),
        #              label="output_range:"+str(skip_array_2[j]))
        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,-1,:,2]),
        #              label ="$\dot{\gamma}="+str(erate[i])+"$")
        data=np.ravel(spherical_coords_tuple[i][:,-500,:,2])
        periodic_data=np.array([data,np.pi-data])  
        sns.kdeplot( data=np.ravel(periodic_data),
                      label ="$\dot{\gamma}="+str(erate[i])+"$")
                   
        #plt.hist(np.ravel(spherical_coords_tuple[i][:,-1,:,2]))

plt.xlabel("$\Phi$")
plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
plt.ylabel('Density')
plt.legend(bbox_to_anchor=[1.1, 0.45])
plt.xlim(0,np.pi/2)
plt.show()

#%% extension_vectors
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
eq_spring_length=3*np.sqrt(3)/2
skip_array=np.arange(0,e_end,3)
for i in range(skip_array.size):
    i=skip_array[i]
# for i in range(e_in,e_end):

    # sns.kdeplot(eq_spring_length-np.ravel(interest_vectors_tuple[i][:,:,2:5]),
    #              label ="$K="+str(K)+"$")
                 #label ="$\dot{\gamma}="+str(erate[i])+"$")
    sns.kdeplot(eq_spring_length+0.125-np.ravel(interest_vectors_tuple[i][:,:,2:5]),
                  label ="$\dot{\gamma}="+str(erate[i])+"$")
    
plt.xlabel("$\Delta x$")

plt.ylabel('Density')
#plt.legend(bbox_to_anchor=[1.1, 0.45])

plt.legend(fontsize=11) 
plt.xlim(-3,2)
plt.show()


#%% plotting stress tensor entry distributions 

skip_array=np.arange(0,e_end,4)
for i in range(skip_array.size):
     i=skip_array[i]
     sns.kdeplot(np.ravel(spring_force_positon_tensor_tuple[i][:,:,:,0]),label ="$\dot{\gamma}="+str(erate[i])+"$")
plt.title("$\sigma_{xx}$ distribution")
plt.xlim(-200,400)
plt.legend()
plt.show()


for i in range(skip_array.size):
     i=skip_array[i]
     sns.kdeplot(np.ravel(spring_force_positon_tensor_tuple[i][:,:,:,1]),label ="$\dot{\gamma}="+str(erate[i])+"$")
plt.title("$\sigma_{yy}$ distribution")
plt.xlim(-200,400)
plt.legend()
plt.show()

for i in range(skip_array.size):
     i=skip_array[i]
     sns.kdeplot(np.ravel(spring_force_positon_tensor_tuple[i][:,:,:,2]),label ="$\dot{\gamma}="+str(erate[i])+"$")
plt.title("$\sigma_{zz}$ distribution")
plt.xlim(-200,400)
plt.legend()
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
