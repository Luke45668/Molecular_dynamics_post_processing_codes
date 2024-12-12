##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file processes the all files from brownian dynamics simulations of many flat elastic particles.


after an MPCD simulation. 
"""
# Importing packages
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

path_2_post_proc_module= '/home/ucahlrl/python_scripts/MPCD_post_processing_codes'
os.chdir(path_2_post_proc_module)

from log2numpy import *
from dump2numpy import *
import glob 
from MPCD_codes.post_MPCD_MP_processing_module import *
import pickle as pck
from numpy.linalg import norm
import pickle as pck


damp=0.035
strain_total=100

erate=np.flip(np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.02,0.01,0.005,0]))

no_timesteps=np.flip(np.array([ 3944000,  4382000,  4929000,  5634000,  6573000,  7887000,
         9859000, 13145000, 19718000,  2253000,  2629000,  3155000,
         3944000,  4929000,  6573000,  9859000, 19718000, 39435000,
        78870000, 10000000]))



thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
K=50
j_=3
box_size=100
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5
n_plates=100


filepath="/home/ucahlrl/Scratch/output/langevin_runs/100_particle/run_312202/sucessful_runs_3_reals/"
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



# save tuples to avoid needing the next stage 
#make sure to comment this out after use
label='damp_'+str(damp)+'_K_'+str(K)+'_'


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

with open(label+"interest_vectors_tuple.pickle",'wb') as f:
    pck.dump(interest_vectors_tuple,f)


