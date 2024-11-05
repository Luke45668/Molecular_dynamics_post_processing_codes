##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code processes data from the continuous extension simulations of plates utlising cfg files to 
apply the correct transformation to the dump outputs


"""
#
import os
from random import sample
import numpy as np
import regex as re
import pandas as pd
import sigfig
import scipy.stats
from datetime import datetime
import mmap
import math as m 
import glob 
from reading_lammps_module import *
path_2_post_proc_module= '/home/ucahlrl/python_scripts/MPCD_post_processing_codes'
os.chdir(path_2_post_proc_module)
import seaborn as sns
import glob 
from post_MPCD_MP_processing_module import *
from post_langevin_module import * 
import pickle as pck
from numpy.linalg import norm
from numpy.linalg import inv as matinv

# simulation parameters
damp=0.035
strain_total=100



erate=np.flip(np.linspace(0.5,0.005,24))
no_timesteps=np.array([[ 394351000,   74345000,  410411000,  283440000,  216470000,
         175098000,  147003000, 1266770000, 1112892000,  992349000,
         895367000,  815654000,  748974000,  692372000,  643724000,
         601464000,  564410000,  531657000,  502497000,  476369000,
         452824000,  431497000,  412089000,  394351000]])

timestep_multiplier=np.array(
[0.00005,0.00005,0.00005,0.00005,
0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,0.00005,
0.0005,0.0005,0.0005,0.0005,0.0005,0.005,
0.005])*4
md_step=0.005071624521210362*timestep_multiplier
thermo_vars='         KinEng         PotEng         Press           Temp         Ecouple       Econserve    c_uniaxnvttemp'

dump_start_line ='ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]'
dump_start_line_posvel = "ITEM: ATOMS id type x y z vx vy vz"

K=300
j_=10
box_size=100
eq_spring_length=3*np.cos(np.pi/6)
mass_pol=5 
n_plates=100
n_particles=6*n_plates
filepath="/home/ucahlrl/Scratch/output/nvt_runs/final_plate_runs/"
path_2_log_files=filepath
Path_2_dump=filepath

# file name search strings

log_general_name_string="log.*K_"+str(K)
cfg_general_name_before_string="before*.cfg"
cfg_general_name_after_string="dump.*K_"+str(K)+"_*.cfg"
log_general_name_string=("log.**K_"+str(K))
posvel_dump_general_name_string="*_UEF_flat_elastic_*K_"+str(K)+".dump"
force_dump_general_name_string="*_UEF_FE_tensor_*K_"+str(K)+".dump"


# grabbing file names and organising 


# grab file names 
(realisation_name_force_dump,
 realisation_name_cfg_before,
 count_mom,count_phantom,
 realisation_name_log,
 count_log,
 realisation_name_posvel_dump,
 count_dump,
 realisation_name_cfg_after,
 count_pol)= VP_and_momentum_data_realisation_name_grabber(cfg_general_name_after_string,
                                                                     log_general_name_string,
                                                                     cfg_general_name_before_string,
                                                                     force_dump_general_name_string,
                                                                     path_2_log_files,
                                                                     posvel_dump_general_name_string)


realisations_for_sorting_after_cfg=[]
# first sort cfg via realisation and erate 
realisation_split_index=6
erate_index=15
realisation_name_after_sorted_final_cfg=org_names(realisations_for_sorting_after_cfg,
                                                      realisation_name_cfg_after,
                                                     realisation_split_index,
                                                     erate_index)
# then sort cfg via timestep and erate 
timestep_split_index=20
erate_index=15
realisations_for_sorting_after_cfg=[]
realisation_name_after_sorted_final_cfg=org_names(realisations_for_sorting_after_cfg,
                                                      realisation_name_after_sorted_final_cfg,
                                                     timestep_split_index,
                                                     erate_index)

# grabbing log files and sorting 
realisation_split_index=6
erate_index=15
realisations_for_sorting_after_log=[]
realisation_name_log_sorted_final=org_names(realisations_for_sorting_after_log,
                                                     realisation_name_log,
                                                     realisation_split_index,
                                                     erate_index)

# grabbing force dump file and sorting 
realisations_for_sorting_force_dump=[]
realisation_name_force_dump_sorted_final=org_names(realisations_for_sorting_force_dump,
                                                     realisation_name_force_dump,
                                                     realisation_split_index,
                                                     erate_index)
# grabbing position velocity dump and sorting 

realisations_for_sorting_posvel_dump=[]
realisation_name_posvel_dump_sorted_final=org_names(realisations_for_sorting_posvel_dump,
                                                     realisation_name_posvel_dump,
                                                     realisation_split_index,
                                                     erate_index)


print(len(realisation_name_after_sorted_final_cfg))
# the following lists should be equal length 
print(len(realisation_name_log_sorted_final))
print(len(realisation_name_force_dump_sorted_final))
print(len(realisation_name_posvel_dump_sorted_final))

# check how many full sets of data we have 
log_file_size_array=np.zeros((2,erate.size,j_))
log_name_list=glob.glob("log.*K_"+str(K))
count=np.zeros((erate.size)).astype("int")
count_failed=np.zeros((erate.size)).astype("int")
failed_files=[]
passed_files=[]
real_target=10
# can scan all the files and produce a list of files that pass test
# check number of files in log file, this will be more clear than size
for file in log_name_list:

    split_name=file.split('_')
    erate_ind=int(np.where(erate==float(split_name[15]))[0][0])
    
    realisation_ind=int(split_name[6])
    spring_stiff=int(split_name[19])


    try:
        file_size_rows=log2numpy_reader(file,
                                filepath,
                                thermo_vars).shape[0]
        #print(file_size_rows)
        log_file_size_array[0,erate_ind,count[erate_ind]]=file_size_rows
        if count[erate_ind]==real_target:
           
            continue

        elif file_size_rows<1000:
            continue
    
        else:
            passed_files.append(file)
            count[erate_ind]+=1
        
       
        

    except:
        # if count[erate_ind]==10:
            failed_files.append(file)
            count_failed[erate_ind]+=1

            continue
        
              
        # log_file_size_array[0,erate_ind,count[erate_ind]]=0
        # count[erate_ind]+=1
        # continue 

print("count array",count)

success_count=list(count).count(j_)

print(success_count)




# file processing loop
Path_2_dump=filepath

# tuples for storing info 
spring_force_positon_tensor_tuple=()
log_file_tuple=()
dir_vector_tuple=()
transformed_vel_tuple=()
transformed_pos_tuple=()

e_in=0
e_end=success_count
count=e_in

for i in range(e_in,e_end):
    i_=(count*j_)
    print("i_",i_)
    i_=0


    # check output dimension of force dump
    outputdim_force_dump=dump2numpy_tensor_1tstep(dump_start_line,
                                    Path_2_dump,
                                     realisation_name_force_dump_sorted_final[i_],
                                     n_plates,300,6).shape[0]
    # check output dimension of pos vel dump   
    outputdim_posvel_dump=int(dump2numpy_f(dump_start_line_posvel,
                                    Path_2_dump,
                                    realisation_name_posvel_dump_sorted_final[i_],
                                    n_plates*6).shape[0]/(n_plates*6))

    # outputdim_cfg_dump=cfg2numpy_coords(Path_2_dump,realisation_name_after_sorted_final_cfg[i_],
    #                   n_plates*2, 4).shape[0]
    # check output dimension of log file dump
    outputdim_log=log2numpy_reader(realisation_name_log_sorted_final[i_],
                                                    path_2_log_files,
                                                    thermo_vars).shape[0]
    
    # creating arrays to store output data 

    spring_force_positon_array=np.zeros((j_,outputdim_force_dump,300,6))
    transform_dump_array=np.zeros((j_,outputdim_posvel_dump,600,3))
    transform_vel_array=np.zeros((j_,outputdim_posvel_dump,600,3))
    dirn_vector_array=np.zeros((j_,outputdim_force_dump,300,3))
    log_file_array=np.zeros((j_,outputdim_log,8))
    

    for j in range(j_):
            # define realisation index to ensure we are looking at correct section of list of realisation names
            # j_ * count skips the loop to the correct position 
            j_index=j+(j_*count)
            print(j_index)


            # extract log file data for realisation
            log_file_array[j,:,:]=log2numpy_reader(realisation_name_log_sorted_final[j_index],
                                                        path_2_log_files,
                                                        thermo_vars)
             
            # extract box vectors from posvel dump
            box_coords_from_dump=dump2numpy_box_coords_1tstep( Path_2_dump,realisation_name_posvel_dump_sorted_final[j_index],n_plates)
            # extract full particle posvel dump 
            posvel_from_dump_all=dump2numpy_f(dump_start_line_posvel,
                                    Path_2_dump,
                                    realisation_name_posvel_dump_sorted_final[j_index],
                                    n_plates*6).astype("float")
            # reshape to timesteps x n particles x n out puts 
            posvel_from_dump_all=np.reshape(posvel_from_dump_all,(1000,600,8))

            # extract forces and directions from force dump
            force_dirn_dump=dump2numpy_tensor_1tstep(dump_start_line,
                                    Path_2_dump,
                                    realisation_name_force_dump_sorted_final[j_index],
                                    n_plates,300,6)
            # columns 1-3 are force components 
            db_forces=force_dirn_dump[:,:,0:3]
            # columns 4-6 are direction components 
            db_dirns=force_dirn_dump[:,:,3:6]
            
            list_box_vec=[] #not used according to finder consider deleting 

            # creatings array to store the box vectors when extracted
            box_vec_array_dump_unsort=np.zeros((1000,3,3))
            # creating array to store the box vectors after sorting, they must match up to the correc timestep
            box_vec_array_dump=np.zeros((1000,3,3))


            count_cfg=0 #not used according to finder consider deleting 
            cfg_indices=np.arange(j,10000+j,10) # enables the next loop to look at the appropriate range of cfg files, 
            for k in range(1000):
                cfg_k_index=cfg_indices[k]
            
                #accquiring dump box vector for timestep k
                

                box_vec_list_dump=dump2numpy_box_coords_1tstep( Path_2_dump,
                                                                realisation_name_posvel_dump_sorted_final[j_index],
                                                                600)[k][5:8]
                
                #accquire cfg box vector and coordinates
                full_cfg_coord_after,box_vector_array_cfg=cfg2numpy_coords(Path_2_dump,
                                                                            realisation_name_after_sorted_final_cfg[cfg_k_index],
                                                                            n_plates*6, 4)
                

                full_cfg_coord_after_sorted=full_cfg_coord_after[full_cfg_coord_after[:,3].argsort()]
                posvel_from_dump_sing=posvel_from_dump_all[k]
                posvel_from_dump_sing_sorted=posvel_from_dump_sing[posvel_from_dump_sing[:,0].argsort()]
                
                
                box_vec_array_dump=np.zeros((3,3))
                box_vec_array_upper_tri=np.zeros((3,3))
                
                # putting dump box vectors into upper triangular matrix 
                # this can be a function 
                # def bounds_2_upper_tri()
                for l in range(3):
                    box_vec_array_dump[l,:]= box_vec_list_dump[l].split(" ")
                    xy= box_vec_array_dump[0,2]
                    xz= box_vec_array_dump[1,2]
                    yz= box_vec_array_dump[2,2]
                    xlo= box_vec_array_dump[0,0]-np.min(np.array([0,xy,xz,xy+xz]))
                    xhi= box_vec_array_dump[0,1]-np.max(np.array([0,xy,xz,xy+xz]))
                    ylo= box_vec_array_dump[1,0]-np.min(np.array([0,yz]))
                    yhi= box_vec_array_dump[1,1]-np.max(np.array([0,yz]))
                    zlo= box_vec_array_dump[2,0]
                    zhi= box_vec_array_dump[2,1]
                    box_vec_array_upper_tri=np.array([[xhi-xlo,xy,xz],
                                           [0,yhi-ylo,yz],
                                           [0,0,zhi-zlo]])
                    
                # sort positions from dump
                
                # calculate q transform and test against cfg

                def q_matrix_transform_plate(box_vector_array_cfg,box_vec_array_upper_tri,
                      full_cfg_coord_after_sorted,posvel_from_dump_sing_sorted,
                       n_particles,n_plates,db_forces,db_dirns,k):
    
                    # inverting lammps frame box vectors 
                    inv_box_vec_array=matinv(box_vec_array_upper_tri) 

                    # multiply  Q= FL^{-1}
                    Q_matrix=np.matmul(box_vector_array_cfg.T,inv_box_vec_array) 
                    
                    unscaled_cfg=np.zeros((n_particles,3))
                    transform_dump_coords=np.zeros((n_particles,3))  
                    transform_force_dump=np.zeros((n_plates*3,6))
                    transform_dump_velocities=np.zeros((n_particles,3)) 


                    
                    for m in range(n_particles):
                        # convert scaled cfg coords to unscaled coords by multiplication by box vector from cfg
                        unscaled_cfg[m]=np.matmul(box_vector_array_cfg.T,full_cfg_coord_after_sorted[m][0:3])
                        # transform dump coords by q matrix 
                        transform_dump_coords[m]=np.matmul(Q_matrix,posvel_from_dump_sing_sorted[m][2:5])
                        transform_dump_velocities[m]=np.matmul(Q_matrix,posvel_from_dump_sing_sorted[m][5:8])

                    
                    # compared unscaled cfg to transformed dump, they should match 
                    # this part needs more work 
                    # print(full_cfg_coord_after_sorted[0][0:3])
                    # comparison=np.abs(unscaled_cfg-transform_dump_coords)
                    # print(np.max(comparison))
                    # plt.plot(comparison)
                    # plt.show()
                    # plt.plot(np.ravel(unscaled_cfg))
                    # plt.show()
                    # plt.plot(np.ravel(transform_dump_coords))
                    # plt.show()



                    #apply Q matrix transform to force components and direction components 
                    for m in range(n_plates*3):
                        transform_force_dump[m,0:3]=np.matmul(Q_matrix,db_forces[k,m])
                        transform_force_dump[m,3:6]=np.matmul(Q_matrix,db_dirns[k,m])



                    
                    return transform_dump_coords,transform_force_dump,transform_dump_velocities





                transform_dump_coords,transform_force_dump,transform_dump_velocties=q_matrix_transform_plate(box_vector_array_cfg,box_vec_array_upper_tri,
                        full_cfg_coord_after_sorted,posvel_from_dump_sing_sorted
                        ,n_particles,n_plates,db_forces,db_dirns,k)
                

                
                dump_data=transform_force_dump
                dirn_vector_array[j,k]=dump_data[:,3:]
                transform_dump_array[j,k]=transform_dump_coords
                transform_vel_array[j,k]=transform_dump_velocties
                

            
                spring_force_positon_array[j,k,:,0]=-dump_data[:,0]*dump_data[:,3]#xx
                spring_force_positon_array[j,k,:,1]=-dump_data[:,1]*dump_data[:,4]#yy
                spring_force_positon_array[j,k,:,2]=-dump_data[:,2]*dump_data[:,5]#zz
                spring_force_positon_array[j,k,:,3]=-dump_data[:,0]*dump_data[:,5]#xz
                spring_force_positon_array[j,k,:,4]=-dump_data[:,0]*dump_data[:,4]#xy
                spring_force_positon_array[j,k,:,5]=-dump_data[:,1]*dump_data[:,5]#yz
    
    
    spring_force_positon_mean=np.mean(np.mean(spring_force_positon_array,axis=0),
                            axis=1)
    lgf_mean=np.mean(log_file_array,axis=0)    
    log_file_tuple=log_file_tuple+(lgf_mean,)
    spring_force_positon_tensor_tuple=spring_force_positon_tensor_tuple+\
        (spring_force_positon_array,)
    dir_vector_tuple=dir_vector_tuple+(dirn_vector_array,)
    transformed_pos_tuple=transformed_pos_tuple +(transform_dump_array,)
    transformed_vel_tuple=transformed_vel_tuple +(transform_vel_array,)
    count+=1
            
                        
                
                    
                
                    
                


                   

    
# save tuples to avoid needing the next stage 

label='damp_'+str(damp)+'_K_'+str(K)+'_'
import pickle as pck

folder_check_or_create(filepath,"saved_tuples")

with open(label+'spring_force_positon_tensor_tuple.pickle', 'wb') as f:
    pck.dump(spring_force_positon_tensor_tuple, f)

with open(label+'log_file_tuple.pickle', 'wb') as f:
    pck.dump(log_file_tuple, f)

with open(label+'dirn_vector_tuple.pickle','wb') as f:
    pck.dump(dir_vector_tuple,f)

with open(label+'transformed_pos_tuple.pickle','wb') as f:
    pck.dump(transformed_pos_tuple,f)

with open(label+'transformed_vel_tuple.pickle','wb') as f:
     pck.dump(transformed_vel_tuple,f)


