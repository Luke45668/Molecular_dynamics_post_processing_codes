#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue April 4th 2023

This is the post processing analysis module, contains velocity profile readers, momentum data output and flux calculator for 
muller plathe method 
@author: lukedebono 
"""
import os
from pyexpat.model import XML_CQUANT_PLUS
#from sys import exception
import numpy as np

import matplotlib.pyplot as plt
import regex as re
import pandas as pd
#import pyswarms as ps

plt.rcParams.update(plt.rcParamsDefault)
#plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
#import seaborn as sns

import scipy.stats
from datetime import datetime
import glob 

def VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string):
    #os.chdir('/Users/lukedebono/Documents/LAMMPS_projects_mac_book/OneDrive_1_24-02-2023/MYRIAD_lammps_runs/'+filepath)
    os.chdir('/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/'+filepath)
    count_VP=0
    realisation_name_VP = []   
    count_mom=0
    realisation_name_Mom=[]  
    count_log=0 
    realisation_name_log=[]
    count_dump=0
    realisation_name_dump =[]
    count_TP=0
    realisation_name_TP=[]

    for name in glob.glob(TP_general_name_string):
   
        count_TP=count_TP+1    
        realisation_name_TP.append(name)

    for name in glob.glob(VP_general_name_string):
   
        count_VP=count_VP+1    
        realisation_name_VP.append(name)
    

    for name in glob.glob(Mom_general_name_string):
    
        count_mom=count_mom+1    
        realisation_name_Mom.append(name)
        
    for name in glob.glob(log_general_name_string):
        
        count_log=count_log+1    
        realisation_name_log.append(name)
        
    for name in glob.glob(dump_general_name_string):
        
        count_dump=count_dump+1    
        realisation_name_dump.append(name)
    
    
    if count_VP!=count_mom:
       breakpoint()
    else:
        print('VP and Mom data consistent')
    
    if count_dump!=count_VP!=count_mom!=count_log:
        print("Should there be dump files on this run? If not ignore this message, if yes there is an ERROR")
    else:
        print("dump file data consistent")
        
    
   #number_of_solutions= int(count_VP/(j_*swap_number.size*swap_rate.size))
        
    return realisation_name_Mom,realisation_name_VP,count_mom,count_VP,realisation_name_log,count_log,realisation_name_dump,count_dump,realisation_name_TP,count_TP



def SRD_counter_solution_grabber_duplicate_remover(loc_no_SRD,count_VP,realisation_name_VP):
    no_SRD=[]
    for i in range(0,count_VP):
        no_srd=realisation_name_VP[i].split('_')
        no_SRD.append(no_srd[loc_no_SRD])
        
    no_SRD.sort(key=int)
    no_SRD_key=[]
    #using list comprehension to remove duplicates
    [no_SRD_key.append(x) for x in no_SRD if x not in no_SRD_key]
    return no_SRD_key

def VP_organiser_and_reader(loc_no_SRD,loc_org_var_1,loc_org_var_2,loc_Realisation_index,box_side_length_scaled,j_,number_of_solutions,org_var_1,org_var_2,no_SRD_key,realisation_name_VP,Path_2_VP,chunk,equilibration_timesteps,VP_ave_freq,no_timesteps,VP_output_col_count,count_VP):
    from velP2numpy import velP2numpy_f
    marker=-1
    error_count=0 
    VP_data_upper=np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,9,int(no_timesteps/VP_ave_freq),j_))
    VP_data_lower=np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,9,int(no_timesteps/VP_ave_freq),j_))
    VP_z_data_upper=np.zeros((number_of_solutions,1,9))
    VP_z_data_lower=np.zeros((number_of_solutions,1,9))
    
    for i in range(0,count_VP):
        filename=realisation_name_VP[i].split('_')
        marker=marker+1
        no_SRD=filename[loc_no_SRD]
        z=no_SRD_key.index(no_SRD)
        realisation_index=filename[loc_Realisation_index]
        j=int(float(realisation_index))-1
        if isinstance(filename[loc_org_var_1], int):
            org_var_1_find_from_file_name=int(filename[loc_org_var_1])
            m=np.where(org_var_1==org_var_1_find_from_file_name)
        else: 
            org_var_1_find_from_file_name=float(filename[loc_org_var_1])
            m=np.where(org_var_1==org_var_1_find_from_file_name)
        if  isinstance(filename[loc_org_var_2], int):
            org_var_2_find_from_file_name=int(filename[loc_org_var_2])
            k=np.where(org_var_2==org_var_2_find_from_file_name)
        else:    
            org_var_2_find_from_file_name=float(filename[loc_org_var_2])
            k=np.where(org_var_2==org_var_2_find_from_file_name)
        
        
        try: 
            realisation_name=realisation_name_VP[i]
            
            VP_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[0]
            #print(VP_data)
            VP_data_upper[z,m,k,:,:,j] = VP_data[1:10,:]
            VP_data_lower[z,m,k,:,:,j] = VP_data[11:,:]
            
        except Exception as e:
            print('Velocity Profile reading failed')
            print(realisation_name)
            error_count=error_count+1 
            break
            
        VP_z_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[1]     
        
        VP_z_data_upper[:,0,:] = VP_z_data[1:10].astype('float64')* box_side_length_scaled.T    
        VP_z_data_lower[:,0,:] = VP_z_data[11:].astype('float64')* box_side_length_scaled.T
        
    return VP_data_upper,VP_data_lower,error_count,filename,VP_z_data_upper,VP_z_data_lower

def single_VP_reader(loc_no_SRD,loc_EF,loc_SN,loc_Realisation_index,box_side_length_scaled,j_,number_of_solutions,swap_number,swap_rate,no_SRD_key,realisation_name_VP,Path_2_VP,chunk,equilibration_timesteps,VP_ave_freq,no_timesteps,VP_output_col_count,count_VP):
    from velP2numpy import velP2numpy_f
    marker=-1
    error_count=0 
    VP_data_upper=np.zeros((number_of_solutions,swap_rate.size,swap_number.size,9,int(no_timesteps/VP_ave_freq),j_))
    VP_data_lower=np.zeros((number_of_solutions,swap_rate.size,swap_number.size,9,int(no_timesteps/VP_ave_freq),j_))

    for i in range(0,count_VP):
        filename=realisation_name_VP[i].split('_')
        marker=marker+1
        no_SRD=filename[loc_no_SRD]
        z=no_SRD_key.index(no_SRD)
        realisation_index=filename[loc_Realisation_index]
        j=int(float(realisation_index))
        EF=int(filename[loc_EF])
        m=np.where(swap_rate==EF)
        SN=int(filename[loc_SN])
        k=np.where(swap_number==SN)
        realisation_name=realisation_name_VP[i]
        try: 
            VP_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[0]
            VP_data_upper= VP_data[1:10,:]
            VP_data_lower= VP_data[11:,:]
            
        except Exception as e:
            print('Velocity Profile Data faulty')
            error_count=error_count+1 
            continue
    VP_z_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[1]     

    VP_z_data_upper = np.array([VP_z_data[1:10].astype('float64')])* box_side_length_scaled.T    
    VP_z_data_lower =np.array([ VP_z_data[11:].astype('float64') ])* box_side_length_scaled.T
    return VP_data_upper,VP_data_lower,error_count,filename,VP_z_data_upper,VP_z_data_lower


def log_file_organiser_and_reader(org_var_log_1,loc_org_var_log_1,org_var_log_2,loc_org_var_log_2,j_,log_file_row_count,log_file_col_count,count_log,realisation_name_log,log_realisation_index,Path_2_log,thermo_vars):
    
    log_file_tuple=()
    from log2numpy import log2numpy_reader
    for i in range(0,org_var_log_1.size):
        log_4d_array_with_all_realisations=np.zeros((org_var_log_2.size,j_,int(log_file_row_count),log_file_col_count))
        log_file_tuple=log_file_tuple+(log_4d_array_with_all_realisations,)
        
    averaged_log_file=np.zeros((org_var_log_1.size,org_var_log_2.size,int(log_file_row_count),4))

    for i in range(0,count_log):
        filename=realisation_name_log[i].split('_')
        realisation_index=int(float(realisation_name_log[i].split('_')[log_realisation_index]))
        print(realisation_index)
        if isinstance(filename[loc_org_var_log_1],int):
            org_var_log_1_find_in_name=int(filename[loc_org_var_log_1])
            tuple_index=np.where(org_var_log_1==org_var_log_1_find_in_name)[0][0]
        else:
            org_var_log_1_find_in_name=float(filename[loc_org_var_log_1])
            tuple_index=np.where(org_var_log_1==org_var_log_1_find_in_name)[0][0]
        
        if isinstance(filename[loc_org_var_log_2],int):
            org_var_log_2_find_in_name=int(filename[loc_org_var_log_2])
            array_index_1= np.where(org_var_log_2==org_var_log_2_find_in_name)[0][0] 
        else:
            org_var_log_2_find_in_name=float(filename[loc_org_var_log_2])
            array_index_1= np.where(org_var_log_2==org_var_log_2_find_in_name)[0][0] 
            
        # multiple swap numbers and swap rates
        #log_file_tuple[np.where(swap_rate==swap_rate_org)[0][0]][np.where(swap_number==swap_number_org)[0][0],realisation_index,:,:]=log2numpy_reader(realisation_name_log[i],Path_2_log,thermo_vars)
        log_file_tuple[tuple_index][array_index_1,realisation_index,:,:]=log2numpy_reader(realisation_name_log[i],Path_2_log,thermo_vars)


    for k in range(0,org_var_log_1.size):
        for i in range(0,org_var_log_2.size):
            averaged_log_file[k,i,:,:]=np.mean(log_file_tuple[k][i],axis=0)
    
    return  averaged_log_file
    

def mom_file_data_size_reader(j_,number_of_solutions,count_mom,realisation_name_Mom,no_SRD_key,org_var_mom_1,org_var_mom_2,Path_2_mom_file):
    from mom2numpy import mom2numpy_f
    pass_count=0
    size_list=[]
    
    for i in range(0,count_mom):
       
        
        realisation_name=realisation_name_Mom[i]
        
        
        
        mom_data_test=mom2numpy_f(realisation_name,Path_2_mom_file)  
        size_list.append(mom_data_test.shape)
        
        pass_count=pass_count+1 
    
    size_list.sort()
    size_list.reverse()
    size_list=list(dict.fromkeys(size_list))
    size_array=np.array(size_list)
    
    mom_data=()
    for i in range(0,org_var_mom_1.size):
    
         mom_data= mom_data+(np.zeros((number_of_solutions,org_var_mom_2.size,j_,(size_array[i,0]))),)
         
         
         
    return size_array,mom_data,pass_count

def Mom_organiser_and_reader(mom_data,count_mom,realisation_name_Mom,no_SRD_key,org_var_mom_1,loc_org_var_mom_1,org_var_mom_2,loc_org_var_mom_2,Path_2_mom_file):
    from mom2numpy import mom2numpy_f
    error_count_mom=0
    failed_list_realisations=[]
    for i in range(0,count_mom):
        filename=realisation_name_Mom[i].split('_')
    
        no_SRD=filename[8]
        z=no_SRD_key.index(no_SRD)
    
        realisation_index=filename[7]
        j=int(float(realisation_index))-1
        if isinstance(filename[loc_org_var_mom_1],int):
            org_var_mom_1_find_in_name=int(filename[loc_org_var_mom_1])
            tuple_index=np.where(org_var_mom_1==org_var_mom_1_find_in_name)[0][0]
        else:
            org_var_mom_1_find_in_name=float(filename[loc_org_var_mom_1])
            tuple_index=np.where(org_var_mom_1==org_var_mom_1_find_in_name)[0][0]
    
        if isinstance(filename[loc_org_var_mom_2],int):
            org_var_mom_2_find_in_name=int(filename[loc_org_var_mom_2])
            array_index_1= np.where(org_var_mom_2==org_var_mom_2_find_in_name)[0][0] 
        else:
            org_var_mom_2_find_in_name=float(filename[loc_org_var_mom_2])
            array_index_1= np.where(org_var_mom_2==org_var_mom_2_find_in_name)[0][0] 
        
        realisation_name=realisation_name_Mom[i]
        
        #try:
        mom_data[tuple_index][z,array_index_1,j,:]=mom2numpy_f(realisation_name,Path_2_mom_file)  
        
        

        
            
        # except Exception as e:
        #     print('Mom Data faulty')
        #     error_count_mom=error_count_mom+1 
        #     failed_list_realisations.append(realisation_name)
        #     break
    return mom_data,error_count_mom,failed_list_realisations

# Averaging for one file 
# VP_z_data_upper_repeated= np.repeat(VP_z_data_upper.T,VP_data_upper.shape[1],axis=1)
# VP_z_data_lower_repeated= np.repeat(VP_z_data_lower.T,VP_data_lower.shape[1],axis=1)
# pearson_coeff_upper= np.zeros(VP_data_upper.shape[1])
# pearson_coeff_lower= np.zeros(VP_data_lower.shape[1])
# shear_rate_upper= np.zeros(VP_data_upper.shape[1])    
# shear_rate_lower= np.zeros(VP_data_lower.shape[1])
# shear_rate_upper_error= np.zeros(VP_data_upper.shape[1]) 
# shear_rate_lower_error= np.zeros(VP_data_lower.shape[1])

# for i in range(0,VP_data_upper.shape[1]):
#     pearson_coeff_upper[i]=scipy.stats.pearsonr(VP_data_upper[:,i],VP_z_data_upper_repeated[:,i])[0]
#     shear_rate_upper[i]= scipy.stats.linregress(VP_z_data_upper_repeated[:,i],VP_data_upper[:,i]).slope
#     shear_rate_upper_error[i]= scipy.stats.linregress(VP_data_upper[:,i],VP_z_data_upper_repeated[:,i] ).stderr
#     pearson_coeff_lower[i] =scipy.stats.pearsonr(VP_data_lower[:,i],VP_z_data_lower_repeated[:,i] )[0]
#     shear_rate_lower[i]= scipy.stats.linregress(VP_z_data_lower_repeated[:,i] ,VP_data_lower[:,i]).slope  
#     shear_rate_lower_error[i]= scipy.stats.linregress(VP_data_lower[:,i],VP_z_data_lower_repeated [:,i] ).stderr 
# timestep_points=np.array([[[np.linspace(1,VP_data_upper.shape[1],int(float(no_timesteps)/float(VP_ave_freq)))]]])*VP_ave_freq

# plt.plot(timestep_points[0,0,0,:],shear_rate_upper[:])
# plt.plot(timestep_points[0,0,0,:],shear_rate_lower[:])
# plt.xlabel('$N_{t}[-]$')
# plt.ylabel('$\dot{\gamma}[\\tau]$',rotation='horizontal')
# plt.title(fluid_name+" simulation run with all $f_{v,x}$ and $N_{v,x}$, $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")

# plt.show()
# import log2numpy
# Path_2_log='/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/Test_data_solid/logs'
# realisation_name='log.H20_solid396988_inc_mom_output_no_rescale_243550_0.0_9112_118.77258268303078_0.001_1781_1000_10000_1000000_T_1.0_lbda_1.3166259218664098_SR_7_SN_1_rparticle_10.0'
# thermo_vars='         KinEng          Temp          TotEng'
# log_data= log2numpy.log2numpy(Path_2_log,thermo_vars,realisation_name)[0]
# fontsize=15
# labelpad=20
# #plotting temp vs time 
# plt.plot(log_data[:,0],log_data[:,2])
# temp=1
# x=np.repeat(temp,log_data[:,0].shape[0])
# plt.plot(log_data[:,0],x[:])
# plt.xlabel('$N_{t}[-]$',fontsize=fontsize)
# plt.ylabel('$T[\\frac{T k_{B}}{\\varepsilon}]$', rotation=0,fontsize=fontsize,labelpad=labelpad)
# plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+", $f_{v,x}=$"+str(swap_rate[0])+", $N_{v,x}=$"+str(swap_number[0])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
                
# plt.show()

# #plotting energy vs time 
# plt.plot(log_data[:,0],log_data[:,3])
# plt.xlabel('$N_{t}[-]$',fontsize=fontsize)
# plt.ylabel('$E_{t}[\\frac{\\tau^{2}}{\mu \ell^{2}}]$', rotation=0,fontsize=fontsize,labelpad=labelpad)
# plt.title(fluid_name+" simulation run $\phi=$"+str(phi)+", $f_{v,x}=$"+str(swap_rate[0])+", $N_{v,x}=$"+str(swap_number[0])+", $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
     
# plt.show()



def VP_data_averaging_and_stat_test_data(VP_z_data_upper,VP_z_data_lower,no_timesteps,VP_data_lower,VP_data_upper,number_of_solutions,org_var_1,org_var_2,VP_ave_freq):
    VP_data_lower_realisation_averaged = np.mean(VP_data_lower,axis=5) 
    VP_data_upper_realisation_averaged = np.mean(VP_data_upper,axis=5) 
    x_u= np.array(VP_data_upper_realisation_averaged)
    x_l= np.array(VP_data_lower_realisation_averaged)
    y_u=np.zeros((number_of_solutions,VP_z_data_upper.shape[2],VP_data_upper.shape[4]))
    y_l=np.zeros((number_of_solutions,VP_z_data_upper.shape[2],VP_data_upper.shape[4]))

    for z in range(number_of_solutions):
        y_u[z,:,:]= np.reshape(np.repeat(VP_z_data_upper[z,0,:],VP_data_upper.shape[4],axis=0),(VP_z_data_upper.shape[2],VP_data_upper.shape[4]))

        y_l[z,:,:]= np.reshape(np.repeat(VP_z_data_lower[z,0,:],VP_data_lower.shape[4],axis=0),(VP_z_data_lower.shape[2],VP_data_lower.shape[4]))

    pearson_coeff_upper= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_upper_realisation_averaged.shape[4]))
    pearson_coeff_lower= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_lower_realisation_averaged.shape[4]))
    shear_rate_upper= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_upper_realisation_averaged.shape[4]))    
    shear_rate_lower= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_lower_realisation_averaged.shape[4]))
    shear_rate_upper_error= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_upper_realisation_averaged.shape[4]))    
    shear_rate_lower_error= np.zeros((number_of_solutions,org_var_1.size,org_var_2.size,VP_data_lower_realisation_averaged.shape[4]))
    timestep_points=np.array([[[np.linspace(1,VP_data_upper.shape[4],int(float(no_timesteps)/float(VP_ave_freq)))]]])*VP_ave_freq
    timestep_points=np.repeat(timestep_points, number_of_solutions,axis=0)
    timestep_points=np.repeat(timestep_points, org_var_1.size,axis=1)     
    timestep_points=np.repeat(timestep_points, org_var_2.size,axis=2)   


    for z in range(0,number_of_solutions): 
        for m in range(0,org_var_1.size):
            for k in range(0,org_var_2.size):
                for i in range(0,VP_data_upper.shape[4]):
                        
                        pearson_coeff_upper[z,m,k,i] =scipy.stats.pearsonr(y_u[z,:,i],x_u[z,m,k,:,i] )[0]
                        shear_rate_upper[z,m,k,i]= scipy.stats.linregress(y_u[z,:,i],x_u[z,m,k,:,i] ).slope
                        shear_rate_upper_error[z,m,k,i]= scipy.stats.linregress(y_u[z,:,i],x_u[z,m,k,:,i] ).stderr
                        pearson_coeff_lower[z,m,k,i] =scipy.stats.pearsonr(y_l[z,:,i],x_l[z,m,k,:,i] )[0]
                        shear_rate_lower[z,m,k,i]= scipy.stats.linregress(y_l[z,:,i],x_l[z,m,k,:,i] ).slope  
                        shear_rate_lower_error[z,m,k,i]= scipy.stats.linregress(y_l[z,:,i],x_l[z,m,k,:,i] ).stderr 
     
    return pearson_coeff_upper,shear_rate_upper,pearson_coeff_lower,shear_rate_lower,timestep_points,VP_data_lower_realisation_averaged,VP_data_upper_realisation_averaged,shear_rate_upper_error,shear_rate_lower_error
        

    
def truncation_step_and_SS_average_of_VP_and_stat_tests(shear_rate_upper_error,shear_rate_lower_error,timestep_points,pearson_coeff_lower,pearson_coeff_upper,shear_rate_upper,shear_rate_lower,VP_ave_freq,truncation_timestep,VP_data_lower_realisation_averaged,VP_data_upper_realisation_averaged):
    truncation_index=int(truncation_timestep/VP_ave_freq)
    shear_rate_upper=shear_rate_upper[:,:,:,truncation_index:]
    shear_rate_upper_error=shear_rate_upper_error[:,:,:,truncation_index:]
    shear_rate_lower=shear_rate_lower[:,:,:,truncation_index:]
    
    shear_rate_lower_error=shear_rate_lower_error[:,:,:,truncation_index:]
    pearson_coeff_upper=pearson_coeff_upper[:,:,:,truncation_index:]
    pearson_coeff_lower=pearson_coeff_lower[:,:,:,truncation_index:]
    timestep_points=timestep_points[:,:,:,truncation_index:]
    VP_steady_state_data_lower_truncated=VP_data_lower_realisation_averaged[:,:,:,:,truncation_index:]
    VP_steady_state_data_upper_truncated=VP_data_upper_realisation_averaged[:,:,:,:,truncation_index:]

    VP_steady_state_data_lower_truncated_time_averaged=np.mean(VP_steady_state_data_lower_truncated,axis=4)
    VP_steady_state_data_upper_truncated_time_averaged=np.mean(VP_steady_state_data_upper_truncated,axis=4)
    shear_rate_upper_steady_state_mean = np.mean(shear_rate_upper, axis=3)
    shear_rate_lower_steady_state_mean = np.mean(shear_rate_lower, axis=3)
    standard_deviation_upper_SS=np.std(shear_rate_upper,axis=3)
    standard_deviation_lower_SS=np.std(shear_rate_lower,axis=3)
    
    
    shear_rate_upper_steady_state_mean_error = np.mean(shear_rate_upper_error, axis=3)
    shear_rate_lower_steady_state_mean_error = np.mean(shear_rate_lower_error, axis=3)
    pearson_coeff_lower_mean_SS=np.mean(pearson_coeff_lower,axis=3)
    pearson_coeff_upper_mean_SS=np.mean(pearson_coeff_upper,axis=3) 
    
    
    
    standard_deviation_upper_error=standard_deviation_upper_SS/ shear_rate_upper_steady_state_mean
    standard_deviation_lower_error=standard_deviation_lower_SS/shear_rate_lower_steady_state_mean

    return standard_deviation_upper_error,standard_deviation_lower_error,pearson_coeff_upper_mean_SS,pearson_coeff_lower_mean_SS,shear_rate_lower_steady_state_mean,shear_rate_upper_steady_state_mean,VP_steady_state_data_lower_truncated_time_averaged,VP_steady_state_data_upper_truncated_time_averaged,shear_rate_upper_steady_state_mean_error,shear_rate_lower_steady_state_mean_error
    
def mom_data_averaging_and_flux_calc(box_size_key,number_of_solutions,swap_number,truncation_timestep,swap_rate,scaled_timestep,no_timesteps,box_side_length_scaled,mom_data):
    mom_data_realisation_averaged=()
    number_swaps_before_truncation=(np.ceil(truncation_timestep/swap_rate)).astype(int)
    mom_data_realisation_averaged_truncated=()
    flux_x_momentum_z_direction=np.zeros((number_of_solutions,swap_number.size,swap_rate.size))
    total_run_time=scaled_timestep* (no_timesteps-truncation_timestep)
    
    flux_ready_for_plotting=np.zeros((number_of_solutions,swap_number.size,swap_rate.size))
    for z in range(0,number_of_solutions):    
        for i in range(0,swap_rate.size):
            box_area_nd=float(box_size_key[z])**2
            mom_data_realisation_averaged=mom_data_realisation_averaged+(np.mean(mom_data[i],axis=2),)


        #for i in range(0,swap_rate.size):

            mom_data_realisation_averaged_truncated=mom_data_realisation_averaged_truncated+(mom_data_realisation_averaged[i][:,:,number_swaps_before_truncation[i]:],)


        # now apply the MP formula 
            mom_difference= mom_data_realisation_averaged_truncated[i][z,:,-1]-mom_data_realisation_averaged_truncated[i][z,:,0]
            flux_x_momentum_z_direction[z,:,i]=(mom_difference)/(2*total_run_time*float(box_area_nd))
            
    flux_ready_for_plotting=np.log((np.abs(flux_x_momentum_z_direction)))
    
    return flux_ready_for_plotting,mom_data_realisation_averaged_truncated

    

###### Plotting section

def plotting_SS_velocity_profiles_with_all_swap_numbers(width_plot,height_plot,swap_number,swap_rate,VP_ave_freq,no_timesteps,VP_steady_state_data_lower_truncated_time_averaged,VP_steady_state_data_upper_truncated_time_averaged,VP_z_data_lower,VP_z_data_upper):
    # this needs labels and a title
    fig=plt.figure(figsize=(width_plot,height_plot))
    gs=GridSpec(nrows=2,ncols=1)
    #fig.suptitle('",$\Delta t_{MD}^{*}$: "+str(scaled_timestep)+", M:"+str(Solvent_bead_SRD_box_density_cp_1[0,0])+", $\Delta x^{*}$"+str(SRD_box_size_wrt_solid_beads[0,3]) ,size='large',ha='center')
    ax1= fig.add_subplot(gs[0,0],projection='3d')
    ax2= fig.add_subplot(gs[1,0],projection='3d')
    


    x_1=VP_steady_state_data_lower_truncated_time_averaged
    x_2=VP_steady_state_data_upper_truncated_time_averaged
    y_1=VP_z_data_lower
    y_2=VP_z_data_upper
    no_vps=int(no_timesteps/VP_ave_freq)
    for k in range(0,swap_number.size):
        for i in range(0,swap_rate.size):
    
            ax1.plot(x_1[0,i,k,:],y_1[:],swap_number[k])
            ax2.plot(x_2[0,i,k,:],y_2[:],swap_number[k])
            ax1.view_init(horizontal_angle, vertical_angle)
            ax2.view_init(horizontal_angle, vertical_angle)
    plt.show()
    
    
# to change the number or sepcfic velocity profiles change the swap rate vector.     
def plotting_SS_velocity_profiles(swap_rate_index_start,swap_rate_index_end,legend_x_pos, legend_y_pos,labelpadx,labelpady,fontsize,number_of_solutions,swap_number_choice_index,width_plot,height_plot,swap_number,swap_rate,VP_ave_freq,no_timesteps,VP_steady_state_data_lower_truncated_time_averaged,VP_steady_state_data_upper_truncated_time_averaged,VP_z_data_lower,VP_z_data_upper):
    for z in range(0,number_of_solutions):
        fig=plt.figure(figsize=(width_plot,height_plot))
        gs=GridSpec(nrows=1,ncols=1)

        ax1= fig.add_subplot(gs[0,0])
        #ax2= fig.add_subplot(gs[1,0])
        k=swap_number_choice_index



        x_1=VP_steady_state_data_lower_truncated_time_averaged[z]
        x_2=VP_steady_state_data_upper_truncated_time_averaged[z]
        y_1=VP_z_data_lower[z,:]
        y_2=VP_z_data_upper[z,:]

        
            
        for i in range(swap_rate_index_start,swap_rate_index_end):
           
                

                ax1.plot(y_1[:],x_1[i,k,:],label='$f_p=${}'.format(swap_rate[i]),marker='x')
                ax1.set_ylabel('$v_{x}\ [\\frac{\\tau}{\ell}]$',rotation=0,labelpad=labelpady, fontsize=fontsize)
                ax1.set_xlabel('$L_{z}\ [\ell^{-1}]$',rotation=0,labelpad=labelpadx,fontsize=fontsize)
                #ax2.plot(x_2[i,k,:],y_2[:],label='$f_p=${}'.format(swap_rate[i]))
                ax1.legend(frameon=False,loc=0,bbox_to_anchor=(legend_x_pos, legend_y_pos),fontsize=fontsize-4)
                #ax2.set_xlabel('$v_{x}\ [\\frac{\\tau}{\ell}]$',fontsize=fontsize)
                #ax2.set_ylabel('$L_{z}\ [\ell^{-1}]$',rotation=0,labelpad=labelpad,fontsize=fontsize)
                #ax2.legend(frameon=False,loc='right')
                
        plt.show()


def plot_shear_rate_to_asses_SS(swap_number_index_end,swap_number_index_start,swap_rate_index_start,swap_rate_index_end,no_timesteps,phi,lengthscale,timestep_points,scaled_temp,number_of_solutions,swap_rate,swap_number,shear_rate_upper,shear_rate_lower,fluid_name,box_size_nd):
    for z in range(0,number_of_solutions): 
        for m in range(swap_rate_index_start,swap_rate_index_end):
            for k in range(swap_number_index_start,swap_number_index_end):
                plt.plot(timestep_points[0,1,1,:],shear_rate_upper[z,m,k,:])
                plt.plot(timestep_points[0,1,1,:],shear_rate_lower[z,m,k,:])
                plt.xlabel('$N_{t}[-]$')
                plt.ylabel('$\dot{\gamma}[\\tau]$',rotation='horizontal')
                plt.title(fluid_name+" simulation run with all $f_{v,x}$ and $N_{v,x}$, $\\bar{T}="+str(scaled_temp)+"$, $\ell="+str(lengthscale)+"$")
                
        plt.show()
        plot_save=input("save figure?, YES/NO")
        if plot_save=='YES':
            plt.savefig(fluid_name+'_T_'+str(scaled_temp)+'_length_scale_'+str(lengthscale)+'_phi_'+str(phi)+'_no_timesteps_'+str(no_timesteps)+'.png')
        else:
            print('Thanks for checking steady state')



def plotting_flux_vs_shear_rate(func4,labelpadx,labelpady,params,fontsize,box_side_length_scaled,number_of_solutions,flux_ready_for_plotting,swap_number_index,shear_rate_mean_of_both_cells):
    
    for z in range(0,number_of_solutions):
        
        
        x=shear_rate_mean_of_both_cells[z,:,:]
        y=flux_ready_for_plotting[z,:,:]
        for i in range(0,swap_number_index):
        
        #for i in range(0,1):
            if z==0:
                j=i
                
                # need to add legend to this 
                plt.scatter(x[:,i],y[i,:],label='$L=${}'.format(np.around(box_side_length_scaled[0,z]),decimals=0))
                plt.plot(x[:,i],func4(x[:,i],params[j][0][0],params[j][0][1]))
                #plt.xscale('log')
                plt.xlabel('log($\dot{\gamma}\ [\\tau]$)', labelpad=labelpadx,fontsize=fontsize)
                #plt.yscale('log')
                plt.ylabel('log($J_{z}(p_{x})$$\ [\\frac{\\tau^{3}}{\mu}]$)',rotation=0,labelpad=labelpady,fontsize=fontsize)
                plt.legend()
            else: 
                j=z*(i+4)
                plt.scatter(x[:,i],y[i,:],label='$L=${}'.format(np.around(box_side_length_scaled[0,z]),decimals=0))
                plt.plot(x[:,i],func4(x[:,i],params[j][0][0],params[j][0][1]))
                #plt.xscale('log')
                plt.xlabel('log($\dot{\gamma}\ [\\tau]$)', labelpad=labelpadx,fontsize=fontsize)
                #plt.yscale('log')
                plt.ylabel('log($J_{z}(p_{x})$$\ [\\frac{\\tau^{3}}{\mu}]$)',rotation=0,labelpad=labelpady,fontsize=fontsize)
                plt.legend()
                 
    plt.show() 
  
                
    


        



