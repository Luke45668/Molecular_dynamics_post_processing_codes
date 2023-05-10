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

def VP_and_momentum_data_realisation_name_grabber(j_,swap_number,swap_rate,VP_general_name_string,Mom_general_name_string,filepath):
    #os.chdir('/Users/lukedebono/Documents/LAMMPS_projects_mac_book/OneDrive_1_24-02-2023/MYRIAD_lammps_runs/'+filepath)
    os.chdir('/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/'+filepath)
    count_VP=0
    realisation_name_VP = []   
    count_mom=0
    realisation_name_Mom=[]  

    for name in glob.glob(VP_general_name_string):
   
        count_VP=count_VP+1    
        realisation_name_VP.append(name)
    

    for name in glob.glob(Mom_general_name_string):
    
        count_mom=count_mom+1    
        realisation_name_Mom.append(name)
    
    if count_VP!=count_mom:
       breakpoint()
    else:
        print('VP and Mom data consistent')
    
    number_of_solutions= int(count_VP/(j_*swap_number.size*swap_rate.size))
        
    return realisation_name_Mom,realisation_name_VP,count_mom,count_VP,number_of_solutions


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

def VP_organiser_and_reader(loc_no_SRD,loc_EF,loc_SN,loc_Realisation_index,box_side_length_scaled,j_,number_of_solutions,swap_number,swap_rate,no_SRD_key,realisation_name_VP,Path_2_VP,chunk,equilibration_timesteps,VP_ave_freq,no_timesteps,VP_output_col_count,count_VP):
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
            VP_data_upper[z,m,k,:,:,j] = VP_data[1:10,:]
            VP_data_lower[z,m,k,:,:,j] = VP_data[11:,:]
            
        except Exception as e:
            print('Velocity Profile Data faulty')
            error_count=error_count+1 
            continue
    VP_z_data = velP2numpy_f(Path_2_VP,chunk,realisation_name,equilibration_timesteps,VP_ave_freq,no_SRD,no_timesteps,VP_output_col_count)[1]     

    VP_z_data_upper = np.array([VP_z_data[1:10].astype('float64')])* box_side_length_scaled.T    
    VP_z_data_lower =np.array([ VP_z_data[11:].astype('float64') ])* box_side_length_scaled.T
    return VP_data_upper,VP_data_lower,error_count,filename,VP_z_data_upper,VP_z_data_lower


def mom_file_data_size_reader(j_,number_of_solutions,count_mom,realisation_name_Mom,no_SRD_key,swap_rate,swap_number,Path_2_mom_file):
    from mom2numpy import mom2numpy_f
    pass_count=0
    size_list=[]
    
    for i in range(0,count_mom):
        filename=realisation_name_Mom[i].split('_')
    
        no_SRD=filename[8]
        z=no_SRD_key.index(no_SRD)
    
        realisation_index=filename[7]
        j=int(float(realisation_index))
        EF=int(filename[20])
        m=np.where(swap_rate==EF)[0][0,]
        
        SN=int(filename[22])
        k=np.where(swap_number==SN)[0][0,]
        
        realisation_name=realisation_name_Mom[i]
        
        
        
        mom_data_test=mom2numpy_f(realisation_name,Path_2_mom_file)  
        size_list.append(mom_data_test.shape)
        
        pass_count=pass_count+1 
    
    size_list.sort()
    size_list.reverse()
    size_list=list(dict.fromkeys(size_list))
    size_array=np.array(size_list)
    
    mom_data=()
    for i in range(0,swap_rate.size):
    
         mom_data= mom_data+(np.zeros((number_of_solutions,swap_number.size,j_,(size_array[i,0]))),)
         
         
         
    return size_array,mom_data,pass_count

def Mom_organiser_and_reader(mom_data,count_mom,realisation_name_Mom,no_SRD_key,swap_rate,swap_number,Path_2_mom_file):
    from mom2numpy import mom2numpy_f
    error_count_mom=0
    failed_list_realisations=[]
    for i in range(0,count_mom):
        filename=realisation_name_Mom[i].split('_')
    
        no_SRD=filename[8]
        z=no_SRD_key.index(no_SRD)
    
        realisation_index=filename[7]
        j=int(float(realisation_index))
        EF=int(filename[20])
        m=np.where(swap_rate==EF)[0][0,]
        
        SN=int(filename[22])
        k=np.where(swap_number==SN)[0][0,]
        
        realisation_name=realisation_name_Mom[i]
        
        try:
           mom_data[m][z,k,j,:]=mom2numpy_f(realisation_name,Path_2_mom_file)  
        
        

        
            
        except Exception as e:
            print('Mom Data faulty')
            error_count_mom=error_count_mom+1 
            failed_list_realisations.append(realisation_name)
            break
    return mom_data,error_count_mom,failed_list_realisations

# this bit needs turning to multi solution version
def mom_data_averaging_and_flux_calc(number_of_solutions,swap_number,truncation_timestep,swap_rate,scaled_timestep,no_timesteps,box_side_length_scaled,mom_data):
    mom_data_realisation_averaged=()
    number_swaps_before_truncation=(np.ceil(truncation_timestep/swap_rate)).astype(int)
    mom_data_realisation_averaged_truncated=()
    flux_x_momentum_z_direction=np.zeros((number_of_solutions,swap_number.size,swap_rate.size))
    total_run_time=scaled_timestep* no_timesteps
    box_area_nd=np.array(box_size_key)
    flux_ready_for_plotting=np.zeros((number_of_solutions,swap_number.size,swap_rate.size))
    for z in range(0,number_of_solutions):    
        for i in range(0,swap_rate.size):
            mom_data_realisation_averaged=mom_data_realisation_averaged+(np.mean(mom_data[i],axis=2),)


        #for i in range(0,swap_rate.size):

            mom_data_realisation_averaged_truncated=mom_data_realisation_averaged_truncated+(mom_data_realisation_averaged[i][:,:,number_swaps_before_truncation[i]:],)


        # now apply the MP formula 
            mom_difference= mom_data_realisation_averaged_truncated[i][z,:,-1]-mom_data_realisation_averaged_truncated[i][z,:,0]
            flux_x_momentum_z_direction[z,:,i]=(mom_difference)/(2*total_run_time*float(box_area_nd[z]))
            
    flux_ready_for_plotting=np.abs(flux_x_momentum_z_direction)
    return flux_x_momentum_z_direction,flux_ready_for_plotting




def VP_data_averaging_and_stat_test_data(VP_z_data_upper,VP_z_data_lower,no_timesteps,VP_data_lower,VP_data_upper,number_of_solutions,swap_rate,swap_number,VP_ave_freq):
    VP_data_lower_realisation_averaged = np.mean(VP_data_lower,axis=5) 
    VP_data_upper_realisation_averaged = np.mean(VP_data_upper,axis=5) 
    x_u= np.array(VP_data_upper_realisation_averaged)
    x_l= np.array(VP_data_lower_realisation_averaged)
    y_u=np.zeros((number_of_solutions,VP_z_data_upper.shape[1],VP_data_upper.shape[4]))
    y_l=np.zeros((number_of_solutions,VP_z_data_upper.shape[1],VP_data_upper.shape[4]))

    for z in range(number_of_solutions):
        y_u[z,:,:]= np.reshape(np.repeat(VP_z_data_upper[z,:],VP_data_upper.shape[4],axis=0),(VP_z_data_upper.shape[1],VP_data_upper.shape[4]))

        y_l[z,:,:]= np.reshape(np.repeat(VP_z_data_lower[z,:],VP_data_lower.shape[4],axis=0),(VP_z_data_lower.shape[1],VP_data_lower.shape[4]))

    pearson_coeff_upper= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_upper_realisation_averaged.shape[4]))
    pearson_coeff_lower= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_lower_realisation_averaged.shape[4]))
    shear_rate_upper= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_upper_realisation_averaged.shape[4]))    
    shear_rate_lower= np.zeros((number_of_solutions,swap_rate.size,swap_number.size,VP_data_lower_realisation_averaged.shape[4]))
    timestep_points=np.array([[[np.linspace(1,VP_data_upper.shape[4],int(float(no_timesteps)/float(VP_ave_freq)))]]])*VP_ave_freq
    timestep_points=np.repeat(timestep_points, number_of_solutions,axis=0)
    timestep_points=np.repeat(timestep_points, swap_rate.size,axis=1)     
    timestep_points=np.repeat(timestep_points, swap_number.size,axis=2)   



    for z in range(0,number_of_solutions): 
        for m in range(0,swap_rate.size):
            for k in range(0,swap_number.size):
                for i in range(0,VP_data_upper.shape[4]):
                        
                        pearson_coeff_upper[z,m,k,i] =scipy.stats.pearsonr(y_u[z,:,i],x_u[z,m,k,:,i] )[0]
                        shear_rate_upper[z,m,k,i]= scipy.stats.linregress(y_u[z,:,i],x_u[z,m,k,:,i] ).slope
                        pearson_coeff_lower[z,m,k,i] =scipy.stats.pearsonr(y_l[z,:,i],x_l[z,m,k,:,i] )[0]
                        shear_rate_lower[z,m,k,i]= scipy.stats.linregress(y_l[z,:,i],x_l[z,m,k,:,i] ).slope  
     
    return pearson_coeff_upper,shear_rate_upper,pearson_coeff_lower,shear_rate_lower,timestep_points,VP_data_lower_realisation_averaged,VP_data_upper_realisation_averaged
        

    
def truncation_step_and_SS_average_of_VP_and_stat_tests(timestep_points,pearson_coeff_lower,pearson_coeff_upper,shear_rate_upper,shear_rate_lower,VP_ave_freq,truncation_timestep,VP_data_lower_realisation_averaged,VP_data_upper_realisation_averaged):
    truncation_index=int(truncation_timestep/VP_ave_freq)
    shear_rate_upper=shear_rate_upper[:,:,:,truncation_index:]
    shear_rate_lower=shear_rate_lower[:,:,:,truncation_index:]
    pearson_coeff_upper=pearson_coeff_upper[:,:,:,truncation_index:]
    pearson_coeff_lower=pearson_coeff_lower[:,:,:,truncation_index:]
    timestep_points=timestep_points[:,:,:,truncation_index:]
    VP_steady_state_data_lower_truncated=VP_data_lower_realisation_averaged[:,:,:,:,truncation_index:]
    VP_steady_state_data_upper_truncated=VP_data_upper_realisation_averaged[:,:,:,:,truncation_index:]

    VP_steady_state_data_lower_truncated_time_averaged=np.mean(VP_steady_state_data_lower_truncated,axis=4)
    VP_steady_state_data_upper_truncated_time_averaged=np.mean(VP_steady_state_data_upper_truncated,axis=4)
    shear_rate_upper_steady_state_mean = np.mean(shear_rate_upper, axis=3)
    shear_rate_lower_steady_state_mean = np.mean(shear_rate_lower, axis=3)
    pearson_coeff_lower_mean_SS=np.mean(pearson_coeff_lower,axis=3)
    pearson_coeff_upper_mean_SS=np.mean(pearson_coeff_upper,axis=3) 
    
    standard_deviation_upper_SS=np.std(pearson_coeff_upper,axis=3)
    standard_deviation_lower_SS=np.std(pearson_coeff_lower,axis=3)
    standard_deviation_upper_error=standard_deviation_upper_SS/ pearson_coeff_lower_mean_SS
    standard_deviation_lower_error=standard_deviation_lower_SS/pearson_coeff_lower_mean_SS

    return standard_deviation_upper_error,standard_deviation_lower_error,pearson_coeff_upper_mean_SS,pearson_coeff_lower_mean_SS,shear_rate_lower_steady_state_mean,shear_rate_upper_steady_state_mean,VP_steady_state_data_lower_truncated_time_averaged,VP_steady_state_data_upper_truncated_time_averaged
    
    

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
    
    
def plotting_SS_velocity_profiles_for_4_swap_numbers(labelpad,fontsize,number_of_solutions,swap_number_choice_index,width_plot,height_plot,swap_number,swap_rate,VP_ave_freq,no_timesteps,VP_steady_state_data_lower_truncated_time_averaged,VP_steady_state_data_upper_truncated_time_averaged,VP_z_data_lower,VP_z_data_upper):
    for z in range(0,number_of_solutions):
        fig=plt.figure(figsize=(width_plot,height_plot))
        gs=GridSpec(nrows=2,ncols=1)

        ax1= fig.add_subplot(gs[0,0])
        ax2= fig.add_subplot(gs[1,0])
        k=swap_number_choice_index



        x_1=VP_steady_state_data_lower_truncated_time_averaged[z,:,:]
        x_2=VP_steady_state_data_upper_truncated_time_averaged[z,:,:]
        y_1=VP_z_data_lower[z,:]
        y_2=VP_z_data_upper[z,:]

        no_vps=int(no_timesteps/VP_ave_freq)
        #for k in range(0,swap_number.size):
        for i in range(0,swap_rate.size):

            ax1.plot(x_1[i,k,:],y_1[:],label='$f_p=${}'.format(swap_rate[i]))
           # ax1.set_xlabel('$v_{x}\ [\\frac{\\tau}{\ell}]$',labelpad=labelpad, fontsize=fontsize)
            ax1.set_ylabel('$L_{z}\ [\ell^{-1}]$',rotation=0,labelpad=labelpad,fontsize=fontsize)
            ax2.plot(x_2[i,k,:],y_2[:],label='$f_p=${}'.format(swap_rate[i]))
            ax1.legend(frameon=False,loc='right')
            ax2.set_xlabel('$v_{x}\ [\\frac{\\tau}{\ell}]$',fontsize=fontsize)
            ax2.set_ylabel('$L_{z}\ [\ell^{-1}]$',rotation=0,labelpad=labelpad,fontsize=fontsize)
            ax2.legend(frameon=False,loc='right')
            
        plt.show()

def plot_shear_rate_to_asses_SS(no_timesteps,phi,lengthscale,timestep_points,scaled_temp,number_of_solutions,swap_rate,swap_number,shear_rate_upper,shear_rate_lower,fluid_name,box_size_nd):
    for z in range(0,number_of_solutions): 
        for m in range(0,swap_rate.size):
            for k in range(0,swap_number.size):
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



def plotting_flux_vs_shear_rate(box_side_length_scaled,number_of_solutions,shear_rate_lower_steady_state_mean,shear_rate_upper_steady_state_mean,flux_ready_for_plotting,swap_number):
    # for z in range(0,number_of_solutions):
    #     shear_rate_mean_of_both_cells=((np.abs(shear_rate_lower_steady_state_mean)+np.abs(shear_rate_upper_steady_state_mean))*0.5)

    #     x=shear_rate_mean_of_both_cells
    #     y=flux_ready_for_plotting
    #     for i in range(0,swap_number.size):
                
    #             # need to add legend to this 
    #             plt.plot(x[0,:,i],y[i,:],label='N={}'.format(swap_number[i]))
    #             plt.xscale('log')
    #             plt.xlabel('$\dot{\gamma}\ [\\tau]$')
    #             plt.yscale('log')
    #             plt.ylabel('$J_{z}(p_{x})\ [\\frac{\\tau^{3}}{\mu}]$',rotation=0,labelpad=25)
    #             plt.legend()
                
    #     plt.show() 
    for z in range(0,number_of_solutions):
        shear_rate_mean_of_both_cells=((np.abs(shear_rate_lower_steady_state_mean)+np.abs(shear_rate_upper_steady_state_mean))*0.5)

        x=shear_rate_mean_of_both_cells[z,:,:]
        y=flux_ready_for_plotting[z,:,:]
        #for i in range(0,swap_number.size):
        
        for i in range(0,1):
                
                # need to add legend to this 
                plt.plot(x[:,i],y[i,:],label='$L=${}'.format(np.around(box_side_length_scaled[0,z]),decimals=0))
                plt.xscale('log')
                plt.xlabel('$\dot{\gamma}\ [\\tau]$')
                plt.yscale('log')
                plt.ylabel('$J_{z}(p_{x})\ [\\frac{\\tau^{3}}{\mu}]$',rotation=0,labelpad=25)
                plt.legend()
plt.show() 


        



