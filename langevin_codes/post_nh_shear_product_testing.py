##!/usr/bin/env python3.11
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
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import regex as re
import pandas as pd
import sigfig
plt.rcParams.update(plt.rcParamsDefault)
#plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats

from datetime import datetime
import mmap
import h5py as h5
from scipy.optimize import curve_fit


# path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
# os.chdir(path_2_post_proc_module)
import seaborn as sns
# from log2numpy import *
# from dump2numpy import *
import glob 
#from MPCD_codes.post_MPCD_MP_processing_module import *
import pickle as pck
from post_langevin_module import *

linestyle_tuple = ['-', 
  'dotted', 
 'dashed', 'dashdot', 
  'solid', 
 'dashed', 'dashdot', '--']

linestyle_tuple = [
    
     ('dotted',                (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dotted',        (0, (1, 1))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

#%% 

#note: currently we have one missing shear rate for k=30,60 so need to run them again with erate 0.02 to get the full picture 
# I have k=20 downloaded, K=120 is running , need to produce run files for k=30,60

damp=np.array([ 0.035, 0.035,0.035,0.035 ])
K=np.array([  30,  60  ,90
            ])
K=np.array([ 120
            ])
K=np.array([ 60
            ,120])


no_timesteps=np.array([1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000, 1999999000,
        1999999000, 1999999000, 1999999000, 1999999000 ])

timestep_multiplier=np.flip(np.array([9.85877894e-06, 1.03045633e-05, 1.07925709e-05, 1.13290990e-05,
        1.19217621e-05, 1.25798566e-05, 1.33148512e-05, 1.41410611e-05,
        1.50765901e-05, 1.61446718e-05, 1.73756257e-05, 1.88097815e-05,
        2.05019815e-05, 2.25287546e-05, 2.50002112e-05, 2.80807326e-05,
        3.20271067e-05, 3.72640781e-05, 4.45485099e-05, 5.53728731e-05,
        7.31457792e-05, 1.07720625e-04, 2.04281005e-04, 1.97175579e-03]))

thermo_vars="         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    "

erate=np.array([0      , 0.00388889, 0.00777778, 0.01166667, 0.01555556,
       0.01944444, 0.02333333, 0.02722222, 0.03111111, 0.035  ,0.07      , 0.13894737, 0.20789474, 0.27684211, 0.34578947,
        0.41473684, 0.48368421, 0.55263158, 0.62157895, 0.69052632,
        0.75947368, 0.82842105, 0.89736842, 0.96631579, 1.03526316,
        1.10421053, 1.17315789,1.24, 1.24210526,1.25444444, 1.26888889, 1.28333333, 1.29777778,  1.31105263,1.31222222,1.32666667, 1.34111111, 1.35555556, 1.37, 1.38,1.4       , 1.42222222, 1.44444444, 1.46666667, 1.48888889,
        1.51111111, 1.53333333, 1.55555556, 1.57777778, 1.6 ])


e_in=0
#e_end=erate.size
n_plates=100

strain_total=250


path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/shear_runs/strain_250_10_reals_erate_up_to_1.34/"



j_=10
sim_fluid=30.315227255599112

eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
damp_ratio=mass_pol/damp

def one_term_poly(x,a,b):
     return b*(x**a)

def quadratic_no_constant(x,a,b):
     return a*x + b*(x**2)
     
def quadratic(x,a):
     return a*(x**2)
#%% load in tuples
label='damp_'+str(damp)+'_K_'+str(K)+'_'


os.chdir(path_2_log_files)
#os.mkdir("tuple_results")
#os.chdir("tuple_results")

def batch_load_tuples(label,tuple_name):

    with open(label+tuple_name, 'rb') as f:
         load_in= pck.load(f)

    return load_in

erate_velocity_batch_tuple=()
spring_force_positon_tensor_batch_tuple=()
COM_velocity_batch_tuple=()
conform_tensor_batch_tuple=()
log_file_batch_tuple=()
area_vector_spherical_batch_tuple=()
interest_vectors_batch_tuple=()
pos_batch_tuple=()
vel_batch_tuple=()
e_end=[]

# loading all data into one 
for i in range(K.size):

    label='damp_'+str(damp[i])+'_K_'+str(K[i])+'_'

   
    spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(batch_load_tuples(label,
                                                            "spring_force_positon_tensor_tuple.pickle"),)
    
    print(len( spring_force_positon_tensor_batch_tuple[i]))
    e_end.append(len(spring_force_positon_tensor_batch_tuple[i]))

    pos_batch_tuple=pos_batch_tuple+(batch_load_tuples(label,"p_positions_tuple.pickle"),)

    vel_batch_tuple=vel_batch_tuple+(batch_load_tuples(label,"p_velocities_tuple.pickle"),)


    log_file_batch_tuple=log_file_batch_tuple+(batch_load_tuples(label,
                                                            "log_file_tuple.pickle"),)
    # print(len(log_file_batch_tuple[i]))
    area_vector_spherical_batch_tuple=area_vector_spherical_batch_tuple+(batch_load_tuples(label,"area_vector_tuple.pickle"),)
    
   
   # e_end.append(len(spring_force_positon_tensor_batch_tuple[i]))


                                                                                      

    


     

#%% strain points for temperatuee data 
strainplot_tuple=()

for i in range(erate.size):
    
    strain_plotting_points= np.linspace(0,strain_total,1002)
    #strain_plotting_points= np.linspace(150,strain_total,501)

    strainplot_tuple=strainplot_tuple+(strain_plotting_points,)  
    print(strainplot_tuple[i].size)

# def strain_plotting_points(total_strain,points_per_iv):
#      #points_per_iv= number of points for the variable measured against strain 
#      strain_unit=total_strain/points_per_iv
#      strain_plotting_points=np.arange(0,total_strain,strain_unit)
#      return  strain_plotting_points
#%%
from fitter import Fitter
#NOTE: have I included the phantom particles in the velocity distributions ?
folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=5
final_temp=np.zeros((erate.size))
mean_temp_tuple=()
plt.rcParams["figure.figsize"] = (24,12 )
plt.rcParams.update({'font.size': 16})
#NOTE need to add an econserve plot as this should be constant 

for j in range(K.size):

    mean_temp_array=np.zeros((erate.size))

    skip_array=np.array([7,8,9,10])
   
    for i in range(erate.size):
    # for i in range(skip_array.size):
    #         i=skip_array[i]
        
           

        
        #for i in range(erate[:e_end[j]].size):
        #i=15
            plt.subplot(2, 3, 1)
            column=5
            signal_std=sigfig.round(np.std(log_file_batch_tuple[j][i][100:,column]), sigfigs=3)
            signal_mean=sigfig.round(np.mean(log_file_batch_tuple[j][i][100:,column]), sigfigs=5)
            plt.plot(strainplot_tuple[i][10:],log_file_batch_tuple[j][i][10:,column],
            label="K="+str(K[j])+",$\\bar{T}="+str(signal_mean)+",\sigma_{T}="+str(signal_std)+"$")
            plt.ylabel("$T$", rotation=0)
            plt.title("$"+str(erate[i])+"$")
            #plt.legend(loc='upper right', bbox_to_anchor=(4.25,1))
            plt.legend()
            mean_temp_array[i]=np.mean(log_file_batch_tuple[j][i][500:,column])


            plt.subplot(2, 3, 2)
            column=2
            grad_pe=np.gradient(log_file_batch_tuple[j][i][:,column])
            grad_mean=np.mean(grad_pe[500:])
            plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column],
            label="K="+str(K[j])+",$\\bar{grad}="+str(grad_mean)+"$")
            #plt.plot(strainplot_tuple[i][:],grad_pe)
            plt.ylabel("$E_{p}$")
            #plt.yscale('log')
            plt.title("$\dot{\gamma}="+str(erate[i])+"$")
            #plt.legend(loc='upper right', bbox_to_anchor=(4.25,0.75))
            plt.legend()
            # final_temp[i]=log_file_batch_tuple[j][i][-1,column]

            
        #for i in range(erate[:e_end[j]].size):
            plt.subplot(2,3 , 3)
            column=6
            plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column],
            label="K="+str(K[j])+",$\dot{\gamma}="+str(erate[i])+"$")
            plt.ylabel("$E_{k}$")
            plt.title("$\dot{\gamma}="+str(erate[i])+"$")

            # assume first 3 particles are stokes beads

           
            vel_data=vel_batch_tuple[j][i]
          

            plt.subplot(2,3,4)
           
            x_vel=np.ravel(vel_data[:,:,:,:,0])
            # f = Fitter(x_vel)
            
            # f.distributions =  ['gennorm']
            # f.fit()
            # # # may take some time since by default, all distributions are tried
            # # # but you call manually provide a smaller set of distributions
            # f.summary()
            sns.kdeplot(x_vel, bw_adjust=1)

            plt.xlabel("$v_{x}$")
            #plt.legend()
            plt.subplot(2,3,5)
           
            y_vel=np.ravel(vel_data[:,:,:,:,1])
            # f = Fitter(y_vel)
            
            # f.distributions =  ['gennorm']
            # f.fit()
            # # # may take some time since by default, all distributions are tried
            # # # but you call manually provide a smaller set of distributions
            # f.summary()
            sns.kdeplot(y_vel, bw_adjust=1)
            plt.xlabel("$v_{y}$")
            #plt.legend()
            plt.subplot(2,3,6)
            z_vel=np.ravel(vel_data[:,:,:,:,2])
            # f = Fitter(z_vel)
            
            # f.distributions =  ['gennorm']
            # f.fit()
            # # # may take some time since by default, all distributions are tried
            # # # but you call manually provide a smaller set of distributions
            # f.summary()
            test=scipy.stats.kstest(z_vel, 'norm')
            sns.kdeplot(z_vel, bw_adjust=1,label="$D="+str(test[0])+",p="+str(test[1])+"$")
            #plt.legend()
           

            plt.xlabel("$v_{z}$")
            
            


        
            #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
           
                #     plt.xlabel("$\gamma$")
                

            
                # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
            #plt.ylim(0,2)
            #plt.yscale('log')
            
            plt.show()

            speed=np.sqrt(x_vel**2 +y_vel**2 +z_vel**2)
            maxwell = scipy.stats.maxwell
            params = maxwell.fit(speed)
            print(params)
            # (0, 4.9808603062591041)

            plt.hist(speed, bins=20,density=True)
            x = np.linspace(np.min(speed),np.max(speed),speed.size)
            plt.plot(x, maxwell.pdf(x, *params), lw=3)
            plt.xlabel("$|v|$")
            plt.ylabel("Density")
           
            plt.show()

            # energy=0.5*5*(speed**2)
            # maxwell = scipy.stats.maxwell
            # params = maxwell.fit(energy)
            # print(params)
            # # (0, 4.9808603062591041)

            # plt.hist(speed, bins=20,density=True)
            # x = np.linspace(np.min(energy),np.max(energy),energy.size)
            # plt.plot(x, maxwell.pdf(x, *params), lw=3)
            # plt.xlabel("$E_{k}$")
            # plt.ylabel("Density")
           
            # plt.show()


    mean_temp_tuple=mean_temp_tuple+(mean_temp_array,)
#%% energy vs time plot
e_end=[34,34]
plt.rcParams["figure.figsize"] = (26,10 )
# pe vs time 

for j in range(K.size):
    for i in range(e_end[j]):
        column=2
        plt.subplot(1,3,1)
        #plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
       #strainplot= np.linspace(0,strain_total,log_file_batch_tuple[j][i].shape[0])
        strainplot= np.linspace(0,strain_total,1002)
        plt.plot(strainplot,log_file_batch_tuple[j][i][:,column])
        #plt.yscale('log')
        plt.xlabel("$\gamma$")
        plt.ylabel("$E_{p}$")
        plt.title("$K="+str(K[j])+"$")
        #plt.show()
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        column=5

        plt.subplot(1,3,2)
        plt.plot(strainplot[50:],log_file_batch_tuple[j][i][50:,column])
       # plt.yscale('log')
        plt.xlabel("$\gamma$")
        plt.ylabel("$T$")
        plt.title("$K="+str(K[j])+"$")
 
        column=6


        plt.subplot(1,3,3)
        plt.plot(strainplot[50:],log_file_batch_tuple[j][i][50:,column])
       # plt.yscale('log')
        plt.xlabel("$\gamma$")
        plt.ylabel("$E_{t}$")
        plt.title("$K="+str(K[j])+"$")
    plt.show()
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
#%%#%% inspecting energies realisation by realisation 
plt.rcParams["figure.figsize"] = (26,10 )
j=0

for i in range(e_end[j]):
    for j in range(K.size):
        for k in range(j_):
    
            column=2
            plt.subplot(1,3,1)
            #plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
        #strainplot= np.linspace(0,strain_total,log_file_batch_tuple[j][i].shape[0])
            strainplot= np.linspace(0,strain_total,1002)
            plt.plot(strainplot,log_file_real_batch_tuple[j][i][k,:,column])
            #plt.yscale('log')
            plt.xlabel("$\gamma$")
            plt.ylabel("$E_{p}$")
            plt.ylim(0,3)
            plt.title("$N_{c}=$"+str(tchain[j]))
            #plt.show()
            # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
            column=6

            plt.subplot(1,3,2)
            plt.plot(strainplot[50:],log_file_real_batch_tuple[j][i][k,50:,column])
        # plt.yscale('log')
            plt.xlabel("$\gamma$")
            plt.ylabel("$T$")
            plt.title("$N_{c}=$"+str(tchain[j]))
    
            column=7


            plt.subplot(1,3,3)
            plt.plot(strainplot[50:],log_file_real_batch_tuple[j][i][k,50:,column])
        # plt.yscale('log')
            plt.xlabel("$\gamma$")
            plt.ylabel("$E_{t}$")
            plt.title("$N_{c}=$"+str(tchain[j]))
        plt.suptitle("$\dot{\gamma}="+str(erate[i])+"$")
        plt.show()
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]       
        
#%%
folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=5
final_temp=np.zeros((erate.size))
mean_temp_array=np.zeros((K.size,erate.size))
for j in range(K.size):
    for i in range(e_end[j]):
        
        # plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
        mean_temp_array[j,i]=np.mean(log_file_batch_tuple[j][i][1000:,column])
      
        #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
    #     plt.ylabel("$T$", rotation=0)
    #     plt.xlabel("$\gamma$")
    

    # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
    #     plt.show()

#

marker=['x','+','^',"1","X","d","*","P","v"]

for j in range(K.size):
    plt.scatter(erate,mean_temp_array[j,:],label="$K="+str(K[j])+"$" ,marker=marker[j])
    plt.ylabel("$T$", rotation=0)
    plt.xlabel('$\dot{\gamma}$')
    plt.xscale('log')
   # plt.yscale('log')
plt.axhline(1,label="$T_{0}=1$")
plt.legend()
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/temp_vs_erate.pdf",dpi=1200,bbox_inches='tight')


plt.show()

#%



#%%
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
                # print(spring_force_positon_tensor_tuple[i][j,:,:,l].shape)
                # print(cutoff)
                # print(aftercutoff)
                data=np.ravel(spring_force_positon_tensor_tuple[i][j,cutoff:aftercutoff,:,l])
              
                stress_tensor_reals[i,j,l]=np.mean(data)
                stress_tensor_std_reals[i,j,l]=np.std(data)
    stress_tensor=np.mean(stress_tensor_reals, axis=1)
    stress_tensor_std=np.mean(stress_tensor_std_reals, axis=1)
    return stress_tensor,stress_tensor_std



e_end=[37,37] # only for low shear rate regime 
aftcut=1
cut=0.6# or 0.4 
# aftcut=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.25,0.25,0.2,0.2,0.175,0.15,0.15,0.1,0.1,0.1]
# cut=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.1,0.1,0.075,0.075,0.075,0.075,0.075,0.05,0.05,0.05]

labels_stress=np.array(["\sigma_{xx}$",
               "\sigma_{yy}$",
               "\sigma_{zz}$",
               "\sigma_{xz}$",
               "\sigma_{xy}$",
               "\sigma_{yz}$"])


#compute stress tensor 
##y_ticks_stress=[-10,0,20,40,60,80] # for plates 
#y_ticks_stress=[0.95,1,1.05,1.1,1.15,1.2,1.25,1.3]


stress_tensor_tuple=()
stress_tensor_std_tuple=()

for j in range(K.size):
    stress_tensor=np.zeros((e_end[j],6))
    stress_tensor_std=np.zeros((e_end[j],6))   
    stress_tensor,stress_tensor_std=stress_tensor_averaging(e_end[j],labels_stress,
                            cut,
                            aftcut,
                           spring_force_positon_tensor_batch_tuple[j],j_)
    
    stress_tensor_tuple=stress_tensor_tuple+(stress_tensor,)
    stress_tensor_std_tuple=stress_tensor_std_tuple+(stress_tensor_std,)



    
    
#%% time series plots of stress
for j in range(K.size):   
    for i in range(e_end[j]):
    
    
        mean_stress_tensor=np.mean(spring_force_positon_tensor_batch_tuple[j][i],axis=0)
        mean_stress_tensor=np.mean(mean_stress_tensor,axis=1)
        std_dev_stress_tensor=np.std(spring_force_positon_tensor_batch_tuple[j][i],axis=0)
        
 
        for l in range(6):
                strain_plot=np.linspace(0,strain_total,mean_stress_tensor[:,l].size)
                grad_cutoff=int(np.round(cut*mean_stress_tensor[:,l].size))
                SS_grad=np.mean(np.gradient(mean_stress_tensor[grad_cutoff:,l],axis=0))
                SS_grad=np.around(SS_grad,5)

                plt.plot(strain_plot,mean_stress_tensor[:,l],label="$SSgrad="+str(SS_grad)+","+labels_stress[l])

        plt.legend(bbox_to_anchor=(1,1))
        plt.title("$\dot{\\gamma}="+str(erate[i])+"K="+str(K[j])+"$")
            #plt.yscale('log')
        plt.ylim(-5,40)
        plt.savefig(path_2_log_files+"/"+str(K[j])+"_stress_time_series_"+str(erate[i])+".pdf",dpi=1200,format="pdf",bbox_inches='tight') 
            
        plt.show()

# for j in range(K.size):
#     for i in range(e_end[j]):
#          mean_stress_tensor=np.mean(spring_force_positon_tensor_batch_tuple[j][i],axis=0)
#          mean_stress_tensor=np.mean(mean_stress_tensor,axis=1)
         
#          for l in range(3,6):
#              strain_plot=np.linspace(0,strain_total,mean_stress_tensor[:,l].size)
#              plt.plot(strain_plot,mean_stress_tensor[:,l],label="$\dot{\gamma}="+str(erate[i])+"$")
#     plt.legend(bbox_to_anchor=(1,1))
#     #plt.yscale('log')
    
#     plt.show()

#%%
sns.set_palette('colorblind')
# sns.color_palette("mako", as_cmap=True)
# sns.color_palette("viridis")
#sns.set_palette('virdris')
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 14})
SIZE_DEFAULT = 14
SIZE_LARGE = 16
#plt.rcParams['text.usetex'] = True
# plt.rc("font", family="Roboto")  # controls default font
# plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
legfont=10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False 

plt.rcParams.update({'font.size': 16})
# for j in range(thermal_damp_multiplier.size): 

linestyle_tuple = [
    
     ('dotted',                (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dotted',        (0, (1, 1))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]



for j in range(K.size): 
    for l in range(3):
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[0])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]), marker=marker[j])
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[j])+","+str(labels_stress[l]),linestyle=linestyle_tuple[j][1], marker=marker[j])
        plt.errorbar(erate[:e_end[j]],stress_tensor_tuple[j][:,l],
                     yerr=stress_tensor_std_tuple[j][:,l]/np.sqrt(n_plates*j_)
                     ,label="$K="+str(K[j])+","+str(labels_stress[l]),linestyle=linestyle_tuple[j][1],
                       marker=marker[j])
        

    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
       
        plt.xlabel("$\dot{\\gamma}$")
        plt.ylabel("$\sigma_{\\alpha \\alpha}$",rotation=0,labelpad=15)
        #plt.yticks(y_ticks_stress)
        #plt.ylim(0.9,1.3)
       
plt.tight_layout()
        #plt.xscale('log')

plt.legend(frameon=False)
plt.savefig(path_2_log_files+"/K_"+str(K[j])+"stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.show()

#%%
for j in range(K.size): 
    for l in range(3,6):
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[0])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        #plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]), marker=marker[j])
       # plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[j])+","+str(labels_stress[l]), marker=marker[j])
        plt.errorbar(erate[:e_end[j]-1],stress_tensor_tuple[j][:-1,l],yerr=stress_tensor_std_tuple[j][:-1,l]/np.sqrt(n_plates*j_),label="$K="+str(K[j])+","+str(labels_stress[l]), marker=marker[j])
        
        

    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
       
       
        plt.xlabel("$\dot{\gamma}$")
        plt.ylabel("$\sigma_{\\alpha \\alpha}$",rotation=0,labelpad=10)
        #plt.yticks(y_ticks_stress)
        #plt.ylim(0.9,1.3)
        #plt.tight_layout()
        #plt.xscale('log')
        #plt.savefig(path_2_log_files+"/plots/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.tight_layout()
        #plt.xscale('log')

plt.legend(frameon=False)
plt.savefig(path_2_log_files+"/stress_tensor_3_6_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.show()

#%% low shear rates n1
# now plot n1 vs erate with y=ax^2
#probably need to turn this into a a function 
n_y_ticks=[-10,0,20,40,60,80]
cutoff=0
quadratic_end=8
cutoff_N1=8
#plt.plot(0,0,marker='none',label="fit: $y=ax^{2}$",linestyle='none')
for j in range(K.size):
    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
  
    n_1,n_1_error=compute_n_stress_diff(stress_tensor_tuple[j], 
                          stress_tensor_std_tuple[j],
                          0,2,
                          j_,n_plates,
                          )
    plt.errorbar(erate[cutoff:cutoff_N1], n_1[cutoff:cutoff_N1], yerr =n_1_error[cutoff:cutoff_N1],
                  ls="none",label="$N_{1},K="+str(K[j])+"$",marker=marker[j] )

    popt,cov_matrix_n1=curve_fit(quadratic,erate[cutoff:quadratic_end], n_1[cutoff:quadratic_end])
    difference=np.sqrt(np.sum((n_1[cutoff:quadratic_end]-(popt[0]*(erate[cutoff:quadratic_end])**2))**2)/(quadratic_end))

    plt.plot(erate[cutoff:quadratic_end],popt[0]*(erate[cutoff:quadratic_end])**2,ls=linestyle_tuple[j][1],#)#,
            label="$N_{1,fit,K="+str(K[j])+"},a="+str(sigfig.round(popt[0],sigfigs=2))+\
                ",\\varepsilon="+str(sigfig.round(difference,sigfigs=2))+"$")

    #plt.xscale('log')
    #plt.show()
    #print(difference)


plt.legend(fontsize=legfont, frameon=False)
#plt.xscale('log')
plt.xlabel("$\dot{\gamma}$")
plt.ylabel("$N_{1}$",rotation=0)
#plt.yticks(n_y_ticks)
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/N1_vs_gdot_ybxa_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()
#print(difference)

#%% all shear rates n1
# now plot n1 vs erate with y=ax^2
#probably need to turn this into a a function 
n_y_ticks=[-10,0,20,40,60,80]
cutoff=0
quadratic_end=8
#plt.plot(0,0,marker='none',label="fit: $y=ax^{2}$",linestyle='none')
for j in range(K.size):
    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
  
    n_1,n_1_error=compute_n_stress_diff(stress_tensor_tuple[j], 
                          stress_tensor_std_tuple[j],
                          0,2,
                          j_,n_plates,
                          )
    plt.errorbar(erate[cutoff:e_end[j]], n_1[cutoff:e_end[j]], yerr =n_1_error[cutoff:e_end[j]],
                  ls="none",label="$N_{1},K="+str(K[j])+"$",marker=marker[j] )

    popt,cov_matrix_n1=curve_fit(quadratic,erate[cutoff:quadratic_end], n_1[cutoff:quadratic_end])
    difference=np.sqrt(np.sum((n_1[cutoff:quadratic_end]-(popt[0]*(erate[cutoff:quadratic_end])**2))**2)/(quadratic_end))

    plt.plot(erate[cutoff:quadratic_end],popt[0]*(erate[cutoff:quadratic_end])**2,ls=linestyle_tuple[j][1],#)#,
            label="$N_{1,fit,K="+str(K[j])+"},a="+str(sigfig.round(popt[0],sigfigs=2))+\
                ",\\varepsilon="+str(sigfig.round(difference,sigfigs=2))+"$")

    #plt.xscale('log')
    #plt.show()
    #print(difference)


plt.legend(fontsize=legfont, frameon=False)
#plt.xscale('log')
plt.xlabel("$\dot{\gamma}$")
plt.ylabel("$N_{1}$",rotation=0)
#plt.yticks(n_y_ticks)
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/N1_vs_gdot_ybxa_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()
#print(difference)


#%% low shear rates n2
# now plot n2 vs erate with y=ax^2
#probably need to turn this into a a function 
n_y_ticks=[-10,0,20,40,60,80]
cutoff=0
quadratic_end=8
cutoff_N_2=[8,8] # low shear rates
#cutoff_N_2=[21,21] # high shear rates

plt.plot(0,0,marker='none',label="fit: $y=ax^{2}$",linestyle='none')
for j in range(K.size):
    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
    
    n_2,n_2_error=compute_n_stress_diff(stress_tensor_tuple[j], 
                          stress_tensor_std_tuple[j],
                          2,1,
                          j_,n_plates,
                          )
    plt.errorbar(erate[cutoff:cutoff_N_2[j]], n_2[cutoff:cutoff_N_2[j]], yerr =n_2_error[cutoff:cutoff_N_2[j]],
                  ls="none",label="$N_{2},K="+str(K[j])+"$",marker=marker[j] )

    popt,cov_matrix_n2=curve_fit(quadratic,erate[cutoff:quadratic_end], n_2[cutoff:quadratic_end])
    difference=np.sqrt(np.sum((n_2[cutoff:quadratic_end]-(popt[0]*(erate[cutoff:quadratic_end])**2))**2)/(quadratic_end))

    plt.plot(erate[cutoff:quadratic_end],popt[0]*(erate[cutoff:quadratic_end])**2,ls=linestyle_tuple[j][1],#)#,
            label="$N_{2,fit,K="+str(K[j])+"},a="+str(sigfig.round(popt[0],sigfigs=2))+\
                ",\\varepsilon="+str(sigfig.round(difference,sigfigs=2))+"$")

    #plt.xscale('log')
    #plt.show()
    #print(difference)


plt.legend(fontsize=9, frameon=False, loc="best")
#plt.xscale('log')
plt.xlabel("$\dot{\gamma}$")
plt.ylabel("$N_{2}$",rotation=0)
#plt.yticks(n_y_ticks)
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/N1_vs_gdot_ybxa_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()
#print(difference)

#%% high shear rates n2
cutoff_N_2=[21,21] 
#plt.plot(0,0,marker='none',label="fit: $y=ax^{2}$",linestyle='none')
for j in range(K.size):
    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
  
    n_2,n_2_error=compute_n_stress_diff(stress_tensor_tuple[j], 
                          stress_tensor_std_tuple[j],
                          2,1,
                          j_,n_plates,
                          )
    plt.errorbar(erate[cutoff:e_end[j]], n_2[cutoff:e_end[j]], yerr =n_2_error[cutoff:e_end[j]],
                  ls="none",label="$N_{2},K="+str(K[j])+"$",marker=marker[j] )

    # popt,cov_matrix_n2=curve_fit(quadratic,erate[cutoff:quadratic_end], n_2[cutoff:quadratic_end])
    # difference=np.sqrt(np.sum((n_2[cutoff:quadratic_end]-(popt[0]*(erate[cutoff:quadratic_end])**2))**2)/(quadratic_end))

    # plt.plot(erate[cutoff:quadratic_end],popt[0]*(erate[cutoff:quadratic_end])**2,ls=linestyle_tuple[j][1],#)#,
    #         label="$N_{2,fit,K="+str(K[j])+"},a="+str(sigfig.round(popt[0],sigfigs=2))+\
    #             ",\\varepsilon="+str(sigfig.round(difference,sigfigs=2))+"$")

    #plt.xscale('log')
    #plt.show()
    #print(difference)


plt.legend(fontsize=9, frameon=False, loc="best")
#plt.xscale('log')
plt.xlabel("$\dot{\gamma}$")
plt.ylabel("$N_{2}$",rotation=0)
#plt.yticks(n_y_ticks)
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/N1_vs_gdot_ybxa_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()
#print(difference)


#%%collapse N1 and N2 /sigma_xz
cutoff=1
for j in range(K.size):
    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
  

    plt.errorbar(erate[cutoff:e_end], n_1[j,cutoff:e_end]/stress_tensor[j,cutoff:e_end,3], yerr =np.abs(n_1_error[j,cutoff:e_end]/stress_tensor[j,cutoff:e_end,3]),
                  ls='none',label="$K="+str(K[j])+"$",marker=marker[j] )
   

    #plt.xscale('log')
    #plt.show()
    #print(difference)

    # plt.errorbar(erate[cutoff:e_end], n_2[j,cutoff:e_end]/K[j], yerr =n_2_error[j,cutoff:e_end]/K[j],
    #               ls='none',label="$N_{2},K="+str(K[j])+"$",marker=marker[j] )
    # popt,cov_matrix_n2=curve_fit(linearthru0,erate[cutoff:e_end], n_2[j,cutoff:e_end])
    # difference=np.sqrt(np.sum((n_2[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

    # plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),
    #         label="$N_{2,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
    #         ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$",ls=linestyle_tuple[j])
    #
    plt.legend(fontsize=legfont,frameon=False)
    plt.ylabel("$\\frac{N_{1}}{\sigma_{xz}}$",rotation=0)
    #plt.xscale('log')
    plt.xlabel("$\dot{\gamma}$")
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/N1_scaled_sigxz_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()
#print(difference)

cutoff=1
for j in range(K.size):
    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
  

    plt.errorbar(erate[cutoff:e_end], n_2[j,cutoff:e_end]/stress_tensor[j,cutoff:e_end,3], yerr =np.abs(n_1_error[j,cutoff:e_end]/stress_tensor[j,cutoff:e_end,3]),
                  ls='none',label="$K="+str(K[j])+"$",marker=marker[j] )
   

    #plt.xscale('log')
    #plt.show()
    #print(difference)

    # plt.errorbar(erate[cutoff:e_end], n_2[j,cutoff:e_end]/K[j], yerr =n_2_error[j,cutoff:e_end]/K[j],
    #               ls='none',label="$N_{2},K="+str(K[j])+"$",marker=marker[j] )
    # popt,cov_matrix_n2=curve_fit(linearthru0,erate[cutoff:e_end], n_2[j,cutoff:e_end])
    # difference=np.sqrt(np.sum((n_2[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

    # plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),
    #         label="$N_{2,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
    #         ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$",ls=linestyle_tuple[j])
    #
    plt.legend(fontsize=legfont,frameon=False)
    plt.ylabel("$\\frac{N_{2}}{\sigma_{xz}}$",rotation=0)
    #plt.xscale('log')
    plt.xlabel("$\dot{\gamma}$")
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/N2_scaled_sigxz_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()



# now do N1 N2 comparison plots
# this plot isnt so useful 
n2_factor=[-10,-10,-15,-20]
for j in range(K.size):

    #sns.set_palette('icefire')
    plt.scatter(erate,n_1[j],label="$N_{1},K="+str(K[j])+"$", marker=marker[j])
    plt.scatter(erate,n2_factor[j]*n_2[j],label="$"+str(n2_factor[j])+"N_{2},K="+str(K[j])+"$", marker=marker[j+2])
    plt.xlabel("$\dot{\gamma}$")
    plt.ylabel("$N_{\\alpha}$",rotation=0)
    #plt.ylabel("$\\frac{N_{1}}{N_{2}}$", rotation=0)
    plt.legend()
    #plt.xscale('log')
    plt.legend(fontsize=legfont) 
    #plt.yticks(n_y_ticks)
    plt.tight_layout()

   # plt.savefig(path_2_log_files+"/plots/N1_N2_multi_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
    plt.show()


#%%viscosity for plate 
# only fit the thickening region not the independent region 
cutoff=10
#plt.plot(0,0,marker='none',label="fit: $y=ax^{n}$",linestyle='none')
for j in range(0,K.size):
    xz_stress= stress_tensor_tuple[j][cutoff:,3]
    xz_stress_std=stress_tensor_std_tuple[j][cutoff:,3]/np.sqrt(j_*n_plates)
    #powerlaw
    plt.errorbar(erate[cutoff:e_end[j]], xz_stress/erate[cutoff:e_end[j]], yerr =xz_stress_std[:]/erate[cutoff:e_end[j]],
                  ls='none',label="$\eta,K="+str(K[j])+"$",marker=marker[j] )
    # plt.plot(erate[cutoff:e_end], xz_stress/erate[cutoff:e_end],
    #               ls='none',label="$\eta,K="+str(K[j])+"$",marker=marker[j] )
    # popt,cov_matrix_xz=curve_fit(powerlaw,erate[cutoff:e_end[j]], xz_stress/erate[cutoff:e_end[j]])
    # y=xz_stress/erate[cutoff:e_end[j]]
    # y_pred=popt[0]*(erate[cutoff:e_end[j]]**(popt[1]))
    # difference=np.sqrt(np.sum((y-y_pred)**2)/e_end[j]-cutoff)
    # plt.plot(erate[cutoff:e_end[j]],popt[0]*(erate[cutoff:e_end[j]]**(popt[1])),
    #       )
    # print("a=",popt[0])
    # print("n=",popt[1])
# label="$K="+str(K[j])+",a="+str(sigfig.round(popt[0],sigfigs=3))+",n="+str(sigfig.round(popt[1],sigfigs=3))+
 #            ",\\varepsilon=\\pm"+str(sigfig.round(difference,sigfigs=3))+"$" 
    plt.legend(fontsize=11,frameon=True) 
    plt.ylabel("$\eta$", rotation=0,labelpad=10)
    plt.xlabel("$\dot{\gamma}$")
    plt.tight_layout()
#plt.xscale('log')
# plt.yscale('log')
#plt.savefig(path_2_log_files+"/plots/eta_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show() 

#%%viscosity for dumbell

cutoff=1 
for j in range(K.size):
    xz_stress= stress_tensor[j,cutoff:,3]
    xz_stress_std=stress_tensor_std[j,:,3]/np.sqrt(j_*n_plates)
    #powerlaw
    plt.errorbar(erate[cutoff:e_end], xz_stress/erate[cutoff:e_end], yerr =xz_stress_std[cutoff:],
                  ls='none',label="$\eta,K="+str(K[j])+"$",marker=marker[j] )
    # popt,cov_matrix_xz=curve_fit(powerlaw,erate[cutoff:e_end], xz_stress/erate[cutoff:e_end])
    # y=xz_stress/erate[cutoff:e_end]
    # y_pred=popt[0]*(erate[cutoff:e_end]**(popt[1]))
    # difference=np.sqrt(np.sum((y-y_pred)**2)/e_end-cutoff)
    # plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end]**(popt[1]))),
    #         label="$\eta_{fit},a="+str(sigfig.round(popt[0],sigfigs=3))+",n="+str(sigfig.round(popt[1],sigfigs=3))+

    #         ",\\varepsilon=\pm"+str(sigfig.round(difference,sigfigs=3))+"$")

   
    plt.ylabel("$\eta/\eta_{s}$", rotation=0,labelpad=10)
    plt.xlabel("$\dot{\gamma}$")
plt.tight_layout()
plt.legend(fontsize=legfont) 
    # plt.xscale('log')
    # plt.yscale('log')
plt.savefig(path_2_log_files+"/plots/eta_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show() 

#%% n1  fit for dumbells

cutoff=0
j=0
plt.errorbar(erate[cutoff:e_end], n_1[j,cutoff:e_end], yerr =n_1_error[j,cutoff:e_end], ls='none',label="$N_{1}$",marker=marker[0] )
popt,cov_matrix_n1=curve_fit(quadfunc,erate[cutoff:e_end], n_1[j,cutoff:e_end])
difference=np.sqrt(np.sum((n_1[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])**2))**2)/(e_end))

plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])**2),
         label="$N_{1,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
            ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$",marker='none')
plt.legend()
plt.ylabel("$N_{1}$", rotation=0, labelpad=20)
plt.xlabel("$\dot{\gamma}$")
plt.tight_layout()
    # plt.xscale('log')
    # plt.yscale('log')
plt.savefig(path_2_log_files+"/plots/N1_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')

plt.show()


#%% area vector plots 
cutoff=0
plt.rcParams['text.usetex'] = "false"
def convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff):
        spherical_coords_tuple=()
        for i in range(len(skip_array)):
            i=skip_array[i]

            
            area_vector_ray=area_vector_spherical_batch_tuple[j][i]
            area_vector_ray[area_vector_ray[:,:,:,2]<0]*=-1
            
            x=area_vector_ray[:,cutoff:,:,0]
            y=area_vector_ray[:,cutoff:,:,1]
            z=area_vector_ray[:,cutoff:,:,2]
        

            spherical_coords_array=np.zeros((j_,area_vector_ray.shape[1]-cutoff,n_plates,3))
        



            # radial coord
            spherical_coords_array[:,:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))

            #  theta coord 
            spherical_coords_array[:,:,:,1]=np.sign(y)*np.arccos(x/(np.sqrt((x**2)+(y**2))))

            #spherical_coords_array[:,:,:,1]=np.sign(x)*np.arccos(y/(np.sqrt((x**2)+(y**2))))
            #spherical_coords_array[:,:,:,1]=np.arctan(y/x)
            
            # phi coord
            # print(spherical_coords_array[spherical_coords_array[:,:,:,0]==0])
            spherical_coords_array[:,:,:,2]=np.arccos(z/np.sqrt((x**2)+(y**2)+(z**2)))

        

            spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_array,)

        return spherical_coords_tuple

def stat_test_on_theta(periodic_data,sample_size):
    KS_test_result=[]
    MW_test_result=[]
    for m in range(100):                
                    # scotts factor 
                    np.random.seed(m)
                    uniform=np.random.uniform(low=-3*np.pi, high=3*np.pi,size=periodic_data.size)
                    sample1 = np.random.choice(uniform,size=sample_size, replace = True, p = None)
                    periodic_sample=np.random.choice( np.ravel(periodic_data) , size = sample_size, replace = True, p = None)
                    print(f'Uniform vs. My data: {scipy.stats.ks_2samp( periodic_sample,sample1)}')
                    KS_test_result.append(scipy.stats.ks_2samp(  periodic_sample,sample1)[1])
                   # MW_test_result.append(scipy.stats.mannwhitneyu(  periodic_sample,sample1)[1])

    return KS_test_result,MW_test_result


                                    
                   
#sns.set_theme(font_scale=1.5, rc={'text.usetex' : True})

def producing_random_points_with_theta(number_of_points,rand_int):

    rng = np.random.default_rng(rand_int)
    Phi=np.arccos(1-2*(rng.random((number_of_points))))
    
    Theta=2*np.pi*rng.random((number_of_points))
    rho=1#7.7942286341
    A=Phi
    B=Theta
    R=np.array([rho*np.sin(A)*np.cos(B),rho*np.sin(B)*np.sin(A),rho*np.cos(A)])


    return Phi,Theta,R
                
                # scotts factor 
def stat_test_on_phi(periodic_data,sample_size):
    KS_test_result=[]
    MW_test_result=[]
    for m in range(100):                
                
                    Phi,Theta,R=producing_random_points_with_theta(periodic_data.size,m)

                    sample_sin=np.random.choice( Phi , size = sample_size, replace = True, p = None)
                    periodic_sample=np.random.choice( np.ravel(periodic_data) , size = sample_size, replace = True, p = None)
                    KS_test_result.append(scipy.stats.ks_2samp( periodic_sample,sample_sin)[1])
                    #MW_test_result.append(scipy.stats.mannwhitneyu( periodic_sample,sample_sin)[1])
                    
                    print(f'sampled sine vs. My data sample KS test: {scipy.stats.ks_2samp( periodic_sample,sample_sin)}')
                    #MW only makes sense in ordinal data - no natural ranking 
                   # print(f'sampled sine vs. My data sample Mannwhitney test: {scipy.stats.mannwhitneyu( periodic_sample,sample_sin)}')
                   # print(f'sampled sine vs. My data sample ranksums test: {scipy.stats.ranksums( periodic_sample,sample_sin)}')

    return KS_test_result,MW_test_result


def plot_MW_test(MW_test_result):

    plt.plot(MW_test_result,label="MW_test_phi")
    plt.axhline(np.mean(MW_test_result),label="MWmean$="+str(np.mean(MW_test_result))+",\pm"+str(np.std(MW_test_result))+"$",color="red",linestyle="dashed")
    plt.ylabel("pvalue")
    plt.legend()
    plt.show()

def plot_KS_test(KS_test_result,):

    plt.plot(KS_test_result,label="KS_test")
    plt.axhline(np.mean(KS_test_result),label="KSmean$="+str(np.mean(KS_test_result))+",\pm"+str(np.std(KS_test_result))+"$",color="green",linestyle="dotted")
    plt.ylabel("pvalue")
    plt.legend()
    plt.show()

#%%

plt.rcParams["figure.figsize"] = (6,4 )
plt.rcParams.update({'font.size': 14})
SIZE_DEFAULT = 14
SIZE_LARGE = 16
legfont=12
# plt.rcParams['text.usetex'] = True


# plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels


pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
phi_y_ticks=[0,0.2,0.4,0.6,0.8,1,1.2]
pi_phi_ticks=[ 0,np.pi/8,np.pi/4,3*np.pi/8, np.pi/2]
pi_phi_tick_labels=[ '0','π/8','π/4','3π/8', 'π/2']
theta_y_ticks=[0,0.02,0.04,0.06,0.08,0.1]
skip_array=[0,15]
spherical_coords_tuple=()
sample_cut=0
cutoff=0
sample_size=500

adjust_factor=0.005 #for all data # 4 smooths the data out 
spherical_coords_batch_tuple=()
# fig = plt.figure(constrained_layout=True)
# spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

for j in range(K.size):

    spherical_coords_tuple=convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff)

   
   

    for l in range(len(skip_array)):
    
  
            data=spherical_coords_tuple[l][:,:,:,1]
            

             
            periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
            #periodic_data=np.ravel(data)
            #if l==0 or l==3:
                
                
            KS_test_result,MW_test_result=stat_test_on_theta(periodic_data,sample_size)

            # plt.title("$\dot{\gamma}="+str(erate[skip_array[l]])+",K="+str(K[j])+"$")
            # plot_MW_test(MW_test_result)
            plt.title("$\dot{\gamma}="+str(erate[skip_array[l]])+",K="+str(K[j])+"$")
            plot_KS_test(KS_test_result)
           
    for l in range(len(skip_array)):
    
  
            data=spherical_coords_tuple[l][:,:,:,1]
            periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
            adjust=adjust_factor#*periodic_data.size**(-1/5)
            #adjust=2
            # with smoothing
            sns.kdeplot( data=periodic_data,
                        label ="$\dot{\gamma}="+str(erate[skip_array[l]],)+"$",linestyle=linestyle_tuple[l][1],bw_method="silverman",bw_adjust=adjust)
            # sns.kdeplot( data=uniform,
            #             label ="$\dot{\gamma}="+str(erate[skip_array[l]],)+"$",linestyle=linestyle_tuple[l],bw_method="silverman",bw_adjust=adjust)
            
   
    plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")         
    plt.xlabel("$\Theta$")
    plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
    #plt.yticks(theta_y_ticks)
    plt.xlim(-np.pi,np.pi)

    plt.ylabel('Density')
    plt.legend(fontsize=legfont) 
    #plt.tight_layout()
    #plt.savefig(path_2_log_files+"/plots/theta_dist_K_"+str(K[j])+"_.pdf",dpi=1200,bbox_inches='tight')
    plt.show()
 

    
    for l in range(len(skip_array)):
       
            
            data=spherical_coords_tuple[l][:,:,:,2]
          
            periodic_data=np.ravel(np.array([data,np.pi-data]))
           

            #if l==0 or l==3:
                 
            KS_test_result,MW_test_result=stat_test_on_phi(periodic_data,sample_size)

            # plt.title("$\dot{\gamma}="+str(erate[skip_array[l]])+",K="+str(K[j])+"$")
            # plot_MW_test(MW_test_result)
            plt.title("$\dot{\gamma}="+str(erate[skip_array[l]])+",K="+str(K[j])+"$")
            plot_KS_test(KS_test_result)

    for l in range(len(skip_array)):
       
            
            data=spherical_coords_tuple[l][:,:,:,2]
          
            periodic_data=np.ravel(np.array([data,np.pi-data]))
            adjust=adjust_factor#*periodic_data.size**(-1/5)
          
            sns.kdeplot( data=periodic_data,
                        label ="$\dot{\gamma}="+str(erate[skip_array[l]])+"$",linestyle=linestyle_tuple[l][1],bw_method="silverman",bw_adjust=adjust)
            # sns.kdeplot( data=Phi,
            #             label ="$\dot{\gamma}="+str(erate[skip_array[l]])+"$",linestyle=linestyle_tuple[l],bw_method="silverman",bw_adjust=adjust)

    plt.plot(0,0,marker='none',ls="none",color='grey',label="$K="+str(K[j])+"$")        
    plt.xlabel("$\Phi$")
    plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
    plt.ylabel('Density')
    plt.legend(fontsize=legfont,loc='upper right') 
    plt.xlim(0,np.pi/2)
    # plt.tight_layout()
    #plt.savefig(path_2_log_files+"/plots/phi_dist_K_"+str(K[j])+"_.pdf",dpi=1200,bbox_inches='tight')
    plt.show()

    spherical_coords_batch_tuple=spherical_coords_batch_tuple+(spherical_coords_tuple,)

#%% phi plot on same plot
skip_array=np.array([0,1,7,14,18,20,21,23,25,27,30,33,35,37,38])

adjust_factor=4
for j in range(K.size):

    spherical_coords_tuple=convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff)
    for i in range(len(skip_array)):

        data=np.ravel( spherical_coords_tuple[i][:,:,:,2])
        periodic_data=np.ravel(np.array([data,np.pi-data]))
        sns.kdeplot( data=periodic_data,
                        label ="$\dot{\gamma}="+str(erate[skip_array[i]])+"$",linestyle=linestyle_tuple[j],bw_adjust=adjust_factor)
    plt.xticks(pi_phi_ticks,pi_phi_tick_labels)

    #plt.yticks(phi_y_ticks)
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1,0.5),frameon=False)
    plt.xlim(0,np.pi/2)
    #plt.xlim(0,np.pi)
    plt.tight_layout()
    #plt.savefig(path_2_log_files+"/plots/phi_dist_.pdf",dpi=1200,bbox_inches='tight')
    plt.show()
        

#%% different style plot of phi 
linestyle_tuple = ['-', 
  'dotted', 
 'dashed', 'dashdot', 
  'solid', 
 'dashed', 'dashdot', '--']
skip_array=[0,10,21,23]
#skip_array=[0,18,19,23]
skip_array=[0,3,5,7]
# skip_array=[8,9,10,11]
# skip_array=[12,13,14,15]
# skip_array=[16,17,18,19]
# skip_array=[20,21,22,23]
# skip_array=[22,24,26,28]
# skip_array=[29,30,31,32]
# skip_array=[33,34,35,36]
#phi 
f, axs = plt.subplots(1, 4, figsize=(15, 6),sharey=True,sharex=True)
adjust_factor=2
#for j in range(1,K.size):
for j in range(0,1):
    spherical_coords_tuple=convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff)
    

    i=0
    data=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    sns.kdeplot( data=periodic_data,
                      label ="K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[0],bw_adjust=adjust_factor)
    axs[0].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=1
    data=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    sns.kdeplot( data=periodic_data,
                      label ="$\dot{\gamma}="+str(erate[skip_array[i]])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[1],bw_adjust=adjust_factor)
    axs[1].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=2
    data=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[2],bw_adjust=adjust_factor)
    axs[2].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=3
    data=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    periodic_data=np.ravel(np.array([data,np.pi-data]))
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[3],bw_adjust=adjust_factor)
    axs[3].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")
f.supxlabel("$\Phi$")
plt.xticks(pi_phi_ticks,pi_phi_tick_labels)

#plt.yticks(phi_y_ticks)
plt.ylabel('Density')
plt.legend(bbox_to_anchor=(1,0.5),frameon=False)
plt.xlim(0,np.pi/2)
#plt.xlim(0,np.pi)
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/phi_dist_.pdf",dpi=1200,bbox_inches='tight')
plt.show()


#%% different style plot of theta

#theta

f, axs = plt.subplots(1, 4, figsize=(15, 6),sharey=True,sharex=True)
adjust_factor=0.5
for j in range(0,K.size):
    spherical_coords_tuple=convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff)

   
    i=0
    data=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[0],bw_adjust=adjust_factor)
    
    axs[0].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=1
    data=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[1],bw_adjust=adjust_factor)
  
    axs[1].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=2
    data=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[2],bw_adjust=adjust_factor)
    
    axs[2].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=3
    data=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    periodic_data=np.ravel(np.array([data-2*np.pi,data,data+2*np.pi]) )
    sns.kdeplot( data=periodic_data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[3],bw_adjust=adjust_factor)
   
    axs[3].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


f.supxlabel("$\Theta$")
plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
plt.legend(bbox_to_anchor=(1,0.55),frameon=False)


#plt.yticks(phi_y_ticks)

plt.ylabel('Density')
plt.xlim(-np.pi,np.pi)
#plt.xlim(0,np.pi)
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/theta_dist_.pdf",dpi=1200,bbox_inches='tight')
plt.show()


  
#%% different style plot of rho

#rho

f, axs = plt.subplots(1, 4, figsize=(15, 6),sharey=True,sharex=True)
adjust_factor=2
for j in range(0,K.size):
    spherical_coords_tuple=convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff)

   
    i=0
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
   
    sns.kdeplot( data=data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[0],bw_adjust=adjust_factor)
    
    axs[0].axvline(np.mean(data))
    
    axs[0].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")
    print("mean rho 0",np.mean(data))


    i=1
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
    
    sns.kdeplot( data=data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[1],bw_adjust=adjust_factor)
    axs[1].axvline(np.mean(data), label="$\\bar{\\rho}="+str(np.mean(data))+"$")
    print("mean rho 1",np.mean(data))
    axs[1].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=2
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
   
    sns.kdeplot( data=data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[2],bw_adjust=adjust_factor)
    
    axs[2].axvline(np.mean(data))
    print("mean rho 2",np.mean(data))
    
    axs[2].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=3
    data=np.ravel( spherical_coords_tuple[i][:,:,:,0])
   
    sns.kdeplot( data=data,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],ax=axs[3],bw_adjust=adjust_factor)
   
    axs[3].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")
    print("mean rho 3",np.mean(data))
    axs[3].axvline(np.mean(data))


f.supxlabel("$\\rho$")

plt.legend(bbox_to_anchor=(1,0.55),frameon=False)


#plt.yticks(phi_y_ticks)

plt.ylabel('Density')

#plt.xlim(0,np.pi)
plt.tight_layout()

plt.show()
#%% plto theta against phi 

#rho
markersize=0.5
f, axs = plt.subplots(1, 4, figsize=(15, 6),sharey=True,sharex=True)
adjust_factor=1
for j in range(0,K.size):
    spherical_coords_tuple=convert_cart_2_spherical_z_inc(j,skip_array,area_vector_spherical_batch_tuple,
                                       n_plates,cutoff)

   
    i=0
    theta=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    phi=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    axs[0].scatter(theta,phi,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],s=markersize)
    
    axs[0].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=1
    theta=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    phi=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    axs[1].scatter(theta,phi,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],s=markersize)
    
   
  
    axs[1].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=2
    theta=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    phi=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    axs[2].scatter(theta,phi,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],s=markersize)
    
    axs[2].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    i=3
    theta=np.ravel( spherical_coords_tuple[i][:,:,:,1])
    phi=np.ravel( spherical_coords_tuple[i][:,:,:,2])
    axs[3].scatter(theta,phi,
                      label ="$K="+str(K[j])+"$",linestyle=linestyle_tuple[j],s=markersize)
   
    axs[3].set_title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$")


    f.supxlabel("$\\Theta$")
    f.supylabel("$\\Phi$", rotation=0)
    plt.legend(bbox_to_anchor=(1,0.55),frameon=False)

    # plt.yticks(pi_phi_ticks,pi_phi_tick_labels)

    # plt.xticks(theta_y_ticks,pi_theta_tick_labels)

    # #plt.ylabel('Density')

    plt.ylim(0,np.pi/2)
    plt.xlim(-np.pi,np.pi)
    plt.tight_layout()

    plt.show()

#%% comparing across K
skip_array=[0,9,13,18]



f, axs = plt.subplots(1, 4, figsize=(15, 6),sharey=True,sharex=True)

#for i in range(len(skip_array)):
adjust=1
for j in range(K.size):
        i=0
        magnitude_spring=interest_vectors_batch_tuple[j][i][:,:,2:5]

        mean_spring_mag=np.mean(magnitude_spring)
          
        
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[0])
        axs[0].set_title("$\dot{\gamma}="+str(erate[i])+"$")
        i=9
        magnitude_spring=interest_vectors_batch_tuple[j][9][:,:,2:5]
       
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[1])
        axs[1].set_title("$\dot{\gamma}="+str(erate[i])+"$")
        i=13
        magnitude_spring=interest_vectors_batch_tuple[j][13][:,:,2:5]
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[2])
        axs[2].set_title("$\dot{\gamma}="+str(erate[i])+"$")
        i=18
        magnitude_spring=interest_vectors_batch_tuple[j][18][:,:,2:5]
        
      
        sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                    label ="$K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[j], ax=axs[3])
        axs[3].set_title("$\dot{\gamma}="+str(erate[i])+"$")
        
        

#plt.yticks(extension_ticks)
#plt.xticks()

plt.legend(bbox_to_anchor=(1,0.55),frameon=False)
f.supxlabel("$\Delta x$")
f.tight_layout()

plt.savefig(path_2_log_files+"/plots/deltax_dist_.pdf",dpi=1200,bbox_inches='tight')
   
plt.show()

#%% comparing erates
skip_array=[0,9,13,18]
#skip_array=[0]

#dist_xticks=([[-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-7.5,-5,-2.5,0,2.5,5]])

#for i in range(len(skip_array)):

adjust=1
for j in range(K.size):
  
    for i in range(len(skip_array)):
            k=skip_array[i]
           
            magnitude_spring=interest_vectors_batch_tuple[j][k][:,:,2:5]

            mean_spring_mag=np.mean(magnitude_spring)
          

            sns.kdeplot(np.ravel(magnitude_spring)-eq_spring_length,
                        label ="$\dot{\gamma}="+str(erate[k])+",K="+str(K[j])+"$",bw_adjust=adjust,linestyle=linestyle_tuple[i])
            plt.axvline(mean_spring_mag,label="$\\bar{\Delta x}="+str(sigfig.round(mean_spring_mag,sigfigs=3))+", \dot{\gamma}="+str(erate[k])+",K="+str(K[j])+"$",linestyle=linestyle_tuple[i])
         
        
            

    #plt.yticks(extension_ticks)

    #plt.xticks()


    plt.xlabel("$\Delta x$")
    plt.legend(fontsize=legfont,bbox_to_anchor=(0.75,0.5), frameon=False)
    plt.xlim(-2,5.5)
    plt.tight_layout()
    

    plt.savefig(path_2_log_files+"/plots/deltax_dist_K"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')
    
    plt.show()





#%% plotting particle velocity against postion 
skip_array=[1,5,10,15,18]
for j in range(K.size):
    for i in range(len(skip_array)):
        i=skip_array[i]


        
        z_position =np.mean(np.mean(pos_batch_tuple[j][i][:,:,:,:,2],axis=0),axis=1)
        x_vel=np.mean(np.mean(vel_batch_tuple[j][i][:,:,:,:,0],axis=0),axis=1)
        pred_x_vel=erate[i]* z_position
        plt.scatter(z_position,x_vel, label="$\dot{\gamma}="+str(erate[i])+"$")
        plt.plot(z_position,pred_x_vel)
        plt.xlabel("$z$")
        plt.ylabel("$v_{x}$",rotation=0,labelpad=20)
        
    plt.legend(bbox_to_anchor=(1,1))
   # plt.savefig(path_2_log_files+"/plots/vx_vs_z_"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')
    plt.show()

#%% plotting initial velocity distribution 

for j in range(K.size):
    # only need the velocity distribution at gammadot=0
    for i in range(erate.size):
        f, axs = plt.subplots(1, 3, figsize=(16, 6),sharey=True,sharex=True)
        
        x_vel=np.ravel(pos_vel_batch_tuple[j][i][0,:,3,0])
        sns.kdeplot( data=x_vel, ax=axs[0])
        
        y_vel=np.ravel(pos_vel_batch_tuple[j][i][0,:,3,1])
        sns.kdeplot( data=y_vel, ax=axs[1])
       
        z_vel=np.ravel(pos_vel_batch_tuple[j][i][0,:,3,2])
        sns.kdeplot( data=z_vel, ax=axs[2])
       
        plt.show()

#%%fitting gaussian for velocity component 
for j in range(K.size):
    # only need the velocity distribution at gammadot=0
    for i in range(1):
        maxwell = scipy.stats.norm
        data = np.ravel(vel_batch_tuple[j][i][:,:,:,:,0])

        params = maxwell.fit(data)
        print(params)
        # (0, 4.9808603062591041)

        plt.hist(data, bins=20,density=True)
        x = np.linspace(np.min(data),np.max(data),data.size)
        plt.plot(x, maxwell.pdf(x, *params), lw=3)
        plt.xlabel("$v_{x}$")
        plt.ylabel("Density")
        #plt.savefig(path_2_log_files+"/plots/vx_dist_"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')
        plt.show()
        data = np.ravel(vel_batch_tuple[j][i][:,:,:,:,1])

        params = maxwell.fit(data)
        print(params)
        # (0, 4.9808603062591041)

        plt.hist(data, bins=20,density=True)
        x = np.linspace(np.min(data),np.max(data),data.size)
        plt.plot(x, maxwell.pdf(x, *params), lw=3)
        plt.xlabel("$v_{y}$")
        plt.ylabel("Density")
       # plt.savefig(path_2_log_files+"/plots/vy_dist_"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')
        plt.show()
        data = np.ravel(vel_batch_tuple[j][i][:,:,:,:,2])

        params = maxwell.fit(data)
        print(params)
        # (0, 4.9808603062591041)

        plt.hist(data, bins=20,density=True)
        x = np.linspace(np.min(data),np.max(data),data.size)
        plt.plot(x, maxwell.pdf(x, *params), lw=3)
        plt.xlabel("$v_{z}$")
        plt.ylabel("Density")
       # plt.savefig(path_2_log_files+"/plots/vz_dist_"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')

        plt.show()

#%% looking at maxwell_ botlzman  for speed dist 

for j in range(K.size):
    # only need the velocity distribution at gammadot=0
    for i in range(1):
        
        x_vel=np.ravel(pos_vel_batch_tuple[j][i][:,:,3:6,0])
       
        
        y_vel=np.ravel(pos_vel_batch_tuple[j][i][:,:,3:6,1])
        
       
        z_vel=np.ravel(pos_vel_batch_tuple[j][i][:,:,3:6,2])
        speed= np.sqrt(x_vel**2+ y_vel**2 + z_vel**2)
        maxwell = scipy.stats.maxwell
        params = maxwell.fit(speed)
        print(params)
        # (0, 4.9808603062591041)

        plt.hist(speed, bins=20,density=True)
        x = np.linspace(np.min(speed),np.max(speed),speed.size)
        plt.plot(x, maxwell.pdf(x, *params), lw=3)
        plt.xlabel("$|v|$")
        plt.ylabel("Density")
      #  plt.savefig(path_2_log_files+"/plots/v_dist_"+str(K[j])+".pdf",dpi=1200,bbox_inches='tight')
        plt.show()
# %% interest vectors

ell_1=interest_vectors_batch_tuple[0][15][:,:,4]
ell_2=interest_vectors_batch_tuple[0][0][:,:,1]

sns.kdeplot(np.ravel(ell_1))
plt.show()

#%% ridgeline plot of phi distributions 


#%% ridgeline plot of theta distributions 


#%% violin plot of phi 
sns.set_palette('colorblind')
# sns.color_palette("mako", as_cmap=True)
# sns.color_palette("viridis")
#sns.set_palette('virdris')

plt.rcParams.update({'font.size': 14})
SIZE_DEFAULT = 14
SIZE_LARGE = 16
#plt.rcParams['text.usetex'] = True
# plt.rc("font", family="Roboto")  # controls default font
# plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
legfont=10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False 

plt.rcParams.update({'font.size': 16})
adjust_factor = 2
erate_1=0
erate_2=37
plt.rcParams["figure.figsize"] = (25,6 )
for j in range(K.size):
    skip_array = np.arange(erate_1, erate_2, 1)
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff)
    periodic_data_list = []
    erate_list = []
    for i in range(skip_array.size):
        data = np.ravel(spherical_coords_tuple[i][:, :, :, 2])  # Assuming this extracts the spherical data
        periodic_data = np.ravel(np.array([data, np.pi - data]))  # Handling the periodic nature
        periodic_data_list.append(periodic_data)


    # Convert lists to DataFrames at the end
    periodic_data_df = pd.DataFrame(periodic_data_list)
    periodic_data_df=periodic_data_df.transpose()
    erate_str=np.around(erate[erate_1:erate_2],3).astype("str")
    periodic_data_df.columns= erate_str
    print(periodic_data_df.isna().sum())
    # erate_df = pd.DataFrame(erate[:e_end[j]])
    # full_df = pd.concat([erate_df, periodic_data_df], axis=0)
    # full_df = full_df.rename(columns={full_df.columns[0]: "erate"})

    # # rename columns 1 to end 
    # full_df.columns = full_df.columns[:1].tolist() + [f"part_angle" for i in range(1, len(full_df.columns))]

    # # Combine both DataFrames into a final DataFrame
  
        


    
   
    sns.violinplot( data=periodic_data_df, inner=None, linewidth=0 ,scale="width")
    plt.yticks(pi_phi_ticks,pi_phi_tick_labels)
    plt.ylim(0,np.pi/2)
    plt.ylabel("$\Phi$")
    plt.xlabel("$\dot{\gamma}$")
    plt.show()

   
#%% violin plot of theta 
adjust_factor = 0.005
erate_1=0
erate_2=21

for j in range(0, K.size):
    skip_array = np.arange(erate_1, erate_2, 1)
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff)
    periodic_data_list = []
    erate_list = []
    for i in range(skip_array.size):
        data = np.ravel(spherical_coords_tuple[i][:, :, :, 1])  # Assuming this extracts the spherical data
        periodic_data = np.ravel(np.array([data, np.pi - data]))  # Handling the periodic nature
        periodic_data_list.append(periodic_data)

        

    # Convert lists to DataFrames at the end
    periodic_data_df = pd.DataFrame(periodic_data_list)
    periodic_data_df=periodic_data_df.transpose()
    
    erate_str=erate[erate_1:erate_2].astype("str")
    periodic_data_df.columns= erate_str
    print(periodic_data_df.isna().sum())
    # erate_df = pd.DataFrame(erate[:e_end[j]])
    # full_df = pd.concat([erate_df, periodic_data_df], axis=0)
    # full_df = full_df.rename(columns={full_df.columns[0]: "erate"})

    # # rename columns 1 to end 
    # full_df.columns = full_df.columns[:1].tolist() + [f"part_angle" for i in range(1, len(full_df.columns))]

    # # Combine both DataFrames into a final DataFrame
  
        


    
   
    sns.violinplot( data=periodic_data_df, inner=None, linewidth=0 ,scale="width")
    plt.ylim(-np.pi,np.pi)
    plt.yticks(pi_theta_ticks,pi_theta_tick_labels)
    plt.ylabel("$\Theta$")
    plt.xlabel("$\dot{\gamma}$")
    plt.show()
# %%
