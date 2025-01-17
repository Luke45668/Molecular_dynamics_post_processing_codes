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
# plt.rcParams["figure.figsize"] = (8,6 )
# plt.rcParams.update({'font.size': 16})
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import mmap
import h5py as h5
from scipy.optimize import curve_fit


#path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
# os.chdir(path_2_post_proc_module)
import seaborn as sns

import glob 
# from post_MPCD_MP_processing_module import *
import pickle as pck
from post_langevin_module import *
from reading_lammps_module import *

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
marker=['x','+','^',"1","X","d","*","P","v","."]

damp=np.array([ 0.035, 0.035 ,0.035,0.035,0.035,0.035])
K=np.array([ 30, 60,100,150,300,600 ])
K=np.array([ 30,60,100,300 ])
K=np.array([ 15,60 ])

#K=np.array([  100,300,600,1200 ])
thermal_damp_multiplier=np.flip(np.array([25,25,25,25,25,25,25,100,100,100,100,100,
100,100,100,100,250,250]))/10

erate=np.flip(np.linspace(1,0.005,24))


e_in=0
e_end=erate.size
n_plates=100

strain_total=100

path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/final_plate_run_x_stretch/no_visc_10_reals_5tstats"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/final_plate_runs_tuples/"

thermo_vars='         KinEng         PotEng         Press           Temp         Ecouple       Econserve    c_uniaxnvttemp'

j_=10

sim_fluid=30.315227255599112

eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
damp_ratio=mass_pol/damp


#%% load in tuples
label='damp_'+str(damp)+'_K_'+str(K)+'_'


os.chdir(path_2_log_files)
#os.mkdir("tuple_results")
#os.chdir("tuple_results")

def batch_load_tuples(label,tuple_name):

    with open(label+tuple_name, 'rb') as f:
         load_in= pck.load(f)

    return load_in


spring_force_positon_tensor_batch_tuple=()
log_file_batch_tuple=()
dirn_vector_batch_tuple=()
transformed_pos_batch_tuple=()
transformed_vel_batch_tuple=()
e_end=[]



for i in range(K.size):
    label='damp_'+str(damp[i])+'_K_'+str(K[i])+'_'
    #label='damp_'+str(thermal_damp_multiplier[i])+'_K_'+str(K[0])+'_'
    print(label)

    spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(batch_load_tuples(label,
                                                            "spring_force_positon_tensor_tuple.pickle"),)
    print(len( spring_force_positon_tensor_batch_tuple[i]))
   
    log_file_batch_tuple=log_file_batch_tuple+(batch_load_tuples(label,
                                                            "log_file_tuple.pickle"),)
    print(len(log_file_batch_tuple[i]))
    
    dirn_vector_batch_tuple=dirn_vector_batch_tuple+(batch_load_tuples(label,
                                                            "dirn_vector_tuple.pickle"),)
    print(len(dirn_vector_batch_tuple[i]))
    
    transformed_pos_batch_tuple=transformed_pos_batch_tuple+(batch_load_tuples(label,
                                                            "transformed_pos_tuple.pickle"),)
    
    print(len(transformed_pos_batch_tuple[i]))

    transformed_vel_batch_tuple=transformed_vel_batch_tuple+(batch_load_tuples(label,
                                                            "transformed_vel_tuple.pickle"),)
    print(len(transformed_vel_batch_tuple[i]))
    
    e_end.append(len(spring_force_positon_tensor_batch_tuple[i]))
    

# need to add velocity and speed distributions, but need an equilirbium simulation to verify 

#%% calculating ensemble means and trajectory means 

# need to make this plot more clear 
cutoff=500
ensemble_mean=np.mean(spring_force_positon_tensor_batch_tuple[0][0][:,-1,:,:],axis=0)
ensemble_mean=np.mean(ensemble_mean,axis=0)
ensemble_std=np.std(spring_force_positon_tensor_batch_tuple[0][0][:,-1,:,:],axis=0)
ensemble_std=np.std(ensemble_std,axis=0)

# take mean over all the trajectories to get an ensemble average set of points 



trajecctory_mean=np.mean(spring_force_positon_tensor_batch_tuple[0][0],axis=1)# over time 
trajecctory_mean=np.mean(trajecctory_mean,axis=1) # over space 

trajectory_std=np.std(spring_force_positon_tensor_batch_tuple[0][0],axis=1)
trajectory_std=np.std(trajectory_std,axis=1)

# take a time average for each trajectory, then compare the mean for selected particles, to the same particle in the ensemble average 



skip_array=[0,50,100,150,200,250,299]
for i in range(j_):
    for j in range(1):
        for k in range(len(skip_array)):
            l=skip_array[k]

           
            # plt.axhline(trajecctory_mean[i,k,j]+trajectory_std[i,k,j], label="std dev upper bound of trajectory",linestyle='--')
            # plt.axhline(trajecctory_mean[i,k,j]-trajectory_std[i,k,j], label="std dev lower bound of trajectory",linestyle=':')

            
            plt.axhline(ensemble_mean[j]) # compare trajectory to mean of ensemble signal 
            plt.axhline(trajecctory_mean[i,j])
            plt.xlabel("output count")
            plt.ylabel("stress value")
            plt.legend()
            plt.show()

# need to extend this test to all shear rates 
# probably use a subplot to show the selected particles for each shear rate. 

# %%
