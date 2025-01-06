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
K=np.array([  20, 40, 60  
            ])


# K=np.array([  50   ,
#             ])
erate=np.flip(np.array([1.   , 0.9  , 0.8  , 0.7  , 0.6  , 0.5  , 0.4  , 0.3  , 0.2  ,
       0.175, 0.15 , 0.125, 0.1  , 0.08 , 0.06 , 0.04 , 0.02 , 0.01 ,
       0.005, 0.  ]))

erate=np.flip(np.array([1.   , 0.9  , 0.8  , 0.7  , 0.6  , 0.5  , 0.4  , 0.3  , 0.2  ,
       0.175, 0.15 , 0.125, 0.1  , 0.08 , 0.06 , 0.04  , 0.01 ,
       0.005, 0.  ]))

# no_timesteps=np.flip(np.array([ 157740000,  175267000,  197175000,  225343000,  262901000,
#          315481000,  394351000,  525801000,  788702000,   90137000,
#          105160000,  126192000,  157740000,  197175000,  262901000,
#          394351000,  394351000,  788702000, 1577404000,   10000000]))

e_in=0
e_end=erate.size
n_plates=100

strain_total=100

path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/run_279865/saved_tuples"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/dumbell_run/log_tensor_files/saved_tuples"
#path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/final_tuples"


thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
j_=5
sim_fluid=30.315227255599112

eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
damp_ratio=mass_pol/damp

def one_term_poly(x,a,b):
     return b*(x**a)

def quadratic_no_constant(x,a,b):
     return a*x + b*(x**2)
     
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
pos_vel_batch_tuple=()

pop_index=3
# loading all data into one 
for i in range(K.size):

    label='damp_'+str(damp[i])+'_K_'+str(K[i])+'_'

    if i==0: # for the 20 run with too many things in 

      
        spring_force_positon_tensor_list= list(batch_load_tuples(label,
                                                                "spring_force_positon_tensor_tuple.pickle"))
        spring_force_positon_tensor_list.pop(pop_index)
        spring_force_positon_tensor_tuple=tuple(spring_force_positon_tensor_list)
        spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(spring_force_positon_tensor_tuple,)

       
        pos_vel_list= list(batch_load_tuples(label,
                                                    "new_pos_vel_tuple.pickle"))
        pos_vel_list.pop(pop_index)
        pos_vel_tuple=tuple(pos_vel_list)
        pos_vel_batch_tuple=pos_vel_batch_tuple+(pos_vel_tuple,)


        log_file_tuple=list(batch_load_tuples(label,
                                                                "log_file_tuple.pickle"))
        log_file_tuple.pop(pop_index)
        log_file_tuple=tuple(log_file_tuple)
        log_file_batch_tuple=log_file_batch_tuple+(log_file_tuple,)

        area_vector_list=list(batch_load_tuples(label,"area_vector_tuple.pickle"))
        area_vector_list.pop(pop_index)
        area_vector_tuple=tuple(area_vector_list)
        area_vector_spherical_batch_tuple=area_vector_spherical_batch_tuple+(area_vector_tuple,)


        interest_vectors_list=list(batch_load_tuples(label,
                                                    "interest_vectors_tuple.pickle"))
        interest_vectors_list.pop(pop_index)
        interest_vector_tuple=tuple(interest_vectors_list)
        interest_vectors_batch_tuple=interest_vectors_batch_tuple+(interest_vector_tuple,)

    else:
        spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(batch_load_tuples(label,
                                                                "spring_force_positon_tensor_tuple.pickle"),)
        # erate_velocity_batch_tuple=erate_velocity_batch_tuple+(batch_load_tuples(label,
        #                                                         "erate_velocity_tuple.pickle"),)
        # COM_velocity_batch_tuple=COM_velocity_batch_tuple+(batch_load_tuples(label,
        #                                                         "COM_velocity_tuple.pickle"),)
        # conform_tensor_batch_tuple=conform_tensor_batch_tuple+(batch_load_tuples(label,
        #                                                         "conform_tensor_tuple.pickle"),)
        pos_vel_batch_tuple=pos_vel_batch_tuple+(batch_load_tuples(label,
                                                                "new_pos_vel_tuple.pickle"),)

        log_file_batch_tuple=log_file_batch_tuple+(batch_load_tuples(label,
                                                                "log_file_tuple.pickle"),)
        area_vector_spherical_batch_tuple=area_vector_spherical_batch_tuple+(batch_load_tuples(label,"area_vector_tuple.pickle"),)
        
        interest_vectors_batch_tuple=interest_vectors_batch_tuple+(batch_load_tuples(label,
                                                                                    "interest_vectors_tuple.pickle"),)


                                                                                        
#%% calculating ensemble means and trajectory means 

# need to make this plot more clear 
cutoff=500
ensemble_mean=np.mean(spring_force_positon_tensor_batch_tuple[0][0],axis=0)
ensemble_std=np.std(spring_force_positon_tensor_batch_tuple[0][0],axis=0)

# take mean over all the trajectories to get an ensemble average set of points 



trajecctory_mean=np.mean(spring_force_positon_tensor_batch_tuple[0][0][:,cutoff:,:,:],axis=1)
trajectory_std=np.std(spring_force_positon_tensor_batch_tuple[0][0][:,cutoff:,:,:],axis=1)
# take a time average for each trajectory, then compare the mean for selected particles, to the same particle in the ensemble average 



skip_array=[0,50,99]
for i in range(j_):
    for j in range(1):
        for k in range(len(skip_array)):
            l=skip_array[k]

            plt.axhline(trajecctory_mean[i,l,j],label="particle num="+str(l),linestyle=linestyle_tuple[k][1])
            # plt.axhline(trajecctory_mean[i,k,j]+trajectory_std[i,k,j], label="std dev upper bound of trajectory",linestyle='--')
            # plt.axhline(trajecctory_mean[i,k,j]-trajectory_std[i,k,j], label="std dev lower bound of trajectory",linestyle=':')

            plt.plot(ensemble_mean[cutoff:,l,j])
            plt.axhline(np.mean(ensemble_mean[cutoff:,l,j])) # compare trajectory to mean of ensemble signal 
            plt.xlabel("output count")
            plt.ylabel("stress value")
            plt.legend()
            plt.show()

# need to extend this test to all shear rates 
# probably use a subplot to show the selected particles for each shear rate. 


     



# %%
