##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script allows me to analyse the data from the flat elastic particles in our simulations, hopefully we can 
obtain oldroyd-A

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
plt.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import mmap

path_2_post_proc_module= '/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/LAMMPS python run and analysis scripts/Analysis codes'
os.chdir(path_2_post_proc_module)

from mom2numpy import *
from velP2numpy import *
from dump2numpy import * 
import glob 
from post_MPCD_MP_processing_module import *


SRD_MD_ratio_ = np.array([100,200,400,800,1000])
bending_stiffness=np.array([50,100,200,400])
realisation_index=np.array([1,2,3])
var_choice_1=bending_stiffness
var_choice_2=SRD_MD_ratio_
fluid_name='flatelastictest'
# grabbing file names
VP_general_name_string='vel.'+fluid_name+'**'

Mom_general_name_string='mom.'+fluid_name+'**'

log_general_name_string='log.'+fluid_name+'**'
                         #log.H20_no466188_wall_VACF_output_no_rescale_
TP_general_name_string='temp.'+fluid_name+'**'

dump_general_name_string='fluid_name'+'**.dump'


filepath='KATHLEEN_LAMMPS_RUNS/flat_elastic_tests/test_1'

realisation_name_info= VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string)
realisation_name_Mom=realisation_name_info[0]
realisation_name_VP=realisation_name_info[1]
count_mom=realisation_name_info[2]
count_VP=realisation_name_info[3]
realisation_name_log=realisation_name_info[4]
count_log=realisation_name_info[5]
realisation_name_dump=realisation_name_info[6]
count_dump=realisation_name_info[7]
realisation_name_TP=realisation_name_info[8]
count_TP=realisation_name_info[9]


#%%
# first task is to scan the batch of log files and determine which ones produced errors, then we can re-define the inputs based on this
def log_error_detector(filepath,var_choice_1,var_choice_2,realisation_name_log, count_log):
    os.chdir('/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/'+filepath)
    error_matrix=np.zeros((1,var_choice_1.size,var_choice_2.size))
    general_error_message=bytes("Loop time",'utf-8') # to check if the simulation finished 


    for i in range(0,count_log):
        split_name_for_org =  realisation_name_log[i].split('_')
        var_choice_1_posn= float(split_name_for_org[17]) # need 
        #print(var_choice_1_posn)
        var_choice_2_posn= float(split_name_for_org[10])
        #print(var_choice_2_posn)
        realisation_posn= int(split_name_for_org[6])
        #print(realisation_posn)
        #print(np.where(var_choice_1==var_choice_1_posn))
        var_choice_1_index= np.where(var_choice_1==var_choice_1_posn)[0][0]
        #print(var_choice_1_index)
        var_choice_2_index= np.where(var_choice_2==var_choice_2_posn)[0][0]
        #print(var_choice_2_index)
        realisation_index_= np.where(realisation_index==realisation_posn)[0][0]
        #print(realisation_index_)
        with open(realisation_name_log[i]) as f:
            read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) 
            

            if read_data.find(general_error_message) != -1:
                print('true, error found')
                error_matrix[realisation_index_,var_choice_1_index,var_choice_2_index]=1.0


            else:
                print('Successful run')
                error_matrix[realisation_index_,var_choice_1_index,var_choice_2_index]=0.0

    return error_matrix

#%%

error_matrix=log_error_detector(filepath,var_choice_1,var_choice_2,realisation_name_log, count_log)


